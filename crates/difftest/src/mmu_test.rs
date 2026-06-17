//! Builder for stage-1 (4KB-granule) translation tables plus the guest code that
//! turns the MMU on, used to drive the MMU differentially against Unicorn.
//!
//! A test maps a few VA->PA pages/blocks and supplies one access instruction.
//! `mmu_vector` assembles a program that seeds TTBR0/TCR/MAIR via `MSR`, sets
//! `SCTLR.M`, `ISB`s, then runs the access — exactly how real software (and
//! Unicorn's own MMU test) enables translation. Both implementations execute the
//! identical program, so the architectural result is directly comparable.

use std::collections::BTreeMap;

use crate::vector::TestVector;
use crate::CODE_START;

/// Physical base where tables are allocated (clear of code @0x80000 and the
/// compared DATA window @0x40000).
const PT_BASE: u64 = 0x1_0000;
const PAGE: u64 = 0x1000;
/// T0SZ=32 -> 32-bit VA -> the walk starts at level 1 (L1/L2/L3). This matches
/// Unicorn's CPU-TLB MMU configuration; TTBR0 points at the L1 table. Our own
/// interpreter computes the same start level from T0SZ, so the two agree.
const START_LEVEL: u32 = 1;
/// 4KB-granule output-address mask (bits [47:12]).
const OA: u64 = 0x0000_ffff_ffff_f000;

const SH_IS: u64 = 0b11 << 8; // inner shareable (Unicorn's walker expects SH set)
const TABLE_DESC: u64 = 0b11; // pointer to next-level table

/// Leaf-descriptor attributes for a mapping. `Leaf::rw()` is the common case
/// (EL0-accessible, read/write, access flag set, executable).
#[derive(Clone, Copy)]
pub struct Leaf {
    pub writable: bool,
    pub af: bool,
    pub el0: bool, // AP[1]: accessible at EL0
    pub uxn: bool, // unprivileged execute never
    pub pxn: bool, // privileged execute never
}

impl Leaf {
    #[must_use]
    pub fn rw() -> Self {
        Self { writable: true, af: true, el0: true, uxn: false, pxn: false }
    }
}

/// Scratch register the setup preamble clobbers (not used by access encodings,
/// which use x0/x1/x2). Both sides clobber it identically, so it still compares.
const SCRATCH: u32 = 9;

/// Exception vector base (identity-mapped). On a fault the same-EL synchronous
/// vector at VBAR+0x200 captures the syndrome so it can be compared.
const VBAR: u64 = 0x9_0000;
/// Registers the fault handler leaves the captured syndrome in.
const ESR_REG: u32 = 10;
const FAR_REG: u32 = 11;

/// A growing set of 4KB translation tables.
pub struct PageTables {
    tables: BTreeMap<u64, [u64; 512]>,
    next: u64,
    l0: u64,
}

impl Default for PageTables {
    fn default() -> Self {
        Self::new()
    }
}

impl PageTables {
    #[must_use]
    pub fn new() -> Self {
        let mut s = Self { tables: BTreeMap::new(), next: PT_BASE, l0: 0 };
        s.l0 = s.alloc();
        s
    }

    fn alloc(&mut self) -> u64 {
        let pa = self.next;
        self.next += PAGE;
        self.tables.insert(pa, [0u64; 512]);
        pa
    }

    /// Map one 4KB page VA -> PA, read/write with the access flag set.
    pub fn map_page(&mut self, va: u64, pa: u64) {
        self.map_page_attr(va, pa, true, true);
    }

    /// Map one 4KB page with explicit `writable` / `af` (access flag) leaf bits,
    /// EL0-accessible and executable.
    pub fn map_page_attr(&mut self, va: u64, pa: u64, writable: bool, af: bool) {
        self.map_leaf(va, pa, 3, Leaf { writable, af, el0: true, uxn: false, pxn: false });
    }

    /// Map one 4KB page with fully-specified leaf attributes.
    pub fn map_page_leaf(&mut self, va: u64, pa: u64, leaf: Leaf) {
        self.map_leaf(va, pa, 3, leaf);
    }

    /// Map one 2MB block VA -> PA (both must be 2MB-aligned).
    pub fn map_block_2m(&mut self, va: u64, pa: u64) {
        self.map_leaf(va, pa, 2, Leaf { writable: true, af: true, el0: true, uxn: false, pxn: false });
    }

    /// Map one 2MB block with fully-specified leaf attributes.
    pub fn map_block_2m_leaf(&mut self, va: u64, pa: u64, leaf: Leaf) {
        self.map_leaf(va, pa, 2, leaf);
    }

    fn map_leaf(&mut self, va: u64, pa: u64, leaf_level: u32, leaf: Leaf) {
        let mut table = self.l0;
        for level in START_LEVEL..leaf_level {
            let shift = 39 - level * 9;
            let idx = ((va >> shift) & 0x1ff) as usize;
            let entry = self.tables[&table][idx];
            table = if entry & 1 == 0 {
                let child = self.alloc();
                self.tables.get_mut(&table).unwrap()[idx] = child | TABLE_DESC;
                child
            } else {
                entry & OA
            };
        }
        let shift = 39 - leaf_level * 9;
        let idx = ((va >> shift) & 0x1ff) as usize;
        // Leaf: type + SH(IS) + AP + AF + execute-never bits.
        let mut desc = (pa & OA) | SH_IS;
        desc |= if leaf_level == 3 { 0b11 } else { 0b01 };
        if leaf.el0 {
            desc |= 1 << 6; // AP[1] = EL0 access
        }
        if !leaf.writable {
            desc |= 1 << 7; // AP[2] = read-only
        }
        if leaf.af {
            desc |= 1 << 10;
        }
        if leaf.pxn {
            desc |= 1 << 53;
        }
        if leaf.uxn {
            desc |= 1 << 54;
        }
        self.tables.get_mut(&table).unwrap()[idx] = desc;
    }

    /// OR hierarchical permission restrictions into the table descriptor at
    /// `level` (1 or 2) on `va`'s path: APTable (no_el0 / no_write) and the
    /// UXNTable / PXNTable execute-never restrictions. The path must already be
    /// mapped. Validates our accumulation of hierarchical permissions.
    pub fn restrict_table(&mut self, va: u64, level: u32, no_el0: bool, no_write: bool, uxn: bool, pxn: bool) {
        let mut table = self.l0;
        for lvl in START_LEVEL..level {
            let shift = 39 - lvl * 9;
            let idx = ((va >> shift) & 0x1ff) as usize;
            table = self.tables[&table][idx] & OA;
        }
        let shift = 39 - level * 9;
        let idx = ((va >> shift) & 0x1ff) as usize;
        let mut d = self.tables[&table][idx];
        if no_el0 {
            d |= 1 << 61; // APTable[0]
        }
        if no_write {
            d |= 1 << 62; // APTable[1]
        }
        if pxn {
            d |= 1 << 59; // PXNTable
        }
        if uxn {
            d |= 1 << 60; // UXNTable
        }
        self.tables.get_mut(&table).unwrap()[idx] = d;
    }

    #[must_use]
    pub fn ttbr0(&self) -> u64 {
        self.l0
    }

    /// Serialize every table to `(physical_addr, little-endian bytes)` patches.
    #[must_use]
    pub fn patches(&self) -> Vec<(u64, Vec<u8>)> {
        self.tables
            .iter()
            .map(|(pa, t)| {
                let mut bytes = Vec::with_capacity(512 * 8);
                for e in t.iter() {
                    bytes.extend_from_slice(&e.to_le_bytes());
                }
                (*pa, bytes)
            })
            .collect()
    }
}

/// The exception level the kernel build uses (CONFIG_ARM64_VA_BITS=39 -> T0SZ =
/// 64-39 = 25). Both 25 and Unicorn's test value 32 give a 3-level walk
/// (START_LEVEL = 1), so the same table layout serves either.
pub const KERNEL_T0SZ: u64 = 25;

/// TCR_EL1 for a 4KB-granule, inner-shareable write-back regime with the given
/// `t0sz`. Our interpreter only reads T0SZ (low 6 bits); the upper bits match
/// Unicorn's CPU-TLB MMU test so its walker is satisfied.
fn tcr(t0sz: u64) -> u64 {
    (0x1_8080_3F20 & !0x3f) | (t0sz & 0x3f)
}

/// MAIR_EL1: Attr0 = Normal memory, write-back (so Unicorn treats it as RAM).
fn mair() -> u64 {
    0xFF
}

// --- minimal AArch64 encoders for the MMU setup preamble ---

fn movz(rd: u32, imm16: u64, shift: u32) -> u32 {
    0xD280_0000 | (shift << 21) | ((imm16 as u32 & 0xffff) << 5) | rd
}
fn movk(rd: u32, imm16: u64, shift: u32) -> u32 {
    0xF280_0000 | (shift << 21) | ((imm16 as u32 & 0xffff) << 5) | rd
}
/// MSR (register) to an EL1 system register (op0=3,op1=0).
fn msr(crn: u32, crm: u32, op2: u32, rt: u32) -> u32 {
    0xD500_0000 | (3 << 19) | (crn << 12) | (crm << 8) | (op2 << 5) | rt
}
/// MRS from an EL1 system register (op0=3,op1=0).
fn mrs(rt: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    0xD530_0000 | (3 << 19) | (crn << 12) | (crm << 8) | (op2 << 5) | rt
}
/// AND Xd, Xn, #0x3f (extract the 6-bit fault status code from ESR).
fn and_dfsc(rd: u32, rn: u32) -> u32 {
    // 64-bit AND immediate, bitmask value 0x3f: N=1, immr=0, imms=5.
    0x9240_0000 | (5 << 10) | (rn << 5) | rd
}
/// Unconditional branch B to absolute `target` from address `from`.
fn b_to(from: u64, target: u64) -> u32 {
    let off = (target as i64 - from as i64) >> 2;
    0x1400_0000 | (off as u32 & 0x03ff_ffff)
}
/// ADR Xd, #imm (PC-relative, byte offset).
fn adr(rd: u32, imm: u32) -> u32 {
    0x1000_0000 | ((imm & 3) << 29) | (((imm >> 2) & 0x7ffff) << 5) | rd
}
const ISB: u32 = 0xD503_3FDF;
const ERET: u32 = 0xD69F_03E0;

/// SCTLR_EL1 value enabling the MMU. This is Unicorn's MAX-model reset value
/// (0x00C5_0838, which carries the architectural RES1 bits) with `M` (bit 0)
/// set. Writing a fixed value (rather than read-modify-write) keeps the scratch
/// register identical on both sides — our interpreter doesn't model the SCTLR
/// reset bits, so an `MRS` of it would read 0 and diverge.
const SCTLR_MMU_ON: u64 = 0x00C5_0839;

fn emit(out: &mut Vec<u8>, word: u32) {
    out.extend_from_slice(&word.to_le_bytes());
}

/// Load a 64-bit constant into `rd` with MOVZ/MOVK.
fn load_imm(out: &mut Vec<u8>, rd: u32, val: u64) {
    emit(out, movz(rd, val & 0xffff, 0));
    for sh in 1..4 {
        let chunk = (val >> (sh * 16)) & 0xffff;
        if chunk != 0 {
            emit(out, movk(rd, chunk, sh));
        }
    }
}

/// A fault handler that records ESR.DFSC into x10 and FAR into x11, then branches
/// to `until`. Placed at a vector slot `base`.
fn fault_handler(base: u64, until: u64) -> Vec<u8> {
    let mut h = Vec::new();
    emit(&mut h, mrs(ESR_REG, 5, 2, 0)); // ESR_EL1
    emit(&mut h, mrs(FAR_REG, 6, 0, 0)); // FAR_EL1
    emit(&mut h, and_dfsc(ESR_REG, ESR_REG));
    emit(&mut h, b_to(base + 12, until));
    h
}

/// Build a TestVector running `access` at exception level `el` (0 or 1) with the
/// MMU on, given a fully-populated table set. Identity-maps the program and
/// vector pages, installs fault handlers (same-EL @VBAR+0x200 and lower-EL
/// @VBAR+0x400) that record ESR.DFSC (x10) and FAR (x11), emits the MMU-enable
/// preamble, and — for `el == 0` — drops to EL0 via ERET just before the access.
fn finish(access: &[u8], mut pt: PageTables, data: Option<Vec<u8>>, el: u8, t0sz: u64) -> TestVector {
    // Identity-map the program pages (preamble+access) and the vector page.
    for i in 0..2 {
        let p = (CODE_START & !(PAGE - 1)) + i * PAGE;
        pt.map_page(p, p);
    }
    pt.map_page(VBAR, VBAR);

    let mut code = Vec::new();
    load_imm(&mut code, SCRATCH, VBAR);
    emit(&mut code, msr(12, 0, 0, SCRATCH)); // VBAR_EL1
    load_imm(&mut code, SCRATCH, pt.ttbr0());
    emit(&mut code, msr(2, 0, 0, SCRATCH)); // TTBR0_EL1
    load_imm(&mut code, SCRATCH, tcr(t0sz));
    emit(&mut code, msr(2, 0, 2, SCRATCH)); // TCR_EL1
    load_imm(&mut code, SCRATCH, mair());
    emit(&mut code, msr(10, 2, 0, SCRATCH)); // MAIR_EL1
    load_imm(&mut code, SCRATCH, SCTLR_MMU_ON);
    emit(&mut code, msr(1, 0, 0, SCRATCH)); // SCTLR_EL1 (enables MMU)
    emit(&mut code, ISB);

    if el == 0 {
        // Drop to EL0: ELR_EL1 = the access (5 instrs / 20 bytes after the ADR),
        // SPSR_EL1 = 0 (EL0t), then ERET. The access then runs at EL0.
        emit(&mut code, adr(SCRATCH, 20));
        emit(&mut code, msr(4, 0, 1, SCRATCH)); // ELR_EL1
        emit(&mut code, movz(SCRATCH, 0, 0)); // SPSR = EL0t
        emit(&mut code, msr(4, 0, 0, SCRATCH)); // SPSR_EL1
        emit(&mut code, ERET);
    }
    code.extend_from_slice(access);

    let until = CODE_START + code.len() as u64;

    let mut patches = pt.patches();
    patches.push((VBAR + 0x200, fault_handler(VBAR + 0x200, until))); // same-EL sync
    patches.push((VBAR + 0x400, fault_handler(VBAR + 0x400, until))); // lower-EL sync

    let mut tv = TestVector::new(&code);
    tv.init_data = data;
    tv.mem_patches = patches;
    tv.cpu_tlb = true;
    tv
}

/// Build a TestVector that enables the MMU then runs `access`. `pages` and
/// `blocks` add read/write VA->PA mappings; `data` seeds (and enables comparison
/// of) the physical DATA window.
#[must_use]
pub fn mmu_vector(
    access: &[u8],
    pages: &[(u64, u64)],
    blocks: &[(u64, u64)],
    data: Option<Vec<u8>>,
) -> TestVector {
    let mut pt = PageTables::new();
    for &(va, pa) in pages {
        pt.map_page(va, pa);
    }
    for &(va, pa) in blocks {
        pt.map_block_2m(va, pa);
    }
    finish(access, pt, data, 1, KERNEL_T0SZ)
}

/// Build a TestVector with arbitrary table contents (`map` populates them via
/// the full `PageTables` API — custom permissions, access flag, blocks, etc.),
/// run at EL1, for permission/access-flag-fault and randomized tests.
#[must_use]
pub fn mmu_vector_with(
    access: &[u8],
    map: impl FnOnce(&mut PageTables),
    data: Option<Vec<u8>>,
) -> TestVector {
    mmu_vector_with_el(access, map, data, 1)
}

/// Like [`mmu_vector_with`] but runs the access at exception level `el` (0 or 1),
/// for EL0 permission differential testing.
#[must_use]
pub fn mmu_vector_with_el(
    access: &[u8],
    map: impl FnOnce(&mut PageTables),
    data: Option<Vec<u8>>,
    el: u8,
) -> TestVector {
    mmu_vector_with_el_t0sz(access, map, data, el, KERNEL_T0SZ)
}

/// Like [`mmu_vector_with_el`] but with an explicit `t0sz` (the translation
/// regime size). Valid 3-level values are 25..=33; the fuzzer varies it to
/// exercise the exact regime the kernel uses (25) and neighbours.
#[must_use]
pub fn mmu_vector_with_el_t0sz(
    access: &[u8],
    map: impl FnOnce(&mut PageTables),
    data: Option<Vec<u8>>,
    el: u8,
    t0sz: u64,
) -> TestVector {
    let mut pt = PageTables::new();
    map(&mut pt);
    finish(access, pt, data, el, t0sz)
}
