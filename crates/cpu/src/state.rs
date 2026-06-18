//! Architectural register state for a single AArch64 core (EL0 subset).

use std::collections::BTreeMap;

use crate::flags::Flags;
use crate::regs::GuestRegs;
use crate::tlb::Tlb;

/// Encoded register index that aliases SP or the zero register.
pub const SP_OR_ZR: u8 = 31;

/// A pending synchronous memory abort produced by stage-1 translation during a
/// *data* access. The run loop drains this after executing an instruction and
/// vectors to EL1 (Data Abort) with the faulting instruction as the return
/// address, so the guest's page-fault handler can map the page and retry. (An
/// instruction-fetch abort is handled directly in the run loop and never lands
/// here.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Abort {
    /// Faulting virtual address (written to FAR_EL1).
    pub far: u64,
    /// True for a store (sets ESR.WnR), false for a load.
    pub write: bool,
    /// Fault Status Code for ESR.DFSC (e.g. translation fault at level n).
    pub fsc: u8,
}

/// Architectural state of a single AArch64 core (EL0 subset).
///
/// Register index 31 is special: depending on the instruction it means either
/// the zero register (`XZR`/`WZR`, reads as 0 and discards writes) or the stack
/// pointer (`SP`). Callers select which via the `sp` flag on [`Self::read_gpr`]
/// / [`Self::write_gpr`].
///
/// **Layout matters.** `#[repr(C)]` pins the leading fields — `x`, `sp`, `pc`,
/// `nzcv` — to the exact byte offsets in [`regs::offsets`](crate::regs::offsets)
/// that generated JIT blocks load/store (asserted below). In the browser the
/// whole `CpuState` lives in the WASM linear memory the blocks share, so the
/// organizer hands a block a pointer straight at this struct and the block
/// mutates the real registers in place — no per-block image copy. Everything
/// after `nzcv` is private to the interpreter and never touched by a block.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct CpuState {
    /// X0..X30. X31 is *not* stored here — see [`Self::read_gpr`].
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    /// Packed NZCV (bit31 N, bit30 Z, bit29 C, bit28 V) — the form generated JIT
    /// blocks read/write at [`offsets::NZCV`](crate::regs::offsets::NZCV). The
    /// interpreter works in [`flags`](Self::flags) (four bools, faster for its
    /// flag-heavy hot path); the organizer bridges the two (one `u64`) around a
    /// block run. Stale outside that window — read [`flags`](Self::flags) instead.
    pub nzcv: u64,
    /// SIMD/FP registers V0..V31 (128-bit; scalar FP uses the low bits). Placed
    /// here so it sits at [`offsets::V`](crate::regs::offsets::V) (asserted), the
    /// offset generated JIT blocks use for inline vector loads/stores.
    pub v: [u128; 32],
    /// Instructions a compiled block reports executing, written by the block just
    /// before it returns (a block that loops internally runs many iterations per
    /// call, so the organizer can't infer the count from the static block length).
    /// Host-side JIT bookkeeping; meaningful only right after a block run. Its
    /// offset is exported as [`JIT_COUNT_OFFSET`](crate::JIT_COUNT_OFFSET).
    pub jit_count: u64,
    /// Set to 1 by a compiled block that **bailed** mid-way — it hit a memory
    /// access its inline fast path can't handle (TLB miss, MMIO, permission
    /// fault, or page-crossing), stored its partial [`jit_count`](Self::jit_count)
    /// and the faulting PC, and returned. The organizer interprets that one
    /// instruction and resumes. Host-side JIT bookkeeping; cleared before each
    /// block run. Offset exported as [`JIT_EXIT_OFFSET`](crate::JIT_EXIT_OFFSET).
    pub jit_exit: u64,
    pub flags: Flags,
    /// Translation control registers, pulled out of [`Self::sysregs`] into hot
    /// fields because the MMU reads them on every translation (SCTLR on *every*
    /// instruction). A `BTreeMap` lookup here dominated the interpreter's
    /// profile; a plain field load removes it. MRS/MSR route to these (like
    /// FPCR/FPSR), and a write flushes the TLB.
    pub sctlr_el1: u64, // SCTLR_EL1: MMU enable (bit 0) + control
    pub tcr_el1: u64,   // TCR_EL1: region sizes (T0SZ/T1SZ)
    pub ttbr0_el1: u64, // TTBR0_EL1: low-half table root
    pub ttbr1_el1: u64, // TTBR1_EL1: high-half table root
    /// Floating-point control register (rounding mode, default-NaN, etc.).
    pub fpcr: u64,
    /// FPSR — floating-point status register. We model the cumulative (sticky)
    /// exception flags (IOC/DZC/OFC/UFC/IXC, bits 0..=4); they are observational
    /// only (no FP result depends on them and we don't trap FP exceptions).
    pub fpsr: u64,
    /// Exclusive monitor: `(address, value)` recorded by LDXR; a later STXR to
    /// the same address succeeds only if memory still holds `value`.
    pub excl: Option<(u64, u64)>,
    /// System registers, keyed by the encoded (op0,op1,CRn,CRm,op2) tuple.
    /// Read/written by MRS/MSR. The foundation of the system-mode model.
    pub sysregs: BTreeMap<u32, u64>,
    /// Current exception level (0 = EL0 user, 1 = EL1 kernel, ...).
    pub el: u8,
    /// Stack-pointer select: false = SP_EL0, true = SP_ELx.
    pub spsel: bool,
    /// Interrupt mask bits packed as `[D,A,I,F]` in the low 4 bits.
    pub daif: u8,
    /// Banked stack pointers SP_EL0..SP_EL3. The *active* one mirrors `sp`; the
    /// others hold the saved value. See [`Self::set_el_spsel`].
    pub sp_el: [u64; 4],
    /// Set by a PSCI `SYSTEM_OFF`/`SYSTEM_RESET` call; the machine loop stops
    /// when it sees this. Not an architectural register — a host-side halt flag.
    pub powered_off: bool,
    /// A data-access translation fault raised mid-instruction, drained by the run
    /// loop after the instruction returns. Not architectural — a host-side
    /// channel to carry the abort out of the memory helpers. See [`Abort`].
    pub pending_abort: Option<Abort>,
    /// Set when the last retired instruction was WFI/WFE. Not architectural — a
    /// host-side hint the machine reads (and clears) to sleep through guest idle
    /// instead of busy-spinning. The pure interpreter leaves it for the caller.
    pub wfi: bool,
    /// Stage-1 translation cache. Not architectural — a host-side accelerator
    /// that remembers page-table walk results so a repeat access to the same
    /// page skips the walk. Flushed via [`Self::flush_tlb`] on TLBI and on writes
    /// to the translation control registers (see the MMU). See [`Tlb`].
    pub tlb: Tlb,
    /// Set when the last retired instruction was an `IC` (instruction-cache
    /// maintenance) — the architecture's signal that guest code changed. Not
    /// architectural — a host-side hint the JIT organizer reads (and clears) to
    /// drop stale compiled blocks. The pure interpreter ignores it.
    pub ic_inval: bool,
    /// Set whenever the TLB is flushed (TTBR/TCR/SCTLR write or `TLBI`) — i.e. a
    /// VA→PA mapping may have changed. Not architectural — a host-side hint the
    /// JIT organizer reads (and clears) to drop VA-keyed compiled blocks, whose
    /// baked-in PCs are only valid for the translation regime they were compiled
    /// in (so they must go when the MMU toggles or a process switches).
    pub tlb_flushed: bool,
}

// The leading fields must sit at exactly the offsets generated JIT blocks use,
// since a block reads/writes them directly in the shared `CpuState`. If this
// fails, the `#[repr(C)]` layout drifted from `regs::offsets` — fix one or the
// other; they are the contract between the interpreter and emitted code.
const _: () = {
    assert!(std::mem::offset_of!(CpuState, x) == crate::regs::offsets::X);
    assert!(std::mem::offset_of!(CpuState, sp) == crate::regs::offsets::SP);
    assert!(std::mem::offset_of!(CpuState, pc) == crate::regs::offsets::PC);
    assert!(std::mem::offset_of!(CpuState, nzcv) == crate::regs::offsets::NZCV);
    assert!(std::mem::offset_of!(CpuState, v) == crate::regs::offsets::V);
};

/// Byte offset of [`CpuState::jit_count`] within `CpuState`, for the JIT emitter
/// to store the per-block instruction count a block executed (it shares the
/// `#[repr(C)]` layout via the asserts above). Not a `GuestRegs` field — it's
/// past the architectural register window.
pub const JIT_COUNT_OFFSET: usize = std::mem::offset_of!(CpuState, jit_count);

/// Byte offset of [`CpuState::jit_exit`] — the block's mid-way bail flag.
pub const JIT_EXIT_OFFSET: usize = std::mem::offset_of!(CpuState, jit_exit);

/// Byte offset of [`CpuState::tlb`]. The JIT's inline memory fast path loads the
/// TLB entry array pointer from here (the `Box<[Entry; N]>` is a thin pointer at
/// this offset) and indexes it — see `tlb::ENTRY_*`.
pub const TLB_OFFSET: usize = std::mem::offset_of!(CpuState, tlb);

/// Byte offset of [`CpuState::el`] — the inline memory fast path reads it for the
/// EL-dependent permission check.
pub const EL_OFFSET: usize = std::mem::offset_of!(CpuState, el);

impl Default for CpuState {
    fn default() -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: 0,
            nzcv: 0,
            jit_count: 0,
            jit_exit: 0,
            flags: Flags::default(),
            v: [0; 32],
            sctlr_el1: 0,
            tcr_el1: 0,
            ttbr0_el1: 0,
            ttbr1_el1: 0,
            fpcr: 0,
            fpsr: 0,
            excl: None,
            sysregs: BTreeMap::new(),
            el: 1,
            spsel: true,
            daif: 0,
            sp_el: [0; 4],
            powered_off: false,
            pending_abort: None,
            wfi: false,
            tlb: Tlb::new(),
            ic_inval: false,
            tlb_flushed: false,
        }
    }
}

impl CpuState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all cached stage-1 translations. Called when a translation may have
    /// changed: a `TLBI` instruction, or a write to TTBR0/TTBR1/TCR/SCTLR. Also
    /// flags [`tlb_flushed`](Self::tlb_flushed) for the JIT organizer.
    pub fn flush_tlb(&mut self) {
        self.tlb.flush();
        self.tlb_flushed = true;
    }

    /// Read a general-purpose register. When `idx == 31`, `sp` chooses between
    /// the stack pointer (`true`) and the zero register (`false`, reads 0).
    #[must_use]
    pub fn read_gpr(&self, idx: u8, sp: bool) -> u64 {
        match idx {
            SP_OR_ZR if sp => self.sp,
            SP_OR_ZR => 0,
            n => self.x[n as usize],
        }
    }

    /// Write a general-purpose register. When `idx == 31`, `sp` chooses between
    /// the stack pointer (`true`) and the zero register (`false`, write
    /// discarded).
    pub fn write_gpr(&mut self, idx: u8, sp: bool, val: u64) {
        match idx {
            SP_OR_ZR if sp => self.sp = val,
            SP_OR_ZR => {}
            n => self.x[n as usize] = val,
        }
    }

    /// Read a register in 32-bit (`W`) view: the low 32 bits, zero-extended.
    #[must_use]
    pub fn read_gpr_w(&self, idx: u8, sp: bool) -> u64 {
        self.read_gpr(idx, sp) & 0xffff_ffff
    }

    /// Write a 32-bit (`W`) result. Writing a W register zeroes the top half of
    /// the X register, per the architecture.
    pub fn write_gpr_w(&mut self, idx: u8, sp: bool, val: u64) {
        self.write_gpr(idx, sp, val & 0xffff_ffff);
    }

    /// Index of the currently active banked stack pointer (SP_EL0 when SPSel=0).
    #[must_use]
    pub fn sp_index(&self) -> usize {
        if self.spsel {
            self.el as usize
        } else {
            0
        }
    }

    /// Change EL and/or SPSel, banking the stack pointer so `sp` always mirrors
    /// the active SP.
    pub fn set_el_spsel(&mut self, el: u8, spsel: bool) {
        let old = self.sp_index();
        self.el = el;
        self.spsel = spsel;
        let new = self.sp_index();
        if old != new {
            self.sp_el[old] = self.sp;
            self.sp = self.sp_el[new];
        }
    }

    /// Read a banked SP_ELx (the active one lives in `sp`).
    #[must_use]
    pub fn read_sp_el(&self, idx: usize) -> u64 {
        if idx == self.sp_index() {
            self.sp
        } else {
            self.sp_el[idx]
        }
    }

    /// Write a banked SP_ELx.
    pub fn write_sp_el(&mut self, idx: usize, val: u64) {
        if idx == self.sp_index() {
            self.sp = val;
        } else {
            self.sp_el[idx] = val;
        }
    }

    /// Pack the current PSTATE into the AArch64 SPSR layout: NZCV at [31:28],
    /// DAIF at [9:6], and M[3:0] = EL<<2 | SPSel (M[4]=0 for AArch64).
    #[must_use]
    pub fn pstate(&self) -> u64 {
        self.flags.to_nzcv()
            | (u64::from(self.daif) << 6)
            | (u64::from(self.el) << 2)
            | u64::from(self.spsel)
    }

    /// Restore PSTATE from a packed SPSR value (used by ERET).
    pub fn set_pstate(&mut self, v: u64) {
        self.flags = Flags::from_nzcv(v);
        self.daif = ((v >> 6) & 0xf) as u8;
        let el = ((v >> 2) & 0x3) as u8;
        let spsel = v & 1 == 1;
        self.set_el_spsel(el, spsel);
    }

    /// Snapshot the hot register file into the flat [`GuestRegs`] image the JIT
    /// operates on (packing `flags` into `nzcv`).
    #[must_use]
    pub fn to_guest_regs(&self) -> GuestRegs {
        GuestRegs {
            x: self.x,
            sp: self.sp,
            pc: self.pc,
            nzcv: self.flags.to_nzcv(),
            v: self.v,
            fpcr: self.fpcr,
        }
    }

    /// Load the hot register file back from a [`GuestRegs`] image (the JIT writes
    /// `nzcv`; unpack it into `flags`). Cold state (sysregs/EL/...) is untouched.
    pub fn load_guest_regs(&mut self, r: &GuestRegs) {
        self.x = r.x;
        self.sp = r.sp;
        self.pc = r.pc;
        self.flags = Flags::from_nzcv(r.nzcv);
        self.v = r.v;
        self.fpcr = r.fpcr;
    }
}
