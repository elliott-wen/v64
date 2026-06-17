//! Stage-1 address translation (4KB granule) with permission and access-flag
//! checks, modelled on the ARM ARM `AArch64.TranslationTableWalk` /
//! `AArch64.S1CheckPermissions` and matching QEMU/Unicorn behaviour.
//!
//! A VA is resolved by walking the tables rooted at TTBR0/TTBR1_EL1 (chosen by
//! VA bit 55), honouring the leaf descriptor's access permissions (AP[2:1],
//! UXN/PXN), the access flag (AF), and the hierarchical table permissions
//! (APTable/UXNTable/PXNTable accumulated down the walk). Any failure returns
//! the ESR fault-status code (`Err(fsc)`), which the caller turns into an
//! Instruction/Data Abort so the guest's handler can demand-page, copy-on-write,
//! or signal as appropriate. `SCTLR_EL1.M` clear means the MMU is off (VA == PA).

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;

use crate::memory::GuestMem;

/// Output-address mask for a 4KB-granule descriptor (bits [47:12]).
const OA_MASK: u64 = 0x0000_ffff_ffff_f000;

// ESR fault status codes; the low two bits carry the level the walk faulted at.
const FSC_TRANSLATION: u8 = 0b0001_00;
const FSC_ACCESS_FLAG: u8 = 0b0010_00;
const FSC_PERMISSION: u8 = 0b0011_00;

/// The kind of access being translated, which selects the permission check.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    Read,
    Write,
    Exec,
}

fn sysreg(cpu: &CpuState, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u64 {
    cpu.sysregs.get(&sysreg_key(op0, op1, crn, crm, op2)).copied().unwrap_or(0)
}

fn mmu_enabled(cpu: &CpuState) -> bool {
    sysreg(cpu, 3, 0, 1, 0, 0) & 1 == 1 // SCTLR_EL1.M
}

/// Translate `va` for an access of kind `access` at the current EL. `Ok(pa)` on
/// success (and always when the MMU is off — identity); `Err(fsc)` is the ESR
/// fault status code for a translation / access-flag / permission fault.
/// `el` is the exception level the permission check is evaluated *at* — normally
/// the current EL, but 0 for unprivileged (LDTR/STTR) accesses even when issued
/// from EL1.
pub fn translate(
    cpu: &CpuState,
    mem: &mut dyn GuestMem,
    va: u64,
    access: Access,
    el: u8,
) -> Result<u64, u8> {
    if !mmu_enabled(cpu) {
        return Ok(va);
    }
    let tcr = sysreg(cpu, 3, 0, 2, 0, 2); // TCR_EL1
    let (ttbr, tsz) = if (va >> 55) & 1 == 1 {
        (sysreg(cpu, 3, 0, 2, 0, 1), (tcr >> 16) & 0x3f) // TTBR1, T1SZ
    } else {
        (sysreg(cpu, 3, 0, 2, 0, 0), tcr & 0x3f) // TTBR0, T0SZ
    };
    walk(mem, ttbr & OA_MASK, va, tsz as u32, access, el)
}

/// Hierarchical (table-descriptor) permission restrictions accumulated as the
/// walk descends. Each only ever tightens permissions.
#[derive(Default, Clone, Copy)]
struct TablePerms {
    no_el0: bool,    // APTable[0]: no EL0 access in the subtree
    no_write: bool,  // APTable[1]: read-only in the subtree
    uxn: bool,       // UXNTable: unprivileged-execute-never in the subtree
    pxn: bool,       // PXNTable: privileged-execute-never in the subtree
}

/// Walk the 4KB-granule tables from `table_base` (a physical address).
fn walk(
    mem: &mut dyn GuestMem,
    mut table_base: u64,
    va: u64,
    tsz: u32,
    access: Access,
    el: u8,
) -> Result<u64, u8> {
    let mut level = starting_level(tsz);
    let mut tp = TablePerms::default();
    loop {
        let shift = 39 - level * 9; // L0=39, L1=30, L2=21, L3=12
        let idx = (va >> shift) & 0x1ff;
        let desc = mem.read_u64(table_base + idx * 8);

        // Bit 0 clear => invalid descriptor: translation fault at this level.
        if desc & 1 == 0 {
            return Err(FSC_TRANSLATION | level as u8);
        }

        // A leaf is an L3 page (bits[1:0]=11) or an L0-L2 block (bit1 clear).
        let is_leaf = level == 3 || desc & 0b10 == 0;
        if is_leaf {
            // Access flag fault takes priority over permission (ARM ARM ordering).
            if desc & (1 << 10) == 0 {
                return Err(FSC_ACCESS_FLAG | level as u8);
            }
            check_perms(el, desc, &tp, access).map_err(|()| FSC_PERMISSION | level as u8)?;
            let block_mask = (1u64 << shift) - 1; // low bits within this leaf
            return Ok((desc & OA_MASK & !block_mask) | (va & block_mask));
        }

        // Table descriptor: accumulate hierarchical permission restrictions.
        let ap_table = (desc >> 61) & 0b11;
        tp.no_el0 |= ap_table & 0b01 != 0; // APTable[0]
        tp.no_write |= ap_table & 0b10 != 0; // APTable[1]
        tp.uxn |= desc & (1 << 60) != 0; // UXNTable
        tp.pxn |= desc & (1 << 59) != 0; // PXNTable

        table_base = desc & OA_MASK;
        level += 1;
    }
}

/// Check a leaf descriptor's permissions for `access` evaluated at exception
/// level `el`. `Err(())` means a permission fault (the caller adds the level).
fn check_perms(el: u8, desc: u64, tp: &TablePerms, access: Access) -> Result<(), ()> {
    // AP[2:1] at bits[7:6]: AP[1]=EL0-access-enable, AP[2]=read-only.
    let el0_access = (desc & (1 << 6) != 0) && !tp.no_el0;
    let read_only = (desc & (1 << 7) != 0) || tp.no_write;
    let uxn = (desc & (1 << 54) != 0) || tp.uxn;
    let pxn = (desc & (1 << 53) != 0) || tp.pxn;
    let el0 = el == 0;

    match access {
        Access::Read => {
            // EL1 may read regardless of AP; EL0 needs the EL0-access bit.
            if el0 && !el0_access {
                return Err(());
            }
        }
        Access::Write => {
            if el0 && !el0_access {
                return Err(());
            }
            if read_only {
                return Err(()); // write to a read-only page (the COW trigger)
            }
        }
        Access::Exec => {
            // Instruction fetch is governed by the execute-never bits only.
            if el0 {
                if uxn {
                    return Err(());
                }
            } else if pxn {
                return Err(());
            }
        }
    }
    Ok(())
}

/// Starting lookup level for a 4KB granule given T0SZ/T1SZ.
fn starting_level(tsz: u32) -> u32 {
    let va_bits = 64 - tsz; // resolvable VA width
    let levels = (va_bits - 12).div_ceil(9); // 9 address bits per level
    4 - levels
}
