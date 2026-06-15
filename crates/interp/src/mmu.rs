//! Stage-1 address translation (4KB granule).
//!
//! When `SCTLR_EL1.M` is clear the MMU is off and VA == PA. When set, a VA is
//! resolved by walking the translation tables rooted at TTBR0/TTBR1_EL1, per the
//! ARM ARM `AArch64.TranslationTableWalk`. Descriptors are read from *physical*
//! memory (the flat `Memory` is the physical address space).
//!
//! Faults (invalid descriptor / permission) are not yet raised as Data Aborts —
//! the translation falls back to identity so valid mappings work; wiring faults
//! into `exception::take_exception` is the follow-up (see `DESIGN_mmu.md`).

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;

use crate::memory::GuestMem;

/// Output-address mask for a 4KB-granule descriptor (bits [47:12]).
const OA_MASK: u64 = 0x0000_ffff_ffff_f000;

fn sysreg(cpu: &CpuState, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u64 {
    cpu.sysregs.get(&sysreg_key(op0, op1, crn, crm, op2)).copied().unwrap_or(0)
}

fn mmu_enabled(cpu: &CpuState) -> bool {
    sysreg(cpu, 3, 0, 1, 0, 0) & 1 == 1 // SCTLR_EL1.M
}

/// Translate a virtual address to physical. Identity when the MMU is off.
#[must_use]
pub fn translate(cpu: &CpuState, mem: &mut dyn GuestMem, va: u64) -> u64 {
    if !mmu_enabled(cpu) {
        return va;
    }
    let tcr = sysreg(cpu, 3, 0, 2, 0, 2); // TCR_EL1
    let (ttbr, tsz) = if (va >> 55) & 1 == 1 {
        (sysreg(cpu, 3, 0, 2, 0, 1), (tcr >> 16) & 0x3f) // TTBR1, T1SZ
    } else {
        (sysreg(cpu, 3, 0, 2, 0, 0), tcr & 0x3f) // TTBR0, T0SZ
    };
    walk(mem, ttbr & OA_MASK, va, tsz as u32)
}

/// Walk the 4KB-granule tables from `table_base` (a physical address).
fn walk(mem: &mut dyn GuestMem, mut table_base: u64, va: u64, tsz: u32) -> u64 {
    let mut level = starting_level(tsz);
    loop {
        let shift = 39 - level * 9; // L0=39, L1=30, L2=21, L3=12
        let idx = (va >> shift) & 0x1ff;
        let desc = mem.read_u64(table_base + idx * 8);

        if desc & 1 == 0 {
            return va; // invalid descriptor -> fault (TODO); fall back to identity
        }
        if level == 3 {
            // Level 3 is always a page; descriptor type bit1 must be 1.
            return (desc & OA_MASK) | (va & 0xfff);
        }
        if desc & 0b10 == 0 {
            // Block descriptor: output a (2^shift)-aligned region.
            let mask = (1u64 << shift) - 1;
            return (desc & OA_MASK & !mask) | (va & mask);
        }
        // Table descriptor: descend.
        table_base = desc & OA_MASK;
        level += 1;
    }
}

/// Starting lookup level for a 4KB granule given T0SZ/T1SZ.
fn starting_level(tsz: u32) -> u32 {
    let va_bits = 64 - tsz; // resolvable VA width
    let levels = (va_bits - 12).div_ceil(9); // 9 address bits per level
    4 - levels
}
