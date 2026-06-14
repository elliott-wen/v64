//! Shared operand read/write helpers honoring W/X width and SP/ZR semantics.

use aarch64_cpu_state::CpuState;

/// Operand width: 64-bit mask for X, 32-bit for W.
#[must_use]
pub fn width_mask(sf: bool) -> u64 {
    if sf {
        u64::MAX
    } else {
        0xffff_ffff
    }
}

/// Datasize in bits.
#[must_use]
pub fn datasize(sf: bool) -> u32 {
    if sf {
        64
    } else {
        32
    }
}

/// Read an operand, honoring the W/X width. `sp_ctx` selects SP-vs-ZR for r31.
#[must_use]
pub fn read(cpu: &CpuState, idx: u8, sf: bool, sp_ctx: bool) -> u64 {
    if sf {
        cpu.read_gpr(idx, sp_ctx)
    } else {
        cpu.read_gpr_w(idx, sp_ctx)
    }
}

/// Write a result with correct W/X zero-extension. `sp` selects SP-vs-ZR for
/// r31.
pub fn write(cpu: &mut CpuState, idx: u8, sf: bool, val: u64, sp: bool) {
    if sf {
        cpu.write_gpr(idx, sp, val);
    } else {
        cpu.write_gpr_w(idx, sp, val);
    }
}
