//! B.cond — conditional branch (immediate).

use aarch64_cpu_state::CpuState;

use crate::cond::eval_cond;

pub(crate) fn exec(cpu: &CpuState, cond: u8, offset: i64, pc: u64) -> Option<u64> {
    if eval_cond(cond, cpu.flags) {
        Some(pc.wrapping_add(offset as u64))
    } else {
        None
    }
}
