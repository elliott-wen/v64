//! ADR / ADRP — PC-relative address computation. `pc` is the instruction's
//! own address.

use aarch64_cpu_state::CpuState;

use crate::regs::write;

pub(crate) fn exec(cpu: &mut CpuState, page: bool, imm: i64, rd: u8, pc: u64) -> Option<u64> {
    let base = if page { pc & !0xfff } else { pc };
    let result = base.wrapping_add(imm as u64);
    write(cpu, rd, true, result, false); // always a 64-bit result
    None
}
