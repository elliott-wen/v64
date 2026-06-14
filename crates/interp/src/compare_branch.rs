//! CBZ / CBNZ — compare and branch on zero / non-zero.

use aarch64_cpu_state::CpuState;

use crate::regs::read;

pub(crate) fn exec(
    cpu: &CpuState,
    sf: bool,
    negate: bool,
    rt: u8,
    offset: i64,
    pc: u64,
) -> Option<u64> {
    let v = read(cpu, rt, sf, false);
    let take = if negate { v != 0 } else { v == 0 };
    take.then(|| pc.wrapping_add(offset as u64))
}
