//! TBZ / TBNZ — test bit and branch on zero / non-zero.

use aarch64_cpu_state::CpuState;

pub(crate) fn exec(
    cpu: &CpuState,
    bit: u8,
    negate: bool,
    rt: u8,
    offset: i64,
    pc: u64,
) -> Option<u64> {
    // The tested bit may be up to 63, so always read the full X register.
    let v = cpu.read_gpr(rt, false);
    let set = (v >> bit) & 1 == 1;
    let take = if negate { set } else { !set };
    take.then(|| pc.wrapping_add(offset as u64))
}
