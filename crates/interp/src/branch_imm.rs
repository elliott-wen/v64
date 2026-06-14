//! B / BL — unconditional branch (immediate). `pc` is the branch's own address.

use aarch64_cpu_state::CpuState;

pub(crate) fn exec(cpu: &mut CpuState, link: bool, offset: i64, pc: u64) -> Option<u64> {
    if link {
        cpu.write_gpr(30, false, pc.wrapping_add(4)); // X30 = return address
    }
    Some(pc.wrapping_add(offset as u64))
}
