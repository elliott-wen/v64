//! BR / BLR / RET — unconditional branch (register).

use aarch64_cpu_state::CpuState;

pub(crate) fn exec(cpu: &mut CpuState, opc: u8, rn: u8, pc: u64) -> Option<u64> {
    let target = cpu.read_gpr(rn, false);
    if opc == 1 {
        // BLR: set the return address.
        cpu.write_gpr(30, false, pc.wrapping_add(4));
    }
    Some(target)
}
