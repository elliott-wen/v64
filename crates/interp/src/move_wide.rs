//! MOVZ / MOVN / MOVK.

use aarch64_cpu_state::CpuState;

use crate::regs::write;

pub(crate) fn exec(cpu: &mut CpuState, sf: bool, opc: u8, hw: u8, imm16: u16, rd: u8) -> Option<u64> {
    let shift = u32::from(hw) * 16;
    let imm = u64::from(imm16) << shift;
    let result = match opc {
        2 => imm,  // MOVZ
        0 => !imm, // MOVN
        3 => {
            // MOVK: keep other bits, replace the 16-bit field.
            let cur = cpu.read_gpr(rd, false);
            let mask = !(0xffff_u64 << shift);
            (cur & mask) | imm
        }
        _ => unreachable!("decoder rejects opc==1"),
    };
    write(cpu, rd, sf, result, false);
    None
}
