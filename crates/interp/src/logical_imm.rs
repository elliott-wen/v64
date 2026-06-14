//! AND/ORR/EOR/ANDS (immediate).

use aarch64_cpu_state::{CpuState, Flags};

use crate::regs::{read, width_mask, write};

pub(crate) fn exec(cpu: &mut CpuState, sf: bool, opc: u8, imm: u64, rn: u8, rd: u8) -> Option<u64> {
    let a = read(cpu, rn, sf, false);
    let result = match opc {
        0 | 3 => a & imm, // AND / ANDS
        1 => a | imm,     // ORR
        2 => a ^ imm,     // EOR
        _ => unreachable!(),
    };
    if opc == 3 {
        // ANDS sets N/Z from the result; C and V are cleared.
        let r = result & width_mask(sf);
        cpu.flags = Flags {
            n: r >> (if sf { 63 } else { 31 }) & 1 == 1,
            z: r == 0,
            c: false,
            v: false,
        };
    }
    // AND/ORR/EOR are SP-capable at Rd; ANDS writes ZR.
    write(cpu, rd, sf, result, opc != 3);
    None
}
