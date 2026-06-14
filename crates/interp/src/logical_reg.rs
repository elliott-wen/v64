//! AND/ORR/EOR/ANDS and inverted forms BIC/ORN/EON/BICS (shifted register).

use aarch64_cpu_state::{CpuState, Flags};
use aarch64_decoder::ShiftType;

use crate::alu::apply_shift;
use crate::regs::{read, width_mask, write};

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    opc: u8,
    negate: bool,
    shift: ShiftType,
    amount: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let mask = width_mask(sf);
    let a = read(cpu, rn, sf, false);
    let mut b = apply_shift(read(cpu, rm, sf, false), shift, amount, sf);
    if negate {
        b = !b & mask; // BIC/ORN/EON/BICS
    }
    let result = match opc {
        0 | 3 => a & b, // AND / ANDS (BIC / BICS with negate)
        1 => a | b,     // ORR (ORN)
        2 => a ^ b,     // EOR (EON)
        _ => unreachable!(),
    };
    if opc == 3 {
        let r = result & mask;
        cpu.flags = Flags {
            n: r >> (if sf { 63 } else { 31 }) & 1 == 1,
            z: r == 0,
            c: false,
            v: false,
        };
    }
    write(cpu, rd, sf, result, false);
    None
}
