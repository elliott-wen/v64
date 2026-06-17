//! Scalar FP compare: FCMP/FCMPE (register and #0.0), setting NZCV.

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let ftype = field(word, 22, 2) as u8;
    // op (bits[15:14]) must be 00; opcode2 low 3 bits must be 0.
    let opcode2 = field(word, 0, 5);
    if !ftype_ok(ftype) || field(word, 14, 2) != 0 || opcode2 & 0b00111 != 0 {
        return Insn::Unsupported { word };
    }
    Insn::FpCompare {
        ftype,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        cmp_zero: (opcode2 >> 3) & 1 == 1,
        signaling: (opcode2 >> 4) & 1 == 1,
    }
}
