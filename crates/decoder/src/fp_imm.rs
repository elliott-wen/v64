//! Scalar FP move immediate (FMOV #imm).

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let ftype = field(word, 22, 2) as u8;
    // imm5 (bits[9:5]) must be zero.
    if !ftype_ok(ftype) || field(word, 5, 5) != 0 {
        return Insn::Unsupported { word };
    }
    Insn::FpImm { ftype, imm8: field(word, 13, 8) as u8, rd: field(word, 0, 5) as u8 }
}
