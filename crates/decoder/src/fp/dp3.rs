//! Scalar FP data-processing, 3 source: FMADD/FMSUB/FNMADD/FNMSUB.

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let ftype = field(word, 22, 2) as u8;
    if !ftype_ok(ftype) {
        return Insn::Unsupported { word };
    }
    Insn::FpDataProc3 {
        ftype,
        o1: field(word, 21, 1) == 1,
        o0: field(word, 15, 1) == 1,
        rm: field(word, 16, 5) as u8,
        ra: field(word, 10, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
