//! Scalar FP conditional select (FCSEL).

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let ftype = field(word, 22, 2) as u8;
    if !ftype_ok(ftype) {
        return Insn::Unsupported { word };
    }
    Insn::FpCondSelect {
        ftype,
        cond: field(word, 12, 4) as u8,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
