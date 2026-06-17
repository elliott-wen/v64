//! Scalar FP conditional compare: FCCMP/FCCMPE.
//!
//! Encoding: `0 0 0 11110 ftype 1 Rm cond 01 Rn op nzcv`, where `op` (bit 4)
//! selects the signaling form (FCCMPE).

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let ftype = field(word, 22, 2) as u8;
    if field(word, 21, 1) != 1 || !ftype_ok(ftype) {
        return Insn::Unsupported { word };
    }
    Insn::FpCondCompare {
        ftype,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        cond: field(word, 12, 4) as u8,
        nzcv: field(word, 0, 4) as u8,
        signaling: field(word, 4, 1) == 1,
    }
}
