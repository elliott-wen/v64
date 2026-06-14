//! B.cond — conditional branch (immediate).

use crate::bits::{field, sfield};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    // bit 4 (o0) and bit 24 (o1) must be 0 for B.cond.
    if field(word, 4, 1) != 0 || field(word, 24, 1) != 0 {
        return Insn::Unsupported { word };
    }
    Insn::CondBranch {
        cond: field(word, 0, 4) as u8,
        offset: sfield(word, 5, 19) * 4,
    }
}
