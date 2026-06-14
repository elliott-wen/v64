//! B / BL — unconditional branch (immediate).

use crate::bits::{field, sfield};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    Insn::BranchImm {
        link: field(word, 31, 1) == 1,
        offset: sfield(word, 0, 26) * 4,
    }
}
