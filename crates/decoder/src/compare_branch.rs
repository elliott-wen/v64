//! CBZ / CBNZ — compare and branch on zero / non-zero.

use crate::bits::{field, sf, sfield};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    Insn::CompareBranch {
        sf: sf(word),
        negate: field(word, 24, 1) == 1, // op: 0=CBZ, 1=CBNZ
        rt: field(word, 0, 5) as u8,
        offset: sfield(word, 5, 19) * 4,
    }
}
