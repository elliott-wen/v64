//! CSEL / CSINC / CSINV / CSNEG — conditional select.

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    // op2 (bits [11:10]) must be 0b0x; the low bit selects the increment/negate
    // form, bit 11 must be zero.
    if field(word, 11, 1) != 0 {
        return Insn::Unsupported { word };
    }
    Insn::CondSelect {
        sf: sf(word),
        op: field(word, 30, 1) == 1,
        o2: field(word, 10, 1) == 1,
        cond: field(word, 12, 4) as u8,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
