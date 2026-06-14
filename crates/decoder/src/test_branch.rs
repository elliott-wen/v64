//! TBZ / TBNZ — test bit and branch on zero / non-zero.

use crate::bits::{field, sfield};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    // Bit position = b5:b40 (bit 31 is the high bit, bits [23:19] the low five).
    let bit = ((field(word, 31, 1) << 5) | field(word, 19, 5)) as u8;
    Insn::TestBranch {
        bit,
        negate: field(word, 24, 1) == 1, // op: 0=TBZ, 1=TBNZ
        rt: field(word, 0, 5) as u8,
        offset: sfield(word, 5, 14) * 4,
    }
}
