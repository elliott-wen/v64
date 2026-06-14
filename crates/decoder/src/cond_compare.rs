//! CCMP / CCMN — conditional compare (register and immediate forms).

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    // bit 29 (S) must be 1; bits 10 and 4 must be 0.
    if field(word, 29, 1) != 1 || field(word, 10, 1) != 0 || field(word, 4, 1) != 0 {
        return Insn::Unsupported { word };
    }
    Insn::CondCompare {
        sf: sf(word),
        sub: field(word, 30, 1) == 1, // op: 1=CCMP, 0=CCMN
        is_imm: field(word, 11, 1) == 1,
        imm_y: field(word, 16, 5) as u8,
        rm: field(word, 16, 5) as u8,
        cond: field(word, 12, 4) as u8,
        nzcv: field(word, 0, 4) as u8,
        rn: field(word, 5, 5) as u8,
    }
}
