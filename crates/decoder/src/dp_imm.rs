//! Router for the "Data processing -- immediate" encoding group.
//!
//! Sub-class is selected by bits [25:23]; each class lives in its own module.

use crate::bits::field;
use crate::insn::Insn;
use crate::{add_sub_imm, bitfield, extract, logical_imm, move_wide, pc_rel};

pub(crate) fn decode(word: u32) -> Insn {
    match field(word, 23, 3) {
        0b000 | 0b001 => pc_rel::decode(word),
        0b010 => add_sub_imm::decode(word),
        // 0b011 = add/sub (immediate, with tags) — not implemented.
        0b100 => logical_imm::decode(word),
        0b101 => move_wide::decode(word),
        0b110 => bitfield::decode(word),
        0b111 => extract::decode(word),
        _ => Insn::Unsupported { word },
    }
}
