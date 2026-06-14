//! Load/store pair: LDP/STP (32- and 64-bit) and LDPSW.

use crate::bits::{field, sfield};
use crate::insn::{Insn, PairIndex};

pub(crate) fn decode(word: u32) -> Insn {
    let idx = field(word, 23, 3);
    let index = match idx {
        0b000 | 0b010 => PairIndex::Offset, // 000 is the non-allocating hint
        0b001 => PairIndex::Post,
        0b011 => PairIndex::Pre,
        _ => return Insn::Unsupported { word },
    };
    let nalloc = idx == 0b000;
    let is_load = field(word, 22, 1) == 1;

    // opc: 00 = 32-bit, 10 = 64-bit, 01 = LDPSW (load only, no non-allocating
    // form), 11 reserved.
    let (width8, signed) = match field(word, 30, 2) {
        0b00 => (false, false),
        0b10 => (true, false),
        0b01 if is_load && !nalloc => (false, true),
        _ => return Insn::Unsupported { word },
    };

    let scale = if width8 { 3 } else { 2 };
    let offset = sfield(word, 15, 7) << scale;
    Insn::LoadStorePair {
        is_load,
        signed,
        width8,
        rt: field(word, 0, 5) as u8,
        rt2: field(word, 10, 5) as u8,
        rn: field(word, 5, 5) as u8,
        offset,
        index,
    }
}
