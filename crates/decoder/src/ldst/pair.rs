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
    let vec = field(word, 26, 1) == 1;

    let (width8, signed, vesize) = if vec {
        // SIMD pair: opc 00=S(2), 01=D(3), 10=Q(4).
        let vesize = match field(word, 30, 2) {
            0b00 => 2u8,
            0b01 => 3,
            0b10 => 4,
            _ => return Insn::Unsupported { word },
        };
        (false, false, vesize)
    } else {
        // opc: 00 = 32-bit, 10 = 64-bit, 01 = LDPSW (load only, no non-alloc), 11 reserved.
        match field(word, 30, 2) {
            0b00 => (false, false, 2),
            0b10 => (true, false, 3),
            0b01 if is_load && !nalloc => (false, true, 2),
            _ => return Insn::Unsupported { word },
        }
    };

    let offset = sfield(word, 15, 7) << vesize;
    Insn::LoadStorePair {
        is_load,
        signed,
        width8,
        vec,
        vesize,
        rt: field(word, 0, 5) as u8,
        rt2: field(word, 10, 5) as u8,
        rn: field(word, 5, 5) as u8,
        offset,
        index,
    }
}
