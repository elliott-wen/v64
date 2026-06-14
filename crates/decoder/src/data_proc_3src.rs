//! Data processing (3 source): MADD/MSUB, S/UMADDL, S/UMSUBL, S/UMULH.

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    // op54 (bits [30:29]) must be 0.
    if field(word, 29, 2) != 0 {
        return Insn::Unsupported { word };
    }
    let sf = sf(word);
    let op31 = field(word, 21, 3) as u8;
    let o0 = field(word, 15, 1) == 1;

    // Validate the allocated combinations. The *L (long) and *MULH forms are
    // 64-bit only.
    let valid = match op31 {
        0b000 => true,                  // MADD / MSUB (both widths)
        0b001 | 0b101 => sf,            // SMADDL/SMSUBL, UMADDL/UMSUBL
        0b010 | 0b110 => sf && !o0,     // SMULH / UMULH
        _ => false,
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::DataProc3Src {
        sf,
        op31,
        o0,
        rm: field(word, 16, 5) as u8,
        ra: field(word, 10, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
