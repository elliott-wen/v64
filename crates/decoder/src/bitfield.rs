//! SBFM / BFM / UBFM — bitfield move (and their many aliases).

use crate::bitmask::decode_bit_masks;
use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let sf = sf(word);
    let opc = field(word, 29, 2) as u8;
    let n = field(word, 22, 1);
    let immr = field(word, 16, 6);
    let imms = field(word, 10, 6);
    let bitsize = if sf { 64 } else { 32 };

    // sf must equal N; immr/imms must be in range; opc==3 is unallocated.
    if (sf as u32) != n || immr >= bitsize || imms >= bitsize || opc > 2 {
        return Insn::Unsupported { word };
    }
    let Some((wmask, tmask)) = decode_bit_masks(n, imms, immr, false) else {
        return Insn::Unsupported { word };
    };
    Insn::Bitfield {
        sf,
        opc,
        wmask,
        tmask,
        immr: immr as u8,
        imms: imms as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
