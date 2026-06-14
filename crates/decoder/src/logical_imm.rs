//! AND/ORR/EOR/ANDS (immediate).
//!
//! The (N, imms, immr) bitmask is resolved here via [`decode_bit_masks`]; an
//! unallocated mask makes the whole encoding unallocated.

use crate::bitmask::decode_bit_masks;
use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let sf = sf(word);
    let n = field(word, 22, 1);
    // For 32-bit operations N must be 0.
    if !sf && n != 0 {
        return Insn::Unsupported { word };
    }
    let immr = field(word, 16, 6);
    let imms = field(word, 10, 6);
    let Some((mask, _)) = decode_bit_masks(n, imms, immr, true) else {
        return Insn::Unsupported { word };
    };
    let imm = if sf { mask } else { mask & 0xffff_ffff };
    Insn::LogicalImm {
        sf,
        opc: field(word, 29, 2) as u8,
        imm,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
