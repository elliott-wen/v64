//! Advanced SIMD three-same extra: SQRDMLAH/SQRDMLSH (FEAT_RDM) and SDOT/UDOT
//! (FEAT_DotProd). FCMLA/FCADD (FEAT_FCMA) are not implemented.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let u = field(word, 29, 1);
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 11, 4) as u8;

    let valid = match u * 16 + u32::from(opcode) {
        0x10 | 0x11 => size == 1 || size == 2, // SQRDMLAH / SQRDMLSH
        0x02 | 0x12 => size == 2,              // SDOT / UDOT
        _ => false,
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdThreeSameExtra {
        q: field(word, 30, 1) == 1,
        u: u == 1,
        size,
        opcode,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
