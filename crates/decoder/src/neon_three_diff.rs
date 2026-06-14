//! Advanced SIMD three-different: widening (L), wide (W) and high-narrowing (HN)
//! ops. The destination element is twice (L/W) or half (HN) the source width, so
//! the `2` variants (Q=1) read/write the upper half.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 4) as u8;

    let valid = size <= 2
        && match opcode {
            // ADDL/ADDW/SUBL/SUBW/ABAL/ABDL/MLAL/MLSL/MULL: signed or unsigned.
            0b0000 | 0b0001 | 0b0010 | 0b0011 | 0b0101 | 0b0111 | 0b1000 | 0b1010
            | 0b1100 => true,
            // ADDHN/SUBHN (U=0) and their rounding forms RADDHN/RSUBHN (U=1).
            0b0100 | 0b0110 => true,
            // SQDMLAL/SQDMLSL/SQDMULL: signed only, source H or S.
            0b1001 | 0b1011 | 0b1101 => !u && (size == 1 || size == 2),
            // PMULL: signed-table poly multiply, byte source only (8->16).
            0b1110 => !u && size == 0,
            _ => false,
        };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdThreeDiff {
        q,
        u,
        size,
        opcode,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
