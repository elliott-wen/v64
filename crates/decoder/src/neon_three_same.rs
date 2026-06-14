//! Advanced SIMD three-same: the implemented subset of element-wise vector ops
//! (logical, add/sub, compares, min/max, multiply).

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 11, 5) as u8;

    let valid = match opcode {
        0b00011 => true,                                // logical
        0b00001 | 0b00101 => size != 3 || q,            // SQADD/SQSUB (saturating)
        0b00110 | 0b00111 | 0b10000 | 0b10001 => size != 3 || q, // CMxx, ADD/SUB, CMTST/CMEQ
        0b01000 | 0b01010 => size != 3 || q,            // SSHL/USHL, SRSHL/URSHL
        0b01001 | 0b01011 => size != 3 || q,            // SQSHL, SQRSHL (saturating)
        0b10010 => size <= 2,                           // MLA/MLS
        0b00000 | 0b00010 | 0b00100 => size <= 2,       // S/U HADD, RHADD, HSUB
        0b01100 | 0b01101 | 0b01110 | 0b01111 => size <= 2, // MAX/MIN, ABD, ABA
        0b10100 | 0b10101 => size <= 2,                 // SMAXP/SMINP
        0b10111 => !u && (size != 3 || q),              // ADDP (U=0 only)
        0b10110 => matches!(size, 1 | 2),               // SQDMULH/SQRDMULH (H/S)
        0b10011 => {
            if u {
                size == 0 // PMUL: bytes only
            } else {
                size <= 2 // MUL: no 64-bit
            }
        }
        _ => false,
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdThreeSame {
        q,
        u,
        size,
        opcode,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
