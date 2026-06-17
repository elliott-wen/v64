//! Advanced SIMD across-lanes: ADDV / SADDLV / UADDLV / SMAXV / UMAXV / SMINV /
//! UMINV.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 5) as u8;

    // No 64-bit reduction; a 32-bit element needs the 4S (Q=1) form.
    let size_ok = size <= 2 && (size != 2 || q);
    let valid = size_ok
        && match opcode {
            0b00011 => true, // SADDLV / UADDLV (widening add across)
            0b11011 => !u, // ADDV
            0b01010 | 0b11010 => true, // S/U MAXV, S/U MINV
            _ => false,
        };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdAcrossLanes {
        q,
        u,
        size,
        opcode,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
