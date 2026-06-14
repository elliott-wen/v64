//! Advanced SIMD permute: ZIP1/ZIP2, UZP1/UZP2, TRN1/TRN2.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 3) as u8;
    // opcode 001 UZP1, 010 TRN1, 011 ZIP1, 101 UZP2, 110 TRN2, 111 ZIP2.
    if opcode == 0 || opcode == 0b100 {
        return Insn::Unsupported { word };
    }
    // A 64-bit (D) element only exists in the 2D (Q=1) form.
    if size == 3 && !q {
        return Insn::Unsupported { word };
    }
    Insn::SimdZipTrn {
        q,
        size,
        opcode,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
