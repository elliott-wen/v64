//! Advanced SIMD extract (EXT): byte extraction from the {Vm:Vn} concatenation.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let imm4 = field(word, 11, 4) as u8;
    // For the 64-bit (Q=0) form, imm4 bit 3 must be 0 (index < 8).
    if !q && imm4 & 0b1000 != 0 {
        return Insn::Unsupported { word };
    }
    Insn::SimdExt {
        q,
        imm4,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
