//! Advanced SIMD modified immediate: MOVI/MVNI/ORR/BIC (the integer cmodes).
//! FMOV-vector (cmode 1111) is not implemented yet.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let cmode = field(word, 12, 4) as u8;
    let op = field(word, 29, 1) == 1;
    // cmode 1111 is FMOV-vector (op=0 single, op=1 double); not implemented.
    if cmode == 0b1111 {
        return Insn::Unsupported { word };
    }
    // The o2 bit (bit 11) must be 0 for the implemented forms.
    if field(word, 11, 1) != 0 {
        return Insn::Unsupported { word };
    }
    let imm8 = ((field(word, 16, 3) << 5) | field(word, 5, 5)) as u8;
    Insn::SimdModImm {
        q: field(word, 30, 1) == 1,
        op,
        cmode,
        imm8,
        rd: field(word, 0, 5) as u8,
    }
}
