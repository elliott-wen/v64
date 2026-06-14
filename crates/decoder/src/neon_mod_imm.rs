//! Advanced SIMD modified immediate: MOVI/MVNI/ORR/BIC (integer cmodes) and
//! FMOV-vector (cmode 1111: single when op=0, double when op=1 & Q=1).

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let cmode = field(word, 12, 4) as u8;
    let op = field(word, 29, 1) == 1;
    let q = field(word, 30, 1) == 1;
    // cmode 1111 FMOV-vector: op=1 & Q=0 is the FP16 form (FEAT_FP16) — skip.
    if cmode == 0b1111 && op && !q {
        return Insn::Unsupported { word };
    }
    // The o2 bit (bit 11) must be 0 for the implemented forms.
    if field(word, 11, 1) != 0 {
        return Insn::Unsupported { word };
    }
    let imm8 = ((field(word, 16, 3) << 5) | field(word, 5, 5)) as u8;
    Insn::SimdModImm {
        q,
        op,
        cmode,
        imm8,
        rd: field(word, 0, 5) as u8,
    }
}
