//! Router for scalar floating-point data-processing.
//!
//! All these share bits[28:24]=11110 and bit21=1; the low bits [11:10] plus a
//! few higher bits select the class. Advanced SIMD (vector) and half-precision
//! are not handled yet — see `DESIGN_neon.md`.

use crate::bits::field;
use crate::insn::Insn;
use crate::{fp_ccmp, fp_compare, fp_csel, fp_cvt, fp_dp1, fp_dp2, fp_dp3, fp_imm};

pub(crate) fn decode(word: u32) -> Insn {
    // 3-source (FMADD/FMSUB/FNMADD/FNMSUB) shares op0 but has bits[28:24]=11111.
    if field(word, 24, 5) == 0b11111 {
        return fp_dp3::decode(word);
    }
    if field(word, 24, 5) != 0b11110 || field(word, 21, 1) != 1 {
        return Insn::Unsupported { word };
    }
    match field(word, 10, 2) {
        0b00 => {
            if field(word, 10, 6) == 0b000000 {
                fp_cvt::decode(word) // convert FP<->int / FMOV gpr
            } else if field(word, 10, 5) == 0b10000 {
                fp_dp1::decode(word) // 1-source
            } else if field(word, 10, 4) == 0b1000 {
                fp_compare::decode(word)
            } else if field(word, 10, 3) == 0b100 {
                fp_imm::decode(word)
            } else {
                Insn::Unsupported { word }
            }
        }
        0b10 => fp_dp2::decode(word), // 2-source
        0b11 => fp_csel::decode(word),
        _ => fp_ccmp::decode(word), // 0b01 = FP conditional compare
    }
}

/// Common ftype check: only single (00) and double (01) are supported.
pub(crate) fn ftype_ok(ftype: u8) -> bool {
    ftype == 0b00 || ftype == 0b01
}
