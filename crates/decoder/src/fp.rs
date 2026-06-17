//! Router for scalar floating-point data-processing.
//!
//! All these share bits[28:24]=11110 and bit21=1; the low bits [11:10] plus a
//! few higher bits select the class. Advanced SIMD (vector) and half-precision
//! are not handled yet — see `DESIGN_neon.md`.

use crate::bits::field;
use crate::insn::Insn;

mod ccmp;
mod compare;
mod csel;
mod cvt;
mod dp1;
mod dp2;
mod dp3;
mod fixed;
mod imm;

pub(crate) fn decode(word: u32) -> Insn {
    // 3-source (FMADD/FMSUB/FNMADD/FNMSUB) shares op0 but has bits[28:24]=11111.
    if field(word, 24, 5) == 0b11111 {
        return dp3::decode(word);
    }
    if field(word, 24, 5) != 0b11110 {
        return Insn::Unsupported { word };
    }
    // bit21=0 selects FP<->fixed-point conversion; bit21=1 is everything else.
    if field(word, 21, 1) == 0 {
        return fixed::decode(word);
    }
    match field(word, 10, 2) {
        0b00 => {
            if field(word, 10, 6) == 0b000000 {
                cvt::decode(word) // convert FP<->int / FMOV gpr
            } else if field(word, 10, 5) == 0b10000 {
                dp1::decode(word) // 1-source
            } else if field(word, 10, 4) == 0b1000 {
                compare::decode(word)
            } else if field(word, 10, 3) == 0b100 {
                imm::decode(word)
            } else {
                Insn::Unsupported { word }
            }
        }
        0b10 => dp2::decode(word), // 2-source
        0b11 => csel::decode(word),
        _ => ccmp::decode(word), // 0b01 = FP conditional compare
    }
}

/// Common ftype check: only single (00) and double (01) are supported.
pub(crate) fn ftype_ok(ftype: u8) -> bool {
    ftype == 0b00 || ftype == 0b01
}
