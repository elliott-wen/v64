//! Convert between floating-point and fixed-point: SCVTF/UCVTF (fixed int -> FP)
//! and FCVTZS/FCVTZU (FP -> fixed int, round toward zero). These share the FP
//! data-processing space but have bit21=0 and a 6-bit `scale` field; the number
//! of fractional bits is `64 - scale`.

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let sf = field(word, 31, 1) == 1;
    let ftype = field(word, 22, 2) as u8;
    let rmode = field(word, 19, 2) as u8;
    let opcode = field(word, 16, 3) as u8;
    let scale = field(word, 10, 6) as u8;

    if !ftype_ok(ftype) {
        return Insn::Unsupported { word };
    }
    // Only the four fixed-point conversions are allocated.
    let valid = matches!(
        (rmode, opcode),
        (0b00, 0b010) | (0b00, 0b011) | (0b11, 0b000) | (0b11, 0b001)
    );
    // For a 32-bit operand, scale[5] must be 1 (fraction bits <= 32); else reserved.
    if !valid || (!sf && scale < 32) {
        return Insn::Unsupported { word };
    }

    Insn::FpCvtFixed {
        sf,
        ftype,
        opcode,
        scale,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
