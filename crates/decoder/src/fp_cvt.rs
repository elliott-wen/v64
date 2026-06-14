//! Convert between FP and integer, plus FMOV gpr<->fpr.
//! SCVTF/UCVTF, FCVTZS/FCVTZU (round toward zero), FMOV.

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let sf = field(word, 31, 1) == 1;
    let ftype = field(word, 22, 2) as u8;
    let rmode = field(word, 19, 2) as u8;
    let opcode = field(word, 16, 3) as u8;
    if !ftype_ok(ftype) {
        return Insn::Unsupported { word };
    }

    let valid = match (rmode, opcode) {
        (0b00, 0b010) | (0b00, 0b011) => true, // SCVTF / UCVTF
        // FCVT{N,P,M,Z}{S,U}: any rmode with opcode 000/001.
        (_, 0b000) | (_, 0b001) => true,
        // FCVTAS / FCVTAU: tie-away, rmode encoded as 00 with opcode 100/101.
        (0b00, 0b100) | (0b00, 0b101) => true,
        // FMOV gpr<->fpr: W<->S (sf=0, single) or X<->D (sf=1, double).
        (0b00, 0b110) | (0b00, 0b111) => (!sf && ftype == 0b00) || (sf && ftype == 0b01),
        _ => false,
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::FpCvtInt {
        sf,
        ftype,
        rmode,
        opcode,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
