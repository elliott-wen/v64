//! Scalar FP data-processing, 1 source: FMOV/FABS/FNEG/FSQRT and FCVT.

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let ftype = field(word, 22, 2) as u8;
    let opcode = field(word, 15, 6) as u8;

    let valid = match opcode {
        // FMOV/FABS/FNEG/FSQRT keep the type.
        0..=3 => ftype_ok(ftype),
        // FCVT to single (from double) / to double (from single).
        4 => ftype == 0b01,
        5 => ftype == 0b00,
        _ => false, // FCVT-to-half, FRINT*, etc. not implemented
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::FpDataProc1 { ftype, opcode, rn: field(word, 5, 5) as u8, rd: field(word, 0, 5) as u8 }
}
