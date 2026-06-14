//! Scalar FP data-processing, 2 source:
//! FMUL/FDIV/FADD/FSUB/FMAX/FMIN/FMAXNM/FMINNM/FNMUL.

use crate::bits::field;
use crate::fp::ftype_ok;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let ftype = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 4) as u8;
    if !ftype_ok(ftype) || opcode > 0b1000 {
        return Insn::Unsupported { word };
    }
    Insn::FpDataProc2 {
        ftype,
        opcode,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
