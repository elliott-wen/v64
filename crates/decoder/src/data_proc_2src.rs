//! Data processing (2 source): UDIV / SDIV / LSLV / LSRV / ASRV / RORV.

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let opcode = field(word, 10, 6) as u8;
    // Implemented subset: UDIV(2) SDIV(3) LSLV(8) LSRV(9) ASRV(10) RORV(11).
    let ok = matches!(opcode, 2 | 3 | 8 | 9 | 10 | 11);
    if !ok {
        return Insn::Unsupported { word };
    }
    Insn::DataProc2Src {
        sf: sf(word),
        opcode,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
