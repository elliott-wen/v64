//! Load/store register, register offset (`[Rn, Rm, ext #amount]`).

use crate::bits::field;
use crate::insn::{AddrMode, Insn};
use crate::ldst;

pub(crate) fn decode(word: u32) -> Insn {
    let size = field(word, 30, 2) as u8;
    let opc = field(word, 22, 2);
    let (size, is_load, signed, dst64, vec) = if field(word, 26, 1) == 1 {
        let Some((is_load, log2)) = ldst::vec_kind(size, opc) else {
            return Insn::Unsupported { word };
        };
        (log2, is_load, false, false, true)
    } else {
        let Some((is_load, signed, dst64)) = ldst::kind(size, opc) else {
            return Insn::Unsupported { word };
        };
        (size, is_load, signed, dst64, false)
    };
    let option = field(word, 13, 3) as u8;
    // Valid extends are UXTW/LSL/SXTW/SXTX (option bit 1 set); others reserved.
    if option & 0b010 == 0 {
        return Insn::Unsupported { word };
    }
    // S (bit 12) scales the index by the access size.
    let shift = if field(word, 12, 1) == 1 { size } else { 0 };
    Insn::LoadStore {
        size,
        is_load,
        signed,
        dst64,
        vec,
        rt: field(word, 0, 5) as u8,
        addr: AddrMode::RegOffset {
            rn: field(word, 5, 5) as u8,
            rm: field(word, 16, 5) as u8,
            option,
            shift,
        },
    }
}
