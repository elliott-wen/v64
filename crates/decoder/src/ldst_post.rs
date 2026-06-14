//! Load/store register, post-indexed signed immediate (`[Rn], #simm`).

use crate::bits::{field, sfield};
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
    Insn::LoadStore {
        size,
        is_load,
        signed,
        dst64,
        vec,
        rt: field(word, 0, 5) as u8,
        addr: AddrMode::PostIndex { rn: field(word, 5, 5) as u8, imm: sfield(word, 12, 9) },
    }
}
