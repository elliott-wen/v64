//! Load/store register, unscaled signed immediate (LDUR/STUR/LDURSB...).

use crate::bits::{field, sfield};
use crate::insn::{AddrMode, Insn};
use crate::ldst;

pub(crate) fn decode(word: u32) -> Insn {
    let size = field(word, 30, 2) as u8;
    let Some((is_load, signed, dst64)) = ldst::kind(size, field(word, 22, 2)) else {
        return Insn::Unsupported { word };
    };
    Insn::LoadStore {
        size,
        is_load,
        signed,
        dst64,
        rt: field(word, 0, 5) as u8,
        addr: AddrMode::Unscaled { rn: field(word, 5, 5) as u8, imm: sfield(word, 12, 9) },
    }
}
