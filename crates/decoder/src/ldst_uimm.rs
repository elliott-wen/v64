//! Load/store register, unsigned (scaled) immediate offset:
//! STRB/LDRB/LDRSB, STRH/LDRH/LDRSH, STR/LDR/LDRSW, STR/LDR (no writeback).

use crate::bits::field;
use crate::insn::{AddrMode, Insn};
use crate::ldst;

pub(crate) fn decode(word: u32) -> Insn {
    let size = field(word, 30, 2) as u8;
    let Some((is_load, signed, dst64)) = ldst::kind(size, field(word, 22, 2)) else {
        return Insn::Unsupported { word };
    };
    let imm = u64::from(field(word, 10, 12)) << size; // scaled by access size
    Insn::LoadStore {
        size,
        is_load,
        signed,
        dst64,
        rt: field(word, 0, 5) as u8,
        addr: AddrMode::UnsignedImm { rn: field(word, 5, 5) as u8, imm },
    }
}
