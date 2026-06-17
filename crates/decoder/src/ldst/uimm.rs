//! Load/store register, unsigned (scaled) immediate offset:
//! STRB/LDRB/LDRSB, STRH/LDRH/LDRSH, STR/LDR/LDRSW, STR/LDR (no writeback).

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
    } else if ldst::is_prefetch(size, opc) {
        return Insn::Prfm; // PRFM (unsigned imm)
    } else {
        let Some((is_load, signed, dst64)) = ldst::kind(size, opc) else {
            return Insn::Unsupported { word };
        };
        (size, is_load, signed, dst64, false)
    };
    let imm = u64::from(field(word, 10, 12)) << size; // scaled by access size
    Insn::LoadStore {
        size,
        is_load,
        signed,
        dst64,
        vec,
        unpriv: false,
        rt: field(word, 0, 5) as u8,
        addr: AddrMode::UnsignedImm { rn: field(word, 5, 5) as u8, imm },
    }
}
