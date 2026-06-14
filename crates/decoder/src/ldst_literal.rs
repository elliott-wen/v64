//! Load register (literal) — PC-relative load: LDR (W/X) and LDRSW.

use crate::bits::{field, sfield};
use crate::insn::{AddrMode, Insn};

pub(crate) fn decode(word: u32) -> Insn {
    // opc: 00 LDR(W), 01 LDR(X), 10 LDRSW, 11 PRFM (not implemented).
    let (size, signed, dst64) = match field(word, 30, 2) {
        0b00 => (2, false, false),
        0b01 => (3, false, true),
        0b10 => (2, true, true), // LDRSW
        _ => return Insn::Unsupported { word },
    };
    let offset = sfield(word, 5, 19) * 4; // imm19, scaled
    Insn::LoadStore {
        size,
        is_load: true,
        signed,
        dst64,
        rt: field(word, 0, 5) as u8,
        addr: AddrMode::Literal { offset },
    }
}
