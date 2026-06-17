//! Load register (literal) — PC-relative load: LDR (W/X) and LDRSW.

use crate::bits::{field, sfield};
use crate::insn::{AddrMode, Insn};

pub(crate) fn decode(word: u32) -> Insn {
    let vec = field(word, 26, 1) == 1;
    // opc: integer 00 LDR(W) 01 LDR(X) 10 LDRSW; SIMD 00 S, 01 D, 10 Q.
    let (size, signed, dst64) = if vec {
        match field(word, 30, 2) {
            0b00 => (2u8, false, false), // S (4 bytes)
            0b01 => (3, false, false),   // D (8 bytes)
            0b10 => (4, false, false),   // Q (16 bytes)
            _ => return Insn::Unsupported { word },
        }
    } else {
        match field(word, 30, 2) {
            0b00 => (2, false, false),
            0b01 => (3, false, true),
            0b10 => (2, true, true),     // LDRSW
            0b11 => return Insn::Prfm,    // PRFM (literal)
            _ => return Insn::Unsupported { word },
        }
    };
    let offset = sfield(word, 5, 19) * 4; // imm19, scaled
    Insn::LoadStore {
        size,
        is_load: true,
        signed,
        dst64,
        vec,
        unpriv: false,
        rt: field(word, 0, 5) as u8,
        addr: AddrMode::Literal { offset },
    }
}
