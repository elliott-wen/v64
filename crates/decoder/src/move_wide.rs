//! MOVZ / MOVN / MOVK — move (wide) immediate.

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let opc = field(word, 29, 2) as u8;
    // opc == 1 is unallocated for move-wide.
    if opc == 1 {
        return Insn::Unsupported { word };
    }
    let hw = field(word, 21, 2) as u8;
    // For 32-bit, hw>=2 is unallocated (shift would exceed the register).
    if !sf(word) && hw >= 2 {
        return Insn::Unsupported { word };
    }
    Insn::MoveWide {
        sf: sf(word),
        opc,
        hw,
        imm16: field(word, 5, 16) as u16,
        rd: field(word, 0, 5) as u8,
    }
}
