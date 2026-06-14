//! AND/ORR/EOR/ANDS and inverted forms BIC/ORN/EON/BICS (shifted register).

use crate::bits::{field, sf};
use crate::insn::{Insn, ShiftType};

pub(crate) fn decode(word: u32) -> Insn {
    let shift = match field(word, 22, 2) {
        0 => ShiftType::Lsl,
        1 => ShiftType::Lsr,
        2 => ShiftType::Asr,
        3 => ShiftType::Ror,
        _ => unreachable!(),
    };
    let amount = field(word, 10, 6);
    // For 32-bit, a shift amount >= 32 is unallocated.
    if !sf(word) && amount >= 32 {
        return Insn::Unsupported { word };
    }
    Insn::LogicalShiftedReg {
        sf: sf(word),
        opc: field(word, 29, 2) as u8,
        negate: field(word, 21, 1) == 1,
        shift,
        amount: amount as u8,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
