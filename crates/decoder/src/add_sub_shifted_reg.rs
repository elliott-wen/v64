//! ADD/SUB (shifted register), including ADDS/SUBS.

use crate::bits::{field, sf};
use crate::insn::{Insn, ShiftType};

pub(crate) fn decode(word: u32) -> Insn {
    let shift = match field(word, 22, 2) {
        0 => ShiftType::Lsl,
        1 => ShiftType::Lsr,
        2 => ShiftType::Asr,
        // 0b11 (ROR) is reserved for add/sub shifted register.
        _ => return Insn::Unsupported { word },
    };
    let amount = field(word, 10, 6);
    // For 32-bit, a shift amount >= 32 is unallocated.
    if !sf(word) && amount >= 32 {
        return Insn::Unsupported { word };
    }
    Insn::AddSubShiftedReg {
        sf: sf(word),
        sub: field(word, 30, 1) == 1,
        set_flags: field(word, 29, 1) == 1,
        shift,
        amount: amount as u8,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
