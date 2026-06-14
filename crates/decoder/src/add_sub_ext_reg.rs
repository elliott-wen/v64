//! ADD/SUB (extended register), including ADDS/SUBS.
//!
//! The second operand is taken from `Rm`, extended (sign/zero, byte..word) per
//! `option`, then left-shifted by `imm3` (0..4).

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let imm3 = field(word, 10, 3);
    // imm3 > 4 is unallocated.
    if imm3 > 4 {
        return Insn::Unsupported { word };
    }
    Insn::AddSubExtReg {
        sf: sf(word),
        sub: field(word, 30, 1) == 1,
        set_flags: field(word, 29, 1) == 1,
        option: field(word, 13, 3) as u8,
        imm3: imm3 as u8,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
