//! ADC/SBC/ADCS/SBCS — add/sub with carry.

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    // opcode2 (bits [15:10]) must be 0.
    if field(word, 10, 6) != 0 {
        return Insn::Unsupported { word };
    }
    Insn::AddSubCarry {
        sf: sf(word),
        sub: field(word, 30, 1) == 1, // op: 0=ADC, 1=SBC
        set_flags: field(word, 29, 1) == 1,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
