//! ADD/SUB (immediate), including the flag-setting ADDS/SUBS.

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let shift = field(word, 22, 2);
    // shift field 1x is unallocated; only LSL #0 and LSL #12 are valid.
    if shift >= 2 {
        return Insn::Unsupported { word };
    }
    Insn::AddSubImm {
        sf: sf(word),
        sub: field(word, 30, 1) == 1,
        set_flags: field(word, 29, 1) == 1,
        shift12: shift == 1,
        imm12: field(word, 10, 12) as u16,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
