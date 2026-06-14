//! EXTR — extract a register from a pair (also the ROR-immediate alias).

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let sf = sf(word);
    let n = field(word, 22, 1);
    let o0 = field(word, 21, 1);
    let op21 = field(word, 29, 2);
    let imms = field(word, 10, 6);
    let bitsize = if sf { 64 } else { 32 };

    // sf must equal N, op21/o0 must be zero, lsb must be in range.
    if (sf as u32) != n || op21 != 0 || o0 != 0 || imms >= bitsize {
        return Insn::Unsupported { word };
    }
    Insn::Extract {
        sf,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        lsb: imms as u8,
        rd: field(word, 0, 5) as u8,
    }
}
