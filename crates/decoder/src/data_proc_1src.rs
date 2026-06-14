//! Data processing (1 source): RBIT / REV16 / REV32 / REV / CLZ / CLS.

use crate::bits::{field, sf};
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    // opcode2 (bits [20:16]) must be 0.
    if field(word, 16, 5) != 0 {
        return Insn::Unsupported { word };
    }
    let opcode = field(word, 10, 6) as u8;
    // Implemented: RBIT(0) REV16(1) REV32(2) REV(3) CLZ(4) CLS(5).
    // REV (opcode 3) is only valid for 64-bit; on 32-bit, REV is opcode 2.
    if opcode > 5 || (opcode == 3 && !sf(word)) {
        return Insn::Unsupported { word };
    }
    Insn::DataProc1Src {
        sf: sf(word),
        opcode,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
