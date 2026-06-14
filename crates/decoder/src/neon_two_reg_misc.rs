//! Advanced SIMD two-register misc: the implemented subset
//! (REV64/REV32/REV16, CLS/CLZ, CNT, NOT/RBIT, ABS, NEG).

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 5) as u8;

    // (u, opcode) -> the size constraint that makes the encoding allocated.
    let valid = match (u, opcode) {
        (false, 0b00000) => size <= 2,            // REV64
        (false, 0b00001) => size == 0,            // REV16
        (true, 0b00000) => size <= 1,             // REV32
        (false, 0b00100) | (true, 0b00100) => size <= 2, // CLS / CLZ
        (false, 0b00101) => size == 0,            // CNT
        (true, 0b00101) => size <= 1,             // NOT (size 0) / RBIT (size 1)
        (false, 0b01011) | (true, 0b01011) => size != 3 || q, // ABS / NEG
        _ => false,
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdTwoRegMisc {
        q,
        u,
        size,
        opcode,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
