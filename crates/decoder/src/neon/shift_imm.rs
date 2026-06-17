//! Advanced SIMD shift by immediate: the full set — same-width shifts
//! (SSHR/SRSHR/SSRA/SRI/SHL/SLI/SQSHL/...), narrowing (SHRN/SQSHRN/...),
//! widening (SSHLL/USHLL) and the fixed-point conversions (SCVTF/FCVTZS).

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let u = field(word, 29, 1) == 1;
    let immh = field(word, 19, 4) as u8;
    let opcode = field(word, 11, 5) as u8;

    // Element size = highest set bit of immh.
    let size = if immh & 0b1000 != 0 {
        3
    } else if immh & 0b0100 != 0 {
        2
    } else if immh & 0b0010 != 0 {
        1
    } else {
        0
    };

    let valid = match opcode {
        0b00000 | 0b00010 | 0b00100 | 0b00110 => size != 3 || q, // SSHR/SSRA/SRSHR/SRSRA
        0b01000 => u && (size != 3 || q),                        // SRI (U=1)
        0b01010 => size != 3 || q,                               // SHL / SLI
        0b01100 => u && (size != 3 || q),                        // SQSHLU (U=1)
        0b01110 => size != 3 || q,                               // SQSHL / UQSHL
        0b10000 | 0b10001 | 0b10010 | 0b10011 => size <= 2,      // narrowing
        0b10100 => size <= 2,                                    // SSHLL / USHLL
        0b11100 | 0b11111 => size >= 2 && (size != 3 || q),      // SCVTF/UCVTF, FCVTZS/U
        _ => false,
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdShiftImm {
        q,
        u,
        immh,
        immb: field(word, 16, 3) as u8,
        opcode,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
