//! Crypto SHA1/SHA256 (three-register and two-register forms).

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn three_reg(word: u32) -> Insn {
    if field(word, 22, 2) != 0 {
        return Insn::Unsupported { word };
    }
    let opcode = field(word, 12, 3) as u8;
    if opcode > 6 {
        return Insn::Unsupported { word };
    }
    Insn::CryptoSha3 {
        opcode,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}

pub(crate) fn two_reg(word: u32) -> Insn {
    if field(word, 22, 2) != 0 {
        return Insn::Unsupported { word };
    }
    let opcode = field(word, 12, 5) as u8;
    if opcode > 2 {
        return Insn::Unsupported { word };
    }
    Insn::CryptoSha2 {
        opcode,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
