//! Crypto AES: AESE/AESD/AESMC/AESIMC.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    // AES requires size == 0.
    if field(word, 22, 2) != 0 {
        return Insn::Unsupported { word };
    }
    let opcode = field(word, 12, 5) as u8;
    if !matches!(opcode, 0x4 | 0x5 | 0x6 | 0x7) {
        return Insn::Unsupported { word };
    }
    Insn::CryptoAes {
        opcode,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
