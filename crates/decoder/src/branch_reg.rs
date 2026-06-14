//! BR / BLR / RET — unconditional branch (register).

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let opc = field(word, 21, 4);
    let op2 = field(word, 16, 5);
    let op3 = field(word, 10, 6);
    let op4 = field(word, 0, 5);

    // Only the plain forms (no pointer auth): op2=11111, op3=0, op4=0.
    if op2 != 0b11111 || op3 != 0 || op4 != 0 {
        return Insn::Unsupported { word };
    }
    // ERET: opc=0100, Rn=11111.
    if opc == 0b0100 && field(word, 5, 5) == 0b11111 {
        return Insn::Eret;
    }
    // opc: 0=BR, 1=BLR, 2=RET.
    if opc > 2 {
        return Insn::Unsupported { word };
    }
    Insn::BranchReg {
        opc: opc as u8,
        rn: field(word, 5, 5) as u8,
    }
}
