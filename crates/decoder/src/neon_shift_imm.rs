//! Advanced SIMD shift by immediate: SHL, SSHR/USHR, SSRA/USRA.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let immh = field(word, 19, 4) as u8;
    let opcode = field(word, 11, 5) as u8;

    // Element size = highest set bit of immh; 64-bit needs the Q=1 form.
    let size = if immh & 0b1000 != 0 {
        3
    } else if immh & 0b0100 != 0 {
        2
    } else if immh & 0b0010 != 0 {
        1
    } else {
        0
    };
    if size == 3 && !q {
        return Insn::Unsupported { word };
    }
    let u = field(word, 29, 1) == 1;
    // SSHR/USHR (00000), SSRA/USRA (00010), SHL (01010, U=0 only; U=1 is SLI).
    let ok = matches!(opcode, 0b00000 | 0b00010) || (opcode == 0b01010 && !u);
    if !ok {
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
