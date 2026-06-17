//! Advanced SIMD table lookup: TBL/TBX. `len+1` consecutive table registers
//! (Vn, Vn+1, ... mod 32) form the lookup table; each byte of Vm indexes it.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    Insn::SimdTableLookup {
        q: field(word, 30, 1) == 1,
        is_tbx: field(word, 12, 1) == 1,
        len: field(word, 13, 2) as u8,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
