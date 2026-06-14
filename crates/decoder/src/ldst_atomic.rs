//! LSE atomic read-modify-write: LDADD/LDCLR/LDEOR/LDSET/LDSMAX/LDSMIN/
//! LDUMAX/LDUMIN and SWP (plus their acquire/release ordering and ST* aliases,
//! which are the same encoding with Rt=31 and don't affect a sequential model).

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let o3 = field(word, 15, 1);
    let opc = field(word, 12, 3);
    let size = field(word, 30, 2) as u8;

    // o3=0: the eight RMW ops (opc 0..7). o3=1, opc=0: SWP.
    let op = match (o3, opc) {
        (0, n) => n as u8,
        (1, 0) => 8, // SWP
        _ => return Insn::Unsupported { word }, // LDAPR and friends not handled
    };
    Insn::AtomicRmw {
        size,
        op,
        rs: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rt: field(word, 0, 5) as u8,
    }
}
