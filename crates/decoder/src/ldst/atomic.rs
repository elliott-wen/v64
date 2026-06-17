//! LSE atomic read-modify-write: LDADD/LDCLR/LDEOR/LDSET/LDSMAX/LDSMIN/
//! LDUMAX/LDUMIN and SWP (plus their acquire/release ordering and ST* aliases,
//! which are the same encoding with Rt=31 and don't affect a sequential model).
//! Also LDAPR (load-acquire RCpc), which in a sequential model is a plain load.

use crate::bits::field;
use crate::insn::{AddrMode, Insn};

pub(crate) fn decode(word: u32) -> Insn {
    let o3 = field(word, 15, 1);
    let opc = field(word, 12, 3);
    let size = field(word, 30, 2) as u8;

    // LDAPR (load-acquire RCpc): A=1, R=0, o3=1, opc=100, Rs=11111. Acquire
    // ordering is a no-op in a single-threaded model, so decode it as a plain
    // zero-offset load.
    if o3 == 1 && opc == 0b100 && field(word, 23, 1) == 1 && field(word, 22, 1) == 0
        && field(word, 16, 5) == 0b11111
    {
        return Insn::LoadStore {
            size,
            is_load: true,
            signed: false,
            dst64: size == 3,
            vec: false,
            unpriv: false,
            rt: field(word, 0, 5) as u8,
            addr: AddrMode::UnsignedImm { rn: field(word, 5, 5) as u8, imm: 0 },
        };
    }

    // o3=0: the eight RMW ops (opc 0..7). o3=1, opc=0: SWP.
    let op = match (o3, opc) {
        (0, n) => n as u8,
        (1, 0) => 8, // SWP
        _ => return Insn::Unsupported { word },
    };
    Insn::AtomicRmw {
        size,
        op,
        rs: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rt: field(word, 0, 5) as u8,
    }
}
