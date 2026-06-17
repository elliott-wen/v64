//! Load/store ordered + exclusive group.
//!
//! Currently only the non-exclusive acquire/release single-register accesses
//! (LDAR/STLR and their byte/half/word forms) are implemented. The true
//! exclusives (LDXR/STXR/LDAXR/STLXR) and LSE atomics (CAS/SWP/LDADD) need an
//! exclusive-monitor model and paired-instruction testing — see
//! `DESIGN_atomics.md`.

use crate::bits::field;
use crate::insn::{AddrMode, Insn};

pub(crate) fn decode(word: u32) -> Insn {
    let o2 = field(word, 23, 1);
    let l = field(word, 22, 1);
    let o1 = field(word, 21, 1);
    let o0 = field(word, 15, 1);
    let rs = field(word, 16, 5);
    let rt2 = field(word, 10, 5);

    // CAS: o2=1, o1=1, Rt2=11111 (L/o0 are the acquire/release variants).
    if o2 == 1 && o1 == 1 && rt2 == 0b11111 {
        return Insn::CompareSwap {
            size: field(word, 30, 2) as u8,
            rs: rs as u8,
            rn: field(word, 5, 5) as u8,
            rt: field(word, 0, 5) as u8,
        };
    }

    // LDAR/STLR: ordered, non-exclusive (o2=1, o1=0, o0=1), Rs and Rt2 unused.
    if o2 == 1 && o1 == 0 && o0 == 1 && rs == 0b11111 && rt2 == 0b11111 {
        let size = field(word, 30, 2) as u8;
        return Insn::LoadStore {
            size,
            is_load: l == 1, // LDAR vs STLR
            signed: false,
            dst64: size == 3,
            vec: false,
            rt: field(word, 0, 5) as u8,
            addr: AddrMode::UnsignedImm { rn: field(word, 5, 5) as u8, imm: 0 },
        };
    }

    // LDXR/LDAXR and STXR/STLXR: single-register exclusives (o2=0, o1=0,
    // Rt2=11111). The pair forms (LDXP/STXP) have Rt2 != 11111 and are not
    // implemented.
    if o2 == 0 && o1 == 0 && rt2 == 0b11111 {
        let size = field(word, 30, 2) as u8;
        let rt = field(word, 0, 5) as u8;
        let rn = field(word, 5, 5) as u8;
        return if l == 1 {
            Insn::LoadExclusive { size, rt, rn }
        } else {
            Insn::StoreExclusive { size, rs: rs as u8, rt, rn }
        };
    }
    Insn::Unsupported { word }
}
