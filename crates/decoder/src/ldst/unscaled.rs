//! Load/store register, unscaled signed immediate (LDUR/STUR/LDURSB...), and —
//! when `unpriv` is set — the unprivileged forms LDTR/STTR/LDTRSB... which share
//! this layout but are permission-checked at EL0.

use crate::bits::{field, sfield};
use crate::insn::{AddrMode, Insn};
use crate::ldst;

pub(crate) fn decode(word: u32, unpriv: bool) -> Insn {
    let size = field(word, 30, 2) as u8;
    let opc = field(word, 22, 2);
    let (size, is_load, signed, dst64, vec) = if field(word, 26, 1) == 1 {
        let Some((is_load, log2)) = ldst::vec_kind(size, opc) else {
            return Insn::Unsupported { word };
        };
        (log2, is_load, false, false, true)
    } else if ldst::is_prefetch(size, opc) {
        return Insn::Prfm; // PRFUM (unscaled)
    } else {
        let Some((is_load, signed, dst64)) = ldst::kind(size, opc) else {
            return Insn::Unsupported { word };
        };
        (size, is_load, signed, dst64, false)
    };
    Insn::LoadStore {
        size,
        is_load,
        signed,
        dst64,
        vec,
        unpriv,
        rt: field(word, 0, 5) as u8,
        addr: AddrMode::Unscaled { rn: field(word, 5, 5) as u8, imm: sfield(word, 12, 9) },
    }
}
