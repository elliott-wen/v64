//! ADR / ADRP — PC-relative address computation.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let page = field(word, 31, 1) == 1; // op: 0=ADR, 1=ADRP
    let immlo = field(word, 29, 2) as i64;
    let immhi = field(word, 5, 19) as i64;
    // 21-bit signed immediate = immhi:immlo, sign-extended.
    let imm21 = (immhi << 2) | immlo;
    let raw = (imm21 << 43) >> 43;
    // ADRP scales by the 4 KiB page size.
    let imm = if page { raw << 12 } else { raw };
    Insn::PcRel {
        page,
        imm,
        rd: field(word, 0, 5) as u8,
    }
}
