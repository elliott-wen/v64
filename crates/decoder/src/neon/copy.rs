//! Advanced SIMD copy group: DUP (general/element), INS (general/element),
//! SMOV/UMOV. ZIP/UZP/TRN/EXT/TBL are separate groups (not here).

use crate::bits::field;
use crate::insn::Insn;

/// Decode (size, index) from imm5: the lowest set bit gives the element size,
/// the higher bits the element index.
fn size_index(imm5: u32) -> Option<(u8, u8)> {
    if imm5 & 1 != 0 {
        Some((0, (imm5 >> 1) as u8))
    } else if imm5 & 2 != 0 {
        Some((1, (imm5 >> 2) as u8))
    } else if imm5 & 4 != 0 {
        Some((2, (imm5 >> 3) as u8))
    } else if imm5 & 8 != 0 {
        Some((3, (imm5 >> 4) as u8))
    } else {
        None
    }
}

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let op = field(word, 29, 1);
    let imm5 = field(word, 16, 5);
    let imm4 = field(word, 11, 4);
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;
    let unsup = Insn::Unsupported { word };

    // INS (element): op=1, requires Q=1; imm4 holds the source index.
    if op == 1 {
        if !q {
            return unsup;
        }
        let Some((size, dst)) = size_index(imm5) else { return unsup };
        return Insn::SimdInsElement { size, dst, src: (imm4 >> size) as u8, rn, rd };
    }

    match imm4 {
        0b0000 => match size_index(imm5) {
            Some((3, _)) if !q => unsup, // DUP 1D reserved
            Some((size, index)) => Insn::SimdDupElement { q, size, index, rn, rd },
            None => unsup,
        },
        0b0001 => dup_general(word, q, imm5, rn, rd),
        0b0011 if q => match size_index(imm5) {
            Some((size, index)) => Insn::SimdInsGeneral { size, index, rn, rd },
            None => unsup,
        },
        0b0101 => mov_to_gpr(true, q, imm5, rn, rd, word),  // SMOV
        0b0111 => mov_to_gpr(false, q, imm5, rn, rd, word), // UMOV
        _ => unsup,
    }
}

fn dup_general(word: u32, q: bool, imm5: u32, rn: u8, rd: u8) -> Insn {
    let Some((size, _)) = size_index(imm5) else {
        return Insn::Unsupported { word };
    };
    if size == 3 && !q {
        return Insn::Unsupported { word }; // D needs Q=1
    }
    Insn::SimdDupGeneral { q, size, rn, rd }
}

fn mov_to_gpr(signed: bool, q: bool, imm5: u32, vn: u8, rd: u8, word: u32) -> Insn {
    let Some((size, index)) = size_index(imm5) else {
        return Insn::Unsupported { word };
    };
    // SMOV: Wd takes B/H, Xd takes B/H/S. UMOV: Wd takes B/H/S, Xd takes D.
    let valid = if signed {
        if q { size <= 2 } else { size <= 1 }
    } else if q {
        size == 3
    } else {
        size <= 2
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdMovToGpr { signed, dst64: q, size, index, vn, rd }
}
