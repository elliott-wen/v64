//! Router for the "Loads and stores" encoding group.
//!
//! Integer single-register forms (unsigned-immediate, unscaled, pre/post-index,
//! register-offset) are implemented. SIMD/FP load/store (V=1), pairs, literals,
//! and exclusives decode to `Unsupported` until their classes land.

use crate::bits::field;
use crate::insn::Insn;
use crate::{
    ldst_atomic, ldst_excl, ldst_literal, ldst_pair, ldst_post, ldst_pre, ldst_reg, ldst_struct,
    ldst_uimm, ldst_unscaled,
};

pub(crate) fn decode(word: u32) -> Insn {
    // Load/store exclusive group: bits[29:24]=001000 (its own layout, no V bit).
    if field(word, 24, 6) == 0b001000 {
        return ldst_excl::decode(word);
    }
    // Advanced SIMD load/store structures (LD1-4/ST1-4): bits[29:26]=0011,
    // bit25=0. These have their own multi-register layout.
    if field(word, 26, 4) == 0b0011 && field(word, 25, 1) == 0 {
        return ldst_struct::decode(word);
    }
    match field(word, 27, 3) {
        0b111 => register_form(word),
        // Load register (literal): bits[25:24]=00.
        0b011 if field(word, 24, 2) == 0b00 => ldst_literal::decode(word),
        // Load/store pair.
        0b101 => ldst_pair::decode(word),
        _ => Insn::Unsupported { word },
    }
}

/// Load/store register (the bits[29:27]=111 single-register addressing classes).
fn register_form(word: u32) -> Insn {
    match field(word, 24, 2) {
        0b01 => ldst_uimm::decode(word),
        0b00 => {
            let indexed = field(word, 21, 1) == 1;
            match (indexed, field(word, 10, 2)) {
                (true, 0b10) => ldst_reg::decode(word),
                (true, 0b00) => ldst_atomic::decode(word), // LSE atomic RMW / SWP
                (false, 0b00) => ldst_unscaled::decode(word),
                (false, 0b01) => ldst_post::decode(word),
                (false, 0b11) => ldst_pre::decode(word),
                // (false, 0b10) is the unprivileged LDTR/STTR — not implemented.
                _ => Insn::Unsupported { word },
            }
        }
        _ => Insn::Unsupported { word },
    }
}

/// Map `(size, opc)` to `(is_load, signed, dst64)`, or `None` for the reserved
/// / prefetch encodings we don't implement. Shared by all integer load/store
/// addressing classes.
pub(crate) fn kind(size: u8, opc: u32) -> Option<(bool, bool, bool)> {
    match (size, opc) {
        (_, 0b00) => Some((false, false, size == 3)),
        (_, 0b01) => Some((true, false, size == 3)),
        // PRFM (size 3 opc 2) and the reserved word/dword signed forms.
        (3, 0b10) | (3, 0b11) | (2, 0b11) => None,
        (_, 0b10) => Some((true, true, true)),  // LDRS* to 64-bit
        (_, 0b11) => Some((true, true, false)), // LDRS* to 32-bit
        _ => None,
    }
}

/// SIMD/FP register load/store (V=1). Returns `(is_load, log2_bytes)` where
/// log2 is 0..4 (B,H,S,D,Q). `opc[1]` (bit 23) widens to 128-bit Q.
pub(crate) fn vec_kind(size: u8, opc: u32) -> Option<(bool, u8)> {
    let is_load = opc & 1 == 1;
    let log2 = if opc & 0b10 != 0 {
        if size == 0 {
            4 // Q (128-bit)
        } else {
            return None; // unallocated
        }
    } else {
        size
    };
    Some((is_load, log2))
}
