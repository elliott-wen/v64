//! AArch64 instruction decoder: pure `u32` instruction word -> typed [`Insn`].
//!
//! No state, no side effects. The top-level dispatch mirrors the encoding
//! groups selected by bits [28:25] in the reference QEMU decoder
//! (`disas_a64_insn` in `unicorn/qemu/target/arm/translate-a64.c`). Each
//! encoding group has a thin router module; each instruction *class* lives in
//! its own file.
//!
//! Anything not yet handled decodes to [`Insn::Unsupported`] so the executor
//! can surface a precise "not implemented" rather than silently misbehaving.

#![allow(clippy::unusual_byte_groupings)]

mod bitmask;
mod bits;
mod insn;

// Encoding-group routers.
mod branch;
mod dp_imm;
mod dp_reg;
mod fp;
mod ldst;
mod neon;

// Data processing -- immediate classes.
mod add_sub_imm;
mod bitfield;
mod extract;
mod logical_imm;
mod move_wide;
mod pc_rel;

// Data processing -- register classes.
mod add_sub_carry;
mod add_sub_ext_reg;
mod add_sub_shifted_reg;
mod cond_compare;
mod cond_select;
mod data_proc_1src;
mod data_proc_2src;
mod data_proc_3src;
mod logical_reg;

// Branch classes.
mod branch_imm;
mod branch_reg;
mod compare_branch;
mod cond_branch;
mod system;
mod test_branch;

// Load/store classes.
mod ldst_atomic;
mod ldst_excl;
mod ldst_literal;
mod ldst_pair;
mod ldst_post;
mod ldst_pre;
mod ldst_reg;
mod ldst_struct;
mod ldst_uimm;
mod ldst_unscaled;

// Advanced SIMD classes.
mod neon_aes;
mod neon_across;
mod neon_copy;
mod neon_ext;
mod neon_indexed;
mod neon_mod_imm;
mod neon_scalar;
mod neon_sha;
mod neon_shift_imm;
mod neon_tbl;
mod neon_three_diff;
mod neon_three_same;
mod neon_three_same_fp;
mod neon_two_reg_misc;
mod neon_zip_trn;

// Scalar floating-point classes.
mod fp_ccmp;
mod fp_compare;
mod fp_csel;
mod fp_cvt;
mod fp_dp1;
mod fp_dp2;
mod fp_dp3;
mod fp_imm;

pub use bitmask::decode_bit_masks;
pub use insn::{AddrMode, Insn, PairIndex, ShiftType};
pub use system::sysreg_key;

/// Decode one 32-bit little-endian instruction word.
#[must_use]
pub fn decode(word: u32) -> Insn {
    let op0 = bits::field(word, 25, 4);
    // Loads and stores occupy the `x1x0` group.
    if op0 & 0b0101 == 0b0100 {
        return ldst::decode(word);
    }
    match op0 {
        // Data processing -- immediate
        0b1000 | 0b1001 => dp_imm::decode(word),
        // Branches, exception generating, and system
        0b1010 | 0b1011 => branch::decode(word),
        // Data processing -- register
        0b0101 | 0b1101 => dp_reg::decode(word),
        // Scalar advanced SIMD (bit30=1) vs scalar floating-point (bit30=0).
        0b1111 => {
            if bits::field(word, 30, 1) == 1 {
                neon_scalar::decode(word)
            } else {
                fp::decode(word)
            }
        }
        // Advanced SIMD (vector)
        0b0111 => neon::decode(word),
        _ => Insn::Unsupported { word },
    }
}
