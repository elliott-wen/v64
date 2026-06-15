//! Encoders for Advanced SIMD classes. They reuse the FP fuzz harness (random
//! V0..V31, compared after the run).
//!
//! Split by sub-family: [`vector`] (the main element-wise / data-movement ops),
//! [`scalar`] (the scalar-SIMD `1110_11110` forms), and [`crypto`] (AES/SHA).
//! The shared `FpEncoded` builders live here.

use crate::encoders::random_v;
use crate::fuzz::FpClass;
use crate::rng::Rng;
use crate::FpEncoded;

mod crypto;
mod scalar;
mod vector;

/// FPCR with DN=1 (default NaN) for deterministic FP results.
pub(super) const FPCR_DN: u64 = 1 << 25;

/// Encode a word with random V state and the default FPCR.
pub(super) fn enc(word: u32, rng: &mut Rng) -> FpEncoded {
    FpEncoded { word, init_v: random_v(rng), gpr_seeds: vec![], fpcr: 0 }
}

/// Encode a word with random V state and `FPCR.DN` set (deterministic NaNs).
pub(super) fn fp_enc(word: u32, rng: &mut Rng) -> FpEncoded {
    FpEncoded { word, init_v: random_v(rng), gpr_seeds: vec![], fpcr: FPCR_DN }
}

pub(super) fn classes() -> Vec<FpClass> {
    vec![
        FpClass { name: "neon_three_same", encode: vector::three_same },
        FpClass { name: "neon_three_diff", encode: vector::three_diff },
        FpClass { name: "neon_indexed", encode: vector::indexed },
        FpClass { name: "neon_three_same_fp", encode: vector::three_same_fp },
        FpClass { name: "neon_two_reg_misc", encode: vector::two_reg_misc },
        FpClass { name: "neon_two_reg_misc_fp", encode: vector::two_reg_misc_fp },
        FpClass { name: "neon_mod_imm", encode: vector::mod_imm },
        FpClass { name: "neon_dup", encode: vector::dup_general },
        FpClass { name: "neon_dup_element", encode: vector::dup_element },
        FpClass { name: "neon_ins", encode: vector::ins },
        FpClass { name: "neon_mov_gpr", encode: vector::mov_gpr },
        FpClass { name: "neon_zip_trn", encode: vector::zip_trn },
        FpClass { name: "neon_ext", encode: vector::ext },
        FpClass { name: "neon_tbl", encode: vector::tbl },
        FpClass { name: "neon_aes", encode: crypto::aes },
        FpClass { name: "neon_sha3", encode: crypto::sha3 },
        FpClass { name: "neon_sha2", encode: crypto::sha2 },
        FpClass { name: "neon_three_same_extra", encode: vector::three_same_extra },
        FpClass { name: "neon_shift_imm", encode: vector::shift_imm },
        FpClass { name: "neon_across", encode: vector::across },
        FpClass { name: "neon_scalar_three_same", encode: scalar::scalar_three_same },
        FpClass { name: "neon_scalar_two_reg_misc", encode: scalar::scalar_two_reg_misc },
        FpClass { name: "neon_scalar_pairwise", encode: scalar::scalar_pairwise },
        FpClass { name: "neon_scalar_three_diff", encode: scalar::scalar_three_diff },
        FpClass { name: "neon_scalar_copy", encode: scalar::scalar_copy },
        FpClass { name: "neon_scalar_indexed", encode: scalar::scalar_indexed },
        FpClass { name: "neon_scalar_shift", encode: scalar::scalar_shift },
    ]
}
