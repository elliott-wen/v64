//! Advanced SIMD / NEON. This module holds the shared lane helpers (below); the
//! per-encoding-class executors live in the `simd/` subdirectory.

pub(crate) mod across;
pub(crate) mod aes;
pub(crate) mod copy;
pub(crate) mod dup;
pub(crate) mod ext;
pub(crate) mod indexed;
pub(crate) mod ldst_struct;
pub(crate) mod mod_imm;
pub(crate) mod permute;
pub(crate) mod scalar;
pub(crate) mod sha;
pub(crate) mod shift_fp;
pub(crate) mod shift_imm;
pub(crate) mod shift_long;
pub(crate) mod shift_narrow;
pub(crate) mod tbl;
pub(crate) mod three_diff;
pub(crate) mod three_same;
pub(crate) mod three_same_extra;
pub(crate) mod three_same_fp;
pub(crate) mod two_reg_long;
pub(crate) mod two_reg_misc;
pub(crate) mod two_reg_misc_fp;
pub(crate) mod two_reg_narrow;

/// Split the low (Q ? 128 : 64) bits of `val` into `8<<size`-bit lanes,
/// each zero-extended into a u64.
pub(crate) fn split(val: u128, size: u8, q: bool) -> Vec<u64> {
    let ebits = 8usize << size;
    let total = if q { 128 } else { 64 };
    let n = total / ebits;
    (0..n)
        .map(|i| {
            let raw = (val >> (i * ebits)) as u64;
            if ebits >= 64 {
                raw
            } else {
                raw & ((1u64 << ebits) - 1)
            }
        })
        .collect()
}

/// Reassemble `8<<size`-bit lanes into a u128 (high bits left zero, which gives
/// the Q=0 upper-half-zeroing for free).
pub(crate) fn join(lanes: &[u64], size: u8) -> u128 {
    let ebits = 8usize << size;
    let mut v = 0u128;
    for (i, &l) in lanes.iter().enumerate() {
        let masked = if ebits >= 64 {
            u128::from(l)
        } else {
            u128::from(l) & ((1u128 << ebits) - 1)
        };
        v |= masked << (i * ebits);
    }
    v
}
