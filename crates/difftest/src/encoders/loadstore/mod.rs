//! Encoders for load/store classes.
//!
//! These produce [`MemEncoded`]: besides the instruction word, they seed the
//! base register to point into the DATA region (so the access is mapped) and
//! provide random initial DATA-region contents to compare after the run.
//!
//! Split by family: [`scalar`] (single register + pair, integer), [`atomic`]
//! (LSE atomics, CAS, acquire/release), and [`simd`] (vector register/pair and
//! the LD1-4/ST1-4 structure forms). The shared seeding helpers live here.

use crate::fuzz::MemClass;
use crate::rng::Rng;
use crate::{DATA_BASE, DATA_SIZE};

mod atomic;
mod scalar;
mod simd;

pub(super) fn classes() -> Vec<MemClass> {
    vec![
        MemClass { name: "ldst_uimm", encode: scalar::ldst_uimm },
        MemClass { name: "ldst_unscaled", encode: scalar::ldst_unscaled },
        MemClass { name: "ldst_post", encode: scalar::ldst_post },
        MemClass { name: "ldst_pre", encode: scalar::ldst_pre },
        MemClass { name: "ldst_reg", encode: scalar::ldst_reg },
        MemClass { name: "ldst_literal", encode: scalar::ldst_literal },
        MemClass { name: "ldst_pair", encode: scalar::ldst_pair },
        MemClass { name: "ldst_ordered", encode: atomic::ldst_ordered },
        MemClass { name: "ldst_atomic", encode: atomic::ldst_atomic },
        MemClass { name: "ldst_cas", encode: atomic::ldst_cas },
        MemClass { name: "ldst_vec_reg", encode: simd::ldst_vec_reg },
        MemClass { name: "ldst_vec_pair", encode: simd::ldst_vec_pair },
        MemClass { name: "ldst_struct_multi", encode: simd::ldst_struct_multi },
        MemClass { name: "ldst_struct_single", encode: simd::ldst_struct_single },
    ]
}

/// `DATA_SIZE` random bytes for the scratch region.
pub(super) fn random_data(rng: &mut Rng) -> Vec<u8> {
    let mut data = vec![0u8; DATA_SIZE];
    for chunk in data.chunks_mut(8) {
        let bytes = rng.next_u64().to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
    data
}

/// 8-aligned base near the region center, leaving `margin` bytes of slack on
/// each side for the offset.
pub(super) fn base_near_mid(rng: &mut Rng, margin: u32) -> u64 {
    let span = DATA_SIZE as u32 - 2 * margin - 8;
    let off = margin + (rng.below(span) & !7);
    DATA_BASE + u64::from(off)
}

/// A data register distinct from `rn` (avoids the UNPREDICTABLE writeback case).
pub(super) fn rt_distinct(rng: &mut Rng, rn: u32) -> u32 {
    loop {
        let rt = rng.below(31);
        if rt != rn {
            return rt;
        }
    }
}
