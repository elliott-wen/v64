//! Per-class instruction encoders for the differential fuzzer.
//!
//! Each encoder fixes a class's discriminator bits and randomizes the operand
//! fields, producing a 32-bit word that should decode to that class. Encoders
//! are grouped by family; [`all_classes`] returns the full registry.

mod branch;
mod fp;
mod integer;
mod loadstore;
mod neon;

use crate::fuzz::{Class, FpClass, MemClass};
use crate::rng::Rng;

/// Random 5-bit register index (0..=31).
pub(crate) fn reg(rng: &mut Rng) -> u32 {
    rng.bits(5)
}

/// Random single bit as a u32.
pub(crate) fn bit(rng: &mut Rng) -> u32 {
    rng.bits(1)
}

/// Random contents for V0..V31 (exercises all bit patterns).
pub(crate) fn random_v(rng: &mut Rng) -> [u128; 32] {
    let mut v = [0u128; 32];
    for slot in &mut v {
        *slot = (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64());
    }
    v
}

/// Every fuzzable simple (word-only) instruction class.
#[must_use]
pub fn all_classes() -> Vec<Class> {
    let mut v = Vec::new();
    v.extend(integer::classes());
    v.extend(branch::classes());
    v
}

/// Every fuzzable memory instruction class.
#[must_use]
pub fn all_mem_classes() -> Vec<MemClass> {
    loadstore::classes()
}

/// Every fuzzable FP/SIMD instruction class.
#[must_use]
pub fn all_fp_classes() -> Vec<FpClass> {
    let mut v = fp::classes();
    v.extend(neon::classes());
    v
}
