//! Encoders for branch classes.
//!
//! Branch offsets are constrained so the target lands inside the mapped region
//! (code sits at the region's midpoint), otherwise Unicorn refuses to execute
//! the branch and every case would be skipped. `branch_reg` is *not* in the
//! generic sweep: its target comes from a register the fuzzer seeds randomly,
//! so it is covered by dedicated tests instead (see tests/oracle.rs).

use super::{bit, reg};
use crate::fuzz::Class;
use crate::rng::Rng;

pub(super) fn classes() -> Vec<Class> {
    vec![
        Class { name: "branch_imm", encode: branch_imm },
        Class { name: "cond_branch", encode: cond_branch },
        Class { name: "compare_branch", encode: compare_branch },
        Class { name: "test_branch", encode: test_branch },
    ]
}

/// A signed instruction-offset (in words) in `[-half, half]`, as raw field bits.
fn off_words(rng: &mut Rng, half: u32) -> u32 {
    let w = rng.below(2 * half + 1) as i32 - half as i32;
    w as u32
}

fn branch_imm(rng: &mut Rng) -> u32 {
    let imm26 = off_words(rng, 0x1_ffff) & 0x3ff_ffff;
    (bit(rng) << 31) | (0b00101 << 26) | imm26
}

fn cond_branch(rng: &mut Rng) -> u32 {
    let imm19 = off_words(rng, 0x1_ffff) & 0x7_ffff;
    (0b0101_0100 << 24) | (imm19 << 5) | rng.bits(4)
}

fn compare_branch(rng: &mut Rng) -> u32 {
    let imm19 = off_words(rng, 0x1_ffff) & 0x7_ffff;
    (bit(rng) << 31) | (0b011010 << 25) | (bit(rng) << 24) | (imm19 << 5) | reg(rng)
}

fn test_branch(rng: &mut Rng) -> u32 {
    let imm14 = off_words(rng, 0x1fff) & 0x3fff;
    (bit(rng) << 31)
        | (0b011011 << 25)
        | (bit(rng) << 24)
        | (rng.bits(5) << 19)
        | (imm14 << 5)
        | reg(rng)
}
