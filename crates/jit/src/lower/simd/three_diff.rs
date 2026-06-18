//! SIMD three-different (widening). Only the pure widening multiply
//! SMULL/UMULL (and the `2` high-half variants) maps cleanly to WASM `extmul`;
//! the widening add/sub/accumulate, SQDMULL, ABDL, PMULL, and the
//! high-narrowing forms fall back.
//!
//! A widening op always produces a full 128-bit result (Q selects the source
//! half, not the result width), so there is no `!q` high-half masking.

use aarch64_cpu_state::regs::offsets;
use wasm_encoder::{Function, Instruction as I};

use super::push_v;
use crate::lower::common::at;

/// Whether [`simd_three_diff`] handles this form — only the widening multiply
/// SMULL/UMULL (opcode `1100`) at source size 8/16/32.
pub(super) fn can_lower(size: u8, opcode: u8) -> bool {
    opcode == 0b1100 && size <= 2
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simd_three_diff(f: &mut Function, q: bool, u: bool, size: u8, opcode: u8, rm: u8, rn: u8, rd: u8) -> bool {
    // opcode 0b1100 = SMULL/UMULL; source element size 0..2 (8/16/32 -> 16/32/64).
    if !can_lower(size, opcode) {
        return false;
    }
    emit!(f, I::LocalGet(0));
    push_v(f, rn);
    push_v(f, rm);
    emit!(f, extmul(q, u, size), I::V128Store(at(offsets::v(rd as usize))));
    true
}

/// The widening multiply for `(size, q-half, signedness)`. `q` picks the high
/// source half (SMULL2/UMULL2); `u` picks unsigned.
fn extmul(q: bool, u: bool, size: u8) -> I<'static> {
    use I::*;
    match (size, q, u) {
        (0, false, false) => I16x8ExtMulLowI8x16S,
        (0, true, false) => I16x8ExtMulHighI8x16S,
        (0, false, true) => I16x8ExtMulLowI8x16U,
        (0, true, true) => I16x8ExtMulHighI8x16U,
        (1, false, false) => I32x4ExtMulLowI16x8S,
        (1, true, false) => I32x4ExtMulHighI16x8S,
        (1, false, true) => I32x4ExtMulLowI16x8U,
        (1, true, true) => I32x4ExtMulHighI16x8U,
        (2, false, false) => I64x2ExtMulLowI32x4S,
        (2, true, false) => I64x2ExtMulHighI32x4S,
        (2, false, true) => I64x2ExtMulLowI32x4U,
        _ => I64x2ExtMulHighI32x4U, // (2, true, true)
    }
}
