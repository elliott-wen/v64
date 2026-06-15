//! Advanced SIMD lowering. Maps the bit-exact integer subset of NEON to WASM
//! `v128` lane ops; anything WASM can't express exactly (saturating, halving,
//! pairwise, FP, polynomial, the i64 ops WASM lacks) falls back to
//! `interpret_one`.
//!
//! The guest V registers are 128-bit, stored little-endian at `offsets::v(n)`,
//! which matches WASM's `v128` lane order, so lane ops map directly. For the
//! 64-bit (`!q`) forms the high half of the result is zeroed.
//!
//! Split by instruction family: [`three_same`] (element-wise integer ops),
//! [`copy`] (modified-immediate, DUP/INS/UMOV/SMOV), [`permute`] (ZIP/UZP/TRN,
//! EXT, TBL), and [`shift`] (shift-by-immediate). The shared `v128` load/store
//! helpers live here.

use aarch64_cpu_state::regs::offsets;
use wasm_encoder::{Function, Instruction as I};

use crate::lower::common::at;

mod copy;
mod permute;
mod shift;
mod three_same;
mod two_reg_misc;

pub(super) use copy::{
    simd_dup_element, simd_dup_general, simd_ins_element, simd_ins_general, simd_mod_imm,
    simd_mov_to_gpr,
};
pub(super) use permute::{simd_ext, simd_tbl, simd_zip_trn};
pub(super) use shift::simd_shift_imm;
pub(super) use three_same::simd_three_same;
pub(super) use two_reg_misc::simd_two_reg_misc;

/// Load V[r] as a `v128` onto the stack.
fn push_v(f: &mut Function, r: u8) {
    emit!(f, I::LocalGet(0), I::V128Load(at(offsets::v(r as usize))));
}

/// Store the `v128` on top of the stack (with `regs_base` beneath it) into V[rd],
/// zeroing the high 64 bits for the 64-bit (`!q`) form.
fn finish_v(f: &mut Function, q: bool, rd: u8) {
    if !q {
        emit!(f, I::V128Const(u64::MAX as i128), I::V128And); // keep low 64 bits only
    }
    emit!(f, I::V128Store(at(offsets::v(rd as usize))));
}

/// Emit `Vd = shuffle(Vn, Vm)` with a constant 16-byte lane pattern, masking the
/// high half for the 64-bit (`!q`) form. (A self-shuffle uses `rn == rm`.)
fn shuffle2(f: &mut Function, q: bool, rn: u8, rm: u8, rd: u8, lanes: [u8; 16]) {
    emit!(f, I::LocalGet(0)); // base for the store
    push_v(f, rn); // operand a (lanes 0..16)
    push_v(f, rm); // operand b (lanes 16..32)
    emit!(f, I::I8x16Shuffle(lanes));
    finish_v(f, q, rd);
}
