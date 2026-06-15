//! Shared building blocks for the inline lowerings: scratch-local conventions,
//! NZCV bit layout, and register-image read/write helpers.
//!
//! Register access is a load/store of the flat image at `offsets::*`; operand
//! width (`sf`) and SP-vs-ZR for r31 follow `CpuState`'s `read`/`write` rules.

use aarch64_cpu_state::regs::offsets;
use wasm_encoder::{Function, Instruction as I, MemArg};

// Scratch i64 locals (local 0 is the i32 `regs_base` parameter). Each lowering
// fully consumes them within itself, so they can be reused across instructions.
pub(super) const T0: u32 = 1;
pub(super) const T1: u32 = 2;
pub(super) const T2: u32 = 3;
pub(super) const T3: u32 = 4;
pub(super) const T4: u32 = 5;
/// Number of scratch i64 locals the block function must declare (indices 1..=5;
/// local 0 is the i32 `regs_base` parameter).
pub(crate) const SCRATCH_I64: u32 = 5;
/// Number of scratch i32 locals (one: the computed linear address, [`ADDR`]).
pub(crate) const SCRATCH_I32: u32 = 1;
/// The i32 scratch local holding a computed linear memory address. Index follows
/// the i64 scratch locals (param + 5 i64 -> index 6).
pub(super) const ADDR: u32 = SCRATCH_I64 + 1;

// NZCV bit positions in the packed word.
pub(super) const N_BIT: i64 = 31;
pub(super) const Z_BIT: i64 = 30;
pub(super) const C_BIT: i64 = 29;
pub(super) const V_BIT: i64 = 28;

pub(super) const W_MASK: i64 = 0xffff_ffff;

/// A byte-aligned `MemArg` on memory 0 with `offset` folded in.
pub(super) fn at(offset: usize) -> MemArg {
    MemArg { offset: offset as u64, align: 0, memory_index: 0 }
}

/// Destination offset for writing register `idx`, or `None` if it is the zero
/// register (write discarded). `sp_ctx` selects SP vs ZR for r31.
pub(super) fn dest_off(idx: u8, sp_ctx: bool) -> Option<usize> {
    match idx {
        31 if sp_ctx => Some(offsets::SP),
        31 => None,
        n => Some(offsets::x(n as usize)),
    }
}

/// Push register `idx`'s value at width `sf`; r31 is SP (`sp_ctx`) or ZR (0).
pub(super) fn push_operand(f: &mut Function, idx: u8, sf: bool, sp_ctx: bool) {
    if idx == 31 && !sp_ctx {
        emit!(f, I::I64Const(0)); // XZR
        return;
    }
    let off = if idx == 31 { offsets::SP } else { offsets::x(idx as usize) };
    emit!(f, I::LocalGet(0), I::I64Load(at(off)));
    if !sf {
        emit!(f, I::I64Const(W_MASK), I::I64And); // W view: low 32, zero-extended
    }
}

/// Mask the top-of-stack i64 to 32 bits when operating on W registers.
pub(super) fn mask_w(f: &mut Function, sf: bool) {
    if !sf {
        emit!(f, I::I64Const(W_MASK), I::I64And);
    }
}

/// Store the i64 in local `tmp` into register `rd` (W results are zero-extended).
pub(super) fn store_local(f: &mut Function, rd: u8, sf: bool, sp: bool, tmp: u32) {
    if let Some(off) = dest_off(rd, sp) {
        emit!(f, I::LocalGet(0), I::LocalGet(tmp));
        mask_w(f, sf);
        emit!(f, I::I64Store(at(off)));
    }
}

/// Store `pc + 4` into the image PC.
pub(super) fn advance_pc(f: &mut Function, pc: u64) {
    emit!(f, I::LocalGet(0), I::I64Const(pc.wrapping_add(4) as i64), I::I64Store(at(offsets::PC)));
}
