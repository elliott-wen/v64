//! Shared building blocks for the inline lowerings: scratch-local conventions,
//! NZCV bit layout, and register-image read/write helpers.
//!
//! Register access is a load/store of the flat image at `offsets::*`; operand
//! width (`sf`) and SP-vs-ZR for r31 follow `CpuState`'s `read`/`write` rules.

use aarch64_cpu_state::regs::offsets;
use aarch64_cpu_state::JIT_COUNT_OFFSET;
use wasm_encoder::{Function, Instruction as I, MemArg};

// The block function takes two i32 params: `regs_base` (the live `CpuState`) and
// `ram_base` (the host base of guest RAM, for the inline memory fast path).
pub(super) const REGS_BASE: u32 = 0;
pub(super) const RAM_BASE: u32 = 1;
/// First scratch-local index (just past the two parameters).
const SCRATCH0: u32 = 2;

// Scratch i64 locals. Each lowering fully consumes them within itself, so they
// can be reused across instructions.
pub(super) const T0: u32 = SCRATCH0;
pub(super) const T1: u32 = SCRATCH0 + 1;
pub(super) const T2: u32 = SCRATCH0 + 2;
pub(super) const T3: u32 = SCRATCH0 + 3;
pub(super) const T4: u32 = SCRATCH0 + 4;
/// Holds the block's runtime entry PC (the image PC at entry), set by
/// [`load_entry_pc`]. PC-derived values are computed as `PC0 + delta`
/// ([`gen_rel_pc`]) so the block is position-independent: the same physical block
/// runs correctly at any virtual address it's mapped to.
pub(super) const PC0: u32 = SCRATCH0 + 5;
/// Region-compilation state (unused by single-block functions, which just leave
/// these declared-but-untouched). `RPC` = the current guest PC the dispatch loop
/// is at; `RCOUNT` = instructions retired so far this call (for `jit_count`).
pub(super) const RPC: u32 = SCRATCH0 + 6;
pub(super) const RCOUNT: u32 = SCRATCH0 + 7;
/// Number of scratch i64 locals the block function declares.
pub(crate) const SCRATCH_I64: u32 = 8;
/// Number of scratch i32 locals: the computed linear address [`ADDR`], the
/// region safety counter [`RITERS`], and the region dispatch index [`RIDX`].
pub(crate) const SCRATCH_I32: u32 = 3;
/// The i32 scratch local holding a computed linear memory address. First i32
/// local, following the i64 scratch locals.
pub(super) const ADDR: u32 = SCRATCH0 + SCRATCH_I64;
/// The i32 region dispatch-loop safety counter (caps iterations before yielding
/// to the organizer to service timers/IRQs — like v86's `LOOP_COUNTER`).
pub(super) const RITERS: u32 = SCRATCH0 + SCRATCH_I64 + 1;
/// The i32 region dispatch index — which basic block to run next. A `br_table`
/// jumps to it in O(1) (the entry block is index 0 = the zero-initialised value).
pub(super) const RIDX: u32 = SCRATCH0 + SCRATCH_I64 + 2;

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

/// Store a constant instruction count into the block's `jit_count` slot, so the
/// organizer learns how many instructions the call retired. Used by
/// straight-line blocks (each call executes the block exactly once).
pub(super) fn store_count_const(f: &mut Function, count: u64) {
    emit!(f, I::LocalGet(0), I::I64Const(count as i64), I::I64Store(at(JIT_COUNT_OFFSET)));
}

/// Load the runtime entry PC (image PC at block entry) into [`PC0`]. Emitted once
/// at the top of every block; the base for all position-independent PC math.
pub(super) fn load_entry_pc(f: &mut Function) {
    emit!(f, I::LocalGet(0), I::I64Load(at(offsets::PC)), I::LocalSet(PC0));
}

/// Push a position-independent guest address: `PC0 + (abs - entry_pc)`. `abs` is
/// the compile-time virtual address; the runtime entry PC ([`PC0`]) supplies the
/// actual base, so the same physical block is correct at whatever VA it runs at.
pub(super) fn gen_rel_pc(f: &mut Function, abs: u64, entry_pc: u64) {
    emit!(
        f,
        I::LocalGet(PC0),
        I::I64Const(abs.wrapping_sub(entry_pc) as i64),
        I::I64Add
    );
}
