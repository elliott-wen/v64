//! Inline LSE atomics — read-modify-write ([`AtomicRmw`]) and compare-and-swap
//! ([`CompareSwap`]) — over the same TLB-checked fast path as [`super::memory`].
//!
//! The emulator is single-threaded, so "atomic" is just load-compute-store and
//! the acquire/release ordering bits are no-ops. Each op computes the base
//! address `[Rn]` (SP for r31), opens the fast path (with store permission, since
//! it writes), does the RMW/CAS directly in host memory, and bails the whole
//! instruction on any miss — exactly like a load/store, with no partial state
//! (the bail happens before any store). The exclusive-monitor forms (LDXR/STXR)
//! are *not* here: their monitor state stays with the interpreter.

use aarch64_cpu_state::regs::offsets;
use aarch64_decoder::Insn;
use wasm_encoder::{BlockType, Function, Instruction as I};

use super::common::*;
use super::memory::{close_fast_path, load_op, open_fast_path, store_op};

/// Inline an LSE [`AtomicRmw`] or [`CompareSwap`] with the TLB fast path. Returns
/// `false` (no emission) for forms it doesn't handle; the caller gates with
/// [`crate::is_inline_atomic`], so that path is unreachable in practice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_atomic(
    f: &mut Function,
    insn: &Insn,
    pc: u64,
    entry_pc: u64,
    insns_before: u64,
    ram_phys: u64,
    ram_size: u64,
    count_base: Option<u32>,
) -> bool {
    match *insn {
        Insn::AtomicRmw { size, op, rs, rn, rt } => {
            if size > 3 || op > 8 {
                return false;
            }
            let bytes = 1u64 << size;
            push_base(f, rn);
            open_fast_path(f, bytes, true, ram_phys, ram_size); // host addr -> ADDR
            // old = zero-extended load(ADDR) -> T1; s = X[Rs] & width-mask -> T2.
            emit!(f, I::LocalGet(ADDR), load_op(size, false), I::LocalSet(T1));
            push_masked_reg(f, rs, size);
            emit!(f, I::LocalSet(T2));
            // new -> T3, store its low `size` bytes, write old to Rt.
            rmw_new(f, op);
            emit!(f, I::LocalSet(T3));
            emit!(f, I::LocalGet(ADDR), I::LocalGet(T3), store_op(size));
            if rt != 31 {
                emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T1), I::I64Store(at(offsets::x(rt as usize))));
            }
            close_fast_path(f, pc, entry_pc, insns_before, count_base);
        }
        Insn::CompareSwap { size, rs, rn, rt } => {
            if size > 3 {
                return false;
            }
            let bytes = 1u64 << size;
            push_base(f, rn);
            open_fast_path(f, bytes, true, ram_phys, ram_size);
            // old -> T1; compare = X[Rs] & mask -> T2.
            emit!(f, I::LocalGet(ADDR), load_op(size, false), I::LocalSet(T1));
            push_masked_reg(f, rs, size);
            emit!(f, I::LocalSet(T2));
            // if old == compare { store Rt's low `size` bytes }.
            emit!(f, I::LocalGet(T1), I::LocalGet(T2), I::I64Eq, I::If(BlockType::Empty));
            emit!(f, I::LocalGet(ADDR));
            if rt == 31 {
                emit!(f, I::I64Const(0));
            } else {
                emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::x(rt as usize))));
            }
            emit!(f, store_op(size), I::End);
            // Rs always receives the old value (zero-extended).
            if rs != 31 {
                emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T1), I::I64Store(at(offsets::x(rs as usize))));
            }
            close_fast_path(f, pc, entry_pc, insns_before, count_base);
        }
        _ => return false,
    }
    true
}

/// Push the base address `[Rn]` (SP for r31) into [`T0`] for the fast path.
fn push_base(f: &mut Function, rn: u8) {
    let off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
    emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(off)), I::LocalSet(T0));
}

/// Push X[rs] (0 for r31 = XZR) masked to the access width — the RMW/compare
/// operand, zero-extended like the interpreter's `read_gpr(rs) & mask`.
fn push_masked_reg(f: &mut Function, rs: u8, size: u8) {
    if rs == 31 {
        emit!(f, I::I64Const(0));
        return;
    }
    emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::x(rs as usize))));
    if size < 3 {
        let mask = (1u64 << (8u32 << size)) - 1;
        emit!(f, I::I64Const(mask as i64), I::I64And);
    }
}

/// Emit the new value for RMW `op` from old ([`T1`]) and the operand ([`T2`]).
/// Signed/unsigned min/max use `select` on a comparison; the operands are
/// zero-extended for sub-word widths, so the signed compare matches the
/// interpreter (which compares the zero-extended values). `op == 8` is SWP.
fn rmw_new(f: &mut Function, op: u8) {
    match op {
        0 => emit!(f, I::LocalGet(T1), I::LocalGet(T2), I::I64Add), // LDADD
        1 => emit!(f, I::LocalGet(T1), I::LocalGet(T2), I::I64Const(-1), I::I64Xor, I::I64And), // LDCLR: old & !s
        2 => emit!(f, I::LocalGet(T1), I::LocalGet(T2), I::I64Xor), // LDEOR
        3 => emit!(f, I::LocalGet(T1), I::LocalGet(T2), I::I64Or),  // LDSET
        4 => sel(f, I::I64GeS), // LDSMAX: old >= s ? old : s
        5 => sel(f, I::I64LeS), // LDSMIN
        6 => sel(f, I::I64GeU), // LDUMAX
        7 => sel(f, I::I64LeU), // LDUMIN
        _ => emit!(f, I::LocalGet(T2)), // SWP
    }
}

/// `cmp(old, s) ? old : s` via `select` (old=[`T1`], s=[`T2`]).
fn sel(f: &mut Function, cmp: I<'static>) {
    emit!(f, I::LocalGet(T1), I::LocalGet(T2), I::LocalGet(T1), I::LocalGet(T2), cmp, I::Select);
}
