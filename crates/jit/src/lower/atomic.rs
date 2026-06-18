//! Inline atomics over the same TLB-checked fast path as [`super::memory`]: LSE
//! read-modify-write ([`AtomicRmw`]) and compare-and-swap ([`CompareSwap`]), plus
//! the LL/SC exclusives ([`LoadExclusive`]/[`StoreExclusive`]).
//!
//! The emulator is single-threaded, so "atomic" is just load-compute-store and
//! the acquire/release bits are no-ops. Each op computes the base address `[Rn]`
//! (SP for r31), checks natural alignment (atomics/exclusives fault if unaligned,
//! so a misaligned address bails to let the interpreter raise the abort), opens
//! the fast path, and works directly in host memory — bailing the whole
//! instruction on any miss, with no partial state (the bail precedes any store or
//! monitor change). LDXR/STXR arm and check the exclusive monitor in `CpuState`
//! (the same fields the interpreter uses, so they stay in lockstep); any event
//! that clears the monitor — exception, context switch — runs through the
//! interpreter, which the JIT bails to.

use aarch64_cpu_state::regs::offsets;
use aarch64_cpu_state::{EXCL_ADDR_OFFSET, EXCL_VALID_OFFSET, EXCL_VAL_OFFSET};
use aarch64_decoder::Insn;
use wasm_encoder::{BlockType, Function, Instruction as I};

use super::common::*;
use super::memory::{close_fast_path, emit_bail, load_op, open_fast_path, store_op};

/// Inline an atomic (LSE RMW/CAS or LL/SC exclusive) with the TLB fast path.
/// Returns `false` (no emission) for forms it doesn't handle; the caller gates
/// with [`crate::is_inline_atomic`], so that path is unreachable in practice.
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
            align_guard(f, bytes, pc, entry_pc, insns_before, count_base);
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
            align_guard(f, bytes, pc, entry_pc, insns_before, count_base);
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
        Insn::LoadExclusive { size, rt, rn } => {
            if size > 3 {
                return false;
            }
            let bytes = 1u64 << size;
            push_base(f, rn);
            align_guard(f, bytes, pc, entry_pc, insns_before, count_base);
            open_fast_path(f, bytes, false, ram_phys, ram_size); // load: read perm
            // val = zero-extended load(ADDR) -> T1.
            emit!(f, I::LocalGet(ADDR), load_op(size, false), I::LocalSet(T1));
            // Arm the monitor: excl_valid=1, excl_addr=VA (T0), excl_val=val (T1).
            emit!(f, I::LocalGet(REGS_BASE), I::I32Const(1), I::I32Store8(at(EXCL_VALID_OFFSET)));
            emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0), I::I64Store(at(EXCL_ADDR_OFFSET)));
            emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T1), I::I64Store(at(EXCL_VAL_OFFSET)));
            if rt != 31 {
                emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T1), I::I64Store(at(offsets::x(rt as usize))));
            }
            close_fast_path(f, pc, entry_pc, insns_before, count_base);
        }
        Insn::StoreExclusive { size, rs, rt, rn } => {
            if size > 3 {
                return false;
            }
            let bytes = 1u64 << size;
            push_base(f, rn);
            align_guard(f, bytes, pc, entry_pc, insns_before, count_base);
            open_fast_path(f, bytes, true, ram_phys, ram_size); // store: write perm
            // success = excl_valid & (excl_addr == VA) & (mem[ADDR] == excl_val),
            // held (zero-extended) in the i64 local T2.
            emit!(f, I::LocalGet(REGS_BASE), I::I32Load8U(at(EXCL_VALID_OFFSET)));
            emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(EXCL_ADDR_OFFSET)), I::LocalGet(T0), I::I64Eq, I::I32And);
            emit!(f, I::LocalGet(ADDR), load_op(size, false), I::LocalGet(REGS_BASE), I::I64Load(at(EXCL_VAL_OFFSET)), I::I64Eq, I::I32And);
            emit!(f, I::I64ExtendI32U, I::LocalSet(T2));
            // if success { store Rt's low `size` bytes }.
            emit!(f, I::LocalGet(T2), I::I32WrapI64, I::If(BlockType::Empty));
            emit!(f, I::LocalGet(ADDR));
            if rt == 31 {
                emit!(f, I::I64Const(0));
            } else {
                emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::x(rt as usize))));
            }
            emit!(f, store_op(size), I::End);
            // Clear the monitor unconditionally; Ws = !success (0 = success).
            emit!(f, I::LocalGet(REGS_BASE), I::I32Const(0), I::I32Store8(at(EXCL_VALID_OFFSET)));
            if rs != 31 {
                emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T2), I::I64Eqz, I::I64ExtendI32U, I::I64Store(at(offsets::x(rs as usize))));
            }
            close_fast_path(f, pc, entry_pc, insns_before, count_base);
        }
        _ => return false,
    }
    true
}

/// Bail (raising an Alignment Data Abort in the interpreter) if the address in
/// [`T0`] isn't naturally aligned to `bytes` — atomics and exclusives require it.
fn align_guard(f: &mut Function, bytes: u64, pc: u64, entry_pc: u64, insns_before: u64, count_base: Option<u32>) {
    if bytes > 1 {
        emit!(f, I::LocalGet(T0), I::I64Const((bytes - 1) as i64), I::I64And, I::I32WrapI64, I::If(BlockType::Empty));
        emit_bail(f, pc, entry_pc, insns_before, count_base);
        emit!(f, I::End);
    }
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
