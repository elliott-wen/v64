//! Inline loads/stores: single register (every addressing mode) and integer
//! pairs, against the shared linear memory. Assumes identity mapping (MMU off —
//! the runtime's only mode today); SIMD/FP, 128-bit, and odd extends fall back.
//!
//! A guest address `a` maps to linear offset `RAM_BASE + (a - guest_base)`. That
//! displacement (`RAM_BASE - guest_base`) is a constant folded into the address
//! arithmetic at emit time. Out-of-bounds accesses trap in WASM and surface as a
//! guest fault (see the runtime), matching the interpreter's failure.

use aarch64_cpu_state::regs::offsets;
use aarch64_decoder::{AddrMode, Insn, PairIndex};
use wasm_encoder::{Function, Instruction as I};

use super::arith::push_ext;
use super::common::*;
use crate::abi;

/// LDR/STR, single register (integer or SIMD/FP), every addressing mode. The
/// 128-bit-and-narrower vector forms are handled; structure loads fall back.
pub(super) fn load_store(f: &mut Function, insn: &Insn, pc: u64, guest_base: u64) -> bool {
    let Insn::LoadStore { size, is_load, signed, dst64, vec, rt, addr } = *insn else {
        return false;
    };
    // Integer access widths are log2 0..=3; vector adds size 4 (the 128-bit Q).
    if size > 4 || (!vec && size > 3) {
        return false;
    }

    // Compute the linear address into the ADDR local, plus any base writeback.
    let Some(writeback) = emit_ea(f, addr, pc, guest_base) else {
        return false; // unsupported addressing form
    };

    match (vec, is_load) {
        (false, true) => int_load(f, size, signed, dst64, rt),
        (false, false) => int_store(f, size, rt),
        (true, true) => vec_load(f, size, rt),
        (true, false) => vec_store(f, size, rt),
    }

    if let Some(rn) = writeback {
        let off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
        emit!(f, I::LocalGet(0), I::LocalGet(T1), I::I64Store(at(off)));
    }
    true
}

/// Emit the addressing-mode computation, leaving the linear i32 address in the
/// [`ADDR`] local. Returns `Some(writeback_reg)` (the base register to update
/// after the access, via `T1`) or `None` if the mode isn't supported.
fn emit_ea(f: &mut Function, addr: AddrMode, pc: u64, guest_base: u64) -> Option<Option<u8>> {
    let delta = i64::from(abi::RAM_BASE) - guest_base as i64;
    let mut writeback = None;
    match addr {
        AddrMode::UnsignedImm { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::I64Const((imm as i64).wrapping_add(delta)), I::I64Add, I::I32WrapI64);
        }
        AddrMode::Unscaled { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::I64Const(imm.wrapping_add(delta)), I::I64Add, I::I32WrapI64);
        }
        AddrMode::PreIndex { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::LocalSet(T0));
            wb_value(f, imm);
            writeback = Some(rn);
            emit_addr_from(f, T0, imm.wrapping_add(delta));
        }
        AddrMode::PostIndex { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::LocalSet(T0));
            wb_value(f, imm);
            writeback = Some(rn);
            emit_addr_from(f, T0, delta); // ea = old base
        }
        AddrMode::Literal { offset } => {
            let lin = (pc.wrapping_add(offset as u64) as i64).wrapping_add(delta);
            emit!(f, I::I64Const(lin), I::I32WrapI64);
        }
        AddrMode::RegOffset { rn, rm, option, shift } => {
            if !matches!(option, 2 | 3 | 6 | 7) {
                return None; // non-standard extend: slow path
            }
            push_base_reg(f, rn);
            push_ext(f, rm, option, shift);
            emit!(f, I::I64Add, I::I64Const(delta), I::I64Add, I::I32WrapI64);
        }
    }
    emit!(f, I::LocalSet(ADDR));
    Some(writeback)
}

/// Integer load from [`ADDR`] into register `rt`.
fn int_load(f: &mut Function, size: u8, signed: bool, dst64: bool, rt: u8) {
    if rt != 31 {
        emit!(f, I::LocalGet(0)); // regs_base for the result store
    }
    emit!(f, I::LocalGet(ADDR), load_op(size, signed));
    if signed && !dst64 {
        emit!(f, I::I64Const(W_MASK), I::I64And); // sign-extend to 32, then zero-extend
    }
    if rt == 31 {
        emit!(f, I::Drop);
    } else {
        emit!(f, I::I64Store(at(offsets::x(rt as usize))));
    }
}

/// Integer store of register `rt` to [`ADDR`].
fn int_store(f: &mut Function, size: u8, rt: u8) {
    emit!(f, I::LocalGet(ADDR));
    if rt == 31 {
        emit!(f, I::I64Const(0)); // STR of XZR
    } else {
        emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rt as usize))));
    }
    emit!(f, store_op(size));
}

/// Vector load from [`ADDR`] into V[rt], zeroing the unused high bytes (a
/// SIMD/FP load writes the whole 128-bit register).
fn vec_load(f: &mut Function, size: u8, rt: u8) {
    if size == 4 {
        // 128-bit: copy 16 bytes straight into the V slot.
        emit!(f, I::LocalGet(0), I::LocalGet(ADDR), I::V128Load(at(0)), I::V128Store(at(offsets::v(rt as usize))));
    } else {
        // 8..64-bit: zero-extended low half, zero the high half.
        emit!(f, I::LocalGet(0), I::LocalGet(ADDR), load_op(size, false), I::I64Store(at(offsets::v(rt as usize))));
        emit!(f, I::LocalGet(0), I::I64Const(0), I::I64Store(at(offsets::v(rt as usize) + 8)));
    }
}

/// Vector store of the low `1 << size` bytes of V[rt] to [`ADDR`].
fn vec_store(f: &mut Function, size: u8, rt: u8) {
    if size == 4 {
        emit!(f, I::LocalGet(ADDR), I::LocalGet(0), I::V128Load(at(offsets::v(rt as usize))), I::V128Store(at(0)));
    } else {
        emit!(f, I::LocalGet(ADDR), I::LocalGet(0), I::I64Load(at(offsets::v(rt as usize))), store_op(size));
    }
}

/// Push the base register (r31 = SP) as a full i64.
fn push_base_reg(f: &mut Function, rn: u8) {
    let off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
    emit!(f, I::LocalGet(0), I::I64Load(at(off)));
}

/// Given a base i64 on the stack, fold in `disp` and wrap to the i32 address.
/// Push `local + disp` wrapped to an i32 linear address.
fn emit_addr_from(f: &mut Function, local: u32, disp: i64) {
    emit!(f, I::LocalGet(local), I::I64Const(disp), I::I64Add, I::I32WrapI64);
}

/// Compute the writeback value `T0 + imm` into `T1`.
fn wb_value(f: &mut Function, imm: i64) {
    emit!(f, I::LocalGet(T0), I::I64Const(imm), I::I64Add, I::LocalSet(T1));
}

fn load_op(size: u8, signed: bool) -> I<'static> {
    if signed {
        match size {
            0 => I::I64Load8S(at(0)),
            1 => I::I64Load16S(at(0)),
            2 => I::I64Load32S(at(0)),
            _ => I::I64Load(at(0)),
        }
    } else {
        match size {
            0 => I::I64Load8U(at(0)),
            1 => I::I64Load16U(at(0)),
            2 => I::I64Load32U(at(0)),
            _ => I::I64Load(at(0)),
        }
    }
}

fn store_op(size: u8) -> I<'static> {
    match size {
        0 => I::I64Store8(at(0)),
        1 => I::I64Store16(at(0)),
        2 => I::I64Store32(at(0)),
        _ => I::I64Store(at(0)),
    }
}

/// LDP/STP/LDPSW, integer and SIMD/FP pairs.
pub(super) fn load_store_pair(f: &mut Function, insn: &Insn, guest_base: u64) -> bool {
    let Insn::LoadStorePair { is_load, signed, width8, vec, vesize, rt, rt2, rn, offset, index } = *insn else {
        return false;
    };
    let delta = i64::from(abi::RAM_BASE) - guest_base as i64;

    // base -> T0; ea displacement depends on the index mode.
    push_base_reg(f, rn);
    emit!(f, I::LocalSet(T0));
    let ea_disp = match index {
        PairIndex::Post => 0,
        PairIndex::Offset | PairIndex::Pre => offset,
    };

    if vec {
        // Two SIMD/FP elements of width `vesize` (2=S, 3=D, 4=Q), `step` apart.
        let step = 1i64 << vesize;
        for (k, vt) in [rt, rt2].into_iter().enumerate() {
            emit_addr_from(f, T0, ea_disp + step * k as i64 + delta);
            emit!(f, I::LocalSet(ADDR));
            if is_load {
                vec_load(f, vesize, vt);
            } else {
                vec_store(f, vesize, vt);
            }
        }
    } else {
        let size = if width8 { 3 } else { 2 };
        let esize = if width8 { 8i64 } else { 4 };
        if is_load {
            // LDPSW and the X form write full X; the W form zero-extends.
            let wide = width8 || signed;
            pair_load(f, T0, ea_disp + delta, size, signed, wide, rt);
            pair_load(f, T0, ea_disp + esize + delta, size, signed, wide, rt2);
        } else {
            pair_store(f, T0, ea_disp + delta, size, rt);
            pair_store(f, T0, ea_disp + esize + delta, size, rt2);
        }
    }

    if matches!(index, PairIndex::Pre | PairIndex::Post) {
        let off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
        emit!(f, I::LocalGet(T0), I::I64Const(offset), I::I64Add); // base + offset
        emit!(f, I::LocalSet(T1), I::LocalGet(0), I::LocalGet(T1), I::I64Store(at(off)));
    }
    true
}

fn pair_load(f: &mut Function, base: u32, disp: i64, size: u8, signed: bool, wide: bool, rt: u8) {
    if rt != 31 {
        emit!(f, I::LocalGet(0)); // regs_base for the result store
    }
    emit_addr_from(f, base, disp);
    emit!(f, load_op(size, signed));
    if signed && !wide {
        emit!(f, I::I64Const(W_MASK), I::I64And);
    }
    if rt == 31 {
        emit!(f, I::Drop);
    } else {
        emit!(f, I::I64Store(at(offsets::x(rt as usize))));
    }
}

fn pair_store(f: &mut Function, base: u32, disp: i64, size: u8, rt: u8) {
    emit_addr_from(f, base, disp);
    if rt == 31 {
        emit!(f, I::I64Const(0));
    } else {
        emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rt as usize))));
    }
    emit!(f, store_op(size));
}
