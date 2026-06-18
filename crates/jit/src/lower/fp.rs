//! Scalar floating-point lowering (single/double; half-precision falls back).
//!
//! The interpreter computes FP with native Rust `f32`/`f64`, and wasm's FP ops
//! are the same IEEE operations, so results match bit-for-bit — *except* NaN
//! payloads, which the interpreter forces to the default NaN (FPCR.DN=1); the
//! emitter canonicalizes likewise ([`canon_d`]/[`canon_s`]). FMOV/FABS/FNEG are
//! raw bit ops (no canonicalization), matching the interpreter.
//!
//! One deliberate approximation: this lowering does **not** maintain FPSR's
//! cumulative exception flags (the interpreter does). No FP result depends on
//! them and almost nothing reads them; the JIT↔interpreter crosscheck compares
//! the V registers and NZCV (which stay exact) but not FPSR.

use aarch64_cpu_state::regs::offsets;
use aarch64_decoder::Insn;
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

use super::common::*;

const DEFAULT_NAN_32: i64 = 0x7FC0_0000;
const DEFAULT_NAN_64: i64 = 0x7FF8_0000_0000_0000;

/// Eligibility gate: the scalar-FP forms lowered here. Single (`ftype` 0) and
/// double (`ftype` 1) only — half-precision and the FPSR/rounding-mode-sensitive
/// forms (FRINT, FMAXNM/FMINNM, conversions, fused multiply-add) fall back.
pub(crate) fn is_inline_fp(insn: &Insn) -> bool {
    match insn {
        Insn::FpDataProc1 { ftype, opcode, .. } => {
            *ftype <= 1 && matches!((ftype, opcode), (_, 0..=3) | (0, 5) | (1, 4))
        }
        Insn::FpDataProc2 { ftype, opcode, .. } => *ftype <= 1 && *opcode <= 5 || (*ftype <= 1 && *opcode == 8),
        Insn::FpImm { ftype, .. } => *ftype <= 1,
        _ => false,
    }
}

/// FpDataProc1 — FMOV/FABS/FNEG (raw bit ops), FSQRT, FCVT single<->double.
pub(crate) fn dp1(f: &mut Function, ftype: u8, opcode: u8, rn: u8, rd: u8) {
    let single = ftype == 0;
    match opcode {
        0 => mov_bits(f, single, rn, rd),       // FMOV
        1 => sign_op(f, single, rn, rd, false), // FABS (clear sign)
        2 => sign_op(f, single, rn, rd, true),  // FNEG (flip sign)
        3 => {
            // FSQRT
            read(f, single, rn);
            emit!(f, if single { I::F32Sqrt } else { I::F64Sqrt });
            canon_write(f, single, rd);
        }
        4 => {
            // FCVT double -> single.
            read_d(f, rn);
            emit!(f, I::F32DemoteF64);
            canon_s(f);
            write_s_t0(f, rd);
        }
        _ => {
            // FCVT single -> double (opcode 5).
            read_s(f, rn);
            emit!(f, I::F64PromoteF32);
            canon_d(f);
            write_d_t0(f, rd);
        }
    }
}

/// FpDataProc2 — FADD/FSUB/FMUL/FDIV/FNMUL and FMAX/FMIN.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dp2(f: &mut Function, ftype: u8, opcode: u8, rm: u8, rn: u8, rd: u8) {
    let single = ftype == 0;
    read(f, single, rn);
    read(f, single, rm);
    // FMAX/FMIN: wasm's max/min match ARM (any-NaN -> NaN, +0 > -0) once the
    // result is canonicalized. FNMUL negates the canonicalized product below.
    emit!(
        f,
        match (opcode, single) {
            (0 | 8, true) => I::F32Mul,
            (0 | 8, false) => I::F64Mul,
            (1, true) => I::F32Div,
            (1, false) => I::F64Div,
            (2, true) => I::F32Add,
            (2, false) => I::F64Add,
            (3, true) => I::F32Sub,
            (3, false) => I::F64Sub,
            (4, true) => I::F32Max,
            (4, false) => I::F64Max,
            (_, true) => I::F32Min,
            (_, false) => I::F64Min,
        }
    );
    canon_write(f, single, rd);
    if opcode == 8 {
        // FNMUL: flip the sign of the canonicalized product (even on default NaN).
        let sign = if single { DEFAULT_SIGN_32 } else { DEFAULT_SIGN_64 };
        emit!(f, I::LocalGet(T0), I::I64Const(sign), I::I64Xor, I::LocalSet(T0));
        write_t0(f, single, rd);
    }
}

const DEFAULT_SIGN_32: i64 = 0x8000_0000;
const DEFAULT_SIGN_64: i64 = 0x8000_0000_0000_0000u64 as i64;

/// FpImm — FMOV (immediate): the expanded constant is known at compile time.
pub(crate) fn imm(f: &mut Function, ftype: u8, imm8: u8, rd: u8) {
    if ftype == 0 {
        emit!(f, I::I64Const(i64::from(expand_imm_s(imm8))), I::LocalSet(T0));
        write_s_t0(f, rd);
    } else {
        emit!(f, I::I64Const(expand_imm_d(imm8) as i64), I::LocalSet(T0));
        write_d_t0(f, rd);
    }
}

// ---- helpers ----

/// Push V[rn] as the working float (low 32 bits as f32 / low 64 as f64).
fn read(f: &mut Function, single: bool, rn: u8) {
    if single {
        read_s(f, rn);
    } else {
        read_d(f, rn);
    }
}
fn read_s(f: &mut Function, rn: u8) {
    emit!(f, I::LocalGet(REGS_BASE), I::F32Load(at(offsets::v(rn as usize))));
}
fn read_d(f: &mut Function, rn: u8) {
    emit!(f, I::LocalGet(REGS_BASE), I::F64Load(at(offsets::v(rn as usize))));
}

/// FMOV Sd/Dd, Sn/Dn — copy raw bits (no NaN canonicalization), zeroing the rest.
fn mov_bits(f: &mut Function, single: bool, rn: u8, rd: u8) {
    if single {
        emit!(f, I::LocalGet(REGS_BASE), I::I32Load(at(offsets::v(rn as usize))), I::I64ExtendI32U, I::LocalSet(T0));
    } else {
        emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::v(rn as usize))), I::LocalSet(T0));
    }
    write_t0(f, single, rd);
}

/// FABS (clear sign) / FNEG (flip sign) — raw bit ops on V[rn] -> V[rd].
fn sign_op(f: &mut Function, single: bool, rn: u8, rd: u8, neg: bool) {
    if single {
        let (op, k) = if neg { (I::I32Xor, 0x8000_0000u32 as i32) } else { (I::I32And, 0x7FFF_FFFF) };
        emit!(f, I::LocalGet(REGS_BASE), I::I32Load(at(offsets::v(rn as usize))), I::I32Const(k), op, I::I64ExtendI32U, I::LocalSet(T0));
    } else {
        let (op, k) = if neg { (I::I64Xor, DEFAULT_SIGN_64) } else { (I::I64And, 0x7FFF_FFFF_FFFF_FFFF) };
        emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::v(rn as usize))), I::I64Const(k), op, I::LocalSet(T0));
    }
    write_t0(f, single, rd);
}

/// Canonicalize the float result on the stack and store it to V[rd].
fn canon_write(f: &mut Function, single: bool, rd: u8) {
    if single {
        canon_s(f);
        write_s_t0(f, rd);
    } else {
        canon_d(f);
        write_d_t0(f, rd);
    }
}

/// f64 on the stack -> canonicalized bits in [`T0`] (any NaN -> default NaN).
fn canon_d(f: &mut Function) {
    emit!(f, I::I64ReinterpretF64, I::LocalSet(T0));
    // NaN iff (bits & ~sign) > +inf bits.
    emit!(f, I::LocalGet(T0), I::I64Const(0x7FFF_FFFF_FFFF_FFFFu64 as i64), I::I64And, I::I64Const(0x7FF0_0000_0000_0000), I::I64GtU);
    emit!(f, I::If(BlockType::Result(ValType::I64)), I::I64Const(DEFAULT_NAN_64), I::Else, I::LocalGet(T0), I::End);
    emit!(f, I::LocalSet(T0));
}

/// f32 on the stack -> canonicalized bits in [`T0`] (low 32; any NaN -> default).
fn canon_s(f: &mut Function) {
    emit!(f, I::I32ReinterpretF32, I::I64ExtendI32U, I::LocalSet(T0));
    emit!(f, I::LocalGet(T0), I::I32WrapI64, I::I32Const(0x7FFF_FFFF), I::I32And, I::I32Const(0x7F80_0000), I::I32GtU);
    emit!(f, I::If(BlockType::Result(ValType::I64)), I::I64Const(DEFAULT_NAN_32), I::Else, I::LocalGet(T0), I::End);
    emit!(f, I::LocalSet(T0));
}

/// Store the bits in [`T0`] to V[rd] as a single- or double-width scalar.
fn write_t0(f: &mut Function, single: bool, rd: u8) {
    if single {
        write_s_t0(f, rd);
    } else {
        write_d_t0(f, rd);
    }
}

/// Store [`T0`]'s low 32 bits to V[rd], zeroing the upper 96 (scalar S write).
fn write_s_t0(f: &mut Function, rd: u8) {
    let v = offsets::v(rd as usize);
    emit!(f, I::LocalGet(REGS_BASE), I::V128Const(0), I::V128Store(at(v)));
    emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0), I::I32WrapI64, I::I32Store(at(v)));
}

/// Store [`T0`] (64 bits) to V[rd], zeroing the upper 64 (scalar D write).
fn write_d_t0(f: &mut Function, rd: u8) {
    let v = offsets::v(rd as usize);
    emit!(f, I::LocalGet(REGS_BASE), I::I64Const(0), I::I64Store(at(v + 8)));
    emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0), I::I64Store(at(v)));
}

/// ARM `VFPExpandImm`, single precision (mirrors `interp::fp::expand_imm_s`).
fn expand_imm_s(imm8: u8) -> u32 {
    let a = u32::from(imm8 >> 7) & 1;
    let b = u32::from(imm8 >> 6) & 1;
    let rep5 = if b == 1 { 0b11111 } else { 0 };
    let exp = ((b ^ 1) << 7) | (rep5 << 2) | (u32::from(imm8 >> 4) & 0b11);
    let frac = (u32::from(imm8) & 0xf) << 19;
    (a << 31) | (exp << 23) | frac
}

/// ARM `VFPExpandImm`, double precision (mirrors `interp::fp::expand_imm_d`).
fn expand_imm_d(imm8: u8) -> u64 {
    let a = u64::from(imm8 >> 7) & 1;
    let b = u64::from(imm8 >> 6) & 1;
    let rep8 = if b == 1 { 0xff } else { 0 };
    let exp = ((b ^ 1) << 10) | (rep8 << 2) | (u64::from(imm8 >> 4) & 0b11);
    let frac = (u64::from(imm8) & 0xf) << 48;
    (a << 63) | (exp << 52) | frac
}
