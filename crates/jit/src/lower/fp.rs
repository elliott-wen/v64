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

use super::cond::emit_cond_test;
use super::common::*;

const DEFAULT_NAN_32: i64 = 0x7FC0_0000;
const DEFAULT_NAN_64: i64 = 0x7FF8_0000_0000_0000;

// NZCV words for an FP compare result (N@31, Z@30, C@29, V@28).
const NZCV_LT: i64 = 1 << 31; // N
const NZCV_EQ: i64 = (1 << 30) | (1 << 29); // Z,C
const NZCV_GT: i64 = 1 << 29; // C
const NZCV_UN: i64 = (1 << 29) | (1 << 28); // C,V (unordered)

/// Eligibility gate: the scalar-FP forms lowered here. Single (`ftype` 0) and
/// double (`ftype` 1) only — half-precision and the FPSR/rounding-mode-sensitive
/// forms (FRINT, FMAXNM/FMINNM, conversions, fused multiply-add) fall back.
pub(crate) fn is_inline_fp(insn: &Insn) -> bool {
    match insn {
        Insn::FpDataProc1 { ftype, opcode, .. } => {
            // FMOV/FABS/FNEG/FSQRT (0..3), FCVT single<->double (5/4), and FRINT
            // N/P/M/Z/X/I (nearest/ceil/floor/trunc). FRINTA (0xc, tie-away) and
            // half-precision fall back.
            *ftype <= 1 && matches!((ftype, opcode), (_, 0..=3) | (0, 5) | (1, 4) | (_, 0x8 | 0x9 | 0xa | 0xb | 0xe | 0xf))
        }
        Insn::FpDataProc2 { ftype, opcode, .. } => *ftype <= 1 && (*opcode <= 5 || *opcode == 8),
        Insn::FpImm { ftype, .. }
        | Insn::FpCompare { ftype, .. }
        | Insn::FpCondCompare { ftype, .. }
        | Insn::FpCondSelect { ftype, .. } => *ftype <= 1,
        // SCVTF/UCVTF, FCVT{N,P,M,Z}{S,U}, FMOV gpr<->fp. Tie-away (FCVTA*,
        // opcode 100/101) has no wasm rounding op and falls back; ftype 10/11
        // (vector-high FMOV, half-precision) too.
        Insn::FpCvtInt { ftype, opcode, .. } => {
            *ftype <= 1 && matches!(opcode, 0b000 | 0b001 | 0b010 | 0b011 | 0b110 | 0b111)
        }
        // Fused multiply-add: see dp3 — emitted as mul+add (double-rounded), an
        // accepted approximation since wasm has no FMA.
        Insn::FpDataProc3 { ftype, .. } => *ftype <= 1,
        _ => false,
    }
}

/// FpDataProc3 — FMADD/FMSUB/FNMADD/FNMSUB. The architecture fuses the
/// multiply-add (one rounding); wasm has no FMA, so this emits `mul` then `add`
/// (two roundings). The result therefore differs from the interpreter in the
/// last bit for a small fraction of inputs — a deliberate, accepted
/// approximation (so the fused ops are excluded from the result crosscheck).
/// `o1` negates Rn and Ra; `o0` negates the product (so Rn is negated iff o1^o0,
/// Ra iff o1).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dp3(f: &mut Function, ftype: u8, o1: bool, o0: bool, rm: u8, ra: u8, rn: u8, rd: u8) {
    let single = ftype == 0;
    let neg = if single { I::F32Neg } else { I::F64Neg };
    let mul = if single { I::F32Mul } else { I::F64Mul };
    let add = if single { I::F32Add } else { I::F64Add };
    read(f, single, rn);
    if o1 ^ o0 {
        emit!(f, neg);
    }
    read(f, single, rm);
    emit!(f, mul);
    read(f, single, ra);
    if o1 {
        emit!(f, neg);
    }
    emit!(f, add);
    canon_write(f, single, rd);
}

/// FpCvtInt — integer<->FP conversions and FMOV between a GPR and an FP register.
pub(crate) fn cvt_int(f: &mut Function, sf: bool, ftype: u8, rmode: u8, opcode: u8, rn: u8, rd: u8) {
    let single = ftype == 0;
    match opcode {
        0b010 | 0b011 => scvtf(f, sf, single, opcode == 0b010, rn, rd), // int -> FP
        0b000 | 0b001 => fcvt_to_int(f, sf, single, opcode == 0b000, rmode, rn, rd),
        0b110 => {
            // FMOV FP -> GPR (raw bits; W<->S takes the low 32, X<->D the low 64).
            if sf {
                emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::v(rn as usize))), I::LocalSet(T0));
            } else {
                emit!(f, I::LocalGet(REGS_BASE), I::I32Load(at(offsets::v(rn as usize))), I::I64ExtendI32U, I::LocalSet(T0));
            }
            if rd != 31 {
                emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0), I::I64Store(at(offsets::x(rd as usize))));
            }
        }
        _ => {
            // FMOV GPR -> FP (opcode 111).
            push_gpr(f, rn);
            emit!(f, I::LocalSet(T0));
            write_t0(f, single, rd);
        }
    }
}

/// SCVTF/UCVTF — integer in `rn` to FP in `rd` (round to nearest; never NaN).
fn scvtf(f: &mut Function, sf: bool, single: bool, signed: bool, rn: u8, rd: u8) {
    push_gpr(f, rn);
    if !sf {
        emit!(f, I::I32WrapI64);
    }
    emit!(
        f,
        match (single, sf, signed) {
            (true, true, true) => I::F32ConvertI64S,
            (true, true, false) => I::F32ConvertI64U,
            (true, false, true) => I::F32ConvertI32S,
            (true, false, false) => I::F32ConvertI32U,
            (false, true, true) => I::F64ConvertI64S,
            (false, true, false) => I::F64ConvertI64U,
            (false, false, true) => I::F64ConvertI32S,
            (false, false, false) => I::F64ConvertI32U,
        }
    );
    // Bits -> T0, then a scalar FP write (zeroing the rest of Vd).
    if single {
        emit!(f, I::I32ReinterpretF32, I::I64ExtendI32U, I::LocalSet(T0));
    } else {
        emit!(f, I::I64ReinterpretF64, I::LocalSet(T0));
    }
    write_t0(f, single, rd);
}

/// FCVT{N,P,M,Z}{S,U} — FP in `rn` to integer in `rd`: round per `rmode` on the
/// source-width float, then truncate-saturate (NaN -> 0, out-of-range saturates,
/// like Rust's `as`). Tie-away (FCVTA*) is gated out (no wasm round op).
fn fcvt_to_int(f: &mut Function, sf: bool, single: bool, signed: bool, rmode: u8, rn: u8, rd: u8) {
    read(f, single, rn);
    // rmode 0=nearest, 1=ceil(+inf), 2=floor(-inf), 3=trunc (toward zero). For
    // trunc, trunc_sat already truncates, so no separate round is emitted.
    match (rmode, single) {
        (0, true) => emit!(f, I::F32Nearest),
        (0, false) => emit!(f, I::F64Nearest),
        (1, true) => emit!(f, I::F32Ceil),
        (1, false) => emit!(f, I::F64Ceil),
        (2, true) => emit!(f, I::F32Floor),
        (2, false) => emit!(f, I::F64Floor),
        _ => {}
    }
    emit!(
        f,
        match (single, sf, signed) {
            (true, true, true) => I::I64TruncSatF32S,
            (true, true, false) => I::I64TruncSatF32U,
            (true, false, true) => I::I32TruncSatF32S,
            (true, false, false) => I::I32TruncSatF32U,
            (false, true, true) => I::I64TruncSatF64S,
            (false, true, false) => I::I64TruncSatF64U,
            (false, false, true) => I::I32TruncSatF64S,
            (false, false, false) => I::I32TruncSatF64U,
        }
    );
    if !sf {
        emit!(f, I::I64ExtendI32U); // W result zero-extends into X
    }
    emit!(f, I::LocalSet(T0));
    if rd != 31 {
        emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0), I::I64Store(at(offsets::x(rd as usize))));
    }
}

/// Push X[rn] as i64 (0 for r31 = XZR).
fn push_gpr(f: &mut Function, rn: u8) {
    if rn == 31 {
        emit!(f, I::I64Const(0));
    } else {
        emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::x(rn as usize))));
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
        5 => {
            // FCVT single -> double.
            read_s(f, rn);
            emit!(f, I::F64PromoteF32);
            canon_d(f);
            write_d_t0(f, rd);
        }
        _ => {
            // FRINT N/P/M/Z/X/I — round to integral. (Tie-away FRINTA is gated out.)
            read(f, single, rn);
            emit!(
                f,
                match (opcode, single) {
                    (0x9, true) => I::F32Ceil,
                    (0x9, false) => I::F64Ceil,
                    (0xa, true) => I::F32Floor,
                    (0xa, false) => I::F64Floor,
                    (0xb, true) => I::F32Trunc,
                    (0xb, false) => I::F64Trunc,
                    // FRINTN (0x8) and FRINTX/I (0xe/0xf, current mode = nearest).
                    (_, true) => I::F32Nearest,
                    (_, false) => I::F64Nearest,
                }
            );
            canon_write(f, single, rd);
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

/// FCMP/FCMPE — set NZCV from the ordered comparison (signaling only affects
/// FPSR, which we don't model, so it has no effect here). `cmp_zero` compares
/// against +0.0.
pub(crate) fn compare(f: &mut Function, ftype: u8, rm: u8, rn: u8, cmp_zero: bool) {
    nzcv_from_cmp(f, ftype == 0, rm, rn, cmp_zero); // -> T0
    emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0), I::I64Store(at(offsets::NZCV)));
}

/// FCCMP/FCCMPE — if `cond` holds, compare and set NZCV; else force NZCV to the
/// 4-bit immediate.
pub(crate) fn ccmp(f: &mut Function, ftype: u8, rm: u8, rn: u8, cond: u8, nzcv: u8) {
    emit_cond_test(f, cond);
    emit!(f, I::If(BlockType::Empty));
    nzcv_from_cmp(f, ftype == 0, rm, rn, false);
    emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0), I::I64Store(at(offsets::NZCV)));
    emit!(f, I::Else);
    let packed = (i64::from(nzcv >> 3 & 1) << 31)
        | (i64::from(nzcv >> 2 & 1) << 30)
        | (i64::from(nzcv >> 1 & 1) << 29)
        | (i64::from(nzcv & 1) << 28);
    emit!(f, I::LocalGet(REGS_BASE), I::I64Const(packed), I::I64Store(at(offsets::NZCV)), I::End);
}

/// FCSEL — select Rn or Rm by `cond` (raw bit copy, zeroing the rest of Vd).
pub(crate) fn csel(f: &mut Function, ftype: u8, cond: u8, rm: u8, rn: u8, rd: u8) {
    let single = ftype == 0;
    emit_cond_test(f, cond);
    emit!(f, I::If(BlockType::Result(ValType::I64)));
    load_bits_ext(f, single, rn);
    emit!(f, I::Else);
    load_bits_ext(f, single, rm);
    emit!(f, I::End, I::LocalSet(T0));
    write_t0(f, single, rd);
}

/// Compute the NZCV word for `cmp(Rn, Rm-or-0)` into [`T0`] via a lt/eq/gt cascade
/// (all-false = unordered, since a NaN compares false every way).
fn nzcv_from_cmp(f: &mut Function, single: bool, rm: u8, rn: u8, cmp_zero: bool) {
    let (lt, eq, gt) =
        if single { (I::F32Lt, I::F32Eq, I::F32Gt) } else { (I::F64Lt, I::F64Eq, I::F64Gt) };
    cmp_pair(f, single, rn, rm, cmp_zero);
    emit!(f, lt, I::If(BlockType::Result(ValType::I64)), I::I64Const(NZCV_LT), I::Else);
    cmp_pair(f, single, rn, rm, cmp_zero);
    emit!(f, eq, I::If(BlockType::Result(ValType::I64)), I::I64Const(NZCV_EQ), I::Else);
    cmp_pair(f, single, rn, rm, cmp_zero);
    emit!(f, gt, I::If(BlockType::Result(ValType::I64)), I::I64Const(NZCV_GT), I::Else);
    emit!(f, I::I64Const(NZCV_UN), I::End, I::End, I::End, I::LocalSet(T0));
}

/// Push Rn then (Rm or +0.0) as the working float for a compare.
fn cmp_pair(f: &mut Function, single: bool, rn: u8, rm: u8, cmp_zero: bool) {
    read(f, single, rn);
    if cmp_zero {
        // +0.0 without an Ieee literal: reinterpret an all-zero word.
        if single {
            emit!(f, I::I32Const(0), I::F32ReinterpretI32);
        } else {
            emit!(f, I::I64Const(0), I::F64ReinterpretI64);
        }
    } else {
        read(f, single, rm);
    }
}

/// Push V[rn]'s scalar bits as i64 (low 32 zero-extended for single).
fn load_bits_ext(f: &mut Function, single: bool, rn: u8) {
    if single {
        emit!(f, I::LocalGet(REGS_BASE), I::I32Load(at(offsets::v(rn as usize))), I::I64ExtendI32U);
    } else {
        emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::v(rn as usize))));
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
