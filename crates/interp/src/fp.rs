//! Scalar floating-point execution.
//!
//! The differential harness seeds FPCR with default-NaN mode (DN=1) and
//! round-to-nearest, so arithmetic NaN results are the canonical default NaN and
//! rounding matches Rust's native `f32`/`f64`. Bit-manipulating ops (FMOV, FABS,
//! FNEG) operate on the raw bits and are *not* NaN-canonicalized.
//!
//! Rounding-mode helpers live in the `round` submodule (`fp/round.rs`).

pub(crate) mod flags;
pub(crate) mod round;

use aarch64_cpu_state::{CpuState, Flags};

use crate::cond::eval_cond;
use crate::fp::flags::Op;
use crate::fp::round::{round_f32, round_f64, Mode};

const DEFAULT_NAN_32: u32 = 0x7FC0_0000;
const DEFAULT_NAN_64: u64 = 0x7FF8_0000_0000_0000;

fn read_s(cpu: &CpuState, idx: u8) -> f32 {
    f32::from_bits(cpu.v[idx as usize] as u32)
}
fn read_d(cpu: &CpuState, idx: u8) -> f64 {
    f64::from_bits(cpu.v[idx as usize] as u64)
}
/// Writing a scalar result zeroes the rest of the 128-bit V register.
fn write_s(cpu: &mut CpuState, idx: u8, bits: u32) {
    cpu.v[idx as usize] = u128::from(bits);
}
fn write_d(cpu: &mut CpuState, idx: u8, bits: u64) {
    cpu.v[idx as usize] = u128::from(bits);
}
pub(crate) fn canon_s(f: f32) -> u32 {
    if f.is_nan() {
        DEFAULT_NAN_32
    } else {
        f.to_bits()
    }
}
pub(crate) fn canon_d(f: f64) -> u64 {
    if f.is_nan() {
        DEFAULT_NAN_64
    } else {
        f.to_bits()
    }
}

fn single(ftype: u8) -> bool {
    ftype == 0b00
}

pub(crate) fn dp1(cpu: &mut CpuState, ftype: u8, opcode: u8, rn: u8, rd: u8) -> Option<u64> {
    if single(ftype) {
        let x = read_s(cpu, rn);
        match opcode {
            0 => write_s(cpu, rd, x.to_bits()),               // FMOV
            1 => write_s(cpu, rd, x.to_bits() & 0x7fff_ffff), // FABS (clear sign)
            2 => write_s(cpu, rd, x.to_bits() ^ 0x8000_0000), // FNEG (flip sign)
            3 => { let r = x.sqrt(); flags::sqrt(cpu, x, r); write_s(cpu, rd, canon_s(r)); } // FSQRT
            5 => { // FCVT single->double: always exact; only an SNaN is Invalid.
                if is_snan_s(x) { flags::raise(cpu, flags::IOC); }
                write_d(cpu, rd, canon_d(f64::from(x)));
            }
            _ => write_s(cpu, rd, canon_s(round_f32(x, frint_mode(opcode)))), // FRINT*
        }
    } else {
        let x = read_d(cpu, rn);
        match opcode {
            0 => write_d(cpu, rd, x.to_bits()),
            1 => write_d(cpu, rd, x.to_bits() & 0x7fff_ffff_ffff_ffff),
            2 => write_d(cpu, rd, x.to_bits() ^ 0x8000_0000_0000_0000),
            3 => { let r = x.sqrt(); flags::sqrt(cpu, x, r); write_d(cpu, rd, canon_d(r)); } // FSQRT
            4 => { // FCVT double->single: can overflow / underflow / lose precision.
                let r = x as f32;
                let mut f = 0u64;
                if is_snan_d(x) {
                    f |= flags::IOC;
                } else if r.is_infinite() && x.is_finite() {
                    f |= flags::OFC | flags::IXC;
                } else if r.is_finite() && f64::from(r) != x {
                    f |= flags::IXC;
                    if r != 0.0 && r.is_subnormal() { f |= flags::UFC; }
                }
                flags::raise(cpu, f);
                write_s(cpu, rd, canon_s(r));
            }
            _ => write_d(cpu, rd, canon_d(round_f64(x, frint_mode(opcode)))), // FRINT*
        }
    }
    None
}

/// FP 1-source FRINT opcode -> rounding mode.
fn frint_mode(opcode: u8) -> Mode {
    match opcode {
        0x8 => Mode::Near,  // FRINTN
        0x9 => Mode::Ceil,  // FRINTP
        0xa => Mode::Floor, // FRINTM
        0xb => Mode::Trunc, // FRINTZ
        0xc => Mode::Away,  // FRINTA
        _ => Mode::Near,    // FRINTX (0xe) / FRINTI (0xf): current mode = nearest-even
    }
}

pub(crate) fn dp2(
    cpu: &mut CpuState,
    ftype: u8,
    opcode: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    if single(ftype) {
        let (a, b) = (read_s(cpu, rn), read_s(cpu, rm));
        let r = match opcode {
            0b0000 | 0b1000 => { let p = a * b; flags::binop(cpu, Op::Mul, a, b, p); p } // FMUL/FNMUL
            0b0001 => { let q = a / b; flags::binop(cpu, Op::Div, a, b, q); q }
            0b0010 => { let s = a + b; flags::binop(cpu, Op::Add, a, b, s); s }
            0b0011 => { let d = a - b; flags::binop(cpu, Op::Sub, a, b, d); d }
            // FMAX/FMIN/FMAXNM/FMINNM: a selection — only a signaling-NaN operand
            // raises Invalid; the chosen value is exact.
            _ => {
                if is_snan_s(a) || is_snan_s(b) {
                    flags::raise(cpu, flags::IOC);
                }
                dp2_op_s(opcode, a, b)
            }
        };
        // FNMUL negates the *canonicalized* product (sign flip even on default NaN).
        let bits = if opcode == 0b1000 { canon_s(r) ^ 0x8000_0000 } else { canon_s(r) };
        write_s(cpu, rd, bits);
    } else {
        let (a, b) = (read_d(cpu, rn), read_d(cpu, rm));
        let r = match opcode {
            0b0000 | 0b1000 => { let p = a * b; flags::binop(cpu, Op::Mul, a, b, p); p }
            0b0001 => { let q = a / b; flags::binop(cpu, Op::Div, a, b, q); q }
            0b0010 => { let s = a + b; flags::binop(cpu, Op::Add, a, b, s); s }
            0b0011 => { let d = a - b; flags::binop(cpu, Op::Sub, a, b, d); d }
            _ => {
                if is_snan_d(a) || is_snan_d(b) {
                    flags::raise(cpu, flags::IOC);
                }
                dp2_op_d(opcode, a, b)
            }
        };
        let bits = if opcode == 0b1000 { canon_d(r) ^ 0x8000_0000_0000_0000 } else { canon_d(r) };
        write_d(cpu, rd, bits);
    }
    None
}

fn dp2_op_s(opcode: u8, a: f32, b: f32) -> f32 {
    match opcode {
        0b0000 => a * b,
        0b0001 => a / b,
        0b0010 => a + b,
        0b0011 => a - b,
        0b0100 => fmax_s(a, b),
        0b0101 => fmin_s(a, b),
        0b0110 => fmaxnm_s(a, b),
        0b0111 => fminnm_s(a, b),
        _ => -(a * b), // FNMUL
    }
}
fn dp2_op_d(opcode: u8, a: f64, b: f64) -> f64 {
    match opcode {
        0b0000 => a * b,
        0b0001 => a / b,
        0b0010 => a + b,
        0b0011 => a - b,
        0b0100 => fmax_d(a, b),
        0b0101 => fmin_d(a, b),
        0b0110 => fmaxnm_d(a, b),
        0b0111 => fminnm_d(a, b),
        _ => -(a * b),
    }
}

/// A signaling NaN has the top fraction bit clear.
macro_rules! is_snan {
    ($name:ident, $t:ty, $top_frac_bit:expr) => {
        /// A signaling NaN has the top fraction bit clear.
        fn $name(f: $t) -> bool {
            f.is_nan() && f.to_bits() & $top_frac_bit == 0
        }
    };
}
is_snan!(is_snan_s, f32, 0x0040_0000);
is_snan!(is_snan_d, f64, 0x0008_0000_0000_0000);

macro_rules! min_max {
    ($t:ty, $snan:ident, $max:ident, $min:ident, $maxnm:ident, $minnm:ident) => {
        // FMAX/FMIN propagate any NaN; signed zeros are ordered (+0 > -0).
        pub(crate) fn $max(a: $t, b: $t) -> $t {
            if a.is_nan() || b.is_nan() { <$t>::NAN } else { a.max(b) }
        }
        pub(crate) fn $min(a: $t, b: $t) -> $t {
            if a.is_nan() || b.is_nan() { <$t>::NAN } else { a.min(b) }
        }
        // FMAXNM/FMINNM use the other operand for a *quiet* NaN; a signaling NaN
        // raises Invalid and yields a NaN result (default NaN under DN=1).
        pub(crate) fn $maxnm(a: $t, b: $t) -> $t {
            if $snan(a) || $snan(b) { <$t>::NAN } else { a.max(b) }
        }
        pub(crate) fn $minnm(a: $t, b: $t) -> $t {
            if $snan(a) || $snan(b) { <$t>::NAN } else { a.min(b) }
        }
    };
}
min_max!(f32, is_snan_s, fmax_s, fmin_s, fmaxnm_s, fminnm_s);
min_max!(f64, is_snan_d, fmax_d, fmin_d, fmaxnm_d, fminnm_d);

macro_rules! mulx {
    ($name:ident, $t:ty) => {
        /// FMULX: like multiply, but `0*inf` (either sign) yields ±2.0, not NaN.
        pub(crate) fn $name(a: $t, b: $t) -> $t {
            if (a == 0.0 && b.is_infinite()) || (a.is_infinite() && b == 0.0) {
                if a.is_sign_negative() ^ b.is_sign_negative() { -2.0 } else { 2.0 }
            } else {
                a * b
            }
        }
    };
}
mulx!(mulx_s, f32);
mulx!(mulx_d, f64);

pub(crate) fn compare(
    cpu: &mut CpuState,
    ftype: u8,
    rm: u8,
    rn: u8,
    cmp_zero: bool,
) -> Option<u64> {
    cpu.flags = if single(ftype) {
        let b = if cmp_zero { 0.0 } else { read_s(cpu, rm) };
        compare_flags(read_s(cpu, rn).partial_cmp(&b))
    } else {
        let b = if cmp_zero { 0.0 } else { read_d(cpu, rm) };
        compare_flags(read_d(cpu, rn).partial_cmp(&b))
    };
    None
}

fn compare_flags(ord: Option<std::cmp::Ordering>) -> Flags {
    use std::cmp::Ordering::*;
    match ord {
        Some(Equal) => Flags { n: false, z: true, c: true, v: false },
        Some(Less) => Flags { n: true, z: false, c: false, v: false },
        Some(Greater) => Flags { n: false, z: false, c: true, v: false },
        None => Flags { n: false, z: false, c: true, v: true }, // unordered (NaN)
    }
}

pub(crate) fn csel(
    cpu: &mut CpuState,
    ftype: u8,
    cond: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let src = if eval_cond(cond, cpu.flags) { rn } else { rm };
    if single(ftype) {
        write_s(cpu, rd, read_s(cpu, src).to_bits());
    } else {
        write_d(cpu, rd, read_d(cpu, src).to_bits());
    }
    None
}

/// FMADD/FMSUB/FNMADD/FNMSUB: fused multiply-add with a single rounding.
/// `o1`/`o0` negate the addend / the product respectively, matching
/// `FPMulAdd(o1?-Ra:Ra, o1^o0?-Rn:Rn, Rm)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dp3(
    cpu: &mut CpuState,
    ftype: u8,
    o1: bool,
    o0: bool,
    rm: u8,
    ra: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    if single(ftype) {
        let (n, m, a) = (read_s(cpu, rn), read_s(cpu, rm), read_s(cpu, ra));
        let n = if o1 { -n } else { n }; // FNMADD/FNMSUB negate Rn
        let a = if o1 { -a } else { a }; // FNMADD/FNMSUB negate Ra
        let n = if o0 { -n } else { n }; // FMSUB/FNMADD negate the product
        let r = n.mul_add(m, a);
        fma_flags_s(cpu, n, m, a, r);
        write_s(cpu, rd, canon_s(r));
    } else {
        let (n, m, a) = (read_d(cpu, rn), read_d(cpu, rm), read_d(cpu, ra));
        let n = if o1 { -n } else { n };
        let a = if o1 { -a } else { a };
        let n = if o0 { -n } else { n };
        let r = n.mul_add(m, a);
        fma_flags_d(cpu, n, m, a, r);
        write_d(cpu, rd, canon_d(r));
    }
    None
}

// FMADD-family flags (best-effort): Invalid for a signaling-NaN operand or a
// NaN produced from finite inputs (e.g. `inf*0`), and Overflow when a finite
// product+addend rounds to infinity. Inexact for the fused op is not modelled
// (it would need extended precision) — acceptable for an observational flag.
macro_rules! fma_flags {
    ($name:ident, $t:ty, $snan:ident) => {
        fn $name(cpu: &mut CpuState, n: $t, m: $t, a: $t, r: $t) {
            if $snan(n) || $snan(m) || $snan(a)
                || (r.is_nan() && !n.is_nan() && !m.is_nan() && !a.is_nan())
            {
                flags::raise(cpu, flags::IOC);
            } else if r.is_infinite() && n.is_finite() && m.is_finite() && a.is_finite() {
                flags::raise(cpu, flags::OFC | flags::IXC);
            }
        }
    };
}
fma_flags!(fma_flags_s, f32, is_snan_s);
fma_flags!(fma_flags_d, f64, is_snan_d);

/// FCCMP/FCCMPE: compare if `cond` holds, else set NZCV to the immediate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ccmp(
    cpu: &mut CpuState,
    ftype: u8,
    rm: u8,
    rn: u8,
    cond: u8,
    nzcv: u8,
    _signaling: bool,
) -> Option<u64> {
    cpu.flags = if eval_cond(cond, cpu.flags) {
        if single(ftype) {
            compare_flags(read_s(cpu, rn).partial_cmp(&read_s(cpu, rm)))
        } else {
            compare_flags(read_d(cpu, rn).partial_cmp(&read_d(cpu, rm)))
        }
    } else {
        Flags {
            n: nzcv & 0b1000 != 0,
            z: nzcv & 0b0100 != 0,
            c: nzcv & 0b0010 != 0,
            v: nzcv & 0b0001 != 0,
        }
    };
    None
}

pub(crate) fn imm(cpu: &mut CpuState, ftype: u8, imm8: u8, rd: u8) -> Option<u64> {
    if single(ftype) {
        write_s(cpu, rd, expand_imm_s(imm8));
    } else {
        write_d(cpu, rd, expand_imm_d(imm8));
    }
    None
}

/// ARM `VFPExpandImm` for single precision.
pub(crate) fn expand_imm_s(imm8: u8) -> u32 {
    let a = u32::from(imm8 >> 7) & 1;
    let b = u32::from(imm8 >> 6) & 1;
    let rep5 = if b == 1 { 0b11111 } else { 0 };
    let exp = ((b ^ 1) << 7) | (rep5 << 2) | (u32::from(imm8 >> 4) & 0b11);
    let frac = (u32::from(imm8) & 0xf) << 19;
    (a << 31) | (exp << 23) | frac
}

/// ARM `VFPExpandImm` for double precision.
pub(crate) fn expand_imm_d(imm8: u8) -> u64 {
    let a = u64::from(imm8 >> 7) & 1;
    let b = u64::from(imm8 >> 6) & 1;
    let rep8 = if b == 1 { 0xff } else { 0 };
    let exp = ((b ^ 1) << 10) | (rep8 << 2) | (u64::from(imm8 >> 4) & 0b11);
    let frac = (u64::from(imm8) & 0xf) << 48;
    (a << 63) | (exp << 52) | frac
}

pub(crate) fn cvt_int(
    cpu: &mut CpuState,
    sf: bool,
    ftype: u8,
    rmode: u8,
    opcode: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let is_single = single(ftype);
    match opcode {
        // SCVTF / UCVTF: integer -> FP (round to nearest).
        0b010 | 0b011 => {
            let signed = opcode == 0b010;
            let gpr = cpu.read_gpr(rn, false);
            int_to_fp(cpu, rd, is_single, sf, signed, gpr);
            flags::i2f(cpu, i2f_exact(gpr, sf, signed, is_single));
        }
        // FCVT{N,P,M,Z}{S,U}: FP -> integer with the rmode rounding (saturating).
        0b000 | 0b001 => {
            let signed = opcode == 0b000;
            let mode = match rmode {
                0b00 => Mode::Near,
                0b01 => Mode::Ceil,
                0b10 => Mode::Floor,
                _ => Mode::Trunc,
            };
            fcvt_to_int_flags(cpu, rn, is_single, sf, signed, mode);
            let result = fp_to_int(cpu, rn, is_single, sf, signed, mode);
            write_gpr(cpu, rd, sf, result);
        }
        // FCVTAS / FCVTAU: tie-away rounding.
        0b100 | 0b101 => {
            let signed = opcode == 0b100;
            fcvt_to_int_flags(cpu, rn, is_single, sf, signed, Mode::Away);
            let result = fp_to_int(cpu, rn, is_single, sf, signed, Mode::Away);
            write_gpr(cpu, rd, sf, result);
        }
        // FMOV gpr <-> fpr.
        0b110 => {
            if ftype == 0b10 {
                // FMOV Xd, Vn.D[1]: high 64 bits of the vector register -> Xd.
                let hi = (cpu.v[rn as usize] >> 64) as u64;
                write_gpr(cpu, rd, true, hi);
            } else {
                // FP -> GPR (W<->S / X<->D).
                let bits =
                    if sf { cpu.v[rn as usize] as u64 } else { u64::from(cpu.v[rn as usize] as u32) };
                write_gpr(cpu, rd, sf, bits);
            }
        }
        0b111 => {
            if ftype == 0b10 {
                // FMOV Vd.D[1], Xn: Xn -> high 64 bits of Vd, low 64 preserved.
                let gpr = cpu.read_gpr(rn, false);
                let lo = cpu.v[rd as usize] & u128::from(u64::MAX);
                cpu.v[rd as usize] = lo | (u128::from(gpr) << 64);
            } else {
                // GPR -> FP (W<->S / X<->D).
                let gpr = cpu.read_gpr(rn, false);
                if sf {
                    write_d(cpu, rd, gpr);
                } else {
                    write_s(cpu, rd, gpr as u32);
                }
            }
        }
        _ => {}
    }
    None
}

fn int_to_fp(cpu: &mut CpuState, rd: u8, is_single: bool, sf: bool, signed: bool, gpr: u64) {
    if signed {
        let iv = if sf { gpr as i64 } else { i64::from(gpr as i32) };
        if is_single {
            write_s(cpu, rd, (iv as f32).to_bits());
        } else {
            write_d(cpu, rd, (iv as f64).to_bits());
        }
    } else {
        let uv = if sf { gpr } else { u64::from(gpr as u32) };
        if is_single {
            write_s(cpu, rd, (uv as f32).to_bits());
        } else {
            write_d(cpu, rd, (uv as f64).to_bits());
        }
    }
}

/// FP -> integer, rounding per `mode` then saturating. Rust's saturating `as`
/// casts match the ARM semantics (NaN -> 0, out-of-range -> saturate). Rounding
/// is done on the source-width float to match QEMU.
/// Convert between FP and fixed-point. `opcode`: 010=SCVTF,011=UCVTF (fixed int
/// -> FP), 000=FCVTZS,001=FCVTZU (FP -> fixed int, toward zero). The fraction
/// has `64 - scale` bits, i.e. scaling by 2^±fbits (an exact power of two).
pub(crate) fn cvt_fixed(
    cpu: &mut CpuState,
    sf: bool,
    ftype: u8,
    opcode: u8,
    scale: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let is_single = single(ftype);
    let fbits = 64 - i32::from(scale); // 1..=64

    match opcode {
        // SCVTF / UCVTF: integer (with `fbits` fraction bits) -> FP.
        0b010 | 0b011 => {
            let signed = opcode == 0b010;
            let gpr = cpu.read_gpr(rn, false);
            // Scaling by 2^-fbits is an exact power of two, so exactness is just
            // whether the raw integer fits the mantissa.
            flags::i2f(cpu, i2f_exact(gpr, sf, signed, is_single));
            if is_single {
                let iv = int_as_f32(gpr, sf, signed);
                write_s(cpu, rd, (iv * 2f32.powi(-fbits)).to_bits());
            } else {
                let iv = int_as_f64(gpr, sf, signed);
                write_d(cpu, rd, (iv * 2f64.powi(-fbits)).to_bits());
            }
        }
        // FCVTZS / FCVTZU: FP -> integer (with `fbits` fraction bits), toward zero.
        0b000 | 0b001 => {
            let signed = opcode == 0b000;
            let scaled = if is_single {
                f64::from(read_s(cpu, rn)) * 2f64.powi(fbits)
            } else {
                read_d(cpu, rn) * 2f64.powi(fbits)
            };
            let (lo, hi) = int_bounds(sf, signed);
            flags::f2i(cpu, scaled, lo, hi, scaled == scaled.trunc());
            // Rust float->int casts truncate toward zero and saturate (NaN -> 0).
            let result = match (sf, signed) {
                (true, true) => scaled as i64 as u64,
                (true, false) => scaled as u64,
                (false, true) => (scaled as i32 as u32) as u64,
                (false, false) => u64::from(scaled as u32),
            };
            write_gpr(cpu, rd, sf, result);
        }
        _ => {}
    }
    None
}

fn int_as_f32(gpr: u64, sf: bool, signed: bool) -> f32 {
    if signed {
        if sf {
            gpr as i64 as f32
        } else {
            (gpr as i32) as f32
        }
    } else if sf {
        gpr as f32
    } else {
        (gpr as u32) as f32
    }
}

fn int_as_f64(gpr: u64, sf: bool, signed: bool) -> f64 {
    if signed {
        if sf {
            gpr as i64 as f64
        } else {
            i64::from(gpr as i32) as f64
        }
    } else if sf {
        gpr as f64
    } else {
        f64::from(gpr as u32)
    }
}

/// Representable-exactly bounds `[lo, hi)` of an integer type, as `f64`.
fn int_bounds(sf: bool, signed: bool) -> (f64, f64) {
    match (sf, signed) {
        (true, true) => (-(2f64.powi(63)), 2f64.powi(63)),
        (true, false) => (0.0, 2f64.powi(64)),
        (false, true) => (-(2f64.powi(31)), 2f64.powi(31)),
        (false, false) => (0.0, 2f64.powi(32)),
    }
}

/// `true` iff the integer in `gpr` is exactly representable in the target float
/// (its significant-bit span fits the mantissa) — i.e. SCVTF/UCVTF is exact.
fn i2f_exact(gpr: u64, sf: bool, signed: bool, is_single: bool) -> bool {
    let mag = if signed {
        let iv = if sf { gpr as i64 } else { i64::from(gpr as i32) };
        iv.unsigned_abs()
    } else if sf {
        gpr
    } else {
        u64::from(gpr as u32)
    };
    let mant_bits = if is_single { 24 } else { 53 };
    mag == 0 || (mag >> mag.trailing_zeros()) < (1u64 << mant_bits)
}

/// Raise FP->integer conversion flags: Invalid for NaN / out-of-range (the cast
/// saturates), else Inexact when the source had a fractional part.
fn fcvt_to_int_flags(cpu: &mut CpuState, rn: u8, is_single: bool, sf: bool, signed: bool, mode: Mode) {
    let src = if is_single { f64::from(read_s(cpu, rn)) } else { read_d(cpu, rn) };
    let rounded = round_f64(src, mode);
    let (lo, hi) = int_bounds(sf, signed);
    flags::f2i(cpu, src, lo, hi, src == rounded);
}

fn fp_to_int(cpu: &CpuState, rn: u8, is_single: bool, sf: bool, signed: bool, mode: Mode) -> u64 {
    let val = if is_single {
        f64::from(round_f32(read_s(cpu, rn), mode))
    } else {
        round_f64(read_d(cpu, rn), mode)
    };
    match (sf, signed) {
        (true, true) => val as i64 as u64,
        (true, false) => val as u64,
        (false, true) => (val as i32 as u32) as u64,
        (false, false) => u64::from(val as u32),
    }
}

fn write_gpr(cpu: &mut CpuState, rd: u8, sf: bool, val: u64) {
    if sf {
        cpu.write_gpr(rd, false, val);
    } else {
        cpu.write_gpr_w(rd, false, val);
    }
}
