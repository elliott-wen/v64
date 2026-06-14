//! Scalar floating-point execution.
//!
//! The differential harness seeds FPCR with default-NaN mode (DN=1) and
//! round-to-nearest, so arithmetic NaN results are the canonical default NaN and
//! rounding matches Rust's native `f32`/`f64`. Bit-manipulating ops (FMOV, FABS,
//! FNEG) operate on the raw bits and are *not* NaN-canonicalized.

use aarch64_cpu_state::{CpuState, Flags};

use crate::cond::eval_cond;
use crate::fp_round::{round_f32, round_f64, Mode};

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
            3 => write_s(cpu, rd, canon_s(x.sqrt())),         // FSQRT
            5 => write_d(cpu, rd, canon_d(f64::from(x))),     // FCVT single->double
            _ => write_s(cpu, rd, canon_s(round_f32(x, frint_mode(opcode)))), // FRINT*
        }
    } else {
        let x = read_d(cpu, rn);
        match opcode {
            0 => write_d(cpu, rd, x.to_bits()),
            1 => write_d(cpu, rd, x.to_bits() & 0x7fff_ffff_ffff_ffff),
            2 => write_d(cpu, rd, x.to_bits() ^ 0x8000_0000_0000_0000),
            3 => write_d(cpu, rd, canon_d(x.sqrt())),
            4 => write_s(cpu, rd, canon_s(x as f32)), // FCVT double->single
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
        let bits = if opcode == 0b1000 {
            // FNMUL negates the *canonicalized* product, so the sign flip
            // applies even when the result is the default NaN.
            canon_s(a * b) ^ 0x8000_0000
        } else {
            canon_s(dp2_op_s(opcode, a, b))
        };
        write_s(cpu, rd, bits);
    } else {
        let (a, b) = (read_d(cpu, rn), read_d(cpu, rm));
        let bits = if opcode == 0b1000 {
            canon_d(a * b) ^ 0x8000_0000_0000_0000
        } else {
            canon_d(dp2_op_d(opcode, a, b))
        };
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
fn is_snan_s(f: f32) -> bool {
    f.is_nan() && f.to_bits() & 0x0040_0000 == 0
}
fn is_snan_d(f: f64) -> bool {
    f.is_nan() && f.to_bits() & 0x0008_0000_0000_0000 == 0
}

// FMAX/FMIN propagate any NaN; signed zeros are ordered (+0 > -0).
pub(crate) fn fmax_s(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
}
pub(crate) fn fmin_s(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.min(b) }
}
pub(crate) fn fmax_d(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}
pub(crate) fn fmin_d(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}

// FMAXNM/FMINNM treat a *quiet* NaN as "use the other operand", but a signaling
// NaN raises Invalid and yields a NaN result (default NaN under DN=1).
pub(crate) fn fmaxnm_s(a: f32, b: f32) -> f32 {
    if is_snan_s(a) || is_snan_s(b) { f32::NAN } else { a.max(b) }
}
pub(crate) fn fminnm_s(a: f32, b: f32) -> f32 {
    if is_snan_s(a) || is_snan_s(b) { f32::NAN } else { a.min(b) }
}
pub(crate) fn fmaxnm_d(a: f64, b: f64) -> f64 {
    if is_snan_d(a) || is_snan_d(b) { f64::NAN } else { a.max(b) }
}
pub(crate) fn fminnm_d(a: f64, b: f64) -> f64 {
    if is_snan_d(a) || is_snan_d(b) { f64::NAN } else { a.min(b) }
}

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
        write_s(cpu, rd, canon_s(n.mul_add(m, a)));
    } else {
        let (n, m, a) = (read_d(cpu, rn), read_d(cpu, rm), read_d(cpu, ra));
        let n = if o1 { -n } else { n };
        let a = if o1 { -a } else { a };
        let n = if o0 { -n } else { n };
        write_d(cpu, rd, canon_d(n.mul_add(m, a)));
    }
    None
}

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
fn expand_imm_s(imm8: u8) -> u32 {
    let a = u32::from(imm8 >> 7) & 1;
    let b = u32::from(imm8 >> 6) & 1;
    let rep5 = if b == 1 { 0b11111 } else { 0 };
    let exp = ((b ^ 1) << 7) | (rep5 << 2) | (u32::from(imm8 >> 4) & 0b11);
    let frac = (u32::from(imm8) & 0xf) << 19;
    (a << 31) | (exp << 23) | frac
}

/// ARM `VFPExpandImm` for double precision.
fn expand_imm_d(imm8: u8) -> u64 {
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
            let result = fp_to_int(cpu, rn, is_single, sf, signed, mode);
            write_gpr(cpu, rd, sf, result);
        }
        // FCVTAS / FCVTAU: tie-away rounding.
        0b100 | 0b101 => {
            let signed = opcode == 0b100;
            let result = fp_to_int(cpu, rn, is_single, sf, signed, Mode::Away);
            write_gpr(cpu, rd, sf, result);
        }
        // FMOV gpr <-> fpr.
        0b110 => {
            // FP -> GPR.
            let bits = if sf { cpu.v[rn as usize] as u64 } else { u64::from(cpu.v[rn as usize] as u32) };
            write_gpr(cpu, rd, sf, bits);
        }
        0b111 => {
            // GPR -> FP.
            let gpr = cpu.read_gpr(rn, false);
            if sf {
                write_d(cpu, rd, gpr);
            } else {
                write_s(cpu, rd, gpr as u32);
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
