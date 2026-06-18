//! Floating-point ↔ integer and fixed-point conversions: `SCVTF`/`UCVTF`,
//! `FCVT*` (FP→int, all rounding modes), `FMOV` GPR↔FPR, and the fixed-point
//! forms. Split out of [`super`] (the scalar-FP arithmetic) as a distinct
//! concern. Rounding is done on the source-width float (matching QEMU); Rust's
//! saturating `as` casts give the ARM out-of-range/NaN behaviour.

use aarch64_cpu_state::CpuState;

use super::flags;
use super::round::{round_f32, round_f64, Mode};
use super::{read_d, read_s, single, write_d, write_s};

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

/// FP -> integer, rounding per `mode` then saturating (Rust's `as` matches ARM:
/// NaN -> 0, out-of-range -> saturate; rounding on the source-width float).
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
