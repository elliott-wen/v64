//! Advanced SIMD two-register misc, floating-point sub-block: FABS/FNEG/FSQRT,
//! the compare-with-zero forms, SCVTF/UCVTF, the rounding FCVT-to-integer family,
//! FRINT[NMPZAXI], and the double<->single FCVTN/FCVTL. `fpop` is the 7-bit
//! remapped opcode; `sz` selects double (true) vs single lanes.

use aarch64_cpu_state::CpuState;

use crate::fp::{canon_d, canon_s};
use crate::fp_round::{round_f32, round_f64, Mode};
use crate::simd::{join, split};

pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    sz: bool,
    fpop: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let a = cpu.v[rn as usize];
    let d = cpu.v[rd as usize];
    let result = match fpop {
        0x16 => fcvtn(q, a, d), // FCVTN: double -> single (narrow)
        0x17 => fcvtl(q, a),    // FCVTL: single -> double (widen)
        _ => {
            let size = if sz { 3u8 } else { 2 };
            let lanes: Vec<u64> = split(a, size, q)
                .into_iter()
                .map(|x| if sz { lane_d(fpop, x) } else { u64::from(lane_s(fpop, x as u32)) })
                .collect();
            join(&lanes, size)
        }
    };
    cpu.v[rd as usize] = result;
    None
}

fn lane_s(fpop: u8, bits: u32) -> u32 {
    let f = f32::from_bits(bits);
    match fpop {
        0x2f => bits & 0x7fff_ffff,    // FABS
        0x6f => bits ^ 0x8000_0000,    // FNEG
        0x7f => canon_s(f.sqrt()),     // FSQRT
        0x2c => mask_s(f > 0.0),       // FCMGT #0
        0x2d => mask_s(f == 0.0),      // FCMEQ #0
        0x2e => mask_s(f < 0.0),       // FCMLT #0
        0x6c => mask_s(f >= 0.0),      // FCMGE #0
        0x6d => mask_s(f <= 0.0),      // FCMLE #0
        0x1d => ((bits as i32) as f32).to_bits(), // SCVTF
        0x5d => (bits as f32).to_bits(),          // UCVTF
        _ => {
            if let Some((signed, mode)) = fcvt_mode(fpop) {
                let r = round_f32(f, mode);
                if signed {
                    (r as i32) as u32
                } else {
                    r as u32
                }
            } else if let Some(mode) = frint_mode(fpop) {
                canon_s(round_f32(f, mode)) // canonicalize NaN under DN=1
            } else {
                bits
            }
        }
    }
}

fn lane_d(fpop: u8, bits: u64) -> u64 {
    let f = f64::from_bits(bits);
    match fpop {
        0x2f => bits & 0x7fff_ffff_ffff_ffff, // FABS
        0x6f => bits ^ 0x8000_0000_0000_0000, // FNEG
        0x7f => canon_d(f.sqrt()),            // FSQRT
        0x2c => mask_d(f > 0.0),
        0x2d => mask_d(f == 0.0),
        0x2e => mask_d(f < 0.0),
        0x6c => mask_d(f >= 0.0),
        0x6d => mask_d(f <= 0.0),
        0x1d => ((bits as i64) as f64).to_bits(), // SCVTF
        0x5d => (bits as f64).to_bits(),          // UCVTF
        _ => {
            if let Some((signed, mode)) = fcvt_mode(fpop) {
                let r = round_f64(f, mode);
                if signed {
                    (r as i64) as u64
                } else {
                    r as u64
                }
            } else if let Some(mode) = frint_mode(fpop) {
                canon_d(round_f64(f, mode)) // canonicalize NaN under DN=1
            } else {
                bits
            }
        }
    }
}

/// (signed, rounding-mode) for the FCVT[NMPZA][SU] opcodes.
fn fcvt_mode(fpop: u8) -> Option<(bool, Mode)> {
    Some(match fpop {
        0x1a => (true, Mode::Near),   // FCVTNS
        0x1b => (true, Mode::Floor),  // FCVTMS
        0x3a => (true, Mode::Ceil),   // FCVTPS
        0x3b => (true, Mode::Trunc),  // FCVTZS
        0x1c => (true, Mode::Away),   // FCVTAS
        0x5a => (false, Mode::Near),  // FCVTNU
        0x5b => (false, Mode::Floor), // FCVTMU
        0x7a => (false, Mode::Ceil),  // FCVTPU
        0x7b => (false, Mode::Trunc), // FCVTZU
        0x5c => (false, Mode::Away),  // FCVTAU
        _ => return None,
    })
}

/// Rounding mode for the FRINT[NMPZAXI] opcodes. FRINTX/FRINTI use the current
/// (default) mode, which is round-to-nearest-even.
fn frint_mode(fpop: u8) -> Option<Mode> {
    Some(match fpop {
        0x18 => Mode::Near,  // FRINTN
        0x19 => Mode::Floor, // FRINTM
        0x38 => Mode::Ceil,  // FRINTP
        0x39 => Mode::Trunc, // FRINTZ
        0x58 => Mode::Away,  // FRINTA
        0x59 | 0x79 => Mode::Near, // FRINTX / FRINTI
        _ => return None,
    })
}

fn mask_s(c: bool) -> u32 {
    if c {
        u32::MAX
    } else {
        0
    }
}
fn mask_d(c: bool) -> u64 {
    if c {
        u64::MAX
    } else {
        0
    }
}

/// FCVTN/FCVTN2: each of the two double lanes of Vn narrows to a single. Q=1
/// writes the upper 64 bits of Vd, preserving the lower half.
fn fcvtn(q: bool, a: u128, d: u128) -> u128 {
    let s0 = canon_s(f64::from_bits(a as u64) as f32);
    let s1 = canon_s(f64::from_bits((a >> 64) as u64) as f32);
    let packed = u64::from(s0) | (u64::from(s1) << 32);
    if q {
        (u128::from(packed) << 64) | (d & u128::from(u64::MAX))
    } else {
        u128::from(packed)
    }
}

/// FCVTL/FCVTL2: two single lanes (low or upper half) widen to two doubles.
fn fcvtl(q: bool, a: u128) -> u128 {
    let half = if q { (a >> 64) as u64 } else { a as u64 };
    let d0 = canon_d(f64::from(f32::from_bits(half as u32)));
    let d1 = canon_d(f64::from(f32::from_bits((half >> 32) as u32)));
    u128::from(d0) | (u128::from(d1) << 64)
}
