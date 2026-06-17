//! Advanced SIMD three-same floating-point: per-lane FP ops, fused multiply-add
//! (FMLA/FMLS), FMULX, the reciprocal steps (FRECPS/FRSQRTS), absolute compares
//! (FACGE/FACGT) and the pairwise reductions (FADDP/FMAXP/...).

use aarch64_cpu_state::CpuState;

use crate::fp::{
    canon_d, canon_s, fmax_d, fmax_s, fmaxnm_d, fmaxnm_s, fmin_d, fmin_s, fminnm_d, fminnm_s,
    mulx_d, mulx_s,
};
use crate::simd::{join, split};

pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    sz: bool,
    fpopcode: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let size = if sz { 3u8 } else { 2 };
    let la = split(cpu.v[rn as usize], size, q);
    let lb = split(cpu.v[rm as usize], size, q);
    let ld = split(cpu.v[rd as usize], size, q);

    let lanes: Vec<u64> = if is_pairwise(fpopcode) {
        pairwise(fpopcode, sz, &la, &lb)
    } else {
        (0..la.len())
            .map(|i| {
                if sz {
                    lane_d(fpopcode, la[i], lb[i], ld[i])
                } else {
                    u64::from(lane_s(fpopcode, la[i] as u32, lb[i] as u32, ld[i] as u32))
                }
            })
            .collect()
    };
    cpu.v[rd as usize] = join(&lanes, size);
    None
}

fn is_pairwise(fpopcode: u8) -> bool {
    matches!(fpopcode, 0x5a | 0x5e | 0x7e | 0x58 | 0x78)
}

/// Pairwise reductions over the concatenation {Vn:Vm}.
fn pairwise(fpopcode: u8, sz: bool, la: &[u64], lb: &[u64]) -> Vec<u64> {
    let mut src: Vec<u64> = la.to_vec();
    src.extend_from_slice(lb);
    (0..src.len() / 2)
        .map(|i| {
            let (x, y) = (src[2 * i], src[2 * i + 1]);
            if sz {
                pair_d(fpopcode, x, y)
            } else {
                u64::from(pair_s(fpopcode, x as u32, y as u32))
            }
        })
        .collect()
}

fn lane_s(fpopcode: u8, xb: u32, yb: u32, db: u32) -> u32 {
    let a = f32::from_bits(xb);
    let b = f32::from_bits(yb);
    let d = f32::from_bits(db);
    match fpopcode {
        0x1a => canon_s(a + b),
        0x3a => canon_s(a - b),
        0x5b => canon_s(a * b),
        0x5f => canon_s(a / b),
        0x1e => canon_s(fmax_s(a, b)),
        0x3e => canon_s(fmin_s(a, b)),
        0x18 => canon_s(fmaxnm_s(a, b)),
        0x38 => canon_s(fminnm_s(a, b)),
        0x7a => canon_s((a - b).abs()),     // FABD
        0x19 => canon_s(a.mul_add(b, d)),   // FMLA: d + a*b
        0x39 => canon_s((-a).mul_add(b, d)), // FMLS: d - a*b
        0x1b => canon_s(mulx_s(a, b)),      // FMULX
        0x1f => canon_s(recps_s(a, b)),     // FRECPS
        0x3f => canon_s(rsqrts_s(a, b)),    // FRSQRTS
        0x1c => bool_s(a == b),             // FCMEQ
        0x5c => bool_s(a >= b),             // FCMGE
        0x5d => bool_s(a.abs() >= b.abs()), // FACGE
        0x7d => bool_s(a.abs() > b.abs()),  // FACGT
        _ => bool_s(a > b),                 // FCMGT (0x7c)
    }
}

fn lane_d(fpopcode: u8, xb: u64, yb: u64, db: u64) -> u64 {
    let a = f64::from_bits(xb);
    let b = f64::from_bits(yb);
    let d = f64::from_bits(db);
    match fpopcode {
        0x1a => canon_d(a + b),
        0x3a => canon_d(a - b),
        0x5b => canon_d(a * b),
        0x5f => canon_d(a / b),
        0x1e => canon_d(fmax_d(a, b)),
        0x3e => canon_d(fmin_d(a, b)),
        0x18 => canon_d(fmaxnm_d(a, b)),
        0x38 => canon_d(fminnm_d(a, b)),
        0x7a => canon_d((a - b).abs()),
        0x19 => canon_d(a.mul_add(b, d)),
        0x39 => canon_d((-a).mul_add(b, d)),
        0x1b => canon_d(mulx_d(a, b)),
        0x1f => canon_d(recps_d(a, b)),
        0x3f => canon_d(rsqrts_d(a, b)),
        0x1c => bool_d(a == b),
        0x5c => bool_d(a >= b),
        0x5d => bool_d(a.abs() >= b.abs()),
        0x7d => bool_d(a.abs() > b.abs()),
        _ => bool_d(a > b),
    }
}

fn pair_s(fpopcode: u8, xb: u32, yb: u32) -> u32 {
    let a = f32::from_bits(xb);
    let b = f32::from_bits(yb);
    match fpopcode {
        0x5a => canon_s(a + b),            // FADDP
        0x5e => canon_s(fmax_s(a, b)),     // FMAXP
        0x7e => canon_s(fmin_s(a, b)),     // FMINP
        0x58 => canon_s(fmaxnm_s(a, b)),   // FMAXNMP
        _ => canon_s(fminnm_s(a, b)),      // FMINNMP (0x78)
    }
}

fn pair_d(fpopcode: u8, xb: u64, yb: u64) -> u64 {
    let a = f64::from_bits(xb);
    let b = f64::from_bits(yb);
    match fpopcode {
        0x5a => canon_d(a + b),
        0x5e => canon_d(fmax_d(a, b)),
        0x7e => canon_d(fmin_d(a, b)),
        0x58 => canon_d(fmaxnm_d(a, b)),
        _ => canon_d(fminnm_d(a, b)),
    }
}

/// FRECPS = 2.0 - a*b (fused); inf*0 yields 2.0.
fn recps_s(a: f32, b: f32) -> f32 {
    if (a.is_infinite() && b == 0.0) || (b.is_infinite() && a == 0.0) {
        2.0
    } else {
        (-a).mul_add(b, 2.0)
    }
}
fn recps_d(a: f64, b: f64) -> f64 {
    if (a.is_infinite() && b == 0.0) || (b.is_infinite() && a == 0.0) {
        2.0
    } else {
        (-a).mul_add(b, 2.0)
    }
}

/// FRSQRTS = (3.0 - a*b) / 2 (fused, single rounding incl. the halving); inf*0
/// yields 1.5. Computed in f64 so the halving folds into one f32 rounding — a
/// plain `mul_add(..) * 0.5` would overflow to inf before the halve.
fn rsqrts_s(a: f32, b: f32) -> f32 {
    if (a.is_infinite() && b == 0.0) || (b.is_infinite() && a == 0.0) {
        1.5
    } else {
        ((-f64::from(a)).mul_add(f64::from(b), 3.0) * 0.5) as f32
    }
}
fn rsqrts_d(a: f64, b: f64) -> f64 {
    if (a.is_infinite() && b == 0.0) || (b.is_infinite() && a == 0.0) {
        return 1.5;
    }
    // (3 - a*b)/2 with the halving folded into one rounding (no f128 available).
    let r = (-a).mul_add(b, 3.0);
    if r.is_infinite() {
        // The unhalved sum overflowed but the halved result is finite; halve the
        // product first. `b * 0.5` is exact for the large-magnitude b that drives
        // the overflow, so this rounds exactly once.
        (-a).mul_add(b * 0.5, 1.5)
    } else {
        r * 0.5
    }
}

fn bool_s(c: bool) -> u32 {
    if c {
        u32::MAX
    } else {
        0
    }
}
fn bool_d(c: bool) -> u64 {
    if c {
        u64::MAX
    } else {
        0
    }
}
