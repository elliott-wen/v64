//! Advanced SIMD three-same floating-point: per-lane FP ops.

use aarch64_cpu_state::CpuState;

use crate::fp::{
    canon_d, canon_s, fmax_d, fmax_s, fmaxnm_d, fmaxnm_s, fmin_d, fmin_s, fminnm_d, fminnm_s,
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
    let size = if sz { 3 } else { 2 };
    let la = split(cpu.v[rn as usize], size, q);
    let lb = split(cpu.v[rm as usize], size, q);
    let lanes: Vec<u64> = la
        .iter()
        .zip(&lb)
        .map(|(&x, &y)| {
            if sz {
                lane_d(fpopcode, x, y)
            } else {
                u64::from(lane_s(fpopcode, x as u32, y as u32))
            }
        })
        .collect();
    cpu.v[rd as usize] = join(&lanes, size);
    None
}

fn lane_s(fpopcode: u8, xb: u32, yb: u32) -> u32 {
    let a = f32::from_bits(xb);
    let b = f32::from_bits(yb);
    match fpopcode {
        0x1a => canon_s(a + b),
        0x3a => canon_s(a - b),
        0x5b => canon_s(a * b),
        0x5f => canon_s(a / b),
        0x1e => canon_s(fmax_s(a, b)),
        0x3e => canon_s(fmin_s(a, b)),
        0x18 => canon_s(fmaxnm_s(a, b)),
        0x38 => canon_s(fminnm_s(a, b)),
        0x7a => canon_s((a - b).abs()),       // FABD
        0x1c => bool_s(a == b),               // FCMEQ
        0x5c => bool_s(a >= b),               // FCMGE
        _ => bool_s(a > b),                   // FCMGT (0x7c)
    }
}

fn lane_d(fpopcode: u8, xb: u64, yb: u64) -> u64 {
    let a = f64::from_bits(xb);
    let b = f64::from_bits(yb);
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
        0x1c => bool_d(a == b),
        0x5c => bool_d(a >= b),
        _ => bool_d(a > b),
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
