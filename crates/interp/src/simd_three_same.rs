//! Advanced SIMD three-same (integer): the full element-wise + pairwise set.

use aarch64_cpu_state::CpuState;

use crate::simd::{join, split};

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    u: bool,
    size: u8,
    opcode: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let (a, b, d) = (cpu.v[rn as usize], cpu.v[rm as usize], cpu.v[rd as usize]);

    let result = match opcode {
        0b00011 => {
            let r = logical(u, size, a, b, d);
            if q { r } else { r & u128::from(u64::MAX) }
        }
        0b10100 | 0b10101 | 0b10111 => pairwise(opcode, u, size, q, a, b),
        _ => {
            let (la, lb, ld) = (split(a, size, q), split(b, size, q), split(d, size, q));
            let lanes: Vec<u64> = (0..la.len())
                .map(|i| lane(opcode, u, size, la[i], lb[i], ld[i]))
                .collect();
            join(&lanes, size)
        }
    };
    cpu.v[rd as usize] = result;
    None
}

fn logical(u: bool, size: u8, a: u128, b: u128, d: u128) -> u128 {
    match (u, size) {
        (false, 0) => a & b,
        (false, 1) => a & !b,
        (false, 2) => a | b,
        (false, _) => a | !b,
        (true, 0) => a ^ b,
        (true, 1) => (a & d) | (b & !d), // BSL
        (true, 2) => (a & b) | (d & !b), // BIT
        (true, _) => (a & !b) | (d & b), // BIF
    }
}

/// Pairwise ops combine adjacent elements of the concatenation {Vn:Vm}.
fn pairwise(opcode: u8, u: bool, size: u8, q: bool, a: u128, b: u128) -> u128 {
    let mut src = split(a, size, q);
    src.extend(split(b, size, q));
    let n = src.len();
    let out: Vec<u64> = (0..n / 2)
        .map(|i| {
            let (x, y) = (src[2 * i], src[2 * i + 1]);
            match opcode {
                0b10111 => x.wrapping_add(y), // ADDP
                0b10100 => max(u, size, x, y), // SMAXP/UMAXP
                _ => min(u, size, x, y),       // SMINP/UMINP
            }
        })
        .collect();
    join(&out, size)
}

fn lane(opcode: u8, u: bool, size: u8, x: u64, y: u64, d: u64) -> u64 {
    let ebits = 8u32 << size;
    let mask = width_mask(ebits);
    match opcode {
        0b00000 => halving_add(u, x, y, ebits, false), // S/U HADD
        0b00010 => halving_add(u, x, y, ebits, true),  // S/U RHADD
        0b00100 => halving_sub(u, x, y, ebits),        // S/U HSUB
        0b00001 => sat_add(u, x, y, ebits),            // SQADD/UQADD
        0b00101 => sat_sub(u, x, y, ebits),            // SQSUB/UQSUB
        0b00110 => bool_lane(if u { x > y } else { sx(x, ebits) > sx(y, ebits) }, mask),
        0b00111 => bool_lane(if u { x >= y } else { sx(x, ebits) >= sx(y, ebits) }, mask),
        0b01000 | 0b01010 => reg_shift(opcode, u, x, y, ebits, mask),
        0b01001 | 0b01011 => sat_reg_shift(opcode, u, x, y, ebits),
        0b01100 => max(u, size, x, y),
        0b01101 => min(u, size, x, y),
        0b01110 => abd(u, x, y, ebits),                // SABD/UABD
        0b01111 => d.wrapping_add(abd(u, x, y, ebits)) & mask, // SABA/UABA
        0b10000 => if u { x.wrapping_sub(y) & mask } else { x.wrapping_add(y) & mask },
        0b10001 => bool_lane(if u { x == y } else { x & y != 0 }, mask),
        0b10010 => {
            let p = x.wrapping_mul(y) & mask;
            if u { d.wrapping_sub(p) & mask } else { d.wrapping_add(p) & mask } // MLS/MLA
        }
        0b10011 => if u { pmul8(x, y) } else { x.wrapping_mul(y) & mask },
        0b10110 => sqdmulh(u, x, y, ebits),
        _ => 0,
    }
}

fn width_mask(ebits: u32) -> u64 {
    if ebits >= 64 {
        u64::MAX
    } else {
        (1u64 << ebits) - 1
    }
}

/// Sign-extend an `ebits`-wide value to i64.
fn sx(v: u64, ebits: u32) -> i64 {
    let s = 64 - ebits;
    ((v << s) as i64) >> s
}

fn bool_lane(c: bool, mask: u64) -> u64 {
    if c {
        mask
    } else {
        0
    }
}

fn max(u: bool, size: u8, x: u64, y: u64) -> u64 {
    let e = 8u32 << size;
    if u {
        x.max(y)
    } else if sx(x, e) >= sx(y, e) {
        x
    } else {
        y
    }
}
fn min(u: bool, size: u8, x: u64, y: u64) -> u64 {
    let e = 8u32 << size;
    if u {
        x.min(y)
    } else if sx(x, e) <= sx(y, e) {
        x
    } else {
        y
    }
}

fn halving_add(u: bool, x: u64, y: u64, ebits: u32, round: bool) -> u64 {
    let r = i64::from(round);
    let sum = if u {
        (x as i128 + y as i128 + r as i128) >> 1
    } else {
        ((sx(x, ebits) + sx(y, ebits) + r) >> 1) as i128
    };
    (sum as u64) & width_mask(ebits)
}

fn halving_sub(u: bool, x: u64, y: u64, ebits: u32) -> u64 {
    let diff = if u { x as i128 - y as i128 } else { (sx(x, ebits) - sx(y, ebits)) as i128 };
    ((diff >> 1) as u64) & width_mask(ebits)
}

fn sat_add(u: bool, x: u64, y: u64, ebits: u32) -> u64 {
    let mask = width_mask(ebits);
    if u {
        (x + y).min(mask)
    } else {
        let (lo, hi) = (-(1i128 << (ebits - 1)), (1i128 << (ebits - 1)) - 1);
        ((sx(x, ebits) as i128 + sx(y, ebits) as i128).clamp(lo, hi) as u64) & mask
    }
}

fn sat_sub(u: bool, x: u64, y: u64, ebits: u32) -> u64 {
    let mask = width_mask(ebits);
    if u {
        x.saturating_sub(y) // unsigned saturates at 0
    } else {
        let (lo, hi) = (-(1i128 << (ebits - 1)), (1i128 << (ebits - 1)) - 1);
        ((sx(x, ebits) as i128 - sx(y, ebits) as i128).clamp(lo, hi) as u64) & mask
    }
}

fn abd(u: bool, x: u64, y: u64, ebits: u32) -> u64 {
    let mask = width_mask(ebits);
    if u {
        if x >= y { x - y } else { y - x }
    } else {
        (sx(x, ebits) - sx(y, ebits)).unsigned_abs() & mask
    }
}

/// SSHL/USHL (opcode 01000) and SRSHL/URSHL (01010, rounding).
fn reg_shift(opcode: u8, u: bool, x: u64, y: u64, ebits: u32, mask: u64) -> u64 {
    let shift = (y & 0xff) as i8 as i32;
    let rounding = opcode == 0b01010;
    if shift >= 0 {
        let s = shift as u32;
        if s >= 64 { 0 } else { (x << s) & mask }
    } else {
        let s = (-shift) as u32;
        if s > ebits {
            // Shift amount past the width: 0, except rounding can round to 0/±0.
            if rounding && s == ebits + 1 { 0 } else { 0 }
        } else {
            let round = if rounding { 1i128 << (s - 1) } else { 0 };
            let v = if u { x as i128 } else { sx(x, ebits) as i128 };
            (((v + round) >> s) as u64) & mask
        }
    }
}

/// SQSHL/UQSHL (01001) and SQRSHL/UQRSHL (01011): saturating register shift.
fn sat_reg_shift(opcode: u8, u: bool, x: u64, y: u64, ebits: u32) -> u64 {
    let shift = (y & 0xff) as i8 as i32;
    let mask = width_mask(ebits);
    let rounding = opcode == 0b01011;
    let v = if u { x as i128 } else { i128::from(sx(x, ebits)) };

    let result = if shift >= 0 {
        v << shift
    } else {
        let s = (-shift) as u32;
        let round = if rounding && s >= 1 { 1i128 << (s - 1) } else { 0 };
        if s >= 128 { 0 } else { (v + round) >> s }
    };

    if u {
        result.clamp(0, mask as i128) as u64
    } else {
        let (lo, hi) = (-(1i128 << (ebits - 1)), (1i128 << (ebits - 1)) - 1);
        (result.clamp(lo, hi) as u64) & mask
    }
}

fn sqdmulh(u: bool, x: u64, y: u64, ebits: u32) -> u64 {
    let mask = width_mask(ebits);
    let prod = 2 * i128::from(sx(x, ebits)) * i128::from(sx(y, ebits));
    let round = if u { 1i128 << (ebits - 1) } else { 0 }; // SQRDMULH rounds
    let shifted = (prod + round) >> ebits;
    let (lo, hi) = (-(1i128 << (ebits - 1)), (1i128 << (ebits - 1)) - 1);
    (shifted.clamp(lo, hi) as u64) & mask
}

fn pmul8(x: u64, y: u64) -> u64 {
    let mut r = 0u64;
    for i in 0..8 {
        if (y >> i) & 1 == 1 {
            r ^= x << i;
        }
    }
    r & 0xff
}
