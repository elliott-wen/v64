//! Advanced SIMD two-register misc (integer). Same-width ops live here; the
//! shape-changing ops route to `simd_two_reg_narrow` (XTN family) and
//! `simd_two_reg_long` (SHLL, SADDLP/SADALP).

use aarch64_cpu_state::CpuState;

use crate::simd::{join, split};
use crate::{simd_two_reg_long, simd_two_reg_narrow};

pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    u: bool,
    size: u8,
    opcode: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let a = cpu.v[rn as usize];
    let d = cpu.v[rd as usize];
    let mask128 = if q { u128::MAX } else { u128::from(u64::MAX) };

    let result = match (u, opcode) {
        (_, 0b10010) | (_, 0b10100) => simd_two_reg_narrow::xtn(u, opcode, size, q, a, d),
        (true, 0b10011) => simd_two_reg_long::shll(size, q, a),
        (_, 0b00010) | (_, 0b00110) => simd_two_reg_long::addlp(u, opcode, size, q, a, d),
        (_, 0b00011) => acc2(a, d, size, q, |x, dd, e| suqadd(u, x, dd, e)), // SUQADD/USQADD
        (false, 0b00000) => rev(a, size, q, 64),  // REV64
        (false, 0b00001) => rev(a, size, q, 16),  // REV16
        (true, 0b00000) => rev(a, size, q, 32),   // REV32
        (true, 0b00101) if size == 0 => !a & mask128, // NOT
        (true, 0b00101) => map(a, 0, q, rbit8) & mask128, // RBIT: always per-byte
        _ => map(a, size, q, |x| lane(u, opcode, size, x)),
    };
    cpu.v[rd as usize] = result;
    None
}

/// Reverse the order of `8<<size`-bit elements within each `container`-bit group.
fn rev(val: u128, size: u8, q: bool, container: usize) -> u128 {
    let ebits = 8usize << size;
    let per = container / ebits;
    let mut lanes = split(val, size, q);
    for chunk in lanes.chunks_mut(per) {
        chunk.reverse();
    }
    join(&lanes, size)
}

/// Apply a per-lane unary function across the register.
fn map(val: u128, size: u8, q: bool, f: impl Fn(u64) -> u64) -> u128 {
    let lanes: Vec<u64> = split(val, size, q).into_iter().map(f).collect();
    join(&lanes, size)
}

/// Apply a function of (src_lane, dst_lane, ebits) across the register.
fn acc2(a: u128, d: u128, size: u8, q: bool, f: impl Fn(u64, u64, u32) -> u64) -> u128 {
    let ebits = 8u32 << size;
    let (la, ld) = (split(a, size, q), split(d, size, q));
    let lanes: Vec<u64> = (0..la.len()).map(|i| f(la[i], ld[i], ebits)).collect();
    join(&lanes, size)
}

fn lane(u: bool, opcode: u8, size: u8, x: u64) -> u64 {
    let ebits = 8u32 << size;
    let mask = width_mask(ebits);
    match (u, opcode) {
        (false, 0b00100) => cls(x, ebits),          // CLS
        // CLZ: cap at the element width (a zero element gives `ebits`, not 64).
        (true, 0b00100) => u64::from((x << (64 - ebits)).leading_zeros()).min(u64::from(ebits)),
        (false, 0b00101) => u64::from((x as u8).count_ones()), // CNT (byte)
        (false, 0b01000) => bool_lane(sx(x, ebits) > 0, mask),   // CMGT zero
        (true, 0b01000) => bool_lane(sx(x, ebits) >= 0, mask),   // CMGE zero
        (false, 0b01001) => bool_lane(x & mask == 0, mask),      // CMEQ zero
        (true, 0b01001) => bool_lane(sx(x, ebits) <= 0, mask),   // CMLE zero
        (false, 0b01010) => bool_lane(sx(x, ebits) < 0, mask),   // CMLT zero
        (false, 0b00111) => sqabs(x, ebits),         // SQABS
        (true, 0b00111) => sqneg(x, ebits),          // SQNEG
        (false, 0b01011) => abs(x, ebits) & mask,    // ABS
        (true, 0b01011) => x.wrapping_neg() & mask,  // NEG
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

fn bool_lane(c: bool, mask: u64) -> u64 {
    if c {
        mask
    } else {
        0
    }
}

/// Sign-extend an `ebits`-wide value to i64.
fn sx(v: u64, ebits: u32) -> i64 {
    let s = 64 - ebits;
    ((v << s) as i64) >> s
}

fn cls(x: u64, ebits: u32) -> u64 {
    let sign = (x >> (ebits - 1)) & 1;
    let mut count = 0u64;
    let mut i = ebits - 1;
    while i > 0 {
        i -= 1;
        if (x >> i) & 1 == sign {
            count += 1;
        } else {
            break;
        }
    }
    count
}

fn abs(x: u64, ebits: u32) -> u64 {
    sx(x, ebits).unsigned_abs()
}

/// Saturating signed absolute value: |MIN| saturates to MAX.
fn sqabs(x: u64, ebits: u32) -> u64 {
    let mask = width_mask(ebits);
    let hi = (1i128 << (ebits - 1)) - 1;
    (i128::from(sx(x, ebits)).abs().min(hi) as u64) & mask
}

/// Saturating signed negate: -MIN saturates to MAX.
fn sqneg(x: u64, ebits: u32) -> u64 {
    let mask = width_mask(ebits);
    let hi = (1i128 << (ebits - 1)) - 1;
    ((-i128::from(sx(x, ebits))).min(hi) as u64) & mask
}

/// SUQADD (u=0): signed-saturating accumulate of an unsigned element into a
/// signed destination. USQADD (u=1): the reverse.
fn suqadd(u: bool, x: u64, d: u64, ebits: u32) -> u64 {
    let mask = width_mask(ebits);
    if u {
        // USQADD: unsigned dst + signed src, saturate to unsigned range.
        let sum = i128::from(d & mask) + i128::from(sx(x, ebits));
        (sum.clamp(0, mask as i128) as u64) & mask
    } else {
        // SUQADD: signed dst + unsigned src, saturate to signed range.
        let (lo, hi) = (-(1i128 << (ebits - 1)), (1i128 << (ebits - 1)) - 1);
        let sum = i128::from(sx(d, ebits)) + i128::from(x & mask);
        (sum.clamp(lo, hi) as u64) & mask
    }
}

fn rbit8(x: u64) -> u64 {
    u64::from((x as u8).reverse_bits())
}
