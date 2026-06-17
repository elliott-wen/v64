//! Advanced SIMD three-different: widening (L), wide (W) and high-narrowing (HN).
//!
//! L ops widen two `esize` source operands to a `2*esize` result; W ops widen
//! only the second operand; HN ops add/subtract two `2*esize` operands and keep
//! the upper `esize` half. The `2` variants (Q=1) take/produce the upper half.

use aarch64_cpu_state::CpuState;

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
    let esize = 8u32 << size; // narrow (source) element width
    let wsize = esize * 2; // wide (result) element width
    let n = 64 / esize; // element count
    let (vn, vm, vd) = (cpu.v[rn as usize], cpu.v[rm as usize], cpu.v[rd as usize]);

    let result = match opcode {
        0b0100 | 0b0110 => narrowing(opcode, u, esize, wsize, n, q, vn, vm, vd),
        _ => widening(opcode, u, esize, wsize, n, q, vn, vm, vd),
    };
    cpu.v[rd as usize] = result;
    None
}

/// L and W forms: produce `n` wide elements filling the full 128-bit result.
#[allow(clippy::too_many_arguments)]
fn widening(opcode: u8, u: bool, esize: u32, wsize: u32, n: u32, q: bool, vn: u128, vm: u128, vd: u128) -> u128 {
    let wmask = width_mask(wsize);
    let an = source_half(vn, esize, n, q);
    let bn = source_half(vm, esize, n, q);
    let aw = wide_elems(vn, wsize, n); // full-width Vn for the W forms
    let dw = wide_elems(vd, wsize, n); // accumulator for *AL / *MLAL / *MLSL

    let mut out = 0u128;
    for i in 0..n as usize {
        let r = wide_op(opcode, u, esize, wsize, an[i], bn[i], aw[i], dw[i]);
        out |= u128::from(r & wmask) << (i as u32 * wsize);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn wide_op(opcode: u8, u: bool, esize: u32, wsize: u32, a: u64, b: u64, awide: u64, dwide: u64) -> u64 {
    let (lo, hi) = signed_range(wsize);
    let ea = ext(u, a, esize);
    let eb = ext(u, b, esize);
    match opcode {
        0b0000 => (ea + eb) as u64,                       // ADDL
        0b0010 => (ea - eb) as u64,                       // SUBL
        0b0001 => (i128::from(sx(awide, wsize)).wrapping_add(eb)) as u64, // ADDW (signed view ok: masked)
        0b0011 => (i128::from(sx(awide, wsize)).wrapping_sub(eb)) as u64, // SUBW
        0b0111 => (ea - eb).unsigned_abs() as u64,        // ABDL
        0b0101 => (i128::from(sx(dwide, wsize)).wrapping_add((ea - eb).unsigned_abs() as i128)) as u64, // ABAL
        0b1100 => (ea * eb) as u64,                        // MULL
        0b1000 => (i128::from(sx(dwide, wsize)) + ea * eb) as u64, // MLAL
        0b1010 => (i128::from(sx(dwide, wsize)) - ea * eb) as u64, // MLSL
        0b1101 => ((2 * ea * eb).clamp(lo, hi)) as u64,   // SQDMULL
        0b1001 => {
            let p = (2 * ea * eb).clamp(lo, hi);
            (i128::from(sx(dwide, wsize)) + p).clamp(lo, hi) as u64 // SQDMLAL
        }
        0b1011 => {
            let p = (2 * ea * eb).clamp(lo, hi);
            (i128::from(sx(dwide, wsize)) - p).clamp(lo, hi) as u64 // SQDMLSL
        }
        0b1110 => poly_mul(a & 0xff, b & 0xff), // PMULL (8 -> 16)
        _ => 0,
    }
}

/// HN forms: keep the upper `esize` half of a `2*esize` add/subtract. Q=1 writes
/// the upper half of Vd, preserving the lower half (and vice versa).
#[allow(clippy::too_many_arguments)]
fn narrowing(opcode: u8, round: bool, esize: u32, wsize: u32, n: u32, q: bool, vn: u128, vm: u128, vd: u128) -> u128 {
    let aw = wide_elems(vn, wsize, n);
    let bw = wide_elems(vm, wsize, n);
    let emask = width_mask(esize);
    let wmask = width_mask(wsize);
    let round_const = if round { 1u64 << (esize - 1) } else { 0 };

    let mut packed = 0u64;
    for i in 0..n as usize {
        let t = if opcode == 0b0100 {
            aw[i].wrapping_add(bw[i]) & wmask // ADDHN/RADDHN
        } else {
            aw[i].wrapping_sub(bw[i]) & wmask // SUBHN/RSUBHN
        };
        let narrowed = (t.wrapping_add(round_const) >> esize) & emask;
        packed |= narrowed << (i as u32 * esize);
    }

    if q {
        (u128::from(packed) << 64) | (vd & u128::from(u64::MAX))
    } else {
        u128::from(packed)
    }
}

/// The `n` narrow source elements from the low half (Q=0) or high half (Q=1).
fn source_half(v: u128, esize: u32, n: u32, q: bool) -> Vec<u64> {
    let half = if q { (v >> 64) as u64 } else { v as u64 };
    (0..n).map(|i| (half >> (i * esize)) & width_mask(esize)).collect()
}

/// The `n` wide elements packed across the full 128-bit register.
fn wide_elems(v: u128, wsize: u32, n: u32) -> Vec<u64> {
    let wmask = if wsize >= 128 { u128::MAX } else { (1u128 << wsize) - 1 };
    (0..n).map(|i| ((v >> (i * wsize)) & wmask) as u64).collect()
}

fn width_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn signed_range(bits: u32) -> (i128, i128) {
    (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
}

/// Sign-extend a `bits`-wide value to i64.
fn sx(v: u64, bits: u32) -> i64 {
    let s = 64 - bits;
    ((v << s) as i64) >> s
}

/// Widen a narrow source element: zero (unsigned) or sign extension.
fn ext(u: bool, x: u64, bits: u32) -> i128 {
    if u {
        i128::from(x & width_mask(bits))
    } else {
        i128::from(sx(x, bits))
    }
}

/// 8x8 -> 16-bit carry-less (polynomial) multiply.
fn poly_mul(x: u64, y: u64) -> u64 {
    let mut r = 0u64;
    for i in 0..8 {
        if (y >> i) & 1 == 1 {
            r ^= x << i;
        }
    }
    r & 0xffff
}
