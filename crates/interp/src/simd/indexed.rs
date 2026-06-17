//! Advanced SIMD vector x indexed element: one operand is a single broadcast
//! element of Vm. Covers MUL/MLA/MLS, the widening MLAL/MLSL/MULL (S/U),
//! SQDMULL/SQDMLAL/SQDMLSL, SQDMULH/SQRDMULH, and FP FMLA/FMLS/FMUL/FMULX.

use aarch64_cpu_state::CpuState;

use crate::fp::{canon_d, canon_s, mulx_d, mulx_s};
use crate::simd::{join, split};

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    u: bool,
    size: u8,
    opcode: u8,
    index: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let key = 16 * u8::from(u) + opcode;
    let (vn, vm, vd) = (cpu.v[rn as usize], cpu.v[rm as usize], cpu.v[rd as usize]);
    let result = match key {
        0x01 | 0x05 | 0x09 | 0x19 => fp_indexed(key, size, q, index, vn, vm, vd),
        0x08 | 0x10 | 0x14 | 0x0c | 0x0d => int_normal(key, size, q, index, vn, vm, vd),
        0x1d | 0x1f => rdmlah_indexed(key, size, q, index, vn, vm, vd),
        0x0e | 0x1e => dot_indexed(key, q, index, vn, vm, vd),
        _ => int_long(key, u, size, q, index, vn, vm, vd),
    };
    cpu.v[rd as usize] = result;
    None
}

fn width_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn sx(v: u64, bits: u32) -> i64 {
    let s = 64 - bits;
    ((v << s) as i64) >> s
}

fn ext(u: bool, x: u64, bits: u32) -> i128 {
    if u {
        i128::from(x & width_mask(bits))
    } else {
        i128::from(sx(x, bits))
    }
}

/// The `index`-th `esize`-bit element of `v`.
fn elem(v: u128, esize: u32, index: u8) -> u64 {
    let m = if esize >= 64 { u128::MAX } else { (1u128 << esize) - 1 };
    ((v >> (u32::from(index) * esize)) & m) as u64
}

fn int_normal(key: u8, size: u8, q: bool, index: u8, vn: u128, vm: u128, vd: u128) -> u128 {
    let esize = 8u32 << size;
    let mask = width_mask(esize);
    let idx = elem(vm, esize, index);
    let n = (if q { 128 } else { 64 }) / esize;

    let mut out = 0u128;
    for i in 0..n {
        let a = ((vn >> (i * esize)) & u128::from(mask)) as u64;
        let d = ((vd >> (i * esize)) & u128::from(mask)) as u64;
        let r = match key {
            0x08 => a.wrapping_mul(idx) & mask,                 // MUL
            0x10 => d.wrapping_add(a.wrapping_mul(idx)) & mask, // MLA
            0x14 => d.wrapping_sub(a.wrapping_mul(idx)) & mask, // MLS
            0x0c => sqdmulh(false, a, idx, esize),              // SQDMULH
            _ => sqdmulh(true, a, idx, esize),                  // SQRDMULH
        };
        out |= u128::from(r) << (i * esize);
    }
    out
}

fn sqdmulh(rounding: bool, x: u64, y: u64, esize: u32) -> u64 {
    let prod = 2 * i128::from(sx(x, esize)) * i128::from(sx(y, esize));
    let round = if rounding { 1i128 << (esize - 1) } else { 0 };
    let shifted = (prod + round) >> esize;
    let (lo, hi) = (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1);
    (shifted.clamp(lo, hi) as u64) & width_mask(esize)
}

fn int_long(key: u8, u: bool, size: u8, q: bool, index: u8, vn: u128, vm: u128, vd: u128) -> u128 {
    let esize = 8u32 << size;
    let wsize = esize * 2;
    let n = 64 / esize;
    let wmask = width_mask(wsize);
    let (lo, hi) = (-(1i128 << (wsize - 1)), (1i128 << (wsize - 1)) - 1);
    let idx = ext(u, elem(vm, esize, index), esize);
    let half = if q { (vn >> 64) as u64 } else { vn as u64 };

    let mut out = 0u128;
    for i in 0..n {
        let src = (half >> (i * esize)) & width_mask(esize);
        let prod = ext(u, src, esize) * idx;
        let dwide = ((vd >> (i * wsize)) & u128::from(wmask)) as u64;
        let r = match key {
            0x0a | 0x1a => prod,                                        // SMULL/UMULL
            0x02 | 0x12 => i128::from(sx_w(dwide, wsize)) + prod,       // SMLAL/UMLAL
            0x06 | 0x16 => i128::from(sx_w(dwide, wsize)) - prod,       // SMLSL/UMLSL
            0x0b => (2 * prod).clamp(lo, hi),                           // SQDMULL
            0x03 => (i128::from(sx_w(dwide, wsize)) + (2 * prod).clamp(lo, hi)).clamp(lo, hi), // SQDMLAL
            _ => (i128::from(sx_w(dwide, wsize)) - (2 * prod).clamp(lo, hi)).clamp(lo, hi),    // SQDMLSL
        };
        out |= u128::from((r as u64) & wmask) << (i * wsize);
    }
    out
}

fn sx_w(v: u64, bits: u32) -> i64 {
    sx(v, bits)
}

/// SQRDMLAH/SQRDMLSH by indexed element.
fn rdmlah_indexed(key: u8, size: u8, q: bool, index: u8, vn: u128, vm: u128, vd: u128) -> u128 {
    let esize = 8u32 << size;
    let m = elem(vm, esize, index);
    let n = (if q { 128 } else { 64 }) / esize;
    let sub = key == 0x1f;
    let mut out = 0u128;
    for i in 0..n {
        let nn = ((vn >> (i * esize)) & u128::from(width_mask(esize))) as u64;
        let a = ((vd >> (i * esize)) & u128::from(width_mask(esize))) as u64;
        out |= u128::from(sqrdmlah(sub, nn, m, a, esize)) << (i * esize);
    }
    out
}

fn sqrdmlah(sub: bool, n: u64, m: u64, a: u64, ebits: u32) -> u64 {
    let prod = i128::from(sx(n, ebits)) * i128::from(sx(m, ebits));
    let acc = i128::from(sx(a, ebits)) << (ebits - 1);
    let round = 1i128 << (ebits - 2);
    let mut ret = if sub { acc - prod + round } else { acc + prod + round };
    ret >>= ebits - 1;
    let (lo, hi) = (-(1i128 << (ebits - 1)), (1i128 << (ebits - 1)) - 1);
    (ret.clamp(lo, hi) as u64) & width_mask(ebits)
}

/// SDOT/UDOT by indexed element: the 4 bytes of the indexed 32-bit element of Vm
/// dot each 4-byte group of Vn into Vd's 32-bit lanes.
fn dot_indexed(key: u8, q: bool, index: u8, vn: u128, vm: u128, vd: u128) -> u128 {
    let unsigned = key == 0x1e;
    let nbytes = vn.to_le_bytes();
    let mbytes = vm.to_le_bytes();
    let lanes = if q { 4 } else { 2 };
    let mut out = vd;
    for i in 0..lanes {
        let mut acc = ((vd >> (i * 32)) & u128::from(u32::MAX)) as u32;
        for k in 0..4 {
            let nb = nbytes[i * 4 + k];
            let mb = mbytes[usize::from(index) * 4 + k];
            let p = if unsigned {
                u32::from(nb).wrapping_mul(u32::from(mb))
            } else {
                (i32::from(nb as i8) * i32::from(mb as i8)) as u32
            };
            acc = acc.wrapping_add(p);
        }
        out &= !(u128::from(u32::MAX) << (i * 32));
        out |= u128::from(acc) << (i * 32);
    }
    if q {
        out
    } else {
        out & u128::from(u64::MAX)
    }
}

fn fp_indexed(key: u8, size: u8, q: bool, index: u8, vn: u128, vm: u128, vd: u128) -> u128 {
    let dbl = size == 3;
    let lane_size = if dbl { 3u8 } else { 2 };
    let idx = elem(vm, if dbl { 64 } else { 32 }, index);

    let la = split(vn, lane_size, q);
    let ld = split(vd, lane_size, q);
    let lanes: Vec<u64> = (0..la.len())
        .map(|i| {
            if dbl {
                lane_d(key, la[i], idx, ld[i])
            } else {
                u64::from(lane_s(key, la[i] as u32, idx as u32, ld[i] as u32))
            }
        })
        .collect();
    join(&lanes, lane_size)
}

fn lane_s(key: u8, xb: u32, vb: u32, db: u32) -> u32 {
    let a = f32::from_bits(xb);
    let v = f32::from_bits(vb);
    let d = f32::from_bits(db);
    match key {
        0x09 => canon_s(a * v),               // FMUL
        0x01 => canon_s(a.mul_add(v, d)),     // FMLA
        0x05 => canon_s((-a).mul_add(v, d)),  // FMLS
        _ => canon_s(mulx_s(a, v)),           // FMULX
    }
}

fn lane_d(key: u8, xb: u64, vb: u64, db: u64) -> u64 {
    let a = f64::from_bits(xb);
    let v = f64::from_bits(vb);
    let d = f64::from_bits(db);
    match key {
        0x09 => canon_d(a * v),
        0x01 => canon_d(a.mul_add(v, d)),
        0x05 => canon_d((-a).mul_add(v, d)),
        _ => canon_d(mulx_d(a, v)),
    }
}

