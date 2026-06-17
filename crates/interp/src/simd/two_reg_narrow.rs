//! Advanced SIMD narrowing two-reg-misc: XTN/SQXTN/UQXTN/SQXTUN. Each `2*esize`
//! source element narrows to `esize`; Q=1 (the `2` form) writes the upper half.

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

/// `n` wide (`wsize`-bit) elements packed across the full 128-bit register.
fn wide_elems(v: u128, wsize: u32, n: u32) -> Vec<u64> {
    let wmask = if wsize >= 128 { u128::MAX } else { (1u128 << wsize) - 1 };
    (0..n).map(|i| ((v >> (i * wsize)) & wmask) as u64).collect()
}

pub(crate) fn xtn(u: bool, opcode: u8, size: u8, q: bool, a: u128, d: u128) -> u128 {
    let esize = 8u32 << size;
    let wsize = esize * 2;
    let n = 64 / esize;
    let emask = width_mask(esize);
    let src = wide_elems(a, wsize, n);

    let mut packed = 0u64;
    for (i, &s) in src.iter().enumerate() {
        let narrowed = match (opcode, u) {
            (0b10010, false) => s & emask,            // XTN: plain truncate
            (0b10010, true) => sqxtun(s, esize, wsize), // SQXTUN
            (0b10100, false) => sqxtn(s, esize, wsize), // SQXTN
            _ => uqxtn(s, esize, wsize),               // UQXTN
        };
        packed |= narrowed << (i as u32 * esize);
    }

    if q {
        (u128::from(packed) << 64) | (d & u128::from(u64::MAX))
    } else {
        u128::from(packed)
    }
}

/// Signed source, signed-saturated to the narrow signed range.
fn sqxtn(s: u64, esize: u32, wsize: u32) -> u64 {
    let v = i128::from(sx(s, wsize));
    let (lo, hi) = (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1);
    (v.clamp(lo, hi) as u64) & width_mask(esize)
}

/// Unsigned source, unsigned-saturated to the narrow unsigned range.
fn uqxtn(s: u64, esize: u32, _wsize: u32) -> u64 {
    let emask = width_mask(esize);
    s.min(emask)
}

/// Signed source, saturated to the narrow *unsigned* range.
fn sqxtun(s: u64, esize: u32, wsize: u32) -> u64 {
    let v = i128::from(sx(s, wsize));
    let emask = width_mask(esize);
    (v.clamp(0, emask as i128) as u64) & emask
}
