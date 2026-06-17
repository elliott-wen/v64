//! Advanced SIMD widening two-reg-misc: SHLL (shift-left-long by element size)
//! and the pairwise long adds SADDLP/UADDLP/SADALP/UADALP.

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

/// `n` narrow elements of the selected 64-bit half (Q=1 -> upper half).
fn half_elems(v: u128, esize: u32, n: u32, q: bool) -> Vec<u64> {
    let half = if q { (v >> 64) as u64 } else { v as u64 };
    (0..n).map(|i| (half >> (i * esize)) & width_mask(esize)).collect()
}

/// SHLL/SHLL2: widen each source element and shift left by its width.
pub(crate) fn shll(size: u8, q: bool, a: u128) -> u128 {
    let esize = 8u32 << size;
    let wsize = esize * 2;
    let n = 64 / esize;
    let src = half_elems(a, esize, n, q);

    let mut out = 0u128;
    for (i, &s) in src.iter().enumerate() {
        let widened = u128::from(s) << esize; // value << esize fits in wsize bits
        out |= widened << (i as u32 * wsize);
    }
    out
}

/// SADDLP/UADDLP (opcode 0x02) and the accumulating SADALP/UADALP (0x06).
pub(crate) fn addlp(u: bool, opcode: u8, size: u8, q: bool, a: u128, d: u128) -> u128 {
    let esize = 8u32 << size;
    let wsize = esize * 2;
    let datasize = if q { 128u32 } else { 64 };
    let n_src = datasize / esize;
    let n_res = n_src / 2;
    let wmask = width_mask(wsize);
    let accumulate = opcode == 0b00110;

    let src: Vec<u64> = (0..n_src).map(|i| ((a >> (i * esize)) & u128::from(width_mask(esize))) as u64).collect();

    let mut out = 0u128;
    for i in 0..n_res as usize {
        let pair = ext(u, src[2 * i], esize) + ext(u, src[2 * i + 1], esize);
        let mut r = (pair as u64) & wmask;
        if accumulate {
            let acc = ((d >> (i as u32 * wsize)) & u128::from(wmask)) as u64;
            r = r.wrapping_add(acc) & wmask;
        }
        out |= u128::from(r) << (i as u32 * wsize);
    }
    out
}
