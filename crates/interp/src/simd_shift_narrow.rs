//! Advanced SIMD narrowing shifts: SHRN/RSHRN, SQSHRUN/SQRSHRUN,
//! SQSHRN/UQSHRN, SQRSHRN/UQRSHRN. The `2` form (Q=1) writes the upper half.

use aarch64_cpu_state::CpuState;

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    u: bool,
    immh: u8,
    immb: u8,
    opcode: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let size = 3 - (immh.leading_zeros() as u32 - 4); // narrow element size
    let esize = 8u32 << size;
    let wsize = esize * 2;
    let immhb = (u32::from(immh) << 3) | u32::from(immb);
    let shift = wsize - immhb;
    let round = opcode & 1 == 1;
    let n = 64 / esize;
    let emask = width_mask(esize);

    let src = wide_elems(cpu.v[rn as usize], wsize, n);
    let mut packed = 0u64;
    for (i, &s) in src.iter().enumerate() {
        let r = narrow_lane(opcode, u, s, shift, esize, wsize, round);
        packed |= (r & emask) << (i as u32 * esize);
    }

    let d = cpu.v[rd as usize];
    cpu.v[rd as usize] = if q {
        (u128::from(packed) << 64) | (d & u128::from(u64::MAX))
    } else {
        u128::from(packed)
    };
    None
}

fn narrow_lane(opcode: u8, u: bool, s: u64, shift: u32, esize: u32, wsize: u32, round: bool) -> u64 {
    let roundc = if round { 1i128 << (shift - 1) } else { 0 };
    match (opcode, u) {
        // SHRN/RSHRN: unsigned shift, plain truncation (no saturation).
        (0b10000, false) | (0b10001, false) => {
            let v = i128::from(s & width_mask(wsize));
            ((v + roundc) >> shift) as u64
        }
        // SQSHRUN/SQRSHRUN: signed source, saturate to the unsigned narrow range.
        (0b10000, true) | (0b10001, true) => {
            let v = i128::from(sx(s, wsize));
            sat((v + roundc) >> shift, 0, width_mask(esize) as i128)
        }
        // UQSHRN/UQRSHRN: unsigned source, unsigned narrow.
        (_, true) => {
            let v = i128::from(s & width_mask(wsize));
            sat((v + roundc) >> shift, 0, width_mask(esize) as i128)
        }
        // SQSHRN/SQRSHRN: signed source, signed narrow.
        (_, false) => {
            let v = i128::from(sx(s, wsize));
            let (lo, hi) = (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1);
            sat((v + roundc) >> shift, lo, hi)
        }
    }
}

fn sat(v: i128, lo: i128, hi: i128) -> u64 {
    v.clamp(lo, hi) as u64
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

fn wide_elems(v: u128, wsize: u32, n: u32) -> Vec<u64> {
    let wmask = if wsize >= 128 { u128::MAX } else { (1u128 << wsize) - 1 };
    (0..n).map(|i| ((v >> (i * wsize)) & wmask) as u64).collect()
}
