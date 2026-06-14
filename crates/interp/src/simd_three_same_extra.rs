//! Advanced SIMD three-same extra: SQRDMLAH/SQRDMLSH and SDOT/UDOT.

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
    let (vn, vm, vd) = (cpu.v[rn as usize], cpu.v[rm as usize], cpu.v[rd as usize]);
    let result = if opcode == 2 {
        dot(u, q, vn, vm, vd) // SDOT / UDOT
    } else {
        // SQRDMLAH (opcode 0) / SQRDMLSH (opcode 1).
        let sub = opcode == 1;
        let la = split(vn, size, q);
        let lb = split(vm, size, q);
        let ld = split(vd, size, q);
        let ebits = 8u32 << size;
        let lanes: Vec<u64> =
            (0..la.len()).map(|i| sqrdmlah(sub, la[i], lb[i], ld[i], ebits)).collect();
        join(&lanes, size)
    };
    cpu.v[rd as usize] = result;
    None
}

fn sx(v: u64, ebits: u32) -> i64 {
    let s = 64 - ebits;
    ((v << s) as i64) >> s
}

/// Signed saturating rounding doubling multiply-(accumulate|subtract) high.
fn sqrdmlah(sub: bool, n: u64, m: u64, a: u64, ebits: u32) -> u64 {
    let prod = i128::from(sx(n, ebits)) * i128::from(sx(m, ebits));
    let acc = i128::from(sx(a, ebits)) << (ebits - 1);
    let round = 1i128 << (ebits - 2);
    let mut ret = if sub { acc - prod + round } else { acc + prod + round };
    ret >>= ebits - 1;
    let (lo, hi) = (-(1i128 << (ebits - 1)), (1i128 << (ebits - 1)) - 1);
    let mask = if ebits >= 64 { u64::MAX } else { (1u64 << ebits) - 1 };
    (ret.clamp(lo, hi) as u64) & mask
}

/// SDOT/UDOT: 4-byte dot product accumulated into each 32-bit lane of Vd.
fn dot(u: bool, q: bool, vn: u128, vm: u128, vd: u128) -> u128 {
    let nbytes = vn.to_le_bytes();
    let mbytes = vm.to_le_bytes();
    let lanes = if q { 4 } else { 2 };
    let mut out = vd;
    for i in 0..lanes {
        let mut acc = ((vd >> (i * 32)) & u128::from(u32::MAX)) as u32;
        for k in 0..4 {
            let (nb, mb) = (nbytes[i * 4 + k], mbytes[i * 4 + k]);
            let p = if u {
                u32::from(nb).wrapping_mul(u32::from(mb))
            } else {
                (i32::from(nb as i8) * i32::from(mb as i8)) as u32
            };
            acc = acc.wrapping_add(p);
        }
        out &= !(u128::from(u32::MAX) << (i * 32));
        out |= u128::from(acc) << (i * 32);
    }
    // Q=0 zeroes the upper 64 bits.
    if q {
        out
    } else {
        out & u128::from(u64::MAX)
    }
}
