//! Advanced SIMD widening shift-left-long: SSHLL/USHLL (and the SXTL/UXTL alias
//! when the shift is zero). Source elements come from the low half (Q=0) or the
//! upper half (Q=1).

use aarch64_cpu_state::CpuState;

pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    u: bool,
    immh: u8,
    immb: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let size = 3 - (immh.leading_zeros() as u32 - 4);
    let esize = 8u32 << size;
    let wsize = esize * 2;
    let immhb = (u32::from(immh) << 3) | u32::from(immb);
    let shift = immhb - esize;
    let n = 64 / esize;
    let wmask = width_mask(wsize);

    let half = if q { (cpu.v[rn as usize] >> 64) as u64 } else { cpu.v[rn as usize] as u64 };
    let mut out = 0u128;
    for i in 0..n {
        let elem = (half >> (i * esize)) & width_mask(esize);
        let val = if u {
            i128::from(elem)
        } else {
            i128::from(sx(elem, esize))
        };
        let r = ((val << shift) as u64) & wmask;
        out |= u128::from(r) << (i * wsize);
    }
    cpu.v[rd as usize] = out;
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
