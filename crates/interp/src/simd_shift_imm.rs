//! Advanced SIMD shift by immediate: SHL, SSHR/USHR, SSRA/USRA.

use aarch64_cpu_state::CpuState;

use crate::simd::{join, split};

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
    let size = 3 - (immh.leading_zeros() as u8 - 4); // highest set bit of the 4-bit immh
    let esize = 8u32 << size;
    let immhb = (u32::from(immh) << 3) | u32::from(immb);
    let mask = if esize >= 64 { u64::MAX } else { (1u64 << esize) - 1 };

    let la = split(cpu.v[rn as usize], size, q);
    let ld = split(cpu.v[rd as usize], size, q);
    let lanes: Vec<u64> = la
        .iter()
        .zip(&ld)
        .map(|(&x, &d)| lane(opcode, u, x, d, esize, immhb, mask))
        .collect();
    cpu.v[rd as usize] = join(&lanes, size);
    None
}

fn lane(opcode: u8, u: bool, x: u64, d: u64, esize: u32, immhb: u32, mask: u64) -> u64 {
    let sext = |v: u64| {
        let s = 64 - esize;
        ((v << s) as i64) >> s
    };
    match opcode {
        0b01010 => (x << (immhb - esize)) & mask, // SHL
        0b00000 => shr(u, x, sext(x), 2 * esize - immhb, mask), // SSHR/USHR
        _ => {
            // SSRA/USRA: accumulate the shifted value into Vd.
            let s = shr(u, x, sext(x), 2 * esize - immhb, mask);
            d.wrapping_add(s) & mask
        }
    }
}

/// One right shift: logical for unsigned, arithmetic for signed.
fn shr(u: bool, x: u64, sx: i64, sh: u32, mask: u64) -> u64 {
    if u {
        if sh >= 64 {
            0
        } else {
            x >> sh
        }
    } else {
        (sx >> sh.min(63)) as u64 & mask
    }
}
