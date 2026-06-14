//! Advanced SIMD two-register misc: REV/CLS/CLZ/CNT/NOT/RBIT/ABS/NEG.

use aarch64_cpu_state::CpuState;

use crate::simd::{join, split};

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
    let mask128 = if q { u128::MAX } else { u128::from(u64::MAX) };

    let result = match (u, opcode) {
        (false, 0b00000) => rev(a, size, q, 64), // REV64
        (false, 0b00001) => rev(a, size, q, 16), // REV16
        (true, 0b00000) => rev(a, size, q, 32),  // REV32
        (true, 0b00101) if size == 0 => !a & mask128, // NOT
        (true, 0b00101) => map(a, 0, q, rbit8) & mask128, // RBIT: always per-byte
        _ => {
            // Element-wise unary ops.
            map(a, size, q, |x| lane(u, opcode, size, x))
        }
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

/// Apply a per-lane function across the register.
fn map(val: u128, size: u8, q: bool, f: impl Fn(u64) -> u64) -> u128 {
    let lanes: Vec<u64> = split(val, size, q).into_iter().map(f).collect();
    join(&lanes, size)
}

fn lane(u: bool, opcode: u8, size: u8, x: u64) -> u64 {
    let ebits = 8u32 << size;
    let mask = if ebits >= 64 { u64::MAX } else { (1u64 << ebits) - 1 };
    match (u, opcode) {
        (false, 0b00100) => cls(x, ebits),          // CLS
        // CLZ: cap at the element width (a zero element gives `ebits`, not 64).
        (true, 0b00100) => u64::from((x << (64 - ebits)).leading_zeros()).min(u64::from(ebits)),
        (false, 0b00101) => u64::from((x as u8).count_ones()), // CNT (byte)
        (false, 0b01011) => abs(x, ebits) & mask,    // ABS
        (true, 0b01011) => x.wrapping_neg() & mask,  // NEG
        _ => 0,
    }
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
    let s = 64 - ebits;
    let v = ((x << s) as i64) >> s;
    v.unsigned_abs()
}

fn rbit8(x: u64) -> u64 {
    u64::from((x as u8).reverse_bits())
}
