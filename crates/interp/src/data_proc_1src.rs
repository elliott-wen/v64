//! Data processing (1 source): RBIT / REV16 / REV32 / REV / CLZ / CLS.

use aarch64_cpu_state::CpuState;

use crate::regs::{datasize, read, write};

/// Reverse the order of bytes within each `group`-byte chunk of the low `ds`
/// bits.
fn rev_groups(val: u64, ds: u32, group_bytes: u32) -> u64 {
    let bytes = (ds / 8) as usize;
    let g = group_bytes as usize;
    let src = val.to_le_bytes();
    let mut out = [0u8; 8];
    for base in (0..bytes).step_by(g) {
        for i in 0..g {
            out[base + i] = src[base + g - 1 - i];
        }
    }
    u64::from_le_bytes(out)
}

/// Count leading sign bits (excluding the sign bit itself) within `ds` bits.
fn cls(val: u64, ds: u32) -> u64 {
    let sign = (val >> (ds - 1)) & 1;
    let mut count = 0u64;
    let mut i = ds - 1;
    while i > 0 {
        i -= 1;
        if (val >> i) & 1 == sign {
            count += 1;
        } else {
            break;
        }
    }
    count
}

pub(crate) fn exec(cpu: &mut CpuState, sf: bool, opcode: u8, rn: u8, rd: u8) -> Option<u64> {
    let ds = datasize(sf);
    let v = read(cpu, rn, sf, false);

    let result = match opcode {
        0 => {
            // RBIT: reverse all bits within ds.
            if sf {
                v.reverse_bits()
            } else {
                u64::from((v as u32).reverse_bits())
            }
        }
        1 => rev_groups(v, ds, 2),  // REV16
        2 => rev_groups(v, ds, 4),  // REV32 (64-bit) / REV (32-bit)
        3 => rev_groups(v, ds, 8),  // REV (64-bit)
        4 => u64::from(if sf { v.leading_zeros() } else { (v as u32).leading_zeros() }), // CLZ
        5 => cls(v, ds), // CLS
        _ => unreachable!(),
    };
    write(cpu, rd, sf, result, false);
    None
}
