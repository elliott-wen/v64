//! Advanced SIMD permute: ZIP1/2, UZP1/2, TRN1/2.

use aarch64_cpu_state::CpuState;

use crate::simd::{join, split};

pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    size: u8,
    opcode: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let a = split(cpu.v[rn as usize], size, q);
    let b = split(cpu.v[rm as usize], size, q);
    let n = a.len();
    let half = n / 2;
    let mut r = vec![0u64; n];

    match opcode {
        0b011 => {
            for i in 0..n {
                r[i] = if i % 2 == 0 { a[i / 2] } else { b[i / 2] }; // ZIP1 (low halves)
            }
        }
        0b111 => {
            for i in 0..n {
                r[i] = if i % 2 == 0 { a[half + i / 2] } else { b[half + i / 2] }; // ZIP2
            }
        }
        0b001 => {
            for i in 0..n {
                r[i] = if i < half { a[2 * i] } else { b[2 * (i - half)] }; // UZP1 (evens)
            }
        }
        0b101 => {
            for i in 0..n {
                r[i] = if i < half { a[2 * i + 1] } else { b[2 * (i - half) + 1] }; // UZP2 (odds)
            }
        }
        0b010 => {
            for k in 0..half {
                r[2 * k] = a[2 * k]; // TRN1
                r[2 * k + 1] = b[2 * k];
            }
        }
        _ => {
            for k in 0..half {
                r[2 * k] = a[2 * k + 1]; // TRN2
                r[2 * k + 1] = b[2 * k + 1];
            }
        }
    }
    cpu.v[rd as usize] = join(&r, size);
    None
}
