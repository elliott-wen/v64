//! Advanced SIMD fixed-point convert: SCVTF/UCVTF (fixed -> float) and
//! FCVTZS/FCVTZU (float -> fixed, round toward zero). `fracbits = (16<<size) -
//! immhb`. Only 32- and 64-bit lanes are handled (FP16 omitted).

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
    let size = if immh & 0b1000 != 0 { 3u8 } else { 2 };
    let immhb = (u32::from(immh) << 3) | u32::from(immb);
    let fracbits = (16u32 << size) - immhb;
    let to_fixed = opcode == 0b11111; // FCVTZS/FCVTZU; else SCVTF/UCVTF

    let lanes: Vec<u64> = split(cpu.v[rn as usize], size, q)
        .into_iter()
        .map(|x| {
            if size == 3 {
                lane64(to_fixed, u, x, fracbits)
            } else {
                u64::from(lane32(to_fixed, u, x as u32, fracbits))
            }
        })
        .collect();
    cpu.v[rd as usize] = join(&lanes, size);
    None
}

fn lane32(to_fixed: bool, u: bool, x: u32, fracbits: u32) -> u32 {
    if to_fixed {
        let f = f32::from_bits(x);
        let scaled = f * 2f32.powi(fracbits as i32);
        if u {
            scaled as u32
        } else {
            (scaled as i32) as u32
        }
    } else {
        let scale = 2f32.powi(-(fracbits as i32));
        let f = if u { (x as f32) * scale } else { (x as i32 as f32) * scale };
        f.to_bits()
    }
}

fn lane64(to_fixed: bool, u: bool, x: u64, fracbits: u32) -> u64 {
    if to_fixed {
        let f = f64::from_bits(x);
        let scaled = f * 2f64.powi(fracbits as i32);
        if u {
            scaled as u64
        } else {
            (scaled as i64) as u64
        }
    } else {
        let scale = 2f64.powi(-(fracbits as i32));
        let f = if u { (x as f64) * scale } else { (x as i64 as f64) * scale };
        f.to_bits()
    }
}
