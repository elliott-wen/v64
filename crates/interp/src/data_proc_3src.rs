//! Data processing (3 source): MADD/MSUB, S/UMADDL, S/UMSUBL, S/UMULH.

use aarch64_cpu_state::CpuState;

use crate::regs::{read, write};

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    op31: u8,
    o0: bool,
    rm: u8,
    ra: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let result = match op31 {
        0b000 => {
            // MADD / MSUB: operands at the instruction width.
            let n = read(cpu, rn, sf, false);
            let m = read(cpu, rm, sf, false);
            let a = read(cpu, ra, sf, false);
            let prod = n.wrapping_mul(m);
            if o0 {
                a.wrapping_sub(prod)
            } else {
                a.wrapping_add(prod)
            }
        }
        0b001 => {
            // SMADDL / SMSUBL: 32-bit signed operands, 64-bit accumulate.
            let n = i64::from(read(cpu, rn, false, false) as i32);
            let m = i64::from(read(cpu, rm, false, false) as i32);
            let a = read(cpu, ra, true, false) as i64;
            let prod = n.wrapping_mul(m);
            (if o0 { a.wrapping_sub(prod) } else { a.wrapping_add(prod) }) as u64
        }
        0b101 => {
            // UMADDL / UMSUBL: 32-bit unsigned operands, 64-bit accumulate.
            let n = u64::from(read(cpu, rn, false, false) as u32);
            let m = u64::from(read(cpu, rm, false, false) as u32);
            let a = read(cpu, ra, true, false);
            let prod = n.wrapping_mul(m);
            if o0 {
                a.wrapping_sub(prod)
            } else {
                a.wrapping_add(prod)
            }
        }
        0b010 => {
            // SMULH: high 64 bits of the signed 128-bit product.
            let n = i128::from(read(cpu, rn, true, false) as i64);
            let m = i128::from(read(cpu, rm, true, false) as i64);
            ((n * m) >> 64) as u64
        }
        0b110 => {
            // UMULH: high 64 bits of the unsigned 128-bit product.
            let n = u128::from(read(cpu, rn, true, false));
            let m = u128::from(read(cpu, rm, true, false));
            ((n * m) >> 64) as u64
        }
        _ => unreachable!(),
    };
    write(cpu, rd, sf, result, false);
    None
}
