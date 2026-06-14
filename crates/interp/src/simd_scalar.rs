//! Advanced SIMD scalar families. Each reuses the vector arithmetic on lane 0
//! and then zeroes everything above the single result element.

use aarch64_cpu_state::CpuState;

use crate::fp::{canon_d, canon_s, fmax_d, fmax_s, fmaxnm_d, fmaxnm_s, fmin_d, fmin_s, fminnm_d, fminnm_s};
use crate::{
    simd_indexed, simd_shift_imm, simd_three_diff, simd_three_same, simd_three_same_fp,
    simd_two_reg_misc, simd_two_reg_misc_fp,
};

/// Zero every bit above the low `bits` of Vd.
fn keep_low(cpu: &mut CpuState, rd: u8, bits: u32) {
    let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
    cpu.v[rd as usize] &= mask;
}

pub(crate) fn three_same(
    cpu: &mut CpuState,
    u: bool,
    size: u8,
    opcode: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    if opcode >= 0x18 {
        let fpopcode = opcode | ((size >> 1) << 5) | (u8::from(u) << 6);
        let sz = size & 1 == 1;
        simd_three_same_fp::exec(cpu, false, sz, fpopcode, rm, rn, rd);
        keep_low(cpu, rd, if sz { 64 } else { 32 });
    } else {
        simd_three_same::exec(cpu, false, u, size, opcode, rm, rn, rd);
        keep_low(cpu, rd, 8u32 << size);
    }
    None
}

pub(crate) fn two_reg_misc(
    cpu: &mut CpuState,
    u: bool,
    size: u8,
    opcode: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    if matches!(opcode, 0xc..=0xf | 0x16..=0x1f) {
        let is_double = size & 1 == 1;
        let fpop = opcode | ((size >> 1) << 5) | (u8::from(u) << 6);
        simd_two_reg_misc_fp::exec(cpu, false, is_double, fpop, rn, rd);
        keep_low(cpu, rd, if is_double { 64 } else { 32 });
    } else {
        simd_two_reg_misc::exec(cpu, false, u, size, opcode, rn, rd);
        keep_low(cpu, rd, 8u32 << size);
    }
    None
}

pub(crate) fn pairwise(
    cpu: &mut CpuState,
    u: bool,
    size: u8,
    opcode: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let v = cpu.v[rn as usize];
    let result = if !u {
        // ADDP (D): add the two 64-bit lanes.
        u128::from((v as u64).wrapping_add((v >> 64) as u64))
    } else {
        let full = opcode | ((size >> 1) << 5);
        let sz = size & 1 == 1;
        if sz {
            let (a, b) = (f64::from_bits(v as u64), f64::from_bits((v >> 64) as u64));
            u128::from(fp_pair_d(full, a, b))
        } else {
            let (a, b) = (f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32));
            u128::from(fp_pair_s(full, a, b))
        }
    };
    cpu.v[rd as usize] = result;
    None
}

fn fp_pair_s(full: u8, a: f32, b: f32) -> u32 {
    match full {
        0xd => canon_s(a + b),          // FADDP
        0xf => canon_s(fmax_s(a, b)),   // FMAXP
        0x2f => canon_s(fmin_s(a, b)),  // FMINP
        0xc => canon_s(fmaxnm_s(a, b)), // FMAXNMP
        _ => canon_s(fminnm_s(a, b)),   // FMINNMP (0x2c)
    }
}
fn fp_pair_d(full: u8, a: f64, b: f64) -> u64 {
    match full {
        0xd => canon_d(a + b),
        0xf => canon_d(fmax_d(a, b)),
        0x2f => canon_d(fmin_d(a, b)),
        0xc => canon_d(fmaxnm_d(a, b)),
        _ => canon_d(fminnm_d(a, b)),
    }
}

pub(crate) fn three_diff(
    cpu: &mut CpuState,
    size: u8,
    opcode: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    simd_three_diff::exec(cpu, false, false, size, opcode, rm, rn, rd);
    keep_low(cpu, rd, 2 * (8u32 << size)); // wide result element
    None
}

pub(crate) fn copy(cpu: &mut CpuState, imm5: u8, rn: u8, rd: u8) -> Option<u64> {
    let size = (imm5 & 0xf).trailing_zeros(); // 0..3
    let esize = 8u32 << size;
    let index = u32::from(imm5) >> (size + 1);
    let mask = if esize >= 128 { u128::MAX } else { (1u128 << esize) - 1 };
    let elem = (cpu.v[rn as usize] >> (index * esize)) & mask;
    cpu.v[rd as usize] = elem;
    None
}

pub(crate) fn indexed(
    cpu: &mut CpuState,
    u: bool,
    size: u8,
    opcode: u8,
    index: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    simd_indexed::exec(cpu, false, u, size, opcode, index, rm, rn, rd);
    let key = 16 * u8::from(u) + opcode;
    let bits = match key {
        0x03 | 0x07 | 0x0b => 2 * (8u32 << size),     // long: SQDMLAL/SQDMLSL/SQDMULL
        0x0c | 0x0d => 8u32 << size,                  // SQDMULH/SQRDMULH
        _ => if size == 3 { 64 } else { 32 },         // FP FMLA/FMLS/FMUL/FMULX
    };
    keep_low(cpu, rd, bits);
    None
}

pub(crate) fn shift_imm(
    cpu: &mut CpuState,
    u: bool,
    immh: u8,
    immb: u8,
    opcode: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    simd_shift_imm::exec(cpu, false, u, immh, immb, opcode, rn, rd);
    let size = 3 - (immh.leading_zeros() - 4); // highest set bit
    keep_low(cpu, rd, 8u32 << size);
    None
}
