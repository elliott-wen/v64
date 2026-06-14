//! Advanced SIMD shift by immediate. Same-width shifts live here; narrowing,
//! widening and fixed-point conversion route to companion modules.

use aarch64_cpu_state::CpuState;

use crate::simd::{join, split};
use crate::{simd_shift_fp, simd_shift_long, simd_shift_narrow};

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
    match opcode {
        0b10000 | 0b10001 | 0b10010 | 0b10011 => {
            return simd_shift_narrow::exec(cpu, q, u, immh, immb, opcode, rn, rd)
        }
        0b10100 => return simd_shift_long::exec(cpu, q, u, immh, immb, rn, rd),
        0b11100 | 0b11111 => return simd_shift_fp::exec(cpu, q, u, immh, immb, opcode, rn, rd),
        _ => {}
    }

    let size = 3 - (immh.leading_zeros() as u8 - 4); // highest set bit of immh
    let esize = 8u32 << size;
    let immhb = (u32::from(immh) << 3) | u32::from(immb);
    let mask = width_mask(esize);

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
    match opcode {
        0b00000 => shr(u, false, x, right_sh(esize, immhb), esize, mask), // SSHR/USHR
        0b00100 => shr(u, true, x, right_sh(esize, immhb), esize, mask),  // SRSHR/URSHR
        0b00010 => d.wrapping_add(shr(u, false, x, right_sh(esize, immhb), esize, mask)) & mask, // SSRA/USRA
        0b00110 => d.wrapping_add(shr(u, true, x, right_sh(esize, immhb), esize, mask)) & mask, // SRSRA/URSRA
        0b01000 => sri(x, d, right_sh(esize, immhb), esize, mask),        // SRI
        0b01010 if !u => (x << left_sh(esize, immhb)) & mask,             // SHL
        0b01010 => sli(x, d, left_sh(esize, immhb), mask),               // SLI
        0b01100 => qshl_imm(QKind::Unsigned, true, x, left_sh(esize, immhb), esize, mask), // SQSHLU
        0b01110 if u => qshl_imm(QKind::Unsigned, false, x, left_sh(esize, immhb), esize, mask), // UQSHL
        0b01110 => qshl_imm(QKind::Signed, true, x, left_sh(esize, immhb), esize, mask), // SQSHL
        _ => 0,
    }
}

fn right_sh(esize: u32, immhb: u32) -> u32 {
    2 * esize - immhb
}
fn left_sh(esize: u32, immhb: u32) -> u32 {
    immhb - esize
}

fn width_mask(esize: u32) -> u64 {
    if esize >= 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    }
}

fn sx(v: u64, esize: u32) -> i64 {
    let s = 64 - esize;
    ((v << s) as i64) >> s
}

/// A right shift by `sh` in [1, esize]; `rounding` adds the round-half term.
fn shr(u: bool, rounding: bool, x: u64, sh: u32, esize: u32, mask: u64) -> u64 {
    let v = if u { i128::from(x & mask) } else { i128::from(sx(x, esize)) };
    let r = if rounding {
        (v + (1i128 << (sh - 1))) >> sh
    } else {
        v >> sh.min(127)
    };
    (r as u64) & mask
}

/// SRI: shift right, inserting the vacated top `sh` bits from Vd.
fn sri(x: u64, d: u64, sh: u32, esize: u32, mask: u64) -> u64 {
    let shifted = if sh >= 64 { 0 } else { (x & mask) >> sh };
    // Preserve the top `sh` bits of d (the bits the shift would otherwise zero).
    let keep = if sh >= esize { mask } else { mask & !(mask >> sh) };
    (shifted | (d & keep)) & mask
}

/// SLI: shift left, inserting the vacated low `sh` bits from Vd.
fn sli(x: u64, d: u64, sh: u32, mask: u64) -> u64 {
    let low = (1u64 << sh) - 1;
    (((x << sh) & mask) | (d & low)) & mask
}

#[derive(Clone, Copy, PartialEq)]
enum QKind {
    Signed,
    Unsigned,
}

/// Saturating left shift by immediate. `signed_src` reads the source as signed;
/// `unsigned_dst` saturates to the unsigned range (SQSHLU/UQSHL).
fn qshl_imm(dst: QKind, signed_src: bool, x: u64, sh: u32, esize: u32, mask: u64) -> u64 {
    // SQSHLU: signed source, unsigned saturate. UQSHL: unsigned source+dst.
    // SQSHL: signed source+dst.
    let v: i128 = if signed_src {
        i128::from(sx(x, esize))
    } else {
        i128::from(x & mask)
    };
    let shifted = v << sh;
    let (lo, hi) = if dst == QKind::Unsigned {
        (0i128, mask as i128)
    } else {
        (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1)
    };
    (shifted.clamp(lo, hi) as u64) & mask
}
