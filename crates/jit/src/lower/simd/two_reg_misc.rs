//! SIMD two-register-misc (integer), the bit-exact same-width subset: NOT, CNT,
//! NEG, ABS, compare-to-zero (CMGT/CMGE/CMEQ/CMLE/CMLT #0), and REV64/16/32.
//!
//! CLS/CLZ (no WASM vector count-leading), RBIT, the saturating ops
//! (SQABS/SQNEG/SUQADD/USQADD), and the shape-changing forms (XTN/SHLL/ADDLP)
//! fall back.

use wasm_encoder::{Function, Instruction as I};

use super::{finish_v, push_v, shuffle2};

/// `Rd = op(Vn)` for a one-operand lane op.
fn unary(f: &mut Function, q: bool, rn: u8, rd: u8, op: I<'static>) {
    emit!(f, I::LocalGet(0)); // regs_base for the store
    push_v(f, rn);
    emit!(f, op);
    finish_v(f, q, rd);
}

/// `Rd = cmp(Vn, 0)` — compare each lane against zero, leaving an all-ones/zero
/// mask (matches the interpreter's `bool_lane`).
fn cmp_zero(f: &mut Function, q: bool, rn: u8, rd: u8, op: I<'static>) {
    emit!(f, I::LocalGet(0));
    push_v(f, rn);
    emit!(f, I::V128Const(0), op);
    finish_v(f, q, rd);
}

/// Whether [`simd_two_reg_misc`] handles this `(u, opcode)` form (NOT is size-0
/// only). Mirrors the dispatch below; the eligibility gate calls it.
pub(super) fn can_lower(u: bool, size: u8, opcode: u8) -> bool {
    match (u, opcode) {
        (true, 0b00101) => size == 0, // NOT
        (false, 0b00101) // CNT
        | (true, 0b01011) // NEG
        | (false, 0b01011) // ABS
        | (false, 0b01000) // CMGT #0
        | (true, 0b01000) // CMGE #0
        | (false, 0b01001) // CMEQ #0
        | (true, 0b01001) // CMLE #0
        | (false, 0b01010) // CMLT #0
        | (false, 0b00000) // REV64
        | (false, 0b00001) // REV16
        | (true, 0b00000) => true, // REV32
        _ => false,
    }
}

pub(crate) fn simd_two_reg_misc(f: &mut Function, q: bool, u: bool, size: u8, opcode: u8, rn: u8, rd: u8) -> bool {
    match (u, opcode) {
        (true, 0b00101) if size == 0 => unary(f, q, rn, rd, I::V128Not), // NOT
        (false, 0b00101) => unary(f, q, rn, rd, I::I8x16Popcnt),         // CNT (per byte)
        (true, 0b01011) => unary(f, q, rn, rd, neg(size)),              // NEG
        (false, 0b01011) => unary(f, q, rn, rd, abs(size)),             // ABS
        (false, 0b01000) => cmp_zero(f, q, rn, rd, gt_s(size)),         // CMGT #0
        (true, 0b01000) => cmp_zero(f, q, rn, rd, ge_s(size)),          // CMGE #0
        (false, 0b01001) => cmp_zero(f, q, rn, rd, eq(size)),           // CMEQ #0
        (true, 0b01001) => cmp_zero(f, q, rn, rd, le_s(size)),          // CMLE #0
        (false, 0b01010) => cmp_zero(f, q, rn, rd, lt_s(size)),         // CMLT #0
        (false, 0b00000) => rev(f, q, size, 8, rn, rd),                 // REV64
        (false, 0b00001) => rev(f, q, size, 2, rn, rd),                 // REV16
        (true, 0b00000) => rev(f, q, size, 4, rn, rd),                  // REV32
        _ => return false, // RBIT, CLS/CLZ, SQABS/SQNEG, SUQADD, XTN/SHLL/ADDLP
    }
    true
}

/// REVxx: reverse the order of `1 << size`-byte elements within each
/// `container`-byte group, via a constant self-shuffle.
fn rev(f: &mut Function, q: bool, size: u8, container: usize, rn: u8, rd: u8) {
    let esize = 1usize << size;
    let per = container / esize; // elements per container
    let nbytes = if q { 16 } else { 8 };
    let mut lanes = [0u8; 16];
    for (j, lane) in lanes.iter_mut().take(nbytes).enumerate() {
        let ei = j / esize; // element index of this byte
        let base = (ei / per) * per; // first element of its container
        let src_elem = base + (per - 1 - (ei - base)); // reversed within container
        *lane = (src_elem * esize + (j % esize)) as u8;
    }
    shuffle2(f, q, rn, rn, rd, lanes);
}

fn neg(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16Neg,
        1 => I::I16x8Neg,
        2 => I::I32x4Neg,
        _ => I::I64x2Neg,
    }
}

fn abs(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16Abs,
        1 => I::I16x8Abs,
        2 => I::I32x4Abs,
        _ => I::I64x2Abs,
    }
}

fn eq(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16Eq,
        1 => I::I16x8Eq,
        2 => I::I32x4Eq,
        _ => I::I64x2Eq,
    }
}

fn gt_s(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16GtS,
        1 => I::I16x8GtS,
        2 => I::I32x4GtS,
        _ => I::I64x2GtS,
    }
}

fn ge_s(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16GeS,
        1 => I::I16x8GeS,
        2 => I::I32x4GeS,
        _ => I::I64x2GeS,
    }
}

fn le_s(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16LeS,
        1 => I::I16x8LeS,
        2 => I::I32x4LeS,
        _ => I::I64x2LeS,
    }
}

fn lt_s(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16LtS,
        1 => I::I16x8LtS,
        2 => I::I32x4LtS,
        _ => I::I64x2LtS,
    }
}
