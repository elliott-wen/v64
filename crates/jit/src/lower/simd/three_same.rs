//! SIMD three-same (integer): element-wise logical, add/sub, multiply,
//! compares, and min/max.

use wasm_encoder::{Function, Instruction as I};

use super::{finish_v, push_v};

/// `Rd = op(Vn, Vm)` for a two-operand lane op.
fn lane2(f: &mut Function, q: bool, rn: u8, rm: u8, rd: u8, op: I<'static>) {
    emit!(f, I::LocalGet(0)); // regs_base for the store
    push_v(f, rn);
    push_v(f, rm);
    emit!(f, op);
    finish_v(f, q, rd);
}

/// Advanced SIMD three-same (integer). Returns whether it was lowered inline;
/// declines (no emission) for forms WASM can't express bit-exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn simd_three_same(f: &mut Function, q: bool, u: bool, size: u8, opcode: u8, rm: u8, rn: u8, rd: u8) -> bool {
    match opcode {
        0b00011 => logical(f, q, u, size, rm, rn, rd), // AND/BIC/ORR/ORN/EOR/BSL/BIT/BIF
        0b10001 if u => lane2(f, q, rn, rm, rd, eq(size)), // CMEQ
        0b10001 => cmtst(f, q, size, rm, rn, rd),          // CMTST
        _ => {
            let Some(op) = simple_op(u, size, opcode) else { return false };
            lane2(f, q, rn, rm, rd, op);
        }
    }
    true
}

/// Logical ops (opcode 0b00011) — all eight `(u, size)` selections map to WASM
/// bitwise/bitselect. Always handled.
fn logical(f: &mut Function, q: bool, u: bool, size: u8, rm: u8, rn: u8, rd: u8) {
    emit!(f, I::LocalGet(0)); // regs_base for the store
    match (u, size) {
        (false, 0) => {
            push_v(f, rn);
            push_v(f, rm);
            emit!(f, I::V128And); // AND
        }
        (false, 1) => {
            push_v(f, rn);
            push_v(f, rm);
            emit!(f, I::V128AndNot); // BIC: a & !b
        }
        (false, 2) => {
            push_v(f, rn);
            push_v(f, rm);
            emit!(f, I::V128Or); // ORR
        }
        (false, _) => {
            push_v(f, rn);
            push_v(f, rm);
            emit!(f, I::V128Not, I::V128Or); // ORN: a | !b
        }
        (true, 0) => {
            push_v(f, rn);
            push_v(f, rm);
            emit!(f, I::V128Xor); // EOR
        }
        (true, 1) => {
            // BSL: (Vn & Vd) | (Vm & !Vd) = bitselect(v1=Vn, v2=Vm, c=Vd)
            push_v(f, rn);
            push_v(f, rm);
            push_v(f, rd);
            emit!(f, I::V128Bitselect);
        }
        (true, 2) => {
            // BIT: (Vn & Vm) | (Vd & !Vm) = bitselect(v1=Vn, v2=Vd, c=Vm)
            push_v(f, rn);
            push_v(f, rd);
            push_v(f, rm);
            emit!(f, I::V128Bitselect);
        }
        (true, _) => {
            // BIF: (Vd & Vm) | (Vn & !Vm) = bitselect(v1=Vd, v2=Vn, c=Vm)
            push_v(f, rd);
            push_v(f, rn);
            push_v(f, rm);
            emit!(f, I::V128Bitselect);
        }
    }
    finish_v(f, q, rd);
}

/// CMTST: per-lane `(Vn & Vm) != 0 ? all-ones : 0` = `!eq(Vn & Vm, 0)`.
fn cmtst(f: &mut Function, q: bool, size: u8, rm: u8, rn: u8, rd: u8) {
    emit!(f, I::LocalGet(0));
    push_v(f, rn);
    push_v(f, rm);
    emit!(f, I::V128And, I::V128Const(0), eq(size), I::V128Not);
    finish_v(f, q, rd);
}

/// The element-equality op for a given size (always available).
fn eq(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16Eq,
        1 => I::I16x8Eq,
        2 => I::I32x4Eq,
        _ => I::I64x2Eq,
    }
}

/// Pick the WASM lane op for the simple two-operand opcodes, or `None` if WASM
/// has no bit-exact equivalent for this `(u, size)` (e.g. i64 unsigned compares
/// and i64 min/max, or 8-bit multiply).
fn simple_op(u: bool, size: u8, opcode: u8) -> Option<I<'static>> {
    use I::*;
    Some(match opcode {
        // SQADD/UQADD and SQSUB/UQSUB — WASM only has 8/16-bit saturating lanes.
        0b00001 if !u => match size {
            0 => I8x16AddSatS,
            1 => I16x8AddSatS,
            _ => return None,
        },
        0b00001 => match size {
            0 => I8x16AddSatU,
            1 => I16x8AddSatU,
            _ => return None,
        },
        0b00101 if !u => match size {
            0 => I8x16SubSatS,
            1 => I16x8SubSatS,
            _ => return None,
        },
        0b00101 => match size {
            0 => I8x16SubSatU,
            1 => I16x8SubSatU,
            _ => return None,
        },
        0b10000 if !u => match size {
            0 => I8x16Add,
            1 => I16x8Add,
            2 => I32x4Add,
            _ => I64x2Add,
        },
        0b10000 => match size {
            0 => I8x16Sub,
            1 => I16x8Sub,
            2 => I32x4Sub,
            _ => I64x2Sub,
        },
        0b10011 if !u => match size {
            1 => I16x8Mul,
            2 => I32x4Mul,
            _ => return None, // no i8x16.mul; 64-bit MUL isn't a NEON form
        },
        0b00110 if !u => match size {
            0 => I8x16GtS,
            1 => I16x8GtS,
            2 => I32x4GtS,
            _ => I64x2GtS,
        },
        0b00110 => match size {
            0 => I8x16GtU,
            1 => I16x8GtU,
            2 => I32x4GtU,
            _ => return None, // no i64x2.gt_u
        },
        0b00111 if !u => match size {
            0 => I8x16GeS,
            1 => I16x8GeS,
            2 => I32x4GeS,
            _ => I64x2GeS,
        },
        0b00111 => match size {
            0 => I8x16GeU,
            1 => I16x8GeU,
            2 => I32x4GeU,
            _ => return None,
        },
        0b01100 if !u => max_min(size, true, true)?,  // SMAX
        0b01100 => max_min(size, false, true)?,       // UMAX
        0b01101 if !u => max_min(size, true, false)?, // SMIN
        0b01101 => max_min(size, false, false)?,      // UMIN
        _ => return None,
    })
}

/// Signed/unsigned max/min by size — WASM lacks the i64 forms.
fn max_min(size: u8, signed: bool, max: bool) -> Option<I<'static>> {
    use I::*;
    Some(match (size, signed, max) {
        (0, true, true) => I8x16MaxS,
        (1, true, true) => I16x8MaxS,
        (2, true, true) => I32x4MaxS,
        (0, false, true) => I8x16MaxU,
        (1, false, true) => I16x8MaxU,
        (2, false, true) => I32x4MaxU,
        (0, true, false) => I8x16MinS,
        (1, true, false) => I16x8MinS,
        (2, true, false) => I32x4MinS,
        (0, false, false) => I8x16MinU,
        (1, false, false) => I16x8MinU,
        (2, false, false) => I32x4MinU,
        _ => return None, // no i64x2 min/max
    })
}
