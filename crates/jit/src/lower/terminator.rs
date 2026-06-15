//! Control-flow terminators: B/BL, BR/BLR/RET, B.cond, CBZ/CBNZ, TBZ/TBNZ.
//!
//! A terminator is always the last instruction in a block. Its lowering leaves
//! the next guest PC on the operand stack as the block function's `i64` result;
//! it does not write the image PC (the runtime writes the returned PC back).

use aarch64_cpu_state::regs::offsets;
use aarch64_decoder::Insn;
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

use super::cond::emit_cond_test;
use super::common::*;

/// Try to lower a control-flow terminator. On success leaves an `i64` (the next
/// guest PC) on the operand stack.
pub(crate) fn lower_terminator(f: &mut Function, insn: &Insn, pc: u64) -> bool {
    match *insn {
        Insn::BranchImm { link, offset } => {
            if link {
                emit!(f, I::LocalGet(0), I::I64Const(pc.wrapping_add(4) as i64), I::I64Store(at(offsets::x(30))));
            }
            emit!(f, I::I64Const(pc.wrapping_add(offset as u64) as i64));
        }
        Insn::BranchReg { opc, rn } => {
            if opc == 1 {
                // BLR sets the return address.
                emit!(f, I::LocalGet(0), I::I64Const(pc.wrapping_add(4) as i64), I::I64Store(at(offsets::x(30))));
            }
            if rn == 31 {
                emit!(f, I::I64Const(0)); // XZR
            } else {
                emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rn as usize))));
            }
        }
        Insn::CondBranch { cond, offset } => {
            emit_cond_test(f, cond);
            branch_select(f, pc.wrapping_add(offset as u64), pc.wrapping_add(4));
        }
        Insn::CompareBranch { sf, negate, rt, offset } => {
            push_operand(f, rt, sf, false);
            emit!(f, I::I64Eqz); // (v == 0)
            if negate {
                emit!(f, I::I32Eqz); // CBNZ: (v != 0)
            }
            branch_select(f, pc.wrapping_add(offset as u64), pc.wrapping_add(4));
        }
        Insn::TestBranch { bit, negate, rt, offset } => {
            if rt == 31 {
                emit!(f, I::I64Const(0));
            } else {
                emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rt as usize))));
            }
            emit!(f, I::I64Const(i64::from(bit)), I::I64ShrU, I::I64Const(1), I::I64And, I::I32WrapI64);
            if !negate {
                emit!(f, I::I32Eqz); // TBZ: take when the bit is clear
            }
            branch_select(f, pc.wrapping_add(offset as u64), pc.wrapping_add(4));
        }
        _ => return false,
    }
    true
}

/// `if <cond i32> { taken } else { fallthrough }`, leaving an `i64` PC.
fn branch_select(f: &mut Function, taken: u64, fallthrough: u64) {
    emit!(f, I::If(BlockType::Result(ValType::I64)));
    emit!(f, I::I64Const(taken as i64));
    emit!(f, I::Else);
    emit!(f, I::I64Const(fallthrough as i64));
    emit!(f, I::End);
}
