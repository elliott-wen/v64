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
pub(crate) fn lower_terminator(f: &mut Function, insn: &Insn, pc: u64, entry_pc: u64) -> bool {
    // All PCs are emitted relative to the runtime entry PC (`gen_rel_pc`) so the
    // block is position-independent — see `PC0` / `gen_rel_pc`.
    let store_link = |f: &mut Function| {
        emit!(f, I::LocalGet(0));
        gen_rel_pc(f, pc.wrapping_add(4), entry_pc);
        emit!(f, I::I64Store(at(offsets::x(30))));
    };
    match *insn {
        Insn::BranchImm { link, offset } => {
            if link {
                store_link(f);
            }
            gen_rel_pc(f, pc.wrapping_add(offset as u64), entry_pc);
        }
        Insn::BranchReg { opc, rn } => {
            if opc == 1 {
                // BLR sets the return address.
                store_link(f);
            }
            if rn == 31 {
                emit!(f, I::I64Const(0)); // XZR
            } else {
                emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rn as usize))));
            }
        }
        Insn::CondBranch { offset, .. }
        | Insn::CompareBranch { offset, .. }
        | Insn::TestBranch { offset, .. } => {
            emit_taken_cond(f, insn); // leaves i32: 1 = take the branch
            branch_select(f, pc.wrapping_add(offset as u64), pc.wrapping_add(4), entry_pc);
        }
        _ => return false,
    }
    true
}

/// Emit the i32 "branch taken?" flag for a conditional terminator
/// (`B.cond` / `CBZ`-`CBNZ` / `TBZ`-`TBNZ`), leaving `1` (take) or `0`. The
/// non-loop terminator feeds it to `branch_select`; `emit_self_loop` uses it as
/// the loop's continue condition. Callers gate on [`taken_target`]; anything
/// else panics.
pub(super) fn emit_taken_cond(f: &mut Function, insn: &Insn) {
    match *insn {
        Insn::CondBranch { cond, .. } => emit_cond_test(f, cond),
        Insn::CompareBranch { sf, negate, rt, .. } => {
            push_operand(f, rt, sf, false);
            emit!(f, I::I64Eqz); // (v == 0)
            if negate {
                emit!(f, I::I32Eqz); // CBNZ: (v != 0)
            }
        }
        Insn::TestBranch { bit, negate, rt, .. } => {
            if rt == 31 {
                emit!(f, I::I64Const(0));
            } else {
                emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rt as usize))));
            }
            emit!(f, I::I64Const(i64::from(bit)), I::I64ShrU, I::I64Const(1), I::I64And, I::I32WrapI64);
            if !negate {
                emit!(f, I::I32Eqz); // TBZ: take when the bit is clear
            }
        }
        _ => unreachable!("emit_taken_cond on a non-conditional terminator"),
    }
}

/// The taken-branch target of a conditional terminator at `pc`, or `None` if
/// `insn` isn't a conditional branch (used to detect a self-loop: target == the
/// block's own entry).
#[must_use]
pub(crate) fn taken_target(insn: &Insn, pc: u64) -> Option<u64> {
    match *insn {
        Insn::CondBranch { offset, .. }
        | Insn::CompareBranch { offset, .. }
        | Insn::TestBranch { offset, .. } => Some(pc.wrapping_add(offset as u64)),
        _ => None,
    }
}

/// `if <cond i32> { taken } else { fallthrough }`, leaving an `i64` PC. Both
/// targets are position-independent (`gen_rel_pc`).
fn branch_select(f: &mut Function, taken: u64, fallthrough: u64, entry_pc: u64) {
    emit!(f, I::If(BlockType::Result(ValType::I64)));
    gen_rel_pc(f, taken, entry_pc);
    emit!(f, I::Else);
    gen_rel_pc(f, fallthrough, entry_pc);
    emit!(f, I::End);
}
