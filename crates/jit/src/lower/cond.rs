//! Condition-code evaluation and the conditional instructions that consume it:
//! CSEL/CSINC/CSINV/CSNEG and CCMP/CCMN. The [`emit_cond_test`] / [`push_flag`]
//! helpers are also used by the branch terminators in [`super::terminator`].

use aarch64_cpu_state::regs::offsets;
use aarch64_decoder::ShiftType;
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

use super::arith::{emit_add_core, setup_add_operands, BOp, Carry};
use super::common::*;

/// CSEL/CSINC/CSINV/CSNEG.
#[allow(clippy::too_many_arguments)]
pub(super) fn cond_select(f: &mut Function, sf: bool, op: bool, o2: bool, cond: u8, rm: u8, rn: u8, rd: u8) {
    emit_cond_test(f, cond);
    emit!(f, I::If(BlockType::Result(ValType::I64)));
    push_operand(f, rn, sf, false);
    emit!(f, I::Else);
    push_operand(f, rm, sf, false);
    match (op, o2) {
        (false, false) => {}                                   // CSEL
        (false, true) => emit!(f, I::I64Const(1), I::I64Add),  // CSINC
        (true, false) => emit!(f, I::I64Const(-1), I::I64Xor), // CSINV
        (true, true) => emit!(f, I::I64Const(-1), I::I64Mul),  // CSNEG (negate)
    }
    mask_w(f, sf);
    emit!(f, I::End, I::LocalSet(T0));
    store_local(f, rd, sf, false, T0);
}

/// CCMP/CCMN — conditional compare. Sets NZCV either from `Rn ± Y` or `nzcv`.
#[allow(clippy::too_many_arguments)]
pub(super) fn cond_compare(f: &mut Function, sf: bool, sub: bool, is_imm: bool, imm_y: u8, rm: u8, cond: u8, nzcv: u8, rn: u8) {
    emit_cond_test(f, cond);
    emit!(f, I::If(BlockType::Empty));
    // Condition holds: flags from add_with_carry(Rn, Y, sub).
    let y = if is_imm { BOp::Imm(u64::from(imm_y)) } else { BOp::shifted(rm, ShiftType::Lsl, 0, false) };
    setup_add_operands(f, sf, sub, y, rn, false);
    emit_add_core(f, sf, true, None, if sub { Carry::One } else { Carry::Zero });
    emit!(f, I::Else);
    // Otherwise force NZCV to the 4-bit immediate.
    let packed = (u64::from(nzcv >> 3 & 1) << 31)
        | (u64::from(nzcv >> 2 & 1) << 30)
        | (u64::from(nzcv >> 1 & 1) << 29)
        | (u64::from(nzcv & 1) << 28);
    emit!(f, I::LocalGet(0), I::I64Const(packed as i64), I::I64Store(at(offsets::NZCV)));
    emit!(f, I::End);
}

/// Push a flag bit (0/1, as i32) from the packed NZCV word.
pub(super) fn push_flag(f: &mut Function, bit: i64) {
    emit!(f, I::LocalGet(0), I::I64Load(at(offsets::NZCV)), I::I64Const(bit), I::I64ShrU, I::I64Const(1), I::I64And, I::I32WrapI64);
}

/// Emit the test for condition code `cond`, leaving its result (0/1 i32) on the
/// stack. Mirrors `interp::eval_cond`; `cond` is constant at emit time.
pub(super) fn emit_cond_test(f: &mut Function, cond: u8) {
    match cond >> 1 {
        0 => push_flag(f, Z_BIT),                  // EQ/NE : Z
        1 => push_flag(f, C_BIT),                  // CS/CC : C
        2 => push_flag(f, N_BIT),                  // MI/PL : N
        3 => push_flag(f, V_BIT),                  // VS/VC : V
        4 => {
            // HI/LS : C && !Z
            push_flag(f, C_BIT);
            push_flag(f, Z_BIT);
            emit!(f, I::I32Eqz, I::I32And);
        }
        5 => {
            // GE/LT : N == V
            push_flag(f, N_BIT);
            push_flag(f, V_BIT);
            emit!(f, I::I32Eq);
        }
        6 => {
            // GT/LE : !Z && (N == V)
            push_flag(f, Z_BIT);
            emit!(f, I::I32Eqz);
            push_flag(f, N_BIT);
            push_flag(f, V_BIT);
            emit!(f, I::I32Eq, I::I32And);
        }
        _ => emit!(f, I::I32Const(1)), // AL
    }
    // Odd condition codes invert the base test (but AL == 0b1111 does not).
    if cond & 1 == 1 && cond != 0b1111 {
        emit!(f, I::I32Eqz);
    }
}
