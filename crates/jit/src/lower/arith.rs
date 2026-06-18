//! Move-wide, add/sub (immediate, shifted-, extended-register, and with-carry),
//! logical ops, constant/variable shift application, EXTR, and ADR/ADRP.

use aarch64_cpu_state::regs::offsets;
use aarch64_decoder::ShiftType;
use wasm_encoder::{Function, Instruction as I};

use super::common::*;

/// MOVZ / MOVN / MOVK.
pub(super) fn move_wide(f: &mut Function, sf: bool, opc: u8, hw: u8, imm16: u16, rd: u8) {
    let shift = u32::from(hw) * 16;
    let imm = u64::from(imm16) << shift;
    let Some(off) = dest_off(rd, false) else { return }; // write to XZR -> nothing

    emit!(f, I::LocalGet(0));
    match opc {
        2 => emit!(f, I::I64Const(imm as i64)),  // MOVZ
        0 => emit!(f, I::I64Const(!imm as i64)), // MOVN
        3 => {
            // MOVK: keep the other bits of Rd, replace the 16-bit field.
            let keep = !(0xffff_u64 << shift);
            emit!(
                f,
                I::LocalGet(0),
                I::I64Load(at(offsets::x(rd as usize))),
                I::I64Const(keep as i64),
                I::I64And,
                I::I64Const(imm as i64),
                I::I64Or,
            );
        }
        _ => unreachable!("decoder rejects MoveWide opc==1"),
    }
    mask_w(f, sf);
    emit!(f, I::I64Store(at(off)));
}

/// Second operand of a data-processing instruction.
pub(super) enum BOp {
    Imm(u64),
    /// Register, shifted by a constant amount, optionally bitwise-negated after
    /// the shift (BIC/ORN/EON forms of the logical ops).
    Shifted { rm: u8, shift: ShiftType, amount: u8, negate: bool },
    /// Extended register (UXTB..SXTX, then left-shift): ADD/SUB extended forms.
    Ext(u8, u8, u8),
}

impl BOp {
    pub(super) fn shifted(rm: u8, shift: ShiftType, amount: u8, negate: bool) -> Self {
        BOp::Shifted { rm, shift, amount, negate }
    }
}

fn push_b(f: &mut Function, b: &BOp, sf: bool) {
    match *b {
        BOp::Imm(v) => emit!(f, I::I64Const(v as i64)),
        BOp::Shifted { rm, shift, amount, negate } => {
            push_operand(f, rm, sf, false);
            emit_shift(f, shift, amount, sf);
            if negate {
                emit!(f, I::I64Const(-1), I::I64Xor);
                mask_w(f, sf);
            }
        }
        BOp::Ext(rm, option, shift) => push_ext(f, rm, option, shift),
    }
}

/// Apply a constant shift to the i64 on the stack (matches `alu::apply_shift`).
fn emit_shift(f: &mut Function, shift: ShiftType, amount: u8, sf: bool) {
    let width = if sf { 64 } else { 32 };
    let amt = u32::from(amount) % width;
    match shift {
        ShiftType::Lsl => {
            if amt != 0 {
                emit!(f, I::I64Const(i64::from(amt)), I::I64Shl);
            }
            mask_w(f, sf);
        }
        ShiftType::Lsr => {
            if amt != 0 {
                emit!(f, I::I64Const(i64::from(amt)), I::I64ShrU);
            }
        }
        ShiftType::Asr => {
            if sf {
                if amt != 0 {
                    emit!(f, I::I64Const(i64::from(amt)), I::I64ShrS);
                }
            } else {
                // Sign-extend the low 32 bits, arithmetic-shift, re-truncate.
                emit!(f, I::I64Const(32), I::I64Shl, I::I64Const(i64::from(32 + amt)), I::I64ShrS);
                mask_w(f, sf);
            }
        }
        ShiftType::Ror => {
            if sf {
                emit!(f, I::I64Const(i64::from(amt)), I::I64Rotr);
            } else if amt != 0 {
                // 32-bit rotate-right: (v >> amt) | ((v << (32-amt)) & 0xffffffff)
                emit!(f, I::LocalSet(T4));
                emit!(f, I::LocalGet(T4), I::I64Const(i64::from(amt)), I::I64ShrU);
                emit!(f, I::LocalGet(T4), I::I64Const(i64::from(32 - amt)), I::I64Shl, I::I64Const(W_MASK), I::I64And);
                emit!(f, I::I64Or);
            }
        }
    }
}

/// Push the extended (UXTB..SXTX then `<< shift`) value of `rm`. Shared with the
/// register-offset addressing mode in [`super::memory`].
pub(super) fn push_ext(f: &mut Function, rm: u8, option: u8, shift: u8) {
    if rm == 31 {
        emit!(f, I::I64Const(0));
    } else {
        emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rm as usize))));
    }
    let (bits, signed) = match option {
        0 => (8, false),
        1 => (16, false),
        2 => (32, false),
        3 => (64, false),
        4 => (8, true),
        5 => (16, true),
        6 => (32, true),
        _ => (64, true),
    };
    if bits < 64 {
        emit!(f, I::I64Const(((1u64 << bits) - 1) as i64), I::I64And);
        if signed {
            let s = 64 - bits;
            emit!(f, I::I64Const(s), I::I64Shl, I::I64Const(s), I::I64ShrS);
        }
    }
    if shift > 0 {
        emit!(f, I::I64Const(i64::from(shift)), I::I64Shl);
    }
}

/// AND/ORR/EOR/ANDS (and BIC/ORN/EON/BICS via a negated `BOp::Shifted`).
pub(super) fn logical(f: &mut Function, sf: bool, opc: u8, b: BOp, rn: u8, rd: u8, dst_sp: bool) {
    push_operand(f, rn, sf, false);
    push_b(f, &b, sf);
    match opc {
        0 | 3 => emit!(f, I::I64And),
        1 => emit!(f, I::I64Or),
        2 => emit!(f, I::I64Xor),
        _ => unreachable!(),
    }
    mask_w(f, sf);
    emit!(f, I::LocalSet(T0));

    store_local(f, rd, sf, dst_sp, T0);

    if opc == 3 {
        // ANDS/BICS: N and Z from the result; C and V cleared.
        let sign = if sf { 63 } else { 31 };
        emit!(f, I::LocalGet(0));
        emit!(f, I::LocalGet(T0), I::I64Const(sign), I::I64ShrU, I::I64Const(N_BIT), I::I64Shl);
        emit!(f, I::LocalGet(T0), I::I64Eqz, I::I64ExtendI32U, I::I64Const(Z_BIT), I::I64Shl, I::I64Or);
        emit!(f, I::I64Store(at(offsets::NZCV)));
    }
}

/// Carry-in source for [`emit_add_core`].
pub(super) enum Carry {
    Zero,
    One,
    CFlag,
}

fn push_carry(f: &mut Function, c: &Carry) {
    match c {
        Carry::Zero => emit!(f, I::I64Const(0)),
        Carry::One => emit!(f, I::I64Const(1)),
        Carry::CFlag => {
            emit!(f, I::LocalGet(0), I::I64Load(at(offsets::NZCV)), I::I64Const(C_BIT), I::I64ShrU, I::I64Const(1), I::I64And);
        }
    }
}

/// Core of `AddWithCarry`: assumes `T0 = a`, `T1 = b_op` (already complemented
/// for subtract and W-masked). Writes the result to `dst` and, if `set_flags`,
/// the NZCV word. Mirrors `alu::add_with_carry_in`.
pub(super) fn emit_add_core(f: &mut Function, sf: bool, set_flags: bool, dst: Option<usize>, carry: Carry) {
    if sf {
        emit!(f, I::LocalGet(T0), I::LocalGet(T1), I::I64Add, I::LocalSet(T2)); // s1
        emit!(f, I::LocalGet(T2));
        push_carry(f, &carry);
        emit!(f, I::I64Add, I::LocalSet(T3)); // s = s1 + carry
        if let Some(off) = dst {
            emit!(f, I::LocalGet(0), I::LocalGet(T3), I::I64Store(at(off)));
        }
        if set_flags {
            emit!(f, I::LocalGet(0));
            emit!(f, I::LocalGet(T3), I::I64Const(63), I::I64ShrU, I::I64Const(N_BIT), I::I64Shl);
            emit!(f, I::LocalGet(T3), I::I64Eqz, I::I64ExtendI32U, I::I64Const(Z_BIT), I::I64Shl, I::I64Or);
            // C = (s1 <u a) | (s <u s1)
            emit!(f, I::LocalGet(T2), I::LocalGet(T0), I::I64LtU);
            emit!(f, I::LocalGet(T3), I::LocalGet(T2), I::I64LtU);
            emit!(f, I::I32Or, I::I64ExtendI32U, I::I64Const(C_BIT), I::I64Shl, I::I64Or);
            // V = (((a ^ s) & (b_op ^ s)) >> 63)
            emit!(f, I::LocalGet(T0), I::LocalGet(T3), I::I64Xor);
            emit!(f, I::LocalGet(T1), I::LocalGet(T3), I::I64Xor);
            emit!(f, I::I64And, I::I64Const(63), I::I64ShrU, I::I64Const(V_BIT), I::I64Shl, I::I64Or);
            emit!(f, I::I64Store(at(offsets::NZCV)));
        }
    } else {
        // 32-bit: operands fit in 32 bits, so a + b_op + carry fits in i64.
        emit!(f, I::LocalGet(T0), I::LocalGet(T1), I::I64Add);
        push_carry(f, &carry);
        emit!(f, I::I64Add, I::LocalSet(T3)); // wide
        emit!(f, I::LocalGet(T3), I::I64Const(W_MASK), I::I64And, I::LocalSet(T2)); // s
        if let Some(off) = dst {
            emit!(f, I::LocalGet(0), I::LocalGet(T2), I::I64Store(at(off)));
        }
        if set_flags {
            emit!(f, I::LocalGet(0));
            emit!(f, I::LocalGet(T2), I::I64Const(31), I::I64ShrU, I::I64Const(N_BIT), I::I64Shl);
            emit!(f, I::LocalGet(T2), I::I64Eqz, I::I64ExtendI32U, I::I64Const(Z_BIT), I::I64Shl, I::I64Or);
            emit!(f, I::LocalGet(T3), I::I64Const(32), I::I64ShrU, I::I64Const(C_BIT), I::I64Shl, I::I64Or); // C = bit 32
            emit!(f, I::LocalGet(T0), I::LocalGet(T2), I::I64Xor);
            emit!(f, I::LocalGet(T1), I::LocalGet(T2), I::I64Xor);
            emit!(f, I::I64And, I::I64Const(31), I::I64ShrU, I::I64Const(V_BIT), I::I64Shl, I::I64Or);
            emit!(f, I::I64Store(at(offsets::NZCV)));
        }
    }
}

/// Set up `T0 = a`, `T1 = b_op` for an add/sub with the given carry-in.
pub(super) fn setup_add_operands(f: &mut Function, sf: bool, sub: bool, b: BOp, rn: u8, rn_sp: bool) {
    push_operand(f, rn, sf, rn_sp);
    emit!(f, I::LocalSet(T0));
    push_b(f, &b, sf);
    emit!(f, I::LocalSet(T1));
    if sub {
        emit!(f, I::LocalGet(T1), I::I64Const(-1), I::I64Xor, I::LocalSet(T1)); // !b
    }
    if !sf {
        emit!(f, I::LocalGet(T1), I::I64Const(W_MASK), I::I64And, I::LocalSet(T1));
    }
}

/// ADD/SUB and ADDS/SUBS (immediate, shifted-reg, extended-reg).
#[allow(clippy::too_many_arguments)]
pub(super) fn add_sub(f: &mut Function, sf: bool, sub: bool, set_flags: bool, b: BOp, rn: u8, rn_sp: bool, rd: u8, rd_sp: bool) {
    setup_add_operands(f, sf, sub, b, rn, rn_sp);
    let carry = if sub { Carry::One } else { Carry::Zero };
    emit_add_core(f, sf, set_flags, dest_off(rd, rd_sp), carry);
}

/// ADC/SBC/ADCS/SBCS — add/sub with the PSTATE carry as carry-in.
pub(super) fn add_sub_carry(f: &mut Function, sf: bool, sub: bool, set_flags: bool, rm: u8, rn: u8, rd: u8) {
    setup_add_operands(f, sf, sub, BOp::shifted(rm, ShiftType::Lsl, 0, false), rn, false);
    emit_add_core(f, sf, set_flags, dest_off(rd, false), Carry::CFlag);
}

/// EXTR — `Rd = (Rn:Rm) >> lsb`, low `datasize` bits.
pub(super) fn extract(f: &mut Function, sf: bool, rm: u8, rn: u8, lsb: u8, rd: u8) {
    let lsb = u32::from(lsb);
    let width = if sf { 64 } else { 32 };
    if lsb == 0 {
        push_operand(f, rm, sf, false);
    } else {
        push_operand(f, rm, sf, false);
        emit!(f, I::I64Const(i64::from(lsb)), I::I64ShrU);
        push_operand(f, rn, sf, false);
        emit!(f, I::I64Const(i64::from(width - lsb)), I::I64Shl);
        mask_w(f, sf);
        emit!(f, I::I64Or);
    }
    emit!(f, I::LocalSet(T0));
    store_local(f, rd, sf, false, T0);
}

/// ADR / ADRP — PC-relative address (always a 64-bit result). Position-independent:
/// the runtime PC is `PC0 + (pc - entry_pc)`, so the result tracks wherever the
/// block is actually mapped (see `gen_rel_pc`).
pub(super) fn pc_rel(f: &mut Function, page: bool, imm: i64, rd: u8, pc: u64, entry_pc: u64) {
    let Some(off) = dest_off(rd, false) else { return };
    emit!(f, I::LocalGet(0)); // regs_base, for the store below
    if page {
        // ADRP: (runtime_pc & ~0xfff) + imm.
        emit!(
            f,
            I::LocalGet(PC0),
            I::I64Const(pc.wrapping_sub(entry_pc) as i64),
            I::I64Add,
            I::I64Const(!0xfff_i64),
            I::I64And,
            I::I64Const(imm),
            I::I64Add
        );
    } else {
        // ADR: runtime_pc + imm = PC0 + (pc + imm - entry_pc).
        gen_rel_pc(f, pc.wrapping_add(imm as u64), entry_pc);
    }
    emit!(f, I::I64Store(at(off)));
}
