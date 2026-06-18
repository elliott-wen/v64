//! Bitfield moves and data-processing 1/2/3-source ops.
//!
//! Inlines what maps cleanly to WASM (RBIT, CLZ, CLS, byte reversals, variable
//! shifts, UDIV/SDIV with AArch64 divide semantics, MADD/MSUB and the widening
//! multiply-accumulates). CRC32 and SMULH/UMULH fall back.

use aarch64_cpu_state::regs::offsets;
use aarch64_decoder::ShiftType;
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

use super::common::*;

/// SBFM/BFM/UBFM — bitfield move.
#[allow(clippy::too_many_arguments)]
pub(super) fn bitfield(f: &mut Function, sf: bool, opc: u8, wmask: u64, tmask: u64, immr: u8, imms: u8, rn: u8, rd: u8) {
    let ds = if sf { 64 } else { 32 };
    let dmask = if sf { u64::MAX } else { W_MASK as u64 };
    let wmask = wmask & dmask;
    let tmask = tmask & dmask;

    // T0 = src
    push_operand(f, rn, sf, false);
    emit!(f, I::LocalSet(T0));

    // Push ROR(src, immr, ds) & wmask, then store to T1.
    let r = u32::from(immr) % ds;
    if sf {
        emit!(f, I::LocalGet(T0), I::I64Const(i64::from(r)), I::I64Rotr);
    } else if r == 0 {
        emit!(f, I::LocalGet(T0));
    } else {
        emit!(f, I::LocalGet(T0), I::I64Const(i64::from(r)), I::I64ShrU);
        emit!(f, I::LocalGet(T0), I::I64Const(i64::from(32 - r)), I::I64Shl, I::I64Const(W_MASK), I::I64And);
        emit!(f, I::I64Or);
    }
    emit!(f, I::I64Const(wmask as i64), I::I64And, I::LocalSet(T1));

    // T3 = result = (top & !tmask) | (bot & tmask)
    match opc {
        2 => {
            // UBFM: top = 0.
            emit!(f, I::LocalGet(T1), I::I64Const(tmask as i64), I::I64And, I::LocalSet(T3));
        }
        0 => {
            // SBFM: top = replicate(src<imms>) ? dmask : 0.
            //   topmasked = signbit * (dmask & !tmask)
            emit!(f, I::LocalGet(T0), I::I64Const(i64::from(imms)), I::I64ShrU, I::I64Const(1), I::I64And);
            emit!(f, I::I64Const((dmask & !tmask) as i64), I::I64Mul);
            emit!(f, I::LocalGet(T1), I::I64Const(tmask as i64), I::I64And, I::I64Or, I::LocalSet(T3));
        }
        _ => {
            // BFM: top = dst; bot = (dst & !wmask) | rotated.
            emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rd as usize))), I::LocalSet(T2)); // dst
            emit!(f, I::LocalGet(T2), I::I64Const(!tmask as i64), I::I64And); // top & !tmask
            emit!(f, I::LocalGet(T2), I::I64Const(!wmask as i64), I::I64And, I::LocalGet(T1), I::I64Or); // bot
            emit!(f, I::I64Const(tmask as i64), I::I64And);
            emit!(f, I::I64Or, I::LocalSet(T3));
        }
    }
    store_local(f, rd, sf, false, T3);
}

/// Data processing (1 source): RBIT, CLZ, CLS, and REV16/REV32/REV inline; CRC32
/// (decoded elsewhere) is the only family left out. Returns whether it was
/// handled (no emission on `false`) — kept in lockstep with `can_inline`.
pub(super) fn data_proc_1src(f: &mut Function, sf: bool, opcode: u8, rn: u8, rd: u8) -> bool {
    let ds: u32 = if sf { 64 } else { 32 };
    match opcode {
        0 => {
            // RBIT: reverse all bits = swap bits within each byte, then reverse
            // the byte order (the two compose to a full bit reversal).
            push_operand(f, rn, sf, false);
            emit!(f, I::LocalSet(T0));
            swap_bits_in_byte(f, ds);
            emit_byte_reverse(f, ds, ds / 8);
            emit!(f, I::LocalSet(T0));
        }
        4 => {
            // CLZ
            push_operand(f, rn, sf, false);
            if sf {
                emit!(f, I::I64Clz);
            } else {
                emit!(f, I::I32WrapI64, I::I32Clz, I::I64ExtendI32U);
            }
            emit!(f, I::LocalSet(T0));
        }
        5 => {
            // CLS: count leading sign bits = CLZ(x ^ sign_extend(msb)) - 1. The
            // xor leaves a 0 MSB, so CLZ >= 1 and the subtraction never underflows.
            push_operand(f, rn, sf, false);
            emit!(f, I::LocalSet(T0));
            if sf {
                emit!(f, I::LocalGet(T0), I::LocalGet(T0), I::I64Const(63), I::I64ShrS, I::I64Xor);
                emit!(f, I::I64Clz, I::I64Const(1), I::I64Sub, I::LocalSet(T0));
            } else {
                emit!(f, I::LocalGet(T0), I::I32WrapI64);
                emit!(f, I::LocalGet(T0), I::I32WrapI64, I::I32Const(31), I::I32ShrS, I::I32Xor);
                emit!(f, I::I32Clz, I::I32Const(1), I::I32Sub, I::I64ExtendI32U, I::LocalSet(T0));
            }
        }
        1..=3 => {
            let group = 1u32 << opcode; // REV16=2, REV32=4, REV(64)=8 bytes
            if group > ds / 8 {
                return false; // invalid width/group combo
            }
            push_operand(f, rn, sf, false);
            emit!(f, I::LocalSet(T0));
            emit_byte_reverse(f, ds, group);
            emit!(f, I::LocalSet(T0));
        }
        _ => return false, // unallocated
    }
    store_local(f, rd, sf, false, T0);
    true
}

/// Reverse the bit order within each byte of the low `ds` bits of `T0`, in place
/// (the three classic mask-swap steps). Combined with [`emit_byte_reverse`] over
/// the whole width this yields RBIT. `push_operand` zero-extends W operands, so
/// the 32-bit masks keep the result in the low 32 bits.
fn swap_bits_in_byte(f: &mut Function, ds: u32) {
    let (m1, m2, m3) = if ds == 64 {
        (0x5555_5555_5555_5555u64, 0x3333_3333_3333_3333, 0x0f0f_0f0f_0f0f_0f0f)
    } else {
        (0x5555_5555, 0x3333_3333, 0x0f0f_0f0f)
    };
    for (mask, sh) in [(m1, 1i64), (m2, 2), (m3, 4)] {
        // T0 = ((T0 & mask) << sh) | ((T0 >> sh) & mask)
        emit!(f, I::LocalGet(T0), I::I64Const(mask as i64), I::I64And, I::I64Const(sh), I::I64Shl);
        emit!(f, I::LocalGet(T0), I::I64Const(sh), I::I64ShrU, I::I64Const(mask as i64), I::I64And);
        emit!(f, I::I64Or, I::LocalSet(T0));
    }
}

/// Reverse bytes within each `group`-byte chunk of the low `ds` bits of `T0`,
/// leaving the result on the stack (mirrors `data_proc_1src::rev_groups`).
fn emit_byte_reverse(f: &mut Function, ds: u32, group: u32) {
    let bytes = (ds / 8) as usize;
    let g = group as usize;
    let mut first = true;
    for d in 0..bytes {
        let base = (d / g) * g;
        let src = base + (g - 1 - (d - base)); // source byte index for dest byte d
        emit!(f, I::LocalGet(T0));
        if src > 0 {
            emit!(f, I::I64Const((src * 8) as i64), I::I64ShrU);
        }
        emit!(f, I::I64Const(0xff), I::I64And);
        if d > 0 {
            emit!(f, I::I64Const((d * 8) as i64), I::I64Shl);
        }
        if first {
            first = false;
        } else {
            emit!(f, I::I64Or);
        }
    }
}

/// Data processing (2 source): variable shifts and UDIV/SDIV inline; CRC32 falls
/// back.
pub(super) fn data_proc_2src(f: &mut Function, sf: bool, opcode: u8, rm: u8, rn: u8, rd: u8) -> bool {
    match opcode {
        2 => div(f, sf, rm, rn, rd, false), // UDIV
        3 => div(f, sf, rm, rn, rd, true),  // SDIV
        8 => var_shift(f, sf, ShiftType::Lsl, rm, rn, rd),
        9 => var_shift(f, sf, ShiftType::Lsr, rm, rn, rd),
        10 => var_shift(f, sf, ShiftType::Asr, rm, rn, rd),
        11 => var_shift(f, sf, ShiftType::Ror, rm, rn, rd),
        _ => return false, // CRC32/CRC32C: slow path
    }
    true
}

/// LSLV/LSRV/ASRV/RORV — variable shift (amount = Rm mod datasize).
fn var_shift(f: &mut Function, sf: bool, shift: ShiftType, rm: u8, rn: u8, rd: u8) {
    if sf {
        // i64 shifts mask the amount mod 64 — exactly the AArch64 semantics.
        push_operand(f, rn, true, false);
        push_operand(f, rm, true, false);
        match shift {
            ShiftType::Lsl => emit!(f, I::I64Shl),
            ShiftType::Lsr => emit!(f, I::I64ShrU),
            ShiftType::Asr => emit!(f, I::I64ShrS),
            ShiftType::Ror => emit!(f, I::I64Rotr),
        }
    } else {
        // i32 shifts mask the amount mod 32; zero-extend the 32-bit result.
        push_operand(f, rn, false, false);
        emit!(f, I::I32WrapI64);
        push_operand(f, rm, false, false);
        emit!(f, I::I32WrapI64);
        match shift {
            ShiftType::Lsl => emit!(f, I::I32Shl),
            ShiftType::Lsr => emit!(f, I::I32ShrU),
            ShiftType::Asr => emit!(f, I::I32ShrS),
            ShiftType::Ror => emit!(f, I::I32Rotr),
        }
        emit!(f, I::I64ExtendI32U);
    }
    emit!(f, I::LocalSet(T0));
    store_local(f, rd, sf, false, T0);
}

/// UDIV/SDIV with AArch64's divide-by-zero (-> 0) and INT_MIN/-1 (-> INT_MIN)
/// behavior, which WASM would otherwise trap on.
fn div(f: &mut Function, sf: bool, rm: u8, rn: u8, rd: u8, signed: bool) {
    // T0 = n, T1 = m, sign-extended to 64 bits for signed W division.
    push_operand(f, rn, sf, false);
    if signed && !sf {
        emit!(f, I::I32WrapI64, I::I64ExtendI32S);
    }
    emit!(f, I::LocalSet(T0));
    push_operand(f, rm, sf, false);
    if signed && !sf {
        emit!(f, I::I32WrapI64, I::I64ExtendI32S);
    }
    emit!(f, I::LocalSet(T1));

    // if m == 0 { 0 } else { ... }
    emit!(f, I::LocalGet(T1), I::I64Eqz, I::If(BlockType::Result(ValType::I64)));
    emit!(f, I::I64Const(0));
    emit!(f, I::Else);
    if signed {
        let min = if sf { i64::MIN } else { i64::from(i32::MIN) };
        // if n == MIN && m == -1 { MIN } else { n / m }
        emit!(f, I::LocalGet(T0), I::I64Const(min), I::I64Eq);
        emit!(f, I::LocalGet(T1), I::I64Const(-1), I::I64Eq);
        emit!(f, I::I32And, I::If(BlockType::Result(ValType::I64)));
        emit!(f, I::I64Const(min));
        emit!(f, I::Else);
        emit!(f, I::LocalGet(T0), I::LocalGet(T1), I::I64DivS);
        emit!(f, I::End);
    } else {
        emit!(f, I::LocalGet(T0), I::LocalGet(T1), I::I64DivU);
    }
    emit!(f, I::End, I::LocalSet(T0));
    store_local(f, rd, sf, false, T0);
}

/// Data processing (3 source): MADD/MSUB and the widening multiply-accumulates.
/// SMULH/UMULH (128-bit) fall back.
#[allow(clippy::too_many_arguments)]
pub(super) fn data_proc_3src(f: &mut Function, sf: bool, op31: u8, o0: bool, rm: u8, ra: u8, rn: u8, rd: u8) -> bool {
    match op31 {
        0b000 => {
            // MADD/MSUB at the instruction width.
            push_operand(f, ra, sf, false);
            push_operand(f, rn, sf, false);
            push_operand(f, rm, sf, false);
            emit!(f, I::I64Mul);
            emit!(f, if o0 { I::I64Sub } else { I::I64Add });
            emit!(f, I::LocalSet(T0));
            store_local(f, rd, sf, false, T0);
        }
        0b001 | 0b101 => {
            // S/UMADDL, S/UMSUBL: 32-bit operands, 64-bit accumulate.
            let signed = op31 == 0b001;
            push_operand(f, ra, true, false); // Ra (full X; r31 = XZR)
            push32(f, rn, signed);
            push32(f, rm, signed);
            emit!(f, I::I64Mul);
            emit!(f, if o0 { I::I64Sub } else { I::I64Add });
            emit!(f, I::LocalSet(T0));
            store_local(f, rd, true, false, T0);
        }
        _ => return false, // SMULH/UMULH
    }
    true
}

/// Push the low 32 bits of `rn`, sign- or zero-extended to 64 bits.
fn push32(f: &mut Function, rn: u8, signed: bool) {
    if rn == 31 {
        emit!(f, I::I64Const(0));
        return;
    }
    emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rn as usize))), I::I32WrapI64);
    emit!(f, if signed { I::I64ExtendI32S } else { I::I64ExtendI32U });
}
