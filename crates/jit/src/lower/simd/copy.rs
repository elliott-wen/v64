//! SIMD copy & modified-immediate: MOVI/MVNI/ORR/BIC-imm, DUP (general and
//! element), INS (general and element), and SMOV/UMOV.
//!
//! The lane-level ops (UMOV/SMOV/INS/DUP-element) are plain sized loads/stores
//! against the V-register image at a constant lane offset — no `v128` needed.

use aarch64_cpu_state::regs::offsets;
use wasm_encoder::{Function, Instruction as I};

use super::{finish_v, push_v};
use crate::lower::common::{at, dest_off, push_operand};

/// Byte offset of lane `lane` (width `1 << size`) within V[v]'s image.
fn lane_off(v: u8, lane: u8, size: u8) -> usize {
    offsets::v(v as usize) + (lane as usize) * (1usize << size)
}

/// A sized integer load at a constant image offset (for lane/element access).
fn iload(size: u8, signed: bool, off: usize) -> I<'static> {
    let m = at(off);
    if signed {
        match size {
            0 => I::I64Load8S(m),
            1 => I::I64Load16S(m),
            2 => I::I64Load32S(m),
            _ => I::I64Load(m),
        }
    } else {
        match size {
            0 => I::I64Load8U(m),
            1 => I::I64Load16U(m),
            2 => I::I64Load32U(m),
            _ => I::I64Load(m),
        }
    }
}

/// A sized integer store at a constant image offset.
fn istore(size: u8, off: usize) -> I<'static> {
    let m = at(off);
    match size {
        0 => I::I64Store8(m),
        1 => I::I64Store16(m),
        2 => I::I64Store32(m),
        _ => I::I64Store(m),
    }
}

/// Splat the i64 on the stack across a `v128` for element width `size`.
fn splat(f: &mut Function, size: u8) {
    match size {
        0 => emit!(f, I::I32WrapI64, I::I8x16Splat),
        1 => emit!(f, I::I32WrapI64, I::I16x8Splat),
        2 => emit!(f, I::I32WrapI64, I::I32x4Splat),
        _ => emit!(f, I::I64x2Splat),
    }
}

/// MOVI/MVNI/ORR/BIC (modified immediate). The 64-bit element is a constant, so
/// MOVI/MVNI become a `v128.const`; ORR/BIC fold it against Vd. The FMOV-vector
/// cmode (1111) needs the FP immediate expansion and falls back.
pub(crate) fn simd_mod_imm(f: &mut Function, q: bool, op: bool, cmode: u8, imm8: u8, rd: u8) -> bool {
    let Some(elem) = expand_imm(op, cmode, imm8) else {
        return false; // cmode 1111 (FMOV-vector): slow path
    };
    let val: u128 = if q { (u128::from(elem) << 64) | u128::from(elem) } else { u128::from(elem) };
    let qmask: u128 = if q { u128::MAX } else { u128::from(u64::MAX) };
    let cmode_hi = cmode >> 1;

    if cmode_hi <= 0b101 && cmode & 1 == 1 {
        // ORR (op=0) / BIC (op=1) immediate against Vd.
        emit!(f, I::LocalGet(0));
        push_v(f, rd);
        emit!(f, I::V128Const(val as i128));
        emit!(f, if op { I::V128AndNot } else { I::V128Or }); // Vd & !val / Vd | val
        finish_v(f, q, rd);
    } else {
        // MOVI (val) or MVNI (!val); both fully constant.
        let result = if op && cmode_hi <= 0b110 { !val & qmask } else { val & qmask };
        emit!(f, I::LocalGet(0), I::V128Const(result as i128), I::V128Store(at(offsets::v(rd as usize))));
    }
    true
}

/// ARM `AdvSIMDExpandImm` for the integer cmodes (everything but FMOV-vector,
/// cmode 1111). Mirrors `simd_mod_imm::expand`.
fn expand_imm(op: bool, cmode: u8, imm8: u8) -> Option<u64> {
    let i = u64::from(imm8);
    let rep32 = |v: u64| (v & 0xffff_ffff) | ((v & 0xffff_ffff) << 32);
    let rep16 = |v: u64| {
        let h = v & 0xffff;
        h | (h << 16) | (h << 32) | (h << 48)
    };
    Some(match cmode >> 1 {
        0b000 => rep32(i),
        0b001 => rep32(i << 8),
        0b010 => rep32(i << 16),
        0b011 => rep32(i << 24),
        0b100 => rep16(i),
        0b101 => rep16(i << 8),
        0b110 => {
            let v = if cmode & 1 == 0 { (i << 8) | 0xff } else { (i << 16) | 0xffff };
            rep32(v)
        }
        _ if cmode & 1 == 0 => {
            // cmode 1110: byte-replicate (op=0) or bit-to-byte (op=1).
            let mut v = 0u64;
            if !op {
                for k in 0..8 {
                    v |= i << (8 * k);
                }
            } else {
                for k in 0..8 {
                    if (imm8 >> k) & 1 == 1 {
                        v |= 0xffu64 << (8 * k);
                    }
                }
            }
            v
        }
        _ => return None, // cmode 1111: FMOV-vector
    })
}

/// DUP (general): replicate GPR `rn` across all lanes of Vd.
pub(crate) fn simd_dup_general(f: &mut Function, q: bool, size: u8, rn: u8, rd: u8) {
    emit!(f, I::LocalGet(0));
    push_operand(f, rn, true, false); // full X (r31 -> ZR)
    splat(f, size);
    finish_v(f, q, rd);
}

/// DUP (element): replicate Vn[index] across all lanes of Vd.
pub(crate) fn simd_dup_element(f: &mut Function, q: bool, size: u8, index: u8, rn: u8, rd: u8) {
    emit!(f, I::LocalGet(0));
    emit!(f, I::LocalGet(0), iload(size, false, lane_off(rn, index, size)));
    splat(f, size);
    finish_v(f, q, rd);
}

/// INS (general): Vd[index] = low bits of Rn (other lanes preserved).
pub(crate) fn simd_ins_general(f: &mut Function, size: u8, index: u8, rn: u8, rd: u8) {
    emit!(f, I::LocalGet(0));
    push_operand(f, rn, true, false);
    emit!(f, istore(size, lane_off(rd, index, size)));
}

/// INS (element): Vd[dst] = Vn[src].
pub(crate) fn simd_ins_element(f: &mut Function, size: u8, dst: u8, src: u8, rn: u8, rd: u8) {
    emit!(f, I::LocalGet(0));
    emit!(f, I::LocalGet(0), iload(size, false, lane_off(rn, src, size)));
    emit!(f, istore(size, lane_off(rd, dst, size)));
}

/// SMOV/UMOV: move Vn[index] to GPR `rd`, sign- or zero-extended.
pub(crate) fn simd_mov_to_gpr(f: &mut Function, signed: bool, dst64: bool, size: u8, index: u8, vn: u8, rd: u8) {
    let Some(rd_off) = dest_off(rd, false) else { return }; // ZR: discard
    emit!(f, I::LocalGet(0)); // base for the GPR store
    emit!(f, I::LocalGet(0), iload(size, signed, lane_off(vn, index, size)));
    if signed && !dst64 {
        emit!(f, I::I64Const(0xffff_ffff), I::I64And); // sign-extend to 32, then zero-extend
    }
    emit!(f, I::I64Store(at(rd_off)));
}
