//! SIMD shift by immediate (same-width): SHL, SSHR, USHR. The rounding,
//! accumulating, inserting, saturating, and narrowing/widening/FP variants fall
//! back.

use aarch64_cpu_state::regs::offsets;
use wasm_encoder::{Function, Instruction as I};

use super::{finish_v, push_v};
use crate::lower::common::at;

#[allow(clippy::too_many_arguments)]
pub(crate) fn simd_shift_imm(f: &mut Function, q: bool, u: bool, immh: u8, immb: u8, opcode: u8, rn: u8, rd: u8) -> bool {
    let size = 3 - (immh.leading_zeros() as u8 - 4); // highest set bit of immh
    let esize: u32 = 8 << size;
    let immhb = (u32::from(immh) << 3) | u32::from(immb);

    match opcode {
        0b01010 if !u => {
            // SHL by (immhb - esize), always < esize.
            let sh = (immhb - esize) as i32;
            emit!(f, I::LocalGet(0));
            push_v(f, rn);
            emit!(f, I::I32Const(sh), shl_op(size));
            finish_v(f, q, rd);
        }
        0b00000 => {
            // SSHR/USHR by (2*esize - immhb), in [1, esize].
            let sh = 2 * esize - immhb;
            if u && sh == esize {
                // USHR by the full width zeroes every lane.
                emit!(f, I::LocalGet(0), I::V128Const(0), I::V128Store(at(offsets::v(rd as usize))));
                return true;
            }
            // For SSHR a full-width shift is sign replication == shr_s by esize-1.
            let sh = if !u && sh == esize { esize - 1 } else { sh } as i32;
            emit!(f, I::LocalGet(0));
            push_v(f, rn);
            emit!(f, I::I32Const(sh), if u { shr_u_op(size) } else { shr_s_op(size) });
            finish_v(f, q, rd);
        }
        _ => return false, // SRSHR/SSRA/SRI/SLI/saturating/narrow/widen/fp
    }
    true
}

fn shl_op(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16Shl,
        1 => I::I16x8Shl,
        2 => I::I32x4Shl,
        _ => I::I64x2Shl,
    }
}

fn shr_s_op(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16ShrS,
        1 => I::I16x8ShrS,
        2 => I::I32x4ShrS,
        _ => I::I64x2ShrS,
    }
}

fn shr_u_op(size: u8) -> I<'static> {
    match size {
        0 => I::I8x16ShrU,
        1 => I::I16x8ShrU,
        2 => I::I32x4ShrU,
        _ => I::I64x2ShrU,
    }
}
