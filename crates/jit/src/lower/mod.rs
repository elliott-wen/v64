//! Inline instruction lowering (Milestones 3–4).
//!
//! Two entry points emit WASM for a single guest instruction directly, avoiding
//! the `interpret_one` escape hatch:
//!
//! - [`lower_sequential`] — non-terminator instructions. Updates registers/flags
//!   in the image, advances the image PC to `pc + 4`, and leaves nothing on the
//!   operand stack.
//! - [`lower_terminator`] — control-flow instructions (always the last in a
//!   block). Computes the next guest PC and leaves it on the stack as the block
//!   function's `i64` result; it does *not* write the image PC (the runtime
//!   writes the returned PC back after the call).
//!
//! Either returns `false` for anything it doesn't handle, so the caller falls
//! back to `interpret_one`; correctness is never at stake, only speed. A
//! lowering that may decline **must do so before emitting anything**, so a
//! `false` return never leaves partial code in the function. Flags are computed
//! inline into the packed NZCV word (no host helper calls).
//!
//! The lowerings are grouped by instruction family across submodules:
//! [`common`] (register/flag image access), [`arith`], [`cond`], [`dataproc`],
//! [`memory`], and [`terminator`].

/// Emit a sequence of instructions into a [`wasm_encoder::Function`]. Defined
/// before the submodule declarations so they inherit it via textual scoping.
macro_rules! emit {
    ($f:expr, $($i:expr),* $(,)?) => {{ $( $f.instruction(&$i); )* }};
}

mod arith;
mod common;
mod cond;
mod dataproc;
mod memory;
mod simd;
mod terminator;

use aarch64_decoder::Insn;
use wasm_encoder::Function;

use arith::BOp;
use common::advance_pc;

pub(crate) use common::{SCRATCH_I32, SCRATCH_I64};
pub(crate) use terminator::lower_terminator;

/// Try to lower a non-terminator instruction. On success advances the image PC.
pub(crate) fn lower_sequential(f: &mut Function, insn: &Insn, pc: u64, guest_base: u64) -> bool {
    match *insn {
        // NOP and PRFM (prefetch hint) have no architectural effect.
        Insn::Nop | Insn::Prfm => {}
        Insn::MoveWide { sf, opc, hw, imm16, rd } => arith::move_wide(f, sf, opc, hw, imm16, rd),
        Insn::LogicalImm { sf, opc, imm, rn, rd } => {
            arith::logical(f, sf, opc, BOp::Imm(imm), rn, rd, opc != 3);
        }
        Insn::LogicalShiftedReg { sf, opc, negate, shift, amount, rm, rn, rd } => {
            arith::logical(f, sf, opc, BOp::shifted(rm, shift, amount, negate), rn, rd, false);
        }
        Insn::AddSubImm { sf, sub, set_flags, shift12, imm12, rn, rd } => {
            let imm = u64::from(imm12) << if shift12 { 12 } else { 0 };
            arith::add_sub(f, sf, sub, set_flags, BOp::Imm(imm), rn, true, rd, !set_flags);
        }
        Insn::AddSubShiftedReg { sf, sub, set_flags, shift, amount, rm, rn, rd } => {
            arith::add_sub(f, sf, sub, set_flags, BOp::shifted(rm, shift, amount, false), rn, false, rd, false);
        }
        Insn::AddSubExtReg { sf, sub, set_flags, option, imm3, rm, rn, rd } => {
            arith::add_sub(f, sf, sub, set_flags, BOp::Ext(rm, option, imm3), rn, true, rd, !set_flags);
        }
        Insn::AddSubCarry { sf, sub, set_flags, rm, rn, rd } => {
            arith::add_sub_carry(f, sf, sub, set_flags, rm, rn, rd);
        }
        Insn::Extract { sf, rm, rn, lsb, rd } => arith::extract(f, sf, rm, rn, lsb, rd),
        Insn::PcRel { page, imm, rd } => arith::pc_rel(f, page, imm, rd, pc),
        Insn::CondSelect { sf, op, o2, cond, rm, rn, rd } => cond::cond_select(f, sf, op, o2, cond, rm, rn, rd),
        Insn::CondCompare { sf, sub, is_imm, imm_y, rm, cond, nzcv, rn } => {
            cond::cond_compare(f, sf, sub, is_imm, imm_y, rm, cond, nzcv, rn);
        }
        Insn::Bitfield { sf, opc, wmask, tmask, immr, imms, rn, rd } => {
            dataproc::bitfield(f, sf, opc, wmask, tmask, immr, imms, rn, rd);
        }
        Insn::DataProc1Src { sf, opcode, rn, rd } => {
            let ok = dataproc::data_proc_1src(f, sf, opcode, rn, rd);
            return finish(f, pc, ok);
        }
        Insn::DataProc2Src { sf, opcode, rm, rn, rd } => {
            let ok = dataproc::data_proc_2src(f, sf, opcode, rm, rn, rd);
            return finish(f, pc, ok);
        }
        Insn::DataProc3Src { sf, op31, o0, rm, ra, rn, rd } => {
            let ok = dataproc::data_proc_3src(f, sf, op31, o0, rm, ra, rn, rd);
            return finish(f, pc, ok);
        }
        Insn::LoadStore { .. } => {
            let ok = memory::load_store(f, insn, pc, guest_base);
            return finish(f, pc, ok);
        }
        Insn::LoadStorePair { .. } => {
            let ok = memory::load_store_pair(f, insn, guest_base);
            return finish(f, pc, ok);
        }
        Insn::SimdThreeSame { q, u, size, opcode, rm, rn, rd } => {
            let ok = simd::simd_three_same(f, q, u, size, opcode, rm, rn, rd);
            return finish(f, pc, ok);
        }
        Insn::SimdTwoRegMisc { q, u, size, opcode, rn, rd } => {
            let ok = simd::simd_two_reg_misc(f, q, u, size, opcode, rn, rd);
            return finish(f, pc, ok);
        }
        Insn::SimdThreeDiff { q, u, size, opcode, rm, rn, rd } => {
            let ok = simd::simd_three_diff(f, q, u, size, opcode, rm, rn, rd);
            return finish(f, pc, ok);
        }
        Insn::SimdModImm { q, op, cmode, imm8, rd } => {
            let ok = simd::simd_mod_imm(f, q, op, cmode, imm8, rd);
            return finish(f, pc, ok);
        }
        Insn::SimdDupGeneral { q, size, rn, rd } => simd::simd_dup_general(f, q, size, rn, rd),
        Insn::SimdDupElement { q, size, index, rn, rd } => simd::simd_dup_element(f, q, size, index, rn, rd),
        Insn::SimdInsGeneral { size, index, rn, rd } => simd::simd_ins_general(f, size, index, rn, rd),
        Insn::SimdInsElement { size, dst, src, rn, rd } => simd::simd_ins_element(f, size, dst, src, rn, rd),
        Insn::SimdMovToGpr { signed, dst64, size, index, vn, rd } => {
            simd::simd_mov_to_gpr(f, signed, dst64, size, index, vn, rd);
        }
        Insn::SimdZipTrn { q, size, opcode, rm, rn, rd } => simd::simd_zip_trn(f, q, size, opcode, rm, rn, rd),
        Insn::SimdExt { q, imm4, rm, rn, rd } => simd::simd_ext(f, q, imm4, rm, rn, rd),
        Insn::SimdTableLookup { q, is_tbx, len, rm, rn, rd } => {
            let ok = simd::simd_tbl(f, q, is_tbx, len, rm, rn, rd);
            return finish(f, pc, ok);
        }
        Insn::SimdShiftImm { q, u, immh, immb, opcode, rn, rd } => {
            let ok = simd::simd_shift_imm(f, q, u, immh, immb, opcode, rn, rd);
            return finish(f, pc, ok);
        }
        _ => return false,
    }
    advance_pc(f, pc);
    true
}

/// Advance the PC and report success only when the inner lowering succeeded.
/// (The fallible lowerings never emit before deciding, so `false` is clean.)
fn finish(f: &mut Function, pc: u64, ok: bool) -> bool {
    if ok {
        advance_pc(f, pc);
    }
    ok
}
