//! Instruction dispatch: decode-time [`Insn`] -> the per-class executor.
//!
//! `pc` is the executing instruction's own address. The return value is an
//! optional PC override: `Some(target)` for taken branches, `None` for
//! sequential instructions (the run loop then advances PC by 4).

use aarch64_cpu_state::CpuState;
use aarch64_decoder::Insn;

use crate::memory::Memory;
use crate::{
    add_sub_carry, add_sub_ext_reg, add_sub_imm, add_sub_shifted_reg, bitfield, branch_imm,
    branch_reg, compare_branch, cond_branch, cond_compare, cond_select, data_proc_1src,
    data_proc_2src, data_proc_3src, exception, extract, fp, ldst, ldst_atomic, ldst_cas, ldst_excl,
    ldst_pair, logical_imm, logical_reg, move_wide, pc_rel, simd_across, simd_copy, simd_dup,
    simd_ext, simd_indexed, simd_ldst_struct, simd_tbl,
    simd_mod_imm, simd_permute, simd_scalar, simd_shift_imm, simd_three_diff, simd_three_same,
    simd_three_same_fp, simd_two_reg_misc, simd_two_reg_misc_fp, system, test_branch,
};

pub(crate) fn execute(cpu: &mut CpuState, mem: &mut Memory, insn: Insn, pc: u64) -> Option<u64> {
    match insn {
        Insn::MoveWide { sf, opc, hw, imm16, rd } => move_wide::exec(cpu, sf, opc, hw, imm16, rd),
        Insn::AddSubImm { sf, sub, set_flags, shift12, imm12, rn, rd } => {
            add_sub_imm::exec(cpu, sf, sub, set_flags, shift12, imm12, rn, rd)
        }
        Insn::AddSubShiftedReg { sf, sub, set_flags, shift, amount, rm, rn, rd } => {
            add_sub_shifted_reg::exec(cpu, sf, sub, set_flags, shift, amount, rm, rn, rd)
        }
        Insn::AddSubCarry { sf, sub, set_flags, rm, rn, rd } => {
            add_sub_carry::exec(cpu, sf, sub, set_flags, rm, rn, rd)
        }
        Insn::AddSubExtReg { sf, sub, set_flags, option, imm3, rm, rn, rd } => {
            add_sub_ext_reg::exec(cpu, sf, sub, set_flags, option, imm3, rm, rn, rd)
        }
        Insn::LogicalImm { sf, opc, imm, rn, rd } => logical_imm::exec(cpu, sf, opc, imm, rn, rd),
        Insn::LogicalShiftedReg { sf, opc, negate, shift, amount, rm, rn, rd } => {
            logical_reg::exec(cpu, sf, opc, negate, shift, amount, rm, rn, rd)
        }
        Insn::Bitfield { sf, opc, wmask, tmask, immr, imms, rn, rd } => {
            bitfield::exec(cpu, sf, opc, wmask, tmask, immr, imms, rn, rd)
        }
        Insn::Extract { sf, rm, rn, lsb, rd } => extract::exec(cpu, sf, rm, rn, lsb, rd),
        Insn::CondSelect { sf, op, o2, cond, rm, rn, rd } => {
            cond_select::exec(cpu, sf, op, o2, cond, rm, rn, rd)
        }
        Insn::CondCompare { sf, sub, is_imm, imm_y, rm, cond, nzcv, rn } => {
            cond_compare::exec(cpu, sf, sub, is_imm, imm_y, rm, cond, nzcv, rn)
        }
        Insn::DataProc1Src { sf, opcode, rn, rd } => data_proc_1src::exec(cpu, sf, opcode, rn, rd),
        Insn::DataProc2Src { sf, opcode, rm, rn, rd } => {
            data_proc_2src::exec(cpu, sf, opcode, rm, rn, rd)
        }
        Insn::DataProc3Src { sf, op31, o0, rm, ra, rn, rd } => {
            data_proc_3src::exec(cpu, sf, op31, o0, rm, ra, rn, rd)
        }
        Insn::PcRel { page, imm, rd } => pc_rel::exec(cpu, page, imm, rd, pc),
        Insn::BranchImm { link, offset } => branch_imm::exec(cpu, link, offset, pc),
        Insn::CondBranch { cond, offset } => cond_branch::exec(cpu, cond, offset, pc),
        Insn::CompareBranch { sf, negate, rt, offset } => {
            compare_branch::exec(cpu, sf, negate, rt, offset, pc)
        }
        Insn::TestBranch { bit, negate, rt, offset } => {
            test_branch::exec(cpu, bit, negate, rt, offset, pc)
        }
        Insn::BranchReg { opc, rn } => branch_reg::exec(cpu, opc, rn, pc),
        Insn::LoadStore { size, is_load, signed, dst64, vec, rt, addr } => {
            ldst::exec(cpu, mem, size, is_load, signed, dst64, vec, rt, addr, pc)
        }
        Insn::LoadStorePair { is_load, signed, width8, vec, vesize, rt, rt2, rn, offset, index } => {
            ldst_pair::exec(cpu, mem, is_load, signed, width8, vec, vesize, rt, rt2, rn, offset, index)
        }
        Insn::AtomicRmw { size, op, rs, rn, rt } => {
            ldst_atomic::exec(cpu, mem, size, op, rs, rn, rt)
        }
        Insn::CompareSwap { size, rs, rn, rt } => ldst_cas::exec(cpu, mem, size, rs, rn, rt),
        Insn::LoadExclusive { size, rt, rn } => ldst_excl::load(cpu, mem, size, rt, rn),
        Insn::StoreExclusive { size, rs, rt, rn } => ldst_excl::store(cpu, mem, size, rs, rt, rn),
        Insn::FpDataProc1 { ftype, opcode, rn, rd } => fp::dp1(cpu, ftype, opcode, rn, rd),
        Insn::FpDataProc2 { ftype, opcode, rm, rn, rd } => fp::dp2(cpu, ftype, opcode, rm, rn, rd),
        Insn::FpDataProc3 { ftype, o1, o0, rm, ra, rn, rd } => {
            fp::dp3(cpu, ftype, o1, o0, rm, ra, rn, rd)
        }
        Insn::FpCondCompare { ftype, rm, rn, cond, nzcv, signaling } => {
            fp::ccmp(cpu, ftype, rm, rn, cond, nzcv, signaling)
        }
        Insn::FpCompare { ftype, rm, rn, cmp_zero, signaling: _ } => {
            fp::compare(cpu, ftype, rm, rn, cmp_zero)
        }
        Insn::FpCondSelect { ftype, cond, rm, rn, rd } => fp::csel(cpu, ftype, cond, rm, rn, rd),
        Insn::FpCvtInt { sf, ftype, rmode, opcode, rn, rd } => {
            fp::cvt_int(cpu, sf, ftype, rmode, opcode, rn, rd)
        }
        Insn::FpImm { ftype, imm8, rd } => fp::imm(cpu, ftype, imm8, rd),
        Insn::SimdThreeSame { q, u, size, opcode, rm, rn, rd } => {
            simd_three_same::exec(cpu, q, u, size, opcode, rm, rn, rd)
        }
        Insn::SimdThreeSameFp { q, sz, fpopcode, rm, rn, rd } => {
            simd_three_same_fp::exec(cpu, q, sz, fpopcode, rm, rn, rd)
        }
        Insn::SimdThreeDiff { q, u, size, opcode, rm, rn, rd } => {
            simd_three_diff::exec(cpu, q, u, size, opcode, rm, rn, rd)
        }
        Insn::SimdIndexed { q, u, size, opcode, index, rm, rn, rd } => {
            simd_indexed::exec(cpu, q, u, size, opcode, index, rm, rn, rd)
        }
        Insn::SimdTableLookup { q, is_tbx, len, rm, rn, rd } => {
            simd_tbl::exec(cpu, q, is_tbx, len, rm, rn, rd)
        }
        Insn::SimdLdStMulti { is_load, q, postidx, rm, rn, rt, size, rpt, selem } => {
            simd_ldst_struct::multi(cpu, mem, is_load, q, postidx, rm, rn, rt, size, rpt, selem)
        }
        Insn::SimdLdStSingle { is_load, replicate, postidx, rm, rn, rt, size, selem, index, q } => {
            simd_ldst_struct::single(
                cpu, mem, is_load, replicate, postidx, rm, rn, rt, size, selem, index, q,
            )
        }
        Insn::SimdScalarThreeSame { u, size, opcode, rm, rn, rd } => {
            simd_scalar::three_same(cpu, u, size, opcode, rm, rn, rd)
        }
        Insn::SimdScalarTwoRegMisc { u, size, opcode, rn, rd } => {
            simd_scalar::two_reg_misc(cpu, u, size, opcode, rn, rd)
        }
        Insn::SimdScalarPairwise { u, size, opcode, rn, rd } => {
            simd_scalar::pairwise(cpu, u, size, opcode, rn, rd)
        }
        Insn::SimdScalarThreeDiff { size, opcode, rm, rn, rd } => {
            simd_scalar::three_diff(cpu, size, opcode, rm, rn, rd)
        }
        Insn::SimdScalarCopy { imm5, rn, rd } => simd_scalar::copy(cpu, imm5, rn, rd),
        Insn::SimdScalarIndexed { u, size, opcode, index, rm, rn, rd } => {
            simd_scalar::indexed(cpu, u, size, opcode, index, rm, rn, rd)
        }
        Insn::SimdScalarShiftImm { u, immh, immb, opcode, rn, rd } => {
            simd_scalar::shift_imm(cpu, u, immh, immb, opcode, rn, rd)
        }
        Insn::SimdAcrossLanes { q, u, size, opcode, rn, rd } => {
            simd_across::exec(cpu, q, u, size, opcode, rn, rd)
        }
        Insn::SimdTwoRegMisc { q, u, size, opcode, rn, rd } => {
            simd_two_reg_misc::exec(cpu, q, u, size, opcode, rn, rd)
        }
        Insn::SimdTwoRegMiscFp { q, sz, fpop, rn, rd } => {
            simd_two_reg_misc_fp::exec(cpu, q, sz, fpop, rn, rd)
        }
        Insn::SimdModImm { q, op, cmode, imm8, rd } => {
            simd_mod_imm::exec(cpu, q, op, cmode, imm8, rd)
        }
        Insn::SimdDupGeneral { q, size, rn, rd } => simd_dup::exec(cpu, q, size, rn, rd),
        Insn::SimdDupElement { q, size, index, rn, rd } => {
            simd_copy::dup_element(cpu, q, size, index, rn, rd)
        }
        Insn::SimdInsGeneral { size, index, rn, rd } => {
            simd_copy::ins_general(cpu, size, index, rn, rd)
        }
        Insn::SimdInsElement { size, dst, src, rn, rd } => {
            simd_copy::ins_element(cpu, size, dst, src, rn, rd)
        }
        Insn::SimdMovToGpr { signed, dst64, size, index, vn, rd } => {
            simd_copy::mov_to_gpr(cpu, signed, dst64, size, index, vn, rd)
        }
        Insn::SimdZipTrn { q, size, opcode, rm, rn, rd } => {
            simd_permute::exec(cpu, q, size, opcode, rm, rn, rd)
        }
        Insn::SimdExt { q, imm4, rm, rn, rd } => simd_ext::exec(cpu, q, imm4, rm, rn, rd),
        Insn::SimdShiftImm { q, u, immh, immb, opcode, rn, rd } => {
            simd_shift_imm::exec(cpu, q, u, immh, immb, opcode, rn, rd)
        }
        Insn::SysRegMove { read, key, rt } => system::exec(cpu, read, key, rt),
        Insn::MsrImm { op1, op2, crm } => exception::msr_imm(cpu, op1, op2, crm),
        Insn::Svc { imm16 } => exception::svc(cpu, imm16, pc),
        Insn::Eret => exception::eret(cpu),
        Insn::Nop => None,
        Insn::Unsupported { .. } => unreachable!("filtered in run()"),
    }
}
