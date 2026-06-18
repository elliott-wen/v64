//! Inline instruction lowering, and block/region assembly.
//!
//! Each guest instruction is emitted directly as native WASM operating on the
//! live `CpuState` in shared linear memory:
//!
//! - [`lower_sequential`] — non-terminator register ops. Updates registers/flags
//!   in place; leaves nothing on the operand stack.
//! - [`lower_mem`] / [`lower_mem_pair`] — loads/stores: an inline TLB-checked
//!   fast path, bailing to the interpreter on a miss.
//! - [`lower_terminator`] — control flow (last in a block). Leaves the next guest
//!   PC on the stack as the function's `i64` result.
//!
//! Block formation ([`crate::form_jit_block`]) only admits instructions these
//! always handle, so lowering never fails mid-block. Flags are computed inline
//! into the packed NZCV word (no host calls). PCs are position-independent
//! (`runtime entry PC + delta`), so a physical block runs at any VA.
//!
//! Higher up, [`emit_self_loop`] turns a self-looping block into an internal wasm
//! `loop`, and [`emit_region_body`] assembles a multi-block region into one
//! function with a `br_table` dispatch loop. Lowerings are grouped by family:
//! [`common`] (register/flag access), [`arith`], [`cond`], [`dataproc`],
//! [`memory`], [`simd`], and [`terminator`].

/// Emit a sequence of instructions into a [`wasm_encoder::Function`]. Defined
/// before the submodule declarations so they inherit it via textual scoping.
macro_rules! emit {
    ($f:expr, $($i:expr),* $(,)?) => {{ $( $f.instruction(&$i); )* }};
}

mod arith;
mod atomic;
mod common;
mod cond;
mod dataproc;
mod fp;
mod memory;
mod simd;
mod terminator;

use aarch64_cpu_state::regs::offsets;
use aarch64_cpu_state::JIT_COUNT_OFFSET;
use aarch64_decoder::{Block, Insn};
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

use arith::BOp;

pub(crate) use atomic::lower_atomic;
pub(crate) use common::{SCRATCH_I32, SCRATCH_I64};
pub(crate) use fp::is_inline_fp;
pub(crate) use memory::{lower_mem, lower_mem_pair};
pub(crate) use simd::is_inline_simd;
pub(crate) use terminator::{lower_terminator, taken_target};

/// Emit the block prologue: cache the runtime entry PC for position-independent
/// PC math. Must be emitted before any lowering. See `common::PC0`.
pub(crate) fn prologue(f: &mut Function) {
    common::load_entry_pc(f);
}

/// Push a position-independent guest PC (`runtime entry PC + (abs - entry_pc)`).
/// Used for a block's sequential fall-through result. See `common::gen_rel_pc`.
pub(crate) fn gen_pc(f: &mut Function, abs: u64, entry_pc: u64) {
    common::gen_rel_pc(f, abs, entry_pc);
}

/// Record, in the block's `jit_count` slot, that it retired `count` instructions
/// — for a straight-line block, its static length (one pass per call).
pub(crate) fn store_count(f: &mut Function, count: u64) {
    common::store_count_const(f, count);
}

/// Dispatch-loop iterations before a region yields to the organizer (so timers
/// and IRQs get serviced) — like v86's `LOOP_COUNTER`. Each iteration runs one
/// basic block, so this caps instructions executed between services.
const MAX_REGION_ITERS: u32 = 8192;

/// Emit a multi-block [`Region`](crate::Region) as one function with an internal
/// dispatch loop. `RIDX` holds the next block index; a `br_table` jumps to it in
/// O(1); each block's terminator either sets `RIDX` (+ `RPC`) and re-loops (for an
/// in-region target) or returns (out-of-region, indirect, call, or fault).
/// `RCOUNT` accumulates retired instructions for `jit_count`; `RITERS` bounds the
/// loop. All PCs are position-independent relative to `region.entry`.
pub(crate) fn emit_region_body(region: &crate::Region, ram_phys: u64, ram_size: u64) -> Function {
    let mut f =
        Function::new([(common::SCRATCH_I64, ValType::I64), (common::SCRATCH_I32, ValType::I32)]);
    prologue(&mut f); // PC0 = runtime region-entry PC
    emit!(f, I::LocalGet(common::PC0), I::LocalSet(common::RPC));

    // RIDX (the dispatch index) is zero-initialised = the entry block, index 0.
    let k = region.blocks.len();

    emit!(f, I::Loop(BlockType::Empty)); // $top
    // safety: yield after MAX_REGION_ITERS block transitions
    emit!(
        f,
        I::LocalGet(common::RITERS),
        I::I32Const(MAX_REGION_ITERS as i32),
        I::I32GeU,
        I::If(BlockType::Empty)
    );
    region_exit(&mut f);
    emit!(f, I::End);
    emit!(f, I::LocalGet(common::RITERS), I::I32Const(1), I::I32Add, I::LocalSet(common::RITERS));

    // O(1) dispatch: a `br_table` on RIDX jumps to the matching block. The classic
    // nested-block trampoline — `k+1` blocks (one per block, plus a default);
    // `br_table` index i lands just past the i-th `end`, where block i's code is.
    for _ in 0..=k {
        emit!(f, I::Block(BlockType::Empty));
    }
    let targets: Vec<u32> = (0..k as u32).collect();
    emit!(f, I::LocalGet(common::RIDX), I::BrTable(targets.into(), k as u32));
    for (i, block) in region.blocks.iter().enumerate() {
        emit!(f, I::End); // close the i-th-innermost dispatch block
        // From here, `k - i` blocks remain open before `$top`, so `br (k - i)`
        // re-dispatches; one deeper (`+1`) inside a conditional terminator's `if`.
        emit_region_block(&mut f, block, region, ram_phys, ram_size, (k - i) as u32);
    }
    emit!(f, I::End); // close the default block
    region_exit(&mut f); // default: RIDX out of range (never happens) — leave.
    emit!(f, I::End); // end $top
    emit!(f, I::Unreachable);
    f.instruction(&I::End);
    f
}

/// Store `jit_count = RCOUNT` and return `RPC` — the region's exit sequence.
fn region_exit(f: &mut Function) {
    emit!(
        f,
        I::LocalGet(common::REGS_BASE),
        I::LocalGet(common::RCOUNT),
        I::I64Store(common::at(JIT_COUNT_OFFSET))
    );
    emit!(f, I::LocalGet(common::RPC), I::Return);
}

/// `RPC = ` the runtime PC for guest VA `abs` (position-independent).
fn set_rpc(f: &mut Function, abs: u64, entry: u64) {
    common::gen_rel_pc(f, abs, entry);
    emit!(f, I::LocalSet(common::RPC));
}

/// Store guest VA `abs` (PI) into register `reg` — the `BL`/`BLR` link address.
fn store_reg_pc(f: &mut Function, reg: usize, abs: u64, entry: u64) {
    emit!(f, I::LocalGet(common::REGS_BASE));
    common::gen_rel_pc(f, abs, entry);
    emit!(f, I::I64Store(common::at(offsets::x(reg))));
}

/// Emit one block inside the dispatch loop: its body, the `RCOUNT += len`, and
/// the terminator routing. An in-region branch sets RIDX (the target block index)
/// + RPC and `br`s to `$top` (`loop_depth` levels out, `+1` inside a conditional
/// `if`); an out-of-region / indirect / call / fall-through-to-non-inline branch
/// returns the PC (exits). RIDX makes re-dispatch O(1) via the `br_table`.
fn emit_region_block(
    f: &mut Function,
    block: &Block,
    region: &crate::Region,
    ram_phys: u64,
    ram_size: u64,
    loop_depth: u32,
) {
    let entry = region.entry;
    let n = block.insns.len();
    let pc = block.insns[n - 1].0;
    let term = block.insns[n - 1].1;
    let has_term = crate::eligible::is_branch(&term);
    let body_len = if has_term { n - 1 } else { n };

    for i in 0..body_len {
        let (ipc, insn) = &block.insns[i];
        if crate::is_inline_load_store(insn) {
            lower_mem(f, insn, *ipc, entry, i as u64, ram_phys, ram_size, Some(common::RCOUNT));
        } else if crate::is_inline_load_store_pair(insn) {
            lower_mem_pair(f, insn, *ipc, entry, i as u64, ram_phys, ram_size, Some(common::RCOUNT));
        } else if crate::is_inline_atomic(insn) {
            lower_atomic(f, insn, *ipc, entry, i as u64, ram_phys, ram_size, Some(common::RCOUNT));
        } else {
            lower_sequential(f, insn, *ipc, entry);
        }
    }
    // The whole block (incl. its terminator branch, if any) retired.
    emit!(f, I::LocalGet(common::RCOUNT), I::I64Const(n as i64), I::I64Add, I::LocalSet(common::RCOUNT));

    if !has_term {
        // Ran to a non-inline instruction at pc+4: exit so the organizer runs it.
        set_rpc(f, pc.wrapping_add(4), entry);
        region_exit(f);
        return;
    }
    match term {
        Insn::BranchImm { link: false, offset } => {
            region_route(f, pc.wrapping_add(offset as u64), entry, region, loop_depth);
        }
        Insn::BranchImm { link: true, offset } => {
            // BL: a call leaves the region; set the link register, then exit.
            store_reg_pc(f, 30, pc.wrapping_add(4), entry);
            set_rpc(f, pc.wrapping_add(offset as u64), entry);
            region_exit(f);
        }
        Insn::BranchReg { opc, rn } => {
            if opc == 1 {
                store_reg_pc(f, 30, pc.wrapping_add(4), entry); // BLR link
            }
            if rn == 31 {
                emit!(f, I::I64Const(0), I::LocalSet(common::RPC));
            } else {
                emit!(
                    f,
                    I::LocalGet(common::REGS_BASE),
                    I::I64Load(common::at(offsets::x(rn as usize))),
                    I::LocalSet(common::RPC)
                );
            }
            region_exit(f);
        }
        Insn::CondBranch { .. } | Insn::CompareBranch { .. } | Insn::TestBranch { .. } => {
            let taken = taken_target(&term, pc).unwrap();
            let ft = pc.wrapping_add(4);
            terminator::emit_taken_cond(f, &term); // i32: 1 = take the branch
            emit!(f, I::If(BlockType::Empty));
            region_route(f, taken, entry, region, loop_depth + 1); // +1: inside the `if`
            emit!(f, I::Else);
            region_route(f, ft, entry, region, loop_depth + 1);
            emit!(f, I::End);
        }
        _ => {
            // Shouldn't happen (has_term implies a known branch); exit safely.
            set_rpc(f, pc.wrapping_add(4), entry);
            region_exit(f);
        }
    }
}

/// Route a terminator edge to `target`: if `target` is an in-region block, set
/// RIDX to its index + RPC and `br loop_depth` to re-dispatch (stay in compiled
/// code); otherwise set RPC and return (leave the region).
fn region_route(f: &mut Function, target: u64, entry: u64, region: &crate::Region, loop_depth: u32) {
    if let Some(idx) = region.blocks.iter().position(|b| b.start == target) {
        emit!(f, I::I32Const(idx as i32), I::LocalSet(common::RIDX));
        set_rpc(f, target, entry);
        emit!(f, I::Br(loop_depth));
    } else {
        set_rpc(f, target, entry);
        region_exit(f);
    }
}

/// Try to lower a non-terminator instruction. On success advances the image PC.
pub(crate) fn lower_sequential(f: &mut Function, insn: &Insn, pc: u64, entry_pc: u64) -> bool {
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
        Insn::PcRel { page, imm, rd } => arith::pc_rel(f, page, imm, rd, pc, entry_pc),
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
        Insn::FpDataProc1 { ftype, opcode, rn, rd } => fp::dp1(f, ftype, opcode, rn, rd),
        Insn::FpDataProc2 { ftype, opcode, rm, rn, rd } => fp::dp2(f, ftype, opcode, rm, rn, rd),
        Insn::FpImm { ftype, imm8, rd } => fp::imm(f, ftype, imm8, rd),
        Insn::FpCompare { ftype, rm, rn, cmp_zero, .. } => fp::compare(f, ftype, rm, rn, cmp_zero),
        Insn::FpCondCompare { ftype, rm, rn, cond, nzcv, .. } => fp::ccmp(f, ftype, rm, rn, cond, nzcv),
        Insn::FpCondSelect { ftype, cond, rm, rn, rd } => fp::csel(f, ftype, cond, rm, rn, rd),
        // LoadStore / LoadStorePair are routed to `lower_mem` / `lower_mem_pair`
        // by the emitter (they need the TLB fast path + bail), never here.
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
    true
}

/// Report whether the inner lowering succeeded. (Inline blocks never write the
/// image PC per instruction — the organizer overwrites `cpu.pc` with the block's
/// returned next PC — so there's nothing to do but propagate the result.)
fn finish(_f: &mut Function, _pc: u64, ok: bool) -> bool {
    ok
}
