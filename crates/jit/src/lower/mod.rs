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

use aarch64_cpu_state::regs::offsets;
use aarch64_cpu_state::JIT_COUNT_OFFSET;
use aarch64_decoder::{Block, Insn};
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

use arith::BOp;

pub(crate) use common::{SCRATCH_I32, SCRATCH_I64};
pub(crate) use memory::{lower_mem, lower_mem_pair};
pub(crate) use terminator::{lower_terminator, taken_target};

/// Iteration cap on an internally-emitted self-loop before it returns to the
/// organizer so timers/IRQs get serviced — our analogue of v86's `LOOP_COUNTER`.
/// Only pure-ALU self-loops (no memory ops) are emitted as loops, so they can't
/// be waiting on external state; this just bounds timer latency.
const MAX_LOOP: u32 = 8192;

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

/// Emit a block whose terminator branches back to its own entry as an internal
/// wasm `loop`, so the iterations run in compiled code instead of one organizer
/// round-trip per iteration. Bounded by [`MAX_LOOP`] (then it returns the entry
/// PC so the organizer re-enters after servicing async events). Reports the
/// instruction count it actually ran via the `jit_count` slot.
///
/// Shape (`$exit` outer, `$top` the loop):
/// ```text
///   block $exit (result i64)
///     loop $top
///       <body>                       ;; the non-terminator instructions
///       <taken?>                      ;; i32: 1 = branch back to entry
///       PASSES += 1
///       if (taken)
///         if (PASSES < MAX_LOOP) { br $top }            ;; keep looping in-wasm
///         else { jit_count = PASSES*len; -> $exit(entry) } ;; yield to organizer
///       else { jit_count = PASSES*len; -> $exit(fallthrough) }  ;; loop done
///     end
///     unreachable                     ;; only reached via the brs above
///   end
/// ```
pub(crate) fn emit_self_loop(f: &mut Function, block: &Block, entry_pc: u64) {
    let n = block.insns.len();
    let (last_pc, last_insn) = &block.insns[n - 1];
    let fallthrough = last_pc.wrapping_add(4);

    emit!(f, I::Block(BlockType::Result(ValType::I64))); // $exit
    emit!(f, I::Loop(BlockType::Empty)); // $top
    for (pc, insn) in &block.insns[..n - 1] {
        let ok = lower_sequential(f, insn, *pc, entry_pc, 0);
        debug_assert!(ok, "self-loop body instruction must be inline-lowerable");
    }
    terminator::emit_taken_cond(f, last_insn); // i32: 1 = take the back-branch
    emit!(f, I::LocalGet(common::PASSES), I::I32Const(1), I::I32Add, I::LocalSet(common::PASSES));
    emit!(f, I::If(BlockType::Empty)); // taken?
    emit!(f, I::LocalGet(common::PASSES), I::I32Const(MAX_LOOP as i32), I::I32LtU);
    emit!(f, I::If(BlockType::Empty)); // under the iteration cap?
    emit!(f, I::Br(2)); // continue: -> loop $top
    emit!(f, I::Else);
    common::store_count_loop(f, n as u64);
    common::gen_rel_pc(f, entry_pc, entry_pc); // yield: re-enter at the loop top
    emit!(f, I::Br(3)); // -> block $exit
    emit!(f, I::End);
    emit!(f, I::Else); // not taken: the loop is done
    common::store_count_loop(f, n as u64);
    common::gen_rel_pc(f, fallthrough, entry_pc);
    emit!(f, I::Br(2)); // -> block $exit
    emit!(f, I::End);
    emit!(f, I::End); // end loop $top
    emit!(f, I::Unreachable); // unreachable: every path above branches out
    emit!(f, I::End); // end block $exit — leaves the next PC (i64) on the stack
}

/// Dispatch-loop iterations before a region yields to the organizer (so timers
/// and IRQs get serviced) — like v86's `LOOP_COUNTER`. Each iteration runs one
/// basic block, so this caps instructions executed between services.
const MAX_REGION_ITERS: u32 = 8192;

/// Emit a multi-block [`Region`](crate::Region) as one function with an internal
/// dispatch loop. `RPC` holds the current guest PC; an if-chain jumps to the
/// block whose entry matches it; each block's terminator either updates `RPC` and
/// re-loops (`br $top`, for an in-region target) or returns (out-of-region,
/// indirect, call, or fault). `RCOUNT` accumulates retired instructions for
/// `jit_count`; `RITERS` bounds the loop. All PCs are position-independent
/// relative to `region.entry`.
pub(crate) fn emit_region_body(region: &crate::Region, ram_phys: u64, ram_size: u64) -> Function {
    let mut f =
        Function::new([(common::SCRATCH_I64, ValType::I64), (common::SCRATCH_I32, ValType::I32)]);
    prologue(&mut f); // PC0 = runtime region-entry PC
    emit!(f, I::LocalGet(common::PC0), I::LocalSet(common::RPC));

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

    // dispatch if-chain: run a block when RPC equals its entry.
    for block in &region.blocks {
        emit!(f, I::LocalGet(common::RPC));
        common::gen_rel_pc(&mut f, block.start, region.entry);
        emit!(f, I::I64Eq, I::If(BlockType::Empty));
        emit_region_block(&mut f, block, region, ram_phys, ram_size);
        emit!(f, I::End);
    }
    // default: RPC isn't a known block entry — leave the region.
    region_exit(&mut f);
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
/// the terminator routing (continue in-region via `br $top`, or exit via return).
fn emit_region_block(f: &mut Function, block: &Block, region: &crate::Region, ram_phys: u64, ram_size: u64) {
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
        } else {
            lower_sequential(f, insn, *ipc, entry, 0);
        }
    }
    // The whole block (incl. its terminator branch, if any) retired.
    emit!(f, I::LocalGet(common::RCOUNT), I::I64Const(n as i64), I::I64Add, I::LocalSet(common::RCOUNT));

    let in_region = |t: u64| region.blocks.iter().any(|b| b.start == t);
    if !has_term {
        // Ran to a non-inline instruction at pc+4: exit so the organizer runs it.
        set_rpc(f, pc.wrapping_add(4), entry);
        region_exit(f);
        return;
    }
    match term {
        Insn::BranchImm { link: false, offset } => {
            let target = pc.wrapping_add(offset as u64);
            set_rpc(f, target, entry);
            if in_region(target) {
                emit!(f, I::Br(1)); // -> $top (enclosing: dispatch-if, loop)
            } else {
                region_exit(f);
            }
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
            region_route(f, taken, entry, in_region(taken));
            emit!(f, I::Else);
            region_route(f, ft, entry, in_region(ft));
            emit!(f, I::End);
        }
        _ => {
            // Shouldn't happen (has_term implies a known branch); exit safely.
            set_rpc(f, pc.wrapping_add(4), entry);
            region_exit(f);
        }
    }
}

/// One arm of a conditional terminator: continue in-region (`br $top`, depth 2 —
/// enclosing cond-if, dispatch-if, loop) or exit.
fn region_route(f: &mut Function, target: u64, entry: u64, in_region: bool) {
    set_rpc(f, target, entry);
    if in_region {
        emit!(f, I::Br(2));
    } else {
        region_exit(f);
    }
}

/// Try to lower a non-terminator instruction. On success advances the image PC.
pub(crate) fn lower_sequential(f: &mut Function, insn: &Insn, pc: u64, entry_pc: u64, guest_base: u64) -> bool {
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
    true
}

/// Report whether the inner lowering succeeded. (Inline blocks never write the
/// image PC per instruction — the organizer overwrites `cpu.pc` with the block's
/// returned next PC — so there's nothing to do but propagate the result.)
fn finish(_f: &mut Function, _pc: u64, ok: bool) -> bool {
    ok
}
