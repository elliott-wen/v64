//! Inline loads/stores with a TLB-checked fast path (the v86 model): single
//! register ([`lower_mem`]) and pairs ([`lower_mem_pair`]), integer and SIMD/FP.
//!
//! Each access reads the live `CpuState` TLB for its page; on a valid, in-RAM,
//! permitted, non-page-crossing hit it loads/stores directly at
//! `ram_base + (pa - ram_phys) + page_offset`. Any miss (TLB miss, MMIO,
//! permission fault, page cross) **bails** ([`emit_bail`]) — the block records
//! its progress, flags `jit_exit`, and returns the faulting PC so the organizer
//! interprets that one access and resumes.

use aarch64_cpu_state::regs::offsets;
use aarch64_cpu_state::{
    EL_OFFSET, ENTRY_PA, ENTRY_PERMS, ENTRY_SIZE, ENTRY_TAG, JIT_COUNT_OFFSET, JIT_EXIT_OFFSET,
    TLB_ENTRIES, TLB_OFFSET,
};
use aarch64_decoder::{AddrMode, Insn, PairIndex};
use wasm_encoder::{BlockType, Function, Instruction as I};

use super::arith::push_ext;
use super::common::*;

/// Inline integer load/store with a **TLB-checked fast path** (the v86 model).
///
/// The block reads the live `CpuState` TLB (in shared linear memory) for the
/// access VA, and if the cached translation is valid for this access — tag
/// matches, target is RAM, the access doesn't cross a page, and the permissions
/// allow it at the current EL — does the load/store directly at
/// `ram_base + (pa - ram_phys) + page_offset`. On *any* miss it **bails**: it
/// records `insns_before` (the instructions this block already retired) into
/// `jit_count`, sets `jit_exit`, and returns this instruction's PC, so the
/// organizer interprets exactly this one access (handling MMIO / page fault /
/// TLB refill) and resumes. `ram_phys`/`ram_size` (the guest-physical RAM window)
/// are baked at compile time.
///
/// Returns `false` (no code emitted) for forms it doesn't inline — caller gates
/// with [`crate::is_inline_load_store`], so that path is unreachable in practice.
/// Emit the slow-path bail (shared by single + pair, region + non-region):
/// record instructions retired, flag `jit_exit`, and return this instruction's
/// PC so the organizer interprets it and resumes. `count_base` is `Some(local)`
/// in a region — `jit_count = local + insns_before` (prior in-region blocks plus
/// this block's progress) — or `None` for a single block (`jit_count =
/// insns_before`).
fn emit_bail(f: &mut Function, pc: u64, entry_pc: u64, insns_before: u64, count_base: Option<u32>) {
    emit!(f, I::LocalGet(REGS_BASE));
    match count_base {
        Some(cl) => emit!(f, I::LocalGet(cl), I::I64Const(insns_before as i64), I::I64Add),
        None => emit!(f, I::I64Const(insns_before as i64)),
    }
    emit!(f, I::I64Store(at(JIT_COUNT_OFFSET)));
    emit!(f, I::LocalGet(REGS_BASE), I::I64Const(1), I::I64Store(at(JIT_EXIT_OFFSET)));
    gen_rel_pc(f, pc, entry_pc);
    emit!(f, I::Return);
}

/// Emit the shared TLB fast-path gate, assuming the access VA is in [`T0`].
/// Computes the TLB entry into [`ADDR`], then checks tag-match + RAM-range +
/// no-page-cross (`span` bytes) + permission (read access; a `store` also needs
/// the read-only bit clear — mirrors `mmu::check_perms`), and opens
/// `if (fast_ok)` with the host linear address `ram_base + (pa - ram_phys) +
/// page_offset` left in [`ADDR`]. The caller emits the access(es) and any base
/// writeback, then calls [`close_fast_path`].
fn open_fast_path(f: &mut Function, span: u64, store: bool, ram_phys: u64, ram_size: u64) {
    // entry = tlb_array + ((VA>>12) & (ENTRIES-1)) * ENTRY_SIZE  -> ADDR.
    // (CpuState.tlb is `Tlb { entries: Box<[Entry; N]> }`; the box is a thin
    // pointer at TLB_OFFSET, i.e. the array base.)
    emit!(f, I::LocalGet(REGS_BASE), I::I32Load(at(TLB_OFFSET)));
    emit!(
        f,
        I::LocalGet(T0),
        I::I64Const(12),
        I::I64ShrU,
        I::I64Const((TLB_ENTRIES - 1) as i64),
        I::I64And,
        I::I32WrapI64,
        I::I32Const(ENTRY_SIZE as i32),
        I::I32Mul,
        I::I32Add,
        I::LocalSet(ADDR)
    );
    // fast_ok = tag-match & pa-in-RAM (& no-page-cross) & permission.
    emit!(
        f,
        I::LocalGet(ADDR),
        I::I64Load(at(ENTRY_TAG)),
        I::LocalGet(T0),
        I::I64Const(!0xFFF_i64),
        I::I64And,
        I::I64Eq // tag == VA & ~0xFFF
    );
    emit!(
        f,
        I::LocalGet(ADDR),
        I::I64Load(at(ENTRY_PA)),
        I::I64Const(ram_phys as i64),
        I::I64Sub,
        I::I64Const(ram_size as i64),
        I::I64LtU, // (pa - ram_phys) <u ram_size
        I::I32And
    );
    if span > 1 {
        emit!(
            f,
            I::LocalGet(T0),
            I::I64Const(0xFFF),
            I::I64And,
            I::I64Const((0x1000 - span) as i64),
            I::I64LeU, // (VA & 0xFFF) <= 0x1000 - span
            I::I32And
        );
    }
    // permission: can_access = (el != 0) | (perms & 1 = EL0-access); a store also
    // needs the read-only bit (perms & 2) clear.
    emit!(
        f,
        I::LocalGet(REGS_BASE),
        I::I32Load8U(at(EL_OFFSET)),
        I::I32Eqz,
        I::I32Eqz, // el != 0
        I::LocalGet(ADDR),
        I::I32Load8U(at(ENTRY_PERMS)),
        I::I32Const(1),
        I::I32And, // perms & 1
        I::I32Or
    );
    if store {
        emit!(
            f,
            I::LocalGet(ADDR),
            I::I32Load8U(at(ENTRY_PERMS)),
            I::I32Const(0b10),
            I::I32And,
            I::I32Eqz, // !(perms & 2)
            I::I32And
        );
    }
    emit!(f, I::I32And); // fold permission into fast_ok

    emit!(f, I::If(BlockType::Empty));
    // host = ram_base + (pa - ram_phys) + (VA & 0xFFF)  -> ADDR
    emit!(f, I::LocalGet(RAM_BASE));
    emit!(
        f,
        I::LocalGet(ADDR),
        I::I64Load(at(ENTRY_PA)),
        I::I64Const(ram_phys as i64),
        I::I64Sub,
        I::I32WrapI64,
        I::I32Add
    );
    emit!(f, I::LocalGet(T0), I::I64Const(0xFFF), I::I64And, I::I32WrapI64, I::I32Add);
    emit!(f, I::LocalSet(ADDR));
}

/// Close the [`open_fast_path`] `if`: the `else` arm bails to the interpreter.
fn close_fast_path(f: &mut Function, pc: u64, entry_pc: u64, insns_before: u64, count_base: Option<u32>) {
    emit!(f, I::Else);
    emit_bail(f, pc, entry_pc, insns_before, count_base);
    emit!(f, I::End);
}

pub(crate) fn lower_mem(
    f: &mut Function,
    insn: &Insn,
    pc: u64,
    entry_pc: u64,
    insns_before: u64,
    ram_phys: u64,
    ram_size: u64,
    count_base: Option<u32>,
) -> bool {
    let Insn::LoadStore { size, is_load, signed, dst64, vec, unpriv, rt, addr } = *insn else {
        return false;
    };
    if unpriv || size > if vec { 4 } else { 3 } {
        return false; // LDTR/STTR, or out-of-range width
    }
    let bytes = 1u64 << size;

    // Compute the access VA into T0, and decide any base-register writeback.
    // `writeback = Some((rn, post, imm))`: update rn after the access — Pre
    // writes the EA itself (which is `T0`), Post writes `base + imm` (`T0 + imm`,
    // since for Post `T0` is the un-incremented base).
    let mut writeback: Option<(u8, bool, i64)> = None;
    match addr {
        AddrMode::UnsignedImm { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::I64Const(imm as i64), I::I64Add, I::LocalSet(T0));
        }
        AddrMode::Unscaled { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::I64Const(imm), I::I64Add, I::LocalSet(T0));
        }
        AddrMode::PreIndex { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::I64Const(imm), I::I64Add, I::LocalSet(T0)); // EA = base + imm
            writeback = Some((rn, false, imm));
        }
        AddrMode::PostIndex { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::LocalSet(T0)); // EA = base
            writeback = Some((rn, true, imm));
        }
        AddrMode::RegOffset { rn, rm, option, shift } => {
            if !matches!(option, 2 | 3 | 6 | 7) {
                return false; // non-standard extend: interpreter
            }
            push_base_reg(f, rn);
            push_ext(f, rm, option, shift);
            emit!(f, I::I64Add, I::LocalSet(T0));
        }
        AddrMode::Literal { .. } => return false,
    }

    open_fast_path(f, bytes, !is_load, ram_phys, ram_size);
    match (vec, is_load) {
        (false, true) => int_load(f, size, signed, dst64, rt),
        (false, false) => int_store(f, size, rt),
        (true, true) => vec_load(f, size, rt),
        (true, false) => vec_store(f, size, rt),
    }
    // Pre/post-index writeback: rn = base + imm (Pre: == EA = T0; Post: T0 + imm).
    if let Some((rn, post, imm)) = writeback {
        let wb_off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
        emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0));
        if post {
            emit!(f, I::I64Const(imm), I::I64Add);
        }
        emit!(f, I::I64Store(at(wb_off)));
    }
    close_fast_path(f, pc, entry_pc, insns_before, count_base);
    true
}

/// Integer load from [`ADDR`] into register `rt`.
fn int_load(f: &mut Function, size: u8, signed: bool, dst64: bool, rt: u8) {
    if rt != 31 {
        emit!(f, I::LocalGet(0)); // regs_base for the result store
    }
    emit!(f, I::LocalGet(ADDR), load_op(size, signed));
    if signed && !dst64 {
        emit!(f, I::I64Const(W_MASK), I::I64And); // sign-extend to 32, then zero-extend
    }
    if rt == 31 {
        emit!(f, I::Drop);
    } else {
        emit!(f, I::I64Store(at(offsets::x(rt as usize))));
    }
}

/// Integer store of register `rt` to [`ADDR`].
fn int_store(f: &mut Function, size: u8, rt: u8) {
    emit!(f, I::LocalGet(ADDR));
    if rt == 31 {
        emit!(f, I::I64Const(0)); // STR of XZR
    } else {
        emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rt as usize))));
    }
    emit!(f, store_op(size));
}

/// Vector load from [`ADDR`] into V[rt], zeroing the unused high bytes (a
/// SIMD/FP load writes the whole 128-bit register).
fn vec_load(f: &mut Function, size: u8, rt: u8) {
    if size == 4 {
        // 128-bit: copy 16 bytes straight into the V slot.
        emit!(f, I::LocalGet(0), I::LocalGet(ADDR), I::V128Load(at(0)), I::V128Store(at(offsets::v(rt as usize))));
    } else {
        // 8..64-bit: zero-extended low half, zero the high half.
        emit!(f, I::LocalGet(0), I::LocalGet(ADDR), load_op(size, false), I::I64Store(at(offsets::v(rt as usize))));
        emit!(f, I::LocalGet(0), I::I64Const(0), I::I64Store(at(offsets::v(rt as usize) + 8)));
    }
}

/// Vector store of the low `1 << size` bytes of V[rt] to [`ADDR`].
fn vec_store(f: &mut Function, size: u8, rt: u8) {
    if size == 4 {
        emit!(f, I::LocalGet(ADDR), I::LocalGet(0), I::V128Load(at(offsets::v(rt as usize))), I::V128Store(at(0)));
    } else {
        emit!(f, I::LocalGet(ADDR), I::LocalGet(0), I::I64Load(at(offsets::v(rt as usize))), store_op(size));
    }
}

/// Push the addressing base register (r31 = SP) as a full i64.
fn push_base_reg(f: &mut Function, rn: u8) {
    let off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
    emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(off)));
}

fn load_op(size: u8, signed: bool) -> I<'static> {
    if signed {
        match size {
            0 => I::I64Load8S(at(0)),
            1 => I::I64Load16S(at(0)),
            2 => I::I64Load32S(at(0)),
            _ => I::I64Load(at(0)),
        }
    } else {
        match size {
            0 => I::I64Load8U(at(0)),
            1 => I::I64Load16U(at(0)),
            2 => I::I64Load32U(at(0)),
            _ => I::I64Load(at(0)),
        }
    }
}

fn store_op(size: u8) -> I<'static> {
    match size {
        0 => I::I64Store8(at(0)),
        1 => I::I64Store16(at(0)),
        2 => I::I64Store32(at(0)),
        _ => I::I64Store(at(0)),
    }
}

/// Inline integer `LDP`/`STP`/`LDPSW` with the same TLB-checked fast path as
/// [`lower_mem`] — the pair's `2 * element_size` bytes share one TLB entry (the
/// page-cross check covers the whole span), so one lookup serves both accesses.
/// Pre/post-index writeback updates the base register after the accesses. Bails
/// (whole instruction, no partial state) on any fast-path miss.
pub(crate) fn lower_mem_pair(
    f: &mut Function,
    insn: &Insn,
    pc: u64,
    entry_pc: u64,
    insns_before: u64,
    ram_phys: u64,
    ram_size: u64,
    count_base: Option<u32>,
) -> bool {
    let Insn::LoadStorePair { is_load, signed, width8, vec, vesize, rt, rt2, rn, offset, index } =
        *insn
    else {
        return false;
    };
    // Integer pair: 4- or 8-byte elements. Vector pair: element size `vesize`
    // (2=S/4B, 3=D/8B, 4=Q/16B).
    let (size, esize) = if vec {
        (vesize, 1i64 << vesize)
    } else {
        (if width8 { 3 } else { 2 }, if width8 { 8i64 } else { 4 })
    };
    let span = 2 * esize as u64;
    let wide = width8 || signed; // X form / LDPSW write full 64-bit; W form zero-extends
    let ea_disp = match index {
        PairIndex::Post => 0,
        PairIndex::Offset | PairIndex::Pre => offset,
    };

    // EA = base(rn; SP for r31) + ea_disp  -> T0
    let base_off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
    emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(base_off)));
    if ea_disp != 0 {
        emit!(f, I::I64Const(ea_disp), I::I64Add);
    }
    emit!(f, I::LocalSet(T0));

    open_fast_path(f, span, !is_load, ram_phys, ram_size);
    // two accesses, esize apart
    match (vec, is_load) {
        (false, true) => {
            inline_pair_load(f, 0, size, signed, wide, rt);
            inline_pair_load(f, esize, size, signed, wide, rt2);
        }
        (false, false) => {
            inline_pair_store(f, 0, size, rt);
            inline_pair_store(f, esize, size, rt2);
        }
        (true, true) => {
            inline_vec_pair_load(f, 0, size, rt);
            inline_vec_pair_load(f, esize, size, rt2);
        }
        (true, false) => {
            inline_vec_pair_store(f, 0, size, rt);
            inline_vec_pair_store(f, esize, size, rt2);
        }
    }
    // pre/post-index writeback: rn = base + offset (Pre: == EA = T0; Post: T0 + offset)
    if matches!(index, PairIndex::Pre | PairIndex::Post) {
        emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0));
        if matches!(index, PairIndex::Post) {
            emit!(f, I::I64Const(offset), I::I64Add);
        }
        emit!(f, I::I64Store(at(base_off)));
    }
    close_fast_path(f, pc, entry_pc, insns_before, count_base);
    true
}

/// One element of an inline pair load, at byte offset `host_off` from [`ADDR`].
fn inline_pair_load(f: &mut Function, host_off: i64, size: u8, signed: bool, wide: bool, rt: u8) {
    if rt != 31 {
        emit!(f, I::LocalGet(REGS_BASE)); // regs_base for the result store
    }
    emit!(f, I::LocalGet(ADDR));
    if host_off != 0 {
        emit!(f, I::I32Const(host_off as i32), I::I32Add);
    }
    emit!(f, load_op(size, signed));
    if signed && !wide {
        emit!(f, I::I64Const(W_MASK), I::I64And);
    }
    if rt == 31 {
        emit!(f, I::Drop);
    } else {
        emit!(f, I::I64Store(at(offsets::x(rt as usize))));
    }
}

/// One element of an inline vector pair load, at byte offset `host_off` from
/// [`ADDR`], into V[rt] (zeroing the unused high bytes — a SIMD load writes the
/// whole 128-bit register).
fn inline_vec_pair_load(f: &mut Function, host_off: i64, size: u8, rt: u8) {
    let v = offsets::v(rt as usize);
    if size == 4 {
        emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(ADDR));
        if host_off != 0 {
            emit!(f, I::I32Const(host_off as i32), I::I32Add);
        }
        emit!(f, I::V128Load(at(0)), I::V128Store(at(v)));
    } else {
        emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(ADDR));
        if host_off != 0 {
            emit!(f, I::I32Const(host_off as i32), I::I32Add);
        }
        emit!(f, load_op(size, false), I::I64Store(at(v)));
        emit!(f, I::LocalGet(REGS_BASE), I::I64Const(0), I::I64Store(at(v + 8)));
    }
}

/// One element of an inline vector pair store, the low `1 << size` bytes of
/// V[rt], at byte offset `host_off` from [`ADDR`].
fn inline_vec_pair_store(f: &mut Function, host_off: i64, size: u8, rt: u8) {
    let v = offsets::v(rt as usize);
    emit!(f, I::LocalGet(ADDR));
    if host_off != 0 {
        emit!(f, I::I32Const(host_off as i32), I::I32Add);
    }
    if size == 4 {
        emit!(f, I::LocalGet(REGS_BASE), I::V128Load(at(v)), I::V128Store(at(0)));
    } else {
        emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(v)), store_op(size));
    }
}

/// One element of an inline pair store, at byte offset `host_off` from [`ADDR`].
fn inline_pair_store(f: &mut Function, host_off: i64, size: u8, rt: u8) {
    emit!(f, I::LocalGet(ADDR));
    if host_off != 0 {
        emit!(f, I::I32Const(host_off as i32), I::I32Add);
    }
    if rt == 31 {
        emit!(f, I::I64Const(0));
    } else {
        emit!(f, I::LocalGet(REGS_BASE), I::I64Load(at(offsets::x(rt as usize))));
    }
    emit!(f, store_op(size));
}

