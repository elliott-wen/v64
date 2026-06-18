//! Inline loads/stores: single register (every addressing mode) and integer
//! pairs, against the shared linear memory. Assumes identity mapping (MMU off —
//! the runtime's only mode today); SIMD/FP, 128-bit, and odd extends fall back.
//!
//! A guest address `a` maps to linear offset `RAM_BASE + (a - guest_base)`. That
//! displacement (`RAM_BASE - guest_base`) is a constant folded into the address
//! arithmetic at emit time. Out-of-bounds accesses trap in WASM and surface as a
//! guest fault (see the runtime), matching the interpreter's failure.

use aarch64_cpu_state::regs::offsets;
use aarch64_cpu_state::{
    EL_OFFSET, ENTRY_PA, ENTRY_PERMS, ENTRY_SIZE, ENTRY_TAG, JIT_COUNT_OFFSET, JIT_EXIT_OFFSET,
    TLB_ENTRIES, TLB_OFFSET,
};
use aarch64_decoder::{AddrMode, Insn, PairIndex};
use wasm_encoder::{BlockType, Function, Instruction as I};

use super::arith::push_ext;
use super::common::*;
use crate::abi;

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
pub(crate) fn lower_mem(
    f: &mut Function,
    insn: &Insn,
    pc: u64,
    entry_pc: u64,
    insns_before: u64,
    ram_phys: u64,
    ram_size: u64,
) -> bool {
    let Insn::LoadStore { size, is_load, signed, dst64, vec, unpriv, rt, addr } = *insn else {
        return false;
    };
    if vec || unpriv || size > 3 {
        return false; // only integer 1/2/4/8-byte forms inline
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

    // entry = tlb_array + ((VA>>12) & (ENTRIES-1)) * ENTRY_SIZE  -> ADDR
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

    // fast_ok = tag-match & pa-in-RAM (& no-page-cross) & permission
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
    if bytes > 1 {
        emit!(
            f,
            I::LocalGet(T0),
            I::I64Const(0xFFF),
            I::I64And,
            I::I64Const((0x1000 - bytes) as i64),
            I::I64LeU, // (VA & 0xFFF) <= 0x1000 - bytes
            I::I32And
        );
    }
    // permission: can_access = (el != 0) | (perms & 1 = EL0-access); a store also
    // needs the read-only bit (perms & 2) clear. (Mirrors mmu::check_perms.)
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
    if !is_load {
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
    // fast path: host = ram_base + (pa - ram_phys) + (VA & 0xFFF)  -> ADDR
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
    if is_load {
        int_load(f, size, signed, dst64, rt);
    } else {
        int_store(f, size, rt);
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
    emit!(f, I::Else);
    // bail: record progress, flag the exit, return this instruction's PC.
    emit!(f, I::LocalGet(REGS_BASE), I::I64Const(insns_before as i64), I::I64Store(at(JIT_COUNT_OFFSET)));
    emit!(f, I::LocalGet(REGS_BASE), I::I64Const(1), I::I64Store(at(JIT_EXIT_OFFSET)));
    gen_rel_pc(f, pc, entry_pc);
    emit!(f, I::Return);
    emit!(f, I::End);
    true
}

/// LDR/STR, single register (integer or SIMD/FP), every addressing mode. The
/// 128-bit-and-narrower vector forms are handled; structure loads fall back.
pub(super) fn load_store(f: &mut Function, insn: &Insn, pc: u64, guest_base: u64) -> bool {
    let Insn::LoadStore { size, is_load, signed, dst64, vec, unpriv, rt, addr } = *insn else {
        return false;
    };
    // Integer access widths are log2 0..=3; vector adds size 4 (the 128-bit Q).
    if size > 4 || (!vec && size > 3) {
        return false;
    }
    // LDTR/STTR need an EL0 permission check the JIT's direct access can't do;
    // fall back to the interpreter.
    if unpriv {
        return false;
    }

    // Compute the linear address into the ADDR local, plus any base writeback.
    let Some(writeback) = emit_ea(f, addr, pc, guest_base) else {
        return false; // unsupported addressing form
    };

    match (vec, is_load) {
        (false, true) => int_load(f, size, signed, dst64, rt),
        (false, false) => int_store(f, size, rt),
        (true, true) => vec_load(f, size, rt),
        (true, false) => vec_store(f, size, rt),
    }

    if let Some(rn) = writeback {
        let off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
        emit!(f, I::LocalGet(0), I::LocalGet(T1), I::I64Store(at(off)));
    }
    true
}

/// Emit the addressing-mode computation, leaving the linear i32 address in the
/// [`ADDR`] local. Returns `Some(writeback_reg)` (the base register to update
/// after the access, via `T1`) or `None` if the mode isn't supported.
fn emit_ea(f: &mut Function, addr: AddrMode, pc: u64, guest_base: u64) -> Option<Option<u8>> {
    let delta = i64::from(abi::RAM_BASE) - guest_base as i64;
    let mut writeback = None;
    match addr {
        AddrMode::UnsignedImm { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::I64Const((imm as i64).wrapping_add(delta)), I::I64Add, I::I32WrapI64);
        }
        AddrMode::Unscaled { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::I64Const(imm.wrapping_add(delta)), I::I64Add, I::I32WrapI64);
        }
        AddrMode::PreIndex { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::LocalSet(T0));
            wb_value(f, imm);
            writeback = Some(rn);
            emit_addr_from(f, T0, imm.wrapping_add(delta));
        }
        AddrMode::PostIndex { rn, imm } => {
            push_base_reg(f, rn);
            emit!(f, I::LocalSet(T0));
            wb_value(f, imm);
            writeback = Some(rn);
            emit_addr_from(f, T0, delta); // ea = old base
        }
        AddrMode::Literal { offset } => {
            let lin = (pc.wrapping_add(offset as u64) as i64).wrapping_add(delta);
            emit!(f, I::I64Const(lin), I::I32WrapI64);
        }
        AddrMode::RegOffset { rn, rm, option, shift } => {
            if !matches!(option, 2 | 3 | 6 | 7) {
                return None; // non-standard extend: slow path
            }
            push_base_reg(f, rn);
            push_ext(f, rm, option, shift);
            emit!(f, I::I64Add, I::I64Const(delta), I::I64Add, I::I32WrapI64);
        }
    }
    emit!(f, I::LocalSet(ADDR));
    Some(writeback)
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

/// Push the base register (r31 = SP) as a full i64.
fn push_base_reg(f: &mut Function, rn: u8) {
    let off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
    emit!(f, I::LocalGet(0), I::I64Load(at(off)));
}

/// Given a base i64 on the stack, fold in `disp` and wrap to the i32 address.
/// Push `local + disp` wrapped to an i32 linear address.
fn emit_addr_from(f: &mut Function, local: u32, disp: i64) {
    emit!(f, I::LocalGet(local), I::I64Const(disp), I::I64Add, I::I32WrapI64);
}

/// Compute the writeback value `T0 + imm` into `T1`.
fn wb_value(f: &mut Function, imm: i64) {
    emit!(f, I::LocalGet(T0), I::I64Const(imm), I::I64Add, I::LocalSet(T1));
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
) -> bool {
    let Insn::LoadStorePair { is_load, signed, width8, vec, rt, rt2, rn, offset, index, .. } = *insn
    else {
        return false;
    };
    if vec {
        return false;
    }
    let size = if width8 { 3 } else { 2 };
    let esize = if width8 { 8i64 } else { 4 };
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

    // entry -> ADDR (same as lower_mem)
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

    // fast_ok = tag & pa-in-RAM & no-page-cross(span) & permission
    emit!(
        f,
        I::LocalGet(ADDR),
        I::I64Load(at(ENTRY_TAG)),
        I::LocalGet(T0),
        I::I64Const(!0xFFF_i64),
        I::I64And,
        I::I64Eq
    );
    emit!(
        f,
        I::LocalGet(ADDR),
        I::I64Load(at(ENTRY_PA)),
        I::I64Const(ram_phys as i64),
        I::I64Sub,
        I::I64Const(ram_size as i64),
        I::I64LtU,
        I::I32And
    );
    emit!(
        f,
        I::LocalGet(T0),
        I::I64Const(0xFFF),
        I::I64And,
        I::I64Const((0x1000 - span) as i64),
        I::I64LeU,
        I::I32And
    );
    emit!(
        f,
        I::LocalGet(REGS_BASE),
        I::I32Load8U(at(EL_OFFSET)),
        I::I32Eqz,
        I::I32Eqz,
        I::LocalGet(ADDR),
        I::I32Load8U(at(ENTRY_PERMS)),
        I::I32Const(1),
        I::I32And,
        I::I32Or
    );
    if !is_load {
        emit!(
            f,
            I::LocalGet(ADDR),
            I::I32Load8U(at(ENTRY_PERMS)),
            I::I32Const(0b10),
            I::I32And,
            I::I32Eqz,
            I::I32And
        );
    }
    emit!(f, I::I32And);

    emit!(f, I::If(BlockType::Empty));
    // host -> ADDR
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
    // two accesses, esize apart
    if is_load {
        inline_pair_load(f, 0, size, signed, wide, rt);
        inline_pair_load(f, esize, size, signed, wide, rt2);
    } else {
        inline_pair_store(f, 0, size, rt);
        inline_pair_store(f, esize, size, rt2);
    }
    // pre/post-index writeback: rn = base + offset (Pre: == EA = T0; Post: T0 + offset)
    if matches!(index, PairIndex::Pre | PairIndex::Post) {
        emit!(f, I::LocalGet(REGS_BASE), I::LocalGet(T0));
        if matches!(index, PairIndex::Post) {
            emit!(f, I::I64Const(offset), I::I64Add);
        }
        emit!(f, I::I64Store(at(base_off)));
    }
    emit!(f, I::Else);
    emit!(f, I::LocalGet(REGS_BASE), I::I64Const(insns_before as i64), I::I64Store(at(JIT_COUNT_OFFSET)));
    emit!(f, I::LocalGet(REGS_BASE), I::I64Const(1), I::I64Store(at(JIT_EXIT_OFFSET)));
    gen_rel_pc(f, pc, entry_pc);
    emit!(f, I::Return);
    emit!(f, I::End);
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

/// LDP/STP/LDPSW, integer and SIMD/FP pairs.
pub(super) fn load_store_pair(f: &mut Function, insn: &Insn, guest_base: u64) -> bool {
    let Insn::LoadStorePair { is_load, signed, width8, vec, vesize, rt, rt2, rn, offset, index } = *insn else {
        return false;
    };
    let delta = i64::from(abi::RAM_BASE) - guest_base as i64;

    // base -> T0; ea displacement depends on the index mode.
    push_base_reg(f, rn);
    emit!(f, I::LocalSet(T0));
    let ea_disp = match index {
        PairIndex::Post => 0,
        PairIndex::Offset | PairIndex::Pre => offset,
    };

    if vec {
        // Two SIMD/FP elements of width `vesize` (2=S, 3=D, 4=Q), `step` apart.
        let step = 1i64 << vesize;
        for (k, vt) in [rt, rt2].into_iter().enumerate() {
            emit_addr_from(f, T0, ea_disp + step * k as i64 + delta);
            emit!(f, I::LocalSet(ADDR));
            if is_load {
                vec_load(f, vesize, vt);
            } else {
                vec_store(f, vesize, vt);
            }
        }
    } else {
        let size = if width8 { 3 } else { 2 };
        let esize = if width8 { 8i64 } else { 4 };
        if is_load {
            // LDPSW and the X form write full X; the W form zero-extends.
            let wide = width8 || signed;
            pair_load(f, T0, ea_disp + delta, size, signed, wide, rt);
            pair_load(f, T0, ea_disp + esize + delta, size, signed, wide, rt2);
        } else {
            pair_store(f, T0, ea_disp + delta, size, rt);
            pair_store(f, T0, ea_disp + esize + delta, size, rt2);
        }
    }

    if matches!(index, PairIndex::Pre | PairIndex::Post) {
        let off = if rn == 31 { offsets::SP } else { offsets::x(rn as usize) };
        emit!(f, I::LocalGet(T0), I::I64Const(offset), I::I64Add); // base + offset
        emit!(f, I::LocalSet(T1), I::LocalGet(0), I::LocalGet(T1), I::I64Store(at(off)));
    }
    true
}

fn pair_load(f: &mut Function, base: u32, disp: i64, size: u8, signed: bool, wide: bool, rt: u8) {
    if rt != 31 {
        emit!(f, I::LocalGet(0)); // regs_base for the result store
    }
    emit_addr_from(f, base, disp);
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

fn pair_store(f: &mut Function, base: u32, disp: i64, size: u8, rt: u8) {
    emit_addr_from(f, base, disp);
    if rt == 31 {
        emit!(f, I::I64Const(0));
    } else {
        emit!(f, I::LocalGet(0), I::I64Load(at(offsets::x(rt as usize))));
    }
    emit!(f, store_op(size));
}
