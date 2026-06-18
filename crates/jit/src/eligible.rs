//! Which instructions the JIT lowers inline vs. hands to the interpreter.
//!
//! The emitter inlines a block's leading run of register-only ALU ops, then
//! executes the block's final "escape" instruction (a branch, load/store,
//! system op — anything) by calling back into the interpreter against the shared
//! CPU + bus. So the JIT covers the whole ISA: the trivial ops are native, the
//! long tail is interpreted, and MMU/MMIO/faults come for free from the
//! interpreter.
//!
//! [`can_inline`] is the gate for *non-terminator, non-escape* instructions —
//! the ones that must be emittable inline because they aren't last in the block.
//! It is restricted to lowerings that *never decline* (so emission can't fail
//! mid-block) and that touch only the hot register image. It is kept in lockstep
//! with `lower::lower_sequential`'s always-succeeding arms; the crosscheck test
//! catches drift.

use aarch64_decoder::{decode, AddrMode, Block, Insn};

/// True if `insn` is a non-terminator the inline lowerings always handle. Most
/// are register-only ALU ops touching the hot register image; integer
/// loads/stores ([`is_inline_load_store`]) are also admitted — they emit a
/// TLB-checked fast path and bail to the interpreter on a miss (see
/// `lower::lower_mem`). Block formation extends the inline run while this holds;
/// the first instruction for which it's false (or any terminator) ends the block.
#[must_use]
pub fn can_inline(insn: &Insn) -> bool {
    is_inline_mem(insn)
        || matches!(
            insn,
            Insn::MoveWide { .. }
                | Insn::AddSubImm { .. }
                | Insn::AddSubShiftedReg { .. }
                | Insn::AddSubExtReg { .. }
                | Insn::AddSubCarry { .. }
                | Insn::LogicalImm { .. }
                | Insn::LogicalShiftedReg { .. }
                | Insn::Bitfield { .. }
                | Insn::Extract { .. }
                | Insn::PcRel { .. }
                | Insn::CondSelect { .. }
                | Insn::CondCompare { .. }
                | Insn::Nop
                | Insn::Prfm
        )
}

/// True for any integer memory access the JIT inlines — a single load/store
/// ([`is_inline_load_store`]) or a load/store pair ([`is_inline_load_store_pair`]).
#[must_use]
pub fn is_inline_mem(insn: &Insn) -> bool {
    is_inline_load_store(insn) || is_inline_load_store_pair(insn)
}

/// True for the single-register integer load/store forms `lower::lower_mem`
/// inlines: a general-purpose register (not SIMD/FP), normal (not unprivileged)
/// access, of 1/2/4/8 bytes, with base+immediate addressing (`[Rn, #imm]` /
/// `LDUR`/`STUR`). Other forms (writeback, register-offset, literal, SIMD,
/// `LDTR`/`STTR`) end the block and run in the interpreter.
#[must_use]
pub fn is_inline_load_store(insn: &Insn) -> bool {
    let Insn::LoadStore { vec: false, unpriv: false, size, addr, .. } = insn else {
        return false;
    };
    if *size > 3 {
        return false;
    }
    match addr {
        // Base+immediate, and pre/post-index (writeback) — all inlined.
        AddrMode::UnsignedImm { .. }
        | AddrMode::Unscaled { .. }
        | AddrMode::PreIndex { .. }
        | AddrMode::PostIndex { .. } => true,
        // Register offset: only the standard extends `lower::lower_mem` emits.
        AddrMode::RegOffset { option, .. } => matches!(option, 2 | 3 | 6 | 7),
        // PC-relative literal: rare; left to the interpreter.
        AddrMode::Literal { .. } => false,
    }
}

/// True for the integer `LDP`/`STP`/`LDPSW` pairs `lower::lower_mem_pair`
/// inlines (all index modes: signed-offset, pre-, post-index). SIMD/FP pairs end
/// the block and run in the interpreter.
#[must_use]
pub fn is_inline_load_store_pair(insn: &Insn) -> bool {
    matches!(insn, Insn::LoadStorePair { vec: false, .. })
}

/// A compiled region: a set of basic blocks reachable from `entry` via
/// statically-known direct branches within one page, plus the connections used
/// to wire intra-region jumps at emit time. Discovered before codegen — see
/// [`form_region`]. `blocks[0]` is always the entry block.
pub struct Region {
    /// Entry VA (the hot PC the region was formed at).
    pub entry: u64,
    /// Basic blocks, `blocks[0]` first; each is a [`form_jit_block`] result.
    pub blocks: Vec<Block>,
}

/// The statically-known in-page successor PCs of a block's terminator — the
/// edges region discovery follows. A direct unconditional branch yields its
/// target; a conditional branch yields both its target and its fall-through. A
/// register/indirect branch, or a block that ended at a non-inline instruction,
/// yields its fall-through PC only (where the interpreter will resume) — which is
/// out-of-block, so it's a region exit unless it happens to be another entry.
fn direct_successors(block: &Block, page_base: u64) -> Vec<u64> {
    let Some((pc, term)) = block.insns.last().copied() else { return Vec::new() };
    let in_page = |a: u64| (a & !0xFFF) == page_base;
    let mut out = Vec::new();
    let mut push = |a: u64| {
        if in_page(a) {
            out.push(a);
        }
    };
    match term {
        Insn::BranchImm { link: false, offset } => push(pc.wrapping_add(offset as u64)),
        Insn::CondBranch { offset, .. }
        | Insn::CompareBranch { offset, .. }
        | Insn::TestBranch { offset, .. } => {
            push(pc.wrapping_add(offset as u64)); // taken
            push(pc.wrapping_add(4)); // fall-through
        }
        // BL/BLR/BR/RET (indirect or call) and any non-branch terminator: the
        // only in-region candidate is the fall-through, handled by the caller as
        // the block's end. No statically-followable target.
        _ => {}
    }
    out
}

/// Discover a [`Region`] at `start`: form the entry basic block, then breadth-
/// first follow [`direct_successors`] within `start`'s page, forming each new
/// block, until no new blocks remain or `max_blocks` is reached. Reads code via
/// `read` (same contract as [`form_jit_block`]); blocks are capped at
/// `max_block_len` instructions each. The CFG is fixed here, before emission.
#[must_use]
pub fn form_region(
    start: u64,
    max_blocks: usize,
    max_block_len: usize,
    read: impl Fn(u64) -> u32,
) -> Region {
    let page_base = start & !0xFFF;
    let mut blocks: Vec<Block> = Vec::new();
    let mut seen: Vec<u64> = Vec::new(); // block start PCs already formed
    let mut work: Vec<u64> = vec![start];

    while let Some(pc) = work.pop() {
        if seen.contains(&pc) || blocks.len() >= max_blocks {
            continue;
        }
        // Bound the block to the end of the page so `read` (and any caller's
        // page-sized buffer) is never indexed past it; blocks don't span pages.
        let to_page_end = ((page_base + 0x1000 - pc) / 4) as usize;
        let block = form_jit_block(pc, max_block_len.min(to_page_end), &read);
        if block.insns.is_empty() {
            continue; // `pc` is a non-inline instruction: not a region block
        }
        seen.push(pc);
        for succ in direct_successors(&block, page_base) {
            if !seen.contains(&succ) {
                work.push(succ);
            }
        }
        blocks.push(block);
    }
    Region { entry: start, blocks }
}

/// True for a branch the emitter lowers inline as a block terminator.
#[must_use]
pub fn is_branch(insn: &Insn) -> bool {
    matches!(
        insn,
        Insn::BranchImm { .. }
            | Insn::CondBranch { .. }
            | Insn::CompareBranch { .. }
            | Insn::TestBranch { .. }
            | Insn::BranchReg { .. }
    )
}

/// Form a JIT block at `start`, reading 32-bit code words via `read`: a maximal
/// run of [`can_inline`] register ops, optionally ended by one inline branch.
/// The block is **fully inline-lowerable** — every instruction in it is emitted
/// as native wasm; there is no escape instruction (the organizer interprets the
/// first non-inline instruction itself). An empty block (`insns.is_empty()`)
/// means `start` is a non-inline instruction; the organizer interprets it.
/// Capped at `max_len`.
#[must_use]
pub fn form_jit_block(start: u64, max_len: usize, read: impl Fn(u64) -> u32) -> Block {
    let mut insns = Vec::new();
    let mut pc = start;
    while insns.len() < max_len {
        let insn = decode(read(pc));
        if can_inline(&insn) {
            insns.push((pc, insn));
            pc = pc.wrapping_add(4);
            continue;
        }
        if is_branch(&insn) {
            insns.push((pc, insn)); // inline branch terminator ends the block
        }
        break; // a branch is included; any other non-inline op is excluded
    }
    Block { start, insns }
}
