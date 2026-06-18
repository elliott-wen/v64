//! JIT block and region discovery — building the control-flow graph before
//! codegen.
//!
//! [`form_jit_block`] forms one straight-line block (a run of inline-lowerable
//! instructions, optionally ended by a branch). [`form_region`] discovers a whole
//! [`Region`] — the blocks reachable from an entry via statically-known direct
//! branches within one page — which the emitter then turns into a single
//! dispatch-loop function. Discovery is a pure read over guest code (via a `read`
//! closure); it performs no codegen and has no side effects.

use aarch64_decoder::{decode, Block, Insn};

use crate::eligible::{can_inline, is_branch};

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

/// Form a JIT block at `start`, reading 32-bit code words via `read`: a maximal
/// run of [`can_inline`] instructions, optionally ended by one inline branch
/// terminator. Every instruction in the block is emittable as native wasm; a
/// non-inline, non-branch instruction ends the block without being included (the
/// organizer interprets it). An empty block (`insns.is_empty()`) means `start`
/// itself is non-inline. Capped at `max_len` instructions.
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

/// Discover a [`Region`] at `start`: form the entry block, then breadth-first
/// follow each block's [`direct_successors`] within `start`'s page, forming new
/// blocks until none remain or `max_blocks` is reached. Reads code via `read`
/// (same contract as [`form_jit_block`]); blocks are capped at `max_block_len`
/// instructions. The CFG is fixed here, before emission.
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

/// The statically-known in-page successor PCs of a block's terminator — the edges
/// region discovery follows. A direct unconditional branch yields its target; a
/// conditional branch yields its target and its fall-through. A register/indirect
/// branch, or a block ending at a non-inline instruction, yields nothing (the
/// fall-through is the block's end, which the caller handles).
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
        // BL/BLR/BR/RET (call or indirect) and any non-branch terminator: no
        // statically-followable in-region target.
        _ => {}
    }
    out
}
