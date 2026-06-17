//! Block discovery: decode forward from a start PC until a terminator.
//!
//! A *block* is a straight-line run of guest instructions ending at the first
//! control-flow/trap boundary. It is the unit the platform execution loop
//! organizes around: the interpreter executes a block by stepping it
//! instruction-by-instruction, and the JIT compiles eligible blocks. Discovery
//! lives here (the decode layer) so every consumer agrees on block boundaries.

use crate::{decode, Insn};

/// A straight-line run of guest instructions ending in a terminator.
pub struct Block {
    pub start: u64,
    /// `(guest_pc, decoded)` in program order; the last entry is the terminator.
    pub insns: Vec<(u64, Insn)>,
}

/// Cap on instructions per block (runaway / huge-function guard).
const MAX_BLOCK: usize = 1024;

/// Form a block starting at `start`, reading 32-bit code words via `read`.
pub fn form_block(start: u64, read: impl Fn(u64) -> u32) -> Block {
    form_block_bounded(start, None, MAX_BLOCK, read)
}

/// Form a block, additionally bounded by a stop address and a length cap:
///
/// - `until`: if the *next* instruction would be at this guest address, end the
///   block first (so the caller can stop there before executing it).
/// - `max_len`: cap the instruction count (also clamped to [`MAX_BLOCK`]).
///
/// The block still ends at the first terminator if one is reached sooner.
pub fn form_block_bounded(
    start: u64,
    until: Option<u64>,
    max_len: usize,
    read: impl Fn(u64) -> u32,
) -> Block {
    let cap = max_len.clamp(1, MAX_BLOCK);
    let mut insns = Vec::new();
    let mut pc = start;
    while insns.len() < cap {
        let insn = decode(read(pc));
        let done = is_terminator(&insn);
        insns.push((pc, insn));
        if done {
            break;
        }
        let next = pc.wrapping_add(4);
        if Some(next) == until {
            break;
        }
        pc = next;
    }
    Block { start, insns }
}

/// Whether an instruction ends a block: it redirects control flow or traps, so
/// it must be the last instruction in a straight-line run.
pub fn is_terminator(insn: &Insn) -> bool {
    matches!(
        insn,
        Insn::BranchImm { .. }
            | Insn::CondBranch { .. }
            | Insn::CompareBranch { .. }
            | Insn::TestBranch { .. }
            | Insn::BranchReg { .. }
            | Insn::Svc { .. }
            | Insn::Eret
            | Insn::Unsupported { .. }
    )
}
