//! Block formation: decode forward from a start PC until a terminator.

use aarch64_decoder::{decode, Insn};

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

/// Form a block, additionally bounded by the dispatcher's stop conditions:
///
/// - `until`: if the *next* instruction would be at this guest address, end the
///   block first (the dispatcher stops there before executing it, mirroring
///   `run()`'s top-of-loop `until` check).
/// - `max_len`: cap the instruction count (used to honor an instruction `count`
///   budget; also clamped to [`MAX_BLOCK`]).
///
/// The block still ends at the first terminator if one is reached sooner. When
/// it ends on a `max_len`/`until` boundary instead, the last instruction is a
/// non-terminator — the emitter handles that by returning the sequential next PC.
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

/// Whether an instruction ends a block: it may redirect control flow or trap.
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
