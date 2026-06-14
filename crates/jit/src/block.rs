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
    let mut insns = Vec::new();
    let mut pc = start;
    loop {
        let insn = decode(read(pc));
        let done = is_terminator(&insn);
        insns.push((pc, insn));
        if done || insns.len() >= MAX_BLOCK {
            break;
        }
        pc = pc.wrapping_add(4);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_at_branch() {
        // NOP; NOP; B . (self-branch). Read returns each word by index.
        let code = [0xd503201fu32, 0xd503201f, 0x1400_0000];
        let block = form_block(0x1000, |pc| code[((pc - 0x1000) / 4) as usize]);
        assert_eq!(block.insns.len(), 3);
        assert!(matches!(block.insns[2].1, Insn::BranchImm { .. }));
    }
}
