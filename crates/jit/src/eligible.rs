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

use aarch64_decoder::{decode, is_terminator, Block, Insn};

/// True if `insn` is a non-terminator the inline lowerings always handle,
/// touching only the hot register image (no memory, system, or cold state).
/// Block formation extends the inline run while this holds; the first
/// instruction for which it's false (or any terminator) ends the block and is
/// executed via the interpreter escape hatch.
#[must_use]
pub fn can_inline(insn: &Insn) -> bool {
    matches!(
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

/// Form a JIT block at `start`, reading 32-bit code words via `read`: a leading
/// run of [`can_inline`] register ops, ended by the first terminator or
/// non-inline ("escape") instruction (which becomes the block's last,
/// interpreter-executed, instruction). Capped at `max_len`. This is the shared
/// definition the emitter relies on — every non-last instruction is inline-
/// lowerable.
#[must_use]
pub fn form_jit_block(start: u64, max_len: usize, read: impl Fn(u64) -> u32) -> Block {
    let mut insns = Vec::new();
    let mut pc = start;
    while insns.len() < max_len {
        let insn = decode(read(pc));
        let stop = is_terminator(&insn) || !can_inline(&insn);
        insns.push((pc, insn));
        if stop {
            break;
        }
        pc = pc.wrapping_add(4);
    }
    Block { start, insns }
}
