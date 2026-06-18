//! Instruction eligibility — which instructions the JIT compiles inline.
//!
//! A JIT block is a straight-line run of [`can_inline`] instructions (register
//! ALU ops, plus integer/SIMD loads/stores via an inline TLB-checked fast path),
//! optionally ended by one inline [branch terminator](is_branch). Any other
//! instruction (system ops, exceptions, unhandled forms) ends the block and is
//! interpreted; a memory access that misses its fast path bails to the
//! interpreter too. So the JIT runs the hot path natively and leaves the long
//! tail — and the MMU/MMIO/fault slow paths — to the interpreter.
//!
//! These predicates are kept in lockstep with the always-succeeding lowerings in
//! [`crate::lower`], so block formation ([`crate::region`]) never yields an
//! instruction the emitter can't handle.

use aarch64_decoder::{AddrMode, Insn};

/// True if `insn` is a non-terminator the inline lowerings always handle: a
/// register-only ALU op, or an inlinable memory access ([`is_inline_mem`]).
/// Block formation extends the inline run while this holds.
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

/// True for any integer/SIMD memory access the JIT inlines — a single load/store
/// ([`is_inline_load_store`]) or a load/store pair ([`is_inline_load_store_pair`]).
#[must_use]
pub fn is_inline_mem(insn: &Insn) -> bool {
    is_inline_load_store(insn) || is_inline_load_store_pair(insn)
}

/// True for the single-register load/store forms [`crate::lower::lower_mem`]
/// inlines: normal (not unprivileged) access, 1/2/4/8-byte integer or up to
/// 16-byte vector, with base+immediate, pre/post-index, or standard
/// register-offset addressing. Literal (PC-relative) and `LDTR`/`STTR` are left
/// to the interpreter.
#[must_use]
pub fn is_inline_load_store(insn: &Insn) -> bool {
    let Insn::LoadStore { vec, unpriv: false, size, addr, .. } = insn else {
        return false;
    };
    // Integer widths are log2 0..=3 (1/2/4/8B); vector adds size 4 (the 128-bit Q).
    if *size > if *vec { 4 } else { 3 } {
        return false;
    }
    match addr {
        AddrMode::UnsignedImm { .. }
        | AddrMode::Unscaled { .. }
        | AddrMode::PreIndex { .. }
        | AddrMode::PostIndex { .. } => true,
        // Register offset: only the standard extends `lower_mem` emits.
        AddrMode::RegOffset { option, .. } => matches!(option, 2 | 3 | 6 | 7),
        AddrMode::Literal { .. } => false,
    }
}

/// True for `LDP`/`STP`/`LDPSW` and the SIMD/FP pair forms
/// [`crate::lower::lower_mem_pair`] inlines — integer or vector, all index modes.
#[must_use]
pub fn is_inline_load_store_pair(insn: &Insn) -> bool {
    matches!(insn, Insn::LoadStorePair { .. })
}
