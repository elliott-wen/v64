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
/// Block formation extends the inline run while this holds. The data-processing
/// classes (1/2/3-source) are partially lowered, so they delegate to per-class
/// predicates that admit only the opcodes [`crate::lower`] emits (the rare
/// CRC32 / SMULH / UMULH forms fall back to the interpreter).
#[must_use]
pub fn can_inline(insn: &Insn) -> bool {
    if is_inline_mem(insn) {
        return true;
    }
    match insn {
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
        | Insn::Prfm => true,
        Insn::DataProc1Src { sf, opcode, .. } => is_inline_dp1(*sf, *opcode),
        Insn::DataProc2Src { opcode, .. } => is_inline_dp2(*opcode),
        Insn::DataProc3Src { op31, .. } => is_inline_dp3(*op31),
        // SIMD eligibility is defined by the lowerings themselves (it reuses
        // their decode helpers), so delegate; returns false for non-SIMD.
        _ => crate::lower::is_inline_simd(insn),
    }
}

/// DataProc (1-source) opcodes [`crate::lower::dataproc::data_proc_1src`] emits:
/// RBIT/CLZ/CLS unconditionally, and REV16/REV32/REV where the reversal group
/// fits the operand width (REV64 is X-only).
fn is_inline_dp1(sf: bool, opcode: u8) -> bool {
    match opcode {
        0 | 4 | 5 => true,                                   // RBIT, CLZ, CLS
        1..=3 => (1u32 << opcode) <= if sf { 8 } else { 4 }, // REV group <= datasize/8
        _ => false,
    }
}

/// DataProc (2-source) opcodes [`crate::lower::dataproc::data_proc_2src`] emits:
/// UDIV/SDIV and the variable shifts LSLV/LSRV/ASRV/RORV (CRC32 falls back).
fn is_inline_dp2(opcode: u8) -> bool {
    matches!(opcode, 2 | 3 | 8 | 9 | 10 | 11)
}

/// DataProc (3-source) `op31` values [`crate::lower::dataproc::data_proc_3src`]
/// emits: MADD/MSUB (`000`), the widening S/UMADDL/S/UMSUBL (`001`/`101`), and
/// SMULH/UMULH (`010`/`110`, built from 32-bit half-products).
fn is_inline_dp3(op31: u8) -> bool {
    matches!(op31, 0b000 | 0b001 | 0b010 | 0b101 | 0b110)
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

/// True for any memory access the JIT inlines via the TLB fast path — a single
/// load/store ([`is_inline_load_store`]), a load/store pair
/// ([`is_inline_load_store_pair`]), or an LSE atomic ([`is_inline_atomic`]).
#[must_use]
pub fn is_inline_mem(insn: &Insn) -> bool {
    is_inline_load_store(insn) || is_inline_load_store_pair(insn) || is_inline_atomic(insn)
}

/// True for the LSE atomics [`crate::lower::lower_atomic`] inlines: read-modify-
/// write (`LDADD`/`LDCLR`/`LDEOR`/`LDSET`/`LD{S,U}{MAX,MIN}`/`SWP`) and
/// compare-and-swap, at 1/2/4/8-byte widths. The exclusive-monitor forms
/// (`LDXR`/`STXR`) keep their monitor state in the interpreter.
#[must_use]
pub fn is_inline_atomic(insn: &Insn) -> bool {
    match insn {
        Insn::AtomicRmw { size, op, .. } => *size <= 3 && *op <= 8,
        Insn::CompareSwap { size, .. } => *size <= 3,
        _ => false,
    }
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
