//! The fetch-decode-execute loop.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::{decode, Insn};

use crate::execute::execute;
use crate::memory::GuestMem;

/// Why execution stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Reached the requested `until` address.
    UntilReached,
    /// Executed the requested instruction count.
    CountReached,
    /// Hit an instruction the interpreter does not implement yet.
    Unsupported { pc: u64, word: u32 },
    /// The guest requested power-off/reset via PSCI (`CpuState::powered_off`).
    PoweredOff,
}

/// Run from `cpu.pc` until reaching `until`, or after `count` instructions
/// (`count == 0` means unbounded until `until`). Mirrors Unicorn's
/// `emu_start(begin, until, _timeout, count)` contract so the two can be
/// compared directly in the differential harness.
pub fn run(cpu: &mut CpuState, mem: &mut dyn GuestMem, until: u64, count: usize) -> StopReason {
    let mut executed = 0usize;
    loop {
        if cpu.pc == until {
            return StopReason::UntilReached;
        }
        if count != 0 && executed >= count {
            return StopReason::CountReached;
        }

        let pc = cpu.pc;
        let fetch_pa = crate::mmu::translate(cpu, mem, pc);
        let word = mem.read_u32(fetch_pa);
        let insn = decode(word);
        if let Insn::Unsupported { word } = insn {
            return StopReason::Unsupported { pc, word };
        }
        // A taken branch returns its target; otherwise advance sequentially.
        cpu.pc = match execute(cpu, mem, insn, pc) {
            Some(target) => target,
            None => pc.wrapping_add(4),
        };
        executed += 1;
    }
}

/// Outcome of a single interpreted instruction step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Executed one instruction; `cpu.pc` now holds the next PC (also returned).
    Next(u64),
    /// Hit an unimplemented instruction. `cpu.pc` is left at the faulting PC and
    /// no architectural state was changed.
    Unsupported { pc: u64, word: u32 },
}

/// Execute exactly one instruction at `cpu.pc`, updating `cpu`/`mem` and
/// advancing `cpu.pc`.
///
/// This is the single-step primitive the JIT's `interpret_one` escape hatch is
/// built on. It mirrors one iteration of [`run`]'s loop body; `run` is left
/// deliberately independent of it so it stays the untouched reference loop.
pub fn step(cpu: &mut CpuState, mem: &mut dyn GuestMem) -> Step {
    let pc = cpu.pc;
    let fetch_pa = crate::mmu::translate(cpu, mem, pc);
    let word = mem.read_u32(fetch_pa);
    let insn = decode(word);
    if let Insn::Unsupported { word } = insn {
        return Step::Unsupported { pc, word };
    }
    cpu.pc = match execute(cpu, mem, insn, pc) {
        Some(target) => target,
        None => pc.wrapping_add(4),
    };
    Step::Next(cpu.pc)
}
