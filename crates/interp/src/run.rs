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
        // Fetch translation can fault (Instruction Abort) — e.g. demand-paged
        // user text. Vector to the handler and retry the fetch after ERET.
        let fetch_pa = match crate::mmu::translate(cpu, mem, pc, crate::mmu::Access::Exec, cpu.el) {
            Ok(pa) => pa,
            Err(fsc) => {
                cpu.pc = crate::exception::inst_abort(cpu, pc, fsc);
                executed += 1;
                continue;
            }
        };
        let word = mem.read_u32(fetch_pa);
        let insn = decode(word);
        if let Insn::Unsupported { word } = insn {
            return StopReason::Unsupported { pc, word };
        }
        let next = execute(cpu, mem, insn, pc);
        // A data access may have raised a translation fault mid-instruction;
        // take the Data Abort instead of advancing (the instruction retries).
        cpu.pc = if let Some(abort) = cpu.pending_abort.take() {
            crate::exception::data_abort(cpu, pc, abort)
        } else {
            // A taken branch returns its target; otherwise advance sequentially.
            next.unwrap_or_else(|| pc.wrapping_add(4))
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
    let fetch_pa = match crate::mmu::translate(cpu, mem, pc, crate::mmu::Access::Exec, cpu.el) {
        Ok(pa) => pa,
        Err(fsc) => {
            cpu.pc = crate::exception::inst_abort(cpu, pc, fsc);
            return Step::Next(cpu.pc);
        }
    };
    let word = mem.read_u32(fetch_pa);
    let insn = decode(word);
    if let Insn::Unsupported { word } = insn {
        return Step::Unsupported { pc, word };
    }
    let next = execute(cpu, mem, insn, pc);
    cpu.pc = if let Some(abort) = cpu.pending_abort.take() {
        crate::exception::data_abort(cpu, pc, abort)
    } else {
        next.unwrap_or_else(|| pc.wrapping_add(4))
    };
    Step::Next(cpu.pc)
}
