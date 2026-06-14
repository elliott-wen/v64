//! The fetch-decode-execute loop.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::{decode, Insn};

use crate::execute::execute;
use crate::memory::Memory;

/// Why execution stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Reached the requested `until` address.
    UntilReached,
    /// Executed the requested instruction count.
    CountReached,
    /// Hit an instruction the interpreter does not implement yet.
    Unsupported { pc: u64, word: u32 },
}

/// Run from `cpu.pc` until reaching `until`, or after `count` instructions
/// (`count == 0` means unbounded until `until`). Mirrors Unicorn's
/// `emu_start(begin, until, _timeout, count)` contract so the two can be
/// compared directly in the differential harness.
pub fn run(cpu: &mut CpuState, mem: &mut Memory, until: u64, count: usize) -> StopReason {
    let mut executed = 0usize;
    loop {
        if cpu.pc == until {
            return StopReason::UntilReached;
        }
        if count != 0 && executed >= count {
            return StopReason::CountReached;
        }

        let pc = cpu.pc;
        let word = mem.read_u32(crate::mmu::translate(cpu, mem, pc));
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
