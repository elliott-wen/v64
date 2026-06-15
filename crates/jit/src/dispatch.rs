//! The block-level dispatcher (Milestone 5): a run loop over compiled blocks,
//! alongside (never replacing) the interpreter's `run()`.
//!
//! The loop mirrors `interp::run`'s contract — stop at `until`, or after `count`
//! instructions — but at block granularity. Policy (block cache, hotness, SMC
//! invalidation) lives here; the mechanics it drives (compile / instantiate /
//! call, image access) are [`Vm`] methods, so this loop ports to an all-wasm/JS
//! host with the cache logic intact.

use std::collections::HashMap;

use aarch64_interp::StopReason;

use crate::abi;
use crate::block::form_block_bounded;
use crate::runtime::Vm;

/// A compiled block plus the source bytes it was compiled from, so a coarse
/// self-modifying-code check can recompile if the guest rewrites that range.
struct Cached {
    instance: wasmtime::Instance,
    code: Vec<u8>,
}

/// Safety net against a lowering bug spinning forever; the fuzz corpus and real
/// code reach `until`/`count` well before this.
const MAX_BLOCKS: u64 = 4_000_000;

impl Vm {
    /// Run from the image PC until reaching `until`, or after `count` guest
    /// instructions (`count == 0` = unbounded until `until`). Mirrors
    /// `interp::run`'s stop semantics so a JIT run is comparable to an interpreter
    /// run on the same vector.
    pub fn run(&mut self, until: u64, count: usize) -> StopReason {
        let mut cache: HashMap<u64, Cached> = HashMap::new();
        let mut executed = 0usize;

        for _ in 0..MAX_BLOCKS {
            let pc = self.image_pc();
            if pc == until {
                return StopReason::UntilReached;
            }
            if count != 0 && executed >= count {
                return StopReason::CountReached;
            }

            // Form a block bounded by the remaining instruction budget and the
            // `until` boundary, then read back its raw bytes for the SMC check.
            let max_len = if count == 0 { usize::MAX } else { count - executed };
            let block = form_block_bounded(pc, Some(until), max_len, |a| self.read_code_word(a));
            let code: Vec<u8> =
                block.insns.iter().flat_map(|(p, _)| self.read_code_word(*p).to_le_bytes()).collect();

            // Cache by start PC; recompile if the underlying bytes changed (SMC).
            let instance = match cache.get(&pc) {
                Some(c) if c.code == code => c.instance,
                _ => {
                    let instance = self.compile_instance(&block);
                    cache.insert(pc, Cached { instance, code });
                    instance
                }
            };

            let exit = self.call_instance(instance);
            match exit.exit_reason {
                abi::EXIT_NONE => executed += block.insns.len(),
                abi::EXIT_UNSUPPORTED => {
                    let word = self.read_code_word(exit.next_pc);
                    return StopReason::Unsupported { pc: exit.next_pc, word };
                }
                other => panic!("JIT block faulted (exit {other:#x}) at pc {:#x}", exit.next_pc),
            }
        }
        panic!("JIT dispatch exceeded {MAX_BLOCKS} blocks without reaching until/count")
    }
}
