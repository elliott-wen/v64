//! The WASM JIT organizer (gated by the `jit` feature; browser/node only).
//!
//! The native interpreter loop ([`Machine::run`](super::Machine::run)) needs no
//! JIT. The browser front-end, where executing through a `WebAssembly` engine
//! beats per-instruction interpretation *if* the per-block overhead is kept
//! down, drives [`run_jit_browser`](super::Machine::run_jit_browser) instead.
//!
//! Following v86: blocks are keyed by **physical address** and emitted
//! **position-independent**, so the same physical code is compiled once and
//! reused under any virtual mapping — a TLB flush / context switch does not
//! invalidate it. Only `IC` (the architectural "code changed" signal) drops the
//! cache. Hotness is tracked per block; cold code is interpreted so only hot
//! loops pay compilation cost.

use aarch64_interp::{step, translate, undefined, Access, GuestMem, Step, StopReason};
use aarch64_jit::{emit_region, form_region};

use super::Machine;

/// 4KB page; a block never spans one (the next page may translate elsewhere).
const PAGE: u64 = 0x1000;

/// Cap on instructions per JIT block.
const MAX_JIT_BLOCK: usize = 256;

/// Cap on basic blocks per compiled region (bounds compiled-function size).
const MAX_REGION_BLOCKS: usize = 16;

/// Executions of a block's entry before it is compiled — below this it's
/// interpreted, so cold code never pays compilation cost; only hot loops do.
///
/// v86 tracks hotness *instruction-weighted* (heat += instructions interpreted
/// on a page) and compiles a whole page at `JIT_THRESHOLD = 200_000`, justified
/// by its expensive page-sized modules. Ours is per-block with much cheaper
/// modules, and our per-block call overhead means over-eager compilation hurts,
/// so we count block-entry executions and use a higher threshold than the
/// bring-up value — compile a block only once it's demonstrably a hot loop.
const JIT_HOTNESS: u32 = 256;

/// Executes the WASM blocks the organizer emits. Implemented by the host that
/// owns a WASM engine — in the browser, JS over the `WebAssembly` API. The
/// Machine emits a block to bytes, hands them to [`compile`](BlockRunner::compile)
/// for a reusable handle, then [`run`](BlockRunner::run)s it against the register
/// image at `regs_base` (a linear-memory address shared with the block).
pub trait BlockRunner {
    /// Compile `wasm` (a module emitted by `aarch64_jit::emit_block`) and return
    /// a handle for [`run`](Self::run).
    fn compile(&mut self, wasm: &[u8]) -> u32;
    /// Run the compiled block `handle`. `regs_base` is the live `CpuState`'s byte
    /// offset in shared memory; `ram_base` is the host byte offset of guest RAM
    /// (for the inline memory fast path). Returns the next guest PC.
    fn run(&mut self, handle: u32, regs_base: u32, ram_base: u32) -> u64;
    /// Drop all compiled blocks (self-modifying code / I-cache maintenance).
    fn invalidate(&mut self);
}

/// A fast hasher for the physical-address `jit_cache` key: the default
/// `SipHash` is overkill for a single `u64` looked up on every region call
/// (tens of millions of times). A Fibonacci multiply avalanches the high bits
/// enough for a power-of-two bucket array.
#[derive(Default)]
pub(super) struct PaHasher(u64);

impl std::hash::Hasher for PaHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        // Only `write_u64` is used for a u64 key; fold defensively otherwise.
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// Physical-address → block-classification cache, keyed for fast lookup.
pub(super) type JitCache =
    std::collections::HashMap<u64, JitBlock, std::hash::BuildHasherDefault<PaHasher>>;

/// How the organizer runs the block at a given physical address.
#[derive(Clone, Copy)]
pub(super) enum JitBlock {
    /// Not yet hot: interpret it, counting executions until [`JIT_HOTNESS`].
    Cold { count: u32 },
    /// Compiled: `handle` runs via the [`BlockRunner`]. The instruction count is
    /// reported by the block (it may loop internally), read from `cpu.jit_count`.
    Hot { handle: u32 },
    /// A lone non-inline instruction (empty block): always interpret.
    Plain,
}

impl Machine {
    /// Guest instructions retired inside hot compiled blocks since boot. Divide
    /// by [`total_insns`](Self::total_insns) for the JIT's coverage fraction.
    #[must_use]
    pub fn jit_insns(&self) -> u64 {
        self.jit_insns
    }

    /// Number of compiled-block invocations since boot. `jit_insns / jit_calls`
    /// is the average instructions retired per block call.
    #[must_use]
    pub fn jit_calls(&self) -> u64 {
        self.jit_calls
    }

    /// Regions compiled and total basic blocks across them; the ratio is the
    /// average region size (how far the dispatch if-chain scans).
    #[must_use]
    pub fn jit_regions(&self) -> u64 {
        self.jit_regions
    }
    #[must_use]
    pub fn jit_region_blocks(&self) -> u64 {
        self.jit_region_blocks
    }

    /// JIT-organized run: same stop contract as [`run`](Self::run), but each
    /// iteration either runs a hot compiled block (via `runner`) or interprets
    /// one instruction. Cold blocks are interpreted while a hotness counter
    /// climbs; once hot, the block is emitted to WASM, compiled by `runner`, and
    /// cached. `IC` (instruction-cache maintenance) drops the cache (SMC).
    pub fn run_jit_browser<R: BlockRunner>(
        &mut self,
        until: u64,
        count: usize,
        runner: &mut R,
    ) -> StopReason {
        let mut executed = 0usize;
        self.idle_until = None;
        loop {
            if self.cpu.powered_off {
                return StopReason::PoweredOff;
            }
            if self.cpu.pc == until {
                return StopReason::UntilReached;
            }
            if count != 0 && executed >= count {
                return StopReason::CountReached;
            }

            self.service_async(); // timers/IRQ once per unit; may vector cpu.pc

            let jitted = self.jit_step(runner);
            if let Some(len) = jitted {
                executed += len;
                self.total_insns += len as u64;
                self.jit_insns += len as u64;
            }
            // Interpret one instruction when the JIT can't make progress itself:
            // the block wasn't hot (`None`), or it ran but bailed at a memory op
            // its inline fast path couldn't handle (`jit_exit`). Either way
            // `cpu.pc` now points at the instruction to interpret.
            if jitted.is_none() || self.cpu.jit_exit != 0 {
                self.cpu.jit_exit = 0;
                if let Step::Unsupported { pc, word } = step(&mut self.cpu, &mut self.bus) {
                    self.undefined_seen.entry(word).or_insert(pc);
                    if self.undef_to_guest {
                        self.cpu.pc = undefined(&mut self.cpu, pc);
                    } else {
                        return StopReason::Unsupported { pc, word };
                    }
                }
                executed += 1;
                self.total_insns += 1;
            }

            // Drop compiled blocks on `IC` — the architectural signal that guest
            // code changed (self-modifying code), so the cached bytes are stale.
            // Blocks are physical-address-keyed and position-independent, so a TLB
            // flush / context switch does *not* invalidate them (the v86 model):
            // the same physical code is re-found by PA and runs correctly at the
            // new VA. Only the bytes changing matters, and that's what `IC` flags.
            if self.cpu.ic_inval {
                self.cpu.ic_inval = false;
                self.cpu.tlb_flushed = false;
                self.jit_cache.clear();
                runner.invalidate();
            }
            // A TLB flush alone changes mappings, not code — keep the cache.
            self.cpu.tlb_flushed = false;

            if self.cpu.wfi {
                self.cpu.wfi = false;
                if !self.irq_deliverable() && self.note_idle() {
                    return StopReason::CountReached;
                }
            }
        }
    }

    /// If the block at `cpu.pc` is hot, run it via `runner` and return the
    /// instruction count; otherwise bump its hotness (compiling once hot) and
    /// return `None` so the caller interprets one instruction.
    fn jit_step<R: BlockRunner>(&mut self, runner: &mut R) -> Option<usize> {
        let pc = self.cpu.pc;
        let el = self.cpu.el;
        // Translate to the physical address: that's both the cache key and where
        // we read the code. Keying by PA (not VA) means the same physical code is
        // compiled once and reused under any mapping; the emitted block is
        // position-independent (it derives PCs from the runtime entry PC), so it
        // runs correctly whatever VA it's entered at. This is the v86 model.
        let pa = translate(&mut self.cpu, &mut self.bus, pc, Access::Exec, el).ok()?;

        let handle = match self.jit_cache.get(&pa).copied() {
            Some(JitBlock::Hot { handle }) => handle,
            Some(JitBlock::Plain) => return None,
            Some(JitBlock::Cold { count }) => {
                if count + 1 < JIT_HOTNESS {
                    self.jit_cache.insert(pa, JitBlock::Cold { count: count + 1 });
                    return None;
                }
                // Hot: discover the region (one or more basic blocks connected by
                // in-page direct branches), emit it, and compile via the runner.
                let region = self.form_jit_region(pc, pa);
                if region.blocks.is_empty() {
                    self.jit_cache.insert(pa, JitBlock::Plain); // a lone non-inline op
                    return None;
                }
                self.jit_regions += 1;
                self.jit_region_blocks += region.blocks.len() as u64;
                let (_, ram_phys, ram_size) = self.bus.ram_jit_info();
                // emit_region picks the shape: a lean one-shot body for a single
                // straight-line block, the dispatch loop for self-loops / multi-block.
                let handle = runner.compile(&emit_region(&region, ram_phys, ram_size));
                self.jit_cache.insert(pa, JitBlock::Hot { handle });
                handle
            }
            None => {
                self.jit_cache.insert(pa, JitBlock::Cold { count: 1 });
                return None;
            }
        };

        // Run the block directly against the live CpuState — no image copy. The
        // block shares this module's linear memory (the host wired `env.memory`
        // to it), and `CpuState` is `#[repr(C)]` with X/SP/PC/NZCV at the offsets
        // the block uses, so its address *is* the register base: the block reads
        // and writes the real X0–X30 / SP in place. Only the packed NZCV word is
        // bridged to/from the interpreter's `flags` (X/SP need no sync at all),
        // and PC comes back as the block's return value.
        //
        // The `as u32` truncation is correct because the JIT runs only under
        // wasm32 (pointers are 32-bit linear-memory offsets); this code is never
        // exercised on a 64-bit host (the `jit` feature is browser/node only).
        self.cpu.nzcv = self.cpu.flags.to_nzcv();
        self.cpu.jit_exit = 0; // the block sets this to 1 only if it bails mid-way
        let regs_base = std::ptr::from_ref(&self.cpu) as u32;
        let ram_base = self.bus.ram_jit_info().0;
        let next = runner.run(handle, regs_base, ram_base);
        self.cpu.flags = aarch64_cpu_state::Flags::from_nzcv(self.cpu.nzcv);
        self.cpu.pc = next;
        self.jit_calls += 1;
        // The block reports how many instructions it retired (a self-loop runs
        // many iterations per call), written to `jit_count` just before it
        // returned. That, not the static block length, is the count to bill.
        Some(self.cpu.jit_count as usize)
    }

    /// Discover a JIT region at `pc` (physical `pa`). Reads the whole containing
    /// page once — a region's blocks may precede the entry (a backward-branch
    /// target) — and forms blocks via direct in-page branches, capped at
    /// [`MAX_REGION_BLOCKS`] blocks of [`MAX_JIT_BLOCK`] instructions each.
    fn form_jit_region(&mut self, pc: u64, pa: u64) -> aarch64_jit::Region {
        let page_va = pc & !(PAGE - 1);
        let page_pa = pa & !(PAGE - 1);
        let words: Vec<u32> = (0..PAGE / 4).map(|i| self.bus.read_u32(page_pa + i * 4)).collect();
        form_region(pc, MAX_REGION_BLOCKS, MAX_JIT_BLOCK, |a| {
            words[((a - page_va) / 4) as usize]
        })
    }
}
