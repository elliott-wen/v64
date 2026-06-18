//! The device-driven execution loop.
//!
//! `aarch64_interp::run` is the pure CPU reference loop with no notion of time
//! or interrupts. [`Machine`] is the system-level driver: it owns the core, the
//! [`Bus`], the [`Gic`], and a [`Clock`]. Each step it advances the generic
//! timer from the clock, asserts the timer's PPI when it fires, and injects an
//! asynchronous IRQ exception when the GIC is asserting a line and `PSTATE.I` is
//! clear.
//!
//! The optional WASM JIT organizer (gated by the `jit` feature, used by the
//! browser/node front-end) lives in the [`jit`] submodule; native builds omit it.

use std::collections::BTreeMap;

use aarch64_cpu_state::CpuState;
use aarch64_interp::{
    next_deadline, physical_fires, set_count, set_frequency, step, take_irq, undefined,
    virtual_fires, Step, StopReason,
};

use crate::clock::{Clock, HostClock, DEFAULT_FREQ_HZ};
use crate::{Bus, DmaDevice, Gic};

#[cfg(feature = "jit")]
mod jit;
#[cfg(feature = "jit")]
pub use jit::BlockRunner;

/// PSTATE.I (IRQ mask) within the packed DAIF nibble `[D,A,I,F]`.
const PSTATE_I: u8 = 0b0010;

/// `virt` generic-timer PPIs: virtual timer = IRQ 27, physical timer = IRQ 30.
const PPI_VIRT_TIMER: u32 = 27;
const PPI_PHYS_TIMER: u32 = 30;

/// Re-sample the host clock (and re-evaluate the comparator) once per this many
/// instructions, rather than every step. Reading the clock and updating the
/// counter per instruction is pure overhead — the scheduler tick is millions of
/// ticks away, so coarse sampling is invisible to the guest while removing a
/// `clock_gettime` and sysreg-map churn from the hot path.
const TIMER_SAMPLE_INTERVAL: u32 = 64;

/// Upper bound on a single idle sleep, expressed as a rate: the machine never
/// sleeps longer than `freq / MAX_IDLE_HZ` ticks (here ~20 ms) before returning
/// to the host loop, so console input and the quit key are serviced promptly
/// even when the next timer deadline is far away (or no timer is armed).
const MAX_IDLE_HZ: u64 = 50;

/// A single-core machine: CPU + physical bus + interrupt controller + clock.
pub struct Machine {
    pub cpu: CpuState,
    pub bus: Bus,
    pub gic: Gic,
    clock: Box<dyn Clock>,
    /// Timer tick frequency (Hz), used to bound how long an idle WFI sleeps.
    freq: u64,
    /// How often (in instructions) to re-sample the clock; `1` = every step.
    timer_interval: u32,
    /// Counts down within the current sampling window; sample when it reaches 0.
    timer_counter: u32,
    /// When true (default), an instruction the interpreter doesn't implement is
    /// delivered to the guest as an Undefined Instruction exception — like real
    /// hardware (the kernel raises SIGILL, or panics in kernel context) — instead
    /// of stopping the machine. When false, `run` stops with `Unsupported`
    /// (useful for bring-up). Each distinct undefined word is recorded either way.
    undef_to_guest: bool,
    /// Distinct undefined instruction words seen -> an example PC, for reporting.
    undefined_seen: BTreeMap<u32, u64>,
    /// DMA-capable devices (virtio) polled on the timer-sampling cadence with
    /// full guest-memory access, so they can drain their virtqueues.
    dma: Vec<Box<dyn DmaDevice>>,
    /// When the last `run` returned because the guest went idle (WFI/WFE with no
    /// pending interrupt), the counter tick the machine should resume at. The
    /// host decides *how* to wait until then — a native binary sleeps, a browser
    /// driver schedules a `setTimeout` — so the Machine itself never blocks and
    /// stays usable on a single-threaded (WASM) host. `None` when not idle.
    idle_until: Option<u64>,
    /// Total guest instructions retired across the machine's lifetime (a
    /// monotonically increasing counter). Not architectural — a host-side stat
    /// for throughput reporting (instructions/sec). Idle WFI sleeps don't count.
    total_insns: u64,
    /// Of [`total_insns`](Self::total_insns), how many were retired inside a hot
    /// compiled block (vs. interpreted). A host-side stat for JIT-coverage
    /// reporting; the ratio tells you how much of the workload the JIT captured.
    #[cfg(feature = "jit")]
    jit_insns: u64,
    /// Number of hot-block invocations (for average-block-length reporting).
    #[cfg(feature = "jit")]
    jit_calls: u64,
    /// Regions compiled, and the total basic blocks across them — their ratio is
    /// the average region size (how much the if-chain dispatch scans).
    #[cfg(feature = "jit")]
    jit_regions: u64,
    #[cfg(feature = "jit")]
    jit_region_blocks: u64,
    /// JIT organizer state: per-physical-address block classification
    /// (cold/hot/plain). See [`jit`]. Compiled blocks read/write the live
    /// `CpuState` directly (it's `#[repr(C)]` in shared linear memory), so there
    /// is no separate register image to hold here.
    #[cfg(feature = "jit")]
    jit_cache: jit::JitCache,
}

impl Machine {
    /// Build a machine with the default host clock at the `virt` timer frequency.
    #[must_use]
    pub fn new(cpu: CpuState, bus: Bus, gic: Gic) -> Self {
        Self::with_clock(cpu, bus, gic, Box::new(HostClock::new(DEFAULT_FREQ_HZ)), DEFAULT_FREQ_HZ)
    }

    /// Build a machine with an explicit clock (e.g. a deterministic test clock or
    /// a browser `performance.now()` source). `freq` is published as `CNTFRQ_EL0`
    /// and must match the clock's tick rate.
    #[must_use]
    pub fn with_clock(mut cpu: CpuState, bus: Bus, gic: Gic, clock: Box<dyn Clock>, freq: u64) -> Self {
        set_frequency(&mut cpu, freq);
        Self {
            cpu,
            bus,
            gic,
            clock,
            freq,
            timer_interval: TIMER_SAMPLE_INTERVAL,
            timer_counter: 0,
            undef_to_guest: true,
            undefined_seen: BTreeMap::new(),
            dma: Vec::new(),
            idle_until: None,
            total_insns: 0,
            #[cfg(feature = "jit")]
            jit_insns: 0,
            #[cfg(feature = "jit")]
            jit_calls: 0,
            #[cfg(feature = "jit")]
            jit_regions: 0,
            #[cfg(feature = "jit")]
            jit_region_blocks: 0,
            #[cfg(feature = "jit")]
            jit_cache: jit::JitCache::default(),
        }
    }

    /// Total guest instructions retired since boot — a host-side throughput
    /// counter (not architectural). A front-end can sample this against wall
    /// time to report instructions/sec. Idle WFI sleeps are not counted.
    #[must_use]
    pub fn total_insns(&self) -> u64 {
        self.total_insns
    }

    /// Register a DMA-capable device (e.g. virtio-blk) to be polled with
    /// guest-memory access on the timer-sampling cadence.
    pub fn add_dma(&mut self, dev: Box<dyn DmaDevice>) {
        self.dma.push(dev);
    }

    /// Choose how an unimplemented instruction is handled: deliver an Undefined
    /// Instruction exception to the guest (`true`, default, faithful) or stop the
    /// machine with `StopReason::Unsupported` (`false`, for bring-up).
    pub fn set_undef_to_guest(&mut self, deliver: bool) {
        self.undef_to_guest = deliver;
    }

    /// Distinct undefined instruction words encountered so far, each mapped to an
    /// example PC where it occurred.
    #[must_use]
    pub fn undefined_seen(&self) -> &BTreeMap<u32, u64> {
        &self.undefined_seen
    }

    /// Override the clock-sampling interval (instructions between samples). `1`
    /// samples every step — used by tests that need deterministic timing.
    pub fn set_timer_interval(&mut self, instructions: u32) {
        self.timer_interval = instructions.max(1);
        self.timer_counter = 0;
    }

    /// Sample the clock into the counter and (de)assert each timer's PPI line.
    /// Level-sensitive: a timer whose condition no longer holds clears its line.
    fn advance_timers(&mut self) {
        set_count(&mut self.cpu, self.clock.now());
        for (fires, ppi) in [
            (virtual_fires(&self.cpu), PPI_VIRT_TIMER),
            (physical_fires(&self.cpu), PPI_PHYS_TIMER),
        ] {
            if fires {
                self.gic.set_pending(ppi);
            } else {
                self.gic.clear_pending(ppi);
            }
        }
    }

    /// True when an IRQ is deliverable: the GIC is asserting and IRQs are
    /// unmasked at the core.
    fn irq_deliverable(&self) -> bool {
        self.cpu.daif & PSTATE_I == 0 && self.gic.pending_irq()
    }

    /// Note guest idle (after a WFI/WFE with no pending interrupt): record the
    /// counter tick to resume at — the next enabled-timer deadline, bounded by
    /// [`MAX_IDLE_HZ`] so injected console input and the quit key stay
    /// responsive. Returns `true` if the machine is now idle (the caller should
    /// stop and wait until [`Self::idle_for`]). Returns `false` if an interrupt
    /// is already pending, so the caller keeps running and takes it immediately.
    ///
    /// Crucially this does *not* block: the host loop decides how to wait (a
    /// native binary sleeps; a browser driver uses `setTimeout`), keeping the
    /// Machine portable to a single-threaded WASM environment.
    fn note_idle(&mut self) -> bool {
        self.advance_timers();
        if self.gic.pending_irq() {
            return false; // already due — wake immediately
        }
        let now = self.clock.now();
        let cap = now + self.freq / MAX_IDLE_HZ;
        let target = match next_deadline(&self.cpu) {
            Some(d) if d > now => d.min(cap),
            Some(_) => return false, // a timer is already past its deadline
            None => cap,             // nothing armed: wait a slice and re-poll
        };
        self.idle_until = Some(target);
        self.timer_counter = 0; // re-sample the clock on the first step after waking
        true
    }

    /// How long the host should wait before re-entering [`Self::run`], when the
    /// last run stopped on guest idle. `None` if the machine isn't idle. The
    /// duration is derived from the recorded deadline and the timer frequency, so
    /// it is host-agnostic: a native loop passes it to `thread::sleep`, a browser
    /// loop converts it to milliseconds for `setTimeout`.
    /// The counter tick the machine will resume at after guest idle, if the last
    /// run stopped on WFI/WFE. `None` otherwise. A tick-based host driver can use
    /// this directly; most native callers want [`Self::idle_for`].
    #[must_use]
    pub fn idle_until_tick(&self) -> Option<u64> {
        self.idle_until
    }

    #[must_use]
    pub fn idle_for(&self) -> Option<std::time::Duration> {
        let target = self.idle_until?;
        let now = self.clock.now();
        let ticks = target.saturating_sub(now);
        let nanos = u128::from(ticks) * 1_000_000_000 / u128::from(self.freq.max(1));
        Some(std::time::Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64))
    }

    /// Service asynchronous events due before the next execution unit: sample the
    /// timer (and poll DMA) on the sampling cadence, then take a pending IRQ if
    /// one is deliverable (which vectors `cpu.pc` to the handler).
    fn service_async(&mut self) {
        if self.timer_counter == 0 {
            self.advance_timers();
            // Service DMA devices (virtio) with full guest-memory access. Disjoint
            // field borrows (`dma` shared, `bus` mutable) — cheap when idle.
            for d in &self.dma {
                d.poll(&mut self.bus);
            }
        }
        self.timer_counter = (self.timer_counter + 1) % self.timer_interval;

        if self.irq_deliverable() {
            self.cpu.pc = take_irq(&mut self.cpu);
        }
    }

    /// Execute one instruction: advance the timer, take a pending IRQ if one is
    /// deliverable, then run the instruction. Taking an IRQ vectors `cpu.pc` to
    /// the handler; the returned [`Step`] reflects the first handler instruction.
    pub fn step(&mut self) -> Step {
        self.service_async();
        step(&mut self.cpu, &mut self.bus)
    }

    /// Run until `cpu.pc == until`, or `count` instructions elapse (`count == 0`
    /// means unbounded). Timers and IRQs are serviced before each instruction.
    /// Mirrors `aarch64_interp::run`'s stop contract.
    pub fn run(&mut self, until: u64, count: usize) -> StopReason {
        let mut executed = 0usize;
        self.idle_until = None; // cleared unless this run stops on guest idle
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
            if let Step::Unsupported { pc, word } = self.step() {
                self.undefined_seen.entry(word).or_insert(pc);
                if self.undef_to_guest {
                    // Faithful: raise an Undefined Instruction exception to the
                    // guest (SIGILL in userspace / panic in kernel), keep running.
                    self.cpu.pc = undefined(&mut self.cpu, pc);
                } else {
                    return StopReason::Unsupported { pc, word };
                }
            }
            executed += 1;
            self.total_insns += 1;

            // Guest idle: the instruction just retired was WFI/WFE. If no IRQ is
            // already deliverable, hand control back to the host loop with an idle
            // deadline (see `idle_for`) instead of spinning the kernel's idle loop
            // at full host speed. The host waits, then re-enters `run`.
            if self.cpu.wfi {
                self.cpu.wfi = false;
                if !self.irq_deliverable() && self.note_idle() {
                    return StopReason::CountReached;
                }
            }
        }
    }
}
