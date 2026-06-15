//! The device-driven execution loop.
//!
//! `aarch64_interp::run` is the pure CPU reference loop with no notion of time
//! or interrupts. [`Machine`] is the system-level driver: it owns the core, the
//! [`Bus`], the [`Gic`], and a [`Clock`]. Each step it advances the generic
//! timer from the clock, asserts the timer's PPI when it fires, and injects an
//! asynchronous IRQ exception when the GIC is asserting a line and `PSTATE.I` is
//! clear.

use aarch64_cpu_state::CpuState;
use aarch64_interp::{
    physical_fires, set_count, set_frequency, step, take_irq, virtual_fires, Step, StopReason,
};

use crate::clock::{Clock, HostClock, DEFAULT_FREQ_HZ};
use crate::{Bus, Gic};

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

/// A single-core machine: CPU + physical bus + interrupt controller + clock.
pub struct Machine {
    pub cpu: CpuState,
    pub bus: Bus,
    pub gic: Gic,
    clock: Box<dyn Clock>,
    /// How often (in instructions) to re-sample the clock; `1` = every step.
    timer_interval: u32,
    /// Counts down within the current sampling window; sample when it reaches 0.
    timer_counter: u32,
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
        Self { cpu, bus, gic, clock, timer_interval: TIMER_SAMPLE_INTERVAL, timer_counter: 0 }
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

    /// Execute one instruction: advance the timer, take a pending IRQ if one is
    /// deliverable, then run the instruction. Taking an IRQ vectors `cpu.pc` to
    /// the handler; the returned [`Step`] reflects the first handler instruction.
    pub fn step(&mut self) -> Step {
        if self.timer_counter == 0 {
            self.advance_timers();
        }
        self.timer_counter = (self.timer_counter + 1) % self.timer_interval;

        if self.irq_deliverable() {
            self.cpu.pc = take_irq(&mut self.cpu);
        }
        step(&mut self.cpu, &mut self.bus)
    }

    /// Run until `cpu.pc == until`, or `count` instructions elapse (`count == 0`
    /// means unbounded). Timers and IRQs are serviced before each instruction.
    /// Mirrors `aarch64_interp::run`'s stop contract.
    pub fn run(&mut self, until: u64, count: usize) -> StopReason {
        let mut executed = 0usize;
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
                return StopReason::Unsupported { pc, word };
            }
            executed += 1;
        }
    }
}
