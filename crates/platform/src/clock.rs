//! The timer's time source, behind a trait so it can be swapped per target.
//!
//! Native builds use [`HostClock`] (a monotonic `Instant` scaled to the timer
//! frequency) — matching QEMU's default `QEMU_CLOCK_VIRTUAL` and v86's
//! `performance.now()` model. A browser build supplies its own `Clock` backed by
//! `performance.now()` fed in from JS; the timer's register and interrupt logic
//! never change. A test can supply a manually-advanced clock for determinism.

use std::time::Instant;

/// `virt`'s architected timer frequency (62.5 MHz, 16 ns/tick).
pub const DEFAULT_FREQ_HZ: u64 = 62_500_000;

/// A source of monotonic counter ticks. The value must never decrease.
///
/// The clock only *reads* time; it never blocks. Idle handling (waiting through
/// guest WFI) is the host loop's job — see [`crate::Machine::idle_for`] — so the
/// machine stays portable to single-threaded hosts (a browser can't sleep).
pub trait Clock {
    /// Current counter value, in timer ticks (at the configured frequency).
    fn now(&self) -> u64;
}

/// Monotonic host clock scaled to a fixed tick frequency.
pub struct HostClock {
    start: Instant,
    freq: u64,
}

impl HostClock {
    #[must_use]
    pub fn new(freq: u64) -> Self {
        Self { start: Instant::now(), freq }
    }
}

impl Clock for HostClock {
    fn now(&self) -> u64 {
        let ns = self.start.elapsed().as_nanos();
        (ns * u128::from(self.freq) / 1_000_000_000) as u64
    }
}
