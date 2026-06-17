//! Randomized differential sweep: every instruction class vs Unicorn.
//!
//! Built only with `--features unicorn`. This is a single `#[test]` that drives
//! one work pool at *iteration* granularity: every class's iterations are split
//! into chunks and distributed across `FUZZ_THREADS` workers, so a slow class no
//! longer bounds wall-time (the way it did when each class was its own `#[test]`
//! and parallelism was only between classes). Any state divergence fails with a
//! reproducible `(class, chunk-seed, iter, word)` description.
//!
//! ```text
//! cargo test -p aarch64-difftest --features unicorn --test fuzz_sweep
//! FUZZ_ITERS=500000 cargo test ... --test fuzz_sweep -- --nocapture
//! FUZZ_CLASS=neon_aes,ldst_pair cargo test ... --test fuzz_sweep   # subset
//! FUZZ_THREADS=8 cargo test ... --test fuzz_sweep                  # cap cores
//! ```

#![cfg(feature = "unicorn")]

use aarch64_difftest::{all_jobs, fuzz_jobs};

/// Iterations per class. Override with `FUZZ_ITERS=<n>` to fuzz harder.
fn iters() -> u32 {
    std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000)
}

/// Worker count. Defaults to the machine's parallelism; override to cap cores.
fn threads() -> usize {
    std::env::var("FUZZ_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
}

/// Per-class fixed seed -> fully reproducible.
fn seed_for(name: &str) -> u64 {
    0x5eed_0000 ^ (u64::from(name.len() as u32) << 8)
}

#[test]
fn sweep() {
    let mut jobs = all_jobs();

    // `FUZZ_CLASS=a,b,c` restricts the sweep to named classes (replaces the old
    // per-class `cargo test <name>` filtering).
    if let Ok(filter) = std::env::var("FUZZ_CLASS") {
        let wanted: Vec<&str> = filter.split(',').map(str::trim).collect();
        jobs.retain(|j| wanted.contains(&j.name.as_str()));
        assert!(!jobs.is_empty(), "FUZZ_CLASS=`{filter}` matched no classes");
    }

    let reports =
        fuzz_jobs(&jobs, iters(), threads(), seed_for).unwrap_or_else(|e| panic!("{e}"));

    for report in &reports {
        eprintln!(
            "{:>22}: compared {:>6}, reserved {:>6}, faulted {:>6}",
            report.class, report.compared, report.reserved, report.faulted
        );
        assert!(report.compared > 0, "class `{}` compared nothing", report.class);
    }
}
