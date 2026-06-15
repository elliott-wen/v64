//! Milestone 6: randomized JIT-vs-interpreter sweep across every instruction
//! class. The interpreter is the trusted reference (itself fuzzed against
//! Unicorn). Built only with `--features jit`.
//!
//! Integer/branch/memory classes exercise the inline lowerings; FP/SIMD classes
//! exercise the `interpret_one` fallback (until SIMD/FP lowering lands). Any
//! divergence in registers, memory, or stop reason fails with a reproducible
//! (class, word, seed) message.
//!
//! Classes are grouped into three `#[test]`s so cargo runs them in parallel.
//! `FUZZ_ITERS=<n>` overrides the per-class iteration count.

#![cfg(feature = "jit")]

use aarch64_difftest::{
    encoders::{all_classes, all_fp_classes, all_mem_classes},
    jit_fuzz_class, jit_fuzz_fp_class, jit_fuzz_mem_class,
};

fn iters() -> u32 {
    std::env::var("FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(3_000)
}

/// Per-class fixed seed -> fully reproducible.
fn seed_for(name: &str) -> u64 {
    0x10ad_0000 ^ (u64::from(name.len() as u32) << 8)
}

#[test]
fn simple_classes() {
    let n = iters();
    for class in all_classes() {
        let compared = jit_fuzz_class(&class, n, seed_for(class.name)).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{:>22}: {compared} compared", class.name);
        assert!(compared > 0);
    }
}

#[test]
fn memory_classes() {
    let n = iters();
    for class in all_mem_classes() {
        let compared = jit_fuzz_mem_class(&class, n, seed_for(class.name)).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{:>22}: {compared} compared", class.name);
        assert!(compared > 0);
    }
}

#[test]
fn fp_simd_classes() {
    let n = iters();
    for class in all_fp_classes() {
        let compared = jit_fuzz_fp_class(&class, n, seed_for(class.name)).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{:>22}: {compared} compared", class.name);
        assert!(compared > 0);
    }
}
