//! Decoder-coverage sweep against Unicorn.
//!
//! Unlike the per-class differential fuzzer (which checks *correctness* of
//! classes we already decode), this finds *missing* classes: it throws random
//! 32-bit words at our decoder, and for every word we reject as `Unsupported` it
//! asks Unicorn whether the word is a real instruction. A word Unicorn executes
//! but we reject is a decoding gap.
//!
//! Unicorn (QEMU) `abort()`s on some malformed encodings (e.g. certain fp16
//! SIMD), which would kill the whole run. So the sweep is split into batches,
//! each run in a *child process* (this same test binary re-invoked with
//! `V64_COV_RANGE`); a child that aborts only loses its batch, and the parent
//! keeps going. Discovery tool, `#[ignore]`d by default:
//!
//! ```text
//! COVERAGE_ITERS=2000000 cargo test -p aarch64-difftest --features unicorn \
//!     --test coverage -- --ignored --nocapture
//! ```

#![cfg(feature = "unicorn")]

use std::collections::BTreeMap;
use std::process::Command;

use aarch64_decoder::{decode, Insn};
use aarch64_difftest::{run_unicorn_outcome, Outcome, TestVector, DATA_BASE, DATA_SIZE};

const BATCH: u64 = 1000;

fn iters() -> u64 {
    std::env::var("COVERAGE_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000)
}

/// Deterministic word for index `i` (splitmix64), so any range is reproducible
/// and the parent can re-run/skip batches.
fn word_at(i: u64) -> u32 {
    let mut z = i.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) as u32
}

/// Probe vector: all GPRs/SP point into the mapped DATA window and V/data are
/// zeroed, so an undecoded load/store still *executes* in Unicorn (rather than
/// faulting, which would hide the gap).
fn probe(word: u32) -> TestVector {
    let mut tv = TestVector::new(&word.to_le_bytes());
    tv.init_x = [DATA_BASE; 31];
    tv.init_sp = DATA_BASE;
    tv.init_data = Some(vec![0u8; DATA_SIZE]);
    tv.init_v = Some([0u128; 32]);
    tv.count = 1;
    tv
}

/// Child mode: process `[start,end)`, print `GAP <hex>` for each decoding gap.
fn run_child(start: u64, end: u64) {
    use std::io::Write;
    let out = std::io::stdout();
    let mut out = out.lock();
    for i in start..end {
        let word = word_at(i);
        if !matches!(decode(word), Insn::Unsupported { .. }) {
            continue; // we decode it
        }
        if let Outcome::Ran(_) = run_unicorn_outcome(&probe(word)) {
            let _ = writeln!(out, "GAP {word:08x}");
            let _ = out.flush(); // flush so a later abort doesn't lose this line
        }
    }
}

#[test]
#[ignore]
fn coverage_vs_unicorn() {
    // Child invocation?
    if let Ok(range) = std::env::var("V64_COV_RANGE") {
        let (s, e) = range.split_once(':').unwrap();
        run_child(s.parse().unwrap(), e.parse().unwrap());
        return;
    }

    let n = iters();
    let exe = std::env::current_exe().unwrap();
    let mut gaps: BTreeMap<u32, (u64, u32)> = BTreeMap::new();
    let mut crashes = 0u64;

    let mut start = 0;
    while start < n {
        let end = (start + BATCH).min(n);
        let output = Command::new(&exe)
            .args(["--ignored", "--nocapture", "coverage_vs_unicorn"])
            .env("V64_COV_RANGE", format!("{start}:{end}"))
            .output()
            .expect("spawn child");
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Some(hex) = line.strip_prefix("GAP ") {
                if let Ok(word) = u32::from_str_radix(hex.trim(), 16) {
                    let e = gaps.entry(word & 0xFFE0_0000).or_insert((0, word));
                    e.0 += 1;
                }
            }
        }
        if !output.status.success() {
            crashes += 1; // QEMU aborted somewhere in this batch; skip the rest of it
        }
        start = end;
    }

    eprintln!("\ncoverage sweep: {n} random words, {crashes} batches hit a Unicorn abort");
    eprintln!("decoding gaps (Unicorn executes, we return Unsupported): {} families", gaps.len());
    let mut fams: Vec<_> = gaps.into_iter().collect();
    fams.sort_by_key(|(_, (count, _))| std::cmp::Reverse(*count));
    for (key, (count, example)) in fams {
        eprintln!("  family {key:#010x}  e.g. {example:#010x}  hits={count}");
    }
}
