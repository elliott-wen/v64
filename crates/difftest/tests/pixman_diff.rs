//! Diff every distinct instruction word from a real libpixman build against the
//! Unicorn oracle, to hunt the NEON instruction the interpreter mis-executes
//! (the one that crashes Xorg). Reads hex words (one per line) from the file in
//! PIXMAN_WORDS (default /tmp/px/words.txt); skips words our interp can't decode
//! or that fault/are-invalid on either side, and reports any word both sides
//! execute but disagree on.
//!
//!   PIXMAN_WORDS=/tmp/px/words.txt cargo test -p aarch64-difftest \
//!       --features unicorn --release --test pixman_diff -- --nocapture

#![cfg(feature = "unicorn")]

use std::collections::BTreeMap;

use aarch64_difftest::{run_ours, run_unicorn_outcome, Outcome, Rng, StopReason, TestVector};

/// FPCR.DN=1 — default-NaN, so FP results are deterministic across impls.
const FPCR_DN: u64 = 1 << 25;
const SEEDS_PER_WORD: usize = 24;

fn rand_vec(rng: &mut Rng) -> [u128; 32] {
    let mut v = [0u128; 32];
    for e in &mut v {
        *e = (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64());
    }
    v
}

/// All GPRs point into the mapped scratch DATA region (0x40000..0x42000), so a
/// word that turns out to be a load/store accesses valid, compared memory
/// instead of a wild address. Mid-region with headroom for offsets/increments.
const DATA_BASE: u64 = 0x4_0000;
fn rand_x(rng: &mut Rng) -> [u64; 31] {
    let mut x = [0u64; 31];
    for e in &mut x {
        *e = DATA_BASE + 0x800 + u64::from(rng.bits(10)); // 0x40800..0x40c00
    }
    x
}

#[test]
fn pixman_words_match_unicorn() {
    let path = std::env::var("PIXMAN_WORDS").unwrap_or_else(|_| "/tmp/px/words.txt".to_string());
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("pixman_diff: no word list at {path}; skipping");
        return;
    };
    let words: Vec<u32> = text
        .lines()
        .filter_map(|l| u32::from_str_radix(l.trim(), 16).ok())
        .collect();
    eprintln!("pixman_diff: {} words", words.len());
    std::panic::set_hook(Box::new(|_| {})); // silence out-of-range/invalid panics

    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15_u64);
    // word -> first divergence description.
    let mut bad: BTreeMap<u32, String> = BTreeMap::new();
    let mut compared = 0u64;

    for &word in &words {
        for _ in 0..SEEDS_PER_WORD {
            let tv = TestVector {
                code: word.to_le_bytes().to_vec(),
                init_x: rand_x(&mut rng),
                init_v: Some(rand_vec(&mut rng)),
                init_data: Some((0..0x2000).map(|_| rng.bits(8) as u8).collect()),
                init_fpcr: FPCR_DN,
                init_nzcv: u64::from(rng.bits(4)) << 28,
                count: 1,
                ..Default::default()
            };
            // A malformed/odd word can still compute a wild address; our memory
            // model panics on out-of-range rather than faulting, so guard it.
            let Ok((ours, stop)) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_ours(&tv)))
            else {
                break;
            };
            if matches!(stop, StopReason::Unsupported { .. }) {
                break; // we don't implement it; not the bug we're hunting
            }
            let theirs_outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_unicorn_outcome(&tv)));
            match theirs_outcome.unwrap_or(Outcome::Fault) {
                Outcome::Ran(theirs) => {
                    compared += 1;
                    if let Some(d) = ours.diff(&theirs) {
                        bad.entry(word).or_insert(d);
                        break;
                    }
                }
                // Faulted (e.g. a load from an unmapped random base) or invalid:
                // not a clean data-processing comparison, skip this seed.
                _ => {}
            }
        }
    }

    eprintln!("pixman_diff: compared {compared} (word,seed) pairs, {} diverging words", bad.len());
    for (word, d) in &bad {
        // Disassemble for readability via the decoder's Debug.
        let insn = aarch64_decoder::decode(*word);
        eprintln!("  {word:08x}  {insn:?}  | {d}");
    }
    assert!(bad.is_empty(), "{} pixman instruction word(s) diverge from Unicorn", bad.len());
}
