//! Iteration-level parallel sweep.
//!
//! The per-class harness in [`crate::fuzz`] runs one class's iterations on a
//! single thread; parallelism then comes only from running different classes at
//! once, so wall-time is bounded by the slowest single class. This module
//! instead flattens *every class's iterations* into one flat list of work units
//! and runs them through a single pool: a slow class's iterations are split
//! across cores too, so wall-time approaches `total_work / threads` regardless
//! of how the work is distributed between classes.
//!
//! Reproducibility is preserved. Each unit gets a deterministic seed derived
//! from its class's base seed and its ordinal, and a divergence reports
//! `(class, chunk-seed, iter, word)` — enough to replay that unit exactly.

#![cfg(feature = "unicorn")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

use crate::encoders::{all_classes, all_fp_classes, all_mem_classes};
use crate::fuzz::{step, tally, vector_fp, vector_mem, vector_simple, word_of, FuzzReport};
use crate::rng::Rng;
use crate::vector::TestVector;

/// One instruction class plus a closure that builds a random single-step vector
/// for it. Boxed so the three encoder kinds share one job type.
pub struct Job {
    pub name: String,
    make: Box<dyn Fn(&mut Rng) -> TestVector + Send + Sync>,
}

/// Iterations per work unit. Small enough that one slow class is spread across
/// many cores; large enough that per-unit dispatch overhead stays negligible.
const CHUNK: u32 = 2_000;

/// Every fuzzable class (simple, memory, FP/SIMD) as a uniform job list.
#[must_use]
pub fn all_jobs() -> Vec<Job> {
    let mut jobs = Vec::new();
    for c in all_classes() {
        jobs.push(Job { name: c.name.to_string(), make: Box::new(move |r| vector_simple(&c, r)) });
    }
    for c in all_mem_classes() {
        jobs.push(Job { name: c.name.to_string(), make: Box::new(move |r| vector_mem(&c, r)) });
    }
    for c in all_fp_classes() {
        jobs.push(Job { name: c.name.to_string(), make: Box::new(move |r| vector_fp(&c, r)) });
    }
    jobs
}

/// A contiguous slice of one job's iterations, seeded independently.
struct Unit {
    job: usize,
    seed: u64,
    iters: u32,
}

#[derive(Default, Clone)]
struct Counts {
    compared: u32,
    reserved: u32,
    faulted: u32,
}

/// Fuzz every job at iteration granularity across `threads` workers.
///
/// `seed_for` maps a class name to its base seed (kept identical to the serial
/// harness so failures replay the same way). Returns one [`FuzzReport`] per job
/// in input order, or the first divergence found.
pub fn fuzz_jobs(
    jobs: &[Job],
    iters: u32,
    threads: usize,
    seed_for: impl Fn(&str) -> u64,
) -> Result<Vec<FuzzReport>, String> {
    // Flatten into work units up front: each job's iterations split into chunks,
    // all chunks of all jobs feeding one queue.
    let mut units = Vec::new();
    for (j, job) in jobs.iter().enumerate() {
        let base = seed_for(&job.name);
        let (mut done, mut ordinal) = (0u32, 0u64);
        while done < iters {
            let n = (iters - done).min(CHUNK);
            // Derive a distinct, deterministic seed per chunk.
            let seed = base ^ ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            units.push(Unit { job: j, seed, iters: n });
            done += n;
            ordinal += 1;
        }
    }

    let next = AtomicUsize::new(0);
    let totals: Mutex<Vec<Counts>> = Mutex::new(vec![Counts::default(); jobs.len()]);
    let failure: Mutex<Option<String>> = Mutex::new(None);

    thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                let mut local = vec![Counts::default(); jobs.len()];
                loop {
                    // Stop pulling new work as soon as any worker has diverged.
                    if failure.lock().unwrap().is_some() {
                        break;
                    }
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    let Some(unit) = units.get(idx) else { break };
                    if let Err(e) = run_unit(&jobs[unit.job], unit, &mut local[unit.job]) {
                        *failure.lock().unwrap() = Some(e);
                        break;
                    }
                }
                let mut t = totals.lock().unwrap();
                for (dst, src) in t.iter_mut().zip(local) {
                    dst.compared += src.compared;
                    dst.reserved += src.reserved;
                    dst.faulted += src.faulted;
                }
            });
        }
    });

    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }

    let totals = totals.into_inner().unwrap();
    Ok(jobs
        .iter()
        .zip(totals)
        .map(|(job, c)| FuzzReport {
            class: job.name.clone(),
            iters,
            compared: c.compared,
            reserved: c.reserved,
            faulted: c.faulted,
        })
        .collect())
}

/// Run one work unit, accumulating into the job's counters.
fn run_unit(job: &Job, unit: &Unit, c: &mut Counts) -> Result<(), String> {
    let mut rng = Rng::new(unit.seed);
    for i in 0..unit.iters {
        let tv = (job.make)(&mut rng);
        let word = word_of(&tv);
        let here = || {
            format!(
                "class `{}` chunk-seed {:#x} iter {i} word {word:#010x}",
                job.name, unit.seed
            )
        };
        tally(step(&tv, &here)?, &mut c.compared, &mut c.reserved, &mut c.faulted);
    }
    Ok(())
}
