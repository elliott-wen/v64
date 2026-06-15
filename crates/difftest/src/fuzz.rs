//! Randomized differential testing.
//!
//! For an instruction class we supply an *encoder*. Simple classes emit just a
//! 32-bit word ([`Class`]); memory classes emit a [`MemEncoded`] that also seeds
//! specific registers (so a base pointer lands in mapped memory) and the DATA
//! region. The harness wraps the result in a single-step [`TestVector`] with
//! otherwise-random state, runs it on both our interpreter and Unicorn, and
//! compares the result.
//!
//! Validity is a *checked invariant*, not a blind spot:
//!
//! | ours          | Unicorn       | result                                  |
//! |---------------|---------------|-----------------------------------------|
//! | executes      | `Ran`         | compare architectural state + memory    |
//! | `Unsupported` | `InvalidInsn` | agree it's reserved -> counted, ok      |
//! | `Unsupported` | `Ran`         | **FAIL** — we reject a real instruction |
//! | executes      | `InvalidInsn` | **FAIL** — we accept an invalid one     |
//! | either        | `Fault`       | runtime fault (e.g. unmapped) -> skipped|

use crate::rng::Rng;
use crate::vector::TestVector;

#[cfg(feature = "unicorn")]
use crate::{
    oracle::{run_unicorn_outcome, Outcome},
    ours::run_ours,
    StopReason,
};

#[cfg(feature = "jit")]
use crate::{jit_run::run_on, MAP_BASE, MEM_SIZE};
#[cfg(feature = "jit")]
use aarch64_jit::Vm;

/// Outcome of fuzzing one class.
#[derive(Debug, Clone)]
pub struct FuzzReport {
    pub class: String,
    pub iters: u32,
    /// Words both sides executed and whose state matched.
    pub compared: u32,
    /// Words both sides agreed were invalid/reserved encodings.
    pub reserved: u32,
    /// Words skipped because a valid instruction faulted at runtime.
    pub faulted: u32,
}

/// A simple instruction-class encoder (just a 32-bit word).
pub struct Class {
    pub name: &'static str,
    pub encode: fn(&mut Rng) -> u32,
}

/// An encoded memory-class instruction: the word, register seeds that point a
/// base register into mapped memory, and the initial DATA-region contents.
pub struct MemEncoded {
    pub word: u32,
    /// `(reg, value)` overrides applied after random seeding. `reg == 31` is SP.
    pub seeds: Vec<(u8, u64)>,
    /// `(vreg, value)` SIMD/FP register seeds (for SIMD stores).
    pub init_v: Vec<(u8, u128)>,
    pub data: Vec<u8>,
}

/// A memory instruction-class encoder.
pub struct MemClass {
    pub name: &'static str,
    pub encode: fn(&mut Rng) -> MemEncoded,
}

/// An encoded FP/SIMD instruction: the word, the initial V0..V31 contents, any
/// GPR seeds (e.g. SCVTF reads an integer register), and the FPCR.
pub struct FpEncoded {
    pub word: u32,
    pub init_v: [u128; 32],
    pub gpr_seeds: Vec<(u8, u64)>,
    pub fpcr: u64,
}

/// An FP/SIMD instruction-class encoder.
pub struct FpClass {
    pub name: &'static str,
    pub encode: fn(&mut Rng) -> FpEncoded,
}

/// Random architectural state for a single-step vector (no code yet).
fn random_state(rng: &mut Rng) -> TestVector {
    let mut tv = TestVector::default();
    tv.count = 1;
    for x in &mut tv.init_x {
        *x = rng.next_u64();
    }
    tv.init_sp = rng.next_u64();
    tv.init_nzcv = u64::from(rng.bits(4)) << 28;
    tv
}

/// One comparison outcome for a prepared vector.
#[cfg(feature = "unicorn")]
enum Step {
    Compared,
    Reserved,
    Faulted,
}

/// Run a prepared vector on both sides and classify, enforcing the validity
/// invariant. `here` lazily builds the failure-context prefix.
#[cfg(feature = "unicorn")]
fn step(tv: &TestVector, here: &dyn Fn() -> String) -> Result<Step, String> {
    let (ours, stop) = run_ours(tv);
    let ours_invalid = matches!(stop, StopReason::Unsupported { .. });

    match run_unicorn_outcome(tv) {
        Outcome::Ran(oracle) => {
            if ours_invalid {
                return Err(format!(
                    "{}: we rejected it as Unsupported, but Unicorn executed it \
                     (decoder too strict / instruction unimplemented)",
                    here()
                ));
            }
            if let Some(diff) = ours.diff(&oracle) {
                return Err(format!("{}: {diff}\n ours:   {ours:?}\n oracle: {oracle:?}", here()));
            }
            Ok(Step::Compared)
        }
        Outcome::InvalidInsn => {
            if !ours_invalid {
                return Err(format!(
                    "{}: we executed it, but Unicorn rejects the encoding as invalid \
                     (decoder too permissive)",
                    here()
                ));
            }
            Ok(Step::Reserved)
        }
        Outcome::Fault => Ok(Step::Faulted),
    }
}

#[cfg(feature = "unicorn")]
fn tally(step: Step, compared: &mut u32, reserved: &mut u32, faulted: &mut u32) {
    match step {
        Step::Compared => *compared += 1,
        Step::Reserved => *reserved += 1,
        Step::Faulted => *faulted += 1,
    }
}

/// Fuzz one simple class against the Unicorn oracle.
#[cfg(feature = "unicorn")]
pub fn fuzz_class(class: &Class, iters: u32, seed: u64) -> Result<FuzzReport, String> {
    let mut rng = Rng::new(seed);
    let (mut compared, mut reserved, mut faulted) = (0, 0, 0);

    for i in 0..iters {
        let word = (class.encode)(&mut rng);
        let mut tv = random_state(&mut rng);
        tv.code = word.to_le_bytes().to_vec();
        let here = || format!("class `{}` iter {i} word {word:#010x} (seed {seed:#x})", class.name);
        tally(step(&tv, &here)?, &mut compared, &mut reserved, &mut faulted);
    }

    Ok(FuzzReport { class: class.name.to_string(), iters, compared, reserved, faulted })
}

/// Fuzz one FP/SIMD class against the Unicorn oracle.
#[cfg(feature = "unicorn")]
pub fn fuzz_fp_class(class: &FpClass, iters: u32, seed: u64) -> Result<FuzzReport, String> {
    let mut rng = Rng::new(seed);
    let (mut compared, mut reserved, mut faulted) = (0, 0, 0);

    for i in 0..iters {
        let enc = (class.encode)(&mut rng);
        let word = enc.word;
        let mut tv = random_state(&mut rng);
        tv.code = word.to_le_bytes().to_vec();
        tv.init_v = Some(enc.init_v);
        tv.init_fpcr = enc.fpcr;
        for (reg, val) in &enc.gpr_seeds {
            tv.init_x[*reg as usize] = *val;
        }
        let here = || format!("class `{}` iter {i} word {word:#010x} (seed {seed:#x})", class.name);
        tally(step(&tv, &here)?, &mut compared, &mut reserved, &mut faulted);
    }

    Ok(FuzzReport { class: class.name.to_string(), iters, compared, reserved, faulted })
}

/// Fuzz one memory class against the Unicorn oracle.
#[cfg(feature = "unicorn")]
pub fn fuzz_mem_class(class: &MemClass, iters: u32, seed: u64) -> Result<FuzzReport, String> {
    let mut rng = Rng::new(seed);
    let (mut compared, mut reserved, mut faulted) = (0, 0, 0);

    for i in 0..iters {
        let enc = (class.encode)(&mut rng);
        let word = enc.word;
        let mut tv = random_state(&mut rng);
        tv.code = word.to_le_bytes().to_vec();
        for (reg, val) in &enc.seeds {
            if *reg == 31 {
                tv.init_sp = *val;
            } else {
                tv.init_x[*reg as usize] = *val;
            }
        }
        tv.init_data = Some(enc.data);
        if !enc.init_v.is_empty() {
            let mut v = [0u128; 32];
            for (reg, val) in &enc.init_v {
                v[*reg as usize] = *val;
            }
            tv.init_v = Some(v);
        }
        let here = || format!("class `{}` iter {i} word {word:#010x} (seed {seed:#x})", class.name);
        tally(step(&tv, &here)?, &mut compared, &mut reserved, &mut faulted);
    }

    Ok(FuzzReport { class: class.name.to_string(), iters, compared, reserved, faulted })
}

// ---------------------------------------------------------------------------
// JIT-vs-interpreter sweep (`jit` feature)
//
// The interpreter is the trusted reference here (it is itself fuzzed against
// Unicorn). Each prepared vector runs on both the interpreter and the JIT; any
// divergence in architectural state, touched memory, or stop reason fails. One
// VM is reused across iterations to amortize wasmtime engine setup.
// ---------------------------------------------------------------------------

/// Recreate the reused VM this often. A wasmtime `Store` caps live instances at
/// 10_000 and the dispatcher compiles a fresh instance per (distinct) block, so
/// a long sweep must periodically drop the store. The interval amortizes engine
/// setup while staying well under the cap.
#[cfg(feature = "jit")]
const VM_REFRESH: u32 = 4096;

/// Compare one prepared vector on the interpreter and the JIT.
#[cfg(feature = "jit")]
fn jit_step(vm: &mut Vm, tv: &TestVector, here: &dyn Fn() -> String) -> Result<(), String> {
    let (ours, ours_stop) = crate::ours::run_ours(tv);
    let (jit, jit_stop) = run_on(vm, tv);
    if let Some(diff) = ours.diff(&jit) {
        return Err(format!("{}: {diff}\n ours: {ours:?}\n jit:  {jit:?}", here()));
    }
    if ours_stop != jit_stop {
        return Err(format!("{}: stop reason ours={ours_stop:?} jit={jit_stop:?}", here()));
    }
    Ok(())
}

/// Fuzz one simple class: JIT vs interpreter. Returns the number compared.
#[cfg(feature = "jit")]
pub fn jit_fuzz_class(class: &Class, iters: u32, seed: u64) -> Result<u32, String> {
    let mut rng = Rng::new(seed);
    let mut vm = Vm::new(MAP_BASE, MEM_SIZE);
    for i in 0..iters {
        if i != 0 && i % VM_REFRESH == 0 {
            vm = Vm::new(MAP_BASE, MEM_SIZE);
        }
        let word = (class.encode)(&mut rng);
        let mut tv = random_state(&mut rng);
        tv.code = word.to_le_bytes().to_vec();
        let here = || format!("class `{}` iter {i} word {word:#010x} (seed {seed:#x})", class.name);
        jit_step(&mut vm, &tv, &here)?;
    }
    Ok(iters)
}

/// Fuzz one FP/SIMD class: JIT vs interpreter (the JIT routes these through the
/// interpreter escape hatch until SIMD/FP lowering lands, so this also guards
/// that fallback path).
#[cfg(feature = "jit")]
pub fn jit_fuzz_fp_class(class: &FpClass, iters: u32, seed: u64) -> Result<u32, String> {
    let mut rng = Rng::new(seed);
    let mut vm = Vm::new(MAP_BASE, MEM_SIZE);
    for i in 0..iters {
        if i != 0 && i % VM_REFRESH == 0 {
            vm = Vm::new(MAP_BASE, MEM_SIZE);
        }
        let enc = (class.encode)(&mut rng);
        let word = enc.word;
        let mut tv = random_state(&mut rng);
        tv.code = word.to_le_bytes().to_vec();
        tv.init_v = Some(enc.init_v);
        tv.init_fpcr = enc.fpcr;
        for (reg, val) in &enc.gpr_seeds {
            tv.init_x[*reg as usize] = *val;
        }
        let here = || format!("class `{}` iter {i} word {word:#010x} (seed {seed:#x})", class.name);
        jit_step(&mut vm, &tv, &here)?;
    }
    Ok(iters)
}

/// Fuzz one memory class: JIT vs interpreter.
#[cfg(feature = "jit")]
pub fn jit_fuzz_mem_class(class: &MemClass, iters: u32, seed: u64) -> Result<u32, String> {
    let mut rng = Rng::new(seed);
    let mut vm = Vm::new(MAP_BASE, MEM_SIZE);
    for i in 0..iters {
        if i != 0 && i % VM_REFRESH == 0 {
            vm = Vm::new(MAP_BASE, MEM_SIZE);
        }
        let enc = (class.encode)(&mut rng);
        let word = enc.word;
        let mut tv = random_state(&mut rng);
        tv.code = word.to_le_bytes().to_vec();
        for (reg, val) in &enc.seeds {
            if *reg == 31 {
                tv.init_sp = *val;
            } else {
                tv.init_x[*reg as usize] = *val;
            }
        }
        tv.init_data = Some(enc.data);
        if !enc.init_v.is_empty() {
            let mut v = [0u128; 32];
            for (reg, val) in &enc.init_v {
                v[*reg as usize] = *val;
            }
            tv.init_v = Some(v);
        }
        let here = || format!("class `{}` iter {i} word {word:#010x} (seed {seed:#x})", class.name);
        jit_step(&mut vm, &tv, &here)?;
    }
    Ok(iters)
}
