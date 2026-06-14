//! The Unicorn oracle and the cross-check entry point.
//!
//! This whole module is behind the `unicorn` feature because building Unicorn
//! compiles QEMU via cmake.

use unicorn_engine::{uc_error, Arch, Arm64CpuModel, Mode, Prot, RegisterARM64, Unicorn};

use crate::ours::run_ours;
use crate::snapshot::StateSnapshot;
use crate::vector::TestVector;
use crate::{CODE_START, DATA_BASE, DATA_SIZE, MAP_BASE, MEM_SIZE};

/// X0..X30 in index order, for batched register I/O against Unicorn.
const XREGS: [RegisterARM64; 31] = [
    RegisterARM64::X0, RegisterARM64::X1, RegisterARM64::X2, RegisterARM64::X3,
    RegisterARM64::X4, RegisterARM64::X5, RegisterARM64::X6, RegisterARM64::X7,
    RegisterARM64::X8, RegisterARM64::X9, RegisterARM64::X10, RegisterARM64::X11,
    RegisterARM64::X12, RegisterARM64::X13, RegisterARM64::X14, RegisterARM64::X15,
    RegisterARM64::X16, RegisterARM64::X17, RegisterARM64::X18, RegisterARM64::X19,
    RegisterARM64::X20, RegisterARM64::X21, RegisterARM64::X22, RegisterARM64::X23,
    RegisterARM64::X24, RegisterARM64::X25, RegisterARM64::X26, RegisterARM64::X27,
    RegisterARM64::X28, RegisterARM64::X29, RegisterARM64::X30,
];

/// V0..V31 in index order, for 128-bit register I/O against Unicorn.
const VREGS: [RegisterARM64; 32] = [
    RegisterARM64::V0, RegisterARM64::V1, RegisterARM64::V2, RegisterARM64::V3,
    RegisterARM64::V4, RegisterARM64::V5, RegisterARM64::V6, RegisterARM64::V7,
    RegisterARM64::V8, RegisterARM64::V9, RegisterARM64::V10, RegisterARM64::V11,
    RegisterARM64::V12, RegisterARM64::V13, RegisterARM64::V14, RegisterARM64::V15,
    RegisterARM64::V16, RegisterARM64::V17, RegisterARM64::V18, RegisterARM64::V19,
    RegisterARM64::V20, RegisterARM64::V21, RegisterARM64::V22, RegisterARM64::V23,
    RegisterARM64::V24, RegisterARM64::V25, RegisterARM64::V26, RegisterARM64::V27,
    RegisterARM64::V28, RegisterARM64::V29, RegisterARM64::V30, RegisterARM64::V31,
];

/// How Unicorn responded to a vector. Distinguishing "this encoding is not a
/// valid instruction" from "a valid instruction faulted at runtime" lets the
/// fuzzer treat decode-validity as a checked invariant while still skipping
/// environment-dependent faults (e.g. an unmapped load address).
#[derive(Debug)]
pub enum Outcome {
    /// Executed cleanly; here is the resulting architectural state.
    Ran(StateSnapshot),
    /// Unicorn rejected the encoding as an invalid/unallocated instruction.
    InvalidInsn,
    /// A valid instruction raised a runtime fault (unmapped memory, etc.), or
    /// the harness could not set up the run.
    Fault,
}

/// Run a vector on Unicorn and classify the outcome.
#[must_use]
pub fn run_unicorn_outcome(tv: &TestVector) -> Outcome {
    let Some(mut uc) = setup(tv) else {
        return Outcome::Fault;
    };
    match uc.emu_start(CODE_START, tv.until(), 0, tv.count) {
        Ok(()) => match read_snapshot(&uc, tv.init_data.is_some(), tv.init_v.is_some()) {
            Some(s) => Outcome::Ran(s),
            None => Outcome::Fault,
        },
        // An unallocated encoding surfaces either as INSN_INVALID or, more
        // commonly on AArch64, as the undefined-instruction trap EXCEPTION.
        // Treating EXCEPTION as "invalid" is safe here because the fuzzer never
        // generates system/exception-generating instructions (SVC/BRK/...),
        // whose traps would otherwise be indistinguishable. Memory faults are
        // the distinct *_UNMAPPED errors and remain runtime Faults — important
        // once load/store lands.
        Err(uc_error::INSN_INVALID | uc_error::EXCEPTION) => Outcome::InvalidInsn,
        Err(_) => Outcome::Fault,
    }
}

fn setup(tv: &TestVector) -> Option<Unicorn<'static, ()>> {
    let mut uc = Unicorn::new(Arch::ARM64, Mode::ARM).ok()?;
    // The default CPU predates ARMv8.1, so it lacks LSE atomics (and other
    // later features). MAX enables the full feature set so the oracle accepts
    // everything we implement. Must be set before anything else.
    uc.ctl_set_cpu_model(Arm64CpuModel::MAX as i32).ok()?;
    uc.mem_map(MAP_BASE, MEM_SIZE as u64, Prot::ALL).ok()?;
    uc.mem_write(CODE_START, &tv.code).ok()?;
    if let Some(data) = &tv.init_data {
        uc.mem_write(DATA_BASE, data).ok()?;
    }
    for (i, reg) in XREGS.iter().enumerate() {
        uc.reg_write(*reg, tv.init_x[i]).ok()?;
    }
    uc.reg_write(RegisterARM64::SP, tv.init_sp).ok()?;
    uc.reg_write(RegisterARM64::NZCV, tv.init_nzcv).ok()?;
    if let Some(v) = &tv.init_v {
        for (i, reg) in VREGS.iter().enumerate() {
            uc.reg_write_long(*reg, &v[i].to_le_bytes()).ok()?;
        }
        uc.reg_write(RegisterARM64::FPCR, tv.init_fpcr).ok()?;
    }
    Some(uc)
}

fn read_snapshot(uc: &Unicorn<'static, ()>, with_data: bool, with_v: bool) -> Option<StateSnapshot> {
    let mut x = [0u64; 31];
    for (i, reg) in XREGS.iter().enumerate() {
        x[i] = uc.reg_read(*reg).ok()?;
    }
    let data = if with_data {
        let mut buf = vec![0u8; DATA_SIZE];
        uc.mem_read(DATA_BASE, &mut buf).ok()?;
        buf
    } else {
        Vec::new()
    };
    let v = if with_v {
        let mut out = Vec::with_capacity(32);
        for reg in &VREGS {
            let bytes = uc.reg_read_long(*reg).ok()?;
            let arr: [u8; 16] = bytes.as_ref().try_into().ok()?;
            out.push(u128::from_le_bytes(arr));
        }
        out
    } else {
        Vec::new()
    };
    Some(StateSnapshot {
        x,
        sp: uc.reg_read(RegisterARM64::SP).ok()?,
        pc: uc.reg_read(RegisterARM64::PC).ok()?,
        nzcv: uc.reg_read(RegisterARM64::NZCV).ok()?,
        data,
        v,
    })
}

/// Execute a vector on Unicorn and snapshot the result. Panics on any Unicorn
/// error (use [`run_unicorn_outcome`] when an error is an expected outcome).
#[must_use]
pub fn run_unicorn(tv: &TestVector) -> StateSnapshot {
    match run_unicorn_outcome(tv) {
        Outcome::Ran(s) => s,
        other => panic!("unicorn run failed: {other:?}"),
    }
}

/// Run a vector through both implementations and assert they agree.
pub fn assert_matches_oracle(tv: &TestVector) {
    let (ours, stop) = run_ours(tv);
    assert!(
        !matches!(stop, aarch64_interp::StopReason::Unsupported { .. }),
        "interpreter could not execute vector: {stop:?}"
    );
    let oracle = run_unicorn(tv);
    if let Some(diff) = ours.diff(&oracle) {
        panic!("differential mismatch: {diff}\n ours:   {ours:?}\n oracle: {oracle:?}");
    }
}
