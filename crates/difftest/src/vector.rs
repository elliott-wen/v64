//! A differential test case.

use crate::CODE_START;

/// Machine code plus the initial architectural state to run it from.
///
/// Both implementations are seeded identically from this vector, so any
/// divergence after execution is a real behavioural difference — not a
/// reset-state artifact. (Unicorn's ARM64 reset leaves `NZCV` with `Z` set, so
/// `init_nzcv` is written explicitly on both sides.)
#[derive(Debug, Clone, Default)]
pub struct TestVector {
    pub code: Vec<u8>,
    /// Initial X0..X30 (index == register number).
    pub init_x: [u64; 31],
    pub init_sp: u64,
    pub init_nzcv: u64,
    /// When set, these bytes are written at `DATA_BASE` before the run and the
    /// region is compared afterwards (load/store testing).
    pub init_data: Option<Vec<u8>>,
    /// When set, V0..V31 are seeded and compared afterwards (FP/SIMD testing).
    pub init_v: Option<[u128; 32]>,
    /// Initial FPCR (rounding mode / default-NaN / flush-to-zero).
    pub init_fpcr: u64,
    /// Instruction count cap (0 = run until end of code).
    pub count: usize,
    /// Arbitrary physical-memory writes applied before the run (e.g. page
    /// tables). `(physical_addr, bytes)`. Applied after `code`/`init_data`.
    pub mem_patches: Vec<(u64, Vec<u8>)>,
    /// When true the oracle uses Unicorn's CPU TLB so it performs real ARM
    /// stage-1 translation-table walks (needed for MMU tests; the guest code
    /// itself enables the MMU via `MSR`). Off for the MMU-disabled ISA fuzzers.
    pub cpu_tlb: bool,
}

impl TestVector {
    #[must_use]
    pub fn new(code: &[u8]) -> Self {
        Self { code: code.to_vec(), ..Default::default() }
    }

    #[must_use]
    pub fn with_x(mut self, idx: usize, val: u64) -> Self {
        self.init_x[idx] = val;
        self
    }

    #[must_use]
    pub fn with_nzcv(mut self, nzcv: u64) -> Self {
        self.init_nzcv = nzcv;
        self
    }

    /// Address one past the last code byte — the natural `until` stop point.
    #[must_use]
    pub fn until(&self) -> u64 {
        CODE_START + self.code.len() as u64
    }
}
