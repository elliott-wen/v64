//! Differential testing harness: run the same machine code through our
//! interpreter and through Unicorn (the oracle), then compare architectural
//! state register-by-register.
//!
//! The vector type, snapshot, and our-interpreter runner are always available.
//! The Unicorn oracle is behind the `unicorn` feature because building it
//! compiles QEMU via cmake. Run the full cross-check with:
//!
//! ```text
//! cargo test -p aarch64-difftest --features unicorn
//! ```

mod fuzz;
mod ours;
mod rng;
mod snapshot;
mod vector;

pub mod encoders;

pub const MAP_BASE: u64 = 0; // mapped region start
pub const MEM_SIZE: usize = 0x10_0000; // 1 MiB image
pub const CODE_START: u64 = 0x8_0000; // code mid-region, so +/- branches stay mapped

/// Scratch region that load/store fuzzing targets. Seeded identically on both
/// sides and compared after execution. Kept well away from the code.
pub const DATA_BASE: u64 = 0x4_0000;
pub const DATA_SIZE: usize = 0x2000; // 8 KiB

pub use aarch64_interp::StopReason;
pub use fuzz::{Class, FpClass, FpEncoded, FuzzReport, MemClass, MemEncoded};
pub use ours::run_ours;
pub use rng::Rng;
pub use snapshot::StateSnapshot;
pub use vector::TestVector;

#[cfg(feature = "unicorn")]
mod oracle;

#[cfg(feature = "unicorn")]
pub use fuzz::{fuzz_class, fuzz_fp_class, fuzz_mem_class};
#[cfg(feature = "unicorn")]
pub use oracle::{assert_matches_oracle, run_unicorn, run_unicorn_outcome, Outcome};
