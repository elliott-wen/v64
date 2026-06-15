//! AArch64 -> WebAssembly JIT.
//!
//! Translates hot blocks of guest code to WebAssembly, runs them on an embedded
//! wasmtime runtime, and (by construction) produces state bit-identical to the
//! interpreter, which remains the cold-path executor and reference oracle.
//!
//! See `docs/jit-plan.md` for the milestone roadmap. The block↔runtime contract
//! (linear-memory layout, ABI, exit convention) lives in [`abi`].

pub mod abi;
pub mod block;
mod dispatch;
pub mod emit;
mod lower;
pub mod runtime;

pub use block::{form_block, Block};
pub use emit::{emit_block, BLOCK_FUNC};
pub use runtime::{BlockExit, Vm};
