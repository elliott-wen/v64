//! AArch64 → WebAssembly block *emitter*.
//!
//! This crate turns a straight-line block of guest instructions into a
//! WebAssembly module ([`emit_block`]): a leading run of inline-lowerable
//! register ops, ended by an "escape" instruction the host runs via the
//! interpreter (`interpret_one`). It does **not** execute anything — the
//! browser's `WebAssembly` engine instantiates and runs the emitted blocks,
//! sharing one linear memory with the wasm-compiled interpreter. Native builds
//! ship no JIT (interpreter only); the JIT and its differential testing live in
//! the browser/node environment.
//!
//! The block↔runtime contract (linear-memory layout, register image, ABI, exit
//! convention) lives in [`abi`].

pub mod abi;
mod eligible;
pub mod emit;
mod lower;

// Block discovery lives in the decoder crate (shared with the interpreter);
// re-export for existing consumers.
pub use aarch64_decoder::{form_block, Block};
pub use eligible::{can_inline, form_jit_block};
pub use emit::{emit_block, BLOCK_FUNC};
