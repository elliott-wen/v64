//! AArch64 → WebAssembly block/region *emitter*.
//!
//! Turns guest code into a WebAssembly module: [`emit_block`] for a single basic
//! block, [`emit_region`] for a region of blocks ([`form_region`]) wired into one
//! function with an internal dispatch loop. Each lowering emits native wasm and,
//! for memory, an inline TLB-checked fast path that **bails** to the interpreter
//! on a miss (the block returns the faulting PC; no escape import). The module
//! shares the host's linear memory, so generated code reads/writes the live
//! `CpuState` and guest RAM directly. This crate does not execute anything — the
//! browser/node `WebAssembly` engine instantiates and runs the modules. Native
//! builds ship no JIT (interpreter only).

mod eligible;
pub mod emit;
mod lower;

// Block discovery lives in the decoder crate (shared with the interpreter);
// re-export for existing consumers.
pub use aarch64_decoder::{form_block, Block};
pub use eligible::{
    can_inline, form_jit_block, form_region, is_inline_load_store, is_inline_load_store_pair,
    is_inline_mem, Region,
};
pub use emit::{emit_block, emit_region, BLOCK_FUNC};
