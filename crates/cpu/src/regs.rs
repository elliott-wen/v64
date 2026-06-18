//! Byte offsets of the JIT-addressable guest registers.
//!
//! Generated WASM reads/writes guest registers as loads/stores at *constant
//! offsets* into the live [`crate::CpuState`], which shares the block's linear
//! memory. `CpuState` is `#[repr(C)]` with its leading register fields pinned to
//! the [`offsets`] table here, asserted at compile time in `state.rs` — so a
//! block mutates the real registers in place with no image copy.
//!
//! The block works in the **packed** condition-flag word `nzcv` (N=bit31,
//! Z=bit30, C=bit29, V=bit28), not the interpreter's four-bool [`crate::Flags`];
//! the organizer packs/unpacks the one `u64` around a block run.

/// Byte offsets of each JIT-addressable register field within [`crate::CpuState`]
/// (`x`, `sp`, `pc`, `nzcv`, `v`), shared by the JIT emitter and the organizer.
/// Derived from the `#[repr(C)]` layout via `offset_of!`, so they track the
/// struct and can't drift. (Expected values for orientation: X=0, SP=248=31*8,
/// PC=256, NZCV=264, V=272.)
pub mod offsets {
    use crate::CpuState;
    use std::mem::offset_of;

    pub const X: usize = offset_of!(CpuState, x);
    pub const SP: usize = offset_of!(CpuState, sp);
    pub const PC: usize = offset_of!(CpuState, pc);
    pub const NZCV: usize = offset_of!(CpuState, nzcv);
    pub const V: usize = offset_of!(CpuState, v);

    /// Byte offset of `X[n]`.
    #[must_use]
    pub const fn x(n: usize) -> usize {
        X + n * 8
    }
    /// Byte offset of `V[n]`.
    #[must_use]
    pub const fn v(n: usize) -> usize {
        V + n * 16
    }
}
