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
/// Pinned by the `#[repr(C)]` offset asserts in `state.rs`.
pub mod offsets {
    pub const X: usize = 0;
    pub const SP: usize = 248; // 31 * 8
    pub const PC: usize = 256;
    pub const NZCV: usize = 264;
    pub const V: usize = 272; // 16-byte aligned (u128)
    pub const FPCR: usize = 784; // V + 32 * 16
    /// Total size of the register image (16-byte aligned for the u128 array).
    pub const SIZE: usize = 800;

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
