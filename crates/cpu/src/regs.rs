//! Flat, offset-addressable guest register block for the JIT.
//!
//! Generated WASM reads/writes guest registers as loads/stores at *constant
//! offsets* into a copy of this struct held in the wasmtime instance's linear
//! memory. The layout is `#[repr(C)]` and pinned by the offset table below
//! (`offsets`), with a unit test that fails if any offset shifts.
//!
//! `nzcv` is the **packed** condition-flag word (N=bit31, Z=bit30, C=bit29,
//! V=bit28), not the interpreter's four-bool [`crate::Flags`]. The interpreter
//! keeps working with `Flags`; conversion happens only at the JIT boundary via
//! [`crate::CpuState::to_guest_regs`] / [`crate::CpuState::load_guest_regs`].

/// The hot, raw-memory-addressable register file the JIT operates on.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestRegs {
    /// X0..X30. X31 is SP or XZR depending on the instruction (not stored here).
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    /// Packed NZCV (bit31 N, bit30 Z, bit29 C, bit28 V).
    pub nzcv: u64,
    /// SIMD/FP registers V0..V31 (128-bit).
    pub v: [u128; 32],
    /// Floating-point control register.
    pub fpcr: u64,
}

impl Default for GuestRegs {
    fn default() -> Self {
        Self { x: [0; 31], sp: 0, pc: 0, nzcv: 0, v: [0; 32], fpcr: 0 }
    }
}

/// Byte offsets of each field within [`GuestRegs`], shared by the JIT emitter
/// and the runtime. Pinned by `tests::offsets_are_stable`.
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
