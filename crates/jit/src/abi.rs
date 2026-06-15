//! The contract between generated WASM blocks and the runtime (Milestone 2).
//!
//! ## Linear-memory layout
//!
//! A block runs inside a wasmtime instance whose single linear memory is shared
//! with the host. The host owns the layout:
//!
//! ```text
//!   offset 0x00000  GuestRegs image (REGS_BASE)          regs::offsets::SIZE bytes
//!   offset 0x00340  control block (CTRL_BASE)            exit_reason, exit_pc
//!   offset 0x10000  guest RAM window (RAM_BASE)          ram_bytes
//! ```
//!
//! `GuestRegs` lives at [`REGS_BASE`] so generated code can read e.g. `X5` as an
//! `i64.load` at the constant offset `regs::offsets::x(5)`. Guest RAM is mapped
//! at [`RAM_BASE`]; a guest address `a` maps to linear offset
//! `RAM_BASE + (a - guest_base)` (see [`ram_offset`]). Little-endian matches
//! WASM natively, so loads/stores need no byte-swapping.
//!
//! ## Block function ABI
//!
//! Every generated block, and the [`interpret_one`](crate::runtime) import it
//! leans on, share the signature:
//!
//! ```text
//!   (func (param $regs_base i32) (result i64))
//! ```
//!
//! `$regs_base` is the base of the `GuestRegs` image (always [`REGS_BASE`] for
//! now; threaded as a parameter so a future dispatcher can relocate it). The
//! `i64` result is the **next guest PC**: sequential instructions don't touch
//! PC, the terminator computes the exit PC.
//!
//! ## Exit convention
//!
//! The return value carries the next PC; *why* a block stopped early — when it
//! cannot continue inline (unsupported instruction, and later exceptions,
//! atomics, MMU slow path) — is reported out of band in the control block at
//! [`CTRL_BASE`]: a 64-bit `exit_reason` (see the `EXIT_*` codes) and a 64-bit
//! `exit_pc`. `interpret_one` writes them; the dispatcher reads them. A normal
//! fall-through / taken branch leaves `exit_reason == EXIT_NONE`.
//!
//! We chose the side-channel control block over packing a reason into the
//! returned `i64` so the full 64-bit guest PC stays unambiguous and the reason
//! space is freely extensible.

use aarch64_cpu_state::regs::offsets;

/// Base of the `GuestRegs` image in linear memory.
pub const REGS_BASE: u32 = 0;

/// Base of the runtime control block (just past the register image, aligned).
pub const CTRL_BASE: u32 = 0x340; // offsets::SIZE == 0x320, rounded up to 0x40.

/// `exit_reason` word: one of the `EXIT_*` codes below.
pub const EXIT_REASON: u32 = CTRL_BASE;
/// `exit_pc` word: guest PC associated with a non-`EXIT_NONE` exit.
pub const EXIT_PC: u32 = CTRL_BASE + 8;

/// Base of the guest RAM window (first WASM page boundary above the regs/ctrl
/// area), so guest RAM starts page-aligned and clear of the register image.
pub const RAM_BASE: u32 = 0x10000;

/// Size of one WASM page, in bytes.
pub const WASM_PAGE: usize = 65536;

// Exit reason codes written to [`EXIT_REASON`].
/// Block completed normally (fall-through or taken branch).
pub const EXIT_NONE: u64 = 0;
/// Block hit an instruction the interpreter does not implement; `exit_pc` is the
/// faulting PC and execution made no progress.
pub const EXIT_UNSUPPORTED: u64 = 1;
/// A generated block trapped (e.g. an out-of-bounds inline memory access).
/// `next_pc` is the PC of the instruction that was executing.
pub const EXIT_FAULT: u64 = 2;

/// Linear-memory offset of guest address `addr`, given the guest base address
/// the RAM window starts at.
#[must_use]
pub fn ram_offset(addr: u64, guest_base: u64) -> usize {
    RAM_BASE as usize + (addr - guest_base) as usize
}

/// Compile-time sanity: the control block must not overlap the register image.
const _: () = assert!(CTRL_BASE as usize >= offsets::SIZE);
