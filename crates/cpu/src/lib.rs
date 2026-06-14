//! AArch64 CPU state.
//!
//! Modelled after `CPUARMState` in the reference QEMU tree
//! (`unicorn/qemu/target/arm/cpu.h`), trimmed to the EL0 user-mode subset we
//! currently execute.

mod flags;
pub mod regs;
mod state;

pub use flags::Flags;
pub use regs::GuestRegs;
pub use state::{CpuState, SP_OR_ZR};
