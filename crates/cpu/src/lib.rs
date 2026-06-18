//! AArch64 CPU state.
//!
//! Modelled after `CPUARMState` in the reference QEMU tree
//! (`unicorn/qemu/target/arm/cpu.h`), trimmed to the EL0 user-mode subset we
//! currently execute.

mod flags;
pub mod regs;
mod state;
mod tlb;

pub use flags::Flags;
pub use state::{
    Abort, CpuState, EL_OFFSET, JIT_COUNT_OFFSET, JIT_EXIT_OFFSET, SP_OR_ZR, TLB_OFFSET,
};
pub use tlb::{Tlb, ENTRIES as TLB_ENTRIES, ENTRY_PA, ENTRY_PERMS, ENTRY_SIZE, ENTRY_TAG};
