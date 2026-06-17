//! AArch64 interpreter: execute decoded instructions against a [`CpuState`]
//! (from `aarch64-cpu-state`) and a flat memory image.
//!
//! Each instruction class has its own executor module; `execute` dispatches to
//! them and `run` drives the fetch-decode-execute loop.

// Shared infrastructure.
mod alu;
mod cond;
mod exception;
mod execute;
mod memory;
mod regs;
mod run;

// Per-class executors.
mod add_sub_carry;
mod add_sub_ext_reg;
mod add_sub_imm;
mod add_sub_shifted_reg;
mod bitfield;
mod branch_imm;
mod branch_reg;
mod compare_branch;
mod cond_branch;
mod cond_compare;
mod cond_select;
mod data_proc_1src;
mod data_proc_2src;
mod data_proc_3src;
mod extract;
mod fp;
mod ldst;
mod mem_access;
pub mod mmu;
mod logical_imm;
mod logical_reg;
mod move_wide;
mod pc_rel;
mod psci;
mod simd;
mod system;
mod test_branch;
mod timer;

pub use alu::{add_with_carry, add_with_carry_in, apply_shift};
pub use cond::eval_cond;
pub use exception::{take_irq, undefined};
pub use memory::{GuestMem, MemView, Memory};
pub use mmu::{translate, Access};
pub use run::{run, step, Step, StopReason};
pub use timer::{next_deadline, physical_fires, set_count, set_frequency, virtual_fires};
