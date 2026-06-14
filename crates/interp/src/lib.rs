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
mod ldst_atomic;
mod ldst_cas;
mod ldst_excl;
mod ldst_pair;
mod mem_access;
pub mod mmu;
mod logical_imm;
mod logical_reg;
mod move_wide;
mod pc_rel;
mod simd;
mod simd_across;
mod simd_copy;
mod simd_dup;
mod simd_ext;
mod simd_indexed;
mod simd_mod_imm;
mod simd_permute;
mod simd_shift_fp;
mod simd_shift_imm;
mod simd_shift_long;
mod simd_shift_narrow;
mod simd_three_diff;
mod simd_three_same;
mod simd_three_same_fp;
mod simd_two_reg_long;
mod simd_two_reg_misc;
mod simd_two_reg_misc_fp;
mod simd_two_reg_narrow;
mod system;
mod test_branch;

pub use alu::{add_with_carry, add_with_carry_in, apply_shift};
pub use cond::eval_cond;
pub use memory::Memory;
pub use mmu::translate;
pub use run::{run, StopReason};
