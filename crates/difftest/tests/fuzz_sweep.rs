//! Randomized differential sweep: every instruction class vs Unicorn.
//!
//! Built only with `--features unicorn`. Each class is its own `#[test]`, so
//! cargo's test runner executes them in parallel across cores (wall-time is
//! roughly the slowest single class rather than the sum of all of them). Any
//! state divergence fails with a reproducible (class, word, seed) description.

#![cfg(feature = "unicorn")]

use aarch64_difftest::{
    encoders::{all_classes, all_fp_classes, all_mem_classes},
    fuzz_class, fuzz_fp_class, fuzz_mem_class, FuzzReport,
};

/// Iterations per class. Override with `FUZZ_ITERS=<n>` to fuzz harder
/// (e.g. `FUZZ_ITERS=500000` for ~30 min across all classes in parallel).
fn iters() -> u32 {
    std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000)
}

fn report_of(name: &str, report: FuzzReport) {
    eprintln!(
        "{:>22}: compared {:>5}, reserved {:>5}, faulted {:>5}",
        report.class, report.compared, report.reserved, report.faulted
    );
    assert!(report.compared > 0, "class `{name}` compared nothing");
}

/// Per-class fixed seed -> fully reproducible.
fn seed_for(name: &str) -> u64 {
    0x5eed_0000 ^ (u64::from(name.len() as u32) << 8)
}

fn run_one(name: &str) {
    let class = all_classes()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("unknown class `{name}`"));
    let report = fuzz_class(&class, iters(), seed_for(name)).unwrap_or_else(|e| panic!("{e}"));
    report_of(name, report);
}

fn run_one_mem(name: &str) {
    let class = all_mem_classes()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("unknown mem class `{name}`"));
    let report = fuzz_mem_class(&class, iters(), seed_for(name)).unwrap_or_else(|e| panic!("{e}"));
    report_of(name, report);
}

fn run_one_fp(name: &str) {
    let class = all_fp_classes()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("unknown fp class `{name}`"));
    let report = fuzz_fp_class(&class, iters(), seed_for(name)).unwrap_or_else(|e| panic!("{e}"));
    report_of(name, report);
}

/// One `#[test]` per class so they run concurrently.
macro_rules! class_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run_one(stringify!($name));
            }
        )*
    };
}

/// One `#[test]` per memory class.
macro_rules! mem_class_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run_one_mem(stringify!($name));
            }
        )*
    };
}

mem_class_tests!(
    ldst_uimm,
    ldst_unscaled,
    ldst_post,
    ldst_pre,
    ldst_reg,
    ldst_literal,
    ldst_pair,
    ldst_ordered,
    ldst_atomic,
    ldst_cas,
);

/// One `#[test]` per FP class.
macro_rules! fp_class_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run_one_fp(stringify!($name));
            }
        )*
    };
}

fp_class_tests!(
    fp_dp1,
    fp_dp2,
    fp_compare,
    fp_csel,
    fp_imm,
    fp_cvt,
    neon_three_same,
    neon_three_diff,
    neon_three_same_fp,
    neon_two_reg_misc,
    neon_two_reg_misc_fp,
    neon_mod_imm,
    neon_dup,
    neon_dup_element,
    neon_ins,
    neon_mov_gpr,
    neon_zip_trn,
    neon_ext,
    neon_shift_imm,
    neon_across,
);

class_tests!(
    move_wide,
    add_sub_imm,
    logical_imm,
    bitfield,
    extract,
    add_sub_shifted_reg,
    add_sub_ext_reg,
    add_sub_carry,
    logical_reg,
    cond_select,
    cond_compare,
    data_proc_1src,
    data_proc_2src,
    data_proc_3src,
    pc_rel,
    branch_imm,
    cond_branch,
    compare_branch,
    test_branch,
);
