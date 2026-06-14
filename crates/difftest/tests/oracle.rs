//! Differential tests against the Unicorn oracle.
//!
//! Built only with `--features unicorn`; otherwise this file compiles to an
//! empty test binary.

#![cfg(feature = "unicorn")]

use aarch64_difftest::{assert_matches_oracle, TestVector};

#[test]
fn matches_unicorn_basic() {
    // mov x16,#1 ; mov x17,#0x20 ; add x28,x28,8
    let tv = TestVector::new(&[
        0x30, 0x00, 0x80, 0xd2,
        0x11, 0x04, 0x80, 0xd2,
        0x9c, 0x23, 0x00, 0x91,
    ])
    .with_x(28, 0x12341234);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_flags() {
    // subs x0, x0, #1  (sets NZCV)
    let tv = TestVector::new(&[0x00, 0x04, 0x00, 0xf1]).with_x(0, 0x1);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_movn_movk() {
    // movn x0, #0  => x0 = 0xFFFF_FFFF_FFFF_FFFF ; movk x0, #0x1234, lsl #16
    let tv = TestVector::new(&[
        0x00, 0x00, 0x80, 0x92, // movn x0, #0
        0x80, 0x46, 0xa2, 0xf2, // movk x0, #0x1234, lsl #16
    ]);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_add_shifted_reg() {
    // add x2, x0, x1, lsl #4
    let tv = TestVector::new(&[0x02, 0x10, 0x01, 0x8b])
        .with_x(0, 0x100)
        .with_x(1, 0xab);
    assert_matches_oracle(&tv);
}

// Register branches: the target register holds a mapped address (CODE_START),
// which the generic fuzzer can't arrange, so they're tested explicitly here.
const CODE_START: u64 = aarch64_difftest::CODE_START;

// Register branches jump back to CODE_START, so single-step (count=1) to avoid
// looping forever.
fn single_step(mut tv: TestVector) -> TestVector {
    tv.count = 1;
    tv
}

#[test]
fn matches_unicorn_br() {
    // br x5  => 0xd61f00a0 ; X5 = CODE_START
    let tv = single_step(TestVector::new(&[0xa0, 0x00, 0x1f, 0xd6]).with_x(5, CODE_START));
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_blr() {
    // blr x9  => 0xd63f0120 ; X9 = CODE_START, expect X30 = return address
    let tv = single_step(TestVector::new(&[0x20, 0x01, 0x3f, 0xd6]).with_x(9, CODE_START));
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_ret() {
    // ret x30  => 0xd65f03c0 ; X30 = CODE_START
    let tv = single_step(TestVector::new(&[0xc0, 0x03, 0x5f, 0xd6]).with_x(30, CODE_START));
    assert_matches_oracle(&tv);
}


// Exclusives: paired LDXR;STXR (success) and a lone STXR (failure), tested as
// short programs since they're stateful (exclusive monitor).
const DATA_BASE: u64 = aarch64_difftest::DATA_BASE;
const DATA_SIZE: usize = aarch64_difftest::DATA_SIZE;

fn excl_word(size: u32, is_load: bool, rs: u32, rn: u32, rt: u32) -> u32 {
    let l = if is_load { 1 } else { 0 };
    (size << 30) | (0b001000 << 24) | (l << 22) | (rs << 16) | (0b11111 << 10) | (rn << 5) | rt
}

fn excl_vector(size: u32) -> TestVector {
    let ldxr = excl_word(size, true, 31, 0, 1); // ldxr (w/x)1, [x0]
    let stxr = excl_word(size, false, 2, 0, 1); // stxr w2, (w/x)1, [x0]
    let mut code = ldxr.to_le_bytes().to_vec();
    code.extend(stxr.to_le_bytes());
    let mut tv = TestVector::new(&code);
    tv.init_x[0] = DATA_BASE; // aligned base
    tv.init_x[1] = 0x1122_3344_5566_7788;
    tv.init_data = Some(vec![0xa5u8; DATA_SIZE]);
    tv
}

#[test]
fn matches_unicorn_ldxr_stxr_w() {
    assert_matches_oracle(&excl_vector(2));
}

#[test]
fn matches_unicorn_ldxr_stxr_x() {
    assert_matches_oracle(&excl_vector(3));
}

#[test]
fn matches_unicorn_stxr_alone_fails() {
    // No preceding LDXR, so the monitor is clear: STXR must fail (Ws=1).
    let stxr = excl_word(3, false, 2, 0, 1);
    let mut tv = TestVector::new(&stxr.to_le_bytes());
    tv.init_x[0] = DATA_BASE;
    tv.init_x[1] = 0xdead_beef;
    tv.init_data = Some(vec![0u8; DATA_SIZE]);
    assert_matches_oracle(&tv);
}

// System registers: write then read TPIDR_EL0 and confirm the round-trip
// matches Unicorn (foundation of the system-mode model).
fn sysreg_word(read: bool, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) -> u32 {
    (0b1101010100 << 22)
        | ((read as u32) << 21)
        | (op0 << 19)
        | (op1 << 16)
        | (crn << 12)
        | (crm << 8)
        | (op2 << 5)
        | rt
}

#[test]
fn matches_unicorn_tpidr_el0_roundtrip() {
    // msr tpidr_el0, x0 ; mrs x1, tpidr_el0   (TPIDR_EL0 = 3,3,13,0,2)
    let msr = sysreg_word(false, 3, 3, 13, 0, 2, 0);
    let mrs = sysreg_word(true, 3, 3, 13, 0, 2, 1);
    let mut code = msr.to_le_bytes().to_vec();
    code.extend(mrs.to_le_bytes());
    let mut tv = TestVector::new(&code);
    tv.init_x[0] = 0xcafe_f00d_1234_5678;
    tv.count = 2;
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_vbar_el1_roundtrip() {
    // msr vbar_el1, x0 ; mrs x1, vbar_el1   (VBAR_EL1 = 3,0,12,0,0)
    let msr = sysreg_word(false, 3, 0, 12, 0, 0, 0);
    let mrs = sysreg_word(true, 3, 0, 12, 0, 0, 1);
    let mut code = msr.to_le_bytes().to_vec();
    code.extend(mrs.to_le_bytes());
    let mut tv = TestVector::new(&code);
    tv.init_x[0] = 0x0000_0000_0020_0800; // a plausible vector base
    tv.count = 2;
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_sp_el0_roundtrip() {
    // msr sp_el0, x0 ; mrs x1, sp_el0   (SP_EL0 = 3,0,4,1,0) — exercises SP banking
    let msr = sysreg_word(false, 3, 0, 4, 1, 0, 0);
    let mrs = sysreg_word(true, 3, 0, 4, 1, 0, 1);
    let mut code = msr.to_le_bytes().to_vec();
    code.extend(mrs.to_le_bytes());
    let mut tv = TestVector::new(&code);
    tv.init_x[0] = 0x1234_5678_9abc_def0;
    tv.count = 2;
    assert_matches_oracle(&tv);
}
