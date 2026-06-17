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

#[test]
fn matches_unicorn_ldapr() {
    // ldapr x1, [x0]  (0xf8bfc001) — load-acquire RCpc; in our model a plain load
    let mut tv = TestVector::new(&0xf8bf_c001u32.to_le_bytes());
    tv.init_x[0] = DATA_BASE;
    let mut data = vec![0u8; DATA_SIZE];
    data[..8].copy_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
    tv.init_data = Some(data);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_ldxp_stxp_x() {
    // ldxp x1,x2,[x0] ; stxp w3,x1,x2,[x0]  — 64-bit exclusive pair round-trip
    let ldxp = 0xc87f_0801u32; // ldxp x1, x2, [x0]
    let stxp = 0xc823_0801u32; // stxp w3, x1, x2, [x0]
    let mut code = ldxp.to_le_bytes().to_vec();
    code.extend(stxp.to_le_bytes());
    let mut tv = TestVector::new(&code);
    tv.init_x[0] = DATA_BASE;
    tv.init_data = Some(vec![0x5au8; DATA_SIZE]);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_casp_success() {
    // casp w0,w1,w2,w3,[x4]  (0x08207c82): x0:x1 match memory, so it swaps in x2:x3
    let mut tv = TestVector::new(&0x0820_7c82u32.to_le_bytes());
    tv.init_x[4] = DATA_BASE;
    // Memory holds two words {0x11111111, 0x22222222}; the compare pair matches.
    tv.init_x[0] = 0x1111_1111;
    tv.init_x[1] = 0x2222_2222;
    tv.init_x[2] = 0xaaaa_aaaa;
    tv.init_x[3] = 0xbbbb_bbbb;
    let mut data = vec![0u8; DATA_SIZE];
    data[..4].copy_from_slice(&0x1111_1111u32.to_le_bytes());
    data[4..8].copy_from_slice(&0x2222_2222u32.to_le_bytes());
    tv.init_data = Some(data);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_casp_fail() {
    // casp with a non-matching compare pair: memory is left unchanged, x0:x1 get
    // the (unchanged) old memory values.
    let mut tv = TestVector::new(&0x0820_7c82u32.to_le_bytes());
    tv.init_x[4] = DATA_BASE;
    tv.init_x[0] = 0xdead_beef; // does not match memory
    tv.init_x[1] = 0xfeed_face;
    tv.init_x[2] = 0xaaaa_aaaa;
    tv.init_x[3] = 0xbbbb_bbbb;
    let mut data = vec![0u8; DATA_SIZE];
    data[..4].copy_from_slice(&0x1111_1111u32.to_le_bytes());
    data[4..8].copy_from_slice(&0x2222_2222u32.to_le_bytes());
    tv.init_data = Some(data);
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

#[test]
fn matches_unicorn_fmov_high_half() {
    // fmov x0, v3.d[1] : high 64 bits of V3 -> X0
    let mut tv = TestVector::new(&0x9eae_0060u32.to_le_bytes());
    let mut v = [0u128; 32];
    v[3] = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00;
    tv.init_v = Some(v);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_fmov_to_high_half() {
    // fmov v0.d[1], x3 : X3 -> high 64 of V0, low 64 preserved
    let mut tv = TestVector::new(&0x9eaf_0060u32.to_le_bytes()).with_x(3, 0xcafe_f00d_1234_5678);
    let mut v = [0u128; 32];
    v[0] = 0x0000_0000_0000_0000_dead_beef_dead_beef; // low half should survive
    tv.init_v = Some(v);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_scvtf_fixed() {
    // scvtf d0, w0, #2 : W0=5 as fixed (2 frac bits) -> 1.25
    let mut tv = TestVector::new(&0x1e42_f800u32.to_le_bytes()).with_x(0, 5);
    tv.init_v = Some([0u128; 32]);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_ucvtf_fixed_single() {
    // ucvtf s0, w0, #4 : W0=20 as fixed (4 frac bits) -> 1.25
    let mut tv = TestVector::new(&0x1e03_f000u32.to_le_bytes()).with_x(0, 20);
    tv.init_v = Some([0u128; 32]);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_fcvtzs_fixed() {
    // fcvtzs w0, d0, #2 : D0=1.25 -> *4 -> 5
    let mut tv = TestVector::new(&0x1e58_f800u32.to_le_bytes());
    let mut v = [0u128; 32];
    v[0] = u128::from(1.25f64.to_bits());
    tv.init_v = Some(v);
    assert_matches_oracle(&tv);
}

#[test]
fn matches_unicorn_fcvtzu_fixed_x() {
    // fcvtzu x0, d0, #3 : D0=2.5 -> *8 -> 20
    let mut tv = TestVector::new(&0x9e59_f400u32.to_le_bytes());
    let mut v = [0u128; 32];
    v[0] = u128::from(2.5f64.to_bits());
    tv.init_v = Some(v);
    assert_matches_oracle(&tv);
}
