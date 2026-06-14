//! ALU primitive tests (AddWithCarry and shifts).

use aarch64_decoder::ShiftType;
use aarch64_interp::{add_with_carry, apply_shift};

#[test]
fn sub_equal_sets_carry_and_zero() {
    // 1 - 1 = 0, no borrow => C=1, Z=1
    let (res, f) = add_with_carry(1, 1, true, true);
    assert_eq!(res, 0);
    assert!(f.z && f.c && !f.n && !f.v);
}

#[test]
fn asr_w_sign_extends() {
    // 0x8000_0000 asr 4, 32-bit => 0xF800_0000
    assert_eq!(apply_shift(0x8000_0000, ShiftType::Asr, 4, false), 0xF800_0000);
}
