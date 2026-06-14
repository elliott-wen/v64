//! Condition-code evaluation tests.

use aarch64_cpu_state::Flags;
use aarch64_interp::eval_cond;

fn flags(n: bool, z: bool, c: bool, v: bool) -> Flags {
    Flags { n, z, c, v }
}

#[test]
fn eq_ne() {
    assert!(eval_cond(0b0000, flags(false, true, false, false))); // EQ, Z=1
    assert!(!eval_cond(0b0001, flags(false, true, false, false))); // NE, Z=1
}

#[test]
fn signed_ge_lt() {
    // N != V -> LT true, GE false
    let f = flags(true, false, false, false);
    assert!(!eval_cond(0b1010, f)); // GE
    assert!(eval_cond(0b1011, f)); // LT
}

#[test]
fn always() {
    assert!(eval_cond(0b1110, flags(false, false, false, false)));
    assert!(eval_cond(0b1111, flags(false, false, false, false)));
}
