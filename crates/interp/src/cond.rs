//! AArch64 condition-code evaluation (the 4-bit `cond` field -> bool).
//!
//! Mirrors `arm_test_cc` in `unicorn/qemu/target/arm/translate.c`.

use aarch64_cpu_state::Flags;

/// Evaluate a 4-bit condition against the current flags.
#[must_use]
pub fn eval_cond(cond: u8, f: Flags) -> bool {
    let base = match cond >> 1 {
        0 => f.z,                  // EQ / NE      : Z
        1 => f.c,                  // CS / CC      : C
        2 => f.n,                  // MI / PL      : N
        3 => f.v,                  // VS / VC      : V
        4 => f.c && !f.z,          // HI / LS      : C && !Z
        5 => f.n == f.v,           // GE / LT      : N == V
        6 => !f.z && (f.n == f.v), // GT / LE      : !Z && N == V
        _ => true,                 // AL (14/15)   : always
    };
    // Odd condition codes invert the base test (but not AL).
    if cond & 1 == 1 && cond != 0b1111 {
        !base
    } else {
        base
    }
}
