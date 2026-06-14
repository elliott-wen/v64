//! Arithmetic primitives shared by the executor: shifts and `AddWithCarry`.
//!
//! Flag semantics follow the ARM ARM pseudocode, cross-checked against the
//! reference helpers in `unicorn/qemu/target/arm/`.

use aarch64_cpu_state::Flags;
use aarch64_decoder::ShiftType;

/// Apply a logical/arithmetic shift to a register operand at the given width.
#[must_use]
pub fn apply_shift(val: u64, shift: ShiftType, amount: u8, sf: bool) -> u64 {
    let width = if sf { 64 } else { 32 };
    let amount = u32::from(amount) % width;
    let v = if sf { val } else { val & 0xffff_ffff };
    let res = match shift {
        ShiftType::Lsl => v.wrapping_shl(amount),
        ShiftType::Lsr => {
            if sf { v >> amount } else { (v & 0xffff_ffff) >> amount }
        }
        ShiftType::Asr => {
            if sf {
                ((v as i64) >> amount) as u64
            } else {
                (((v as u32) as i32) >> amount) as u32 as u64
            }
        }
        ShiftType::Ror => {
            if sf {
                v.rotate_right(amount)
            } else {
                u64::from((v as u32).rotate_right(amount))
            }
        }
    };
    if sf { res } else { res & 0xffff_ffff }
}

/// ARM `AddWithCarry`: returns (result, NZCV). `sub` computes `a - b` as
/// `a + !b + 1`. Operates at the operand width given by `sf`.
#[must_use]
pub fn add_with_carry(a: u64, b: u64, sub: bool, sf: bool) -> (u64, Flags) {
    // Subtraction is add of the ones'-complement with carry-in 1.
    let (b_op, carry_in) = if sub { (!b, 1) } else { (b, 0) };
    add_with_carry_in(a, b_op, carry_in, sf)
}

/// ARM `AddWithCarry` with an explicit carry-in (used by ADC/SBC, which take
/// the PSTATE C flag). `b` is the operand *after* any complement.
#[must_use]
pub fn add_with_carry_in(a: u64, b: u64, carry_in: u64, sf: bool) -> (u64, Flags) {
    if sf {
        let (s1, c1) = a.overflowing_add(b);
        let (s, c2) = s1.overflowing_add(carry_in);
        let carry = c1 || c2;
        let overflow = (((a ^ s) & (b ^ s)) >> 63) & 1 == 1;
        (s, flags_from(s, 64, carry, overflow))
    } else {
        let a = a & 0xffff_ffff;
        let b = b & 0xffff_ffff;
        let wide = a + b + carry_in; // fits in u64, no wrap at 33 bits
        let s = wide & 0xffff_ffff;
        let carry = (wide >> 32) & 1 == 1;
        let overflow = (((a ^ s) & (b ^ s)) >> 31) & 1 == 1;
        (s, flags_from(s, 32, carry, overflow))
    }
}

fn flags_from(result: u64, width: u32, carry: bool, overflow: bool) -> Flags {
    let sign_bit = 1u64 << (width - 1);
    Flags {
        n: result & sign_bit != 0,
        z: (result & ((1u128 << width) - 1) as u64) == 0,
        c: carry,
        v: overflow,
    }
}
