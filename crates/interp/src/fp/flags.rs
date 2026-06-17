//! FPSR cumulative (sticky) floating-point exception flags.
//!
//! These bits are *observational*: no FP result depends on them, and we don't
//! model trapped FP exceptions (the `FPCR` trap-enable bits are treated as RES0,
//! a valid ARMv8 configuration), so only an explicit `MRS FPSR` / `fetestexcept`
//! ever reads them. We therefore compute them cheaply:
//!
//! * IOC / DZC / OFC — exact (a few native-float predicates).
//! * IXC — exact, via error-free transforms (TwoSum for +/-, the FMA residual
//!   for * / / / sqrt): the rounding error is nonzero iff the result was rounded.
//! * UFC — best-effort: result subnormal && inexact (after-rounding tininess),
//!   which differs from ARM's before-rounding rule only at a vanishing boundary.
//!
//! Conversions use simpler range/integrality checks. None of this needs a
//! 128-bit type or a soft-float library.

use aarch64_cpu_state::CpuState;

pub(crate) const IOC: u64 = 1 << 0; // Invalid Operation
pub(crate) const DZC: u64 = 1 << 1; // Divide by Zero
pub(crate) const OFC: u64 = 1 << 2; // Overflow
pub(crate) const UFC: u64 = 1 << 3; // Underflow
pub(crate) const IXC: u64 = 1 << 4; // Inexact

#[inline]
pub(crate) fn raise(cpu: &mut CpuState, bits: u64) {
    cpu.fpsr |= bits;
}

/// The two binary-arithmetic operations whose flags we model precisely.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// Float operations the flag logic needs, abstracted over `f32`/`f64`.
pub(crate) trait Fp: Copy + PartialEq {
    const ZERO: Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_subnormal(self) -> bool;
    fn is_snan(self) -> bool;
    fn neg(self) -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul_add(self, b: Self, c: Self) -> Self;
}

macro_rules! impl_fp {
    ($t:ty, $snan_bit:expr) => {
        impl Fp for $t {
            const ZERO: Self = 0.0;
            #[inline] fn is_nan(self) -> bool { <$t>::is_nan(self) }
            #[inline] fn is_finite(self) -> bool { <$t>::is_finite(self) }
            #[inline] fn is_infinite(self) -> bool { <$t>::is_infinite(self) }
            #[inline] fn is_subnormal(self) -> bool { <$t>::is_subnormal(self) }
            #[inline] fn is_snan(self) -> bool { self.is_nan() && self.to_bits() & $snan_bit == 0 }
            #[inline] fn neg(self) -> Self { -self }
            #[inline] fn add(self, o: Self) -> Self { self + o }
            #[inline] fn sub(self, o: Self) -> Self { self - o }
            #[inline] fn mul_add(self, b: Self, c: Self) -> Self { <$t>::mul_add(self, b, c) }
        }
    };
}
impl_fp!(f32, 0x0040_0000);
impl_fp!(f64, 0x0008_0000_0000_0000);

/// `true` iff `s == a + b` exactly (TwoSum: the rounding error term is zero).
fn add_is_exact<T: Fp>(a: T, b: T, s: T) -> bool {
    let bv = s.sub(a);
    let av = s.sub(bv);
    a.sub(av).add(b.sub(bv)) == T::ZERO
}

/// Compute and raise the FPSR flags for a binary arithmetic op `a OP b = r`.
pub(crate) fn binop<T: Fp>(cpu: &mut CpuState, op: Op, a: T, b: T, r: T) {
    let mut f = 0u64;
    // Invalid: a signaling-NaN operand, or a NaN produced from non-NaN inputs
    // (0/0, inf-inf, 0*inf, inf/inf).
    if a.is_snan() || b.is_snan() || (r.is_nan() && !a.is_nan() && !b.is_nan()) {
        f |= IOC;
    }
    let div_by_zero = op == Op::Div && b == T::ZERO;
    if div_by_zero && a.is_finite() && a != T::ZERO {
        f |= DZC; // finite-nonzero / 0
    }
    if r.is_infinite() && a.is_finite() && b.is_finite() && !div_by_zero {
        f |= OFC | IXC; // overflow from finite operands is always inexact
    } else if r.is_finite() && f & IOC == 0 {
        let exact = match op {
            Op::Add => add_is_exact(a, b, r),
            Op::Sub => add_is_exact(a, b.neg(), r), // a - b = a + (-b)
            Op::Mul => a.mul_add(b, r.neg()) == T::ZERO, // r + (a*b - r) ; residual == 0
            Op::Div => r.neg().mul_add(b, a) == T::ZERO, // a - r*b == 0
        };
        if !exact {
            f |= IXC;
            if r != T::ZERO && r.is_subnormal() {
                f |= UFC;
            }
        }
    }
    raise(cpu, f);
}

/// FSQRT flags: invalid for sqrt of a negative (NaN result) or SNaN; inexact via
/// the FMA residual `r*r - a`.
pub(crate) fn sqrt<T: Fp>(cpu: &mut CpuState, a: T, r: T) {
    let mut f = 0u64;
    if a.is_snan() || (r.is_nan() && !a.is_nan()) {
        f |= IOC;
    } else if r.is_finite() && r.mul_add(r, a.neg()) != T::ZERO {
        f |= IXC;
    }
    raise(cpu, f);
}

/// FP -> integer conversion flags (FCVT*, fixed-point FCVTZS/U). `lo`/`hi` are the
/// inclusive-low / exclusive-high bounds of the integer type as `f64`. Out of
/// range or NaN raises Invalid (the value saturates); an in-range value with a
/// fractional part raises Inexact.
pub(crate) fn f2i(cpu: &mut CpuState, src: f64, lo: f64, hi: f64, integral: bool) {
    let mut f = 0u64;
    if src.is_nan() || src < lo || src >= hi {
        f |= IOC;
    } else if !integral {
        f |= IXC;
    }
    raise(cpu, f);
}

/// Integer -> FP conversion flags: inexact when the integer wasn't exactly
/// representable (`exact` is the caller's round-trip check).
pub(crate) fn i2f(cpu: &mut CpuState, exact: bool) {
    if !exact {
        raise(cpu, IXC);
    }
}
