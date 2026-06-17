//! Shared floating-point round-to-integral helpers (used by FRINT* and the
//! rounding FCVT conversions, scalar and vector).

#[derive(Clone, Copy)]
pub(crate) enum Mode {
    Near, // round to nearest, ties to even
    Floor,
    Ceil,
    Trunc,
    Away, // round to nearest, ties away from zero
}

macro_rules! round_impl {
    ($name:ident, $t:ty) => {
        /// Round `x` to an integral value per `mode`. Non-finite values pass
        /// through; a zero result keeps the input's sign (e.g. trunc(-0.3) = -0.0).
        pub(crate) fn $name(x: $t, mode: Mode) -> $t {
            if !x.is_finite() {
                return x;
            }
            let r = match mode {
                Mode::Floor => x.floor(),
                Mode::Ceil => x.ceil(),
                Mode::Trunc => x.trunc(),
                Mode::Away => x.round(),           // ties away from zero
                Mode::Near => x.round_ties_even(), // ties to even
            };
            if r == 0.0 && x.is_sign_negative() {
                -0.0
            } else {
                r
            }
        }
    };
}
round_impl!(round_f32, f32);
round_impl!(round_f64, f64);
