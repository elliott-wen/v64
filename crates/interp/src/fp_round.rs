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

pub(crate) fn round_f32(x: f32, mode: Mode) -> f32 {
    if !x.is_finite() {
        return x;
    }
    match mode {
        Mode::Floor => x.floor(),
        Mode::Ceil => x.ceil(),
        Mode::Trunc => x.trunc(),
        Mode::Away => keep_sign_f32(x.round(), x),
        Mode::Near => round_even_f32(x),
    }
}

pub(crate) fn round_f64(x: f64, mode: Mode) -> f64 {
    if !x.is_finite() {
        return x;
    }
    match mode {
        Mode::Floor => x.floor(),
        Mode::Ceil => x.ceil(),
        Mode::Trunc => x.trunc(),
        Mode::Away => keep_sign_f64(x.round(), x),
        Mode::Near => round_even_f64(x),
    }
}

fn round_even_f32(x: f32) -> f32 {
    let t = x.trunc();
    let diff = x - t;
    let r = if diff.abs() < 0.5 {
        t
    } else if diff.abs() > 0.5 {
        t + diff.signum()
    } else if (t / 2.0).trunc() * 2.0 == t {
        t
    } else {
        t + diff.signum()
    };
    keep_sign_f32(r, x)
}

fn round_even_f64(x: f64) -> f64 {
    let t = x.trunc();
    let diff = x - t;
    let r = if diff.abs() < 0.5 {
        t
    } else if diff.abs() > 0.5 {
        t + diff.signum()
    } else if (t / 2.0).trunc() * 2.0 == t {
        t
    } else {
        t + diff.signum()
    };
    keep_sign_f64(r, x)
}

/// A zero result keeps the sign of the input (e.g. trunc(-0.3) = -0.0).
fn keep_sign_f32(r: f32, x: f32) -> f32 {
    if r == 0.0 && x.is_sign_negative() {
        -0.0
    } else {
        r
    }
}
fn keep_sign_f64(r: f64, x: f64) -> f64 {
    if r == 0.0 && x.is_sign_negative() {
        -0.0
    } else {
        r
    }
}
