//! Small shared bit-field helpers for the decoders.

/// The `sf` bit (bit 31): set means 64-bit operands.
#[must_use]
pub fn sf(word: u32) -> bool {
    (word >> 31) & 1 == 1
}

/// Extract `len` bits starting at `lsb` (zero-extended).
#[must_use]
pub fn field(word: u32, lsb: u32, len: u32) -> u32 {
    (word >> lsb) & ((1 << len) - 1)
}

/// Extract `len` bits starting at `lsb` and sign-extend to i64.
#[must_use]
pub fn sfield(word: u32, lsb: u32, len: u32) -> i64 {
    let v = field(word, lsb, len) as i64;
    let shift = 64 - len;
    (v << shift) >> shift
}
