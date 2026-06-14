//! ARM `DecodeBitMasks`: resolve (N, imms, immr) into the work/tail masks used
//! by logical-immediate and bitfield instructions.
//!
//! Mirrors `logic_imm_decode_wmask` in
//! `unicorn/qemu/target/arm/translate-a64.c` (which computes only `wmask`) and
//! the full ARM ARM `DecodeBitMasks` pseudocode (which also yields `tmask`,
//! needed by SBFM/UBFM/BFM).

/// `len`-bit run of ones, for `1 <= len <= 64`.
fn ones(len: u32) -> u64 {
    if len >= 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

/// Rotate the low `esize` bits of `val` right by `r` (0 <= r < esize).
fn ror_within(val: u64, r: u32, esize: u32) -> u64 {
    if r == 0 {
        return val;
    }
    let mask = ones(esize);
    ((val >> r) | (val << (esize - r))) & mask
}

/// Replicate an `esize`-bit element across the full 64-bit value.
fn replicate(mut elem: u64, esize: u32) -> u64 {
    let mut e = esize;
    while e < 64 {
        elem |= elem << e;
        e *= 2;
    }
    elem
}

/// Returns `(wmask, tmask)`, or `None` for reserved/unallocated encodings.
///
/// `immediate` is true for logical-immediate instructions (where an all-ones
/// run length is reserved) and false for bitfield instructions.
#[must_use]
pub fn decode_bit_masks(n: u32, imms: u32, immr: u32, immediate: bool) -> Option<(u64, u64)> {
    // Element size: len = highest set bit of (N : ~imms<5:0>), over 7 bits.
    let x = (n << 6) | ((!imms) & 0x3f);
    if x == 0 {
        return None;
    }
    let len = 31 - x.leading_zeros() as i32;
    if len < 1 {
        return None;
    }
    let esize = 1u32 << len;
    let levels = esize - 1; // `len` low bits set

    let s = imms & levels;
    let r = immr & levels;

    // For logical immediates a run length of all-ones is reserved.
    if immediate && s == levels {
        return None;
    }

    let diff = s.wrapping_sub(r) & levels;

    let welem = ones(s + 1);
    let telem = ones(diff + 1);

    let wmask = replicate(ror_within(welem, r, esize), esize);
    let tmask = replicate(telem, esize);
    Some((wmask, tmask))
}
