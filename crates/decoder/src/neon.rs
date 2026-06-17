//! Router for the Advanced SIMD (vector) data-processing group (op0 = 0b0111).
//!
//! Class discriminators use the (value, mask) pairs from QEMU's decode table.
//! Implemented: three-same (int + FP), two-reg-misc, modified-immediate, copy,
//! permute (ZIP/UZP/TRN), and extract (EXT). Others -> `Unsupported`.

use crate::bits::field;
use crate::insn::Insn;

mod aes;
mod across;
mod copy;
mod ext;
mod indexed;
mod mod_imm;
pub(crate) mod scalar; // reached from the top-level decoder for scalar SIMD
mod sha;
mod shift_imm;
mod tbl;
mod three_diff;
mod three_same;
mod three_same_extra;
mod three_same_fp;
mod two_reg_misc;
mod zip_trn;

fn matches(word: u32, mask: u32, value: u32) -> bool {
    word & mask == value
}

pub(crate) fn decode(word: u32) -> Insn {
    // bits[28:24]=01111 covers both modified-immediate (immh==0) and
    // shift-by-immediate (immh!=0).
    if field(word, 24, 5) == 0b01111 {
        // Indexed-element ops share this block but have bit10 = 0.
        if field(word, 10, 1) == 0 {
            return indexed::decode(word);
        }
        return if field(word, 19, 4) == 0 {
            mod_imm::decode(word)
        } else {
            shift_imm::decode(word)
        };
    }
    if matches(word, 0xbf20_8c00, 0x0e00_0800) {
        return zip_trn::decode(word);
    }
    if matches(word, 0xbf20_8c00, 0x0e00_0000) {
        return tbl::decode(word);
    }
    if matches(word, 0xff3e_0c00, 0x4e28_0800) {
        return aes::decode(word);
    }
    // (mask, value) from QEMU's table, value pre-masked (bit29=U is outside the mask).
    if matches(word, 0x9f20_c400, 0x0e00_8400) {
        return three_same_extra::decode(word);
    }
    if matches(word, 0xbf20_8400, 0x2e00_0000) {
        return ext::decode(word);
    }
    if matches(word, 0x9f3e_0c00, 0x0e30_0800) {
        return across::decode(word);
    }
    if matches(word, 0x9f3e_0c00, 0x0e20_0800) {
        return two_reg_misc::decode(word);
    }
    if matches(word, 0x9fe0_8400, 0x0e00_0400) {
        return copy::decode(word);
    }
    // Three-different: bits[28:24]=01110, bit21=1, bits[11:10]=00.
    if field(word, 24, 5) == 0b01110 && field(word, 21, 1) == 1 && field(word, 10, 2) == 0 {
        return three_diff::decode(word);
    }
    // Three-same (integer + FP): bits[28:24]=01110, bit21=1, bit10=1.
    if field(word, 24, 5) == 0b01110 && field(word, 21, 1) == 1 && field(word, 10, 1) == 1 {
        if field(word, 11, 5) >= 0b11000 {
            return three_same_fp::decode(word);
        }
        return three_same::decode(word);
    }
    Insn::Unsupported { word }
}
