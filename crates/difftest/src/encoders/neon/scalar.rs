//! Scalar-SIMD encoders (the `...1110_11110...` scalar forms): three-same,
//! two-reg-misc, pairwise, three-different, copy, by-element, and shift.

use super::{enc, fp_enc, FPCR_DN};
use crate::encoders::random_v;
use crate::rng::Rng;
use crate::FpEncoded;

pub(super) fn scalar_three_same(rng: &mut Rng) -> FpEncoded {
    if rng.below(2) == 0 {
        let fps = [0x1bu32, 0x1f, 0x3f, 0x5d, 0x7d, 0x1c, 0x5c, 0x7c, 0x7a];
        let fp = fps[rng.below(fps.len() as u32) as usize];
        let (u, b5, op5) = ((fp >> 6) & 1, (fp >> 5) & 1, fp & 0x1f);
        let size = (b5 << 1) | rng.below(2);
        let word = sts_word(u, size, op5, rng);
        return fp_enc(word, rng);
    }
    // (opcode, size-rule): 0=any, 1=size3, 2=size 1|2.
    let table: &[(u32, u32)] = &[
        (1, 0), (5, 0), (9, 0), (0xb, 0),
        (8, 1), (0xa, 1), (6, 1), (7, 1), (0x11, 1), (0x10, 1),
        (0x16, 2),
    ];
    let (op, rule) = table[rng.below(table.len() as u32) as usize];
    let u = rng.below(2);
    let size = match rule {
        1 => 3,
        2 => 1 + rng.below(2),
        _ => rng.below(4),
    };
    let word = sts_word(u, size, op, rng);
    enc(word, rng)
}

fn sts_word(u: u32, size: u32, opcode: u32, rng: &mut Rng) -> u32 {
    (1 << 30)
        | (u << 29)
        | (0b11110 << 24)
        | (size << 22)
        | (1 << 21)
        | (rng.bits(5) << 16)
        | (opcode << 11)
        | (1 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5)
}

pub(super) fn scalar_two_reg_misc(rng: &mut Rng) -> FpEncoded {
    if rng.below(2) == 0 {
        let fps = [
            0x2cu32, 0x2d, 0x2e, 0x6c, 0x6d, 0x1d, 0x5d, 0x1a, 0x1b, 0x3a, 0x3b, 0x5a, 0x5b, 0x7a,
            0x7b, 0x1c, 0x5c,
        ];
        let fp = fps[rng.below(fps.len() as u32) as usize];
        let (u, b5, op5) = ((fp >> 6) & 1, (fp >> 5) & 1, fp & 0x1f);
        let size = (b5 << 1) | rng.below(2);
        let word = strm_word(u, size, op5, rng);
        return fp_enc(word, rng);
    }
    // (opcode, u-fixed (2=either), size-rule)
    let table: &[(u32, u32, u32)] = &[
        (0x3, 2, 0), (0x7, 2, 0),
        (0x8, 2, 1), (0x9, 2, 1), (0xa, 0, 1), (0xb, 2, 1),
        (0x12, 1, 2), (0x14, 2, 2),
    ];
    let (op, ufix, rule) = table[rng.below(table.len() as u32) as usize];
    let u = if ufix == 2 { rng.below(2) } else { ufix };
    let size = match rule {
        1 => 3,
        2 => rng.below(3),
        _ => rng.below(4),
    };
    let word = strm_word(u, size, op, rng);
    enc(word, rng)
}

fn strm_word(u: u32, size: u32, opcode: u32, rng: &mut Rng) -> u32 {
    (1 << 30)
        | (u << 29)
        | (0b11110 << 24)
        | (size << 22)
        | (0b10000 << 17)
        | (opcode << 12)
        | (0b10 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5)
}

pub(super) fn scalar_pairwise(rng: &mut Rng) -> FpEncoded {
    if rng.below(2) == 0 {
        // ADDP: u=0, size=3, opcode5=0x1b.
        let word = sp_word(0, 3, 0x1b, rng);
        return enc(word, rng);
    }
    let fulls = [0xcu32, 0xd, 0xf, 0x2c, 0x2f];
    let full = fulls[rng.below(fulls.len() as u32) as usize];
    let (b5, op5) = ((full >> 5) & 1, full & 0x1f);
    let size = (b5 << 1) | rng.below(2);
    let word = sp_word(1, size, op5, rng);
    fp_enc(word, rng)
}

fn sp_word(u: u32, size: u32, opcode: u32, rng: &mut Rng) -> u32 {
    (1 << 30)
        | (u << 29)
        | (0b11110 << 24)
        | (size << 22)
        | (0b11000 << 17)
        | (opcode << 12)
        | (0b10 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5)
}

pub(super) fn scalar_three_diff(rng: &mut Rng) -> FpEncoded {
    let op = [0x9u32, 0xb, 0xd][rng.below(3) as usize];
    let size = 1 + rng.below(2);
    let word = (1 << 30)
        | (0b11110 << 24)
        | (size << 22)
        | (1 << 21)
        | (rng.bits(5) << 16)
        | (op << 12)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    enc(word, rng)
}

pub(super) fn scalar_copy(rng: &mut Rng) -> FpEncoded {
    let size = rng.below(4);
    let index = rng.below(16 >> size);
    let imm5 = (index << (size + 1)) | (1 << size);
    let word = (1 << 30)
        | (0b11110 << 24)
        | (imm5 << 16)
        | (1 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    enc(word, rng)
}

pub(super) fn scalar_indexed(rng: &mut Rng) -> FpEncoded {
    let ops: &[(u32, u32, bool)] = &[
        (0, 0b0011, false), // SQDMLAL
        (0, 0b0111, false), // SQDMLSL
        (0, 0b1011, false), // SQDMULL
        (0, 0b1100, false), // SQDMULH
        (0, 0b1101, false), // SQRDMULH
        (0, 0b0001, true),  // FMLA
        (0, 0b0101, true),  // FMLS
        (0, 0b1001, true),  // FMUL
        (1, 0b1001, true),  // FMULX
    ];
    let (u, opcode, is_fp) = ops[rng.below(ops.len() as u32) as usize];
    let (size, h, l, m, rm4) = if is_fp {
        if rng.below(2) == 0 {
            let idx = rng.below(4);
            let rm5 = rng.bits(5);
            (2u32, idx >> 1, idx & 1, rm5 >> 4, rm5 & 0xf)
        } else {
            let idx = rng.below(2);
            let rm5 = rng.bits(5);
            (3u32, idx, 0, rm5 >> 4, rm5 & 0xf)
        }
    } else if rng.below(2) == 0 {
        let idx = rng.below(8);
        (1u32, idx >> 2, (idx >> 1) & 1, idx & 1, rng.bits(4))
    } else {
        let idx = rng.below(4);
        let rm5 = rng.bits(5);
        (2u32, idx >> 1, idx & 1, rm5 >> 4, rm5 & 0xf)
    };
    let word = (1 << 30)
        | (u << 29)
        | (0b11111 << 24)
        | (size << 22)
        | (l << 21)
        | (m << 20)
        | (rm4 << 16)
        | (opcode << 12)
        | (h << 11)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    FpEncoded { word, init_v: random_v(rng), gpr_seeds: vec![], fpcr: if is_fp { FPCR_DN } else { 0 } }
}

pub(super) fn scalar_shift(rng: &mut Rng) -> FpEncoded {
    // (opcode, u-fixed (2=either), size-kind): 0=D-only,1=any,2=narrow(<=2),3=conv(2|3)
    let table: &[(u32, u32, u32)] = &[
        (0b00000, 2, 0), (0b00010, 2, 0), (0b00100, 2, 0), (0b00110, 2, 0),
        (0b01000, 1, 0), (0b01010, 2, 0),
        (0b01100, 1, 1), (0b01110, 2, 1),
        (0b10000, 1, 2), (0b10001, 1, 2), (0b10010, 2, 2), (0b10011, 2, 2),
        (0b11100, 2, 3), (0b11111, 2, 3),
    ];
    let (opcode, ufix, kind) = table[rng.below(table.len() as u32) as usize];
    let u = if ufix == 2 { rng.below(2) } else { ufix };
    let (immh, immb) = match kind {
        0 => (0b1000 | rng.below(8), rng.below(8)),     // D-only (size 3)
        3 => {
            let size = 2 + rng.below(2);
            ((1 << size) | rng.below(1 << size), rng.below(8))
        }
        2 => {
            let size = rng.below(3);
            ((1 << size) | rng.below(1 << size), rng.below(8))
        }
        _ => {
            let size = rng.below(4);
            ((1 << size) | rng.below(1 << size), rng.below(8))
        }
    };
    let word = (1 << 30)
        | (u << 29)
        | (0b11111 << 24)
        | (immh << 19)
        | (immb << 16)
        | (opcode << 11)
        | (1 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    let mut e = enc(word, rng);
    if matches!(opcode, 0b11100 | 0b11111) {
        e.fpcr = FPCR_DN;
    }
    e
}
