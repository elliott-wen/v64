//! Integer single-register and pair load/store encoders: unsigned-immediate,
//! unscaled (LDUR), pre/post-index, register-offset, PC-literal, and LDP/STP.

use super::{base_near_mid, random_data, rt_distinct};
use crate::rng::Rng;
use crate::{MemEncoded, CODE_START, DATA_BASE, DATA_SIZE};

/// Pick a valid `opc` for the given access `size` (excludes PRFM and the
/// reserved word/dword signed forms, which we don't implement).
fn valid_opc(rng: &mut Rng, size: u32) -> u32 {
    let choices: &[u32] = match size {
        0 | 1 => &[0, 1, 2, 3],
        2 => &[0, 1, 2],
        _ => &[0, 1],
    };
    choices[rng.below(choices.len() as u32) as usize]
}

/// A raw 9-bit signed-immediate field (`-256..=255`).
fn imm9_field(rng: &mut Rng) -> u32 {
    let v = rng.below(512) as i64 - 256;
    (v as u32) & 0x1ff
}

/// Shared body for the imm9 addressing classes; `op1110` selects the form.
fn imm9_form(rng: &mut Rng, op1110: u32, indexed: bool) -> MemEncoded {
    let size = rng.below(4);
    let opc = valid_opc(rng, size);
    let imm_field = imm9_field(rng);
    let rn = rng.below(31);
    let rt = if indexed { rt_distinct(rng, rn) } else { rng.below(32) };

    let word = (size << 30)
        | (0b111 << 27)
        | (opc << 22)
        | (imm_field << 12)
        | (op1110 << 10)
        | (rn << 5)
        | rt;

    MemEncoded {
        init_v: Vec::new(),
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x400))],
        data: random_data(rng),
    }
}

pub(super) fn ldst_unscaled(rng: &mut Rng) -> MemEncoded {
    imm9_form(rng, 0b00, false)
}

pub(super) fn ldst_post(rng: &mut Rng) -> MemEncoded {
    imm9_form(rng, 0b01, true)
}

pub(super) fn ldst_pre(rng: &mut Rng) -> MemEncoded {
    imm9_form(rng, 0b11, true)
}

pub(super) fn ldst_reg(rng: &mut Rng) -> MemEncoded {
    let size = rng.below(4);
    let opc = valid_opc(rng, size);
    let option = [0b010u32, 0b011, 0b110, 0b111][rng.below(4) as usize];
    let s = rng.below(2);
    let rn = rng.below(31);
    let rm = rt_distinct(rng, rn); // distinct base/index so their seeds don't collide
    let rt = rng.below(32);

    let word = (size << 30)
        | (0b111 << 27)
        | (1 << 21)
        | (opc << 22)
        | (rm << 16)
        | (option << 13)
        | (s << 12)
        | (0b10 << 10)
        | (rn << 5)
        | rt;

    // Seed base near center and the index to a small value so the access stays
    // mapped regardless of extend/scale.
    MemEncoded {
        init_v: Vec::new(),
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x400)), (rm as u8, u64::from(rng.below(64)))],
        data: random_data(rng),
    }
}

pub(super) fn ldst_uimm(rng: &mut Rng) -> MemEncoded {
    let size = rng.below(4);
    let opc = valid_opc(rng, size);
    let imm12 = rng.below(256); // keep scaled offset inside the DATA region
    let rn = rng.below(31); // base 0..30 (SP base handled by other tests)
    let rt = rng.below(32);

    let word = (size << 30)
        | (0b111 << 27)
        | (0b01 << 24)
        | (opc << 22)
        | (imm12 << 10)
        | (rn << 5)
        | rt;

    // Seed the base so base + (imm12 << size) lands inside [DATA_BASE, +SIZE).
    let offset = imm12 << size;
    let slack = DATA_SIZE as u32 - offset - 8;
    let base_off = rng.below(slack) & !7; // 8-aligned
    let base_addr = DATA_BASE + u64::from(base_off);

    MemEncoded { init_v: Vec::new(), word, seeds: vec![(rn as u8, base_addr)], data: random_data(rng) }
}

pub(super) fn ldst_literal(rng: &mut Rng) -> MemEncoded {
    let opc = rng.below(3); // LDR(W) / LDR(X) / LDRSW
    let rt = rng.below(32);
    // Point the literal at a 4-aligned slot inside the (seeded) DATA region.
    let target = DATA_BASE + u64::from(rng.below(DATA_SIZE as u32 - 8) & !3);
    let off = target as i64 - CODE_START as i64;
    let imm19 = ((off >> 2) as u32) & 0x7_ffff;
    let word = (opc << 30) | (0b011 << 27) | (imm19 << 5) | rt;
    MemEncoded { init_v: Vec::new(), word, seeds: vec![], data: random_data(rng) }
}

pub(super) fn ldst_pair(rng: &mut Rng) -> MemEncoded {
    // kind: 0 = 32-bit, 1 = 64-bit, 2 = LDPSW (load-only).
    let (opc_bits, width8, force_load) = match rng.below(3) {
        0 => (0b00u32, false, false),
        1 => (0b10, true, false),
        _ => (0b01, false, true),
    };
    let l = if force_load { 1 } else { rng.below(2) };
    // LDPSW (the signed/force_load kind) has no non-allocating (000) form.
    let idx_choices: &[u32] =
        if force_load { &[0b001, 0b010, 0b011] } else { &[0b000, 0b001, 0b010, 0b011] };
    let idx_field = idx_choices[rng.below(idx_choices.len() as u32) as usize];
    let indexed = idx_field == 0b001 || idx_field == 0b011;

    let rn = rng.below(31);
    // rt != rt2 always; for indexed, both must differ from the base.
    let bad = |r: u32| r == rn && indexed;
    let rt = loop {
        let r = rng.below(32);
        if !bad(r) {
            break r;
        }
    };
    let rt2 = loop {
        let r = rng.below(32);
        if r != rt && !bad(r) {
            break r;
        }
    };

    let imm7_val = rng.below(17) as i64 - 8; // small signed
    let imm7_field = (imm7_val as u32) & 0x7f;
    let _ = width8; // element width is encoded by `opc_bits`; kept for readability

    let word = (opc_bits << 30)
        | (0b101 << 27)
        | (idx_field << 23)
        | (l << 22)
        | (imm7_field << 15)
        | (rt2 << 10)
        | (rn << 5)
        | rt;

    MemEncoded {
        init_v: Vec::new(),
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x100))],
        data: random_data(rng),
    }
}
