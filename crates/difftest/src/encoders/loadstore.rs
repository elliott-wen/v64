//! Encoders for load/store classes.
//!
//! These produce [`MemEncoded`]: besides the instruction word, they seed the
//! base register to point into the DATA region (so the access is mapped) and
//! provide random initial DATA-region contents to compare after the run.

use crate::fuzz::MemClass;
use crate::rng::Rng;
use crate::{MemEncoded, CODE_START, DATA_BASE, DATA_SIZE};

pub(super) fn classes() -> Vec<MemClass> {
    vec![
        MemClass { name: "ldst_uimm", encode: ldst_uimm },
        MemClass { name: "ldst_unscaled", encode: ldst_unscaled },
        MemClass { name: "ldst_post", encode: ldst_post },
        MemClass { name: "ldst_pre", encode: ldst_pre },
        MemClass { name: "ldst_reg", encode: ldst_reg },
        MemClass { name: "ldst_literal", encode: ldst_literal },
        MemClass { name: "ldst_pair", encode: ldst_pair },
        MemClass { name: "ldst_ordered", encode: ldst_ordered },
        MemClass { name: "ldst_atomic", encode: ldst_atomic },
        MemClass { name: "ldst_cas", encode: ldst_cas },
    ]
}

fn ldst_atomic(rng: &mut Rng) -> MemEncoded {
    let size = rng.below(4);
    // op 0..7 are the RMW ops; 8 is SWP.
    let op = rng.below(9);
    let (o3, opc) = if op == 8 { (1, 0) } else { (0, op) };
    let (a, r) = (rng.below(2), rng.below(2)); // ordering — no effect
    let rn = rng.below(31);
    let rs = rng.below(32);
    let rt = rt_distinct(rng, rn); // keep Rt != base
    let word = (size << 30)
        | (0b111 << 27)
        | (a << 23)
        | (r << 22)
        | (1 << 21)
        | (rs << 16)
        | (o3 << 15)
        | (opc << 12)
        | (rn << 5)
        | rt;
    // Atomics require natural alignment; base_near_mid is 8-aligned.
    MemEncoded { word, seeds: vec![(rn as u8, base_near_mid(rng, 0x100))], data: random_data(rng) }
}

fn ldst_cas(rng: &mut Rng) -> MemEncoded {
    let size = rng.below(4);
    let (l, o0) = (rng.below(2), rng.below(2));
    let rn = rng.below(31);
    let rs = rt_distinct(rng, rn); // Rs holds compare value + receives old; keep != base
    let rt = rng.below(32);
    let word = (size << 30)
        | (0b001000 << 24)
        | (1 << 23)
        | (l << 22)
        | (1 << 21)
        | (rs << 16)
        | (o0 << 15)
        | (0b11111 << 10)
        | (rn << 5)
        | rt;

    let base = base_near_mid(rng, 0x100);
    let data = random_data(rng);
    let mut seeds = vec![(rn as u8, base)];
    // Half the time, seed Rs to the in-memory value so the swap succeeds.
    if rng.below(2) == 0 {
        let off = (base - DATA_BASE) as usize;
        let mut val = 0u64;
        for i in 0..(1usize << size) {
            val |= u64::from(data[off + i]) << (8 * i);
        }
        seeds.push((rs as u8, val));
    }
    MemEncoded { word, seeds, data }
}

fn ldst_ordered(rng: &mut Rng) -> MemEncoded {
    let size = rng.below(4);
    let l = rng.below(2); // LDAR vs STLR
    let rn = rng.below(31);
    let rt = rng.below(32);
    let word = (size << 30)
        | (0b001000 << 24)
        | (1 << 23)
        | (l << 22)
        | (0b11111 << 16)
        | (1 << 15)
        | (0b11111 << 10)
        | (rn << 5)
        | rt;
    // Ordered accesses require natural alignment; base_near_mid is 8-aligned.
    MemEncoded {
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x100))],
        data: random_data(rng),
    }
}

fn ldst_literal(rng: &mut Rng) -> MemEncoded {
    let opc = rng.below(3); // LDR(W) / LDR(X) / LDRSW
    let rt = rng.below(32);
    // Point the literal at a 4-aligned slot inside the (seeded) DATA region.
    let target = DATA_BASE + u64::from(rng.below(DATA_SIZE as u32 - 8) & !3);
    let off = target as i64 - CODE_START as i64;
    let imm19 = ((off >> 2) as u32) & 0x7_ffff;
    let word = (opc << 30) | (0b011 << 27) | (imm19 << 5) | rt;
    MemEncoded { word, seeds: vec![], data: random_data(rng) }
}

fn ldst_pair(rng: &mut Rng) -> MemEncoded {
    // kind: 0 = 32-bit, 1 = 64-bit, 2 = LDPSW (load-only).
    let (opc_bits, width8, force_load) = match rng.below(3) {
        0 => (0b00u32, false, false),
        1 => (0b10, true, false),
        _ => (0b01, false, true),
    };
    let l = if force_load { 1 } else { rng.below(2) };
    // LDPSW (the signed/force_load kind) has no non-allocating (000) form.
    let idx_choices: &[u32] = if force_load {
        &[0b001, 0b010, 0b011]
    } else {
        &[0b000, 0b001, 0b010, 0b011]
    };
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

    let scale = if width8 { 3 } else { 2 };
    let imm7_val = rng.below(17) as i64 - 8; // small signed
    let imm7_field = (imm7_val as u32) & 0x7f;

    let word = (opc_bits << 30)
        | (0b101 << 27)
        | (idx_field << 23)
        | (l << 22)
        | (imm7_field << 15)
        | (rt2 << 10)
        | (rn << 5)
        | rt;

    MemEncoded {
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x100))],
        data: random_data(rng),
    }
}

/// Center of the DATA region, so a signed offset stays mapped either way.
const MID: u64 = DATA_BASE + (DATA_SIZE as u64) / 2;

/// A signed imm9 (`-256..=255`) and its raw 9-bit field.
fn imm9(rng: &mut Rng) -> (i64, u32) {
    let v = rng.below(512) as i64 - 256;
    (v, (v as u32) & 0x1ff)
}

/// `DATA_SIZE` random bytes for the scratch region.
fn random_data(rng: &mut Rng) -> Vec<u8> {
    let mut data = vec![0u8; DATA_SIZE];
    for chunk in data.chunks_mut(8) {
        let bytes = rng.next_u64().to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
    data
}

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

/// 8-aligned base near the region center, leaving `margin` bytes of slack on
/// each side for the offset.
fn base_near_mid(rng: &mut Rng, margin: u32) -> u64 {
    let span = DATA_SIZE as u32 - 2 * margin - 8;
    let off = margin + (rng.below(span) & !7);
    DATA_BASE + u64::from(off)
}

/// A data register distinct from `rn` (avoids the UNPREDICTABLE writeback case).
fn rt_distinct(rng: &mut Rng, rn: u32) -> u32 {
    loop {
        let rt = rng.below(31);
        if rt != rn {
            return rt;
        }
    }
}

/// Shared body for the imm9 addressing classes; `op1110` selects the form.
fn imm9_form(rng: &mut Rng, op1110: u32, indexed: bool) -> MemEncoded {
    let size = rng.below(4);
    let opc = valid_opc(rng, size);
    let (_, imm_field) = imm9(rng);
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
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x400))],
        data: random_data(rng),
    }
}

fn ldst_unscaled(rng: &mut Rng) -> MemEncoded {
    imm9_form(rng, 0b00, false)
}

fn ldst_post(rng: &mut Rng) -> MemEncoded {
    imm9_form(rng, 0b01, true)
}

fn ldst_pre(rng: &mut Rng) -> MemEncoded {
    imm9_form(rng, 0b11, true)
}

fn ldst_reg(rng: &mut Rng) -> MemEncoded {
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
        word,
        seeds: vec![
            (rn as u8, base_near_mid(rng, 0x400)),
            (rm as u8, u64::from(rng.below(64))),
        ],
        data: random_data(rng),
    }
}

fn ldst_uimm(rng: &mut Rng) -> MemEncoded {
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

    MemEncoded {
        word,
        seeds: vec![(rn as u8, base_addr)],
        data: random_data(rng),
    }
}
