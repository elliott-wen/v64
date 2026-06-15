//! Atomic and ordered access encoders: LSE atomics (LDADD/SWP/...), CAS, and
//! the acquire/release LDAR/STLR forms.

use super::{base_near_mid, random_data, rt_distinct};
use crate::rng::Rng;
use crate::{MemEncoded, DATA_BASE};

pub(super) fn ldst_atomic(rng: &mut Rng) -> MemEncoded {
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
    MemEncoded {
        init_v: Vec::new(),
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x100))],
        data: random_data(rng),
    }
}

pub(super) fn ldst_cas(rng: &mut Rng) -> MemEncoded {
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
    MemEncoded { init_v: Vec::new(), word, seeds, data }
}

pub(super) fn ldst_ordered(rng: &mut Rng) -> MemEncoded {
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
        init_v: Vec::new(),
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x100))],
        data: random_data(rng),
    }
}
