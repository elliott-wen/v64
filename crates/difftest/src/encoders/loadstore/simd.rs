//! SIMD/FP load/store encoders: single register and pair (V=1), plus the
//! LD1-4/ST1-4 structure forms (multiple and single/replicate).

use super::{base_near_mid, random_data, rt_distinct};
use crate::rng::Rng;
use crate::{MemEncoded, DATA_BASE, DATA_SIZE};

/// Seed all 32 SIMD/FP registers with random values (for store coverage).
fn random_v_seeds(rng: &mut Rng) -> Vec<(u8, u128)> {
    (0..32).map(|r| (r, (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64()))).collect()
}

/// Pick a SIMD/FP access size: `(size[31:30], opc_hi[23], log2 bytes)` for
/// B/H/S/D/Q. `opc_hi` (bit 23) widens to the 128-bit Q form.
fn vec_size(rng: &mut Rng) -> (u32, u32, u32) {
    [(0u32, 0u32, 0u32), (1, 0, 1), (2, 0, 2), (3, 0, 3), (0, 1, 4)][rng.below(5) as usize]
}

/// SIMD/FP load/store single register (V=1), B/H/S/D/Q, across every addressing
/// form: unsigned-immediate, unscaled (LDUR), post/pre-index, register-offset.
pub(super) fn ldst_vec_reg(rng: &mut Rng) -> MemEncoded {
    match rng.below(5) {
        0 => vec_uimm(rng),
        1 => vec_imm9(rng, 0b00), // LDUR/STUR (unscaled)
        2 => vec_imm9(rng, 0b01), // post-index
        3 => vec_imm9(rng, 0b11), // pre-index
        _ => vec_reg_off(rng),    // register offset
    }
}

/// Unsigned-immediate vec single register.
fn vec_uimm(rng: &mut Rng) -> MemEncoded {
    let (size30, opc_hi, log2) = vec_size(rng);
    let opc = (opc_hi << 1) | rng.below(2);
    let imm12 = rng.below(32);
    let rn = rng.below(31);
    let rt = rng.below(32);

    let word = (size30 << 30)
        | (0b111 << 27)
        | (1 << 26) // V
        | (0b01 << 24)
        | (opc << 22)
        | (imm12 << 10)
        | (rn << 5)
        | rt;

    let offset = imm12 << log2;
    let slack = DATA_SIZE as u32 - offset - 16;
    let base = DATA_BASE + u64::from(rng.below(slack) & !15);
    MemEncoded { init_v: random_v_seeds(rng), word, seeds: vec![(rn as u8, base)], data: random_data(rng) }
}

/// imm9 vec single register: unscaled / post-index / pre-index (selected by
/// `op1110`). The data register Vt and base Xn are different register files, so
/// no rt/rn distinctness is needed even for the writeback forms.
fn vec_imm9(rng: &mut Rng, op1110: u32) -> MemEncoded {
    let (size30, opc_hi, _) = vec_size(rng);
    let opc = (opc_hi << 1) | rng.below(2);
    let imm9 = ((rng.below(512) as i64 - 256) as u32) & 0x1ff;
    let rn = rng.below(31);
    let rt = rng.below(32);

    let word = (size30 << 30)
        | (0b111 << 27)
        | (1 << 26) // V
        | (opc << 22)
        | (imm9 << 12)
        | (op1110 << 10)
        | (rn << 5)
        | rt;

    MemEncoded {
        init_v: random_v_seeds(rng),
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x400))],
        data: random_data(rng),
    }
}

/// Register-offset vec single register. Base Xn and index Xm are both GPRs, so
/// they must be distinct to keep their seeds from colliding; Vt is unrelated.
fn vec_reg_off(rng: &mut Rng) -> MemEncoded {
    let (size30, opc_hi, _) = vec_size(rng);
    let opc = (opc_hi << 1) | rng.below(2);
    let option = [0b010u32, 0b011, 0b110, 0b111][rng.below(4) as usize];
    let s = rng.below(2);
    let rn = rng.below(31);
    let rm = rt_distinct(rng, rn);
    let rt = rng.below(32);

    let word = (size30 << 30)
        | (0b111 << 27)
        | (1 << 26) // V
        | (opc << 22)
        | (1 << 21)
        | (rm << 16)
        | (option << 13)
        | (s << 12)
        | (0b10 << 10)
        | (rn << 5)
        | rt;

    // Small index so base + (index << scale) stays mapped even for Q (16 B).
    MemEncoded {
        init_v: random_v_seeds(rng),
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x400)), (rm as u8, u64::from(rng.below(32)))],
        data: random_data(rng),
    }
}

/// SIMD/FP load/store pair (LDP/STP S/D/Q).
pub(super) fn ldst_vec_pair(rng: &mut Rng) -> MemEncoded {
    let opc = [0u32, 1, 2][rng.below(3) as usize]; // S/D/Q
    // non-allocating (LDNP/STNP) / post / offset / pre
    let idx = [0b00u32, 0b01, 0b10, 0b11][rng.below(4) as usize];
    let l = rng.below(2);
    let rn = rng.below(31);
    let rt = rng.below(32);
    let rt2 = rng.below(32);
    let imm7 = rng.below(16); // small positive offset

    let word = (opc << 30)
        | (0b101 << 27)
        | (1 << 26) // V
        | (idx << 23)
        | (l << 22)
        | (imm7 << 15)
        | (rt2 << 10)
        | (rn << 5)
        | rt;

    MemEncoded {
        init_v: random_v_seeds(rng),
        word,
        seeds: vec![(rn as u8, base_near_mid(rng, 0x400))],
        data: random_data(rng),
    }
}

/// SIMD load/store multiple structures (LD1-4/ST1-4).
pub(super) fn ldst_struct_multi(rng: &mut Rng) -> MemEncoded {
    let q = rng.below(2);
    // (opcode, selem).
    let (opcode, selem) =
        [(0x0u32, 4u32), (0x2, 1), (0x4, 3), (0x6, 1), (0x7, 1), (0x8, 2), (0xa, 1)][rng.below(7) as usize];
    // size 0..3; size==3 && !q && selem!=1 is reserved.
    let size = if q == 0 && selem != 1 { rng.below(3) } else { rng.below(4) };
    let l = rng.below(2);

    let (postidx, rm, seeds) = addr_mode(rng);

    let word = (q << 30)
        | (0b001100 << 24)
        | (postidx << 23)
        | (l << 22)
        | (rm << 16)
        | (opcode << 12)
        | (size << 10)
        | (seeds.0 << 5)
        | rng.bits(5);
    MemEncoded { init_v: random_v_seeds(rng), word, seeds: seeds.1, data: random_data(rng) }
}

/// SIMD load/store single structure / replicate (LD1..4 lane, LD1R..LD4R).
pub(super) fn ldst_struct_single(rng: &mut Rng) -> MemEncoded {
    let selem = 1 + rng.below(4);
    let opc0 = (selem - 1) >> 1;
    let r = (selem - 1) & 1;
    let is_load = rng.below(2);
    let replicate = is_load == 1 && rng.below(3) == 0;

    let (q, s, size, scale) = if replicate {
        (rng.below(2), 0, rng.below(4), 3) // LDxR: size = element size, opc[2:1]=3
    } else {
        match rng.below(4) {
            0 => {
                let idx = rng.below(16);
                (idx >> 3, (idx >> 2) & 1, idx & 3, 0) // B
            }
            1 => {
                let idx = rng.below(8);
                (idx >> 2, (idx >> 1) & 1, (idx & 1) << 1, 1) // H
            }
            2 => {
                let idx = rng.below(4);
                (idx >> 1, idx & 1, 0, 2) // S
            }
            _ => (rng.below(2), 0, 1, 2), // D
        }
    };
    let opc = (scale << 1) | opc0;
    let (postidx, rm, (rn, seeds)) = addr_mode(rng);

    let word = (q << 30)
        | (0b001101 << 24)
        | (postidx << 23)
        | (is_load << 22)
        | (r << 21)
        | (rm << 16)
        | (opc << 13)
        | (s << 12)
        | (size << 10)
        | (rn << 5)
        | rng.bits(5);
    MemEncoded { init_v: random_v_seeds(rng), word, seeds, data: random_data(rng) }
}

/// Pick a structure-load addressing form: no-offset, post-index immediate, or
/// post-index register. Returns `(postidx, rm, (rn, seeds))`.
fn addr_mode(rng: &mut Rng) -> (u32, u32, (u32, Vec<(u8, u64)>)) {
    let rn = rng.below(31);
    let base = base_near_mid(rng, 0x400);
    match rng.below(3) {
        0 => (0, 0, (rn, vec![(rn as u8, base)])),  // no offset
        1 => (1, 31, (rn, vec![(rn as u8, base)])), // post-index immediate
        _ => {
            let rm = rt_distinct(rng, rn);
            (1, rm, (rn, vec![(rn as u8, base), (rm as u8, u64::from(rng.below(64)))]))
        }
    }
}
