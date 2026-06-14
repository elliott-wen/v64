//! Encoders for Advanced SIMD (vector) classes. They reuse the FP fuzz harness
//! (random V0..V31, compared after the run).

use super::random_v;
use crate::fuzz::FpClass;
use crate::rng::Rng;
use crate::FpEncoded;

/// FPCR with DN=1 (default NaN) for deterministic FP results.
const FPCR_DN: u64 = 1 << 25;

fn enc(word: u32, rng: &mut Rng) -> FpEncoded {
    FpEncoded { word, init_v: random_v(rng), gpr_seeds: vec![], fpcr: 0 }
}

pub(super) fn classes() -> Vec<FpClass> {
    vec![
        FpClass { name: "neon_three_same", encode: three_same },
        FpClass { name: "neon_three_same_fp", encode: three_same_fp },
        FpClass { name: "neon_two_reg_misc", encode: two_reg_misc },
        FpClass { name: "neon_mod_imm", encode: mod_imm },
        FpClass { name: "neon_dup", encode: dup_general },
        FpClass { name: "neon_dup_element", encode: dup_element },
        FpClass { name: "neon_ins", encode: ins },
        FpClass { name: "neon_mov_gpr", encode: mov_gpr },
        FpClass { name: "neon_zip_trn", encode: zip_trn },
        FpClass { name: "neon_ext", encode: ext },
        FpClass { name: "neon_shift_imm", encode: shift_imm },
        FpClass { name: "neon_across", encode: across },
    ]
}

fn across(rng: &mut Rng) -> FpEncoded {
    let (opcode, u) = match rng.below(5) {
        0 => (0b11011u32, 0), // ADDV
        1 => (0b01010, 0),    // SMAXV
        2 => (0b01010, 1),    // UMAXV
        3 => (0b11010, 0),    // SMINV
        _ => (0b11010, 1),    // UMINV
    };
    let q = rng.below(2);
    let size = if q == 1 { rng.below(3) } else { rng.below(2) }; // 32-bit needs Q=1
    let word = (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (0b11000 << 17)
        | (opcode << 12)
        | (0b10 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    enc(word, rng)
}

fn shift_imm(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    // SSHR/USHR (00000), SSRA/USRA (00010), SHL (01010, U=0 only).
    let (opcode, u) = match rng.below(5) {
        0 => (0b00000u32, 0),
        1 => (0b00000, 1),
        2 => (0b00010, 0),
        3 => (0b00010, 1),
        _ => (0b01010, 0), // SHL
    };
    let size = if q == 1 { rng.below(4) } else { rng.below(3) };
    let immh = (1 << size) | rng.below(1 << size); // highest set bit picks the size
    let immb = rng.below(8);
    let word = (q << 30)
        | (u << 29)
        | (0b011110 << 23)
        | (immh << 19)
        | (immb << 16)
        | (opcode << 11)
        | (1 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    enc(word, rng)
}

fn zip_trn(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    let opcode = [0b001u32, 0b010, 0b011, 0b101, 0b110, 0b111][rng.below(6) as usize];
    let size = if q == 1 { rng.below(4) } else { rng.below(3) }; // D needs Q=1
    let word = (q << 30)
        | (0b01110 << 24)
        | (size << 22)
        | (rng.bits(5) << 16)
        | (opcode << 12)
        | (1 << 11)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    enc(word, rng)
}

fn ext(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    let imm4 = if q == 1 { rng.below(16) } else { rng.below(8) };
    let word = (q << 30)
        | (1 << 29)
        | (0b01110 << 24)
        | (rng.bits(5) << 16)
        | (imm4 << 11)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    enc(word, rng)
}

fn copy_word(q: u32, op: u32, imm5: u32, imm4: u32, rn: u32, rd: u32) -> u32 {
    (q << 30) | (op << 29) | (0b01110000 << 21) | (imm5 << 16) | (imm4 << 11) | (1 << 10) | (rn << 5) | rd
}

fn imm5_for(size: u32, index: u32) -> u32 {
    (index << (size + 1)) | (1 << size)
}

fn dup_element(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    let size = if q == 1 { rng.below(4) } else { rng.below(3) }; // D needs Q=1
    let index = rng.below(16 >> size);
    let word = copy_word(q, 0, imm5_for(size, index), 0b0000, rng.bits(5), rng.bits(5));
    enc(word, rng)
}

fn ins(rng: &mut Rng) -> FpEncoded {
    let size = rng.below(4);
    let dst = rng.below(16 >> size);
    let imm5 = imm5_for(size, dst);
    let word = if rng.below(2) == 0 {
        // INS (general): Q=1, imm4=0011.
        copy_word(1, 0, imm5, 0b0011, rng.bits(5), rng.bits(5))
    } else {
        // INS (element): op=1, Q=1, imm4 = src index << size.
        let src = rng.below(16 >> size);
        copy_word(1, 1, imm5, (src << size) & 0xf, rng.bits(5), rng.bits(5))
    };
    enc(word, rng)
}

fn mov_gpr(rng: &mut Rng) -> FpEncoded {
    let signed = rng.below(2) == 0;
    let q = rng.below(2);
    let size = match (signed, q == 1) {
        (true, true) => rng.below(3),  // SMOV Xd: B/H/S
        (true, false) => rng.below(2), // SMOV Wd: B/H
        (false, true) => 3,            // UMOV Xd: D
        (false, false) => rng.below(3), // UMOV Wd: B/H/S
    };
    let index = rng.below(16 >> size);
    let imm4 = if signed { 0b0101 } else { 0b0111 };
    let word = copy_word(q, 0, imm5_for(size, index), imm4, rng.bits(5), rng.bits(5));
    enc(word, rng)
}

fn three_same_fp(rng: &mut Rng) -> FpEncoded {
    let ops = [0x1au32, 0x3a, 0x5b, 0x5f, 0x1e, 0x3e, 0x18, 0x38, 0x7a, 0x1c, 0x5c, 0x7c];
    let fpopcode = ops[rng.below(ops.len() as u32) as usize];
    let u = (fpopcode >> 6) & 1;
    let bit23 = (fpopcode >> 5) & 1;
    let opcode = fpopcode & 0x1f;
    let sz = rng.below(2);
    let q = if sz == 1 { 1 } else { rng.below(2) }; // double needs Q=1
    let word = (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (bit23 << 23)
        | (sz << 22)
        | (1 << 21)
        | (rng.bits(5) << 16)
        | (opcode << 11)
        | (1 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    FpEncoded { word, init_v: random_v(rng), gpr_seeds: vec![], fpcr: FPCR_DN }
}

fn two_reg_misc(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    // (u, opcode, max_size) for the implemented ops.
    let variants: [(u32, u32, u32); 9] = [
        (0, 0b00000, 3), // REV64 (size<=2)
        (0, 0b00001, 1), // REV16 (size 0)
        (1, 0b00000, 2), // REV32 (size<=1)
        (0, 0b00100, 3), // CLS (size<=2)
        (1, 0b00100, 3), // CLZ (size<=2)
        (0, 0b00101, 1), // CNT (size 0)
        (1, 0b00101, 2), // NOT/RBIT (size<=1)
        (0, 0b01011, 4), // ABS
        (1, 0b01011, 4), // NEG
    ];
    let (u, opcode, maxs) = variants[rng.below(variants.len() as u32) as usize];
    let size = pick_size(rng, opcode, maxs, q);
    let word = (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (0b10000 << 17)
        | (opcode << 12)
        | (0b10 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    enc(word, rng)
}

fn pick_size(rng: &mut Rng, opcode: u32, maxs: u32, q: u32) -> u32 {
    match opcode {
        // REV16/REV32/CLS/CLZ have a hard size cap; CNT is size 0.
        0b00001 => 0,
        _ if maxs <= 2 => rng.below(maxs.min(3)),
        _ => {
            // ABS/NEG: a 64-bit element requires Q=1.
            let s = rng.below(4);
            if s == 3 && q == 0 { rng.below(3) } else { s }
        }
    }
}

fn mod_imm(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    let op = rng.below(2);
    // cmode 0..14 (exclude 1111 = FMOV-vector, not implemented).
    let cmode = rng.below(15);
    let imm8 = rng.bits(8);
    let word = (q << 30)
        | (op << 29)
        | (0b01111 << 24)
        | ((imm8 >> 5) << 16)
        | (cmode << 12)
        | (0b01 << 10)
        | ((imm8 & 0x1f) << 5)
        | rng.bits(5);
    enc(word, rng)
}

fn dup_general(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    // imm5 low set bit picks the size: B=1,H=2,S=4,D=8; D needs Q=1.
    let sizes: &[u32] = if q == 1 { &[1, 2, 4, 8] } else { &[1, 2, 4] };
    let imm5 = sizes[rng.below(sizes.len() as u32) as usize];
    let word = (q << 30)
        | (0b01110000 << 21)
        | (imm5 << 16)
        | (0b0001 << 11)
        | (1 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    // Seed GPRs with random values (DUP reads Rn); init_v is overwritten anyway.
    FpEncoded {
        word,
        init_v: random_v(rng),
        gpr_seeds: (0..31).map(|r| (r as u8, rng.next_u64())).collect(),
        fpcr: 0,
    }
}

fn three_same(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    let opcode = rng.below(0b11000); // 0x00..0x17
    let size_q = |rng: &mut Rng| if q == 1 { rng.below(4) } else { rng.below(3) };
    let (u, size) = match opcode {
        0b10111 => (0, size_q(rng)),               // ADDP (U=0)
        0b10110 => (rng.below(2), 1 + rng.below(2)), // SQDMULH/SQRDMULH: size 1,2
        0b10011 => {
            let u = rng.below(2);
            (u, if u == 1 { 0 } else { rng.below(3) }) // PMUL byte / MUL no-64
        }
        // No-64-bit ops.
        0b00000 | 0b00010 | 0b00100 | 0b01100 | 0b01101 | 0b01110 | 0b01111 | 0b10010
        | 0b10100 | 0b10101 => (rng.below(2), rng.below(3)),
        _ => (rng.below(2), size_q(rng)), // 64-bit allowed (needs Q=1)
    };
    let word = (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (1 << 21)
        | (rng.bits(5) << 16)
        | (opcode << 11)
        | (1 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    FpEncoded { word, init_v: random_v(rng), gpr_seeds: vec![], fpcr: 0 }
}
