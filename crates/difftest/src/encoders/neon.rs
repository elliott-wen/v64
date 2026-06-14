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
        FpClass { name: "neon_three_diff", encode: three_diff },
        FpClass { name: "neon_indexed", encode: indexed },
        FpClass { name: "neon_three_same_fp", encode: three_same_fp },
        FpClass { name: "neon_two_reg_misc", encode: two_reg_misc },
        FpClass { name: "neon_two_reg_misc_fp", encode: two_reg_misc_fp },
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
    // (opcode, u). u==2 means "either" (random U with both allocated).
    let table: &[(u32, u32)] = &[
        (0b00000, 2), // SSHR/USHR
        (0b00010, 2), // SSRA/USRA
        (0b00100, 2), // SRSHR/URSHR
        (0b00110, 2), // SRSRA/URSRA
        (0b01000, 1), // SRI (U=1)
        (0b01010, 2), // SHL / SLI
        (0b01100, 1), // SQSHLU (U=1)
        (0b01110, 2), // SQSHL / UQSHL
        (0b10000, 2), // SHRN / SQSHRUN
        (0b10001, 2), // RSHRN / SQRSHRUN
        (0b10010, 2), // SQSHRN / UQSHRN
        (0b10011, 2), // SQRSHRN / UQRSHRN
        (0b10100, 2), // SSHLL / USHLL
        (0b11100, 2), // SCVTF / UCVTF
        (0b11111, 2), // FCVTZS / FCVTZU
    ];
    let (opcode, ufix) = table[rng.below(table.len() as u32) as usize];
    let u = if ufix == 2 { rng.below(2) } else { ufix };

    // Pick the element size and immh:immb consistent with the op's shape.
    let (immh, immb) = match opcode {
        // Narrowing / widening: element size 0..2 (immh top bit not 0b1000).
        0b10000..=0b10100 => {
            let size = rng.below(3);
            ((1 << size) | rng.below(1 << size), rng.below(8))
        }
        // Fixed-point convert: only 32-bit (immh=01xx) or 64-bit (immh=1xxx, Q=1).
        0b11100 | 0b11111 => {
            let size = if q == 1 { 2 + rng.below(2) } else { 2 };
            ((1 << size) | rng.below(1 << size), rng.below(8))
        }
        // Same-width: any element size (64-bit needs Q=1).
        _ => {
            let size = if q == 1 { rng.below(4) } else { rng.below(3) };
            ((1 << size) | rng.below(1 << size), rng.below(8))
        }
    };
    let word = (q << 30)
        | (u << 29)
        | (0b011110 << 23)
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
    let ops = [
        0x1au32, 0x3a, 0x5b, 0x5f, 0x1e, 0x3e, 0x18, 0x38, 0x7a, 0x1c, 0x5c, 0x7c, // base
        0x19, 0x39, 0x1b, 0x1f, 0x3f, 0x5d, 0x7d, // FMLA/FMLS/FMULX/FRECPS/FRSQRTS/FACGE/FACGT
        0x5a, 0x5e, 0x7e, 0x58, 0x78, // FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP
    ];
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

/// Size rule for a two-reg-misc op.
#[derive(Clone, Copy)]
enum SzRule {
    Le2,   // size in 0..=2
    Eq0,   // size == 0
    Le1,   // size in 0..=1
    Eq1,   // size == 1
    Not3,  // size in 0..=3, but 3 requires Q=1
}

fn two_reg_misc(rng: &mut Rng) -> FpEncoded {
    use SzRule::*;
    let q = rng.below(2);
    // (u, opcode, size rule).
    let variants: &[(u32, u32, SzRule)] = &[
        (0, 0b00000, Le2),  // REV64
        (0, 0b00001, Eq0),  // REV16
        (1, 0b00000, Le1),  // REV32
        (0, 0b00010, Le2),  // SADDLP
        (1, 0b00010, Le2),  // UADDLP
        (0, 0b00110, Le2),  // SADALP
        (1, 0b00110, Le2),  // UADALP
        (0, 0b00011, Not3), // SUQADD
        (1, 0b00011, Not3), // USQADD
        (0, 0b00100, Le2),  // CLS
        (1, 0b00100, Le2),  // CLZ
        (0, 0b00101, Eq0),  // CNT
        (1, 0b00101, Eq0),  // NOT
        (1, 0b00101, Eq1),  // RBIT
        (0, 0b00111, Not3), // SQABS
        (1, 0b00111, Not3), // SQNEG
        (0, 0b01000, Not3), // CMGT #0
        (1, 0b01000, Not3), // CMGE #0
        (0, 0b01001, Not3), // CMEQ #0
        (1, 0b01001, Not3), // CMLE #0
        (0, 0b01010, Not3), // CMLT #0
        (0, 0b01011, Not3), // ABS
        (1, 0b01011, Not3), // NEG
        (0, 0b10010, Le2),  // XTN
        (1, 0b10010, Le2),  // SQXTUN
        (0, 0b10100, Le2),  // SQXTN
        (1, 0b10100, Le2),  // UQXTN
        (1, 0b10011, Le2),  // SHLL
    ];
    let (u, opcode, rule) = variants[rng.below(variants.len() as u32) as usize];
    let size = pick_size(rng, rule, q);
    let word = misc_word(q, u, size, opcode, rng);
    enc(word, rng)
}

fn misc_word(q: u32, u: u32, size: u32, opcode: u32, rng: &mut Rng) -> u32 {
    (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (0b10000 << 17)
        | (opcode << 12)
        | (0b10 << 10)
        | (rng.bits(5) << 5)
        | rng.bits(5)
}

fn pick_size(rng: &mut Rng, rule: SzRule, q: u32) -> u32 {
    match rule {
        SzRule::Eq0 => 0,
        SzRule::Eq1 => 1,
        SzRule::Le1 => rng.below(2),
        SzRule::Le2 => rng.below(3),
        SzRule::Not3 => {
            let s = rng.below(4);
            if s == 3 && q == 0 {
                rng.below(3)
            } else {
                s
            }
        }
    }
}

fn two_reg_misc_fp(rng: &mut Rng) -> FpEncoded {
    // (u, opcode-low-5) pairs; size[1] folds into the high opcode bit, size[0]
    // selects single/double. We build the raw `size`/`u`/`opcode` fields here.
    // (u, opcode5, a) where the 7-bit op = opcode5 | a<<5 | u<<6.
    let variants: &[(u32, u32, u32)] = &[
        (0, 0b01111, 1), // FABS (0x2f)
        (1, 0b01111, 1), // FNEG (0x6f)
        (1, 0b11111, 1), // FSQRT (0x7f)
        (0, 0b01100, 1), // FCMGT #0 (0x2c)
        (0, 0b01101, 1), // FCMEQ #0 (0x2d)
        (0, 0b01110, 1), // FCMLT #0 (0x2e)
        (1, 0b01100, 1), // FCMGE #0 (0x6c)
        (1, 0b01101, 1), // FCMLE #0 (0x6d)
        (0, 0b11101, 0), // SCVTF (0x1d)
        (1, 0b11101, 0), // UCVTF (0x5d)
        (0, 0b11010, 0), // FCVTNS (0x1a)
        (0, 0b11011, 0), // FCVTMS (0x1b)
        (0, 0b11010, 1), // FCVTPS (0x3a)
        (0, 0b11011, 1), // FCVTZS (0x3b)
        (1, 0b11010, 0), // FCVTNU (0x5a)
        (1, 0b11011, 0), // FCVTMU (0x5b)
        (1, 0b11010, 1), // FCVTPU (0x7a)
        (1, 0b11011, 1), // FCVTZU (0x7b)
        (0, 0b11100, 0), // FCVTAS (0x1c)
        (1, 0b11100, 0), // FCVTAU (0x5c)
        (0, 0b11000, 0), // FRINTN (0x18)
        (0, 0b11001, 0), // FRINTM (0x19)
        (0, 0b11000, 1), // FRINTP (0x38)
        (0, 0b11001, 1), // FRINTZ (0x39)
        (1, 0b11000, 0), // FRINTA (0x58)
        (1, 0b11001, 0), // FRINTX (0x59)
        (1, 0b11001, 1), // FRINTI (0x79)
        (0, 0b10110, 0), // FCVTN (0x16) — double->single
        (0, 0b10111, 0), // FCVTL (0x17) — single->double
    ];
    let (u, opcode, a) = variants[rng.below(variants.len() as u32) as usize];
    // size[1] = a. size[0] = is_double. FCVTN/FCVTL require double (is_double=1).
    let needs_double = matches!(opcode | (a << 5) | (u << 6), 0x16 | 0x17);
    let q = rng.below(2);
    let is_double = if needs_double {
        1
    } else if q == 0 {
        // double lanes need Q=1; with Q=0 force single to stay allocated.
        0
    } else {
        rng.below(2)
    };
    let q = if needs_double { 1 } else { q };
    let size = (a << 1) | is_double;
    let word = misc_word(q, u, size, opcode, rng);
    FpEncoded { word, init_v: random_v(rng), gpr_seeds: vec![], fpcr: FPCR_DN }
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

fn indexed(rng: &mut Rng) -> FpEncoded {
    // (u, opcode, is_fp).
    let ops: &[(u32, u32, bool)] = &[
        (0, 0b1000, false), // MUL
        (1, 0b0000, false), // MLA
        (1, 0b0100, false), // MLS
        (0, 0b0010, false), // SMLAL
        (1, 0b0010, false), // UMLAL
        (0, 0b0110, false), // SMLSL
        (1, 0b0110, false), // UMLSL
        (0, 0b1010, false), // SMULL
        (1, 0b1010, false), // UMULL
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
    let q = rng.below(2);

    // Pick size, then derive index bits (h, l, m) and the Rm register.
    let (size, h, l, m, rm4, q) = if is_fp {
        // single (size 2) or double (size 3, needs L=0 and Q=1).
        if rng.below(2) == 0 {
            let idx = rng.below(4); // h:l
            let rm5 = rng.bits(5);
            (2u32, idx >> 1, idx & 1, rm5 >> 4, rm5 & 0xf, q)
        } else {
            let idx = rng.below(2); // h
            let rm5 = rng.bits(5);
            (3u32, idx, 0, rm5 >> 4, rm5 & 0xf, 1)
        }
    } else if rng.below(2) == 0 {
        // MO_16: index = h:l:m (0..7), Rm in 0..15.
        let idx = rng.below(8);
        (1u32, idx >> 2, (idx >> 1) & 1, idx & 1, rng.bits(4), q)
    } else {
        // MO_32: index = h:l (0..3), Rm 5-bit via m.
        let idx = rng.below(4);
        let rm5 = rng.bits(5);
        (2u32, idx >> 1, idx & 1, rm5 >> 4, rm5 & 0xf, q)
    };

    let word = (q << 30)
        | (u << 29)
        | (0b01111 << 24)
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

fn three_diff(rng: &mut Rng) -> FpEncoded {
    let q = rng.below(2);
    // opcode (bits 15:12) with its (u, size) constraints.
    let opcode = match rng.below(15) {
        14 => 0b1110u32, // PMULL (U=0, size 0)
        n => n,          // 0..13
    };
    let (u, size) = match opcode {
        0b1001 | 0b1011 | 0b1101 => (0, 1 + rng.below(2)), // SQDM*: H/S only
        0b1110 => (0, 0),                                  // PMULL: byte source
        _ => (rng.below(2), rng.below(3)),                 // size 0..2
    };
    let word = (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (1 << 21)
        | (rng.bits(5) << 16)
        | (opcode << 12)
        | (rng.bits(5) << 5)
        | rng.bits(5);
    enc(word, rng)
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
