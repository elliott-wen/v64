//! Encoders for scalar floating-point classes.
//!
//! FPCR is seeded with default-NaN mode (DN=1) and round-to-nearest so results
//! are deterministic and match Rust's native float behavior. V0..V31 are seeded
//! with random bits (exercising finite, NaN, and infinity inputs).

use super::reg;
use crate::fuzz::FpClass;
use crate::rng::Rng;
use crate::FpEncoded;

/// FPCR with DN=1 (default NaN), round-to-nearest, no flush-to-zero.
const FPCR_DN: u64 = 1 << 25;

/// Fixed top bits `0 0 0 11110` shared by the FP data-processing encodings.
const FP_HDR: u32 = 0b0001_1110 << 24;

pub(super) fn classes() -> Vec<FpClass> {
    vec![
        FpClass { name: "fp_dp1", encode: fp_dp1 },
        FpClass { name: "fp_dp2", encode: fp_dp2 },
        FpClass { name: "fp_compare", encode: fp_compare },
        FpClass { name: "fp_csel", encode: fp_csel },
        FpClass { name: "fp_imm", encode: fp_imm },
        FpClass { name: "fp_cvt", encode: fp_cvt },
        FpClass { name: "fp_dp3", encode: fp_dp3 },
        FpClass { name: "fp_ccmp", encode: fp_ccmp },
    ]
}

fn fp_dp3(rng: &mut Rng) -> FpEncoded {
    let ftype = rng.below(2);
    let o1 = rng.below(2);
    let o0 = rng.below(2);
    let word = (0b0011111 << 24)
        | (ftype << 22)
        | (o1 << 21)
        | (reg(rng) << 16)
        | (o0 << 15)
        | (reg(rng) << 10)
        | (reg(rng) << 5)
        | reg(rng);
    enc(word, rng)
}

fn fp_ccmp(rng: &mut Rng) -> FpEncoded {
    let ftype = rng.below(2);
    let op = rng.below(2); // FCCMP / FCCMPE
    let word = FP_HDR
        | (ftype << 22)
        | (1 << 21)
        | (reg(rng) << 16)
        | (rng.bits(4) << 12)
        | (0b01 << 10)
        | (reg(rng) << 5)
        | (op << 4)
        | rng.bits(4);
    // Seed NZCV via random flags init? The harness sets flags; just fuzz.
    enc(word, rng)
}

fn random_v(rng: &mut Rng) -> [u128; 32] {
    let mut v = [0u128; 32];
    for slot in &mut v {
        *slot = (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64());
    }
    v
}

fn enc(word: u32, rng: &mut Rng) -> FpEncoded {
    FpEncoded { word, init_v: random_v(rng), gpr_seeds: vec![], fpcr: FPCR_DN }
}

fn fp_dp1(rng: &mut Rng) -> FpEncoded {
    let ftype = rng.below(2);
    // FMOV/FABS/FNEG/FSQRT keep the type; FCVT flips single<->double; FRINT* keep.
    let choices: &[u32] = if ftype == 0 {
        &[0, 1, 2, 3, 5, 0x8, 0x9, 0xa, 0xb, 0xc, 0xe, 0xf]
    } else {
        &[0, 1, 2, 3, 4, 0x8, 0x9, 0xa, 0xb, 0xc, 0xe, 0xf]
    };
    let opcode = choices[rng.below(choices.len() as u32) as usize];
    let word =
        FP_HDR | (ftype << 22) | (1 << 21) | (opcode << 15) | (0b10000 << 10) | (reg(rng) << 5) | reg(rng);
    enc(word, rng)
}

fn fp_dp2(rng: &mut Rng) -> FpEncoded {
    let ftype = rng.below(2);
    let opcode = rng.below(9); // FMUL..FNMUL
    let word = FP_HDR
        | (ftype << 22)
        | (1 << 21)
        | (reg(rng) << 16)
        | (opcode << 12)
        | (0b10 << 10)
        | (reg(rng) << 5)
        | reg(rng);
    enc(word, rng)
}

fn fp_compare(rng: &mut Rng) -> FpEncoded {
    let ftype = rng.below(2);
    let cmp_zero = rng.below(2);
    let signaling = rng.below(2);
    let opcode2 = (signaling << 4) | (cmp_zero << 3);
    let rm = if cmp_zero == 1 { 0 } else { reg(rng) }; // #0.0 form needs Rm=0
    let word =
        FP_HDR | (ftype << 22) | (1 << 21) | (rm << 16) | (0b1000 << 10) | (reg(rng) << 5) | opcode2;
    enc(word, rng)
}

fn fp_csel(rng: &mut Rng) -> FpEncoded {
    let ftype = rng.below(2);
    let word = FP_HDR
        | (ftype << 22)
        | (1 << 21)
        | (reg(rng) << 16)
        | (rng.bits(4) << 12)
        | (0b11 << 10)
        | (reg(rng) << 5)
        | reg(rng);
    enc(word, rng)
}

fn fp_imm(rng: &mut Rng) -> FpEncoded {
    let ftype = rng.below(2);
    let word = FP_HDR | (ftype << 22) | (1 << 21) | (rng.bits(8) << 13) | (0b100 << 10) | reg(rng);
    enc(word, rng)
}

fn fp_cvt(rng: &mut Rng) -> FpEncoded {
    // SCVTF/UCVTF, FCVT{N,P,M,Z,A}{S,U}, FMOV.
    let (mut sf, mut ftype) = (rng.below(2), rng.below(2));
    let (rmode, opcode) = match rng.below(9) {
        0 => (0b00, 0b010), // SCVTF
        1 => (0b00, 0b011), // UCVTF
        2 => (rng.below(4), 0b000), // FCVT{N,P,M,Z}S
        3 => (rng.below(4), 0b001), // FCVT{N,P,M,Z}U
        4 => (0b00, 0b100), // FCVTAS
        5 => (0b00, 0b101), // FCVTAU
        _ => {
            // FMOV requires matching width: W<->S or X<->D.
            if rng.below(2) == 0 {
                sf = 0;
                ftype = 0;
            } else {
                sf = 1;
                ftype = 1;
            }
            (0b00, if rng.below(2) == 0 { 0b110 } else { 0b111 })
        }
    };
    let word = (sf << 31)
        | (0b0011110 << 24)
        | (ftype << 22)
        | (1 << 21)
        | (rmode << 19)
        | (opcode << 16)
        | (reg(rng) << 5)
        | reg(rng);
    enc(word, rng)
}
