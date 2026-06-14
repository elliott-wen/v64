//! Advanced SIMD two-register misc. Integer ops stay in `SimdTwoRegMisc`; the
//! floating-point sub-block (opcodes 0xc..0xf, 0x16..0x1f) is remapped exactly
//! as QEMU does — `opcode |= size[1]<<5 | u<<6`, `sz = size[0]` — and emitted as
//! `SimdTwoRegMiscFp`.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 5) as u8;
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;

    // The FP sub-block.
    if matches!(opcode, 0xc..=0xf | 0x16..=0x1f) {
        let is_double = size & 1 == 1;
        let fpop = opcode | ((size >> 1) << 5) | (u8::from(u) << 6);
        if !fp_valid(fpop, is_double, q) {
            return Insn::Unsupported { word };
        }
        return Insn::SimdTwoRegMiscFp { q, sz: is_double, fpop, rn, rd };
    }

    let valid = match opcode {
        0b00000 => {
            if u {
                size <= 1
            } else {
                size <= 2
            }
        } // REV32 / REV64
        0b00001 => !u && size == 0,                 // REV16
        0b00010 | 0b00110 => size <= 2,             // SADDLP/UADDLP, SADALP/UADALP
        0b00011 => size != 3 || q,                  // SUQADD/USQADD
        0b00100 => size <= 2,                        // CLS/CLZ
        0b00101 => (!u && size == 0) || (u && size <= 1), // CNT / NOT / RBIT
        0b00111 => size != 3 || q,                  // SQABS/SQNEG
        0b01000 | 0b01001 => size != 3 || q,        // CMGT/CMGE, CMEQ/CMLE (zero)
        0b01010 => !u && (size != 3 || q),          // CMLT (zero)
        0b01011 => size != 3 || q,                  // ABS/NEG
        0b10010 | 0b10100 => size <= 2,             // XTN/SQXTUN, SQXTN/UQXTN
        0b10011 => u && size <= 2,                   // SHLL
        _ => false,
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdTwoRegMisc { q, u, size, opcode, rn, rd }
}

/// Validity for the remapped 7-bit FP opcode. Estimate ops (FRECPE/FRSQRTE/
/// URECPE/URSQRTE) and the FRINT32/64 family are left unsupported for now.
fn fp_valid(fpop: u8, is_double: bool, q: bool) -> bool {
    let not_d_or_q = !is_double || q; // the "size==3 && !is_q" guard
    match fpop {
        0x2f | 0x6f => not_d_or_q,                  // FABS / FNEG
        0x7f => not_d_or_q,                          // FSQRT
        0x1d | 0x5d => not_d_or_q,                   // SCVTF / UCVTF
        0x2c | 0x2d | 0x2e | 0x6c | 0x6d => not_d_or_q, // FCM{GT,EQ,LT,GE,LE} zero
        0x1a | 0x1b | 0x3a | 0x3b | 0x5a | 0x5b | 0x7a | 0x7b => not_d_or_q, // FCVT[NMPZ][SU]
        0x1c | 0x5c => not_d_or_q,                   // FCVTAS / FCVTAU
        // FCVTN/FCVTL: only the double<->single forms (skip FP16 narrowing).
        0x16 => is_double,                            // FCVTN (double -> single)
        0x17 => is_double,                            // FCVTL (single -> double)
        0x18 | 0x19 | 0x38 | 0x39 | 0x58 | 0x59 | 0x79 => not_d_or_q, // FRINT[NMPZAXI]
        _ => false,
    }
}
