//! Router + decoders for the Advanced SIMD *scalar* families. Each instruction
//! operates on a single element; the interpreter reuses the vector arithmetic on
//! lane 0 and zeroes the rest. (bit30 == 1 already separated these from scalar
//! floating-point at the top level.)

use crate::bits::field;
use crate::insn::Insn;

fn m(word: u32, mask: u32, val: u32) -> bool {
    word & mask == val
}

pub(crate) fn decode(word: u32) -> Insn {
    // Crypto SHA shares the op0=1111/bit30=1 region but has bit29=0.
    if m(word, 0xff20_8c00, 0x5e00_0000) {
        return super::sha::three_reg(word);
    }
    if m(word, 0xff3e_0c00, 0x5e28_0800) {
        return super::sha::two_reg(word);
    }
    if m(word, 0xdf20_0c00, 0x5e20_0000) {
        return three_diff(word);
    }
    if m(word, 0xdf20_0400, 0x5e20_0400) {
        return three_same(word);
    }
    if m(word, 0xdf3e_0c00, 0x5e20_0800) {
        return two_reg_misc(word);
    }
    if m(word, 0xdf3e_0c00, 0x5e30_0800) {
        return pairwise(word);
    }
    if m(word, 0xdfe0_8400, 0x5e00_0400) {
        return copy(word);
    }
    if m(word, 0xdf00_0400, 0x5f00_0000) {
        return indexed(word);
    }
    if m(word, 0xdf80_0400, 0x5f00_0400) {
        return shift_imm(word);
    }
    Insn::Unsupported { word }
}

fn three_same(word: u32) -> Insn {
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 11, 5) as u8;
    let rm = field(word, 16, 5) as u8;
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;

    let valid = if opcode >= 0x18 {
        let fpopcode = opcode | ((size >> 1) << 5) | (u8::from(u) << 6);
        matches!(fpopcode, 0x1b | 0x1f | 0x3f | 0x5d | 0x7d | 0x1c | 0x5c | 0x7c | 0x7a)
    } else {
        match opcode {
            0x1 | 0x5 | 0x9 | 0xb => true,                  // SQADD/SQSUB/SQSHL/SQRSHL
            0x8 | 0xa | 0x6 | 0x7 | 0x11 | 0x10 => size == 3, // SSHL/SRSHL/CM*/ADD
            0x16 => size == 1 || size == 2,                 // SQDMULH/SQRDMULH
            _ => false,
        }
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdScalarThreeSame { u, size, opcode, rm, rn, rd }
}

fn two_reg_misc(word: u32) -> Insn {
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 5) as u8;
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;

    let valid = if matches!(opcode, 0xc..=0xf | 0x16..=0x1f) {
        let fpop = opcode | ((size >> 1) << 5) | (u8::from(u) << 6);
        // FCM*-zero, SCVTF/UCVTF, FCVT[NMPZA][SU] (estimates + FCVTXN skipped).
        matches!(
            fpop,
            0x2c | 0x2d | 0x2e | 0x6c | 0x6d
            | 0x1d | 0x5d
            | 0x1a | 0x1b | 0x3a | 0x3b | 0x5a | 0x5b | 0x7a | 0x7b | 0x1c | 0x5c
        )
    } else {
        match opcode {
            0x3 | 0x7 => true,                  // SUQADD/USQADD, SQABS/SQNEG
            0x8 | 0x9 | 0xb => size == 3,       // CMGT/CMGE, CMEQ/CMLE, ABS/NEG
            0xa => !u && size == 3,             // CMLT
            0x12 => u && size <= 2,             // SQXTUN
            0x14 => size <= 2,                  // SQXTN/UQXTN
            _ => false,
        }
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdScalarTwoRegMisc { u, size, opcode, rn, rd }
}

fn pairwise(word: u32) -> Insn {
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 5) as u8;
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;

    let full = opcode | ((size >> 1) << 5);
    let valid = if !u {
        opcode == 0x1b && size == 3 // ADDP (full == 0x3b)
    } else {
        matches!(full, 0xc | 0xd | 0xf | 0x2c | 0x2f) // FMAXNMP/FADDP/FMAXP/FMINNMP/FMINP
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdScalarPairwise { u, size, opcode, rn, rd }
}

fn three_diff(word: u32) -> Insn {
    let u = field(word, 29, 1) == 1;
    let size = field(word, 22, 2) as u8;
    let opcode = field(word, 12, 4) as u8;
    let rm = field(word, 16, 5) as u8;
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;

    let valid = !u && matches!(opcode, 0x9 | 0xb | 0xd) && (size == 1 || size == 2);
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdScalarThreeDiff { size, opcode, rm, rn, rd }
}

fn copy(word: u32) -> Insn {
    let imm5 = field(word, 16, 5) as u8;
    if field(word, 29, 1) != 0 || field(word, 11, 4) != 0 || imm5 & 0xf == 0 {
        return Insn::Unsupported { word };
    }
    Insn::SimdScalarCopy {
        imm5,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}

fn indexed(word: u32) -> Insn {
    let u = field(word, 29, 1);
    let size = field(word, 22, 2) as u8;
    let l = field(word, 21, 1);
    let mbit = field(word, 20, 1);
    let rm4 = field(word, 16, 4);
    let opcode = field(word, 12, 4) as u8;
    let h = field(word, 11, 1);
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;

    let key = 16 * u + u32::from(opcode);
    let is_fp = matches!(key, 0x01 | 0x05 | 0x09 | 0x19);
    let allocated = matches!(key, 0x03 | 0x07 | 0x0b | 0x0c | 0x0d | 0x01 | 0x05 | 0x09 | 0x19);
    let size_ok = if is_fp { size == 2 || size == 3 } else { size == 1 || size == 2 };
    if !allocated || !size_ok {
        return Insn::Unsupported { word };
    }
    let (index, rm) = match size {
        1 => ((h << 2) | (l << 1) | mbit, rm4),
        2 => ((h << 1) | l, rm4 | (mbit << 4)),
        _ => {
            if l == 1 {
                return Insn::Unsupported { word };
            }
            (h, rm4 | (mbit << 4))
        }
    };
    Insn::SimdScalarIndexed {
        u: u == 1,
        size,
        opcode,
        index: index as u8,
        rm: rm as u8,
        rn,
        rd,
    }
}

fn shift_imm(word: u32) -> Insn {
    let u = field(word, 29, 1) == 1;
    let immh = field(word, 19, 4) as u8;
    let immb = field(word, 16, 3) as u8;
    let opcode = field(word, 11, 5) as u8;
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;
    if immh == 0 {
        return Insn::Unsupported { word };
    }
    let d_only = immh & 0b1000 != 0; // scalar same-width shifts are D-form only
    let size_le2 = immh & 0b1000 == 0;

    let valid = match opcode {
        0b00000 | 0b00010 | 0b00100 | 0b00110 => d_only, // SSHR/SSRA/SRSHR/SRSRA
        0b01000 => u && d_only,                          // SRI
        0b01010 => d_only,                               // SHL/SLI
        0b01100 => u,                                    // SQSHLU (any size)
        0b01110 => true,                                 // SQSHL/UQSHL (any size)
        0b10000 | 0b10001 => u && size_le2,              // SQSHRUN/SQRSHRUN
        0b10010 | 0b10011 => size_le2,                   // SQSHRN/UQSHRN/SQRSHRN/UQRSHRN
        0b11100 | 0b11111 => immh & 0b1100 != 0,         // SCVTF/UCVTF, FCVTZS/U (32/64)
        _ => false,
    };
    if !valid {
        return Insn::Unsupported { word };
    }
    Insn::SimdScalarShiftImm { u, immh, immb, opcode, rn, rd }
}
