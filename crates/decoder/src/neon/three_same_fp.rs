//! Advanced SIMD three-same (floating-point): per-lane FADD/FSUB/FMUL/FDIV,
//! FMAX/FMIN/FMAXNM/FMINNM, FABD, and FCMEQ/FCMGE/FCMGT.
//!
//! The 7-bit `fpopcode` = opcode[15:11] | (bit23 << 5) | (U << 6), matching
//! QEMU's `disas_simd_3same_float`.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let fpopcode = (field(word, 11, 5) | (field(word, 23, 1) << 5) | (field(word, 29, 1) << 6)) as u8;
    let implemented = matches!(
        fpopcode,
        // element-wise arithmetic / compare
        0x1a | 0x3a | 0x5b | 0x5f | 0x1e | 0x3e | 0x18 | 0x38 | 0x7a | 0x1c | 0x5c | 0x7c
        // FMLA/FMLS, FMULX, FRECPS/FRSQRTS, FACGE/FACGT
        | 0x19 | 0x39 | 0x1b | 0x1f | 0x3f | 0x5d | 0x7d
        // pairwise FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP
        | 0x5a | 0x5e | 0x7e | 0x58 | 0x78
    );
    if !implemented {
        return Insn::Unsupported { word };
    }
    let q = field(word, 30, 1) == 1;
    let sz = field(word, 22, 1) == 1; // false = single (2S/4S), true = double (2D)
    // A 1D arrangement (double, Q=0) is reserved for vector FP.
    if sz && !q {
        return Insn::Unsupported { word };
    }
    Insn::SimdThreeSameFp {
        q,
        sz,
        fpopcode,
        rm: field(word, 16, 5) as u8,
        rn: field(word, 5, 5) as u8,
        rd: field(word, 0, 5) as u8,
    }
}
