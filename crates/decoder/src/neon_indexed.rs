//! Advanced SIMD vector x indexed element. Implemented: MUL/MLA/MLS,
//! S/U MLAL/MLSL/MULL, SQDMULL/SQDMLAL/SQDMLSL, SQDMULH/SQRDMULH, and the FP
//! FMLA/FMLS/FMUL/FMULX. Skipped (FEAT-gated): SDOT/UDOT, FCMLA, FMLAL/FMLSL,
//! SQRDMLAH/SQRDMLSH and all FP16 forms.

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    let q = field(word, 30, 1) == 1;
    let u = field(word, 29, 1);
    let size = field(word, 22, 2) as u8;
    let l = field(word, 21, 1);
    let m = field(word, 20, 1);
    let rm4 = field(word, 16, 4);
    let opcode = field(word, 12, 4) as u8;
    let h = field(word, 11, 1);
    let rn = field(word, 5, 5) as u8;
    let rd = field(word, 0, 5) as u8;

    let key = 16 * u + u32::from(opcode); // 16*U + opcode
    let is_fp = matches!(key, 0x01 | 0x05 | 0x09 | 0x19);

    // Element size validity.
    let size_ok = if is_fp {
        size == 2 || size == 3 // single / double (FP16 skipped)
    } else {
        size == 1 || size == 2 // H / S
    };
    let allocated = matches!(
        key,
        0x08 | 0x10 | 0x14            // MUL / MLA / MLS
        | 0x02 | 0x12 | 0x06 | 0x16 | 0x0a | 0x1a // MLAL/MLSL/MULL (S/U)
        | 0x03 | 0x07 | 0x0b          // SQDMLAL / SQDMLSL / SQDMULL
        | 0x0c | 0x0d                 // SQDMULH / SQRDMULH
        | 0x01 | 0x05 | 0x09 | 0x19   // FMLA / FMLS / FMUL / FMULX
    );
    if !allocated || !size_ok {
        return Insn::Unsupported { word };
    }

    // Index and the (possibly 5-bit) Rm depend on the element size.
    let (index, rm) = match size {
        1 => ((h << 2) | (l << 1) | m, rm4), // MO_16
        2 => ((h << 1) | l, rm4 | (m << 4)), // MO_32
        _ => {
            // MO_64: needs L==0 and Q==1.
            if l == 1 || !q {
                return Insn::Unsupported { word };
            }
            (h, rm4 | (m << 4))
        }
    };

    Insn::SimdIndexed {
        q,
        u: u == 1,
        size,
        opcode,
        index: index as u8,
        rm: rm as u8,
        rn,
        rd,
    }
}
