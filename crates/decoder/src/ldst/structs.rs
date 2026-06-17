//! Advanced SIMD load/store structures: LD1-4 / ST1-4 (multiple registers and
//! single structure / replicate). bit24 selects multiple (0) vs single (1).

use crate::bits::field;
use crate::insn::Insn;

pub(crate) fn decode(word: u32) -> Insn {
    if field(word, 24, 1) == 0 {
        multiple(word)
    } else {
        single(word)
    }
}

fn multiple(word: u32) -> Insn {
    if field(word, 31, 1) == 1 || field(word, 21, 1) == 1 {
        return Insn::Unsupported { word };
    }
    let postidx = field(word, 23, 1) == 1;
    let rm = field(word, 16, 5) as u8;
    if !postidx && rm != 0 {
        return Insn::Unsupported { word };
    }
    let (rpt, selem) = match field(word, 12, 4) {
        0x0 => (1u8, 4u8), // LD4/ST4
        0x2 => (4, 1),     // LD1/ST1 (4 regs)
        0x4 => (1, 3),     // LD3/ST3
        0x6 => (3, 1),     // LD1/ST1 (3 regs)
        0x7 => (1, 1),     // LD1/ST1 (1 reg)
        0x8 => (1, 2),     // LD2/ST2
        0xa => (2, 1),     // LD1/ST1 (2 regs)
        _ => return Insn::Unsupported { word },
    };
    let size = field(word, 10, 2) as u8;
    let q = field(word, 30, 1) == 1;
    if size == 3 && !q && selem != 1 {
        return Insn::Unsupported { word };
    }
    Insn::SimdLdStMulti {
        is_load: field(word, 22, 1) == 1,
        q,
        postidx,
        rm,
        rn: field(word, 5, 5) as u8,
        rt: field(word, 0, 5) as u8,
        size,
        rpt,
        selem,
    }
}

fn single(word: u32) -> Insn {
    if field(word, 31, 1) == 1 {
        return Insn::Unsupported { word };
    }
    let postidx = field(word, 23, 1) == 1;
    let rm = field(word, 16, 5) as u8;
    if !postidx && rm != 0 {
        return Insn::Unsupported { word };
    }
    let size = field(word, 10, 2);
    let s_bit = field(word, 12, 1);
    let opc = field(word, 13, 3);
    let r = field(word, 21, 1);
    let is_load = field(word, 22, 1) == 1;
    let q = field(word, 30, 1) == 1;

    let mut scale = opc >> 1; // opc[2:1]
    let selem = (((opc & 1) << 1) | r) + 1;
    let mut replicate = false;
    let mut index = (u32::from(q) << 3) | (s_bit << 2) | size;

    match scale {
        3 => {
            if !is_load || s_bit == 1 {
                return Insn::Unsupported { word };
            }
            replicate = true;
            scale = size;
        }
        0 => {}
        1 => {
            if size & 1 == 1 {
                return Insn::Unsupported { word };
            }
            index >>= 1;
        }
        2 => {
            if size & 2 != 0 {
                return Insn::Unsupported { word };
            }
            if size & 1 == 0 {
                index >>= 2;
            } else {
                if s_bit == 1 {
                    return Insn::Unsupported { word };
                }
                index >>= 3;
                scale = 3;
            }
        }
        _ => unreachable!(),
    }

    Insn::SimdLdStSingle {
        is_load,
        replicate,
        postidx,
        rm,
        rn: field(word, 5, 5) as u8,
        rt: field(word, 0, 5) as u8,
        size: scale as u8,
        selem: selem as u8,
        index: index as u8,
        q,
    }
}
