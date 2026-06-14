//! Router for the "Data processing -- register" encoding group.

use crate::bits::field;
use crate::insn::Insn;
use crate::{
    add_sub_carry, add_sub_ext_reg, add_sub_shifted_reg, cond_compare, cond_select,
    data_proc_1src, data_proc_2src, data_proc_3src, logical_reg,
};

pub(crate) fn decode(word: u32) -> Insn {
    // bits [28:24] coarsely separate the families.
    match field(word, 24, 5) {
        0b01010 => logical_reg::decode(word),
        0b01011 => {
            if field(word, 21, 1) == 0 {
                add_sub_shifted_reg::decode(word)
            } else {
                add_sub_ext_reg::decode(word)
            }
        }
        0b11011 => data_proc_3src::decode(word),
        0b11010 => match field(word, 21, 3) {
            0b000 => add_sub_carry::decode(word),
            0b010 => cond_compare::decode(word),
            0b100 => cond_select::decode(word),
            0b110 => {
                if field(word, 30, 1) == 1 {
                    data_proc_1src::decode(word)
                } else {
                    data_proc_2src::decode(word)
                }
            }
            _ => Insn::Unsupported { word },
        },
        _ => Insn::Unsupported { word },
    }
}
