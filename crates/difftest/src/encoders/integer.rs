//! Encoders for integer data-processing classes (immediate and register).

use super::{bit, reg};
use crate::fuzz::Class;
use crate::rng::Rng;

pub(super) fn classes() -> Vec<Class> {
    vec![
        Class { name: "move_wide", encode: move_wide },
        Class { name: "add_sub_imm", encode: add_sub_imm },
        Class { name: "logical_imm", encode: logical_imm },
        Class { name: "bitfield", encode: bitfield },
        Class { name: "extract", encode: extract },
        Class { name: "add_sub_shifted_reg", encode: add_sub_shifted_reg },
        Class { name: "add_sub_ext_reg", encode: add_sub_ext_reg },
        Class { name: "add_sub_carry", encode: add_sub_carry },
        Class { name: "logical_reg", encode: logical_reg },
        Class { name: "cond_select", encode: cond_select },
        Class { name: "cond_compare", encode: cond_compare },
        Class { name: "data_proc_1src", encode: data_proc_1src },
        Class { name: "data_proc_2src", encode: data_proc_2src },
        Class { name: "data_proc_3src", encode: data_proc_3src },
        Class { name: "pc_rel", encode: pc_rel },
    ]
}

fn move_wide(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    let opc = [0u32, 2, 3][rng.below(3) as usize];
    let hw = if sf == 1 { rng.below(4) } else { rng.below(2) };
    (sf << 31) | (opc << 29) | (0b100101 << 23) | (hw << 21) | (rng.bits(16) << 5) | reg(rng)
}

fn add_sub_imm(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    (sf << 31)
        | (bit(rng) << 30)
        | (bit(rng) << 29)
        | (0b100010 << 23)
        | (bit(rng) << 22)
        | (rng.bits(12) << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn logical_imm(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    let n = if sf == 1 { bit(rng) } else { 0 };
    (sf << 31)
        | (rng.below(4) << 29)
        | (0b100100 << 23)
        | (n << 22)
        | (rng.bits(6) << 16)
        | (rng.bits(6) << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn bitfield(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    let bound = if sf == 1 { 64 } else { 32 };
    (sf << 31)
        | (rng.below(3) << 29)
        | (0b100110 << 23)
        | (sf << 22)
        | (rng.below(bound) << 16)
        | (rng.below(bound) << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn extract(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    let bound = if sf == 1 { 64 } else { 32 };
    (sf << 31)
        | (0b100111 << 23)
        | (sf << 22)
        | (reg(rng) << 16)
        | (rng.below(bound) << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn add_sub_shifted_reg(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    let amount = if sf == 1 { rng.below(64) } else { rng.below(32) };
    (sf << 31)
        | (bit(rng) << 30)
        | (bit(rng) << 29)
        | (0b01011 << 24)
        | (rng.below(3) << 22)
        | (reg(rng) << 16)
        | (amount << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn add_sub_ext_reg(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    (sf << 31)
        | (bit(rng) << 30)
        | (bit(rng) << 29)
        | (0b01011 << 24)
        | (1 << 21)
        | (reg(rng) << 16)
        | (rng.below(8) << 13)
        | (rng.below(5) << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn add_sub_carry(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    (sf << 31)
        | (bit(rng) << 30)
        | (bit(rng) << 29)
        | (0b11010000 << 21)
        | (reg(rng) << 16)
        | (reg(rng) << 5)
        | reg(rng)
}

fn logical_reg(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    let amount = if sf == 1 { rng.below(64) } else { rng.below(32) };
    (sf << 31)
        | (rng.below(4) << 29)
        | (0b01010 << 24)
        | (rng.below(4) << 22)
        | (bit(rng) << 21)
        | (reg(rng) << 16)
        | (amount << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn cond_select(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    (sf << 31)
        | (bit(rng) << 30)
        | (0b11010100 << 21)
        | (reg(rng) << 16)
        | (rng.bits(4) << 12)
        | (bit(rng) << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn cond_compare(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    let is_imm = bit(rng);
    (sf << 31)
        | (bit(rng) << 30)
        | (1 << 29)
        | (0b11010010 << 21)
        | (rng.bits(5) << 16) // Rm or imm5
        | (rng.bits(4) << 12)
        | (is_imm << 11)
        | (reg(rng) << 5)
        | rng.bits(4) // nzcv
}

fn data_proc_1src(rng: &mut Rng) -> u32 {
    let sf = bit(rng);
    // Avoid opcode 3 (REV, 64-bit only) when sf==0.
    let opcode = if sf == 1 { rng.below(6) } else { [0u32, 1, 2, 4, 5][rng.below(5) as usize] };
    // bit 30 == 1 selects the 1-source family (0 would be 2-source).
    (sf << 31) | (1 << 30) | (0b11010110 << 21) | (opcode << 10) | (reg(rng) << 5) | reg(rng)
}

fn data_proc_2src(rng: &mut Rng) -> u32 {
    // Plain 2-src ops (random sf), and CRC32/CRC32C (sf tied to the size).
    let (sf, opcode) = if rng.below(2) == 0 {
        (bit(rng), [2u32, 3, 8, 9, 10, 11][rng.below(6) as usize])
    } else {
        let op = 0x10 + rng.below(8); // CRC32{B,H,W,X}, CRC32C{B,H,W,X}
        ((op & 3 == 3) as u32, op)
    };
    (sf << 31)
        | (0b11010110 << 21)
        | (reg(rng) << 16)
        | (opcode << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn data_proc_3src(rng: &mut Rng) -> u32 {
    // (op31, o0_allowed, needs_sf)
    let variants: [(u32, bool, bool); 5] = [
        (0b000, true, false), // MADD/MSUB
        (0b001, true, true),  // SMADDL/SMSUBL
        (0b101, true, true),  // UMADDL/UMSUBL
        (0b010, false, true), // SMULH
        (0b110, false, true), // UMULH
    ];
    let (op31, o0_allowed, needs_sf) = variants[rng.below(5) as usize];
    let sf = if needs_sf { 1 } else { bit(rng) };
    let o0 = if o0_allowed { bit(rng) } else { 0 };
    (sf << 31)
        | (0b11011 << 24)
        | (op31 << 21)
        | (reg(rng) << 16)
        | (o0 << 15)
        | (reg(rng) << 10)
        | (reg(rng) << 5)
        | reg(rng)
}

fn pc_rel(rng: &mut Rng) -> u32 {
    (bit(rng) << 31)
        | (rng.bits(2) << 29)
        | (0b10000 << 24)
        | (rng.bits(19) << 5)
        | reg(rng)
}
