//! Encoding-level decode tests. Instruction words are taken from `as`/objdump
//! output and cross-checked against the comments.

use aarch64_decoder::{decode, Insn};

#[test]
fn movz_x16_1() {
    // mov x16, #1  => 0xd2800030
    assert_eq!(
        decode(0xd280_0030),
        Insn::MoveWide { sf: true, opc: 2, hw: 0, imm16: 1, rd: 16 }
    );
}

#[test]
fn add_imm_x28() {
    // add x28, x28, #8  => 0x9100239c
    assert_eq!(
        decode(0x9100_239c),
        Insn::AddSubImm {
            sf: true,
            sub: false,
            set_flags: false,
            shift12: false,
            imm12: 8,
            rn: 28,
            rd: 28,
        }
    );
}

#[test]
fn add_w_imm() {
    // add w0, w0, #1 => 0x11000400  (little-endian bytes 00 04 00 11)
    assert_eq!(
        decode(0x1100_0400),
        Insn::AddSubImm {
            sf: false,
            sub: false,
            set_flags: false,
            shift12: false,
            imm12: 1,
            rn: 0,
            rd: 0,
        }
    );
}

#[test]
fn nop_decodes() {
    assert_eq!(decode(0xD503_201F), Insn::Nop);
}
