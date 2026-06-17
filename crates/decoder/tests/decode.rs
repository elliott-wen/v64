//! Encoding-level decode tests. Instruction words are taken from `as`/objdump
//! output and cross-checked against the comments.

use aarch64_decoder::{decode, AddrMode, Insn};

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

#[test]
fn wfi_wfe_decode_distinctly() {
    assert_eq!(decode(0xd503_207f), Insn::Wfi); // wfi (CRm=0, op2=3)
    assert_eq!(decode(0xd503_205f), Insn::Wfe); // wfe (CRm=0, op2=2)
    // Neighbouring hints stay NOPs: yield (op2=1), sev (op2=4).
    assert_eq!(decode(0xd503_203f), Insn::Nop);
    assert_eq!(decode(0xd503_209f), Insn::Nop);
}

#[test]
fn prfm_decodes() {
    // prfm pldl1keep, [x0, #0] => 0xf9800000  (size=11, opc=10, unsigned imm)
    assert_eq!(decode(0xf980_0000), Insn::Prfm);
}

#[test]
fn dc_zva_decodes() {
    // dc zva, x3 => 0xd50b7423  (SYS op1=3 CRn=7 CRm=4 op2=1)
    assert_eq!(decode(0xd50b_7423), Insn::DcZva { rt: 3 });
}

#[test]
fn tlbi_is_nop() {
    // tlbi vmalle1 => 0xd508871f  (SYS, no architectural effect in our model)
    assert_eq!(decode(0xd508_871f), Insn::Nop);
}

#[test]
fn brk_decodes() {
    // brk #0 => 0xd4200000 ; brk #1 => 0xd4200020
    assert_eq!(decode(0xd420_0000), Insn::Brk { imm16: 0 });
    assert_eq!(decode(0xd420_0020), Insn::Brk { imm16: 1 });
}

#[test]
fn ldapr_decodes_as_load() {
    // ldapr x1, [x0] => 0xf8bfc001  (acquire load -> plain zero-offset load)
    assert_eq!(
        decode(0xf8bf_c001),
        Insn::LoadStore {
            size: 3,
            is_load: true,
            signed: false,
            dst64: true,
            vec: false,
            unpriv: false,
            rt: 1,
            addr: AddrMode::UnsignedImm { rn: 0, imm: 0 },
        }
    );
}

#[test]
fn ldxp_stxp_decode() {
    // ldxp x1, x2, [x0] => 0xc87f0801 ; stxp w3, x1, x2, [x0] => 0xc8230801
    assert_eq!(
        decode(0xc87f_0801),
        Insn::LoadExclusivePair { size: 3, rt: 1, rt2: 2, rn: 0 }
    );
    assert_eq!(
        decode(0xc823_0801),
        Insn::StoreExclusivePair { size: 3, rs: 3, rt: 1, rt2: 2, rn: 0 }
    );
}

#[test]
fn casp_decodes() {
    // casp w0, w1, w2, w3, [x4] => 0x08207c82  (sz=0 => 32-bit pair)
    assert_eq!(decode(0x0820_7c82), Insn::CompareSwapPair { sz: 0, rs: 0, rn: 4, rt: 2 });
}

#[test]
fn fmov_vd_d1_high_half() {
    // fmov x0, v3.d[1]  => 0x9eae0060  (high 64 of V3 -> X0)
    assert_eq!(
        decode(0x9eae_0060),
        Insn::FpCvtInt { sf: true, ftype: 0b10, rmode: 0b01, opcode: 0b110, rn: 3, rd: 0 }
    );
    // fmov v0.d[1], x3  => 0x9eaf0060  (X3 -> high 64 of V0)
    assert_eq!(
        decode(0x9eaf_0060),
        Insn::FpCvtInt { sf: true, ftype: 0b10, rmode: 0b01, opcode: 0b111, rn: 3, rd: 0 }
    );
}

#[test]
fn fp_fixed_point_convert() {
    // scvtf d0, w0, #2  => 0x1e42f800 (opcode 010, scale 62 => 2 frac bits)
    assert_eq!(
        decode(0x1e42_f800),
        Insn::FpCvtFixed { sf: false, ftype: 0b01, opcode: 0b010, scale: 62, rn: 0, rd: 0 }
    );
    // fcvtzs w0, d0, #2 => 0x1e58f800 (opcode 000)
    assert_eq!(
        decode(0x1e58_f800),
        Insn::FpCvtFixed { sf: false, ftype: 0b01, opcode: 0b000, scale: 62, rn: 0, rd: 0 }
    );
}
