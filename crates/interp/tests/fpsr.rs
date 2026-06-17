//! FPSR cumulative exception flags — validated against the IEEE-754 / ARM spec
//! directly (these flags are well-defined, so no Unicorn oracle is needed).

use aarch64_cpu_state::CpuState;
use aarch64_interp::{step, GuestMem, Memory};

// FPSR exception bits.
const IOC: u64 = 1 << 0;
const DZC: u64 = 1 << 1;
const OFC: u64 = 1 << 2;
const IXC: u64 = 1 << 4;

const PD: u32 = 0b01; // ptype = double

// FP data-processing (2 source): FMUL=0000 FDIV=0001 FADD=0010 FSUB=0011.
fn fp2(opcode: u32, rm: u32, rn: u32, rd: u32) -> u32 {
    0x1E20_0000 | (PD << 22) | (rm << 16) | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
}
// FP data-processing (1 source): FSQRT = opcode 000011.
fn fp1(opcode: u32, rn: u32, rd: u32) -> u32 {
    0x1E20_0000 | (PD << 22) | (opcode << 15) | (0b10000 << 10) | (rn << 5) | rd
}
// FCVTZS Wd, Dn (sf=0, ptype=D, rmode=11 toward-zero, opcode=000).
fn fcvtzs_w(rn: u32, rd: u32) -> u32 {
    0x1E20_0000 | (PD << 22) | (0b11 << 19) | (rn << 5) | rd
}

/// Run one instruction with D1/D2 preset, return the resulting FPSR.
fn run1(word: u32, d1: f64, d2: f64) -> u64 {
    let mut mem = Memory::new(0, 0x1000);
    mem.write(0, &word.to_le_bytes());
    let mut cpu = CpuState::new();
    cpu.v[1] = u128::from(d1.to_bits());
    cpu.v[2] = u128::from(d2.to_bits());
    assert!(matches!(step(&mut cpu, &mut mem), aarch64_interp::Step::Next(_)), "decoded+ran");
    cpu.fpsr
}

#[test]
fn fdiv_by_zero_raises_dzc_only() {
    let f = run1(fp2(0b0001, 2, 1, 0), 1.0, 0.0); // FDIV D0, D1, D2
    assert_eq!(f & DZC, DZC, "divide-by-zero");
    assert_eq!(f & IOC, 0, "not invalid");
}

#[test]
fn zero_over_zero_raises_ioc() {
    let f = run1(fp2(0b0001, 2, 1, 0), 0.0, 0.0);
    assert_eq!(f & IOC, IOC, "0/0 is invalid");
    assert_eq!(f & DZC, 0, "not divide-by-zero");
}

#[test]
fn inexact_division_raises_ixc() {
    let f = run1(fp2(0b0001, 2, 1, 0), 1.0, 3.0); // 1/3 not representable
    assert_eq!(f & IXC, IXC);
    assert_eq!(f & (IOC | DZC | OFC), 0);
}

#[test]
fn exact_add_raises_nothing() {
    assert_eq!(run1(fp2(0b0010, 2, 1, 0), 2.0, 2.0), 0, "2+2 is exact");
}

#[test]
fn inexact_add_raises_ixc() {
    let f = run1(fp2(0b0010, 2, 1, 0), 0.1, 0.2); // 0.1+0.2 rounds
    assert_eq!(f & IXC, IXC);
}

#[test]
fn overflow_raises_ofc_and_ixc() {
    let f = run1(fp2(0b0000, 2, 1, 0), 1e308, 1e308); // FMUL -> +inf
    assert_eq!(f & OFC, OFC);
    assert_eq!(f & IXC, IXC, "overflow implies inexact");
}

#[test]
fn sqrt_of_negative_raises_ioc() {
    let f = run1(fp1(0b000011, 1, 0), -1.0, 0.0);
    assert_eq!(f & IOC, IOC);
}

#[test]
fn sqrt_exact_raises_nothing() {
    assert_eq!(run1(fp1(0b000011, 1, 0), 4.0, 0.0), 0, "sqrt(4)=2 exact");
}

#[test]
fn fcvtzs_of_nan_raises_ioc() {
    let f = run1(fcvtzs_w(1, 0), f64::NAN, 0.0);
    assert_eq!(f & IOC, IOC);
}

#[test]
fn fcvtzs_with_fraction_raises_ixc() {
    let f = run1(fcvtzs_w(1, 0), 1.5, 0.0); // truncates to 1
    assert_eq!(f & IXC, IXC);
    assert_eq!(f & IOC, 0);
}

#[test]
fn flags_are_cumulative_across_ops() {
    // First an inexact divide, then a divide-by-zero — both bits stay set.
    let mut mem = Memory::new(0, 0x1000);
    mem.write(0, &fp2(0b0001, 2, 1, 0).to_le_bytes()); // FDIV D0,D1,D2
    mem.write(4, &fp2(0b0001, 3, 1, 0).to_le_bytes()); // FDIV D0,D1,D3
    let mut cpu = CpuState::new();
    cpu.v[1] = u128::from(1.0f64.to_bits());
    cpu.v[2] = u128::from(3.0f64.to_bits()); // inexact
    cpu.v[3] = u128::from(0.0f64.to_bits()); // div by zero
    step(&mut cpu, &mut mem);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.fpsr & IXC, IXC, "inexact retained");
    assert_eq!(cpu.fpsr & DZC, DZC, "divide-by-zero added");
}
