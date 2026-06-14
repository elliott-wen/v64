//! Interpreter integration tests using small real programs.

use aarch64_cpu_state::CpuState;
use aarch64_interp::{run, Memory, StopReason};

const CODE_START: u64 = 0x1000;

fn run_code(code: &[u8], setup: impl FnOnce(&mut CpuState)) -> CpuState {
    let mut mem = Memory::new(CODE_START, 0x1000);
    mem.write(CODE_START, code);
    let mut cpu = CpuState::new();
    cpu.pc = CODE_START;
    setup(&mut cpu);
    let stop = run(&mut cpu, &mut mem, CODE_START + code.len() as u64, 0);
    assert_eq!(stop, StopReason::UntilReached, "unexpected stop");
    cpu
}

/// Same program as Unicorn's `test_arm64_until`.
#[test]
fn arm64_until() {
    let code = &[
        0x30, 0x00, 0x80, 0xd2, // mov x16, #1
        0x11, 0x04, 0x80, 0xd2, // mov x17, #0x20
        0x9c, 0x23, 0x00, 0x91, // add x28, x28, 8
    ];
    let cpu = run_code(code, |c| {
        c.x[16] = 0x12341234;
        c.x[17] = 0x78907890;
        c.x[28] = 0x12341234;
    });
    assert_eq!(cpu.x[16], 0x1);
    assert_eq!(cpu.x[17], 0x20);
    assert_eq!(cpu.x[28], 0x1234123c);
    assert_eq!(cpu.pc, CODE_START + code.len() as u64);
}

#[test]
fn add_w_zero_extends() {
    // add w0, w0, #0x7FF  => 0x111ffc00
    let code = &[0x00, 0xfc, 0x1f, 0x11];
    let cpu = run_code(code, |c| c.x[0] = 0xffff_ffff_0000_0000);
    assert_eq!(cpu.x[0], 0x7ff, "top half must be cleared on W write");
}

#[test]
fn subs_sets_zero_flag() {
    // subs x0, x0, #1  => 0xf1000400 ; start x0=1 -> result 0, Z=1, C=1
    let code = &[0x00, 0x04, 0x00, 0xf1];
    let cpu = run_code(code, |c| c.x[0] = 1);
    assert_eq!(cpu.x[0], 0);
    assert!(cpu.flags.z);
    assert!(cpu.flags.c); // no borrow
    assert!(!cpu.flags.n);
    assert!(!cpu.flags.v);
}
