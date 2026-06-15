//! Milestone 5: the block dispatcher (`Vm::run`) over multi-block programs,
//! cross-checked against the interpreter's `run()` for identical final state and
//! stop reason — including loops (cache reuse) and `until`/`count` boundaries.

use aarch64_cpu_state::{CpuState, GuestRegs};
use aarch64_interp::{run, Memory, StopReason};
use aarch64_jit::Vm;

const BASE: u64 = 0x1000;
const RAM: usize = 0x10000;

/// Run `prog` from `init` through both engines and assert identical results.
fn run_both(prog: &[u32], init: &GuestRegs, until: u64, count: usize, expect: StopReason) {
    // Interpreter.
    let mut cpu = CpuState::new();
    cpu.load_guest_regs(init);
    let mut mem = Memory::new(BASE, RAM);
    for (i, w) in prog.iter().enumerate() {
        mem.write(BASE + 4 * i as u64, &w.to_le_bytes());
    }
    let istop = run(&mut cpu, &mut mem, until, count);

    // JIT dispatcher.
    let mut vm = Vm::new(BASE, RAM);
    for (i, w) in prog.iter().enumerate() {
        vm.write_ram(BASE + 4 * i as u64, &w.to_le_bytes());
    }
    vm.load_regs(init);
    let jstop = vm.run(until, count);

    assert_eq!(jstop, istop, "stop reason");
    assert_eq!(jstop, expect, "stop reason vs expected");
    assert_eq!(vm.store_regs(), cpu.to_guest_regs(), "register state");
}

fn init(x0: u64, x30: u64) -> GuestRegs {
    GuestRegs { pc: BASE, x: { let mut x = [0; 31]; x[0] = x0; x[30] = x30; x }, ..GuestRegs::default() }
}

/// A countdown loop: acc += counter; counter--; b.ne loop; ret. Exercises a
/// backward branch (block-cache reuse) across many iterations.
#[test]
fn dispatch_loop_sums() {
    let prog = [
        0x8B000021u32, // add  x1, x1, x0
        0xF1000400,    // subs x0, x0, #1
        0x54FFFFC1,    // b.ne loop (-8 -> 0x1000)
        0xD65F03C0,    // ret
    ];
    let until = BASE + 4 * prog.len() as u64; // 0x1010
    // x30 = until so RET lands exactly on the stop address.
    run_both(&prog, &init(5, until), until, 0, StopReason::UntilReached);
}

/// `until` falls in the middle of the code: stop before executing it.
#[test]
fn dispatch_until_midcode() {
    let prog = [
        0xD2800020u32, // movz x0, #1
        0xD2800041,    // movz x1, #2
        0xD2800062,    // movz x2, #3
        0xD65F03C0,    // ret
    ];
    // Stop before the third movz: only x0, x1 should be set.
    run_both(&prog, &init(0, 0), BASE + 8, 0, StopReason::UntilReached);
}

/// An instruction `count` cap stops mid-program regardless of `until`.
#[test]
fn dispatch_count_cap() {
    let prog = [
        0xD2800020u32, // movz x0, #1
        0xD2800041,    // movz x1, #2
        0xD2800062,    // movz x2, #3
        0xD65F03C0,    // ret
    ];
    // Run exactly two instructions.
    run_both(&prog, &init(0, 0), BASE + 4 * prog.len() as u64, 2, StopReason::CountReached);
}
