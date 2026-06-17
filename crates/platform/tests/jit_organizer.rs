//! The Machine is the execution organizer; with the JIT enabled it runs
//! register-only blocks through the JIT backend and interprets everything else.
//! This checks that an ALU loop produces identical state with the JIT on and off
//! (the interpreter is the reference), exercising the organizer's compiled-block
//! path and register-image sync.

use aarch64_cpu_state::CpuState;
use aarch64_interp::Memory;
use aarch64_platform::{Bus, Gic, Machine};

const BASE: u64 = 0x4000_0000;

/// `movz x0,#10; loop: sub x0,x0,#1; cbnz x0,loop; nop` — a register-only loop
/// (the `sub`+`cbnz` body is exactly the kind of block the JIT compiles). The
/// loop falls through to the trailing `nop` when `x0` hits 0.
const PROGRAM: [u32; 4] = [0xD280_0140, 0xD100_0400, 0xB5FF_FFE0, 0xD503_201F];

fn machine(jit: bool) -> Machine {
    let mut ram = Memory::new(BASE, 0x1_0000);
    for (i, w) in PROGRAM.iter().enumerate() {
        ram.write(BASE + 4 * i as u64, &w.to_le_bytes());
    }
    let mut cpu = CpuState::new(); // MMU off: VA == PA
    cpu.pc = BASE;
    let mut m = Machine::new(cpu, Bus::new(ram), Gic::new());
    if jit {
        m.enable_jit();
    }
    m
}

#[test]
fn jit_organizer_matches_interpreter() {
    // Stop at the trailing nop (the loop's fall-through target once x0 == 0).
    let until = BASE + 12;

    let mut interp = machine(false);
    interp.run(until, 0);

    let mut jit = machine(true);
    jit.run(until, 0);

    assert_eq!(interp.cpu.x[0], 0, "loop should count x0 down to 0");
    assert_eq!(jit.cpu.x, interp.cpu.x, "X registers diverge (jit vs interp)");
    assert_eq!(jit.cpu.pc, interp.cpu.pc, "PC diverges");
    assert_eq!(jit.cpu.flags.to_nzcv(), interp.cpu.flags.to_nzcv(), "NZCV diverges");
    // Same number of guest instructions retired either way (1 movz + 10×(sub+cbnz)).
    assert_eq!(jit.total_insns(), interp.total_insns(), "instruction count diverges");
    assert_eq!(jit.total_insns(), 21);
}
