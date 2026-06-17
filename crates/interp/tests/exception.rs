//! Exception entry/return self-consistency (SVC vectors to EL1, ERET returns).
//! Unicorn intercepts SVC rather than vectoring, so this is validated against
//! the ARM spec directly; the register round-trips are checked vs Unicorn in
//! the difftest oracle tests.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;
use aarch64_interp::{run, Memory, StopReason};

const CODE: u64 = 0x1000;
const VBAR: u64 = 0x2000;
const EL0_SP: u64 = 0x7000;
const EL1_SP: u64 = 0x8000;

fn vbar_el1() -> u32 {
    sysreg_key(3, 0, 12, 0, 0)
}
fn elr_el1() -> u32 {
    sysreg_key(3, 0, 4, 0, 1)
}
fn spsr_el1() -> u32 {
    sysreg_key(3, 0, 4, 0, 0)
}
fn esr_el1() -> u32 {
    sysreg_key(3, 0, 5, 2, 0)
}
fn far_el1() -> u32 {
    sysreg_key(3, 0, 6, 0, 0)
}

#[test]
fn svc_then_eret_roundtrips() {
    let mut mem = Memory::new(0, 0x10000);
    mem.write(CODE, &0xD400_0001u32.to_le_bytes()); // svc #0
    mem.write(VBAR + 0x400, &0xD69F_03E0u32.to_le_bytes()); // eret

    let mut cpu = CpuState::new();
    cpu.el = 0;
    cpu.spsel = false; // EL0t
    cpu.pc = CODE;
    cpu.sp = EL0_SP;
    cpu.sp_el[1] = EL1_SP; // banked SP_EL1
    cpu.sysregs.insert(vbar_el1(), VBAR);

    // Take the SVC: vector to EL1 at VBAR + 0x400 (lower-EL synchronous).
    assert_eq!(run(&mut cpu, &mut mem, VBAR + 0x400, 0), StopReason::UntilReached);
    assert_eq!(cpu.pc, VBAR + 0x400);
    assert_eq!(cpu.el, 1);
    assert!(cpu.spsel);
    assert_eq!(cpu.sp, EL1_SP, "switched to SP_EL1");
    assert_eq!(cpu.sysregs.get(&elr_el1()), Some(&(CODE + 4)), "ELR_EL1 = next PC");
    assert_eq!(cpu.sysregs.get(&spsr_el1()), Some(&0), "SPSR_EL1 = EL0t PSTATE");
    assert_eq!(cpu.sysregs.get(&esr_el1()), Some(&0x5600_0000), "ESR: EC=0x15, IL=1");
    assert_eq!(cpu.daif, 0b1111, "interrupts masked on entry");

    // Return: ERET restores PC from ELR and PSTATE from SPSR (back to EL0).
    assert_eq!(run(&mut cpu, &mut mem, CODE + 4, 0), StopReason::UntilReached);
    assert_eq!(cpu.pc, CODE + 4);
    assert_eq!(cpu.el, 0);
    assert!(!cpu.spsel);
    assert_eq!(cpu.sp, EL0_SP, "restored SP_EL0");
}

#[test]
fn unaligned_exclusive_takes_alignment_abort() {
    // Exclusive/atomic accesses must be naturally aligned regardless of SCTLR.A;
    // an unaligned address takes an Alignment Data Abort (DFSC = 0b100001).
    let mut mem = Memory::new(0, 0x10000);
    mem.write(CODE, &0xC85F_7C01u32.to_le_bytes()); // ldxr x1, [x0]

    let mut cpu = CpuState::new(); // EL1h, IRQs unmasked
    cpu.pc = CODE;
    cpu.x[0] = 0x1004; // 8-byte access, not 8-byte aligned
    cpu.sysregs.insert(vbar_el1(), VBAR);

    // Same-EL synchronous exceptions vector to VBAR + 0x200.
    assert_eq!(run(&mut cpu, &mut mem, VBAR + 0x200, 0), StopReason::UntilReached);
    assert_eq!(cpu.pc, VBAR + 0x200);
    assert_eq!(cpu.sysregs.get(&far_el1()), Some(&0x1004), "FAR = faulting VA");
    let esr = *cpu.sysregs.get(&esr_el1()).unwrap();
    assert_eq!(esr >> 26, 0x25, "EC = Data Abort, same EL");
    assert_eq!(esr & 0x3f, 0x21, "DFSC = Alignment fault");
    assert_eq!((esr >> 6) & 1, 0, "WnR = 0 for a load");
    assert_eq!(cpu.x[1], 0, "destination register untouched on fault");
    assert!(cpu.excl.is_none(), "monitor not armed on a faulting LDXR");
}
