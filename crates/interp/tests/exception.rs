//! Exception entry/return self-consistency (SVC vectors to EL1, ERET returns).
//! Unicorn intercepts SVC rather than vectoring, so this is validated against
//! the ARM spec directly; the register round-trips are checked vs Unicorn in
//! the difftest oracle tests.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;
use aarch64_interp::{run, undefined, GuestMem, Memory, StopReason};

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

#[test]
fn instruction_fetch_fault_takes_inst_abort() {
    // MMU on, but the page holding PC is unmapped (invalid L1 descriptor), so the
    // instruction *fetch* itself faults -> Instruction Abort to EL1. This is the
    // demand-paged-user-text path: the handler maps the page and ERET retries.
    const FAULT_VA: u64 = 0x1_0000; // L0/L1 index 0 (VA < 1 GiB), so it hits L1[0]
    let mut mem = Memory::new(0, 0x1_0000);
    // TTBR0 -> L0 @ 0x3000; L0[0] -> L1 @ 0x4000 (valid); L1[0] invalid (bit0=0).
    mem.write_u64(0x3000, 0x4000 | 0b11);
    mem.write_u64(0x4000, 0x0000);

    let mut cpu = CpuState::new();
    cpu.el = 0;
    cpu.spsel = false; // EL0t (lower EL)
    cpu.ttbr0_el1 = 0x3000;
    cpu.tcr_el1 = 16; // T0SZ=16 (48-bit VA)
    cpu.sctlr_el1 = 1; // SCTLR_EL1.M = MMU on
    cpu.sp_el[1] = EL1_SP;
    cpu.pc = FAULT_VA;
    cpu.sysregs.insert(vbar_el1(), VBAR);

    // Lower-EL synchronous exceptions vector to VBAR + 0x400.
    assert_eq!(run(&mut cpu, &mut mem, VBAR + 0x400, 0), StopReason::UntilReached);
    assert_eq!(cpu.pc, VBAR + 0x400);
    assert_eq!(cpu.el, 1);
    assert!(cpu.spsel, "entered EL1h");
    assert_eq!(cpu.sp, EL1_SP, "switched to SP_EL1");
    assert_eq!(cpu.sysregs.get(&far_el1()), Some(&FAULT_VA), "FAR = faulting fetch VA");
    assert_eq!(cpu.sysregs.get(&elr_el1()), Some(&FAULT_VA), "ELR = faulting PC (retry on ERET)");
    let esr = *cpu.sysregs.get(&esr_el1()).unwrap();
    assert_eq!(esr >> 26, 0x20, "EC = Instruction Abort, lower EL");
    assert_eq!(esr & 0x3f, 0x05, "IFSC = translation fault, level 1");
    assert_eq!(cpu.daif, 0b1111, "interrupts masked on entry");
}

#[test]
fn undefined_instruction_vectors_to_el1() {
    // An unallocated/unimplemented encoding is delivered (by the machine loop) as
    // a synchronous "Unknown reason" exception (ESR.EC = 0) — the path the kernel
    // turns into SIGILL for a userspace process. `undefined` returns the vector PC
    // (the caller assigns it to `cpu.pc`); it does not branch itself.
    let mut cpu = CpuState::new(); // EL1h
    cpu.sysregs.insert(vbar_el1(), VBAR);

    let vector = undefined(&mut cpu, CODE);

    // Same-EL (SP_ELx) synchronous exception -> VBAR + 0x200.
    assert_eq!(vector, VBAR + 0x200);
    assert_eq!(cpu.el, 1);
    assert!(cpu.spsel);
    assert_eq!(cpu.sysregs.get(&elr_el1()), Some(&CODE), "ELR = the offending instruction");
    let esr = *cpu.sysregs.get(&esr_el1()).unwrap();
    assert_eq!(esr >> 26, 0x00, "EC = Unknown reason");
    assert_eq!((esr >> 25) & 1, 1, "IL = 1");
    assert_eq!(esr & 0x1ff_ffff, 0, "ISS = 0");
    assert_eq!(cpu.daif, 0b1111, "interrupts masked on entry");
}
