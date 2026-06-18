//! End-to-end asynchronous IRQ injection through the `Machine` loop: vectoring
//! through VBAR_EL1, masking semantics, and ERET return.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;
use aarch64_interp::{GuestMem, Memory};
use aarch64_platform::{Bus, Gic, Machine};

const GICD: u64 = 0x0800_0000;
const GICC: u64 = 0x0801_0000;

const RAM_BASE: u64 = 0x4000_0000;
const MAIN: u64 = 0x4000_0000;
const VBAR: u64 = 0x4000_2000;
/// Current-EL-with-SP_ELx IRQ vector = group 0x200 + type 0x80.
const IRQ_VECTOR: u64 = VBAR + 0x280;

const IRQ: u32 = 33;
const PSTATE_I: u8 = 0b0010;

// Instruction encodings.
const NOP: u32 = 0xd503_201f;
const B_SELF: u32 = 0x1400_0000; // B . (branch to itself)
const ERET: u32 = 0xd69f_03e0;

fn vbar_key() -> u32 {
    sysreg_key(3, 0, 12, 0, 0)
}
fn elr_key() -> u32 {
    sysreg_key(3, 0, 4, 0, 1)
}

/// Build a machine with the GIC mapped, IRQ 33 enabled and unmasked at the
/// controller, and VBAR_EL1 set. The caller writes the code and arms `set_pending`.
fn setup() -> Machine {
    let gic = Gic::new();
    let mut bus = Bus::new(Memory::new(RAM_BASE, 0x1_0000));
    bus.map(GICD, 0x10000, Box::new(gic.distributor()));
    bus.map(GICC, 0x10000, Box::new(gic.cpu_interface()));
    bus.write_u32(GICD + 0x104, 1 << 1);
    bus.write_u32(GICD + 0x000, 1);
    bus.write_u32(GICC + 0x000, 1);
    bus.write_u32(GICC + 0x004, 0xF0);

    let mut cpu = CpuState::new(); // EL1h, DAIF clear
    cpu.pc = MAIN;
    cpu.sysregs.insert(vbar_key(), VBAR);

    Machine::new(cpu, bus, gic)
}

fn write_insn(bus: &mut Bus, addr: u64, insn: u32) {
    bus.ram_mut().write(addr, &insn.to_le_bytes());
}

#[test]
fn pending_irq_vectors_through_vbar_and_masks() {
    let mut m = setup();
    write_insn(&mut m.bus, MAIN, B_SELF); // main spins
    write_insn(&mut m.bus, IRQ_VECTOR, B_SELF); // handler spins
    m.gic.set_pending(IRQ);

    m.step();

    assert_eq!(m.cpu.pc, IRQ_VECTOR, "vectored to the IRQ slot");
    assert_eq!(m.cpu.sysregs[&elr_key()], MAIN, "ELR holds the interrupted PC");
    assert_ne!(m.cpu.daif & PSTATE_I, 0, "IRQ masked on entry");
}

#[test]
fn masked_irq_is_not_taken() {
    let mut m = setup();
    write_insn(&mut m.bus, MAIN, B_SELF);
    m.cpu.daif = PSTATE_I; // I masked
    m.gic.set_pending(IRQ);

    m.step();

    assert_eq!(m.cpu.pc, MAIN, "stayed in main; IRQ not delivered while masked");
}

#[test]
fn eret_returns_to_interrupted_instruction() {
    let mut m = setup();
    write_insn(&mut m.bus, MAIN, NOP); // 0x..00: NOP
    write_insn(&mut m.bus, MAIN + 4, B_SELF); // 0x..04: spin
    write_insn(&mut m.bus, IRQ_VECTOR, ERET); // handler immediately returns
    m.gic.set_pending(IRQ);

    // Vector to the handler and execute its ERET.
    m.step();
    assert_eq!(m.cpu.pc, MAIN, "ERET restored the interrupted PC");
    assert_eq!(m.cpu.daif & PSTATE_I, 0, "ERET restored unmasked PSTATE.I");

    // The device's line is still asserted; simulate the handler servicing it.
    m.gic.clear_pending(IRQ);

    // Now main runs undisturbed: the NOP advances to the spin.
    m.step();
    assert_eq!(m.cpu.pc, MAIN + 4);
}

/// GICD_SGIR offset within the distributor.
const GICD_SGIR: u64 = 0xF00;

#[test]
fn sgir_self_delivers_software_interrupt() {
    let mut m = setup();
    write_insn(&mut m.bus, MAIN, B_SELF); // main spins
    write_insn(&mut m.bus, IRQ_VECTOR, B_SELF); // handler spins
    // TargetListFilter = 0b10 (requester == CPU0), SGI ID 1. SGIs are always
    // enabled, so this alone must make the SGI pending and deliverable.
    m.bus.write_u32(GICD + GICD_SGIR, (0b10 << 24) | 1);

    m.step();

    assert_eq!(m.cpu.pc, IRQ_VECTOR, "SGI vectored to the IRQ slot");
    assert_eq!(m.bus.read_u32(GICC + 0x0C), 1, "GICC_IAR returns the SGI ID");
}

#[test]
fn sgir_all_but_self_is_dropped_on_single_core() {
    let mut m = setup();
    write_insn(&mut m.bus, MAIN, B_SELF);
    // TargetListFilter = 0b01 (all CPUs but the requester); there is no other
    // core, so nothing is delivered and main keeps spinning.
    m.bus.write_u32(GICD + GICD_SGIR, (0b01 << 24) | 1);

    m.step();

    assert_eq!(m.cpu.pc, MAIN, "no other core to target; SGI dropped");
}
