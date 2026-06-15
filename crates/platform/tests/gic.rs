//! GICv2 register model: enable/pending programming and the IAR/EOIR cycle,
//! all driven through the bus exactly as guest code would.

use aarch64_interp::{GuestMem, Memory};
use aarch64_platform::{Bus, Gic};

// `virt` GICv2 placement.
const GICD: u64 = 0x0800_0000;
const GICC: u64 = 0x0801_0000;

// SPI 1 == interrupt ID 33 (word 1, bit 1 in the per-32 bitmaps).
const IRQ: u32 = 33;
const SPURIOUS: u32 = 1023;

fn machine_bus() -> (Gic, Bus) {
    let gic = Gic::new();
    let mut bus = Bus::new(Memory::new(0x4000_0000, 0x1000));
    bus.map(GICD, 0x10000, Box::new(gic.distributor()));
    bus.map(GICC, 0x10000, Box::new(gic.cpu_interface()));
    (gic, bus)
}

/// Enable IRQ 33, enable both CTLRs, open the priority mask. Priority defaults
/// to 0 (highest), so it passes any non-zero PMR.
fn program_baseline(bus: &mut Bus) {
    bus.write_u32(GICD + 0x104, 1 << 1); // GICD_ISENABLER1, bit for IRQ 33
    bus.write_u32(GICD + 0x000, 1); // GICD_CTLR enable
    bus.write_u32(GICC + 0x000, 1); // GICC_CTLR enable
    bus.write_u32(GICC + 0x004, 0xF0); // GICC_PMR open
}

#[test]
fn typer_reports_interrupt_lines() {
    let (_gic, mut bus) = machine_bus();
    // ITLinesNumber for 1024 IDs == 1024/32 - 1 == 31.
    assert_eq!(bus.read_u32(GICD + 0x004) & 0x1f, 31);
}

#[test]
fn enable_bit_reads_back() {
    let (_gic, mut bus) = machine_bus();
    bus.write_u32(GICD + 0x104, 1 << 1);
    assert_eq!(bus.read_u32(GICD + 0x104) & (1 << 1), 1 << 1);
    // Clear-enable register clears the same bit.
    bus.write_u32(GICD + 0x184, 1 << 1);
    assert_eq!(bus.read_u32(GICD + 0x104) & (1 << 1), 0);
}

#[test]
fn pending_signals_then_iar_eoir_cycle() {
    let (gic, mut bus) = machine_bus();
    program_baseline(&mut bus);

    assert!(!gic.pending_irq(), "nothing pending yet");

    // A device raises the line via the set-pending register.
    bus.write_u32(GICD + 0x204, 1 << 1); // GICD_ISPENDR1
    assert!(gic.pending_irq(), "enabled+pending+unmasked should signal");

    // Acknowledge: IAR returns the ID and moves it pending->active.
    let id = bus.read_u32(GICC + 0x00C);
    assert_eq!(id, IRQ);
    assert!(!gic.pending_irq(), "active interrupt raises running priority");

    // Deactivate: EOIR drops the running priority back to idle.
    bus.write_u32(GICC + 0x010, IRQ);
    assert!(!gic.pending_irq(), "no longer pending after ack");
}

#[test]
fn masked_by_pmr() {
    let (gic, mut bus) = machine_bus();
    program_baseline(&mut bus);
    bus.write_u32(GICC + 0x004, 0); // PMR = 0 masks everything
    bus.write_u32(GICD + 0x204, 1 << 1);
    assert!(!gic.pending_irq(), "priority 0 is not < PMR 0");
    // Reading IAR while masked yields the spurious ID.
    assert_eq!(bus.read_u32(GICC + 0x00C), SPURIOUS);
}

#[test]
fn set_pending_handle_matches_register() {
    let (gic, mut bus) = machine_bus();
    program_baseline(&mut bus);
    gic.set_pending(IRQ); // peripheral-facing API
    assert!(gic.pending_irq());
    assert_eq!(bus.read_u32(GICD + 0x204) & (1 << 1), 1 << 1, "visible as pending");
}
