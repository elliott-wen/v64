//! Subsystem mini-kernels: tiny guest programs that each drive one part of the
//! machine end to end through the real boot path — MMU translation, the generic
//! timer interrupt, and a full IRQ handler lifecycle (IAR/EOIR/ERET).

mod common;

use aarch64_interp::{GuestMem, StopReason};
use aarch64_platform::{GICC_BASE, GICD_BASE, RAM_BASE, UART_BASE};
use common::*;

/// MMU: build a 1 GiB block mapping, enable translation, store through a virtual
/// address, and confirm it landed at the *aliased* physical address.
#[test]
fn mmu_translates_a_store_through_a_block_mapping() {
    // Level-1 table at RAM_BASE+0x2000. With T0SZ=25 the walk starts at level 1,
    // where each entry maps a 1 GiB block.
    let pt_base = RAM_BASE + 0x2000;
    // Block descriptor for the RAM gigabyte: output 0x4000_0000, valid (bit0),
    // block (bit1 clear), access flag set (bit10) so the walk doesn't take an
    // access-flag fault. AP=00 => EL1 read/write (the mini-kernel runs at EL1).
    let desc = RAM_BASE | 1 | (1 << 10);

    let mut a = Asm::new();
    a.load_imm64(1, pt_base);
    a.load_imm64(2, desc);
    a.ins(str64(2, 1, 8)); // L1[1]: identity-map the code/RAM gigabyte (VA 0x4xxx_xxxx)
    a.ins(str64(2, 1, 16)); // L1[2]: alias VA 0x8000_0000.. -> PA 0x4000_0000..
    a.msr(TTBR0_EL1, 1); // TTBR0 = page-table base
    a.load_imm64(3, 25);
    a.msr(TCR_EL1, 3); // T0SZ = 25 -> 39-bit VA, start at level 1
    a.load_imm64(4, 1);
    a.msr(SCTLR_EL1, 4); // SCTLR.M = 1: MMU on
    // Store a sentinel through the aliased VA; it must reach PA 0x4000_1234.
    a.load_imm64(5, 0x8000_1234);
    a.load_imm64(6, 0xABCD);
    a.ins(str32(6, 5, 0));
    a.power_off();

    let (stop, mut board) = boot_and_run(&a.image(), |_| {});
    assert_eq!(stop, StopReason::PoweredOff);
    // The aliased physical address holds the sentinel — proving translation ran.
    // (With the MMU off, the VA 0x8000_1234 would be unmapped and dropped.)
    assert_eq!(board.machine.bus.read_u32(RAM_BASE + 0x1234), 0xABCD);
}

/// Shared GIC/PMR/CTLR bring-up used by the interrupt kernels.
fn enable_gic(a: &mut Asm) {
    a.store_u32(GICD_BASE + 0x000, 1, 1, 2); // GICD_CTLR enable
    a.store_u32(GICC_BASE + 0x000, 1, 1, 2); // GICC_CTLR enable
    a.store_u32(GICC_BASE + 0x004, 0xF0, 1, 2); // PMR open
}

/// Timer: program the virtual timer to fire immediately, take the PPI as an IRQ,
/// and have the handler print and power off.
#[test]
fn virtual_timer_interrupt_reaches_handler() {
    let vbar = kaddr(0x1000);

    let mut a = Asm::new();
    a.load_imm64(1, vbar);
    a.msr(VBAR_EL1, 1);
    enable_gic(&mut a);
    a.store_u32(GICD_BASE + 0x100, 1 << 27, 1, 2); // ISENABLER0: enable PPI 27
    a.load_imm64(2, 0);
    a.msr(CNTV_CVAL_EL0, 2); // compare value 0 -> condition holds at once
    a.load_imm64(3, 1);
    a.msr(CNTV_CTL_EL0, 3); // enable the virtual timer
    a.ins(msr_daifclr(2)); // unmask IRQ -> taken on the next step
    a.ins(B_SELF); // spin until the timer fires

    // IRQ vector (current-EL/SP_ELx IRQ slot = VBAR + 0x280).
    a.pad_to(0x1280);
    a.load_imm64(9, UART_BASE);
    a.ins(movz32(10, u32::from(b'T')));
    a.ins(strb(10, 9));
    a.power_off();

    // Sample every instruction so the fire is prompt and deterministic.
    let (stop, board) = boot_and_run(&a.image(), |b| b.machine.set_timer_interval(1));
    assert_eq!(stop, StopReason::PoweredOff);
    assert_eq!(board.uart.take_tx(), b"T", "timer handler ran");
}

/// Interrupt lifecycle: pend an SPI from guest code, take it, and run a real
/// handler that acknowledges (IAR), services, deactivates (EOIR), and returns
/// (ERET) — after which main resumes and powers off. Output "IM" proves the
/// handler ran ('I') and control returned to main ('M').
#[test]
fn full_irq_handler_cycle_with_iar_eoir_eret() {
    let vbar = kaddr(0x1000);

    let mut a = Asm::new();
    a.load_imm64(1, vbar);
    a.msr(VBAR_EL1, 1);
    enable_gic(&mut a);
    a.store_u32(GICD_BASE + 0x104, 0x2, 1, 2); // ISENABLER1: enable IRQ 33 (SPI 1)
    a.store_u32(GICD_BASE + 0x204, 0x2, 1, 2); // ISPENDR1: pend IRQ 33
    a.ins(msr_daifclr(2)); // unmask -> IRQ taken before the next instruction
    // --- main resumes here after the handler's ERET ---
    a.load_imm64(9, UART_BASE);
    a.ins(movz32(10, u32::from(b'M')));
    a.ins(strb(10, 9)); // main writes 'M'
    a.power_off();
    a.ins(B_SELF);

    // IRQ handler at VBAR + 0x280.
    a.pad_to(0x1280);
    a.load_imm64(11, GICC_BASE);
    a.ins(ldr32(12, 11, 0x0C)); // w12 = GICC_IAR (acknowledge; returns 33)
    a.load_imm64(9, UART_BASE);
    a.ins(movz32(10, u32::from(b'I')));
    a.ins(strb(10, 9)); // handler writes 'I'
    a.ins(str32(12, 11, 0x10)); // GICC_EOIR = 33 (deactivate)
    a.ins(ERET); // return to main

    let (stop, board) = boot_and_run(&a.image(), |_| {});
    assert_eq!(stop, StopReason::PoweredOff);
    assert_eq!(board.uart.take_tx(), b"IM", "handler ran, then main resumed");
}

/// ID/cache registers: the board seeds realistic ARMv8.0 values at reset, so a
/// guest MRS sees 64-byte cache lines, a 64-byte DC ZVA block, a single-core
/// MPIDR, and an AArch64 FP+AdvSIMD feature profile — not the bare-reset zeros.
#[test]
fn id_registers_read_seeded_values() {
    let mut a = Asm::new();
    a.mrs(10, (3, 3, 0, 0, 1)); // CTR_EL0
    a.mrs(11, (3, 3, 0, 0, 7)); // DCZID_EL0
    a.mrs(12, (3, 0, 0, 0, 5)); // MPIDR_EL1
    a.mrs(13, (3, 0, 0, 4, 0)); // ID_AA64PFR0_EL1
    a.mrs(14, (3, 0, 0, 0, 0)); // MIDR_EL1
    a.power_off();

    let (stop, board) = boot_and_run(&a.image(), |_| {});
    assert_eq!(stop, StopReason::PoweredOff);
    assert_eq!(board.machine.cpu.x[10], 0x8444_8004, "CTR_EL0: 64-byte lines");
    assert_eq!(board.machine.cpu.x[11], 0x4, "DCZID_EL0: 64-byte ZVA block");
    assert_eq!(board.machine.cpu.x[12], 0x8000_0000, "MPIDR_EL1: single core");
    assert_eq!(board.machine.cpu.x[13], 0x11, "ID_AA64PFR0: AArch64 EL0/EL1");
    assert_eq!(board.machine.cpu.x[14], 0x410f_d000, "MIDR_EL1: synthetic ARM part");
}
