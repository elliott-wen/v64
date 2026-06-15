//! Boot integration: the arm64 entry-register protocol, and a synthetic
//! "mini-kernel" that drives the whole stack — fetch/decode/execute, the bus,
//! the PL011, and PSCI — end to end.

mod common;

use aarch64_interp::{GuestMem, StopReason};
use aarch64_platform::{Board, DTB_LOAD, KERNEL_LOAD};
use common::*;

/// Assemble: write "BOOT\n" to the PL011 data register, then PSCI SYSTEM_OFF.
/// Receives only `x0` = DTB (per the boot protocol); builds everything else.
fn mini_kernel() -> Vec<u8> {
    let mut a = Asm::new();
    a.load_imm64(9, aarch64_platform::UART_BASE); // x9 = UART base (DR at offset 0)
    for &c in b"BOOT\n" {
        a.ins(movz32(10, u32::from(c)));
        a.ins(strb(10, 9));
    }
    a.power_off();
    a.image()
}

#[test]
fn boot_sets_entry_registers_and_loads_images() {
    let mut board = Board::new(RAM_SIZE);
    let kernel = [0xAAu8, 0xBB, 0xCC, 0xDD];
    let dtb = board.dtb(RAM_SIZE as u64, "console=ttyAMA0", None);

    board.boot(&kernel, &dtb);

    let cpu = &board.machine.cpu;
    assert_eq!(cpu.pc, KERNEL_LOAD, "PC at kernel entry");
    assert_eq!(cpu.x[0], DTB_LOAD, "x0 = DTB physical address");
    assert_eq!((cpu.x[1], cpu.x[2], cpu.x[3]), (0, 0, 0), "x1..x3 zeroed");
    assert_eq!(cpu.el, 1, "entered at EL1");
    assert_eq!(cpu.daif, 0b1111, "all interrupts masked on entry");

    // Images landed in RAM. The DTB magic is stored big-endian (bytes d0 0d fe
    // ed), so a little-endian word read returns it byte-swapped.
    assert_eq!(board.machine.bus.read_u32(KERNEL_LOAD), 0xDDCC_BBAA);
    assert_eq!(board.machine.bus.read_u32(DTB_LOAD), 0xedfe_0dd0);
}

#[test]
fn mini_kernel_prints_and_powers_off() {
    let (stop, board) = boot_and_run(&mini_kernel(), |_| {});
    assert_eq!(stop, StopReason::PoweredOff, "kernel powered off via PSCI");
    assert_eq!(board.uart.take_tx(), b"BOOT\n", "console output captured");
}
