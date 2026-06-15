//! Boot integration: the arm64 entry-register protocol, and a synthetic
//! "mini-kernel" that drives the whole stack — fetch/decode/execute, the bus,
//! the PL011, and PSCI — end to end.

mod common;

use aarch64_interp::{GuestMem, StopReason};
use aarch64_platform::{parse_image_header, Board, DTB_LOAD, KERNEL_LOAD, RAM_BASE};
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
    let mut board = Board::with_ram(RAM_SIZE);
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

/// A fake arm64 `Image`: header fields set, magic at offset 56, body padded.
fn fake_image(text_offset: u64, image_size: u64, body_len: usize) -> Vec<u8> {
    let mut v = vec![0u8; body_len.max(64)];
    v[8..16].copy_from_slice(&text_offset.to_le_bytes());
    v[16..24].copy_from_slice(&image_size.to_le_bytes());
    v[56..60].copy_from_slice(&0x644d_5241u32.to_le_bytes()); // "ARM\x64"
    v
}

#[test]
fn parses_image_header_and_rejects_headerless() {
    let img = fake_image(0x8_0000, 0x20_0000, 0x1000);
    let h = parse_image_header(&img).expect("valid magic");
    assert_eq!(h.text_offset, 0x8_0000);
    assert_eq!(h.image_size, 0x20_0000);
    // A raw blob (our mini-kernels) has no magic -> None, so it falls back.
    assert!(parse_image_header(&[0u8; 64]).is_none());
}

#[test]
fn boot_image_lays_out_kernel_initrd_dtb_without_overlap() {
    let mut board = Board::new(); // 1 GiB
    let image = fake_image(0x8_0000, 0x20_0000, 0x1000); // 2 MiB span
    let initrd = vec![0x5Au8; 0x1000];

    let layout = board.boot_image(&image, Some(&initrd), "console=ttyAMA0");

    // Kernel at base + text_offset; initrd 2 MiB-aligned above the 2 MiB span;
    // DTB 2 MiB-aligned above the initrd.
    assert_eq!(layout.kernel, RAM_BASE + 0x8_0000);
    assert_eq!(layout.initrd, Some((RAM_BASE + 0x40_0000, RAM_BASE + 0x40_1000)));
    assert_eq!(layout.dtb, RAM_BASE + 0x60_0000);

    // Regions are strictly increasing — no overlap.
    let (istart, iend) = layout.initrd.unwrap();
    assert!(layout.kernel + 0x20_0000 <= istart && iend <= layout.dtb);

    // Entry state per the boot protocol.
    assert_eq!(board.machine.cpu.pc, layout.kernel);
    assert_eq!(board.machine.cpu.x[0], layout.dtb);
    assert_eq!(board.machine.cpu.daif, 0b1111);

    // Bytes actually landed: kernel magic, initrd fill, DTB magic.
    assert_eq!(board.machine.bus.read_u32(layout.kernel + 56), 0x644d_5241);
    assert_eq!(board.machine.bus.read_u8(istart), 0x5A);
    assert_eq!(board.machine.bus.read_u32(layout.dtb), 0xedfe_0dd0); // DTB magic, BE->LE
}
