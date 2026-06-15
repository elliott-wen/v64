//! PL011 UART: transmit capture, receive, flag register, PrimeCell IDs, and the
//! receive interrupt into the GIC. Driven through the bus as guest code would.

use aarch64_interp::{GuestMem, Memory};
use aarch64_platform::{Bus, Gic, Uart};

const UART: u64 = 0x0900_0000; // virt UART0
const GICD: u64 = 0x0800_0000;
const GICC: u64 = 0x0801_0000;
const UART_IRQ: u32 = 33; // virt UART0 = SPI 1

// Register offsets / bits used by the tests.
const DR: u64 = 0x000;
const FR: u64 = 0x018;
const IMSC: u64 = 0x038;
const MIS: u64 = 0x040;
const FR_RXFE: u64 = 1 << 4;
const FR_TXFE: u64 = 1 << 7;
const INT_RX: u64 = 1 << 4;

fn bus_with_uart() -> (Uart, Gic, Bus) {
    let gic = Gic::new();
    let uart = Uart::new(gic.clone(), UART_IRQ);
    let mut bus = Bus::new(Memory::new(0x4000_0000, 0x1000));
    bus.map(UART, 0x1000, Box::new(uart.device()));
    // GIC too, so the interrupt test can observe delivery.
    bus.map(GICD, 0x10000, Box::new(gic.distributor()));
    bus.map(GICC, 0x10000, Box::new(gic.cpu_interface()));
    (uart, gic, bus)
}

#[test]
fn transmit_captures_bytes() {
    let (uart, _gic, mut bus) = bus_with_uart();
    for &b in b"Hi!\n" {
        // The driver polls TXFE before writing; it's always set here.
        assert_ne!(bus.read_u32(UART + FR) & FR_TXFE as u32, 0);
        bus.write_u32(UART + DR, u32::from(b));
    }
    assert_eq!(uart.take_tx(), b"Hi!\n");
    assert!(uart.take_tx().is_empty(), "drained");
}

#[test]
fn receive_reads_in_order_and_tracks_empty() {
    let (uart, _gic, mut bus) = bus_with_uart();
    assert_ne!(bus.read_u32(UART + FR) & FR_RXFE as u32, 0, "RX empty initially");

    uart.feed(b"AB");
    assert_eq!(bus.read_u32(UART + FR) & FR_RXFE as u32, 0, "RX not empty after feed");
    assert_eq!(bus.read_u32(UART + DR) as u8, b'A');
    assert_eq!(bus.read_u32(UART + DR) as u8, b'B');
    assert_ne!(bus.read_u32(UART + FR) & FR_RXFE as u32, 0, "RX empty after draining");
    assert_eq!(bus.read_u32(UART + DR) as u8, 0, "empty read returns 0");
}

#[test]
fn primecell_id_signature() {
    let (_uart, _gic, mut bus) = bus_with_uart();
    let id: Vec<u8> = (0..8).map(|i| bus.read_u32(UART + 0xFE0 + i * 4) as u8).collect();
    assert_eq!(id, [0x11, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1]);
}

#[test]
fn receive_interrupt_asserts_and_clears() {
    let (uart, gic, mut bus) = bus_with_uart();
    // Enable the UART SPI at the GIC and open the controller.
    bus.write_u32(GICD + 0x104, 1 << 1); // ISENABLER for IRQ 33
    bus.write_u32(GICD + 0x000, 1);
    bus.write_u32(GICC + 0x000, 1);
    bus.write_u32(GICC + 0x004, 0xF0);
    // Unmask the UART receive interrupt.
    bus.write_u32(UART + IMSC, INT_RX as u32);

    assert!(!gic.pending_irq(), "no interrupt before any input");

    uart.feed(b"x");
    assert_ne!(bus.read_u32(UART + MIS) & INT_RX as u32, 0, "masked status set");
    assert!(gic.pending_irq(), "UART asserted its line into the GIC");

    // Drain the byte; the receive line should deassert at both UART and GIC.
    assert_eq!(bus.read_u32(UART + DR) as u8, b'x');
    assert_eq!(bus.read_u32(UART + MIS) & INT_RX as u32, 0, "cleared after read");
    assert!(!gic.pending_irq(), "GIC line deasserted after drain");
}
