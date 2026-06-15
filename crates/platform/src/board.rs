//! The `virt`-style board: assembles the bus, GIC, UART, and timer at their
//! fixed physical addresses, and implements the arm64 boot protocol (load
//! kernel + DTB, set up entry registers).

use aarch64_cpu_state::CpuState;
use aarch64_interp::Memory;

use crate::fdt::{virt_dtb, DtbConfig};
use crate::{Bus, Gic, Machine, Uart};

// Physical memory map (mirrors QEMU `virt`).
pub const RAM_BASE: u64 = 0x4000_0000;
pub const GICD_BASE: u64 = 0x0800_0000;
pub const GICC_BASE: u64 = 0x0801_0000;
pub const UART_BASE: u64 = 0x0900_0000;
/// PL011 (UART0) interrupt: SPI 1 == GIC interrupt ID 33.
pub const UART_IRQ: u32 = 33;

/// Where the kernel image is loaded (the conventional 512 KiB text offset).
pub const KERNEL_LOAD: u64 = RAM_BASE + 0x8_0000;
/// Where the DTB is placed — clear of the kernel image, 8-byte aligned.
pub const DTB_LOAD: u64 = RAM_BASE + 0x100_0000;

/// PSTATE.DAIF all-masked, as required on kernel entry.
const DAIF_MASKED: u8 = 0b1111;

/// A fully-wired single-core `virt` machine plus the UART handle (so the host
/// can drain console output / feed input).
pub struct Board {
    pub machine: Machine,
    pub uart: Uart,
}

impl Board {
    /// Build the board with `ram_size` bytes of RAM and all devices mapped.
    #[must_use]
    pub fn new(ram_size: usize) -> Self {
        let gic = Gic::new();
        let uart = Uart::new(gic.clone(), UART_IRQ);

        let mut bus = Bus::new(Memory::new(RAM_BASE, ram_size));
        bus.map(GICD_BASE, 0x10000, Box::new(gic.distributor()));
        bus.map(GICC_BASE, 0x10000, Box::new(gic.cpu_interface()));
        bus.map(UART_BASE, 0x1000, Box::new(uart.device()));

        let machine = Machine::new(CpuState::new(), bus, gic);
        Board { machine, uart }
    }

    /// Generate a device tree describing this board, for `boot`.
    #[must_use]
    pub fn dtb(&self, ram_size: u64, bootargs: &str, initrd: Option<(u64, u64)>) -> Vec<u8> {
        virt_dtb(&DtbConfig {
            mem_base: RAM_BASE,
            mem_size: ram_size,
            gicd_base: GICD_BASE,
            gicc_base: GICC_BASE,
            uart_base: UART_BASE,
            uart_irq: UART_IRQ,
            bootargs,
            initrd,
        })
    }

    /// Load `kernel` and `dtb` into RAM and set up the entry state per the arm64
    /// boot protocol: `x0` = DTB physical address, `x1..x3 = 0`, PC = kernel load
    /// address, EL1h with all interrupts masked (MMU is off by default).
    pub fn boot(&mut self, kernel: &[u8], dtb: &[u8]) {
        let ram = self.machine.bus.ram_mut();
        ram.write(KERNEL_LOAD, kernel);
        ram.write(DTB_LOAD, dtb);

        let cpu = &mut self.machine.cpu;
        cpu.x = [0; 31];
        cpu.x[0] = DTB_LOAD;
        cpu.pc = KERNEL_LOAD;
        cpu.set_el_spsel(1, true); // EL1h
        cpu.daif = DAIF_MASKED;
        cpu.powered_off = false;
    }
}
