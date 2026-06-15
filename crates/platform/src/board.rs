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

/// Default kernel load offset when the `Image` header gives none — the
/// conventional 512 KiB text offset from the 2 MiB-aligned RAM base.
pub const KERNEL_LOAD: u64 = RAM_BASE + 0x8_0000;
/// DTB placement for the simple [`Board::boot`] path (small, header-less images).
/// [`Board::boot_image`] computes placement dynamically instead.
pub const DTB_LOAD: u64 = RAM_BASE + 0x100_0000;

/// arm64 `Image` header magic ("ARM\x64", little-endian) at byte offset 56.
const ARM64_IMAGE_MAGIC: u32 = 0x644d_5241;
/// 2 MiB alignment for image / initrd / DTB placement.
const ALIGN_2M: u64 = 0x20_0000;

fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

fn read_u64_le(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// The arm64 `Image` header fields the boot protocol needs (booting.rst).
#[derive(Debug, Clone, Copy)]
pub struct ImageHeader {
    /// Load offset from the 2 MiB-aligned base.
    pub text_offset: u64,
    /// Effective image size (includes BSS); 0 on very old kernels.
    pub image_size: u64,
}

/// Parse an arm64 `Image` header. Returns `None` if the magic is absent (e.g. a
/// raw, header-less test blob).
#[must_use]
pub fn parse_image_header(image: &[u8]) -> Option<ImageHeader> {
    if image.len() < 64 {
        return None;
    }
    let magic = u32::from_le_bytes(image[56..60].try_into().unwrap());
    if magic != ARM64_IMAGE_MAGIC {
        return None;
    }
    Some(ImageHeader { text_offset: read_u64_le(image, 8), image_size: read_u64_le(image, 16) })
}

/// Resulting physical placement of the loaded images.
#[derive(Debug, Clone, Copy)]
pub struct BootLayout {
    pub kernel: u64,
    pub initrd: Option<(u64, u64)>,
    pub dtb: u64,
}

/// Default guest RAM: 1 GiB — enough headroom for a real `defconfig` kernel
/// (~tens of MiB) plus its decompression/init and an initramfs.
pub const DEFAULT_RAM_SIZE: usize = 1 << 30;

/// PSTATE.DAIF all-masked, as required on kernel entry.
const DAIF_MASKED: u8 = 0b1111;

/// A fully-wired single-core `virt` machine plus the UART handle (so the host
/// can drain console output / feed input).
pub struct Board {
    pub machine: Machine,
    pub uart: Uart,
}

impl Board {
    /// Build the board with the default 1 GiB of RAM and all devices mapped.
    #[must_use]
    pub fn new() -> Self {
        Self::with_ram(DEFAULT_RAM_SIZE)
    }

    /// Build the board with `ram_size` bytes of RAM and all devices mapped.
    #[must_use]
    pub fn with_ram(ram_size: usize) -> Self {
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

    /// Load a real arm64 kernel `Image` (and optional initramfs), generating the
    /// device tree and laying everything out without collisions:
    ///
    /// ```text
    ///   RAM_BASE ┌─ (base, 2 MiB aligned)
    ///            │  kernel  @ base + text_offset, spanning image_size
    ///            │  initrd  @ 2 MiB-aligned, above the kernel
    ///            │  DTB     @ 2 MiB-aligned, above the initrd
    /// ```
    ///
    /// Sets up entry per the boot protocol (x0 = DTB, PC = kernel, EL1h, DAIF
    /// masked) and returns where everything landed.
    pub fn boot_image(&mut self, image: &[u8], initrd: Option<&[u8]>, bootargs: &str) -> BootLayout {
        let ram_len = self.machine.bus.ram_mut().bytes.len() as u64;

        // Kernel: base (= RAM_BASE, 2 MiB aligned) + text_offset.
        let header = parse_image_header(image);
        let text_offset = header.map_or(0x8_0000, |h| h.text_offset);
        let kernel_addr = RAM_BASE + text_offset;
        let span = header.map_or(0, |h| h.image_size).max(image.len() as u64);
        let kernel_end = kernel_addr + span;

        // initrd: 2 MiB-aligned, above the kernel.
        let initrd_range = initrd.map(|data| {
            let start = align_up(kernel_end, ALIGN_2M);
            (start, start + data.len() as u64)
        });

        // DTB: 2 MiB-aligned, above the initrd (or the kernel if no initrd).
        let dtb_addr = align_up(initrd_range.map_or(kernel_end, |(_, end)| end), ALIGN_2M);
        let dtb = self.dtb(ram_len, bootargs, initrd_range);

        let dtb_end = dtb_addr + dtb.len() as u64;
        assert!(
            dtb_end <= RAM_BASE + ram_len,
            "images do not fit in {ram_len:#x} of RAM (need up to {dtb_end:#x})",
        );

        // Place everything.
        let ram = self.machine.bus.ram_mut();
        ram.write(kernel_addr, image);
        if let (Some(data), Some((start, _))) = (initrd, initrd_range) {
            ram.write(start, data);
        }
        ram.write(dtb_addr, &dtb);

        let cpu = &mut self.machine.cpu;
        cpu.x = [0; 31];
        cpu.x[0] = dtb_addr;
        cpu.pc = kernel_addr;
        cpu.set_el_spsel(1, true); // EL1h
        cpu.daif = DAIF_MASKED;
        cpu.powered_off = false;

        BootLayout { kernel: kernel_addr, initrd: initrd_range, dtb: dtb_addr }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}
