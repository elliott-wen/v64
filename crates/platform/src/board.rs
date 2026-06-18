//! The `virt`-style board: assembles the bus, GIC, UART, and timer at their
//! fixed physical addresses. Kernel-image parsing and load layout live in
//! [`crate::loader`].

use aarch64_cpu_state::CpuState;
use aarch64_interp::Memory;

use crate::clock::{Clock, HostClock, DEFAULT_FREQ_HZ};
use crate::fdt::{virt_dtb, DtbConfig};
use crate::{Bus, Gic, Machine, Uart};

// Physical memory map (mirrors QEMU `virt`).
pub const RAM_BASE: u64 = 0x4000_0000;
pub const GICD_BASE: u64 = 0x0800_0000;
pub const GICC_BASE: u64 = 0x0801_0000;
pub const UART_BASE: u64 = 0x0900_0000;
/// PL011 (UART0) interrupt: SPI 1 == GIC interrupt ID 33.
pub const UART_IRQ: u32 = 33;

/// virtio-mmio device region: up to 8 transports at [`VIRTIO_STRIDE`] spacing,
/// each with its own SPI starting at [`VIRTIO_IRQ`]. (Sits below RAM and the
/// other devices, like the low end of QEMU virt's virtio area.)
pub const VIRTIO_BASE: u64 = 0x0a00_0000;
pub const VIRTIO_STRIDE: u64 = 0x200;
/// First virtio interrupt: GIC ID 48 == SPI 16. Device *n* uses `VIRTIO_IRQ + n`.
pub const VIRTIO_IRQ: u32 = 48;

/// Default kernel load offset when the `Image` header gives none — the
/// conventional 512 KiB text offset from the 2 MiB-aligned RAM base.
pub const KERNEL_LOAD: u64 = RAM_BASE + 0x8_0000;
/// DTB placement for the simple [`Board::boot`] path (small, header-less images).
/// [`Board::boot_image`] computes placement dynamically instead.
pub const DTB_LOAD: u64 = RAM_BASE + 0x100_0000;

/// Default guest RAM: 1 GiB — enough headroom for a real `defconfig` kernel
/// (~tens of MiB) plus its decompression/init and an initramfs.
pub const DEFAULT_RAM_SIZE: usize = 1 << 30;

/// PSTATE.DAIF all-masked, as required on kernel entry.
const DAIF_MASKED: u8 = 0b1111;

/// Pack a system register's (op0,op1,CRn,CRm,op2) tuple into the flat key the
/// interpreter's sysreg map uses (matches `aarch64_decoder::sysreg_key`).
const fn sys_key(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    (op0 << 16) | (op1 << 12) | (crn << 8) | (crm << 4) | op2
}

/// Seed the read-only ID / cache registers with values describing this core:
/// an ARMv8.0-A AArch64 implementation with FP + AdvSIMD, 64-byte cache lines,
/// and a 64-byte DC ZVA block. Reset leaves them zero, which advertises 4-byte
/// cache lines and a 4-byte DC ZVA block — correct but pathologically slow for
/// the kernel's cache/clear-page loops, and a poor `/proc/cpuinfo`.
///
/// Feature registers we don't set stay zero, i.e. "extension absent" — a clean
/// v8.0 baseline (no LSE/crypto/SVE/PAuth), matching what the interpreter
/// actually implements, so the guest never takes a code path we can't honour.
fn reset_id_registers(cpu: &mut CpuState) {
    let regs = [
        // MIDR_EL1: implementer ARM (0x41), architecture "use ID regs" (0xF),
        // synthetic part number 0xD00 so the kernel matches no errata range.
        (sys_key(3, 0, 0, 0, 0), 0x410f_d000),
        // MPIDR_EL1: RES1 bit[31], single core, affinity 0.
        (sys_key(3, 0, 0, 0, 5), 0x8000_0000),
        // ID_AA64PFR0_EL1: EL0/EL1 = AArch64-only (0b0001); FP & AdvSIMD = 0
        // (implemented). All other fields 0 (GICv2 via MMIO, no SVE, etc.).
        (sys_key(3, 0, 0, 4, 0), 0x0000_0000_0000_0011),
        // CTR_EL0: 64-byte I/D cache lines, 64-byte CWG/ERG (Cortex-A53 value).
        (sys_key(3, 3, 0, 0, 1), 0x8444_8004),
        // DCZID_EL0: DC ZVA permitted (DZP=0), block size 4<<4 = 64 bytes.
        (sys_key(3, 3, 0, 0, 7), 0x0000_0004),
    ];
    for (key, val) in regs {
        cpu.sysregs.insert(key, val);
    }
}

/// A fully-wired single-core `virt` machine plus the UART handle (so the host
/// can drain console output / feed input).
pub struct Board {
    pub machine: Machine,
    pub uart: Uart,
    /// `(base, irq)` of each mapped virtio-mmio transport, in slot order — used
    /// to emit their device-tree nodes.
    virtio_slots: Vec<(u64, u32)>,
}

impl Board {
    /// Build the board with the default 1 GiB of RAM and all devices mapped.
    #[must_use]
    pub fn new() -> Self {
        Self::with_ram(DEFAULT_RAM_SIZE)
    }

    /// Build the board with `ram_size` bytes of RAM and all devices mapped, using
    /// the default host clock (native only — `HostClock` reads `Instant`).
    #[must_use]
    pub fn with_ram(ram_size: usize) -> Self {
        Self::with_ram_and_clock(ram_size, Box::new(HostClock::new(DEFAULT_FREQ_HZ)))
    }

    /// Build the default-RAM board with a caller-supplied [`Clock`]. The seam for
    /// non-native hosts: a browser/node build passes a clock backed by
    /// `Date.now()`/`performance.now()` (where `HostClock`'s `Instant` is
    /// unavailable).
    #[must_use]
    pub fn with_clock(clock: Box<dyn Clock>) -> Self {
        Self::with_ram_and_clock(DEFAULT_RAM_SIZE, clock)
    }

    /// Build the board with `ram_size` bytes of RAM, all devices mapped, and the
    /// given timer [`Clock`].
    #[must_use]
    pub fn with_ram_and_clock(ram_size: usize, clock: Box<dyn Clock>) -> Self {
        let gic = Gic::new();
        let uart = Uart::new(gic.clone(), UART_IRQ);

        let mut bus = Bus::new(Memory::new(RAM_BASE, ram_size));
        bus.map(GICD_BASE, 0x10000, Box::new(gic.distributor()));
        bus.map(GICC_BASE, 0x10000, Box::new(gic.cpu_interface()));
        bus.map(UART_BASE, 0x1000, Box::new(uart.device()));

        let mut cpu = CpuState::new();
        reset_id_registers(&mut cpu);
        let machine = Machine::with_clock(cpu, bus, gic, clock, DEFAULT_FREQ_HZ);
        Board { machine, uart, virtio_slots: Vec::new() }
    }

    /// Allocate the next virtio-mmio slot (`base`, `irq`) and record it for the
    /// device tree.
    fn next_virtio_slot(&mut self) -> (u64, u32) {
        let n = self.virtio_slots.len() as u64;
        let slot = (VIRTIO_BASE + n * VIRTIO_STRIDE, VIRTIO_IRQ + n as u32);
        self.virtio_slots.push(slot);
        slot
    }

    /// Attach a virtio-blk disk backed by `image` (appears as `/dev/vda`). Maps
    /// its MMIO window, registers it for DMA polling, and adds its FDT node.
    /// Returns the handle (e.g. to flush the image back to a file on shutdown).
    pub fn attach_disk(&mut self, image: Vec<u8>) -> crate::VirtioBlk {
        let (base, irq) = self.next_virtio_slot();
        let blk = crate::VirtioBlk::new(self.machine.gic.clone(), irq, image);
        self.machine.bus.map(base, VIRTIO_STRIDE, Box::new(blk.device()));
        self.machine.add_dma(Box::new(blk.clone()));
        blk
    }

    /// Attach a virtio-rng (entropy) device. Maps its MMIO window, registers it
    /// for DMA polling, and adds its FDT node. The guest uses it to seed its
    /// CRNG early, silencing the "uninitialized urandom read" boot warnings.
    pub fn attach_rng(&mut self) -> crate::VirtioRng {
        let (base, irq) = self.next_virtio_slot();
        let dev = crate::VirtioRng::new(self.machine.gic.clone(), irq);
        self.machine.bus.map(base, VIRTIO_STRIDE, Box::new(dev.device()));
        self.machine.add_dma(Box::new(dev.clone()));
        dev
    }

    /// Attach a virtio-input device (keyboard or mouse). The returned handle's
    /// `key`/`motion` methods inject host input events to the guest.
    pub fn attach_input(&mut self, kind: crate::InputKind) -> crate::VirtioInput {
        let (base, irq) = self.next_virtio_slot();
        let dev = crate::VirtioInput::new(self.machine.gic.clone(), irq, kind);
        self.machine.bus.map(base, VIRTIO_STRIDE, Box::new(dev.device()));
        self.machine.add_dma(Box::new(dev.clone()));
        dev
    }

    /// Attach a virtio-gpu with one `width` x `height` scanout. The returned
    /// handle's `take_frame` yields the composed image for the host to display.
    pub fn attach_gpu(&mut self, width: u32, height: u32) -> crate::VirtioGpu {
        let (base, irq) = self.next_virtio_slot();
        let dev = crate::VirtioGpu::new(self.machine.gic.clone(), irq, width, height);
        self.machine.bus.map(base, VIRTIO_STRIDE, Box::new(dev.device()));
        self.machine.add_dma(Box::new(dev.clone()));
        dev
    }

    /// Bytes of RAM the board was built with.
    #[must_use]
    pub fn ram_size(&mut self) -> u64 {
        self.machine.bus.ram_mut().bytes.len() as u64
    }

    /// Generate a device tree describing this board.
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
            virtio: &self.virtio_slots,
        })
    }

    /// Set entry state per the arm64 boot protocol: `x0` = DTB physical address,
    /// `x1..x3 = 0`, PC = kernel address, EL1h with all interrupts masked (the
    /// MMU is off by default).
    pub(crate) fn enter(&mut self, kernel_addr: u64, dtb_addr: u64) {
        let cpu = &mut self.machine.cpu;
        cpu.x = [0; 31];
        cpu.x[0] = dtb_addr;
        cpu.pc = kernel_addr;
        cpu.set_el_spsel(1, true); // EL1h
        cpu.daif = DAIF_MASKED;
        cpu.powered_off = false;
    }

    /// Load `kernel` and `dtb` at the fixed [`KERNEL_LOAD`]/[`DTB_LOAD`] addresses
    /// and set up entry. For small, header-less images (the test mini-kernels);
    /// real kernels use [`Self::boot_image`].
    pub fn boot(&mut self, kernel: &[u8], dtb: &[u8]) {
        let ram = self.machine.bus.ram_mut();
        ram.write(KERNEL_LOAD, kernel);
        ram.write(DTB_LOAD, dtb);
        self.enter(KERNEL_LOAD, DTB_LOAD);
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}
