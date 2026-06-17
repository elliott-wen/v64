//! The physical-address bus: routes guest memory accesses to either RAM or a
//! memory-mapped device.
//!
//! The interpreter (and, later, the JIT) reach memory through
//! [`GuestMem`](aarch64_interp::GuestMem). [`Bus`] is a `GuestMem` impl that
//! owns the RAM image plus a table of devices keyed by physical-address range,
//! and dispatches each sized access to the right backing. This is the single
//! seam every peripheral hangs off — the GIC, timer, UART, and virtio all
//! register as [`MmioDevice`]s here.

use aarch64_interp::{GuestMem, Memory};

/// A memory-mapped device. Offsets are *relative to the device base*; `size` is
/// the access width in bytes (1, 2, 4, or 8).
///
/// Both directions take `&mut self`: MMIO reads commonly mutate (a UART data
/// register pops its RX FIFO; `GICC_IAR` acknowledges the active interrupt).
pub trait MmioDevice {
    /// Human-readable name, for logging unmapped/odd accesses.
    fn name(&self) -> &str;
    /// Read `size` bytes at `offset` within the device's window.
    fn read(&mut self, offset: u64, size: u8) -> u64;
    /// Write the low `size` bytes of `val` at `offset` within the window.
    fn write(&mut self, offset: u64, size: u8, val: u64);
}

/// A device that performs DMA: it reads/writes guest memory *outside* its own
/// MMIO window (e.g. virtio walking a virtqueue and moving block/framebuffer
/// data). The [`crate::Machine`] calls [`poll`](DmaDevice::poll) periodically
/// with full guest-memory access; the device drains any pending work and posts
/// completions. It must be cheap when idle (typically a flag check), since it is
/// polled on the machine's timer-sampling cadence.
pub trait DmaDevice {
    fn poll(&self, mem: &mut dyn GuestMem);
}

/// One device mapped into the physical-address space at `[base, base + size)`.
struct DeviceEntry {
    base: u64,
    size: u64,
    dev: Box<dyn MmioDevice>,
}

/// Where a physical address lands.
enum Route {
    Ram,
    Device(usize),
    Unmapped,
}

/// Routes physical accesses to RAM or a device. Implements
/// [`GuestMem`](aarch64_interp::GuestMem) so it drops straight into the
/// interpreter's run loop in place of a flat `Memory`.
pub struct Bus {
    ram: Memory,
    /// Devices, kept sorted by `base` and guaranteed non-overlapping by `map`.
    devices: Vec<DeviceEntry>,
}

impl Bus {
    /// Create a bus over `ram` with no devices yet.
    #[must_use]
    pub fn new(ram: Memory) -> Self {
        Self { ram, devices: Vec::new() }
    }

    /// Map `dev` into `[base, base + size)`. Panics if the window overlaps RAM
    /// or an already-mapped device — overlap is a wiring bug, not a runtime
    /// condition.
    pub fn map(&mut self, base: u64, size: u64, dev: Box<dyn MmioDevice>) {
        let end = base + size;
        let ram_base = self.ram.base;
        let ram_end = ram_base + self.ram.bytes.len() as u64;
        assert!(
            end <= ram_base || base >= ram_end,
            "device {} @ {base:#x}..{end:#x} overlaps RAM {ram_base:#x}..{ram_end:#x}",
            dev.name(),
        );
        for d in &self.devices {
            assert!(
                end <= d.base || base >= d.base + d.size,
                "device {} @ {base:#x}..{end:#x} overlaps existing device @ {:#x}",
                dev.name(),
                d.base,
            );
        }
        let idx = self.devices.partition_point(|d| d.base < base);
        self.devices.insert(idx, DeviceEntry { base, size, dev });
    }

    /// Direct access to the RAM image, for loading the kernel/DTB before boot.
    #[must_use]
    pub fn ram_mut(&mut self) -> &mut Memory {
        &mut self.ram
    }

    fn route(&self, addr: u64) -> Route {
        let ram_end = self.ram.base + self.ram.bytes.len() as u64;
        if addr >= self.ram.base && addr < ram_end {
            return Route::Ram;
        }
        // Linear scan: the device count is tiny (a handful for `virt`).
        for (i, d) in self.devices.iter().enumerate() {
            if addr >= d.base && addr < d.base + d.size {
                return Route::Device(i);
            }
        }
        Route::Unmapped
    }

    /// True when `[addr, addr+size)` lies wholly within RAM. A `route` of `Ram`
    /// only guarantees the *start* is in RAM; a wide access near the top edge
    /// would otherwise index past the backing slice and panic the host.
    fn ram_fits(&self, addr: u64, size: u8) -> bool {
        let ram_end = self.ram.base + self.ram.bytes.len() as u64;
        addr.checked_add(u64::from(size)).is_some_and(|end| end <= ram_end)
    }

    fn load(&mut self, addr: u64, size: u8) -> u64 {
        match self.route(addr) {
            Route::Ram if self.ram_fits(addr, size) => match size {
                1 => u64::from(self.ram.read_u8(addr)),
                2 => u64::from(self.ram.read_u16(addr)),
                4 => u64::from(self.ram.read_u32(addr)),
                _ => self.ram.read_u64(addr),
            },
            Route::Device(i) if addr + u64::from(size) <= self.devices[i].base + self.devices[i].size => {
                let off = addr - self.devices[i].base;
                self.devices[i].dev.read(off, size)
            }
            // Out-of-range (access straddles the end of a region) or unmapped:
            // read 0 rather than panic the host.
            _ => {
                eprintln!("bus: unmapped/oob read{} @ {addr:#x} -> 0", size * 8);
                0
            }
        }
    }

    fn store(&mut self, addr: u64, size: u8, val: u64) {
        match self.route(addr) {
            Route::Ram if self.ram_fits(addr, size) => match size {
                1 => self.ram.write_u8(addr, val as u8),
                2 => self.ram.write_u16(addr, val as u16),
                4 => self.ram.write_u32(addr, val as u32),
                _ => self.ram.write_u64(addr, val),
            },
            Route::Device(i) if addr + u64::from(size) <= self.devices[i].base + self.devices[i].size => {
                let off = addr - self.devices[i].base;
                self.devices[i].dev.write(off, size, val);
            }
            _ => {
                eprintln!("bus: unmapped/oob write{} @ {addr:#x} = {val:#x} (dropped)", size * 8);
            }
        }
    }
}

impl GuestMem for Bus {
    fn base(&self) -> u64 {
        self.ram.base
    }
    fn read_u8(&mut self, addr: u64) -> u8 {
        self.load(addr, 1) as u8
    }
    fn read_u16(&mut self, addr: u64) -> u16 {
        self.load(addr, 2) as u16
    }
    fn read_u32(&mut self, addr: u64) -> u32 {
        self.load(addr, 4) as u32
    }
    fn read_u64(&mut self, addr: u64) -> u64 {
        self.load(addr, 8)
    }
    fn write_u8(&mut self, addr: u64, val: u8) {
        self.store(addr, 1, u64::from(val));
    }
    fn write_u16(&mut self, addr: u64, val: u16) {
        self.store(addr, 2, u64::from(val));
    }
    fn write_u32(&mut self, addr: u64, val: u32) {
        self.store(addr, 4, u64::from(val));
    }
    fn write_u64(&mut self, addr: u64, val: u64) {
        self.store(addr, 8, val);
    }
}
