//! Platform / peripheral emulation for the AArch64 machine: the physical bus
//! and the memory-mapped devices that hang off it (GIC, timer, UART, PSCI,
//! virtio — added incrementally).
//!
//! The CPU semantics live in `aarch64-interp`; this crate is everything *around*
//! the core needed to boot a real OS. The entry point is [`Bus`], a
//! [`GuestMem`](aarch64_interp::GuestMem) that routes physical accesses to RAM
//! or a registered [`MmioDevice`].

mod board;
mod bus;
mod clock;
mod fdt;
mod gic;
mod machine;
mod uart;

pub use board::{
    parse_image_header, Board, BootLayout, ImageHeader, DEFAULT_RAM_SIZE, DTB_LOAD, GICC_BASE,
    GICD_BASE, KERNEL_LOAD, RAM_BASE, UART_BASE, UART_IRQ,
};
pub use bus::{Bus, MmioDevice};
pub use clock::{Clock, HostClock, DEFAULT_FREQ_HZ};
pub use fdt::{virt_dtb, DtbConfig, FdtBuilder};
pub use gic::{Gic, GicCpu, GicDist};
pub use machine::Machine;
pub use uart::{Uart, UartDevice};
