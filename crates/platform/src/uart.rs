//! PL011 UART — the `virt` machine's serial console (UART0 @ 0x0900_0000).
//!
//! This is the device that makes the machine *visible*: the kernel's console
//! writes land in the transmit register, and early boot output flows out. A
//! minimal-but-faithful subset:
//!
//! * **TX** is instantaneous — bytes written to the data register are appended
//!   to a buffer the host drains (and, in the browser, forwards to a terminal).
//! * **RX** is a FIFO the host feeds; the guest reads it through the data
//!   register, with the flag register reflecting empty/non-empty.
//! * **Interrupts**: the masked status (`RIS & IMSC`) drives the UART's SPI line
//!   into the GIC, updated on every register access and host feed — event
//!   driven, never polled.
//! * **PrimeCell/Peripheral ID** registers carry the AMBA signature so Linux's
//!   `amba-pl011` driver probes and binds (without these you get earlycon output
//!   but no `ttyAMA0`).
//!
//! Shared via an [`Uart`] handle (`Rc<RefCell>`), mirroring [`crate::Gic`]: the
//! bus maps [`Uart::device`]; the host drains TX with [`Uart::take_tx`] and
//! supplies input with [`Uart::feed`].

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::{Gic, MmioDevice};

// Register offsets.
const UARTDR: u64 = 0x000; // data
const UARTFR: u64 = 0x018; // flags
const UARTIBRD: u64 = 0x024; // integer baud divisor
const UARTFBRD: u64 = 0x028; // fractional baud divisor
const UARTLCR_H: u64 = 0x02C; // line control
const UARTCR: u64 = 0x030; // control
const UARTIFLS: u64 = 0x034; // FIFO level select
const UARTIMSC: u64 = 0x038; // interrupt mask set/clear
const UARTRIS: u64 = 0x03C; // raw interrupt status
const UARTMIS: u64 = 0x040; // masked interrupt status
const UARTICR: u64 = 0x044; // interrupt clear

// Flag register (UARTFR) bits we assert. (BUSY/TXFF are never set in this
// always-ready model: transmit is instantaneous, so the TX FIFO is never full
// or busy.)
const FR_RXFE: u64 = 1 << 4; // RX FIFO empty
const FR_TXFE: u64 = 1 << 7; // TX FIFO empty

// Interrupt bits (shared by RIS/IMSC/MIS).
const INT_RX: u16 = 1 << 4; // receive
const INT_TX: u16 = 1 << 5; // transmit
const INT_RT: u16 = 1 << 6; // receive timeout

/// AMBA PrimeCell ID bytes at 0xFE0..0x1000 (spelling part 0x011, PrimeCell
/// signature 0xB105F00D). Same values QEMU's pl011 reports.
const ID_REGS: [u8; 8] = [0x11, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1];

struct UartInner {
    /// Transmitted bytes awaiting drain by the host.
    tx: Vec<u8>,
    /// Bytes received from the host, awaiting read by the guest.
    rx: VecDeque<u8>,
    /// Interrupt mask (UARTIMSC).
    imsc: u16,
    // Stored config registers (behaviourally inert here, but read back).
    cr: u32,
    lcr_h: u32,
    ibrd: u32,
    fbrd: u32,
    ifls: u32,
    /// Interrupt controller and the SPI line this UART drives.
    gic: Gic,
    irq: u32,
}

impl UartInner {
    /// Raw interrupt status: TX is always ready (FIFO empty); RX/RT assert while
    /// the receive FIFO holds data.
    fn ris(&self) -> u16 {
        let mut r = INT_TX;
        if !self.rx.is_empty() {
            r |= INT_RX | INT_RT;
        }
        r
    }

    /// (De)assert the SPI line based on the masked interrupt status.
    fn update_irq(&self) {
        if self.ris() & self.imsc != 0 {
            self.gic.set_pending(self.irq);
        } else {
            self.gic.clear_pending(self.irq);
        }
    }

    fn read(&mut self, offset: u64) -> u64 {
        match offset {
            UARTDR => {
                let byte = self.rx.pop_front().unwrap_or(0);
                self.update_irq(); // RX FIFO may now be empty
                u64::from(byte)
            }
            UARTFR => {
                let mut fr = FR_TXFE; // TX always empty (instant transmit)
                if self.rx.is_empty() {
                    fr |= FR_RXFE;
                }
                fr
            }
            UARTIBRD => u64::from(self.ibrd),
            UARTFBRD => u64::from(self.fbrd),
            UARTLCR_H => u64::from(self.lcr_h),
            UARTCR => u64::from(self.cr),
            UARTIFLS => u64::from(self.ifls),
            UARTIMSC => u64::from(self.imsc),
            UARTRIS => u64::from(self.ris()),
            UARTMIS => u64::from(self.ris() & self.imsc),
            // PrimeCell / Peripheral ID block.
            0xFE0..0x1000 => {
                let idx = (offset - 0xFE0) / 4;
                ID_REGS.get(idx as usize).map_or(0, |b| u64::from(*b))
            }
            _ => 0,
        }
    }

    fn write(&mut self, offset: u64, val: u64) {
        match offset {
            UARTDR => {
                self.tx.push(val as u8);
                self.update_irq();
            }
            UARTIBRD => self.ibrd = val as u32,
            UARTFBRD => self.fbrd = val as u32,
            UARTLCR_H => self.lcr_h = val as u32,
            UARTCR => self.cr = val as u32,
            UARTIFLS => self.ifls = val as u32,
            UARTIMSC => {
                self.imsc = val as u16;
                self.update_irq();
            }
            // ICR clears interrupts; our RX/RT status is derived from the FIFO
            // (cleared by reading data), so just re-evaluate the line.
            UARTICR => self.update_irq(),
            _ => {}
        }
    }
}

/// A cloneable handle to a shared PL011. Clones reference the same device.
#[derive(Clone)]
pub struct Uart(Rc<RefCell<UartInner>>);

impl Uart {
    /// Create a UART that raises `irq` on `gic` for receive interrupts.
    #[must_use]
    pub fn new(gic: Gic, irq: u32) -> Self {
        Uart(Rc::new(RefCell::new(UartInner {
            tx: Vec::new(),
            rx: VecDeque::new(),
            imsc: 0,
            cr: 0,
            lcr_h: 0,
            ibrd: 0,
            fbrd: 0,
            ifls: 0,
            gic,
            irq,
        })))
    }

    /// The MMIO register block, to map on the bus.
    #[must_use]
    pub fn device(&self) -> UartDevice {
        UartDevice(self.clone())
    }

    /// Drain transmitted bytes (host side: print them / forward to a terminal).
    #[must_use]
    pub fn take_tx(&self) -> Vec<u8> {
        std::mem::take(&mut self.0.borrow_mut().tx)
    }

    /// Supply received input from the host; may assert the RX interrupt.
    pub fn feed(&self, bytes: &[u8]) {
        let mut u = self.0.borrow_mut();
        u.rx.extend(bytes.iter().copied());
        u.update_irq();
    }
}

/// The PL011 register block as an [`MmioDevice`].
pub struct UartDevice(Uart);

impl MmioDevice for UartDevice {
    fn name(&self) -> &str {
        "pl011"
    }
    fn read(&mut self, offset: u64, _size: u8) -> u64 {
        self.0 .0.borrow_mut().read(offset)
    }
    fn write(&mut self, offset: u64, _size: u8, val: u64) {
        self.0 .0.borrow_mut().write(offset, val);
    }
}
