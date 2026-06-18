//! virtio-entropy (virtio-rng): hands the guest entropy on demand. The driver
//! posts device-writable buffers on the request queue; we fill them and post the
//! completion. Linux uses this to seed its CRNG early in boot, which silences
//! the "uninitialized urandom read" warnings (udev, etc.).
//!
//! This is an emulator, not a security boundary, so faithfulness to a real
//! entropy source buys nothing: we simply return zero bytes. The guest CRNG
//! still initializes (it credits the bytes as entropy); the values just aren't
//! unpredictable — which is fine here and keeps runs reproducible.

use std::cell::RefCell;
use std::rc::Rc;

use aarch64_interp::GuestMem;

use crate::bus::DmaDevice;
use crate::virtio::queue::Virtq;
use crate::{Gic, MmioDevice};

// Common virtio-mmio registers (see virtio/blk.rs for the full list).
const MAGIC: u64 = 0x000;
const VERSION: u64 = 0x004;
const DEVICE_ID: u64 = 0x008;
const VENDOR_ID: u64 = 0x00c;
const DEVICE_FEATURES: u64 = 0x010;
const DEVICE_FEATURES_SEL: u64 = 0x014;
const DRIVER_FEATURES_SEL: u64 = 0x024;
const QUEUE_SEL: u64 = 0x030;
const QUEUE_NUM_MAX: u64 = 0x034;
const QUEUE_READY: u64 = 0x044;
const QUEUE_NOTIFY: u64 = 0x050;
const INTERRUPT_STATUS: u64 = 0x060;
const INTERRUPT_ACK: u64 = 0x064;
const STATUS: u64 = 0x070;

const DEV_ENTROPY: u64 = 4;
const QUEUE_MAX: u32 = 64;
const F_VERSION_1_HI: u32 = 1;

struct Inner {
    gic: Gic,
    irq: u32,
    dev_feat_sel: u32,
    status: u32,
    int_status: u32,
    queue: Virtq, // single request queue (queue 0)
    notified: bool,
}

impl Inner {
    fn read(&mut self, off: u64, _size: u8) -> u64 {
        match off {
            MAGIC => 0x7472_6976,
            VERSION => 2,
            DEVICE_ID => DEV_ENTROPY,
            VENDOR_ID => 0x3436_3676,
            DEVICE_FEATURES => {
                if self.dev_feat_sel == 1 { u64::from(F_VERSION_1_HI) } else { 0 }
            }
            QUEUE_NUM_MAX => u64::from(QUEUE_MAX),
            QUEUE_READY => u64::from(self.queue.ready),
            INTERRUPT_STATUS => u64::from(self.int_status),
            STATUS => u64::from(self.status),
            // SHM_LEN_{LOW,HIGH}: no shared-memory region (sentinel is all-ones).
            0x0b0 | 0x0b4 => 0xffff_ffff,
            _ => 0,
        }
    }

    fn write(&mut self, off: u64, val: u64) {
        let v = val as u32;
        // Queue-config registers act on the single request queue.
        if self.queue.write_reg(off, v) {
            return;
        }
        match off {
            DEVICE_FEATURES_SEL => self.dev_feat_sel = v,
            DRIVER_FEATURES_SEL => {}
            QUEUE_SEL => {} // only queue 0 exists
            QUEUE_NOTIFY => self.notified = true,
            INTERRUPT_ACK => {
                self.int_status &= !v;
                if self.int_status == 0 {
                    self.gic.clear_pending(self.irq);
                }
            }
            STATUS => {
                self.status = v;
                if v == 0 {
                    self.reset();
                }
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.queue = Virtq::default();
        self.int_status = 0;
        self.notified = false;
        self.gic.clear_pending(self.irq);
    }

    fn process(&mut self, mem: &mut dyn GuestMem) {
        if !self.notified {
            return;
        }
        self.notified = false;
        let mut raised = false;
        // Each posted chain is device-writable buffers to fill with entropy. We
        // return zeros (see module docs) and report the full byte count.
        while let Some((head, segs)) = self.queue.pop(mem) {
            let mut written = 0u32;
            for seg in &segs {
                if seg.writable {
                    for i in 0..u64::from(seg.len) {
                        mem.write_u8(seg.addr + i, 0);
                    }
                    written += seg.len;
                }
            }
            self.queue.push_used(mem, head, written);
            raised = true;
        }
        if raised {
            self.int_status |= 1; // used-buffer notification
            self.gic.set_pending(self.irq);
        }
    }
}

/// A cloneable handle to a shared virtio-rng device (mirrors [`crate::Uart`]).
#[derive(Clone)]
pub struct VirtioRng(Rc<RefCell<Inner>>);

impl VirtioRng {
    /// Build an entropy device raising `irq` on `gic` when requests complete.
    #[must_use]
    pub fn new(gic: Gic, irq: u32) -> Self {
        VirtioRng(Rc::new(RefCell::new(Inner {
            gic,
            irq,
            dev_feat_sel: 0,
            status: 0,
            int_status: 0,
            queue: Virtq::default(),
            notified: false,
        })))
    }

    /// The MMIO register block, to map on the bus.
    #[must_use]
    pub fn device(&self) -> VirtioRngMmio {
        VirtioRngMmio(self.clone())
    }
}

impl DmaDevice for VirtioRng {
    fn poll(&self, mem: &mut dyn GuestMem) {
        self.0.borrow_mut().process(mem);
    }
}

/// The virtio-mmio register block as an [`MmioDevice`].
pub struct VirtioRngMmio(VirtioRng);

impl MmioDevice for VirtioRngMmio {
    fn name(&self) -> &str {
        "virtio-rng"
    }
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.0 .0.borrow_mut().read(offset, size)
    }
    fn write(&mut self, offset: u64, _size: u8, val: u64) {
        self.0 .0.borrow_mut().write(offset, val);
    }
}
