//! virtio-mmio transport + virtio-blk device (modern / VIRTIO 1.0, version 2).
//!
//! This is the first DMA-capable peripheral: unlike the UART/GIC, a virtio
//! device reads command rings and data buffers straight out of guest RAM. The
//! register block is an [`MmioDevice`] like any other, but the actual work
//! (walking the virtqueue, doing block I/O, posting completions) happens in
//! [`DmaDevice::poll`], which the [`crate::Machine`] calls with guest-memory
//! access. A `QueueNotify` write just flags "work pending"; `poll` drains it.
//!
//! Scope: a single request virtqueue, split-ring layout, in-memory disk image.
//! Enough to mount an ext4 root filesystem off `/dev/vda`.

use std::cell::RefCell;
use std::rc::Rc;

use aarch64_interp::GuestMem;

use crate::bus::DmaDevice;
use crate::{Gic, MmioDevice};

// --- virtio-mmio register offsets (VIRTIO 1.1 §4.2.2). ---
const MAGIC: u64 = 0x000; // R: 0x74726976 "virt"
const VERSION: u64 = 0x004; // R: 2 (modern)
const DEVICE_ID: u64 = 0x008; // R: 2 = block
const VENDOR_ID: u64 = 0x00c; // R
const DEVICE_FEATURES: u64 = 0x010; // R (windowed by DEVICE_FEATURES_SEL)
const DEVICE_FEATURES_SEL: u64 = 0x014; // W
const DRIVER_FEATURES: u64 = 0x020; // W
const DRIVER_FEATURES_SEL: u64 = 0x024; // W
const QUEUE_SEL: u64 = 0x030; // W
const QUEUE_NUM_MAX: u64 = 0x034; // R
const QUEUE_NUM: u64 = 0x038; // W
const QUEUE_READY: u64 = 0x044; // RW
const QUEUE_NOTIFY: u64 = 0x050; // W
const INTERRUPT_STATUS: u64 = 0x060; // R
const INTERRUPT_ACK: u64 = 0x064; // W
const STATUS: u64 = 0x070; // RW
const QUEUE_DESC_LOW: u64 = 0x080; // W
const QUEUE_DESC_HIGH: u64 = 0x084; // W
const QUEUE_AVAIL_LOW: u64 = 0x090; // W
const QUEUE_AVAIL_HIGH: u64 = 0x094; // W
const QUEUE_USED_LOW: u64 = 0x0a0; // W
const QUEUE_USED_HIGH: u64 = 0x0a4; // W
const CONFIG: u64 = 0x100; // device-specific config space

const MAGIC_VALUE: u64 = 0x7472_6976;
const VENDOR: u64 = 0x3436_3676; // "v64\0"-ish
const DEV_BLOCK: u64 = 2;
const QUEUE_MAX: u32 = 256; // max ring size we accept (power of two)

/// VIRTIO_F_VERSION_1 (feature bit 32) — mandatory for a modern device.
const F_VERSION_1_HI: u32 = 1; // bit 0 of the high (sel=1) feature word

// Split-virtqueue descriptor flags.
const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2; // device writes (vs reads) this buffer

// virtio-blk request types + status.
const BLK_T_IN: u32 = 0; // read disk -> memory
const BLK_T_OUT: u32 = 1; // write memory -> disk
const BLK_S_OK: u8 = 0;
const BLK_S_UNSUPP: u8 = 2;

const SECTOR: u64 = 512;

struct Inner {
    disk: Vec<u8>,
    gic: Gic,
    irq: u32,
    // Negotiation / transport state.
    dev_feat_sel: u32,
    drv_feat_sel: u32,
    status: u32,
    int_status: u32,
    // Single queue (queue 0).
    queue_num: u32,
    queue_ready: bool,
    desc: u64,
    avail: u64,
    used: u64,
    last_avail: u16, // next avail-ring index we haven't consumed
    notified: bool,
}

impl Inner {
    fn read(&mut self, off: u64, size: u8) -> u64 {
        match off {
            MAGIC => MAGIC_VALUE,
            VERSION => 2,
            DEVICE_ID => DEV_BLOCK,
            VENDOR_ID => VENDOR,
            DEVICE_FEATURES => {
                if self.dev_feat_sel == 1 { u64::from(F_VERSION_1_HI) } else { 0 }
            }
            QUEUE_NUM_MAX => u64::from(QUEUE_MAX),
            QUEUE_READY => u64::from(self.queue_ready),
            INTERRUPT_STATUS => u64::from(self.int_status),
            STATUS => u64::from(self.status),
            // SHM_LEN_{LOW,HIGH}: no shared-memory region (sentinel is all-ones).
            0x0b0 | 0x0b4 => 0xffff_ffff,
            // Block config space: capacity in 512-byte sectors (u64 @ +0x00).
            CONFIG.. => self.config_read(off - CONFIG, size),
            _ => 0,
        }
    }

    fn config_read(&self, coff: u64, size: u8) -> u64 {
        let capacity = (self.disk.len() as u64) / SECTOR;
        let mut bytes = [0u8; 8];
        if coff < 8 {
            bytes = capacity.to_le_bytes();
        }
        // Return `size` bytes starting at `coff & 7` (config is little-endian).
        let start = (coff & 7) as usize;
        let mut v = 0u64;
        for i in 0..size as usize {
            v |= u64::from(*bytes.get(start + i).unwrap_or(&0)) << (8 * i);
        }
        v
    }

    fn write(&mut self, off: u64, val: u64) {
        let v = val as u32;
        match off {
            DEVICE_FEATURES_SEL => self.dev_feat_sel = v,
            DRIVER_FEATURES_SEL => self.drv_feat_sel = v,
            DRIVER_FEATURES => {} // accepted as-is (we only require VERSION_1)
            QUEUE_SEL => {} // single queue; ignore (only 0 is valid)
            QUEUE_NUM => self.queue_num = v,
            QUEUE_READY => self.queue_ready = v & 1 != 0,
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
            QUEUE_DESC_LOW => self.desc = (self.desc & !0xffff_ffff) | u64::from(v),
            QUEUE_DESC_HIGH => self.desc = (self.desc & 0xffff_ffff) | (u64::from(v) << 32),
            QUEUE_AVAIL_LOW => self.avail = (self.avail & !0xffff_ffff) | u64::from(v),
            QUEUE_AVAIL_HIGH => self.avail = (self.avail & 0xffff_ffff) | (u64::from(v) << 32),
            QUEUE_USED_LOW => self.used = (self.used & !0xffff_ffff) | u64::from(v),
            QUEUE_USED_HIGH => self.used = (self.used & 0xffff_ffff) | (u64::from(v) << 32),
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.queue_ready = false;
        self.int_status = 0;
        self.last_avail = 0;
        self.notified = false;
        self.gic.clear_pending(self.irq);
    }

    /// Drain the avail ring: handle every request the driver has posted since we
    /// last looked, append completions to the used ring, and raise the IRQ.
    fn process(&mut self, mem: &mut dyn GuestMem) {
        if !self.notified {
            return;
        }
        self.notified = false;
        if !self.queue_ready || self.queue_num == 0 {
            return;
        }
        let qsz = self.queue_num as u16;
        // avail ring: { u16 flags; u16 idx; u16 ring[qsz]; ... }
        let avail_idx = mem.read_u16(self.avail + 2);
        let mut did_work = false;
        while self.last_avail != avail_idx {
            let slot = self.last_avail % qsz;
            let head = mem.read_u16(self.avail + 4 + u64::from(slot) * 2);
            let written = self.handle_request(mem, head);
            self.push_used(mem, qsz, head, written);
            self.last_avail = self.last_avail.wrapping_add(1);
            did_work = true;
        }
        if did_work {
            self.int_status |= 1; // used-buffer notification
            self.gic.set_pending(self.irq);
        }
    }

    /// Walk one descriptor chain and perform the block request. Returns the
    /// number of bytes the device wrote into guest buffers (for the used ring).
    fn handle_request(&mut self, mem: &mut dyn GuestMem, head: u16) -> u32 {
        // Gather the chain: (addr, len, device_writable).
        let mut chain: Vec<(u64, u32, bool)> = Vec::new();
        let mut idx = head;
        loop {
            let d = self.desc + u64::from(idx) * 16;
            let addr = mem.read_u64(d);
            let len = mem.read_u32(d + 8);
            let flags = mem.read_u16(d + 12);
            let next = mem.read_u16(d + 14);
            chain.push((addr, len, flags & VIRTQ_DESC_F_WRITE != 0));
            if flags & VIRTQ_DESC_F_NEXT == 0 {
                break;
            }
            idx = next;
            if chain.len() > QUEUE_MAX as usize {
                break; // malformed/cyclic chain guard
            }
        }
        if chain.len() < 2 {
            return 0;
        }
        // First descriptor: 16-byte request header (type, _, sector).
        let (haddr, _, _) = chain[0];
        let req_type = mem.read_u32(haddr);
        let sector = mem.read_u64(haddr + 8);
        // Last descriptor: 1-byte status (device-writable).
        let (status_addr, _, _) = chain[chain.len() - 1];
        // Middle descriptors: the data buffers.
        let data = &chain[1..chain.len() - 1];

        let status = match req_type {
            BLK_T_IN => self.do_read(mem, sector, data),
            BLK_T_OUT => self.do_write(mem, sector, data),
            _ => BLK_S_UNSUPP,
        };
        mem.write_u8(status_addr, status);

        // Used `len` = device-written bytes: data (on read) + the status byte.
        let data_written: u32 = if status == BLK_S_OK && req_type == BLK_T_IN {
            data.iter().map(|&(_, l, _)| l).sum()
        } else {
            0
        };
        data_written + 1
    }

    fn do_read(&self, mem: &mut dyn GuestMem, sector: u64, data: &[(u64, u32, bool)]) -> u8 {
        let mut off = sector * SECTOR;
        for &(addr, len, _) in data {
            for i in 0..u64::from(len) {
                let byte = self.disk.get((off + i) as usize).copied().unwrap_or(0);
                mem.write_u8(addr + i, byte);
            }
            off += u64::from(len);
        }
        BLK_S_OK
    }

    fn do_write(&mut self, mem: &mut dyn GuestMem, sector: u64, data: &[(u64, u32, bool)]) -> u8 {
        let mut off = (sector * SECTOR) as usize;
        for &(addr, len, _) in data {
            for i in 0..u64::from(len) {
                let byte = mem.read_u8(addr + i);
                if off < self.disk.len() {
                    self.disk[off] = byte;
                }
                off += 1;
            }
        }
        BLK_S_OK
    }

    /// Append `(head, len)` to the used ring and bump its index.
    fn push_used(&self, mem: &mut dyn GuestMem, qsz: u16, head: u16, len: u32) {
        // used ring: { u16 flags; u16 idx; struct{u32 id; u32 len} ring[qsz]; ... }
        let used_idx = mem.read_u16(self.used + 2);
        let slot = used_idx % qsz;
        let elem = self.used + 4 + u64::from(slot) * 8;
        mem.write_u32(elem, u32::from(head));
        mem.write_u32(elem + 4, len);
        mem.write_u16(self.used + 2, used_idx.wrapping_add(1));
    }
}

/// A cloneable handle to a shared virtio-blk device (mirrors [`crate::Uart`]).
#[derive(Clone)]
pub struct VirtioBlk(Rc<RefCell<Inner>>);

impl VirtioBlk {
    /// Build a block device backed by the in-memory `disk` image, raising `irq`
    /// on `gic` when requests complete.
    #[must_use]
    pub fn new(gic: Gic, irq: u32, disk: Vec<u8>) -> Self {
        VirtioBlk(Rc::new(RefCell::new(Inner {
            disk,
            gic,
            irq,
            dev_feat_sel: 0,
            drv_feat_sel: 0,
            status: 0,
            int_status: 0,
            queue_num: 0,
            queue_ready: false,
            desc: 0,
            avail: 0,
            used: 0,
            last_avail: 0,
            notified: false,
        })))
    }

    /// The MMIO register block, to map on the bus.
    #[must_use]
    pub fn device(&self) -> VirtioBlkMmio {
        VirtioBlkMmio(self.clone())
    }

    /// The current disk image (e.g. to flush back to a file on shutdown).
    #[must_use]
    pub fn disk_image(&self) -> Vec<u8> {
        self.0.borrow().disk.clone()
    }
}

impl DmaDevice for VirtioBlk {
    fn poll(&self, mem: &mut dyn GuestMem) {
        self.0.borrow_mut().process(mem);
    }
}

/// The virtio-mmio register block as an [`MmioDevice`].
pub struct VirtioBlkMmio(VirtioBlk);

impl MmioDevice for VirtioBlkMmio {
    fn name(&self) -> &str {
        "virtio-blk"
    }
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.0 .0.borrow_mut().read(offset, size)
    }
    fn write(&mut self, offset: u64, _size: u8, val: u64) {
        self.0 .0.borrow_mut().write(offset, val);
    }
}
