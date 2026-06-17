//! Shared split-virtqueue plumbing used by every virtio device.
//!
//! A device owns one or more [`Virtq`]s. The driver fills the descriptor/avail
//! rings in guest RAM; [`Virtq::pop`] hands back the next chain, the device does
//! its work, then [`Virtq::push_used`] records the completion. The MMIO
//! queue-config registers route through [`Virtq::write_reg`].

use aarch64_interp::GuestMem;

const F_NEXT: u16 = 1;
const F_WRITE: u16 = 2;

/// One descriptor of a gathered chain.
pub(crate) struct Seg {
    pub addr: u64,
    pub len: u32,
    /// Device-writable (`VIRTQ_DESC_F_WRITE`) — a buffer we fill, vs one we read.
    pub writable: bool,
}

/// A single split virtqueue: the driver-configured ring addresses plus our read
/// cursor into the avail ring.
#[derive(Default)]
pub(crate) struct Virtq {
    pub num: u32,
    pub ready: bool,
    pub desc: u64,
    pub avail: u64,
    pub used: u64,
    pub last_avail: u16,
}

impl Virtq {
    /// Pop the next available descriptor chain as `(head, segments)`, or `None`
    /// if the driver hasn't posted anything new.
    pub fn pop(&mut self, mem: &mut dyn GuestMem) -> Option<(u16, Vec<Seg>)> {
        if !self.ready || self.num == 0 || self.last_avail == mem.read_u16(self.avail + 2) {
            return None;
        }
        let slot = self.last_avail % self.num as u16;
        let head = mem.read_u16(self.avail + 4 + u64::from(slot) * 2);
        let mut segs = Vec::new();
        let mut idx = head;
        loop {
            let d = self.desc + u64::from(idx) * 16;
            let flags = mem.read_u16(d + 12);
            segs.push(Seg {
                addr: mem.read_u64(d),
                len: mem.read_u32(d + 8),
                writable: flags & F_WRITE != 0,
            });
            if flags & F_NEXT == 0 || segs.len() > self.num as usize {
                break;
            }
            idx = mem.read_u16(d + 14);
        }
        self.last_avail = self.last_avail.wrapping_add(1);
        Some((head, segs))
    }

    /// Append `(head, bytes_written)` to the used ring and advance its index.
    pub fn push_used(&self, mem: &mut dyn GuestMem, head: u16, written: u32) {
        let used_idx = mem.read_u16(self.used + 2);
        let elem = self.used + 4 + u64::from(used_idx % self.num as u16) * 8;
        mem.write_u32(elem, u32::from(head));
        mem.write_u32(elem + 4, written);
        mem.write_u16(self.used + 2, used_idx.wrapping_add(1));
    }

    /// Apply a queue-config MMIO write (`off` = virtio-mmio register offset).
    /// Returns `true` if `off` was a queue register this consumed.
    pub fn write_reg(&mut self, off: u64, v: u32) -> bool {
        match off {
            0x038 => self.num = v,            // QueueNum
            0x044 => self.ready = v & 1 != 0, // QueueReady
            0x080 => self.desc = (self.desc & !0xffff_ffff) | u64::from(v),
            0x084 => self.desc = (self.desc & 0xffff_ffff) | (u64::from(v) << 32),
            0x090 => self.avail = (self.avail & !0xffff_ffff) | u64::from(v),
            0x094 => self.avail = (self.avail & 0xffff_ffff) | (u64::from(v) << 32),
            0x0a0 => self.used = (self.used & !0xffff_ffff) | u64::from(v),
            0x0a4 => self.used = (self.used & 0xffff_ffff) | (u64::from(v) << 32),
            _ => return false,
        }
        true
    }
}

/// Copy `len` bytes from guest memory at `addr` into a `Vec` (8-byte chunks).
pub(crate) fn dma_read(mem: &mut dyn GuestMem, addr: u64, len: u32, out: &mut Vec<u8>) {
    let mut i = 0u64;
    while i + 8 <= u64::from(len) {
        out.extend_from_slice(&mem.read_u64(addr + i).to_le_bytes());
        i += 8;
    }
    while i < u64::from(len) {
        out.push(mem.read_u8(addr + i));
        i += 1;
    }
}

/// Copy `bytes` into guest memory at `addr` (8-byte chunks).
pub(crate) fn dma_write(mem: &mut dyn GuestMem, addr: u64, bytes: &[u8]) {
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let w = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        mem.write_u64(addr + i as u64, w);
        i += 8;
    }
    while i < bytes.len() {
        mem.write_u8(addr + i as u64, bytes[i]);
        i += 1;
    }
}
