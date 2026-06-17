//! virtio-gpu: a minimal 2D (VGA-style) display. Supports the command subset a
//! simple framebuffer driver (Linux `virtio_gpu`/`drm`, or `simpledrm`) needs to
//! bring up one scanout and blit a guest framebuffer to it:
//! GET_DISPLAY_INFO, RESOURCE_CREATE_2D, (DE)ATTACH_BACKING, SET_SCANOUT,
//! TRANSFER_TO_HOST_2D, RESOURCE_FLUSH. The composed scanout image is exposed to
//! the host via [`VirtioGpu::take_frame`] (e.g. for an SDL window or xterm.js
//! canvas). 3D/virgl, cursor rendering, and multi-head are out of scope.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use aarch64_interp::GuestMem;

use crate::bus::DmaDevice;
use crate::virtio::queue::{dma_read, dma_write, Virtq};
use crate::{Gic, MmioDevice};

const MAGIC: u64 = 0x000;
const VERSION: u64 = 0x004;
const DEVICE_ID: u64 = 0x008;
const VENDOR_ID: u64 = 0x00c;
const DEVICE_FEATURES: u64 = 0x010;
const DEVICE_FEATURES_SEL: u64 = 0x014;
const DRIVER_FEATURES_SEL: u64 = 0x024;
const QUEUE_SEL: u64 = 0x030;
const QUEUE_NUM_MAX: u64 = 0x034;
const QUEUE_NOTIFY: u64 = 0x050;
const INTERRUPT_STATUS: u64 = 0x060;
const INTERRUPT_ACK: u64 = 0x064;
const STATUS: u64 = 0x070;
const SHM_LEN_LOW: u64 = 0x0b0;
const SHM_LEN_HIGH: u64 = 0x0b4;
const CONFIG: u64 = 0x100;

const DEV_GPU: u64 = 16;
const QUEUE_MAX: u32 = 64;
const F_VERSION_1_HI: u32 = 1;

// virtio-gpu control commands / responses.
const CMD_GET_DISPLAY_INFO: u32 = 0x0100;
const CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
const CMD_RESOURCE_UNREF: u32 = 0x0102;
const CMD_SET_SCANOUT: u32 = 0x0103;
const CMD_RESOURCE_FLUSH: u32 = 0x0104;
const CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
const CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
const CMD_RESOURCE_DETACH_BACKING: u32 = 0x0107;
const RESP_OK_NODATA: u32 = 0x1100;
const RESP_OK_DISPLAY_INFO: u32 = 0x1101;
const RESP_ERR_UNSPEC: u32 = 0x1200;

const HDR_LEN: usize = 24; // virtio_gpu_ctrl_hdr

struct Resource {
    width: u32,
    height: u32,
    /// Guest backing pages `(addr, len)`.
    backing: Vec<(u64, u32)>,
    /// Host copy of the resource pixels (BGRA), updated by TRANSFER_TO_HOST_2D.
    pixels: Vec<u8>,
}

struct Inner {
    gic: Gic,
    irq: u32,
    width: u32,
    height: u32,
    dev_feat_sel: u32,
    status: u32,
    int_status: u32,
    queue_sel: usize,
    queues: [Virtq; 2], // 0 = controlq, 1 = cursorq
    resources: BTreeMap<u32, Resource>,
    scanout_res: u32, // resource bound to scanout 0 (0 = none)
    dirty: bool,
    notified: bool,
}

/// Read a little-endian u32 from `b` at byte offset `o` (0 if out of range).
fn rd_u32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}
fn rd_u64(b: &[u8], o: usize) -> u64 {
    b.get(o..o + 8).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()))
}

impl Inner {
    fn read(&mut self, off: u64, _size: u8) -> u64 {
        match off {
            MAGIC => 0x7472_6976,
            VERSION => 2,
            DEVICE_ID => DEV_GPU,
            VENDOR_ID => 0x3436_3676,
            DEVICE_FEATURES => {
                if self.dev_feat_sel == 1 { u64::from(F_VERSION_1_HI) } else { 0 }
            }
            QUEUE_NUM_MAX => u64::from(QUEUE_MAX),
            0x044 => u64::from(self.queues[self.queue_sel].ready),
            INTERRUPT_STATUS => u64::from(self.int_status),
            STATUS => u64::from(self.status),
            // SHM_LEN_{LOW,HIGH}: we expose no shared-memory regions. The "absent"
            // sentinel is all-ones (-1); returning 0 would look like a valid
            // zero-length region and fail virtio-gpu probe with -EBUSY.
            SHM_LEN_LOW | SHM_LEN_HIGH => 0xffff_ffff,
            // config: virtio_gpu_config { events_read; events_clear; num_scanouts; num_capsets }.
            CONFIG.. => {
                if off - CONFIG == 8 { 1 } else { 0 } // num_scanouts = 1
            }
            _ => 0,
        }
    }

    fn write(&mut self, off: u64, val: u64) {
        let v = val as u32;
        if self.queues[self.queue_sel].write_reg(off, v) {
            return;
        }
        match off {
            DEVICE_FEATURES_SEL => self.dev_feat_sel = v,
            DRIVER_FEATURES_SEL => {}
            QUEUE_SEL => self.queue_sel = (v as usize).min(1),
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
        self.queues = [Virtq::default(), Virtq::default()];
        self.resources.clear();
        self.scanout_res = 0;
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
        // controlq: handle commands.
        while let Some((head, segs)) = self.queues[0].pop(mem) {
            // Concatenate readable segments into the request; first writable
            // segment is the response buffer.
            let mut req = Vec::new();
            let mut resp_at: Option<(u64, u32)> = None;
            for s in &segs {
                if s.writable {
                    if resp_at.is_none() {
                        resp_at = Some((s.addr, s.len));
                    }
                } else {
                    dma_read(mem, s.addr, s.len, &mut req);
                }
            }
            let resp = self.handle_cmd(mem, &req);
            if let Some((addr, len)) = resp_at {
                let n = resp.len().min(len as usize);
                dma_write(mem, addr, &resp[..n]);
                self.queues[0].push_used(mem, head, n as u32);
            } else {
                self.queues[0].push_used(mem, head, 0);
            }
            raised = true;
        }
        // cursorq: consume + ignore (no hardware cursor).
        while let Some((head, _)) = self.queues[1].pop(mem) {
            self.queues[1].push_used(mem, head, 0);
            raised = true;
        }
        if raised {
            self.int_status |= 1;
            self.gic.set_pending(self.irq);
        }
    }

    /// Dispatch one control command `req`; return the response bytes.
    fn handle_cmd(&mut self, mem: &mut dyn GuestMem, req: &[u8]) -> Vec<u8> {
        let cmd = rd_u32(req, 0);
        match cmd {
            CMD_GET_DISPLAY_INFO => self.resp_display_info(),
            CMD_RESOURCE_CREATE_2D => {
                let id = rd_u32(req, HDR_LEN);
                let width = rd_u32(req, HDR_LEN + 8);
                let height = rd_u32(req, HDR_LEN + 12);
                let pixels = vec![0u8; (width as usize) * (height as usize) * 4];
                self.resources.insert(id, Resource { width, height, backing: Vec::new(), pixels });
                resp_hdr(RESP_OK_NODATA)
            }
            CMD_RESOURCE_UNREF => {
                self.resources.remove(&rd_u32(req, HDR_LEN));
                resp_hdr(RESP_OK_NODATA)
            }
            CMD_RESOURCE_ATTACH_BACKING => {
                let id = rd_u32(req, HDR_LEN);
                let nr = rd_u32(req, HDR_LEN + 4) as usize;
                if let Some(res) = self.resources.get_mut(&id) {
                    res.backing.clear();
                    for i in 0..nr {
                        let base = HDR_LEN + 8 + i * 16;
                        res.backing.push((rd_u64(req, base), rd_u32(req, base + 8)));
                    }
                }
                resp_hdr(RESP_OK_NODATA)
            }
            CMD_RESOURCE_DETACH_BACKING => {
                if let Some(res) = self.resources.get_mut(&rd_u32(req, HDR_LEN)) {
                    res.backing.clear();
                }
                resp_hdr(RESP_OK_NODATA)
            }
            CMD_SET_SCANOUT => {
                self.scanout_res = rd_u32(req, HDR_LEN + 20);
                resp_hdr(RESP_OK_NODATA)
            }
            CMD_TRANSFER_TO_HOST_2D => {
                let id = rd_u32(req, HDR_LEN + 24);
                self.transfer(mem, id);
                resp_hdr(RESP_OK_NODATA)
            }
            CMD_RESOURCE_FLUSH => {
                self.dirty = true; // host can now pull the composed scanout
                resp_hdr(RESP_OK_NODATA)
            }
            _ => resp_hdr(RESP_ERR_UNSPEC),
        }
    }

    /// Pull the resource's guest backing into its host pixel buffer.
    fn transfer(&mut self, mem: &mut dyn GuestMem, id: u32) {
        let Some(res) = self.resources.get_mut(&id) else { return };
        let mut buf = Vec::with_capacity(res.pixels.len());
        for &(addr, len) in &res.backing {
            dma_read(mem, addr, len, &mut buf);
            if buf.len() >= res.pixels.len() {
                break;
            }
        }
        let n = buf.len().min(res.pixels.len());
        res.pixels[..n].copy_from_slice(&buf[..n]);
    }

    fn resp_display_info(&self) -> Vec<u8> {
        let mut r = resp_hdr(RESP_OK_DISPLAY_INFO);
        // pmodes[16]: { virtio_gpu_rect{x,y,w,h}; enabled; flags } — 24 bytes each.
        for i in 0..16 {
            let (w, h, en) = if i == 0 { (self.width, self.height, 1u32) } else { (0, 0, 0) };
            r.extend_from_slice(&0u32.to_le_bytes()); // x
            r.extend_from_slice(&0u32.to_le_bytes()); // y
            r.extend_from_slice(&w.to_le_bytes());
            r.extend_from_slice(&h.to_le_bytes());
            r.extend_from_slice(&en.to_le_bytes());
            r.extend_from_slice(&0u32.to_le_bytes()); // flags
        }
        r
    }
}

/// A 24-byte virtio_gpu_ctrl_hdr response with the given type.
fn resp_hdr(ty: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(HDR_LEN);
    v.extend_from_slice(&ty.to_le_bytes());
    v.resize(HDR_LEN, 0); // flags, fence_id, ctx_id, padding all zero
    v
}

/// A cloneable handle to a shared virtio-gpu device.
#[derive(Clone)]
pub struct VirtioGpu(Rc<RefCell<Inner>>);

impl VirtioGpu {
    /// Create a GPU with one scanout of `width` x `height` pixels.
    #[must_use]
    pub fn new(gic: Gic, irq: u32, width: u32, height: u32) -> Self {
        VirtioGpu(Rc::new(RefCell::new(Inner {
            gic,
            irq,
            width,
            height,
            dev_feat_sel: 0,
            status: 0,
            int_status: 0,
            queue_sel: 0,
            queues: [Virtq::default(), Virtq::default()],
            resources: BTreeMap::new(),
            scanout_res: 0,
            dirty: false,
            notified: false,
        })))
    }

    #[must_use]
    pub fn device(&self) -> VirtioGpuMmio {
        VirtioGpuMmio(self.clone())
    }

    /// The current scanout image `(width, height, BGRA pixels)`, if it changed
    /// since the last call (a FLUSH happened). `None` if nothing new to show.
    #[must_use]
    pub fn take_frame(&self) -> Option<(u32, u32, Vec<u8>)> {
        let mut g = self.0.borrow_mut();
        if !g.dirty {
            return None;
        }
        g.dirty = false;
        let res = g.resources.get(&g.scanout_res)?;
        Some((res.width, res.height, res.pixels.clone()))
    }
}

impl DmaDevice for VirtioGpu {
    fn poll(&self, mem: &mut dyn GuestMem) {
        self.0.borrow_mut().process(mem);
    }
}

/// The virtio-gpu register block as an [`MmioDevice`].
pub struct VirtioGpuMmio(VirtioGpu);

impl MmioDevice for VirtioGpuMmio {
    fn name(&self) -> &str {
        "virtio-gpu"
    }
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.0 .0.borrow_mut().read(offset, size)
    }
    fn write(&mut self, offset: u64, _size: u8, val: u64) {
        self.0 .0.borrow_mut().write(offset, val);
    }
}
