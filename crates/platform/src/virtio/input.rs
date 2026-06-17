//! virtio-input: a keyboard or relative pointer (mouse). The host injects
//! events with [`VirtioInput::key`] / [`VirtioInput::motion`] / etc.; they are
//! delivered to the guest through the event virtqueue. Linux's `virtio_input`
//! driver learns the device's capabilities from the config space (name + the
//! per-event-type code bitmaps), which we synthesize from [`InputKind`].

use std::cell::RefCell;
use std::collections::VecDeque;
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
const QUEUE_NOTIFY: u64 = 0x050;
const INTERRUPT_STATUS: u64 = 0x060;
const INTERRUPT_ACK: u64 = 0x064;
const STATUS: u64 = 0x070;
const CONFIG: u64 = 0x100;

const DEV_INPUT: u64 = 18;
const QUEUE_MAX: u32 = 64;
const F_VERSION_1_HI: u32 = 1;

// Linux input-event type/code constants we use.
const EV_SYN: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const EV_REL: u16 = 0x02;
const REL_X: u16 = 0x00;
const REL_Y: u16 = 0x01;
const REL_WHEEL: u16 = 0x08;

// virtio-input config selects.
const CFG_ID_NAME: u8 = 0x01;
const CFG_ID_DEVIDS: u8 = 0x03;
const CFG_EV_BITS: u8 = 0x11;

/// Which kind of input device this is — it determines the advertised name and
/// capability bitmaps.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Keyboard,
    Mouse,
}

struct Inner {
    kind: InputKind,
    gic: Gic,
    irq: u32,
    dev_feat_sel: u32,
    status: u32,
    int_status: u32,
    queue_sel: usize,
    queues: [Virtq; 2], // 0 = eventq (device->driver), 1 = statusq (driver->device)
    cfg_select: u8,
    cfg_subsel: u8,
    /// Pending input events `(type, code, value)` awaiting an event-queue buffer.
    pending: VecDeque<(u16, u16, u32)>,
    notified: bool,
}

impl Inner {
    fn read(&mut self, off: u64, size: u8) -> u64 {
        match off {
            MAGIC => 0x7472_6976,
            VERSION => 2,
            DEVICE_ID => DEV_INPUT,
            VENDOR_ID => 0x3436_3676,
            DEVICE_FEATURES => {
                if self.dev_feat_sel == 1 { u64::from(F_VERSION_1_HI) } else { 0 }
            }
            QUEUE_NUM_MAX => u64::from(QUEUE_MAX),
            0x044 => u64::from(self.queues[self.queue_sel].ready),
            INTERRUPT_STATUS => u64::from(self.int_status),
            STATUS => u64::from(self.status),
            // SHM_LEN_{LOW,HIGH}: no shared-memory region (sentinel is all-ones).
            0x0b0 | 0x0b4 => 0xffff_ffff,
            CONFIG.. => self.config_read(off - CONFIG, size),
            _ => 0,
        }
    }

    /// virtio-input config space: { u8 select; u8 subsel; u8 size; u8 _[5];
    /// u8 payload[128] }. `size`/`payload` are derived from select/subsel.
    fn config_read(&self, coff: u64, size: u8) -> u64 {
        let payload = self.config_payload();
        let mut bytes = [0u8; 136];
        bytes[0] = self.cfg_select;
        bytes[1] = self.cfg_subsel;
        bytes[2] = payload.len() as u8;
        bytes[8..8 + payload.len()].copy_from_slice(&payload);
        let mut v = 0u64;
        for i in 0..size as usize {
            v |= u64::from(*bytes.get(coff as usize + i).unwrap_or(&0)) << (8 * i);
        }
        v
    }

    /// The payload bytes for the current (select, subsel).
    fn config_payload(&self) -> Vec<u8> {
        match (self.cfg_select, self.cfg_subsel, self.kind) {
            (CFG_ID_NAME, _, InputKind::Keyboard) => b"v64-keyboard".to_vec(),
            (CFG_ID_NAME, _, InputKind::Mouse) => b"v64-mouse".to_vec(),
            // devids: bustype=BUS_VIRTUAL(0x06), vendor, product, version.
            (CFG_ID_DEVIDS, _, _) => {
                vec![0x06, 0x00, 0x64, 0x00, 0x01, 0x00, 0x01, 0x00]
            }
            // EV_BITS: bitmap of supported codes for the requested event type.
            (CFG_EV_BITS, sub, InputKind::Keyboard) if u16::from(sub) == EV_KEY => {
                vec![0xff; 96] // advertise all KEY_* codes
            }
            (CFG_EV_BITS, sub, InputKind::Mouse) if u16::from(sub) == EV_KEY => {
                // BTN_LEFT/RIGHT/MIDDLE = 0x110/0x111/0x112.
                let mut b = vec![0u8; 35];
                b[0x110 / 8] = 0b0000_0111;
                b
            }
            (CFG_EV_BITS, sub, InputKind::Mouse) if u16::from(sub) == EV_REL => {
                // REL_X(0), REL_Y(1), REL_WHEEL(8).
                vec![0b0000_0011, 0b0000_0001]
            }
            _ => Vec::new(),
        }
    }

    fn write(&mut self, off: u64, val: u64) {
        let v = val as u32;
        // Queue-config registers act on the selected queue.
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
            // Config select/subsel are byte writes at CONFIG+0 / CONFIG+1.
            CONFIG => self.cfg_select = val as u8,
            x if x == CONFIG + 1 => self.cfg_subsel = val as u8,
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.queues = [Virtq::default(), Virtq::default()];
        self.int_status = 0;
        self.pending.clear();
        self.notified = false;
        self.gic.clear_pending(self.irq);
    }

    fn process(&mut self, mem: &mut dyn GuestMem) {
        if !self.notified && self.pending.is_empty() {
            return;
        }
        self.notified = false;
        let mut raised = false;
        // statusq (queue 1): the driver writes status reports; consume + drop.
        while let Some((head, _segs)) = self.queues[1].pop(mem) {
            self.queues[1].push_used(mem, head, 0);
            raised = true;
        }
        // eventq (queue 0): fill each posted buffer with one pending event.
        while !self.pending.is_empty() {
            let Some((head, segs)) = self.queues[0].pop(mem) else { break };
            if let (Some(seg), Some((ty, code, value))) = (segs.first(), self.pending.pop_front()) {
                // struct virtio_input_event { __le16 type; __le16 code; __le32 value }.
                mem.write_u16(seg.addr, ty);
                mem.write_u16(seg.addr + 2, code);
                mem.write_u32(seg.addr + 4, value);
                self.queues[0].push_used(mem, head, 8);
                raised = true;
            }
        }
        if raised {
            self.int_status |= 1;
            self.gic.set_pending(self.irq);
        }
    }
}

/// A cloneable handle to a shared virtio-input device.
#[derive(Clone)]
pub struct VirtioInput(Rc<RefCell<Inner>>);

impl VirtioInput {
    #[must_use]
    pub fn new(gic: Gic, irq: u32, kind: InputKind) -> Self {
        VirtioInput(Rc::new(RefCell::new(Inner {
            kind,
            gic,
            irq,
            dev_feat_sel: 0,
            status: 0,
            int_status: 0,
            queue_sel: 0,
            queues: [Virtq::default(), Virtq::default()],
            cfg_select: 0,
            cfg_subsel: 0,
            pending: VecDeque::new(),
            notified: false,
        })))
    }

    #[must_use]
    pub fn device(&self) -> VirtioInputMmio {
        VirtioInputMmio(self.clone())
    }

    fn push(&self, ty: u16, code: u16, value: u32) {
        self.0.borrow_mut().pending.push_back((ty, code, value));
    }

    /// A key/button press (`down=true`) or release, followed by a SYN report.
    pub fn key(&self, code: u16, down: bool) {
        self.push(EV_KEY, code, u32::from(down));
        self.push(EV_SYN, 0, 0);
    }

    /// Relative pointer motion (and optional wheel notches), then a SYN report.
    pub fn motion(&self, dx: i32, dy: i32, wheel: i32) {
        if dx != 0 {
            self.push(EV_REL, REL_X, dx as u32);
        }
        if dy != 0 {
            self.push(EV_REL, REL_Y, dy as u32);
        }
        if wheel != 0 {
            self.push(EV_REL, REL_WHEEL, wheel as u32);
        }
        self.push(EV_SYN, 0, 0);
    }
}

impl DmaDevice for VirtioInput {
    fn poll(&self, mem: &mut dyn GuestMem) {
        self.0.borrow_mut().process(mem);
    }
}

/// The virtio-input register block as an [`MmioDevice`].
pub struct VirtioInputMmio(VirtioInput);

impl MmioDevice for VirtioInputMmio {
    fn name(&self) -> &str {
        "virtio-input"
    }
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.0 .0.borrow_mut().read(offset, size)
    }
    fn write(&mut self, offset: u64, _size: u8, val: u64) {
        self.0 .0.borrow_mut().write(offset, val);
    }
}
