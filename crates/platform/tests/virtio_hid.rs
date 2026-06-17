//! virtio-input (keyboard/mouse) and virtio-gpu (2D) queue-level tests: drive
//! the rings by hand and confirm events are delivered and the framebuffer path
//! works end to end.

use aarch64_interp::{GuestMem, Memory};
use aarch64_platform::{DmaDevice, Gic, InputKind, MmioDevice, VirtioGpu, VirtioInput};

const QUEUE_SEL: u64 = 0x030;
const QUEUE_NUM: u64 = 0x038;
const QUEUE_READY: u64 = 0x044;
const QUEUE_NOTIFY: u64 = 0x050;
const STATUS: u64 = 0x070;
const QUEUE_DESC_LOW: u64 = 0x080;
const QUEUE_AVAIL_LOW: u64 = 0x090;
const QUEUE_USED_LOW: u64 = 0x0a0;

const F_NEXT: u16 = 1;
const F_WRITE: u16 = 2;

const DESC: u64 = 0x1000;
const AVAIL: u64 = 0x2000;
const USED: u64 = 0x3000;

fn write_desc(mem: &mut Memory, i: u64, addr: u64, len: u32, flags: u16, next: u16) {
    let d = DESC + i * 16;
    mem.write_u64(d, addr);
    mem.write_u32(d + 8, len);
    mem.write_u16(d + 12, flags);
    mem.write_u16(d + 14, next);
}

fn setup_queue(dev: &mut dyn MmioDevice, sel: u32, num: u32) {
    dev.write(STATUS, 4, 0xf);
    dev.write(QUEUE_SEL, 4, u64::from(sel));
    dev.write(QUEUE_NUM, 4, u64::from(num));
    dev.write(QUEUE_DESC_LOW, 4, DESC);
    dev.write(QUEUE_AVAIL_LOW, 4, AVAIL);
    dev.write(QUEUE_USED_LOW, 4, USED);
    dev.write(QUEUE_READY, 4, 1);
}

#[test]
fn keyboard_events_reach_the_event_queue() {
    let mut mem = Memory::new(0, 0x10000);
    let kbd = VirtioInput::new(Gic::new(), 48, InputKind::Keyboard);
    let mut dev = kbd.device();
    setup_queue(&mut dev, 0, 4); // eventq

    // Post two 8-byte device-writable event buffers.
    write_desc(&mut mem, 0, 0x4000, 8, F_WRITE, 0);
    write_desc(&mut mem, 1, 0x4100, 8, F_WRITE, 0);
    mem.write_u16(AVAIL + 2, 2);
    mem.write_u16(AVAIL + 4, 0);
    mem.write_u16(AVAIL + 6, 1);

    kbd.key(30, true); // KEY_A press -> EV_KEY(30)=1 then EV_SYN
    kbd.poll(&mut mem);

    // Buffer 0: virtio_input_event { type=EV_KEY(1), code=30, value=1 }.
    assert_eq!(mem.read_u16(0x4000), 1);
    assert_eq!(mem.read_u16(0x4002), 30);
    assert_eq!(mem.read_u32(0x4004), 1);
    // Buffer 1: the SYN report.
    assert_eq!(mem.read_u16(0x4100), 0, "EV_SYN");
    assert_eq!(mem.read_u16(USED + 2), 2, "two buffers used");
}

#[test]
fn mouse_motion_reaches_the_event_queue() {
    let mut mem = Memory::new(0, 0x10000);
    let mouse = VirtioInput::new(Gic::new(), 49, InputKind::Mouse);
    let mut dev = mouse.device();
    setup_queue(&mut dev, 0, 8);

    // Three buffers: REL_X, REL_Y, SYN.
    for i in 0..3u64 {
        write_desc(&mut mem, i, 0x4000 + i * 16, 8, F_WRITE, 0);
        mem.write_u16(AVAIL + 4 + i * 2, i as u16);
    }
    mem.write_u16(AVAIL + 2, 3);

    mouse.motion(5, -3, 0);
    mouse.poll(&mut mem);

    assert_eq!(mem.read_u16(0x4000), 2, "EV_REL"); // type
    assert_eq!(mem.read_u16(0x4002), 0, "REL_X");
    assert_eq!(mem.read_u32(0x4004), 5);
    assert_eq!(mem.read_u16(0x4012), 1, "REL_Y");
    assert_eq!(mem.read_u32(0x4014) as i32, -3);
    assert_eq!(mem.read_u16(USED + 2), 3);
}

/// Submit one virtio-gpu control command (single cmd+response descriptor chain),
/// poll, and return the 32-byte response header region.
fn gpu_cmd(gpu: &VirtioGpu, dev: &mut dyn MmioDevice, mem: &mut Memory, cmd: &[u8], idx: u16) -> [u8; 32] {
    const CMD: u64 = 0x6000;
    const RESP: u64 = 0x7000;
    for (i, b) in cmd.iter().enumerate() {
        mem.write_u8(CMD + i as u64, *b);
    }
    write_desc(mem, 0, CMD, cmd.len() as u32, F_NEXT, 1);
    write_desc(mem, 1, RESP, 1024, F_WRITE, 0);
    mem.write_u16(AVAIL + 4 + u64::from(idx) * 2, 0);
    mem.write_u16(AVAIL + 2, idx + 1);
    dev.write(QUEUE_NOTIFY, 4, 0);
    gpu.poll(mem);
    let mut r = [0u8; 32];
    for (i, b) in r.iter_mut().enumerate() {
        *b = mem.read_u8(RESP + i as u64);
    }
    r
}

fn le32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

#[test]
fn gpu_display_info_and_framebuffer() {
    let mut mem = Memory::new(0, 0x40000);
    let gpu = VirtioGpu::new(Gic::new(), 50, 4, 4); // 4x4 scanout
    let mut dev = gpu.device();
    setup_queue(&mut dev, 0, 16); // controlq

    // GET_DISPLAY_INFO -> RESP_OK_DISPLAY_INFO with pmodes[0] = 4x4 enabled.
    let resp = gpu_cmd(&gpu, &mut dev, &mut mem, &le32(0x0100), 0);
    assert_eq!(u32::from_le_bytes(resp[0..4].try_into().unwrap()), 0x1101);
    // pmodes[0] starts at +24: rect{x,y,w,h}, so width @ +24+8, height @ +24+12.
    assert_eq!(mem.read_u32(0x7000 + 24 + 8), 4, "width");
    assert_eq!(mem.read_u32(0x7000 + 24 + 12), 4, "height");
    assert_eq!(mem.read_u32(0x7000 + 24 + 16), 1, "enabled");

    // Put a 4x4 BGRA pattern in a guest backing buffer.
    const BACKING: u64 = 0x8000;
    let pixels: Vec<u8> = (0..4 * 4 * 4).map(|i| i as u8).collect();
    for (i, b) in pixels.iter().enumerate() {
        mem.write_u8(BACKING + i as u64, *b);
    }

    // RESOURCE_CREATE_2D(id=1, format=1, w=4, h=4).
    let mut c = Vec::new();
    c.extend_from_slice(&le32(0x0101));
    c.resize(24, 0);
    c.extend_from_slice(&le32(1)); // resource_id
    c.extend_from_slice(&le32(1)); // format B8G8R8A8
    c.extend_from_slice(&le32(4)); // width
    c.extend_from_slice(&le32(4)); // height
    let r = gpu_cmd(&gpu, &mut dev, &mut mem, &c, 1);
    assert_eq!(u32::from_le_bytes(r[0..4].try_into().unwrap()), 0x1100, "create OK");

    // RESOURCE_ATTACH_BACKING(id=1, 1 entry -> BACKING/64).
    let mut c = Vec::new();
    c.extend_from_slice(&le32(0x0106));
    c.resize(24, 0);
    c.extend_from_slice(&le32(1)); // resource_id
    c.extend_from_slice(&le32(1)); // nr_entries
    c.extend_from_slice(&BACKING.to_le_bytes()); // addr
    c.extend_from_slice(&le32(64)); // length
    c.extend_from_slice(&le32(0)); // padding
    gpu_cmd(&gpu, &mut dev, &mut mem, &c, 2);

    // SET_SCANOUT(scanout 0 -> resource 1): hdr + rect(16) + scanout + resource.
    let mut c = Vec::new();
    c.extend_from_slice(&le32(0x0103));
    c.resize(24 + 16, 0);
    c.extend_from_slice(&le32(0)); // scanout_id
    c.extend_from_slice(&le32(1)); // resource_id
    gpu_cmd(&gpu, &mut dev, &mut mem, &c, 3);

    // TRANSFER_TO_HOST_2D(resource 1): hdr + rect(16) + offset(8) + resource.
    let mut c = Vec::new();
    c.extend_from_slice(&le32(0x0105));
    c.resize(24 + 16 + 8, 0);
    c.extend_from_slice(&le32(1)); // resource_id
    gpu_cmd(&gpu, &mut dev, &mut mem, &c, 4);

    // RESOURCE_FLUSH(resource 1).
    let mut c = Vec::new();
    c.extend_from_slice(&le32(0x0104));
    c.resize(24 + 16, 0);
    c.extend_from_slice(&le32(1));
    gpu_cmd(&gpu, &mut dev, &mut mem, &c, 5);

    // The composed scanout frame now matches the guest backing.
    let (w, h, frame) = gpu.take_frame().expect("a frame after flush");
    assert_eq!((w, h), (4, 4));
    assert_eq!(frame, pixels, "framebuffer == transferred backing");
    assert!(gpu.take_frame().is_none(), "frame consumed (not dirty)");
}
