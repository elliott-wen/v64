//! virtio-blk: drive the MMIO register handshake + a split virtqueue by hand and
//! confirm the device does block I/O via DMA and posts a completion.

use aarch64_interp::{GuestMem, Memory};
use aarch64_platform::{DmaDevice, Gic, MmioDevice, VirtioBlk};

// Register offsets we drive (subset of the virtio-mmio layout).
const QUEUE_NUM: u64 = 0x038;
const QUEUE_READY: u64 = 0x044;
const QUEUE_NOTIFY: u64 = 0x050;
const INTERRUPT_STATUS: u64 = 0x060;
const INTERRUPT_ACK: u64 = 0x064;
const STATUS: u64 = 0x070;
const QUEUE_DESC_LOW: u64 = 0x080;
const QUEUE_AVAIL_LOW: u64 = 0x090;
const QUEUE_USED_LOW: u64 = 0x0a0;

const F_NEXT: u16 = 1;
const F_WRITE: u16 = 2;
const T_IN: u32 = 0; // read disk -> memory
const T_OUT: u32 = 1; // write memory -> disk

// Guest-memory layout for the test rings/buffers.
const DESC: u64 = 0x1000;
const AVAIL: u64 = 0x2000;
const USED: u64 = 0x3000;
const HDR: u64 = 0x4000;
const DATA: u64 = 0x5000;
const STAT: u64 = 0x6000;
const QSZ: u32 = 4;

fn write_desc(mem: &mut Memory, i: u64, addr: u64, len: u32, flags: u16, next: u16) {
    let d = DESC + i * 16;
    mem.write_u64(d, addr);
    mem.write_u32(d + 8, len);
    mem.write_u16(d + 12, flags);
    mem.write_u16(d + 14, next);
}

/// Configure queue 0 and post a 3-descriptor request (header, data, status) with
/// the given block type/sector; then notify and poll.
fn submit(blk: &VirtioBlk, dev: &mut dyn MmioDevice, mem: &mut Memory, ty: u32, sector: u64, avail_idx: u16) {
    // Queue setup.
    dev.write(STATUS, 4, 0xf); // ACK|DRIVER|FEATURES_OK|DRIVER_OK
    dev.write(QUEUE_NUM, 4, u64::from(QSZ));
    dev.write(QUEUE_DESC_LOW, 4, DESC);
    dev.write(QUEUE_AVAIL_LOW, 4, AVAIL);
    dev.write(QUEUE_USED_LOW, 4, USED);
    dev.write(QUEUE_READY, 4, 1);

    // Request header at HDR: { u32 type; u32 reserved; u64 sector }.
    mem.write_u32(HDR, ty);
    mem.write_u32(HDR + 4, 0);
    mem.write_u64(HDR + 8, sector);

    // Descriptor chain: header (R) -> data -> status (W).
    let data_writable = if ty == T_IN { F_WRITE } else { 0 };
    write_desc(mem, 0, HDR, 16, F_NEXT, 1);
    write_desc(mem, 1, DATA, 512, F_NEXT | data_writable, 2);
    write_desc(mem, 2, STAT, 1, F_WRITE, 0);

    // Avail ring: flags, idx, ring[0]=head(0).
    mem.write_u16(AVAIL, 0);
    mem.write_u16(AVAIL + 2, avail_idx);
    mem.write_u16(AVAIL + 4, 0);

    dev.write(QUEUE_NOTIFY, 4, 0);
    blk.poll(mem);
}

#[test]
fn read_request_dmas_sector_into_guest_and_completes() {
    let mut mem = Memory::new(0, 0x10000);
    // Disk: byte i = i mod 256, so sector 1 is bytes 512..1024.
    let disk: Vec<u8> = (0..2048u32).map(|i| i as u8).collect();
    let blk = VirtioBlk::new(Gic::new(), 48, disk);
    let mut dev = blk.device();

    submit(&blk, &mut dev, &mut mem, T_IN, 1, 1);

    // Data buffer now holds sector 1.
    for i in 0..512u64 {
        assert_eq!(mem.read_u8(DATA + i), ((512 + i) & 0xff) as u8, "byte {i}");
    }
    assert_eq!(mem.read_u8(STAT), 0, "status OK");
    // Used ring advanced; element = (head=0, len=512+1).
    assert_eq!(mem.read_u16(USED + 2), 1, "used.idx");
    assert_eq!(mem.read_u32(USED + 4), 0, "used.ring[0].id = head");
    assert_eq!(mem.read_u32(USED + 8), 513, "used.ring[0].len = data+status");
    // Completion interrupt asserted; ACK clears it.
    assert_eq!(dev.read(INTERRUPT_STATUS, 4), 1, "used-buffer notification");
    dev.write(INTERRUPT_ACK, 4, 1);
    assert_eq!(dev.read(INTERRUPT_STATUS, 4), 0, "ACK clears");
}

#[test]
fn write_request_dmas_guest_into_disk() {
    let mut mem = Memory::new(0, 0x10000);
    let blk = VirtioBlk::new(Gic::new(), 48, vec![0u8; 2048]);
    let mut dev = blk.device();

    // Fill the data buffer with a pattern to write to sector 2.
    for i in 0..512u64 {
        mem.write_u8(DATA + i, (i as u8) ^ 0x5a);
    }
    submit(&blk, &mut dev, &mut mem, T_OUT, 2, 1);

    assert_eq!(mem.read_u8(STAT), 0, "status OK");
    let disk = blk.disk_image();
    for i in 0..512usize {
        assert_eq!(disk[1024 + i], (i as u8) ^ 0x5a, "disk byte {i}");
    }
}

#[test]
fn capacity_reported_in_config_space() {
    let blk = VirtioBlk::new(Gic::new(), 48, vec![0u8; 8 * 512]);
    let mut dev = blk.device();
    // Config offset 0 = capacity in sectors (u64). 8*512 bytes -> 8 sectors.
    assert_eq!(dev.read(0x100, 4), 8, "capacity low word");
    assert_eq!(dev.read(0x000, 4), 0x7472_6976, "magic 'virt'");
    assert_eq!(dev.read(0x004, 4), 2, "version 2");
    assert_eq!(dev.read(0x008, 4), 2, "device id = block");
}
