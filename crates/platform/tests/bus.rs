//! Bus dispatch: RAM passthrough, device routing, offset translation, overlap.

use aarch64_interp::{GuestMem, Memory};
use aarch64_platform::{Bus, MmioDevice};

/// A device that echoes a programmable value on read; used to observe routing.
/// (Routing of writes and offset translation is checked separately via `Latch`,
/// which reads back what was written.)
#[derive(Default)]
struct ProbeDevice {
    value: u64,
}

impl MmioDevice for ProbeDevice {
    fn name(&self) -> &str {
        "probe"
    }
    fn read(&mut self, _offset: u64, _size: u8) -> u64 {
        self.value
    }
    fn write(&mut self, _offset: u64, _size: u8, _val: u64) {}
}

fn ram_bus() -> Bus {
    Bus::new(Memory::new(0x4000_0000, 0x1000))
}

#[test]
fn ram_passthrough_roundtrips() {
    let mut bus = ram_bus();
    bus.write_u32(0x4000_0010, 0xdead_beef);
    assert_eq!(bus.read_u32(0x4000_0010), 0xdead_beef);
    // Sized writes land little-endian in the underlying image.
    bus.write_u8(0x4000_0020, 0xAB);
    assert_eq!(bus.read_u8(0x4000_0020), 0xAB);
    assert_eq!(bus.base(), 0x4000_0000);
}

#[test]
fn device_read_routes_with_relative_offset() {
    let mut bus = ram_bus();
    let dev = Box::new(ProbeDevice { value: 0x1234_5678, ..Default::default() });
    bus.map(0x0900_0000, 0x1000, dev);

    // Read at base + 0x18 should reach the device at offset 0x18, width 4.
    assert_eq!(bus.read_u32(0x0900_0018), 0x1234_5678);
}

#[test]
fn device_write_is_recorded_via_readback() {
    // A device that stores the last written value and returns it on read.
    #[derive(Default)]
    struct Latch(u64);
    impl MmioDevice for Latch {
        fn name(&self) -> &str {
            "latch"
        }
        fn read(&mut self, _o: u64, _s: u8) -> u64 {
            self.0
        }
        fn write(&mut self, _o: u64, _s: u8, v: u64) {
            self.0 = v;
        }
    }
    let mut bus = ram_bus();
    bus.map(0x0900_0000, 0x1000, Box::new(Latch::default()));
    bus.write_u32(0x0900_0004, 0xcafe_f00d);
    assert_eq!(bus.read_u32(0x0900_0000), 0xcafe_f00d);
}

#[test]
fn unmapped_read_returns_zero() {
    let mut bus = ram_bus();
    // 0x5000_0000 is neither RAM nor a device.
    assert_eq!(bus.read_u32(0x5000_0000), 0);
}

#[test]
#[should_panic(expected = "overlaps RAM")]
fn mapping_over_ram_panics() {
    let mut bus = ram_bus();
    bus.map(0x4000_0800, 0x1000, Box::new(ProbeDevice::default()));
}

#[test]
#[should_panic(expected = "overlaps existing device")]
fn overlapping_devices_panic() {
    let mut bus = ram_bus();
    bus.map(0x0900_0000, 0x1000, Box::new(ProbeDevice::default()));
    bus.map(0x0900_0800, 0x1000, Box::new(ProbeDevice::default()));
}
