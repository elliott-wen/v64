//! Robustness: no guest-reachable bus access may panic the host. A wide access
//! straddling the end of RAM (or any region) must read 0 / drop, not index past
//! the backing slice. Fetches (`mem.read_u32`) and the MMU fast path can both
//! land at the RAM edge, so the Bus itself must be bounds-safe.

use aarch64_interp::{GuestMem, Memory};
use aarch64_platform::{Bus, RAM_BASE};

fn mix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn fresh_bus(size: u64) -> Bus {
    Bus::new(Memory::new(RAM_BASE, size as usize))
}

#[test]
fn read_u64_straddling_ram_end_returns_zero() {
    // The exact pre-fix crash: an 8-byte read starting 4 bytes before the end of
    // RAM would slice past the backing buffer.
    let size = 0x1_0000u64;
    let mut bus = fresh_bus(size);
    assert_eq!(bus.read_u64(RAM_BASE + size - 4), 0);
    bus.write_u64(RAM_BASE + size - 4, 0xdead_beef); // must not panic either
}

#[test]
fn bus_accesses_never_panic() {
    let size = 0x1_0000u64;
    let ram_end = RAM_BASE + size;
    let mut bus = fresh_bus(size);
    for i in 0..50_000u64 {
        let p = mix(i);
        let addr = match p % 6 {
            0 => RAM_BASE + (p >> 8) % size,    // inside RAM
            1 => ram_end.wrapping_sub(p % 16),  // straddling the top edge
            2 => RAM_BASE.wrapping_sub(p % 16), // straddling the bottom edge
            3 => u64::MAX - (p % 16),           // near u64::MAX (addr+size overflows)
            4 => 0,                             // far below RAM
            _ => p,                             // anything
        };
        match (p >> 3) % 8 {
            0 => drop(bus.read_u8(addr)),
            1 => drop(bus.read_u16(addr)),
            2 => drop(bus.read_u32(addr)),
            3 => drop(bus.read_u64(addr)),
            4 => bus.write_u8(addr, p as u8),
            5 => bus.write_u16(addr, p as u16),
            6 => bus.write_u32(addr, p as u32),
            _ => bus.write_u64(addr, p),
        }
    }
}
