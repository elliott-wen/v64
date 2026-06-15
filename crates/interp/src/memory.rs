//! Guest memory behind the [`GuestMem`] trait, so the interpreter can run
//! against either an owned buffer (native execution, tests, the difftest oracle)
//! or a borrowed view of the JIT's shared wasmtime linear memory — without
//! copying. The trait's primitive is the *sized little-endian access*, so a
//! future MMIO/paged backing can dispatch to a handler instead of indexing a
//! buffer (a flat slice couldn't).

/// Sized little-endian guest memory access. Addresses are *guest* addresses.
///
/// Reads take `&mut self` because an MMIO backing can mutate on read — e.g. a
/// UART data register pops its RX FIFO, and `GICC_IAR` acknowledges the active
/// interrupt. Flat RAM (`Memory`/`MemView`) ignores the mutability.
pub trait GuestMem {
    fn base(&self) -> u64;
    fn read_u8(&mut self, addr: u64) -> u8;
    fn read_u16(&mut self, addr: u64) -> u16;
    fn read_u32(&mut self, addr: u64) -> u32;
    fn read_u64(&mut self, addr: u64) -> u64;
    fn write_u8(&mut self, addr: u64, val: u8);
    fn write_u16(&mut self, addr: u64, val: u16);
    fn write_u32(&mut self, addr: u64, val: u32);
    fn write_u64(&mut self, addr: u64, val: u64);
}

/// An owned flat memory image (native execution, tests, the difftest oracle).
#[derive(Debug, Clone)]
pub struct Memory {
    pub base: u64,
    pub bytes: Vec<u8>,
}

impl Memory {
    #[must_use]
    pub fn new(base: u64, size: usize) -> Self {
        Self { base, bytes: vec![0; size] }
    }

    /// Load `data` at `addr`. Panics if it would overflow the image.
    pub fn write(&mut self, addr: u64, data: &[u8]) {
        let off = (addr - self.base) as usize;
        self.bytes[off..off + data.len()].copy_from_slice(data);
    }
}

/// A borrowed view over an external buffer — e.g. the JIT's wasmtime linear
/// memory — letting the interpreter read/write shared bytes in place.
pub struct MemView<'a> {
    pub base: u64,
    pub bytes: &'a mut [u8],
}

/// Implement [`GuestMem`] for a flat, contiguous, slice-backed type with `base`
/// and `bytes` fields (the common RAM case). MMIO/paged backings would write a
/// bespoke impl that dispatches instead.
macro_rules! impl_guest_mem {
    ($t:ty) => {
        impl GuestMem for $t {
            fn base(&self) -> u64 {
                self.base
            }
            fn read_u8(&mut self, addr: u64) -> u8 {
                self.bytes[(addr - self.base) as usize]
            }
            fn read_u16(&mut self, addr: u64) -> u16 {
                let o = (addr - self.base) as usize;
                u16::from_le_bytes(self.bytes[o..o + 2].try_into().unwrap())
            }
            fn read_u32(&mut self, addr: u64) -> u32 {
                let o = (addr - self.base) as usize;
                u32::from_le_bytes(self.bytes[o..o + 4].try_into().unwrap())
            }
            fn read_u64(&mut self, addr: u64) -> u64 {
                let o = (addr - self.base) as usize;
                u64::from_le_bytes(self.bytes[o..o + 8].try_into().unwrap())
            }
            fn write_u8(&mut self, addr: u64, val: u8) {
                self.bytes[(addr - self.base) as usize] = val;
            }
            fn write_u16(&mut self, addr: u64, val: u16) {
                let o = (addr - self.base) as usize;
                self.bytes[o..o + 2].copy_from_slice(&val.to_le_bytes());
            }
            fn write_u32(&mut self, addr: u64, val: u32) {
                let o = (addr - self.base) as usize;
                self.bytes[o..o + 4].copy_from_slice(&val.to_le_bytes());
            }
            fn write_u64(&mut self, addr: u64, val: u64) {
                let o = (addr - self.base) as usize;
                self.bytes[o..o + 8].copy_from_slice(&val.to_le_bytes());
            }
        }
    };
}

impl_guest_mem!(Memory);
impl_guest_mem!(MemView<'_>);
