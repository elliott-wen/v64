//! Flat little-endian memory. Phase 1 backing store; swap for a paged MMU later.

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

    fn off(&self, addr: u64) -> usize {
        (addr - self.base) as usize
    }

    #[must_use]
    pub fn read_u8(&self, addr: u64) -> u8 {
        self.bytes[self.off(addr)]
    }

    #[must_use]
    pub fn read_u16(&self, addr: u64) -> u16 {
        let o = self.off(addr);
        u16::from_le_bytes(self.bytes[o..o + 2].try_into().unwrap())
    }

    #[must_use]
    pub fn read_u32(&self, addr: u64) -> u32 {
        let o = self.off(addr);
        u32::from_le_bytes(self.bytes[o..o + 4].try_into().unwrap())
    }

    #[must_use]
    pub fn read_u64(&self, addr: u64) -> u64 {
        let o = self.off(addr);
        u64::from_le_bytes(self.bytes[o..o + 8].try_into().unwrap())
    }

    pub fn write_u8(&mut self, addr: u64, val: u8) {
        let o = self.off(addr);
        self.bytes[o] = val;
    }

    pub fn write_u16(&mut self, addr: u64, val: u16) {
        let o = self.off(addr);
        self.bytes[o..o + 2].copy_from_slice(&val.to_le_bytes());
    }

    pub fn write_u32(&mut self, addr: u64, val: u32) {
        let o = self.off(addr);
        self.bytes[o..o + 4].copy_from_slice(&val.to_le_bytes());
    }

    pub fn write_u64(&mut self, addr: u64, val: u64) {
        let o = self.off(addr);
        self.bytes[o..o + 8].copy_from_slice(&val.to_le_bytes());
    }
}
