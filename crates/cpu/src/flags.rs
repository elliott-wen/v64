//! NZCV condition flags (PSTATE bits 31..28).

/// NZCV condition flags.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Flags {
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
}

impl Flags {
    /// Pack into the NZCV layout Unicorn/QEMU use for the `NZCV` pseudo-register
    /// (N=bit31, Z=bit30, C=bit29, V=bit28).
    #[must_use]
    pub fn to_nzcv(self) -> u64 {
        (u64::from(self.n) << 31)
            | (u64::from(self.z) << 30)
            | (u64::from(self.c) << 29)
            | (u64::from(self.v) << 28)
    }

    #[must_use]
    pub fn from_nzcv(v: u64) -> Self {
        Self {
            n: (v >> 31) & 1 == 1,
            z: (v >> 30) & 1 == 1,
            c: (v >> 29) & 1 == 1,
            v: (v >> 28) & 1 == 1,
        }
    }
}
