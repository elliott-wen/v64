//! Tiny deterministic PRNG (xorshift64*) for reproducible fuzzing.
//!
//! Reproducibility is the point: a failing case is fully described by its seed
//! and iteration index, so it can be replayed exactly.

pub struct Rng(u64);

impl Rng {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which xorshift cannot escape.
        Self(seed ^ 0x9e37_79b9_7f4a_7c15 | 1)
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Low `n` bits as a u32 (`n <= 32`).
    #[must_use]
    pub fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        (self.next_u64() & ((1u64 << n) - 1)) as u32
    }

    /// Uniform value in `0..bound` (`bound > 0`).
    #[must_use]
    pub fn below(&mut self, bound: u32) -> u32 {
        (self.next_u64() % u64::from(bound)) as u32
    }
}
