//! Crypto extension encoders: AES and SHA1/SHA256.

use super::enc;
use crate::rng::Rng;
use crate::FpEncoded;

pub(super) fn aes(rng: &mut Rng) -> FpEncoded {
    let opcode = 0x4 + rng.below(4); // AESE/AESD/AESMC/AESIMC
    let word = 0x4e28_0800 | (opcode << 12) | (rng.bits(5) << 5) | rng.bits(5);
    enc(word, rng)
}

pub(super) fn sha3(rng: &mut Rng) -> FpEncoded {
    let opcode = rng.below(7); // SHA1C/P/M/SU0, SHA256H/H2/SU1
    let word = 0x5e00_0000 | (rng.bits(5) << 16) | (opcode << 12) | (rng.bits(5) << 5) | rng.bits(5);
    enc(word, rng)
}

pub(super) fn sha2(rng: &mut Rng) -> FpEncoded {
    let opcode = rng.below(3); // SHA1H/SHA1SU1/SHA256SU0
    let word = 0x5e28_0800 | (opcode << 12) | (rng.bits(5) << 5) | rng.bits(5);
    enc(word, rng)
}
