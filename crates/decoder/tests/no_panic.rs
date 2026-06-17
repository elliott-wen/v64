//! Robustness: `decode` must never panic, for *any* 32-bit word. A real CPU
//! treats an unallocated encoding as Undefined (the platform delivers SIGILL);
//! the decoder's job is to return `Insn::Unsupported`, never crash the host.

use aarch64_decoder::decode;

/// splitmix64 for a reproducible random sweep.
fn mix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[test]
fn decode_never_panics() {
    // Large random sample.
    for i in 0..8_000_000u64 {
        let _ = decode(mix(i) as u32);
    }
    // Structured sweep over the top 11 bits (the major encoding-group selector)
    // crossed with a few revealing low patterns, to hit every group router.
    for hi in 0..(1u32 << 11) {
        let top = hi << 21;
        for lo in [0u32, 0x1f, 0x3ff, 0x1f_ffff, 0x1fff_ffff, 0xffff_ffff] {
            let _ = decode(top | (lo & 0x001f_ffff));
        }
    }
}
