//! Crypto SHA1/SHA256. State words are the four little-endian 32-bit lanes of a
//! V register (word 0 = low 32 bits). Matches QEMU's helpers.

use aarch64_cpu_state::CpuState;

fn words(v: u128) -> [u32; 4] {
    [v as u32, (v >> 32) as u32, (v >> 64) as u32, (v >> 96) as u32]
}
fn pack(w: [u32; 4]) -> u128 {
    u128::from(w[0]) | (u128::from(w[1]) << 32) | (u128::from(w[2]) << 64) | (u128::from(w[3]) << 96)
}

fn cho(x: u32, y: u32, z: u32) -> u32 {
    (x & (y ^ z)) ^ z
}
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}
fn par(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}
fn big_s0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
fn big_s1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}
fn sml_s0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
fn sml_s1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

pub(crate) fn three_reg(cpu: &mut CpuState, opcode: u8, rm: u8, rn: u8, rd: u8) -> Option<u64> {
    let mut d = words(cpu.v[rd as usize]);
    let n = words(cpu.v[rn as usize]);
    let m = words(cpu.v[rm as usize]);

    let out = match opcode {
        3 => {
            // SHA1SU0
            let mut dl = [d[0] as u64 | ((d[1] as u64) << 32), d[2] as u64 | ((d[3] as u64) << 32)];
            let nl = n[0] as u64 | ((n[1] as u64) << 32);
            let ml = [m[0] as u64 | ((m[1] as u64) << 32), m[2] as u64 | ((m[3] as u64) << 32)];
            dl[0] ^= dl[1] ^ ml[0];
            dl[1] ^= nl ^ ml[1];
            u128::from(dl[0]) | (u128::from(dl[1]) << 64)
        }
        0 | 1 | 2 => {
            let mut nn = n;
            for i in 0..4 {
                let f = match opcode {
                    0 => cho(d[1], d[2], d[3]),
                    1 => par(d[1], d[2], d[3]),
                    _ => maj(d[1], d[2], d[3]),
                };
                let t = f
                    .wrapping_add(d[0].rotate_left(5))
                    .wrapping_add(nn[0])
                    .wrapping_add(m[i]);
                nn[0] = d[3];
                d[3] = d[2];
                d[2] = d[1].rotate_right(2);
                d[1] = d[0];
                d[0] = t;
            }
            pack(d)
        }
        4 => sha256h(d, n, m, false), // SHA256H
        5 => sha256h(d, n, m, true),  // SHA256H2
        _ => {
            // SHA256SU1
            d[0] = d[0].wrapping_add(sml_s1(m[2])).wrapping_add(n[1]);
            d[1] = d[1].wrapping_add(sml_s1(m[3])).wrapping_add(n[2]);
            d[2] = d[2].wrapping_add(sml_s1(d[0])).wrapping_add(n[3]);
            d[3] = d[3].wrapping_add(sml_s1(d[1])).wrapping_add(m[0]);
            pack(d)
        }
    };
    cpu.v[rd as usize] = out;
    None
}

/// SHA256H (part2=false) / SHA256H2 (part2=true).
fn sha256h(mut d: [u32; 4], mut n: [u32; 4], m: [u32; 4], part2: bool) -> u128 {
    for i in 0..4 {
        if part2 {
            let t = cho(d[0], d[1], d[2])
                .wrapping_add(d[3])
                .wrapping_add(big_s1(d[0]))
                .wrapping_add(m[i]);
            d[3] = d[2];
            d[2] = d[1];
            d[1] = d[0];
            d[0] = n[3 - i].wrapping_add(t);
        } else {
            let mut t = cho(n[0], n[1], n[2])
                .wrapping_add(n[3])
                .wrapping_add(big_s1(n[0]))
                .wrapping_add(m[i]);
            n[3] = n[2];
            n[2] = n[1];
            n[1] = n[0];
            n[0] = d[3].wrapping_add(t);
            t = t.wrapping_add(maj(d[0], d[1], d[2])).wrapping_add(big_s0(d[0]));
            d[3] = d[2];
            d[2] = d[1];
            d[1] = d[0];
            d[0] = t;
        }
    }
    pack(d)
}

pub(crate) fn two_reg(cpu: &mut CpuState, opcode: u8, rn: u8, rd: u8) -> Option<u64> {
    let m = words(cpu.v[rn as usize]);
    let mut d = words(cpu.v[rd as usize]);
    let out = match opcode {
        0 => {
            // SHA1H
            pack([m[0].rotate_right(2), 0, 0, 0])
        }
        1 => {
            // SHA1SU1
            d[0] = (d[0] ^ m[1]).rotate_left(1);
            d[1] = (d[1] ^ m[2]).rotate_left(1);
            d[2] = (d[2] ^ m[3]).rotate_left(1);
            d[3] = (d[3] ^ d[0]).rotate_left(1);
            pack(d)
        }
        _ => {
            // SHA256SU0
            d[0] = d[0].wrapping_add(sml_s0(d[1]));
            d[1] = d[1].wrapping_add(sml_s0(d[2]));
            d[2] = d[2].wrapping_add(sml_s0(d[3]));
            d[3] = d[3].wrapping_add(sml_s0(m[0]));
            pack(d)
        }
    };
    cpu.v[rd as usize] = out;
    None
}
