//! Advanced SIMD modified immediate: MOVI/MVNI/ORR/BIC (integer cmodes).

use aarch64_cpu_state::CpuState;

pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    op: bool,
    cmode: u8,
    imm8: u8,
    rd: u8,
) -> Option<u64> {
    let imm = expand(op, cmode, imm8);
    let val = if q {
        (u128::from(imm) << 64) | u128::from(imm) // replicate to 128 bits
    } else {
        u128::from(imm)
    };
    let cmode_hi = cmode >> 1;

    let result = if cmode_hi <= 0b101 && cmode & 1 == 1 {
        // ORR (op=0) / BIC (op=1) immediate: combine with Vd.
        if op {
            cpu.v[rd as usize] & !val
        } else {
            cpu.v[rd as usize] | val
        }
    } else if op && cmode_hi <= 0b110 {
        // MVNI: invert (cmode 1110 with op=1 bakes op into `expand`, so excluded).
        !val
    } else {
        val // MOVI
    };

    // A 64-bit (Q=0) result zeroes the upper half of Vd.
    let mask = if q { u128::MAX } else { u128::from(u64::MAX) };
    cpu.v[rd as usize] = result & mask;
    None
}

/// ARM `AdvSIMDExpandImm` for the integer cmodes -> a 64-bit element value.
fn expand(op: bool, cmode: u8, imm8: u8) -> u64 {
    let i = u64::from(imm8);
    let rep32 = |v: u64| (v & 0xffff_ffff) | ((v & 0xffff_ffff) << 32);
    let rep16 = |v: u64| {
        let h = v & 0xffff;
        h | (h << 16) | (h << 32) | (h << 48)
    };
    match cmode >> 1 {
        0b000 => rep32(i),
        0b001 => rep32(i << 8),
        0b010 => rep32(i << 16),
        0b011 => rep32(i << 24),
        0b100 => rep16(i),
        0b101 => rep16(i << 8),
        0b110 => {
            // Shifting ones: low byte(s) all ones.
            let v = if cmode & 1 == 0 { (i << 8) | 0xff } else { (i << 16) | 0xffff };
            rep32(v)
        }
        _ => {
            // cmode 1110: MOVI byte-replicate (op=0) or bit-to-byte (op=1).
            if !op {
                let mut v = 0u64;
                for k in 0..8 {
                    v |= i << (8 * k);
                }
                v
            } else {
                let mut v = 0u64;
                for k in 0..8 {
                    if (imm8 >> k) & 1 == 1 {
                        v |= 0xffu64 << (8 * k);
                    }
                }
                v
            }
        }
    }
}
