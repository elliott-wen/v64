//! Advanced SIMD across-lanes reductions: ADDV/SMAXV/UMAXV/SMINV/UMINV.

use aarch64_cpu_state::CpuState;

use crate::simd::split;

pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    u: bool,
    size: u8,
    opcode: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let ebits = 8u32 << size;
    let mask = if ebits >= 64 { u64::MAX } else { (1u64 << ebits) - 1 };
    let sext = |v: u64| {
        let s = 64 - ebits;
        ((v << s) as i64) >> s
    };
    let lanes = split(cpu.v[rn as usize], size, q);

    // SADDLV/UADDLV: sum all lanes into a *double-width* scalar (no overflow).
    if opcode == 0b00011 {
        let sum: i128 = lanes
            .iter()
            .map(|&x| if u { i128::from(x & mask) } else { i128::from(sext(x)) })
            .sum();
        let rbits = ebits * 2;
        let rmask = if rbits >= 64 { u64::MAX } else { (1u64 << rbits) - 1 };
        cpu.v[rd as usize] = u128::from((sum as u64) & rmask);
        return None;
    }

    let result = match opcode {
        0b11011 => lanes.iter().fold(0u64, |a, &x| a.wrapping_add(x)) & mask, // ADDV
        0b01010 => lanes
            .iter()
            .copied()
            .reduce(|a, b| if u { a.max(b) } else if sext(a) >= sext(b) { a } else { b })
            .unwrap_or(0), // S/U MAXV
        _ => lanes
            .iter()
            .copied()
            .reduce(|a, b| if u { a.min(b) } else if sext(a) <= sext(b) { a } else { b })
            .unwrap_or(0), // S/U MINV
    };

    // The scalar result goes in lane 0; the rest of Vd is zeroed.
    cpu.v[rd as usize] = u128::from(result & mask);
    None
}
