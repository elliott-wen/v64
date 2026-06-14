//! DUP (general register): replicate a GPR element across all lanes.

use aarch64_cpu_state::CpuState;

pub(crate) fn exec(cpu: &mut CpuState, q: bool, size: u8, rn: u8, rd: u8) -> Option<u64> {
    let ebits = 8usize << size;
    let src = cpu.read_gpr(rn, false);
    let elem = if ebits >= 64 {
        u128::from(src)
    } else {
        u128::from(src & ((1u64 << ebits) - 1))
    };

    let n = if q { 128 } else { 64 } / ebits;
    let mut v = 0u128;
    for i in 0..n {
        v |= elem << (i * ebits);
    }
    cpu.v[rd as usize] = v;
    None
}
