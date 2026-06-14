//! Advanced SIMD table lookup: TBL/TBX.

use aarch64_cpu_state::CpuState;

pub(crate) fn exec(
    cpu: &mut CpuState,
    q: bool,
    is_tbx: bool,
    len: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let nregs = u32::from(len) + 1;
    let table_bytes = nregs * 16;
    let idx = cpu.v[rm as usize].to_le_bytes();
    let old = cpu.v[rd as usize].to_le_bytes();
    let nbytes = if q { 16 } else { 8 };

    let mut out = [0u8; 16];
    for i in 0..nbytes {
        let sel = u32::from(idx[i]);
        if sel < table_bytes {
            // Table register and byte within it; registers wrap modulo 32.
            let reg = (u32::from(rn) + sel / 16) % 32;
            let byte = (sel % 16) as usize;
            out[i] = cpu.v[reg as usize].to_le_bytes()[byte];
        } else if is_tbx {
            out[i] = old[i]; // TBX keeps the destination byte on out-of-range
        }
        // TBL leaves the byte zero on out-of-range.
    }
    cpu.v[rd as usize] = u128::from_le_bytes(out);
    None
}
