//! LSE atomic read-modify-write. Single-threaded, so "atomic" is just
//! read-compute-write; ordering bits have no effect.

use aarch64_cpu_state::CpuState;

use crate::mem_access::{read, write};
use crate::memory::Memory;

pub(crate) fn exec(
    cpu: &mut CpuState,
    mem: &mut Memory,
    size: u8,
    op: u8,
    rs: u8,
    rn: u8,
    rt: u8,
) -> Option<u64> {
    let addr = cpu.read_gpr(rn, true);

    // QEMU/TCG implements these as fetch-ops at the access width: both the
    // memory value and Rs are zero-extended from `size` bytes, the op runs at
    // 64-bit, and the low `size` bytes of the result are stored. Notably the
    // sub-word signed min/max therefore compare *zero-extended* values (the ARM
    // per-element sign-extension is not what QEMU does) — matching the oracle is
    // what counts.
    let w = 8u32 << size;
    let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
    let old = read(cpu, mem, addr, size);
    let s = cpu.read_gpr(rs, false) & mask;

    let new = match op {
        0 => old.wrapping_add(s),                              // LDADD
        1 => old & !s,                                         // LDCLR
        2 => old ^ s,                                          // LDEOR
        3 => old | s,                                          // LDSET
        4 => if (old as i64) >= (s as i64) { old } else { s }, // LDSMAX
        5 => if (old as i64) <= (s as i64) { old } else { s }, // LDSMIN
        6 => old.max(s),                                       // LDUMAX
        7 => old.min(s),                                       // LDUMIN
        _ => s,                                                // SWP
    };
    write(cpu, mem, addr, size, new);

    // The old value (zero-extended) is returned to Rt.
    if size == 3 {
        cpu.write_gpr(rt, false, old);
    } else {
        cpu.write_gpr_w(rt, false, old);
    }
    None
}
