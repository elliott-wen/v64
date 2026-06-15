//! Compare-and-swap (CAS): if `[Rn] == Rs`, store Rt. Rs always receives the
//! old memory value.

use aarch64_cpu_state::CpuState;

use crate::mem_access::{read, write};
use crate::memory::GuestMem;

pub(crate) fn exec(
    cpu: &mut CpuState,
    mem: &mut dyn GuestMem,
    size: u8,
    rs: u8,
    rn: u8,
    rt: u8,
) -> Option<u64> {
    let addr = cpu.read_gpr(rn, true);
    let w = 8u32 << size;
    let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };

    let old = read(cpu, mem, addr, size);
    let compare = cpu.read_gpr(rs, false) & mask;
    if old == compare {
        write(cpu, mem, addr, size, cpu.read_gpr(rt, false) & mask);
    }
    if size == 3 {
        cpu.write_gpr(rs, false, old);
    } else {
        cpu.write_gpr_w(rs, false, old);
    }
    None
}
