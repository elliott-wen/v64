//! Load/store exclusive (LDXR/LDAXR, STXR/STLXR) via the exclusive monitor.

use aarch64_cpu_state::CpuState;

use crate::mem_access::{read, write};
use crate::memory::GuestMem;

/// LDXR/LDAXR: load and arm the exclusive monitor with `(addr, value)`.
pub(crate) fn load(cpu: &mut CpuState, mem: &dyn GuestMem, size: u8, rt: u8, rn: u8) -> Option<u64> {
    let addr = cpu.read_gpr(rn, true);
    let val = read(cpu, mem, addr, size);
    cpu.excl = Some((addr, val));
    if size == 3 {
        cpu.write_gpr(rt, false, val);
    } else {
        cpu.write_gpr_w(rt, false, val);
    }
    None
}

/// STXR/STLXR: store iff the monitor is still armed for this address and memory
/// is unchanged (an intervening write changes the value and fails the store).
/// Ws gets 0 on success, 1 on failure; the monitor is always cleared.
pub(crate) fn store(
    cpu: &mut CpuState,
    mem: &mut dyn GuestMem,
    size: u8,
    rs: u8,
    rt: u8,
    rn: u8,
) -> Option<u64> {
    let addr = cpu.read_gpr(rn, true);
    let success = matches!(cpu.excl, Some((a, v)) if a == addr && read(cpu, mem, addr, size) == v);
    if success {
        write(cpu, mem, addr, size, cpu.read_gpr(rt, false));
    }
    cpu.excl = None;
    cpu.write_gpr_w(rs, false, u64::from(!success)); // 0 = success
    None
}
