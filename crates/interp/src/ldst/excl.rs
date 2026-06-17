//! Load/store exclusive (LDXR/LDAXR, STXR/STLXR) via the exclusive monitor.

use aarch64_cpu_state::CpuState;

use crate::mem_access::{align_check, read, write};
use crate::memory::GuestMem;

/// LDXR/LDAXR: load and arm the exclusive monitor with `(addr, value)`.
pub(crate) fn load<M: GuestMem>(cpu: &mut CpuState, mem: &mut M, size: u8, rt: u8, rn: u8) -> Option<u64> {
    let addr = cpu.read_gpr(rn, true);
    if align_check(cpu, addr, 1 << size, false) {
        return None; // unaligned exclusive: Alignment Data Abort
    }
    let val = read(cpu, mem, addr, size);
    if cpu.pending_abort.is_some() {
        return None; // faulted: don't arm the monitor or write the destination
    }
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
pub(crate) fn store<M: GuestMem>(
    cpu: &mut CpuState,
    mem: &mut M,
    size: u8,
    rs: u8,
    rt: u8,
    rn: u8,
) -> Option<u64> {
    let addr = cpu.read_gpr(rn, true);
    if align_check(cpu, addr, 1 << size, true) {
        return None; // unaligned exclusive: Alignment Data Abort
    }
    let armed = cpu.excl; // copy out so the monitor check doesn't borrow `cpu`
    let success = matches!(armed, Some((a, v)) if a == addr && read(cpu, mem, addr, size) == v);
    if success {
        let store_val = cpu.read_gpr(rt, false);
        write(cpu, mem, addr, size, store_val);
    }
    cpu.excl = None;
    cpu.write_gpr_w(rs, false, u64::from(!success)); // 0 = success
    None
}

/// LDXP/LDAXP: load two `1<<size`-byte elements and arm the monitor. In a
/// single-threaded model the monitor only needs to detect intervening writes,
/// so it tracks the base address and the first element's value.
pub(crate) fn load_pair<M: GuestMem>(
    cpu: &mut CpuState,
    mem: &mut M,
    size: u8,
    rt: u8,
    rt2: u8,
    rn: u8,
) -> Option<u64> {
    let addr = cpu.read_gpr(rn, true);
    let esize = 1u64 << size;
    if align_check(cpu, addr, 2 * esize, false) {
        return None; // unaligned exclusive pair: aligned to the whole pair size
    }
    let v1 = read(cpu, mem, addr, size);
    let v2 = read(cpu, mem, addr + esize, size);
    if cpu.pending_abort.is_some() {
        return None;
    }
    cpu.excl = Some((addr, v1));
    if size == 3 {
        cpu.write_gpr(rt, false, v1);
        cpu.write_gpr(rt2, false, v2);
    } else {
        cpu.write_gpr_w(rt, false, v1);
        cpu.write_gpr_w(rt2, false, v2);
    }
    None
}

/// STXP/STLXP: store the pair iff the monitor is still armed for this address;
/// Ws gets 0 on success, 1 on failure. The monitor is always cleared.
pub(crate) fn store_pair<M: GuestMem>(
    cpu: &mut CpuState,
    mem: &mut M,
    size: u8,
    rs: u8,
    rt: u8,
    rt2: u8,
    rn: u8,
) -> Option<u64> {
    let addr = cpu.read_gpr(rn, true);
    let esize = 1u64 << size;
    if align_check(cpu, addr, 2 * esize, true) {
        return None; // unaligned exclusive pair: aligned to the whole pair size
    }
    let armed = cpu.excl;
    let success = matches!(armed, Some((a, v)) if a == addr && read(cpu, mem, addr, size) == v);
    if success {
        let v1 = cpu.read_gpr(rt, false);
        let v2 = cpu.read_gpr(rt2, false);
        write(cpu, mem, addr, size, v1);
        write(cpu, mem, addr + esize, size, v2);
    }
    cpu.excl = None;
    cpu.write_gpr_w(rs, false, u64::from(!success));
    None
}
