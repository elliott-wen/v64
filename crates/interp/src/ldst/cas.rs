//! Compare-and-swap (CAS): if `[Rn] == Rs`, store Rt. Rs always receives the
//! old memory value.

use aarch64_cpu_state::CpuState;

use crate::mem_access::{align_check, read, write};
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
    if align_check(cpu, addr, 1 << size, true) {
        return None; // unaligned CAS: Alignment Data Abort
    }
    let w = 8u32 << size;
    let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };

    let old = read(cpu, mem, addr, size);
    if cpu.pending_abort.is_some() {
        return None; // faulted: don't write Rs; the instruction retries
    }
    let compare = cpu.read_gpr(rs, false) & mask;
    if old == compare {
        let store_val = cpu.read_gpr(rt, false) & mask;
        write(cpu, mem, addr, size, store_val);
        if cpu.pending_abort.is_some() {
            return None;
        }
    }
    if size == 3 {
        cpu.write_gpr(rs, false, old);
    } else {
        cpu.write_gpr_w(rs, false, old);
    }
    None
}

/// CASP: compare-and-swap pair. Compares the two elements at `[Rn]` against
/// Rs:Rs+1; on a match stores Rt:Rt+1. Rs:Rs+1 always receive the old pair.
/// `sz` selects 4-byte (0) vs 8-byte (1) elements.
pub(crate) fn cas_pair(
    cpu: &mut CpuState,
    mem: &mut dyn GuestMem,
    sz: u8,
    rs: u8,
    rn: u8,
    rt: u8,
) -> Option<u64> {
    let size = if sz == 1 { 3 } else { 2 };
    let esize = 1u64 << size;
    let addr = cpu.read_gpr(rn, true);
    if align_check(cpu, addr, 2 * esize, true) {
        return None; // unaligned CASP: aligned to the whole pair size
    }

    let mask = if size == 3 { u64::MAX } else { 0xffff_ffff };
    let old1 = read(cpu, mem, addr, size);
    let old2 = read(cpu, mem, addr + esize, size);
    if cpu.pending_abort.is_some() {
        return None;
    }
    let cmp1 = cpu.read_gpr(rs, false) & mask;
    let cmp2 = cpu.read_gpr(rs + 1, false) & mask;
    if old1 == cmp1 && old2 == cmp2 {
        let new1 = cpu.read_gpr(rt, false);
        let new2 = cpu.read_gpr(rt + 1, false);
        write(cpu, mem, addr, size, new1);
        write(cpu, mem, addr + esize, size, new2);
        if cpu.pending_abort.is_some() {
            return None;
        }
    }
    if size == 3 {
        cpu.write_gpr(rs, false, old1);
        cpu.write_gpr(rs + 1, false, old2);
    } else {
        cpu.write_gpr_w(rs, false, old1);
        cpu.write_gpr_w(rs + 1, false, old2);
    }
    None
}
