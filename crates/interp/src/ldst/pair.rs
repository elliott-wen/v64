//! Load/store pair: LDP/STP and LDPSW.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::PairIndex;

use crate::mem_access;
use crate::memory::GuestMem;

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec<M: GuestMem>(
    cpu: &mut CpuState,
    mem: &mut M,
    is_load: bool,
    signed: bool,
    width8: bool,
    vec: bool,
    vesize: u8,
    rt: u8,
    rt2: u8,
    rn: u8,
    offset: i64,
    index: PairIndex,
) -> Option<u64> {
    let base = cpu.read_gpr(rn, true);
    let ea = match index {
        PairIndex::Post => base,
        PairIndex::Offset | PairIndex::Pre => base.wrapping_add(offset as u64),
    };

    let do_writeback =
        |cpu: &mut CpuState| if matches!(index, PairIndex::Pre | PairIndex::Post) {
            // Suppress writeback if a fault is pending — the instruction retries.
            if cpu.pending_abort.is_none() {
                cpu.write_gpr(rn, true, base.wrapping_add(offset as u64));
            }
        };

    if vec {
        let step = 1u64 << vesize;
        if is_load {
            let v1 = mem_access::read_vec(cpu, mem, ea, vesize);
            let v2 = mem_access::read_vec(cpu, mem, ea + step, vesize);
            if cpu.pending_abort.is_some() {
                return None; // faulted: commit nothing; the instruction retries
            }
            cpu.v[rt as usize] = v1;
            cpu.v[rt2 as usize] = v2;
        } else {
            let v1 = cpu.v[rt as usize];
            let v2 = cpu.v[rt2 as usize];
            mem_access::write_vec(cpu, mem, ea, vesize, v1);
            mem_access::write_vec(cpu, mem, ea + step, vesize, v2);
            if cpu.pending_abort.is_some() {
                return None;
            }
        }
        do_writeback(cpu);
        return None;
    }

    let esize = if width8 { 8 } else { 4 };

    if is_load {
        let v1 = load_elem(cpu, mem, ea, width8, signed);
        let v2 = load_elem(cpu, mem, ea + esize, width8, signed);
        // A faulting load commits no destination (it may alias the base, and the
        // instruction re-runs after the handler).
        if cpu.pending_abort.is_some() {
            return None;
        }
        // LDPSW and the 64-bit form write X; the 32-bit form writes W.
        if width8 || signed {
            cpu.write_gpr(rt, false, v1);
            cpu.write_gpr(rt2, false, v2);
        } else {
            cpu.write_gpr_w(rt, false, v1);
            cpu.write_gpr_w(rt2, false, v2);
        }
    } else {
        let size = if width8 { 3 } else { 2 };
        let v1 = cpu.read_gpr(rt, false);
        let v2 = cpu.read_gpr(rt2, false);
        mem_access::write(cpu, mem, ea, size, v1);
        mem_access::write(cpu, mem, ea + esize, size, v2);
        if cpu.pending_abort.is_some() {
            return None;
        }
    }

    do_writeback(cpu);
    None
}

fn load_elem<M: GuestMem>(cpu: &mut CpuState, mem: &mut M, addr: u64, width8: bool, signed: bool) -> u64 {
    let size = if width8 { 3 } else { 2 };
    let raw = mem_access::read(cpu, mem, addr, size);
    if signed {
        i64::from(raw as u32 as i32) as u64 // LDPSW: sign-extend the 32-bit element
    } else {
        raw
    }
}
