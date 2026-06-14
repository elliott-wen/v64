//! Load/store pair: LDP/STP and LDPSW.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::PairIndex;

use crate::mem_access;
use crate::memory::Memory;

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    mem: &mut Memory,
    is_load: bool,
    signed: bool,
    width8: bool,
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
    let esize = if width8 { 8 } else { 4 };

    if is_load {
        let (v1, v2) = (
            load_elem(cpu, mem, ea, width8, signed),
            load_elem(cpu, mem, ea + esize, width8, signed),
        );
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
        mem_access::write(cpu, mem, ea, size, cpu.read_gpr(rt, false));
        mem_access::write(cpu, mem, ea + esize, size, cpu.read_gpr(rt2, false));
    }

    if matches!(index, PairIndex::Pre | PairIndex::Post) {
        cpu.write_gpr(rn, true, base.wrapping_add(offset as u64));
    }
    None
}

fn load_elem(cpu: &CpuState, mem: &Memory, addr: u64, width8: bool, signed: bool) -> u64 {
    let size = if width8 { 3 } else { 2 };
    let raw = mem_access::read(cpu, mem, addr, size);
    if signed {
        i64::from(raw as u32 as i32) as u64 // LDPSW: sign-extend the 32-bit element
    } else {
        raw
    }
}
