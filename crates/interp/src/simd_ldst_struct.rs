//! Advanced SIMD load/store structures: LD1-4/ST1-4 (multiple and single).

use aarch64_cpu_state::CpuState;

use crate::mem_access;
use crate::memory::Memory;

/// Write the low `1<<size` bytes of `val` into element `byte_off` of V[tt].
fn set_elem(cpu: &mut CpuState, tt: u8, byte_off: u32, size: u8, val: u64) {
    let mut v = cpu.v[tt as usize];
    for i in 0..(1u32 << size) {
        let sh = (byte_off + i) * 8;
        v &= !(0xffu128 << sh);
        v |= u128::from((val >> (i * 8)) as u8) << sh;
    }
    cpu.v[tt as usize] = v;
}

fn get_elem(cpu: &CpuState, tt: u8, byte_off: u32, size: u8) -> u64 {
    let v = cpu.v[tt as usize];
    let mut r = 0u64;
    for i in 0..(1u32 << size) {
        r |= u64::from((v >> ((byte_off + i) * 8)) as u8) << (i * 8);
    }
    r
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn multi(
    cpu: &mut CpuState,
    mem: &mut Memory,
    is_load: bool,
    q: bool,
    postidx: bool,
    rm: u8,
    rn: u8,
    rt: u8,
    size: u8,
    rpt: u8,
    selem: u8,
) -> Option<u64> {
    let ebytes = 1u64 << size;
    let elements = (if q { 16 } else { 8 }) / (1u32 << size);
    let start = cpu.read_gpr(rn, true);
    let mut addr = start;

    for r in 0..u32::from(rpt) {
        for e in 0..elements {
            for xs in 0..u32::from(selem) {
                let tt = ((u32::from(rt) + r + xs) % 32) as u8;
                let off = e * (1u32 << size);
                if is_load {
                    let val = mem_access::read(cpu, mem, addr, size);
                    set_elem(cpu, tt, off, size, val);
                } else {
                    let val = get_elem(cpu, tt, off, size);
                    mem_access::write(cpu, mem, addr, size, val);
                }
                addr = addr.wrapping_add(ebytes);
            }
        }
    }

    // Non-Q loads zero the upper 64 bits of every written register.
    if is_load && !q {
        for k in 0..u32::from(rpt) * u32::from(selem) {
            let tt = ((u32::from(rt) + k) % 32) as usize;
            cpu.v[tt] &= u128::from(u64::MAX);
        }
    }

    if postidx {
        let inc = if rm == 31 {
            u64::from(rpt) * u64::from(elements) * u64::from(selem) * ebytes
        } else {
            cpu.read_gpr(rm, false)
        };
        cpu.write_gpr(rn, true, start.wrapping_add(inc));
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn single(
    cpu: &mut CpuState,
    mem: &mut Memory,
    is_load: bool,
    replicate: bool,
    postidx: bool,
    rm: u8,
    rn: u8,
    rt: u8,
    size: u8,
    selem: u8,
    index: u8,
    q: bool,
) -> Option<u64> {
    let ebytes = 1u64 << size;
    let start = cpu.read_gpr(rn, true);
    let mut addr = start;
    let mut tt = rt;

    for _ in 0..selem {
        if replicate {
            let val = mem_access::read(cpu, mem, addr, size);
            cpu.v[tt as usize] = broadcast(val, size, q);
        } else if is_load {
            let off = u32::from(index) * (1u32 << size);
            let val = mem_access::read(cpu, mem, addr, size);
            set_elem(cpu, tt, off, size, val);
        } else {
            let off = u32::from(index) * (1u32 << size);
            let val = get_elem(cpu, tt, off, size);
            mem_access::write(cpu, mem, addr, size, val);
        }
        addr = addr.wrapping_add(ebytes);
        tt = (tt + 1) % 32;
    }

    if postidx {
        let inc = if rm == 31 { u64::from(selem) * ebytes } else { cpu.read_gpr(rm, false) };
        cpu.write_gpr(rn, true, start.wrapping_add(inc));
    }
    None
}

/// LDxR: replicate a `1<<size`-byte element across the (Q?16:8)-byte register.
fn broadcast(val: u64, size: u8, q: bool) -> u128 {
    let bytes = 1u32 << size;
    let nbytes = if q { 16u32 } else { 8 };
    let mask = if bytes >= 8 { u64::MAX } else { (1u64 << (bytes * 8)) - 1 };
    let elem = u128::from(val & mask);
    let mut v = 0u128;
    let mut i = 0;
    while i * bytes < nbytes {
        v |= elem << (i * bytes * 8);
        i += 1;
    }
    v
}
