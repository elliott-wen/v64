//! Sized, MMU-translated memory read/write helpers shared by the load/store,
//! atomic, compare-swap, and exclusive executors. `size` is log2 of the width
//! in bytes. The virtual address is translated to physical via [`crate::mmu`]
//! (identity when the MMU is off).

use aarch64_cpu_state::CpuState;

use crate::memory::Memory;
use crate::mmu;

pub(crate) fn read(cpu: &CpuState, mem: &Memory, va: u64, size: u8) -> u64 {
    let pa = mmu::translate(cpu, mem, va);
    match size {
        0 => u64::from(mem.read_u8(pa)),
        1 => u64::from(mem.read_u16(pa)),
        2 => u64::from(mem.read_u32(pa)),
        _ => mem.read_u64(pa),
    }
}

pub(crate) fn write(cpu: &CpuState, mem: &mut Memory, va: u64, size: u8, val: u64) {
    let pa = mmu::translate(cpu, &*mem, va);
    match size {
        0 => mem.write_u8(pa, val as u8),
        1 => mem.write_u16(pa, val as u16),
        2 => mem.write_u32(pa, val as u32),
        _ => mem.write_u64(pa, val),
    }
}
