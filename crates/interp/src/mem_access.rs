//! Sized, MMU-translated memory read/write helpers shared by the load/store,
//! atomic, compare-swap, and exclusive executors. `size` is log2 of the width
//! in bytes. The virtual address is translated to physical via [`crate::mmu`]
//! (identity when the MMU is off).
//!
//! When translation faults, the access is dropped (reads return 0, writes are
//! suppressed) and the fault is recorded on [`CpuState::pending_abort`]; the run
//! loop drains it after the instruction and vectors to a Data Abort. Callers
//! should check `cpu.pending_abort` before committing any base-register
//! writeback so the instruction can be retried cleanly after the handler.

use aarch64_cpu_state::{Abort, CpuState};
use aarch64_decoder::sysreg_key;

use crate::memory::GuestMem;
use crate::mmu::{self, Access};

/// 4KB translation granule (the only granule we model).
const PAGE: u64 = 0x1000;

/// Translate for a data access at exception level `el` (the current EL for
/// ordinary accesses, 0 for unprivileged LDTR/STTR), recording a Data Abort on
/// failure. Returns the physical address, or `None` if the access faulted.
fn xlate(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, write: bool, el: u8) -> Option<u64> {
    let access = if write { Access::Write } else { Access::Read };
    match mmu::translate(cpu, mem, va, access, el) {
        Ok(pa) => Some(pa),
        Err(fsc) => {
            cpu.pending_abort = Some(Abort { far: va, write, fsc });
            None
        }
    }
}

/// True when an `n`-byte access starting at `va` stays within a single page, so
/// one translation covers it. Consecutive VA pages can map to *non-adjacent*
/// physical pages, so a crossing access must translate each page separately.
fn in_one_page(va: u64, n: u64) -> bool {
    (va & (PAGE - 1)) + n <= PAGE
}

/// Gather `n` (<=16) bytes from `va` into a little-endian u128, translating each
/// page the access spans at EL `el`. Sets `pending_abort` and returns 0 on fault.
fn gather(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, n: u64, el: u8) -> u128 {
    let mut val = 0u128;
    let mut done = 0u64;
    while done < n {
        let cur = va + done;
        let page_end = (cur & !(PAGE - 1)) + PAGE;
        let chunk = (n - done).min(page_end - cur);
        let Some(pa) = xlate(cpu, mem, cur, false, el) else { return 0 };
        for i in 0..chunk {
            val |= u128::from(mem.read_u8(pa + i)) << ((done + i) * 8);
        }
        done += chunk;
    }
    val
}

/// Scatter the low `n` (<=16) bytes of `val` to `va` at EL `el`, translating each
/// page the access spans. Sets `pending_abort` (and stops) on a fault.
fn scatter(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, n: u64, val: u128, el: u8) {
    let mut done = 0u64;
    while done < n {
        let cur = va + done;
        let page_end = (cur & !(PAGE - 1)) + PAGE;
        let chunk = (n - done).min(page_end - cur);
        let Some(pa) = xlate(cpu, mem, cur, true, el) else { return };
        for i in 0..chunk {
            mem.write_u8(pa + i, (val >> ((done + i) * 8)) as u8);
        }
        done += chunk;
    }
}

/// Sized read evaluated at exception level `el`.
fn read_at(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, size: u8, el: u8) -> u64 {
    let n = 1u64 << size;
    // Fast path: the whole access is within one page -> a single translation.
    if in_one_page(va, n) {
        let Some(pa) = xlate(cpu, mem, va, false, el) else { return 0 };
        return match size {
            0 => u64::from(mem.read_u8(pa)),
            1 => u64::from(mem.read_u16(pa)),
            2 => u64::from(mem.read_u32(pa)),
            _ => mem.read_u64(pa),
        };
    }
    gather(cpu, mem, va, n, el) as u64
}

/// Sized write evaluated at exception level `el`.
fn write_at(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, size: u8, val: u64, el: u8) {
    let n = 1u64 << size;
    if in_one_page(va, n) {
        let Some(pa) = xlate(cpu, mem, va, true, el) else { return };
        match size {
            0 => mem.write_u8(pa, val as u8),
            1 => mem.write_u16(pa, val as u16),
            2 => mem.write_u32(pa, val as u32),
            _ => mem.write_u64(pa, val),
        }
        return;
    }
    scatter(cpu, mem, va, n, u128::from(val), el);
}

/// Ordinary (privileged-at-current-EL) sized read.
pub(crate) fn read(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, size: u8) -> u64 {
    read_at(cpu, mem, va, size, cpu.el)
}

/// Ordinary (privileged-at-current-EL) sized write.
pub(crate) fn write(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, size: u8, val: u64) {
    write_at(cpu, mem, va, size, val, cpu.el);
}

/// ESR Data Fault Status Code for an Alignment fault (`0b100001`).
const FSC_ALIGNMENT: u8 = 0b10_0001;

/// Exclusive (LDXR/STXR) and LSE atomic/CAS accesses must be naturally aligned
/// to their transfer size regardless of `SCTLR.A`; an unaligned address takes an
/// Alignment Data Abort. Records the abort and returns `true` if `va` is
/// misaligned for a `bytes`-byte access, so the caller bails before touching
/// memory or the destination register (the run loop then vectors the abort).
pub(crate) fn align_check(cpu: &mut CpuState, va: u64, bytes: u64, write: bool) -> bool {
    if va & (bytes - 1) != 0 {
        cpu.pending_abort = Some(Abort { far: va, write, fsc: FSC_ALIGNMENT });
        true
    } else {
        false
    }
}

/// Unprivileged sized read (LDTR): permission-checked as EL0 even from EL1.
pub(crate) fn read_unpriv(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, size: u8) -> u64 {
    read_at(cpu, mem, va, size, 0)
}

/// Unprivileged sized write (STTR): permission-checked as EL0 even from EL1.
pub(crate) fn write_unpriv(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, size: u8, val: u64) {
    write_at(cpu, mem, va, size, val, 0);
}

/// Read `1 << log2` bytes (log2 0..=4) into a u128, for SIMD/FP accesses.
pub(crate) fn read_vec(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, log2: u8) -> u128 {
    gather(cpu, mem, va, 1u64 << log2, cpu.el)
}

/// Write the low `1 << log2` bytes of `val` to memory.
pub(crate) fn write_vec(cpu: &mut CpuState, mem: &mut dyn GuestMem, va: u64, log2: u8, val: u128) {
    scatter(cpu, mem, va, 1u64 << log2, val, cpu.el);
}

/// DC ZVA — zero the naturally-aligned block containing `X[rt]`. The block size
/// is `4 << DCZID_EL0.BS` bytes (the same value the guest reads to size its
/// loop, so they agree). Stops early if a store faults; the instruction retries.
pub(crate) fn dc_zva(cpu: &mut CpuState, mem: &mut dyn GuestMem, rt: u8) {
    let dczid = cpu.sysregs.get(&sysreg_key(3, 3, 0, 0, 7)).copied().unwrap_or(0);
    let bytes = 4u64 << (dczid & 0xf);
    let base = cpu.read_gpr(rt, false) & !(bytes - 1);
    for off in 0..bytes {
        write(cpu, mem, base + off, 0, 0);
        if cpu.pending_abort.is_some() {
            return;
        }
    }
}
