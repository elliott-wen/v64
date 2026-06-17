//! System-register moves (MRS/MSR).
//!
//! The system register file is a flat map keyed by the encoded
//! (op0,op1,CRn,CRm,op2) tuple. Reads of an unwritten register return 0 for now
//! (ID/feature registers with architectural reset values come with the wider
//! system-mode work). Writable registers round-trip, which is what early
//! differential tests exercise.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;

use crate::exception::{key_sp_el0, key_sp_el1};

// FPCR/FPSR live in dedicated CpuState fields (the FP executors read/update them
// directly), so route their MRS/MSR there rather than to the generic map.
fn key_fpcr() -> u32 {
    sysreg_key(3, 3, 4, 4, 0)
}
fn key_fpsr() -> u32 {
    sysreg_key(3, 3, 4, 4, 1)
}

// The translation-control registers live in dedicated `CpuState` fields (the
// MMU reads them on the hot path), so MRS/MSR route here instead of the map.
fn key_sctlr_el1() -> u32 {
    sysreg_key(3, 0, 1, 0, 0)
}
fn key_tcr_el1() -> u32 {
    sysreg_key(3, 0, 2, 0, 2)
}
fn key_ttbr0_el1() -> u32 {
    sysreg_key(3, 0, 2, 0, 0)
}
fn key_ttbr1_el1() -> u32 {
    sysreg_key(3, 0, 2, 0, 1)
}

/// Read a translation-control register from its dedicated field, or `None` if
/// `key` isn't one of them.
fn read_xlate_reg(cpu: &CpuState, key: u32) -> Option<u64> {
    if key == key_sctlr_el1() {
        Some(cpu.sctlr_el1)
    } else if key == key_tcr_el1() {
        Some(cpu.tcr_el1)
    } else if key == key_ttbr0_el1() {
        Some(cpu.ttbr0_el1)
    } else if key == key_ttbr1_el1() {
        Some(cpu.ttbr1_el1)
    } else {
        None
    }
}

/// Write a translation-control register to its dedicated field and flush the TLB
/// (the cached walks are now stale). Returns `true` if `key` was one of them.
fn write_xlate_reg(cpu: &mut CpuState, key: u32, val: u64) -> bool {
    let field = if key == key_sctlr_el1() {
        &mut cpu.sctlr_el1
    } else if key == key_tcr_el1() {
        &mut cpu.tcr_el1
    } else if key == key_ttbr0_el1() {
        &mut cpu.ttbr0_el1
    } else if key == key_ttbr1_el1() {
        &mut cpu.ttbr1_el1
    } else {
        return false;
    };
    *field = val;
    cpu.flush_tlb();
    true
}

pub(crate) fn exec(cpu: &mut CpuState, read: bool, key: u32, rt: u8) -> Option<u64> {
    // SP_EL0/SP_EL1 are banked stack pointers, not plain map entries.
    let sp_bank = if key == key_sp_el0() {
        Some(0)
    } else if key == key_sp_el1() {
        Some(1)
    } else {
        None
    };

    if read {
        // MRS: sysreg -> Rt (ZR discards at r31, but reads still happen).
        let val = if key == key_fpcr() {
            cpu.fpcr
        } else if key == key_fpsr() {
            cpu.fpsr
        } else if let Some(v) = read_xlate_reg(cpu, key) {
            v
        } else {
            match sp_bank {
                Some(idx) => cpu.read_sp_el(idx),
                // Timer registers have computed reads (TVAL/ISTATUS); others fall
                // back to the plain map.
                None => crate::timer::read(cpu, key)
                    .unwrap_or_else(|| cpu.sysregs.get(&key).copied().unwrap_or(0)),
            }
        };
        cpu.write_gpr(rt, false, val);
    } else {
        // MSR: Rt -> sysreg (Rt == 31 reads as zero).
        let val = cpu.read_gpr(rt, false);
        if key == key_fpcr() {
            cpu.fpcr = val;
        } else if key == key_fpsr() {
            cpu.fpsr = val;
        } else if write_xlate_reg(cpu, key, val) {
            // Routed to a dedicated translation-control field; TLB flushed.
        } else {
            match sp_bank {
                Some(idx) => cpu.write_sp_el(idx, val),
                // A timer TVAL write becomes a CVAL store; others round-trip.
                None if crate::timer::write(cpu, key, val) => {}
                None => {
                    cpu.sysregs.insert(key, val);
                }
            }
        }
    }
    None
}
