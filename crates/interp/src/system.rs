//! System-register moves (MRS/MSR).
//!
//! The system register file is a flat map keyed by the encoded
//! (op0,op1,CRn,CRm,op2) tuple. Reads of an unwritten register return 0 for now
//! (ID/feature registers with architectural reset values come with the wider
//! system-mode work). Writable registers round-trip, which is what early
//! differential tests exercise.

use aarch64_cpu_state::CpuState;

use crate::exception::{key_sp_el0, key_sp_el1};

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
        let val = match sp_bank {
            Some(idx) => cpu.read_sp_el(idx),
            // Timer registers have computed reads (TVAL/ISTATUS); others fall
            // back to the plain map.
            None => crate::timer::read(cpu, key)
                .unwrap_or_else(|| cpu.sysregs.get(&key).copied().unwrap_or(0)),
        };
        cpu.write_gpr(rt, false, val);
    } else {
        // MSR: Rt -> sysreg (Rt == 31 reads as zero).
        let val = cpu.read_gpr(rt, false);
        match sp_bank {
            Some(idx) => cpu.write_sp_el(idx, val),
            // A timer TVAL write becomes a CVAL store; others round-trip.
            None if crate::timer::write(cpu, key, val) => {}
            None => {
                cpu.sysregs.insert(key, val);
            }
        }
    }
    None
}
