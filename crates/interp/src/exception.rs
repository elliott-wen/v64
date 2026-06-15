//! Exception entry (SVC) and return (ERET), plus the MSR-immediate PSTATE
//! writes (SPSel/DAIF).
//!
//! Modelled on the ARM ARM `AArch64.TakeException` / `ExceptionReturn`. The
//! EL1 exception registers (VBAR/ELR/SPSR/ESR/FAR) live in the system-register
//! map so MRS/MSR and this logic stay consistent.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;

/// Exception class for an SVC taken from AArch64.
const EC_SVC: u32 = 0x15;

fn key_vbar_el1() -> u32 {
    sysreg_key(3, 0, 12, 0, 0)
}
fn key_elr_el1() -> u32 {
    sysreg_key(3, 0, 4, 0, 1)
}
fn key_spsr_el1() -> u32 {
    sysreg_key(3, 0, 4, 0, 0)
}
fn key_esr_el1() -> u32 {
    sysreg_key(3, 0, 5, 2, 0)
}

/// SP_EL0 / SP_EL1 system-register keys (handled via SP banking, not the map).
pub(crate) fn key_sp_el0() -> u32 {
    sysreg_key(3, 0, 4, 1, 0)
}
pub(crate) fn key_sp_el1() -> u32 {
    sysreg_key(3, 4, 4, 1, 0)
}

/// Vector-table *type* offsets within a source group: synchronous = 0x000,
/// IRQ = 0x080, FIQ = 0x100, SError = 0x180 (ARM ARM, "Exception vectors").
const VEC_SYNC: u64 = 0x000;
const VEC_IRQ: u64 = 0x080;

/// Common exception-entry sequence to EL1: pick the vector slot, save
/// ELR/SPSR, switch to EL1h, mask interrupts, and return the new PC. `vec_type`
/// is the within-group type offset (`VEC_SYNC`/`VEC_IRQ`).
fn enter_el1(cpu: &mut CpuState, return_addr: u64, vec_type: u64) -> u64 {
    let target_el = 1u8;
    // Source group: lower EL (AArch64) = 0x400; same EL = 0x200 (SP_ELx) or
    // 0x000 (SP_EL0). The type offset selects sync/IRQ/FIQ/SError within it.
    let group = if cpu.el < target_el {
        0x400
    } else if cpu.spsel {
        0x200
    } else {
        0x000
    };
    let vbar = cpu.sysregs.get(&key_vbar_el1()).copied().unwrap_or(0);
    let spsr = cpu.pstate();

    cpu.sysregs.insert(key_elr_el1(), return_addr);
    cpu.sysregs.insert(key_spsr_el1(), spsr);

    cpu.set_el_spsel(target_el, true); // EL1h
    cpu.daif = 0b1111; // mask D, A, I, F
    vbar.wrapping_add(group + vec_type)
}

/// Take a synchronous exception to EL1, also recording ESR_EL1. Returns the new
/// PC.
fn take_exception(cpu: &mut CpuState, ec: u32, iss: u32, return_addr: u64) -> u64 {
    let pc = enter_el1(cpu, return_addr, VEC_SYNC);
    cpu.sysregs
        .insert(key_esr_el1(), (u64::from(ec) << 26) | (1 << 25) | u64::from(iss));
    pc
}

/// Take an asynchronous IRQ to EL1. Called by the machine loop *before*
/// executing the instruction at `cpu.pc`, so the saved ELR is `cpu.pc` itself
/// (execution resumes there on ERET). ESR is UNKNOWN for IRQ, so it is left
/// untouched. Returns the IRQ vector PC.
pub fn take_irq(cpu: &mut CpuState) -> u64 {
    enter_el1(cpu, cpu.pc, VEC_IRQ)
}

/// SVC #imm — exception to EL1. `pc` is the SVC's own address.
pub(crate) fn svc(cpu: &mut CpuState, imm16: u16, pc: u64) -> Option<u64> {
    Some(take_exception(cpu, EC_SVC, u32::from(imm16), pc.wrapping_add(4)))
}

/// ERET — restore PC from ELR_EL1 and PSTATE from SPSR_EL1.
pub(crate) fn eret(cpu: &mut CpuState) -> Option<u64> {
    let elr = cpu.sysregs.get(&key_elr_el1()).copied().unwrap_or(0);
    let spsr = cpu.sysregs.get(&key_spsr_el1()).copied().unwrap_or(0);
    cpu.set_pstate(spsr);
    Some(elr)
}

/// MSR (immediate): SPSel / DAIFSet / DAIFClr.
pub(crate) fn msr_imm(cpu: &mut CpuState, op1: u8, op2: u8, crm: u8) -> Option<u64> {
    match (op1, op2) {
        (0, 5) => cpu.set_el_spsel(cpu.el, crm & 1 == 1), // SPSel
        (3, 6) => cpu.daif |= crm,                        // DAIFSet
        (3, 7) => cpu.daif &= !crm,                       // DAIFClr
        _ => {}                                           // other PSTATE fields ignored
    }
    None
}
