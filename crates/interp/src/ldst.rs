//! Load/store of a single register, for every addressing mode.
//!
//! The decoder normalizes each encoding class into an [`AddrMode`]; this routine
//! computes the effective address (and any base writeback), then performs the
//! sized, optionally sign-extending load or the sized store. The other
//! load/store classes (pairs, exclusives, atomics, compare-and-swap) live in the
//! sibling submodules below.

pub(crate) mod atomic;
pub(crate) mod cas;
pub(crate) mod excl;
pub(crate) mod pair;

use aarch64_cpu_state::CpuState;
use aarch64_decoder::AddrMode;

use crate::mem_access;
use crate::memory::GuestMem;

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    mem: &mut dyn GuestMem,
    size: u8,
    is_load: bool,
    signed: bool,
    dst64: bool,
    vec: bool,
    unpriv: bool,
    rt: u8,
    addr: AddrMode,
    pc: u64,
) -> Option<u64> {
    let (ea, writeback) = effective_address(cpu, addr, pc);

    if vec {
        // SIMD/FP register access: `size` is log2 bytes (0..=4). Loads zero the
        // rest of the V register; no sign extension.
        if is_load {
            let val = mem_access::read_vec(cpu, mem, ea, size);
            if cpu.pending_abort.is_some() {
                return None; // faulted: commit nothing; the instruction retries
            }
            cpu.v[rt as usize] = val;
        } else {
            let val = cpu.v[rt as usize];
            mem_access::write_vec(cpu, mem, ea, size, val);
            if cpu.pending_abort.is_some() {
                return None;
            }
        }
        writeback_base(cpu, writeback);
        return None;
    }

    if is_load {
        // LDTR (`unpriv`) is permission-checked at EL0 even from EL1.
        let raw = if unpriv {
            mem_access::read_unpriv(cpu, mem, ea, size)
        } else {
            mem_access::read(cpu, mem, ea, size)
        };
        // A faulting load must NOT write its destination: the value is bogus, and
        // if the destination doubles as the base register (e.g. `ldr x9, [x9]`)
        // clobbering it would corrupt the address when the instruction retries.
        if cpu.pending_abort.is_some() {
            return None;
        }
        let value = if signed {
            let bits = 8u32 << size;
            let sh = 64 - bits;
            ((raw << sh) as i64 >> sh) as u64
        } else {
            raw
        };
        // `dst64` picks the X/W result width; the W write zero-extends.
        if dst64 {
            cpu.write_gpr(rt, false, value);
        } else {
            cpu.write_gpr_w(rt, false, value);
        }
    } else {
        // Store the low `1<<size` bytes of Rt (ZR reads as 0). STTR (`unpriv`)
        // is permission-checked at EL0.
        let val = cpu.read_gpr(rt, false);
        if unpriv {
            mem_access::write_unpriv(cpu, mem, ea, size, val);
        } else {
            mem_access::write(cpu, mem, ea, size, val);
        }
        if cpu.pending_abort.is_some() {
            return None;
        }
    }

    writeback_base(cpu, writeback);
    None
}

/// Apply a pre/post-index base writeback — but only if the access did not fault.
/// On a translation fault the instruction is retried after the handler, so the
/// writeback must not have already been committed (it would double-apply).
fn writeback_base(cpu: &mut CpuState, writeback: Option<(u8, u64)>) {
    if cpu.pending_abort.is_some() {
        return;
    }
    if let Some((rn, new_base)) = writeback {
        cpu.write_gpr(rn, true, new_base);
    }
}

/// Returns `(effective_address, optional base writeback)`.
fn effective_address(cpu: &CpuState, addr: AddrMode, pc: u64) -> (u64, Option<(u8, u64)>) {
    match addr {
        AddrMode::UnsignedImm { rn, imm } => (cpu.read_gpr(rn, true).wrapping_add(imm), None),
        AddrMode::Unscaled { rn, imm } => {
            (cpu.read_gpr(rn, true).wrapping_add(imm as u64), None)
        }
        AddrMode::PreIndex { rn, imm } => {
            let base = cpu.read_gpr(rn, true).wrapping_add(imm as u64);
            (base, Some((rn, base)))
        }
        AddrMode::PostIndex { rn, imm } => {
            let base = cpu.read_gpr(rn, true);
            (base, Some((rn, base.wrapping_add(imm as u64))))
        }
        AddrMode::RegOffset { rn, rm, option, shift } => {
            let off = extend_offset(cpu, rm, option, shift);
            (cpu.read_gpr(rn, true).wrapping_add(off), None)
        }
        AddrMode::Literal { offset } => (pc.wrapping_add(offset as u64), None),
    }
}

/// Extend/scale the index register for a register-offset address.
fn extend_offset(cpu: &CpuState, rm: u8, option: u8, shift: u8) -> u64 {
    let v = cpu.read_gpr(rm, false);
    let extended = match option {
        0b010 => u64::from(v as u32),                  // UXTW
        0b110 => (v as u32 as i32 as i64) as u64,      // SXTW
        0b111 => v,                                    // SXTX
        _ => v,                                        // LSL / UXTX (option 011)
    };
    extended << shift
}
