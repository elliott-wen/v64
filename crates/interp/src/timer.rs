//! ARM generic timer register semantics (MRS/MSR side).
//!
//! The timer is not a device — it's a set of CPU system registers plus one
//! output line per timer. This module implements the register *behaviour* that
//! a plain sysreg-map round-trip can't capture:
//!
//! * `CNTV_TVAL`/`CNTP_TVAL` are a convenience view of the compare value:
//!   writing `TVAL = N` means "fire `N` ticks from now" (`CVAL = count + N`),
//!   and reading returns the signed ticks remaining (`CVAL - count`).
//! * `CNTV_CTL`/`CNTP_CTL` expose a read-only `ISTATUS` bit (bit 2) that is
//!   computed live from the count and compare value, not stored.
//!
//! The live count (`CNTVCT`/`CNTPCT`) and frequency (`CNTFRQ`) are kept in the
//! sysreg map by the machine's clock; everything here reads them from there, so
//! the time source stays out of the interpreter.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;

// Counter / frequency (read directly from the map; maintained by the machine).
fn key_cntfrq() -> u32 {
    sysreg_key(3, 3, 14, 0, 0)
}
fn key_cntpct() -> u32 {
    sysreg_key(3, 3, 14, 0, 1)
}
fn key_cntvct() -> u32 {
    sysreg_key(3, 3, 14, 0, 2)
}
// Physical timer (EL0 view).
fn key_cntp_tval() -> u32 {
    sysreg_key(3, 3, 14, 2, 0)
}
fn key_cntp_ctl() -> u32 {
    sysreg_key(3, 3, 14, 2, 1)
}
fn key_cntp_cval() -> u32 {
    sysreg_key(3, 3, 14, 2, 2)
}
// Virtual timer (EL0 view) — the one Linux uses by default on arm64.
fn key_cntv_tval() -> u32 {
    sysreg_key(3, 3, 14, 3, 0)
}
fn key_cntv_ctl() -> u32 {
    sysreg_key(3, 3, 14, 3, 1)
}
fn key_cntv_cval() -> u32 {
    sysreg_key(3, 3, 14, 3, 2)
}

fn get(cpu: &CpuState, key: u32) -> u64 {
    cpu.sysregs.get(&key).copied().unwrap_or(0)
}

/// CTL read with the live `ISTATUS` bit: bit0 ENABLE, bit1 IMASK (both stored),
/// bit2 ISTATUS = ENABLE && count >= compare.
fn ctl_read(cpu: &CpuState, ctl_key: u32, cval_key: u32, cnt_key: u32) -> u64 {
    let ctl = get(cpu, ctl_key) & 0b11;
    let enabled = ctl & 1 != 0;
    let istatus = enabled && get(cpu, cnt_key) >= get(cpu, cval_key);
    ctl | (u64::from(istatus) << 2)
}

/// TVAL read: signed ticks remaining, truncated to 32 bits (TVAL is a W reg).
fn tval_read(cpu: &CpuState, cval_key: u32, cnt_key: u32) -> u64 {
    let remaining = get(cpu, cval_key).wrapping_sub(get(cpu, cnt_key));
    u64::from(remaining as u32)
}

/// TVAL write: `CVAL = count + sext32(val)`.
fn tval_write(cpu: &mut CpuState, cval_key: u32, cnt_key: u32, val: u64) {
    let delta = i64::from(val as u32 as i32) as u64; // sign-extend 32 -> 64
    let cval = get(cpu, cnt_key).wrapping_add(delta);
    cpu.sysregs.insert(cval_key, cval);
}

/// Handle an MRS of a timer register. Returns `Some(value)` if `key` is a timer
/// register needing computed behaviour; `None` falls back to the plain map read
/// (used for `CNTVCT`/`CNTPCT`/`CNTFRQ`/`CVAL`, which are stored directly).
pub(crate) fn read(cpu: &CpuState, key: u32) -> Option<u64> {
    if key == key_cntv_ctl() {
        Some(ctl_read(cpu, key_cntv_ctl(), key_cntv_cval(), key_cntvct()))
    } else if key == key_cntp_ctl() {
        Some(ctl_read(cpu, key_cntp_ctl(), key_cntp_cval(), key_cntpct()))
    } else if key == key_cntv_tval() {
        Some(tval_read(cpu, key_cntv_cval(), key_cntvct()))
    } else if key == key_cntp_tval() {
        Some(tval_read(cpu, key_cntp_cval(), key_cntpct()))
    } else {
        None
    }
}

/// Handle an MSR of a timer register. Returns `true` if it was a TVAL write
/// (converted to a CVAL store); `false` falls back to the plain map write
/// (CTL/CVAL/CNTFRQ store as-is, ISTATUS being recomputed on read).
pub(crate) fn write(cpu: &mut CpuState, key: u32, val: u64) -> bool {
    if key == key_cntv_tval() {
        tval_write(cpu, key_cntv_cval(), key_cntvct(), val);
        true
    } else if key == key_cntp_tval() {
        tval_write(cpu, key_cntp_cval(), key_cntpct(), val);
        true
    } else {
        false
    }
}

/// The earliest count value at which an *enabled* timer will next fire, if any
/// timer is enabled. The machine uses this to sleep until the next timer
/// interrupt while the guest is in WFI, instead of busy-spinning. A masked timer
/// (IMASK set) still counts toward the deadline — WFI wakes on the pending
/// interrupt regardless of the mask.
#[must_use]
pub fn next_deadline(cpu: &CpuState) -> Option<u64> {
    let mut deadline: Option<u64> = None;
    for (ctl, cval) in [
        (key_cntv_ctl(), key_cntv_cval()),
        (key_cntp_ctl(), key_cntp_cval()),
    ] {
        if get(cpu, ctl) & 1 != 0 {
            let c = get(cpu, cval);
            deadline = Some(deadline.map_or(c, |d| d.min(c)));
        }
    }
    deadline
}

/// Set the timer frequency (`CNTFRQ_EL0`, Hz). Called once at machine init.
pub fn set_frequency(cpu: &mut CpuState, hz: u64) {
    cpu.sysregs.insert(key_cntfrq(), hz);
}

/// Publish the current counter value into `CNTVCT_EL0`/`CNTPCT_EL0`. The machine
/// calls this each step from its [`crate`]-external clock; MRS reads then see a
/// live count. (Single counter; no virtual offset is modelled.)
pub fn set_count(cpu: &mut CpuState, ticks: u64) {
    cpu.sysregs.insert(key_cntvct(), ticks);
    cpu.sysregs.insert(key_cntpct(), ticks);
}

/// Whether a timer's output line is asserted: ENABLE set, IMASK clear, and
/// count past the compare value.
fn fires(cpu: &CpuState, ctl_key: u32, cval_key: u32, cnt_key: u32) -> bool {
    let ctl = get(cpu, ctl_key);
    let enabled = ctl & 1 != 0;
    let masked = ctl & 0b10 != 0;
    enabled && !masked && get(cpu, cnt_key) >= get(cpu, cval_key)
}

/// Is the virtual timer (`CNTV_*`) asserting its interrupt line?
#[must_use]
pub fn virtual_fires(cpu: &CpuState) -> bool {
    fires(cpu, key_cntv_ctl(), key_cntv_cval(), key_cntvct())
}

/// Is the physical timer (`CNTP_*`) asserting its interrupt line?
#[must_use]
pub fn physical_fires(cpu: &CpuState) -> bool {
    fires(cpu, key_cntp_ctl(), key_cntp_cval(), key_cntpct())
}
