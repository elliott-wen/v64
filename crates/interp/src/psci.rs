//! PSCI (Power State Coordination Interface) — the firmware ABI Linux invokes
//! via `HVC`/`SMC` for power management.
//!
//! Conceptually this is "firmware," not CPU semantics, but it's small and
//! stateless apart from a halt flag, so it lives here next to the conduit
//! instruction. A single-core subset sufficient for `virt`:
//!
//! * `PSCI_VERSION` — report v1.0 so Linux's PSCI driver binds.
//! * `SYSTEM_OFF` / `SYSTEM_RESET` — set [`CpuState::powered_off`]; the machine
//!   loop stops on it.
//! * `CPU_AFFINITY_INFO` for CPU 0 — report ON.
//! * everything else (notably `CPU_ON`, never called by a single-CPU DTB) —
//!   `NOT_SUPPORTED`.
//!
//! The function ID is in `x0`; the 32-bit signed result is returned in `x0`.

use aarch64_cpu_state::CpuState;

// Function IDs (the SMC64 variants set bit 30; we match on the low bits so both
// the 32- and 64-bit calling conventions hit the same arm).
const PSCI_VERSION: u32 = 0x8400_0000;
const CPU_AFFINITY_INFO: u32 = 0x8400_0004;
const MIGRATE_INFO_TYPE: u32 = 0x8400_0006;
const SYSTEM_OFF: u32 = 0x8400_0008;
const SYSTEM_RESET: u32 = 0x8400_0009;

// Return codes (32-bit signed).
const SUCCESS: i64 = 0;
const NOT_SUPPORTED: i64 = -1;

/// PSCI version 1.0, packed as (major << 16) | minor.
const VERSION_1_0: i64 = 1 << 16;
/// MIGRATE_INFO_TYPE: "Trusted OS not present / migration not required."
const MIGRATE_NOT_REQUIRED: i64 = 2;

/// Handle a PSCI call (the conduit instruction's effect). Reads the function ID
/// from `x0` and writes the result back to `x0`. `cpu.pc` advances normally.
pub(crate) fn call(cpu: &mut CpuState) {
    // Match on the SMC32 form (clear bit 30) so SMC64 calls share the arm.
    let fid = (cpu.x[0] as u32) & !(1 << 30);
    let ret = match fid {
        PSCI_VERSION => VERSION_1_0,
        SYSTEM_OFF | SYSTEM_RESET => {
            cpu.powered_off = true;
            SUCCESS
        }
        // CPU 0 (the only core) is always on; affinity arg in x1 is ignored.
        CPU_AFFINITY_INFO => SUCCESS, // 0 == AFFINITY_INFO_ON
        MIGRATE_INFO_TYPE => MIGRATE_NOT_REQUIRED,
        _ => NOT_SUPPORTED,
    };
    cpu.x[0] = ret as u64;
}
