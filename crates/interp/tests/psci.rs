//! PSCI calls via the HVC conduit: version query, power-off, and the
//! not-supported default.

use aarch64_cpu_state::CpuState;
use aarch64_interp::{step, Memory};

const HVC0: u32 = 0xD400_0002; // HVC #0

fn run_hvc(x0: u64) -> CpuState {
    let mut mem = Memory::new(0, 0x100);
    mem.write(0, &HVC0.to_le_bytes());
    let mut cpu = CpuState::new();
    cpu.x[0] = x0;
    step(&mut cpu, &mut mem);
    cpu
}

#[test]
fn version_returns_1_0() {
    let cpu = run_hvc(0x8400_0000); // PSCI_VERSION
    assert_eq!(cpu.x[0], 1 << 16, "v1.0");
    assert_eq!(cpu.pc, 4, "HVC returns to the next instruction");
}

#[test]
fn system_off_powers_down() {
    let cpu = run_hvc(0x8400_0008); // SYSTEM_OFF
    assert!(cpu.powered_off);
}

#[test]
fn system_reset_powers_down() {
    let cpu = run_hvc(0x8400_0009); // SYSTEM_RESET
    assert!(cpu.powered_off);
}

#[test]
fn affinity_info_cpu0_is_on() {
    let cpu = run_hvc(0x8400_0004); // CPU_AFFINITY_INFO
    assert_eq!(cpu.x[0], 0, "0 == ON");
}

#[test]
fn unknown_call_is_not_supported() {
    let cpu = run_hvc(0xC400_0003); // CPU_ON (SMC64) — unsupported single-core
    assert_eq!(cpu.x[0], (-1i64) as u64, "PSCI_RET_NOT_SUPPORTED");
    assert!(!cpu.powered_off);
}
