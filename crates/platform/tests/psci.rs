//! PSCI SYSTEM_OFF stops the machine loop with `StopReason::PoweredOff`.

use aarch64_cpu_state::CpuState;
use aarch64_interp::{Memory, StopReason};
use aarch64_platform::{Bus, Gic, Machine};

const RAM_BASE: u64 = 0x4000_0000;
const HVC0: u32 = 0xD400_0002;
const B_SELF: u32 = 0x1400_0000;

#[test]
fn system_off_halts_the_machine() {
    let mut mem = Memory::new(RAM_BASE, 0x1000);
    mem.write(RAM_BASE, &HVC0.to_le_bytes()); // 0x..00: HVC #0
    mem.write(RAM_BASE + 4, &B_SELF.to_le_bytes()); // 0x..04: spin (should never run)

    let mut cpu = CpuState::new();
    cpu.pc = RAM_BASE;
    cpu.x[0] = 0x8400_0008; // SYSTEM_OFF

    let mut m = Machine::new(cpu, Bus::new(mem), Gic::new());

    // Unbounded run within a generous instruction cap; should stop on power-off.
    assert_eq!(m.run(0xdead_beef, 100), StopReason::PoweredOff);
    assert!(m.cpu.powered_off);
}
