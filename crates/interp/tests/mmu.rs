//! Stage-1 translation-table walk tests.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;
use aarch64_interp::{translate, Access, GuestMem, Memory};

fn set(cpu: &mut CpuState, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, v: u64) {
    cpu.sysregs.insert(sysreg_key(op0, op1, crn, crm, op2), v);
}

/// Enable the MMU with TTBR0 at `l0_base` and a 48-bit VA (T0SZ=16).
fn enable_mmu(cpu: &mut CpuState, l0_base: u64) {
    set(cpu, 3, 0, 2, 0, 0, l0_base); // TTBR0_EL1
    set(cpu, 3, 0, 2, 0, 2, 16); // TCR_EL1: T0SZ=16
    set(cpu, 3, 0, 1, 0, 0, 1); // SCTLR_EL1.M = 1
}

#[test]
fn mmu_off_is_identity() {
    let mut mem = Memory::new(0, 0x1000);
    let cpu = CpuState::new();
    assert_eq!(translate(&cpu, &mut mem, 0xdead_beef, Access::Read, 1), Ok(0xdead_beef));
}

#[test]
fn walk_4level_4k_page() {
    let mut mem = Memory::new(0, 0x1_0000);
    // L0[0] -> L1 -> L2 -> L3 -> page @ 0x5000 (descriptor type 0b11).
    mem.write_u64(0x1000, 0x2000 | 0b11);
    mem.write_u64(0x2000, 0x3000 | 0b11);
    mem.write_u64(0x3000, 0x4000 | 0b11);
    mem.write_u64(0x4000, 0x5000 | 0b11 | (1 << 10)); // L3 page, AF set

    let mut cpu = CpuState::new();
    enable_mmu(&mut cpu, 0x1000);

    assert_eq!(translate(&cpu, &mut mem, 0x000, Access::Read, 1), Ok(0x5000));
    assert_eq!(translate(&cpu, &mut mem, 0x123, Access::Read, 1), Ok(0x5123), "page offset preserved");
}

#[test]
fn walk_2mb_block_at_l2() {
    let mut mem = Memory::new(0, 0x1_0000);
    // L0 -> L1 -> L2 block descriptor (type 0b01) mapping a 2 MiB region.
    mem.write_u64(0x1000, 0x2000 | 0b11);
    mem.write_u64(0x2000, 0x3000 | 0b11);
    mem.write_u64(0x3000, 0x4000_0000 | 0b01 | (1 << 10)); // 2 MiB block, AF set

    let mut cpu = CpuState::new();
    enable_mmu(&mut cpu, 0x1000);

    // VA offset within the 2 MiB block is carried into the PA.
    assert_eq!(translate(&cpu, &mut mem, 0x1234, Access::Read, 1), Ok(0x4000_1234));
}

#[test]
fn invalid_descriptor_faults() {
    let mut mem = Memory::new(0, 0x1_0000);
    // L0 -> L1 present, but the L1 entry is invalid (bit 0 clear).
    mem.write_u64(0x1000, 0x2000 | 0b11);
    mem.write_u64(0x2000, 0x0000); // invalid

    let mut cpu = CpuState::new();
    enable_mmu(&mut cpu, 0x1000);

    // VA 0 indexes entry 0 at every level, so the invalid L1 entry at 0x2000 is
    // hit: translation fault at level 1 -> FSC 0b0001_01 = 0x05. The walk reports
    // the fault instead of silently aliasing to identity.
    assert_eq!(translate(&cpu, &mut mem, 0x0, Access::Read, 1), Err(0x05));
}
