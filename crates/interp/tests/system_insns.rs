//! DC ZVA (block zero) and BRK (debug exception) behaviour.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;
use aarch64_interp::{run, GuestMem, Memory, StopReason};

fn dczid_el0() -> u32 {
    sysreg_key(3, 3, 0, 0, 7)
}
fn vbar_el1() -> u32 {
    sysreg_key(3, 0, 12, 0, 0)
}
fn esr_el1() -> u32 {
    sysreg_key(3, 0, 5, 2, 0)
}
fn elr_el1() -> u32 {
    sysreg_key(3, 0, 4, 0, 1)
}

#[test]
fn dc_zva_zeros_aligned_block() {
    let code = 0x1000u64;
    let mut mem = Memory::new(0, 0x10000);
    mem.write(code, &0xd50b_7420u32.to_le_bytes()); // dc zva, x0

    let mut cpu = CpuState::new();
    cpu.pc = code;
    // DCZID_EL0.BS = 2 -> block size 4 << 2 = 16 bytes.
    cpu.sysregs.insert(dczid_el0(), 2);
    // Fill 0x2000..0x2030 with 0xFF, point x0 mid-block (0x2014).
    for a in 0x2000u64..0x2030 {
        mem.write(a, &[0xff]);
    }
    cpu.x[0] = 0x2014;

    assert_eq!(run(&mut cpu, &mut mem, code + 4, 0), StopReason::UntilReached);

    let read = |m: &mut Memory, a: u64| m.read_u8(a);
    // The 16-byte block containing 0x2014 is 0x2010..0x2020: zeroed.
    for a in 0x2010u64..0x2020 {
        assert_eq!(read(&mut mem, a), 0, "byte {a:#x} should be zeroed");
    }
    // Bytes just outside the block are untouched.
    assert_eq!(read(&mut mem, 0x200f), 0xff, "below block untouched");
    assert_eq!(read(&mut mem, 0x2020), 0xff, "above block untouched");
}

#[test]
fn brk_vectors_to_el1() {
    let code = 0x1000u64;
    let vbar = 0x4000u64;
    let mut mem = Memory::new(0, 0x10000);
    mem.write(code, &0xd420_0020u32.to_le_bytes()); // brk #1

    let mut cpu = CpuState::new();
    cpu.el = 1;
    cpu.spsel = true; // EL1h -> same-EL SP_ELx group (0x200)
    cpu.pc = code;
    cpu.sysregs.insert(vbar_el1(), vbar);

    // BRK is a synchronous exception: vector to VBAR + 0x200 (same EL, SP_ELx).
    assert_eq!(run(&mut cpu, &mut mem, vbar + 0x200, 0), StopReason::UntilReached);
    assert_eq!(cpu.pc, vbar + 0x200);
    // ESR: EC=0x3C (BRK), IL=1, ISS = comment (#1).
    assert_eq!(cpu.sysregs.get(&esr_el1()), Some(&0xf200_0001));
    // ELR points at the BRK itself (the handler decides whether to skip it).
    assert_eq!(cpu.sysregs.get(&elr_el1()), Some(&code));
}
