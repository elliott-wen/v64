//! ADC/SBC/ADCS/SBCS — add/sub with carry. The carry-in is the PSTATE C flag.

use aarch64_cpu_state::CpuState;

use crate::alu::add_with_carry_in;
use crate::regs::{read, write};

pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    sub: bool,
    set_flags: bool,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let a = read(cpu, rn, sf, false);
    let b = read(cpu, rm, sf, false);
    let carry_in = u64::from(cpu.flags.c);
    // SBC computes Rn + NOT(Rm) + C; ADC computes Rn + Rm + C.
    let b_op = if sub { !b } else { b };
    let (result, flags) = add_with_carry_in(a, b_op, carry_in, sf);
    if set_flags {
        cpu.flags = flags;
    }
    // Add/sub-with-carry Rd is always ZR (no SP form).
    write(cpu, rd, sf, result, false);
    None
}
