//! ADD/SUB (shifted register), including ADDS/SUBS.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::ShiftType;

use crate::alu::{add_with_carry, apply_shift};
use crate::regs::{read, write};

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    sub: bool,
    set_flags: bool,
    shift: ShiftType,
    amount: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    // All operands are ZR at r31 (no SP form for the shifted variant).
    let a = read(cpu, rn, sf, false);
    let b = apply_shift(read(cpu, rm, sf, false), shift, amount, sf);
    let (result, flags) = add_with_carry(a, b, sub, sf);
    if set_flags {
        cpu.flags = flags;
    }
    // Shifted-register Rd is always ZR (no SP form).
    write(cpu, rd, sf, result, false);
    None
}
