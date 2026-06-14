//! ADD/SUB (immediate), including ADDS/SUBS.

use aarch64_cpu_state::CpuState;

use crate::alu::add_with_carry;
use crate::regs::{read, write};

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    sub: bool,
    set_flags: bool,
    shift12: bool,
    imm12: u16,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let imm = u64::from(imm12) << if shift12 { 12 } else { 0 };
    // Rn is SP-capable; Rd is SP-capable only when flags are not set.
    let a = read(cpu, rn, sf, true);
    let (result, flags) = add_with_carry(a, imm, sub, sf);
    if set_flags {
        cpu.flags = flags;
    }
    // Rd is SP-capable here, but only when flags are not being set.
    write(cpu, rd, sf, result, !set_flags);
    None
}
