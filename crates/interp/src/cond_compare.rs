//! CCMP / CCMN — conditional compare.
//!
//! If the condition holds, set the flags from `Rn - Y` (CCMP) or `Rn + Y`
//! (CCMN); otherwise force NZCV to the 4-bit immediate.

use aarch64_cpu_state::{CpuState, Flags};

use crate::alu::add_with_carry;
use crate::cond::eval_cond;
use crate::regs::read;

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    sub: bool,
    is_imm: bool,
    imm_y: u8,
    rm: u8,
    cond: u8,
    nzcv: u8,
    rn: u8,
) -> Option<u64> {
    if eval_cond(cond, cpu.flags) {
        let y = if is_imm {
            u64::from(imm_y)
        } else {
            read(cpu, rm, sf, false)
        };
        let a = read(cpu, rn, sf, false);
        let (_, flags) = add_with_carry(a, y, sub, sf);
        cpu.flags = flags;
    } else {
        cpu.flags = Flags {
            n: nzcv & 0b1000 != 0,
            z: nzcv & 0b0100 != 0,
            c: nzcv & 0b0010 != 0,
            v: nzcv & 0b0001 != 0,
        };
    }
    None
}
