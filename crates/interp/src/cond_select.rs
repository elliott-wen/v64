//! CSEL / CSINC / CSINV / CSNEG — conditional select.

use aarch64_cpu_state::CpuState;

use crate::cond::eval_cond;
use crate::regs::{read, width_mask, write};

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    op: bool,
    o2: bool,
    cond: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let result = if eval_cond(cond, cpu.flags) {
        read(cpu, rn, sf, false)
    } else {
        let m = read(cpu, rm, sf, false);
        let mask = width_mask(sf);
        match (op, o2) {
            (false, false) => m,                       // CSEL
            (false, true) => m.wrapping_add(1) & mask, // CSINC
            (true, false) => !m & mask,                // CSINV
            (true, true) => m.wrapping_neg() & mask,   // CSNEG
        }
    };
    write(cpu, rd, sf, result, false);
    None
}
