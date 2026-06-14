//! ADD/SUB (extended register), including ADDS/SUBS.

use aarch64_cpu_state::CpuState;

use crate::alu::add_with_carry;
use crate::regs::{read, write};

/// ARM `ExtendReg`: extract the low byte/half/word/double of `Rm` per `option`,
/// sign- or zero-extend it, then left-shift by `shift` (0..4).
fn extend_reg(cpu: &CpuState, rm: u8, option: u8, shift: u8) -> u64 {
    let v = cpu.read_gpr(rm, false);
    let (bits, signed) = match option {
        0 => (8, false),   // UXTB
        1 => (16, false),  // UXTH
        2 => (32, false),  // UXTW
        3 => (64, false),  // UXTX
        4 => (8, true),    // SXTB
        5 => (16, true),   // SXTH
        6 => (32, true),   // SXTW
        _ => (64, true),   // SXTX
    };
    let extracted = if bits >= 64 {
        v
    } else {
        let m = (1u64 << bits) - 1;
        let x = v & m;
        if signed {
            let s = 64 - bits;
            (((x as i64) << s) >> s) as u64
        } else {
            x
        }
    };
    extracted << shift
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    sub: bool,
    set_flags: bool,
    option: u8,
    imm3: u8,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    // Rn is SP-capable; Rd is SP-capable only when flags are not set.
    let a = read(cpu, rn, sf, true);
    let b = extend_reg(cpu, rm, option, imm3);
    let (result, flags) = add_with_carry(a, b, sub, sf);
    if set_flags {
        cpu.flags = flags;
    }
    // Rd is SP-capable here, but only when flags are not being set.
    write(cpu, rd, sf, result, !set_flags);
    None
}
