//! EXTR — `Rd = (Rn:Rm) >> lsb`, low `datasize` bits.

use aarch64_cpu_state::CpuState;

use crate::regs::{read, write};

pub(crate) fn exec(cpu: &mut CpuState, sf: bool, rm: u8, rn: u8, lsb: u8, rd: u8) -> Option<u64> {
    let m = read(cpu, rm, sf, false);
    let n = read(cpu, rn, sf, false);
    let lsb = u32::from(lsb);

    let result = if sf {
        if lsb == 0 {
            m
        } else {
            (m >> lsb) | (n << (64 - lsb))
        }
    } else {
        let m = m as u32;
        let n = n as u32;
        let r = if lsb == 0 {
            m
        } else {
            (m >> lsb) | (n << (32 - lsb))
        };
        u64::from(r)
    };
    write(cpu, rd, sf, result, false);
    None
}
