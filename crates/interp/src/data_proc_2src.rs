//! Data processing (2 source): UDIV / SDIV / LSLV / LSRV / ASRV / RORV.

use aarch64_cpu_state::CpuState;

use crate::regs::{datasize, read, write};

pub(crate) fn exec(cpu: &mut CpuState, sf: bool, opcode: u8, rm: u8, rn: u8, rd: u8) -> Option<u64> {
    let ds = datasize(sf);
    let n = read(cpu, rn, sf, false);
    let m = read(cpu, rm, sf, false);

    // CRC32/CRC32C accumulate into a 32-bit result regardless of `sf`.
    if (0x10..=0x17).contains(&opcode) {
        let acc = (n & 0xffff_ffff) as u32;
        let size = 1u32 << (opcode & 3); // B=1, H=2, W=4, X=8 bytes
        let poly = if opcode & 4 == 0 { 0xEDB8_8320 } else { 0x82F6_3B78 }; // CRC32 / CRC32C
        let result = u64::from(crc(acc, m, size, poly));
        write(cpu, rd, false, result, false);
        return None;
    }

    let result = match opcode {
        2 => udiv(n, m, sf),               // UDIV
        3 => sdiv(n, m, sf),               // SDIV
        8 => shift_var(n, m, ds, Shift::Lsl, sf),
        9 => shift_var(n, m, ds, Shift::Lsr, sf),
        10 => shift_var(n, m, ds, Shift::Asr, sf),
        11 => shift_var(n, m, ds, Shift::Ror, sf),
        _ => unreachable!(),
    };
    write(cpu, rd, sf, result, false);
    None
}

/// Bit-reflected CRC update of `acc` over the low `size` bytes of `val`. Matches
/// QEMU's CRC32/CRC32C helpers (which reduce to a plain reflected update).
fn crc(acc: u32, val: u64, size: u32, poly: u32) -> u32 {
    let mut crc = acc;
    for i in 0..size {
        crc ^= ((val >> (i * 8)) & 0xff) as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ poly } else { crc >> 1 };
        }
    }
    crc
}

fn udiv(n: u64, m: u64, sf: bool) -> u64 {
    if sf {
        if m == 0 { 0 } else { n / m }
    } else {
        let (n, m) = (n as u32, m as u32);
        u64::from(if m == 0 { 0 } else { n / m })
    }
}

fn sdiv(n: u64, m: u64, sf: bool) -> u64 {
    if sf {
        let (n, m) = (n as i64, m as i64);
        if m == 0 {
            0
        } else if n == i64::MIN && m == -1 {
            i64::MIN as u64
        } else {
            (n / m) as u64
        }
    } else {
        let (n, m) = (n as i32, m as i32);
        let r = if m == 0 {
            0
        } else if n == i32::MIN && m == -1 {
            i32::MIN
        } else {
            n / m
        };
        r as u32 as u64
    }
}

enum Shift {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

fn shift_var(n: u64, m: u64, ds: u32, kind: Shift, sf: bool) -> u64 {
    let amount = (m % u64::from(ds)) as u32;
    if sf {
        match kind {
            Shift::Lsl => n << amount,
            Shift::Lsr => n >> amount,
            Shift::Asr => ((n as i64) >> amount) as u64,
            Shift::Ror => n.rotate_right(amount),
        }
    } else {
        let n = n as u32;
        let r = match kind {
            Shift::Lsl => n << amount,
            Shift::Lsr => n >> amount,
            Shift::Asr => ((n as i32) >> amount) as u32,
            Shift::Ror => n.rotate_right(amount),
        };
        u64::from(r)
    }
}
