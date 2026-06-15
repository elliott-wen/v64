//! Generic-timer register semantics through the real MRS/MSR path: TVAL<->CVAL
//! conversion and the computed ISTATUS bit.

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;
use aarch64_interp::{set_count, step, virtual_fires, Memory};

/// Encode an MRS/MSR (register) system move. `read` true = MRS, false = MSR.
fn sysreg_move(read: bool, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) -> u32 {
    (0b1101010100 << 22)
        | (u32::from(read) << 21)
        | (op0 << 19)
        | (op1 << 16)
        | (crn << 12)
        | (crm << 8)
        | (op2 << 5)
        | rt
}

// CNTV_* virtual-timer register coordinates.
const CTL: (u32, u32, u32, u32, u32) = (3, 3, 14, 3, 1);
const CVAL: (u32, u32, u32, u32, u32) = (3, 3, 14, 3, 2);
const TVAL: (u32, u32, u32, u32, u32) = (3, 3, 14, 3, 0);

fn key(c: (u32, u32, u32, u32, u32)) -> u32 {
    sysreg_key(c.0, c.1, c.2, c.3, c.4)
}

#[test]
fn tval_cval_roundtrip_and_istatus() {
    // Program:
    //  0: MSR CNTV_TVAL_EL0, X0   ; CVAL = count + X0
    //  4: MRS X1, CNTV_TVAL_EL0   ; X1 = ticks remaining
    //  8: MSR CNTV_CTL_EL0,  X3   ; enable
    // 12: MRS X2, CNTV_CTL_EL0    ; X2 = CTL (no ISTATUS yet)
    // 16: MRS X4, CNTV_CTL_EL0    ; read again after the count passes CVAL
    let prog = [
        sysreg_move(false, TVAL.0, TVAL.1, TVAL.2, TVAL.3, TVAL.4, 0),
        sysreg_move(true, TVAL.0, TVAL.1, TVAL.2, TVAL.3, TVAL.4, 1),
        sysreg_move(false, CTL.0, CTL.1, CTL.2, CTL.3, CTL.4, 3),
        sysreg_move(true, CTL.0, CTL.1, CTL.2, CTL.3, CTL.4, 2),
        sysreg_move(true, CTL.0, CTL.1, CTL.2, CTL.3, CTL.4, 4),
    ];
    let mut mem = Memory::new(0, 0x100);
    for (i, w) in prog.iter().enumerate() {
        mem.write(i as u64 * 4, &w.to_le_bytes());
    }

    let mut cpu = CpuState::new();
    cpu.x[0] = 500; // fire 500 ticks from now
    cpu.x[3] = 1; // CTL enable
    set_count(&mut cpu, 1000);

    // Run the first four instructions.
    for _ in 0..4 {
        step(&mut cpu, &mut mem);
    }

    assert_eq!(cpu.sysregs[&key(CVAL)], 1500, "TVAL write set CVAL = count + 500");
    assert_eq!(cpu.x[1], 500, "TVAL read = CVAL - count");
    assert_eq!(cpu.x[2], 0b001, "enabled, count < CVAL: ISTATUS clear");
    assert!(!virtual_fires(&cpu));

    // Advance the count past CVAL; the comparator condition is now met.
    set_count(&mut cpu, 2000);
    step(&mut cpu, &mut mem); // the MRS at offset 16

    assert_eq!(cpu.x[4], 0b101, "enabled + count >= CVAL: ISTATUS set");
    assert!(virtual_fires(&cpu));
}

#[test]
fn tval_remaining_is_signed_when_overdue() {
    // CVAL behind the count -> remaining is negative, truncated to 32 bits.
    let prog = [sysreg_move(true, TVAL.0, TVAL.1, TVAL.2, TVAL.3, TVAL.4, 1)];
    let mut mem = Memory::new(0, 0x100);
    mem.write(0, &prog[0].to_le_bytes());

    let mut cpu = CpuState::new();
    set_count(&mut cpu, 1000);
    cpu.sysregs.insert(key(CVAL), 900); // 100 ticks overdue

    step(&mut cpu, &mut mem);
    assert_eq!(cpu.x[1], (-100i32) as u32 as u64, "remaining = -100, 32-bit");
}
