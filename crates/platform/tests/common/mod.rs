//! Shared test scaffolding: a tiny AArch64 assembler and a boot+run helper, so
//! each integration test can express a focused "mini-kernel" as readable code
//! rather than magic instruction words.
#![allow(dead_code)]

use aarch64_interp::StopReason;
use aarch64_platform::{Board, KERNEL_LOAD};

pub const RAM_SIZE: usize = 0x200_0000; // 32 MiB
/// Generous instruction cap so a runaway mini-kernel fails fast instead of hanging.
pub const RUN_LIMIT: usize = 100_000;

// ---- Instruction encoders (just the subset the mini-kernels need). ----

/// MOVZ Xd, #imm16, LSL #(16*hw).
pub fn movz64(rd: u32, imm16: u32, hw: u32) -> u32 {
    0xD280_0000 | (hw << 21) | (imm16 << 5) | rd
}
/// MOVK Xd, #imm16, LSL #(16*hw).
pub fn movk64(rd: u32, imm16: u32, hw: u32) -> u32 {
    0xF280_0000 | (hw << 21) | (imm16 << 5) | rd
}
/// MOVZ Wd, #imm16.
pub fn movz32(rd: u32, imm16: u32) -> u32 {
    0x5280_0000 | (imm16 << 5) | rd
}
/// ADD Xd, Xn, #imm12.
pub fn add_imm64(rd: u32, rn: u32, imm12: u32) -> u32 {
    0x9100_0000 | (imm12 << 10) | (rn << 5) | rd
}
/// STR Xt, [Xn, #imm] (imm is a byte offset, must be 8-aligned).
pub fn str64(rt: u32, rn: u32, imm_bytes: u32) -> u32 {
    0xF900_0000 | ((imm_bytes / 8) << 10) | (rn << 5) | rt
}
/// STR Wt, [Xn, #imm] (imm byte offset, 4-aligned).
pub fn str32(rt: u32, rn: u32, imm_bytes: u32) -> u32 {
    0xB900_0000 | ((imm_bytes / 4) << 10) | (rn << 5) | rt
}
/// LDR Wt, [Xn, #imm] (imm byte offset, 4-aligned).
pub fn ldr32(rt: u32, rn: u32, imm_bytes: u32) -> u32 {
    0xB940_0000 | ((imm_bytes / 4) << 10) | (rn << 5) | rt
}
/// STRB Wt, [Xn] (offset 0).
pub fn strb(rt: u32, rn: u32) -> u32 {
    0x3900_0000 | (rn << 5) | rt
}
/// MSR (Sx_x_Cx_Cx_x), Xt — write a system register.
pub fn msr(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) -> u32 {
    sysreg_move(false, op0, op1, crn, crm, op2, rt)
}
/// MRS Xt, (Sx_x_Cx_Cx_x) — read a system register.
pub fn mrs(rt: u32, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    sysreg_move(true, op0, op1, crn, crm, op2, rt)
}
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
/// MSR DAIFClr, #imm — unmask PSTATE.{D,A,I,F} bits.
pub fn msr_daifclr(imm: u32) -> u32 {
    0xD500_0000 | (3 << 16) | (4 << 12) | (imm << 8) | (7 << 5) | 31
}

pub const NOP: u32 = 0xD503_201F;
pub const B_SELF: u32 = 0x1400_0000; // B .
pub const ERET: u32 = 0xD69F_03E0;
pub const HVC0: u32 = 0xD400_0002;

// Common system-register coordinates.
pub const SCTLR_EL1: (u32, u32, u32, u32, u32) = (3, 0, 1, 0, 0);
pub const TTBR0_EL1: (u32, u32, u32, u32, u32) = (3, 0, 2, 0, 0);
pub const TCR_EL1: (u32, u32, u32, u32, u32) = (3, 0, 2, 0, 2);
pub const VBAR_EL1: (u32, u32, u32, u32, u32) = (3, 0, 12, 0, 0);
pub const CNTV_CTL_EL0: (u32, u32, u32, u32, u32) = (3, 3, 14, 3, 1);
pub const CNTV_CVAL_EL0: (u32, u32, u32, u32, u32) = (3, 3, 14, 3, 2);

/// PSCI SYSTEM_OFF function ID.
pub const PSCI_SYSTEM_OFF: u64 = 0x8400_0008;

// ---- A small program builder that tracks byte offsets. ----

#[derive(Default)]
pub struct Asm {
    words: Vec<u32>,
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one instruction.
    pub fn ins(&mut self, w: u32) -> &mut Self {
        self.words.push(w);
        self
    }

    /// Current byte offset from the program start.
    pub fn offset(&self) -> u64 {
        (self.words.len() * 4) as u64
    }

    /// Pad with NOPs up to `byte_off` (e.g. to place an exception vector).
    pub fn pad_to(&mut self, byte_off: u64) -> &mut Self {
        while self.offset() < byte_off {
            self.words.push(NOP);
        }
        self
    }

    /// MSR helper taking a register-coordinate tuple.
    pub fn msr(&mut self, reg: (u32, u32, u32, u32, u32), rt: u32) -> &mut Self {
        self.ins(msr(reg.0, reg.1, reg.2, reg.3, reg.4, rt))
    }
    /// MRS helper taking a register-coordinate tuple.
    pub fn mrs(&mut self, rt: u32, reg: (u32, u32, u32, u32, u32)) -> &mut Self {
        self.ins(mrs(rt, reg.0, reg.1, reg.2, reg.3, reg.4))
    }

    /// Materialize a 64-bit immediate into `rd` (1 MOVZ + up to 3 MOVK).
    pub fn load_imm64(&mut self, rd: u32, value: u64) -> &mut Self {
        let chunks = [
            (value & 0xffff) as u32,
            ((value >> 16) & 0xffff) as u32,
            ((value >> 32) & 0xffff) as u32,
            ((value >> 48) & 0xffff) as u32,
        ];
        self.ins(movz64(rd, chunks[0], 0));
        for (hw, &c) in chunks.iter().enumerate().skip(1) {
            if c != 0 {
                self.ins(movk64(rd, c, hw as u32));
            }
        }
        self
    }

    /// Store the 32-bit `value` to absolute physical `addr`, clobbering
    /// `addr_reg` and `val_reg`.
    pub fn store_u32(&mut self, addr: u64, value: u32, addr_reg: u32, val_reg: u32) -> &mut Self {
        self.load_imm64(addr_reg, addr);
        self.load_imm64(val_reg, u64::from(value));
        self.ins(str32(val_reg, addr_reg, 0))
    }

    /// Emit a PSCI SYSTEM_OFF call (clobbers x0).
    pub fn power_off(&mut self) -> &mut Self {
        self.load_imm64(0, PSCI_SYSTEM_OFF);
        self.ins(HVC0)
    }

    /// Flatten to little-endian bytes (a loadable "kernel image").
    pub fn image(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

/// Boot `image` on a fresh board, apply `prepare` (e.g. set the timer interval),
/// run to power-off/limit, and return the stop reason plus the board (so tests
/// can inspect UART output, memory, or registers afterward).
pub fn boot_and_run(image: &[u8], prepare: impl FnOnce(&mut Board)) -> (StopReason, Board) {
    let mut board = Board::new(RAM_SIZE);
    let dtb = board.dtb(RAM_SIZE as u64, "console=ttyAMA0", None);
    board.boot(image, &dtb);
    prepare(&mut board);
    let stop = board.machine.run(0, RUN_LIMIT);
    (stop, board)
}

/// Absolute address of byte `off` within the loaded kernel image.
pub fn kaddr(off: u64) -> u64 {
    KERNEL_LOAD + off
}
