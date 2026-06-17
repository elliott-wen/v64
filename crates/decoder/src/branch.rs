//! Router for the "Branches, exception generating and system" group.
//!
//! System and exception-generating instructions (SVC/HVC/SMC, MSR/MRS/SYS) are
//! intentionally left as `Unsupported`: Unicorn hooks and re-implements several
//! of them, so they are not faithfully comparable against it as an oracle.

use crate::bits::field;
use crate::insn::Insn;
use crate::{branch_imm, branch_reg, compare_branch, cond_branch, system, test_branch};

pub(crate) fn decode(word: u32) -> Insn {
    // Unconditional branch (immediate): bits [30:26] == 00101.
    if field(word, 26, 5) == 0b00101 {
        return branch_imm::decode(word);
    }
    // Compare and branch (immediate): bits [30:25] == 011010.
    if field(word, 25, 6) == 0b011010 {
        return compare_branch::decode(word);
    }
    // Test and branch (immediate): bits [30:25] == 011011.
    if field(word, 25, 6) == 0b011011 {
        return test_branch::decode(word);
    }
    // Conditional branch (immediate): bits [31:24] == 0101_0100.
    if field(word, 24, 8) == 0b0101_0100 {
        return cond_branch::decode(word);
    }
    // Unconditional branch (register): bits [31:25] == 1101011.
    if field(word, 25, 7) == 0b1101011 {
        return branch_reg::decode(word);
    }
    // System instructions (incl. NOP): bits [31:22] == 1101010100.
    if field(word, 22, 10) == 0b1101010100 {
        return system::decode(word);
    }
    // Exception generating: bits [31:24] == 1101_0100. opc = [23:21], LL = [1:0].
    if field(word, 24, 8) == 0b1101_0100 {
        let imm16 = field(word, 5, 16) as u16;
        let opc = field(word, 21, 3);
        let ll = field(word, 0, 2);
        if opc == 0 {
            match ll {
                0b01 => return Insn::Svc { imm16 }, // SVC -> EL1
                0b10 => return Insn::Hvc { imm16 }, // HVC -> PSCI conduit
                0b11 => return Insn::Smc { imm16 }, // SMC -> PSCI conduit
                _ => {}
            }
        }
        // BRK #imm: opc=001, LL=00 — software breakpoint (debug exception).
        if opc == 0b001 && ll == 0 {
            return Insn::Brk { imm16 };
        }
        return Insn::Unsupported { word }; // HLT/DCPS not implemented
    }
    Insn::Unsupported { word }
}
