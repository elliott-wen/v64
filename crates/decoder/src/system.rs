//! System instructions: MRS/MSR (register), hints, barriers, MSR-immediate.
//!
//! Identified by bits[31:22] = 1101010100. `op0` (bits[20:19]) == 0 selects the
//! hint/barrier/MSR-immediate space (treated as NOPs for now); otherwise it's a
//! system-register move.

use crate::bits::field;
use crate::insn::Insn;

/// Pack (op0,op1,CRn,CRm,op2) into a single key for the system register file.
#[must_use]
pub fn sysreg_key(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    (op0 << 16) | (op1 << 12) | (crn << 8) | (crm << 4) | op2
}

pub(crate) fn decode(word: u32) -> Insn {
    let l = field(word, 21, 1);
    let op0 = field(word, 19, 2);
    let crn = field(word, 12, 4);
    let rt = field(word, 0, 5) as u8;

    if op0 == 0 {
        // Hints (CRn=2), barriers (CRn=3), MSR-immediate (CRn=4): all require
        // L=0 and Rt=31. We treat them as no-ops in a sequential interpreter.
        if l == 1 || rt != 31 {
            return Insn::Unsupported { word };
        }
        return match crn {
            // Hints (NOP/YIELD/...) and barriers (DSB/DMB/ISB/CLREX): no-ops.
            2 | 3 => Insn::Nop,
            // MSR (immediate): SPSel / DAIFSet / DAIFClr.
            4 => Insn::MsrImm {
                op1: field(word, 16, 3) as u8,
                op2: field(word, 5, 3) as u8,
                crm: field(word, 8, 4) as u8,
            },
            _ => Insn::Unsupported { word },
        };
    }

    // MRS (L=1) / MSR (L=0) register move.
    let key = sysreg_key(op0, field(word, 16, 3), crn, field(word, 8, 4), field(word, 5, 3));
    Insn::SysRegMove { read: l == 1, key, rt }
}
