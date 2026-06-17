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
            // Hint space (CRn=2): WFI/WFE are wait-for-interrupt/event — the
            // machine fast-forwards over guest idle instead of busy-spinning, so
            // they decode distinctly. The rest (NOP/YIELD/SEV/...) are no-ops.
            2 => match (field(word, 8, 4), field(word, 5, 3)) {
                (0, 2) => Insn::Wfe,
                (0, 3) => Insn::Wfi,
                _ => Insn::Nop,
            },
            // Barriers (DSB/DMB/ISB/CLREX): no-ops in a sequential interpreter.
            3 => Insn::Nop,
            // MSR (immediate): SPSel / DAIFSet / DAIFClr.
            4 => Insn::MsrImm {
                op1: field(word, 16, 3) as u8,
                op2: field(word, 5, 3) as u8,
                crm: field(word, 8, 4) as u8,
            },
            _ => Insn::Unsupported { word },
        };
    }

    // SYS/SYSL (op0=1): cache/TLB/address-translate maintenance. With no TLB or
    // cache model these are no-ops — *except* DC ZVA, which architecturally zeros
    // a block of memory and so must be modelled to keep guest memory correct.
    if op0 == 1 {
        let op1 = field(word, 16, 3);
        let crm = field(word, 8, 4);
        let op2 = field(word, 5, 3);
        // DC ZVA: op1=011, CRn=0111, CRm=0100, op2=001 (L=0).
        if l == 0 && op1 == 0b011 && crn == 0b0111 && crm == 0b0100 && op2 == 0b001 {
            return Insn::DcZva { rt };
        }
        // TLBI (CRn=1000): TLB maintenance. We model a single unified TLB and
        // invalidate it wholesale, so every TLBI variant maps to the same flush.
        if l == 0 && crn == 0b1000 {
            return Insn::Tlbi;
        }
        // IC (CRn=0111, CRm=0001 IALLUIS / CRm=0101 IALLU/IVAU): instruction-cache
        // maintenance — the architecture's "code changed" signal, used by the JIT
        // to drop stale compiled blocks.
        if l == 0 && crn == 0b0111 && (crm == 0b0001 || crm == 0b0101) {
            return Insn::Ic;
        }
        // Other DC / AT: no cache/translation state we model. A SYSL read
        // (L=1) is rarer still; treat as a no-op (its Rt is left unchanged).
        return Insn::Nop;
    }

    // MRS (L=1) / MSR (L=0) register move (op0 = 2 or 3).
    let key = sysreg_key(op0, field(word, 16, 3), crn, field(word, 8, 4), field(word, 5, 3));
    Insn::SysRegMove { read: l == 1, key, rt }
}
