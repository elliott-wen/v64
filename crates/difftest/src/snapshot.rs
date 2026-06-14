//! Observable architectural state, captured after a run for comparison.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSnapshot {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub nzcv: u64,
    /// Contents of the scratch DATA region after the run (empty for tests that
    /// don't touch memory).
    pub data: Vec<u8>,
    /// V0..V31 after the run (empty for tests that don't touch FP/SIMD).
    pub v: Vec<u128>,
}

impl StateSnapshot {
    /// First differing field, as a human-readable string, or `None` if equal.
    #[must_use]
    pub fn diff(&self, other: &Self) -> Option<String> {
        for i in 0..31 {
            if self.x[i] != other.x[i] {
                return Some(format!(
                    "X{i}: ours={:#018x} oracle={:#018x}",
                    self.x[i], other.x[i]
                ));
            }
        }
        if self.sp != other.sp {
            return Some(format!("SP: ours={:#018x} oracle={:#018x}", self.sp, other.sp));
        }
        if self.pc != other.pc {
            return Some(format!("PC: ours={:#018x} oracle={:#018x}", self.pc, other.pc));
        }
        if self.nzcv != other.nzcv {
            return Some(format!(
                "NZCV: ours={:#06x} oracle={:#06x}",
                self.nzcv, other.nzcv
            ));
        }
        if self.data != other.data {
            // Report the first differing byte offset for a concise message.
            let at = self
                .data
                .iter()
                .zip(&other.data)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Some(format!(
                "DATA[{at:#x}]: ours={:#04x} oracle={:#04x}",
                self.data.get(at).copied().unwrap_or(0),
                other.data.get(at).copied().unwrap_or(0)
            ));
        }
        for i in 0..self.v.len().min(other.v.len()) {
            if self.v[i] != other.v[i] {
                return Some(format!(
                    "V{i}: ours={:#034x} oracle={:#034x}",
                    self.v[i], other.v[i]
                ));
            }
        }
        None
    }
}
