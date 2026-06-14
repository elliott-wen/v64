//! SBFM / BFM / UBFM — bitfield move.
//!
//! Implements the ARM ARM pseudocode:
//!   bot = ROR(src, R) AND wmask
//!   top = SBFM: Replicate(src<imms>) | UBFM: 0 | BFM: Rd
//!   Rd  = (top AND NOT tmask) OR (bot AND tmask)

use aarch64_cpu_state::CpuState;

use crate::regs::{datasize, read, write};

/// Rotate the low `ds` bits of `val` right by `r`.
fn ror(val: u64, r: u32, ds: u32) -> u64 {
    if ds == 64 {
        val.rotate_right(r)
    } else {
        let v = (val as u32).rotate_right(r);
        u64::from(v)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec(
    cpu: &mut CpuState,
    sf: bool,
    opc: u8,
    wmask: u64,
    tmask: u64,
    immr: u8,
    imms: u8,
    rn: u8,
    rd: u8,
) -> Option<u64> {
    let ds = datasize(sf);
    let dmask = if ds == 64 { u64::MAX } else { 0xffff_ffff };
    let wmask = wmask & dmask;
    let tmask = tmask & dmask;

    let src = read(cpu, rn, sf, false);
    let rotated = ror(src, u32::from(immr), ds) & wmask;
    let dst = read(cpu, rd, sf, false);

    // BFM merges the existing destination bits where wmask is 0; SBFM/UBFM
    // start the low half from zero (their `top` provides the rest).
    let bot = if opc == 1 {
        (dst & !wmask) | rotated
    } else {
        rotated
    };

    let top = match opc {
        0 => {
            // SBFM: replicate the source bit at position `imms`.
            if (src >> imms) & 1 == 1 {
                dmask
            } else {
                0
            }
        }
        1 => dst, // BFM: keep existing destination bits
        _ => 0,   // UBFM
    };

    let result = (top & !tmask) | (bot & tmask);
    write(cpu, rd, sf, result, false);
    None
}
