//! SIMD permutes as constant byte shuffles: ZIP1/2, UZP1/2, TRN1/2, EXT, and
//! single-table TBL (→ `i8x16.swizzle`).

use wasm_encoder::{Function, Instruction as I};

use super::{finish_v, push_v, shuffle2};

/// ZIP1/2, UZP1/2, TRN1/2 — interleave/deinterleave/transpose. Always handled.
pub(crate) fn simd_zip_trn(f: &mut Function, q: bool, size: u8, opcode: u8, rm: u8, rn: u8, rd: u8) {
    let esize = 1usize << size;
    let n = (if q { 16 } else { 8 }) / esize; // element count
    let half = n / 2;
    let mut lanes = [0u8; 16];
    for oi in 0..n {
        // Output element `oi` is sourced from element `si` of a (Vn) or b (Vm).
        let (from_b, si) = match opcode {
            0b011 => (oi % 2 == 1, oi / 2),             // ZIP1 (low halves)
            0b111 => (oi % 2 == 1, half + oi / 2),      // ZIP2 (high halves)
            0b001 => (oi >= half, 2 * (oi % half)),     // UZP1 (evens)
            0b101 => (oi >= half, 2 * (oi % half) + 1), // UZP2 (odds)
            0b010 => (oi % 2 == 1, oi & !1),            // TRN1: a[2k], b[2k]
            _ => (oi % 2 == 1, (oi & !1) | 1),          // TRN2: a[2k+1], b[2k+1]
        };
        let src = (if from_b { 16 } else { 0 }) + si * esize;
        for b in 0..esize {
            lanes[oi * esize + b] = (src + b) as u8;
        }
    }
    shuffle2(f, q, rn, rm, rd, lanes);
}

/// EXT: `Vd = bytes[imm4 .. imm4+nbytes)` of the concatenation {Vn:Vm} (Vn low).
pub(crate) fn simd_ext(f: &mut Function, q: bool, imm4: u8, rm: u8, rn: u8, rd: u8) {
    let nbytes = if q { 16 } else { 8 };
    let mut lanes = [0u8; 16];
    for (j, lane) in lanes.iter_mut().take(nbytes).enumerate() {
        let k = imm4 as usize + j; // index into the concatenation
        *lane = if q || k < 8 {
            k as u8 // a = Vn at 0..16; for q, b = Vm at 16..32 lines up directly
        } else {
            (16 + (k - 8)) as u8 // !q: bytes 8..16 of the concat come from Vm's low 8
        };
    }
    shuffle2(f, q, rn, rm, rd, lanes);
}

/// TBL with a single table register maps to `i8x16.swizzle` (index >= 16 -> 0,
/// matching AArch64). Multi-register tables and TBX (keep-on-miss) fall back.
pub(crate) fn simd_tbl(f: &mut Function, q: bool, is_tbx: bool, len: u8, rm: u8, rn: u8, rd: u8) -> bool {
    if is_tbx || len != 0 {
        return false;
    }
    emit!(f, I::LocalGet(0)); // base for the store
    push_v(f, rn); // table
    push_v(f, rm); // byte indices
    emit!(f, I::I8x16Swizzle);
    finish_v(f, q, rd);
    true
}
