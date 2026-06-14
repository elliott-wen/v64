//! Advanced SIMD copy: DUP (element), INS (general/element), SMOV/UMOV.

use aarch64_cpu_state::CpuState;

fn elem_mask(ebits: u32) -> u128 {
    if ebits >= 128 {
        u128::MAX
    } else {
        (1u128 << ebits) - 1
    }
}

fn get_lane(v: u128, index: u8, ebits: u32) -> u128 {
    (v >> (u32::from(index) * ebits)) & elem_mask(ebits)
}

fn set_lane(v: u128, index: u8, ebits: u32, elem: u128) -> u128 {
    let shift = u32::from(index) * ebits;
    (v & !(elem_mask(ebits) << shift)) | ((elem & elem_mask(ebits)) << shift)
}

/// DUP (element): replicate Vn[index] across all lanes of Vd.
pub(crate) fn dup_element(cpu: &mut CpuState, q: bool, size: u8, index: u8, rn: u8, rd: u8) -> Option<u64> {
    let ebits = 8u32 << size;
    let elem = get_lane(cpu.v[rn as usize], index, ebits);
    let n = (if q { 128 } else { 64 }) / ebits;
    let mut v = 0u128;
    for i in 0..n {
        v |= elem << (i * ebits);
    }
    cpu.v[rd as usize] = v;
    None
}

/// INS (general): Vd[index] = Rn; other lanes preserved (no upper-half zeroing).
pub(crate) fn ins_general(cpu: &mut CpuState, size: u8, index: u8, rn: u8, rd: u8) -> Option<u64> {
    let ebits = 8u32 << size;
    let elem = u128::from(cpu.read_gpr(rn, false));
    cpu.v[rd as usize] = set_lane(cpu.v[rd as usize], index, ebits, elem);
    None
}

/// INS (element): Vd[dst] = Vn[src].
pub(crate) fn ins_element(cpu: &mut CpuState, size: u8, dst: u8, src: u8, rn: u8, rd: u8) -> Option<u64> {
    let ebits = 8u32 << size;
    let elem = get_lane(cpu.v[rn as usize], src, ebits);
    cpu.v[rd as usize] = set_lane(cpu.v[rd as usize], dst, ebits, elem);
    None
}

/// SMOV/UMOV: move Vn[index] to a GPR, sign- or zero-extended.
pub(crate) fn mov_to_gpr(
    cpu: &mut CpuState,
    signed: bool,
    dst64: bool,
    size: u8,
    index: u8,
    vn: u8,
    rd: u8,
) -> Option<u64> {
    let ebits = 8u32 << size;
    let elem = get_lane(cpu.v[vn as usize], index, ebits) as u64;
    let val = if signed {
        let s = 64 - ebits;
        ((elem << s) as i64 >> s) as u64
    } else {
        elem
    };
    if dst64 {
        cpu.write_gpr(rd, false, val);
    } else {
        cpu.write_gpr_w(rd, false, val);
    }
    None
}
