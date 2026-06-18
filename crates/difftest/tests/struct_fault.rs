//! Structured SIMD load/store that FAULTS partway must be cleanly restartable:
//! it must not commit the post-index base writeback (nor keep accessing past the
//! faulting element). Otherwise, under demand paging, the kernel maps the page
//! and re-executes the instruction — but with a base register already advanced,
//! so the retry reads the wrong memory. This is what corrupts pixman's NEON
//! `ld4/st4 [x], #64` streaming loops at EL0. Interp-only (no oracle needed).

use aarch64_difftest::{mmu_test::mmu_vector_with, run_ours, DATA_BASE};

/// VA page that IS mapped; the next page (VA_A + 0x1000) is intentionally left
/// unmapped so an access straddling the boundary faults.
const VA_A: u64 = 0x10000;

fn snap_after(insn: u32, base: u64) -> aarch64_difftest::StateSnapshot {
    let mut tv = mmu_vector_with(
        &insn.to_le_bytes(),
        |pt| pt.map_page(VA_A, DATA_BASE), // only this page is mapped
        Some((0..0x2000u32).map(|i| i as u8).collect()),
    );
    tv.init_x[1] = base;
    let (snap, _stop) = run_ours(&tv);
    snap
}

/// `base` is chosen so the first part of the access lands in the mapped page and
/// the tail crosses into the unmapped next page.
fn straddling_base(access_bytes: u64) -> u64 {
    VA_A + 0x1000 - (access_bytes / 2)
}

#[test]
fn ld1_4regs_postindex_fault_no_writeback() {
    // ld1 {v0.16b-v3.16b}, [x1], #64
    let base = straddling_base(64);
    let s = snap_after(0x4cdf_2020, base);
    assert_ne!(s.x[10], 0, "expected a translation fault (DFSC in x10)");
    assert_eq!(s.x[1], base, "post-index writeback must be suppressed on fault");
}

#[test]
fn ld4_postindex_fault_no_writeback() {
    // ld4 {v0.16b-v3.16b}, [x1], #64
    let base = straddling_base(64);
    let s = snap_after(0x4cdf_0020, base);
    assert_ne!(s.x[10], 0, "expected a translation fault");
    assert_eq!(s.x[1], base, "ld4 post-index writeback must be suppressed on fault");
}

#[test]
fn st4_postindex_fault_no_writeback() {
    // st4 {v0.16b-v3.16b}, [x1], #64
    let base = straddling_base(64);
    let s = snap_after(0x4c9f_0020, base);
    assert_ne!(s.x[10], 0, "expected a translation fault");
    assert_eq!(s.x[1], base, "st4 post-index writeback must be suppressed on fault");
}

#[test]
fn ld1_single_postindex_fault_no_writeback() {
    // ld1 {v0.16b}, [x1], #16  (single register, still a structure form)
    let base = straddling_base(16);
    let s = snap_after(0x4cdf_7020, base);
    assert_ne!(s.x[10], 0, "expected a translation fault");
    assert_eq!(s.x[1], base, "post-index writeback must be suppressed on fault");
}
