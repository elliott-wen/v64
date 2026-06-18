//! Cross-page SIMD load/store vs Unicorn, with the MMU on and consecutive
//! virtual pages mapped to NON-adjacent physical pages.
//!
//! This is the gap the MMU-off ISA sweep and the single-page word-diff can't
//! reach: a 16/32/64-byte NEON access that straddles a page boundary must
//! translate EACH page separately (the second VA page can map anywhere). If the
//! interpreter translates once and reads/writes contiguous physical bytes, the
//! tail lands in the wrong place — exactly the kind of memory corruption that
//! crashes pixman's NEON fast paths while every instruction is individually
//! correct. Built only with `--features unicorn`.

#![cfg(feature = "unicorn")]

use aarch64_difftest::{
    mmu_test::mmu_vector_with, run_ours, run_unicorn_mmu, MmuOutcome, StopReason, TestVector,
    DATA_BASE,
};

// Two consecutive virtual pages, deliberately mapped to reversed (non-adjacent)
// physical pages inside the compared DATA window.
const VA_A: u64 = 0x10000; // -> PA DATA_BASE + 0x1000
const VA_B: u64 = 0x11000; // -> PA DATA_BASE + 0x0000

fn xpage_vector(insn: u32, va: u64) -> TestVector {
    let mut tv = mmu_vector_with(
        &insn.to_le_bytes(),
        |pt| {
            pt.map_page(VA_A, DATA_BASE + 0x1000);
            pt.map_page(VA_B, DATA_BASE);
        },
        Some((0..0x2000).map(|i| (i * 7 + 1) as u8).collect()),
    );
    tv.init_x[1] = va; // base register x1 = straddling VA
    let mut v = [0u128; 32];
    for (i, e) in v.iter_mut().enumerate() {
        *e = (0x0102_0304_0506_0708u128 << 64) | (0xa0b0_c0d0_e0f0_0011u128 + i as u128);
    }
    tv.init_v = Some(v);
    tv
}

#[track_caller]
fn assert_xpage(name: &str, insn: u32, va: u64) {
    let tv = xpage_vector(insn, va);
    let (ours, stop) = run_ours(&tv);
    assert!(!matches!(stop, StopReason::Unsupported { .. }), "{name}: interp can't run {insn:08x}");
    match run_unicorn_mmu(&tv).expect("unicorn run failed") {
        MmuOutcome::Ran(oracle) => {
            if let Some(diff) = ours.diff(&oracle) {
                panic!("{name} ({insn:08x} @ {va:#x}) DIVERGES: {diff}");
            }
        }
        MmuOutcome::Faulted { .. } => panic!("{name}: unexpected fault"),
    }
}

#[test]
fn ldr_q_straddle() {
    // ldr q0, [x1]  (0x3dc00020) — 16 bytes, 8 in each page.
    assert_xpage("ldr q0", 0x3dc0_0020, VA_B - 8);
}

#[test]
fn str_q_straddle() {
    // str q0, [x1]  (0x3d800020) — 16 bytes across the boundary.
    assert_xpage("str q0", 0x3d80_0020, VA_B - 8);
}

#[test]
fn ldp_q_straddle() {
    // ldp q0, q1, [x1]  (0xad400420) — 32 bytes across the boundary.
    assert_xpage("ldp q0,q1", 0xad40_0420, VA_B - 16);
}

#[test]
fn ld1_4regs_straddle() {
    // ld1 {v0.16b-v3.16b}, [x1] (0x4c402020) — 64 bytes across the boundary.
    assert_xpage("ld1 x4", 0x4c40_2020, VA_B - 32);
}

#[test]
fn ld4_straddle() {
    // ld4 {v0.16b-v3.16b}, [x1] (0x4c400020) — 64 bytes, de-interleaved.
    assert_xpage("ld4", 0x4c40_0020, VA_B - 32);
}

#[test]
fn st4_straddle() {
    // st4 {v0.16b-v3.16b}, [x1] (0x4c000020) — 64-byte de-interleaved store.
    assert_xpage("st4", 0x4c00_0020, VA_B - 32);
}

#[test]
fn ld2_straddle() {
    // ld2 {v0.16b,v1.16b}, [x1] (0x4c408020) — 32 bytes de-interleaved.
    assert_xpage("ld2", 0x4c40_8020, VA_B - 16);
}
