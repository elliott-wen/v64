//! Differential MMU tests against Unicorn.
//!
//! Each test builds 4KB-granule page tables (identity-mapping the code), enables
//! the MMU via seeded SCTLR/TCR/TTBR0/MAIR, then runs a single load/store at a
//! virtual address. Both our interpreter and Unicorn do a real translation-table
//! walk, and the architectural result (registers and the physical DATA window)
//! is compared. Built only with `--features unicorn`.

#![cfg(feature = "unicorn")]

use aarch64_difftest::{
    mmu_test::{mmu_vector, mmu_vector_with, mmu_vector_with_el, mmu_vector_with_el_t0sz, Leaf},
    run_ours, run_unicorn_mmu, MmuOutcome, Rng, StopReason, TestVector, DATA_BASE, DATA_SIZE,
};

const LDTR_X0_X1: [u8; 4] = 0xf840_0820u32.to_le_bytes(); // ldtr x0, [x1] (unprivileged)
const STTR_X2_X1: [u8; 4] = 0xf800_0822u32.to_le_bytes(); // sttr x2, [x1] (unprivileged)

/// Run a vector on our interpreter and report whether it faulted. On a fault our
/// interpreter vectors to the in-guest handler, which leaves ESR.DFSC in x10 and
/// FAR in x11; `Some((dfsc, far))` is that syndrome, `None` if it ran cleanly.
fn our_outcome(tv: &TestVector) -> (aarch64_difftest::StateSnapshot, Option<(u64, u64)>) {
    let (ours, stop) = run_ours(tv);
    assert!(
        !matches!(stop, StopReason::Unsupported { .. }),
        "interpreter could not run vector: {stop:?}"
    );
    let fault = if ours.x[10] != 0 { Some((ours.x[10], ours.x[11])) } else { None };
    (ours, fault)
}

/// Compare a vector against Unicorn. The key cross-check is the **fault
/// decision**: our MMU must fault exactly when Unicorn's does (this is what makes
/// demand-paging / COW correct). On success the full architectural state is
/// compared. (Unicorn stops on a guest MMU fault without populating ESR/FAR, so
/// the exact DFSC/FAR is validated against the ARM spec in the dedicated tests
/// rather than against Unicorn.)
#[track_caller]
fn assert_mmu(tv: &TestVector) {
    let (ours, fault) = our_outcome(tv);
    match run_unicorn_mmu(tv).expect("unicorn run failed") {
        MmuOutcome::Ran(oracle) => {
            assert!(fault.is_none(), "we faulted ({fault:?}) but Unicorn ran cleanly");
            if let Some(diff) = ours.diff(&oracle) {
                panic!("mismatch: {diff}\n ours:   {ours:?}\n oracle: {oracle:?}");
            }
        }
        MmuOutcome::Faulted { .. } => {
            assert!(fault.is_some(), "Unicorn faulted but we did not");
        }
    }
}

/// Our exact fault syndrome (DFSC, FAR) for spec-level assertions.
#[track_caller]
fn our_syndrome(tv: &TestVector) -> (u64, u64) {
    our_outcome(tv).1.expect("expected a fault but the access succeeded")
}

const LDR_X0_X1: [u8; 4] = 0xf940_0020u32.to_le_bytes(); // ldr x0, [x1]
const STR_X2_X1: [u8; 4] = 0xf900_0022u32.to_le_bytes(); // str x2, [x1]

// PA pages inside the compared DATA window [0x40000, 0x42000).
const PA0: u64 = DATA_BASE; // 0x40000
const PA1: u64 = DATA_BASE + 0x1000; // 0x41000

/// 8KB of initial DATA content with a position-dependent, per-byte-distinct
/// pattern, so any byte sourced from the wrong physical page is caught.
fn patterned_data() -> Vec<u8> {
    (0..DATA_SIZE).map(|i| (i as u8) ^ ((i >> 8) as u8).wrapping_mul(31)).collect()
}

#[test]
fn mmu_simple_page_load() {
    // VA 64GiB -> PA0; load the 8 bytes there and compare.
    let va = 0x3000_0000u64;
    let tv = mmu_vector(&LDR_X0_X1, &[(va, PA0)], &[], Some(patterned_data()))
        .with_x(1, va);
    assert_mmu(&tv);
}

#[test]
fn mmu_cross_page_load() {
    // Two adjacent VA pages mapped to *reversed* (non-adjacent) physical pages,
    // so an 8-byte load straddling the boundary must translate each half
    // independently. The buggy single-translate path reads the wrong PA for the
    // upper half.
    let va = 0x3000_0000u64;
    let tv = mmu_vector(
        &LDR_X0_X1,
        &[(va, PA1), (va + 0x1000, PA0)], // page0->PA1, page1->PA0
        &[],
        Some(patterned_data()),
    )
    .with_x(1, va + 0xffc); // 4 bytes before the page boundary
    assert_mmu(&tv);
}

#[test]
fn mmu_cross_page_store() {
    let va = 0x3000_0000u64;
    let tv = mmu_vector(
        &STR_X2_X1,
        &[(va, PA1), (va + 0x1000, PA0)],
        &[],
        Some(patterned_data()),
    )
    .with_x(1, va + 0xffc)
    .with_x(2, 0xdead_beef_cafe_babe);
    assert_mmu(&tv);
}

#[test]
fn mmu_block_2mb_load() {
    // 2MB block at VA 1GiB -> PA 0; load from offset 0x40000 (== PA0) to check
    // the L2 block descriptor and the large offset are honoured.
    let va_block = 0x4000_0000u64;
    let tv = mmu_vector(&LDR_X0_X1, &[], &[(va_block, 0)], Some(patterned_data()))
        .with_x(1, va_block + PA0);
    assert_mmu(&tv);
}

#[test]
fn mmu_unaligned_within_page_load() {
    // Misaligned but within a single page: a sanity check that the fast path and
    // page-offset handling agree with Unicorn.
    let va = 0x3000_0000u64;
    let tv = mmu_vector(&LDR_X0_X1, &[(va, PA0)], &[], Some(patterned_data()))
        .with_x(1, va + 0x7); // unaligned offset, no page crossing
    assert_mmu(&tv);
}

// --- Fault-syndrome tests: the guest fault handler captures ESR.DFSC (x10) and
// FAR (x11); assert_matches_oracle compares them against Unicorn. ---

#[test]
fn mmu_permission_fault_on_ro_store() {
    // Map the page read-only, then store to it: a permission fault (the COW
    // trigger). DFSC and FAR must match Unicorn.
    let va = 0x3000_0000u64;
    let tv = mmu_vector_with(
        &STR_X2_X1,
        |pt| pt.map_page_attr(va, PA0, /*writable=*/ false, /*af=*/ true),
        Some(patterned_data()),
    )
    .with_x(1, va)
    .with_x(2, 0x1234_5678_9abc_def0);
    assert_mmu(&tv); // fault decision matches Unicorn
    // ...and the exact syndrome is the ARM-spec permission fault at level 3.
    assert_eq!(our_syndrome(&tv), (0x0f, va), "permission fault DFSC/FAR");
}

#[test]
fn mmu_access_flag_fault_on_load() {
    // Map the page with AF clear, then load: an access-flag fault.
    let va = 0x3000_0000u64;
    let tv = mmu_vector_with(
        &LDR_X0_X1,
        |pt| pt.map_page_attr(va, PA0, /*writable=*/ true, /*af=*/ false),
        Some(patterned_data()),
    )
    .with_x(1, va);
    assert_mmu(&tv);
    assert_eq!(our_syndrome(&tv), (0x0b, va), "access-flag fault DFSC/FAR (level 3)");
}

#[test]
fn mmu_translation_fault_on_unmapped() {
    // Load from a VA with no mapping at all: a translation fault.
    let mapped = 0x3000_0000u64;
    let unmapped = 0x4000_0000u64; // different L1 slot, never mapped
    let tv = mmu_vector_with(
        &LDR_X0_X1,
        |pt| pt.map_page(mapped, PA0),
        Some(patterned_data()),
    )
    .with_x(1, unmapped);
    assert_mmu(&tv);
    // Unmapped at the L1 slot -> translation fault at level 1.
    assert_eq!(our_syndrome(&tv), (0x05, unmapped), "translation fault DFSC/FAR (level 1)");
}

// --- EL0 permission tests (access runs at EL0 via ERET) ---

#[test]
fn mmu_el0_load_ok() {
    // EL0-accessible page, load at EL0: succeeds and matches Unicorn.
    let va = 0x3000_0000u64;
    let tv = mmu_vector_with_el(
        &LDR_X0_X1,
        |pt| pt.map_page_leaf(va, PA0, Leaf::rw()),
        Some(patterned_data()),
        0,
    )
    .with_x(1, va);
    assert_mmu(&tv);
}

#[test]
fn mmu_el0_fault_on_privileged_page() {
    // Page NOT accessible at EL0 (AP[1]=0): an EL0 load faults (permission),
    // though the same access would succeed at EL1. Validates the EL0 rule.
    let va = 0x3000_0000u64;
    let leaf = Leaf { el0: false, ..Leaf::rw() };
    let tv = mmu_vector_with_el(
        &LDR_X0_X1,
        |pt| pt.map_page_leaf(va, PA0, leaf),
        Some(patterned_data()),
        0,
    )
    .with_x(1, va);
    assert_mmu(&tv);
    assert_eq!(our_syndrome(&tv), (0x0f, va), "EL0 permission fault DFSC/FAR");
}

#[test]
fn mmu_el0_store_to_ro_faults() {
    // EL0 store to a read-only page: permission fault.
    let va = 0x3000_0000u64;
    let leaf = Leaf { writable: false, ..Leaf::rw() };
    let tv = mmu_vector_with_el(
        &STR_X2_X1,
        |pt| pt.map_page_leaf(va, PA0, leaf),
        Some(patterned_data()),
        0,
    )
    .with_x(1, va)
    .with_x(2, 0xa5a5_a5a5_a5a5_a5a5);
    assert_mmu(&tv);
    assert_eq!(our_syndrome(&tv), (0x0f, va), "EL0 RO-store permission fault");
}

// --- Hierarchical (table-descriptor) permission tests ---

#[test]
fn mmu_hierarchical_no_write_faults_store() {
    // Leaf is writable, but an APTable[1] (read-only) restriction on the L2 table
    // descriptor must make a store fault. Validates hierarchical accumulation.
    let va = 0x3000_0000u64;
    let tv = mmu_vector_with(
        &STR_X2_X1,
        |pt| {
            pt.map_page_leaf(va, PA0, Leaf::rw());
            pt.restrict_table(va, 2, /*no_el0=*/ false, /*no_write=*/ true, false, false);
        },
        Some(patterned_data()),
    )
    .with_x(1, va)
    .with_x(2, 0x1111_2222_3333_4444);
    assert_mmu(&tv);
    assert_eq!(our_syndrome(&tv), (0x0f, va), "hierarchical RO permission fault");
}

#[test]
fn mmu_hierarchical_no_el0_faults_el0() {
    // Leaf is EL0-accessible, but an APTable[0] (no-EL0) restriction on the L1
    // table descriptor must make an EL0 access fault.
    let va = 0x3000_0000u64;
    let tv = mmu_vector_with_el(
        &LDR_X0_X1,
        |pt| {
            pt.map_page_leaf(va, PA0, Leaf::rw());
            pt.restrict_table(va, 1, /*no_el0=*/ true, false, false, false);
        },
        Some(patterned_data()),
        0,
    )
    .with_x(1, va);
    assert_mmu(&tv);
}

// --- Unprivileged load/store (LDTR/STTR): permission-checked at EL0 even from
// EL1. Run at EL1 so the EL0-check behaviour is observable. ---

#[test]
fn mmu_sttr_unpriv_faults_on_non_el0_page() {
    // Page is NOT EL0-accessible. A normal STR at EL1 would succeed, but STTR is
    // unprivileged, so it must fault (permission) — matching Unicorn.
    let va = 0x3000_0000u64;
    let leaf = Leaf { el0: false, ..Leaf::rw() };
    let tv = mmu_vector_with(
        &STTR_X2_X1,
        |pt| pt.map_page_leaf(va, PA0, leaf),
        Some(patterned_data()),
    )
    .with_x(1, va)
    .with_x(2, 0xcafe_f00d_dead_beef);
    assert_mmu(&tv);
    assert_eq!(our_syndrome(&tv), (0x0f, va), "STTR unprivileged permission fault");
}

#[test]
fn mmu_ldtr_unpriv_ok_on_el0_page() {
    // EL0-accessible page: LDTR at EL1 succeeds and reads the same value Unicorn
    // does (confirms the unprivileged path still translates correctly).
    let va = 0x3000_0000u64;
    let tv = mmu_vector_with(
        &LDTR_X0_X1,
        |pt| pt.map_page_leaf(va, PA0, Leaf::rw()),
        Some(patterned_data()),
    )
    .with_x(1, va);
    assert_mmu(&tv);
}

#[test]
fn mmu_sttr_unpriv_ok_on_el0_page() {
    // EL0-accessible RW page: STTR at EL1 succeeds; the written bytes match
    // Unicorn (the DATA window is compared).
    let va = 0x3000_0000u64;
    let tv = mmu_vector_with(
        &STTR_X2_X1,
        |pt| pt.map_page_leaf(va, PA0, Leaf::rw()),
        Some(patterned_data()),
    )
    .with_x(1, va)
    .with_x(2, 0x0011_2233_4455_6677);
    assert_mmu(&tv);
}

// --- Randomized MMU fuzzing against Unicorn ---

/// Iterations for the randomized MMU sweep (override with `MMU_FUZZ_ITERS`).
fn fuzz_iters() -> u32 {
    std::env::var("MMU_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(2000)
}

#[test]
fn mmu_random_sweep() {
    let mut rng = Rng::new(0x4d4d_5530); // "MMU0"
    for i in 0..fuzz_iters() {
        // Two adjacent VA pages mapped to a random permutation of two physical
        // pages in the compared DATA window, with random permissions/AF/EL0 bits,
        // a random access EL, optional 2MB block, and optional hierarchical
        // (table-descriptor) restrictions.
        let vbase = 0x2000_0000u64 + u64::from(rng.bits(10)) * 0x2000;
        let swap = rng.bits(1) == 1;
        let (p0, p1) = if swap { (PA1, PA0) } else { (PA0, PA1) };
        let leaf = |rng: &mut Rng| Leaf {
            writable: rng.bits(1) == 1,
            af: rng.bits(1) == 1,
            el0: rng.bits(1) == 1,
            uxn: false,
            pxn: false,
        };
        let l0 = leaf(&mut rng);
        let l1 = leaf(&mut rng);
        // ~1/8 of the time leave the region unmapped to exercise translation faults.
        let map_pages = rng.bits(3) != 0;
        let el = rng.bits(1) as u8; // run the access at EL0 or EL1
        // Vary the translation-regime size. 25 is the kernel's actual T0SZ
        // (VA_BITS=39); all three start at level 1. (Unicorn's CPU TLB only
        // cleanly accepts these 3-level values in a single-table regime — it
        // rejects T0SZ 26..31 as a config it won't walk — so we stick to the set
        // it supports; our own walker handles the full range.)
        let t0sz = [25u64, 32, 33][(rng.bits(2) % 3) as usize];
        // Occasionally add a hierarchical restriction on the L1 table descriptor.
        let restrict = rng.bits(2) == 0;
        let r_no_el0 = rng.bits(1) == 1;
        let r_no_write = rng.bits(1) == 1;

        let is_store = rng.bits(1) == 1;
        // Offset spans both pages and the boundary (incl. unaligned/cross-page).
        let off = u64::from(rng.bits(13)) % 0x1ff9;
        let store_val = rng.next_u64();

        let access = if is_store { STR_X2_X1 } else { LDR_X0_X1 };
        let tv = mmu_vector_with_el_t0sz(
            &access,
            |pt| {
                if map_pages {
                    pt.map_page_leaf(vbase, p0, l0);
                    pt.map_page_leaf(vbase + 0x1000, p1, l1);
                    if restrict {
                        pt.restrict_table(vbase, 1, r_no_el0, r_no_write, false, false);
                    }
                }
            },
            Some(patterned_data()),
            el,
            t0sz,
        )
        .with_x(1, vbase + off)
        .with_x(2, store_val);

        let ctx = format!(
            "iter {i} el={el} t0sz={t0sz} vbase={vbase:#x} off={off:#x} store={is_store} \
             map={map_pages} swap={swap} restrict={restrict}({r_no_el0},{r_no_write}) \
             l0=(w{},af{},e{}) l1=(w{},af{},e{})",
            l0.writable, l0.af, l0.el0, l1.writable, l1.af, l1.el0
        );
        let (ours, fault) = our_outcome(&tv);
        match run_unicorn_mmu(&tv).expect("unicorn run failed") {
            MmuOutcome::Ran(oracle) => {
                assert!(fault.is_none(), "{ctx}: we faulted ({fault:?}) but Unicorn ran");
                if let Some(diff) = ours.diff(&oracle) {
                    panic!("{ctx}: {diff}\n ours:   {ours:?}\n oracle: {oracle:?}");
                }
            }
            MmuOutcome::Faulted { .. } => {
                assert!(fault.is_some(), "{ctx}: Unicorn faulted but we did not");
            }
        }
    }
}
