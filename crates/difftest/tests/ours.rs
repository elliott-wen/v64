//! Tests that run only our interpreter (no oracle, always available).

use aarch64_difftest::{run_ours, StopReason, TestVector};

#[test]
fn ours_runs_until_program() {
    // mov x16,#1 ; mov x17,#0x20 ; add x28,x28,8
    let tv = TestVector::new(&[
        0x30, 0x00, 0x80, 0xd2,
        0x11, 0x04, 0x80, 0xd2,
        0x9c, 0x23, 0x00, 0x91,
    ])
    .with_x(28, 0x12341234);

    let (snap, stop) = run_ours(&tv);
    assert_eq!(stop, StopReason::UntilReached);
    assert_eq!(snap.x[16], 0x1);
    assert_eq!(snap.x[17], 0x20);
    assert_eq!(snap.x[28], 0x1234123c);
}

/// DAIF is a PSTATE field with dedicated `CpuState` storage. The *immediate*
/// MSR forms (DAIFSet/DAIFClr) and the *register* forms (`msr daif,x` /
/// `mrs x,daif`) must operate on that same state — `local_irq_save`/`restore`
/// mix them, and if the register form fell through to the generic sysreg map it
/// would desync from `cpu.daif` and leave IRQs stuck masked. This pins the
/// cross-form round-trip: clear DAIF via the register form, set the I bit via
/// the immediate form, read it back via the register form.
#[test]
fn ours_daif_cross_form_roundtrips() {
    let tv = TestVector::new(&[
        0x01, 0x00, 0x80, 0xd2, // mov x1, #0
        0x21, 0x42, 0x1b, 0xd5, // msr daif, x1    (register form -> clears DAIF)
        0xdf, 0x42, 0x03, 0xd5, // msr daifset, #2 (immediate form -> sets I)
        0x20, 0x42, 0x3b, 0xd5, // mrs x0, daif    (register form -> reads it back)
    ]);
    let (snap, stop) = run_ours(&tv);
    assert_eq!(stop, StopReason::UntilReached);
    // DAIF reads back with the I bit (PSTATE bit 7) set, the rest clear.
    assert_eq!(snap.x[0], 0x80, "mrs daif must observe the daifset I bit");
}

/// The register-form `msr nzcv,x` must reach the dedicated flags state (not the
/// generic sysreg map), so a subsequent `mrs x,nzcv` and the PSTATE flags agree.
#[test]
fn ours_nzcv_reg_form_writes_flags() {
    let tv = TestVector::new(&[
        0x02, 0x00, 0xbe, 0xd2, // mov x2, #0xf0000000
        0x02, 0x42, 0x1b, 0xd5, // msr nzcv, x2
        0x03, 0x42, 0x3b, 0xd5, // mrs x3, nzcv
    ]);
    let (snap, stop) = run_ours(&tv);
    assert_eq!(stop, StopReason::UntilReached);
    assert_eq!(snap.nzcv, 0xf000_0000, "msr nzcv must update the flags state");
    assert_eq!(snap.x[3], 0xf000_0000, "mrs nzcv must read the flags state back");
}
