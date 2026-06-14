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
