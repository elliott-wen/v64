//! The JIT and runtime agree on `GuestRegs`'s layout via `regs::offsets`. This
//! guard fails if any field offset, the total size, or the alignment shifts.

use aarch64_cpu_state::{regs::offsets, GuestRegs};
use std::mem::{align_of, offset_of, size_of};

#[test]
fn offsets_are_stable() {
    assert_eq!(offset_of!(GuestRegs, x), offsets::X);
    assert_eq!(offset_of!(GuestRegs, sp), offsets::SP);
    assert_eq!(offset_of!(GuestRegs, pc), offsets::PC);
    assert_eq!(offset_of!(GuestRegs, nzcv), offsets::NZCV);
    assert_eq!(offset_of!(GuestRegs, v), offsets::V);
    assert_eq!(offset_of!(GuestRegs, fpcr), offsets::FPCR);
    assert_eq!(size_of::<GuestRegs>(), offsets::SIZE);
    // u128 forces 16-byte alignment; the JIT relies on this.
    assert_eq!(align_of::<GuestRegs>(), 16);
}
