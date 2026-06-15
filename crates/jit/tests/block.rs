//! Block-formation tests: decode forward, stop at the first terminator.

use aarch64_decoder::Insn;
use aarch64_jit::form_block;

#[test]
fn stops_at_branch() {
    // NOP; NOP; B . (self-branch). `read` returns each word by index.
    let code = [0xd503201fu32, 0xd503201f, 0x1400_0000];
    let block = form_block(0x1000, |pc| code[((pc - 0x1000) / 4) as usize]);
    assert_eq!(block.insns.len(), 3);
    assert!(matches!(block.insns[2].1, Insn::BranchImm { .. }));
}
