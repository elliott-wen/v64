//! Advanced SIMD extract (EXT): `Vd = bytes[imm4 .. imm4+datasize)` of the
//! concatenation {Vm:Vn} (Vn is the low operand).

use aarch64_cpu_state::CpuState;

pub(crate) fn exec(cpu: &mut CpuState, q: bool, imm4: u8, rm: u8, rn: u8, rd: u8) -> Option<u64> {
    let nbytes = if q { 16 } else { 8 };
    let vn = cpu.v[rn as usize].to_le_bytes();
    let vm = cpu.v[rm as usize].to_le_bytes();

    // Concatenate the low `nbytes` of Vn then Vm.
    let mut cat = [0u8; 32];
    cat[..nbytes].copy_from_slice(&vn[..nbytes]);
    cat[nbytes..2 * nbytes].copy_from_slice(&vm[..nbytes]);

    let mut out = [0u8; 16];
    let start = imm4 as usize;
    out[..nbytes].copy_from_slice(&cat[start..start + nbytes]);
    cpu.v[rd as usize] = u128::from_le_bytes(out); // Q=0 leaves the upper half zero
    None
}
