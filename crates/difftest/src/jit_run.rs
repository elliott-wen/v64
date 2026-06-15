//! Run a test vector through the WebAssembly JIT — the third differential
//! runner, alongside [`run_ours`](crate::run_ours) and the Unicorn oracle.
//!
//! Available only under the `jit` feature (it pulls in wasmtime). Builds the
//! same linear-memory image and initial state `run_ours` does, runs it through
//! the JIT dispatcher, and snapshots identically so the two are comparable.

use aarch64_cpu_state::GuestRegs;
use aarch64_interp::StopReason;
use aarch64_jit::Vm;

use crate::snapshot::StateSnapshot;
use crate::vector::TestVector;
use crate::{CODE_START, DATA_BASE, DATA_SIZE, MAP_BASE, MEM_SIZE};

/// Execute a vector on the JIT and snapshot the result. A fresh VM per call —
/// the clean API; the sweep reuses one VM via [`run_on`] for speed.
#[must_use]
pub fn run_jit(tv: &TestVector) -> (StateSnapshot, StopReason) {
    let mut vm = Vm::new(MAP_BASE, MEM_SIZE);
    run_on(&mut vm, tv)
}

/// Run `tv` on an existing VM (reused across fuzz iterations to amortize the
/// engine/linear-memory setup). Code, data, and the register image are fully
/// overwritten each call; the fuzz classes never touch cold CPU state, so the
/// VM's persisted cold state stays at its default between vectors.
pub(crate) fn run_on(vm: &mut Vm, tv: &TestVector) -> (StateSnapshot, StopReason) {
    vm.write_ram(CODE_START, &tv.code);
    if let Some(data) = &tv.init_data {
        vm.write_ram(DATA_BASE, data);
    }

    let mut regs = GuestRegs { sp: tv.init_sp, pc: CODE_START, nzcv: tv.init_nzcv, ..GuestRegs::default() };
    regs.x = tv.init_x;
    if let Some(v) = &tv.init_v {
        regs.v = *v;
        regs.fpcr = tv.init_fpcr;
    }
    vm.load_regs(&regs);

    let stop = vm.run(tv.until(), tv.count);

    let out = vm.store_regs();
    let data = if tv.init_data.is_some() { vm.read_ram(DATA_BASE, DATA_SIZE) } else { Vec::new() };
    let v = if tv.init_v.is_some() { out.v.to_vec() } else { Vec::new() };
    (StateSnapshot { x: out.x, sp: out.sp, pc: out.pc, nzcv: out.nzcv, data, v }, stop)
}
