//! Run a test vector through our own interpreter.

use aarch64_cpu_state::{CpuState, Flags};
use aarch64_interp::{run, Memory, StopReason};

use crate::snapshot::StateSnapshot;
use crate::vector::TestVector;
use crate::{CODE_START, DATA_BASE, DATA_SIZE, MAP_BASE, MEM_SIZE};

/// Execute a vector on our interpreter and snapshot the result.
#[must_use]
pub fn run_ours(tv: &TestVector) -> (StateSnapshot, StopReason) {
    let mut mem = Memory::new(MAP_BASE, MEM_SIZE);
    mem.write(CODE_START, &tv.code);
    if let Some(data) = &tv.init_data {
        mem.write(DATA_BASE, data);
    }

    let mut cpu = CpuState::new();
    cpu.pc = CODE_START;
    cpu.x = tv.init_x;
    cpu.sp = tv.init_sp;
    cpu.flags = Flags::from_nzcv(tv.init_nzcv);
    if let Some(v) = &tv.init_v {
        cpu.v = *v;
        cpu.fpcr = tv.init_fpcr;
    }

    let stop = run(&mut cpu, &mut mem, tv.until(), tv.count);
    (snapshot(&cpu, &mem, tv.init_data.is_some(), tv.init_v.is_some()), stop)
}

fn snapshot(cpu: &CpuState, mem: &Memory, with_data: bool, with_v: bool) -> StateSnapshot {
    let data = if with_data {
        let off = (DATA_BASE - mem.base) as usize;
        mem.bytes[off..off + DATA_SIZE].to_vec()
    } else {
        Vec::new()
    };
    let v = if with_v { cpu.v.to_vec() } else { Vec::new() };
    StateSnapshot {
        x: cpu.x,
        sp: cpu.sp,
        pc: cpu.pc,
        nzcv: cpu.flags.to_nzcv(),
        data,
        v,
    }
}
