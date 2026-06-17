//! The JIT backend: compile blocks to WASM and run them against the *shared*
//! guest CPU state and memory the interpreter uses.
//!
//! The platform execution loop is the organizer — it owns guest state and
//! decides which blocks run here. This type is the wasmtime substrate: a linear
//! memory holding the guest register image, a cache of compiled block instances,
//! and a `Ctx<M>` in the store that the host helper `interpret_one` reads/writes.
//!
//! Shared state (Option A): rather than copy guest RAM into the wasm linear
//! memory, the organizer **lends** its `CpuState` + memory `M` to the store for
//! the duration of a block run via a cheap swap (the TLB is boxed and the bus's
//! buffers are heap-allocated, so swapping the structs only moves pointers). A
//! block's leading register ops run inline against the register image; its escape
//! instruction calls [`interpret_one`], which single-steps the interpreter on the
//! lent `cpu` + `mem` — so memory, system instructions, MMU translation, MMIO,
//! and faults are all handled exactly as in the pure interpreter.

use std::collections::HashMap;

use aarch64_cpu_state::{regs::offsets, CpuState, GuestRegs};
use aarch64_decoder::Block;
use aarch64_interp::{GuestMem, Step};
use wasmtime::{Caller, Engine, Instance, Linker, Memory as WasmMemory, MemoryType, Module, Store};

use crate::abi;
use crate::emit::{emit_block, BLOCK_FUNC};

/// How a block run finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockExit {
    /// Guest PC to resume at.
    pub next_pc: u64,
    /// One of the `abi::EXIT_*` codes. `EXIT_NONE` means a clean exit.
    pub exit_reason: u64,
}

/// Store state: the guest CPU + memory, lent by the organizer for a block run,
/// plus the linear-memory handle the host helper uses to reach the register
/// image. `cpu`/`mem` are swapped in before a run and back out after.
struct Ctx<M: GuestMem + 'static> {
    cpu: CpuState,
    mem: M,
    wmem: Option<WasmMemory>,
}

/// A wasmtime substrate that compiles and runs guest blocks against shared state.
pub struct Vm<M: GuestMem + 'static> {
    engine: Engine,
    linker: Linker<Ctx<M>>,
    store: Store<Ctx<M>>,
    memory: WasmMemory,
    /// Compiled block instances keyed by an organizer-chosen key (physical addr).
    blocks: HashMap<u64, Instance>,
}

impl<M: GuestMem + 'static> Vm<M> {
    /// Create an empty backend. `spare` is a throwaway memory of the same type as
    /// the organizer's, parked in the store between runs (the real memory is
    /// swapped in for each run). The linear memory is a single WASM page — enough
    /// for the register image + control block (below [`abi::RAM_BASE`]); guest RAM
    /// is *not* mirrored here (it lives in `mem`, reached via `interpret_one`).
    #[must_use]
    pub fn new(spare: M) -> Self {
        let engine = Engine::default();
        let mut store =
            Store::new(&engine, Ctx { cpu: CpuState::new(), mem: spare, wmem: None });
        let memory = WasmMemory::new(&mut store, MemoryType::new(1, None)).expect("create memory");
        store.data_mut().wmem = Some(memory);

        let mut linker = Linker::new(&engine);
        linker.define(&store, "env", "memory", memory).expect("define memory");
        linker.func_wrap("env", "interpret_one", interpret_one::<M>).expect("define interpret_one");

        Vm { engine, linker, store, memory, blocks: HashMap::new() }
    }

    /// Compile and cache the block at `key` if not already present.
    pub fn ensure(&mut self, key: u64, block: &Block) {
        if self.blocks.contains_key(&key) {
            return;
        }
        let wasm = emit_block(block, 0);
        let module = Module::new(&self.engine, &wasm).expect("valid module");
        let inst = self.linker.instantiate(&mut self.store, &module).expect("instantiate");
        self.blocks.insert(key, inst);
    }

    /// Run the compiled block at `key` against the organizer's `cpu` + `mem`,
    /// which are lent to the store for the call and restored after. Returns the
    /// next PC and exit reason. Precondition: [`ensure`](Self::ensure) was called.
    pub fn run(&mut self, key: u64, cpu: &mut CpuState, mem: &mut M) -> BlockExit {
        let inst = self.blocks[&key];

        // Lend the real CPU + memory to the store (cheap pointer-swapping move).
        std::mem::swap(cpu, &mut self.store.data_mut().cpu);
        std::mem::swap(mem, &mut self.store.data_mut().mem);

        // Hot registers -> image; clear the exit control word.
        let regs = self.store.data().cpu.to_guest_regs();
        let bytes = self.memory.data_mut(&mut self.store);
        write_regs(bytes, abi::REGS_BASE as usize, &regs);
        write_u64(bytes, abi::EXIT_REASON as usize, abi::EXIT_NONE);

        let func = inst
            .get_typed_func::<i32, i64>(&mut self.store, BLOCK_FUNC)
            .expect("block export");
        let next =
            func.call(&mut self.store, abi::REGS_BASE as i32).expect("block call") as u64;

        // Image -> hot registers (capture inline updates), set PC, read exit.
        let img = read_regs(self.memory.data(&self.store), abi::REGS_BASE as usize);
        let ctx = self.store.data_mut();
        ctx.cpu.load_guest_regs(&img);
        ctx.cpu.pc = next;
        let reason = read_u64(self.memory.data(&self.store), abi::EXIT_REASON as usize);

        // Take the CPU + memory back out of the store.
        std::mem::swap(cpu, &mut self.store.data_mut().cpu);
        std::mem::swap(mem, &mut self.store.data_mut().mem);

        BlockExit { next_pc: next, exit_reason: reason }
    }

    /// Drop all compiled blocks — call when guest code may have changed
    /// (self-modifying code / I-cache maintenance).
    pub fn invalidate(&mut self) {
        self.blocks.clear();
    }
}

/// The `interpret_one(regs_base) -> i64` host import: single-step the interpreter
/// on the lent `cpu` + `mem` (so MMU/MMIO/faults are handled), syncing the hot
/// register image around the step. Returns the next guest PC; stamps the exit
/// control word on an unsupported instruction.
fn interpret_one<M: GuestMem + 'static>(mut caller: Caller<'_, Ctx<M>>, regs_base: i32) -> i64 {
    let wmem = caller.data().wmem.expect("memory set");
    let (bytes, ctx) = wmem.data_and_store_mut(&mut caller);
    let base = regs_base as usize;

    // Image -> CPU hot regs (cold state in ctx.cpu is preserved), step, regs back.
    ctx.cpu.load_guest_regs(&read_regs(bytes, base));
    let outcome = aarch64_interp::step(&mut ctx.cpu, &mut ctx.mem);
    write_regs(bytes, base, &ctx.cpu.to_guest_regs());

    match outcome {
        Step::Next(next) => {
            write_u64(bytes, abi::EXIT_REASON as usize, abi::EXIT_NONE);
            next as i64
        }
        Step::Unsupported { pc, .. } => {
            write_u64(bytes, abi::EXIT_REASON as usize, abi::EXIT_UNSUPPORTED);
            write_u64(bytes, abi::EXIT_PC as usize, pc);
            pc as i64
        }
    }
}

// --- register-image (de)serialization ---------------------------------------

fn read_u64(mem: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(mem[off..off + 8].try_into().unwrap())
}

fn write_u64(mem: &mut [u8], off: usize, val: u64) {
    mem[off..off + 8].copy_from_slice(&val.to_le_bytes());
}

fn read_regs(mem: &[u8], base: usize) -> GuestRegs {
    let rd16 =
        |off: usize| u128::from_le_bytes(mem[base + off..base + off + 16].try_into().unwrap());
    let mut x = [0u64; 31];
    for (i, slot) in x.iter_mut().enumerate() {
        *slot = read_u64(mem, base + offsets::x(i));
    }
    let mut v = [0u128; 32];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = rd16(offsets::v(i));
    }
    GuestRegs {
        x,
        sp: read_u64(mem, base + offsets::SP),
        pc: read_u64(mem, base + offsets::PC),
        nzcv: read_u64(mem, base + offsets::NZCV),
        v,
        fpcr: read_u64(mem, base + offsets::FPCR),
    }
}

fn write_regs(mem: &mut [u8], base: usize, regs: &GuestRegs) {
    for (i, val) in regs.x.iter().enumerate() {
        write_u64(mem, base + offsets::x(i), *val);
    }
    write_u64(mem, base + offsets::SP, regs.sp);
    write_u64(mem, base + offsets::PC, regs.pc);
    write_u64(mem, base + offsets::NZCV, regs.nzcv);
    for (i, val) in regs.v.iter().enumerate() {
        let off = base + offsets::v(i);
        mem[off..off + 16].copy_from_slice(&val.to_le_bytes());
    }
    write_u64(mem, base + offsets::FPCR, regs.fpcr);
}
