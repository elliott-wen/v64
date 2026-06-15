//! The JIT runtime: an embedded wasmtime instance whose linear memory is shared
//! with the host, plus the `interpret_one` host import generated blocks call
//! into for anything not lowered inline (Milestone 2).
//!
//! For now [`Vm`] runs a single formed block end to end — enough to exercise the
//! full block ABI through the interpreter escape hatch. Milestone 5 grows this
//! into a dispatcher with a block cache; the linear-memory layout and the
//! `interpret_one` contract established here are what it builds on.

use aarch64_cpu_state::{regs::offsets, CpuState, GuestRegs};
use aarch64_interp::Memory as GuestMem;
use wasmtime::{
    Caller, Engine, Instance, Linker, MemoryType, Module, Store, Memory as WasmMemory,
};

use crate::abi;
use crate::block::Block;
use crate::emit::{emit_block, BLOCK_FUNC};

/// Per-instance host state threaded through the wasmtime `Store`.
///
/// The hot register file and guest RAM live in the wasmtime linear memory (the
/// single source of truth). The **cold** CPU state — sysregs, EL, exclusives,
/// etc. — has no flat layout and lives here, persisted across calls; only its
/// hot fields are synced to/from the linear-memory image on each step.
struct Runtime {
    cpu: CpuState,
    memory: Option<WasmMemory>,
    guest_base: u64,
    ram_bytes: usize,
    /// Number of `interpret_one` calls (escape-hatch fallbacks) so far. Lets
    /// callers confirm how much of a block was lowered inline.
    interp_calls: u64,
}

/// How a block run finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockExit {
    /// Guest PC to resume at.
    pub next_pc: u64,
    /// One of the `abi::EXIT_*` codes. `EXIT_NONE` means a clean exit.
    pub exit_reason: u64,
}

/// An embedded wasmtime instance that runs JIT-compiled guest blocks.
pub struct Vm {
    engine: Engine,
    linker: Linker<Runtime>,
    store: Store<Runtime>,
    memory: WasmMemory,
    guest_base: u64,
}

impl Vm {
    /// Create a VM whose guest RAM window starts at guest address `guest_base`
    /// and is `ram_bytes` long (mapped at [`abi::RAM_BASE`] in linear memory).
    #[must_use]
    pub fn new(guest_base: u64, ram_bytes: usize) -> Self {
        let engine = Engine::default();
        let mut store = Store::new(
            &engine,
            Runtime { cpu: CpuState::new(), memory: None, guest_base, ram_bytes, interp_calls: 0 },
        );

        let total = abi::RAM_BASE as usize + ram_bytes;
        let pages = total.div_ceil(abi::WASM_PAGE) as u32;
        let memory =
            WasmMemory::new(&mut store, MemoryType::new(pages, None)).expect("create memory");
        store.data_mut().memory = Some(memory);

        let mut linker = Linker::new(&engine);
        linker.define(&store, "env", "memory", memory).expect("define memory");
        linker
            .func_wrap("env", "interpret_one", interpret_one)
            .expect("define interpret_one");

        Vm { engine, linker, store, memory, guest_base }
    }

    /// Write the initial register image into linear memory.
    pub fn load_regs(&mut self, regs: &GuestRegs) {
        let bytes = self.memory.data_mut(&mut self.store);
        write_regs(bytes, abi::REGS_BASE as usize, regs);
    }

    /// Read the current register image back out of linear memory.
    #[must_use]
    pub fn store_regs(&mut self) -> GuestRegs {
        read_regs(self.memory.data(&self.store), abi::REGS_BASE as usize)
    }

    /// Write `data` into guest RAM at guest address `addr` (e.g. code or data).
    pub fn write_ram(&mut self, addr: u64, data: &[u8]) {
        let off = abi::ram_offset(addr, self.guest_base);
        self.memory.data_mut(&mut self.store)[off..off + data.len()].copy_from_slice(data);
    }

    /// Read `len` bytes of guest RAM starting at guest address `addr`.
    #[must_use]
    pub fn read_ram(&mut self, addr: u64, len: usize) -> Vec<u8> {
        let off = abi::ram_offset(addr, self.guest_base);
        self.memory.data(&self.store)[off..off + len].to_vec()
    }

    /// Total number of `interpret_one` (escape-hatch) calls so far. Zero means
    /// every instruction run was lowered inline.
    #[must_use]
    pub fn interp_calls(&self) -> u64 {
        self.store.data().interp_calls
    }

    /// Compile `block` to WASM and instantiate it against the shared memory and
    /// the `interpret_one` import. The returned [`Instance`] can be cached and
    /// re-run with [`call_instance`](Self::call_instance) as long as this `Vm`
    /// (and its store) lives.
    pub(crate) fn compile_instance(&mut self, block: &Block) -> Instance {
        let wasm = emit_block(block, self.guest_base);
        let module = Module::new(&self.engine, &wasm).expect("valid module");
        self.linker.instantiate(&mut self.store, &module).expect("instantiate")
    }

    /// Run a compiled block instance and report where it exited.
    pub(crate) fn call_instance(&mut self, instance: Instance) -> BlockExit {
        // Clear the control block; interpret_one re-stamps it on each step.
        write_u64(self.memory.data_mut(&mut self.store), abi::EXIT_REASON as usize, abi::EXIT_NONE);

        let func = instance
            .get_typed_func::<i32, i64>(&mut self.store, BLOCK_FUNC)
            .expect("block export");

        match func.call(&mut self.store, abi::REGS_BASE as i32) {
            Ok(next) => {
                // The returned PC is authoritative; write it into the image so
                // an inline terminator (which leaves the PC on the stack but
                // doesn't touch the image) and a helper exit agree. Then read
                // the exit reason the last step stamped into the control block.
                let bytes = self.memory.data_mut(&mut self.store);
                write_u64(bytes, abi::REGS_BASE as usize + offsets::PC, next as u64);
                let reason = read_u64(bytes, abi::EXIT_REASON as usize);
                BlockExit { next_pc: next as u64, exit_reason: reason }
            }
            Err(_trap) => {
                // A WASM trap (e.g. an out-of-bounds inline memory access) aborts
                // the block. Surface it as a guest fault at the instruction that
                // was executing — whose PC is still in the image, since lowerings
                // advance the image PC only after the access completes — rather
                // than letting it propagate as a host panic.
                let bytes = self.memory.data_mut(&mut self.store);
                let pc = read_u64(bytes, abi::REGS_BASE as usize + offsets::PC);
                write_u64(bytes, abi::EXIT_REASON as usize, abi::EXIT_FAULT);
                BlockExit { next_pc: pc, exit_reason: abi::EXIT_FAULT }
            }
        }
    }

    /// Compile, run, and discard a single block. Convenience for tests; the
    /// dispatcher ([`Vm::run`]) caches instances instead.
    pub fn run_block(&mut self, block: &Block) -> BlockExit {
        let instance = self.compile_instance(block);
        self.call_instance(instance)
    }

    /// Current guest PC from the register image.
    pub(crate) fn image_pc(&self) -> u64 {
        read_u64(self.memory.data(&self.store), abi::REGS_BASE as usize + offsets::PC)
    }

    /// Read the 32-bit guest instruction word at guest address `addr`.
    pub(crate) fn read_code_word(&self, addr: u64) -> u32 {
        let off = abi::ram_offset(addr, self.guest_base);
        let b = self.memory.data(&self.store);
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }
}

/// The `interpret_one(regs_base) -> i64` host import: step exactly one guest
/// instruction at the current PC through the interpreter, against the shared
/// linear memory, and return the next guest PC.
///
/// This is the long-tail escape hatch — generated code calls it for anything not
/// lowered inline. It is deliberately the slow path: it copies the register
/// image and guest-RAM window across the interpreter boundary each call. Later
/// milestones lower common instructions so this runs far less often.
fn interpret_one(mut caller: Caller<'_, Runtime>, regs_base: i32) -> i64 {
    let wmem = caller.data().memory.expect("memory set");
    let (bytes, rt) = wmem.data_and_store_mut(&mut caller);
    rt.interp_calls += 1;
    let base = regs_base as usize;

    // Hot regs: linear memory -> CpuState (cold state in rt.cpu is preserved).
    let gr = read_regs(bytes, base);
    rt.cpu.load_guest_regs(&gr);

    // Guest RAM window: linear memory -> a Memory the interpreter can step on.
    //
    // PORTABILITY SEAM: this copy-out/copy-back exists only because the native
    // interpreter's `Memory` owns a `Vec<u8>` and can't view wasmtime's bytes
    // in place. It is the one piece of the runtime that does NOT survive an
    // all-wasm deployment: there, the interpreter would itself be a WASM module
    // importing this same linear memory and would read/write it directly (no
    // copy). When/if `interp::Memory` becomes a view over a shared backing
    // store, this whole window-copy disappears and `interpret_one` operates on
    // `bytes` directly. Inline lowerings (see `lower.rs`) already bypass it.
    let ram_lo = abi::RAM_BASE as usize;
    let ram_hi = ram_lo + rt.ram_bytes;
    let mut gmem = GuestMem { base: rt.guest_base, bytes: bytes[ram_lo..ram_hi].to_vec() };

    let outcome = aarch64_interp::step(&mut rt.cpu, &mut gmem);

    // Write back: RAM window first, then the hot register image.
    bytes[ram_lo..ram_hi].copy_from_slice(&gmem.bytes);
    write_regs(bytes, base, &rt.cpu.to_guest_regs());

    match outcome {
        aarch64_interp::Step::Next(next) => {
            write_u64(bytes, abi::EXIT_REASON as usize, abi::EXIT_NONE);
            next as i64
        }
        aarch64_interp::Step::Unsupported { pc, .. } => {
            write_u64(bytes, abi::EXIT_REASON as usize, abi::EXIT_UNSUPPORTED);
            write_u64(bytes, abi::EXIT_PC as usize, pc);
            pc as i64
        }
    }
}

// --- Register-image (de)serialization at the JIT boundary -------------------

fn read_u64(mem: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(mem[off..off + 8].try_into().unwrap())
}

fn write_u64(mem: &mut [u8], off: usize, val: u64) {
    mem[off..off + 8].copy_from_slice(&val.to_le_bytes());
}

/// Decode a [`GuestRegs`] image at `base` out of a linear-memory byte slice.
fn read_regs(mem: &[u8], base: usize) -> GuestRegs {
    let rd16 = |off: usize| u128::from_le_bytes(mem[base + off..base + off + 16].try_into().unwrap());
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

/// Encode a [`GuestRegs`] image at `base` into a linear-memory byte slice.
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
