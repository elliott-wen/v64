//! wasm-bindgen entry points for running the AArch64 emulator in a browser or
//! node. Two layers:
//!
//! - [`run_code`] — a one-shot interpreter smoke test (run a code blob, read X0).
//! - [`Emulator`] — the full system Machine (CPU + bus + virtio devices) booting
//!   a real Linux `Image`, with a JS-backed [`Clock`] (the one piece native
//!   `HostClock` can't provide under wasm, since `Instant` is unavailable).

use aarch64_cpu_state::CpuState;
use aarch64_interp::{run, Memory, StopReason};
use aarch64_platform::{Board, BlockRunner, Clock, InputKind, DEFAULT_FREQ_HZ, RAM_BASE};
use wasm_bindgen::prelude::*;

/// Guest base address the code blob is loaded at (for [`run_code`]).
const BASE: u64 = 0x1000;

/// Run `code` for `steps` instructions through the interpreter and return X0.
/// A minimal smoke test that the interpreter executes real AArch64 under wasm.
#[wasm_bindgen]
pub fn run_code(code: &[u8], steps: usize) -> u64 {
    let mut mem = Memory::new(BASE, 0x1_0000);
    mem.write(BASE, code);
    let mut cpu = CpuState::new();
    cpu.pc = BASE;
    run(&mut cpu, &mut mem, u64::MAX, steps);
    cpu.x[0]
}

// JS time source for the guest timer. `Date.now()` is wall-clock milliseconds.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Date, js_name = now)]
    fn date_now() -> f64;
}

/// Timer clock backed by JS `Date.now()`, scaled to the architected tick rate —
/// the browser/node stand-in for native `HostClock`.
struct WasmClock {
    freq: u64,
}

impl Clock for WasmClock {
    fn now(&self) -> u64 {
        (date_now() * (self.freq as f64) / 1000.0) as u64
    }
}

// The JIT compile backend, in JS over the `WebAssembly` API. `jit_set_memory`
// gives it this module's linear memory (so compiled blocks share it and read the
// register image — the live `CpuState` — at `regs_base`); `jit_set_table` gives
// it this module's indirect function table. `jit_compile` instantiates a block
// module (importing that memory), appends its `block` export to the table, and
// returns the slot index.
//
// Crucially, *running* a block is **not** here: a table slot index is just a
// Rust fn-pointer value in wasm, so [`WasmRunner::run`] calls it with a direct
// `call_indirect` — no JS round-trip per block (the previous `jit_run` hop was
// the dominant per-block cost). JS is touched only at compile time.
#[wasm_bindgen(inline_js = "
let mem = null;
let table = null;
export function jit_set_memory(m) { mem = m; }
export function jit_set_table(t) { table = t; }
export function jit_compile(bytes) {
    const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory: mem } });
    const idx = table.grow(1);               // append a slot; returns its index
    table.set(idx, inst.exports.block);
    return idx;
}
export function jit_invalidate() { /* table slots leak until the table is reused; the organizer drops the handles so stale blocks are never called */ }
")]
extern "C" {
    fn jit_set_memory(mem: JsValue);
    fn jit_set_table(table: JsValue);
    fn jit_compile(bytes: &[u8]) -> u32;
    fn jit_invalidate();
}

/// [`BlockRunner`] over the browser/node `WebAssembly` API. Compilation goes
/// through JS; execution is a direct in-wasm `call_indirect`.
struct WasmRunner;

impl BlockRunner for WasmRunner {
    fn compile(&mut self, wasm: &[u8]) -> u32 {
        jit_compile(wasm)
    }
    fn run(&mut self, handle: u32, regs_base: u32, ram_base: u32) -> u64 {
        // `handle` is the block's slot in this module's indirect function table.
        // A wasm fn pointer *is* that table index, so transmuting it yields a
        // callable pointer and the call lowers to `call_indirect` — entirely
        // inside wasm, with no JS boundary crossing. The block's type
        // `(i32, i32) -> i64` matches this signature, so `call_indirect`
        // type-checks. (wasm32-only: a fn pointer and `usize` are both the
        // 32-bit table index.)
        let block: extern "C" fn(u32, u32) -> u64 =
            unsafe { core::mem::transmute(handle as usize) };
        block(regs_base, ram_base)
    }
    fn invalidate(&mut self) {
        jit_invalidate();
    }
}

/// A microbenchmark harness: a bare machine (no Linux) running a raw code blob
/// at the RAM base, MMU off (identity-mapped, EL1). Isolates a workload pattern
/// — e.g. a tight loop — to measure the JIT's per-pattern speedup, away from the
/// boot's cold-code-heavy mix. (MMU off means no TLB, so inline memory bails;
/// use register/branch kernels to measure compute throughput.)
#[wasm_bindgen]
pub struct Kernel {
    board: Board,
}

#[wasm_bindgen]
impl Kernel {
    /// Load `code` at the RAM base and point the PC at it.
    #[wasm_bindgen(constructor)]
    pub fn new(code: &[u8]) -> Kernel {
        // A few MiB is ample for a tiny in-place loop (and avoids the 1 GiB
        // default — these are created per kernel).
        let clock = Box::new(WasmClock { freq: DEFAULT_FREQ_HZ });
        let mut board = Board::with_ram_and_clock(4 << 20, clock);
        board.machine.bus.ram_mut().write(RAM_BASE, code);
        board.machine.cpu.pc = RAM_BASE;
        Kernel { board }
    }

    /// Run `budget` instructions through the interpreter; return the count run.
    pub fn run(&mut self, budget: usize) -> u64 {
        self.board.machine.run(0, budget);
        self.board.machine.total_insns()
    }

    /// Run `budget` instructions JIT-organized; return the count run.
    pub fn run_jit(&mut self, budget: usize) -> u64 {
        jit_set_memory(wasm_bindgen::memory());
        jit_set_table(wasm_bindgen::function_table());
        let mut runner = WasmRunner;
        self.board.machine.run_jit_browser(0, budget, &mut runner);
        self.board.machine.total_insns()
    }

    /// Read X[i] — a sanity check that the kernel actually executed.
    pub fn x(&self, i: usize) -> u64 {
        self.board.machine.cpu.x[i]
    }
}

/// Map a stop reason to the JS status code (0 = ran/idle, 1 = off, 2 = unsupported).
fn status_code(stop: StopReason) -> u32 {
    match stop {
        StopReason::PoweredOff => 1,
        StopReason::Unsupported { .. } => 2,
        _ => 0,
    }
}

/// A full `virt` machine booting a Linux `Image`, driven from JS. The host calls
/// [`run`](Emulator::run) in a loop and drains [`take_uart`](Emulator::take_uart)
/// for console output.
#[wasm_bindgen]
pub struct Emulator {
    board: Board,
}

#[wasm_bindgen]
impl Emulator {
    /// Build an emulator with a JS-backed timer clock.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Emulator {
        Emulator { board: Board::with_clock(Box::new(WasmClock { freq: DEFAULT_FREQ_HZ })) }
    }

    /// Attach the standard virtio devices and boot `image` (a Linux `Image`) with
    /// an optional `initrd` (empty = none) and the `bootargs` kernel command line.
    pub fn boot(&mut self, image: &[u8], initrd: &[u8], bootargs: &str) {
        self.board.attach_input(InputKind::Keyboard);
        self.board.attach_input(InputKind::Mouse);
        self.board.attach_gpu(1024, 768);
        let initrd = (!initrd.is_empty()).then_some(initrd);
        self.board.boot_image(image, initrd, bootargs);
    }

    /// Run up to `budget` guest instructions (interpreter only). Returns a status
    /// code: 0 = ran (or went idle), 1 = powered off, 2 = unsupported instruction.
    pub fn run(&mut self, budget: usize) -> u32 {
        status_code(self.board.machine.run(0, budget))
    }

    /// Like [`run`](Self::run), but JIT-organized: hot blocks are compiled to
    /// WASM and run via the browser `WebAssembly` engine; cold code is
    /// interpreted. Blocks share this module's linear memory.
    pub fn run_jit(&mut self, budget: usize) -> u32 {
        // Hand the JIT engine this module's memory (compiled blocks share it) and
        // its indirect function table (blocks are appended there and called via
        // `call_indirect`).
        jit_set_memory(wasm_bindgen::memory());
        jit_set_table(wasm_bindgen::function_table());
        let mut runner = WasmRunner;
        status_code(self.board.machine.run_jit_browser(0, budget, &mut runner))
    }

    /// Drain and return pending UART (serial console) output.
    pub fn take_uart(&mut self) -> Vec<u8> {
        self.board.uart.take_tx()
    }

    /// Total guest instructions retired since boot.
    pub fn total_insns(&self) -> u64 {
        self.board.machine.total_insns()
    }

    /// Guest instructions retired inside hot compiled blocks (JIT coverage).
    pub fn jit_insns(&self) -> u64 {
        self.board.machine.jit_insns()
    }

    /// Number of compiled-block invocations (for average-block-length stats).
    pub fn jit_calls(&self) -> u64 {
        self.board.machine.jit_calls()
    }

    /// Regions compiled and total blocks across them (avg region size = ratio).
    pub fn jit_regions(&self) -> u64 {
        self.board.machine.jit_regions()
    }
    pub fn jit_region_blocks(&self) -> u64 {
        self.board.machine.jit_region_blocks()
    }
}

/// JIT ↔ interpreter differential crosscheck, run in node.
///
/// The JIT only executes under a `WebAssembly` host, so this is the third leg of
/// the trust chain: Unicorn validates the interpreter (the native fuzz sweep),
/// and this validates the JIT *against that interpreter*. Each kernel is a tight
/// loop run two ways through the same `Machine` — once interpreted, once
/// JIT-organized — and the full architectural state must come out identical.
/// Loops iterate past `JIT_HOTNESS` (256) so the block actually compiles and the
/// hot, compiled path is what we compare (asserted via `jit_insns() > 0`).
///
/// Run with:
/// ```text
/// CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
///   cargo test -p aarch64-web --target wasm32-unknown-unknown
/// ```
#[cfg(test)]
mod jit_crosscheck {
    use super::{jit_set_memory, jit_set_table, WasmClock, WasmRunner};
    use aarch64_platform::{Board, Machine, DEFAULT_FREQ_HZ, RAM_BASE};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_node);

    // --- A handful of instruction encoders (OR-composed so the field math is
    // self-evidently correct), covering the lowered families under test. ---
    fn add_reg(rd: u32, rn: u32, rm: u32) -> u32 {
        0x8B00_0000 | (rm << 16) | (rn << 5) | rd
    }
    fn eor_reg(rd: u32, rn: u32, rm: u32) -> u32 {
        0xCA00_0000 | (rm << 16) | (rn << 5) | rd
    }
    fn orr_reg(rd: u32, rn: u32, rm: u32) -> u32 {
        0xAA00_0000 | (rm << 16) | (rn << 5) | rd
    }
    fn subs_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
        0xF100_0000 | (imm12 << 10) | (rn << 5) | rd
    }
    fn madd(rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
        0x9B00_0000 | (rm << 16) | (ra << 10) | (rn << 5) | rd
    }
    fn udiv(rd: u32, rn: u32, rm: u32) -> u32 {
        0x9AC0_0800 | (rm << 16) | (rn << 5) | rd
    }
    fn csel(rd: u32, rn: u32, rm: u32, cond: u32) -> u32 {
        0x9A80_0000 | (rm << 16) | (cond << 12) | (rn << 5) | rd
    }
    /// ADD Vd.4S, Vn.4S, Vm.4S (SIMD three-same, 32-bit lanes, full 128-bit).
    fn add_v4s(rd: u32, rn: u32, rm: u32) -> u32 {
        0x4EA0_8400 | (rm << 16) | (rn << 5) | rd
    }
    /// CBNZ Xt, #off (off is a signed byte displacement from this instruction).
    fn cbnz(rt: u32, off: i32) -> u32 {
        0xB500_0000 | ((((off >> 2) as u32) & 0x7_ffff) << 5) | rt
    }
    /// B.cond #off (off is a signed byte displacement from this instruction).
    fn bcond(cond: u32, off: i32) -> u32 {
        0x5400_0000 | ((((off >> 2) as u32) & 0x7_ffff) << 5) | cond
    }

    fn image(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A bare machine: code at `RAM_BASE`, PC there, MMU off (EL1h), registers
    /// seeded. Two of these built identically are the two sides of a crosscheck.
    fn machine(words: &[u32], xs: &[(usize, u64)], vs: &[(usize, u128)]) -> Board {
        let clock = Box::new(WasmClock { freq: DEFAULT_FREQ_HZ });
        let mut board = Board::with_ram_and_clock(8 << 20, clock);
        board.machine.bus.ram_mut().write(RAM_BASE, &image(words));
        board.machine.cpu.pc = RAM_BASE;
        for &(i, v) in xs {
            board.machine.cpu.x[i] = v;
        }
        for &(i, v) in vs {
            board.machine.cpu.v[i] = v;
        }
        board
    }

    fn assert_arch_state_eq(interp: &Machine, jit: &Machine) {
        assert_eq!(jit.cpu.x, interp.cpu.x, "X registers");
        assert_eq!(jit.cpu.sp, interp.cpu.sp, "SP");
        assert_eq!(jit.cpu.pc, interp.cpu.pc, "PC");
        assert_eq!(jit.cpu.flags.to_nzcv(), interp.cpu.flags.to_nzcv(), "NZCV");
        assert_eq!(jit.cpu.v, interp.cpu.v, "V registers");
        assert_eq!(jit.cpu.fpcr, interp.cpu.fpcr, "FPCR");
    }

    /// Run `words` to `exit_off` two ways and assert identical architectural
    /// state. `until` (a PC, not a count) makes both stop at the same point
    /// regardless of block granularity. A 2M-instruction cap turns a runaway
    /// kernel into a loud failure rather than a hang.
    fn crosscheck(words: &[u32], xs: &[(usize, u64)], vs: &[(usize, u128)], exit_off: u64) {
        const CAP: usize = 2_000_000;
        let until = RAM_BASE + exit_off;

        let mut interp = machine(words, xs, vs);
        interp.machine.run(until, CAP);

        let mut jit = machine(words, xs, vs);
        jit_set_memory(wasm_bindgen::memory());
        jit_set_table(wasm_bindgen::function_table());
        let mut runner = WasmRunner;
        jit.machine.run_jit_browser(until, CAP, &mut runner);

        assert_eq!(interp.machine.cpu.pc, until, "interpreter reached exit");
        assert_eq!(jit.machine.cpu.pc, until, "JIT reached exit");
        assert!(jit.machine.jit_insns() > 0, "a hot block actually compiled and ran");
        assert_arch_state_eq(&interp.machine, &jit.machine);
    }

    // EOR/ORR pattern. Both x0 (loop counter, sets flags via SUBS) and the
    // accumulators must match — exercises ADD/EOR/ORR shifted-reg, SUBS imm, CBNZ.
    #[wasm_bindgen_test]
    fn arith_logical_flags() {
        let code = [
            add_reg(1, 1, 2),    // x1 += x2
            eor_reg(3, 1, 2),    // x3 = x1 ^ x2
            orr_reg(4, 4, 3),    // x4 |= x3
            subs_imm(0, 0, 1),   // x0--, set NZCV
            cbnz(0, -16),        // loop while x0 != 0
        ];
        crosscheck(&code, &[(0, 1000), (1, 0), (2, 0x0123_4567_89AB_CDEF), (3, 0), (4, 0)], &[], 20);
    }

    // MADD / UDIV / CSEL — the multiply, divide, and conditional-select lowering.
    #[wasm_bindgen_test]
    fn mul_div_csel() {
        let code = [
            madd(1, 1, 2, 3),    // x1 = x1*x2 + x3
            udiv(5, 1, 7),       // x5 = x1 / x7
            subs_imm(0, 0, 1),   // x0--, set flags
            csel(8, 1, 5, 0),    // x8 = (Z) ? x1 : x5   (cond EQ)
            cbnz(0, -16),
        ];
        crosscheck(&code, &[(0, 500), (1, 1), (2, 3), (3, 7), (5, 0), (7, 9), (8, 0)], &[], 20);
    }

    // B.cond terminator (distinct from CBNZ): loop on NE.
    #[wasm_bindgen_test]
    fn bcond_loop() {
        let code = [
            add_reg(1, 1, 2),    // x1 += x2
            subs_imm(0, 0, 1),   // x0--, set flags
            bcond(0b0001, -8),   // B.NE loop
        ];
        crosscheck(&code, &[(0, 800), (1, 0), (2, 5)], &[], 12);
    }

    // Inline SIMD: vector ADD on 4×32-bit lanes accumulated across the loop.
    #[wasm_bindgen_test]
    fn simd_vector_add() {
        let code = [
            add_v4s(0, 0, 1),    // v0.4s += v1.4s
            subs_imm(0, 0, 1),   // x0--, set flags
            cbnz(0, -8),
        ];
        // v1 lanes = [4, 3, 2, 1] (little-endian 32-bit lanes).
        let v1 = 0x0000_0001_0000_0002_0000_0003_0000_0004u128;
        crosscheck(&code, &[(0, 600)], &[(0, 0), (1, v1)], 12);
    }
}
