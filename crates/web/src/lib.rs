//! wasm-bindgen entry points for running the AArch64 emulator in a browser or
//! node. Two layers:
//!
//! - [`run_code`] — a one-shot interpreter smoke test (run a code blob, read X0).
//! - [`Emulator`] — the full system Machine (CPU + bus + virtio devices) booting
//!   a real Linux `Image`, with a JS-backed [`Clock`] (the one piece native
//!   `HostClock` can't provide under wasm, since `Instant` is unavailable).

use aarch64_cpu_state::CpuState;
use aarch64_interp::{run, Memory, StopReason};
use aarch64_platform::{
    Board, BlockRunner, Clock, InputKind, VirtioGpu, VirtioInput, DEFAULT_FREQ_HZ, RAM_BASE,
};
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
    // Device handles retained so the page can read the framebuffer and inject
    // input (the keyboard and mouse virtio-input devices, and the GPU scanout).
    gpu: Option<VirtioGpu>,
    kbd: Option<VirtioInput>,
    mouse: Option<VirtioInput>,
    fb_w: u32,
    fb_h: u32,
}

#[wasm_bindgen]
impl Emulator {
    /// Build an emulator with a JS-backed timer clock.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Emulator {
        Emulator {
            board: Board::with_clock(Box::new(WasmClock { freq: DEFAULT_FREQ_HZ })),
            gpu: None,
            kbd: None,
            mouse: None,
            fb_w: 0,
            fb_h: 0,
        }
    }

    /// Attach the standard virtio devices and boot `image` (a Linux `Image`) with
    /// an optional `initrd` (empty = none) and the `bootargs` kernel command line.
    /// The keyboard/mouse/GPU handles are kept for [`key`](Self::key) /
    /// [`mouse_motion`](Self::mouse_motion) / [`take_frame`](Self::take_frame).
    pub fn boot(&mut self, image: &[u8], initrd: &[u8], bootargs: &str) {
        self.kbd = Some(self.board.attach_input(InputKind::Keyboard));
        self.mouse = Some(self.board.attach_input(InputKind::Mouse));
        self.gpu = Some(self.board.attach_gpu(1024, 768));
        let initrd = (!initrd.is_empty()).then_some(initrd);
        self.board.boot_image(image, initrd, bootargs);
    }

    /// The latest virtio-gpu scanout as B8G8R8A8 bytes, or an empty buffer if the
    /// guest hasn't drawn a new frame since the last call. On a new frame the
    /// dimensions are cached for [`fb_width`](Self::fb_width)/[`fb_height`](Self::fb_height).
    pub fn take_frame(&mut self) -> Vec<u8> {
        if let Some(gpu) = &self.gpu {
            if let Some((w, h, px)) = gpu.take_frame() {
                self.fb_w = w;
                self.fb_h = h;
                return px;
            }
        }
        Vec::new()
    }

    /// Width/height of the most recent frame from [`take_frame`](Self::take_frame).
    pub fn fb_width(&self) -> u32 {
        self.fb_w
    }
    pub fn fb_height(&self) -> u32 {
        self.fb_h
    }

    /// Inject a guest keyboard event: `code` is a Linux evdev keycode (KEY_*),
    /// `down` is press vs release.
    pub fn key(&self, code: u16, down: bool) {
        if let Some(kbd) = &self.kbd {
            kbd.key(code, down);
        }
    }

    /// Inject relative pointer motion (and/or `wheel` ticks) into the mouse.
    pub fn mouse_motion(&self, dx: i32, dy: i32, wheel: i32) {
        if let Some(mouse) = &self.mouse {
            mouse.motion(dx, dy, wheel);
        }
    }

    /// Inject a mouse button event: `code` is a Linux `BTN_*` code.
    pub fn mouse_button(&self, code: u16, down: bool) {
        if let Some(mouse) = &self.mouse {
            mouse.key(code, down);
        }
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
