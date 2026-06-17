//! wasm-bindgen entry points for running the AArch64 emulator in a browser or
//! node. Two layers:
//!
//! - [`run_code`] — a one-shot interpreter smoke test (run a code blob, read X0).
//! - [`Emulator`] — the full system Machine (CPU + bus + virtio devices) booting
//!   a real Linux `Image`, with a JS-backed [`Clock`] (the one piece native
//!   `HostClock` can't provide under wasm, since `Instant` is unavailable).

use aarch64_cpu_state::CpuState;
use aarch64_interp::{run, Memory, StopReason};
use aarch64_platform::{Board, Clock, InputKind, DEFAULT_FREQ_HZ};
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

    /// Run up to `budget` guest instructions. Returns a status code:
    /// 0 = ran (or went idle), 1 = powered off, 2 = unsupported instruction.
    pub fn run(&mut self, budget: usize) -> u32 {
        match self.board.machine.run(0, budget) {
            StopReason::PoweredOff => 1,
            StopReason::Unsupported { .. } => 2,
            _ => 0,
        }
    }

    /// Drain and return pending UART (serial console) output.
    pub fn take_uart(&mut self) -> Vec<u8> {
        self.board.uart.take_tx()
    }

    /// Total guest instructions retired since boot.
    pub fn total_insns(&self) -> u64 {
        self.board.machine.total_insns()
    }
}
