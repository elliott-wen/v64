//! `bench` — headless throughput benchmark: boot a kernel and run a fixed
//! instruction budget as fast as possible, reporting MIPS. No SDL, no window,
//! no idle sleeping — pure interpreter speed, for comparing optimizations.
//!
//! Usage:
//!   cargo run --release -p aarch64-platform --bin bench -- <Image> [initramfs] [insns]

use std::process::ExitCode;
use std::time::Instant;

use aarch64_interp::StopReason;
use aarch64_platform::{Board, InputKind};

const BOOTARGS: &str = "earlycon=pl011,0x9000000 console=ttyAMA0 rdinit=/init";
const BATCH: usize = 5_000_000;

fn main() -> ExitCode {
    // `--jit` (anywhere) enables the JIT organizer; remaining args are positional
    // <Image> [initramfs] [insn-budget].
    let use_jit = std::env::args().any(|a| a == "--jit");
    let mut args = std::env::args().skip(1).filter(|a| a != "--jit");
    let Some(image_path) = args.next() else {
        eprintln!("usage: bench [--jit] <Image> [initramfs] [insn-budget]");
        return ExitCode::FAILURE;
    };
    let initrd_path = args.next();
    let budget: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200_000_000);

    let image = std::fs::read(&image_path).expect("read kernel");
    let initrd = initrd_path.as_ref().map(|p| std::fs::read(p).expect("read initramfs"));

    let mut board = Board::new();
    board.attach_input(InputKind::Keyboard);
    board.attach_input(InputKind::Mouse);
    board.attach_gpu(1024, 768);
    board.boot_image(&image, initrd.as_deref(), BOOTARGS);
    if use_jit {
        board.machine.enable_jit();
    }

    eprintln!("bench: running ~{budget} instructions (jit={use_jit})...");
    let start = Instant::now();
    let mut last = start;
    let mut last_insns = 0u64;
    loop {
        let stop = board.machine.run(0, BATCH);
        // Drain (and discard) serial output so the UART FIFO doesn't back up.
        let _ = board.uart.take_tx();

        let total = board.machine.total_insns();
        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64();
        if dt >= 1.0 {
            let mips = (total - last_insns) as f64 / dt / 1e6;
            eprintln!("bench: {total} insns | {mips:.1} MIPS");
            last = now;
            last_insns = total;
        }

        match stop {
            StopReason::PoweredOff => break,
            StopReason::Unsupported { pc, word } => {
                eprintln!("bench: unsupported {word:#010x} @ {pc:#x}");
                break;
            }
            _ => {}
        }
        if total >= budget {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total = board.machine.total_insns();
    eprintln!(
        "bench: {total} insns in {elapsed:.2}s = {:.2} MIPS average",
        total as f64 / elapsed / 1e6
    );
    ExitCode::SUCCESS
}
