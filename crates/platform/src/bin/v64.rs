//! `v64` — boot a real arm64 Linux `Image` (with optional initramfs) on the
//! emulated virt board and stream the serial console to stdout.
//!
//! Usage:
//!   cargo run -p aarch64-platform --bin v64 -- <Image> [initramfs.cpio.gz]
//!
//! The kernel runs until it powers off (PSCI), hits an instruction the
//! interpreter doesn't implement yet (reported, for bring-up), or you Ctrl-C.

use std::io::Write;
use std::process::ExitCode;

use aarch64_interp::StopReason;
use aarch64_platform::Board;

/// Default console: earlycon writes to the PL011 immediately (before the driver
/// binds), so we see boot output from the very first kernel print.
const BOOTARGS: &str = "earlycon=pl011,0x9000000 console=ttyAMA0 rdinit=/init";

/// Instructions to run between UART drains — small enough that output streams,
/// large enough to amortize the loop overhead.
const BATCH: usize = 2_000_000;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(image_path) = args.next() else {
        eprintln!("usage: v64 <Image> [initramfs.cpio.gz]");
        return ExitCode::FAILURE;
    };
    let initrd_path = args.next();

    let image = match std::fs::read(&image_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("v64: cannot read kernel {image_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let initrd = match &initrd_path {
        Some(p) => match std::fs::read(p) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("v64: cannot read initramfs {p}: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    let mut board = Board::new();
    let layout = board.boot_image(&image, initrd.as_deref(), BOOTARGS);
    eprintln!(
        "v64: kernel@{:#x} dtb@{:#x} initrd={:?} ram={} MiB",
        layout.kernel,
        layout.dtb,
        layout.initrd,
        aarch64_platform::DEFAULT_RAM_SIZE / (1024 * 1024),
    );
    eprintln!("v64: ---- console ----");

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    loop {
        let stop = board.machine.run(0, BATCH);
        let tx = board.uart.take_tx();
        if !tx.is_empty() {
            let _ = out.write_all(&tx);
            let _ = out.flush();
        }
        match stop {
            StopReason::CountReached => {} // keep running
            StopReason::PoweredOff => {
                eprintln!("\nv64: machine powered off");
                return ExitCode::SUCCESS;
            }
            StopReason::Unsupported { pc, word } => {
                eprintln!("\nv64: unimplemented instruction {word:#010x} at pc {pc:#x}");
                return ExitCode::FAILURE;
            }
            StopReason::UntilReached => return ExitCode::SUCCESS,
        }
    }
}
