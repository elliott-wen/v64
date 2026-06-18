//! `diskboot` — headless harness that boots a kernel `Image` with a virtio-blk
//! disk as the ext4 root (`root=/dev/vda`), with no SDL/GPU/input. Used to
//! reproduce and debug the disk-root mount path fast (~1-2s), where the desktop
//! normally boots via initramfs.
//!
//! Usage:
//!   cargo run -p aarch64-platform --bin diskboot -- <Image> <rootfs.ext4>

use std::io::Write;
use std::process::ExitCode;

use aarch64_interp::StopReason;
use aarch64_platform::Board;

const BOOTARGS: &str =
    "earlycon=pl011,0x9000000 console=ttyAMA0 root=/dev/vda rootfstype=ext4 rw rdinit=/sbin/init";
const BATCH: usize = 2_000_000;
/// Safety cap so a livelock/panic loop doesn't run forever.
const MAX_INSNS: u64 = 4_000_000_000;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(image_path), Some(disk_path)) = (args.next(), args.next()) else {
        eprintln!("usage: diskboot <Image> <rootfs.ext4>");
        return ExitCode::FAILURE;
    };

    let image = match std::fs::read(&image_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("diskboot: cannot read kernel {image_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let disk = match std::fs::read(&disk_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("diskboot: cannot read disk {disk_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut board = Board::new();
    board.attach_disk(disk);
    board.attach_rng();
    let layout = board.boot_image(&image, None, BOOTARGS);
    eprintln!("diskboot: kernel@{:#x} dtb@{:#x}", layout.kernel, layout.dtb);

    let mut out = std::io::stdout();
    loop {
        let stop = board.machine.run(0, BATCH);
        let tx = board.uart.take_tx();
        if !tx.is_empty() {
            let _ = out.write_all(&tx);
            let _ = out.flush();
        }
        match stop {
            StopReason::PoweredOff => {
                eprintln!("\ndiskboot: guest powered off");
                return ExitCode::SUCCESS;
            }
            StopReason::Unsupported { pc, word } => {
                eprintln!("\ndiskboot: unimplemented insn {word:#010x} @ {pc:#x}");
                return ExitCode::FAILURE;
            }
            _ => {}
        }
        if board.machine.total_insns() > MAX_INSNS {
            eprintln!("\ndiskboot: instruction cap reached");
            return ExitCode::SUCCESS;
        }
    }
}
