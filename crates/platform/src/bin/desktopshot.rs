//! `desktopshot` — headless desktop smoke test. Boots a kernel `Image` with the
//! ext4 disk root plus virtio-gpu/input/rng (no SDL), runs until X has painted,
//! and writes the latest composed scanout to a PPM so the framebuffer pipeline
//! can be checked without a display. Reports the dominant colour so a script can
//! tell "steelblue desktop" from "blank console".
//!
//! Usage:
//!   cargo run --release -p aarch64-platform --bin desktopshot -- <Image> <rootfs.ext4> [out.ppm] [secs]

use std::collections::HashMap;
use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use aarch64_platform::{Board, InputKind};

const BOOTARGS: &str =
    "earlycon=pl011,0x9000000 console=ttyAMA0 root=/dev/vda rootfstype=ext4 rw";
const BATCH: usize = 2_000_000;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(image_path), Some(disk_path)) = (args.next(), args.next()) else {
        eprintln!("usage: desktopshot <Image> <rootfs.ext4> [out.ppm] [secs]");
        return ExitCode::FAILURE;
    };
    let out_path = args.next().unwrap_or_else(|| "/tmp/desktop.ppm".to_string());
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(90);

    let image = std::fs::read(&image_path).expect("read kernel Image");
    let disk = std::fs::read(&disk_path).expect("read rootfs.ext4");

    let mut board = Board::new();
    board.attach_disk(disk);
    let _kbd = board.attach_input(InputKind::Keyboard);
    let _mouse = board.attach_input(InputKind::Mouse);
    let gpu = board.attach_gpu(1024, 768);
    board.attach_rng();
    board.boot_image(&image, None, BOOTARGS);

    let mut out = std::io::stdout();
    let mut last_frame: Option<(u32, u32, Vec<u8>)> = None;
    let mut flushes = 0u64;
    let mut seen_steel = false;
    let start = Instant::now();
    while start.elapsed().as_secs() < secs {
        board.machine.run(0, BATCH);
        let tx = board.uart.take_tx();
        if !tx.is_empty() {
            let _ = out.write_all(&tx);
            let _ = out.flush();
        }
        if let Some(frame) = gpu.take_frame() {
            flushes += 1;
            // Did this frame contain a meaningful amount of the steelblue root?
            let steel = frame.2.chunks_exact(4).filter(|p| p[2] == 70 && p[1] == 130 && p[0] == 180).count();
            if steel > 1000 && !seen_steel {
                seen_steel = true;
                eprintln!("\ndesktopshot: steelblue root appeared at flush {flushes} ({steel} px)");
                write_ppm("/tmp/desktop-steel.ppm", frame.0, frame.1, &frame.2);
            }
            last_frame = Some(frame);
        }
    }

    eprintln!("\ndesktopshot: {flushes} scanout flush(es)");
    let Some((w, h, px)) = last_frame else {
        eprintln!("desktopshot: NO frame was ever flushed — X/GPU never presented");
        return ExitCode::FAILURE;
    };
    report_dominant(&px);
    write_ppm(&out_path, w, h, &px);
    eprintln!("desktopshot: wrote {out_path} ({w}x{h})");
    ExitCode::SUCCESS
}

/// Tally the most common BGRA pixels so a human/script can see what's on screen.
fn report_dominant(px: &[u8]) {
    let mut counts: HashMap<[u8; 3], u64> = HashMap::new();
    for p in px.chunks_exact(4) {
        *counts.entry([p[2], p[1], p[0]]).or_default() += 1; // store as RGB
    }
    let mut top: Vec<_> = counts.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    let total = (px.len() / 4) as f64;
    eprintln!("desktopshot: top colours (RGB):");
    for ([r, g, b], n) in top.into_iter().take(4) {
        let pct = 100.0 * n as f64 / total;
        let tag = if [r, g, b] == [70, 130, 180] { "  <- steelblue root!" } else { "" };
        eprintln!("  #{r:02x}{g:02x}{b:02x}  {pct:5.1}%{tag}");
    }
}

fn write_ppm(path: &str, w: u32, h: u32, px: &[u8]) {
    let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
    for p in px.chunks_exact(4) {
        buf.push(p[2]); // R
        buf.push(p[1]); // G
        buf.push(p[0]); // B
    }
    let _ = std::fs::write(path, buf);
}
