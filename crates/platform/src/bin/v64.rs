//! `v64` — boot a real arm64 Linux `Image` (with optional initramfs) on the
//! emulated virt board in an SDL window.
//!
//! Usage:
//!   cargo run --release -p aarch64-platform --bin v64 -- <Image> [initramfs.cpio.gz]
//!
//! The native front-end is always SDL: a window shows the virtio-gpu scanout and
//! host keyboard/mouse drive the guest's virtio-input devices. The kernel serial
//! console is logged to stdout (output only — not interactive). The run ends when
//! the guest powers off (PSCI) or the window is closed / Esc is pressed.

use std::process::ExitCode;

use aarch64_platform::{Board, InputKind};

/// Default virtio-gpu scanout size.
const GPU_WIDTH: u32 = 1024;
const GPU_HEIGHT: u32 = 768;

/// Default console: earlycon writes to the PL011 immediately (before the driver
/// binds), so we see boot output from the very first kernel print.
const BOOTARGS: &str = "earlycon=pl011,0x9000000 console=ttyAMA0 rdinit=/init";

/// Instructions to run between UART drains — small enough that output streams,
/// large enough to amortize the loop overhead.
const BATCH: usize = 2_000_000;

const USAGE: &str = "usage: v64 <Image> [initramfs.cpio.gz] [options]\n  \
    --disk <img>     attach a virtio-blk disk (/dev/vda)\n  \
    --append <args>  kernel command line\n  \
    --halt-on-undef  stop on the first unimplemented instruction";

fn main() -> ExitCode {
    // CLI flags (no environment variables — keeps runs reproducible).
    let mut image_path: Option<String> = None;
    let mut initrd_path: Option<String> = None;
    let mut disk_path: Option<String> = None;
    let mut append: Option<String> = None;
    let mut halt_on_undef = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--disk" => disk_path = args.next(),
            "--append" => append = args.next(),
            "--halt-on-undef" => halt_on_undef = true,
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            s if s.starts_with('-') => {
                eprintln!("v64: unknown option {s}\n{USAGE}");
                return ExitCode::FAILURE;
            }
            _ if image_path.is_none() => image_path = Some(a),
            _ if initrd_path.is_none() => initrd_path = Some(a),
            _ => {
                eprintln!("v64: unexpected argument {a}\n{USAGE}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(image_path) = image_path else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };

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
    // By default an unimplemented instruction is delivered to the guest as an
    // Undefined Instruction exception (faithful: SIGILL/panic, machine keeps
    // running). `--halt-on-undef` reverts to halting on the first one.
    if halt_on_undef {
        board.machine.set_undef_to_guest(false);
    }
    // Optional virtio-blk disk (`--disk <image>`) — appears as /dev/vda. Boot it
    // as root with e.g. `--append "root=/dev/vda rw rootfstype=ext4"`. Must be
    // attached before boot_image so the DTB advertises the device.
    if let Some(path) = &disk_path {
        match std::fs::read(path) {
            Ok(img) => {
                eprintln!("v64: virtio-blk /dev/vda <- {path} ({} MiB)", img.len() / (1024 * 1024));
                board.attach_disk(img);
            }
            Err(e) => {
                eprintln!("v64: cannot read disk {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // Always attach the HID + display virtio devices: a keyboard and mouse
    // (/dev/input/eventN) and a 2D GPU (/dev/fb0, /dev/dri/card0).
    let kbd = board.attach_input(InputKind::Keyboard);
    let mouse = board.attach_input(InputKind::Mouse);
    let (gw, gh) = (GPU_WIDTH, GPU_HEIGHT);
    let gpu = board.attach_gpu(gw, gh);
    eprintln!("v64: virtio devices: keyboard, mouse, gpu {gw}x{gh}");

    let bootargs = append.as_deref().unwrap_or(BOOTARGS);
    let layout = board.boot_image(&image, initrd.as_deref(), bootargs);
    eprintln!(
        "v64: kernel@{:#x} dtb@{:#x} initrd={:?} ram={} MiB",
        layout.kernel,
        layout.dtb,
        layout.initrd,
        aarch64_platform::DEFAULT_RAM_SIZE / (1024 * 1024),
    );

    // Native front-end is always SDL: a window + real keyboard/mouse.
    sdl_ui::run(board, gpu, kbd, mouse)
}

/// SDL front-end: a live window showing the virtio-gpu scanout, with host
/// keyboard/mouse fed into the guest's virtio-input devices. Serial output still
/// goes to stdout.
mod sdl_ui {
    use std::io::Write;
    use std::process::ExitCode;
    use std::time::Duration;

    use aarch64_interp::StopReason;
    use aarch64_platform::{Board, VirtioGpu, VirtioInput};
    use sdl2::event::Event;
    use sdl2::keyboard::Scancode;
    use sdl2::mouse::MouseButton;
    use sdl2::pixels::PixelFormatEnum;

    pub fn run(mut board: Board, gpu: VirtioGpu, kbd: VirtioInput, mouse: VirtioInput) -> ExitCode {
        let (w, h) = (super::GPU_WIDTH, super::GPU_HEIGHT);
        let sdl = sdl2::init().expect("sdl init");
        let video = sdl.video().expect("sdl video");
        let window = video.window("v64", w, h).position_centered().build().expect("window");
        let mut canvas = window.into_canvas().build().expect("canvas");
        let tc = canvas.texture_creator();
        // Our framebuffer is B8G8R8A8 in memory == SDL ARGB8888 (LE) as a u32.
        let mut tex =
            tc.create_texture_streaming(PixelFormatEnum::ARGB8888, w, h).expect("texture");
        let mut events = sdl.event_pump().expect("event pump");
        sdl.mouse().set_relative_mouse_mode(true); // deliver relative motion
        let mut out = std::io::stdout();

        'main: loop {
            for ev in events.poll_iter() {
                match ev {
                    Event::Quit { .. } => break 'main,
                    Event::KeyDown { scancode: Some(Scancode::Escape), .. } => break 'main,
                    Event::KeyDown { scancode: Some(sc), repeat: false, .. } => {
                        if let Some(code) = key(sc) {
                            kbd.key(code, true);
                        }
                    }
                    Event::KeyUp { scancode: Some(sc), .. } => {
                        if let Some(code) = key(sc) {
                            kbd.key(code, false);
                        }
                    }
                    Event::MouseMotion { xrel, yrel, .. } => mouse.motion(xrel, yrel, 0),
                    Event::MouseButtonDown { mouse_btn, .. } => {
                        if let Some(b) = btn(mouse_btn) {
                            mouse.key(b, true);
                        }
                    }
                    Event::MouseButtonUp { mouse_btn, .. } => {
                        if let Some(b) = btn(mouse_btn) {
                            mouse.key(b, false);
                        }
                    }
                    Event::MouseWheel { y, .. } => mouse.motion(0, 0, y),
                    _ => {}
                }
            }

            let stop = board.machine.run(0, super::BATCH);
            let tx = board.uart.take_tx();
            if !tx.is_empty() {
                let _ = out.write_all(&tx);
                let _ = out.flush();
            }
            if let Some((fw, fh, px)) = gpu.take_frame() {
                if fw == w && fh == h {
                    let _ = tex.update(None, &px, (w * 4) as usize);
                    let _ = canvas.copy(&tex, None, None);
                    canvas.present();
                }
            }
            match stop {
                StopReason::PoweredOff => {
                    eprintln!("\nv64: machine powered off");
                    break 'main;
                }
                StopReason::Unsupported { pc, word } => {
                    eprintln!("\nv64: unimplemented instruction {word:#010x} at pc {pc:#x}");
                    break 'main;
                }
                StopReason::UntilReached => break 'main,
                StopReason::CountReached => {
                    // Cap idle so the window stays responsive (~60 Hz event pump).
                    if let Some(d) = board.machine.idle_for() {
                        std::thread::sleep(d.min(Duration::from_millis(16)));
                    }
                }
            }
        }
        ExitCode::SUCCESS
    }

    /// SDL mouse button -> Linux `BTN_*` code.
    fn btn(b: MouseButton) -> Option<u16> {
        match b {
            MouseButton::Left => Some(0x110),
            MouseButton::Right => Some(0x111),
            MouseButton::Middle => Some(0x112),
            _ => None,
        }
    }

    /// SDL scancode -> Linux evdev keycode (common keys; enough to drive a UI).
    #[rustfmt::skip]
    fn key(sc: Scancode) -> Option<u16> {
        use Scancode as S;
        Some(match sc {
            S::A=>30, S::B=>48, S::C=>46, S::D=>32, S::E=>18, S::F=>33, S::G=>34, S::H=>35,
            S::I=>23, S::J=>36, S::K=>37, S::L=>38, S::M=>50, S::N=>49, S::O=>24, S::P=>25,
            S::Q=>16, S::R=>19, S::S=>31, S::T=>20, S::U=>22, S::V=>47, S::W=>17, S::X=>45,
            S::Y=>21, S::Z=>44,
            S::Num1=>2, S::Num2=>3, S::Num3=>4, S::Num4=>5, S::Num5=>6,
            S::Num6=>7, S::Num7=>8, S::Num8=>9, S::Num9=>10, S::Num0=>11,
            S::Return=>28, S::Backspace=>14, S::Tab=>15, S::Space=>57,
            S::Minus=>12, S::Equals=>13, S::LeftBracket=>26, S::RightBracket=>27, S::Backslash=>43,
            S::Semicolon=>39, S::Apostrophe=>40, S::Grave=>41, S::Comma=>51, S::Period=>52, S::Slash=>53,
            S::CapsLock=>58,
            S::F1=>59, S::F2=>60, S::F3=>61, S::F4=>62, S::F5=>63, S::F6=>64,
            S::F7=>65, S::F8=>66, S::F9=>67, S::F10=>68, S::F11=>87, S::F12=>88,
            S::Up=>103, S::Left=>105, S::Right=>106, S::Down=>108,
            S::LCtrl=>29, S::LShift=>42, S::LAlt=>56, S::RShift=>54, S::RCtrl=>97, S::RAlt=>100,
            _ => return None,
        })
    }
}
