# guest/desktop — a small aarch64 X11 desktop for the v64 emulator (Buildroot)

A lightweight, *bootable* X11 rootfs built with **Buildroot**: Xorg on the
framebuffer (`/dev/fb0`), the matchbox window manager, and an xterm. It exercises
the emulator's virtio-gpu + virtio-input end to end — a "real desktop" without
XFCE's weight (which would be minutes-to-boot and sluggish at ~50–100 MIPS).

Buildroot cross-compiles everything on your x86 host — **no Docker, no qemu, no
arm64 box**. We build *only the rootfs*; the kernel stays the existing
`guest/prebuilt/Image-tiny` (it already has DRM virtio-gpu + fbdev, evdev,
virtio-blk, ext4, VT/fbcon).

## Files here

- `v64_desktop_defconfig` — Buildroot config: aarch64 + musl, ext4 rootfs, no
  kernel, eudev, and the X packages (server + `xf86-video-fbdev` +
  `xf86-input-libinput` + xinit + matchbox + xterm + fonts).
- `overlay/` — dropped into the rootfs as-is:
  - `etc/init.d/S99xdesktop` — starts X at boot.
  - `root/.xinitrc` — the session (matchbox + xterm).
  - `etc/X11/xorg.conf.d/10-fbdev.conf` — force the fbdev video driver.
- `build.sh` — clones Buildroot, applies the defconfig, builds, copies the image.

## Build

```sh
guest/desktop/build.sh
# clones buildroot/ at the repo root, builds, then writes:
#   guest/prebuilt/rootfs.ext4  (+ .gz for the browser)
```

First build is slow (~30–90 min — Buildroot builds a toolchain + Xorg from
source). Needs the usual Buildroot host deps (`gcc make git rsync cpio unzip bc`,
etc.). It's all driven by the committed `defconfig`, so it's reproducible.

## Run

Native (SDL window):
```sh
cargo build --release -p aarch64-platform --bin v64
./target/release/v64 guest/prebuilt/Image-tiny --disk guest/prebuilt/rootfs.ext4 \
    --append 'root=/dev/vda rw rootfstype=ext4 console=ttyAMA0 random.trust_cpu=on'
```

Browser:
```sh
crates/web/build.sh web            # if not already built
python3 -m http.server 8000        # from the repo root
# open http://localhost:8000/crates/web/uitest.html
# pick "desktop (ext4 disk)", click Boot
```
The page fetches `guest/prebuilt/rootfs.ext4.gz` and decompresses it in-browser
(the kernel inflates an initrd but not a raw block device), then boots `/dev/vda`.

## Notes / tweak points (you'll likely iterate on these)

- **Entropy:** `random.trust_cpu=on` (already in the bootargs) keeps Xorg from
  stalling on `getrandom()`. If it still hangs early, add a virtio-rng to the
  kernel or `rng_core.default_quality=1000`.
- **Input not working?** It relies on libinput + eudev autodetecting the virtio
  keyboard/mouse. If X sees no input, check `/var/log/x.log` in the guest; you
  can fall back to the `evdev` driver with explicit `InputDevice` sections
  (`/dev/input/event0` = keyboard, `event1` = mouse, in attach order).
- **Window manager / apps:** edit the `BR2_PACKAGE_*` lines in the defconfig —
  e.g. add another WM or more X apps. (Buildroot has no XFCE package; that's why
  this is matchbox-based.)
- **Boot is slow:** a cold X start is billions of instructions; give it time. The
  framebuffer console means you'll see kernel/boot text on the canvas first, then
  X takes over vt1. JIT (the page default) helps a lot.
- **Rebuilds:** after editing the overlay or defconfig, just re-run `build.sh`
  (Buildroot is incremental; `make` in `buildroot/` rebuilds only what changed).
