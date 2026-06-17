# guest/ — reproducing the test kernel & initramfs images

These are the **sources and config** for the guest-side artifacts v64 boots
(kernel, initramfs images, the UI/loopback helpers). The large *binaries* and
*toolchains* are intentionally not committed — this README is how to rebuild
them on a fresh machine.

```
guest/
  kernel.config          Linux 6.6.x config: a broad "VM" kernel (proc/sysfs,
                         file-locking, futex, namespaces, cgroups, seccomp,
                         io_uring, virtio mmio/blk/net/gpu/input, DRM + fbdev,
                         ext4/overlay, ...). Drivers for real hardware are off.
  src/
    uitest.c             GUI smoke test: mmaps /dev/fb0, opens /dev/input/event*,
                         draws bg+HUD+crosshair, stamps a square per key, logs
                         every input event to stdout. ESC quits.
    init.c               Freestanding PID1 (raw aarch64 syscalls, no libc) for a
                         minimal "userspace alive" initramfs.
    ifup-lo.c            Brings up loopback (127.0.0.1) — busybox here has no
                         ifconfig/ip and `lo` starts down.
  initramfs/
    uitest.init          /init for the uitest image (mounts, runs /bin/uitest).
    libctest.init        /init that runs the musl libc-test suite + prints results.
    busybox.init         /init for an interactive busybox shell.
```

## Prerequisites (host)
- **Kernel cross toolchain:** `aarch64-linux-gnu-gcc` (any recent; we used 12.1).
- **Userspace cross toolchain (musl):** `aarch64-linux-musl-cross` from
  <https://musl.cc/aarch64-linux-musl-cross.tgz> (ships musl 1.2.2).
- `cpio`, `gzip`, and for v64's window: **libSDL2** (`libsdl2-dev`).

Set `CROSS=aarch64-linux-gnu-` and put both toolchains on `PATH`.

## 1. Kernel
```sh
curl -LO https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.58.tar.xz
tar xf linux-6.6.58.tar.xz && cd linux-6.6.58
cp ../guest/kernel.config .config
make ARCH=arm64 CROSS_COMPILE=$CROSS olddefconfig
make ARCH=arm64 CROSS_COMPILE=$CROSS -j"$(nproc)" Image
cp arch/arm64/boot/Image ../Image-tiny     # ~6 MiB
```

## 2. Userspace binaries (static, musl)
```sh
MUSL=aarch64-linux-musl-gcc
$MUSL -static -no-pie -O2 -o uitest  guest/src/uitest.c  && aarch64-linux-musl-strip uitest
$MUSL -static -no-pie -Os -o ifup-lo guest/src/ifup-lo.c
# Freestanding PID1 for the tiny image (no libc):
$CROSS-gcc -nostdlib -static -O2 -o tiny-init guest/src/init.c
```
**busybox** (for the busybox/libctest/uitest images):
```sh
curl -LO https://busybox.net/downloads/busybox-1.36.1.tar.bz2
tar xf busybox-1.36.1.tar.bz2 && cd busybox-1.36.1
make defconfig
# set CONFIG_STATIC=y and CC=aarch64-linux-musl-gcc, build -no-pie static
make CROSS_COMPILE=aarch64-linux-musl- CONFIG_STATIC=y -j"$(nproc)"
```

## 3. Assemble an initramfs (example: uitest)
```sh
mkdir -p root/bin root/dev root/proc root/sys
cp busybox-1.36.1/busybox root/bin/
cp uitest root/bin/uitest
cp guest/initramfs/uitest.init root/init && chmod +x root/init
( cd root && find . | LC_ALL=C sort | cpio -o -H newc | gzip -9 ) > uitest.cpio.gz
```
Same pattern for `busybox.init` (interactive shell) and `libctest.init` (build
the musl `libc-test` suite static-no-pie, drop the `*.exe` binaries into
`root/tests/{functional,regression}/`, add `ifup-lo` to `root/bin`).

> The libc-test images want a **newer musl** (1.2.5+/git) than musl.cc's 1.2.2
> to pass strptime/mntent/strtold — build musl from source into the toolchain
> sysroot and rebuild the suite. See the project memory note for the exact steps.

## 4. Run
```sh
cargo build --release -p aarch64-platform --bin v64
# GUI (SDL window, real keyboard/mouse) — needs a display + libSDL2:
./target/release/v64 Image-tiny uitest.cpio.gz
# Headless serial console (busybox, libc-test):
./target/release/v64 --headless Image-tiny busybox.cpio.gz
# Boot a real disk as root:
./target/release/v64 --headless Image-tiny --disk rootfs.ext4 \
    --append "root=/dev/vda rw rootfstype=ext4 console=ttyAMA0"
```

## Prebuilt binaries (temporary)
`prebuilt/` holds ready-to-run artifacts so you can test immediately on another
machine without rebuilding — **temporary; not meant to live in git long-term**
(rebuild from the sources above and drop them once set up):
- `prebuilt/Image-tiny` — the kernel (built from `kernel.config`).
- `prebuilt/uitest.cpio.gz` — the uitest initramfs.

```sh
cargo build --release -p aarch64-platform --bin v64
./target/release/v64 guest/prebuilt/Image-tiny guest/prebuilt/uitest.cpio.gz
```

## Optional: differential-test oracle
`qemu-aarch64-static` (run the same static binaries on real QEMU as a reference):
```sh
curl -LO https://github.com/multiarch/qemu-user-static/releases/download/v7.2.0-1/qemu-aarch64-static
chmod +x qemu-aarch64-static && ./qemu-aarch64-static ./some-test.exe
```
