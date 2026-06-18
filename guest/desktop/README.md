# guest/desktop — a small aarch64 X11 desktop for the v64 emulator

A lightweight, *bootable* X11 rootfs built from Alpine's prebuilt arm64 packages:
Xorg on the framebuffer (`/dev/fb0`), a tiny window manager (JWM), and an xterm.
It exercises the emulator's virtio-gpu + virtio-input end to end — a "real
desktop" without XFCE's weight (which would be minutes-to-boot and sluggish at
the emulator's ~50–100 MIPS).

The kernel is the existing `guest/prebuilt/Image-tiny` (it already has DRM
virtio-gpu + fbdev, evdev, virtio-blk, ext4). We build **only the rootfs**.

## Prerequisites (host)

- **Docker** with arm64 emulation. One-time:
  ```sh
  docker run --privileged --rm tonistiigi/binfmt --install arm64
  ```
- **e2fsprogs** (`mke2fs`), `gzip`, `tar`. No root or loopback mount is needed —
  `mke2fs -d` populates the image from a directory directly.

## Build

```sh
guest/desktop/mkrootfs.sh          # SIZE=512M by default
# -> guest/prebuilt/rootfs.ext4 and rootfs.ext4.gz
```

This builds the Dockerfile (Alpine arm64 + the X stack), exports the container
filesystem, and packs it into an ext4 image. The rootfs is ~150–250 MB; the
gzip (for the browser) is much smaller since the free space compresses away.

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
# pick "desktop (ext4 disk)" in the image dropdown, click Boot
```
The page fetches `guest/prebuilt/rootfs.ext4.gz` and decompresses it in-browser
(the kernel doesn't inflate a block device), then boots it as `/dev/vda`.

## Notes / tuning

- **Entropy:** `random.trust_cpu=on` keeps Xorg from stalling on `getrandom()`.
  If it still hangs early, enable a virtio-rng in the kernel or add
  `rng_core.default_quality=1000`.
- **Window manager / apps:** edit the Dockerfile's `apk add` line — e.g. add
  `xeyes xclock` for more on-screen motion, or swap `jwm` for `icewm`/`fluxbox`.
- **Driver:** X uses the `fbdev` driver on `/dev/fb0` (see `10-fbdev.conf`), the
  same path `uitest` uses. The KMS/`modesetting` driver also works with
  virtio-gpu but isn't needed for a 2D test.
- **XFCE:** not packaged here (Alpine has it via `apk add xfce4`, but it's heavy
  — expect minutes to boot and laggy interaction). Try it only once the
  lightweight stack is proven.
- **Boot is slow:** a cold X start is billions of instructions; give it time.
  JIT (the default in the page) helps a lot.
