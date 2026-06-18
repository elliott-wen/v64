#!/usr/bin/env bash
# Build a bootable aarch64 X11 desktop ext4 image for the v64 emulator from the
# Dockerfile here: build the arm64 rootfs, export its filesystem, and pack it
# into an ext4 image (no root / no mount needed, via `mke2fs -d`).
#
# Prereqs on the host:
#   - Docker with arm64 emulation:
#       docker run --privileged --rm tonistiigi/binfmt --install arm64
#   - e2fsprogs (provides mke2fs), gzip, tar
#
# Output: guest/prebuilt/rootfs.ext4 (+ .gz for the browser). Boot it with
#   root=/dev/vda rw rootfstype=ext4 console=ttyAMA0 random.trust_cpu=on
set -euo pipefail
cd "$(dirname "$0")"

IMG=../prebuilt/rootfs.ext4
SIZE=${SIZE:-512M}          # ext4 image size (the rootfs uses ~150-250MB)
TAG=v64-desktop

echo "==> building arm64 rootfs image ($TAG)"
docker build --platform linux/arm64 -t "$TAG" .

echo "==> exporting container filesystem"
cid=$(docker create --platform linux/arm64 "$TAG")
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
rm -rf rootdir && mkdir rootdir
docker export "$cid" | tar -x -C rootdir

echo "==> packing into ext4 ($SIZE)"
mkdir -p ../prebuilt
rm -f "$IMG" "$IMG.gz"
# -d populates the image from a directory with no mount and no privileges.
mke2fs -q -t ext4 -d rootdir -L v64root "$IMG" "$SIZE"
rm -rf rootdir

echo "==> gzipping for the browser (the kernel does not decompress a disk)"
gzip -9 -k "$IMG"
ls -lh "$IMG" "$IMG.gz"
echo "done. Native:  v64 --headless ../prebuilt/Image-tiny --disk $IMG \\"
echo "                   --append 'root=/dev/vda rw rootfstype=ext4 console=ttyAMA0 random.trust_cpu=on'"
echo "      Browser: copy rootfs.ext4.gz where uitest.html can fetch it (see guest/desktop/README.md)"
