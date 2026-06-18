#!/usr/bin/env bash
# Build a small aarch64 X11 desktop ext4 rootfs for the v64 emulator with
# Buildroot. Clones Buildroot into the repo root (sibling of guest/), applies the
# defconfig here, builds, and copies the image to guest/prebuilt/.
#
# First build is slow (~30-90 min: Buildroot builds a cross-toolchain + Xorg from
# source). No Docker/qemu needed — it cross-compiles on this host. Prereqs are
# the usual Buildroot host tools (gcc, make, git, rsync, cpio, unzip, bc, ...).
#
# Output: guest/prebuilt/rootfs.ext4 (+ .gz for the browser).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

BR_TAG="${BR_TAG:-2024.02.9}"   # a stable Buildroot release

if [ ! -d buildroot ]; then
  echo "==> cloning Buildroot $BR_TAG"
  git clone --depth 1 --branch "$BR_TAG" https://github.com/buildroot/buildroot.git
fi

echo "==> applying defconfig"
cp guest/desktop/v64_desktop_defconfig buildroot/configs/v64_desktop_defconfig
make -C buildroot v64_desktop_defconfig

echo "==> building (this takes a while the first time)"
make -C buildroot -j"$(nproc)"

echo "==> copying image to guest/prebuilt/"
mkdir -p guest/prebuilt
cp buildroot/output/images/rootfs.ext2 guest/prebuilt/rootfs.ext4
gzip -9 -f -k guest/prebuilt/rootfs.ext4
ls -lh guest/prebuilt/rootfs.ext4 guest/prebuilt/rootfs.ext4.gz
echo "done."
echo "  Native:  ./target/release/v64 guest/prebuilt/Image-tiny --disk guest/prebuilt/rootfs.ext4 \\"
echo "               --append 'root=/dev/vda rw rootfstype=ext4 console=ttyAMA0 random.trust_cpu=on'"
echo "  Browser: open crates/web/uitest.html, pick 'desktop (ext4 disk)', Boot"
