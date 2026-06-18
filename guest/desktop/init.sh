#!/bin/sh
# PID 1 for the desktop rootfs: mount the basics and start X on the framebuffer.
# devtmpfs is auto-mounted by the kernel (CONFIG_DEVTMPFS_MOUNT=y), so /dev/fb0
# and /dev/input/event* already exist — no udev needed.
mount -t proc     proc     /proc
mount -t sysfs    sysfs    /sys
mount -t tmpfs    tmpfs    /tmp
mkdir -p /dev/pts /run
mount -t devpts   devpts   /dev/pts 2>/dev/null

export HOME=/root USER=root XDG_RUNTIME_DIR=/run

echo "=== v64 desktop: starting X on /dev/fb0 ==="
# Log X output to the serial console; -nolisten tcp since there's no network need.
startx -- -nolisten tcp vt1

echo "=== X exited — powering off ==="
poweroff -f
