# A minimal aarch64 X11 desktop rootfs for the v64 emulator, built from Alpine's
# prebuilt arm64 packages (fast — no compiling). Turned into a bootable ext4 by
# mkrootfs.sh. Lightweight on purpose: Xorg on the fbdev driver (/dev/fb0, the
# same path uitest uses), a tiny window manager, and a terminal.
#
# Build needs arm64 emulation on the host:
#   docker run --privileged --rm tonistiigi/binfmt --install arm64
FROM --platform=linux/arm64 alpine:3.20

# Xorg + the framebuffer video driver + libinput (for the virtio keyboard/mouse),
# xinit to launch a session, a tiny WM (jwm), a terminal, and fonts. mesa lets
# simple GL apps fall back to software. Add xeyes/xclock here if you want more
# moving things to watch.
RUN apk add --no-cache \
        xorg-server \
        xf86-video-fbdev \
        xf86-input-libinput \
        xinit \
        jwm \
        xterm \
        xsetroot \
        font-misc-misc \
        font-cursor-misc \
        mesa-dri-gallium \
        busybox

# Boot init: PID 1 is a plain script (devtmpfs auto-populates /dev/fb0 and
# /dev/input, so no udev is needed). It mounts the pseudo-filesystems and starts X.
COPY init.sh /sbin/init
COPY xinitrc /root/.xinitrc
COPY 10-fbdev.conf /etc/X11/xorg.conf.d/10-fbdev.conf
RUN chmod +x /sbin/init /root/.xinitrc
