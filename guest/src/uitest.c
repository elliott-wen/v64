/*
 * uitest — minimal framebuffer + input test for v64 (virtio-gpu / virtio-input).
 *
 * - mmaps /dev/fb0 (works with virtio-gpu via DRM_FBDEV_EMULATION, or simplefb)
 * - opens every /dev/input/event* (virtio keyboard / mouse / tablet via evdev)
 * - draws a background, a HUD bar, and a cursor that follows the pointer;
 *   each key press stamps a colored square at the cursor.
 * - logs every input event to stdout in a parseable form so a headless,
 *   scripted harness can assert on the console; redraws reflect the same state.
 * - exits (clean) on ESC so /init can PSCI-poweroff.
 *
 * Build (static, no-pie) with the musl cross toolchain:
 *   aarch64-linux-musl-gcc -static -no-pie -O2 -o uitest uitest.c
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <poll.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <linux/input.h>

static void logs(const char *s) { write(1, s, strlen(s)); }
static void logn(const char *pfx, long v) {
    char b[64]; size_t i = 0; for (const char *p = pfx; *p; p++) b[i++] = *p;
    if (v < 0) { b[i++] = '-'; v = -v; }
    char t[24]; int j = 0; if (v == 0) t[j++] = '0';
    while (v) { t[j++] = '0' + (v % 10); v /= 10; }
    while (j) b[i++] = t[--j];
    b[i++] = '\n'; write(1, b, i);
}

/* ---- framebuffer ---- */
struct fb {
    uint8_t *back;      /* shadow we draw into */
    uint8_t *mem;       /* mmap of /dev/fb0   */
    uint32_t w, h, bpp, stride, size;
} FB;

static int fb_open(void) {
    int fd = open("/dev/fb0", O_RDWR);
    if (fd < 0) { logs("uitest: no /dev/fb0 (need DRM_FBDEV_EMULATION/FB_SIMPLE)\n"); return -1; }
    struct fb_var_screeninfo var; struct fb_fix_screeninfo fix;
    if (ioctl(fd, FBIOGET_VSCREENINFO, &var) || ioctl(fd, FBIOGET_FSCREENINFO, &fix)) {
        logs("uitest: FBIOGET ioctl failed\n"); return -1;
    }
    FB.w = var.xres; FB.h = var.yres; FB.bpp = var.bits_per_pixel;
    FB.stride = fix.line_length; FB.size = FB.stride * FB.h;
    FB.mem = mmap(0, FB.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (FB.mem == MAP_FAILED) { logs("uitest: fb mmap failed\n"); return -1; }
    FB.back = malloc(FB.size); if (!FB.back) return -1;
    logn("uitest: fb width=", FB.w); logn("uitest: fb height=", FB.h);
    logn("uitest: fb bpp=", FB.bpp); logn("uitest: fb stride=", FB.stride);
    return 0;
}

static inline uint32_t pack(uint8_t r, uint8_t g, uint8_t b) {
    /* assume little-endian XRGB8888 (virtio-gpu / simplefb default) or RGB565 */
    return (FB.bpp == 16)
        ? (uint32_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3))
        : (uint32_t)((r << 16) | (g << 8) | b);
}
static void px(int x, int y, uint32_t c) {
    if ((unsigned)x >= FB.w || (unsigned)y >= FB.h) return;
    uint8_t *p = FB.back + (size_t)y * FB.stride + (size_t)x * (FB.bpp / 8);
    if (FB.bpp == 16) { *(uint16_t *)p = (uint16_t)c; } else { *(uint32_t *)p = c; }
}
static void fill_rect(int x, int y, int w, int h, uint32_t c) {
    for (int j = 0; j < h; j++) for (int i = 0; i < w; i++) px(x + i, y + j, c);
}
static void flush(void) { memcpy(FB.mem, FB.back, FB.size); }

/* ---- input ---- */
#define MAXFD 16
static int infd[MAXFD], nfd;
/* tablet (EV_ABS) scaling */
static int abs_x_max = 0, abs_y_max = 0;

static void inputs_open(void) {
    DIR *d = opendir("/dev/input"); if (!d) { logs("uitest: no /dev/input\n"); return; }
    struct dirent *e;
    while ((e = readdir(d)) && nfd < MAXFD) {
        if (strncmp(e->d_name, "event", 5)) continue;
        char path[64]; strcpy(path, "/dev/input/"); strcat(path, e->d_name);
        int fd = open(path, O_RDONLY | O_NONBLOCK); if (fd < 0) continue;
        char name[64] = {0}; ioctl(fd, EVIOCGNAME(sizeof name), name);
        logs("uitest: opened "); logs(path); logs(" = "); logs(name); logs("\n");
        struct input_absinfo ai;
        if (ioctl(fd, EVIOCGABS(ABS_X), &ai) == 0 && ai.maximum > 0) abs_x_max = ai.maximum;
        if (ioctl(fd, EVIOCGABS(ABS_Y), &ai) == 0 && ai.maximum > 0) abs_y_max = ai.maximum;
        infd[nfd++] = fd;
    }
    closedir(d);
    if (!nfd) logs("uitest: WARNING no input devices found\n");
}

int main(void) {
    logs("uitest: starting\n");
    if (fb_open() < 0) logs("uitest: continuing without display (input-only)\n");
    inputs_open();

    int W = FB.w ? FB.w : 640, H = FB.h ? FB.h : 480;
    int cx = W / 2, cy = H / 2;          /* cursor */
    const uint32_t palette[] = {
        0xE0392B, 0x27AE60, 0x2980D9, 0xF1C40F, 0x8E44AD, 0xE67E22, 0x1ABC9C };
    int pal = 0;
    long nstamp = 0, nkey = 0, nmotion = 0;

    /* initial frame */
    if (FB.mem) {
        fill_rect(0, 0, W, H, pack(20, 22, 28));                 /* bg     */
        fill_rect(0, 0, W, 24, pack(40, 44, 56));                /* HUD    */
        fill_rect(2, 2, W - 4, 20, pack(60, 66, 84));
        flush();
        logs("uitest: drew initial frame\n");
    }

    int running = 1;
    while (running) {
        struct pollfd pfd[MAXFD];
        for (int i = 0; i < nfd; i++) { pfd[i].fd = infd[i]; pfd[i].events = POLLIN; }
        int r = poll(pfd, nfd, -1);
        if (r < 0) { if (errno == EINTR) continue; break; }

        int dirty = 0;
        for (int i = 0; i < nfd; i++) {
            if (!(pfd[i].revents & POLLIN)) continue;
            struct input_event ev;
            while (read(infd[i], &ev, sizeof ev) == (ssize_t)sizeof ev) {
                if (ev.type == EV_REL) {
                    if (ev.code == REL_X) { cx += ev.value; nmotion++; }
                    else if (ev.code == REL_Y) { cy += ev.value; nmotion++; }
                    logn("EV rel code=", ev.code); logn("EV rel val=", ev.value);
                    dirty = 1;
                } else if (ev.type == EV_ABS) {
                    if (ev.code == ABS_X && abs_x_max) { cx = (int)((long)ev.value * W / abs_x_max); nmotion++; }
                    else if (ev.code == ABS_Y && abs_y_max) { cy = (int)((long)ev.value * H / abs_y_max); nmotion++; }
                    logn("EV abs code=", ev.code); logn("EV abs val=", ev.value);
                    dirty = 1;
                } else if (ev.type == EV_KEY) {
                    logs(ev.value ? "EV key DOWN code=" : "EV key UP   code=");
                    logn("", ev.code);
                    if (ev.value == 1) {  /* press */
                        if (ev.code == KEY_ESC) { running = 0; }
                        else if (ev.code == BTN_LEFT || ev.code == BTN_RIGHT || ev.code == BTN_MIDDLE) {
                            nkey++; pal = (pal + 1) % 7;
                            fill_rect(cx - 12, cy - 12, 24, 24, pack(255, 255, 255));
                            nstamp++;
                        } else {
                            nkey++;
                            uint32_t c = palette[pal]; pal = (pal + 1) % 7;
                            fill_rect(cx - 10, cy - 10, 20, 20,
                                      pack((c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff));
                            nstamp++;
                        }
                        dirty = 1;
                    }
                }
            }
        }

        if (dirty && FB.mem) {
            /* clamp + redraw cursor over a freshly cleared HUD */
            if (cx < 0) cx = 0; if (cx >= W) cx = W - 1;
            if (cy < 0) cy = 0; if (cy >= H) cy = H - 1;
            fill_rect(0, 0, W, 24, pack(40, 44, 56));
            fill_rect(2, 2, W - 4, 20, pack(60, 66, 84));
            /* crosshair cursor */
            fill_rect(cx - 8, cy - 1, 17, 3, pack(255, 255, 0));
            fill_rect(cx - 1, cy - 8, 3, 17, pack(255, 255, 0));
            flush();
            logn("cursor x=", cx); logn("cursor y=", cy);
        }
        if (!FB.mem && (nmotion + nkey) % 1 == 0) { /* input-only: still log totals */
        }
    }

    logn("uitest: done stamps=", nstamp);
    logn("uitest: done keys=", nkey);
    logn("uitest: done motion=", nmotion);
    logs("uitest: exiting on ESC\n");
    return 0;
}
