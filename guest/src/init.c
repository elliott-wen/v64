/* Freestanding static PID 1 for platform bring-up — no libc, raw aarch64 syscalls. */
static long sys(long n, long a, long b, long c, long d, long e) {
    register long x8 __asm__("x8") = n;
    register long x0 __asm__("x0") = a;
    register long x1 __asm__("x1") = b;
    register long x2 __asm__("x2") = c;
    register long x3 __asm__("x3") = d;
    register long x4 __asm__("x4") = e;
    __asm__ volatile("svc #0" : "+r"(x0)
        : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4) : "memory", "cc");
    return x0;
}
#define SYS_write 64
#define SYS_ppoll 73
#define SYS_exit_group 94

static unsigned slen(const char *s){unsigned n=0;while(s[n])n++;return n;}

void _start(void) {
    const char *m = "v64 tiny-init: userspace alive, PID 1 running on aarch64\n";
    sys(SYS_write, 1, (long)m, slen(m), 0, 0);
    /* Block forever so the kernel never panics on init exit. */
    for (;;) sys(SYS_ppoll, 0, 0, 0, 0, 0);
}
