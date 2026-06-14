# Design: System calls and the system/exception instruction surface

Status: **not implemented.** These are the instructions deliberately excluded
from the differential fuzzer because **Unicorn diverges from real hardware/QEMU**
on them (it hooks and re-implements MRS/MSR/SYS and routes SVC/exceptions to its
own callbacks). So this group is verified against *expected/spec* behavior, not
against Unicorn.

## What's in this surface

Decoded today: only `NOP`. Everything else in the branch/exception/system group
(`op0 = 0b101x`) is `Unsupported`:

- **Exception generating**: `SVC #imm` (syscall), `HVC`, `SMC`, `BRK`, `HLT`.
- **Hints**: NOP/YIELD/WFE/WFI/SEV/SEVL — all architecturally no-ops for an
  EL0 interpreter; decode the hint space and treat as nops.
- **Barriers**: DMB/DSB/ISB — no-ops in a sequential interpreter.
- **System register move**: MRS/MSR (read/write `TPIDR_EL0`, `FPCR`, `FPSR`,
  `NZCV`, `DCZID_EL0`, counter regs, ...). `SYS`/`SYSL` (cache ops `DC`/`IC`).

## Recommended approach

### 1. Hints + barriers (trivial)
Decode the hint/barrier encodings and execute as nops. This alone lets a lot of
compiler output run (prologues emit `nop`, libc emits barriers).

### 2. System register file
Add a small `sysregs` map (or named fields) to `CpuState`. Implement MRS/MSR for
the EL0-visible registers a program actually touches:
- `TPIDR_EL0` — thread pointer (read/write; libc/TLS relies on it).
- `FPCR`/`FPSR` — already partially modeled (`cpu.fpcr`); expose via MRS/MSR.
- `NZCV` — already modeled; alias to PSTATE flags.
- `DCZID_EL0`, `CTR_EL0`, `MIDR_EL1` — return plausible constants.
- `CNTVCT_EL0`/`CNTFRQ_EL0` — virtual counter; back with a host clock or a
  monotonically increasing counter (pass time in, since `Date.now` is banned in
  some contexts — thread it through the run config).

### 3. SVC → syscall dispatch
`SVC #0` traps to a syscall handler keyed on `X8` (Linux AArch64 ABI), args in
`X0..X5`, return in `X0`. Architecture:

```
trait SyscallHandler { fn dispatch(&mut self, cpu: &mut CpuState, mem: &mut Memory); }
```

The interpreter's `run` loop, on decoding `SVC`, calls the handler instead of
faulting. Implement a **Linux-user subset** first, enough for static binaries:
- `write`(64), `read`(63), `exit`(93)/`exit_group`(94), `brk`(214),
  `mmap`(222)/`munmap`(215), `set_tid_address`(96), `ioctl`(29) for isatty,
  `clock_gettime`(113), `writev`(66), `getrandom`(278), `rt_sigprocmask`,
  `uname`(160).
- A simple program loader: parse a static AArch64 ELF, map PT_LOAD segments,
  set up the initial stack (argc/argv/envp/auxv), set PC to entry, SP to stack
  top. This is the milestone that makes "run a real `hello` binary" work.

### 4. Exceptions
`BRK`/`HLT` stop the run with a reason; full exception-level modeling (vector
table, ELR/SPSR, EL transitions) is only needed for system-mode emulation —
defer until booting a kernel is a goal.

## Testing strategy (not Unicorn)

- Syscall handlers: unit tests with a mock memory + asserting the side effects
  (bytes written, X0 return value).
- End-to-end: compile tiny C/asm with `aarch64-linux-gnu-gcc -static`, run under
  our interpreter, compare stdout/exit code against `qemu-aarch64` (the user-mode
  QEMU), which *is* an honest oracle for syscalls (unlike Unicorn).
- The instruction decoding of MRS/MSR/hints can still be sanity-checked against
  Unicorn for the *non-hooked* subset, but treat results cautiously.
