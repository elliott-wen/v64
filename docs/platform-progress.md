# Platform / device emulation — progress & handoff

_Last updated: 2026-06-16. All five device pieces needed to boot a minimal arm64
Linux on a QEMU-`virt`-style board are implemented and tested; **no real kernel
has been booted yet** — that's the next milestone._

## Goal

Emulate enough peripherals to boot a minimal arm64 Linux kernel (target board:
QEMU `virt`, GICv2). The CPU/ISA interpreter (`crates/interp`) and the WASM JIT
(`crates/jit`) already exist; this work is everything *around* the core, in a new
**`crates/platform`** crate, plus the supporting interpreter changes.

Endgame is the browser (WASM + JS), so design choices favour: device *models* in
Rust behind narrow interfaces, host I/O pluggable behind traits, no external
toolchains, deterministic where cheap.

## Where things stand — all device pieces done

| Piece | File(s) | State |
|---|---|---|
| Bus / MMIO dispatch | `platform/src/bus.rs` | done |
| GICv2 (GICD+GICC) + async IRQ injection | `platform/src/gic.rs`, `platform/src/machine.rs`, `interp/src/exception.rs` | done |
| Generic timer | `interp/src/timer.rs`, `platform/src/clock.rs`, `platform/src/machine.rs` | done |
| PL011 UART | `platform/src/uart.rs` | done |
| PSCI (HVC/SMC) | `interp/src/psci.rs`, `interp/src/execute.rs`, decoder | done |
| Boot protocol + DTB + board | `platform/src/board.rs`, `platform/src/loader.rs`, `platform/src/fdt.rs` | done |
| `v64` runner binary | `platform/src/bin/v64.rs` | done |

Workspace builds clean, all tests green (`cargo test --workspace`).

### Interpreter changes (in `crates/interp` / `crates/cpu` / `crates/decoder`)

- **`GuestMem` reads now take `&mut self`** (`interp/src/memory.rs`). MMIO reads
  mutate (UART RX FIFO pop, `GICC_IAR` acknowledge), so the trait's `read_u*`
  became `&mut`. Rippled through `mmu::translate`, `mem_access::{read,read_vec}`,
  the fetch line in `run`/`step`, and two executor helpers (`ldst_excl::load`,
  `ldst_pair::load_elem`). RAM/`MemView` impls are unchanged behaviourally.
- **`interp::take_irq(cpu) -> u64`** (`exception.rs`). Factored the EL1 entry
  sequence into `enter_el1(cpu, return_addr, vec_type)` (vec_type = `VEC_SYNC`
  0x000 / `VEC_IRQ` 0x080); `take_irq` saves `ELR = cpu.pc` and vectors through
  the IRQ slot. SVC path unchanged (still writes ESR).
- **Generic-timer registers** (`interp/src/timer.rs`, hooked into
  `system.rs`): `CNTV_TVAL`↔`CNTV_CVAL` conversion, computed `ISTATUS`. The live
  count (`CNTVCT`/`CNTPCT`) and `CNTFRQ` are kept in the sysreg map by the
  machine's clock; the interpreter only reads them. Public API:
  `set_frequency`, `set_count`, `virtual_fires`, `physical_fires`.
- **PSCI** (`interp/src/psci.rs`): `HVC`/`SMC` decoded by `decoder/src/branch.rs`
  (`Insn::Hvc`/`Smc`, LL bits 10/11), dispatched in `execute.rs`. Handles
  `PSCI_VERSION` (→1.0), `SYSTEM_OFF`/`SYSTEM_RESET` (set `CpuState.powered_off`),
  `CPU_AFFINITY_INFO`, `MIGRATE_INFO_TYPE`; else `NOT_SUPPORTED`. Bit 30 masked so
  SMC32/64 share an arm. New `CpuState.powered_off` halt flag;
  `StopReason::PoweredOff`.

### platform crate

- **`Bus`** (`bus.rs`): owns RAM (`interp::Memory`) + a sorted, non-overlapping
  device table; `impl GuestMem` routes a physical access to RAM or an
  `MmioDevice` (linear scan, tiny N). Unmapped access logs + reads 0 / drops
  (a stray probe doesn't kill boot). `MmioDevice` trait: `name`/`read`/`write`,
  both `&mut self`. `ram_mut()` for loading images.
- **`Gic`** (`gic.rs`): GICv2 subset, single CPU. `GicInner` behind
  `Rc<RefCell>`; the bus maps `gic.distributor()` (GICD) and `gic.cpu_interface()`
  (GICC) as two adapters over the same state. Peripherals raise lines via
  `set_pending(id)`; the machine polls `pending_irq()`. `GICC_IAR` read is the
  side-effecting acknowledge (pending→active, pushes running priority);
  `GICC_EOIR` deactivates. **`pending_irq` is O(1)** — a `cached_pending` field
  recomputed only on state change (`set/clear_pending`, enable/priority writes,
  IAR), not scanned per instruction.
- **Timer**: host clock behind a **`Clock` trait** (`clock.rs`); `HostClock` =
  monotonic `Instant` scaled to 62.5 MHz (matches QEMU default `QEMU_CLOCK_VIRTUAL`
  and v86's `performance.now()`). The browser build swaps in a `performance.now()`
  source behind the same trait. `Machine` samples the clock into `CNTVCT`/`CNTPCT`
  and (de)asserts the timer PPIs (virtual=27, physical=30) — **every
  `TIMER_SAMPLE_INTERVAL` (64) instructions**, not per instruction (perf).
  `Machine::set_timer_interval(1)` forces per-step sampling for deterministic tests.
- **PL011 UART** (`uart.rs`): `Uart` handle (`Rc<RefCell>`) like `Gic`. TX is
  instantaneous → a buffer drained by `take_tx()` (host prints / forwards to a
  terminal); RX is a FIFO filled by `feed()`. `RIS & IMSC` drives the UART SPI
  into the GIC, updated event-driven (never polled). PrimeCell/Peripheral IDs at
  `0xFE0..` carry the AMBA signature so `amba-pl011` binds `ttyAMA0`.
- **`Machine`** (`machine.rs`): owns CPU + `Bus` + `Gic` + `Clock`. Each step:
  advance timers (throttled), inject an IRQ if `pending_irq() && !PSTATE.I`, then
  execute one instruction. `run(until, count)` mirrors `interp::run`'s stop
  contract and also stops on `powered_off`. `interp::run` itself is untouched.
- **Board + boot** (`board.rs`, `loader.rs`): `Board::new()` wires the `virt`
  memory map (RAM @ `0x4000_0000`, GICD @ `0x0800_0000`, GICC @ `0x0801_0000`,
  UART @ `0x0900_0000`, UART IRQ = SPI 1 = 33) with **1 GiB default RAM**
  (`with_ram(n)` for the small tests). `boot()` is the simple header-less path
  (test mini-kernels). `boot_image(image, initrd, bootargs)` is the real path:
  parses the arm64 `Image` header (`parse_image_header`), places kernel /
  initrd / DTB at non-overlapping 2 MiB-aligned addresses, builds the DTB with the
  initrd range + bootargs, and sets entry state. Both share `Board::enter`.
- **FDT** (`fdt.rs`): `FdtBuilder` emits the DTB binary format (spec v17,
  big-endian) — no `dtc` dependency, works in WASM. `virt_dtb()` builds the node
  set: memory, `cpus/cpu@0` (`enable-method="psci"`), `psci` (`method="hvc"`),
  `timer` (PPIs 11/14 = IRQ 27/30), GICv2 `intc`, `apb-pclk`, `pl011` with SPI +
  clocks, `chosen` (bootargs / stdout-path / initrd).
- **`v64` runner** (`bin/v64.rs`):
  `cargo run -p aarch64-platform --bin v64 -- <Image> [initramfs.cpio.gz]`.
  Boots with `earlycon=pl011,0x9000000 console=ttyAMA0 rdinit=/init`, streams the
  PL011 console to stdout, and on stop reports power-off **or the unimplemented
  instruction (pc + opcode)** — the bring-up feedback loop.

## Tests

Device unit/integration tests live in `crates/platform/tests/` (never inline in
`src/` — project preference): `bus`, `gic`, `irq`, `timer`, `uart`, `psci`,
`fdt`, `boot`. Plus interp-level `interp/tests/{timer,psci}.rs`.

**Subsystem mini-kernels** (`platform/tests/kernels.rs`) are the headline: tiny
guest programs assembled in-test (shared assembler in `tests/common/mod.rs`) that
drive one subsystem each through the *real* `boot()` path:
- `mmu_translates_a_store_through_a_block_mapping` — guest builds an L1 block
  table, enables `SCTLR.M`, stores through an aliased VA; test verifies the
  physical alias holds the value (would be dropped if translation didn't run).
- `virtual_timer_interrupt_reaches_handler` — guest arms `CNTV` + GIC, the timer
  PPI vectors to a handler.
- `full_irq_handler_cycle_with_iar_eoir_eret` — guest pends an SPI; handler reads
  IAR, writes EOIR, ERETs; main resumes (`"IM"` proves both).

These are deterministic (timer uses interval=1 + `CVAL=0`; interrupt uses a
software-pended SPI), so no flakiness.

## Next milestone: boot a real kernel

Nothing has run an actual Linux kernel yet — that's where unmodeled corners
surface. The loop is: build an `Image`, run `v64`, fix what it trips on, repeat.

**Getting a kernel (build on Linux, or in a container — native arm64 = no
cross-compiler):**
```
# arm64 Linux container (Colima/Docker on Apple Silicon, or a real Linux box):
apt-get install -y build-essential bc bison flex libssl-dev libelf-dev \
                   wget xz-utils cpio busybox-static
# kernel: defconfig already includes GIC, PSCI, arch timer, PL011, virtio
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.<x>.tar.xz
tar xf linux-*.tar.xz && cd linux-*
make ARCH=arm64 defconfig && make ARCH=arm64 -j"$(nproc)" Image   # -> arch/arm64/boot/Image
# tiny busybox initramfs:
mkdir -p ir/bin && cp /bin/busybox ir/bin/
for c in sh mount ls cat echo dmesg; do ln -s busybox ir/bin/$c; done
printf '#!/bin/busybox sh\n/bin/busybox mount -t proc proc /proc\necho userspace alive\nexec /bin/busybox sh\n' > ir/init
chmod +x ir/init
(cd ir && find . | cpio -o -H newc | gzip) > initramfs.cpio.gz
```
**Important:** load the *raw uncompressed* `Image` (magic `ARM\x64` at offset 56),
**not** `vmlinuz`/`Image.gz`/zboot — we have no decompressor/EFI stub.

First proof-of-life is the kernel banner via `earlycon` — no initramfs required.
Then add the initramfs to reach a busybox shell.

```
cargo run -p aarch64-platform --bin v64 -- out/Image out/initramfs.cpio.gz
```

## Known gaps / likely bring-up work

- **MMU faults fall back to identity.** `interp/src/mmu.rs` `walk()` returns the
  VA (identity) on an invalid descriptor instead of raising a Data Abort
  (long-standing TODO). A real kernel that expects a fault won't get one — may
  need real fault injection via `take_exception`.
- **Unimplemented instructions.** A real kernel will use instructions the
  interpreter doesn't cover yet; `v64` prints the pc+opcode so each is a discrete
  fix. (The ISA fuzz sweep is green for userspace, but kernel/system corners
  differ.)
- **GIC is single-core.** `CPU_ON` returns `NOT_SUPPORTED`; a single-CPU DTB
  never calls it. SMP needs real `CPU_ON` + a second core.
- **No EL2/EL3.** `HVC` and `SMC` both route straight to PSCI.
- **No virtio.** initramfs avoids needing a block device; virtio-mmio block/net
  is the follow-up for a real rootfs / networking (the DTB doesn't declare it yet).
- **`WFI` is not optimized** — it busy-steps (correct under wall-clock, since the
  counter advances on real time, but wastes host CPU). Fast-forward to the next
  timer deadline later.
- **Performance.** Booting a ~40 MB kernel in the interpreter is billions of
  instructions = minutes. The JIT (`crates/jit`) is *not* wired into the platform
  `Machine` loop yet — the natural place to also move the per-step `RefCell`
  borrow / IRQ-flag check to block boundaries.
- **DTB content is plausible but unvalidated by a real kernel** — node set is
  modelled on `virt` but only structurally tested (`tests/fdt.rs`). Expect tweaks
  (clock bindings, interrupt flags) once a kernel parses it.

## Useful facts

- Memory map / IRQs: `aarch64_platform::{RAM_BASE, GICD_BASE, GICC_BASE,
  UART_BASE, UART_IRQ, KERNEL_LOAD, DTB_LOAD, DEFAULT_RAM_SIZE}`.
- DAIF nibble is `[D,A,I,F]`; IRQ mask (`PSTATE.I`) = bit 1 (`0b0010`).
- Timer PPIs: virtual = 27, physical = 30. UART = SPI 1 = IRQ 33.
- platform exports: `Board`, `BootLayout`, `ImageHeader`, `parse_image_header`,
  `Bus`, `MmioDevice`, `Gic`/`GicDist`/`GicCpu`, `Uart`/`UartDevice`, `Machine`,
  `Clock`/`HostClock`/`DEFAULT_FREQ_HZ`, `FdtBuilder`/`virt_dtb`/`DtbConfig`.
- interp exports for this work: `take_irq`, `set_count`, `set_frequency`,
  `virtual_fires`, `physical_fires` (plus the existing `step`, `run`, `Step`,
  `StopReason`, `translate`, `GuestMem`, `Memory`, `MemView`).
- Mini-kernel assembler: `crates/platform/tests/common/mod.rs` (encoders +
  `Asm` builder + `boot_and_run`).

## How to run things
```
# everything
cargo test --workspace
# just the device crate
cargo test -p aarch64-platform
# boot a real kernel and watch the console
cargo run -p aarch64-platform --bin v64 -- path/to/Image [path/to/initramfs.cpio.gz]
```
Run cargo from the workspace root.
