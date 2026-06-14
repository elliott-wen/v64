# aarch64-emu

An AArch64 (ARM64) CPU interpreter in Rust, built to be validated against
Unicorn via differential testing, with a WebAssembly JIT planned as a later
acceleration layer.

## Workspace layout

```
crates/
  cpu/        aarch64-cpu-state  — register file, NZCV flags (CpuState)
  decoder/    aarch64-decoder    — pure u32 -> Insn, split per encoding group
  interp/     aarch64-interp     — fetch/decode/execute loop, ALU, memory
  difftest/   aarch64-difftest   — cross-check our interpreter vs Unicorn
```

Each crate keeps modules small and single-purpose (e.g. the decoder has
`dp_imm.rs`, `dp_reg.rs`, `branch.rs`; the interpreter has `alu.rs`,
`execute.rs`, `run.rs`, `memory.rs`).

## Building & testing

The core (everything except the Unicorn oracle) needs only a Rust toolchain:

```sh
cargo test
```

### Differential testing against Unicorn

The oracle is behind the `unicorn` feature because it compiles QEMU via cmake.

```sh
cargo test -p aarch64-difftest --features unicorn
```

This build has a few **system prerequisites** (discovered the hard way):

- **clang / libclang** — `bindgen` needs it to generate Unicorn's FFI bindings.
  On Arch: `sudo pacman -S clang`. With system clang installed, no
  `LIBCLANG_PATH` or `BINDGEN_EXTRA_CLANG_ARGS` are required.
- **libatomic** — QEMU's 128-bit atomics (`__atomic_compare_exchange_16`)
  resolve to libatomic on x86-64. The difftest `build.rs` links it
  automatically when the `unicorn` feature is on; the lib itself ships with gcc
  (`/usr/lib/libatomic.so`).

The dependency is pinned to `default-features = false, features =
["arch_aarch64"]` so cmake only builds the ARM/AArch64 QEMU target rather than
all ten architectures.

## How differential testing works

A [`TestVector`](crates/difftest/src/vector.rs) carries machine code plus the
initial register/flag state. Both implementations are seeded **identically**
(Unicorn's ARM64 reset leaves `NZCV.Z` set, so `init_nzcv` is written explicitly
on both sides), run to the same `until` address, and their
[`StateSnapshot`](crates/difftest/src/snapshot.rs)s are compared field by field.
The first divergence is reported with the offending register.

## Currently implemented

All differentially fuzzed against Unicorn (33 classes, ~130k comparisons/run).

**Integer / data-processing**
- MOVZ/MOVN/MOVK; ADD/SUB (immediate, shifted-reg, extended-reg) + flag-setting
- ADC/SBC/ADCS/SBCS; logical (immediate + shifted-reg incl. BIC/ORN/EON)
- Bitfield (SBFM/BFM/UBFM), EXTR; conditional select & conditional compare
- Data-processing 1/2/3-source (RBIT/REV/CLZ/CLS, U/SDIV, shifts, MADD/MSUB/MULH)
- PC-relative (ADR/ADRP)

**Branches**
- B/BL, B.cond, CBZ/CBNZ, TBZ/TBNZ, BR/BLR/RET

**Load/store**
- LDR/STR all sizes + sign-extend, addressing: unsigned-imm, unscaled (LDUR),
  pre/post-index, register-offset, PC-literal
- LDP/STP/LDPSW (offset/pre/post); LDAR/STLR (acquire/release)

**Scalar floating-point**
- FMOV (reg/imm/gpr<->fpr), FABS/FNEG/FSQRT, FCVT (single<->double)
- FADD/FSUB/FMUL/FDIV/FMAX/FMIN/FMAXNM/FMINNM/FNMUL
- FCMP/FCMPE, FCSEL; SCVTF/UCVTF/FCVTZS/FCVTZU

Anything else decodes to `Insn::Unsupported`, which stops the run with the PC
and instruction word rather than silently misbehaving.

## Not yet implemented (design docs included)

See `DESIGN_*.md`: NEON/Advanced SIMD, syscalls + system registers, MMU/paged
memory, exclusives + LSE atomics, and the FP edge-case gaps (FPSR, non-default
rounding, FP16, FMADD, FCCMP).

## How the harness handles each instruction family

The differential fuzzer ([crates/difftest](crates/difftest)) seeds both
implementations identically and compares **registers, NZCV, the DATA region
(load/store), and V0–V31 (FP)**. Validity is a checked invariant: every word is
either state-matched, agreed-reserved, or a runtime fault — never silently
skipped. System/exception instructions are excluded because Unicorn re-implements
them (see `DESIGN_syscalls.md`).

## Reference material

- `../unicorn/qemu/target/arm/translate-a64.c` — reference instruction decoder
- `../unicorn/bindings/rust/unicorn-engine/src/tests/arm64.rs` — Unicorn's own
  ARM64 test vectors (a good source of more cases)
