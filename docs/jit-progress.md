# JIT progress / handoff

_Last updated: 2026-06-15_

## Where things stand

### ISA (done — the prerequisite the JIT builds on)
The full AArch64 interpreter is implemented and **differential-fuzzed against
Unicorn (QEMU `max` CPU) at 1.5M iterations/class across all 68 classes — green.**
The deep fuzz found and fixed one real bug (FRSQRTS double-precision
overflow-before-halve). Coverage and the deferred optional extensions (FP16,
FCMLA, v8.3+ crypto, PAuth, BF16 — none needed for Linux userspace) are listed
in [isa-coverage.md](isa-coverage.md).

### JIT (in progress — following [jit-plan.md](jit-plan.md))

- **M0 — done.** `crates/cpu/src/regs.rs`: `#[repr(C)] GuestRegs`
  (x/sp/pc/nzcv/v/fpcr) — the flat, offset-addressable register image the JIT
  mirrors in wasmtime linear memory. Pinned offset table (`regs::offsets`) with
  a unit test that fails if any offset shifts. `CpuState::to_guest_regs()` /
  `load_guest_regs()` convert at the JIT boundary (flags<->packed nzcv).
  **Chose the "convert" form over embedding `regs` in CpuState** so the
  interpreter storage is untouched and the suite stays trivially green.
- **M1 — done.** New `crates/jit` crate (wasm-encoder 0.252 + wasmtime 45).
  - `block.rs`: `form_block(start, read)` decodes forward to a terminator
    (branch / Svc / Eret / Unsupported). `is_terminator()`.
  - `emit.rs`: `emit_block()` produces a WASM module exporting `block` with the
    ABI signature `(param $regs_base i32) -> i64` (next guest PC). **Body is a
    placeholder** that returns the fall-through PC (`start + 4*len`).
  - `lib.rs`: `run_block_placeholder()` does emit -> wasmtime compile ->
    instantiate -> call. Test `emit_compile_run_roundtrip` passes.

## Next: M2 — block ABI & helper imports

The contract between generated blocks and the runtime. Key tasks:

1. **Linear-memory layout** in the wasmtime instance:
   - `GuestRegs` image at `REGS_BASE` (use offset 0). Size = `regs::offsets::SIZE`
     (800 bytes).
   - Guest RAM region at `RAM_BASE`; guest addr `a` -> linear offset
     `RAM_BASE + (a - Memory.base)`.
2. **Define the function ABI + exit convention.** Pick ONE and document it:
   either an `exit_reason`/`exit_pc` word added to the state image, or encode the
   reason in the high bits of the returned `i64`. The block returns the next
   guest PC; non-inlineable cases (exception, unsupported, atomics, MMU slow
   path) signal via the exit convention.
3. **`interpret_one(regs_base: i32) -> i64` import** — the long-tail escape
   hatch. The host implementation:
   - reads the `GuestRegs` image out of linear memory,
   - `CpuState::load_guest_regs(&gr)` (cold state — sysregs/EL/... — lives
     host-side in the runtime struct, persisted across calls),
   - runs exactly one instruction via the existing `interp` `execute()` path
     (decode the word at PC, execute, update PC),
   - `cpu.to_guest_regs()` written back into linear memory,
   - returns the next PC.
   Wire it as a wasmtime `Func`/`Linker` import named e.g. `env.interpret_one`.
4. Optionally also import `add_with_carry` / `apply_shift` / `eval_cond`
   (already `pub` from `interp`) to bootstrap M4 lowerings without
   reimplementing flag logic in WASM.

### Useful facts for M2+
- `interp` re-exports: `add_with_carry`, `add_with_carry_in`, `apply_shift`,
  `eval_cond`, `Memory`, `mmu::translate`, `run`, `StopReason`. Dispatch entry is
  `interp::execute(cpu, mem, insn, pc) -> Option<u64>` — `Some(target)` = branch,
  `None` = fall through to pc+4 (this maps directly onto block exits).
- `Memory { base: u64, bytes: Vec<u8> }`, flat little-endian, identity-mapped
  when MMU is off (the current default / the fuzz harness case).
- Reference run loop to mirror for the dispatcher (M5): `interp/src/run.rs`
  (do **not** modify `run()` — it stays the reference).
- Offset constants for the JIT: `aarch64_cpu_state::regs::offsets`
  (`X`, `SP`, `PC`, `NZCV`, `V`, `FPCR`, `SIZE`, `x(n)`, `v(n)`).

## Remaining milestones (from jit-plan.md)
- M2 — block ABI + helper imports (`interpret_one` first). **<-- next**
- M3 — identity-mapped memory fast path.
- M4 — lower the ~12 common integer/branch instructions (MoveWide, AddSub*,
  Logical*, Branch*, Cmp/TestBranch, basic LDR/STR, Nop); everything else ->
  `interpret_one`.
- M5 — dispatcher + block cache + coarse SMC invalidation (new loop alongside
  `run.rs`).
- M6 — `run_jit` backend in `crates/difftest`; fuzz interp-vs-JIT to green
  (reuse the existing encoders/fuzzers; keep periodic three-way Unicorn checks
  since JIT and interp share the decoder).

## How to run things
```
# full ISA differential fuzz (default 50k/class)
cargo test -p aarch64-difftest --features unicorn --test fuzz_sweep
# deep version
FUZZ_ITERS=1500000 cargo test -p aarch64-difftest --features unicorn --test fuzz_sweep
# JIT tests
cargo test -p aarch64-jit
```
Note: the workspace lives in `emu/` (run cargo from there).
