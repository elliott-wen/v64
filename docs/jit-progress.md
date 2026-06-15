# JIT progress / handoff

_Last updated: 2026-06-16 (M0–M6 complete; SIMD Tier 1 lowered incl. two-reg-misc / saturating / widening multiply; interp memory behind the `GuestMem` trait so the JIT shares linear memory with no copy)_

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
- **M2 — done.** Block ABI, shared linear memory, and the `interpret_one`
  escape hatch. Generated blocks now actually execute (the M1 placeholder is
  gone). All wiring lives behind the public `crates/jit` API; tests moved to
  `crates/jit/tests/` (`block.rs`, `runtime.rs`).
  - `abi.rs`: the documented contract. Linear-memory layout — `GuestRegs` image
    at `REGS_BASE=0`, a control block at `CTRL_BASE=0x340` (`exit_reason`,
    `exit_pc`), guest RAM at `RAM_BASE=0x10000`; guest addr `a` -> linear
    `RAM_BASE + (a - guest_base)` (`ram_offset`). **Exit convention: chose the
    side-channel control block** (`exit_reason`/`exit_pc`) over packing bits into
    the returned `i64`, so the full 64-bit next-PC stays unambiguous. Reasons:
    `EXIT_NONE`, `EXIT_UNSUPPORTED` (extensible).
  - `emit.rs`: module imports `env.interpret_one : (i32)->i64` and `env.memory`,
    exports `block`. Body lowers **every** instruction to a `call interpret_one`
    (one per insn, in order; non-terminators `drop` their PC, the terminator's
    return is the block's result). M4 swaps individual calls for inline code.
  - `runtime.rs`: `Vm` owns the wasmtime engine/store/linker and the shared
    memory; `interpret_one` host fn syncs hot regs + the RAM window across the
    interpreter boundary and steps one instruction via the new
    `interp::step`/`interp::Step`. **Cold** CPU state (sysregs/EL/excl/...) lives
    in the host `Runtime` struct, persisted across calls. `Vm::{new, load_regs,
    store_regs, write_ram, read_ram, run_block}`.
  - `interp`: added `pub fn step(cpu, mem) -> Step` (single-step primitive,
    mirrors one `run()` iteration). `run()` itself is **untouched** per the plan.
- **M3 — done.** Identity-mapped memory fast path + per-instruction lowering
  dispatch (the scaffold M4 extends).
  - `lower.rs`: `lower_inline(f, insn, pc, guest_base) -> bool` — emits inline
    WASM for an instruction or returns false (caller falls back to
    `interpret_one`; correctness is never at stake, only speed). Implements
    single-register `LoadStore` with `AddrMode::UnsignedImm`, integer, **unsigned**
    (signed loads still fall back). Address = `base_reg + imm + (RAM_BASE -
    guest_base)`, folded into one displacement; sized `i64.load{,8u,16u,32u}` /
    `i64.store{,8,16,32}`. Handles Rn==31 (SP base) and Rt==31 (XZR: store 0 /
    load-and-discard). Each inline op advances the image PC to `pc+4` itself, so
    inline and helper instructions interleave freely.
  - `emit.rs`: `emit_block(block, guest_base)`; body now tries `lower_inline`
    first for each non-terminator, else `interpret_one`. Terminator is never
    inlined (it's a branch/exception) so it always returns via `interpret_one`.
  - `runtime.rs`: `run_block` catches WASM traps (e.g. inline OOB access) and
    surfaces them as `EXIT_FAULT` at the faulting PC instead of a host panic.
    `interpret_one` runs the interpreter **directly on the shared linear memory**
    via `interp::MemView` (no copy) — see the GuestMem-trait refactor below.
  - `abi.rs`: added `EXIT_FAULT`.
  - **Inline-vs-helper assumption** (documented in `lower.rs`): inline accesses
    assume identity mapping (MMU off), which is the runtime's only mode today.
    MMU-on / MMIO would need the slow path — deferred.
- **M4 — done (comprehensive).** The full integer + branch core lowered inline,
  with **flags computed inline** (no host helpers — NZCV math is emitted in
  WASM). Two entry points: `lower_sequential` (non-terminators: leave nothing on
  the stack, advance image PC to `pc+4`) and `lower_terminator` (control flow:
  leave the next-PC `i64` as the block result, don't touch image PC). 5 scratch
  i64 locals (`SCRATCH_I64`).
  - **`lower.rs` is now a module dir** `lower/` split by family for readability:
    `common` (image access, scratch locals, NZCV bits), `arith` (move-wide,
    add/sub all forms incl. extended & carry, logical, const/var shift apply,
    EXTR, ADR/ADRP), `cond` (condition-code eval, CSEL family, CCMP/CCMN),
    `dataproc` (bitfield, DP-1/2/3-source), `memory` (loads/stores + pairs),
    `terminator` (branches). The `emit!` macro lives in `mod.rs` ahead of the
    submodule declarations (textual scoping).
  - Lowered inline: `Nop`; `MoveWide`; `LogicalImm`/`LogicalShiftedReg`
    (any shift/amount, incl. BIC/ORN/EON/BICS); `AddSubImm`/`AddSubShiftedReg`
    (any shift/amount)/`AddSubExtReg`/`AddSubCarry` (ADC/SBC) — full NZCV inline;
    `Extract`; `PcRel`; `CondSelect` (CSEL/CSINC/CSINV/CSNEG); `CondCompare`
    (CCMP/CCMN); `Bitfield` (SBFM/BFM/UBFM); `DataProc1Src` (CLZ, REV16/32/64);
    `DataProc2Src` (LSLV/LSRV/ASRV/RORV, UDIV/SDIV with AArch64 /0 and INT_MIN/-1
    semantics); `DataProc3Src` (MADD/MSUB, S/UMADDL, S/UMSUBL); `LoadStore` all
    addressing modes (unsigned/unscaled imm, pre/post-index with writeback,
    register-offset, PC-literal) + signed loads + W/X width; `LoadStorePair`
    (LDP/STP/LDPSW). Branch terminators: B/BL, BR/BLR/RET, B.cond, CBZ/CBNZ,
    TBZ/TBNZ.
  - **Still fall back** (correct via `interpret_one`): RBIT, CLS, CRC32,
    SMULH/UMULH (128-bit), all SIMD/FP, atomics/exclusives, system. Each declines
    *before emitting* so a fallback never leaves partial code.
  - Tests: `tests/crosscheck.rs` runs curated programs (seeded registers, 8
    seeds each) through the JIT and the interpreter and asserts bit-identical
    GuestRegs + memory **and** `interp_calls()==0` — proving the inline path ran
    rather than silently falling back. `tests/lower.rs` adds focused
    branch/flag cases.

- **M5 — done.** Block dispatcher (`Vm::run(until, count) -> StopReason`) in
  `dispatch.rs`, a run loop alongside `run.rs` (which is untouched).
  - Mirrors `interp::run`'s stop contract exactly, at block granularity:
    top-of-loop checks `pc == until` then `count` (same order as `run()`), so the
    two agree even when a single-instruction vector lands its PC on `until`.
  - `block.rs` gained `form_block_bounded(start, until, max_len, read)` — caps
    the block at the `until` boundary and remaining instruction budget. `emit.rs`
    now also lowers a **non-terminator last instruction inline** (emitting the
    sequential next PC as the block result), so single-instruction blocks (the
    `count=1` fuzz vectors) still exercise inline lowering instead of falling to
    `interpret_one`.
  - `runtime.rs` split `run_block` into `compile_instance` + `call_instance`
    (mechanics); the dispatcher (policy) holds a per-run `HashMap<pc, instance>`
    cache with a coarse SMC guard (recompile if the block's source bytes changed).
    Block chaining and a persistent cross-run cache are deferred (perf, not
    correctness). `image_pc`/`read_code_word` expose what the loop needs.
  - Per the architecture discussion, policy (cache/SMC) is kept separate from
    mechanics (`Module::new`/instantiate/call) so the loop ports to a JS host.
  - Tests: `tests/dispatch.rs` cross-checks `Vm::run` against `interp::run` on a
    backward-branch loop (cache reuse) and the `until`/`count` boundaries.
- **M6 — done.** `run_jit` differential backend + JIT-vs-interpreter sweep.
  - `crates/difftest` gained a `jit` feature (optional `aarch64-jit` dep, since
    it pulls wasmtime). `jit_run::run_jit(tv) -> (StateSnapshot, StopReason)`
    mirrors `run_ours`: same image, same seeding, runs via `Vm::run`, snapshots
    identically. `fuzz.rs` adds `jit_fuzz_{class,fp_class,mem_class}` comparing
    `run_jit` against the trusted `run_ours` (reusing one `Vm` across iterations).
  - `tests/jit_sweep.rs` (`#![cfg(feature = "jit")]`): all 68 classes,
    `FUZZ_ITERS` per class (default 3000), three parallel `#[test]`s. **Green**:
    integer/branch/memory exercise inline lowering; FP/SIMD exercise the
    `interpret_one` fallback. Run with
    `cargo test -p aarch64-difftest --features jit --test jit_sweep`.
  - The sweep immediately caught a real lowering bug: S/UMADDL read the
    accumulator `Ra` via `offsets::x(ra)`, but `ra==31` is XZR and `x(31)` aliases
    the SP slot — it read SP instead of 0. Fixed to go through `push_operand`
    (r31 → ZR). The interpreter (the reference) was right; exactly what M6 is for.

## SIMD lowering — Tier 1 done (bit-exact integer + data movement)

With M6 green, every new lowering is validated against the interpreter by the
sweep. The first SIMD increment (all bit-exact, no FP) is in `lower/`:
- **Vector load/store** (`memory.rs`): SIMD/FP `LDR/STR` (B/H/S/D/Q, every
  addressing mode) and `LDP/STP` pairs. The 128-bit form is `v128.load/store`;
  narrower loads zero the unused high bytes of the V register. The integer and
  vector paths now share address computation via the new `ADDR` i32 scratch local
  (`SCRATCH_I32`). The guest V regs are little-endian at `offsets::v(n)`, matching
  WASM `v128` lane order, so this is pure byte movement.
- **`SimdThreeSame` integer** (`simd.rs`): logical (AND/BIC/ORR/ORN/EOR and the
  BSL/BIT/BIF selects via `v128.bitselect`), ADD/SUB (all sizes), MUL (16/32-bit),
  compares (CMEQ/CMTST/CMGT/CMGE and unsigned CMHI/CMHS), and S/U MAX/MIN. The
  `!q` (64-bit) forms zero the high half with a `v128.and` mask. Forms WASM can't
  express bit-exactly **fall back** (i64 unsigned compares, i64 min/max, 8-bit
  MUL, and all saturating/halving/pairwise/shift/abd/mla/sqdmulh/polynomial ops),
  each declining before emitting.
- **Modified-immediate** (`simd.rs`): MOVI/MVNI → `v128.const` (the element is a
  compile-time constant); ORR/BIC-immediate fold it against Vd. The integer
  cmodes are expanded inline (mirrors `simd_mod_imm::expand`); the FMOV-vector
  cmode 1111 falls back (needs the FP immediate helpers).
- **Copy family** (`simd.rs`), all as plain image loads/stores — no v128 needed:
  `UMOV/SMOV` (lane → GPR, zero/sign-extend), `INS` general (GPR → lane) and
  element (lane → lane), `DUP` general (GPR → `iNxM.splat`) and element
  (lane → splat).
- **Permutes** (`simd/permute.rs`): `ZIP1/2`, `UZP1/2`, `TRN1/2`, `EXT`, and
  single-table `TBL` (→ `i8x16.swizzle`) as constant shuffles computed at emit
  time; `!q` masks the high half.
- **Shift-by-immediate** (`simd/shift.rs`): SHL, SSHR, USHR (full-width shifts
  handled specially since WASM lane shifts mask the amount).
- **Two-register-misc integer** (`simd/two_reg_misc.rs`): NOT, CNT
  (`i8x16.popcnt`), NEG, ABS, compare-to-zero (CMGT/CMGE/CMEQ/CMLE/CMLT #0), and
  REV64/16/32 (constant self-shuffle). CLS/CLZ, RBIT, the saturating ops, and the
  shape-changing forms (XTN/SHLL/ADDLP) fall back.
- **Saturating add/sub** (in `simd/three_same.rs`): SQADD/UQADD/SQSUB/UQSUB for
  8/16-bit lanes (`iNxM.add_sat_*`/`sub_sat_*`); 32/64-bit fall back (no WASM op).
- **Widening multiply** (`simd/three_diff.rs`): SMULL/UMULL (and the `2`
  high-half variants) via `extmul_low/high` — Q selects the source half, U the
  sign; the result is always full-width. The other three-different forms (widening
  add/sub/accumulate, SQDMULL, ABDL, PMULL, high-narrowing) fall back.
- The `lower/simd.rs` file is now a `lower/simd/` dir split by family
  (`three_same`, `copy`, `permute`, `shift`, `two_reg_misc`, `mod`).
- Validated: `tests/runtime.rs` proves vector LDR/STR Q and D are fully inline
  (`interp_calls()==0`); the `jit_sweep` `neon_*` and `ldst_vec_*` classes pass
  against the interpreter at 8k+ iters.

## Shared memory: the `GuestMem` trait (browser-port prerequisite)

`interp::Memory` was an *owned* `Vec<u8>`, so `interpret_one` had to copy the
guest-RAM window out of wasmtime linear memory and back on every call. That copy
is gone:
- `interp/src/memory.rs` now defines a **`GuestMem` trait** (sized little-endian
  loads/stores), with two implementations: the owned `Memory { base, bytes:
  Vec<u8> }` (native execution, tests, the Unicorn oracle) and a borrowed
  `MemView<'a> { base, bytes: &'a mut [u8] }`. The interpreter's executors take
  `&mut dyn GuestMem`; callers passing a concrete `&mut Memory` unsize-coerce
  automatically, so only interp-internal signatures changed.
- `jit::interpret_one` `split_at_mut`s the shared linear memory into the register
  image (head) and the guest RAM window (tail), wraps the window in a `MemView`,
  and steps the interpreter on it **in place** — no copy.
- Why a trait (not a lifetime-`Memory<'a>`): the trait's primitive is the *sized
  access*, so a future MMU/MMIO backing can dispatch to a handler instead of
  indexing a buffer (a flat slice can't). `dyn` is fine here — the interpreter is
  the cold/reference path; if it ever needs to be fast, switch to generic
  `<M: GuestMem>` for zero-overhead monomorphization, same trait.
- This is the one structural change the all-wasm/browser port needs: there the
  interpreter is itself a WASM module importing the shared `WebAssembly.Memory`,
  which it accesses as exactly this kind of borrowed view.

## Next: push SIMD further

Still on the interpreter fallback (good follow-ups, in rough value order):
**Tier 1 cont.** — pairwise add (`extadd_pairwise`), across-lanes reductions
(need log-step shuffle+op), RBIT/CLS/CLZ (no direct WASM op), the widening
add/sub/accumulate forms. **Tier 2 (careful)** — FP three-same/two-reg
(`f32x4`/`f64x2`), watching the
sweep for NaN/rounding/FPCR divergence WASM won't match bit-for-bit. The decoder
variants are all in `decoder/src/insn.rs` (`Simd*`); each interp handler in
`interp/src/simd_*.rs` is the semantics reference.

### Useful facts
- `interp` re-exports: `add_with_carry`, `add_with_carry_in`, `apply_shift`,
  `eval_cond`, `Memory`, `step`, `Step`, `mmu::translate`, `run`, `StopReason`.
  Single-step entry is `interp::step(cpu, mem) -> Step` (`Step::Next(pc)` /
  `Step::Unsupported{pc,word}`); `execute()` itself stays `pub(crate)`.
- `Memory { base: u64, bytes: Vec<u8> }`, flat little-endian, identity-mapped
  when MMU is off (the current default / the fuzz harness case).
- Reference run loop to mirror for the dispatcher (M5): `interp/src/run.rs`
  (do **not** modify `run()` — it stays the reference).
- Offset constants for the JIT: `aarch64_cpu_state::regs::offsets`
  (`X`, `SP`, `PC`, `NZCV`, `V`, `FPCR`, `SIZE`, `x(n)`, `v(n)`).
- JIT ABI constants: `aarch64_jit::abi` (`REGS_BASE`, `CTRL_BASE`, `EXIT_REASON`,
  `EXIT_PC`, `RAM_BASE`, `WASM_PAGE`, `EXIT_*`, `ram_offset`).

## Remaining milestones (from jit-plan.md)
- M0–M6 — **done** (see above). The seven-step plan is complete: the JIT runs
  the full integer/branch/memory core inline and matches the interpreter across
  the whole ISA fuzz sweep.
- Post-M6 (optional, perf/coverage): SIMD/FP lowering (**next focus**), then
  register-caching in WASM locals, lazy flags, block chaining, a persistent
  cross-run block cache, and MMU/MMIO inline paths.
- Periodic three-way check: since JIT and interp share the decoder, keep running
  the Unicorn sweep (`--features unicorn`) too, so decoder bugs can't hide.

## How to run things
```
# full ISA differential fuzz (default 50k/class)
cargo test -p aarch64-difftest --features unicorn --test fuzz_sweep
# deep version
FUZZ_ITERS=1500000 cargo test -p aarch64-difftest --features unicorn --test fuzz_sweep
# JIT unit/integration tests (block formation, lowering, dispatcher)
cargo test -p aarch64-jit
# JIT-vs-interpreter differential sweep (all 68 classes)
cargo test -p aarch64-difftest --features jit --test jit_sweep
FUZZ_ITERS=200000 cargo test -p aarch64-difftest --features jit --test jit_sweep  # deeper
```
Note: run cargo from the workspace root (the repo root, where `Cargo.toml` is).
