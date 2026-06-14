# AArch64 → WebAssembly JIT — Implementation Plan

## Goal

Add a JIT backend that translates hot blocks of guest AArch64 code into
WebAssembly, runs the generated WASM via an embedded **wasmtime** runtime, and
produces guest-visible state **bit-identical** to the existing interpreter.

The interpreter (`crates/interp`) stays as the cold-path executor and the
deterministic reference oracle. The JIT is a faster tier layered on top, not a
replacement.

**Out of scope for now:** Node.js / browser execution, register-caching and
lazy-flag optimizations, SIMD/FP lowering, multi-core. Get a *correct,
helper-heavy* JIT passing the differential harness first; optimize later.

---

## Current architecture (what we build on)

Already JIT-friendly — **do not change**:

- **Decoder** (`crates/decoder`): pure `decode(u32) -> Insn`. Reused verbatim.
- **Handler convention**: every interpreter handler returns `Option<u64>`
  (`Some(target)` = branch, `None` = fall through to `pc+4`). Maps directly
  onto block exits.
- **Per-class handlers** (`crates/interp/src/*.rs`, e.g. `add_sub_imm.rs`,
  `branch_imm.rs`): a ready-made 1:1 checklist for lowering.
- **Shared helpers** exported from `interp`: `add_with_carry`, `apply_shift`,
  `eval_cond`. The JIT can *call these as imports* to bootstrap correctness.
- **Differential harness** (`crates/difftest`): `TestVector` → `run_ours` /
  `run_unicorn` → `StateSnapshot` → `diff`. A JIT backend slots in as a third
  `run_*`.

Key types:
- `CpuState` — `crates/cpu/src/state.rs`: `x:[u64;31]`, `sp`, `pc`,
  `flags:Flags`, `v:[u128;32]`, `fpcr`, `excl:Option<(u64,u64)>`,
  `sysregs:BTreeMap`, `el`, `spsel`, `daif`, `sp_el:[u64;4]`.
- `Memory` — `crates/interp/src/memory.rs`: `{ base:u64, bytes:Vec<u8> }`,
  flat, little-endian.
- Dispatch — `crates/interp/src/execute.rs`: `execute(cpu, mem, insn, pc) -> Option<u64>`.
- Loop — `crates/interp/src/run.rs`: fetch (`mmu::translate` + `read_u32`) →
  `decode` → `execute` → PC update.

---

## Milestone 0 — Flat, offset-addressable state (`crates/cpu`)

Generated WASM must read `X5` as an `i64.load` at a *constant offset*. Split
`CpuState` so the hot, raw-memory-addressable fields have a stable layout, and
the non-flat fields stay Rust-side (reachable only via helper calls).

1. Introduce a `#[repr(C)]` hot register block:

   ```rust
   #[repr(C)]
   pub struct GuestRegs {
       pub x:    [u64; 31],   // offset 0
       pub sp:   u64,
       pub pc:   u64,
       pub nzcv: u64,         // PACKED flags (bit31 N, bit30 Z, bit29 C, bit28 V)
       pub v:    [u128; 32],
       pub fpcr: u64,
   }
   ```

   Note `nzcv` is a packed word, **not** the four-bool `Flags`. Keep `Flags`
   for the interpreter's internal use, but add conversions
   `Flags::from_nzcv_word(u64)` / `to_nzcv_word(&self) -> u64` and keep the
   `GuestRegs.nzcv` word as the source of truth the JIT writes.

2. Embed `GuestRegs` in `CpuState` (`regs: GuestRegs`) and keep the cold fields
   (`excl`, `sysregs`, `el`, `spsel`, `daif`, `sp_el`) outside it. Update
   `read_gpr`/`write_gpr`/`read_gpr_w`/`write_gpr_w` and flag access to go
   through `regs` — the **method API stays the same**, only storage moves.

3. Add a documented offset table (a small `const` module or test that asserts
   `offset_of!` values) so the JIT and the runtime agree on layout. A unit test
   must fail if any offset changes.

Acceptance: existing interpreter + difftest suite still green after the refactor
(pure storage move, no behavior change).

---

## Milestone 1 — JIT crate skeleton (`crates/jit`)

Create a new workspace crate `crates/jit` depending on `decoder`, `cpu`,
`interp`, and:
- `wasm-encoder` (emit WASM; do **not** hand-roll LEB128),
- `wasmtime` (embedded runtime).

Deliverables:
- A `Block` formation step: given a start guest PC and a `&Memory`, decode
  forward with the existing `decode()` until a terminator (branch / exception /
  `Insn::Unsupported`). Produce an ordered `Vec<(pc, Insn)>`.
- A `lower` module with one function per instruction class (mirrors
  `execute.rs` arms). For Milestone 1 only the minimal set below.
- An emitter that wraps the lowered instructions into one WASM function with
  the block ABI (Milestone 2) and returns the module bytes.

---

## Milestone 2 — Block ABI & helper imports

Define the contract between generated blocks and the runtime.

**Linear memory layout (in the wasmtime instance):**
- `GuestRegs` image at a fixed offset `REGS_BASE` (e.g. 0).
- Guest RAM region at `RAM_BASE`; guest address `a` maps to linear offset
  `RAM_BASE + (a - Memory.base)`.

**Block function signature:**

```
(func (param $regs_base i32) (result i64))
```

Returns the **next guest PC**. Sequential instructions don't touch PC; the
terminator computes the exit PC. Internal conditional branches become WASM
`if` / `br`.

**Exit / status convention** for cases a block can't handle inline (exception
taken, unsupported instruction, atomics/exclusives, MMU slow path): write an
exit-reason word into a known `GuestRegs`-adjacent field (e.g. add
`exit_reason: u64`, `exit_pc: u64` to the state image) and return a sentinel,
OR encode reason in the high bits of the returned i64. Pick one and document it.

**Imported helpers** (provided by the runtime, callable from generated code):
- `interpret_one(regs_base: i32) -> i64` — runs exactly one instruction at the
  current PC through the existing `execute()` path. **This is the long-tail
  escape hatch**: anything not yet lowered emits a call to it. Requires
  reconstructing `&mut CpuState` / `&mut Memory` over the same linear-memory
  bytes (see Milestone 0/3).
- `mem_translate(regs_base: i32, va: i64) -> i64` — MMU slow path (wraps
  `mmu::translate`).
- `take_exception(regs_base: i32, ...)` — for SVC/faults later.

Bootstrapping: early lowerings may also import and call the existing
`add_with_carry` / `apply_shift` / `eval_cond` to avoid re-implementing flag
logic in WASM on day one.

---

## Milestone 3 — Memory access

Guest RAM is a region of the wasmtime instance's linear memory.

- **Fast path (MMU off / identity, the current default):** lower a guest load
  to `i64.load`/`i32.load` at `RAM_BASE + (va - Memory.base)`; stores
  symmetrically. Little-endian matches WASM natively.
- **Slow path (MMU on, MMIO):** emit a call to `mem_translate`, then access.
  For Milestone 3 it is acceptable to *only* fast-path identity and route
  everything else through `interpret_one`.
- **Trap semantics:** WASM linear-memory OOB traps; ensure that surfaces as a
  guest exception/exit, not a host abort. Match the interpreter's behavior.

Decision to make explicit in code comments: when is inline vs. helper access
chosen. Start conservative (inline only when MMU is known-off).

---

## Milestone 4 — Minimal instruction lowering

Lower the common integer + branch core, **naively** (every register access is a
load/store; flags via helper call or inline packed write; everything else →
`interpret_one`). Target set:

- `MoveWide` (MOVZ/MOVN/MOVK)
- `AddSubImm`, `AddSubShiftedReg` (with and without flags)
- `LogicalImm`, `LogicalShiftedReg`
- `BranchImm` (B, BL)
- `BranchCond` (B.cond — internal `if`/exit)
- `BranchReg` (BR, BLR, RET)
- `CmpBranch` / `TestBranch` (CBZ/CBNZ/TBZ/TBNZ)
- `LoadStore` (immediate offset, identity-mapped) — basic LDR/STR
- `Nop`

Everything else in the block → emit `call interpret_one` and, if it returns a
non-sequential PC or sets an exit reason, end the block.

---

## Milestone 5 — Dispatcher + block cache

New run loop alongside `run.rs` (do not modify `run()` — keep it as the
reference):

```
loop {
  if let Some(f) = cache.get(pc)      { pc = f.call(regs_base) }
  else if is_hot(pc)                  { let f = compile(pc); pc = f.call(...) }
  else                                { pc = interpret_one(...) ; bump_hotness(pc) }
  // honor `until` / `count` stop conditions exactly like run()
}
```

- Cache keyed by guest PC → compiled wasmtime function.
- Invalidate on writes to a compiled code page (self-modifying code). For
  Milestone 5 a coarse "clear cache on any store into a compiled range" is
  acceptable; refine later.
- Block chaining (patching block→block jumps) is a later optimization — omit.

---

## Milestone 6 — Testing (wasmtime, in-process)

Add a JIT backend to `crates/difftest`, mirroring `run_ours`:

```rust
pub fn run_jit(tv: &TestVector) -> (StateSnapshot, StopReason)
```

Steps:
1. Build a linear-memory image from `TestVector` init (`GuestRegs` at
   `REGS_BASE`, code + `init_data` into the RAM region).
2. Compile the block(s) and run via the embedded wasmtime dispatcher
   (Milestone 5) honoring `tv.until()` / `tv.count`.
3. Snapshot `GuestRegs` + touched RAM back into a `StateSnapshot`.

Then:
- **Primary loop:** reuse the existing fuzzers/encoders to diff
  `run_jit` vs `run_ours` (the trusted interpreter) on full architectural
  state. Add `assert_jit_matches_ours(tv)`.
- **Shared-decoder blind spot:** the JIT and interpreter share the decoder, so
  interp-vs-JIT can't catch decoder bugs. Keep periodic three-way checks
  against Unicorn (`run_unicorn`) for the lowered instruction set.
- **Trap parity:** add cases asserting the JIT exits at the *same* PC the
  interpreter does for branches and (later) exceptions.

Acceptance for the first cut: every instruction in the Milestone 4 set, and any
mixed block of them, produces `diff == None` against `run_ours` across the
fuzz corpus; non-lowered instructions correctly fall back via `interpret_one`
and still match.

---

## Sequencing summary

1. M0 — `repr(C)` `GuestRegs` split, offset table, suite still green.
2. M1 — `crates/jit` skeleton, block formation, emitter.
3. M2 — block ABI + helper imports (`interpret_one` first).
4. M3 — identity-mapped memory fast path.
5. M4 — lower the ~12 common integer/branch instructions, naively.
6. M5 — dispatcher + block cache + coarse SMC invalidation.
7. M6 — `run_jit` in difftest; fuzz interp-vs-JIT to green.

Only after M6 is green: register-caching in WASM locals, lazy flags, block
chaining, SIMD/FP lowering, MMU/MMIO inline paths.

## Guiding principles

- **Correct-but-slow first.** Lean on `interpret_one` and imported helpers; a
  block that's mostly helper calls but matches the oracle beats a fast block
  that's wrong.
- **Never change guest-visible semantics.** The interpreter is ground truth;
  the JIT must match it bit-for-bit (GPRs, SP, PC, NZCV, and later V/FPCR/FPSR
  and touched memory).
- **Keep `run()` untouched** as the deterministic reference and fallback.
- **One lowering per existing handler** — use `execute.rs` as the checklist.
