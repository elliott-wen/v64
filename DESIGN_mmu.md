# Design: Memory Management Unit (MMU) and paged memory

Status: **not implemented.** Today `interp::Memory` is a single flat `Vec<u8>`
with a base offset. That's correct for EL0 user-mode where addresses are already
"virtual = what the program sees". An MMU is needed for (a) system-mode emulation
(translating VA→PA through page tables), (b) faithful fault behavior, and (c)
running an OS kernel.

## Current state

`crates/interp/src/memory.rs`:
- `Memory { base, bytes }`, with `read_u8/16/32/64`, `write_u8/...`, panicking on
  out-of-range. No permissions, no translation, no fault signaling.

The load/store executor (`ldst.rs`) computes a 64-bit effective address and calls
`Memory` directly. **This is the single choke point** to route through an MMU.

## Recommended design

### Phase 1 — Paged user memory with permissions (no translation)
Replace the flat `Vec` with a sparse page table of host-backed pages:

```
struct Page { data: Box<[u8; 4096]>, prot: Prot }   // Prot = R|W|X bits
struct Memory { pages: HashMap<u64 /*vpn*/, Page> }
```

- `map(addr, len, prot)`, `unmap`, `protect` — mirror Unicorn's `mem_map`.
- Reads/writes resolve `addr >> 12` to a page; a miss or permission violation
  returns a typed `Fault { kind: ReadUnmapped|WriteUnmapped|FetchUnmapped|Perm,
  addr }` instead of panicking.
- `run`/`execute`/`ldst` thread `Result<_, Fault>` upward; a fault becomes a
  `StopReason::Fault { .. }` (the differential harness already distinguishes
  Unicorn's `*_UNMAPPED` faults from invalid instructions — see `oracle.rs`).

This phase makes the harness's `Outcome::Fault` path testable from our side too,
and lets load/store fuzzing drop the "keep the base in-region" constraint.

### Phase 2 — Address translation (system mode)
Implement the ARMv8-A translation table walk driven by the system registers:
- `TTBR0_EL1`/`TTBR1_EL1` (table base), `TCR_EL1` (granule 4K/16K/64K, T0SZ/T1SZ,
  TG0/TG1), `SCTLR_EL1.M` (MMU enable).
- Multi-level walk (up to 4 levels for 4K/48-bit): index bits per level, read
  descriptor, check valid/table/block/page, accumulate output address + AP/XN
  permissions. Model the AArch64 descriptor format (bits[1:0] type, bits[47:12]
  output addr, AttrIndx, AP[2:1], SH, AF, nG, UXN/PXN).
- A **TLB** keyed on VA→(PA, perms, ASID) to avoid walking every access; flush on
  `TLBI`, TTBR/TCR writes, and ASID change.

QEMU's `target/arm/ptw.c` (in `../unicorn/qemu/target/arm/`) is the reference for
the exact descriptor decoding and fault priority — mirror it.

### Phase 3 — Faults
Translation/permission failures raise a synchronous Data/Instruction Abort:
compose `ESR_EL1` (EC + ISS, DFSC/IFSC status), set `FAR_EL1`, vector through
`VBAR_EL1`. Only needed once exception handling (see `DESIGN_syscalls.md` §4) and
EL1 exist.

## Interaction with the JIT (future)
For the WASM-JIT phase, the MMU is the main correctness/perf tension: either
(a) inline a software TLB lookup in generated code, or (b) for user-mode, use a
flat guest-address space and rely on WASM linear-memory bounds — much faster,
but no per-page perms. Recommend starting JIT against the **flat user model**
and only adding software-TLB codegen if system-mode JIT is needed.

## Testing
- Phase 1: differential vs Unicorn — map/unmap/protect regions, confirm our
  fault kind matches Unicorn's `uc_error` (the `Outcome` split already encodes
  this distinction).
- Phase 2: build page tables in guest memory, set TTBR/TCR/SCTLR, and compare
  translations + access results against Unicorn in MMU mode (Unicorn supports
  AArch64 MMU; `tests/unit/test_arm64.c::test_arm64_mmu` is a worked example).
