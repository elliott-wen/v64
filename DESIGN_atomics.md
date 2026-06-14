# Design: Exclusives and LSE atomics

Status: **partially implemented.** The non-exclusive ordered accesses LDAR/STLR
(and byte/half/word forms) are done and fuzzed (`ldst_excl.rs` decoder reuses the
single-register `LoadStore` executor). The true exclusives and the LSE atomic
group are deferred — they need an exclusive-monitor model and *paired-instruction*
testing, which the current single-step fuzzer can't express.

## What's left

1. **Load/store exclusive**: LDXR/LDAXR, STXR/STLXR (byte..dword), and the pair
   forms LDXP/STXP. Same encoding group as LDAR/STLR (`ldst_excl.rs`), selected
   by `o2=0`.
2. **LSE atomics** (ARMv8.1, separate encoding `op0=x1x0`, bits[29:24]=111000
   with bit21=1): LDADD/LDCLR/LDEOR/LDSET/LDSMAX/LDSMIN/LDUMAX/LDUMIN, SWP, and
   compare-and-swap CAS/CASP (plus the A/L acquire/release and B/H size
   variants).

## Exclusive monitor model

STXR semantics depend on a monitor set by a prior LDXR:

```
struct ExclusiveMonitor { addr: Option<u64>, size: u8 }  // in CpuState
```

- `LDXR Xt, [Xn]`: load, set `monitor = Some((aligned_addr, size))`.
- `STXR Ws, Xt, [Xn]`: if `monitor == Some(addr)` → store, `Ws = 0` (success),
  clear monitor; else `Ws = 1` (fail), no store. Real hardware may also fail
  spuriously; for a deterministic interpreter, succeed iff the monitor matches.
- Any store to the monitored region (or a context switch / CLREX) clears it.

Single-step differential testing breaks here: a lone STXR (no preceding LDXR)
sees `monitor == None` and must fail — but Unicorn's monitor state after
`emu_start` of one instruction may differ, and a standalone STXR is not a
meaningful test.

## Testing strategy (paired, not single-step)

Extend the harness with a **multi-instruction** vector for this group:
- Emit a 2-instruction body `LDXR; STXR` (or `LDADD` alone for LSE, which *is*
  single-step-testable since it has no monitor) at `CODE_START`, run with
  `count = 2`, and compare registers + the DATA region against Unicorn.
- For LSE atomics (LDADD/SWP/CAS): these are **single-instruction read-modify-
  write** with no monitor — they fit the existing `MemClass` fuzzer directly.
  Implement these first; they're the common case in modern compiler output
  (`-moutline-atomics` / `-march=armv8.1-a`) and need no new harness.
- For CAS: seed the comparison register sometimes equal to memory (success) and
  sometimes not (failure) so both paths are exercised.

## Recommended order

1. **LSE atomics** (LDADD/SWP/CAS/...) — single-step fuzzable now, high real-world
   value. Add an `Insn::AtomicRmw { op, size, acquire, release, rs, rt, rn }` and
   a `Insn::CompareSwap { .. }`; execute as read-modify-write on `Memory`.
2. **LDXR/STXR** with the monitor + the paired-vector harness extension.
3. **LDXP/STXP** pair exclusives.

## Memory ordering
All of acquire/release/barrier ordering is a no-op in our sequential interpreter
(single thread, in-order). Model the bits for decode completeness but they don't
affect results until multi-core/weak-memory modeling — out of scope.
