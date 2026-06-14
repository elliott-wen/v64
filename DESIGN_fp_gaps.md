# Design: Floating-point gaps and edge cases

Status: scalar FP **implemented and fuzzed** (FMOV/FABS/FNEG/FSQRT, FCVT single<->
double, FADD/FSUB/FMUL/FDIV/FMAX/FMIN/FMAXNM/FMINNM/FNMUL, FCMP/FCMPE, FCSEL,
SCVTF/UCVTF/FCVTZS/FCVTZU, FMOV gpr<->fpr, FMOV #imm). This note records the
deliberate simplifications so they aren't mistaken for bugs.

## What the harness pins down

The differential harness seeds **FPCR = DN(1), RMode=RN(00), FZ=0**. This makes
results deterministic and matches Rust's native `f32`/`f64`:
- **DN=1** ⇒ any *arithmetic* NaN result is the default NaN, so NaN *payloads*
  never have to match. Bit-ops (FMOV/FABS/FNEG) keep raw bits (correct).
- **RN** ⇒ Rust's round-to-nearest-even matches; **FCVTZ\*** uses round-toward-
  zero, matching Rust's saturating `as` casts (NaN→0, overflow→saturate).
- Signaling-NaN handling is implemented for FMAXNM/FMINNM (sNaN ⇒ Invalid ⇒ NaN);
  FMAX/FMIN propagate any NaN. See `fp.rs`.

## Gaps to close later

1. **FPSR is not compared.** The cumulative exception flags (IOC/DZC/OFC/UFC/IXC)
   and QC are not in the snapshot. Add `FPSR` to `StateSnapshot` and set the
   sticky bits in each op (inexact is the fiddly one). Until then, FP correctness
   is verified on *results*, not on raised exceptions.
2. **Non-default FPCR.** Only DN=1/RN/FZ=0 is exercised. To cover:
   - **Other rounding modes** (RMode 01/10/11): can't use plain Rust `as`/ops;
     need explicit rounding (e.g. via `libm`-style or manual round control).
   - **FZ=1 (flush-to-zero)**: denormal inputs/outputs flush to zero. Add a
     pre/post flush step gated on FPCR.FZ.
   - **DN=0**: NaN-payload propagation rules (which input NaN wins, quieting) —
     mirror QEMU `target/arm/vfp_helper.c`.
3. **Half-precision (FP16, ftype=11).** Decoder rejects it today. Needs a soft
   `f16` (no native Rust type) — implement via `half` crate or manual. The
   encoders avoid generating it, so the validity invariant stays satisfied.
4. **FRINT family** (FRINTN/P/M/Z/A/X/I) and **FCVTNS/AS/PS/MS** (FP→int with a
   specific rounding) — not decoded yet. Straightforward once per-mode rounding
   exists.
5. **FCCMP/FCCMPE** (FP conditional compare) — `fp.rs` router leaves bits[11:10]
   == 01 unimplemented. Mirrors integer CCMP: on condition-false, force NZCV.
6. **FMADD/FMSUB/FNMADD/FNMSUB** (fused multiply-add, a separate 3-source FP
   encoding group) — use `f32::mul_add`/`f64::mul_add` for correct single-
   rounding fused semantics.
7. **FP load/store** (V=1 in the load/store group) — the SIMD/FP `LDR/STR (Sd/Dd/
   Qd)` forms; needs the V-register harness (already built) joined to the memory
   harness. Easy add.

## Why Rust floats are a safe oracle base
With DN/RN/FZ pinned as above, IEEE-754 binary32/binary64 arithmetic is fully
specified and both QEMU softfloat and Rust/host hardware implement it correctly,
so the *only* divergences are the categorized edge cases above — which the
differential fuzzer surfaces precisely (it already caught the FMAXNM sNaN case).
