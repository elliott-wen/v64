# Design: Advanced SIMD (NEON) and Vector FP

Status: **not implemented.** The scalar-FP work already built the prerequisites
(128-bit `v: [u128; 32]` in `CpuState`, V-register seeding/compare in the
differential harness, FPCR with DN=1). NEON reuses all of it.

## Scope

AArch64 Advanced SIMD operates on the same V0–V31 registers as scalar FP, but
treats each as a **vector of lanes**. The encoding group is `op0 == 0b0111`
(bit28 = 0), which the top-level decoder currently routes to `Unsupported`. (The
existing `fp::decode` handles `op0 == 0b1111`, the scalar forms.)

Element arrangements (`<Vd>.<T>`): `8B/16B`, `4H/8H`, `2S/4S`, `1D/2D`. The `Q`
bit (bit30) selects 64-bit (D, lower half only) vs 128-bit (Q, full register)
operation; `size` (bits[23:22]) selects element width.

## Recommended build order

Each step is independently fuzzable against Unicorn (the harness already compares
full 128-bit V registers), so follow the same implement-then-fuzz loop used for
the integer ISA.

1. **Three-same integer** (`Advanced SIMD three same`, bits[28:24]=0_1110):
   ADD/SUB/MUL/AND/ORR/EOR/ORN/BIC, the compares (CMEQ/CMGT/CMHI...), min/max
   (SMAX/UMAX/SMIN/UMIN), and the shifts (SSHL/USHL). Pure lane-wise integer —
   no FP rounding, so these fuzz cleanly first.
2. **Three-same FP**: FADD/FSUB/FMUL/FDIV/FMAX/FMIN/FMAXNM/FMINNM per lane.
   Reuse the scalar `fp.rs` lane kernels (including the signaling-NaN handling
   in FMAXNM/FMINNM — see the comment there) applied across lanes.
3. **Two-register misc** (bits[28:24]=0_1110, bit21=1, ...): NEG/ABS/CNT/NOT/
   REV*/CLZ/CLS per lane, plus FP FNEG/FABS/FSQRT and the FCVT family.
4. **Copy / permute**: DUP (element & general), INS, SMOV/UMOV, ZIP/UZP/TRN,
   EXT, TBL/TBX.
5. **Modified immediate**: MOVI/MVNI/FMOV-vector/ORR/BIC-immediate — note the
   8-bit `abcdefgh` + `cmode` expansion (`AdvSIMDExpandImm`).
6. **Across-lanes**: ADDV/SADDLV/UADDLV/SMAXV/UMINV/FMAXV...
7. **Pairwise**: ADDP/FADDP/SMAXP...
8. **Scalar SIMD** (`op0 == 0b1111` with bits[28:24] ≠ 11110): the per-element
   scalar forms that sit beside the FP encodings.
9. **SIMD load/store**: LD1–LD4 / ST1–ST4 (single + multiple structures), with
   the de-interleaving. These need the memory harness *and* the V-register
   harness together — straightforward but high lane-count bookkeeping.

## Modeling notes

- Add a `Insn::SimdThreeSame { q, size, opcode, u, rm, rn, rd }` style variant
  per encoding class (keep one decoder file per class, as elsewhere).
- A lane helper module (`simd_lanes.rs`) that splits a `u128` into N lanes of a
  given width, maps a kernel over them, and reassembles — parameterized by
  element width — keeps each instruction file tiny.
- For 64-bit (`Q=0`) ops, compute on the low 8 bytes and **zero the upper 64
  bits** of Vd (same zeroing rule as scalar writes).
- Saturating ops (SQADD/UQADD/...) set FPSR.QC; we don't compare FPSR yet (see
  `DESIGN_fp_gaps.md`), so they'll compare on the result lanes only.

## Fuzzer extension

The `FpClass` encoder kind already produces `(word, init_v, gpr_seeds, fpcr)` and
the harness compares all V registers. NEON needs **no new harness** — just new
encoder functions that emit vector words (randomize `Q`, `size`, lane regs).
Seed FPCR with DN=1 as for scalar FP. Add a `simd` encoder module mirroring
`encoders/fp.rs`.

## Known divergence risks (handle as the fuzzer surfaces them)

- FP lane NaN canonicalization (already solved for scalar — reuse).
- Signed-zero ordering in per-lane FMAX/FMIN (solved for scalar).
- Polynomial multiply (PMUL) and AES/SHA crypto — separate, optional.
