# AArch64 ISA coverage

Status of the interpreter relative to the Unicorn (QEMU `max` CPU) oracle. Every
"implemented" item below is differential-fuzzed against Unicorn and passes.

## Implemented (ARMv8.0-A baseline + selected extensions)

### Integer
- Move-wide (MOVZ/MOVN/MOVK), ADR/ADRP
- Add/sub (immediate, shifted-register, extended-register), with/without flags
- Logical (immediate, shifted-register), bitfield, extract, conditional select
- Conditional compare (CCMP/CCMN), CSEL/CSINC/CSINV/CSNEG
- Multiply (MADD/MSUB/SMADDL/UMADDL/SMULH/UMULH), divide (UDIV/SDIV)
- Variable shifts (LSLV/LSRV/ASRV/RORV), CRC32/CRC32C
- Branches: B/BL/B.cond/CBZ/CBNZ/TBZ/TBNZ/BR/BLR/RET

### Loads / stores
- Single register, all addressing modes (uimm, unscaled, pre/post-index,
  register-offset, literal) for integer **and** SIMD/FP (B/H/S/D/Q)
- Pairs (LDP/STP/LDPSW) integer and SIMD/FP
- LSE atomics (LDADD/LDCLR/LDEOR/LDSET/LDSMAX/.../SWP), CAS, LDXR/STXR, LDAR/STLR
- Advanced SIMD structures: LD1-4/ST1-4 (multiple and single), LD1R-LD4R

### Floating point (scalar)
- FMOV/FABS/FNEG/FSQRT/FCVT, FRINT[NPMZAXI]
- FADD/FSUB/FMUL/FDIV/FNMUL/FMAX/FMIN/FMAXNM/FMINNM
- FMADD/FMSUB/FNMADD/FNMSUB, FCMP/FCMPE, FCCMP/FCCMPE, FCSEL, FMOV-immediate
- Int<->FP conversions: SCVTF/UCVTF, FCVT{N,P,M,Z,A}{S,U}, fixed-point convert

### Advanced SIMD (NEON)
- Three-same integer (full), three-same FP + pairwise + FMLA/FMLS/FMULX/FRECPS/
  FRSQRTS/FACGE/FACGT
- Three-different (widening/narrowing), three-same-extra (SQRDMLAH/SQRDMLSH,
  SDOT/UDOT)
- Two-register-misc integer (full) and FP (FABS/FNEG/FSQRT, FCM*-zero, SCVTF/
  UCVTF, FCVT-to-int, FRINT, FCVTN/FCVTL)
- Across-lanes, copy (DUP/INS/SMOV/UMOV), permute (ZIP/UZP/TRN), EXT, TBL/TBX
- Modified-immediate (MOVI/MVNI/ORR/BIC, FMOV-vector), shift-by-immediate (full)
- By-element (indexed): MUL/MLA/MLS, MLAL/MLSL/MULL, SQDMULL/SQDMLAL/SQDMLSL,
  SQDMULH/SQRDMULH, FMLA/FMLS/FMUL/FMULX, SQRDMLAH/SQRDMLSH, SDOT/UDOT
- Scalar SIMD: three-same, two-reg-misc, pairwise, three-different, copy,
  by-element, shift-by-immediate

### Crypto
- AES (AESE/AESD/AESMC/AESIMC)
- SHA1 (SHA1C/P/M/SU0/H/SU1), SHA256 (SHA256H/H2/SU0/SU1)

### System
- MRS/MSR system registers, SVC/exception vectoring, ERET, SP banking (SP_ELx),
  PSTATE/SPSR, 4 KB-granule MMU translation-table walk

## Deferred (Unicorn `max` supports these; encoders do not yet generate them)

These are optional ARMv8.1+ feature extensions, none required by typical
AArch64 Linux userspace:

- **FEAT_FP16** — half-precision SIMD/scalar FP (large surface; needs a software
  f16). The FP16 forms of FCVTN/FCVTL, indexed FP, three-same FP, etc.
- **FEAT_FCMA** — FCMLA/FCADD complex arithmetic.
- **FEAT_FRINTTS** — FRINT32Z/64Z/32X/64X.
- **Scalar FEAT_RDM** — scalar SQRDMLAH/SQRDMLSH (vector + indexed are done).
- **FCVTXN** (round-to-odd narrowing), **FJCVTZS** (FEAT_JSCVT).
- **v8.3+ crypto** — SHA512, SHA3 (EOR3/RAX1/XAR/BCAX), SM3, SM4.
- **FEAT_PAuth** — pointer authentication (PACIA/AUTIA/...).
- **FEAT_BF16** — bfloat16.
- Unprivileged LDTR/STTR, LDAPR.

## Methodology

`crates/difftest` fuzzes each instruction class against Unicorn, enforcing a
validity invariant in both directions (a divergence, or a decode-agreement
mismatch, fails the test). Run `cargo test -p aarch64-difftest --features
unicorn --test fuzz_sweep`; `FUZZ_ITERS` controls iterations per class.
