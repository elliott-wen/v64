//! Decoded instruction types.
//!
//! One enum variant per instruction class. Fields are pre-resolved at decode
//! time where that keeps the interpreter simple (e.g. logical-immediate masks
//! and bitfield work/tail masks are computed here, not during execution).

/// Logical/arithmetic shift kind for shifted-register data processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftType {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

/// How a load/store computes its effective address (and any base writeback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrMode {
    /// `[Rn, #imm]` — scaled unsigned immediate, no writeback.
    UnsignedImm { rn: u8, imm: u64 },
    /// `[Rn, #simm]` — unscaled signed imm9 (LDUR/STUR), no writeback.
    Unscaled { rn: u8, imm: i64 },
    /// `[Rn, #simm]!` — pre-indexed: base += imm, then access.
    PreIndex { rn: u8, imm: i64 },
    /// `[Rn], #simm` — post-indexed: access, then base += imm.
    PostIndex { rn: u8, imm: i64 },
    /// `[Rn, Rm, ext #shift]` — register offset with extend/scale.
    RegOffset { rn: u8, rm: u8, option: u8, shift: u8 },
    /// `label` — PC-relative literal (loads only).
    Literal { offset: i64 },
}

/// Indexing mode for load/store pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairIndex {
    /// `[Rn, #imm]` — no writeback (also covers the non-allocating hint).
    Offset,
    /// `[Rn, #imm]!` — base += imm before access.
    Pre,
    /// `[Rn], #imm` — base += imm after access.
    Post,
}

/// A decoded instruction. `sf` (set => 64-bit) is carried per-variant where the
/// operand width matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Insn {
    /// MOVZ/MOVN/MOVK. `opc`: 0=MOVN, 2=MOVZ, 3=MOVK.
    MoveWide { sf: bool, opc: u8, hw: u8, imm16: u16, rd: u8 },

    /// ADD/SUB (immediate), optionally setting flags (ADDS/SUBS).
    AddSubImm {
        sf: bool,
        sub: bool,
        set_flags: bool,
        shift12: bool,
        imm12: u16,
        rn: u8,
        rd: u8,
    },

    /// ADD/SUB (shifted register), optionally setting flags.
    AddSubShiftedReg {
        sf: bool,
        sub: bool,
        set_flags: bool,
        shift: ShiftType,
        amount: u8,
        rm: u8,
        rn: u8,
        rd: u8,
    },

    /// ADC/SBC (add/sub with carry), optionally setting flags.
    AddSubCarry { sf: bool, sub: bool, set_flags: bool, rm: u8, rn: u8, rd: u8 },

    /// ADD/SUB (extended register), optionally setting flags.
    AddSubExtReg {
        sf: bool,
        sub: bool,
        set_flags: bool,
        option: u8,
        imm3: u8,
        rm: u8,
        rn: u8,
        rd: u8,
    },

    /// AND/ORR/EOR/ANDS (immediate). `opc`: 0=AND,1=ORR,2=EOR,3=ANDS.
    /// `imm` is the resolved 64-bit bitmask.
    LogicalImm { sf: bool, opc: u8, imm: u64, rn: u8, rd: u8 },

    /// AND/ORR/EOR/ANDS and their inverted forms BIC/ORN/EON (shifted register).
    /// `opc`: 0=AND,1=ORR,2=EOR,3=ANDS; `negate` selects the inverted operand.
    LogicalShiftedReg {
        sf: bool,
        opc: u8,
        negate: bool,
        shift: ShiftType,
        amount: u8,
        rm: u8,
        rn: u8,
        rd: u8,
    },

    /// SBFM/BFM/UBFM. `opc`: 0=SBFM,1=BFM,2=UBFM. Carries the resolved
    /// work/tail masks plus `immr`/`imms` for rotation and the SBFM sign bit.
    Bitfield {
        sf: bool,
        opc: u8,
        wmask: u64,
        tmask: u64,
        immr: u8,
        imms: u8,
        rn: u8,
        rd: u8,
    },

    /// EXTR: `Rd = (Rn:Rm) >> lsb`, low `datasize` bits.
    Extract { sf: bool, rm: u8, rn: u8, lsb: u8, rd: u8 },

    /// CSEL/CSINC/CSINV/CSNEG. `op` and `o2` select the false-branch transform.
    CondSelect { sf: bool, op: bool, o2: bool, cond: u8, rm: u8, rn: u8, rd: u8 },

    /// CCMP/CCMN. `sub` => CCMP (compare), else CCMN. `imm_y` is the rm value or
    /// the 5-bit immediate depending on `is_imm`.
    CondCompare {
        sf: bool,
        sub: bool,
        is_imm: bool,
        imm_y: u8,
        rm: u8,
        cond: u8,
        nzcv: u8,
        rn: u8,
    },

    /// Data processing (1 source): RBIT/REV16/REV32/REV/CLZ/CLS.
    DataProc1Src { sf: bool, opcode: u8, rn: u8, rd: u8 },

    /// Data processing (2 source): UDIV/SDIV/LSLV/LSRV/ASRV/RORV.
    DataProc2Src { sf: bool, opcode: u8, rm: u8, rn: u8, rd: u8 },

    /// Data processing (3 source): MADD/MSUB/SMADDL/SMSUBL/UMADDL/SMULH/UMULH...
    DataProc3Src { sf: bool, op31: u8, o0: bool, rm: u8, ra: u8, rn: u8, rd: u8 },

    /// ADR (page=false) / ADRP (page=true). `imm` is the signed displacement.
    PcRel { page: bool, imm: i64, rd: u8 },

    /// Unconditional branch (immediate): B (link=false) / BL (link=true).
    BranchImm { link: bool, offset: i64 },

    /// Conditional branch (immediate): B.cond.
    CondBranch { cond: u8, offset: i64 },

    /// CBZ/CBNZ. `negate` => CBNZ.
    CompareBranch { sf: bool, negate: bool, rt: u8, offset: i64 },

    /// TBZ/TBNZ. `negate` => TBNZ. `bit` is the tested bit position (0..63).
    TestBranch { bit: u8, negate: bool, rt: u8, offset: i64 },

    /// Unconditional branch (register): `opc` 0=BR, 1=BLR, 2=RET.
    BranchReg { opc: u8, rn: u8 },

    /// Load/store a single register. `size` is log2 of the access width in bytes
    /// (0=byte..3=dword); `signed` sign-extends loaded values; `dst64` selects
    /// the X vs W result width for loads.
    LoadStore {
        size: u8,
        is_load: bool,
        signed: bool,
        dst64: bool,
        rt: u8,
        addr: AddrMode,
    },

    /// LSE atomic read-modify-write (LDADD/LDCLR/LDEOR/LDSET/LDS/UMAX/MIN, SWP).
    /// `op` 0..7 = the RMW kind, 8 = SWP. Loads the old value into Rt.
    AtomicRmw { size: u8, op: u8, rs: u8, rn: u8, rt: u8 },

    /// Compare-and-swap (CAS). If `[Rn] == Rs`, store Rt; Rs always gets the old
    /// memory value.
    CompareSwap { size: u8, rs: u8, rn: u8, rt: u8 },

    /// Load exclusive (LDXR/LDAXR): load and arm the exclusive monitor.
    LoadExclusive { size: u8, rt: u8, rn: u8 },

    /// Store exclusive (STXR/STLXR): store if the monitor is still armed; Ws
    /// receives 0 on success, 1 on failure.
    StoreExclusive { size: u8, rs: u8, rt: u8, rn: u8 },

    /// Load/store pair. `width8` selects 8-byte (X) vs 4-byte (W) elements;
    /// `signed` is LDPSW (4-byte signed elements into X registers). `offset` is
    /// the already-scaled byte displacement.
    LoadStorePair {
        is_load: bool,
        signed: bool,
        width8: bool,
        rt: u8,
        rt2: u8,
        rn: u8,
        offset: i64,
        index: PairIndex,
    },

    /// Scalar FP data-processing, 1 source (FMOV/FABS/FNEG/FSQRT/FCVT).
    /// `ftype`: 0=single, 1=double. `opcode` is bits[20:15].
    FpDataProc1 { ftype: u8, opcode: u8, rn: u8, rd: u8 },

    /// Scalar FP data-processing, 2 source (FADD/FSUB/FMUL/FDIV/FMAX/FMIN/...).
    /// `opcode` is bits[15:12].
    FpDataProc2 { ftype: u8, opcode: u8, rm: u8, rn: u8, rd: u8 },

    /// Scalar FP compare (FCMP/FCMPE), optionally against zero / signaling.
    FpCompare { ftype: u8, rm: u8, rn: u8, cmp_zero: bool, signaling: bool },

    /// Scalar FP conditional select (FCSEL).
    FpCondSelect { ftype: u8, cond: u8, rm: u8, rn: u8, rd: u8 },

    /// Convert between FP and integer, or FMOV gpr<->fpr.
    /// `sf` selects 64-bit GPR; `rmode`/`opcode` are the encoding fields.
    FpCvtInt { sf: bool, ftype: u8, rmode: u8, opcode: u8, rn: u8, rd: u8 },

    /// FMOV (scalar, immediate).
    FpImm { ftype: u8, imm8: u8, rd: u8 },

    /// MRS/MSR (register): move to/from a system register. `read` selects MRS
    /// (sysreg -> Rt) vs MSR (Rt -> sysreg). `key` is the encoded
    /// (op0,op1,CRn,CRm,op2) tuple.
    SysRegMove { read: bool, key: u32, rt: u8 },

    /// MSR (immediate): write a PSTATE field (SPSel / DAIFSet / DAIFClr).
    MsrImm { op1: u8, op2: u8, crm: u8 },

    /// SVC #imm — supervisor call (exception to EL1).
    Svc { imm16: u16 },

    /// ERET — exception return.
    Eret,

    /// Advanced SIMD three-same (vector op on two V registers). `q` selects
    /// 128 vs 64-bit; `u` and `opcode`/`size` select the operation and lane
    /// width.
    SimdThreeSame { q: bool, u: bool, size: u8, opcode: u8, rm: u8, rn: u8, rd: u8 },

    /// Advanced SIMD three-same floating-point (per-lane FADD/FMUL/FMAX/...).
    /// `sz` selects single vs double lanes; `fpopcode` is the 7-bit combined op.
    SimdThreeSameFp { q: bool, sz: bool, fpopcode: u8, rm: u8, rn: u8, rd: u8 },

    /// Advanced SIMD permute: ZIP1/2, UZP1/2, TRN1/2. `opcode` is bits[14:12].
    SimdZipTrn { q: bool, size: u8, opcode: u8, rm: u8, rn: u8, rd: u8 },

    /// Advanced SIMD extract (EXT): bytewise extraction from {Vm:Vn} at `imm4`.
    SimdExt { q: bool, imm4: u8, rm: u8, rn: u8, rd: u8 },

    /// Advanced SIMD shift by immediate (SHL/SSHR/USHR/SSRA/USRA). `immh:immb`
    /// encode the element size and shift amount.
    SimdShiftImm { q: bool, u: bool, immh: u8, immb: u8, opcode: u8, rn: u8, rd: u8 },

    /// Advanced SIMD across-lanes reduction (ADDV/SMAXV/UMAXV/SMINV/UMINV).
    SimdAcrossLanes { q: bool, u: bool, size: u8, opcode: u8, rn: u8, rd: u8 },

    /// Advanced SIMD two-register misc (NEG/ABS/NOT/CNT/CLZ/CLS/REV).
    SimdTwoRegMisc { q: bool, u: bool, size: u8, opcode: u8, rn: u8, rd: u8 },

    /// Advanced SIMD modified immediate (MOVI/MVNI/ORR/BIC). `op`/`cmode` select
    /// how the expanded `imm8` combines with Vd.
    SimdModImm { q: bool, op: bool, cmode: u8, imm8: u8, rd: u8 },

    /// DUP (general register -> all lanes).
    SimdDupGeneral { q: bool, size: u8, rn: u8, rd: u8 },

    /// DUP (element -> all lanes).
    SimdDupElement { q: bool, size: u8, index: u8, rn: u8, rd: u8 },

    /// INS (general register -> one lane).
    SimdInsGeneral { size: u8, index: u8, rn: u8, rd: u8 },

    /// INS (element -> element).
    SimdInsElement { size: u8, dst: u8, src: u8, rn: u8, rd: u8 },

    /// SMOV/UMOV: move a vector element to a GPR. `signed` sign-extends;
    /// `dst64` selects the X vs W destination.
    SimdMovToGpr { signed: bool, dst64: bool, size: u8, index: u8, vn: u8, rd: u8 },

    /// NOP (and the wider hint space we treat as nops for now).
    Nop,

    /// Recognised encoding space but operand/variant not yet implemented.
    Unsupported { word: u32 },
}
