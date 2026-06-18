//! JIT ↔ interpreter differential crosscheck, run in node.
//!
//! The JIT only executes under a `WebAssembly` host, so this is the third leg of
//! the trust chain: Unicorn validates the interpreter (the native fuzz sweep),
//! and this validates the JIT *against that interpreter*. Each kernel is run two
//! ways through the same `Machine` — once interpreted, once JIT-organized — and
//! the full architectural state (and RAM) must come out identical.
//!
//! Blocks only compile once *hot* (`JIT_HOTNESS` = 256 executions), so every
//! kernel is a **loop** that runs well past that threshold; the compiled path is
//! then what we compare, asserted via `jit_insns() > 0`. The loop counter lives
//! in `x0` and stopping is by target PC (`until`), so the interpreter and the
//! JIT halt at the same architectural point regardless of how the JIT batches a
//! block.
//!
//! This is a standalone integration test: it carries its own tiny JS JIT backend
//! and `BlockRunner`, depending only on the public `aarch64-platform` API.
//!
//! Run with (runner + growable-table link args come from `.cargo/config.toml`):
//! ```text
//! cargo test -p aarch64-web --target wasm32-unknown-unknown
//! ```

#![cfg(target_arch = "wasm32")]

use aarch64_decoder::{decode, Insn};
use aarch64_platform::{BlockRunner, Board, Clock, Machine, RAM_BASE};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

// --- The JS-side JIT backend (mirrors `crates/web/src/lib.rs`). `jit_compile`
// instantiates a block module against this test module's shared memory and
// appends its `block` export to the indirect function table; *running* is a
// direct in-wasm `call_indirect` (see `TestRunner::run`). ---
#[wasm_bindgen(inline_js = "
let mem = null;
let table = null;
export function jit_set_memory(m) { mem = m; }
export function jit_set_table(t) { table = t; }
export function jit_compile(bytes) {
    const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory: mem } });
    const idx = table.grow(1);
    table.set(idx, inst.exports.block);
    return idx;
}
export function jit_invalidate() {}
")]
extern "C" {
    fn jit_set_memory(mem: JsValue);
    fn jit_set_table(table: JsValue);
    fn jit_compile(bytes: &[u8]) -> u32;
    fn jit_invalidate();
}

/// [`BlockRunner`] over the node `WebAssembly` API; same shape as the production
/// `WasmRunner` — compile through JS, run via in-wasm `call_indirect`.
struct TestRunner;

impl BlockRunner for TestRunner {
    fn compile(&mut self, wasm: &[u8]) -> u32 {
        jit_compile(wasm)
    }
    fn run(&mut self, handle: u32, regs_base: u32, ram_base: u32) -> u64 {
        // A wasm fn pointer *is* the table index, so transmuting yields a
        // callable pointer and the call lowers to `call_indirect`.
        let block: extern "C" fn(u32, u32) -> u64 =
            unsafe { core::mem::transmute(handle as usize) };
        block(regs_base, ram_base)
    }
    fn invalidate(&mut self) {
        jit_invalidate();
    }
}

/// Constant clock — no IRQs/timers are armed in these kernels, so time is inert.
struct ZeroClock;
impl Clock for ZeroClock {
    fn now(&self) -> u64 {
        0
    }
}

// --- Instruction encoders (OR-composed so the field math is self-evident). ---
fn add_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0x8B00_0000 | (rm << 16) | (rn << 5) | rd
}
fn eor_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xCA00_0000 | (rm << 16) | (rn << 5) | rd
}
fn orr_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xAA00_0000 | (rm << 16) | (rn << 5) | rd
}
fn add_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    0x9100_0000 | (imm12 << 10) | (rn << 5) | rd
}
fn subs_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    0xF100_0000 | (imm12 << 10) | (rn << 5) | rd
}
fn madd(rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    0x9B00_0000 | (rm << 16) | (ra << 10) | (rn << 5) | rd
}
fn udiv(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9AC0_0800 | (rm << 16) | (rn << 5) | rd
}
fn sdiv(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9AC0_0C00 | (rm << 16) | (rn << 5) | rd
}
fn udiv_w(rd: u32, rn: u32, rm: u32) -> u32 {
    0x1AC0_0800 | (rm << 16) | (rn << 5) | rd
}
fn sdiv_w(rd: u32, rn: u32, rm: u32) -> u32 {
    0x1AC0_0C00 | (rm << 16) | (rn << 5) | rd
}
fn csel(rd: u32, rn: u32, rm: u32, cond: u32) -> u32 {
    0x9A80_0000 | (rm << 16) | (cond << 12) | (rn << 5) | rd
}
/// STR Xt, [Xn] / LDR Xt, [Xn] (unsigned offset 0).
fn str64(rt: u32, rn: u32) -> u32 {
    0xF900_0000 | (rn << 5) | rt
}
fn ldr64(rt: u32, rn: u32) -> u32 {
    0xF940_0000 | (rn << 5) | rt
}
/// SIMD three-same (Q, U, size, opcode); covers ADD/SUB/MUL/AND/ORR/EOR lanes.
fn three_same(q: u32, u: u32, size: u32, opcode: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    (q << 30) | (u << 29) | (0b01110 << 24) | (size << 22) | (1 << 21) | (rm << 16)
        | (opcode << 11) | (1 << 10) | (rn << 5) | rd
}
/// CBNZ Xt / B.cond, signed byte displacement from the instruction.
fn cbnz(rt: u32, off: i32) -> u32 {
    0xB500_0000 | ((((off >> 2) as u32) & 0x7_ffff) << 5) | rt
}
fn bcond(cond: u32, off: i32) -> u32 {
    0x5400_0000 | ((((off >> 2) as u32) & 0x7_ffff) << 5) | cond
}

fn image(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A bare machine: code at `RAM_BASE`, PC there, MMU off (EL1h), registers seeded.
fn machine(words: &[u32], xs: &[(usize, u64)], vs: &[(usize, u128)]) -> Board {
    let mut board = Board::with_ram_and_clock(1 << 16, Box::new(ZeroClock));
    board.machine.bus.ram_mut().write(RAM_BASE, &image(words));
    board.machine.cpu.pc = RAM_BASE;
    for &(i, v) in xs {
        board.machine.cpu.x[i] = v;
    }
    for &(i, v) in vs {
        board.machine.cpu.v[i] = v;
    }
    board
}

/// Turn on an identity MMU: one L1 block descriptor maps the whole RAM gigabyte
/// VA==PA, so code, page tables, and data are all mapped — the regime the inline
/// memory fast path needs (it bails when the MMU is off).
fn enable_identity_mmu(board: &mut Board) {
    let pt_base = RAM_BASE + 0x2000;
    // 1 GiB block @ RAM_BASE: valid (bit0), block (bit1=0), AF set (bit10),
    // AP=00 => EL1 read/write. VA 0x4000_0000 indexes L1 entry 1.
    let desc: u64 = RAM_BASE | 1 | (1 << 10);
    board.machine.bus.ram_mut().write(pt_base + 8, &desc.to_le_bytes());
    board.machine.cpu.ttbr0_el1 = pt_base;
    board.machine.cpu.tcr_el1 = 25; // T0SZ=25 -> 39-bit VA, walk starts at L1
    board.machine.cpu.sctlr_el1 = 1; // SCTLR.M = MMU on
}

fn assert_arch_state_eq(interp: &Machine, jit: &Machine, ctx: &str) {
    assert_eq!(jit.cpu.x, interp.cpu.x, "X registers ({ctx})");
    assert_eq!(jit.cpu.sp, interp.cpu.sp, "SP ({ctx})");
    assert_eq!(jit.cpu.pc, interp.cpu.pc, "PC ({ctx})");
    assert_eq!(jit.cpu.flags.to_nzcv(), interp.cpu.flags.to_nzcv(), "NZCV ({ctx})");
    assert_eq!(jit.cpu.v, interp.cpu.v, "V registers ({ctx})");
    assert_eq!(jit.cpu.fpcr, interp.cpu.fpcr, "FPCR ({ctx})");
}

/// Run `words` to `exit_off` two ways (interpreter, then JIT-organized) and
/// assert identical architectural state *and* RAM. `prepare` runs on both
/// machines after seeding (e.g. enable the MMU). `until` (a PC, not a count)
/// stops both at the same point; a 2M cap turns a runaway kernel into a loud
/// failure. `ctx` labels failures.
fn crosscheck_with(
    words: &[u32],
    xs: &[(usize, u64)],
    vs: &[(usize, u128)],
    exit_off: u64,
    ctx: &str,
    prepare: impl Fn(&mut Board),
) {
    const CAP: usize = 2_000_000;
    let until = RAM_BASE + exit_off;

    let mut interp = machine(words, xs, vs);
    prepare(&mut interp);
    interp.machine.run(until, CAP);

    let mut jit = machine(words, xs, vs);
    prepare(&mut jit);
    jit_set_memory(wasm_bindgen::memory());
    jit_set_table(wasm_bindgen::function_table());
    jit.machine.run_jit_browser(until, CAP, &mut TestRunner);

    assert_eq!(interp.machine.cpu.pc, until, "interpreter reached exit ({ctx})");
    assert_eq!(jit.machine.cpu.pc, until, "JIT reached exit ({ctx})");
    assert!(jit.machine.jit_insns() > 0, "a hot block compiled and ran ({ctx})");
    assert_arch_state_eq(&interp.machine, &jit.machine, ctx);
    assert!(
        interp.machine.bus.ram_mut().bytes == jit.machine.bus.ram_mut().bytes,
        "RAM differs ({ctx})"
    );
}

fn crosscheck(words: &[u32], xs: &[(usize, u64)], vs: &[(usize, u128)], exit_off: u64, ctx: &str) {
    crosscheck_with(words, xs, vs, exit_off, ctx, |_| {});
}

// ---- Curated kernels: one per lowered family. ----

#[wasm_bindgen_test]
fn arith_logical_flags() {
    let code = [
        add_reg(1, 1, 2),  // x1 += x2
        eor_reg(3, 1, 2),  // x3 = x1 ^ x2
        orr_reg(4, 4, 3),  // x4 |= x3
        subs_imm(0, 0, 1), // x0--, set NZCV
        cbnz(0, -16),      // loop while x0 != 0
    ];
    crosscheck(&code, &[(0, 1000), (1, 0), (2, 0x0123_4567_89AB_CDEF), (3, 0), (4, 0)], &[], 20, "arith");
}

#[wasm_bindgen_test]
fn mul_div_csel() {
    let code = [
        madd(1, 1, 2, 3),  // x1 = x1*x2 + x3
        udiv(5, 1, 7),     // x5 = x1 / x7
        subs_imm(0, 0, 1), // x0--, set flags
        csel(8, 1, 5, 0),  // x8 = (Z) ? x1 : x5   (cond EQ)
        cbnz(0, -16),
    ];
    crosscheck(&code, &[(0, 500), (1, 1), (2, 3), (3, 7), (5, 0), (7, 9), (8, 0)], &[], 20, "muldiv");
}

#[wasm_bindgen_test]
fn bcond_loop() {
    let code = [
        add_reg(1, 1, 2),  // x1 += x2
        subs_imm(0, 0, 1), // x0--, set flags
        bcond(0b0001, -8), // B.NE loop
    ];
    crosscheck(&code, &[(0, 800), (1, 0), (2, 5)], &[], 12, "bcond");
}

#[wasm_bindgen_test]
fn simd_vector_add() {
    let code = [
        three_same(1, 0, 0b10, 0b10000, 0, 0, 1), // add v0.4s, v0.4s, v1.4s
        subs_imm(0, 0, 1),
        cbnz(0, -8),
    ];
    let v1 = 0x0000_0001_0000_0002_0000_0003_0000_0004u128; // lanes [4,3,2,1]
    crosscheck(&code, &[(0, 600)], &[(0, 0), (1, v1)], 12, "simd_add");
}

#[wasm_bindgen_test]
fn division_edge_cases() {
    // AArch64 defines x/0 == 0 and SDIV(INT_MIN, -1) == INT_MIN; wasm `i64.div_*`
    // *traps* on both, so the lowering must guard them. A bug here makes the
    // compiled block trap (caught as a JIT-vs-interp divergence / failure).
    let code = [
        sdiv(5, 1, 2), // INT_MIN / -1  -> INT_MIN
        udiv(6, 4, 3), // 100 / 0       -> 0
        sdiv(7, 4, 3), // 100 / 0       -> 0
        subs_imm(0, 0, 1),
        cbnz(0, -16),
    ];
    let xs = [
        (0, 500u64),
        (1, 0x8000_0000_0000_0000),
        (2, 0xFFFF_FFFF_FFFF_FFFF),
        (3, 0),
        (4, 100),
        (5, 0),
        (6, 0),
        (7, 0),
    ];
    crosscheck(&code, &xs, &[], 20, "division_edge");
}

#[wasm_bindgen_test]
fn division_edge_cases_w() {
    // Same guards at 32-bit width: W INT_MIN/-1 and W x/0. Results zero-extend to
    // 64 bits, so the lowering must mask to 32 bits *and* guard the trap cases.
    let code = [
        sdiv_w(5, 1, 2), // (i32)0x8000_0000 / -1 -> 0x8000_0000 (zero-ext to X)
        udiv_w(6, 4, 3), // 100 / 0 -> 0
        sdiv_w(7, 4, 3), // 100 / 0 -> 0
        subs_imm(0, 0, 1),
        cbnz(0, -16),
    ];
    let xs = [
        (0, 500u64),
        (1, 0x8000_0000),       // W INT_MIN
        (2, 0xFFFF_FFFF),       // W -1
        (3, 0),
        (4, 100),
        (5, 0),
        (6, 0),
        (7, 0),
    ];
    crosscheck(&code, &xs, &[], 20, "division_edge_w");
}

#[wasm_bindgen_test]
fn memory_loads_stores_mmu_on() {
    let data = RAM_BASE + 0x4000;
    let code = [
        str64(2, 1),       // [x1] = x2
        ldr64(3, 1),       // x3 = [x1]
        add_imm(2, 2, 1),  // x2++
        subs_imm(0, 0, 1),
        cbnz(0, -16),
    ];
    crosscheck_with(
        &code,
        &[(0, 600), (1, data), (2, 0xAAAA), (3, 0)],
        &[],
        20,
        "memory",
        enable_identity_mmu,
    );
}

// ---- Randomized-operand sweep over the lowered families. ----

/// xorshift64* — deterministic so any failure is reproducible from the seed.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }
}

/// A GPR in x1..x15 — never x0 (the loop counter) so the random op can't clobber
/// control flow.
fn greg(rng: &mut Rng) -> u32 {
    1 + rng.below(15)
}
/// A vector reg in v0..v15.
fn vreg(rng: &mut Rng) -> u32 {
    rng.below(16)
}

/// A random valid integer data-processing instruction over safe registers
/// (x1..x15), at 32- or 64-bit width, spanning arith/logical/carry, mul/div,
/// variable shifts, bitfield, EXTR, and 1-source ops. (Any rare invalid
/// width/opcode combo is dropped by the decoder filter in the sweep.)
fn rand_dp(rng: &mut Rng) -> u32 {
    let (rd, rn, rm) = (greg(rng), greg(rng), greg(rng));
    let w = rng.below(2) == 0; // exercise the 32-bit width too
    // Pick the X or W base for a two-register op.
    let two = |xb: u32, wb: u32| (if w { wb } else { xb }) | (rm << 16) | (rn << 5) | rd;
    match rng.below(21) {
        0 => two(0x8B00_0000, 0x0B00_0000),  // ADD
        1 => two(0xCB00_0000, 0x4B00_0000),  // SUB
        2 => two(0x8A00_0000, 0x0A00_0000),  // AND
        3 => two(0xAA00_0000, 0x2A00_0000),  // ORR
        4 => two(0xCA00_0000, 0x4A00_0000),  // EOR
        5 => two(0xAB00_0000, 0x2B00_0000),  // ADDS
        6 => two(0xEB00_0000, 0x6B00_0000),  // SUBS
        7 => two(0x9A00_0000, 0x1A00_0000),  // ADC
        8 => two(0xDA00_0000, 0x5A00_0000),  // SBC
        9 => (if w { 0x1B00_0000 } else { 0x9B00_0000 }) | (rm << 16) | (greg(rng) << 10) | (rn << 5) | rd, // MADD
        10 => (if w { 0x1B00_8000 } else { 0x9B00_8000 }) | (rm << 16) | (greg(rng) << 10) | (rn << 5) | rd, // MSUB
        11 => two(0x9AC0_0800, 0x1AC0_0800), // UDIV (÷0 -> 0, defined)
        12 => two(0x9AC0_0C00, 0x1AC0_0C00), // SDIV
        13 => two(0x9AC0_2000, 0x1AC0_2000), // LSLV
        14 => two(0x9AC0_2400, 0x1AC0_2400), // LSRV
        15 => two(0x9AC0_2800, 0x1AC0_2800), // ASRV
        16 => two(0x9AC0_2C00, 0x1AC0_2C00), // RORV
        17 => (if w { 0x1A80_0000 } else { 0x9A80_0000 }) | (rm << 16) | (rng.below(16) << 12) | (rn << 5) | rd, // CSEL
        18 => {
            // Bitfield: UBFM / SBFM / BFM. 64-bit sets N=1 (the 0x_x340 bases);
            // 32-bit clears it, and immr/imms are 0..31.
            let bits = if w { 32 } else { 64 };
            let (immr, imms) = (rng.below(bits), rng.below(bits));
            let base = if w {
                [0x5300_0000u32, 0x1300_0000, 0x3300_0000]
            } else {
                [0xD340_0000u32, 0x9340_0000, 0xB340_0000]
            }[rng.below(3) as usize];
            base | (immr << 16) | (imms << 10) | (rn << 5) | rd
        }
        19 => {
            // EXTR (64-bit N=1 / 32-bit N=0), `imms` = lsb in 0..width.
            let bits = if w { 32 } else { 64 };
            (if w { 0x1380_0000 } else { 0x93C0_0000 }) | (rm << 16) | (rng.below(bits) << 10) | (rn << 5) | rd
        }
        _ => {
            // Data-proc 1-source: RBIT/REV16/REV(32)/REV64/CLZ/CLS. REV64 (op 3)
            // is X-only; the decoder filter drops it at W width.
            let op = [0u32, 1, 2, 3, 4, 5][rng.below(6) as usize];
            (if w { 0x5AC0_0000 } else { 0xDAC0_0000 }) | (op << 10) | (rn << 5) | rd
        }
    }
}

/// `copy_word` / `imm5_for`: the DUP/INS/UMOV/SMOV "advanced SIMD copy" layout
/// (matches the difftest encoders, which are validated against Unicorn).
fn copy_word(q: u32, op: u32, imm5: u32, imm4: u32, rn: u32, rd: u32) -> u32 {
    (q << 30) | (op << 29) | (0b01110000 << 21) | (imm5 << 16) | (imm4 << 11) | (1 << 10) | (rn << 5) | rd
}
fn imm5_for(size: u32, index: u32) -> u32 {
    (index << (size + 1)) | (1 << size)
}

/// A random valid SIMD instruction over safe registers: three-same,
/// two-register-misc, shift-by-immediate, widening multiply (three-diff),
/// modified-immediate, copy (DUP/INS/UMOV/SMOV), and permute (ZIP/TRN/UZP/EXT).
/// V operands are v0..v15; the UMOV/SMOV destination GPR is x1..x15 so it can't
/// touch the x0 loop counter.
fn rand_simd(rng: &mut Rng) -> u32 {
    let (rd, rn, rm) = (vreg(rng), vreg(rng), vreg(rng));
    let q = rng.below(2);
    match rng.below(8) {
        3 => {
            // three-diff widening multiply SMULL/UMULL (Q selects source half).
            let u = rng.below(2);
            let size = rng.below(3);
            (q << 30) | (u << 29) | (0b01110 << 24) | (size << 22) | (1 << 21)
                | (rm << 16) | (0b1100 << 12) | (rn << 5) | rd
        }
        4 => {
            // modified-immediate: MOVI/MVNI/ORR/BIC-imm (cmode + op select).
            let mut q = q;
            let op = rng.below(2);
            let cmode = rng.below(16);
            if cmode == 0b1111 && op == 1 {
                q = 1; // FMOV-vector form needs Q=1
            }
            let imm8 = rng.below(256);
            (q << 30) | (op << 29) | (0b01111 << 24) | ((imm8 >> 5) << 16)
                | (cmode << 12) | (0b01 << 10) | ((imm8 & 0x1f) << 5) | rd
        }
        5 => {
            // DUP (element): broadcast one lane of Vn.
            let size = if q == 1 { rng.below(4) } else { rng.below(3) };
            let index = rng.below(16 >> size);
            copy_word(q, 0, imm5_for(size, index), 0b0000, rn, rd)
        }
        6 => {
            // INS general (GPR -> lane) or INS element (lane -> lane), Q=1.
            let size = rng.below(4);
            let dst = rng.below(16 >> size);
            let imm5 = imm5_for(size, dst);
            if rng.below(2) == 0 {
                copy_word(1, 0, imm5, 0b0011, greg(rng), rd) // INS (general): reads a GPR
            } else {
                let src = rng.below(16 >> size);
                copy_word(1, 1, imm5, (src << size) & 0xf, rn, rd) // INS (element)
            }
        }
        7 => {
            // UMOV/SMOV: lane -> GPR. Destination is a safe GPR (x1..x15).
            let signed = rng.below(2) == 0;
            let q2 = rng.below(2);
            let size = match (signed, q2 == 1) {
                (true, true) => rng.below(3),  // SMOV Xd: B/H/S
                (true, false) => rng.below(2), // SMOV Wd: B/H
                (false, true) => 3,            // UMOV Xd: D
                (false, false) => rng.below(3),// UMOV Wd: B/H/S
            };
            let index = rng.below(16 >> size);
            let imm4 = if signed { 0b0101 } else { 0b0111 };
            copy_word(q2, 0, imm5_for(size, index), imm4, vreg(rng), greg(rng))
        }
        _ => rand_simd_three_same_misc_shift(rng, rd, rn, rm, q),
    }
}

/// The first three SIMD families (kept separate so `rand_simd` stays readable).
fn rand_simd_three_same_misc_shift(rng: &mut Rng, rd: u32, rn: u32, rm: u32, q: u32) -> u32 {
    match rng.below(4) {
        3 => match rng.below(3) {
            // permute: ZIP1/2, UZP1/2, TRN1/2.
            0 => {
                let opcode = [0b001u32, 0b010, 0b011, 0b101, 0b110, 0b111][rng.below(6) as usize];
                let size = if q == 1 { rng.below(4) } else { rng.below(3) };
                (q << 30) | (0b01110 << 24) | (size << 22) | (rm << 16)
                    | (opcode << 12) | (1 << 11) | (rn << 5) | rd
            }
            // EXT: byte extract.
            1 => {
                let imm4 = if q == 1 { rng.below(16) } else { rng.below(8) };
                (q << 30) | (1 << 29) | (0b01110 << 24) | (rm << 16) | (imm4 << 11) | (rn << 5) | rd
            }
            // TBL, single table register (len=0, op=0); the only inlined table form.
            _ => (q << 30) | (0b001110 << 24) | (rm << 16) | (rn << 5) | rd,
        },
        0 => {
            // three-same: the full bit-exact handled set — ADD/SUB (2D needs Q=1),
            // MUL, all eight logical (incl. BSL/BIT/BIF/BIC/ORN), CMEQ/CMTST,
            // saturating add/sub (8/16-bit), CMGT/CMHI, CMGE, S/UMAX, S/UMIN.
            let mut q = q;
            let sz4 = |q: &mut u32, rng: &mut Rng| {
                let s = rng.below(4);
                if s == 3 {
                    *q = 1;
                }
                s
            };
            let (u, size, opcode) = match rng.below(13) {
                0 => (0, sz4(&mut q, rng), 0b10000),         // ADD
                1 => (1, sz4(&mut q, rng), 0b10000),         // SUB
                2 => (0, 1 + rng.below(2), 0b10011),         // MUL (size 1/2)
                3 => (rng.below(2), rng.below(4), 0b00011),  // logical (all 8)
                4 => (1, sz4(&mut q, rng), 0b10001),         // CMEQ
                5 => (0, sz4(&mut q, rng), 0b10001),         // CMTST
                6 => (rng.below(2), rng.below(2), 0b00001),  // SQADD/UQADD (8/16-bit)
                7 => (rng.below(2), rng.below(2), 0b00101),  // SQSUB/UQSUB (8/16-bit)
                8 => (0, sz4(&mut q, rng), 0b00110),         // CMGT (signed, any size)
                9 => (1, rng.below(3), 0b00110),             // CMHI (unsigned, size 0-2)
                10 => (0, sz4(&mut q, rng), 0b00111),        // CMGE (signed, any size)
                11 => (rng.below(2), rng.below(3), 0b01100), // S/UMAX (size 0-2)
                _ => (rng.below(2), rng.below(3), 0b01101),  // S/UMIN (size 0-2)
            };
            three_same(q, u, size, opcode, rd, rn, rm)
        }
        1 => {
            // two-register-misc: NEG/ABS/NOT/CNT, compare-to-zero
            // CMGT/CMGE/CMEQ/CMLE/CMLT #0, and REV64/REV16.
            let mut q = q;
            let sz4 = |q: &mut u32, rng: &mut Rng| {
                let s = rng.below(4);
                if s == 3 {
                    *q = 1;
                }
                s
            };
            let (u, opcode, size) = match rng.below(11) {
                0 => (1, 0b01011, sz4(&mut q, rng)), // NEG
                1 => (0, 0b01011, sz4(&mut q, rng)), // ABS
                2 => (1, 0b00101, 0),                // NOT
                3 => (0, 0b00101, 0),                // CNT
                4 => (0, 0b01001, sz4(&mut q, rng)), // CMEQ #0
                5 => (0, 0b01000, sz4(&mut q, rng)), // CMGT #0
                6 => (1, 0b01000, sz4(&mut q, rng)), // CMGE #0
                7 => (1, 0b01001, sz4(&mut q, rng)), // CMLE #0
                8 => (0, 0b01010, sz4(&mut q, rng)), // CMLT #0
                9 => (0, 0b00000, rng.below(3)),     // REV64 (size 0..2)
                _ => (0, 0b00001, rng.below(2)),     // REV16 (size 0..1)
            };
            (q << 30) | (u << 29) | (0b01110 << 24) | (size << 22)
                | (0b10000 << 17) | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
        }
        _ => {
            // shift-by-immediate across element sizes: SHL / SSHR / USHR. immh
            // selects the size (64-bit needs Q=1); immh:immb keeps the shift
            // amount in [0, esize) for SHL and [1, esize] for the right shifts.
            let mut q = q;
            let (u, opcode) = [(0u32, 0b01010u32), (0, 0b00000), (1, 0b00000)][rng.below(3) as usize];
            let size = rng.below(4);
            if size == 3 {
                q = 1;
            }
            let immh = (1u32 << size) | rng.below(1u32 << size);
            let immb = rng.below(8);
            (q << 30) | (u << 29) | (0b011110 << 23) | (immh << 19) | (immb << 16)
                | (opcode << 11) | (1 << 10) | (rn << 5) | rd
        }
    }
}

/// A random scalar FP instruction (S/D): dp2/dp1/dp3, compare, csel, and
/// int<->fp conversions. FP isn't lowered yet (it bails to the interpreter, so
/// these pass today); once a lowering lands this exercises it for real. GPR
/// destinations/sources use x1..x15 so the x0 counter is never disturbed.
fn rand_fp(rng: &mut Rng) -> u32 {
    let ftype = rng.below(2); // 0 = single, 1 = double
    let (rd, rn, rm) = (vreg(rng), vreg(rng), vreg(rng));
    const HDR: u32 = 0b0001_1110 << 24;
    match rng.below(7) {
        0 => {
            // dp2: FMUL/FDIV/FADD/FSUB/FMAX/FMIN/FMAXNM/FMINNM/FNMUL
            let opcode = rng.below(9);
            HDR | (ftype << 22) | (1 << 21) | (rm << 16) | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
        }
        1 => {
            // dp1: FMOV/FABS/FNEG/FSQRT/FRINT{N,P,M,Z,A,X,I}
            let opcode = [0u32, 1, 2, 3, 0x8, 0x9, 0xa, 0xb, 0xc, 0xe, 0xf][rng.below(11) as usize];
            HDR | (ftype << 22) | (1 << 21) | (opcode << 15) | (0b10000 << 10) | (rn << 5) | rd
        }
        2 => {
            // dp3: FMADD/FMSUB/FNMADD/FNMSUB
            let (o1, o0) = (rng.below(2), rng.below(2));
            (0b0011111 << 24) | (ftype << 22) | (o1 << 21) | (rm << 16) | (o0 << 15) | (vreg(rng) << 10) | (rn << 5) | rd
        }
        3 => {
            // FCMP / FCMPE (reg or #0.0). Sets NZCV; the loop's SUBS overwrites
            // flags before the CBNZ, so it doesn't disturb control flow.
            let (cmp_zero, sig) = (rng.below(2), rng.below(2));
            let opcode2 = (sig << 4) | (cmp_zero << 3);
            let rmf = if cmp_zero == 1 { 0 } else { rm };
            HDR | (ftype << 22) | (1 << 21) | (rmf << 16) | (0b1000 << 10) | (rn << 5) | opcode2
        }
        4 => {
            // FCSEL
            let cond = rng.below(16);
            HDR | (ftype << 22) | (1 << 21) | (rm << 16) | (cond << 12) | (0b11 << 10) | (rn << 5) | rd
        }
        5 => {
            // SCVTF / UCVTF (GPR -> FP): reads a GPR (x1..x15).
            let opcode = if rng.below(2) == 0 { 0b010 } else { 0b011 };
            let sf = rng.below(2);
            (sf << 31) | (0b0011110 << 24) | (ftype << 22) | (1 << 21) | (opcode << 16) | (greg(rng) << 5) | rd
        }
        _ => {
            // FCVTZS / FCVTZU (FP -> GPR, round to zero): writes a GPR (x1..x15).
            let opcode = if rng.below(2) == 0 { 0b000 } else { 0b001 };
            let sf = rng.below(2);
            (sf << 31) | (0b0011110 << 24) | (ftype << 22) | (1 << 21) | (0b11 << 19) | (opcode << 16) | (rn << 5) | greg(rng)
        }
    }
}

#[wasm_bindgen_test]
fn random_lowering_sweep() {
    const TARGET: u32 = 1200;
    let mut rng = Rng::new(0xC0FFEE_1234);
    let (mut tested, mut tries) = (0u32, 0u32);
    while tested < TARGET && tries < TARGET * 20 {
        tries += 1;
        // 0/1 = integer DP, 2/3 = SIMD, 4 = scalar FP.
        let kind = rng.below(5);
        let (w, fp) = match kind {
            0 | 1 => (rand_dp(&mut rng), false),
            2 | 3 => (rand_simd(&mut rng), false),
            _ => (rand_fp(&mut rng), true),
        };
        // Never stall the loop on a word our own decoder rejects.
        if matches!(decode(w), Insn::Unsupported { .. }) {
            continue;
        }
        // [W ; subs x0,#1 ; cbnz x0] — W over x1..x15 / v0..v15 can't touch x0.
        let code = [w, subs_imm(0, 0, 1), cbnz(0, -8)];

        let mut xs = vec![(0usize, 400u64)]; // counter past JIT_HOTNESS
        for r in 1..16usize {
            xs.push((r, rng.next()));
        }
        let mut vs = Vec::new();
        for r in 0..16usize {
            vs.push((r, (u128::from(rng.next()) << 64) | u128::from(rng.next())));
        }
        let ctx = format!("sweep #{tested} kind={kind} w={w:#010x}");
        // FP wants default-NaN mode (DN=1) so Rust-float and wasm-float results
        // agree bit-for-bit once a lowering exists.
        if fp {
            crosscheck_with(&code, &xs, &vs, 12, &ctx, |b| b.machine.cpu.fpcr = 1 << 25);
        } else {
            crosscheck(&code, &xs, &vs, 12, &ctx);
        }
        tested += 1;
    }
    assert!(tested >= TARGET, "sweep produced too few decodable instructions ({tested})");
}
