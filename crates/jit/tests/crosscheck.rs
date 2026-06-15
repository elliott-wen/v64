//! Differential cross-check: run a block through the JIT and through the
//! interpreter from identical state and require bit-identical results.
//!
//! The JIT and interpreter share the decoder, so this validates the *lowering*,
//! not the decode. Each program here is fully inline-lowerable, so we also
//! assert `interp_calls() == 0` — proving the inline path was taken rather than
//! silently falling back to `interpret_one` (which would match trivially).

use aarch64_cpu_state::{CpuState, GuestRegs};
use aarch64_interp::{step, Memory, Step};
use aarch64_jit::{abi, form_block, Vm};

const BASE: u64 = 0x1000;
const RAM: usize = 0x10000;

// --- tiny instruction encoders (64-bit forms unless noted) -----------------

const fn movz(rd: u32, imm16: u32, hw: u32) -> u32 {
    0xD280_0000 | (hw << 21) | (imm16 << 5) | rd
}
const fn add_sub_imm(sub: u32, s: u32, rd: u32, rn: u32, imm12: u32) -> u32 {
    0x9100_0000 | (sub << 30) | (s << 29) | (imm12 << 10) | (rn << 5) | rd
}
const fn add_sub_sr(sub: u32, s: u32, rd: u32, rn: u32, rm: u32, shift: u32, amt: u32) -> u32 {
    0x8B00_0000 | (sub << 30) | (s << 29) | (shift << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd
}
const fn add_sub_ext(sub: u32, s: u32, rd: u32, rn: u32, rm: u32, option: u32, imm3: u32) -> u32 {
    0x8B20_0000 | (sub << 30) | (s << 29) | (rm << 16) | (option << 13) | (imm3 << 10) | (rn << 5) | rd
}
const fn adc(sub: u32, s: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    0x9A00_0000 | (sub << 30) | (s << 29) | (rm << 16) | (rn << 5) | rd
}
const fn logical_sr(opc: u32, n: u32, rd: u32, rn: u32, rm: u32, shift: u32, amt: u32) -> u32 {
    0x8A00_0000 | (opc << 29) | (shift << 22) | (n << 21) | (rm << 16) | (amt << 10) | (rn << 5) | rd
}
const fn dp2(opcode: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    0x9AC0_0000 | (rm << 16) | (opcode << 10) | (rn << 5) | rd
}
const fn dp1(opcode: u32, rd: u32, rn: u32) -> u32 {
    0xDAC0_0000 | (opcode << 10) | (rn << 5) | rd
}
const fn madd(o0: u32, rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    0x9B00_0000 | (o0 << 15) | (rm << 16) | (ra << 10) | (rn << 5) | rd
}
const fn csel(op: u32, o2: u32, rd: u32, rn: u32, rm: u32, cond: u32) -> u32 {
    0x9A80_0000 | (op << 30) | (rm << 16) | (cond << 12) | (o2 << 10) | (rn << 5) | rd
}
const fn ccmp_imm(sub: u32, rn: u32, imm5: u32, cond: u32, nzcv: u32) -> u32 {
    // CCMP/CCMN immediate, 64-bit.
    0x3A40_0800 | (sub << 30) | (imm5 << 16) | (cond << 12) | (rn << 5) | nzcv
}
const fn extr(rd: u32, rn: u32, rm: u32, lsb: u32) -> u32 {
    0x93C0_0000 | (rm << 16) | (lsb << 10) | (rn << 5) | rd
}
const fn ubfm(opc: u32, rd: u32, rn: u32, immr: u32, imms: u32) -> u32 {
    // 64-bit bitfield (N=1). opc: 0=SBFM,1=BFM,2=UBFM.
    0x9300_0000 | (opc << 29) | (1 << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd
}
const fn adr(page: u32, rd: u32, imm: i32) -> u32 {
    let imm = (imm as u32) & 0x1f_ffff;
    (page << 31) | ((imm & 3) << 29) | 0x1000_0000 | ((imm >> 2) << 5) | rd
}
const fn ldr_uimm(rd: u32, rn: u32, imm12: u32) -> u32 {
    0xF940_0000 | (imm12 << 10) | (rn << 5) | rd
}
const fn str_uimm(rt: u32, rn: u32, imm12: u32) -> u32 {
    0xF900_0000 | (imm12 << 10) | (rn << 5) | rt
}
const fn stp(rt: u32, rt2: u32, rn: u32, imm7: u32) -> u32 {
    0xA900_0000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt
}
const fn ldp(rt: u32, rt2: u32, rn: u32, imm7: u32) -> u32 {
    0xA940_0000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt
}
const RET: u32 = 0xD65F_03C0;
const NOP: u32 = 0xD503_201F;

// --- harness ----------------------------------------------------------------

fn reader(prog: &[u32]) -> impl Fn(u64) -> u32 + '_ {
    move |pc| prog[((pc - BASE) / 4) as usize]
}

fn run_interp(prog: &[u32], init: &GuestRegs, steps: usize) -> (GuestRegs, Vec<u8>) {
    let mut cpu = CpuState::new();
    cpu.load_guest_regs(init);
    let mut mem = Memory::new(BASE, RAM);
    for (i, w) in prog.iter().enumerate() {
        mem.write(BASE + 4 * i as u64, &w.to_le_bytes());
    }
    for _ in 0..steps {
        if let Step::Unsupported { pc, word } = step(&mut cpu, &mut mem) {
            panic!("interp hit unsupported {word:#010x} at {pc:#x}");
        }
    }
    (cpu.to_guest_regs(), mem.bytes)
}

fn run_jit(prog: &[u32], init: &GuestRegs) -> (GuestRegs, Vec<u8>, u64) {
    let mut vm = Vm::new(BASE, RAM);
    for (i, w) in prog.iter().enumerate() {
        vm.write_ram(BASE + 4 * i as u64, &w.to_le_bytes());
    }
    vm.load_regs(init);
    let block = form_block(BASE, reader(prog));
    let exit = vm.run_block(&block);
    assert_eq!(exit.exit_reason, abi::EXIT_NONE, "unexpected exit {:#x}", exit.exit_reason);
    (vm.store_regs(), vm.read_ram(BASE, RAM), vm.interp_calls())
}

/// Run `prog` (which must end in a single terminator) through both engines from
/// `init` and assert identical architectural state and that nothing fell back.
fn check(name: &str, prog: &[u32], init: &GuestRegs) {
    let block_len = form_block(BASE, reader(prog)).insns.len();
    let (ir, imem) = run_interp(prog, init, block_len);
    let (jr, jmem, calls) = run_jit(prog, init);

    assert_eq!(calls, 0, "[{name}] expected fully-inline block, but {calls} interp_one call(s)");
    assert_eq!(jr.x, ir.x, "[{name}] X registers diverge\n jit={:x?}\n int={:x?}", jr.x, ir.x);
    assert_eq!(jr.sp, ir.sp, "[{name}] SP diverges");
    assert_eq!(jr.pc, ir.pc, "[{name}] PC diverges");
    assert_eq!(jr.nzcv, ir.nzcv, "[{name}] NZCV diverges (jit={:#x} int={:#x})", jr.nzcv, ir.nzcv);
    assert_eq!(jmem, imem, "[{name}] memory diverges");
}

/// Deterministic register seeds (no rand dependency).
fn seeds() -> Vec<GuestRegs> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s
    };
    let mut out = Vec::new();
    for _ in 0..8 {
        let mut r = GuestRegs::default();
        for x in r.x.iter_mut() {
            *x = next();
        }
        r.sp = BASE + 0x800; // a valid, aligned in-window stack pointer
        r.pc = BASE;
        r.nzcv = (next() & 0xf) << 28; // exercise flag-reading instructions
        out.push(r);
    }
    out
}

/// Append a terminator and run the program against every seed.
fn check_all(name: &str, body: &[u32]) {
    let mut prog = body.to_vec();
    prog.push(RET);
    for (i, seed) in seeds().into_iter().enumerate() {
        check(&format!("{name}#{i}"), &prog, &seed);
    }
}

#[test]
fn arithmetic_imm_and_reg() {
    check_all("add_sub_imm", &[
        add_sub_imm(0, 0, 0, 1, 0x123),  // add  x0, x1, #0x123
        add_sub_imm(1, 0, 2, 3, 0xfff),  // sub  x2, x3, #0xfff
        add_sub_imm(0, 1, 4, 5, 0x010),  // adds x4, x5, #0x10
        add_sub_imm(1, 1, 6, 7, 0x001),  // subs x6, x7, #1
    ]);
    check_all("add_sub_shifted", &[
        add_sub_sr(0, 0, 0, 1, 2, 0, 0),   // add  x0, x1, x2
        add_sub_sr(0, 0, 3, 4, 5, 0, 7),   // add  x3, x4, x5, lsl #7
        add_sub_sr(1, 1, 6, 7, 0, 1, 3),   // subs x6, x7, x0, lsr #3
        add_sub_sr(1, 1, 1, 2, 3, 2, 5),   // subs x1, x2, x3, asr #5
        // (ROR is reserved for add/sub shifted-register, so it is not tested here.)
    ]);
    check_all("add_sub_extended", &[
        add_sub_ext(0, 0, 0, 1, 2, 0, 0),  // add x0, x1, w2, uxtb
        add_sub_ext(1, 1, 3, 4, 5, 6, 2),  // subs x3, x4, w5, sxtw #2
        add_sub_ext(0, 1, 6, 7, 0, 3, 4),  // adds x6, x7, x0, uxtx #4
    ]);
    check_all("adc_sbc", &[
        adc(0, 0, 0, 1, 2), // adc  x0, x1, x2
        adc(1, 1, 3, 4, 5), // sbcs x3, x4, x5
        adc(0, 1, 6, 7, 0), // adcs x6, x7, x0
    ]);
}

#[test]
fn logical_and_shifts() {
    check_all("logical", &[
        logical_sr(0, 0, 0, 1, 2, 0, 0),  // and x0, x1, x2
        logical_sr(1, 0, 3, 4, 5, 0, 4),  // orr x3, x4, x5, lsl #4
        logical_sr(2, 0, 6, 7, 0, 1, 8),  // eor x6, x7, x0, lsr #8
        logical_sr(3, 0, 1, 2, 3, 2, 1),  // ands x1, x2, x3, asr #1
        logical_sr(0, 1, 4, 5, 6, 0, 0),  // bic x4, x5, x6
        logical_sr(1, 1, 7, 0, 1, 3, 5),  // orn x7, x0, x1, ror #5
    ]);
    check_all("var_shift", &[
        dp2(8, 0, 1, 2),   // lslv x0, x1, x2
        dp2(9, 3, 4, 5),   // lsrv x3, x4, x5
        dp2(10, 6, 7, 0),  // asrv x6, x7, x0
        dp2(11, 1, 2, 3),  // rorv x1, x2, x3
    ]);
}

#[test]
fn mul_div() {
    check_all("madd_msub", &[
        madd(0, 0, 1, 2, 3),  // madd x0, x1, x2, x3
        madd(1, 4, 5, 6, 7),  // msub x4, x5, x6, x7
    ]);
    check_all("div", &[
        dp2(2, 0, 1, 2),  // udiv x0, x1, x2
        dp2(3, 3, 4, 5),  // sdiv x3, x4, x5
    ]);
    // Division corner cases: /0 -> 0, and INT_MIN / -1 -> INT_MIN.
    let mut seed = GuestRegs::default();
    seed.pc = BASE;
    seed.sp = BASE + 0x800;
    seed.x[1] = 100;
    seed.x[2] = 0; // divisor zero
    seed.x[4] = i64::MIN as u64;
    seed.x[5] = u64::MAX; // -1
    let prog = [dp2(2, 0, 1, 2), dp2(3, 3, 4, 5), RET];
    check("div_corners", &prog, &seed);
}

#[test]
fn bitfield_extract_movewide() {
    check_all("movewide", &[
        movz(0, 0xBEEF, 0),
        movz(1, 0xDEAD, 1),
        0xF2C0_0AE2u32, // movk x2, #0x57, lsl #32
        0x9280_0003,    // movn x3, #0
    ]);
    check_all("bitfield", &[
        ubfm(0, 0, 1, 0, 7),    // sbfm x0, x1, #0, #7   (sxtb)
        ubfm(2, 3, 4, 8, 15),   // ubfm x2, x4, #8, #15  (ubfx)
        ubfm(1, 5, 6, 4, 20),   // bfm  x5(rd=5)->x5 ... bfxil-ish
    ]);
    check_all("extract", &[
        extr(0, 1, 2, 17), // extr x0, x1, x2, #17
        extr(3, 4, 5, 1),  // extr x3, x4, x5, #1
    ]);
}

#[test]
fn cond_select_compare() {
    check_all("csel", &[
        csel(0, 0, 0, 1, 2, 0),  // csel  x0, x1, x2, eq
        csel(0, 1, 3, 4, 5, 1),  // csinc x3, x4, x5, ne
        csel(1, 0, 6, 7, 0, 11), // csinv x6, x7, x0, lt
        csel(1, 1, 1, 2, 3, 12), // csneg x1, x2, x3, gt
    ]);
    check_all("ccmp", &[
        ccmp_imm(1, 1, 5, 0, 0b0010),  // ccmp x1, #5, #2, eq
        ccmp_imm(0, 3, 9, 12, 0b1100), // ccmn x3, #9, #12, gt
    ]);
}

#[test]
fn data_proc_1src() {
    check_all("clz_rev", &[
        dp1(4, 0, 1),  // clz   x0, x1
        dp1(1, 2, 3),  // rev16 x2, x3
        dp1(2, 4, 5),  // rev32 x4, x5
        dp1(3, 6, 7),  // rev   x6, x7
    ]);
}

#[test]
fn pc_relative() {
    check_all("adr_adrp", &[
        adr(0, 0, 0x40),    // adr  x0, .+0x40
        adr(1, 1, -0x20),   // adrp x1, ...
        NOP,
    ]);
}

#[test]
fn memory_single_and_pair() {
    // Build a safe in-window base in x9, then exercise loads/stores/pairs that
    // round-trip through memory. Seeded data in x0..x3.
    check_all("ldr_str", &[
        movz(9, 0x1400, 0),   // x9 = 0x1400 (in-window, past the code)
        str_uimm(0, 9, 0),    // str x0, [x9]
        str_uimm(1, 9, 1),    // str x1, [x9, #8]
        ldr_uimm(2, 9, 0),    // ldr x2, [x9]      -> x2 == x0
        ldr_uimm(3, 9, 1),    // ldr x3, [x9, #8]  -> x3 == x1
    ]);
    check_all("ldp_stp", &[
        movz(9, 0x1400, 0),   // x9 = 0x1400
        stp(0, 1, 9, 0),      // stp x0, x1, [x9]
        ldp(4, 5, 9, 0),      // ldp x4, x5, [x9]  -> x4==x0, x5==x1
        stp(2, 3, 9, 2),      // stp x2, x3, [x9, #16]
        ldp(6, 7, 9, 2),      // ldp x6, x7, [x9, #16]
    ]);
}
