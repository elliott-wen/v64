//! Milestone 4 lowering tests: run a single block through the JIT and through
//! the interpreter and assert bit-identical register/flag state. The interpreter
//! is the reference oracle, so this validates every inline lowering against it.
//!
//! Each program is fully lowered inline, so `vm.interp_calls()` must be 0 — that
//! *proves* the inline path ran rather than silently falling back (a fallback
//! would also match the interpreter and hide a broken lowering).

use aarch64_cpu_state::{CpuState, GuestRegs};
use aarch64_interp::{step, Memory, Step};
use aarch64_jit::{form_block, Vm};

const BASE: u64 = 0x1000;
const RAM: usize = 0x1000;

/// Run `prog` (starting from register state `init`, PC forced to BASE) as one
/// block through both the JIT and the interpreter and assert they agree.
fn check(init: GuestRegs, prog: &[u32]) {
    let block = form_block(BASE, |pc| prog[((pc - BASE) / 4) as usize]);
    let steps = block.insns.len();

    // JIT.
    let mut vm = Vm::new(BASE, RAM);
    for (i, w) in prog.iter().enumerate() {
        vm.write_ram(BASE + 4 * i as u64, &w.to_le_bytes());
    }
    let mut jit_init = init.clone();
    jit_init.pc = BASE;
    vm.load_regs(&jit_init);
    let exit = vm.run_block(&block);
    let jit_regs = vm.store_regs();

    // Interpreter reference.
    let mut cpu = CpuState::new();
    let mut ref_init = init;
    ref_init.pc = BASE;
    cpu.load_guest_regs(&ref_init);
    let mut mem = Memory::new(BASE, RAM);
    let mut code = Vec::new();
    for w in prog {
        code.extend_from_slice(&w.to_le_bytes());
    }
    mem.write(BASE, &code);
    let mut next = BASE;
    for _ in 0..steps {
        match step(&mut cpu, &mut mem) {
            Step::Next(p) => next = p,
            Step::Unsupported { .. } => break,
        }
    }
    let ref_regs = cpu.to_guest_regs();

    assert_eq!(vm.interp_calls(), 0, "block was not fully lowered inline");
    assert_eq!(exit.next_pc, next, "next-PC mismatch");
    assert_eq!(jit_regs, ref_regs, "register/flag state mismatch vs interpreter");
}

fn regs(setup: impl FnOnce(&mut GuestRegs)) -> GuestRegs {
    let mut r = GuestRegs::default();
    setup(&mut r);
    r
}

#[test]
fn move_and_logical() {
    check(
        GuestRegs::default(),
        &[
            0xD2824680, // movz  x0, #0x1234
            0xF2B579A0, // movk  x0, #0xABCD, lsl #16
            0x92800001, // movn  x1, #0            ; x1 = !0
            0x8A010002, // and   x2, x0, x1
            0xAA010003, // orr   x3, x0, x1
            0xCA010004, // eor   x4, x0, x1
            0xD65F03C0, // ret
        ],
    );
}

#[test]
fn add_sub_no_flags() {
    check(
        regs(|r| {
            r.x[0] = 100;
            r.x[1] = 40;
        }),
        &[
            0x8B010002, // add x2, x0, x1   ; 140
            0xCB010003, // sub x3, x0, x1   ; 60
            0x9100A404, // add x4, x0, #41  ; 141
            0xD65F03C0, // ret
        ],
    );
}

#[test]
fn adds_subs_flags_64() {
    // Carry + zero on add, and a borrow case on subtract.
    check(
        regs(|r| {
            r.x[0] = u64::MAX;
            r.x[1] = 1;
        }),
        &[
            0xAB010002, // adds x2, x0, x1  ; result 0, Z=1, C=1
            0xEB010003, // subs x3, x0, x1  ; result MAX-1, no borrow -> C=1
            0xD65F03C0, // ret
        ],
    );
}

#[test]
fn adds_subs_flags_32_overflow() {
    // 0x8000_0000 + 0x8000_0000 (W) overflows and carries; result 0.
    check(
        regs(|r| {
            r.x[0] = 0x8000_0000;
            r.x[1] = 0x8000_0000;
        }),
        &[
            0x2B010002, // adds w2, w0, w1  ; V=1, C=1, Z=1
            0x6B010003, // subs w3, w0, w1  ; result 0, Z=1, C=1
            0xD65F03C0, // ret
        ],
    );
}

#[test]
fn cmp_immediate_then_beq_taken() {
    check(
        regs(|r| r.x[0] = 7),
        &[
            0xF1001C1F, // cmp x0, #7   (subs xzr, x0, #7) -> Z=1
            0x54000040, // b.eq #8      -> taken
        ],
    );
}

#[test]
fn bcond_not_taken_falls_through() {
    check(
        regs(|r| r.x[0] = 7),
        &[
            0xF1001C1F, // cmp x0, #7   -> Z=1
            0x54000041, // b.ne #8      -> not taken, fall through to pc+4
        ],
    );
}

#[test]
fn cbz_taken_and_cbnz() {
    check(GuestRegs::default(), &[0xB4000040 /* cbz x0, #8 (x0==0) -> taken */]);
    check(regs(|r| r.x[0] = 1), &[0xB5000040 /* cbnz x0, #8 (x0!=0) -> taken */]);
}

#[test]
fn tbz_tbnz() {
    // bit 2 is set: TBNZ takes, TBZ does not.
    check(regs(|r| r.x[0] = 0b100), &[0x37100040 /* tbnz x0, #2, #8 */]);
    check(regs(|r| r.x[0] = 0b100), &[0x36100040 /* tbz  x0, #2, #8 -> fall through */]);
}

#[test]
fn bl_sets_link_register() {
    check(GuestRegs::default(), &[0x94000002 /* bl #8 -> x30 = BASE+4, target BASE+8 */]);
}

#[test]
fn br_and_blr() {
    check(regs(|r| r.x[5] = 0x1234), &[0xD61F00A0 /* br x5 -> target 0x1234 */]);
    check(regs(|r| r.x[5] = 0x1234), &[0xD63F00A0 /* blr x5 -> x30=BASE+4, target 0x1234 */]);
}
