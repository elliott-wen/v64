//! Milestone 2 end-to-end tests: run formed blocks through the embedded
//! wasmtime runtime and the `interpret_one` escape hatch, exercising the block
//! ABI (register-image round-trip, helper import, shared memory, exit codes).

use aarch64_cpu_state::GuestRegs;
use aarch64_jit::{abi, form_block, Vm};

/// Read code words for a block placed at `base` out of a slice.
fn reader(base: u64, code: &[u32]) -> impl Fn(u64) -> u32 + '_ {
    move |pc| code[((pc - base) / 4) as usize]
}

/// Load a program into a fresh VM at `base`, seed PC, and return the VM + block.
fn setup(base: u64, ram_bytes: usize, prog: &[u32]) -> (Vm, aarch64_jit::Block) {
    let block = form_block(base, reader(base, prog));
    let mut vm = Vm::new(base, ram_bytes);
    for (i, w) in prog.iter().enumerate() {
        vm.write_ram(base + 4 * i as u64, &w.to_le_bytes());
    }
    let regs = GuestRegs { pc: base, ..GuestRegs::default() };
    vm.load_regs(&regs);
    (vm, block)
}

/// Vector LDR Q / STR Q move a full 128-bit register through memory, inlined
/// (no interpret_one). Proves the SIMD load/store lowering and that V0 is
/// written/read via the register image.
#[test]
fn inline_vector_load_store_q() {
    let base = 0x4000;
    let prog = [
        0xD28C0001u32, // movz x1, #0x6000  (source)
        0xD28C2002,    // movz x2, #0x6100  (dest)
        0x3DC00020,    // ldr  q0, [x1]
        0x3D800040,    // str  q0, [x2]
        0xD65F03C0,    // ret
    ];
    let val: u128 = 0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10;
    let (mut vm, block) = setup(base, 0x8000, &prog);
    vm.write_ram(0x6000, &val.to_le_bytes());

    let exit = vm.run_block(&block);

    assert_eq!(exit.exit_reason, abi::EXIT_NONE);
    assert_eq!(vm.interp_calls(), 0, "vector load/store should be fully inline");
    assert_eq!(vm.store_regs().v[0], val, "V0 loaded");
    assert_eq!(u128::from_le_bytes(vm.read_ram(0x6100, 16).try_into().unwrap()), val, "stored");
}

/// Vector LDR D loads 64 bits and zeroes the upper half of V (a SIMD load writes
/// the whole 128-bit register).
#[test]
fn inline_vector_load_d_zero_extends() {
    let base = 0x4000;
    let prog = [
        0xD28C0001u32, // movz x1, #0x6000
        0xFD400020,    // ldr  d0, [x1]
        0xD65F03C0,    // ret
    ];
    let (mut vm, block) = setup(base, 0x8000, &prog);
    // Pre-set V0 to all-ones to prove the upper half is cleared by the load.
    let mut regs = GuestRegs { pc: base, ..GuestRegs::default() };
    regs.v[0] = u128::MAX;
    vm.load_regs(&regs);
    vm.write_ram(0x6000, &0xDEAD_BEEF_CAFE_F00Du64.to_le_bytes());

    let exit = vm.run_block(&block);

    assert_eq!(exit.exit_reason, abi::EXIT_NONE);
    assert_eq!(vm.interp_calls(), 0);
    assert_eq!(vm.store_regs().v[0], 0xDEAD_BEEF_CAFE_F00D, "low 64 loaded, high 64 zeroed");
}

/// MOVZ X0,#5; MOVZ X1,#7; ADD X2,X0,X1; B #+8 — a straight-line block run
/// entirely through `interpret_one`. The terminator yields the exit PC.
#[test]
fn runs_block_via_interpret_one() {
    let base = 0x1000;
    let prog = [
        0xD28000A0u32, // movz x0, #5
        0xD28000E1,    // movz x1, #7
        0x8B010002,    // add  x2, x0, x1
        0x14000002,    // b    #+8  -> 0x100C + 8 = 0x1014
    ];
    let (mut vm, block) = setup(base, 0x1000, &prog);

    let exit = vm.run_block(&block);

    assert_eq!(exit.exit_reason, abi::EXIT_NONE);
    assert_eq!(exit.next_pc, 0x1014);
    let out = vm.store_regs();
    assert_eq!(out.x[0], 5);
    assert_eq!(out.x[1], 7);
    assert_eq!(out.x[2], 12);
    assert_eq!(out.pc, 0x1014);
}

/// A guest store must land in the shared linear memory and be visible to the
/// host afterwards — proves guest RAM is the single source of truth.
#[test]
fn block_store_is_visible_to_host() {
    let base = 0x2000;
    let prog = [
        0xD2800020u32, // movz x0, #1
        0xD2880001,    // movz x1, #0x4000   ; store address (within the window)
        0xF9000020,    // str  x0, [x1]
        0xD65F03C0,    // ret  (x30 = 0 -> next pc 0)
    ];
    let (mut vm, block) = setup(base, 0x8000, &prog);

    let exit = vm.run_block(&block);
    assert_eq!(exit.exit_reason, abi::EXIT_NONE);

    let stored = vm.read_ram(0x4000, 8);
    assert_eq!(u64::from_le_bytes(stored.try_into().unwrap()), 1);
}

/// An inline (M3) LDR reads guest memory directly — no interpret_one — and the
/// loaded value lands in the destination register.
#[test]
fn inline_load_reads_memory() {
    let base = 0x4000;
    let prog = [
        0xD28C0001u32, // movz x1, #0x6000   ; load address (within the window)
        0xF9400020,    // ldr  x0, [x1]
        0xD65F03C0,    // ret
    ];
    let (mut vm, block) = setup(base, 0x8000, &prog);
    vm.write_ram(0x6000, &0xDEAD_BEEFu64.to_le_bytes());

    let exit = vm.run_block(&block);

    assert_eq!(exit.exit_reason, abi::EXIT_NONE);
    assert_eq!(vm.store_regs().x[0], 0xDEAD_BEEF);
}

/// An inline (M3) word store writes only its low 4 bytes, and a store of XZR
/// writes zero — exercises the size and Rt==31 paths of the lowering.
#[test]
fn inline_store_sizes_and_xzr() {
    let base = 0x5000;
    let prog = [
        0xD2802460u32, // movz x0, #0x123       ; data
        0xD28C0001,    // movz x1, #0x6000      ; address
        0xB9000020,    // str  w0, [x1]         ; 32-bit store
        0xF900043F,    // str  xzr, [x1, #8]    ; 64-bit store of zero
        0xD65F03C0,    // ret
    ];
    let (mut vm, block) = setup(base, 0x8000, &prog);
    // Pre-fill so we can see the word store clear the upper half and xzr zero it.
    vm.write_ram(0x6000, &0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());
    vm.write_ram(0x6008, &0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());

    let exit = vm.run_block(&block);
    assert_eq!(exit.exit_reason, abi::EXIT_NONE);

    let lo = u64::from_le_bytes(vm.read_ram(0x6000, 8).try_into().unwrap());
    // The 32-bit store touched only the low 4 bytes; the upper 4 keep their 0xFF.
    assert_eq!(lo, 0xFFFF_FFFF_0000_0123);
    let hi = u64::from_le_bytes(vm.read_ram(0x6008, 8).try_into().unwrap());
    assert_eq!(hi, 0); // str xzr cleared it
}

/// An inline store whose address is out of bounds traps in WASM and surfaces as
/// `EXIT_FAULT` at the faulting instruction. This also *proves* the store was
/// lowered inline: had it fallen back to interpret_one, the interpreter's Vec
/// would panic instead of producing a clean fault exit.
#[test]
fn inline_oob_store_faults() {
    let base = 0x1000;
    let prog = [
        0xD2A00081u32, // movz x1, #0x4, lsl #16   ; x1 = 0x40000 (far past the window)
        0xF9000020,    // str  x0, [x1]            ; inline, out of bounds -> trap
        0xD65F03C0,    // ret
    ];
    let (mut vm, block) = setup(base, 0x1000, &prog);

    let exit = vm.run_block(&block);

    assert_eq!(exit.exit_reason, abi::EXIT_FAULT);
    assert_eq!(exit.next_pc, 0x1004); // the str instruction's own PC
}

/// An unsupported instruction routes through interpret_one and surfaces as an
/// `EXIT_UNSUPPORTED` exit at the faulting PC, with prior work preserved.
#[test]
fn unsupported_instruction_exits() {
    let base = 0x3000;
    let prog = [
        0xD28000A0u32, // movz x0, #5
        0x0000_0000,   // udf #0 -> Unsupported (terminator)
    ];
    let (mut vm, block) = setup(base, 0x1000, &prog);

    let exit = vm.run_block(&block);

    assert_eq!(exit.exit_reason, abi::EXIT_UNSUPPORTED);
    assert_eq!(exit.next_pc, base + 4); // faulted at the second instruction
    assert_eq!(vm.store_regs().x[0], 5); // first instruction took effect
}
