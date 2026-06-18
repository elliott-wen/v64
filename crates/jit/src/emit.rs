//! WASM module emitter for a JIT block or region.
//!
//! Wraps a function body (from [`emit_body`] for a single block, or
//! [`crate::lower::emit_region_body`] for a multi-block region) in a
//! self-contained module: one import — the shared linear memory holding the live
//! `CpuState` and guest RAM — and the export [`BLOCK_FUNC`] with the ABI
//! `(param $regs_base i32, $ram_base i32) -> i64` (the next guest PC).
//!
//! Every instruction is lowered to native wasm; memory accesses take an inline
//! TLB-checked fast path and **bail** to the interpreter on a miss (returning the
//! faulting PC). The block reads/writes the real registers in place at
//! `$regs_base` (the live `CpuState`) — no escape import, no register-image copy.

use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    Instruction, MemoryType, Module, TypeSection, ValType,
};

use aarch64_decoder::{is_terminator, Block};

/// Name of the exported block entry function.
pub const BLOCK_FUNC: &str = "block";

/// Emit a self-contained WASM module exporting [`BLOCK_FUNC`]. The only import
/// is `env.memory` (the shared linear memory holding the register image and guest
/// RAM). `ram_phys`/`ram_size` bound the guest-physical RAM window, baked into the
/// inline memory fast path (see `lower::lower_mem`).
pub fn emit_block(block: &Block, ram_phys: u64, ram_size: u64) -> Vec<u8> {
    wrap_module(emit_body(block, ram_phys, ram_size))
}

/// Emit a self-contained WASM module for a multi-block [`Region`](crate::Region)
/// — one function with an internal dispatch loop, so control stays in compiled
/// code across in-region branches (see [`crate::lower::emit_region_body`]). Same
/// ABI and single `env.memory` import as [`emit_block`].
#[must_use]
pub fn emit_region(region: &crate::Region, ram_phys: u64, ram_size: u64) -> Vec<u8> {
    wrap_module(crate::lower::emit_region_body(region, ram_phys, ram_size))
}

/// Wrap a block/region function body in the module scaffolding: the
/// `(i32, i32) -> i64` type, the `env.memory` import, and the `block` export.
fn wrap_module(body: Function) -> Vec<u8> {
    let mut module = Module::new();

    // Block signature: (regs_base: i32, ram_base: i32) -> i64 (next guest PC).
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32, ValType::I32], [ValType::I64]);
    module.section(&types);

    // Only import: the shared linear memory (mem 0) holding the live CpuState and
    // guest RAM.
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 0,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    module.section(&imports);

    // The block function: type 0, func index 0 (there are no function imports).
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);

    let mut exports = ExportSection::new();
    exports.export(BLOCK_FUNC, ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    code.function(&body);
    module.section(&code);

    module.finish()
}

/// Build the block function body: lower every instruction inline. The last is
/// either an inline branch (which leaves the next PC) or a register op followed
/// by the sequential next PC; either way the function returns the next guest PC.
fn emit_body(block: &Block, ram_phys: u64, ram_size: u64) -> Function {
    let mut f = Function::new([
        (crate::lower::SCRATCH_I64, ValType::I64),
        (crate::lower::SCRATCH_I32, ValType::I32),
    ]);
    let n = block.insns.len();
    debug_assert!(n > 0, "form produces at least one instruction for a non-empty block");

    // The block's entry virtual address at compile time. Every emitted PC is
    // `runtime_entry_pc + (this - entry_pc)`, so the same physical block is
    // correct at whatever VA it is mapped to (position independence).
    let entry_pc = block.start;
    crate::lower::prologue(&mut f);

    // A memory-free block whose terminator branches back to its own entry is a
    // self-loop: emit it as an internal wasm `loop` so iterations stay in
    // compiled code. (Blocks containing a load/store can bail mid-way, which
    // doesn't compose with the loop, so those go straight-line below.)
    let (last_pc, last_insn) = &block.insns[n - 1];
    let has_mem = block.insns.iter().any(|(_, i)| crate::is_inline_mem(i));
    if !has_mem && crate::lower::taken_target(last_insn, *last_pc) == Some(entry_pc) {
        crate::lower::emit_self_loop(&mut f, block, entry_pc);
        f.instruction(&Instruction::End);
        return f;
    }

    // Straight-line block: it runs once per call, so it retired `n` instructions
    // (a mid-block memory bail overwrites this with the count it actually ran).
    crate::lower::store_count(&mut f, n as u64);
    for (i, (pc, insn)) in block.insns.iter().enumerate() {
        let is_last = i + 1 == n;
        if is_last && is_terminator(insn) {
            // Inline branch terminator: leaves the next PC on the stack.
            let ok = crate::lower::lower_terminator(&mut f, insn, *pc, entry_pc);
            debug_assert!(ok, "block terminator must be an inline-lowerable branch");
        } else if crate::is_inline_load_store(insn) {
            // Load/store: TLB-checked fast path, bail (return) on a miss. On a
            // bail `i` instructions already ran (the count to bill).
            let ok =
                crate::lower::lower_mem(&mut f, insn, *pc, entry_pc, i as u64, ram_phys, ram_size, None);
            debug_assert!(ok, "inline load/store must be lowerable");
            if is_last {
                crate::lower::gen_pc(&mut f, pc.wrapping_add(4), entry_pc);
            }
        } else if crate::is_inline_load_store_pair(insn) {
            // LDP/STP: same fast-path + bail, two registers.
            let ok = crate::lower::lower_mem_pair(
                &mut f, insn, *pc, entry_pc, i as u64, ram_phys, ram_size, None,
            );
            debug_assert!(ok, "inline load/store pair must be lowerable");
            if is_last {
                crate::lower::gen_pc(&mut f, pc.wrapping_add(4), entry_pc);
            }
        } else {
            // Register op: updates the image; leaves nothing.
            let ok = crate::lower::lower_sequential(&mut f, insn, *pc, entry_pc);
            debug_assert!(ok, "non-terminator block instruction must be inline-lowerable");
            if is_last {
                // Block ended before a non-inline instruction: return its PC.
                crate::lower::gen_pc(&mut f, pc.wrapping_add(4), entry_pc);
            }
        }
    }
    f.instruction(&Instruction::End);
    f
}
