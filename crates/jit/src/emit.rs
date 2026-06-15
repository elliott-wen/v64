//! WASM emitter for a formed block (Milestone 2 block ABI).
//!
//! The emitted module imports the shared linear memory and the `interpret_one`
//! host helper, and exports a single function [`BLOCK_FUNC`] with the block ABI
//! signature `(param $regs_base i32) -> i64` (next guest PC).
//!
//! Each instruction is either **lowered inline** (see [`crate::lower`]) or, if
//! not yet handled, emitted as a `call interpret_one` — the escape hatch that
//! steps one guest instruction through the interpreter. Because a block is
//! straight-line and terminated by its last instruction, processing it in order
//! reproduces it exactly: non-terminators leave nothing on the stack (inline) or
//! drop their sequential PC (helper), and the terminator's `interpret_one` call
//! yields the real next guest PC, which the function returns.
//!
//! Inline lowerings keep the register-image PC consistent themselves, so they
//! and helper calls can be freely interleaved. The terminator is never lowered
//! inline (it's a branch/exception), so it always goes through `interpret_one`.

use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    Instruction, MemoryType, Module, TypeSection, ValType,
};

use crate::block::{is_terminator, Block};

/// Name of the exported block entry function.
pub const BLOCK_FUNC: &str = "block";

/// Imported-function index of `interpret_one` (it is the only function import,
/// so it occupies func index 0; the block function follows at index 1).
const INTERPRET_ONE: u32 = 0;

/// Emit a self-contained WASM module exporting [`BLOCK_FUNC`].
///
/// `guest_base` is the base guest address of the VM's RAM window, needed to fold
/// the guest→linear address displacement into inline memory accesses.
///
/// Imports (resolved by the runtime's linker):
/// - `env.interpret_one : (i32) -> i64` — single-instruction escape hatch.
/// - `env.memory` — the shared linear memory holding the register image + RAM.
pub fn emit_block(block: &Block, guest_base: u64) -> Vec<u8> {
    let mut module = Module::new();

    // One shared signature for both interpret_one and the block function.
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I64]);
    module.section(&types);

    // Imports: the helper function (func 0) and the shared memory (mem 0).
    let mut imports = ImportSection::new();
    imports.import("env", "interpret_one", EntityType::Function(0));
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

    // The block function: type 0, becoming func index 1 (after the import).
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);

    let mut exports = ExportSection::new();
    exports.export(BLOCK_FUNC, ExportKind::Func, 1);
    module.section(&exports);

    let mut code = CodeSection::new();
    code.function(&emit_body(block, guest_base));
    module.section(&code);

    module.finish()
}

/// Build the block function body: each instruction is lowered inline where
/// possible, otherwise via `interpret_one`; the terminator's next guest PC is
/// the function result.
fn emit_body(block: &Block, guest_base: u64) -> Function {
    let mut f = Function::new([
        (crate::lower::SCRATCH_I64, ValType::I64),
        (crate::lower::SCRATCH_I32, ValType::I32),
    ]);
    let n = block.insns.len();
    debug_assert!(n > 0, "form_block always produces at least one instruction");

    for (i, (pc, insn)) in block.insns.iter().enumerate() {
        let is_last = i + 1 == n;

        if is_last {
            // The last instruction must leave the next-PC i64 result on the
            // stack. Three ways, in order of preference:
            //   1. an inlined terminator (computes its own next PC);
            //   2. an inlined non-terminator, followed by the sequential PC;
            //   3. interpret_one (returns the next PC) for anything not lowered.
            if is_terminator(insn) {
                if crate::lower::lower_terminator(&mut f, insn, *pc) {
                    continue;
                }
            } else if crate::lower::lower_sequential(&mut f, insn, *pc, guest_base) {
                f.instruction(&Instruction::I64Const(pc.wrapping_add(4) as i64));
                continue;
            }
            f.instruction(&Instruction::LocalGet(0)); // $regs_base
            f.instruction(&Instruction::Call(INTERPRET_ONE));
        } else {
            // Non-terminators leave nothing on the stack: inline, or
            // interpret_one whose (sequential) next-PC result is discarded.
            if crate::lower::lower_sequential(&mut f, insn, *pc, guest_base) {
                continue;
            }
            f.instruction(&Instruction::LocalGet(0));
            f.instruction(&Instruction::Call(INTERPRET_ONE));
            f.instruction(&Instruction::Drop);
        }
    }
    f.instruction(&Instruction::End);
    f
}
