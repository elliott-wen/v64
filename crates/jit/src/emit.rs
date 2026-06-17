//! WASM emitter for a JIT block.
//!
//! The emitted module imports the shared linear memory (holding the guest
//! register image) and the `interpret_one` host helper, and exports a single
//! function [`BLOCK_FUNC`] with the block ABI signature
//! `(param $regs_base i32) -> i64` (next guest PC).
//!
//! A block is a leading run of always-inline-lowerable register ops (see
//! [`crate::can_inline`]) terminated by exactly one *escape* instruction — its
//! last. The leading ops are lowered inline ([`crate::lower`]); the escape (a
//! branch, load/store, system op, …) is executed by `call interpret_one`, which
//! single-steps the interpreter against the shared CPU + bus and returns the
//! next guest PC. Because the escape is always last, nothing it triggers (a
//! taken branch, a fault that vectors) can corrupt later inline ops — there are
//! none. A register-only branch terminator is lowered inline instead of escaped.

use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    Instruction, MemoryType, Module, TypeSection, ValType,
};

use aarch64_decoder::{is_terminator, Block};

/// Name of the exported block entry function.
pub const BLOCK_FUNC: &str = "block";

/// Imported-function index of `interpret_one` (the only function import, so func
/// index 0; the block function follows at index 1).
const INTERPRET_ONE: u32 = 0;

/// Emit a self-contained WASM module exporting [`BLOCK_FUNC`].
///
/// Imports (resolved by the runtime's linker):
/// - `env.interpret_one : (i32) -> i64` — single-step the interpreter on the
///   shared CPU + bus, for the block's escape instruction.
/// - `env.memory` — the shared linear memory holding the register image.
///
/// `guest_base` is accepted for ABI symmetry; inline register ops never touch
/// memory, so it is unused.
pub fn emit_block(block: &Block, guest_base: u64) -> Vec<u8> {
    let mut module = Module::new();

    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I64]);
    module.section(&types);

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

/// Build the block function body: inline every instruction but the last, and the
/// last either as an inline branch terminator or an `interpret_one` escape whose
/// returned next-PC is the function result.
fn emit_body(block: &Block, guest_base: u64) -> Function {
    let mut f = Function::new([
        (crate::lower::SCRATCH_I64, ValType::I64),
        (crate::lower::SCRATCH_I32, ValType::I32),
    ]);
    let n = block.insns.len();
    debug_assert!(n > 0, "form produces at least one instruction");

    for (i, (pc, insn)) in block.insns.iter().enumerate() {
        let is_last = i + 1 == n;
        if is_last {
            // Inline a register-only branch terminator; otherwise escape to the
            // interpreter, whose returned i64 next-PC is the function result.
            if is_terminator(insn) && crate::lower::lower_terminator(&mut f, insn, *pc) {
                continue;
            }
            f.instruction(&Instruction::LocalGet(0)); // $regs_base
            f.instruction(&Instruction::Call(INTERPRET_ONE));
        } else {
            // Non-last instructions are always inline-lowerable by construction
            // (block formation stops the inline run at the first that isn't).
            debug_assert!(crate::can_inline(insn), "non-last block instruction must be inlinable");
            let ok = crate::lower::lower_sequential(&mut f, insn, *pc, guest_base);
            debug_assert!(ok, "can_inline instruction failed to lower");
        }
    }
    f.instruction(&Instruction::End);
    f
}
