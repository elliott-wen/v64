//! WASM emitter for a formed block.
//!
//! Milestone 1 is scaffolding: the emitted function ignores its register-base
//! argument and returns the fall-through guest PC. Real instruction lowering and
//! the full block ABI (helper imports, linear-memory register access) arrive in
//! later milestones.

use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
    TypeSection, ValType,
};

use crate::block::Block;

/// Name of the exported block entry function.
pub const BLOCK_FUNC: &str = "block";

/// Emit a self-contained WASM module exporting [`BLOCK_FUNC`] with the block ABI
/// signature `(param $regs_base i32) -> i64` (next guest PC).
pub fn emit_block(block: &Block) -> Vec<u8> {
    let mut module = Module::new();

    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I64]);
    module.section(&types);

    let mut funcs = FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);

    let mut exports = ExportSection::new();
    exports.export(BLOCK_FUNC, ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    let mut f = Function::new([]);
    // Placeholder body: return start + 4*len (sequential fall-through).
    let exit_pc = block.start.wrapping_add(4 * block.insns.len() as u64);
    f.instruction(&Instruction::I64Const(exit_pc as i64));
    f.instruction(&Instruction::End);
    code.function(&f);
    module.section(&code);

    module.finish()
}
