//! AArch64 -> WebAssembly JIT.
//!
//! Translates hot blocks of guest code to WebAssembly, runs them on an embedded
//! wasmtime runtime, and (by construction) produces state bit-identical to the
//! interpreter, which remains the cold-path executor and reference oracle.
//!
//! See `docs/jit-plan.md` for the milestone roadmap.

pub mod block;
pub mod emit;

pub use block::{form_block, Block};
pub use emit::{emit_block, BLOCK_FUNC};

/// Compile a block's WASM with wasmtime and run it, returning the next guest PC.
///
/// Milestone-1 smoke path: it exercises emit -> compile -> instantiate -> call
/// end to end. The register-base argument is currently ignored by the emitted
/// code.
pub fn run_block_placeholder(block: &Block) -> i64 {
    use wasmtime::{Engine, Instance, Module, Store};

    let bytes = emit_block(block);
    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("valid module");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let func = instance
        .get_typed_func::<i32, i64>(&mut store, BLOCK_FUNC)
        .expect("block export");
    func.call(&mut store, 0).expect("call block")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_compile_run_roundtrip() {
        // NOP; NOP; B . — three instructions; fall-through PC = start + 12.
        let code = [0xd503201fu32, 0xd503201f, 0x1400_0000];
        let block = form_block(0x1000, |pc| code[((pc - 0x1000) / 4) as usize]);
        assert_eq!(run_block_placeholder(&block), 0x1000 + 12);
    }
}
