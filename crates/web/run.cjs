// Run the AArch64 interpreter (compiled to wasm, bound via wasm-bindgen) under
// node. wasm-bindgen handles loading + the Uint8Array marshalling, so this is
// just a function call.
//
//   cargo build -p aarch64-web --target wasm32-unknown-unknown --release
//   wasm-bindgen target/wasm32-unknown-unknown/release/aarch64_web.wasm \
//       --out-dir crates/web/pkg --target nodejs
//   node crates/web/run.cjs
const { run_code } = require("./pkg/aarch64_web.js");

// Guest program: movz x0,#3 ; movz x1,#4 ; add x0,x0,x1  -> x0 = 7
const prog = Uint32Array.from([0xd280_0060, 0xd280_0081, 0x8b01_0000]);
const code = new Uint8Array(prog.buffer);

const x0 = run_code(code, prog.length);
console.log(`interpreter ran ${prog.length} instructions in node; x0 = ${x0}`);
if (x0 !== 7n) {
  console.error(`FAIL: expected 7, got ${x0}`);
  process.exit(1);
}
console.log("OK — AArch64 interpreter runs in node (via wasm-bindgen).");
