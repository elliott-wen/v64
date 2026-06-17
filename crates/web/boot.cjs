// Boot a real arm64 Linux Image in node, via the wasm-compiled emulator. Streams
// the serial console and stops once the kernel banner appears (or on a budget).
//
//   crates/web/build.sh
//   node crates/web/boot.cjs
const { readFileSync } = require("node:fs");
const { Emulator } = require("./pkg/aarch64_web.js");

const image = readFileSync("guest/prebuilt/Image-tiny");
const initrd = readFileSync("guest/prebuilt/uitest.cpio.gz");
const bootargs = "earlycon=pl011,0x9000000 console=ttyAMA0 rdinit=/init";

const emu = new Emulator();
emu.boot(image, initrd, bootargs);

const BATCH = 2_000_000;
const MAX_BATCHES = 400; // up to ~800M instructions
const t0 = Date.now();
let out = "";
let status = 0;

for (let i = 0; i < MAX_BATCHES; i++) {
  status = emu.run(BATCH);
  const u = emu.take_uart();
  if (u.length) {
    const s = Buffer.from(u).toString("latin1");
    process.stdout.write(s);
    out += s;
  }
  if (status === 1) {
    console.log("\n[guest powered off]");
    break;
  }
  if (out.includes("Linux version")) {
    console.log("\n[kernel banner reached — Linux boots in node]");
    break;
  }
}

const secs = ((Date.now() - t0) / 1000).toFixed(1);
const mips = (Number(emu.total_insns()) / 1e6 / secs).toFixed(1);
console.log(`\nran ${emu.total_insns()} instructions in ${secs}s (~${mips} MIPS in node)`);
if (!out.includes("Linux")) {
  console.error("FAIL: no kernel output");
  process.exit(1);
}
