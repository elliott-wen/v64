// Throughput benchmark: interp vs JIT, in node, on a real Linux boot workload.
// Reports MIPS for each over a fixed instruction budget, plus the JIT's coverage
// (fraction of instructions retired inside compiled blocks). A long budget
// amortizes JIT warmup (compilation) so steady-state shows.
//
//   crates/web/build.sh && node crates/web/bench.cjs [budgetInsns]
const { readFileSync } = require("node:fs");
const { Emulator } = require("./pkg/aarch64_web.js");

const image = readFileSync("guest/prebuilt/Image-tiny");
const initrd = readFileSync("guest/prebuilt/uitest.cpio.gz");
const bootargs = "earlycon=pl011,0x9000000 console=ttyAMA0 rdinit=/init";

const BATCH = 5_000_000;
const BUDGET = Number(process.argv[2] || 400_000_000);

function bench(useJit) {
  const emu = new Emulator();
  emu.boot(image, initrd, bootargs);
  const run = useJit ? (b) => emu.run_jit(b) : (b) => emu.run(b);
  // Warm: don't time the first batch (alloc, initial compiles, page-in).
  run(BATCH);
  const start = Date.now();
  let stopped = false;
  while (Number(emu.total_insns()) < BUDGET) {
    if (run(BATCH) === 1) { stopped = true; break; }
  }
  const secs = (Date.now() - start) / 1000;
  emu.take_uart();
  const insns = Number(emu.total_insns());
  const jit = Number(emu.jit_insns());
  const calls = Number(emu.jit_calls());
  return { secs, insns, jit, calls, stopped };
}

console.log(`budget ${(BUDGET / 1e6).toFixed(0)}M instructions, batch ${(BATCH / 1e6).toFixed(0)}M\n`);

const i = bench(false);
console.log(`interp : ${(i.insns / 1e6 / i.secs).toFixed(1)} MIPS  (${i.insns} insns in ${i.secs.toFixed(2)}s)`);

const j = bench(true);
const cov = ((j.jit / j.insns) * 100).toFixed(1);
const avgLen = (j.jit / j.calls).toFixed(1);
console.log(`jit    : ${(j.insns / 1e6 / j.secs).toFixed(1)} MIPS  (${j.insns} insns in ${j.secs.toFixed(2)}s, ${cov}% via ${j.calls} block calls, avg ${avgLen} insns/call)`);

const interpRate = i.insns / i.secs, jitRate = j.insns / j.secs;
const ratio = jitRate >= interpRate ? jitRate / interpRate : interpRate / jitRate;
console.log(`\njit is ${ratio.toFixed(2)}x ${jitRate >= interpRate ? "faster" : "slower"} than interp`);
