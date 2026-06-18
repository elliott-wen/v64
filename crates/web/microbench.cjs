// Microbenchmarks: isolate workload patterns to see the JIT's per-pattern boost,
// away from the boot's cold-code-heavy mix. Each kernel is a tiny hand-encoded
// AArch64 loop run on a bare machine (MMU off), via interpreter vs JIT.
//
//   crates/web/build.sh && node crates/web/microbench.cjs
//
// Note: MMU off ⇒ no TLB ⇒ inline memory would bail, so these kernels are
// register/branch only (compute throughput). The boot bench covers memory.
const { Kernel } = require("./pkg/aarch64_web.js");

const u32 = (words) => {
  const b = Buffer.alloc(words.length * 4);
  words.forEach((w, i) => b.writeUInt32LE(w >>> 0, i * 4));
  return b;
};

// Each loops forever (cmp x1,x1 sets Z, b.eq always taken); the budget caps it.
const KERNELS = {
  // add x1,#1 ; eor x2,x2,x1 ; cmp x1,x1 ; b.eq -3   (tight ALU self-loop)
  "alu-tight  (3 ops/iter, self-loop)": u32([0x91000421, 0xca010042, 0xeb01003f, 0x54ffffa0]),
  // 6x add ; eor x7,x7,x1 ; cmp x1,x1 ; b.eq -8       (wider self-loop body)
  "alu-wide   (7 ops/iter, self-loop)": u32([
    0x91000421, 0x91000442, 0x91000463, 0x91000484, 0x910004a5, 0x910004c6, 0xca0100e7, 0xeb01003f,
    0x54ffff00,
  ]),
  // add x1,#1 ; tbz x1,#0,+2 ; add x2,#1 ; cmp x1,x1 ; b.eq -4  (multi-block loop)
  "branchy    (data-dependent branch each iter)": u32([
    0x91000421, 0x36000041, 0x91000442, 0xeb01003f, 0x54ffff80,
  ]),
};

const BUDGET = 300_000_000;
console.log(`budget ${(BUDGET / 1e6).toFixed(0)}M instructions/kernel\n`);

for (const [name, code] of Object.entries(KERNELS)) {
  const ki = new Kernel(code);
  let t = Date.now();
  const ni = Number(ki.run(BUDGET));
  const si = (Date.now() - t) / 1000;
  const x1i = Number(ki.x(1));

  const kj = new Kernel(code);
  t = Date.now();
  const nj = Number(kj.run_jit(BUDGET));
  const sj = (Date.now() - t) / 1000;
  const x1j = Number(kj.x(1));

  const mi = ni / 1e6 / si;
  const mj = nj / 1e6 / sj;
  // x1 counts loop iterations; interp and JIT should agree to within JIT batch
  // overshoot (<<1%). A mismatch means a miscompiled kernel.
  const drift = Math.abs(x1i - x1j) / Math.max(1, x1i);
  const warn = x1i > 0 && drift < 0.01 ? "" : `  [WARN x1 interp=${x1i} jit=${x1j}]`;
  console.log(`${name}`);
  console.log(`  interp ${mi.toFixed(1)} MIPS | jit ${mj.toFixed(1)} MIPS | ${(mj / mi).toFixed(2)}x faster${warn}`);
}
