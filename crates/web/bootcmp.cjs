// Differential boot: run interp and JIT to a deep marker, diff the console.
// Validates the JIT stays byte-correct well past the banner (through MMU-on,
// context switches, TLB flushes) — not just to the banner.
const { readFileSync } = require("node:fs");
const { Emulator } = require("./pkg/aarch64_web.js");

const image = readFileSync("guest/prebuilt/Image-tiny");
const initrd = readFileSync("guest/prebuilt/uitest.cpio.gz");
const bootargs = "earlycon=pl011,0x9000000 console=ttyAMA0 rdinit=/init";

// Deterministic-ish deep target: run a fixed instruction budget, no wall-clock
// stop. (Timestamps may differ, but early/mid boot text is deterministic.)
const BATCH = 2_000_000;
const BUDGET = Number(process.env.BUDGET || 80_000_000);

function boot(useJit) {
  const emu = new Emulator();
  emu.boot(image, initrd, bootargs);
  const run = useJit ? (b) => emu.run_jit(b) : (b) => emu.run(b);
  let out = "";
  while (Number(emu.total_insns()) < BUDGET) {
    const st = run(BATCH);
    const u = emu.take_uart();
    if (u.length) out += Buffer.from(u).toString("latin1");
    if (st === 1) break;
  }
  return { out, insns: Number(emu.total_insns()) };
}

const a = boot(false);
const b = boot(true);
console.log(`interp: ${a.out.length} bytes / ${a.insns} insns`);
console.log(`jit:    ${b.out.length} bytes / ${b.insns} insns`);

// Compare ignoring the "[   12.345678]" printk timestamps (wall-clock dependent).
const strip = (s) => s.replace(/\[\s*\d+\.\d+\]/g, "[TS]");
const sa = strip(a.out), sb = strip(b.out);
if (sa === sb) {
  console.log(`MATCH: interp and JIT consoles identical (${sa.length} bytes, timestamps masked)`);
} else {
  let i = 0;
  while (i < sa.length && i < sb.length && sa[i] === sb[i]) i++;
  console.log(`DIVERGE at byte ${i}:`);
  console.log("  interp: " + JSON.stringify(sa.slice(Math.max(0, i - 40), i + 40)));
  console.log("  jit:    " + JSON.stringify(sb.slice(Math.max(0, i - 40), i + 40)));
  process.exit(1);
}
