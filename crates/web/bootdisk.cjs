// Boot the X-desktop ext4 rootfs in node via the wasm-compiled emulator, JIT
// mode, streaming the serial console. Headless functional check of the full
// stack (kernel + virtio-blk root + userspace init) — no browser needed.
//   crates/web/build.sh && node crates/web/bootdisk.cjs
const { readFileSync } = require("node:fs");
const { Emulator } = require("./pkg/aarch64_web.js");

const image = readFileSync("guest/prebuilt/Image-tiny");
const disk = readFileSync("guest/prebuilt/rootfs.ext4");
const bootargs =
  "console=ttyAMA0 root=/dev/vda rw rootfstype=ext4 random.trust_cpu=on";

const emu = new Emulator();
emu.boot_disk(image, disk, bootargs);
jitwarm();
function jitwarm() {}

const BATCH = 4_000_000;
const MAX_BATCHES = 1500;
const t0 = Date.now();
let out = "";
for (let i = 0; i < MAX_BATCHES; i++) {
  const status = emu.run_jit(BATCH);
  const u = emu.take_uart();
  if (u.length) {
    const s = Buffer.from(u).toString("latin1");
    process.stdout.write(s);
    out += s;
  }
  if (status === 1) { console.log("\n[guest powered off]"); break; }
  // Stop once X is starting (proves init ran the desktop autostart).
  if (out.includes("starting X desktop") || /v64 login:/.test(out)) {
    console.log("\n[reached login / X autostart]");
    break;
  }
}
console.log(`\n[${((Date.now() - t0) / 1000).toFixed(1)}s elapsed]`);
