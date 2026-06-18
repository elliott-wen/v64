// uitest emulator worker: runs the wasm emulator off the main thread and draws
// the virtio-gpu scanout straight into an OffscreenCanvas, so the page's main
// thread only forwards input and shows the console — it never blocks on
// emulation or pixel work. See uitest.html for the main-thread half.

import init, { Emulator } from './pkg-web/aarch64_web.js';

const KERNEL = '../../guest/prebuilt/Image-tiny';
const INITRD = '../../guest/prebuilt/uitest.cpio.gz';
const BOOTARGS = 'earlycon=pl011,0x9000000 console=ttyAMA0 rdinit=/init';
const CHUNK = 250_000;   // guest instructions per run() call
const SLICE_MS = 8;      // run chunks up to this long, then yield to messages

let emu = null;
let ctx = null;          // OffscreenCanvas 2d context
let canvas = null;
let imageData = null;
let useJit = true;
let running = false;

const dec = new TextDecoder('latin1');
let uart = '';
let statT = 0;
let statN = 0;

async function bytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function start(offscreen) {
  canvas = offscreen;
  ctx = canvas.getContext('2d');
  post({ ev: 'log', text: '[loading wasm + guest image…]\n' });
  await init();
  const [image, initrd] = await Promise.all([bytes(KERNEL), bytes(INITRD)]);
  emu = new Emulator();
  emu.boot(image, initrd, BOOTARGS);
  post({ ev: 'log', text: `[booted: kernel ${image.length} B, initrd ${initrd.length} B]\n` });
  running = true;
  statT = performance.now();
  statN = 0;
  schedule(false);
}

function draw() {
  const px = emu.take_frame(); // B8G8R8A8, empty if no new frame
  if (px.length === 0) return;
  const w = emu.fb_width(), h = emu.fb_height();
  if (w === 0 || h === 0) return;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h; imageData = null;
  }
  if (!imageData) imageData = ctx.createImageData(w, h);
  // BGRA (LE u32 = A<<24|R<<16|G<<8|B) -> RGBA, opaque, a word at a time.
  const src = new Uint32Array(px.buffer, px.byteOffset, px.length >> 2);
  const dst = new Uint32Array(imageData.data.buffer);
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    dst[i] = 0xff000000 | ((v & 0xff) << 16) | (v & 0x0000ff00) | ((v >> 16) & 0xff);
  }
  ctx.putImageData(imageData, 0, 0);
}

function tick() {
  if (!running) return;
  const t = performance.now();
  let idle = false, powered = false;
  do {
    const before = Number(emu.total_insns());
    const status = useJit ? emu.run_jit(CHUNK) : emu.run(CHUNK);
    const u = emu.take_uart();
    if (u.length) uart += dec.decode(u);
    if (status === 1) { powered = true; break; }
    // A chunk that didn't fill means the guest hit WFI on poll() — it's idle.
    if (Number(emu.total_insns()) - before < CHUNK) { idle = true; break; }
  } while (performance.now() - t < SLICE_MS);

  draw();
  if (uart) { post({ ev: 'uart', text: uart }); uart = ''; }

  const now = performance.now();
  const total = Number(emu.total_insns());
  if (now - statT > 250) {
    const mips = (total - statN) / 1e6 / ((now - statT) / 1000);
    const cov = total ? (100 * Number(emu.jit_insns()) / total) : 0;
    post({ ev: 'stats', insns: total, mips, cov, jit: useJit });
    statT = now; statN = total;
  }

  if (powered) { post({ ev: 'off' }); running = false; return; }
  schedule(idle);
}

// Reschedule the next tick: when busy, ASAP via a MessageChannel (no setTimeout
// clamp, and incoming input messages are drained between ticks); when idle,
// back off so an idle guest doesn't spin the CPU.
const chan = new MessageChannel();
chan.port1.onmessage = tick;
function schedule(idle) {
  if (idle) setTimeout(tick, 12);
  else chan.port2.postMessage(0);
}

onmessage = (e) => {
  const m = e.data;
  switch (m.cmd) {
    case 'start': start(m.canvas).catch((err) => post({ ev: 'log', text: `\n[error: ${err.message || err}]\n` })); break;
    case 'jit': useJit = m.on; break;
    case 'key': if (emu) emu.key(m.code, m.down); break;
    case 'motion': if (emu) emu.mouse_motion(m.dx, m.dy, m.wheel); break;
    case 'button': if (emu) emu.mouse_button(m.code, m.down); break;
  }
};

function post(msg) { postMessage(msg); }
