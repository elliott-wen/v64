#!/usr/bin/env node
// Static server for the v64 web front-end. Sets the cross-origin isolation
// headers SharedArrayBuffer needs (the emulator runs in a Web Worker that shares
// linear memory with the WebGL renderer), and serves from the repo root so
// uitest.html's `../../guest/prebuilt/*` asset paths resolve.
//
//   node crates/web/serve.cjs [port]      # default 8000
//   then open http://<host>:<port>/  -> pick a target -> Boot
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');
const PORT = parseInt(process.argv[2] || '8000', 10);
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.css': 'text/css', '.json': 'application/json',
};

http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/crates/web/uitest.html';
  const fp = path.normalize(path.join(ROOT, p));
  if (!fp.startsWith(ROOT)) { res.writeHead(403); return res.end('forbidden'); }
  fs.stat(fp, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, {
      // Cross-origin isolation -> enables SharedArrayBuffer.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Content-Type': MIME[path.extname(fp)] || 'application/octet-stream',
      'Content-Length': st.size,
      'Cache-Control': 'no-cache',
    });
    fs.createReadStream(fp).pipe(res);
  });
}).listen(PORT, '0.0.0.0', () => {
  console.log(`v64 web server: http://0.0.0.0:${PORT}/crates/web/uitest.html`);
});
