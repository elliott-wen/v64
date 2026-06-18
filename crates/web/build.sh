#!/usr/bin/env bash
# Build the wasm module and generate node/browser bindings in one step.
#
#   crates/web/build.sh            # node bindings (default)
#   crates/web/build.sh web        # browser (ESM) bindings
#
# Then: `node crates/web/run.cjs` (node), or open crates/web/uitest.html (web).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

TARGET="${1:-nodejs}"
WASM=target/wasm32-unknown-unknown/release/aarch64_web.wasm
# Node and browser bindings live side by side: node scripts use pkg/, the
# browser page (uitest.html) imports pkg-web/.
OUT=crates/web/pkg
[ "$TARGET" = "web" ] && OUT=crates/web/pkg-web

# The JIT appends compiled block functions to the indirect function table and
# calls them via `call_indirect`, so the table must be exported (for JS to
# `table.set`) and growable (`table.grow`). `wasm_bindgen::function_table()`
# handles the export; `--growable-table` makes it growable.
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=--growable-table -C link-arg=--export-table"
cargo build -p aarch64-web --target wasm32-unknown-unknown --release
wasm-bindgen "$WASM" --out-dir "$OUT" --target "$TARGET"
echo "bindings generated in $OUT ($TARGET target)"
