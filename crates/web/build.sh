#!/usr/bin/env bash
# Build the wasm module and generate node/browser bindings in one step.
#
#   crates/web/build.sh            # node bindings (default)
#   crates/web/build.sh web        # browser (ESM) bindings
#
# Then: `node crates/web/run.cjs` (node), or import crates/web/pkg in a page.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

TARGET="${1:-nodejs}"
WASM=target/wasm32-unknown-unknown/release/aarch64_web.wasm

# The JIT appends compiled block functions to the indirect function table and
# calls them via `call_indirect`, so the table must be exported (for JS to
# `table.set`) and growable (`table.grow`). `wasm_bindgen::function_table()`
# handles the export; `--growable-table` makes it growable.
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=--growable-table -C link-arg=--export-table"
cargo build -p aarch64-web --target wasm32-unknown-unknown --release
wasm-bindgen "$WASM" --out-dir crates/web/pkg --target "$TARGET"
echo "bindings generated in crates/web/pkg ($TARGET target)"
