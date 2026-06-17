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

cargo build -p aarch64-web --target wasm32-unknown-unknown --release
wasm-bindgen "$WASM" --out-dir crates/web/pkg --target "$TARGET"
echo "bindings generated in crates/web/pkg ($TARGET target)"
