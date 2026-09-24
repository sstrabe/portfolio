#!/usr/bin/env bash
# Build the physics/rendering engine to WebAssembly and generate JS bindings
# into web/src/wasm/pkg. Requires the wasm32 target and a wasm-bindgen CLI
# matching the crate version (cargo install wasm-bindgen-cli --version 0.2.128).
set -euo pipefail
cd "$(dirname "$0")/.."

profile="${1:-release}"
dir="$profile"
[ "$profile" = dev ] && dir=debug
cargo build -p engine --target wasm32-unknown-unknown --profile "$profile"
out=web/src/wasm/pkg
rm -rf "$out"
wasm-bindgen --target web --out-dir "$out" --out-name engine \
  "target/wasm32-unknown-unknown/$dir/engine.wasm"

if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$out/engine_bg.wasm" -o "$out/engine_bg.wasm"
fi
ls -l "$out"
