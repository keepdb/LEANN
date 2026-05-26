#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/dist"

mkdir -p "${OUT_DIR}"

if command -v emcc >/dev/null 2>&1; then
  emcc "${ROOT_DIR}/wasm/leann_wasm.c" \
    -O3 \
    --no-entry \
    -s STANDALONE_WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_malloc","_free","_leann_wasm_version","_leann_wasm_score_dot","_leann_wasm_flat_search","_leann_wasm_hnsw_search"]' \
    -o "${OUT_DIR}/leann-wasm.wasm"
elif command -v wat2wasm >/dev/null 2>&1; then
  wat2wasm "${ROOT_DIR}/wasm/leann_wasm.wat" \
    -o "${OUT_DIR}/leann-wasm.wasm"
elif command -v clang >/dev/null 2>&1; then
  clang \
    --target=wasm32-unknown-unknown-wasm \
    -O3 \
    -nostdlib \
    -Wl,--no-entry \
    -Wl,--export-memory \
    -Wl,--export=malloc \
    -Wl,--export=free \
    -Wl,--export=leann_wasm_version \
    -Wl,--export=leann_wasm_score_dot \
    -Wl,--export=leann_wasm_flat_search \
    -Wl,--export=leann_wasm_hnsw_search \
    -Wl,--allow-undefined \
    "${ROOT_DIR}/wasm/leann_wasm.c" \
    -o "${OUT_DIR}/leann-wasm.wasm"
else
  echo "emcc or clang is required to build WASM." >&2
  exit 1
fi

ls -lh "${OUT_DIR}/leann-wasm.wasm"
