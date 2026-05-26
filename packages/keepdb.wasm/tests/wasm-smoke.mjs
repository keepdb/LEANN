import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";

import { loadLeannIndex } from "../src/index.js";

const wasmPath = new URL("../dist/leann-wasm.wasm", import.meta.url);

if (!existsSync(wasmPath)) {
  console.log("WASM smoke skipped: dist/leann-wasm.wasm not found");
  process.exit(0);
}

const wasmBytes = await readFile(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const version = instance.exports.leann_wasm_version ?? instance.exports._leann_wasm_version;
const dot = instance.exports.leann_wasm_score_dot ?? instance.exports._leann_wasm_score_dot;
const flatSearch = instance.exports.leann_wasm_flat_search
  ?? instance.exports._leann_wasm_flat_search;

assert.equal(version(), 1);
assert.equal(typeof dot, "function");
assert.equal(typeof flatSearch, "function");

const index = await loadLeannIndex({
  wasmBytes,
  dimension: 3,
  metric: "cosine",
  vectors: [
    { id: "wasm-doc", vector: [1, 0, 0] },
    { id: "embedding-doc", vector: [0, 1, 0] },
    { id: "provider-doc", vector: [0, 0, 1] },
  ],
});

assert.equal(index.mode, "wasm");

const results = index.search({
  query: new Float32Array([0, 0, 1]),
  topK: 1,
});

assert.equal(results.length, 1);
assert.equal(results[0].id, "provider-doc");

index.dispose();

console.log("keepdb.wasm native search smoke passed");
