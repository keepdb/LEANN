import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";

import { loadLeannIndex } from "../src/index.js";

const wasmPath = new URL("../dist/leann-wasm.wasm", import.meta.url);
if (!existsSync(wasmPath)) {
  console.log("HNSW WASM smoke skipped: dist/leann-wasm.wasm is missing.");
  process.exit(0);
}
const wasmBytes = await readFile(wasmPath);
const wasmProbe = await WebAssembly.instantiate(wasmBytes, {});
if (!("leann_wasm_hnsw_search" in wasmProbe.instance.exports)) {
  console.log("HNSW WASM smoke skipped: current local WASM lacks leann_wasm_hnsw_search.");
  process.exit(0);
}

const meta = {
  backend_name: "hnsw",
  dimensions: 4,
  is_compact: false,
  is_pruned: false,
  backend_kwargs: {
    distance_metric: "cosine",
    is_compact: false,
    is_recompute: false,
  },
};

const index = await loadLeannIndex({
  metaJson: meta,
  indexBytes: createSyntheticHnswIndex([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0.7, 0.2, 0.1, 0],
    [0, 0, 0, 1],
  ]),
  idsText: "doc-wasm\ndoc-bigmodel\ndoc-leann\ndoc-weather\n",
  wasmBytes,
  metric: "cosine",
});

const results = index.search({
  query: new Float32Array([1, 0, 0, 0]),
  topK: 2,
  complexity: 4,
});

assert.equal(results[0]?.id, "doc-wasm");
assert.equal(results.length, 2);
console.log("HNSW WASM smoke passed");

function createSyntheticHnswIndex(rows) {
  const writer = new BinaryWriter();
  writer.ascii("IHNf");
  writer.indexHeader({ dimension: 4, ntotal: rows.length, metricType: 0 });
  writer.float64Vector([1]);
  writer.int32Vector([0, 4]);
  writer.int32Vector([1, 1, 1, 1]);
  writer.bool(false);
  writer.u64Vector([0, 4, 8, 12, 16]);
  writer.int32Vector([
    1, 2, 3, -1,
    0, 2, 3, -1,
    0, 1, 3, -1,
    0, 1, 2, -1,
  ]);
  writer.int32(0);
  writer.int32(0);
  writer.int32(40);
  writer.int32(16);
  writer.int32(1);
  writer.ascii("IxFI");
  writer.indexHeader({ dimension: 4, ntotal: rows.length, metricType: 0 });
  writer.u64(rows.length * 4);
  for (const row of rows) {
    for (const value of row) writer.float32(value);
  }
  return writer.toUint8Array();
}

class BinaryWriter {
  constructor() {
    this.bytes = [];
  }

  ascii(value) {
    for (let i = 0; i < value.length; i += 1) this.bytes.push(value.charCodeAt(i));
  }

  bool(value) {
    this.bytes.push(value ? 1 : 0);
  }

  int32(value) {
    this.write(4, (view) => view.setInt32(0, value, true));
  }

  int64(value) {
    this.write(8, (view) => view.setBigInt64(0, BigInt(value), true));
  }

  u64(value) {
    this.write(8, (view) => view.setBigUint64(0, BigInt(value), true));
  }

  float32(value) {
    this.write(4, (view) => view.setFloat32(0, value, true));
  }

  float64(value) {
    this.write(8, (view) => view.setFloat64(0, value, true));
  }

  indexHeader({ dimension, ntotal, metricType }) {
    this.int32(dimension);
    this.int64(ntotal);
    this.int64(1 << 20);
    this.int64(1 << 20);
    this.bool(true);
    this.int32(metricType);
  }

  float64Vector(values) {
    this.u64(values.length);
    for (const value of values) this.float64(value);
  }

  int32Vector(values) {
    this.u64(values.length);
    for (const value of values) this.int32(value);
  }

  u64Vector(values) {
    this.u64(values.length);
    for (const value of values) this.u64(value);
  }

  write(length, callback) {
    const buffer = new ArrayBuffer(length);
    callback(new DataView(buffer));
    this.bytes.push(...new Uint8Array(buffer));
  }

  toUint8Array() {
    return new Uint8Array(this.bytes);
  }
}
