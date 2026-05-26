import assert from "node:assert/strict";

import { parseLeannIndex } from "../src/leann-index-parser.js";

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

const vectors = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0.7, 0.2, 0.1, 0],
  [0, 0, 0, 1],
];

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
    for (const value of row) {
      writer.float32(value);
    }
  }
  return writer.toUint8Array();
}

class BinaryWriter {
  constructor() {
    this.bytes = [];
  }

  ascii(value) {
    for (let i = 0; i < value.length; i += 1) {
      this.bytes.push(value.charCodeAt(i));
    }
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

const parsed = parseLeannIndex({
  meta,
  indexBytes: createSyntheticHnswIndex(vectors),
  ids: ["doc-wasm", "doc-bigmodel", "doc-leann", "doc-weather"],
});

assert.equal(parsed.kind, "hnsw");
assert.equal(parsed.vectors.length, 4);
assert.equal(parsed.vectors[0].id, "doc-wasm");
assert.deepEqual(Array.from(parsed.vectors[0].vector), vectors[0]);
assert.equal(parsed.hnsw.entryPoint, 0);
assert.equal(parsed.hnsw.maxLevel, 0);
assert.deepEqual(Array.from(parsed.hnsw.levels), [1, 1, 1, 1]);
assert.deepEqual(Array.from(parsed.hnsw.cumNeighbors), [0, 4]);
assert.deepEqual(Array.from(parsed.hnsw.offsets, Number), [0, 4, 8, 12, 16]);
assert.deepEqual(Array.from(parsed.hnsw.neighbors.slice(0, 4)), [1, 2, 3, -1]);

console.log("HNSW parser smoke passed");
