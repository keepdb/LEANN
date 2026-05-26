import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { loadLeannIndex } from "../src/index.js";

const tmpDir = new URL("../.tmp/hnsw-capability/", import.meta.url);
const requireHnsw = process.env.LEANN_WASM_REQUIRE_HNSW === "1";

if (!existsSync(tmpDir)) {
  const payload = {
    status: "missing-hnsw-fixture",
    required: requireHnsw,
    fixtureDir: tmpDir.pathname,
  };
  console.log(JSON.stringify(payload, null, 2));
  if (requireHnsw) {
    process.exit(1);
  }
  process.exit(0);
}

const artifacts = [
  "hnsw-capability.leann.meta.json",
  "hnsw-capability.index",
  "hnsw-capability.ids.txt",
  "hnsw-capability.leann.passages.jsonl",
];

for (const artifact of artifacts) {
  if (!existsSync(new URL(artifact, tmpDir))) {
    const payload = {
      status: "missing-hnsw-fixture",
      required: requireHnsw,
      fixtureDir: tmpDir.pathname,
      missing: artifact,
    };
    console.log(JSON.stringify(payload, null, 2));
    if (requireHnsw) {
      process.exit(1);
    }
    process.exit(0);
  }
}

const meta = JSON.parse(await readFile(new URL("hnsw-capability.leann.meta.json", tmpDir), "utf8"));
assert.equal(meta.backend_name, "hnsw");
assert.equal(meta.is_compact, false);
assert.equal(meta.is_pruned, false);

const wasmPath = new URL("../dist/leann-wasm.wasm", import.meta.url);
assert.ok(existsSync(wasmPath), "dist/leann-wasm.wasm missing. Run build:wasm before HNSW capability.");

const index = await loadLeannIndex({
  metaJson: await readFile(new URL("hnsw-capability.leann.meta.json", tmpDir), "utf8"),
  indexBytes: await readFile(new URL("hnsw-capability.index", tmpDir)),
  idsText: await readFile(new URL("hnsw-capability.ids.txt", tmpDir), "utf8"),
  passagesJsonl: await readFile(new URL("hnsw-capability.leann.passages.jsonl", tmpDir), "utf8"),
  wasmBytes: await readFile(wasmPath),
  metric: "cosine",
});
const results = index.search({ query: new Float32Array([1, 0, 0, 0]), topK: 2, complexity: 8 });
assert.equal(results[0]?.id, "doc-wasm");

console.log(JSON.stringify({
  status: "hnsw-fixture-generated-and-searched",
  artifacts,
  results,
  meta: {
    backend_name: meta.backend_name,
    dimensions: meta.dimensions,
    is_compact: meta.is_compact,
    is_pruned: meta.is_pruned,
    backend_kwargs: meta.backend_kwargs,
  },
}, null, 2));
