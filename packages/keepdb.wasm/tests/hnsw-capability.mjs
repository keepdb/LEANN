import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { loadLeannIndex } from "../src/index.js";

const rootUrl = new URL("..", import.meta.url);
const tmpDir = new URL("../.tmp/hnsw-capability/", import.meta.url);
const requireHnsw = process.env.LEANN_WASM_REQUIRE_HNSW === "1";

await rm(tmpDir, { recursive: true, force: true });
await mkdir(tmpDir, { recursive: true });

const docs = [
  {
    id: "doc-wasm",
    text: "WebAssembly lets JavaScript run a compiled search engine locally.",
  },
  {
    id: "doc-bigmodel",
    text: "Embedding providers convert text into vectors for semantic retrieval.",
  },
  {
    id: "doc-leann",
    text: "LEANN stores search indexes and passage files for local retrieval.",
  },
  {
    id: "doc-weather",
    text: "Weather reports describe temperature, rain, wind, and humidity.",
  },
];

const embeddings = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0.7, 0.2, 0.1, 0],
  [0, 0, 0, 1],
];

await writeFile(
  new URL("docs-embeddings.json", tmpDir),
  JSON.stringify({ documents: docs, embeddings }, null, 2),
);

const pythonPath = [
  new URL("../leann-core/src", rootUrl).pathname,
  new URL("../leann-backend-ivf", rootUrl).pathname,
  new URL("../leann-backend-hnsw", rootUrl).pathname,
  process.env.PYTHONPATH || "",
].filter(Boolean).join(":");

const pythonCommand = buildPythonCommand();
const result = spawnSync(
  pythonCommand.command,
  [
    ...pythonCommand.args,
    new URL("../scripts/build-leann-index-from-embeddings.py", import.meta.url).pathname,
    new URL("docs-embeddings.json", tmpDir).pathname,
    tmpDir.pathname,
    "hnsw-capability.leann",
  ],
  {
    encoding: "utf8",
    env: {
      ...process.env,
      PYTHONPATH: pythonPath,
      LEANN_WASM_E2E_BACKEND: "hnsw",
      BIGMODEL_EMBEDDING_MODEL: "embedding-3",
      BIGMODEL_BASE_URL: "https://open.bigmodel.cn/api/paas/v4",
      BIGMODEL_API_KEY: process.env.BIGMODEL_API_KEY || "not-used-for-precomputed-embeddings",
    },
  },
);

if (result.status !== 0) {
  const output = `${result.stdout || ""}\n${result.stderr || ""}`.trim();
  const payload = {
    status: "missing-hnsw-native-extension",
    required: requireHnsw,
    output,
  };
  console.log(JSON.stringify(payload, null, 2));
  if (requireHnsw) {
    process.exit(result.status || 1);
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
  assert.ok(existsSync(new URL(artifact, tmpDir)), `Missing HNSW artifact: ${artifact}`);
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

function buildPythonCommand() {
  if (process.env.LEANN_WASM_PYTHON) {
    return { command: process.env.LEANN_WASM_PYTHON, args: [] };
  }

  return {
    command: "uv",
    args: [
      "run",
      "--no-project",
      "--python",
      "3.11",
      "--with",
      "numpy",
      "--with",
      "faiss-cpu",
      "--with",
      "tqdm",
      "--with",
      "psutil",
      "--with",
      "pyzmq",
      "--with",
      "msgpack",
      "--with",
      "openai",
      "--with",
      "python-dotenv",
      "--with",
      "tiktoken",
      "--with",
      "requests",
      "--with",
      "torch",
    ],
  };
}
