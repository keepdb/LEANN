import assert from "node:assert/strict";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";

import { loadLeannIndex } from "../src/index.js";
import { writeIvfNlistOneArtifacts } from "./leann-fixture-utils.mjs";

const envPath = new URL("../.env", import.meta.url);
loadDotEnv(envPath);

const apiKey = process.env.BIGMODEL_API_KEY || process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.log("BigModel LEANN e2e skipped: BIGMODEL_API_KEY is not set in packages/keepdb.wasm/.env");
  process.exit(0);
}

const baseUrl = process.env.BIGMODEL_BASE_URL || "https://open.bigmodel.cn/api/paas/v4";
const model = process.env.BIGMODEL_EMBEDDING_MODEL || "embedding-3";
const dimensions = optionalInt(process.env.BIGMODEL_EMBEDDING_DIMENSIONS);
const docs = JSON.parse(await readFile(new URL("../fixtures/bigmodel-e2e-docs.json", import.meta.url), "utf8"));
const queryText = "How can JavaScript use WebAssembly to run local vector search?";
const expectedTop1 = "doc-wasm";

const tmpDir = new URL("../.tmp/bigmodel-e2e/", import.meta.url);
await rm(tmpDir, { recursive: true, force: true });
await mkdir(tmpDir, { recursive: true });

console.log("Embedding documents with BigModel embedding-3...");
const docEmbeddings = await embedWithBigModel(docs.map((doc) => doc.text), {
  apiKey,
  baseUrl,
  model,
  dimensions,
});

await writeFile(
  new URL("docs-embeddings.json", tmpDir),
  JSON.stringify({ documents: docs, embeddings: docEmbeddings }, null, 2),
);

console.log("Generating JS-only LEANN IVF nlist=1 fixture artifacts...");
await writeIvfNlistOneArtifacts({
  outputUrl: tmpDir,
  indexName: "bigmodel.leann",
  documents: docs,
  embeddings: docEmbeddings,
  embeddingModel: model,
  baseUrl,
});

const metaJson = await readFile(new URL("bigmodel.leann.meta.json", tmpDir), "utf8");
const indexBytes = await readFile(new URL("bigmodel.index", tmpDir));
const idsText = await readFile(new URL("bigmodel.ids.txt", tmpDir), "utf8");
const passagesJsonl = await readFile(new URL("bigmodel.leann.passages.jsonl", tmpDir), "utf8");

console.log("Embedding query with BigModel embedding-3...");
const [queryEmbedding] = await embedWithBigModel([queryText], {
  apiKey,
  baseUrl,
  model,
  dimensions,
});

console.log("Loading JS-generated LEANN-compatible IVF artifacts through @keepdb/leann-wasm...");
const wasmPath = new URL("../dist/leann-wasm.wasm", import.meta.url);
if (!existsSync(wasmPath)) {
  throw new Error("dist/leann-wasm.wasm missing. Run `pnpm --dir packages/keepdb.wasm build:wasm` first.");
}

const wasmBytes = await readFile(wasmPath);
const index = await loadLeannIndex({
  metaJson,
  indexBytes,
  idsText,
  passagesJsonl,
  wasmBytes,
  embeddingProvider: {
    type: "custom",
    async embed() {
      return queryEmbedding;
    },
  },
});

const results = await index.searchText(queryText, { topK: 3 });
assert.equal(results[0]?.id, expectedTop1);
assert.ok(results.length >= 1);

console.log(JSON.stringify({ expectedTop1, results }, null, 2));
console.log("BigModel LEANN WASM e2e passed");

function loadDotEnv(url) {
  if (!existsSync(url)) {
    return;
  }
  const text = readFileSync(url, "utf8");
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 0) continue;
    const key = trimmed.slice(0, eq).trim();
    const value = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
    if (!(key in process.env)) {
      process.env[key] = value;
    }
  }
}

async function embedWithBigModel(input, { apiKey, baseUrl, model, dimensions }) {
  const body = { model, input };
  if (dimensions) {
    body.dimensions = dimensions;
  }

  const response = await fetch(`${baseUrl.replace(/\/+$/, "")}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`BigModel embedding failed: ${response.status} ${text}`);
  }

  const payload = await response.json();
  return payload.data.map((item) => item.embedding);
}

function optionalInt(value) {
  if (!value) return undefined;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}
