import { mkdir, writeFile } from "node:fs/promises";

export const hnswCapabilityDocs = [
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

export const hnswCapabilityEmbeddings = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0.7, 0.2, 0.1, 0],
  [0, 0, 0, 1],
];

export async function writeHnswCapabilityInput(outputUrl) {
  await mkdir(outputUrl, { recursive: true });
  await writeFile(
    new URL("docs-embeddings.json", outputUrl),
    JSON.stringify({
      documents: hnswCapabilityDocs,
      embeddings: hnswCapabilityEmbeddings,
    }, null, 2),
  );
}

export async function writeIvfNlistOneArtifacts({
  outputUrl,
  indexName,
  documents,
  embeddings,
  embeddingModel = "embedding-3",
  baseUrl = "https://open.bigmodel.cn/api/paas/v4",
} = {}) {
  const dimension = embeddings[0]?.length;
  if (!dimension) {
    throw new Error("At least one embedding is required.");
  }
  if (documents.length !== embeddings.length) {
    throw new Error("documents and embeddings length mismatch.");
  }
  for (const [index, vector] of embeddings.entries()) {
    if (vector.length !== dimension) {
      throw new Error(`Embedding ${index} dimension mismatch.`);
    }
  }

  await mkdir(outputUrl, { recursive: true });
  const stem = indexName.replace(/\.leann$/, "");
  const idsText = `${documents.map((doc) => doc.id).join("\n")}\n`;
  const passagesJsonl = documents
    .map((doc) => JSON.stringify({
      id: doc.id,
      text: doc.text,
      metadata: {
        id: doc.id,
        source: "bigmodel-e2e",
      },
    }))
    .join("\n") + "\n";
  const meta = {
    version: "1.0",
    backend_name: "ivf",
    embedding_model: embeddingModel,
    dimensions: dimension,
    backend_kwargs: {
      distance_metric: "cosine",
      nlist: 1,
    },
    embedding_mode: "openai",
    passage_sources: [
      {
        type: "jsonl",
        path: `${indexName}.passages.jsonl`,
        path_relative: `${indexName}.passages.jsonl`,
      },
    ],
    built_from_precomputed_embeddings: true,
    embedding_options: {
      base_url: baseUrl,
    },
  };

  await Promise.all([
    writeFile(new URL(`${indexName}.meta.json`, outputUrl), JSON.stringify(meta, null, 2)),
    writeFile(new URL(`${stem}.index`, outputUrl), createIvfNlistOneIndex(embeddings)),
    writeFile(new URL(`${stem}.ids.txt`, outputUrl), idsText),
    writeFile(new URL(`${indexName}.passages.jsonl`, outputUrl), passagesJsonl),
  ]);
}

function createIvfNlistOneIndex(embeddings) {
  const dimension = embeddings[0].length;
  const writer = new BinaryWriter();
  writer.ascii("IwFl");
  writer.int32(dimension);
  writer.ascii("ilar");
  writer.u64(1);
  writer.u64(dimension * Float32Array.BYTES_PER_ELEMENT);
  writer.ascii("full");
  writer.u64(1);
  writer.u64(embeddings.length);
  for (const vector of embeddings) {
    for (const value of vector) {
      writer.float32(value);
    }
  }
  for (let id = 0; id < embeddings.length; id += 1) {
    writer.u64(id);
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

  int32(value) {
    this.write(4, (view) => view.setInt32(0, value, true));
  }

  u64(value) {
    this.write(8, (view) => view.setBigUint64(0, BigInt(value), true));
  }

  float32(value) {
    this.write(4, (view) => view.setFloat32(0, value, true));
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
