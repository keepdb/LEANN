import assert from "node:assert/strict";

import { loadLeannIndex } from "../src/index.js";

const index = await loadLeannIndex({
  dimension: 3,
  metric: "cosine",
  vectors: [
    { id: "wasm-doc", vector: [1, 0, 0] },
    { id: "embedding-doc", vector: [0, 1, 0] },
    { id: "provider-doc", vector: [0, 0, 1] },
  ],
  embeddingProvider: {
    type: "custom",
    async embed(text) {
      if (text.includes("provider")) {
        return [0, 0, 1];
      }
      return [1, 0, 0];
    },
  },
});

const vectorResults = index.search({
  query: new Float32Array([1, 0, 0]),
  topK: 1,
});

assert.equal(vectorResults.length, 1);
assert.equal(vectorResults[0].id, "wasm-doc");

const textResults = await index.searchText("provider config", {
  topK: 1,
});

assert.equal(textResults.length, 1);
assert.equal(textResults[0].id, "provider-doc");

index.dispose();

console.log("keepdb.wasm smoke passed");
