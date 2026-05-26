import { createEmbeddingProvider } from "./providers.js";
import { createMockSearchCore } from "./mock-search-core.js";
import { loadWasmModule } from "./wasm-loader.js";
import { createWasmSearchCore } from "./wasm-search-core.js";
import { parseLeannIndex } from "./leann-index-parser.js";
import { parseLeannSidecars } from "./leann-sidecars.js";

export const version = "0.0.0-m3";

export async function loadLeannIndex(options = {}) {
  const {
    dimension,
    metric = "cosine",
    embeddingProvider,
    vectors,
    metaJson,
    passagesJsonl,
    idsText,
    indexBytes,
    wasmBytes,
    wasmUrl,
  } = options;

  const provider = embeddingProvider ? createEmbeddingProvider(embeddingProvider) : null;
  const sidecars = parseLeannSidecars({ metaJson, idsText, passagesJsonl });
  const parsedIndex = indexBytes
    ? parseLeannIndex({
        meta: sidecars.meta,
        indexBytes,
        ids: sidecars.ids,
      })
    : undefined;
  const resolvedVectors = vectors ?? parsedIndex?.vectors;
  const resolvedDimension = dimension ?? sidecars.meta?.dimensions;
  const wasmCore = await createWasmSearchCore({
    dimension: resolvedDimension,
    metric,
    vectors: resolvedVectors,
    hnsw: parsedIndex?.hnsw,
    indexBytes,
    idsText,
    wasmBytes,
    wasmUrl,
  });
  const core = wasmCore ?? await createMockSearchCore({
    dimension: resolvedDimension,
    metric,
    vectors: resolvedVectors,
    indexBytes,
    idsText,
  });

  return {
    search({ query, topK = 10, complexity = 64 } = {}) {
      return enrichResults(core.search({ query, topK, complexity }), sidecars);
    },

    async searchText(text, { topK = 10, complexity = 64 } = {}) {
      if (!provider) {
        throw new Error("searchText() requires embeddingProvider in loadLeannIndex().");
      }

      const query = await provider.embed(text);
      return enrichResults(core.search({ query, topK, complexity }), sidecars);
    },

    dispose() {
      core.dispose();
    },

    mode: wasmCore ? "wasm" : "mock",
  };
}

export async function loadLeannWasm(options = {}) {
  const wasm = await loadWasmModule(options);
  return wasm ?? {
    version,
    mode: "mock",
  };
}

export { createEmbeddingProvider };

function enrichResults(results, sidecars) {
  if (!sidecars?.passageById?.size) {
    return results;
  }

  return results.map((result) => {
    const passage = sidecars.passageById.get(result.id);
    if (!passage) {
      return result;
    }
    return {
      ...result,
      text: passage.text,
      metadata: passage.metadata ?? {},
    };
  });
}
