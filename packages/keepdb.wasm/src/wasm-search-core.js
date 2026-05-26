import { loadWasmModule } from "./wasm-loader.js";

export async function createWasmSearchCore(options = {}) {
  const { wasmBytes, wasmUrl } = options;
  if (!wasmBytes && !wasmUrl) {
    return null;
  }

  const wasm = await loadWasmModule({ wasmBytes, wasmUrl });
  if (!wasm?.instance) {
    return null;
  }

  const { instance } = wasm;
  const exports = instance.exports;
  assertExport(exports, "memory");
  const malloc = getExport(exports, "malloc");
  const free = getExport(exports, "free");
  const flatSearch = getExport(exports, "leann_wasm_flat_search");
  const hnswSearch = exports.leann_wasm_hnsw_search ?? exports._leann_wasm_hnsw_search;

  if (options.hnsw) {
    if (!hnswSearch) {
      throw new Error("WASM export 'leann_wasm_hnsw_search' is required for HNSW indexes.");
    }
    return createHnswSearchCore({ exports, malloc, free, hnswSearch, ...options });
  }

  return createFlatSearchCore({ exports, malloc, free, flatSearch, ...options });
}

function createFlatSearchCore(options) {
  const {
    exports,
    malloc,
    free,
    flatSearch,
    dimension,
    metric = "cosine",
    vectors = defaultVectors(),
  } = options;

  const normalizedVectors = vectors.map((item, label) => ({
    id: item.id ?? String(label),
    label,
    vector: toFloat32Array(item.vector),
  }));

  const inferredDimension = dimension ?? normalizedVectors[0]?.vector.length;
  if (!inferredDimension) {
    throw new Error("dimension is required when vectors are empty.");
  }

  const vectorCount = normalizedVectors.length;
  const flatVectors = new Float32Array(vectorCount * inferredDimension);

  normalizedVectors.forEach((item, row) => {
    assertDimension(item.vector, inferredDimension, `vector '${item.id}'`);
    flatVectors.set(item.vector, row * inferredDimension);
  });

  const vectorPtr = malloc(flatVectors.byteLength);
  if (!vectorPtr) {
    throw new Error("Failed to allocate WASM vector memory.");
  }
  floatView(exports, vectorPtr, flatVectors.length).set(flatVectors);

  let disposed = false;

  return {
    search({ query, topK = 10 } = {}) {
      if (disposed) {
        throw new Error("Search core has been disposed.");
      }

      const queryVector = toFloat32Array(query);
      assertDimension(queryVector, inferredDimension, "query");

      const k = Math.max(0, Math.min(topK, vectorCount));
      if (k === 0) {
        return [];
      }

      const queryPtr = malloc(queryVector.byteLength);
      const labelsPtr = malloc(k * Int32Array.BYTES_PER_ELEMENT);
      const scoresPtr = malloc(k * Float32Array.BYTES_PER_ELEMENT);

      try {
        if (!queryPtr || !labelsPtr || !scoresPtr) {
          throw new Error("Failed to allocate WASM search memory.");
        }

        floatView(exports, queryPtr, queryVector.length).set(queryVector);

        const status = flatSearch(
          vectorPtr,
          vectorCount,
          inferredDimension,
          queryPtr,
          k,
          metricCode(metric),
          labelsPtr,
          scoresPtr,
        );

        if (status !== 0) {
          throw new Error(`WASM flat search failed with status ${status}.`);
        }

        const labels = new Int32Array(exports.memory.buffer, labelsPtr, k);
        const scores = new Float32Array(exports.memory.buffer, scoresPtr, k);

        return Array.from({ length: k }, (_, rank) => {
          const label = labels[rank];
          const item = normalizedVectors[label];
          return {
            id: item?.id ?? String(label),
            label,
            score: scores[rank],
          };
        });
      } finally {
        if (queryPtr) free(queryPtr);
        if (labelsPtr) free(labelsPtr);
        if (scoresPtr) free(scoresPtr);
      }
    },

    dispose() {
      if (!disposed) {
        free(vectorPtr);
        disposed = true;
      }
    },
  };
}

function createHnswSearchCore(options) {
  const {
    exports,
    malloc,
    free,
    hnswSearch,
    dimension,
    metric = "cosine",
    vectors = [],
    hnsw,
  } = options;

  if (!hnsw) {
    throw new Error("hnsw graph data is required.");
  }

  const normalizedVectors = vectors.map((item, label) => ({
    id: item.id ?? String(label),
    label,
    vector: toFloat32Array(item.vector),
  }));
  const inferredDimension = dimension ?? normalizedVectors[0]?.vector.length;
  if (!inferredDimension) {
    throw new Error("dimension is required when vectors are empty.");
  }

  const vectorCount = normalizedVectors.length;
  const flatVectors = new Float32Array(vectorCount * inferredDimension);
  normalizedVectors.forEach((item, row) => {
    assertDimension(item.vector, inferredDimension, `vector '${item.id}'`);
    flatVectors.set(item.vector, row * inferredDimension);
  });

  const offsets = toBigUint64Array(hnsw.offsets);
  const neighbors = toInt32Array(hnsw.neighbors);
  const levels = toInt32Array(hnsw.levels);
  const cumNeighbors = toInt32Array(hnsw.cumNeighbors);

  const vectorPtr = copyToWasm(exports, malloc, flatVectors);
  const offsetsPtr = copyToWasm(exports, malloc, offsets);
  const neighborsPtr = copyToWasm(exports, malloc, neighbors);
  const levelsPtr = copyToWasm(exports, malloc, levels);
  const cumNeighborsPtr = copyToWasm(exports, malloc, cumNeighbors);
  let disposed = false;

  return {
    search({ query, topK = 10, complexity = 64 } = {}) {
      if (disposed) {
        throw new Error("Search core has been disposed.");
      }

      const queryVector = toFloat32Array(query);
      assertDimension(queryVector, inferredDimension, "query");
      const k = Math.max(0, Math.min(topK, vectorCount));
      if (k === 0) {
        return [];
      }

      const queryPtr = malloc(queryVector.byteLength);
      const labelsPtr = malloc(k * Int32Array.BYTES_PER_ELEMENT);
      const scoresPtr = malloc(k * Float32Array.BYTES_PER_ELEMENT);
      try {
        if (!queryPtr || !labelsPtr || !scoresPtr) {
          throw new Error("Failed to allocate WASM HNSW search memory.");
        }

        floatView(exports, queryPtr, queryVector.length).set(queryVector);

        const status = hnswSearch(
          vectorPtr,
          vectorCount,
          inferredDimension,
          offsetsPtr,
          neighborsPtr,
          levelsPtr,
          cumNeighborsPtr,
          cumNeighbors.length,
          hnsw.entryPoint,
          hnsw.maxLevel,
          queryPtr,
          k,
          Math.max(k, complexity),
          metricCode(metric),
          labelsPtr,
          scoresPtr,
        );

        if (status !== 0) {
          throw new Error(`WASM HNSW search failed with status ${status}.`);
        }

        const labels = new Int32Array(exports.memory.buffer, labelsPtr, k);
        const scores = new Float32Array(exports.memory.buffer, scoresPtr, k);
        return Array.from({ length: k }, (_, rank) => {
          const label = labels[rank];
          const item = normalizedVectors[label];
          return {
            id: item?.id ?? String(label),
            label,
            score: scores[rank],
          };
        });
      } finally {
        if (queryPtr) free(queryPtr);
        if (labelsPtr) free(labelsPtr);
        if (scoresPtr) free(scoresPtr);
      }
    },

    dispose() {
      if (!disposed) {
        free(vectorPtr);
        free(offsetsPtr);
        free(neighborsPtr);
        free(levelsPtr);
        free(cumNeighborsPtr);
        disposed = true;
      }
    },
  };
}

function assertExport(exports, name) {
  if (!(name in exports) && !(`_${name}` in exports)) {
    throw new Error(`WASM export '${name}' is required.`);
  }
}

function getExport(exports, name) {
  const value = exports[name] ?? exports[`_${name}`];
  if (!value) {
    throw new Error(`WASM export '${name}' is required.`);
  }
  return value;
}

function defaultVectors() {
  return [
    { id: "wasm", vector: [1, 0, 0] },
    { id: "embedding", vector: [0, 1, 0] },
    { id: "provider", vector: [0, 0, 1] },
  ];
}

function toFloat32Array(value) {
  if (value instanceof Float32Array) {
    return value;
  }
  if (Array.isArray(value)) {
    return new Float32Array(value);
  }
  if (value instanceof ArrayBuffer) {
    return new Float32Array(value);
  }
  throw new TypeError("Expected vector to be Float32Array, Array, or ArrayBuffer.");
}

function toInt32Array(value) {
  if (value instanceof Int32Array) {
    return value;
  }
  if (Array.isArray(value)) {
    return new Int32Array(value);
  }
  throw new TypeError("Expected Int32Array or number[].");
}

function toBigUint64Array(value) {
  if (value instanceof BigUint64Array) {
    return value;
  }
  if (Array.isArray(value)) {
    return new BigUint64Array(value.map((item) => BigInt(item)));
  }
  throw new TypeError("Expected BigUint64Array or number[].");
}

function assertDimension(vector, dimension, name) {
  if (vector.length !== dimension) {
    throw new Error(`${name} dimension mismatch: expected ${dimension}, got ${vector.length}.`);
  }
}

function floatView(exports, ptr, length) {
  return new Float32Array(exports.memory.buffer, ptr, length);
}

function copyToWasm(exports, malloc, typedArray) {
  const ptr = malloc(typedArray.byteLength);
  if (!ptr) {
    throw new Error("Failed to allocate WASM memory.");
  }
  new Uint8Array(exports.memory.buffer, ptr, typedArray.byteLength)
    .set(new Uint8Array(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength));
  return ptr;
}

function metricCode(metric) {
  if (metric === "mips") return 0;
  if (metric === "cosine") return 1;
  if (metric === "l2") return 2;
  throw new Error(`Unsupported metric: ${metric}`);
}
