export async function createMockSearchCore(options = {}) {
  const {
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

  for (const item of normalizedVectors) {
    assertDimension(item.vector, inferredDimension, `vector '${item.id}'`);
  }

  return {
    search({ query, topK = 10 } = {}) {
      const queryVector = toFloat32Array(query);
      assertDimension(queryVector, inferredDimension, "query");

      return normalizedVectors
        .map((item) => ({
          id: item.id,
          label: item.label,
          score: score(queryVector, item.vector, metric),
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
    },

    dispose() {},
  };
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

function assertDimension(vector, dimension, name) {
  if (vector.length !== dimension) {
    throw new Error(`${name} dimension mismatch: expected ${dimension}, got ${vector.length}.`);
  }
}

function score(a, b, metric) {
  if (metric === "l2") {
    return -l2Distance(a, b);
  }
  if (metric === "mips") {
    return dot(a, b);
  }
  if (metric === "cosine") {
    return cosine(a, b);
  }
  throw new Error(`Unsupported metric: ${metric}`);
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += a[i] * b[i];
  }
  return total;
}

function l2Distance(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    total += diff * diff;
  }
  return Math.sqrt(total);
}

function cosine(a, b) {
  const denom = Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b));
  if (denom === 0) {
    return 0;
  }
  return dot(a, b) / denom;
}
