export function createEmbeddingProvider(options = {}) {
  if (options.type === "openai-compatible") {
    return createOpenAICompatibleProvider(options);
  }
  if (options.type === "remote") {
    return createRemoteProvider(options);
  }
  if (options.type === "custom") {
    return createCustomProvider(options);
  }
  throw new Error(`Unsupported embeddingProvider.type: ${options.type}`);
}

function createOpenAICompatibleProvider(options) {
  const {
    baseUrl,
    apiKey,
    model,
    dimensions,
    headers = {},
    fetchImpl = globalThis.fetch,
  } = options;

  if (!baseUrl) {
    throw new Error("openai-compatible embeddingProvider requires baseUrl.");
  }
  if (!apiKey) {
    throw new Error("openai-compatible embeddingProvider requires apiKey.");
  }
  if (!model) {
    throw new Error("openai-compatible embeddingProvider requires model.");
  }
  if (!fetchImpl) {
    throw new Error("fetch is unavailable. Pass fetchImpl or use a runtime with fetch.");
  }

  return {
    async embed(text) {
      const body = {
        model,
        input: text,
      };
      if (dimensions) {
        body.dimensions = dimensions;
      }

      const response = await fetchImpl(joinUrl(baseUrl, "embeddings"), {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          ...headers,
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`Embedding request failed: ${response.status}`);
      }

      const payload = await response.json();
      return embeddingFromPayload(payload);
    },
  };
}

function createRemoteProvider(options) {
  const {
    endpoint,
    headers = {},
    fetchImpl = globalThis.fetch,
  } = options;

  if (!endpoint) {
    throw new Error("remote embeddingProvider requires endpoint.");
  }
  if (!fetchImpl) {
    throw new Error("fetch is unavailable. Pass fetchImpl or use a runtime with fetch.");
  }

  return {
    async embed(text) {
      const response = await fetchImpl(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...headers,
        },
        body: JSON.stringify({ input: text }),
      });

      if (!response.ok) {
        throw new Error(`Remote embedding request failed: ${response.status}`);
      }

      const payload = await response.json();
      return embeddingFromPayload(payload);
    },
  };
}

function createCustomProvider(options) {
  if (typeof options.embed !== "function") {
    throw new Error("custom embeddingProvider requires embed(text).");
  }

  return {
    async embed(text) {
      const value = await options.embed(text);
      return toFloat32Array(value);
    },
  };
}

function embeddingFromPayload(payload) {
  if (Array.isArray(payload?.embedding)) {
    return new Float32Array(payload.embedding);
  }
  if (Array.isArray(payload?.data?.[0]?.embedding)) {
    return new Float32Array(payload.data[0].embedding);
  }
  throw new Error("Embedding response must include embedding or data[0].embedding.");
}

function toFloat32Array(value) {
  if (value instanceof Float32Array) {
    return value;
  }
  if (Array.isArray(value)) {
    return new Float32Array(value);
  }
  throw new TypeError("Embedding provider must return Float32Array or number[].");
}

function joinUrl(baseUrl, path) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}
