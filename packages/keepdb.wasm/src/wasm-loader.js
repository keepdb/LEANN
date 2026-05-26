export async function loadWasmModule(options = {}) {
  const {
    wasmBytes,
    wasmUrl,
    imports = {},
    fetchImpl = globalThis.fetch,
  } = options;

  if (wasmBytes) {
    const bytes = wasmBytes instanceof Uint8Array ? wasmBytes : new Uint8Array(wasmBytes);
    return WebAssembly.instantiate(bytes, imports);
  }

  if (wasmUrl) {
    if (!fetchImpl) {
      throw new Error("fetch is unavailable. Pass wasmBytes or fetchImpl.");
    }
    const response = await fetchImpl(wasmUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch WASM: ${response.status}`);
    }
    const bytes = await response.arrayBuffer();
    return WebAssembly.instantiate(bytes, imports);
  }

  return null;
}
