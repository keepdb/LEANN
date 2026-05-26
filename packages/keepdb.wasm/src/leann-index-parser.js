export function parseLeannIndex({ meta, indexBytes, ids = [] } = {}) {
  if (!meta) {
    throw new Error("metaJson is required when loading a real LEANN index.");
  }
  if (!indexBytes) {
    throw new Error("indexBytes is required when loading a real LEANN index.");
  }

  if (meta.backend_name === "ivf") {
    return {
      kind: "flat",
      vectors: parseIvfNlistOne({ meta, indexBytes, ids }),
    };
  }

  if (meta.backend_name === "hnsw") {
    return parseHnswFlatIndex({ meta, indexBytes, ids });
  }

  throw new Error(`Unsupported LEANN backend for WASM reader: ${meta.backend_name}`);
}

export function parseLeannIndexVectors({ meta, indexBytes, ids = [] } = {}) {
  return parseLeannIndex({ meta, indexBytes, ids }).vectors;
}

function parseIvfNlistOne({ meta, indexBytes, ids }) {
  const bytes = toUint8Array(indexBytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const header = readAscii(bytes, 0, 4);
  if (header !== "IwFl") {
    throw new Error(`Expected Faiss IndexIVFFlat FourCC 'IwFl', got '${header}'.`);
  }

  const dimension = view.getInt32(4, true);
  if (dimension !== meta.dimensions) {
    throw new Error(`Index dimension mismatch: meta=${meta.dimensions}, index=${dimension}.`);
  }

  const ilarOffset = indexOfAscii(bytes, "ilar");
  const fullOffset = indexOfAscii(bytes, "full", ilarOffset);
  if (ilarOffset < 0 || fullOffset < 0) {
    throw new Error("Unsupported IVF index: missing ArrayInvertedLists/full sections.");
  }

  const nlist = readU64AsNumber(view, ilarOffset + 4);
  const codeSize = readU64AsNumber(view, ilarOffset + 12);
  if (nlist !== 1) {
    throw new Error(`M1 IVF reader supports nlist=1 only, got nlist=${nlist}.`);
  }
  if (codeSize !== dimension * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(`Unexpected IVF code size: ${codeSize}; expected ${dimension * 4}.`);
  }

  const fullNlist = readU64AsNumber(view, fullOffset + 4);
  if (fullNlist !== nlist) {
    throw new Error(`IVF nlist mismatch: ilar=${nlist}, full=${fullNlist}.`);
  }

  const listSize = readU64AsNumber(view, fullOffset + 12);
  const codesOffset = fullOffset + 20;
  const codesByteLength = listSize * codeSize;
  const idsOffset = codesOffset + codesByteLength;
  const vectors = [];

  for (let row = 0; row < listSize; row += 1) {
    const codeOffset = codesOffset + row * codeSize;
    const numericId = readU64AsNumber(view, idsOffset + row * 8);
    const id = ids[numericId] ?? String(numericId);
    const vector = new Float32Array(dimension);
    for (let dim = 0; dim < dimension; dim += 1) {
      vector[dim] = view.getFloat32(codeOffset + dim * 4, true);
    }
    vectors.push({
      id,
      label: numericId,
      vector,
    });
  }

  return vectors;
}

function parseHnswFlatIndex({ meta, indexBytes, ids }) {
  const bytes = toUint8Array(indexBytes);
  const reader = new BinaryReader(bytes);
  const header = reader.readAscii(4);
  if (header !== "IHNf") {
    throw new Error(`Expected Faiss IndexHNSWFlat FourCC 'IHNf', got '${header}'.`);
  }

  const indexHeader = reader.readIndexHeader();
  if (indexHeader.dimension !== meta.dimensions) {
    throw new Error(
      `HNSW dimension mismatch: meta=${meta.dimensions}, index=${indexHeader.dimension}.`,
    );
  }

  const backend = meta.backend_kwargs ?? {};
  const isCompact = Boolean(meta.is_compact ?? backend.is_compact);
  const isPruned = Boolean(meta.is_pruned ?? backend.is_recompute);
  if (isCompact || isPruned) {
    throw new Error("M3 HNSW reader supports non-compact, non-pruned HNSW only.");
  }

  const assignProbas = reader.readFloat64Vector();
  const cumNeighbors = reader.readInt32Vector();
  const levels = reader.readInt32Vector();
  const storageIsCompact = reader.readBool();
  if (storageIsCompact) {
    throw new Error("M3 HNSW reader does not support compact CSR storage yet.");
  }

  const offsets = reader.readU64Vector();
  const neighbors = reader.readInt32Vector();
  const entryPoint = reader.readInt32();
  const maxLevel = reader.readInt32();
  const efConstruction = reader.readInt32();
  const efSearch = reader.readInt32();
  reader.readInt32(); // Deprecated upper_beam placeholder.

  const storageFourcc = reader.readAscii(4);
  if (storageFourcc !== "IxFI" && storageFourcc !== "IxF2" && storageFourcc !== "IxFl") {
    throw new Error(`Unsupported HNSW storage FourCC '${storageFourcc}'.`);
  }
  const storageHeader = reader.readIndexHeader();
  const codeCount = reader.readU64();
  const expectedCodes = storageHeader.ntotal * storageHeader.dimension;
  if (codeCount !== expectedCodes) {
    throw new Error(`Unexpected HNSW storage code count: ${codeCount}; expected ${expectedCodes}.`);
  }

  const vectorCount = storageHeader.ntotal;
  const dimension = storageHeader.dimension;
  const vectors = [];
  for (let row = 0; row < vectorCount; row += 1) {
    const vector = new Float32Array(dimension);
    for (let dim = 0; dim < dimension; dim += 1) {
      vector[dim] = reader.readFloat32();
    }
    vectors.push({
      id: ids[row] ?? String(row),
      label: row,
      vector,
    });
  }

  return {
    kind: "hnsw",
    vectors,
    hnsw: {
      assignProbas,
      cumNeighbors,
      levels,
      offsets,
      neighbors,
      entryPoint,
      maxLevel,
      efConstruction,
      efSearch,
      metric: metricFromFaiss(storageHeader.metricType),
    },
  };
}

function toUint8Array(value) {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new TypeError("indexBytes must be Uint8Array, ArrayBuffer, or ArrayBuffer view.");
}

function readAscii(bytes, offset, length) {
  return String.fromCharCode(...bytes.subarray(offset, offset + length));
}

function indexOfAscii(bytes, text, start = 0) {
  const pattern = new TextEncoder().encode(text);
  for (let i = Math.max(0, start); i <= bytes.length - pattern.length; i += 1) {
    let matched = true;
    for (let j = 0; j < pattern.length; j += 1) {
      if (bytes[i + j] !== pattern[j]) {
        matched = false;
        break;
      }
    }
    if (matched) return i;
  }
  return -1;
}

function readU64AsNumber(view, offset) {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`u64 value exceeds Number.MAX_SAFE_INTEGER at offset ${offset}.`);
  }
  return Number(value);
}

function metricFromFaiss(metricType) {
  if (metricType === 0) return "mips";
  if (metricType === 1) return "l2";
  return undefined;
}

class BinaryReader {
  constructor(bytes) {
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.offset = 0;
  }

  readAscii(length) {
    this.ensure(length);
    const value = readAscii(this.bytes, this.offset, length);
    this.offset += length;
    return value;
  }

  readBool() {
    this.ensure(1);
    const value = this.view.getUint8(this.offset) !== 0;
    this.offset += 1;
    return value;
  }

  readInt32() {
    this.ensure(4);
    const value = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readInt64() {
    this.ensure(8);
    const value = this.view.getBigInt64(this.offset, true);
    this.offset += 8;
    if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new Error(`i64 value exceeds safe integer range at offset ${this.offset - 8}.`);
    }
    return Number(value);
  }

  readU64() {
    const value = readU64AsNumber(this.view, this.offset);
    this.offset += 8;
    return value;
  }

  readFloat32() {
    this.ensure(4);
    const value = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readFloat64() {
    this.ensure(8);
    const value = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return value;
  }

  readIndexHeader() {
    const dimension = this.readInt32();
    const ntotal = this.readInt64();
    this.readInt64();
    this.readInt64();
    const isTrained = this.readBool();
    const metricType = this.readInt32();
    const metricArg = metricType > 1 ? this.readFloat32() : undefined;
    return {
      dimension,
      ntotal,
      isTrained,
      metricType,
      metricArg,
    };
  }

  readFloat64Vector() {
    const count = this.readU64();
    const values = new Float64Array(count);
    for (let i = 0; i < count; i += 1) {
      values[i] = this.readFloat64();
    }
    return values;
  }

  readInt32Vector() {
    const count = this.readU64();
    const values = new Int32Array(count);
    for (let i = 0; i < count; i += 1) {
      values[i] = this.readInt32();
    }
    return values;
  }

  readU64Vector() {
    const count = this.readU64();
    const values = new BigUint64Array(count);
    for (let i = 0; i < count; i += 1) {
      const value = BigInt(this.readU64());
      values[i] = value;
    }
    return values;
  }

  ensure(length) {
    if (this.offset + length > this.bytes.byteLength) {
      throw new Error(`Unexpected end of index at offset ${this.offset}.`);
    }
  }
}
