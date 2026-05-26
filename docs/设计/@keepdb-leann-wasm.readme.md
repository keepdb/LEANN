# @keepdb/leann-wasm

面向浏览器、Node.js 和边缘运行时的 LEANN 只读向量搜索 WASM 包。

> 状态：预生成的包使用文档，并同步当前验证进展。M1/M2 已验证；M3 的 `non-compact/non-pruned` HNSW 子集已由 [GitHub Actions run `26465439258`](https://github.com/keepdb/LEANN/actions/runs/26465439258) 验证，compact/pruned 与 recompute adapter 仍未完成。

## 目标

`@keepdb/leann-wasm` 把 LEANN 的只读搜索内核封装为 JavaScript ES6 API + WebAssembly bundle，让应用可以在不启动 Python、不启动本地 daemon、不依赖服务器向量数据库的情况下，对已经构建好的 LEANN 索引执行本地向量检索。

首版只解决一个问题：

```text
给定一个已经存在的 LEANN 向量索引 + 一个预计算好的 query embedding，返回 top-k 搜索结果。
```

## 安装

```bash
pnpm add @keepdb/leann-wasm
```

## 快速开始

```js
import { loadLeannIndex } from "@keepdb/leann-wasm";

const indexBytes = await fetch("/indexes/demo.index").then((res) => res.arrayBuffer());
const idsText = await fetch("/indexes/demo.ids.txt").then((res) => res.text());

const index = await loadLeannIndex({
  indexBytes,
  idsText,
});

const results = index.search({
  query: new Float32Array([0.12, -0.03, 0.44, 0.91]),
  topK: 5,
});

console.log(results);
```

返回结构：

```js
[
  { id: "passage-17", score: 0.8321 },
  { id: "passage-04", score: 0.8018 },
  { id: "passage-29", score: 0.7764 },
];
```

## 与 BigModel Embedding-3 对齐

支持。

`@keepdb/leann-wasm` 与 BigModel 的关系是分层协作：

```text
BigModel Embedding-3
  -> 负责把文本向量化
  -> 输出 Float32Array 兼容的 embedding

@keepdb/leann-wasm
  -> 负责加载 LEANN 索引
  -> 负责用 query embedding 做本地 top-k 搜索
  -> 不保存 API Key
  -> 不直接承担模型推理
```

也就是说，向量化存储和查询都可以使用 BigModel `embedding-3`，但 WASM 包本身只消费向量，不在搜索内核里调用 BigModel API。

### 构建索引

索引构建仍由原生 LEANN 执行。构建时使用 BigModel OpenAI-compatible endpoint：

```bash
export OPENAI_API_KEY="$BIGMODEL_API_KEY"
export OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4"

leann build demo \
  --docs ./documents \
  --backend hnsw \
  --embedding-mode openai \
  --embedding-model embedding-3 \
  --no-recompute \
  --no-compact
```

如果需要固定 BigModel `embedding-3` 的输出维度，必须让构建阶段和查询阶段使用相同维度。首版建议使用一个项目级约定，例如：

```json
{
  "embeddingProvider": "bigmodel",
  "embeddingModel": "embedding-3",
  "embeddingDimensions": 1024,
  "embeddingBaseUrl": "https://open.bigmodel.cn/api/paas/v4"
}
```

维度不一致会导致 query embedding 无法搜索对应索引。

### 查询向量化

查询时，应用侧先调用 BigModel `/embeddings` 生成 query embedding，再把 `Float32Array` 传给 WASM 搜索器。

服务端或边缘运行时示例：

```js
async function embedWithBigModel(text, {
  apiKey,
  dimensions = 1024,
  baseUrl = "https://open.bigmodel.cn/api/paas/v4",
} = {}) {
  const response = await fetch(`${baseUrl}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "embedding-3",
      input: text,
      dimensions,
    }),
  });

  if (!response.ok) {
    throw new Error(`BigModel embedding failed: ${response.status}`);
  }

  const payload = await response.json();
  return new Float32Array(payload.data[0].embedding);
}

const query = await embedWithBigModel("要查询的问题", {
  apiKey: process.env.BIGMODEL_API_KEY,
  dimensions: 1024,
});

const results = index.search({ query, topK: 10 });
```

本地验证示例页可以由验证者临时输入 BigModel API Key，并在浏览器内直接调用 BigModel；Key 仅保存在当前页面内存中。正式应用不建议要求用户在浏览器提供长期 API Key，推荐通过自己的后端、边缘函数或受控代理生成 query embedding。

## 包含什么

```text
@keepdb/leann-wasm
  dist/
    leann-wasm.mjs
    leann-wasm.wasm
  package.json
  README.md
```

导出 API：

```js
import {
  loadLeannIndex,
  loadLeannWasm,
  version,
} from "@keepdb/leann-wasm";
```

## 不包含什么

首版不包含以下能力：

- 不计算文本 embedding。
- 不运行 `torch`、`sentence-transformers`、`transformers`、MLX 或 Ollama。
- WASM 搜索内核不调用 OpenAI、Hugging Face 或其他远程模型服务；ES6 wrapper 可通过显式传入的 `embeddingProvider` 调用 BigModel 等兼容服务。
- 不构建索引。
- 不执行文本切分。
- 不启动 Python。
- 不启动 daemon。
- 不使用 ZMQ、TCP socket 或 subprocess。
- 不支持 DiskANN。
- 不承诺完整兼容 `leann-core` Python API。

## 核心概念

### Query Embedding

`query embedding` 是已经由外部模型算好的 `Float32Array`。WASM 包只负责搜索，不负责把自然语言 query 转成向量。

推荐调用链：

```text
用户输入文本
  -> 应用侧 embedding provider
  -> Float32Array query embedding
  -> @keepdb/leann-wasm search
  -> passage id + score
  -> 应用侧读取 passage 文本
```

### Index Bytes

`indexBytes` 是 LEANN 后端生成的只读索引二进制内容。首版优先支持非 pruned HNSW 索引，避免依赖 selective recomputation。

### ID Map

`idsText` 是索引内部整数 label 到 passage id 的映射文本，格式与 LEANN HNSW 后端生成的 `.ids.txt` 保持一致：每行一个 passage id，行号对应内部 label。

## API

### 简单消费模式

如果传入 `embeddingProvider`，调用方可以直接用自然语言查询，不需要自己手动调用 embedding API。

```js
import { loadLeannIndex } from "@keepdb/leann-wasm";

const index = await loadLeannIndex({
  indexBytes,
  idsText,
  dimension: 1024,
  metric: "cosine",
  embeddingProvider: {
    type: "openai-compatible",
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    apiKey: process.env.BIGMODEL_API_KEY,
    model: "embedding-3",
    dimensions: 1024,
  },
});

const results = await index.searchText("这个项目如何封装 WASM？", {
  topK: 10,
});
```

这个模式的内部流程：

```text
index.searchText(text)
  -> embeddingProvider.embed(text)
  -> Float32Array query embedding
  -> index.search({ query })
  -> SearchResult[]
```

### 安全建议

`embeddingProvider.apiKey` 不应直接放进浏览器 bundle。推荐两种方式：

1. 服务端或边缘运行时使用 provider config，直接调用 `searchText()`。
2. 浏览器使用 `embeddingProvider.type = "remote"`，把文本发给自己的受控 API，由后端持有 BigModel API Key。

用于真实链路验收的本地示例页可额外提供临时 Key 输入框，以确认浏览器、BigModel、真实 LEANN 索引与 WASM 的组合行为；该方式不是生产鉴权方案。

浏览器远程代理示例：

```js
const index = await loadLeannIndex({
  indexBytes,
  idsText,
  dimension: 1024,
  embeddingProvider: {
    type: "remote",
    endpoint: "/api/embed",
  },
});

const results = await index.searchText("查询问题", { topK: 10 });
```

`/api/embed` 返回结构：

```json
{
  "embedding": [0.12, -0.03, 0.44]
}
```

### `loadLeannIndex(options)`

加载一个 LEANN 索引并返回搜索器。

```ts
type EmbeddingProviderOptions =
  | {
      type: "openai-compatible";
      baseUrl: string;
      apiKey: string;
      model: string;
      dimensions?: number;
      headers?: Record<string, string>;
    }
  | {
      type: "remote";
      endpoint: string;
      headers?: Record<string, string>;
    }
  | {
      type: "custom";
      embed: (text: string) => Promise<Float32Array>;
    };

type LoadLeannIndexOptions = {
  indexBytes: ArrayBuffer | Uint8Array;
  idsText?: string;
  wasmUrl?: string | URL;
  dimension?: number;
  metric?: "mips" | "l2" | "cosine";
  embeddingProvider?: EmbeddingProviderOptions;
};
```

示例：

```js
const index = await loadLeannIndex({
  indexBytes,
  idsText,
  dimension: 384,
  metric: "mips",
});
```

### `index.search(options)`

执行 top-k 向量搜索。

```ts
type SearchOptions = {
  query: Float32Array;
  topK?: number;
  complexity?: number;
};

type SearchResult = {
  id: string;
  label: number;
  score: number;
};
```

示例：

```js
const results = index.search({
  query,
  topK: 10,
  complexity: 64,
});
```

### `index.searchText(text, options)`

使用配置好的 embedding provider 先向量化文本，再执行搜索。

```ts
type SearchTextOptions = {
  topK?: number;
  complexity?: number;
};
```

示例：

```js
const results = await index.searchText("LEANN 的 WASM 封装范围是什么？", {
  topK: 10,
});
```

如果初始化时没有传入 `embeddingProvider`，调用 `searchText()` 会抛出错误。此时应改用 `search({ query })`。

### `index.dispose()`

释放 WASM 侧资源。

```js
index.dispose();
```

## 浏览器使用

```js
import { loadLeannIndex } from "@keepdb/leann-wasm";

const [indexBytes, idsText] = await Promise.all([
  fetch("/leann/demo.index").then((res) => res.arrayBuffer()),
  fetch("/leann/demo.ids.txt").then((res) => res.text()),
]);

const index = await loadLeannIndex({ indexBytes, idsText });
const results = index.search({ query, topK: 8 });
```

对于较大索引，建议在 Web Worker 中加载和搜索，避免阻塞浏览器主线程。

## Node.js 使用

```js
import { readFile } from "node:fs/promises";
import { loadLeannIndex } from "@keepdb/leann-wasm";

const indexBytes = await readFile("./demo.index");
const idsText = await readFile("./demo.ids.txt", "utf8");

const index = await loadLeannIndex({
  indexBytes,
  idsText,
});

const results = index.search({
  query: new Float32Array([0.12, -0.03, 0.44, 0.91]),
  topK: 5,
});
```

## 边缘运行时

目标兼容：

- Cloudflare Workers
- Vercel Edge Runtime
- Deno Deploy
- Bun

边缘运行时通常不支持本地文件系统，建议通过 `fetch()`、KV、R2、对象存储或 bundle asset 加载 `indexBytes`。

## 索引生成

索引构建仍由原生 LEANN 完成。

推荐第一阶段生成非 pruned HNSW 索引：

```bash
leann build demo --docs ./documents --backend hnsw --no-recompute --no-compact
```

然后发布以下文件给 JS 应用：

```text
demo.index
demo.ids.txt
demo.meta.json
demo.passages.jsonl
```

WASM 包只需要 `demo.index` 和 `demo.ids.txt` 完成搜索。应用如果需要展示文本，需要自行根据 passage id 读取 `demo.passages.jsonl` 或自己的文档存储。

## 构建

本包的 WASM 构建不要求用户本机安装 Emscripten、CMake 或 Ninja。官方构建通过 GitHub Actions 执行。

当前 workflow：

```text
.github/workflows/keepdb-wasm.yml
```

构建步骤：

1. checkout 源码和 submodules。
2. 安装 Emscripten SDK。
3. 编译带有 HNSW search 导出的 `dist/leann-wasm.wasm`。
4. 构建 LEANN HNSW native extension 并生成真实 `non-compact/non-pruned` fixture。
5. 使用 Node.js 运行 WASM、parser 和真实 HNSW fixture smoke tests。
6. 上传真实 `.wasm`、npm tarball 和 HNSW fixture artifacts。

run `26465439258` 已完成上述流程，真实 HNSW fixture 查询返回 `top1=doc-wasm`。该 run 的 BigModel `embedding-3` E2E 因仓库没有配置 `BIGMODEL_API_KEY` secret 而跳过；本地 `.env` 链路已验证通过。

## 限制

- 首版只支持只读搜索。
- 首版要求 query embedding 维度与索引维度一致。
- 首版优先支持非 pruned HNSW 索引。
- 浏览器内存有限，不建议直接加载超大索引。
- 搜索结果的文本召回由应用层负责，WASM 包只返回 id 和 score。
- 如果索引格式发生变化，必须通过 meta version 或 package version 明确兼容范围。

## 设计原则

- JS API 保持 ES6 原生模块风格。
- WASM 只承载搜索内核，不承载模型推理。
- 输入输出使用 `ArrayBuffer`、`Uint8Array`、`Float32Array` 等浏览器原生结构。
- 默认路径适合浏览器，Node.js 只是额外运行环境。
- 构建环境只放在 CI，不要求业务用户本机具备原生编译工具链。

## 路线图

### v0.1 / M3 已验证子集

- [x] HNSW 非 compact、非 pruned index 只读搜索。
- [x] ES6 wrapper。
- [x] Node.js smoke test。
- [x] GitHub Actions `.wasm`、npm tarball 和真实 HNSW fixture artifact。

### v0.2

- Web Worker 示例。
- 浏览器 smoke test。
- 更完整的错误码和维度校验。
- 小型 fixture index。

### v0.3 / M4

- [ ] 评估 compact/CSR HNSW 兼容。
- [ ] 评估 streaming 或分片加载。
- [ ] 设计 JS callback/provider adapter 形式的 selective recomputation，替代浏览器不能直接复用的 ZMQ server。

## License

MIT
