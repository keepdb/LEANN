# @keepdb/leann-wasm PoC

这是 `@keepdb/leann-wasm` 的 PoC 目录，用于先验证 JS@ES6 对外 API、embedding provider 配置和消费方式。

当前 PoC 通过 WASM 执行搜索，并可读取 LEANN 真实生成的 IVF 最小索引产物。当前实现仍限定 `nlist=1`，尚未达到 LEANN HNSW/compact 后端支持；HNSW search core 属于后续 M3。

## 目标

- 验证 `loadLeannIndex()` 的配置形态。
- 验证 `search({ query })` 的向量搜索调用。
- 验证 `searchText(text)` 通过 `embeddingProvider` 自动向量化后搜索。
- 支持 BigModel `embedding-3` 这类 OpenAI-compatible embedding provider。
- 支持浏览器安全场景下的 remote embedding endpoint。

## 快速验证

从仓库根目录运行：

```bash
node packages/keepdb.wasm/tests/smoke.mjs
```

期望输出：

```text
keepdb.wasm smoke passed
```

## BigModel + LEANN 真实索引端到端验证

准备本地配置：

```bash
cp packages/keepdb.wasm/.env.example packages/keepdb.wasm/.env
```

在 `.env` 中填入：

```text
BIGMODEL_API_KEY=...
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4
BIGMODEL_EMBEDDING_MODEL=embedding-3
BIGMODEL_EMBEDDING_DIMENSIONS=1024
```

运行：

```bash
pnpm --dir packages/keepdb.wasm build:wasm
pnpm --dir packages/keepdb.wasm test:e2e:bigmodel
```

这条测试会执行完整链路：

1. 调用 BigModel `/embeddings` 的 `embedding-3` 生成公开小样本文档向量。
2. 用 LEANN `build_index_from_arrays()` 生成真实索引产物：
   - `.meta.json`
   - `.index`
   - `.ids.txt`
   - `.passages.jsonl`
3. `@keepdb/leann-wasm` 读取真实 LEANN IVF `.index`。
4. 查询时再次调用 BigModel `embedding-3` 生成 query 向量。
5. WASM 返回预期 top-1/top-k，并补齐 passage text 与 metadata。

当前 M1 读取器的范围是 LEANN IVF 最小索引：`nlist=1`、`distance_metric=cosine`。HNSW/Faiss 完整后端支持属于后续 M3。

## 浏览器示例页

在完成上述 E2E 命令、生成真实测试索引后运行：

```bash
pnpm --dir packages/keepdb.wasm demo
```

示例预览进程只托管 ES module、真实 `.wasm` 和 `.tmp/bigmodel-e2e/` 下的 LEANN 索引文件；它会通过 `npx @keepdb/cli port` 获取本地端口。页面允许输入 BigModel API Key，并由浏览器直接请求 `https://open.bigmodel.cn/api/paas/v4/embeddings` 的 `embedding-3`。Key 只保存在当前页面内存，不读取 `.env`、不发给本地预览进程。

该直连方式适合本地 PoC 验证。正式应用不应要求终端用户在浏览器提供长期凭据；应改用 `remote` provider 配合受控后端或边缘函数。

## GitHub Actions 定向打包

定向 workflow：

```text
.github/workflows/keepdb-wasm.yml
```

触发条件：

- 手动触发：`workflow_dispatch`
- PR 或 push 命中以下路径：
  - `packages/keepdb.wasm/**`
  - `docs/wasm.todo.md`
  - `docs/设计/**`
  - `.github/workflows/keepdb-wasm.yml`

流程：

1. checkout 源码。
2. 使用 Node.js 20。
3. 启用 `pnpm@10.12.1`。
4. 运行 `pnpm test`。
5. 安装 Emscripten SDK。
6. 执行 `pnpm build:wasm` 生成 `dist/leann-wasm.wasm`。
7. 运行 `pnpm test:wasm`。
8. 构建 LEANN HNSW 原生扩展并运行 `pnpm test:hnsw-capability`，作为 M3 的 HNSW fixture 闸门。
9. 执行 `pnpm pack`。
10. 上传 `keepdb-leann-wasm-package` artifact。

当前产物是 PoC npm tarball，包含 ES6 wrapper、LEANN IVF `nlist=1` 最小 `.index` 读取器，以及 WASM search core。CI 的 Emscripten 构建会导出 `leann_wasm_flat_search` 和 `leann_wasm_hnsw_search`；本地 `wat2wasm` fallback 仍只覆盖 flat search，用于不安装打包环境时的基础验证。

M3 方案见：

```text
docs/设计/leann-wasm-m3-hnsw.md
```

本地不安装 HNSW/Faiss 打包环境时，可以运行能力探针：

```bash
pnpm --dir packages/keepdb.wasm test:hnsw-parser
pnpm --dir packages/keepdb.wasm test:hnsw-wasm
pnpm --dir packages/keepdb.wasm test:hnsw-capability
```

`test:hnsw-parser` 使用合成二进制验证 `IHNf` non-compact parser。`test:hnsw-wasm` 在当前 `.wasm` 导出 `leann_wasm_hnsw_search` 时验证 HNSW WASM search；本地 fallback 产物没有该导出时会跳过。`test:hnsw-capability` 如果输出 `missing-hnsw-native-extension`，表示本机缺少 `leann_backend_hnsw.faiss` 原生扩展；这不是 M3 通过，只是明确阻塞。GitHub Actions 中会设置 `LEANN_WASM_REQUIRE_HNSW=1`，必须真实生成 HNSW fixture，并通过 WASM `leann_wasm_hnsw_search()` 返回 `top1=doc-wasm` 才算通过。

Actions 成功后会上传三个 artifact：

- `keepdb-leann-wasm-package`：npm tarball。
- `keepdb-leann-wasm-binary`：原始 `dist/leann-wasm.wasm`。
- `keepdb-leann-hnsw-fixture`：真实 non-compact/non-pruned HNSW fixture。

## API 示例

### 直接传入 query embedding

```js
import { loadLeannIndex } from "./src/index.js";

const index = await loadLeannIndex({
  vectors: [
    { id: "doc-1", vector: [1, 0, 0] },
    { id: "doc-2", vector: [0, 1, 0] },
  ],
  dimension: 3,
  metric: "cosine",
});

const results = index.search({
  query: new Float32Array([1, 0, 0]),
  topK: 1,
});
```

### 配置 BigModel Embedding-3

服务端或边缘运行时可以直接传入 API Key：

```js
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

const results = await index.searchText("查询问题", { topK: 10 });
```

本地示例页可以让验证者临时输入 API Key，浏览器会直接调用 BigModel：

```js
const index = await loadLeannIndex({
  indexBytes,
  idsText,
  dimension: 1024,
  embeddingProvider: {
    type: "openai-compatible",
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    apiKey: inputApiKey,
    model: "embedding-3",
    dimensions: 1024,
  },
});
```

正式浏览器应用建议使用 remote provider，避免把长期 API Key 作为产品交互暴露给前端：

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
```

## 当前实现边界

- `vectors` 是 PoC mock 数据，用来验证 API，不代表真实索引格式。
- `indexBytes` 和 `idsText` 参数已预留，当前 mock core 不解析真实 LEANN HNSW index。
- `embeddingProvider.type = "openai-compatible"` 会调用 `${baseUrl}/embeddings`。
- `embeddingProvider.type = "remote"` 会调用自有 endpoint，适合浏览器。
- `embeddingProvider.type = "custom"` 允许传入 `embed(text)` 函数做测试。

## 后续替换点

LEANN HNSW 内核接入时，需要扩展 `src/leann-index-parser.js` 和 `src/wasm-search-core.js` 的底层加载/search adapter，保持 `src/index.js` 对外 API 不变。

当前 WASM 源码：

```text
wasm/leann_wasm.c
```

它导出：

- `leann_wasm_version()`
- `leann_wasm_score_dot()`
- `leann_wasm_flat_search()`

这些导出用于验证 GitHub Actions 中 Emscripten 构建、Node.js WASM 加载、真实向量检索和 npm tarball artifact 链路。
