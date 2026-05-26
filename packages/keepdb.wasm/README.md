# @keepdb/leann-wasm

这是 `@keepdb/leann-wasm` 的专用目录，用于封装 LEANN 索引的浏览器/Node.js WASM 搜索能力，并提供 JS@ES6 API、embedding provider 配置和端到端验证。

真实状态：

- M0：flat vector search WASM 仅作为 toolchain 验证，不能称为 LEANN WASM。
- M1：已能读取 LEANN 真实生成的 IVF 最小索引产物，限定 `backend_name=ivf`、`nlist=1`、`distance_metric=cosine`。
- M2：已接入 BigModel `embedding-3`，端到端验证 build/query 同模型同维度，`searchText()` 可用。
- M3：`non-compact/non-pruned` HNSW 子集已通过 [GitHub Actions run `26465439258`](https://github.com/keepdb/LEANN/actions/runs/26465439258) 真实验证：真实 `.wasm` 读取真实 HNSW 四文件 fixture，并返回 `top1=doc-wasm`。
- M4/M5：compact/pruned index、JS callback/provider adapter、正式 npm 发布仍未完成。

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

## BigModel + JS-only IVF 兼容 fixture 验证

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
2. 用 JS@ES6 写出当前 M1 reader 支持的 LEANN IVF `nlist=1` 兼容 fixture：
   - `.meta.json`
   - `.index`
   - `.ids.txt`
   - `.passages.jsonl`
3. `@keepdb/leann-wasm` 读取该 IVF 兼容 `.index`。
4. 查询时再次调用 BigModel `embedding-3` 生成 query 向量。
5. WASM 返回预期 top-1/top-k，并补齐 passage text 与 metadata。

这一命令证明 JS package 不依赖 Python 即可串通 BigModel provider、sidecar、IVF 最小格式 reader 与 WASM 查询；它不证明本轮索引由原生 LEANN builder 生成。真实 LEANN 索引消费证据来自 M3 Actions：已验证由 LEANN HNSW/Faiss 后端生成的 `non-compact/non-pruned` 索引读取与 HNSW graph traversal 查询。compact/pruned 索引仍不在支持范围内。

## 浏览器示例页

在完成上述 E2E 命令、生成 JS-only IVF 兼容 fixture 后运行：

```bash
pnpm --dir packages/keepdb.wasm demo
```

示例预览进程只托管 ES module、真实 `.wasm` 和 `.tmp/bigmodel-e2e/` 下的 IVF 兼容 fixture；它会通过 `npx @keepdb/cli port` 获取本地端口。页面允许输入 BigModel API Key，并由浏览器直接请求 `https://open.bigmodel.cn/api/paas/v4/embeddings` 的 `embedding-3`。Key 只保存在当前页面内存，不读取 `.env`、不发给本地预览进程。

该直连方式适合本地验证。正式应用不应要求终端用户在浏览器提供长期凭据；应改用 `remote` provider 配合受控后端或边缘函数。

## GitHub Actions 定向打包

定向 workflow：

```text
.github/workflows/keepdb-wasm.yml
```

触发条件：

- 手动触发：`workflow_dispatch`
- PR 或 push 命中以下路径：
  - `packages/keepdb.wasm/**`
  - `.github/scripts/keepdb-wasm/**`
  - `docs/wasm.todo.md`
  - `docs/设计/**`
  - `.github/workflows/keepdb-wasm.yml`

流程：

1. checkout 源码。
2. 使用 Node.js 20。
3. 启用 `pnpm@10.12.1`。
4. 运行 `pnpm test`，其中 `tests/js-only-guard.mjs` 会确认包内无 Python 文件、无 shell build 脚本、无 Python/`uv` 调用。
5. 安装 Emscripten SDK。
6. 执行 `pnpm build:wasm` 生成 `dist/leann-wasm.wasm`。
7. 运行 `pnpm test:wasm`。
8. 在仓库级 CI helper 中构建 LEANN HNSW 原生扩展并生成真实 fixture，再运行 `pnpm test:hnsw-capability`，作为 M3 的 HNSW fixture 消费闸门。
9. 执行 `pnpm pack`。
10. 上传 `keepdb-leann-wasm-package`、`keepdb-leann-wasm-binary` 和 `keepdb-leann-hnsw-fixture` artifacts。

Actions run [`26465439258`](https://github.com/keepdb/LEANN/actions/runs/26465439258) 已成功产出 M3 validation npm tarball 和真实 `.wasm`。包包含 ES6 wrapper、LEANN IVF `nlist=1` 最小 `.index` 读取器、non-compact/non-pruned HNSW parser，以及 WASM search core。CI 的 Emscripten 构建导出 `leann_wasm_flat_search` 和 `leann_wasm_hnsw_search`；本地 `wat2wasm` fallback 仍只覆盖 flat search，用于不安装打包环境时的基础验证。

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

`test:hnsw-parser` 使用合成二进制验证 `IHNf` non-compact parser。`test:hnsw-wasm` 在当前 `.wasm` 导出 `leann_wasm_hnsw_search` 时验证 HNSW WASM search；本地 fallback 产物没有该导出时会跳过。`test:hnsw-capability` 如果输出 `missing-hnsw-fixture`，表示本机没有准备真实 HNSW fixture；这不是 M3 通过，只是明确阻塞。GitHub Actions 中会设置 `LEANN_WASM_REQUIRE_HNSW=1`，先通过仓库级 helper 真实生成 HNSW fixture，再要求 WASM `leann_wasm_hnsw_search()` 返回 `top1=doc-wasm` 才算通过。

上述 CI 闸门已在 run `26465439258` 通过。该 run 的 BigModel E2E 步骤因为仓库没有配置 `BIGMODEL_API_KEY` secret 而跳过；本地通过 `.env` 执行 `pnpm --dir packages/keepdb.wasm test:e2e:bigmodel` 已验证 BigModel `embedding-3` 的 build/query 链路。

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

- `vectors` 是 mock/测试数据入口，用来验证 API，不代表真实索引格式。
- `indexBytes`、`metaJson`、`idsText`、`passagesJsonl` 是真实 LEANN sidecar/index 入口；当前支持 IVF `nlist=1` 和 non-compact/non-pruned HNSW reader。
- `embeddingProvider.type = "openai-compatible"` 会调用 `${baseUrl}/embeddings`。
- `embeddingProvider.type = "remote"` 会调用自有 endpoint，适合浏览器。
- `embeddingProvider.type = "custom"` 允许传入 `embed(text)` 函数做测试。

## 后续替换点

LEANN `non-compact/non-pruned` HNSW 内核已通过 `src/leann-index-parser.js` 和 `src/wasm-search-core.js` 接入，并保持 `src/index.js` 对外 API 不变。后续替换点是 compact/pruned index reader 以及以 JS callback/provider adapter 替代 ZMQ recompute 的 M4 路径。

当前 WASM 源码：

```text
wasm/leann_wasm.c
```

包内构建脚本是 JS@ES6：

```text
scripts/build-wasm.mjs
```

它导出：

- `leann_wasm_version()`
- `leann_wasm_score_dot()`
- `leann_wasm_flat_search()`
- `leann_wasm_hnsw_search()`

这些导出用于验证 GitHub Actions 中 Emscripten 构建、Node.js WASM 加载、真实 IVF/HNSW 索引检索和 npm tarball artifact 链路。
