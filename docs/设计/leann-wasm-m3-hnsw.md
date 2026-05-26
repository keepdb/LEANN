# LEANN WASM M3：HNSW/Faiss 后端方案

日期：2026-05-26

## 当前结论

M3 的 `non-compact/non-pruned` HNSW 子集已完成真实验证。GitHub Actions run [`26465439258`](https://github.com/keepdb/LEANN/actions/runs/26465439258) 在 2026-05-27 完成以下链路：

1. 用 Emscripten 构建真实 `dist/leann-wasm.wasm`。
2. 构建 `leann_backend_hnsw.faiss` 原生扩展，并生成 LEANN HNSW 四文件 fixture。
3. 通过 `loadLeannIndex()` 读取真实 `.meta.json`、`.index`、`.ids.txt`、`.passages.jsonl`。
4. 在 WASM 中运行 `leann_wasm_hnsw_search()` 图遍历查询，得到 `top1=doc-wasm`。

本机探针结果是：

```text
missing-hnsw-native-extension
```

原因是本机没有 `leann_backend_hnsw.faiss` 原生扩展。按照用户约束，本机不安装 HNSW/Faiss 打包环境，因此 HNSW fixture 生成必须放到 GitHub Actions。

该结论仅覆盖非 compact、非 pruned 的全量向量 HNSW 路径。compact/pruned index 和依赖 selective recomputation 的 ZMQ 替代方案仍属于 M4，不能声称完整 LEANN WASM 已封装完成。

## M3 验收目标

M3 不能继续把 IVF `.index` 中的 vectors 抽出来交给 flat search。M3 的验收目标必须是：

1. 生成真实 LEANN HNSW index：
   - `backend_name=hnsw`
   - `is_recompute=false`
   - `is_compact=false`
   - `distance_metric=cosine`
2. WASM 读取 HNSW graph 和 storage：
   - `IHNf` index header
   - HNSW graph metadata
   - non-compact `offsets` / `neighbors`
   - embedded `IndexFlat` storage
   - `.ids.txt`
3. WASM 中执行 HNSW graph traversal search。
4. 对固定 query embedding 断言已知 topK；增加与 Python `HNSWSearcher(..., recompute_embeddings=false)` 的通用 topK 一致性矩阵作为后续强化验证。

## 为什么先选 non-compact / non-pruned HNSW

LEANN HNSW 默认的 compact/pruned 模式依赖 selective recomputation。当前 Python 搜索路径通过 ZMQ embedding server 获取需要重算的节点向量，浏览器 WASM 不能直接复用这个本地 daemon 模型。

因此 M3 的最小真实目标是非 compact、非 pruned HNSW：

```json
{
  "backend_name": "hnsw",
  "backend_kwargs": {
    "is_recompute": false,
    "is_compact": false
  },
  "is_compact": false,
  "is_pruned": false
}
```

这个模式仍然是 LEANN HNSW/Faiss 后端生成的真实 `.index`，但保留全量向量，不需要 ZMQ。

## 已新增闸门

本地命令：

```bash
pnpm --dir packages/keepdb.wasm test:hnsw-capability
```

本地没有原生扩展时输出 `missing-hnsw-native-extension` 并退出 0，避免误伤不安装打包环境的开发机。

CI 命令：

```bash
LEANN_WASM_REQUIRE_HNSW=1 pnpm --dir packages/keepdb.wasm test:hnsw-capability
```

CI 中该命令必须真正生成 HNSW fixture，否则失败。

Actions run `26465439258` 中，`test:hnsw-capability` 已执行：

1. 读取真实 HNSW `.meta.json`、`.index`、`.ids.txt`、`.passages.jsonl`。
2. 通过 `loadLeannIndex()` 加载 HNSW graph 和 storage。
3. 调用 WASM `leann_wasm_hnsw_search()`。
4. 断言 `query=[1,0,0,0]` 的 `top1=doc-wasm`。

## GitHub Actions 设计

`.github/workflows/keepdb-wasm.yml` 已加入：

1. `actions/checkout` 使用 `submodules: recursive`，拉取 LEANN HNSW 使用的 Faiss fork。
2. 安装 HNSW 原生构建依赖：
   - `cmake`
   - `ninja-build`
   - `pkg-config`
   - `swig`
   - `libomp-dev`
   - `libzmq3-dev`
3. 用 `uv` 创建 `.venv-keepdb-hnsw`。
4. 在 CI 里 editable 安装：
   - `packages/leann-core`
   - `packages/leann-backend-hnsw`
5. 运行 `test:hnsw-capability`，生成真实 HNSW fixture。

Actions 成功后上传：

- `keepdb-leann-wasm-package`
- `keepdb-leann-wasm-binary`
- `keepdb-leann-hnsw-fixture`

实际 run `26465439258` 已上传上述三类 artifact：

- `keepdb-leann-wasm-binary`：真实 `dist/leann-wasm.wasm`。
- `keepdb-leann-wasm-package`：`keepdb-leann-wasm-0.0.0-m3.tgz`。
- `keepdb-leann-hnsw-fixture`：真实 HNSW `.meta.json`、`.index`、`.ids.txt`、`.passages.jsonl`。

该 run 中 BigModel E2E 因仓库没有配置 `BIGMODEL_API_KEY` secret 而跳过；本地使用 `.env` 的 `pnpm --dir packages/keepdb.wasm test:e2e:bigmodel` 已验证 `embedding-3` 链路。

## 已实现路径

M3 的代码实现分两段：

### M3-A：HNSW index parser

已在 `src/leann-index-parser.js` 中新增 `IHNf` non-compact parser：

- index header：`d`、`ntotal`、`is_trained`、`metric_type`
- HNSW metadata：`assign_probas`、`cum_nneighbor_per_level`、`levels`
- non-compact storage flag：`false`
- graph arrays：`offsets`、`neighbors`
- HNSW params：`entry_point`、`max_level`、`efConstruction`、`efSearch`
- embedded storage：`IxFI` 或 `IxF2` 的 `IndexFlat` codes

### M3-B：WASM HNSW search core

已扩展 `wasm/leann_wasm.c`，新增导出：

```c
int leann_wasm_hnsw_search(
  const float* vectors,
  int vector_count,
  int dimension,
  const uint64_t* offsets,
  const int32_t* neighbors,
  const int32_t* levels,
  int entry_point,
  int max_level,
  const float* query,
  int top_k,
  int ef_search,
  int metric,
  int32_t* out_labels,
  float* out_scores
);
```

这个函数执行 HNSW traversal，而不是遍历全量 vectors。当前本地 `wat2wasm` fallback 仍只覆盖 flat search；Actions run `26465439258` 的 Emscripten 产物已验证 `leann_wasm_hnsw_search` 可加载合成 HNSW，并可查询真实 LEANN HNSW fixture。

## M4 延后项

compact/pruned HNSW 暂不并入 M3。M4 再处理：

- compact CSR graph reader。
- pruned vectors 缺失时的 JS callback / provider adapter。
- 替代 Python ZMQ recompute 的浏览器可用协议。
