# LEANN WASM 封装评估与 TODO

日期：2026-05-26

## 结论

LEANN 当前不适合把完整 Python 包直接封装成浏览器 WASM。

可行方向是先做一个收窄范围的 WASM MVP：只封装只读向量索引搜索内核，输入由外部提前计算好的 query embedding，输出候选 passage id 与距离分数。索引构建、文本切分、embedding 模型推理、LLM 调用、daemon、ZMQ 重算服务暂不进入 WASM 范围。

原因：

- `leann-core` 是 Python API 层，依赖 `torch`、`sentence-transformers`、`transformers`、`pyzmq`、`llama-index`、`openai` 等运行时；这些能力不适合作为浏览器 WASM 的第一阶段目标。
- 默认 `hnsw` 后端通过 `scikit-build-core` + CMake + SWIG 构建定制 Faiss Python 扩展，当前接口面向 CPython 扩展模块，不是面向 JavaScript 或 WASI 的稳定 C ABI。
- LEANN 的核心压缩收益来自 pruned/compact index + selective recomputation；当前搜索重算路径依赖本地 ZMQ embedding server。浏览器 WASM 不能直接复用这个本地 TCP/ZMQ daemon 模型。
- `diskann` 后端依赖 DiskANN、磁盘索引、线程、BLAS/MKL/OpenBLAS、protobuf 等原生能力，作为 WASM 首发目标风险过高。
- `ivf` 后端依赖 `faiss-cpu` Python wheel；除非单独把 Faiss 编译到 WASM，否则不能直接复用。

所以判断是：可以推进 WASM，但必须先定义一个明确、可验证、较小的封装面。

## 推荐首发范围

首发目标：`leann-wasm-search`。

能力边界：

- 支持读取已构建好的索引文件。
- 支持输入 `float32` query embedding。
- 支持返回 top-k label 与 distance。
- 首版只支持不需要重算 embedding 的搜索路径。
- 不在 WASM 中计算 embedding。
- 不在 WASM 中构建索引。
- 不在 WASM 中启动 daemon、socket 或 ZMQ 服务。
- 不在 WASM 中调用 OpenAI、Hugging Face、Ollama、MLX 或本地模型。

首选后端顺序：

1. HNSW read-only search，优先验证非 pruned index。
2. HNSW compact/CSR read-only search，但前提是确认不需要 ZMQ 重算，或新增同步 embedding callback。
3. IVF/Faiss WASM，作为第二阶段。
4. DiskANN WASM，暂不建议进入近期范围。

## 需要满足的封装条件

### 代码条件

- 从 `packages/leann-backend-hnsw` 中拆出一个纯 C++ search core 入口，避免直接暴露 CPython/SWIG API。
- 新增一个稳定 C ABI 或 Embind API，例如：
  - `leann_wasm_load_index(bytes | path)`
  - `leann_wasm_search(query_embedding_float32, top_k, complexity)`
  - `leann_wasm_free(handle)`
- 搜索路径不能依赖：
  - Python interpreter
  - `pyzmq`
  - TCP socket
  - subprocess/daemon
  - runtime model loading
  - platform-specific mmap
- 如需读取 index/passages/meta，优先使用 Emscripten MEMFS/WORKERFS 或由 JS 层传入 bytes，避免假设本地文件系统存在。
- 如果复用 Faiss，需要确认当前 fork 的 HNSW 改动能在 Emscripten 下关闭不兼容依赖并完成编译。

### 产品条件

- WASM 包的 API 必须要求调用方传入 query embedding；embedding 生成交给 JS 应用、服务端或独立模型运行时。
- WASM 首版只保证搜索内核，不承诺完整 LEANN CLI/API 行为。
- 索引格式需要版本标记，避免未来 native index 与 WASM reader 不兼容。
- 对大索引必须明确内存预算；浏览器环境下不能默认加载百万级或千万级全量索引。

### CI 条件

用户本机不安装打包环境。所有 WASM 构建、测试和产物生成都通过 GitHub Actions 完成。

## GitHub Actions 实施方案

新增独立 workflow：`.github/workflows/keepdb-wasm.yml`。

触发方式：

- `workflow_dispatch`
- `pull_request`，仅当改动命中 WASM 相关路径时运行

建议 job：

 1. checkout
   - 使用 `actions/checkout`
   - M3 HNSW/Faiss 验证需要 `submodules: recursive`
2. setup
   - 安装 Emscripten SDK
   - M3 HNSW/Faiss 验证需要 CMake/Ninja 和 LEANN HNSW native build 依赖
   - 不安装项目完整 Python runtime，除非需要生成 fixture
3. build
   - 编译真实 flat vector search WASM core
   - 输出 `dist/leann-wasm.wasm`
4. fixture
   - 使用已提交的小型 fixture index
   - 或在 CI 中用 Python 生成一个极小非 pruned HNSW index
5. smoke test
   - Node.js 加载 WASM
   - 传入固定 query embedding
   - 断言 top-k label 与距离稳定
6. artifact
   - 上传包含 WASM bundle 的 npm tarball

不建议把 WASM job 合并进现有 `.github/workflows/build-reusable.yml` 的主矩阵。现有矩阵已经覆盖 Linux/macOS/Windows 多 Python 版本和原生 wheel 修复，WASM 应保持独立，避免扩大主 CI 的失败面。

## 目录建议

已新增专用验证目录：

```text
packages/keepdb.wasm/
  README.md
  src/
    index.js
    providers.js
    wasm-loader.js
    wasm-search-core.js
    mock-search-core.js
  wasm/
    leann_wasm.c
  scripts/
    build-wasm.sh
  tests/
    smoke.mjs
    wasm-smoke.mjs
```

职责边界：

- `src/index.js` 只负责收敛导出入口，不堆搜索细节。
- `src/wasm-search-core.js` 只处理 WASM search adapter。
- `wasm/leann_wasm.c` 当前实现真实 flat vector top-k search，后续替换为 HNSW/LEANN index search core。
- `README.md` 说明构建方式、API、限制、验证命令和 GitHub Actions 产物位置。
- fixture 必须足够小，便于进 Git；如果 index 二进制不适合进 Git，则 workflow 在 CI 中生成。

## TODO

### P0：确认最小可行封装面

- [ ] 明确首版只支持 `search(query_embedding)`，不支持 `search(query_text)`。
- [ ] 明确首版只支持非 pruned HNSW index，避免 ZMQ recompute 阻塞。
- [ ] 梳理 HNSW 搜索路径中真实需要的 Faiss 类型与方法：
  - `read_index`
  - `IndexHNSWFlat` 或当前 fork 的 CSR reader
  - `SearchParametersHNSW`
  - `search`
- [ ] 确认搜索路径是否强依赖 `IO_FLAG_MMAP`；如果强依赖，需要改成可从 bytes 或普通 stream 加载。
- [ ] 确认当前 fork 的 Faiss 是否可以禁用 OpenMP、ZMQ、Python/SWIG，只编译最小 HNSW search core。

### P1：建立 CI 原型

- [x] 新增 `.github/workflows/keepdb-wasm.yml`。
- [x] 在 GitHub Actions 中安装 Emscripten SDK。
- [x] 新增 `packages/keepdb.wasm/scripts/build-wasm.sh`，编译真实 flat search WASM core。
- [x] 新增 Node.js smoke test，验证 WASM 可以加载并完成一次 top-k 搜索。
- [x] 上传 npm tarball artifact，其中包含 `dist/leann-wasm.wasm`。
- [ ] 将 flat search core 替换为 LEANN HNSW index reader/searcher。

### P2：索引 fixture 与兼容性

- [x] 准备一个 BigModel 公开小样本文档 fixture，维度和向量数量足够小。
- [x] fixture 记录生成命令和后端参数；当前后端为 LEANN IVF `nlist=1`、`distance_metric=cosine`。
- [ ] 为 index meta 增加或复用格式版本字段。
- [x] 在 BigModel e2e test 中断言：
  - load 成功
  - query embedding 维度校验生效
  - top-k 数量正确
  - label 映射正确
  - distance dtype/排序稳定
  - passage text 与 metadata 可回填

### M1/M2 当前状态

- [x] M1：读取 LEANN 真实最小索引产物并完成 `search(queryEmbedding)`。
  - 当前限定：`backend_name=ivf`、`nlist=1`、`distance_metric=cosine`。
  - 读取文件：`.meta.json`、`.index`、`.ids.txt`、`.passages.jsonl`。
- [x] M2：接入 BigModel `embedding-3` 做文档向量化和 query 向量化。
  - 当前测试：`pnpm --dir packages/keepdb.wasm test:e2e:bigmodel`。
  - 测试结果：预期 `top1=doc-wasm`，WASM 返回 `doc-wasm`。
  - 示例页：`packages/keepdb.wasm/demo/` 通过 `npx @keepdb/cli port` 启动静态预览，验证者在浏览器输入临时 BigModel API Key 后直连 `/embeddings`，本地预览进程不读取或转发 Key。
- [ ] M3：支持 LEANN HNSW/Faiss 后端，不再局限 IVF `nlist=1` 最小 reader。
  - M3 方案文档：`docs/设计/leann-wasm-m3-hnsw.md`。
  - 已实现 non-compact HNSW `IHNf` parser 与 WASM `leann_wasm_hnsw_search()` 调用路径。
  - 未确认：GitHub Actions 能否成功构建 `leann_backend_hnsw.faiss` 原生扩展并产出真实 HNSW fixture。

### P3：扩展能力

- [x] 新增 M3 HNSW capability 探针：`pnpm --dir packages/keepdb.wasm test:hnsw-capability`。
  - 本地无 `leann_backend_hnsw.faiss` 原生扩展时只报告 `missing-hnsw-native-extension`，不伪装成 HNSW WASM 已完成。
  - GitHub Actions 通过 `LEANN_WASM_REQUIRE_HNSW=1` 把 HNSW fixture 生成变成强制闸门。
- [x] 新增 HNSW parser smoke：`pnpm --dir packages/keepdb.wasm test:hnsw-parser`。
- [x] 新增 HNSW WASM smoke：`pnpm --dir packages/keepdb.wasm test:hnsw-wasm`。
  - 本地 `wat2wasm` fallback 无 HNSW 导出时跳过。
  - CI Emscripten 构建必须导出 `leann_wasm_hnsw_search` 并通过合成 HNSW 查询。
- [ ] 在 GitHub Actions 中确认 `packages/leann-backend-hnsw` 原生 wheel 可构建，并产出非 compact、非 pruned HNSW fixture。
- [ ] 基于真实 HNSW fixture 实现 WASM HNSW graph search，而不是继续从索引中抽向量做 flat search。
- [ ] 评估 compact/CSR HNSW 在无 recompute 情况下是否有价值。
- [ ] 如果需要保留 selective recomputation，设计 JS callback 或外部 embedding provider API，替代当前 ZMQ server。
- [ ] 评估 IVF/Faiss WASM 的体积、性能和构建复杂度。
- [ ] 评估 Web Worker 运行搜索，避免阻塞浏览器主线程。
- [ ] 评估 streaming/分片加载大索引，避免一次性加载超出浏览器内存。

## 暂不做

- 不把完整 `leann-core` Python API 通过 Pyodide 直接搬进浏览器。
- 不在浏览器 WASM 中运行 `torch`、`sentence-transformers` 或 `transformers`。
- 不在首版中支持 DiskANN。
- 不在首版中支持本地 daemon、ZMQ、TCP socket。
- 不在本机安装 WASM 打包环境。

## 验收标准

第一阶段完成的定义：

- GitHub Actions 可以手动触发 WASM build。
- CI 产出可下载的 WASM bundle。
- Node.js smoke test 通过。
- README 说明最小 API、限制和示例调用。
- 本机无需安装 Emscripten、CMake、Ninja 或其他 WASM 打包环境。
