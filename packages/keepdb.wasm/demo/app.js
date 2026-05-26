import { loadLeannIndex } from "/src/index.js";

const elements = {
  form: document.querySelector("#query-form"),
  apiKey: document.querySelector("#api-key"),
  toggleKey: document.querySelector("#toggle-key"),
  query: document.querySelector("#query"),
  button: document.querySelector("#search-button"),
  results: document.querySelector("#results"),
  documents: document.querySelector("#documents"),
  latency: document.querySelector("#latency"),
  status: document.querySelector("#runtime-status"),
  backend: document.querySelector("#meta-backend"),
  dimension: document.querySelector("#meta-dimension"),
  metric: document.querySelector("#meta-metric"),
  runtime: document.querySelector("#meta-runtime"),
};

let index;
let resources;

initializeAssets().catch((error) => {
  setStatus("不可用", "error");
  renderError(error.message);
});

elements.apiKey.addEventListener("input", () => {
  disposeIndex();
  elements.button.disabled = !resources || !elements.apiKey.value.trim();
  setStatus(elements.apiKey.value.trim() ? "可检索" : "请输入 Key", "idle");
});

elements.toggleKey.addEventListener("click", () => {
  const isVisible = elements.apiKey.type === "text";
  elements.apiKey.type = isVisible ? "password" : "text";
  elements.toggleKey.textContent = isVisible ? "显示" : "隐藏";
  elements.toggleKey.setAttribute("aria-label", isVisible ? "显示 API Key" : "隐藏 API Key");
});

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const apiKey = elements.apiKey.value.trim();
  const text = elements.query.value.trim();
  if (!apiKey || !text || !resources) return;

  elements.button.disabled = true;
  elements.latency.value = "";
  setStatus("查询中", "working");
  const start = performance.now();

  try {
    if (!index) {
      index = await createIndex(apiKey);
    }
    const results = await index.searchText(text, { topK: 3 });
    elements.latency.value = `${Math.round(performance.now() - start)} ms`;
    renderResults(results);
    setStatus("WASM Ready", "success");
  } catch (error) {
    renderError(error.message);
    setStatus("请求失败", "error");
  } finally {
    elements.button.disabled = false;
  }
});

async function initializeAssets() {
  const [meta, indexBytes, idsText, passagesJsonl, wasmBytes] = await Promise.all([
    fetchJson("/artifacts/bigmodel.leann.meta.json"),
    fetchBytes("/artifacts/bigmodel.index"),
    fetchText("/artifacts/bigmodel.ids.txt"),
    fetchText("/artifacts/bigmodel.leann.passages.jsonl"),
    fetchBytes("/assets/leann-wasm.wasm"),
  ]);

  elements.backend.textContent = meta.backend_name.toUpperCase();
  elements.dimension.textContent = String(meta.dimensions);
  elements.metric.textContent = meta.backend_kwargs.distance_metric;
  elements.runtime.textContent = "待查询";
  renderDocuments(passagesJsonl);
  resources = { meta, indexBytes, idsText, passagesJsonl, wasmBytes };
  elements.button.disabled = !elements.apiKey.value.trim();
  setStatus("请输入 Key", "idle");
}

async function createIndex(apiKey) {
  const loadedIndex = await loadLeannIndex({
    metaJson: resources.meta,
    indexBytes: resources.indexBytes,
    idsText: resources.idsText,
    passagesJsonl: resources.passagesJsonl,
    wasmBytes: resources.wasmBytes,
    metric: resources.meta.backend_kwargs.distance_metric,
    embeddingProvider: {
      type: "openai-compatible",
      baseUrl: "https://open.bigmodel.cn/api/paas/v4",
      apiKey,
      model: "embedding-3",
      dimensions: resources.meta.dimensions,
    },
  });
  elements.runtime.textContent = loadedIndex.mode.toUpperCase();
  return loadedIndex;
}

function renderResults(results) {
  elements.results.replaceChildren(
    ...results.map((result, rank) => {
      const item = document.createElement("article");
      item.className = "result-item";
      item.innerHTML = `
        <div class="rank">${rank + 1}</div>
        <div class="result-content">
          <div class="result-row">
            <strong>${escapeHtml(result.id)}</strong>
            <span class="score">${result.score.toFixed(4)}</span>
          </div>
          <p>${escapeHtml(result.text || "")}</p>
        </div>
      `;
      return item;
    }),
  );
}

function renderDocuments(jsonl) {
  const docs = jsonl
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));

  elements.documents.replaceChildren(
    ...docs.map((doc) => {
      const row = document.createElement("div");
      row.className = "document-row";
      row.innerHTML = `<strong>${escapeHtml(doc.id)}</strong><span>${escapeHtml(doc.text)}</span>`;
      return row;
    }),
  );
}

function renderError(message) {
  const displayMessage = message === "Failed to fetch"
    ? "浏览器无法直连 BigModel，请检查网络或跨域策略。"
    : message;
  elements.results.innerHTML = `<div class="error-message">${escapeHtml(displayMessage)}</div>`;
}

function setStatus(text, state) {
  elements.status.textContent = text;
  elements.status.dataset.state = state;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`读取失败: ${url}`);
  return response.json();
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`读取失败: ${url}`);
  return response.text();
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`读取失败: ${url}`);
  return response.arrayBuffer();
}

function disposeIndex() {
  index?.dispose();
  index = undefined;
  elements.runtime.textContent = "待查询";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
