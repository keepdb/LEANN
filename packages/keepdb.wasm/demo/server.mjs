import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageDir = resolve(fileURLToPath(new URL("..", import.meta.url)));
const demoDir = resolve(packageDir, "demo");
const fixtureDir = resolve(packageDir, ".tmp", "bigmodel-e2e");
const wasmFile = resolve(packageDir, "dist", "leann-wasm.wasm");

const port = Number.parseInt(
  process.env.PORT || execFileSync("npx", ["--loglevel=error", "@keepdb/cli", "port"], {
    cwd: packageDir,
    encoding: "utf8",
  }).trim(),
  10,
);

const routes = new Map([
  ["/", resolve(demoDir, "index.html")],
  ["/app.js", resolve(demoDir, "app.js")],
  ["/styles.css", resolve(demoDir, "styles.css")],
  ["/src/index.js", resolve(packageDir, "src", "index.js")],
  ["/src/providers.js", resolve(packageDir, "src", "providers.js")],
  ["/src/wasm-loader.js", resolve(packageDir, "src", "wasm-loader.js")],
  ["/src/wasm-search-core.js", resolve(packageDir, "src", "wasm-search-core.js")],
  ["/src/mock-search-core.js", resolve(packageDir, "src", "mock-search-core.js")],
  ["/src/leann-index-parser.js", resolve(packageDir, "src", "leann-index-parser.js")],
  ["/src/leann-sidecars.js", resolve(packageDir, "src", "leann-sidecars.js")],
  ["/assets/leann-wasm.wasm", wasmFile],
  ["/artifacts/bigmodel.leann.meta.json", resolve(fixtureDir, "bigmodel.leann.meta.json")],
  ["/artifacts/bigmodel.index", resolve(fixtureDir, "bigmodel.index")],
  ["/artifacts/bigmodel.ids.txt", resolve(fixtureDir, "bigmodel.ids.txt")],
  ["/artifacts/bigmodel.leann.passages.jsonl", resolve(fixtureDir, "bigmodel.leann.passages.jsonl")],
]);

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

    const file = routes.get(url.pathname);
    if (!file) {
      sendJson(res, 404, { error: "Not found" });
      return;
    }
    if (!existsSync(file)) {
      sendJson(res, 404, { error: "Required demo artifact is missing." });
      return;
    }

    const bytes = await readFile(file);
    res.writeHead(200, {
      "Content-Type": contentType(file),
      "Cache-Control": "no-store",
    });
    res.end(bytes);
  } catch (error) {
    sendJson(res, 500, { error: error.message });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`keepdb.wasm demo: http://127.0.0.1:${port}`);
});

function sendJson(res, status, value) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(value));
}

function contentType(file) {
  const types = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".index": "application/octet-stream",
    ".wasm": "application/wasm",
  };
  return types[extname(file)] || "application/octet-stream";
}
