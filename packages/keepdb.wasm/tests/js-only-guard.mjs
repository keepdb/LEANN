import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { basename, extname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = new URL("..", import.meta.url);
const packageRootPath = fileURLToPath(packageRoot);
const ignoredDirs = new Set([".tmp", "dist", "node_modules"]);
const prohibitedExtensions = new Set([".py", ".pyc", ".pyo"]);
const prohibitedScriptPattern = /\b(python|uv)\b|\.py\b|\.sh\b/;
const prohibitedSourcePattern = /\b(python|uv)\b|\.py\b|LEANN_WASM_PYTHON/;
const scannedSourceExtensions = new Set([".js", ".mjs"]);
const allowSourceFiles = new Set(["tests/js-only-guard.mjs"]);

const files = await listFiles(packageRootPath);
const prohibitedFiles = files
  .filter((file) => prohibitedExtensions.has(extname(file)))
  .map((file) => relative(packageRootPath, file));

assert.deepEqual(prohibitedFiles, [], "packages/keepdb.wasm must not contain Python files.");

const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
assert.equal(packageJson.type, "module", 'packages/keepdb.wasm package.json must use "type": "module".');

const prohibitedScripts = Object.entries(packageJson.scripts ?? {})
  .filter(([, command]) => prohibitedScriptPattern.test(command));

assert.deepEqual(
  prohibitedScripts,
  [],
  "packages/keepdb.wasm package scripts must not invoke Python, uv, .py, or .sh.",
);

const prohibitedSources = [];
for (const file of files) {
  const rel = relative(packageRootPath, file);
  if (!scannedSourceExtensions.has(extname(file)) || allowSourceFiles.has(rel)) {
    continue;
  }
  const text = await readFile(file, "utf8");
  if (prohibitedSourcePattern.test(text)) {
    prohibitedSources.push(rel);
  }
}

assert.deepEqual(
  prohibitedSources,
  [],
  "packages/keepdb.wasm JS sources/tests must not reference Python, uv, or .py helpers.",
);

console.log("keepdb.wasm JS-only guard passed");

async function listFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (entry.isDirectory()) {
      if (ignoredDirs.has(entry.name) || entry.name === "__pycache__") {
        continue;
      }
      files.push(...await listFiles(join(dir, entry.name)));
    } else if (entry.isFile()) {
      files.push(join(dir, entry.name));
    }
  }
  return files;
}
