#!/usr/bin/env node
import { access, mkdir, stat } from "node:fs/promises";
import { constants } from "node:fs";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));
const outDir = join(rootDir, "dist");
const wasmPath = join(outDir, "leann-wasm.wasm");

await mkdir(outDir, { recursive: true });

const emcc = await findExecutable("emcc");
const wat2wasm = await findExecutable("wat2wasm");
const clang = await findExecutable("clang");

if (emcc) {
  run(emcc, [
    join(rootDir, "wasm/leann_wasm.c"),
    "-O3",
    "--no-entry",
    "-s",
    "STANDALONE_WASM=1",
    "-s",
    "ALLOW_MEMORY_GROWTH=1",
    "-s",
    'EXPORTED_FUNCTIONS=["_malloc","_free","_leann_wasm_version","_leann_wasm_score_dot","_leann_wasm_flat_search","_leann_wasm_hnsw_search"]',
    "-o",
    wasmPath,
  ]);
} else if (wat2wasm) {
  run(wat2wasm, [
    join(rootDir, "wasm/leann_wasm.wat"),
    "-o",
    wasmPath,
  ]);
} else if (clang) {
  run(clang, [
    "--target=wasm32-unknown-unknown-wasm",
    "-O3",
    "-nostdlib",
    "-Wl,--no-entry",
    "-Wl,--export-memory",
    "-Wl,--export=malloc",
    "-Wl,--export=free",
    "-Wl,--export=leann_wasm_version",
    "-Wl,--export=leann_wasm_score_dot",
    "-Wl,--export=leann_wasm_flat_search",
    "-Wl,--export=leann_wasm_hnsw_search",
    "-Wl,--allow-undefined",
    join(rootDir, "wasm/leann_wasm.c"),
    "-o",
    wasmPath,
  ]);
} else {
  throw new Error("emcc, wat2wasm, or clang is required to build WASM.");
}

const output = await stat(wasmPath);
console.log(`${wasmPath} ${(output.size / 1024).toFixed(1)} KiB`);

async function findExecutable(name) {
  const paths = (process.env.PATH || "").split(delimiter).filter(Boolean);
  for (const dir of paths) {
    const candidate = join(dir, name);
    try {
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Continue searching PATH.
    }
  }
  return null;
}

function run(command, args) {
  const result = spawnSync(command, args, { stdio: "inherit" });
  if (result.status !== 0) {
    throw new Error(`${command} failed with status ${result.status}`);
  }
}
