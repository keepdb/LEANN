import { writeHnswCapabilityInput } from "./leann-fixture-utils.mjs";

const outputUrl = new URL("../.tmp/hnsw-capability/", import.meta.url);
await writeHnswCapabilityInput(outputUrl);
console.log(`Wrote ${new URL("docs-embeddings.json", outputUrl).pathname}`);
