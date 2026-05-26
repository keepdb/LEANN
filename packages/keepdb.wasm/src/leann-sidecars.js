export function parseLeannSidecars({ metaJson, idsText, passagesJsonl } = {}) {
  const meta = parseMeta(metaJson);
  const ids = parseIds(idsText);
  const passages = parsePassages(passagesJsonl);

  return {
    meta,
    ids,
    passages,
    passageById: new Map(passages.map((item) => [item.id, item])),
  };
}

function parseMeta(metaJson) {
  if (!metaJson) {
    return null;
  }
  if (typeof metaJson === "string") {
    return JSON.parse(metaJson);
  }
  return metaJson;
}

function parseIds(idsText) {
  if (!idsText) {
    return [];
  }
  return idsText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function parsePassages(passagesJsonl) {
  if (!passagesJsonl) {
    return [];
  }
  return passagesJsonl
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}
