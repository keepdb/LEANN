import json
import os
import pickle
import sys
import types
from importlib import util as importlib_util
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "Usage: build-leann-index-from-embeddings.py <docs_embeddings.json> <output_dir> <index_name>",
            file=sys.stderr,
        )
        return 2

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    index_name = sys.argv[3]

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    docs = payload["documents"]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    ids = [doc["id"] for doc in docs]

    if len(ids) != embeddings.shape[0]:
        raise ValueError("documents and embeddings length mismatch")

    backend = os.getenv("LEANN_WASM_E2E_BACKEND", "ivf")
    dimensions = int(embeddings.shape[1])

    # Import backend modules explicitly so their register_backend decorators run
    # even when the packages are loaded from PYTHONPATH instead of installed dists.
    if backend == "ivf":
        import leann_backend_ivf  # noqa: F401
    elif backend == "hnsw":
        build_hnsw_index_from_arrays(output_dir, index_name, docs, ids, embeddings)
        return 0
    else:
        raise ValueError(f"Unsupported LEANN_WASM_E2E_BACKEND={backend!r}")

    from leann.api import LeannBuilder

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / index_name

    backend_kwargs = {"distance_metric": "cosine"}
    if backend == "ivf":
        backend_kwargs["nlist"] = int(os.getenv("LEANN_WASM_E2E_IVF_NLIST", "1"))
    if backend == "hnsw":
        backend_kwargs.update({"is_recompute": False, "is_compact": False, "M": 8})

    builder = LeannBuilder(
        backend_name=backend,
        embedding_model=os.getenv("BIGMODEL_EMBEDDING_MODEL", "embedding-3"),
        embedding_mode="openai",
        dimensions=dimensions,
        embedding_options={
            "base_url": os.getenv("BIGMODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        },
        **backend_kwargs,
    )

    for doc in docs:
        builder.add_text(doc["text"], metadata={"id": doc["id"], "source": "bigmodel-e2e"})

    builder.build_index_from_arrays(str(index_path), ids, embeddings)

    artifact_names = [
        f"{index_name}.meta.json",
        f"{Path(index_name).stem}.index",
        f"{Path(index_name).stem}.ids.txt",
        f"{index_name}.passages.jsonl",
    ]
    missing = [name for name in artifact_names if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected LEANN artifacts: {missing}")

    print(json.dumps({"index_path": str(index_path), "artifacts": artifact_names}, indent=2))
    return 0


def build_hnsw_index_from_arrays(
    output_dir: Path,
    index_name: str,
    docs: list[dict],
    ids: list[str],
    embeddings: np.ndarray,
) -> None:
    faiss = load_hnsw_faiss_extension()
    dimensions = int(embeddings.shape[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / index_name
    index_stem = Path(index_name).stem
    passages_file = output_dir / f"{index_name}.passages.jsonl"
    offset_file = output_dir / f"{index_name}.passages.idx"
    idmap_file = output_dir / f"{index_stem}.ids.txt"
    index_file = output_dir / f"{index_stem}.index"
    meta_file = output_dir / f"{index_name}.meta.json"

    offset_map = {}
    with open(passages_file, "w", encoding="utf-8") as f:
        for doc in docs:
            offset = f.tell()
            metadata = {"id": doc["id"], "source": "bigmodel-e2e"}
            json.dump(
                {"id": doc["id"], "text": doc["text"], "metadata": metadata},
                f,
                ensure_ascii=False,
            )
            f.write("\n")
            offset_map[doc["id"]] = offset

    with open(offset_file, "wb") as f:
        pickle.dump(offset_map, f)

    with open(idmap_file, "w", encoding="utf-8") as f:
        for id_value in ids:
            f.write(str(id_value) + "\n")

    data = embeddings.astype(np.float32, copy=True)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1
    data = data / norms

    metric = faiss.METRIC_INNER_PRODUCT
    index = faiss.IndexHNSWFlat(dimensions, int(os.getenv("LEANN_WASM_E2E_HNSW_M", "8")), metric)
    index.hnsw.efConstruction = int(os.getenv("LEANN_WASM_E2E_HNSW_EF_CONSTRUCTION", "40"))
    index.add(data.shape[0], faiss.swig_ptr(data))
    faiss.write_index(index, str(index_file))

    backend_kwargs = {
        "distance_metric": "cosine",
        "is_recompute": False,
        "is_compact": False,
        "M": int(os.getenv("LEANN_WASM_E2E_HNSW_M", "8")),
    }
    meta_data = {
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": os.getenv("BIGMODEL_EMBEDDING_MODEL", "embedding-3"),
        "dimensions": dimensions,
        "backend_kwargs": backend_kwargs,
        "embedding_mode": "openai",
        "passage_sources": [
            {
                "type": "jsonl",
                "path": passages_file.name,
                "index_path": offset_file.name,
                "path_relative": passages_file.name,
                "index_path_relative": offset_file.name,
            }
        ],
        "built_from_precomputed_embeddings": True,
        "embedding_options": {
            "base_url": os.getenv("BIGMODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        },
        "is_compact": False,
        "is_pruned": False,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    artifact_names = [
        f"{index_name}.meta.json",
        f"{index_stem}.index",
        f"{index_stem}.ids.txt",
        f"{index_name}.passages.jsonl",
    ]
    missing = [name for name in artifact_names if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected LEANN artifacts: {missing}")

    print(json.dumps({"index_path": str(index_path), "artifacts": artifact_names}, indent=2))


def load_hnsw_faiss_extension():
    candidate_paths = []
    repo_root = Path(__file__).resolve().parents[3]
    search_roots = [
        repo_root / "packages" / "leann-backend-hnsw" / "leann_backend_hnsw",
        *[Path(entry) / "leann_backend_hnsw" for entry in sys.path if entry],
    ]
    for root in search_roots:
        if root.exists():
            candidate_paths.extend(root.glob("faiss*.so"))
            candidate_paths.extend(root.glob("faiss*.pyd"))

    if not candidate_paths:
        raise RuntimeError(
            "LEANN_WASM_E2E_BACKEND=hnsw requires the native "
            "leann_backend_hnsw.faiss extension. This is expected on local "
            "machines without the LEANN HNSW build toolchain; run the M3 "
            "fixture/build job in GitHub Actions instead."
        )

    extension_path = candidate_paths[0]
    package = types.ModuleType("leann_backend_hnsw")
    package.__path__ = [str(extension_path.parent)]
    sys.modules.setdefault("leann_backend_hnsw", package)

    spec = importlib_util.spec_from_file_location("leann_backend_hnsw.faiss", extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load HNSW Faiss extension from {extension_path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules["leann_backend_hnsw.faiss"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
