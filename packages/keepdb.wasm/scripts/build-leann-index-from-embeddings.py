import json
import os
import sys
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
        import leann_backend_hnsw  # noqa: F401
        try:
            from leann_backend_hnsw import faiss as _hnsw_faiss  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LEANN_WASM_E2E_BACKEND=hnsw requires the native "
                "leann_backend_hnsw.faiss extension. This is expected on local "
                "machines without the LEANN HNSW build toolchain; run the M3 "
                "fixture/build job in GitHub Actions instead."
            ) from exc
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
            "api_key": os.getenv("BIGMODEL_API_KEY", ""),
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


if __name__ == "__main__":
    raise SystemExit(main())
