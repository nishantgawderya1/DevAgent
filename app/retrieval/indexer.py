from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from app.retrieval import chunker


INDEX_ROOT = Path("/tmp/devagent")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 100
VECTOR_SIZE = 384
SKIP_DIRS = {"node_modules", ".venv", "__pycache__", "dist", "build", ".git"}

_embedding_model: Any | None = None


@dataclass(frozen=True)
class IndexSummary:
    repo_full_name: str
    collection_name: str
    repo_path: Path
    files_indexed: int
    chunks_indexed: int


try:
    from qdrant_client.http import models as qdrant_models
except Exception:  # pragma: no cover - dependency is mocked in tests
    from types import SimpleNamespace

    @dataclass(frozen=True)
    class _Distance:
        COSINE: str = "Cosine"

    @dataclass(frozen=True)
    class _VectorParams:
        size: int
        distance: str

    @dataclass(frozen=True)
    class _PointStruct:
        id: str
        vector: list[float]
        payload: dict[str, Any]

    qdrant_models = SimpleNamespace(Distance=_Distance(), VectorParams=_VectorParams, PointStruct=_PointStruct)


def index_repo(repo_full_name: str) -> IndexSummary:
    repo_path = _ensure_repo_checkout(repo_full_name)
    collection_name = _collection_name(repo_full_name)
    qdrant_client = _get_qdrant_client()

    if _collection_has_points(qdrant_client, collection_name):
        return IndexSummary(repo_full_name, collection_name, repo_path, 0, 0)

    _create_collection_if_needed(qdrant_client, collection_name)

    files_indexed = 0
    chunks: list[dict[str, Any]] = []
    for file_path in _iter_source_files(repo_path):
        files_indexed += 1
        chunks.extend(chunker.chunk_file(file_path))

    embeddings = _embed_chunks(chunks)
    _upsert_chunks(qdrant_client, collection_name, chunks, embeddings)

    return IndexSummary(repo_full_name, collection_name, repo_path, files_indexed, len(chunks))


def _ensure_repo_checkout(repo_full_name: str) -> Path:
    repo_name = repo_full_name.split("/")[-1]
    repo_path = INDEX_ROOT / repo_name
    if repo_path.exists() and any(repo_path.iterdir()):
        return repo_path

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if repo_path.exists():
        subprocess.run(["rm", "-rf", str(repo_path)], check=True)
    subprocess.run(["git", "clone", f"https://github.com/{repo_full_name}.git", str(repo_path)], check=True)
    return repo_path


def _iter_source_files(repo_path: Path) -> Iterator[Path]:
    for file_path in sorted(repo_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix not in {".py", ".js"}:
            continue
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        yield file_path


def _collection_name(repo_full_name: str) -> str:
    return repo_full_name.replace("/", "__")


def _get_qdrant_client() -> Any:
    from qdrant_client import QdrantClient

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def _collection_has_points(qdrant_client: Any, collection_name: str) -> bool:
    if not _collection_exists(qdrant_client, collection_name):
        return False

    result = qdrant_client.count(collection_name=collection_name, exact=True)
    return int(getattr(result, "count", result)) > 0


def _collection_exists(qdrant_client: Any, collection_name: str) -> bool:
    collection_exists = getattr(qdrant_client, "collection_exists", None)
    if callable(collection_exists):
        return bool(collection_exists(collection_name=collection_name))

    get_collection = getattr(qdrant_client, "get_collection", None)
    if callable(get_collection):
        try:
            get_collection(collection_name=collection_name)
        except Exception:
            return False
        return True

    return False


def _create_collection_if_needed(qdrant_client: Any, collection_name: str) -> None:
    if _collection_exists(qdrant_client, collection_name):
        return

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(size=VECTOR_SIZE, distance=qdrant_models.Distance.COSINE),
    )


def _load_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _embed_chunks(chunks: Sequence[dict[str, Any]]) -> list[list[float]]:
    if not chunks:
        return []

    model = _load_embedding_model()
    embeddings: list[list[float]] = []
    for batch in _batched([chunk["text"] for chunk in chunks], BATCH_SIZE):
        batch_embeddings = model.encode(batch, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
        embeddings.extend(_normalize_embeddings(batch_embeddings))
    return embeddings


def _normalize_embeddings(batch_embeddings: Any) -> list[list[float]]:
    if hasattr(batch_embeddings, "tolist"):
        return [list(vector) for vector in batch_embeddings.tolist()]
    return [list(vector) for vector in batch_embeddings]


def _upsert_chunks(
    qdrant_client: Any,
    collection_name: str,
    chunks: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
) -> None:
    points = [qdrant_models.PointStruct(id=chunk["id"], vector=list(vector), payload=chunk) for chunk, vector in zip(chunks, embeddings, strict=True)]
    qdrant_client.upsert(collection_name=collection_name, points=points)


def _batched(items: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])
