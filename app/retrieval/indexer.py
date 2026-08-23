from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from app.retrieval import chunker


logger = logging.getLogger(__name__)

INDEX_ROOT = Path(os.getenv("DEVAGENT_INDEX_ROOT", Path(tempfile.gettempdir()) / "devagent"))
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 100
VECTOR_SIZE = 384
SKIP_DIRS = {"node_modules", ".venv", "__pycache__", "dist", "build", ".git"}

# Qdrant only accepts unsigned-int or UUID point ids. Chunk ids are strings
# ("path::name"), so we map them to deterministic UUIDs: the same chunk id
# always yields the same point id, which keeps upserts idempotent.
_POINT_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

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
    """Clone, chunk, embed and upsert a repository into Qdrant.

    Errors are caught and logged; on failure a zeroed summary is returned rather
    than propagating an exception to the caller.
    """
    collection_name = _collection_name(repo_full_name)
    try:
        repo_path = _ensure_repo_checkout(repo_full_name)
        qdrant_client = _get_qdrant_client()

        head_sha = _head_sha(repo_path)
        indexed_sha = _read_indexed_sha(collection_name)
        populated = _collection_has_points(qdrant_client, collection_name)

        if populated and head_sha is not None and head_sha == indexed_sha:
            logger.info("Collection %s is current at %s; skipping re-index.", collection_name, head_sha[:8])
            return IndexSummary(repo_full_name, collection_name, repo_path, 0, 0)

        if populated:
            # The checkout moved. Points are keyed by deterministic uuid5, so a
            # plain upsert would refresh surviving chunks but leave behind any
            # whose defining code was deleted upstream. Rebuild instead.
            logger.info(
                "Collection %s is stale (indexed %s, head %s); rebuilding.",
                collection_name,
                (indexed_sha or "unknown")[:8],
                (head_sha or "unknown")[:8],
            )
            _drop_collection(qdrant_client, collection_name)

        _create_collection_if_needed(qdrant_client, collection_name)

        files_indexed = 0
        chunks: list[dict[str, Any]] = []
        for file_path in _iter_source_files(repo_path):
            files_indexed += 1
            chunks.extend(_safe_chunk_file(file_path))

        embeddings = _embed_chunks(chunks)
        _upsert_chunks(qdrant_client, collection_name, chunks, embeddings)

        if head_sha is not None:
            _write_indexed_sha(collection_name, head_sha)

        logger.info("Indexed %s: %d files, %d chunks.", repo_full_name, files_indexed, len(chunks))
        return IndexSummary(repo_full_name, collection_name, repo_path, files_indexed, len(chunks))
    except Exception:
        logger.exception("Failed to index repository %s", repo_full_name)
        return IndexSummary(repo_full_name, collection_name, INDEX_ROOT, 0, 0)


def _safe_chunk_file(file_path: Path) -> list[dict[str, Any]]:
    try:
        return chunker.chunk_file(file_path)
    except Exception:
        logger.warning("Skipping file that failed to chunk: %s", file_path, exc_info=True)
        return []


def _ensure_repo_checkout(repo_full_name: str) -> Path:
    """Return a checkout of ``repo_full_name`` synced to its default branch.

    Two things here are load-bearing.

    The directory is keyed on the *full* name via :func:`_collection_name`, not
    on the trailing path segment. Keying on the segment made ``alice/utils`` and
    ``bob/utils`` share one directory while Qdrant kept them apart, so a patch
    could be applied to the wrong repository's working tree.

    An existing checkout is fetched and hard-reset rather than returned as-is.
    Downstream, the PR node builds its branch from this tree and opens the pull
    request against the remote's default branch; if the tree were stale the diff
    would contain a revert of every upstream commit made since the clone.
    """
    repo_path = INDEX_ROOT / _collection_name(repo_full_name)

    if (repo_path / ".git").is_dir():
        try:
            _sync_checkout(repo_path)
            return repo_path
        except subprocess.CalledProcessError:
            # A corrupt or half-cloned tree is not worth diagnosing; recloning is
            # cheap and always correct.
            logger.warning("Could not sync %s; recloning.", repo_path, exc_info=True)
            shutil.rmtree(repo_path, ignore_errors=True)

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if repo_path.exists():
        shutil.rmtree(repo_path)
    subprocess.run(
        ["git", "clone", f"https://github.com/{repo_full_name}.git", str(repo_path)], check=True
    )
    return repo_path


def _sync_checkout(repo_path: Path) -> None:
    """Fetch and hard-reset an existing checkout onto the remote default branch."""
    _run_git(["fetch", "--quiet", "origin"], repo_path)
    branch = _default_branch(repo_path)
    _run_git(["checkout", "--quiet", "--force", "-B", branch, f"origin/{branch}"], repo_path)
    _run_git(["reset", "--quiet", "--hard", f"origin/{branch}"], repo_path)
    _run_git(["clean", "-qfd"], repo_path)


def _default_branch(repo_path: Path) -> str:
    """Resolve the remote's default branch, falling back to main."""
    try:
        ref = _run_git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"], repo_path).strip()
        if ref:
            return ref.rsplit("/", 1)[-1]
    except subprocess.CalledProcessError:
        # origin/HEAD is not always set on a local or shallow clone.
        logger.debug("origin/HEAD unset in %s; falling back.", repo_path)

    try:
        return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_path).strip() or "main"
    except subprocess.CalledProcessError:
        return "main"


def _head_sha(repo_path: Path) -> str | None:
    """Current commit of a checkout, or None when it cannot be read."""
    try:
        return _run_git(["rev-parse", "HEAD"], repo_path).strip() or None
    except subprocess.CalledProcessError:
        return None


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return result.stdout


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


def _indexed_sha_path(collection_name: str) -> Path:
    return INDEX_ROOT / ".index-state" / f"{collection_name}.sha"


def _read_indexed_sha(collection_name: str) -> str | None:
    """The commit the collection was last built from, if we recorded one.

    Paired with a points check rather than trusted alone: if Qdrant is wiped
    independently the file would still claim the repo is indexed, so both the
    marker and the collection must agree before a re-index is skipped.
    """
    path = _indexed_sha_path(collection_name)
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_indexed_sha(collection_name: str, sha: str) -> None:
    path = _indexed_sha_path(collection_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sha, encoding="utf-8")
    except OSError:
        # Losing the marker costs a redundant re-index, never correctness.
        logger.warning("Could not record indexed sha for %s.", collection_name, exc_info=True)


def _drop_collection(qdrant_client: Any, collection_name: str) -> None:
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        logger.warning("Could not drop collection %s.", collection_name, exc_info=True)


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
    points = [
        qdrant_models.PointStruct(id=_point_id(chunk["id"]), vector=list(vector), payload=chunk)
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)


def _point_id(chunk_id: str) -> str:
    """Map a string chunk id to a deterministic UUID accepted by Qdrant."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def _batched(items: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])
