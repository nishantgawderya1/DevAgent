from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

from app.retrieval import indexer


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size=100, convert_to_numpy=True, show_progress_bar=False):
        batch = list(texts)
        self.calls.append(batch)
        return [[float(len(batch))] * indexer.VECTOR_SIZE for _ in batch]


class FakeQdrantClient:
    def __init__(self, collection_exists: bool = False, point_count: int = 0) -> None:
        self.collection_exists_value = collection_exists
        self.point_count = point_count
        self.created: list[tuple[str, object]] = []
        self.upserts: list[tuple[str, list[object]]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.collection_exists_value

    def count(self, collection_name: str, exact: bool = True):
        return SimpleNamespace(count=self.point_count)

    def create_collection(self, collection_name: str, vectors_config):
        self.created.append((collection_name, vectors_config))

    def upsert(self, collection_name: str, points):
        self.upserts.append((collection_name, list(points)))


def test_index_repo_walks_files_batches_embeddings_and_upserts(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "flask"
    (repo_path / "pkg").mkdir(parents=True)
    for index in range(205):
        suffix = ".py" if index % 2 == 0 else ".js"
        (repo_path / "pkg" / f"file_{index}{suffix}").write_text("content", encoding="utf-8")
    for ignored in [
        repo_path / "node_modules" / "skip.py",
        repo_path / ".venv" / "skip.py",
        repo_path / "__pycache__" / "skip.py",
        repo_path / "dist" / "skip.js",
        repo_path / "build" / "skip.py",
        repo_path / ".git" / "skip.py",
    ]:
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text("ignored = True\n", encoding="utf-8")

    fake_model = FakeEmbeddingModel()
    fake_client = FakeQdrantClient(collection_exists=False, point_count=0)
    seen_files: list[Path] = []

    def fake_chunk_file(file_path: str | Path, root: str | Path | None = None):
        path = Path(file_path)
        seen_files.append(path)
        return [
            {
                "id": f"{path.as_posix()}::chunk",
                "text": f"chunk:{path.name}",
                "metadata": {
                    "file_path": path.as_posix(),
                    "name": path.stem,
                    "node_type": "function",
                    "start_line": 1,
                    "end_line": 1,
                    "imports": [],
                },
            }
        ]

    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda repo_full_name: repo_path)
    monkeypatch.setattr(indexer, "_get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(indexer, "_load_embedding_model", lambda: fake_model)
    monkeypatch.setattr(indexer.chunker, "chunk_file", fake_chunk_file)

    summary = indexer.index_repo("owner/flask")

    assert summary.collection_name == "owner__flask"
    assert summary.repo_path == repo_path
    assert summary.files_indexed == 205
    assert summary.chunks_indexed == 205
    assert len(seen_files) == 205
    assert all(path.suffix in {".py", ".js"} for path in seen_files)
    assert all("node_modules" not in path.parts and ".venv" not in path.parts and "__pycache__" not in path.parts for path in seen_files)
    assert [len(batch) for batch in fake_model.calls] == [100, 100, 5]
    assert len(fake_client.created) == 1

    collection_name, vectors_config = fake_client.created[0]
    assert collection_name == "owner__flask"
    assert vectors_config.size == indexer.VECTOR_SIZE
    assert vectors_config.distance == indexer.qdrant_models.Distance.COSINE

    assert len(fake_client.upserts) == 1
    upsert_collection, points = fake_client.upserts[0]
    assert upsert_collection == "owner__flask"
    assert len(points) == 205
    assert points[0].payload["text"] == "chunk:file_0.py"
    assert len(points[0].vector) == indexer.VECTOR_SIZE
    # Point id is a deterministic UUID, while the original string id is kept in the payload.
    assert points[0].id == indexer._point_id(points[0].payload["id"])
    assert uuid.UUID(points[0].id)


def test_point_id_is_deterministic_uuid() -> None:
    chunk_id = "app/retrieval/indexer.py::index_repo"

    first = indexer._point_id(chunk_id)
    second = indexer._point_id(chunk_id)

    assert first == second
    assert uuid.UUID(first)  # raises if not a valid UUID
    assert indexer._point_id("other::chunk") != first


def test_index_repo_skips_when_collection_already_has_points(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "flask"
    repo_path.mkdir()

    fake_client = FakeQdrantClient(collection_exists=True, point_count=11)
    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda repo_full_name: repo_path)
    monkeypatch.setattr(indexer, "_get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(indexer, "_load_embedding_model", lambda: fake_model)

    summary = indexer.index_repo("owner/flask")

    assert summary.files_indexed == 0
    assert summary.chunks_indexed == 0
    assert fake_client.created == []
    assert fake_client.upserts == []
    assert fake_model.calls == []


def test_ensure_repo_checkout_clones_when_missing(tmp_path: Path, monkeypatch) -> None:
    # Pin the token off so the clone URL is deterministic regardless of the
    # developer's environment; the authenticated form is covered separately.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    destination = tmp_path / "owner__flask"
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command, check=True, **kwargs):
        calls.append((list(command), check))
        destination.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(indexer, "INDEX_ROOT", tmp_path)
    monkeypatch.setattr(indexer.subprocess, "run", fake_run)

    result = indexer._ensure_repo_checkout("owner/flask")

    assert result == destination
    assert calls == [(["git", "clone", "https://github.com/owner/flask.git", str(destination)], True)]


def test_checkout_dirs_do_not_collide_across_owners(tmp_path: Path, monkeypatch) -> None:
    """Two owners can publish the same repo name; they must not share a tree.

    Qdrant already namespaced these apart, so a shared directory meant retrieval
    read one repo's index while the patch was applied to the other's code.
    """
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    created: list[str] = []

    def fake_run(command, check=True, **kwargs):
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        created.append(destination.name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(indexer, "INDEX_ROOT", tmp_path)
    monkeypatch.setattr(indexer.subprocess, "run", fake_run)

    alice = indexer._ensure_repo_checkout("alice/utils")
    bob = indexer._ensure_repo_checkout("bob/utils")

    assert alice != bob
    assert created == ["alice__utils", "bob__utils"]
    # The directory key must match the Qdrant collection key, or the two drift.
    assert alice.name == indexer._collection_name("alice/utils")


def test_existing_checkout_is_synced_not_returned_stale(tmp_path: Path, monkeypatch) -> None:
    """A stale tree would make the PR revert every upstream commit since cloning."""
    repo_path = tmp_path / "owner__flask"
    (repo_path / ".git").mkdir(parents=True)

    git_calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        git_calls.append(list(command)[1:])
        if "--show-toplevel" in command:
            # git agrees this directory is its own repository root
            return SimpleNamespace(returncode=0, stdout=f"{repo_path.as_posix()}\n", stderr="")
        if "symbolic-ref" in command:
            return SimpleNamespace(returncode=0, stdout="refs/remotes/origin/main\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(indexer, "INDEX_ROOT", tmp_path)
    monkeypatch.setattr(indexer.subprocess, "run", fake_run)

    result = indexer._ensure_repo_checkout("owner/flask")

    assert result == repo_path
    verbs = [call[0] for call in git_calls]
    assert "fetch" in verbs
    assert "reset" in verbs
    assert "clone" not in verbs  # reuse the tree, but only after syncing it


def test_partially_deleted_checkout_is_never_synced(tmp_path: Path, monkeypatch) -> None:
    """A .git husk must not aim fetch/reset/clean at the *enclosing* repository.

    On Windows a failed rmtree leaves .git behind because git marks its objects
    read-only. The directory then looks like a checkout but is not a valid repo,
    so git walks up to the nearest real one. With the index root living inside
    another checkout, syncing would hard-reset and clean a tree that was never
    meant to be touched. This reclones instead.
    """
    repo_path = tmp_path / "owner__flask"
    (repo_path / ".git").mkdir(parents=True)  # husk: exists, but not a repository

    enclosing = tmp_path.parent  # what git would walk up to
    git_calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        git_calls.append(list(command)[1:])
        if "--show-toplevel" in command:
            # git reports the ENCLOSING repo, not our directory
            return SimpleNamespace(returncode=0, stdout=f"{enclosing.as_posix()}\n", stderr="")
        repo_path.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(indexer, "INDEX_ROOT", tmp_path)
    monkeypatch.setattr(indexer.subprocess, "run", fake_run)

    indexer._ensure_repo_checkout("owner/flask")

    verbs = [call[0] for call in git_calls]
    assert "clone" in verbs, "must reclone rather than trust the husk"
    for destructive in ("fetch", "reset", "clean", "checkout"):
        assert destructive not in verbs, f"{destructive} would have hit the enclosing repo"


def test_reindex_is_skipped_only_when_the_sha_still_matches(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "owner__flask"
    repo_path.mkdir(parents=True)

    monkeypatch.setattr(indexer, "INDEX_ROOT", tmp_path)
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: repo_path)
    monkeypatch.setattr(indexer, "_get_qdrant_client", lambda: SimpleNamespace())
    monkeypatch.setattr(indexer, "_collection_has_points", lambda client, name: True)
    monkeypatch.setattr(indexer, "_head_sha", lambda path: "abc123")

    indexer._write_indexed_sha("owner__flask", "abc123")
    summary = indexer.index_repo("owner/flask")
    assert summary.chunks_indexed == 0  # current: skipped

    # The checkout moved on; the collection now describes code that is gone.
    dropped: list[str] = []
    monkeypatch.setattr(indexer, "_head_sha", lambda path: "def456")
    monkeypatch.setattr(indexer, "_drop_collection", lambda c, n: dropped.append(n))
    monkeypatch.setattr(indexer, "_create_collection_if_needed", lambda c, n: None)
    monkeypatch.setattr(indexer, "_iter_source_files", lambda path: iter(()))
    monkeypatch.setattr(indexer, "_embed_chunks", lambda chunks: [])
    monkeypatch.setattr(indexer, "_upsert_chunks", lambda c, n, ch, e: None)

    indexer.index_repo("owner/flask")

    assert dropped == ["owner__flask"]
    assert indexer._read_indexed_sha("owner__flask") == "def456"
