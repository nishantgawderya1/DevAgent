from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.retrieval import indexer


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):
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

    def fake_chunk_file(file_path: str | Path):
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
    destination = tmp_path / "flask"
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command, check):
        calls.append((list(command), check))
        destination.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(indexer, "INDEX_ROOT", tmp_path)
    monkeypatch.setattr(indexer.subprocess, "run", fake_run)

    result = indexer._ensure_repo_checkout("owner/flask")

    assert result == destination
    assert calls == [(["git", "clone", "https://github.com/owner/flask.git", str(destination)], True)]
