from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.retrieval import indexer, retriever


def _chunk(name: str, text: str, calls: list[str]) -> dict[str, Any]:
    return {
        "id": f"mod.py::{name}",
        "text": text,
        "metadata": {
            "file_path": "mod.py",
            "name": name,
            "node_type": "function",
            "start_line": 1,
            "end_line": 1,
            "imports": [],
            "calls": calls,
        },
    }


class FakeEmbeddingModel:
    def encode(self, texts, batch_size=1, convert_to_numpy=True, show_progress_bar=False):
        return [[0.1] * indexer.VECTOR_SIZE for _ in texts]


class FakeQdrantClient:
    """Fake Qdrant returning a fixed corpus (scroll) and a fixed semantic order (search)."""

    def __init__(self, corpus: list[dict[str, Any]], semantic_order: list[str]) -> None:
        self._corpus = corpus
        self._semantic_order = semantic_order

    def scroll(self, collection_name, limit, offset=None, with_payload=True, with_vectors=False):
        if offset is not None:
            return [], None
        return [SimpleNamespace(payload=payload) for payload in self._corpus], None

    def search(self, collection_name, query_vector, limit, with_payload=True):
        by_id = {payload["id"]: payload for payload in self._corpus}
        return [SimpleNamespace(payload=by_id[chunk_id]) for chunk_id in self._semantic_order[:limit]]


def test_retrieve_fuses_semantic_and_keyword(monkeypatch) -> None:
    corpus = [
        _chunk("login", "def login(user): handle the user login flow", calls=[]),
        _chunk("authenticate", "def authenticate(token): validate the token", calls=[]),
        _chunk("logout", "def logout(): end the session", calls=[]),
    ]
    # Semantic ranks `authenticate` first; keyword (query "login") ranks `login` first.
    client = FakeQdrantClient(corpus, semantic_order=["mod.py::authenticate", "mod.py::login"])
    monkeypatch.setattr(indexer, "_load_embedding_model", lambda: FakeEmbeddingModel())

    results = retriever.retrieve("login", "owner/repo", top_k=5, qdrant_client=client)

    ids = [result.chunk_id for result in results]
    # `login` appears in BOTH rankings, so fusion lifts it above `authenticate`.
    assert ids[0] == "mod.py::login"
    assert "mod.py::authenticate" in ids
    assert all(result.source == "fused" for result in results)


def test_retrieve_expands_dependencies(monkeypatch) -> None:
    corpus = [
        _chunk("login", "def login(user): login", calls=["authenticate"]),
        _chunk("authenticate", "def authenticate(token): token", calls=[]),
    ]
    client = FakeQdrantClient(corpus, semantic_order=["mod.py::login"])
    monkeypatch.setattr(indexer, "_load_embedding_model", lambda: FakeEmbeddingModel())

    # top_k=1 keeps only `login` in the fused set; `authenticate` must arrive via expansion.
    results = retriever.retrieve("login", "owner/repo", top_k=1, qdrant_client=client)

    assert [(r.chunk_id, r.source) for r in results] == [
        ("mod.py::login", "fused"),
        ("mod.py::authenticate", "expanded"),
    ]


def test_retrieve_returns_empty_when_corpus_empty(monkeypatch) -> None:
    client = FakeQdrantClient(corpus=[], semantic_order=[])
    monkeypatch.setattr(indexer, "_load_embedding_model", lambda: FakeEmbeddingModel())

    results = retriever.retrieve("anything", "owner/repo", qdrant_client=client)

    assert results == []


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    fused = retriever._reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]])
    ordering = [chunk_id for chunk_id, _ in fused]

    # `a` is high in both lists; `c` is low in one but top of the other.
    assert ordering[0] == "a"
    assert set(ordering) == {"a", "b", "c"}


def test_tokenize_splits_identifiers() -> None:
    tokens = retriever._tokenize("indexRepo index_repo")

    assert "index" in tokens
    assert "repo" in tokens
    assert "index_repo" in tokens


class ModernFakeQdrantClient:
    """Exposes query_points() only, like qdrant-client >= 1.10.

    The original fake defined search(), which every test used -- so the removal
    of search() from the real client sailed through the suite and only surfaced
    against a live Qdrant. This fake pins the modern path.
    """

    def __init__(self, corpus: list[dict[str, Any]], semantic_order: list[str]) -> None:
        self._corpus = corpus
        self._semantic_order = semantic_order
        self.query_calls: list[dict[str, Any]] = []

    def scroll(self, collection_name, limit, offset=None, with_payload=True, with_vectors=False):
        if offset is not None:
            return [], None
        return [SimpleNamespace(payload=payload) for payload in self._corpus], None

    def query_points(self, collection_name, query, limit, with_payload=True):
        self.query_calls.append({"collection_name": collection_name, "limit": limit})
        by_id = {payload["id"]: payload for payload in self._corpus}
        points = [SimpleNamespace(payload=by_id[cid]) for cid in self._semantic_order[:limit]]
        # The real client wraps hits in a response object rather than returning them.
        return SimpleNamespace(points=points)


def test_retrieve_works_against_the_query_points_api(monkeypatch) -> None:
    # Same corpus as the search()-based fusion test, so this isolates the API
    # difference rather than re-testing ranking behaviour.
    corpus = [
        _chunk("login", "def login(user): handle the user login flow", calls=[]),
        _chunk("authenticate", "def authenticate(token): validate the token", calls=[]),
        _chunk("logout", "def logout(): end the session", calls=[]),
    ]
    client = ModernFakeQdrantClient(corpus, semantic_order=["mod.py::authenticate", "mod.py::login"])
    monkeypatch.setattr(indexer, "_load_embedding_model", lambda: FakeEmbeddingModel())

    results = retriever.retrieve("login", "owner/repo", top_k=5, qdrant_client=client)

    assert [r.chunk_id for r in results][0] == "mod.py::login"
    assert client.query_calls and client.query_calls[0]["limit"] == retriever.SEMANTIC_TOP_N


def test_dependency_expansion_works_on_the_modern_api(monkeypatch) -> None:
    corpus = [
        _chunk("paginate", "def paginate(items, page_size): ...", calls=["clamp_page_size"]),
        _chunk("clamp_page_size", "def clamp_page_size(requested, maximum): ...", calls=[]),
    ]
    client = ModernFakeQdrantClient(corpus, semantic_order=["mod.py::paginate"])
    monkeypatch.setattr(indexer, "_load_embedding_model", lambda: FakeEmbeddingModel())

    results = retriever.retrieve("page size", "owner/repo", top_k=1, qdrant_client=client)

    assert [(r.chunk_id, r.source) for r in results] == [
        ("mod.py::paginate", "fused"),
        ("mod.py::clamp_page_size", "expanded"),
    ]
