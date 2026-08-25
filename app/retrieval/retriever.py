from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from app.retrieval import indexer


logger = logging.getLogger(__name__)

SEMANTIC_TOP_N = 20  # candidates pulled from vector search
BM25_TOP_N = 20  # candidates pulled from keyword search
DEFAULT_TOP_K = 10  # fused results returned to the caller
RRF_K = 60  # reciprocal-rank-fusion damping constant

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SUBTOKEN_RE = re.compile(r"_|(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    source: str  # "fused" (semantic+keyword) or "expanded" (dependency)
    payload: dict[str, Any]


def retrieve(
    query: str,
    repo_full_name: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    qdrant_client: Any | None = None,
) -> list[RetrievalResult]:
    """Hybrid retrieval over an indexed repo.

    Runs semantic (vector) and keyword (BM25) search, fuses the two rankings
    with reciprocal rank fusion, then expands the top results with their direct
    callee chunks so the caller gets self-contained, AST-aware context.
    """
    try:
        collection_name = indexer._collection_name(repo_full_name)
        client = qdrant_client or indexer._get_qdrant_client()

        corpus = _load_corpus(client, collection_name)
        if not corpus:
            logger.info("No indexed chunks found for %s; returning no results.", repo_full_name)
            return []

        by_id = {payload["id"]: payload for payload in corpus}

        semantic_ranking = _semantic_search(client, collection_name, query)
        keyword_ranking = _keyword_search(query, corpus)
        fused = _reciprocal_rank_fusion([semantic_ranking, keyword_ranking])

        results = [
            RetrievalResult(chunk_id=chunk_id, score=score, source="fused", payload=by_id[chunk_id])
            for chunk_id, score in fused[:top_k]
            if chunk_id in by_id
        ]
        return results + _expand_dependencies(results, by_id)
    except Exception:
        logger.exception("Retrieval failed for query %r on %s", query, repo_full_name)
        return []


def _load_corpus(client: Any, collection_name: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(point.payload for point in points if point.payload)
        if offset is None:
            break
    return payloads


def _semantic_search(client: Any, collection_name: str, query: str) -> list[str]:
    model = indexer._load_embedding_model()
    vector = _embed_query(model, query)
    hits = _vector_search(client, collection_name, vector)
    return [hit.payload["id"] for hit in hits if hit.payload]


def _vector_search(client: Any, collection_name: str, vector: list[float]) -> Any:
    """Run the vector query across qdrant-client API generations.

    ``search()`` was superseded by ``query_points()``, which returns a response
    object wrapping the hits rather than the hits themselves. requirements.txt
    allows ``qdrant-client>=1.7``, which spans both, and the installed 1.18 has
    dropped ``search()`` entirely -- caught only against a live Qdrant, because
    the test fake defined the method the real client no longer has.
    """
    query_points = getattr(client, "query_points", None)
    if callable(query_points):
        response = query_points(
            collection_name=collection_name,
            query=vector,
            limit=SEMANTIC_TOP_N,
            with_payload=True,
        )
        return getattr(response, "points", response)

    return client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=SEMANTIC_TOP_N,
        with_payload=True,
    )


def _embed_query(model: Any, query: str) -> list[float]:
    embeddings = model.encode([query], batch_size=1, convert_to_numpy=True, show_progress_bar=False)
    return indexer._normalize_embeddings(embeddings)[0]


def _keyword_search(query: str, corpus: Sequence[dict[str, Any]]) -> list[str]:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(payload["text"]) for payload in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
    return [payload["id"] for payload, score in ranked[:BM25_TOP_N] if score > 0]


def _reciprocal_rank_fusion(rankings: Sequence[Sequence[str]]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _expand_dependencies(
    results: Sequence[RetrievalResult],
    by_id: dict[str, dict[str, Any]],
) -> list[RetrievalResult]:
    name_index: dict[str, dict[str, Any]] = {}
    for payload in by_id.values():
        name_index.setdefault(payload["metadata"]["name"], payload)

    seen = {result.chunk_id for result in results}
    expanded: list[RetrievalResult] = []
    for result in results:
        for callee in result.payload["metadata"].get("calls", []):
            dependency = name_index.get(callee)
            if dependency is None or dependency["id"] in seen:
                continue
            seen.add(dependency["id"])
            expanded.append(
                RetrievalResult(chunk_id=dependency["id"], score=0.0, source="expanded", payload=dependency)
            )
    return expanded


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lowered = raw.lower()
        tokens.append(lowered)
        # split snake_case / camelCase identifiers into sub-tokens for recall
        for part in _SUBTOKEN_RE.split(raw):
            part = part.lower()
            if part and part != lowered:
                tokens.append(part)
    return tokens
