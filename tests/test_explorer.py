from __future__ import annotations

from typing import Any

from app.agent import state as agent_state
from app.agent.nodes import explorer
from app.retrieval.retriever import RetrievalResult


def _state() -> agent_state.AgentState:
    state = agent_state.create_initial_state(
        "owner/repo", 7, "Fix pagination", "Off-by-one in page size"
    )
    state["plan"] = [
        agent_state.Subtask(id=1, description="Locate paginate() in the API layer"),
        agent_state.Subtask(id=2, description="Correct the upper bound"),
    ]
    return state


def _result(name: str, source: str = "fused", score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"api.py::{name}",
        score=score,
        source=source,
        payload={
            "id": f"api.py::{name}",
            "text": f"def {name}(): ...",
            "metadata": {
                "file_path": "api.py",
                "name": name,
                "node_type": "function",
                "start_line": 10,
                "end_line": 20,
                "imports": [],
                "calls": [],
            },
        },
    )


def test_explore_flattens_results_into_serialisable_context() -> None:
    def fake_retrieve(query: str, repo: str, top_k: int = 10, **kwargs: Any):
        return [_result("paginate"), _result("page_size", source="expanded", score=0.0)]

    update = explorer.explore_codebase(_state(), retrieve_fn=fake_retrieve)

    assert update["status"] == "writing"
    context = update["retrieved_context"]
    assert [chunk["name"] for chunk in context] == ["paginate", "page_size"]
    assert context[0]["file_path"] == "api.py"
    assert context[0]["start_line"] == 10
    assert context[0]["text"] == "def paginate(): ..."
    assert context[1]["source"] == "expanded"
    # Context must stay plain dicts so the state can be checkpointed / logged.
    assert all(isinstance(chunk, dict) for chunk in context)


def test_explore_query_includes_issue_and_plan() -> None:
    captured: dict[str, Any] = {}

    def fake_retrieve(query: str, repo: str, top_k: int = 10, **kwargs: Any):
        captured["query"] = query
        captured["repo"] = repo
        captured["top_k"] = top_k
        return [_result("paginate")]

    explorer.explore_codebase(_state(), retrieve_fn=fake_retrieve)

    assert "Fix pagination" in captured["query"]
    assert "Off-by-one in page size" in captured["query"]
    # The plan names the symbol the writer will need; it must reach BM25.
    assert "paginate()" in captured["query"]
    assert captured["repo"] == "owner/repo"
    assert captured["top_k"] == explorer.CONTEXT_TOP_K


def test_explore_fails_when_nothing_retrieved() -> None:
    update = explorer.explore_codebase(_state(), retrieve_fn=lambda *a, **k: [])

    assert update["status"] == "failed"
    assert "no relevant code" in update["error"].lower()


def test_explore_fails_without_raising_when_retriever_errors() -> None:
    def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("qdrant unreachable")

    update = explorer.explore_codebase(_state(), retrieve_fn=boom)

    assert update["status"] == "failed"
    assert "qdrant unreachable" in update["error"]
