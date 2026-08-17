from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.agent import graph as agent_graph
from app.agent import state as agent_state
from app.retrieval.retriever import RetrievalResult


PLAN = json.dumps({"subtasks": ["Locate paginate()", "Correct the upper bound"]})
DIFF = "--- a/api.py\n+++ b/api.py\n@@ -1 +1 @@\n-old\n+new\n"


class ScriptedLLMClient:
    """Returns queued responses in order, repeating the last one once drained."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeGitHubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_pull_request(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "https://github.com/owner/repo/pull/12"


def _fake_retrieve(query: str, repo: str, top_k: int = 10, **kwargs: Any):
    return [
        RetrievalResult(
            chunk_id="api.py::paginate",
            score=0.9,
            source="fused",
            payload={
                "id": "api.py::paginate",
                "text": "def paginate(items, page_size): ...",
                "metadata": {
                    "file_path": "api.py",
                    "name": "paginate",
                    "node_type": "function",
                    "start_line": 10,
                    "end_line": 12,
                    "imports": [],
                    "calls": [],
                },
            },
        )
    ]


def _run(llm_client: Any, runner: Any, github_client: Any = None) -> agent_state.AgentState:
    return agent_graph.run(
        "owner/repo",
        7,
        "Fix pagination",
        "Off-by-one in page size",
        llm_client=llm_client,
        retrieve_fn=_fake_retrieve,
        test_runner=runner,
        github_client=github_client if github_client is not None else FakeGitHubClient(),
    )


def test_graph_runs_issue_to_pr_on_the_happy_path() -> None:
    github = FakeGitHubClient()

    final = _run(
        ScriptedLLMClient(PLAN, DIFF),
        runner=lambda repo, diff: (0, "5 passed in 0.10s"),
        github_client=github,
    )

    assert final["status"] == "success"
    assert final["pr_url"] == "https://github.com/owner/repo/pull/12"
    assert [task["description"] for task in final["plan"]] == [
        "Locate paginate()",
        "Correct the upper bound",
    ]
    assert final["retrieved_context"][0]["name"] == "paginate"
    assert final["target_files"] == ["api.py"]
    assert final["test_report"]["passed"] is True
    assert final["retry_count"] == 0
    assert len(github.calls) == 1


def test_graph_self_repairs_after_a_failing_test_run() -> None:
    attempts = {"count": 0}

    def flaky_runner(repo: str, diff: str) -> tuple[int, str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return 1, "FAILED tests/test_api.py::test_paginate - assert 11 == 10"
        return 0, "5 passed in 0.10s"

    client = ScriptedLLMClient(PLAN, DIFF)
    final = _run(client, runner=flaky_runner)

    assert final["status"] == "success"
    # Writer ran twice: the initial patch, then the repair.
    assert attempts["count"] == 2
    assert final["retry_count"] == 1
    assert final["test_report"]["passed"] is True

    # The repair prompt must carry the failure back to the model.
    repair_prompt = client.calls[-1]["messages"][1]["content"]
    assert "Previous attempt failed" in repair_prompt
    assert "tests/test_api.py::test_paginate" in repair_prompt


def test_graph_gives_up_after_exhausting_the_retry_budget() -> None:
    attempts = {"count": 0}

    def always_failing(repo: str, diff: str) -> tuple[int, str]:
        attempts["count"] += 1
        return 1, "FAILED tests/test_api.py::test_paginate - assert 11 == 10"

    final = _run(ScriptedLLMClient(PLAN, DIFF), runner=always_failing)

    assert final["status"] == "failed"
    assert "still failing" in final["error"].lower()
    # Initial attempt plus MAX_RETRIES repairs, then the loop stops.
    assert attempts["count"] == agent_state.MAX_RETRIES + 1
    assert final["retry_count"] == agent_state.MAX_RETRIES
    assert final["pr_url"] == ""


def test_graph_short_circuits_when_the_planner_fails() -> None:
    github = FakeGitHubClient()

    def unused_runner(repo: str, diff: str) -> tuple[int, str]:
        raise AssertionError("test runner should not be reached")

    final = _run(
        ScriptedLLMClient("not json at all"), runner=unused_runner, github_client=github
    )

    assert final["status"] == "failed"
    assert "planning failed" in final["error"].lower()
    assert final["retrieved_context"] == []
    assert final["diff"] == ""
    assert github.calls == []


def test_graph_short_circuits_when_retrieval_finds_nothing() -> None:
    github = FakeGitHubClient()

    final = agent_graph.run(
        "owner/repo",
        7,
        "Fix pagination",
        "Off-by-one",
        llm_client=ScriptedLLMClient(PLAN, DIFF),
        retrieve_fn=lambda *a, **k: [],
        test_runner=lambda repo, diff: (0, "ok"),
        github_client=github,
    )

    assert final["status"] == "failed"
    assert "no relevant code" in final["error"].lower()
    assert final["diff"] == ""
    assert github.calls == []


def test_build_graph_compiles_without_dependencies() -> None:
    # The graph must be constructible without any backend wired up, so the
    # FastAPI entrypoint (Phase 3) can build it at import time.
    assert agent_graph.build_graph() is not None
