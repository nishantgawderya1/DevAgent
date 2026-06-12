from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.agent import state as agent_state
from app.agent.nodes import planner


class FakeLLMClient:
    """Minimal stand-in for the OpenAI/OpenRouter client used by ``llm.complete``."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _state() -> agent_state.AgentState:
    return agent_state.create_initial_state("owner/repo", 7, "Fix pagination", "Off-by-one in page size")


def test_plan_issue_parses_subtasks_object() -> None:
    client = FakeLLMClient(json.dumps({"subtasks": ["Locate pagination logic", "Fix the bound", "Add a test"]}))

    update = planner.plan_issue(_state(), client=client)

    assert update["status"] == "exploring"
    assert [task["description"] for task in update["plan"]] == [
        "Locate pagination logic",
        "Fix the bound",
        "Add a test",
    ]
    assert [task["id"] for task in update["plan"]] == [1, 2, 3]


def test_plan_issue_accepts_bare_list() -> None:
    client = FakeLLMClient(json.dumps(["Step one", "Step two"]))

    update = planner.plan_issue(_state(), client=client)

    assert update["status"] == "exploring"
    assert len(update["plan"]) == 2


def test_plan_issue_requests_json_object_format() -> None:
    client = FakeLLMClient(json.dumps({"subtasks": ["only step"]}))

    planner.plan_issue(_state(), client=client)

    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["messages"][0]["role"] == "system"


def test_plan_issue_fails_on_empty_plan() -> None:
    client = FakeLLMClient(json.dumps({"subtasks": []}))

    update = planner.plan_issue(_state(), client=client)

    assert update["status"] == "failed"
    assert "no subtasks" in update["error"].lower()


def test_plan_issue_fails_on_invalid_json() -> None:
    client = FakeLLMClient("not json at all")

    update = planner.plan_issue(_state(), client=client)

    assert update["status"] == "failed"
    assert "planning failed" in update["error"].lower()
