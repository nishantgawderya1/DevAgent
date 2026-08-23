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


class FlakyJSONModeClient:
    """Rejects response_format the way a model without JSON mode does."""

    def __init__(self, content: str, error: Exception) -> None:
        self._content = content
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _bad_request(message: str) -> Exception:
    error = ValueError(message)
    error.status_code = 400  # type: ignore[attr-defined]
    return error


def test_plan_issue_retries_without_json_mode_when_rejected() -> None:
    client = FlakyJSONModeClient(
        json.dumps({"subtasks": ["Locate the bug", "Fix it"]}),
        _bad_request("400: response_format is not supported for this model"),
    )

    update = planner.plan_issue(_state(), client=client)

    assert update["status"] == "exploring"
    assert len(update["plan"]) == 2
    # Tried JSON mode first, then fell back once.
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]
    assert len(client.calls) == 2


def test_plan_issue_does_not_retry_on_an_unrelated_error() -> None:
    """A real outage must surface, not cost a second call and the same failure."""

    class DownClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise RuntimeError("503 upstream connect error")

    client = DownClient()
    update = planner.plan_issue(_state(), client=client)

    assert update["status"] == "failed"
    assert "503" in update["error"]
    assert len(client.calls) == 1


def test_plan_issue_unwraps_a_fenced_reply() -> None:
    fenced = '```json\n{"subtasks": ["Step one", "Step two"]}\n```'

    update = planner.plan_issue(_state(), client=FakeLLMClient(fenced))

    assert update["status"] == "exploring"
    assert [task["description"] for task in update["plan"]] == ["Step one", "Step two"]


def test_plan_issue_recovers_json_from_surrounding_prose() -> None:
    chatty = 'Sure! Here is the plan:\n{"subtasks": ["Only step"]}\nHope that helps.'

    update = planner.plan_issue(_state(), client=FakeLLMClient(chatty))

    assert update["status"] == "exploring"
    assert update["plan"][0]["description"] == "Only step"


def test_plan_issue_still_fails_when_there_is_no_json_at_all() -> None:
    update = planner.plan_issue(_state(), client=FakeLLMClient("I could not understand the issue."))

    assert update["status"] == "failed"
    assert "planning failed" in update["error"].lower()
