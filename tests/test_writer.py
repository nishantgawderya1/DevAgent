from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.agent import state as agent_state
from app.agent.nodes import writer


DIFF = """--- a/api.py
+++ b/api.py
@@ -10,3 +10,3 @@
-    return items[: page_size + 1]
+    return items[:page_size]
"""


class FakeLLMClient:
    """Stand-in for the OpenRouter client; records the prompts it was sent."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _state() -> agent_state.AgentState:
    state = agent_state.create_initial_state(
        "owner/repo", 7, "Fix pagination", "Off-by-one in page size"
    )
    state["plan"] = [agent_state.Subtask(id=1, description="Correct the upper bound")]
    state["retrieved_context"] = [
        {
            "chunk_id": "api.py::paginate",
            "score": 0.5,
            "source": "fused",
            "file_path": "api.py",
            "name": "paginate",
            "node_type": "function",
            "start_line": 10,
            "end_line": 12,
            "text": "def paginate(items, page_size):\n    return items[: page_size + 1]",
        }
    ]
    return state


def test_write_patch_returns_diff_and_target_files() -> None:
    update = writer.write_patch(_state(), client=FakeLLMClient(DIFF))

    assert update["status"] == "testing"
    assert update["diff"].startswith("--- a/api.py")
    assert update["target_files"] == ["api.py"]
    assert update["retry_count"] == 0


def test_write_patch_strips_markdown_fences() -> None:
    fenced = f"Here is the patch:\n\n```diff\n{DIFF.strip()}\n```\n"

    update = writer.write_patch(_state(), client=FakeLLMClient(fenced))

    assert update["diff"].startswith("--- a/api.py")
    assert "```" not in update["diff"]
    assert update["target_files"] == ["api.py"]


def test_write_patch_prompt_includes_plan_and_context() -> None:
    client = FakeLLMClient(DIFF)

    writer.write_patch(_state(), client=client)

    prompt = client.calls[0]["messages"][1]["content"]
    assert "Correct the upper bound" in prompt
    assert "api.py:10-12" in prompt
    assert "def paginate(items, page_size):" in prompt
    # Nothing failed yet, so no repair section.
    assert "Previous attempt failed" not in prompt


def test_write_patch_feeds_failures_back_on_repair() -> None:
    state = _state()
    state["diff"] = DIFF
    state["test_report"] = {
        "passed": False,
        "output": "E   assert 11 == 10",
        "failed_tests": ["tests/test_api.py::test_paginate"],
    }
    client = FakeLLMClient(DIFF)

    update = writer.write_patch(state, client=client)

    prompt = client.calls[0]["messages"][1]["content"]
    assert "Previous attempt failed" in prompt
    assert "tests/test_api.py::test_paginate" in prompt
    assert "assert 11 == 10" in prompt
    # A repair attempt consumes one unit of the retry budget.
    assert update["retry_count"] == 1


def test_write_patch_does_not_consume_budget_when_tests_passed() -> None:
    state = _state()
    state["test_report"] = {"passed": True, "output": "ok", "failed_tests": []}
    state["retry_count"] = 2

    update = writer.write_patch(state, client=FakeLLMClient(DIFF))

    assert update["retry_count"] == 2


def test_write_patch_collects_multiple_target_files() -> None:
    multi = (
        "--- a/api.py\n+++ b/api.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/utils.py\n+++ b/utils.py\n@@ -1 +1 @@\n-a\n+b\n"
    )

    update = writer.write_patch(_state(), client=FakeLLMClient(multi))

    assert update["target_files"] == ["api.py", "utils.py"]


def test_write_patch_fails_when_reply_has_no_diff() -> None:
    update = writer.write_patch(_state(), client=FakeLLMClient("I could not find the bug."))

    assert update["status"] == "failed"
    assert "no usable diff" in update["error"].lower()
