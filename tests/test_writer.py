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


def test_malformed_hunk_header_is_repaired() -> None:
    """Verbatim from a live run: the +range was missing entirely.

    git rejects this as a corrupt patch before it looks at the change, and
    `git apply --recount` cannot help because it still needs a parseable header.
    The model had the edit exactly right; only the arithmetic was wrong.
    """
    reply = (
        "--- a/pagelib/utils.py\n"
        "+++ b/pagelib/utils.py\n"
        "@@ -7,7 @@ def clamp_page_size(requested, maximum):\n"
        '     """Clamp a requested page size into the allowed range."""\n'
        "     if requested < 1:\n"
        "         return 1\n"
        "-    return min(requested, maximum + 1)\n"
        "+    return min(requested, maximum)\n"
    )

    diff = writer._extract_diff(reply)

    # 3 context + 1 removed = 4 old; 3 context + 1 added = 4 new.
    assert "@@ -7,4 +7,4 @@" in diff
    assert "-    return min(requested, maximum + 1)" in diff
    assert "+    return min(requested, maximum)" in diff


def test_correct_hunk_headers_are_left_alone() -> None:
    """An already-correct header must survive normalisation untouched."""
    reply = (
        "--- a/api.py\n"
        "+++ b/api.py\n"
        "@@ -10,3 +10,3 @@\n"
        " before\n"
        "-old\n"
        "+new\n"
        " after\n"
    )

    # 2 context + 1 removed = 3 old; 2 context + 1 added = 3 new. Already right.
    assert "@@ -10,3 +10,3 @@" in writer._extract_diff(reply)


def test_wrong_counts_are_recomputed_from_the_body() -> None:
    reply = (
        "--- a/api.py\n"
        "+++ b/api.py\n"
        "@@ -1,99 +1,99 @@\n"
        " keep\n"
        "-drop\n"
        "+add\n"
        "+add two\n"
    )

    # Body is the source of truth: 2 old (keep+drop), 3 new (keep+2 adds).
    assert "@@ -1,2 +1,3 @@" in writer._extract_diff(reply)


def test_multiple_hunks_are_each_recomputed() -> None:
    reply = (
        "--- a/api.py\n"
        "+++ b/api.py\n"
        "@@ -1 +1 @@\n"
        " a\n"
        "-b\n"
        "+c\n"
        "@@ -20 +20 @@\n"
        " x\n"
        " y\n"
        "-z\n"
        "+w\n"
    )

    diff = writer._extract_diff(reply)

    assert "@@ -1,2 +1,2 @@" in diff
    assert "@@ -20,3 +20,3 @@" in diff
