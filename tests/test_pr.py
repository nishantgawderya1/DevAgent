from __future__ import annotations

from typing import Any

from app.agent import state as agent_state
from app.agent.nodes import pr


class FakeGitHubClient:
    def __init__(self, url: str = "https://github.com/owner/repo/pull/12") -> None:
        self._url = url
        self.calls: list[dict[str, Any]] = []

    def create_pull_request(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self._url


def _state(**overrides) -> agent_state.AgentState:
    state = agent_state.create_initial_state("owner/repo", 7, "Fix pagination", "Off-by-one")
    state["plan"] = [agent_state.Subtask(id=1, description="Correct the upper bound")]
    state["diff"] = "--- a/api.py\n+++ b/api.py\n"
    state["target_files"] = ["api.py"]
    state["test_report"] = {"passed": True, "output": "5 passed", "failed_tests": []}
    state.update(overrides)
    return state


def test_create_pull_request_returns_url_and_succeeds() -> None:
    client = FakeGitHubClient()

    update = pr.create_pull_request(_state(), github_client=client)

    assert update["status"] == "success"
    assert update["pr_url"] == "https://github.com/owner/repo/pull/12"


def test_create_pull_request_composes_branch_and_title() -> None:
    client = FakeGitHubClient()

    pr.create_pull_request(_state(), github_client=client)

    call = client.calls[0]
    assert call["repo_full_name"] == "owner/repo"
    assert call["branch"] == "devagent/issue-7"
    assert call["title"] == "Fix #7: Fix pagination"
    assert call["diff"].startswith("--- a/api.py")


def test_pr_body_shows_the_work() -> None:
    client = FakeGitHubClient()

    pr.create_pull_request(_state(), github_client=client)

    body = client.calls[0]["body"]
    assert "Closes #7." in body
    assert "Correct the upper bound" in body
    assert "`api.py`" in body
    assert "test suite passes" in body
    # No repairs happened, so the retry note is omitted.
    assert "self-repair" not in body


def test_pr_body_notes_self_repair_attempts() -> None:
    client = FakeGitHubClient()

    pr.create_pull_request(_state(retry_count=2), github_client=client)

    assert "2 self-repair attempt(s)" in client.calls[0]["body"]


def test_create_pull_request_requires_a_client() -> None:
    update = pr.create_pull_request(_state())

    assert update["status"] == "failed"
    assert "no github client configured" in update["error"].lower()


def test_create_pull_request_fails_without_raising_on_api_error() -> None:
    class BoomClient:
        def create_pull_request(self, **kwargs: Any) -> str:
            raise RuntimeError("403 Forbidden")

    update = pr.create_pull_request(_state(), github_client=BoomClient())

    assert update["status"] == "failed"
    assert "403 Forbidden" in update["error"]
