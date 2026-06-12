from __future__ import annotations

from app.agent import state as agent_state


def test_create_initial_state_populates_defaults() -> None:
    result = agent_state.create_initial_state("owner/repo", 42, "Fix bug", "Steps to reproduce")

    assert result["repo_full_name"] == "owner/repo"
    assert result["issue_number"] == 42
    assert result["issue_title"] == "Fix bug"
    assert result["issue_body"] == "Steps to reproduce"
    assert result["status"] == "pending"
    assert result["plan"] == []
    assert result["retrieved_context"] == []
    assert result["diff"] == ""
    assert result["target_files"] == []
    assert result["test_report"] is None
    assert result["pr_url"] == ""
    assert result["retry_count"] == 0
    assert result["max_retries"] == agent_state.MAX_RETRIES
    assert result["error"] == ""


def test_create_initial_state_accepts_custom_max_retries() -> None:
    result = agent_state.create_initial_state("o/r", 1, "t", "b", max_retries=5)

    assert result["max_retries"] == 5


def test_should_retry_false_when_no_report() -> None:
    state = agent_state.create_initial_state("o/r", 1, "t", "b")

    assert agent_state.should_retry(state) is False


def test_should_retry_false_when_tests_pass() -> None:
    state = agent_state.create_initial_state("o/r", 1, "t", "b")
    state["test_report"] = {"passed": True, "output": "ok", "failed_tests": []}

    assert agent_state.should_retry(state) is False


def test_should_retry_true_when_failing_under_budget() -> None:
    state = agent_state.create_initial_state("o/r", 1, "t", "b")
    state["test_report"] = {"passed": False, "output": "boom", "failed_tests": ["test_x"]}
    state["retry_count"] = 1

    assert agent_state.should_retry(state) is True


def test_should_retry_false_when_budget_exhausted() -> None:
    state = agent_state.create_initial_state("o/r", 1, "t", "b")
    state["test_report"] = {"passed": False, "output": "boom", "failed_tests": ["test_x"]}
    state["retry_count"] = agent_state.MAX_RETRIES

    assert agent_state.should_retry(state) is False


def test_record_failure_marks_failed() -> None:
    update = agent_state.record_failure(agent_state.create_initial_state("o/r", 1, "t", "b"), "clone failed")

    assert update["status"] == "failed"
    assert update["error"] == "clone failed"
