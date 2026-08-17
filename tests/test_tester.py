from __future__ import annotations

from app.agent import state as agent_state
from app.agent.nodes import tester


PYTEST_FAILURE = """\
=================================== FAILURES ===================================
E   assert 11 == 10
=========================== short test summary info ============================
FAILED tests/test_api.py::test_paginate - assert 11 == 10
FAILED tests/test_api.py::test_bounds - IndexError
1 failed, 4 passed in 0.11s
"""


def _state(**overrides) -> agent_state.AgentState:
    state = agent_state.create_initial_state("owner/repo", 7, "Fix pagination", "Off-by-one")
    state["diff"] = "--- a/api.py\n+++ b/api.py\n"
    state.update(overrides)
    return state


def test_run_tests_routes_to_pr_when_suite_passes() -> None:
    update = tester.run_tests(_state(), runner=lambda repo, diff: (0, "5 passed in 0.10s"))

    assert update["status"] == "creating_pr"
    assert update["test_report"]["passed"] is True
    assert update["test_report"]["failed_tests"] == []


def test_run_tests_routes_back_to_writer_when_under_budget() -> None:
    update = tester.run_tests(_state(), runner=lambda repo, diff: (1, PYTEST_FAILURE))

    assert update["status"] == "writing"
    report = update["test_report"]
    assert report["passed"] is False
    assert report["failed_tests"] == [
        "tests/test_api.py::test_paginate",
        "tests/test_api.py::test_bounds",
    ]


def test_run_tests_fails_run_when_budget_exhausted() -> None:
    state = _state(retry_count=agent_state.MAX_RETRIES)

    update = tester.run_tests(state, runner=lambda repo, diff: (1, PYTEST_FAILURE))

    assert update["status"] == "failed"
    assert "still failing" in update["error"].lower()
    # The report is still recorded so the dashboard can show the last failure.
    assert update["test_report"]["passed"] is False


def test_run_tests_parses_jest_failures() -> None:
    jest_output = "  ✓ adds numbers (2 ms)\n  ✕ renders the header (4 ms)\n  ✕ paginates\n"

    update = tester.run_tests(_state(), runner=lambda repo, diff: (1, jest_output))

    assert update["test_report"]["failed_tests"] == ["renders the header", "paginates"]


def test_run_tests_truncates_long_output_keeping_the_tail() -> None:
    output = "noise\n" * 5000 + "FAILED tests/test_api.py::test_paginate - boom"

    update = tester.run_tests(_state(), runner=lambda repo, diff: (1, output))

    recorded = update["test_report"]["output"]
    assert len(recorded) < len(output)
    assert recorded.startswith("...(truncated)...")
    # The summary lives at the end, so the tail is the part worth keeping.
    assert recorded.endswith("boom")


def test_run_tests_requires_an_injected_runner() -> None:
    update = tester.run_tests(_state())

    assert update["status"] == "failed"
    assert "no test runner configured" in update["error"].lower()


def test_run_tests_fails_without_raising_when_runner_errors() -> None:
    def boom(repo: str, diff: str):
        raise RuntimeError("docker daemon not running")

    update = tester.run_tests(_state(), runner=boom)

    assert update["status"] == "failed"
    assert "docker daemon not running" in update["error"]


def test_run_tests_rejects_empty_diff() -> None:
    update = tester.run_tests(_state(diff=""), runner=lambda repo, diff: (0, "ok"))

    assert update["status"] == "failed"
    assert "empty diff" in update["error"].lower()
