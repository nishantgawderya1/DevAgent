"""Test Runner node — executes the patch against the repository's own tests.

Fourth node in the graph, and the one that decides whether the run finishes or
loops. This module owns *orchestration and parsing only*: applying the diff and
executing the suite is delegated to an injected ``runner`` so the execution
backend stays swappable. The sandboxed Docker implementation is Phase 3
(``app/sandbox/docker.py``); until then a runner must be supplied explicitly,
which is deliberate — model-generated patches should not execute unsandboxed by
default.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from app.agent.state import AgentState, TestReport, record_failure, should_retry


logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 8000


class TestRunner(Protocol):
    """Applies ``diff`` to a checkout of ``repo_full_name`` and runs its tests.

    Returns ``(exit_code, output)`` where a zero exit code means the suite
    passed. Implementations are responsible for isolation.
    """

    def __call__(self, repo_full_name: str, diff: str) -> tuple[int, str]: ...


# pytest: "FAILED tests/test_x.py::test_y - AssertionError"
_PYTEST_FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)
# jest: "  ✕ renders the header (4 ms)"
_JEST_FAILED_RE = re.compile(r"^\s*[✕×]\s+(.+?)(?:\s+\(\d+\s*ms\))?$", re.MULTILINE)


def run_tests(state: AgentState, *, runner: TestRunner | Any | None = None) -> AgentState:
    """Run the suite against the generated patch and route the run onward."""
    if runner is None:
        return record_failure(
            state,
            "No test runner configured; inject one via run_tests(runner=...) "
            "(the sandboxed runner lands in Phase 3).",
        )

    diff = state.get("diff", "")
    if not diff:
        return record_failure(state, "Test runner received an empty diff.")

    try:
        exit_code, output = runner(state.get("repo_full_name", ""), diff)
    except Exception as error:  # noqa: BLE001 - nodes never raise; they fail the run
        logger.exception("Test execution failed for issue #%s", state.get("issue_number"))
        return record_failure(state, f"Test execution failed: {error}")

    passed = exit_code == 0
    report = TestReport(
        passed=passed,
        output=_truncate(output),
        failed_tests=[] if passed else _parse_failed_tests(output),
    )

    if passed:
        logger.info("Tests passed for issue #%s", state.get("issue_number"))
        return AgentState(test_report=report, status="creating_pr")

    # Evaluate the retry budget against the report we just produced, since the
    # caller's state does not have it yet.
    if should_retry({**state, "test_report": report}):
        logger.info(
            "Tests failed for issue #%s (%d failing); routing back to the writer.",
            state.get("issue_number"),
            len(report["failed_tests"]),
        )
        return AgentState(test_report=report, status="writing")

    logger.warning("Tests failed for issue #%s and the retry budget is spent.", state.get("issue_number"))
    return AgentState(
        test_report=report,
        status="failed",
        error=f"Tests still failing after {state.get('retry_count', 0)} repair attempt(s).",
    )


def _parse_failed_tests(output: str) -> list[str]:
    """Extract failing test identifiers from pytest or jest output."""
    names: list[str] = []
    for pattern in (_PYTEST_FAILED_RE, _JEST_FAILED_RE):
        for name in pattern.findall(output):
            name = name.strip()
            if name and name not in names:
                names.append(name)
    return names


def _truncate(output: str) -> str:
    """Keep the tail of long output — failures and the summary land at the end."""
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    return "...(truncated)...\n" + output[-MAX_OUTPUT_CHARS:]
