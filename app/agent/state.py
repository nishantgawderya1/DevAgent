"""Typed state schema for the DevAgent LangGraph state machine.

Every node receives an :class:`AgentState`, reads the fields produced by earlier
nodes, and returns a partial update. The state is a plain ``TypedDict`` (no
LangGraph import here) so it stays dependency-light and trivially serialisable
for checkpointing / MLflow logging.

Flow::

    pending → planning → exploring → writing → testing ─┬─ success → creating_pr
                                          ▲              │
                                          └── writing ◀──┘  (self-repair, max 3)
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


MAX_RETRIES = 3

RunStatus = Literal[
    "pending",
    "planning",
    "exploring",
    "writing",
    "testing",
    "creating_pr",
    "success",
    "failed",
]


class Subtask(TypedDict):
    """A single structured step the planner decomposes the issue into."""

    id: int
    description: str


class TestReport(TypedDict):
    """Parsed result of one sandboxed test run."""

    passed: bool
    output: str
    failed_tests: list[str]


class AgentState(TypedDict, total=False):
    """State threaded through every node of the graph.

    ``total=False`` so nodes can return partial updates; use
    :func:`create_initial_state` to build a fully-populated starting state.
    """

    # --- Input (set when the run is created) ---
    repo_full_name: str
    issue_number: int
    issue_title: str
    issue_body: str

    # --- Planner ---
    plan: list[Subtask]

    # --- Explorer ---
    retrieved_context: list[dict[str, Any]]

    # --- Code Writer ---
    diff: str
    target_files: list[str]

    # --- Test Runner ---
    test_report: TestReport | None

    # --- PR Creator ---
    pr_url: str

    # --- Control / bookkeeping ---
    status: RunStatus
    retry_count: int
    max_retries: int
    error: str


def create_initial_state(
    repo_full_name: str,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    *,
    max_retries: int = MAX_RETRIES,
) -> AgentState:
    """Build a fully-populated starting state for a new run."""
    return AgentState(
        repo_full_name=repo_full_name,
        issue_number=issue_number,
        issue_title=issue_title,
        issue_body=issue_body,
        plan=[],
        retrieved_context=[],
        diff="",
        target_files=[],
        test_report=None,
        pr_url="",
        status="pending",
        retry_count=0,
        max_retries=max_retries,
        error="",
    )


def should_retry(state: AgentState) -> bool:
    """Self-repair loop condition: retry only on a failing run still under budget."""
    report = state.get("test_report")
    if report is None or report["passed"]:
        return False
    return state.get("retry_count", 0) < state.get("max_retries", MAX_RETRIES)


def record_failure(state: AgentState, message: str) -> AgentState:
    """Return a partial update marking the run as failed with an error message."""
    return AgentState(status="failed", error=message)
