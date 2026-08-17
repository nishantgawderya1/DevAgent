"""PR Creator node — opens the pull request that closes the issue.

Terminal node on the success path. Like the test runner, this module owns
composition only — branch name, title, body — and delegates the GitHub calls to
an injected client so the node stays testable offline. The PyGithub-backed
implementation is Phase 3 (``app/github/client.py``).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.agent.state import AgentState, record_failure


logger = logging.getLogger(__name__)

BRANCH_PREFIX = "devagent/issue-"


class GitHubClient(Protocol):
    """Pushes ``diff`` to ``branch`` and opens a PR, returning its URL."""

    def create_pull_request(
        self, *, repo_full_name: str, branch: str, title: str, body: str, diff: str
    ) -> str: ...


def create_pull_request(state: AgentState, *, github_client: GitHubClient | Any | None = None) -> AgentState:
    """Open the pull request and mark the run successful."""
    if github_client is None:
        return record_failure(
            state,
            "No GitHub client configured; inject one via create_pull_request(github_client=...) "
            "(the PyGithub client lands in Phase 3).",
        )

    diff = state.get("diff", "")
    if not diff:
        return record_failure(state, "PR creator received an empty diff.")

    try:
        pr_url = github_client.create_pull_request(
            repo_full_name=state.get("repo_full_name", ""),
            branch=_branch_name(state),
            title=_title(state),
            body=_body(state),
            diff=diff,
        )
    except Exception as error:  # noqa: BLE001 - nodes never raise; they fail the run
        logger.exception("PR creation failed for issue #%s", state.get("issue_number"))
        return record_failure(state, f"PR creation failed: {error}")

    if not pr_url:
        return record_failure(state, "GitHub client returned no pull request URL.")

    logger.info("Opened PR for issue #%s: %s", state.get("issue_number"), pr_url)
    return AgentState(pr_url=pr_url, status="success")


def _branch_name(state: AgentState) -> str:
    return f"{BRANCH_PREFIX}{state.get('issue_number', 0)}"


def _title(state: AgentState) -> str:
    return f"Fix #{state.get('issue_number', 0)}: {state.get('issue_title', '')}".strip()


def _body(state: AgentState) -> str:
    """Compose a PR body that shows the work rather than just the result."""
    issue_number = state.get("issue_number", 0)
    sections = [f"Closes #{issue_number}.", ""]

    plan = state.get("plan", [])
    if plan:
        sections.append("### Plan")
        sections.extend(f"{task['id']}. {task['description']}" for task in plan)
        sections.append("")

    target_files = state.get("target_files", [])
    if target_files:
        sections.append("### Files changed")
        sections.extend(f"- `{path}`" for path in target_files)
        sections.append("")

    sections.append("### Tests")
    report = state.get("test_report")
    if report is not None and report.get("passed"):
        sections.append("The repository's existing test suite passes against this patch.")
    else:
        sections.append("Test results unavailable.")

    retry_count = state.get("retry_count", 0)
    if retry_count:
        sections.append(
            f"\nReached after {retry_count} self-repair attempt(s) following an initial test failure."
        )

    sections.extend(["", "---", "_Opened automatically by DevAgent._"])
    return "\n".join(sections)
