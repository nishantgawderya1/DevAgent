"""Code Writer node — turns the plan and retrieved context into a unified diff.

Third node in the graph, and the only one that runs more than once: when the
test runner reports a failure, the graph routes back here with the failing
output folded into the prompt. That return path is the self-repair loop, and it
is why this node owns incrementing ``retry_count``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.agent import llm
from app.agent.state import AgentState, record_failure


logger = logging.getLogger(__name__)

MAX_CONTEXT_CHUNKS = 12

_SYSTEM_PROMPT = (
    "You are the code-writing module of an autonomous software engineering agent. "
    "You are given a GitHub issue, a plan, and the relevant source code. "
    "Produce a patch that resolves the issue.\n\n"
    "Respond ONLY with a unified diff. No prose, no explanation, no markdown fences. "
    "Use paths relative to the repository root, in the form '--- a/path' and '+++ b/path'. "
    "Include at least 3 lines of context around each hunk. "
    "Only modify files you were shown."
)

# `+++ b/path/to/file.py` (optionally followed by a tab-separated timestamp).
_TARGET_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(\S+)", re.MULTILINE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n```", re.DOTALL | re.MULTILINE)
_DIFF_START_RE = re.compile(r"^(diff --git |--- )", re.MULTILINE)


def write_patch(state: AgentState, *, client: Any | None = None) -> AgentState:
    """Generate a unified diff for the issue, or repair the previous attempt."""
    try:
        content = llm.complete(_build_messages(state), temperature=0.0, client=client)
        diff = _extract_diff(content)
        if not diff:
            return record_failure(state, "Code writer returned no usable diff.")

        target_files = _target_files(diff)
        if not target_files:
            return record_failure(state, "Code writer produced a diff with no target files.")

        retry_count = _next_retry_count(state)
        logger.info(
            "Writer produced a patch touching %d file(s) for issue #%s (attempt %d)",
            len(target_files),
            state.get("issue_number"),
            retry_count + 1,
        )
        return AgentState(
            diff=diff,
            target_files=target_files,
            status="testing",
            retry_count=retry_count,
        )
    except Exception as error:  # noqa: BLE001 - nodes never raise; they fail the run
        logger.exception("Patch generation failed for issue #%s", state.get("issue_number"))
        return record_failure(state, f"Patch generation failed: {error}")


def _build_messages(state: AgentState) -> list[dict[str, str]]:
    sections = [
        f"Repository: {state.get('repo_full_name', '')}",
        f"Issue #{state.get('issue_number', '')}: {state.get('issue_title', '')}",
        "",
        state.get("issue_body", ""),
        "",
        "## Plan",
        _format_plan(state),
        "",
        "## Relevant code",
        _format_context(state),
    ]

    repair = _format_repair(state)
    if repair:
        sections.extend(["", "## Previous attempt failed", repair])

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(sections)},
    ]


def _format_plan(state: AgentState) -> str:
    plan = state.get("plan", [])
    if not plan:
        return "(no plan available)"
    return "\n".join(f"{task['id']}. {task['description']}" for task in plan)


def _format_context(state: AgentState) -> str:
    chunks = state.get("retrieved_context", [])[:MAX_CONTEXT_CHUNKS]
    if not chunks:
        return "(no context retrieved)"

    blocks = []
    for chunk in chunks:
        header = (
            f"### {chunk.get('file_path', '')}:"
            f"{chunk.get('start_line', 0)}-{chunk.get('end_line', 0)} "
            f"({chunk.get('node_type', '')} {chunk.get('name', '')})"
        )
        blocks.append(f"{header}\n```\n{chunk.get('text', '')}\n```")
    return "\n\n".join(blocks)


def _format_repair(state: AgentState) -> str:
    """Render the failing test output so the model can repair its own patch."""
    report = state.get("test_report")
    if report is None or report.get("passed"):
        return ""

    lines = ["Your previous patch was applied but the tests failed."]
    failed = report.get("failed_tests", [])
    if failed:
        lines.append("Failing tests:")
        lines.extend(f"- {name}" for name in failed)

    previous = state.get("diff", "")
    if previous:
        lines.extend(["", "Your previous diff:", "```", previous, "```"])

    lines.extend(["", "Test output:", "```", report.get("output", ""), "```"])
    lines.append("\nProduce a corrected unified diff that fixes these failures.")
    return "\n".join(lines)


def _extract_diff(content: str) -> str:
    """Pull the diff out of the model's reply.

    The prompt forbids markdown fences, but models emit them anyway, so unwrap a
    fenced block when present and otherwise trim any preamble before the first
    diff marker.
    """
    fenced = _FENCE_RE.search(content)
    if fenced:
        content = fenced.group(1)

    start = _DIFF_START_RE.search(content)
    if start is None:
        return ""
    return content[start.start() :].strip()


def _target_files(diff: str) -> list[str]:
    """Collect the files a diff writes to, preserving order and dropping /dev/null."""
    files: list[str] = []
    for path in _TARGET_FILE_RE.findall(diff):
        if path != "/dev/null" and path not in files:
            files.append(path)
    return files


def _next_retry_count(state: AgentState) -> int:
    """Count a repair attempt only when re-entered from a failing test run."""
    report = state.get("test_report")
    current = state.get("retry_count", 0)
    if report is not None and not report.get("passed"):
        return current + 1
    return current
