"""Planner node — decomposes a GitHub issue into structured subtasks.

First node in the graph. Reads the issue from state, asks the LLM to break it
into an ordered list of concrete steps (returned as JSON), and writes them back
as ``plan``. On success the run advances to the explorer stage.

Being first makes this node a single point of failure for the whole run, so it
is deliberately tolerant about *how* it gets its JSON. Structured-output support
varies across providers — NVIDIA NIM accepts ``response_format`` for some models
and rejects it for others — and a model that ignores the instruction and wraps
its answer in a markdown fence is still giving us a usable answer. Neither should
kill a run before the codebase has even been looked at.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent import llm
from app.agent.state import AgentState, Subtask, record_failure


logger = logging.getLogger(__name__)

_JSON_OBJECT = {"type": "json_object"}

_SYSTEM_PROMPT = (
    "You are the planning module of an autonomous software engineering agent. "
    "Given a GitHub issue, decompose it into a short ordered list of concrete, "
    "actionable engineering subtasks that will resolve it. Prefer 2-6 steps. "
    'Respond ONLY with JSON of the form {"subtasks": ["step one", "step two"]}.'
)

_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)


def plan_issue(state: AgentState, *, client: Any | None = None) -> AgentState:
    try:
        content = _complete(_build_messages(state), client)
        plan = _parse_plan(content)
        if not plan:
            return record_failure(state, "Planner returned no subtasks.")

        logger.info("Planner produced %d subtasks for issue #%s", len(plan), state.get("issue_number"))
        return AgentState(plan=plan, status="exploring")
    except Exception as error:  # noqa: BLE001 - nodes never raise; they fail the run
        logger.exception("Planning failed for issue #%s", state.get("issue_number"))
        return record_failure(state, f"Planning failed: {error}")


def _complete(messages: list[dict[str, str]], client: Any | None) -> str:
    """Ask for JSON mode, retrying once without it if the provider refuses.

    The prompt already spells out the required shape, so dropping
    ``response_format`` costs reliability rather than capability.
    """
    try:
        return llm.complete(messages, temperature=0.0, response_format=_JSON_OBJECT, client=client)
    except Exception as error:  # noqa: BLE001 - inspected, then re-raised if unrelated
        if not _is_response_format_rejection(error):
            raise
        logger.info("Provider rejected response_format; retrying without JSON mode.")
        return llm.complete(messages, temperature=0.0, client=client)


def _is_response_format_rejection(error: Exception) -> bool:
    """Distinguish 'this model has no JSON mode' from a real outage.

    Retrying a genuinely failing call would just double the cost and the latency
    before reporting the same error.
    """
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    message = str(error).lower()
    mentions_format = "response_format" in message or "response format" in message
    return mentions_format and (status is None or int(status) == 400)


def _build_messages(state: AgentState) -> list[dict[str, str]]:
    user_prompt = (
        f"Repository: {state.get('repo_full_name', '')}\n"
        f"Issue #{state.get('issue_number', '')}: {state.get('issue_title', '')}\n\n"
        f"{state.get('issue_body', '')}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_plan(content: str) -> list[Subtask]:
    data = _load_json(content)

    if isinstance(data, dict):
        raw_subtasks = data.get("subtasks", [])
    elif isinstance(data, list):
        raw_subtasks = data
    else:
        raw_subtasks = []

    plan: list[Subtask] = []
    for index, item in enumerate(raw_subtasks, start=1):
        description = item.get("description") if isinstance(item, dict) else item
        if isinstance(description, str) and description.strip():
            plan.append(Subtask(id=index, description=description.strip()))
    return plan


def _load_json(content: str) -> Any:
    """Parse the model's reply, tolerating fences and surrounding prose.

    Without JSON mode a model will often comply in substance while wrapping the
    answer in ```json or prefacing it with a sentence. That is a formatting
    quirk, not a failed plan, so unwrap it rather than fail the run.
    """
    for candidate in _json_candidates(content):
        try:
            return json.loads(candidate)
        except (TypeError, ValueError):
            continue
    raise ValueError("Planner reply contained no parseable JSON.")


def _json_candidates(content: str) -> list[str]:
    text = (content or "").strip()
    candidates = [text]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    # Last resort: the outermost braces or brackets in the reply.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])

    return candidates
