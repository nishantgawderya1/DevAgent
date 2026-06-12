"""Planner node — decomposes a GitHub issue into structured subtasks.

First node in the graph. Reads the issue from state, asks the LLM to break it
into an ordered list of concrete steps (returned as JSON), and writes them back
as ``plan``. On success the run advances to the explorer stage.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent import llm
from app.agent.state import AgentState, Subtask, record_failure


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the planning module of an autonomous software engineering agent. "
    "Given a GitHub issue, decompose it into a short ordered list of concrete, "
    "actionable engineering subtasks that will resolve it. Prefer 2-6 steps. "
    'Respond ONLY with JSON of the form {"subtasks": ["step one", "step two"]}.'
)


def plan_issue(state: AgentState, *, client: Any | None = None) -> AgentState:
    try:
        content = llm.complete(
            _build_messages(state),
            temperature=0.0,
            response_format={"type": "json_object"},
            client=client,
        )
        plan = _parse_plan(content)
        if not plan:
            return record_failure(state, "Planner returned no subtasks.")

        logger.info("Planner produced %d subtasks for issue #%s", len(plan), state.get("issue_number"))
        return AgentState(plan=plan, status="exploring")
    except Exception as error:  # noqa: BLE001 - nodes never raise; they fail the run
        logger.exception("Planning failed for issue #%s", state.get("issue_number"))
        return record_failure(state, f"Planning failed: {error}")


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
    data = json.loads(content)

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
