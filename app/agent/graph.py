"""LangGraph state machine wiring the DevAgent nodes together.

Assembles the five nodes into the flow the README describes::

    planner → explorer → writer → tester ─┬─ pass ──→ pr → END
                           ▲              │
                           └── repair ────┘  (self-repair, max 3)

Two routing rules cover the whole graph:

* every node may fail, so each edge is gated on ``status != "failed"``;
* the tester is the only real branch — it has already decided between
  ``creating_pr``, ``writing`` (repair) and ``failed`` (budget spent), so the
  router here just reads the status it set.

External dependencies are injected at build time rather than imported inside
the nodes, which keeps the whole graph runnable offline in tests.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import explorer, planner, pr, tester, writer
from app.agent.state import AgentState, create_initial_state


logger = logging.getLogger(__name__)


def build_graph(
    *,
    llm_client: Any | None = None,
    qdrant_client: Any | None = None,
    retrieve_fn: Callable[..., Any] | None = None,
    test_runner: Any | None = None,
    github_client: Any | None = None,
) -> Any:
    """Build and compile the DevAgent graph with its dependencies bound."""
    builder = StateGraph(AgentState)

    builder.add_node("planner", lambda state: planner.plan_issue(state, client=llm_client))
    builder.add_node(
        "explorer",
        lambda state: explorer.explore_codebase(
            state, retrieve_fn=retrieve_fn, qdrant_client=qdrant_client
        ),
    )
    builder.add_node("writer", lambda state: writer.write_patch(state, client=llm_client))
    builder.add_node("tester", lambda state: tester.run_tests(state, runner=test_runner))
    builder.add_node("pr", lambda state: pr.create_pull_request(state, github_client=github_client))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", _gate("explorer"), {"explorer": "explorer", END: END})
    builder.add_conditional_edges("explorer", _gate("writer"), {"writer": "writer", END: END})
    builder.add_conditional_edges("writer", _gate("tester"), {"tester": "tester", END: END})
    builder.add_conditional_edges(
        "tester", _route_after_tests, {"pr": "pr", "writer": "writer", END: END}
    )
    builder.add_edge("pr", END)

    return builder.compile()


def _gate(next_node: str) -> Callable[[AgentState], str]:
    """Continue to ``next_node`` unless the node just marked the run failed."""

    def route(state: AgentState) -> str:
        if state.get("status") == "failed":
            return END
        return next_node

    return route


def _route_after_tests(state: AgentState) -> str:
    """Read the decision the tester already made."""
    status = state.get("status")
    if status == "creating_pr":
        return "pr"
    if status == "writing":
        return "writer"
    return END


def run(
    repo_full_name: str,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    **dependencies: Any,
) -> AgentState:
    """Convenience entrypoint: build the graph and run one issue through it."""
    graph = build_graph(**dependencies)
    initial_state = create_initial_state(repo_full_name, issue_number, issue_title, issue_body)

    final_state: AgentState = graph.invoke(initial_state)
    logger.info(
        "Run finished for %s#%s with status=%s", repo_full_name, issue_number, final_state.get("status")
    )
    return final_state
