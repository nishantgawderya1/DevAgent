"""Explorer node — retrieves the code the writer needs to see.

Second node in the graph. Turns the issue plus the planner's subtasks into a
retrieval query, calls the hybrid retriever, and flattens the hits into
``retrieved_context``. The flattening matters: results are stored as plain
dicts rather than :class:`~app.retrieval.retriever.RetrievalResult` objects so
the state stays JSON-serialisable for checkpointing and MLflow logging.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.agent.state import AgentState, record_failure
from app.retrieval import retriever


logger = logging.getLogger(__name__)

CONTEXT_TOP_K = 10


def explore_codebase(
    state: AgentState,
    *,
    retrieve_fn: Callable[..., Any] | None = None,
    qdrant_client: Any | None = None,
) -> AgentState:
    """Retrieve relevant chunks for the issue and hand them to the writer."""
    try:
        retrieve = retrieve_fn or retriever.retrieve
        results = retrieve(
            _build_query(state),
            state.get("repo_full_name", ""),
            top_k=CONTEXT_TOP_K,
            qdrant_client=qdrant_client,
        )

        context = [_as_context(result) for result in results]
        if not context:
            return record_failure(
                state,
                "Explorer found no relevant code; the repository may not be indexed yet.",
            )

        logger.info(
            "Explorer retrieved %d chunks for issue #%s", len(context), state.get("issue_number")
        )
        return AgentState(retrieved_context=context, status="writing")
    except Exception as error:  # noqa: BLE001 - nodes never raise; they fail the run
        logger.exception("Exploration failed for issue #%s", state.get("issue_number"))
        return record_failure(state, f"Exploration failed: {error}")


def _build_query(state: AgentState) -> str:
    """Combine the issue and the plan into a single retrieval query.

    The plan's subtasks usually name the concrete symbols the writer will need,
    so folding them in measurably widens what BM25 can match on.
    """
    parts = [state.get("issue_title", ""), state.get("issue_body", "")]
    parts.extend(subtask["description"] for subtask in state.get("plan", []))
    return "\n".join(part for part in parts if part)


def _as_context(result: Any) -> dict[str, Any]:
    payload = result.payload
    metadata = payload.get("metadata", {})
    return {
        "chunk_id": result.chunk_id,
        "score": result.score,
        "source": result.source,
        "file_path": metadata.get("file_path", ""),
        "name": metadata.get("name", ""),
        "node_type": metadata.get("node_type", ""),
        "start_line": metadata.get("start_line", 0),
        "end_line": metadata.get("end_line", 0),
        "text": payload.get("text", ""),
    }
