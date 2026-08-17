"""FastAPI entrypoint — receives issues and runs them through the agent.

Two ways in: a GitHub App webhook (``/webhook``, fired on ``issues.opened``) and
a manual trigger (``/webhook/manual``) for testing without a webhook round-trip.
Both do the same thing — schedule a background run and return immediately, since
a full run takes minutes and GitHub expects a webhook response in seconds.

Each run indexes the repository before invoking the graph. ``index_repo`` is
idempotent and skips collections that already have points, so the cost is paid
once per repository rather than once per issue — but it has to happen, because
the explorer node has nothing to retrieve from otherwise.

Run state is kept in memory. That is deliberate for now: it is enough for the
Phase 5 dashboard to read, and swapping it for a real store is a contained
change behind :func:`get_run` / :func:`list_runs`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent import graph
from app.agent.state import AgentState
from app.github.client import GitHubClient
from app.retrieval import indexer
from app.sandbox.docker import DockerTestRunner


load_dotenv()

logging.basicConfig(level=os.getenv("DEVAGENT_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="DevAgent", description="Autonomous GitHub issue resolver")

_runs: dict[str, dict[str, Any]] = {}


class ManualRequest(BaseModel):
    repo: str = Field(..., description="Repository in owner/name form")
    issue_number: int = Field(..., ge=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    return sorted(_runs.values(), key=lambda run: run["started_at"], reverse=True)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
) -> dict[str, Any]:
    """Handle a GitHub App webhook delivery."""
    raw_body = await request.body()
    _verify_signature(raw_body, x_hub_signature_256)

    if x_github_event != "issues":
        return {"status": "ignored", "reason": f"event {x_github_event} is not handled"}

    payload = json.loads(raw_body or b"{}")
    if payload.get("action") != "opened":
        return {"status": "ignored", "reason": f"action {payload.get('action')} is not handled"}

    issue = payload.get("issue", {})
    repo_full_name = payload.get("repository", {}).get("full_name", "")
    if not repo_full_name or "number" not in issue:
        raise HTTPException(status_code=400, detail="Payload missing repository or issue")

    run_id = _schedule(
        background_tasks,
        repo_full_name=repo_full_name,
        issue_number=int(issue["number"]),
        issue_title=issue.get("title") or "",
        issue_body=issue.get("body") or "",
    )
    return {"status": "accepted", "run_id": run_id}


@app.post("/webhook/manual")
def manual_trigger(payload: ManualRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Trigger a run by repo and issue number, fetching the issue text ourselves."""
    try:
        issue = GitHubClient().get_issue(payload.repo, payload.issue_number)
    except Exception as error:  # noqa: BLE001 - surfaced to the caller as a 502
        logger.exception("Could not read %s#%s", payload.repo, payload.issue_number)
        raise HTTPException(status_code=502, detail=f"Could not read issue: {error}") from error

    run_id = _schedule(
        background_tasks,
        repo_full_name=issue["repo_full_name"],
        issue_number=issue["issue_number"],
        issue_title=issue["issue_title"],
        issue_body=issue["issue_body"],
    )
    return {"status": "accepted", "run_id": run_id}


def _schedule(
    background_tasks: BackgroundTasks,
    *,
    repo_full_name: str,
    issue_number: int,
    issue_title: str,
    issue_body: str,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    _runs[run_id] = {
        "run_id": run_id,
        "repo_full_name": repo_full_name,
        "issue_number": issue_number,
        "issue_title": issue_title,
        "status": "queued",
        "started_at": _now(),
        "finished_at": None,
        "state": None,
    }
    background_tasks.add_task(
        _execute_run, run_id, repo_full_name, issue_number, issue_title, issue_body
    )
    logger.info("Queued run %s for %s#%s", run_id, repo_full_name, issue_number)
    return run_id


def _execute_run(
    run_id: str,
    repo_full_name: str,
    issue_number: int,
    issue_title: str,
    issue_body: str,
) -> None:
    """Index the repo, then run the graph. Never raises — the run records the failure."""
    record = _runs[run_id]
    record["status"] = "running"

    try:
        summary = indexer.index_repo(repo_full_name)
        logger.info(
            "Run %s indexed %s (%d files, %d chunks)",
            run_id,
            repo_full_name,
            summary.files_indexed,
            summary.chunks_indexed,
        )

        final_state: AgentState = graph.run(
            repo_full_name,
            issue_number,
            issue_title,
            issue_body,
            test_runner=DockerTestRunner(),
            github_client=GitHubClient(),
        )
        record["state"] = dict(final_state)
        record["status"] = final_state.get("status", "failed")
    except Exception as error:  # noqa: BLE001 - a crashed run is a failed run
        logger.exception("Run %s crashed", run_id)
        record["status"] = "failed"
        record["state"] = {"status": "failed", "error": str(error)}
    finally:
        record["finished_at"] = _now()


def _verify_signature(raw_body: bytes, signature: str | None) -> None:
    """Reject deliveries that are not signed with the configured secret.

    Fails closed: an unset secret rejects every delivery rather than accepting
    unauthenticated ones, since this endpoint starts agent runs.
    """
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="GITHUB_WEBHOOK_SECRET is not configured; refusing unauthenticated deliveries.",
        )
    if not signature:
        raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header")

    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Signature mismatch")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
