"""FastAPI entrypoint — receives issues and runs them through the agent.

Two ways in: a GitHub App webhook (``/webhook``, fired on ``issues.opened``) and
a manual trigger (``/webhook/manual``) for testing without a webhook round-trip.
Both do the same thing — schedule a background run and return immediately, since
a full run takes minutes and GitHub expects a webhook response in seconds.

Each run indexes the repository before invoking the graph. ``index_repo`` skips
the work when the collection is already current for the checked-out commit, so
the cost is paid per repository revision rather than per issue — but it has to
happen, because the explorer node has nothing to retrieve from otherwise.

Run state lives in SQLite (:mod:`app.store`) so it survives a restart. Every
endpoint that starts a run or exposes run contents is behind a bearer token
(:mod:`app.auth`); ``/webhook`` stays on HMAC because GitHub cannot send one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

# Load .env BEFORE importing app modules. Some of them resolve configuration at
# import time -- indexer.INDEX_ROOT is the live example -- so loading afterwards
# silently discarded DEVAGENT_INDEX_ROOT and cloned every repo into system temp.
load_dotenv()

from app import auth, store  # noqa: E402
from app.agent import graph  # noqa: E402
from app.agent.state import AgentState  # noqa: E402
from app.github.client import GitHubClient  # noqa: E402
from app.retrieval import indexer  # noqa: E402
from app.sandbox.docker import DockerTestRunner  # noqa: E402


logging.basicConfig(level=os.getenv("DEVAGENT_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

def _log_resolved_config() -> None:
    """Surface the paths we actually resolved, not the ones .env asked for.

    Config that silently falls back is the hardest kind to debug, so report what
    took effect at boot rather than discovering it three steps into a run.
    """
    logger.info("Index root: %s", indexer.INDEX_ROOT)
    logger.info(
        "Webhook secret configured: %s | GitHub credentials: %s",
        bool(os.getenv("GITHUB_WEBHOOK_SECRET")),
        bool(os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_APP_ID")),
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _log_resolved_config()
    store.init_db()
    yield


app = FastAPI(
    title="DevAgent",
    description="Autonomous GitHub issue resolver",
    lifespan=_lifespan,
)


class ManualRequest(BaseModel):
    repo: str = Field(..., description="Repository in owner/name form")
    issue_number: int = Field(..., ge=1)
    triggered_by: str | None = Field(
        default=None, description="Who asked for this run; recorded for audit"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs", dependencies=[Depends(auth.require_api_token)])
def list_runs() -> list[dict[str, Any]]:
    return store.list_runs()


@app.get("/runs/{run_id}", dependencies=[Depends(auth.require_api_token)])
def get_run(run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
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


@app.post("/webhook/manual", dependencies=[Depends(auth.require_api_token)])
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
        triggered_by=payload.triggered_by,
    )
    return {"status": "accepted", "run_id": run_id}


def _schedule(
    background_tasks: BackgroundTasks,
    *,
    repo_full_name: str,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    triggered_by: str | None = None,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    store.create_run(
        run_id,
        repo_full_name=repo_full_name,
        issue_number=issue_number,
        issue_title=issue_title,
        triggered_by=triggered_by,
    )
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
    store.update_run(run_id, status="running")

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
        store.update_run(
            run_id, status=final_state.get("status", "failed"), state=dict(final_state)
        )
    except Exception as error:  # noqa: BLE001 - a crashed run is a failed run
        logger.exception("Run %s crashed", run_id)
        store.update_run(
            run_id, status="failed", state={"status": "failed", "error": str(error)}
        )
    finally:
        store.update_run(run_id, finished=True)


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

