"""Durable run storage.

Runs used to live in a module-level dict, so a restart erased every run the
agent had ever done. This is the same data behind the same shape --
``create_run`` / ``update_run`` / ``get_run`` / ``list_runs`` -- backed by SQLite
so it survives.

Stdlib ``sqlite3`` on purpose: the project has stayed dependency-light and a run
registry does not justify an ORM. Connections are opened per call rather than
shared, because FastAPI runs background tasks on a threadpool and sqlite3
connections are not safe to pass between threads.

The ``state`` column holds a JSON-serialised :class:`~app.agent.state.AgentState`.
That works because the state is plain data end to end -- ``explorer.py``
deliberately flattens retrieval hits to dicts rather than keeping dataclasses.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.retrieval import indexer


logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    repo_full_name TEXT NOT NULL,
    issue_number   INTEGER NOT NULL,
    issue_title    TEXT,
    status         TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    triggered_by   TEXT,
    state          TEXT
);
CREATE INDEX IF NOT EXISTS runs_started_at ON runs (started_at DESC);
"""


def db_path() -> Path:
    """Resolved per call so tests and .env changes are picked up."""
    configured = os.getenv("DEVAGENT_DB_PATH")
    if configured:
        return Path(configured)
    return indexer.INDEX_ROOT / "devagent.db"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Create the schema if absent. Safe to call repeatedly."""
    with _connect() as connection:
        connection.executescript(_SCHEMA)


def create_run(
    run_id: str,
    *,
    repo_full_name: str,
    issue_number: int,
    issue_title: str,
    triggered_by: str | None = None,
) -> dict[str, Any]:
    """Record a queued run and return it."""
    started_at = _now()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO runs (run_id, repo_full_name, issue_number, issue_title, "
            "status, started_at, finished_at, triggered_by, state) "
            "VALUES (?, ?, ?, ?, 'queued', ?, NULL, ?, NULL)",
            (run_id, repo_full_name, issue_number, issue_title, started_at, triggered_by),
        )

    return {
        "run_id": run_id,
        "repo_full_name": repo_full_name,
        "issue_number": issue_number,
        "issue_title": issue_title,
        "status": "queued",
        "started_at": started_at,
        "finished_at": None,
        "triggered_by": triggered_by,
        "state": None,
    }


def update_run(
    run_id: str,
    *,
    status: str | None = None,
    state: dict[str, Any] | None = None,
    finished: bool = False,
) -> None:
    """Patch only the fields passed.

    Partial by design: marking a run running must not clear the state a later
    call will set.
    """
    assignments: list[str] = []
    values: list[Any] = []

    if status is not None:
        assignments.append("status = ?")
        values.append(status)
    if state is not None:
        assignments.append("state = ?")
        values.append(json.dumps(state, default=str))
    if finished:
        assignments.append("finished_at = ?")
        values.append(_now())

    if not assignments:
        return

    values.append(run_id)
    with _connect() as connection:
        connection.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?", values)


def get_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return _as_dict(row) if row is not None else None


def list_runs(limit: int = 100) -> list[dict[str, Any]]:
    """Most recent first."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY started_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_as_dict(row) for row in rows]


def _as_dict(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["state"] = _load_state(record.get("state"), record.get("run_id"))
    return record


def _load_state(raw: Any, run_id: Any) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        # A corrupt blob should degrade one run's detail view, not break the list.
        logger.warning("Run %s has unreadable state JSON; returning None.", run_id)
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
