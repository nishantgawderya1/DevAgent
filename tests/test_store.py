from __future__ import annotations

from pathlib import Path

import pytest

from app import store


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DEVAGENT_DB_PATH", str(tmp_path / "runs.db"))
    store.init_db()


def _create(run_id: str = "run1", **overrides):
    kwargs = {
        "repo_full_name": "owner/repo",
        "issue_number": 7,
        "issue_title": "Fix pagination",
        "triggered_by": None,
    }
    kwargs.update(overrides)
    return store.create_run(run_id, **kwargs)


def test_create_run_round_trips() -> None:
    created = _create(triggered_by="nishant")

    stored = store.get_run("run1")

    assert stored["run_id"] == "run1"
    assert stored["repo_full_name"] == "owner/repo"
    assert stored["issue_number"] == 7
    assert stored["issue_title"] == "Fix pagination"
    assert stored["triggered_by"] == "nishant"
    assert stored["status"] == "queued"
    assert stored["finished_at"] is None
    assert stored["state"] is None
    assert stored["started_at"] == created["started_at"]


def test_get_run_returns_none_when_absent() -> None:
    assert store.get_run("nope") is None


def test_init_db_is_idempotent() -> None:
    _create()
    store.init_db()  # must not wipe or fail

    assert store.get_run("run1") is not None


def test_update_run_is_partial() -> None:
    _create()
    state = {"status": "success", "diff": "--- a/x\n+++ b/x\n"}

    store.update_run("run1", status="running")
    store.update_run("run1", state=state)

    stored = store.get_run("run1")
    # Setting state must not have reverted the status set by the earlier call.
    assert stored["status"] == "running"
    assert stored["state"] == state
    assert stored["finished_at"] is None


def test_update_run_marks_finished_separately() -> None:
    _create()

    store.update_run("run1", status="success", finished=True)

    stored = store.get_run("run1")
    assert stored["status"] == "success"
    assert stored["finished_at"] is not None


def test_update_run_with_nothing_to_set_is_a_noop() -> None:
    _create()

    store.update_run("run1")

    assert store.get_run("run1")["status"] == "queued"


def test_state_survives_a_full_agent_state_shape() -> None:
    """The state blob carries nested lists and dicts; JSON must not flatten them."""
    _create()
    state = {
        "plan": [{"id": 1, "description": "Correct the upper bound"}],
        "retrieved_context": [
            {"chunk_id": "api.py::paginate", "source": "expanded", "text": "def paginate(): ..."}
        ],
        "test_report": {"passed": False, "output": "boom", "failed_tests": ["t::a"]},
        "target_files": ["api.py", "utils.py"],
        "retry_count": 2,
    }

    store.update_run("run1", state=state)

    assert store.get_run("run1")["state"] == state


def test_unserialisable_state_does_not_crash_the_write() -> None:
    """default=str keeps an exotic value from losing the whole run record."""
    _create()

    store.update_run("run1", state={"error": ValueError("boom")})

    assert "boom" in store.get_run("run1")["state"]["error"]


def test_corrupt_state_degrades_to_none(tmp_path: Path) -> None:
    import sqlite3

    _create()
    connection = sqlite3.connect(store.db_path())
    connection.execute("UPDATE runs SET state = 'not json' WHERE run_id = 'run1'")
    connection.commit()
    connection.close()

    # One unreadable blob must not take out the listing.
    assert store.get_run("run1")["state"] is None
    assert len(store.list_runs()) == 1


def test_list_runs_is_newest_first() -> None:
    _create("run1")
    _create("run2")
    _create("run3")

    assert [run["run_id"] for run in store.list_runs()] == ["run3", "run2", "run1"]


def test_list_runs_honours_the_limit() -> None:
    for index in range(5):
        _create(f"run{index}")

    assert len(store.list_runs(limit=2)) == 2


def test_list_runs_is_empty_on_a_fresh_db() -> None:
    assert store.list_runs() == []


def test_db_path_prefers_the_env_var(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEVAGENT_DB_PATH", str(tmp_path / "custom.db"))
    assert store.db_path() == tmp_path / "custom.db"

    monkeypatch.delenv("DEVAGENT_DB_PATH")
    # Falls back beside the index root rather than the working directory.
    assert store.db_path().name == "devagent.db"
