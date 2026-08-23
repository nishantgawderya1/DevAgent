from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main, store


SECRET = "test-secret"
TOKEN = "test-api-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    """Test client on a throwaway DB, with runs stubbed out."""
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("DEVAGENT_API_TOKEN", TOKEN)
    monkeypatch.setenv("DEVAGENT_DB_PATH", str(tmp_path / "runs.db"))

    executed: list[tuple[Any, ...]] = []
    monkeypatch.setattr(main, "_execute_run", lambda *args: executed.append(args))

    with TestClient(main.app) as test_client:
        test_client.executed = executed  # type: ignore[attr-defined]
        yield test_client


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _issue_payload(number: int = 7, action: str = "opened") -> bytes:
    return json.dumps(
        {
            "action": action,
            "issue": {"number": number, "title": "Fix pagination", "body": "Off-by-one"},
            "repository": {"full_name": "owner/repo"},
        }
    ).encode()


def _post_webhook(client, body: bytes | None = None, event: str = "issues"):
    body = _issue_payload() if body is None else body
    return client.post(
        "/webhook",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": event},
    )


# --- health -------------------------------------------------------------------


def test_health_needs_no_token(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


# --- API token ----------------------------------------------------------------


def test_read_endpoints_reject_a_missing_token(client) -> None:
    assert client.get("/runs").status_code == 401
    assert client.get("/runs/anything").status_code == 401


def test_read_endpoints_reject_a_wrong_token(client) -> None:
    response = client.get("/runs", headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API token"


def test_malformed_authorization_header_is_rejected(client) -> None:
    assert client.get("/runs", headers={"Authorization": TOKEN}).status_code == 401
    assert client.get("/runs", headers={"Authorization": "Basic abc"}).status_code == 401
    assert client.get("/runs", headers={"Authorization": "Bearer "}).status_code == 401


def test_manual_trigger_requires_a_token(client) -> None:
    response = client.post("/webhook/manual", json={"repo": "owner/repo", "issue_number": 1})

    assert response.status_code == 401
    assert client.executed == []


def test_endpoints_fail_closed_when_no_token_is_configured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DEVAGENT_API_TOKEN", raising=False)
    monkeypatch.setenv("DEVAGENT_DB_PATH", str(tmp_path / "runs.db"))

    with TestClient(main.app) as unconfigured:
        response = unconfigured.get("/runs", headers=AUTH)

    # An unset token must not mean "allow everyone" on endpoints this sensitive.
    assert response.status_code == 503
    assert "DEVAGENT_API_TOKEN" in response.json()["detail"]


# --- webhook ------------------------------------------------------------------


def test_webhook_accepts_a_signed_issue_opened_event(client) -> None:
    response = _post_webhook(client)

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    stored = store.get_run(run_id)
    assert stored["repo_full_name"] == "owner/repo"
    assert stored["issue_number"] == 7
    assert stored["status"] == "queued"
    assert client.executed == [(run_id, "owner/repo", 7, "Fix pagination", "Off-by-one")]


def test_webhook_needs_no_api_token(client) -> None:
    # GitHub cannot send a bearer token, so this path stays on HMAC alone.
    assert _post_webhook(client).status_code == 200


def test_webhook_rejects_a_bad_signature(client) -> None:
    body = _issue_payload()

    response = client.post(
        "/webhook",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body, "wrong"), "X-GitHub-Event": "issues"},
    )

    assert response.status_code == 401
    assert client.executed == []


def test_webhook_rejects_a_missing_signature(client) -> None:
    response = client.post(
        "/webhook", content=_issue_payload(), headers={"X-GitHub-Event": "issues"}
    )

    assert response.status_code == 401
    assert client.executed == []


def test_webhook_fails_closed_without_a_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    monkeypatch.setenv("DEVAGENT_DB_PATH", str(tmp_path / "runs.db"))
    body = _issue_payload()

    with TestClient(main.app) as unconfigured:
        response = unconfigured.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": "issues"},
        )

    assert response.status_code == 503


def test_webhook_ignores_other_events(client) -> None:
    assert _post_webhook(client, event="push").json()["status"] == "ignored"
    assert client.executed == []


def test_webhook_ignores_non_opened_actions(client) -> None:
    response = _post_webhook(client, body=_issue_payload(action="closed"))

    assert response.json()["status"] == "ignored"
    assert client.executed == []


# --- manual trigger -----------------------------------------------------------


class _FakeGitHubClient:
    def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        return {
            "repo_full_name": repo,
            "issue_number": number,
            "issue_title": "Fix pagination",
            "issue_body": "Off-by-one",
        }


def test_manual_trigger_reads_the_issue_and_schedules(client, monkeypatch) -> None:
    monkeypatch.setattr(main, "GitHubClient", _FakeGitHubClient)

    response = client.post(
        "/webhook/manual",
        json={"repo": "owner/repo", "issue_number": 42, "triggered_by": "nishant"},
        headers=AUTH,
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert store.get_run(run_id)["triggered_by"] == "nishant"
    assert client.executed == [(run_id, "owner/repo", 42, "Fix pagination", "Off-by-one")]


def test_manual_trigger_surfaces_a_github_failure(client, monkeypatch) -> None:
    class BoomClient:
        def get_issue(self, repo: str, number: int) -> dict[str, Any]:
            raise RuntimeError("404 Not Found")

    monkeypatch.setattr(main, "GitHubClient", BoomClient)

    response = client.post(
        "/webhook/manual", json={"repo": "owner/repo", "issue_number": 42}, headers=AUTH
    )

    assert response.status_code == 502
    assert "404 Not Found" in response.json()["detail"]
    assert client.executed == []


def test_manual_trigger_validates_its_payload(client) -> None:
    assert client.post("/webhook/manual", json={"repo": "o/r"}, headers=AUTH).status_code == 422
    assert (
        client.post(
            "/webhook/manual", json={"repo": "o/r", "issue_number": 0}, headers=AUTH
        ).status_code
        == 422
    )


# --- run registry -------------------------------------------------------------


def test_runs_endpoints_expose_stored_runs(client) -> None:
    run_id = _post_webhook(client).json()["run_id"]

    listed = client.get("/runs", headers=AUTH).json()
    assert [run["run_id"] for run in listed] == [run_id]

    detail = client.get(f"/runs/{run_id}", headers=AUTH).json()
    assert detail["issue_title"] == "Fix pagination"
    assert detail["status"] == "queued"

    assert client.get("/runs/does-not-exist", headers=AUTH).status_code == 404


def test_runs_survive_an_app_restart(client) -> None:
    """The whole point of the store: a restart used to erase every run."""
    run_id = _post_webhook(client).json()["run_id"]

    # Same DB path from the fixture env, brand new app instance.
    with TestClient(main.app) as restarted:
        listed = restarted.get("/runs", headers=AUTH).json()

    assert [run["run_id"] for run in listed] == [run_id]


# --- run deadline ---------------------------------------------------------------


def test_run_timeout_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("DEVAGENT_RUN_TIMEOUT_SECONDS", raising=False)
    assert main._run_timeout() == main.DEFAULT_RUN_TIMEOUT_SECONDS

    monkeypatch.setenv("DEVAGENT_RUN_TIMEOUT_SECONDS", "45")
    assert main._run_timeout() == 45.0

    # A typo must not remove the ceiling entirely.
    monkeypatch.setenv("DEVAGENT_RUN_TIMEOUT_SECONDS", "half an hour")
    assert main._run_timeout() == main.DEFAULT_RUN_TIMEOUT_SECONDS


def test_a_hung_graph_is_recorded_as_a_failed_run(monkeypatch, tmp_path: Path) -> None:
    import time

    monkeypatch.setenv("DEVAGENT_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("DEVAGENT_RUN_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setattr(main, "DockerTestRunner", lambda *a, **k: None)
    monkeypatch.setattr(main, "GitHubClient", lambda *a, **k: None)
    monkeypatch.setattr(main.indexer, "index_repo", lambda repo: SimpleNamespace(
        files_indexed=0, chunks_indexed=0
    ))

    def hang(*args, **kwargs):
        time.sleep(5)

    monkeypatch.setattr(main.graph, "run", hang)
    store.init_db()
    store.create_run("hung", repo_full_name="owner/repo", issue_number=1, issue_title="t")

    main._execute_run("hung", "owner/repo", 1, "t", "b")

    record = store.get_run("hung")
    assert record["status"] == "failed"
    assert "exceeded" in record["state"]["error"]
    assert record["finished_at"] is not None
