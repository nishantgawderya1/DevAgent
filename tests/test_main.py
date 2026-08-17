from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main


SECRET = "test-secret"


@pytest.fixture
def client(monkeypatch):
    """A test client with the run registry cleared and runs stubbed out."""
    main._runs.clear()
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)

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


def test_health(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_webhook_accepts_a_signed_issue_opened_event(client) -> None:
    body = _issue_payload()

    response = client.post(
        "/webhook",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": "issues"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    run_id = response.json()["run_id"]
    assert main._runs[run_id]["repo_full_name"] == "owner/repo"
    assert main._runs[run_id]["issue_number"] == 7
    assert client.executed == [(run_id, "owner/repo", 7, "Fix pagination", "Off-by-one")]


def test_webhook_rejects_a_bad_signature(client) -> None:
    body = _issue_payload()

    response = client.post(
        "/webhook",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body, "wrong-secret"), "X-GitHub-Event": "issues"},
    )

    assert response.status_code == 401
    assert client.executed == []


def test_webhook_rejects_a_missing_signature(client) -> None:
    body = _issue_payload()

    response = client.post("/webhook", content=body, headers={"X-GitHub-Event": "issues"})

    assert response.status_code == 401
    assert client.executed == []


def test_webhook_fails_closed_when_no_secret_is_configured(monkeypatch) -> None:
    main._runs.clear()
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    body = _issue_payload()

    with TestClient(main.app) as unconfigured:
        response = unconfigured.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": "issues"},
        )

    # An unset secret must not mean "accept everything" — this endpoint starts runs.
    assert response.status_code == 503
    assert main._runs == {}


def test_webhook_ignores_other_events(client) -> None:
    body = _issue_payload()

    response = client.post(
        "/webhook",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": "push"},
    )

    assert response.json()["status"] == "ignored"
    assert client.executed == []


def test_webhook_ignores_non_opened_actions(client) -> None:
    body = _issue_payload(action="closed")

    response = client.post(
        "/webhook",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": "issues"},
    )

    assert response.json()["status"] == "ignored"
    assert client.executed == []


def test_manual_trigger_reads_the_issue_and_schedules(client, monkeypatch) -> None:
    class FakeGitHubClient:
        def get_issue(self, repo: str, number: int) -> dict[str, Any]:
            return {
                "repo_full_name": repo,
                "issue_number": number,
                "issue_title": "Fix pagination",
                "issue_body": "Off-by-one",
            }

    monkeypatch.setattr(main, "GitHubClient", FakeGitHubClient)

    response = client.post("/webhook/manual", json={"repo": "owner/repo", "issue_number": 42})

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert client.executed == [
        (response.json()["run_id"], "owner/repo", 42, "Fix pagination", "Off-by-one")
    ]


def test_manual_trigger_surfaces_a_github_failure(client, monkeypatch) -> None:
    class BoomClient:
        def get_issue(self, repo: str, number: int) -> dict[str, Any]:
            raise RuntimeError("404 Not Found")

    monkeypatch.setattr(main, "GitHubClient", BoomClient)

    response = client.post("/webhook/manual", json={"repo": "owner/repo", "issue_number": 42})

    assert response.status_code == 502
    assert "404 Not Found" in response.json()["detail"]
    assert client.executed == []


def test_manual_trigger_validates_its_payload(client) -> None:
    assert client.post("/webhook/manual", json={"repo": "owner/repo"}).status_code == 422
    assert (
        client.post("/webhook/manual", json={"repo": "owner/repo", "issue_number": 0}).status_code
        == 422
    )


def test_runs_endpoints_expose_run_state(client) -> None:
    body = _issue_payload()
    run_id = client.post(
        "/webhook",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": "issues"},
    ).json()["run_id"]

    listed = client.get("/runs").json()
    assert [run["run_id"] for run in listed] == [run_id]

    detail = client.get(f"/runs/{run_id}").json()
    assert detail["issue_title"] == "Fix pagination"
    assert detail["status"] == "queued"

    assert client.get("/runs/does-not-exist").status_code == 404
