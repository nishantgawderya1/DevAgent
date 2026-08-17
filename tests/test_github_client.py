from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.github import client as github_client
from app.retrieval import indexer


PATCH = """\
--- a/api.py
+++ b/api.py
@@ -1,2 +1,2 @@
 def paginate(items, page_size):
-    return items[: page_size + 1]
+    return items[:page_size]
"""

SOURCE = "def paginate(items, page_size):\n    return items[: page_size + 1]\n"


class FakeRepo:
    def __init__(self, default_branch: str = "main") -> None:
        self.default_branch = default_branch
        self.pulls: list[dict[str, Any]] = []

    def create_pull(self, **kwargs: Any) -> Any:
        self.pulls.append(kwargs)
        return SimpleNamespace(html_url="https://github.com/owner/repo/pull/12")

    def get_issue(self, number: int) -> Any:
        return SimpleNamespace(title="Fix pagination", body="Off-by-one in page size")


class FakeGithub:
    def __init__(self, repo: FakeRepo) -> None:
        self._repo = repo
        self.requested: list[str] = []

    def get_repo(self, full_name: str) -> FakeRepo:
        self.requested.append(full_name)
        return self._repo


def _bare_remote(path: Path) -> Path:
    subprocess.run(["git", "init", "--bare", "--quiet", str(path)], check=True, capture_output=True)
    return path


def test_get_issue_returns_the_fields_a_run_needs() -> None:
    client = github_client.GitHubClient(token="t", github=FakeGithub(FakeRepo()))

    issue = client.get_issue("owner/repo", 7)

    assert issue == {
        "repo_full_name": "owner/repo",
        "issue_number": 7,
        "issue_title": "Fix pagination",
        "issue_body": "Off-by-one in page size",
    }


def test_create_pull_request_pushes_the_patched_branch(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    source = make_git_repo(tmp_path / "src", {"api.py": SOURCE})
    remote = _bare_remote(tmp_path / "remote.git")
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: source)

    repo = FakeRepo()
    client = github_client.GitHubClient(
        token="t", github=FakeGithub(repo), workspace_root=tmp_path / "prs"
    )
    # Point the push at a local bare repo so the real git path runs end to end.
    monkeypatch.setattr(client, "_authenticated_url", lambda name: str(remote))

    url = client.create_pull_request(
        repo_full_name="owner/repo",
        branch="devagent/issue-7",
        title="Fix #7: pagination",
        body="Closes #7.",
        diff=PATCH,
    )

    assert url == "https://github.com/owner/repo/pull/12"

    # The branch really exists on the remote, with the patch applied.
    shown = subprocess.run(
        ["git", "show", "devagent/issue-7:api.py"],
        cwd=remote,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "return items[:page_size]" in shown.stdout
    # The patch file must not be committed alongside the change.
    listing = subprocess.run(
        ["git", "ls-tree", "--name-only", "devagent/issue-7"],
        cwd=remote,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ".devagent.patch" not in listing.stdout


def test_create_pull_request_targets_the_default_branch(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    source = make_git_repo(tmp_path / "src", {"api.py": SOURCE})
    remote = _bare_remote(tmp_path / "remote.git")
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: source)

    repo = FakeRepo(default_branch="develop")
    client = github_client.GitHubClient(
        token="t", github=FakeGithub(repo), workspace_root=tmp_path / "prs"
    )
    monkeypatch.setattr(client, "_authenticated_url", lambda name: str(remote))

    client.create_pull_request(
        repo_full_name="owner/repo",
        branch="devagent/issue-7",
        title="Fix #7",
        body="body",
        diff=PATCH,
    )

    call = repo.pulls[0]
    assert call["base"] == "develop"
    assert call["head"] == "devagent/issue-7"
    assert call["title"] == "Fix #7"


def test_create_pull_request_raises_when_the_patch_does_not_apply(
    tmp_path: Path, monkeypatch, make_git_repo
) -> None:
    source = make_git_repo(tmp_path / "src", {"api.py": "def other():\n    pass\n"})
    remote = _bare_remote(tmp_path / "remote.git")
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: source)

    client = github_client.GitHubClient(
        token="t", github=FakeGithub(FakeRepo()), workspace_root=tmp_path / "prs"
    )
    monkeypatch.setattr(client, "_authenticated_url", lambda name: str(remote))

    with pytest.raises(github_client.GitHubError, match="git apply failed"):
        client.create_pull_request(
            repo_full_name="owner/repo",
            branch="devagent/issue-7",
            title="Fix #7",
            body="body",
            diff=PATCH,
        )


def test_cleans_up_its_workspace(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    source = make_git_repo(tmp_path / "src", {"api.py": SOURCE})
    remote = _bare_remote(tmp_path / "remote.git")
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: source)
    workspace_root = tmp_path / "prs"

    client = github_client.GitHubClient(
        token="t", github=FakeGithub(FakeRepo()), workspace_root=workspace_root
    )
    monkeypatch.setattr(client, "_authenticated_url", lambda name: str(remote))
    client.create_pull_request(
        repo_full_name="owner/repo", branch="b", title="t", body="b", diff=PATCH
    )

    assert not workspace_root.exists() or list(workspace_root.iterdir()) == []


def test_token_resolution_prefers_explicit_then_env(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")

    assert github_client.GitHubClient(token="explicit")._resolve_token() == "explicit"
    assert github_client.GitHubClient()._resolve_token() == "from-env"


def test_missing_credentials_raise_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_PRIVATE_KEY_PATH", raising=False)

    with pytest.raises(github_client.GitHubError, match="No GitHub credentials"):
        github_client.GitHubClient()._resolve_token()


def test_authenticated_url_embeds_the_token() -> None:
    url = github_client.GitHubClient(token="abc123")._authenticated_url("owner/repo")

    assert url == "https://x-access-token:abc123@github.com/owner/repo.git"


def test_app_package_does_not_shadow_pygithub() -> None:
    # `app/github/` shares a name with PyGithub's top-level `github` package.
    # Absolute imports mean the real one wins, but that is worth pinning.
    from github import Auth, Github  # noqa: F401

    assert Github.__module__.startswith("github")
