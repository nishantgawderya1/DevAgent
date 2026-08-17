"""GitHub backend — reads issues and opens pull requests.

Satisfies the ``GitHubClient`` protocol in :mod:`app.agent.nodes.pr`, and adds
the issue read the webhook needs to build a run.

Opening the PR is split across two mechanisms on purpose. The branch is built
with **git** — clone locally, apply the patch, commit, push — because a unified
diff is exactly what ``git apply`` consumes, and reconstructing arbitrary
multi-hunk patches through the REST contents API would mean reimplementing patch
application. The pull request itself is opened through **PyGithub**, which is
where the API is pleasant.

Authentication accepts either a token (``GITHUB_TOKEN``, personal or
installation) or GitHub App credentials, in which case an installation token is
minted for the target repository.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from app.fsutil import remove_tree
from app.retrieval import indexer


logger = logging.getLogger(__name__)

DEFAULT_COMMIT_NAME = "DevAgent"
DEFAULT_COMMIT_EMAIL = "devagent@users.noreply.github.com"


class GitHubError(RuntimeError):
    """Raised when a GitHub operation cannot be completed."""


class GitHubClient:
    """Reads issues and opens pull requests for a repository."""

    def __init__(
        self,
        *,
        token: str | None = None,
        github: Any | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self._token = token
        self._github = github
        self._workspace_root = workspace_root or Path(tempfile.gettempdir()) / "devagent-prs"

    # --- issues -----------------------------------------------------------

    def get_issue(self, repo_full_name: str, issue_number: int) -> dict[str, Any]:
        """Fetch the fields a run needs to start."""
        repo = self._repo(repo_full_name)
        issue = repo.get_issue(number=issue_number)
        return {
            "repo_full_name": repo_full_name,
            "issue_number": issue_number,
            "issue_title": issue.title or "",
            "issue_body": issue.body or "",
        }

    # --- pull requests ----------------------------------------------------

    def create_pull_request(
        self, *, repo_full_name: str, branch: str, title: str, body: str, diff: str
    ) -> str:
        """Push ``diff`` to ``branch`` and open a PR against the default branch."""
        repo = self._repo(repo_full_name)
        base = repo.default_branch

        workspace = self._workspace_root / f"pr-{uuid.uuid4().hex[:12]}"
        try:
            self._build_branch(repo_full_name, workspace, branch, title, diff)
        finally:
            remove_tree(workspace)

        pull = repo.create_pull(title=title, body=body, head=branch, base=base)
        logger.info("Opened PR %s", pull.html_url)
        return pull.html_url

    def _build_branch(
        self, repo_full_name: str, workspace: Path, branch: str, title: str, diff: str
    ) -> None:
        """Clone, branch, apply, commit, push."""
        source = indexer._ensure_repo_checkout(repo_full_name)
        workspace.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--local", "--quiet", str(source), str(workspace)])

        # The local clone's origin points at the on-disk checkout, so retarget it
        # at GitHub with credentials before pushing.
        _git(["remote", "set-url", "origin", self._authenticated_url(repo_full_name)], cwd=workspace)
        _git(["checkout", "-b", branch], cwd=workspace)

        patch_path = workspace / ".devagent.patch"
        patch_path.write_text(diff if diff.endswith("\n") else diff + "\n", encoding="utf-8")
        try:
            _git(["apply", str(patch_path)], cwd=workspace)
        finally:
            patch_path.unlink(missing_ok=True)

        _git(["add", "-A"], cwd=workspace)
        _git(
            [
                "-c",
                f"user.name={os.getenv('DEVAGENT_GIT_NAME', DEFAULT_COMMIT_NAME)}",
                "-c",
                f"user.email={os.getenv('DEVAGENT_GIT_EMAIL', DEFAULT_COMMIT_EMAIL)}",
                "commit",
                "-m",
                title,
            ],
            cwd=workspace,
        )
        _git(["push", "--quiet", "origin", branch], cwd=workspace)

    # --- auth -------------------------------------------------------------

    def _authenticated_url(self, repo_full_name: str) -> str:
        return f"https://x-access-token:{self._resolve_token()}@github.com/{repo_full_name}.git"

    def _resolve_token(self) -> str:
        if self._token:
            return self._token

        token = os.getenv("GITHUB_TOKEN")
        if token:
            self._token = token
            return token

        self._token = _mint_app_token()
        return self._token

    def _repo(self, repo_full_name: str) -> Any:
        return self._client().get_repo(repo_full_name)

    def _client(self) -> Any:
        if self._github is None:
            from github import Auth, Github

            self._github = Github(auth=Auth.Token(self._resolve_token()))
        return self._github


def _mint_app_token() -> str:
    """Derive an installation token from GitHub App credentials."""
    app_id = os.getenv("GITHUB_APP_ID")
    key_path = os.getenv("GITHUB_PRIVATE_KEY_PATH")
    if not app_id or not key_path:
        raise GitHubError(
            "No GitHub credentials found. Set GITHUB_TOKEN, or GITHUB_APP_ID plus "
            "GITHUB_PRIVATE_KEY_PATH for GitHub App authentication."
        )

    key_file = Path(key_path)
    if not key_file.exists():
        raise GitHubError(f"GitHub App private key not found at {key_path}")

    from github import Auth, GithubIntegration

    app_auth = Auth.AppAuth(int(app_id), key_file.read_text(encoding="utf-8"))
    installations = list(GithubIntegration(auth=app_auth).get_installations())
    if not installations:
        raise GitHubError("GitHub App has no installations; install it on the target repository.")

    return app_auth.get_installation_auth(installations[0].id).token


def _git(args: list[str], cwd: Path | None = None) -> str:
    """Run a git command, raising with git's own stderr when it fails."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitHubError(f"git {args[0]} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout
