from __future__ import annotations

from pathlib import Path
from typing import Any

from app.retrieval import indexer
from app.sandbox import docker as sandbox


class FakeContainer:
    def __init__(self, exit_code: int, logs: bytes) -> None:
        self._exit_code = exit_code
        self._logs = logs
        self.removed = False
        self.killed = False

    def wait(self, timeout: int | None = None) -> dict[str, int]:
        return {"StatusCode": self._exit_code}

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        return self._logs

    def remove(self, force: bool = False) -> None:
        self.removed = True

    def kill(self) -> None:
        self.killed = True


class FakeDockerClient:
    def __init__(self, container: FakeContainer) -> None:
        self._container = container
        self.runs: list[tuple[str, dict[str, Any]]] = []
        self.containers = self

    def run(self, image: str, **kwargs: Any) -> FakeContainer:
        self.runs.append((image, kwargs))
        return self._container


PATCH = """\
--- a/api.py
+++ b/api.py
@@ -1,2 +1,2 @@
 def paginate(items, page_size):
-    return items[: page_size + 1]
+    return items[:page_size]
"""

SOURCE = "def paginate(items, page_size):\n    return items[: page_size + 1]\n"


def test_apply_diff_applies_a_valid_patch(tmp_path: Path, make_git_repo) -> None:
    repo = make_git_repo(tmp_path / "repo", {"api.py": SOURCE})

    applied, output = sandbox._apply_diff(repo, PATCH)

    assert applied is True
    assert output == ""
    assert (repo / "api.py").read_text(encoding="utf-8") == (
        "def paginate(items, page_size):\n    return items[:page_size]\n"
    )
    # The temporary patch file must not be left behind for `git add -A` to catch.
    assert not (repo / ".devagent.patch").exists()


def test_apply_diff_reports_actionable_failure(tmp_path: Path, make_git_repo) -> None:
    repo = make_git_repo(tmp_path / "repo", {"api.py": "def paginate():\n    return []\n"})

    applied, output = sandbox._apply_diff(repo, PATCH)

    assert applied is False
    assert "could not be applied" in output
    # The message is fed back to the writer, so it must say what to do next.
    assert "Regenerate the diff" in output
    assert not (repo / ".devagent.patch").exists()


def test_runner_reports_unapplied_patch_as_a_test_failure(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    repo = make_git_repo(tmp_path / "repo", {"api.py": "def other():\n    pass\n"})
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: repo)

    runner = sandbox.DockerTestRunner(
        docker_client=FakeDockerClient(FakeContainer(0, b"")),
        workspace_root=tmp_path / "runs",
    )
    exit_code, output = runner("owner/repo", PATCH)

    # Non-zero so the tester routes it back to the writer for repair.
    assert exit_code == 1
    assert "could not be applied" in output


def test_runner_runs_tests_and_returns_output(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    repo = make_git_repo(tmp_path / "repo", {"api.py": SOURCE})
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: repo)

    container = FakeContainer(0, b"5 passed in 0.10s")
    client = FakeDockerClient(container)
    runner = sandbox.DockerTestRunner(
        docker_client=client, workspace_root=tmp_path / "runs"
    )

    exit_code, output = runner("owner/repo", PATCH)

    assert exit_code == 0
    assert output == "5 passed in 0.10s"
    assert container.removed is True


def test_runner_applies_isolation_limits(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    repo = make_git_repo(tmp_path / "repo", {"api.py": SOURCE})
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: repo)

    client = FakeDockerClient(FakeContainer(0, b"ok"))
    config = sandbox.SandboxConfig(allow_network=False, mem_limit="512m")
    sandbox.DockerTestRunner(
        config=config, docker_client=client, workspace_root=tmp_path / "runs"
    )("owner/repo", PATCH)

    image, kwargs = client.runs[0]
    assert image == sandbox.PYTHON_IMAGE
    assert kwargs["network_disabled"] is True
    assert kwargs["mem_limit"] == "512m"
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges"]
    assert kwargs["working_dir"] == sandbox.WORKDIR


def test_runner_never_mutates_the_indexed_checkout(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    repo = make_git_repo(tmp_path / "repo", {"api.py": SOURCE})
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: repo)

    sandbox.DockerTestRunner(
        docker_client=FakeDockerClient(FakeContainer(0, b"ok")),
        workspace_root=tmp_path / "runs",
    )("owner/repo", PATCH)

    # The patch was applied to a throwaway clone, not to the tree the indexer built from.
    assert (repo / "api.py").read_text(encoding="utf-8") == SOURCE


def test_runner_cleans_up_its_workspace(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    repo = make_git_repo(tmp_path / "repo", {"api.py": SOURCE})
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: repo)
    workspace_root = tmp_path / "runs"

    sandbox.DockerTestRunner(
        docker_client=FakeDockerClient(FakeContainer(0, b"ok")), workspace_root=workspace_root
    )("owner/repo", PATCH)

    assert not workspace_root.exists() or list(workspace_root.iterdir()) == []


def test_runner_returns_failure_when_docker_is_unavailable(tmp_path: Path, monkeypatch, make_git_repo) -> None:
    repo = make_git_repo(tmp_path / "repo", {"api.py": SOURCE})
    monkeypatch.setattr(indexer, "_ensure_repo_checkout", lambda name: repo)

    class BoomClient:
        containers = None

        def run(self, *args: Any, **kwargs: Any):
            raise RuntimeError("Cannot connect to the Docker daemon")

    boom = BoomClient()
    boom.containers = boom

    exit_code, output = sandbox.DockerTestRunner(
        docker_client=boom, workspace_root=tmp_path / "runs"
    )("owner/repo", PATCH)

    assert exit_code == 1
    assert "Docker daemon" in output


def test_detect_suite_picks_node_for_package_json(tmp_path: Path) -> None:
    workspace = tmp_path / "js"
    workspace.mkdir()
    (workspace / "package.json").write_text("{}", encoding="utf-8")

    image, command = sandbox._detect_suite(workspace, sandbox.SandboxConfig())

    assert image == sandbox.NODE_IMAGE
    assert "jest" in command


def test_detect_suite_defaults_to_python(tmp_path: Path) -> None:
    image, command = sandbox._detect_suite(tmp_path, sandbox.SandboxConfig())

    assert image == sandbox.PYTHON_IMAGE
    assert "pytest" in command
