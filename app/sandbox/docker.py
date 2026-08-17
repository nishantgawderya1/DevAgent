"""Sandboxed test execution — the ``TestRunner`` backend for the tester node.

Satisfies the protocol in :mod:`app.agent.nodes.tester`: given a repository and
a diff, apply the patch to a disposable checkout, run the project's own test
suite inside a container, and return ``(exit_code, output)``.

Three things are worth knowing about the design:

**Patch failures are test failures.** If ``git apply`` rejects the diff we return
a non-zero exit code with git's own error text rather than raising. The tester
node turns that into a failing report, which routes back to the writer — so a
malformed patch is repaired by the same loop that repairs a wrong one.

**The indexed checkout is never mutated.** Each run clones from the local
checkout into a throwaway workspace, so applying a patch cannot corrupt the tree
the indexer built its vectors from.

**Network is on by default.** Installing a project's dependencies needs it, and
without dependencies almost no real suite runs. The container is still isolated
from the host filesystem and process space, capabilities are dropped, and memory
and PIDs are capped. Set ``allow_network=False`` when the target repo vendors its
dependencies and you want the tighter box.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.fsutil import remove_tree
from app.retrieval import indexer


logger = logging.getLogger(__name__)

PYTHON_IMAGE = "python:3.11-slim"
NODE_IMAGE = "node:20-slim"
WORKDIR = "/workspace"

_PYTHON_COMMAND = (
    "pip install -q -r requirements.txt 2>/dev/null; "
    "pip install -q -e . 2>/dev/null; "
    "python -m pytest -q"
)
_NODE_COMMAND = "npm install --silent 2>/dev/null; npx --yes jest --ci"


@dataclass(frozen=True)
class SandboxConfig:
    """Resource and isolation limits applied to every run."""

    python_image: str = PYTHON_IMAGE
    node_image: str = NODE_IMAGE
    timeout_seconds: int = 600
    mem_limit: str = "2g"
    nano_cpus: int = 2_000_000_000  # 2 CPUs
    pids_limit: int = 512
    allow_network: bool = True


class DockerTestRunner:
    """Callable satisfying :class:`app.agent.nodes.tester.TestRunner`."""

    def __init__(
        self,
        *,
        config: SandboxConfig | None = None,
        docker_client: Any | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self._config = config or SandboxConfig()
        self._docker_client = docker_client
        self._workspace_root = workspace_root or Path(tempfile.gettempdir()) / "devagent-runs"

    def __call__(self, repo_full_name: str, diff: str) -> tuple[int, str]:
        workspace = self._workspace_root / f"run-{uuid.uuid4().hex[:12]}"
        try:
            source = indexer._ensure_repo_checkout(repo_full_name)
            self._clone_local(source, workspace)

            applied, apply_output = _apply_diff(workspace, diff)
            if not applied:
                logger.info("Patch did not apply for %s", repo_full_name)
                return 1, apply_output

            return self._run_in_container(workspace)
        except Exception as error:  # noqa: BLE001 - surfaced as a failing run, not a crash
            logger.exception("Sandboxed run failed for %s", repo_full_name)
            return 1, f"Sandbox error: {error}"
        finally:
            remove_tree(workspace)

    def _client(self) -> Any:
        if self._docker_client is None:
            import docker

            self._docker_client = docker.from_env()
        return self._docker_client

    @staticmethod
    def _clone_local(source: Path, workspace: Path) -> None:
        """Clone the indexed checkout so the patch lands on a disposable copy."""
        workspace.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--local", "--quiet", str(source), str(workspace)],
            check=True,
            capture_output=True,
        )

    def _run_in_container(self, workspace: Path) -> tuple[int, str]:
        image, command = _detect_suite(workspace, self._config)
        client = self._client()
        config = self._config

        logger.info("Running tests in %s", image)
        container = client.containers.run(
            image,
            command=["sh", "-c", command],
            volumes={str(workspace.resolve()): {"bind": WORKDIR, "mode": "rw"}},
            working_dir=WORKDIR,
            detach=True,
            network_disabled=not config.allow_network,
            mem_limit=config.mem_limit,
            nano_cpus=config.nano_cpus,
            pids_limit=config.pids_limit,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
        )

        try:
            result = container.wait(timeout=config.timeout_seconds)
            exit_code = int(result.get("StatusCode", 1))
            output = _decode(container.logs(stdout=True, stderr=True))
        except Exception as error:  # noqa: BLE001 - timeout or daemon hiccup
            output = _decode(_safe_logs(container))
            exit_code = 1
            output += f"\n\nRun aborted after {config.timeout_seconds}s: {error}"
            _safe_kill(container)
        finally:
            _safe_remove(container)

        return exit_code, output


def _apply_diff(workspace: Path, diff: str) -> tuple[bool, str]:
    """Apply the patch, reporting git's own error text when it is rejected."""
    patch_path = workspace / ".devagent.patch"
    patch_path.write_text(diff if diff.endswith("\n") else diff + "\n", encoding="utf-8")

    try:
        for args in (["--check"], []):
            result = subprocess.run(
                ["git", "apply", *args, str(patch_path)],
                cwd=workspace,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, (
                    "The patch could not be applied to the repository.\n\n"
                    f"git apply said:\n{result.stderr.strip()}\n\n"
                    "Regenerate the diff against the current file contents, "
                    "with correct line numbers and at least 3 lines of context."
                )
        return True, ""
    finally:
        patch_path.unlink(missing_ok=True)


def _detect_suite(workspace: Path, config: SandboxConfig) -> tuple[str, str]:
    """Pick the image and command from what the repository actually contains."""
    if (workspace / "package.json").exists():
        return config.node_image, _NODE_COMMAND
    return config.python_image, _PYTHON_COMMAND


def _decode(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw or "")


def _safe_logs(container: Any) -> Any:
    try:
        return container.logs(stdout=True, stderr=True)
    except Exception:  # pragma: no cover - best-effort during cleanup
        return b""


def _safe_kill(container: Any) -> None:
    try:
        container.kill()
    except Exception:  # pragma: no cover - already dead is fine
        logger.debug("Container kill failed; it had likely already exited.")


def _safe_remove(container: Any) -> None:
    try:
        container.remove(force=True)
    except Exception:  # pragma: no cover - best-effort during cleanup
        logger.debug("Container removal failed.")
