from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def make_git_repo():
    """Build a real git repository on disk.

    The sandbox and GitHub backends both drive git directly, so their tests run
    against real repositories rather than mocks — patch application and pushes
    are exactly the parts worth not faking.
    """

    def _make(path: Path, files: dict[str, str]) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            file_path = path / name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        for args in (
            ["init", "--quiet", "--initial-branch=main"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "Test"],
            ["add", "-A"],
            ["commit", "--quiet", "-m", "initial"],
        ):
            subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
        return path

    return _make
