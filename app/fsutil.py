"""Filesystem helpers shared by the sandbox and GitHub backends."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


def remove_tree(path: Path) -> None:
    """Delete a directory tree, including read-only files.

    ``shutil.rmtree`` fails on read-only files, which matters here because git
    marks everything under ``.git/objects`` read-only on Windows. Passing
    ``ignore_errors=True`` hides that failure rather than fixing it, so every
    run would silently leak its workspace. Clear the bit and retry instead.
    """

    def on_error(func: Callable[..., Any], target: str, _exc: Any) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:  # pragma: no cover - best effort during cleanup
            logger.debug("Could not remove %s during cleanup.", target)

    if not Path(path).exists():
        return

    # onerror was replaced by onexc in 3.12; the handler signature is compatible
    # for our purposes since we ignore the third argument either way.
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=on_error)
    else:  # pragma: no cover - exercised only on older interpreters
        shutil.rmtree(path, onerror=on_error)
