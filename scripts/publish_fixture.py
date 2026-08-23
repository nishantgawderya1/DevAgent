"""Publish a fixture directory to GitHub as a standalone repository.

The fixtures live inside this repo so they are version-controlled, but DevAgent
has to be pointed at a real GitHub repo with real issues. This copies one out,
initialises it as its own repository, tags the clean state as ``pristine`` so
runs can be reset, and pushes it.

    python scripts/publish_fixture.py pagination-bug --repo owner/devagent-fixture

Nothing is pushed without ``--push``; the default is a dry run that prints the
commands so you can see what it would do first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

# Not part of the fixture's own source; they describe it to us, not to the repo.
EXCLUDED = {"ISSUE.md", "__pycache__", ".pytest_cache"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", help="Directory name under fixtures/")
    parser.add_argument("--repo", required=True, help="Target repo as owner/name")
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--push", action="store_true", help="Actually push (default is a dry run)"
    )
    args = parser.parse_args(argv)

    source = FIXTURES_DIR / args.fixture
    if not source.is_dir():
        available = ", ".join(sorted(p.name for p in FIXTURES_DIR.iterdir() if p.is_dir()))
        parser.error(f"No fixture {args.fixture!r}. Available: {available}")

    token = os.getenv("GITHUB_TOKEN")
    if args.push and not token:
        parser.error("GITHUB_TOKEN is required to push.")

    workspace = Path(tempfile.mkdtemp(prefix="devagent-fixture-"))
    try:
        staged = workspace / args.fixture
        shutil.copytree(
            source, staged, ignore=shutil.ignore_patterns(*EXCLUDED)
        )

        commands = [
            ["git", "init", f"--initial-branch={args.branch}", "--quiet"],
            ["git", "add", "-A"],
            ["git", "commit", "--quiet", "-m", "Initial commit"],
            ["git", "tag", "pristine"],
            [
                "git",
                "remote",
                "add",
                "origin",
                f"https://x-access-token:{token or '$GITHUB_TOKEN'}@github.com/{args.repo}.git",
            ],
            ["git", "push", "--force", "origin", args.branch],
            ["git", "push", "--force", "origin", "pristine"],
        ]

        if not args.push:
            print(f"Dry run — would publish {source} to {args.repo}:\n")
            for command in commands:
                print("   ", " ".join(_redact(part, token) for part in command))
            print("\nRe-run with --push to do it. Create the repo on GitHub first.")
            return 0

        for command in commands:
            print("  ", " ".join(_redact(part, token) for part in command))
            subprocess.run(command, cwd=staged, check=True, capture_output=True, text=True)

        print(f"\nPublished to https://github.com/{args.repo}")
        print(f"Now open {source / 'ISSUE.md'} as an issue there.")
        return 0
    except subprocess.CalledProcessError as error:
        print(f"\nFailed: {error.stderr.strip() or error}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _redact(part: str, token: str | None) -> str:
    return part.replace(token, "***") if token and token in part else part


if __name__ == "__main__":
    raise SystemExit(main())
