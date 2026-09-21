"""Open a review PR. Never push to main."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def propose_pull_request(
    repo_root: Path,
    experiment_id: str,
    title: str,
    body: str,
    open_pr: bool,
) -> str:
    branch = f"agent/{experiment_id}"
    if not open_pr:
        return f"dry-run:{branch}"
    if not shutil.which("gh"):
        return f"dry-run:{branch}:gh-not-installed"
    result = subprocess.run(
        ["gh", "pr", "create", "--title", title, "--body", body, "--head", branch],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"dry-run:{branch}: {result.stderr.strip() or result.stdout.strip()}"
    return result.stdout.strip()
