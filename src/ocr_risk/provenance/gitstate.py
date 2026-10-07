"""Capture of code identity at run time."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ocr_risk.io.hashing import hash_str
from ocr_risk.io.paths import project_root
from ocr_risk.schemas.run_record import GitState

__all__ = ["capture_git_state"]

_TIMEOUT = 15


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a read-only git command, returning ``None`` when git or the repo is absent."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def capture_git_state(root: Path | None = None) -> GitState:
    """Record commit, branch, and working-tree cleanliness.

    A dirty tree is recorded rather than rejected. Refusing to run on uncommitted changes
    would push people toward running outside the pipeline, where nothing is recorded at
    all; capturing a hash of the diff at least makes the run identifiable, and downstream
    tooling can weight the claim accordingly.
    """
    cwd = root or project_root()
    commit = _git(["rev-parse", "HEAD"], cwd) or "unknown"
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or "unknown"
    status = _git(["status", "--porcelain"], cwd)
    dirty = bool(status)
    diff_sha256 = None
    if dirty:
        diff = _git(["diff", "HEAD"], cwd)
        combined = f"{status}\n{diff or ''}"
        diff_sha256 = hash_str(combined)
    return GitState(
        commit=commit,
        branch=branch,
        dirty=dirty,
        diff_sha256=diff_sha256,
        remote=_git(["config", "--get", "remote.origin.url"], cwd),
    )
