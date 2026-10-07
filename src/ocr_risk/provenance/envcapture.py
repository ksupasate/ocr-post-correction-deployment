"""Capture of the execution environment: interpreter, platform, hardware, dependencies."""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from pathlib import Path

from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.io.paths import project_root
from ocr_risk.schemas.run_record import EnvironmentState

__all__ = ["capture_environment", "code_sha256", "installed_package_digest"]

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def installed_package_digest() -> str:
    """Order-independent hash over installed distribution name/version pairs.

    Cheaper and more portable than embedding a full freeze, and sufficient to detect that
    the dependency set changed between a run and its attempted reproduction.
    """
    pairs = sorted(
        (str(dist.metadata["Name"]), str(dist.version))
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    )
    return canonical_hash(pairs)


def code_sha256(source_root: Path | None = None) -> str:
    """Hash over the package source tree.

    Complements the git commit: it also changes for uncommitted edits, so a run whose
    ``git.dirty`` is true still carries a precise identity for the logic that ran.
    """
    root = source_root or (project_root() / "src" / "ocr_risk")
    if not root.exists():
        return "unknown"
    entries = [
        (p.relative_to(root).as_posix(), file_sha256(p))
        for p in sorted(root.rglob("*.py"))
        if "__pycache__" not in p.parts
    ]
    return canonical_hash(entries)


def _total_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def capture_environment() -> EnvironmentState:
    """Snapshot everything that could make a rerun produce different bytes."""
    lock = project_root() / "uv.lock"
    return EnvironmentState(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        total_memory_bytes=_total_memory_bytes(),
        uv_lock_sha256=file_sha256(lock) if lock.exists() else None,
        package_digest=installed_package_digest(),
        thread_limits={var: os.environ[var] for var in _THREAD_VARS if var in os.environ},
    )
