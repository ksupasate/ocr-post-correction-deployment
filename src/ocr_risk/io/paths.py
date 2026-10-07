"""Resolution of the repository's data, artifact, and cache roots.

All three are environment-overridable so heavy trees can live off the system volume. Paths
are resolved lazily on each call rather than captured at import time, so tests can redirect
them per-test without module reloading.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ProjectPaths",
    "artifact_root",
    "cache_root",
    "data_root",
    "project_root",
    "raw_ocr_dir",
    "raw_source_dir",
    "resolve_paths",
]


def project_root() -> Path:
    """Repository root, located from this file rather than the working directory."""
    return Path(__file__).resolve().parents[3]


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


def data_root() -> Path:
    return _env_path("OCR_RISK_DATA_ROOT", project_root() / "data")


def artifact_root() -> Path:
    return _env_path("OCR_RISK_ARTIFACT_ROOT", project_root() / "artifacts")


def cache_root() -> Path:
    return _env_path("OCR_RISK_CACHE_DIR", project_root() / "cache")


def raw_source_dir(dataset_id: str) -> Path:
    """Immutable download location for one dataset's source files."""
    return data_root() / "raw" / "source" / dataset_id


def raw_ocr_dir(dataset_id: str, engine_id: str, fingerprint: str) -> Path:
    """Immutable location for verbatim engine responses.

    Keyed by fingerprint so re-running an engine after a version or config change writes a
    sibling directory instead of colliding with, or overwriting, earlier evidence.
    """
    return data_root() / "raw" / "ocr" / dataset_id / engine_id / fingerprint[:8]


class ProjectPaths:
    """Snapshot of the resolved roots, for logging into a run record."""

    __slots__ = ("artifacts", "cache", "data", "project")

    def __init__(self) -> None:
        self.project = project_root()
        self.data = data_root()
        self.artifacts = artifact_root()
        self.cache = cache_root()

    def as_dict(self) -> dict[str, str]:
        return {
            "project": self.project.as_posix(),
            "data": self.data.as_posix(),
            "artifacts": self.artifacts.as_posix(),
            "cache": self.cache.as_posix(),
        }


def resolve_paths() -> ProjectPaths:
    return ProjectPaths()


def manifest_root() -> Path:
    """Where committed manifests live.

    Overridable so a test can freeze and load a partition without writing into the
    repository's tracked manifests -- which a test did, committing a stale 16-document
    partition that then refused to load against the 48-document corpus it was named for.
    """
    return _env_path("OCR_RISK_MANIFEST_ROOT", project_root() / "manifests")


def partition_manifest_path(partition_id: str) -> Path:
    """Where a frozen document partition lives.

    Keyed by ``splits.partition_id``, not by experiment name. Experiments that must be
    compared -- the two arms of a transfer contrast -- have to share one partition, and
    experiments over different corpora must not; the name of the run is the wrong key for
    both.
    """
    return manifest_root() / "splits" / f"{partition_id}.document_partition.json"
