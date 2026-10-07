"""Content hashing, the write-once raw store, parquet IO, manifests, and the artifact store.

Layer 1. Intra-layer ordering is fixed: ``io.hashing`` and ``io.paths`` are leaves,
``provenance`` builds on them, and ``io.artifacts`` builds on ``provenance``. The reverse
edge (``provenance`` importing ``io.artifacts``) is forbidden and tested.
"""

from __future__ import annotations

from ocr_risk.io.artifacts import ArtifactStore, ImmutableRunError, RunWriter
from ocr_risk.io.hashing import (
    canonical_hash,
    canonical_json,
    canonicalize,
    file_sha256,
    hash_str,
    sha256_of_bytes,
    short,
    stable_string_set_hash,
)
from ocr_risk.io.parquet import read_dataframe, read_records, read_table, write_records, write_table
from ocr_risk.io.paths import (
    artifact_root,
    cache_root,
    data_root,
    project_root,
    raw_ocr_dir,
    raw_source_dir,
    resolve_paths,
)
from ocr_risk.io.raw_store import ImmutableWriteError, RawStore

__all__ = [
    "ArtifactStore",
    "ImmutableRunError",
    "ImmutableWriteError",
    "RawStore",
    "RunWriter",
    "artifact_root",
    "cache_root",
    "canonical_hash",
    "canonical_json",
    "canonicalize",
    "data_root",
    "file_sha256",
    "hash_str",
    "project_root",
    "raw_ocr_dir",
    "raw_source_dir",
    "read_dataframe",
    "read_records",
    "read_table",
    "resolve_paths",
    "sha256_of_bytes",
    "short",
    "stable_string_set_hash",
    "write_records",
    "write_table",
]
