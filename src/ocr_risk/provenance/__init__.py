"""Run records, git/environment capture, and the artifact provenance DAG.

Layer 1. May import ``io.hashing`` and ``io.paths``; must **not** import
``io.artifacts``, which builds on this module.
"""

from __future__ import annotations

from ocr_risk.provenance.dag import LineageNode, LineageReport, build_lineage, render_lineage
from ocr_risk.provenance.envcapture import (
    capture_environment,
    code_sha256,
    installed_package_digest,
)
from ocr_risk.provenance.gitstate import capture_git_state
from ocr_risk.provenance.record import build_run_record, make_run_id, utc_now_iso

__all__ = [
    "LineageNode",
    "LineageReport",
    "build_lineage",
    "build_run_record",
    "capture_environment",
    "capture_git_state",
    "code_sha256",
    "installed_package_digest",
    "make_run_id",
    "render_lineage",
    "utc_now_iso",
]
