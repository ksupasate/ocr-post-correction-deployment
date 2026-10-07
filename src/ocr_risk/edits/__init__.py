"""Correction sites, edit application, and the correction-outcome taxonomy.

Layer 5.
"""

from __future__ import annotations

from ocr_risk.edits.apply import AppliedEdit, apply_edits
from ocr_risk.edits.outcome import (
    BENEFICIAL_OUTCOMES,
    classify_accepted,
    classify_rejected,
    distance,
    is_beneficial,
    is_harmful,
)
from ocr_risk.edits.sites import build_sites, site_kind_for

__all__ = [
    "BENEFICIAL_OUTCOMES",
    "AppliedEdit",
    "apply_edits",
    "build_sites",
    "classify_accepted",
    "classify_rejected",
    "distance",
    "is_beneficial",
    "is_harmful",
    "site_kind_for",
]
