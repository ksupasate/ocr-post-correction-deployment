"""Correction sites: the decision points of the experiment.

A site is where the system must choose PRESERVE or CORRECT. Sites where the OCR is already
correct (``d_before == 0``) are enumerated deliberately — without them, overcorrection is
unobservable, and a benchmark that cannot observe overcorrection cannot measure harm.
"""

from __future__ import annotations

from pydantic import Field

from ocr_risk.schemas.base import BBox, RecordModel
from ocr_risk.schemas.enums import SiteKind

__all__ = ["CorrectionSite"]


class CorrectionSite(RecordModel):
    """One candidate-editable region of a page's OCR, paired with its ground truth.

    ``d_before`` is a *raw* character edit distance, not a normalized rate: the comparison
    that matters is ``d_after`` against ``d_before`` for the same ``gt_text``, so dividing
    both by the same constant would only add noise.
    """

    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    alignment_ids: tuple[str, ...]
    ocr_span_ids: tuple[str, ...]
    gt_token_ids: tuple[str, ...]
    ocr_text: str
    """``O`` — the original OCR span text."""
    gt_text: str
    """``G`` — the aligned ground-truth text."""
    d_before: int = Field(ge=0)
    """``lev(O, G)`` in characters."""
    site_kind: SiteKind
    evaluable: bool
    """False for sites built from ambiguous or unresolved alignments. Such sites are kept
    in the artifact so their volume is reportable, but contribute to no metric."""
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    """Offsets into the linearized OCR stream for this (document, engine)."""
    reading_order_start: int = Field(ge=0)
    bbox: BBox | None = None
    min_align_confidence: float
    """Weakest component confidence in this site; propagated so downstream analysis can
    stratify results by alignment quality instead of assuming it is uniform."""
