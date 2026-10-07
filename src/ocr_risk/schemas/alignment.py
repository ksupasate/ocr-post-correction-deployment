"""Alignment between canonical OCR spans and ground-truth tokens.

One record per connected component of the OCR-to-GT bipartite graph. Ambiguity is a
first-class outcome: a component the aligner cannot resolve is stored with a reason code
and excluded from evaluation, never forced into a match to raise coverage.
"""

from __future__ import annotations

from pydantic import Field

from ocr_risk.schemas.base import BBox, RecordModel, UnitInterval
from ocr_risk.schemas.enums import AlignmentRelation, AlignmentStatus

__all__ = ["AlignmentRecord", "RegionAnchor"]


class RegionAnchor(RecordModel):
    """A matched (OCR region, GT region) pair from geometric anchoring.

    Produced only when the ground truth carries geometry. Anchors bound the character-level
    dynamic program to a small window, which is what keeps full-page alignment tractable.
    """

    region_id: str
    document_id: str
    engine_id: str
    ocr_line_ids: tuple[str, ...]
    gt_line_ids: tuple[str, ...]
    iou: float | None = None
    center_distance: float | None = None
    cost: float
    """Assignment cost actually paid; anchors above the ceiling are dropped, not kept."""


class AlignmentRecord(RecordModel):
    """One OCR-to-GT correspondence component with its trust assessment.

    ``align_confidence`` combines character agreement, geometric plausibility, and the
    margin over the runner-up assignment. When ground truth has no geometry the geometric
    term is dropped and the remaining weights are renormalized, which is recorded in
    ``diagnostics`` rather than silently treated as zero evidence.
    """

    alignment_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    ocr_span_ids: tuple[str, ...]
    gt_token_ids: tuple[str, ...]
    relation: AlignmentRelation
    status: AlignmentStatus
    ocr_text: str
    """Denormalized concatenation of the aligned OCR spans, for auditability."""
    gt_text: str
    """Denormalized concatenation of the aligned GT tokens, for auditability."""
    align_confidence: UnitInterval
    char_agreement: UnitInterval
    geom_score: float | None = None
    """IoU of the OCR and GT bounding-box unions; ``None`` when GT has no geometry."""
    uniqueness_margin: UnitInterval
    """Normalized gap between the chosen alignment and the best alternative. A low margin
    means the aligner had a near-tie, which is exactly when a forced match is dangerous."""
    edit_distance: int = Field(ge=0)
    region_id: str | None = None
    bbox: BBox | None = None
    reason_code: str | None = None
    """Why a component is not ``RESOLVED``, e.g. ``below_confidence_floor``."""
    diagnostics: dict[str, str] = Field(default_factory=dict)
