"""Accept/preserve decisions produced by the risk controller.

One row per (site, method, fold, epsilon). The threshold ``tau`` is carried on every row
because it is a fitted quantity selected on the calibration split — recording it here is
what lets an auditor confirm the same threshold was applied to every test row, rather than
re-derived per row from data it should not have seen.
"""

from __future__ import annotations

from pydantic import Field

from ocr_risk.schemas.base import RecordModel
from ocr_risk.schemas.enums import DecisionAction

__all__ = ["Decision"]


class Decision(RecordModel):
    """The action taken at one correction site under one risk tolerance."""

    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    method_id: str
    """Verifier + evidence configuration + calibrator, as one comparable method identity."""
    fold_id: str
    epsilon: float = Field(ge=0.0, le=1.0)
    """Target upper bound on harmful accepted-edit risk."""
    tau: float
    """Score threshold selected on the calibration split for this ``epsilon``."""
    action: DecisionAction
    accepted_candidate_id: str | None = None
    accepted_score: float | None = None
    n_candidates_considered: int = Field(ge=0)
