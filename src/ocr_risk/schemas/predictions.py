"""Verifier scores, before and after calibration.

Raw scores and calibrated probabilities are kept in separate columns on purpose. A raw
score is whatever the model emits on its own scale; a calibrated score claims to be a
probability. Collapsing them would make it impossible to audit, months later, whether a
reported probability was ever actually calibrated, or by which frozen transform.
"""

from __future__ import annotations

from ocr_risk.schemas.base import Probability, RecordModel

__all__ = ["Prediction"]


class Prediction(RecordModel):
    """One verifier's judgement of one candidate edit, within one experimental fold."""

    candidate_id: str
    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    verifier_id: str
    evidence_config: str
    """Ablation configuration the verifier was given, e.g. ``v3_text_conf``."""
    fold_id: str
    """Identifies the held-out engine and protocol; scores from different folds come from
    different fitted models and must never be pooled without it."""
    raw_score: float
    calibrated_score: Probability | None = None
    calibrator_id: str | None = None
    """Identity of the frozen calibration transform applied. ``None`` means the score is
    uncalibrated and must not be interpreted as a probability."""
