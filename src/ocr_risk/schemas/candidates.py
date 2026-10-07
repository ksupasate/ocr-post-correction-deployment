"""Proposed edits and their counterfactual outcome labels.

A :class:`Candidate` is the research object of this project: the proposed edit ``O -> Y``
whose acceptance the verifier must judge. A :class:`CandidateLabel` records what would
happen if that edit were accepted, computed from ground truth.

Labels are the *only* place ground truth enters the pipeline downstream of alignment.
Evidence construction never sees them.
"""

from __future__ import annotations

from pydantic import Field

from ocr_risk.schemas.base import RecordModel
from ocr_risk.schemas.enums import CandidatePool, OutcomeIfAccepted, OutcomeIfRejected

__all__ = ["Candidate", "CandidateLabel"]


class Candidate(RecordModel):
    """One proposed correction for a site, from one generator.

    Generators are ground-truth-blind by construction, so a candidate carries no signal
    about whether it is correct. Multiple candidates per site are expected; the decision
    policy accepts at most one.
    """

    candidate_id: str
    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    candidate_text: str
    """``Y`` — the proposed replacement for the site's OCR text."""
    generator_id: str
    generator_version: str
    generator_rank: int = Field(ge=0)
    """Rank within this generator's n-best list for the site (0 is best)."""
    generator_score: float | None = None
    """The generator's own score on its own scale. Not comparable across generators and
    never used as a verifier score."""
    is_synthetic_hard_negative: bool = False
    """Whether this candidate was synthesized adversarially rather than proposed from OCR.

    The **stored** fact. Read it through :attr:`pool`, which gives it the type and the name
    the denominators are actually about."""
    hard_negative_family: str | None = None
    """e.g. ``numeric_magnitude``, ``unit_substitution``, ``homoglyph``."""
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def pool(self) -> CandidatePool:
        """Which population this candidate belongs to. Headline denominators use NATURAL.

        Derived, not stored. It was briefly a second column kept in step with
        ``is_synthetic_hard_negative`` by a validator, which is two records of one fact and
        exactly how a natural denominator quietly acquires adversarial members — one call
        site writes the boolean, another reads the enum, and nothing raises. A property
        cannot disagree with itself, and it leaves every artifact written before the name
        existed readable without migration.
        """
        return CandidatePool.CHALLENGE if self.is_synthetic_hard_negative else CandidatePool.NATURAL


class CandidateLabel(RecordModel):
    """Counterfactual outcome of accepting (or rejecting) a candidate.

    Harmfulness is deliberately *not* stored: it is derived at analysis time from
    ``outcome_if_accepted`` and the configured
    :class:`~ocr_risk.schemas.enums.HarmPolicy`. Materializing it would let the stored
    booleans drift from the enum, and would privilege one policy over the sensitivity
    analysis that must accompany it.
    """

    candidate_id: str
    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    d_before: int = Field(ge=0)
    """``lev(O, G)``."""
    d_after: int = Field(ge=0)
    """``lev(Y, G)``."""
    delta: int
    """``d_before - d_after``. Positive means the edit moved toward ground truth."""
    outcome_if_accepted: OutcomeIfAccepted
    outcome_if_rejected: OutcomeIfRejected
    gt_text: str
    """Denormalized for auditability of the label itself. Must never be joined into an
    evidence bundle — see ``tests/architecture/test_evidence_has_no_gt.py``."""
