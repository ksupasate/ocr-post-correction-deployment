"""The verifier contract.

A verifier answers one question: *should this proposed edit be accepted?* It scores masked
evidence bundles, and it is fitted only through a :class:`~ocr_risk.splits.plan.FitView`.

The signature is the enforcement. ``fit`` takes a view, not a frame, so a verifier cannot
be handed the evaluation split by mistake — and because ``verify`` may not import
``metrics`` (layering), it cannot reach the evaluation labels by another route either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ocr_risk.schemas.enums import EvidenceField
from ocr_risk.schemas.evidence import EvidenceBundle

__all__ = ["ScoredBatch", "VerificationInput", "Verifier"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class VerificationInput:
    """One masked bundle plus the crop it is allowed to read, if any."""

    bundle: EvidenceBundle
    crop: Path | None = None

    @property
    def candidate_id(self) -> str:
        return self.bundle.candidate_id


@dataclass(frozen=True, slots=True)
class ScoredBatch:
    """Scores aligned to candidate ids, so a score can never drift from its row."""

    candidate_ids: tuple[str, ...]
    scores: FloatArray

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != self.scores.size:
            msg = f"{len(self.candidate_ids)} ids but {self.scores.size} scores"
            raise ValueError(msg)


@runtime_checkable
class Verifier(Protocol):
    """Contract every edit verifier satisfies."""

    verifier_id: str
    evidence_config: str
    required_evidence: frozenset[EvidenceField]

    def fit(self, inputs: Sequence[VerificationInput], harmful: Sequence[bool]) -> None:
        """Fit on fit-scope rows only.

        The caller obtains ``inputs`` and ``harmful`` from ``SplitPlan.view(role=FIT)``;
        this signature takes them already extracted so the verifier never touches a frame
        and cannot widen its own scope.
        """
        ...

    def score(self, inputs: Sequence[VerificationInput]) -> ScoredBatch:
        """Score candidates. Higher means more confident the edit is safe to accept."""
        ...
