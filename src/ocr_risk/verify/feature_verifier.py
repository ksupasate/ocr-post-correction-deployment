"""Feature-based verifiers: the V0-V6 ladder, on CPU, with no torch.

Two implementations cover the whole ablation:

- :class:`AcceptAllVerifier` for V0, the degenerate reference point.
- :class:`FeatureVerifier` for V1-V6, a logistic regression or gradient boosting over the
  feature blocks the ablation permits.

Using a shallow model here is deliberate and is what makes the bootstrap honest: the whole
pipeline — masking, fitting, calibration, threshold selection, risk accounting — runs end
to end on CPU, so CI exercises the real code path rather than a mock. A learned encoder
replaces :func:`~ocr_risk.verify.featurizers.image_block` without touching anything else,
and the V3-vs-V6 comparison is structurally identical either way.

Scores are oriented so that **higher means safer to accept**: the model predicts harm, and
the score is its complement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ocr_risk.evidence.fields import get_evidence_config
from ocr_risk.verify.base import FloatArray, ScoredBatch, VerificationInput
from ocr_risk.verify.featurizers import blocks_for
from ocr_risk.verify.registry import register_verifier

__all__ = ["AcceptAllVerifier", "FeatureVerifier", "NotFittedVerifierError"]


class NotFittedVerifierError(RuntimeError):
    """Raised when a verifier scores before it has been fitted."""


@register_verifier("accept_all")
class AcceptAllVerifier:
    """V0: accept everything.

    Not a straw man — it is the operating point every risk-coverage curve is read
    against. Its coverage is 1.0 and its risk is whatever the candidate pool carries, which
    is the number a proposed method has to improve on.
    """

    def __init__(self, verifier_id: str = "v0_accept_all", evidence_config: str = "v0") -> None:
        self.verifier_id = verifier_id
        self.evidence_config = evidence_config
        self.required_evidence = get_evidence_config(evidence_config).fields

    def fit(self, inputs: Sequence[VerificationInput], harmful: Sequence[bool]) -> None:
        return None

    def score(self, inputs: Sequence[VerificationInput]) -> ScoredBatch:
        return ScoredBatch(
            candidate_ids=tuple(i.candidate_id for i in inputs),
            scores=np.ones(len(inputs), dtype=np.float64),
        )


@register_verifier("feature_logistic")
class FeatureVerifier:
    """V1-V6: a shallow model over the permitted feature blocks."""

    def __init__(
        self,
        verifier_id: str,
        evidence_config: str,
        model: str = "logistic",
        C: float = 1.0,
        max_iter: int = 2000,
        class_weight: str | None = "balanced",
        random_state: int = 0,
    ) -> None:
        self.verifier_id = verifier_id
        self.evidence_config = evidence_config
        self.required_evidence = get_evidence_config(evidence_config).fields
        self.model_kind = model
        self.C = C
        self.max_iter = max_iter
        # Harmful candidates outnumber beneficial ones roughly 50:1 in this pool, because
        # most sites are already correct. Without reweighting the model would minimize
        # loss by rejecting everything, which is a degenerate solution, not a verifier.
        self.class_weight = class_weight
        self.random_state = random_state

        self._model: LogisticRegression | HistGradientBoostingClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: tuple[str, ...] = ()
        self._constant_score: float | None = None

    @property
    def fitted(self) -> bool:
        return self._model is not None or self._constant_score is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def _matrix(self, inputs: Sequence[VerificationInput]) -> FloatArray:
        rows: list[FloatArray] = []
        names: tuple[str, ...] = ()
        for item in inputs:
            blocks = blocks_for(item.bundle, self.required_evidence, item.crop)
            if not blocks:
                rows.append(np.zeros(1, dtype=np.float64))
                names = ("constant",)
                continue
            rows.append(np.concatenate([b.values for b in blocks]))
            names = tuple(name for block in blocks for name in block.names)
        if not self._feature_names:
            self._feature_names = names
        return np.vstack(rows) if rows else np.zeros((0, max(len(names), 1)), dtype=np.float64)

    def fit(self, inputs: Sequence[VerificationInput], harmful: Sequence[bool]) -> None:
        """Fit on fit-scope rows. ``harmful`` is the target; scores invert it."""
        if len(inputs) != len(harmful):
            msg = f"{len(inputs)} inputs but {len(harmful)} labels"
            raise ValueError(msg)

        y = np.asarray(harmful, dtype=np.int32)
        if y.size == 0 or len(np.unique(y)) < 2:
            # One class only: there is nothing to learn, and a fitted model would encode
            # an accident of the split. Fall back to a constant and say so.
            self._constant_score = 0.5 if y.size == 0 else float(1 - y[0])
            self._model = None
            return

        X = self._matrix(inputs)
        # The scaler is fitted here, inside fit scope, and frozen. Standardizing at score
        # time against the batch being scored would pool the evaluation split into its own
        # features (leakage vector L3).
        self._scaler = StandardScaler().fit(X)
        scaled = self._scaler.transform(X)

        if self.model_kind == "gradient_boosting":
            self._model = HistGradientBoostingClassifier(
                max_iter=200, random_state=self.random_state
            ).fit(scaled, y)
        else:
            self._model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
            ).fit(scaled, y)
        self._constant_score = None

    def score(self, inputs: Sequence[VerificationInput]) -> ScoredBatch:
        if not self.fitted:
            msg = f"verifier {self.verifier_id!r} scored before fit()"
            raise NotFittedVerifierError(msg)

        ids = tuple(i.candidate_id for i in inputs)
        if not inputs:
            return ScoredBatch(candidate_ids=(), scores=np.zeros(0, dtype=np.float64))

        if self._model is None or self._scaler is None:
            constant = self._constant_score if self._constant_score is not None else 0.5
            return ScoredBatch(
                candidate_ids=ids, scores=np.full(len(inputs), constant, dtype=np.float64)
            )

        scaled = self._scaler.transform(self._matrix(inputs))
        harm_probability = self._model.predict_proba(scaled)[:, 1]
        # Higher score = safer to accept.
        return ScoredBatch(candidate_ids=ids, scores=1.0 - harm_probability.astype(np.float64))
