"""Derived confidence features, as a **fitted** transformer.

Any statistic that needs a population — a z-score, a rank, a quantile — must be estimated
on the fit split and then frozen. Computing it over the pooled frame is leakage vector L3:
subtle, easy to write by accident, and undetectable downstream because the resulting
number looks perfectly reasonable.

Making the fit explicit is the point. ``transform`` before ``fit`` raises rather than
falling back to whatever data is in hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ocr_risk.canonical.confidence import normalize_confidence
from ocr_risk.schemas.evidence import ConfidenceFeatures

__all__ = ["ConfidenceFeaturizer", "NotFittedError", "unfitted_confidence_features"]


class NotFittedError(RuntimeError):
    """Raised when a transformer is used before its statistics are estimated."""


@dataclass(slots=True)
class ConfidenceFeaturizer:
    """Normalizes and standardizes native confidences within each engine.

    Standardization is **per engine**, not global: Tesseract's 0-100 word confidence and
    PaddleOCR's 0-1 score have different distributions and different meanings, so pooling
    them into one z-score would compare incomparable quantities — which is precisely the
    cross-engine shift this project studies rather than something to normalize away.
    """

    _means: dict[str, float] = field(default_factory=dict)
    _stds: dict[str, float] = field(default_factory=dict)
    _fitted: bool = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def engines(self) -> tuple[str, ...]:
        return tuple(sorted(self._means))

    def fit(self, samples: Sequence[tuple[str, float | None, str | None]]) -> None:
        """Estimate per-engine statistics from ``(engine_id, native_conf, scale)`` triples.

        The caller must pass fit-split rows only. This signature exists so that obligation
        is visible at the call site instead of hidden inside a frame filter.
        """
        buckets: dict[str, list[float]] = {}
        for engine_id, value, scale in samples:
            normalized = normalize_confidence(value, scale)
            if normalized is not None:
                buckets.setdefault(engine_id, []).append(normalized)

        self._means = {}
        self._stds = {}
        for engine_id, values in buckets.items():
            array = np.asarray(values, dtype=np.float64)
            self._means[engine_id] = float(array.mean())
            # A degenerate spread would make every z-score infinite; 1.0 leaves the
            # centered value unscaled, which is the honest fallback.
            std = float(array.std(ddof=0))
            self._stds[engine_id] = std if std > 1e-12 else 1.0
        self._fitted = True

    def transform(
        self,
        native_confidences: Sequence[float | None],
        conf_scale: str | None,
        engine_id: str,
    ) -> ConfidenceFeatures:
        """Derive features for one site's spans."""
        if not self._fitted:
            msg = (
                "ConfidenceFeaturizer.transform called before fit(). Population statistics "
                "must come from the fit split; estimating them here would pool the split "
                "being evaluated into its own features."
            )
            raise NotFittedError(msg)

        n_spans = len(native_confidences)
        normalized = [
            v
            for v in (normalize_confidence(c, conf_scale) for c in native_confidences)
            if v is not None
        ]
        if not normalized:
            return ConfidenceFeatures(n_spans=n_spans, has_native_confidence=False)

        raw = [c for c in native_confidences if c is not None]
        mean = self._means.get(engine_id)
        std = self._stds.get(engine_id, 1.0)
        minimum = min(normalized)
        average = float(np.mean(normalized))

        return ConfidenceFeatures(
            native_min=min(raw) if raw else None,
            native_mean=float(np.mean(raw)) if raw else None,
            conf_normalized=minimum,
            # The weakest span in a multi-span site drives the risk, so the site-level
            # summary is the minimum rather than the mean.
            conf_rank_in_line=None,
            conf_zscore_within_engine=(average - mean) / std if mean is not None else None,
            n_spans=n_spans,
            has_native_confidence=True,
        )


def unfitted_confidence_features(
    native_confidences: Sequence[float | None], conf_scale: str | None
) -> ConfidenceFeatures:
    """Every confidence feature that needs no fitted population.

    Used when a bundle is built outside any fold, so it cannot carry a z-score computed
    against a population that includes the engine some fold will hold out. The z-score is
    filled in per fold by :meth:`ConfidenceFeaturizer.transform`.
    """
    normalized = [
        v
        for v in (normalize_confidence(c, conf_scale) for c in native_confidences)
        if v is not None
    ]
    n_spans = len(native_confidences)
    if not normalized:
        return ConfidenceFeatures(n_spans=n_spans, has_native_confidence=False)
    raw = [c for c in native_confidences if c is not None]
    return ConfidenceFeatures(
        native_min=min(raw) if raw else None,
        native_mean=float(np.mean(raw)) if raw else None,
        conf_normalized=min(normalized),
        conf_rank_in_line=None,
        conf_zscore_within_engine=None,
        n_spans=n_spans,
        has_native_confidence=True,
    )
