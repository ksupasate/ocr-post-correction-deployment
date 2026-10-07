"""Discrimination metrics: can the scores rank a beneficial edit above a harmful one?

Separate from ``calibration.py`` on purpose. The H1 pilot found the two moving in opposite
directions under engine transfer — Murphy calibration error degraded on 0 of 4 engines
while Brier degraded on 4 of 4 and resolution fell 9.6% to 35.2% — and a reader who has
only "calibration" and "Brier" cannot tell those apart. Calibration asks whether a score
of 0.8 means 80%; discrimination asks whether the safe edits score above the harmful ones
at all. Recalibration fixes the first and cannot touch the second.

Everything here is computable from a saved predictions table alone, months later, with no
model. ``metrics/`` may not import ``verify/``, ``candidates/``, or ``engines/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DiscriminationReport",
    "average_precision",
    "discrimination_report",
    "resolution",
    "roc_auc",
    "score_separation",
]

FloatArray = NDArray[np.float64]


def _clean(scores: FloatArray, positive: FloatArray) -> tuple[FloatArray, FloatArray]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(positive, dtype=np.float64).ravel()
    if scores.shape != positive.shape:
        msg = f"scores {scores.shape} and labels {positive.shape} disagree in length"
        raise ValueError(msg)
    return scores, positive


def roc_auc(scores: FloatArray, positive: FloatArray) -> float:
    """Probability a random positive outscores a random negative, ties counting a half.

    Computed from the rank sum (the Mann-Whitney U identity) rather than by sweeping
    thresholds, because ties are the common case here: a verifier that emits a constant
    score must score exactly 0.5, and a threshold sweep with a ``>=`` comparison silently
    reports 1.0 for it. The accept-everything reference does emit a constant score.

    Returns NaN when one class is absent — the quantity is undefined, and 0.5 would be a
    measurement.
    """
    scores, positive = _clean(scores, positive)
    n_pos = float(positive.sum())
    n_neg = float(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranked = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    index = 0
    while index < ranked.size:
        stop = index
        while stop + 1 < ranked.size and ranked[stop + 1] == ranked[index]:
            stop += 1
        # Average rank within a tie group; 1-based, as the U statistic expects.
        ranks[order[index : stop + 1]] = 0.5 * (index + stop) + 1.0
        index = stop + 1

    rank_sum = float(ranks[positive > 0].sum())
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def average_precision(scores: FloatArray, positive: FloatArray) -> float:
    """Area under the precision-recall curve, as the step-wise sum over recall gains.

    Reported beside ROC AUC because the classes here are far from balanced — the natural
    candidate pool is 60-85% harmful depending on the policy — and ROC AUC is insensitive
    to the base rate in a way that flatters a method on the rare class.
    """
    scores, positive = _clean(scores, positive)
    n_pos = float(positive.sum())
    if n_pos == 0 or positive.size == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    labels = positive[order]
    ordered_scores = scores[order]
    true_positives = np.cumsum(labels)
    seen = np.arange(1, labels.size + 1, dtype=np.float64)

    # Tied scores are one operating point: a threshold cannot separate them, so precision
    # must be read at the END of each tie group, not partway through it.
    last_of_group = np.append(ordered_scores[:-1] != ordered_scores[1:], True)
    precision = true_positives[last_of_group] / seen[last_of_group]
    recall = true_positives[last_of_group] / n_pos
    recall_gain = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * recall_gain))


def score_separation(scores: FloatArray, positive: FloatArray) -> float:
    """Standardized mean difference between the two classes (Cohen's d, pooled SD).

    A magnitude to accompany the rank statistics: AUC saturates, and two verifiers with
    the same AUC can put the classes very differently far apart. NaN when either class is
    absent or both are degenerate.
    """
    scores, positive = _clean(scores, positive)
    good = scores[positive > 0]
    bad = scores[positive <= 0]
    if good.size < 2 or bad.size < 2:
        return float("nan")
    pooled = np.sqrt(
        ((good.size - 1) * good.var(ddof=1) + (bad.size - 1) * bad.var(ddof=1))
        / (good.size + bad.size - 2)
    )
    if pooled <= 1e-12:
        return 0.0
    return float((good.mean() - bad.mean()) / pooled)


def resolution(base_rate: float, refinement: float) -> float:
    """``base(1 - base) - refinement``: how much uncertainty the scores actually removed.

    The Murphy decomposition's refinement term is what is left *unexplained* within bins,
    so it falls as a model gets sharper. Resolution is the complement, and is the term to
    quote when the claim is "transfer cost the model its ability to tell the classes
    apart" — it moves in the intuitive direction.
    """
    return base_rate * (1.0 - base_rate) - refinement


@dataclass(frozen=True, slots=True)
class DiscriminationReport:
    """Every discrimination endpoint for one (arm, engine, verifier), plus its n."""

    n: int
    n_positive: int
    base_rate: float
    roc_auc: float
    average_precision: float
    score_separation: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "n_positive": self.n_positive,
            "base_rate": self.base_rate,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "score_separation": self.score_separation,
        }


def discrimination_report(scores: FloatArray, positive: FloatArray) -> DiscriminationReport:
    """All four endpoints at once, so a table row is one call.

    ``positive`` is the **safe** class, matching the orientation of the calibrated score
    everywhere else in the repository: the model predicts "accepting this edit is safe".
    """
    scores, positive = _clean(scores, positive)
    n_positive = int(positive.sum())
    return DiscriminationReport(
        n=int(scores.size),
        n_positive=n_positive,
        base_rate=float(positive.mean()) if scores.size else float("nan"),
        roc_auc=roc_auc(scores, positive),
        average_precision=average_precision(scores, positive),
        score_separation=score_separation(scores, positive),
    )
