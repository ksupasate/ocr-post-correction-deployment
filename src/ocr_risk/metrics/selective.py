"""Selective-prediction metrics: the risk-coverage frontier.

The headline comparison of this project is read off these curves: *at a fixed harmful-edit
tolerance, which method safely automates the most genuine repair?*

Two properties are enforced rather than assumed:

- **Coverage is non-increasing in the threshold.** Raising the bar can only reject more.
- **Risk is not monotone**, and pretending otherwise is a real error. Selective risk can
  rise as coverage shrinks, because the last edits a model is most confident about are not
  guaranteed to be its safest. ``Coverage@Risk`` therefore searches the whole curve for
  the largest coverage whose risk is within tolerance, instead of walking down from the
  top and stopping at the first crossing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RiskCoverageCurve",
    "RiskCoveragePoint",
    "aurc",
    "coverage_at_risk",
    "risk_coverage_curve",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    """One threshold and what it buys."""

    threshold: float
    n_considered: int
    n_accepted: int
    n_harmful: int

    @property
    def coverage(self) -> float:
        return self.n_accepted / self.n_considered if self.n_considered else 0.0

    @property
    def risk(self) -> float:
        """Selective risk: harmful over accepted. Undefined at zero coverage, reported as
        0.0, which is why coverage must always be read alongside it."""
        return self.n_harmful / self.n_accepted if self.n_accepted else 0.0

    @property
    def joint_harm_rate(self) -> float:
        """Harmful over everything considered: stable where selective risk is not."""
        return self.n_harmful / self.n_considered if self.n_considered else 0.0


@dataclass(frozen=True, slots=True)
class RiskCoverageCurve:
    """A method's whole frontier."""

    points: tuple[RiskCoveragePoint, ...]

    @property
    def coverages(self) -> list[float]:
        return [p.coverage for p in self.points]

    @property
    def risks(self) -> list[float]:
        return [p.risk for p in self.points]

    def as_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "threshold": [p.threshold for p in self.points],
            "coverage": [p.coverage for p in self.points],
            "risk": [p.risk for p in self.points],
            "joint_harm_rate": [p.joint_harm_rate for p in self.points],
            "n_accepted": [p.n_accepted for p in self.points],
            "n_harmful": [p.n_harmful for p in self.points],
        }


def risk_coverage_curve(
    scores: FloatArray, harmful: BoolArray, n_thresholds: int = 200
) -> RiskCoverageCurve:
    """Sweep an acceptance threshold over ``scores``.

    A candidate is accepted when ``score >= threshold``. Thresholds are drawn from the
    observed score quantiles rather than a uniform grid, so the curve has resolution where
    the scores actually are — a uniform grid wastes most of its points on empty regions
    when scores cluster.
    """
    scores = np.asarray(scores, dtype=np.float64)
    harmful = np.asarray(harmful, dtype=bool)
    if scores.shape != harmful.shape:
        msg = f"shape mismatch: {scores.shape} scores vs {harmful.shape} harm flags"
        raise ValueError(msg)

    n = int(scores.size)
    if n == 0:
        return RiskCoverageCurve(points=())

    quantiles = np.linspace(0.0, 1.0, min(n_thresholds, max(n, 2)))
    thresholds = np.unique(np.quantile(scores, quantiles))
    # A threshold above every score gives the zero-coverage endpoint, which anchors the
    # curve at "accept nothing" and makes AURC well defined.
    thresholds = np.append(thresholds, np.nextafter(float(scores.max()), np.inf))

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_harmful = harmful[order]
    cumulative_harmful = np.cumsum(sorted_harmful)

    points: list[RiskCoveragePoint] = []
    for threshold in thresholds:
        n_accepted = int(np.searchsorted(-sorted_scores, -threshold, side="right"))
        n_harmful = int(cumulative_harmful[n_accepted - 1]) if n_accepted else 0
        points.append(
            RiskCoveragePoint(
                threshold=float(threshold),
                n_considered=n,
                n_accepted=n_accepted,
                n_harmful=n_harmful,
            )
        )
    return RiskCoverageCurve(points=tuple(points))


def coverage_at_risk(curve: RiskCoverageCurve, epsilon: float) -> RiskCoveragePoint | None:
    """Largest coverage whose selective risk is within ``epsilon``.

    Searches the entire curve rather than walking down from full coverage. Selective risk
    is not monotone in the threshold, so the first crossing is not necessarily the best
    operating point, and taking it would understate what a method can safely do.
    """
    feasible = [p for p in curve.points if p.n_accepted > 0 and p.risk <= epsilon]
    if not feasible:
        return None
    return max(feasible, key=lambda p: (p.n_accepted, -p.threshold))


def aurc(curve: RiskCoverageCurve) -> float:
    """Area under the risk-coverage curve, integrated over coverage.

    A summary of the whole frontier, useful when two methods cross. It is *not* a
    substitute for Coverage@Risk: a method can win on AURC while being unusable at the
    tolerance anyone would actually deploy at.
    """
    usable = [p for p in curve.points if p.n_accepted > 0]
    if not usable:
        # No operating point accepted anything, so there is no frontier. NaN, not 0.0:
        # zero is the BEST attainable area, and returning it would rank a method that
        # produced no curve above every method that did.
        return float("nan")
    if len(usable) == 1:
        # A degenerate frontier -- one point, which happens whenever a method emits a
        # constant score. The accept-everything reference does exactly that, and under
        # the old `return 0.0` it scored a perfect area while being the worst possible
        # method. The area-average of a single point is its own risk, which for
        # accept-everything is precisely the base harm rate: interpretable, and correctly
        # unflattering.
        return float(usable[0].risk)
    ordered = sorted(usable, key=lambda p: p.coverage)
    coverage = np.array([p.coverage for p in ordered])
    risk = np.array([p.risk for p in ordered])
    span = coverage[-1] - coverage[0]
    if span <= 0:
        return float(risk.mean())
    return float(np.trapezoid(risk, coverage) / span)
