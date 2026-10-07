"""Calibration metrics.

**Brier is primary.** It is a strictly proper scoring rule, it needs no binning, and it
decomposes (Murphy) into calibration, refinement, and uncertainty — so a single number can
be unpacked into "is it calibrated?" and "is it informative?" separately.

**ECE is secondary and is reported under both binning schemes.** Equal-width ECE is
biased and depends on bin count in ways that can be tuned, intentionally or not, to
flatter a method. Equal-mass binning fixes the count per bin instead. Reporting one number
without saying which scheme produced it is not a reportable result, so ``n_bins`` and the
scheme travel with the value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CalibrationReport",
    "ReliabilityBin",
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "murphy_decomposition",
    "reliability_bins",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of a reliability diagram, with the counts behind it."""

    lower: float
    upper: float
    n: int
    mean_predicted: float
    mean_observed: float

    @property
    def gap(self) -> float:
        return abs(self.mean_predicted - self.mean_observed)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Everything needed to state a calibration claim, including how it was measured."""

    n: int
    brier: float
    ece_equal_width: float
    ece_equal_mass: float
    max_calibration_error: float
    n_bins: int
    binning: str
    reliability: tuple[ReliabilityBin, ...]
    calibration_term: float
    refinement_term: float
    base_rate: float

    @property
    def ece(self) -> float:
        """ECE under the configured scheme. Prefer Brier for headline claims."""
        return self.ece_equal_mass if self.binning == "equal_mass" else self.ece_equal_width

    def as_dict(self) -> dict[str, float | int | str | list[dict[str, float | int]]]:
        return {
            "n": self.n,
            "brier": self.brier,
            "ece": self.ece,
            "ece_equal_width": self.ece_equal_width,
            "ece_equal_mass": self.ece_equal_mass,
            "max_calibration_error": self.max_calibration_error,
            "n_bins": self.n_bins,
            "binning": self.binning,
            "calibration_term": self.calibration_term,
            "refinement_term": self.refinement_term,
            "base_rate": self.base_rate,
            "reliability": [
                {
                    "lower": b.lower,
                    "upper": b.upper,
                    "n": b.n,
                    "mean_predicted": b.mean_predicted,
                    "mean_observed": b.mean_observed,
                }
                for b in self.reliability
            ],
        }


def brier_score(probabilities: FloatArray, outcomes: FloatArray) -> float:
    """Mean squared error of probabilistic predictions. Lower is better."""
    if probabilities.size == 0:
        return 0.0
    return float(np.mean((probabilities - outcomes) ** 2))


def reliability_bins(
    probabilities: FloatArray, outcomes: FloatArray, n_bins: int, binning: str
) -> list[ReliabilityBin]:
    """Bin predictions for a reliability diagram.

    ``equal_width`` splits [0, 1] into equal intervals; ``equal_mass`` splits into bins
    with (nearly) equal counts. Empty bins are omitted rather than reported as perfectly
    calibrated, which is what an all-zeros bin would otherwise imply.
    """
    if probabilities.size == 0:
        return []

    if binning == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif binning == "equal_mass":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(probabilities, quantiles))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        msg = f"unknown binning scheme {binning!r}; use 'equal_width' or 'equal_mass'"
        raise ValueError(msg)

    bins: list[ReliabilityBin] = []
    for index in range(len(edges) - 1):
        lower, upper = float(edges[index]), float(edges[index + 1])
        last = index == len(edges) - 2
        mask = (probabilities >= lower) & (
            probabilities <= upper if last else probabilities < upper
        )
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                n=count,
                mean_predicted=float(np.mean(probabilities[mask])),
                mean_observed=float(np.mean(outcomes[mask])),
            )
        )
    return bins


def expected_calibration_error(
    probabilities: FloatArray, outcomes: FloatArray, n_bins: int, binning: str
) -> tuple[float, float]:
    """Return ``(ece, max_calibration_error)`` under one binning scheme."""
    bins = reliability_bins(probabilities, outcomes, n_bins, binning)
    if not bins:
        return 0.0, 0.0
    total = sum(b.n for b in bins)
    ece = sum(b.n * b.gap for b in bins) / total
    return float(ece), float(max(b.gap for b in bins))


def murphy_decomposition(
    probabilities: FloatArray, outcomes: FloatArray, n_bins: int, binning: str
) -> tuple[float, float, float]:
    """Decompose Brier into ``(calibration, refinement, uncertainty)``.

    Calibration is the part a recalibration can remove; refinement is the part it cannot.
    Separating them answers "is this model miscalibrated, or just uninformative?" — which
    matters under engine shift, where the two failure modes call for different responses.
    """
    if probabilities.size == 0:
        return 0.0, 0.0, 0.0
    base_rate = float(np.mean(outcomes))
    uncertainty = base_rate * (1.0 - base_rate)

    bins = reliability_bins(probabilities, outcomes, n_bins, binning)
    total = sum(b.n for b in bins) or 1
    calibration = sum(b.n * (b.mean_predicted - b.mean_observed) ** 2 for b in bins) / total
    refinement = sum(b.n * b.mean_observed * (1.0 - b.mean_observed) for b in bins) / total
    return float(calibration), float(refinement), float(uncertainty)


def calibration_report(
    probabilities: FloatArray,
    outcomes: FloatArray,
    n_bins: int = 15,
    binning: str = "equal_mass",
) -> CalibrationReport:
    """Full calibration assessment, with the measurement choices attached."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if probabilities.shape != outcomes.shape:
        msg = f"shape mismatch: {probabilities.shape} predictions vs {outcomes.shape} outcomes"
        raise ValueError(msg)

    ece_width, mce_width = expected_calibration_error(
        probabilities, outcomes, n_bins, "equal_width"
    )
    ece_mass, mce_mass = expected_calibration_error(probabilities, outcomes, n_bins, "equal_mass")
    calibration, refinement, _ = murphy_decomposition(probabilities, outcomes, n_bins, binning)

    return CalibrationReport(
        n=int(probabilities.size),
        brier=brier_score(probabilities, outcomes),
        ece_equal_width=ece_width,
        ece_equal_mass=ece_mass,
        max_calibration_error=mce_mass if binning == "equal_mass" else mce_width,
        n_bins=n_bins,
        binning=binning,
        reliability=tuple(reliability_bins(probabilities, outcomes, n_bins, binning)),
        calibration_term=calibration,
        refinement_term=refinement,
        base_rate=float(np.mean(outcomes)) if outcomes.size else 0.0,
    )
