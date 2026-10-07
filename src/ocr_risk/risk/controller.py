"""Threshold selection under a harmful-edit tolerance.

Two controllers, and the difference between them is the honesty of the guarantee:

``empirical``
    Pick the threshold whose *observed* risk on the calibration split is within epsilon.
    Simple, and optimistically biased: it is selected on the same sample it is measured on,
    so realized risk on new data exceeds epsilon roughly half the time.

``ltt_bentkus``
    Learn-then-Test. Sweep thresholds, compute a finite-sample p-value for
    ``H0: risk >= epsilon`` at each, and take the most permissive threshold that rejects
    at level delta with multiplicity controlled. Gives ``P(risk <= epsilon) >= 1 - delta``
    on exchangeable data.

Both are provided deliberately. The gap between them at deployment is what quantifies the
selection bias in the naive approach, and the gap between the *guarantee* and the realized
risk on a held-out engine is what H1 measures — cross-engine shift violates exchangeability
by construction, so the bound is expected to degrade, and the size of that degradation is
the finding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ocr_risk.risk.bounds import bentkus_p_value, hoeffding_p_value

__all__ = ["CONTROLLERS", "ThresholdDecision", "select_threshold"]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CONTROLLERS = ("empirical", "ltt_bentkus", "ltt_clopper_pearson", "ltt_beta_binomial")
"""``empirical`` takes the observed rate at face value; the two ``ltt_`` members require a
finite-sample upper bound to clear the tolerance. ``ltt_bentkus`` is distribution-free over
any loss bounded in [0, 1]; ``ltt_clopper_pearson`` is exact for the Bernoulli harm indicator
and materially tighter at small calibration sizes; ``ltt_beta_binomial`` is a Bayesian credible
bound that needs a ``prior`` and offers no finite-sample guarantee when that prior is wrong.
Which one a stage uses is a pre-registered choice."""


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """A chosen threshold, and everything needed to judge whether to trust it."""

    tau: float
    epsilon: float
    delta: float
    controller: str
    n_calibration: int
    n_accepted: int
    n_harmful_accepted: int
    observed_risk: float
    risk_upper_bound: float
    coverage: float
    feasible: bool
    """False when no threshold satisfies the tolerance. The controller then abstains
    entirely (``tau`` above every score) rather than returning its least-bad option."""
    note: str = ""

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "tau": self.tau,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "controller": self.controller,
            "n_calibration": self.n_calibration,
            "n_accepted": self.n_accepted,
            "n_harmful_accepted": self.n_harmful_accepted,
            "observed_risk": self.observed_risk,
            "risk_upper_bound": self.risk_upper_bound,
            "coverage": self.coverage,
            "feasible": self.feasible,
            "note": self.note,
        }


def _candidate_thresholds(scores: FloatArray, n_grid: int) -> FloatArray:
    """Thresholds drawn from the observed score quantiles, plus an abstain-everything point.

    Quantiles rather than a uniform grid, because scores cluster and a uniform grid spends
    most of its resolution where there is no data.
    """
    unique = np.unique(scores)
    if unique.size <= n_grid:
        grid = unique
    else:
        grid = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_grid)))
    return np.append(grid, np.nextafter(float(unique.max()), np.inf))


def select_threshold(
    scores: FloatArray,
    harmful: BoolArray,
    epsilon: float,
    *,
    delta: float = 0.1,
    controller: str = "ltt_bentkus",
    n_grid: int = 200,
    thresholds: Sequence[float] | None = None,
    prior: tuple[float, float] | None = None,
) -> ThresholdDecision:
    """Choose the most permissive threshold whose harmful-edit risk is within ``epsilon``.

    Called with a **calibration-split** view. Handing it evaluation scores would select the
    threshold on the data it is then measured on, which is leakage vector L1.

    ``prior`` is ``(pseudo_harmful, pseudo_total)`` and is required by -- and only used by --
    the ``ltt_beta_binomial`` controller.

    ``thresholds`` supplies a pre-registered cut grid in place of the score quantiles. The
    multiplicity correction is over whatever grid is searched, so a caller that wants a small
    declared family -- and the wider per-test level that comes with it -- passes one here
    rather than searching every row and quoting a pointwise interval as a guarantee.
    """
    if controller not in CONTROLLERS:
        msg = f"unknown risk controller {controller!r}; known: {', '.join(CONTROLLERS)}"
        raise ValueError(msg)
    if controller == "ltt_beta_binomial" and prior is None:
        msg = "the ltt_beta_binomial controller needs a prior; refusing to invent one"
        raise ValueError(msg)

    scores = np.asarray(scores, dtype=np.float64).ravel()
    harmful = np.asarray(harmful, dtype=bool).ravel()
    if scores.shape != harmful.shape:
        msg = f"shape mismatch: {scores.shape} scores vs {harmful.shape} harm flags"
        raise ValueError(msg)

    n = int(scores.size)
    if n == 0:
        return ThresholdDecision(
            tau=float("inf"),
            epsilon=epsilon,
            delta=delta,
            controller=controller,
            n_calibration=0,
            n_accepted=0,
            n_harmful_accepted=0,
            observed_risk=0.0,
            risk_upper_bound=1.0,
            coverage=0.0,
            feasible=False,
            note="empty calibration set; abstaining rather than guessing a threshold",
        )

    grid = (
        _candidate_thresholds(scores, n_grid)
        if thresholds is None
        else np.unique(np.asarray(list(thresholds), dtype=np.float64))
    )
    # Multiplicity: the threshold is chosen by testing every grid point, so the level is
    # split across them. Without this the guarantee would be void by construction.
    per_test_delta = delta / max(len(grid), 1)

    # Most permissive feasible threshold = most coverage. Risk is not monotone in tau, so
    # the whole grid is searched rather than stopping at the first crossing. The upper
    # bound is deferred to the winner: it costs a bisection, and only the chosen threshold
    # needs one.
    best: tuple[float, int, int, float] | None = None
    for tau in grid:
        accepted = scores >= tau
        n_accepted = int(np.count_nonzero(accepted))
        if n_accepted == 0:
            continue
        n_harmful = int(np.count_nonzero(harmful & accepted))
        observed = n_harmful / n_accepted

        if controller == "empirical":
            passes = observed <= epsilon
        elif controller in ("ltt_clopper_pearson", "ltt_beta_binomial"):
            passes = (
                _upper_bound(n_harmful, n_accepted, per_test_delta, controller, prior) <= epsilon
            )
        else:
            p_value = min(
                hoeffding_p_value(n_accepted, observed, epsilon),
                bentkus_p_value(n_accepted, observed, epsilon),
            )
            passes = p_value <= per_test_delta

        if passes and (best is None or n_accepted > best[1]):
            best = (float(tau), n_accepted, n_harmful, observed)

    if best is not None:
        tau, n_accepted, n_harmful, observed = best
        bound = (
            observed
            if controller == "empirical"
            else _upper_bound(n_harmful, n_accepted, per_test_delta, controller, prior)
        )
        return ThresholdDecision(
            tau=tau,
            epsilon=epsilon,
            delta=delta,
            controller=controller,
            n_calibration=n,
            n_accepted=n_accepted,
            n_harmful_accepted=n_harmful,
            observed_risk=observed,
            risk_upper_bound=bound,
            coverage=n_accepted / n,
            feasible=True,
        )

    return ThresholdDecision(
        tau=float(np.nextafter(float(scores.max()), np.inf)),
        epsilon=epsilon,
        delta=delta,
        controller=controller,
        n_calibration=n,
        n_accepted=0,
        n_harmful_accepted=0,
        observed_risk=0.0,
        risk_upper_bound=1.0,
        coverage=0.0,
        feasible=False,
        note=(
            f"no threshold controls risk at epsilon={epsilon} with delta={delta}; "
            "abstaining from every edit. Zero coverage at zero risk is a valid and "
            "honest operating point, and reporting the least-bad threshold instead would "
            "silently exceed the tolerance the experiment promised."
        ),
    )


def _upper_bound(
    n_harmful: int,
    n_accepted: int,
    delta: float,
    controller: str = "ltt_bentkus",
    prior: tuple[float, float] | None = None,
) -> float:
    from ocr_risk.risk.bounds import (
        beta_binomial_upper,
        clopper_pearson_upper,
        risk_upper_bound,
    )

    if controller == "ltt_clopper_pearson":
        return clopper_pearson_upper(n_harmful, n_accepted, delta)
    if controller == "ltt_beta_binomial":
        assert prior is not None
        return beta_binomial_upper(n_harmful, n_accepted, prior[0], prior[1], delta)
    return risk_upper_bound(n_harmful, n_accepted, delta)
