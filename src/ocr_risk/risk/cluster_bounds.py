"""Risk bounds when the sampling unit is a document, not a candidate.

``bounds.py`` answers "how large can a Bernoulli rate be, given ``k`` harms in ``n`` draws".
That question is the right one only when the ``n`` draws are independent. Candidates are not:
several of them come from one page, they share a scan, a typeface and an engine failure mode,
and SGV15 measured design effects between 7 and 38 on the harm rate of its deployed accepted
sets. An exact Clopper-Pearson interval read on such a sample is exact about the wrong model.

This module bounds the same **candidate-weighted** quantity -- ``sum_d H_d / sum_d A_d``, the
harm rate a deployment actually experiences -- while treating the *document* as the independent
unit. The reduction that makes it possible is elementary: ``R <= epsilon`` is equivalent to
``E[H_d - epsilon * A_d] <= 0``, so a ratio null becomes an upper confidence bound on the mean
of one bounded per-document variable, and three standard distribution-free bounds apply.

The price is the support width. ``Z_d = H_d - epsilon * A_d`` ranges over an interval whose
width is the *largest* per-document accepted count, while the signal is carried by the *average*
one, so a page that contributes two hundred accepted edits widens every bound. That ratio is the
design effect arrived at from the other direction, and it is why honest document-level inference
costs what it costs.

Everything here is distribution-free and finite-sample except :func:`cluster_robust_ratio_upper`,
which is a limit theorem and is labelled as one everywhere it is used.

The full specification -- estimand, assumptions, degenerate cases, multiplicity, and the
duplication-invariance argument that separates the two inference units -- is in
``docs/sgv15b/risk_bound_design.md``, which was frozen before any result was computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

__all__ = [
    "BETTING_TRUNCATION",
    "CLUSTER_RATIO_METHODS",
    "FINITE_SAMPLE_METHODS",
    "ClusterRatioBound",
    "betting_mean_upper",
    "cluster_ratio_bound",
    "cluster_ratio_upper",
    "cluster_robust_ratio_upper",
    "design_effect",
    "empirical_bernstein_mean_upper",
    "hoeffding_mean_upper",
]

FloatArray = NDArray[np.float64]

BETTING_TRUNCATION = 0.75
"""The cap on the betting fraction. Any predictable ``lambda`` in ``[0, 1)`` keeps the capital
process a supermartingale; the cap keeps ``-log(1 - lambda)`` finite and is the value Waudby-Smith
and Ramdas recommend for the one-sided plug-in."""

FINITE_SAMPLE_METHODS = ("hoeffding", "empirical_bernstein", "betting")
"""The three bounds that hold at their nominal level for every sample size. Their union at
``delta / 3`` is the declared primary certificate."""

CLUSTER_RATIO_METHODS = (*FINITE_SAMPLE_METHODS, "union_finite", "cluster_robust_asymptotic")
"""``union_finite`` is the declared primary. ``cluster_robust_asymptotic`` is a comparator and is
never a certificate: its guarantee is a limit, not a bound."""


def _validated(values: NDArray[np.float64] | list[float], delta: float, span: float) -> FloatArray:
    """Shared argument check. A bound with a nonsense level is a bug, not a wide interval."""
    if not 0.0 < delta < 1.0:
        msg = f"delta must lie strictly inside (0, 1); got {delta}"
        raise ValueError(msg)
    if not math.isfinite(span) or span < 0.0:
        msg = f"span must be finite and non-negative; got {span}"
        raise ValueError(msg)
    return np.asarray(values, dtype=np.float64).ravel()


def hoeffding_mean_upper(
    values: NDArray[np.float64] | list[float], delta: float, *, span: float
) -> float:
    """One-sided ``1 - delta`` upper bound on ``E[X]`` for i.i.d. ``X`` in an interval of width
    ``span``.

    Distribution-free, exact at every sample size, and free of any variance estimate, which is
    what makes it the floor: it cannot be anti-conservative under its stated assumptions, so it is
    the bound the others are checked against.

    Returns ``+inf`` with no observations, because nothing observed rules nothing out.
    """
    observations = _validated(values, delta, span)
    n = observations.size
    if n == 0:
        return float("inf")
    return float(observations.mean() + span * math.sqrt(math.log(1.0 / delta) / (2.0 * n)))


def empirical_bernstein_mean_upper(
    values: NDArray[np.float64] | list[float], delta: float, *, span: float
) -> float:
    """Maurer-Pontil empirical-Bernstein upper bound, rescaled to a support of width ``span``.

    Adapts to the observed variance, which matters here because per-document harm counts are
    concentrated near zero on most pages and the variance-free Hoeffding width is then far larger
    than the data warrant. The ``O(span / n)`` additive term is the cost of estimating that
    variance from the same sample, and it is what dominates at small document counts.

    Falls back to Hoeffding below two observations, where a sample variance does not exist.
    """
    observations = _validated(values, delta, span)
    n = observations.size
    if n < 2:
        return hoeffding_mean_upper(observations, delta, span=span)
    log_term = math.log(2.0 / delta)
    variance = float(observations.var(ddof=1))
    return float(
        observations.mean()
        + math.sqrt(2.0 * variance * log_term / n)
        + 7.0 * span * log_term / (3.0 * (n - 1))
    )


def betting_mean_upper(
    values: NDArray[np.float64] | list[float], delta: float, *, lower: float, span: float
) -> float:
    """Waudby-Smith-Ramdas predictable-plug-in empirical-Bernstein bound on ``E[X]``.

    The capital process ``K(m) = prod_i exp(lambda_i (m - Y_i) - psi(lambda_i)(Y_i - muhat_i-1)^2)``
    on the rescaled ``Y = (X - lower) / span`` is a non-negative supermartingale under
    ``E[Y] >= m`` for predictable ``lambda_i``, so ``P(K(m) >= 1 / delta) <= delta`` and the
    smallest rejected ``m`` is a valid upper confidence bound.

    ``log K(m)`` is **affine** in ``m`` -- only the ``lambda_i (m - Y_i)`` term moves -- so the
    crossing point is solved in closed form after one pass rather than by bisection. That is what
    makes this bound affordable inside a per-prefix certification loop.

    The bound is **order-dependent**: ``lambda_i`` is predictable, so it sees only the first
    ``i - 1`` observations. The caller must fix the order before reading the data; SGV15b sorts by
    document identifier, which is content-blind.
    """
    observations = _validated(values, delta, span)
    n = observations.size
    if n == 0:
        return float("inf")
    if span == 0.0:
        return float(lower)

    y = (observations - lower) / span
    steps = np.arange(1, n + 1, dtype=np.float64)
    # muhat_i and sigmahat_i are the plug-ins the source prescribes, seeded at 1/2 and 1/4 so
    # that the first bet is defined before any observation has been seen.
    running_mean = (0.5 + np.cumsum(y)) / (1.0 + steps)
    running_var = (0.25 + np.cumsum((y - running_mean) ** 2)) / (1.0 + steps)
    previous_mean = np.concatenate(([0.5], running_mean[:-1]))
    previous_var = np.concatenate(([0.25], running_var[:-1]))
    lam = np.minimum(
        BETTING_TRUNCATION,
        np.sqrt(2.0 * math.log(1.0 / delta) / (previous_var * steps * np.log1p(steps))),
    )
    psi = -np.log1p(-lam) - lam
    # log K(m) = m * sum(lambda) - constant, so the m at which the process reaches 1 / delta is
    # available directly.
    slope = float(lam.sum())
    intercept = float((lam * y + psi * (y - previous_mean) ** 2).sum())
    crossing = (math.log(1.0 / delta) + intercept) / slope
    return float(lower + span * min(1.0, max(0.0, crossing)))


def cluster_robust_ratio_upper(
    harmful: NDArray[np.float64] | list[float],
    accepted: NDArray[np.float64] | list[float],
    delta: float,
) -> float:
    """Delta-method upper confidence limit on ``sum_d H_d / sum_d A_d`` over document clusters.

    The classical cluster ratio estimator: the residual ``H_d - Rhat * A_d`` carries the
    variance, and Student's ``t`` on ``D - 1`` degrees of freedom is the small-sample courtesy.

    **This is asymptotic.** It is a comparator, never a certificate, and the gap between it and
    the finite-sample bounds is what separates "the documents do not support this prefix" from
    "a distribution-free bound cannot see the support that is there". Returns ``1.0`` when it
    cannot be formed at all.
    """
    numerators = np.asarray(harmful, dtype=np.float64).ravel()
    denominators = np.asarray(accepted, dtype=np.float64).ravel()
    if numerators.shape != denominators.shape:
        msg = f"shape mismatch: {numerators.shape} numerators vs {denominators.shape} denominators"
        raise ValueError(msg)
    documents = numerators.size
    total = float(denominators.sum())
    if documents < 2 or total <= 0.0:
        return 1.0
    ratio = float(numerators.sum()) / total
    mean_denominator = float(denominators.mean())
    residual = numerators - ratio * denominators
    variance = float((residual**2).sum()) / (
        (documents - 1) * documents * mean_denominator * mean_denominator
    )
    quantile = float(student_t.ppf(1.0 - delta, documents - 1))
    return float(min(1.0, ratio + quantile * math.sqrt(variance)))


def design_effect(
    harmful: NDArray[np.float64] | list[float], accepted: NDArray[np.float64] | list[float]
) -> float:
    """Ratio of the document-clustered variance of the harm rate to its binomial variance.

    A diagnostic, not a bound: it says how many candidates one independent observation is worth,
    which is the intuition the certificates make precise. Returns ``nan`` where it is undefined.
    """
    numerators = np.asarray(harmful, dtype=np.float64).ravel()
    denominators = np.asarray(accepted, dtype=np.float64).ravel()
    documents = numerators.size
    total = float(denominators.sum())
    if documents < 2 or total <= 0.0:
        return float("nan")
    ratio = float(numerators.sum()) / total
    if ratio <= 0.0 or ratio >= 1.0:
        return float("nan")
    residual = numerators - ratio * denominators
    clustered = documents * float((residual**2).sum()) / (documents - 1) / (total * total)
    return float(clustered / (ratio * (1.0 - ratio) / total))


@dataclass(frozen=True, slots=True)
class ClusterRatioBound:
    """One prefix's certification decision under one inference model."""

    method: str
    guarantee: str
    """``finite-sample`` or ``asymptotic``. Only the former may be called a certificate."""
    certifies: bool
    documents: int
    accepted: int
    harmful: int
    observed_ratio: float
    span: float
    z_upper: float
    """Upper confidence limit on ``E[H_d - epsilon A_d]``; ``nan`` for the asymptotic method,
    which bounds the ratio directly."""
    ratio_upper: float
    """Upper confidence limit on the ratio itself; ``nan`` for the finite-sample methods, whose
    natural output is a test at a declared tolerance. :func:`cluster_ratio_upper` inverts them."""
    degenerate_denominator: bool
    """True when no sampled document accepted anything at this prefix. It is a diagnostic and not
    a decision: a sample that accepted nothing has seen no evidence, and the bound is still
    widened by the population support, so such a prefix certifies only when nothing anywhere can
    be accepted at it."""

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "method": self.method,
            "guarantee": self.guarantee,
            "certifies": self.certifies,
            "documents": self.documents,
            "accepted": self.accepted,
            "harmful": self.harmful,
            "observed_ratio": self.observed_ratio,
            "span": self.span,
            "z_upper": self.z_upper,
            "ratio_upper": self.ratio_upper,
            "degenerate_denominator": self.degenerate_denominator,
        }


def cluster_ratio_bound(
    harmful: NDArray[np.float64] | list[float],
    accepted: NDArray[np.float64] | list[float],
    epsilon: float,
    *,
    delta: float,
    span: float,
    method: str = "union_finite",
) -> ClusterRatioBound:
    """Can ``sum_d H_d / sum_d A_d <= epsilon`` be certified from these documents?

    ``harmful`` and ``accepted`` are per-document counts, in the order the caller declared.
    ``span`` bounds the per-document accepted count over the *population*, not over this sample;
    it is a design constant computed from unlabelled scores and it must not depend on an outcome.

    ``union_finite`` -- the declared primary -- is the tightest of the three finite-sample bounds
    with the level split three ways, so no bound is chosen after seeing a result.

    A sample in which no document accepted anything does **not** certify. `Z_d` is then zero on
    every sampled document, but the bound is still widened by ``span``, which describes what the
    population can accept at this prefix rather than what this draw happened to see. Certifying
    there would deploy a threshold that no observation had constrained, which is the failure mode
    the whole stage exists to remove.
    """
    if method not in CLUSTER_RATIO_METHODS:
        msg = f"unknown cluster ratio method {method!r}; known: {', '.join(CLUSTER_RATIO_METHODS)}"
        raise ValueError(msg)
    numerators = np.asarray(harmful, dtype=np.float64).ravel()
    denominators = np.asarray(accepted, dtype=np.float64).ravel()
    if numerators.shape != denominators.shape:
        msg = f"shape mismatch: {numerators.shape} numerators vs {denominators.shape} denominators"
        raise ValueError(msg)

    documents = int(numerators.size)
    total = float(denominators.sum())
    harm_total = float(numerators.sum())
    guarantee = "asymptotic" if method == "cluster_robust_asymptotic" else "finite-sample"
    common = {
        "method": method,
        "guarantee": guarantee,
        "documents": documents,
        "accepted": int(total),
        "harmful": int(harm_total),
        "observed_ratio": harm_total / total if total > 0.0 else float("nan"),
        "span": float(span),
    }

    degenerate = total <= 0.0
    if documents == 0:
        # Nothing observed rules nothing out, and a prefix cannot be deployed on no evidence.
        return ClusterRatioBound(
            certifies=False,
            z_upper=float("inf"),
            ratio_upper=1.0,
            degenerate_denominator=False,
            **common,  # type: ignore[arg-type]
        )

    if method == "cluster_robust_asymptotic":
        ratio_upper = cluster_robust_ratio_upper(numerators, denominators, delta)
        return ClusterRatioBound(
            certifies=bool(ratio_upper <= epsilon),
            z_upper=float("nan"),
            ratio_upper=ratio_upper,
            degenerate_denominator=degenerate,
            **common,  # type: ignore[arg-type]
        )

    z = numerators - epsilon * denominators
    lower = -epsilon * span
    if method == "union_finite":
        level = delta / len(FINITE_SAMPLE_METHODS)
        z_upper = min(
            hoeffding_mean_upper(z, level, span=span),
            empirical_bernstein_mean_upper(z, level, span=span),
            betting_mean_upper(z, level, lower=lower, span=span),
        )
    elif method == "hoeffding":
        z_upper = hoeffding_mean_upper(z, delta, span=span)
    elif method == "empirical_bernstein":
        z_upper = empirical_bernstein_mean_upper(z, delta, span=span)
    else:
        z_upper = betting_mean_upper(z, delta, lower=lower, span=span)

    return ClusterRatioBound(
        certifies=bool(z_upper <= 0.0),
        z_upper=float(z_upper),
        ratio_upper=float("nan"),
        degenerate_denominator=degenerate,
        **common,  # type: ignore[arg-type]
    )


def cluster_ratio_upper(
    harmful: NDArray[np.float64] | list[float],
    accepted: NDArray[np.float64] | list[float],
    *,
    delta: float,
    span: float,
    method: str = "union_finite",
    resolution: int = 1000,
) -> float:
    """Invert a finite-sample certificate into an upper confidence limit on the ratio.

    The finite-sample bounds are tests at a declared tolerance, and the tolerance enters
    ``Z_d = H_d - epsilon A_d`` in a way that is monotone for Hoeffding but not guaranteed
    monotone for the variance-adaptive members. The reported limit is therefore the smallest grid
    tolerance from which certification holds at *every* larger grid tolerance -- the monotone
    closure, which is conservative by construction and never reports a limit the method would
    fail to certify.

    Reporting only. Nothing selects a prefix from this number.
    """
    grid = np.linspace(0.0, 1.0, resolution + 1)
    limit = 1.0
    for epsilon in reversed(grid.tolist()):
        if not cluster_ratio_bound(
            harmful, accepted, float(epsilon), delta=delta, span=span, method=method
        ).certifies:
            return float(limit)
        limit = float(epsilon)
    return float(limit)
