"""Multiplicity control and effect sizes.

The leave-one-engine-out protocol runs four folds against several risk tolerances, so a
hypothesis is tested many times. Reporting the smallest p-value from that family without
adjustment is how a null result becomes a finding.

Holm is the default: it controls the family-wise error rate under arbitrary dependence,
which is the honest assumption here — folds share documents and the epsilon levels are
nested, so the tests are dependent in ways not worth modelling. Benjamini-Hochberg is
available for exploratory families where controlling the false discovery rate is the
appropriate weaker guarantee, and the choice is recorded either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AdjustedTest",
    "benjamini_hochberg",
    "cliffs_delta",
    "cohens_h",
    "holm_bonferroni",
    "minimum_detectable_effect",
]


@dataclass(frozen=True, slots=True)
class AdjustedTest:
    """One test within a family, after adjustment."""

    label: str
    p_value: float
    adjusted_p_value: float
    significant: bool


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> list[AdjustedTest]:
    """Holm step-down adjustment. Controls FWER under arbitrary dependence."""
    if not p_values:
        return []
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    results: list[AdjustedTest] = []
    running_max = 0.0
    for index, (label, p) in enumerate(ordered):
        adjusted = min(1.0, (m - index) * p)
        # Enforce monotonicity: an adjusted p-value may never fall below an earlier one.
        running_max = max(running_max, adjusted)
        results.append(
            AdjustedTest(
                label=label,
                p_value=p,
                adjusted_p_value=running_max,
                significant=running_max <= alpha,
            )
        )
    return results


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> list[AdjustedTest]:
    """Benjamini-Hochberg adjustment. Controls the false discovery rate."""
    if not p_values:
        return []
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: list[float] = []
    running_min = 1.0
    for index in reversed(range(m)):
        value = min(1.0, ordered[index][1] * m / (index + 1))
        running_min = min(running_min, value)
        adjusted.append(running_min)
    adjusted.reverse()
    return [
        AdjustedTest(label=label, p_value=p, adjusted_p_value=q, significant=q <= alpha)
        for (label, p), q in zip(ordered, adjusted, strict=True)
    ]


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference of two proportions.

    Reported alongside a difference in rates because a statistically distinguishable
    change of 0.2 percentage points is not the same claim as a meaningful one.
    """
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Non-parametric effect size in [-1, 1]: how often ``a`` exceeds ``b``.

    Distribution-free, so it does not assume the per-document rates are normal — which
    they are not, being bounded and often zero-inflated.
    """
    if not a or not b:
        return 0.0
    left = np.asarray(a, dtype=np.float64)[:, None]
    right = np.asarray(b, dtype=np.float64)[None, :]
    greater = int(np.count_nonzero(left > right))
    lesser = int(np.count_nonzero(left < right))
    return (greater - lesser) / (left.size * right.size)


def minimum_detectable_effect(
    standard_error: float, alpha: float = 0.05, power: float = 0.80, n_tests: int = 1
) -> float:
    """The smallest true effect this design would detect, at the given power.

    Reported beside every non-significant result, because "we did not detect an effect"
    and "there is no effect" are different claims and only the first is supported by a
    null. A null with an MDE larger than the quantity being compared is uninformative,
    and saying so is the difference between an honest negative result and a misleading
    one.

    ``n_tests`` applies the Holm worst case: the smallest p in a family of ``m`` must
    clear ``alpha / m``, so that is the level a study is actually powered against.
    """
    if not math.isfinite(standard_error) or standard_error <= 0:
        return float("nan")
    adjusted = alpha / max(n_tests, 1)
    return float(
        (_normal_quantile(1.0 - adjusted / 2.0) + _normal_quantile(power)) * standard_error
    )


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, via scipy so the tail is accurate."""
    from scipy.stats import norm

    return float(norm.ppf(p))
