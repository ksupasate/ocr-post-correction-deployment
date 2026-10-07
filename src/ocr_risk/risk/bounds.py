"""Finite-sample upper confidence bounds on a binomial risk.

Used to choose a threshold that provably controls harmful-edit risk on *exchangeable*
data. Two bounds, and the tighter of the two is taken pointwise:

- **Hoeffding** — distribution-free, loose, and reliable everywhere.
- **Bentkus** — much tighter in the small-risk regime this project operates in, where the
  target is a 1% or 0.1% harm rate and Hoeffding's variance-free bound is very
  conservative.

The exchangeability assumption is the interesting part. Under cross-engine shift it is
**violated by construction** — that is the shift H1 is about — so a bound calibrated on
training engines is not guaranteed on the held-out one. Measuring how badly it fails is a
result, not a bug, and the analysis reports realized risk against the nominal bound rather
than assuming the guarantee holds.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom

__all__ = [
    "bentkus_p_value",
    "beta_binomial_upper",
    "clopper_pearson_upper",
    "hoeffding_p_value",
    "risk_upper_bound",
    "smallest_controlled_risk",
]


def hoeffding_p_value(n: int, observed_mean: float, null_risk: float) -> float:
    """P-value for ``H0: true risk >= null_risk`` under Hoeffding's inequality.

    Distribution-free and valid for any bounded loss, which is why it is the safe floor.
    """
    if n <= 0:
        return 1.0
    if observed_mean >= null_risk:
        return 1.0
    gap = null_risk - observed_mean
    return float(np.exp(-2.0 * n * gap * gap))


def bentkus_p_value(n: int, observed_mean: float, null_risk: float) -> float:
    """Bentkus' binomial-tail p-value for the same null.

    Substantially tighter than Hoeffding when the risk is small, which is the regime that
    matters here: at a 1% tolerance, Hoeffding needs far more calibration data than a
    per-fold split has.
    """
    if n <= 0:
        return 1.0
    if observed_mean >= null_risk:
        return 1.0
    successes = int(np.ceil(n * observed_mean))
    # The factor of e is Bentkus' constant; it is what makes the binomial tail a valid
    # bound for a general bounded loss rather than only for a Bernoulli one.
    return float(min(1.0, np.e * binom.cdf(successes, n, null_risk)))


def risk_upper_bound(n_harmful: int, n_total: int, delta: float = 0.1) -> float:
    """Smallest ``r`` such that ``H0: risk >= r`` cannot be rejected at level ``delta``.

    Returns ``1.0`` when there is no data: with nothing observed, nothing is ruled out,
    and reporting a low bound would be worse than reporting none.
    """
    if n_total <= 0:
        return 1.0
    observed = n_harmful / n_total

    # The p-value is monotone in the null, so bisection finds the crossing exactly.
    low, high = observed, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        p = min(hoeffding_p_value(n_total, observed, mid), bentkus_p_value(n_total, observed, mid))
        if p <= delta:
            high = mid
        else:
            low = mid
    return float(high)


def smallest_controlled_risk(n_harmful: int, n_total: int, delta: float = 0.1) -> float:
    """Alias for :func:`risk_upper_bound`, named for how the controller reads it."""
    return risk_upper_bound(n_harmful, n_total, delta)


def clopper_pearson_upper(n_harmful: int, n_total: int, delta: float = 0.1) -> float:
    """Exact one-sided Clopper-Pearson upper bound on a Bernoulli rate.

    ``risk_upper_bound`` above is the tighter of Hoeffding and Bentkus, and Bentkus carries a
    factor of ``e`` precisely because it must hold for any loss bounded in ``[0, 1]``. The
    harmful-edit indicator is not a general bounded loss: it is a Bernoulli draw, one per
    accepted candidate. For that case the Clopper-Pearson interval is exact -- its coverage is
    at least ``1 - delta`` for every true rate -- and it is materially tighter than Bentkus in
    the small-sample regime, which is the regime a target certification sample lives in.

    Both are valid; which one is used is a pre-registered choice, not a tuning knob. Returns
    ``1.0`` with no data, because nothing observed rules nothing out.
    """
    if n_total <= 0:
        return 1.0
    if n_harmful >= n_total:
        return 1.0
    # Beta(k + 1, n - k) at the 1 - delta quantile. The k = 0 case is the one that matters
    # most here and reduces to 1 - delta ** (1 / n), which is why observing no harm in a
    # small sample certifies so little.
    return float(beta.ppf(1.0 - delta, n_harmful + 1, n_total - n_harmful))


def beta_binomial_upper(
    n_harmful: int,
    n_total: int,
    prior_harmful: float,
    prior_total: float,
    delta: float = 0.1,
) -> float:
    """Upper quantile of a Beta posterior with an informative prior.

    ``prior_harmful`` and ``prior_total`` are pseudo-counts: a prior belief worth
    ``prior_total`` observations of which ``prior_harmful`` were harmful. The posterior is
    ``Beta(prior_harmful + n_harmful, prior_total - prior_harmful + n_total - n_harmful)`` and
    the bound is its ``1 - delta`` quantile.

    This is a Bayesian credible bound and NOT a finite-sample guarantee: its coverage holds only
    if the prior is right. That is exactly why it is worth measuring against the exact bound when
    the prior comes from a source domain the deployment has shifted away from -- a prior that is
    wrong in the safe direction produces a bound that is confidently too low.
    """
    if n_total <= 0 and prior_total <= 0:
        return 1.0
    alpha = prior_harmful + n_harmful
    beta_parameter = (prior_total - prior_harmful) + (n_total - n_harmful)
    if alpha <= 0.0 or beta_parameter <= 0.0:
        return 1.0
    return float(beta.ppf(1.0 - delta, alpha, beta_parameter))
