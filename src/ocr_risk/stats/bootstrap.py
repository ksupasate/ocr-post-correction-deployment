"""Resampling at the document level.

**The resampling unit is the document, not the edit.** Edits within a page are strongly
correlated — they share a scan, a font, a degradation, and often a systematic engine
failure — so resampling individual edits treats correlated observations as independent and
produces confidence intervals that are far too narrow. That is not a conservative
approximation; it is a wrong answer that makes non-results look significant.

``cluster_bootstrap`` resamples whole documents with replacement, which respects the
dependence. ``paired_cluster_bootstrap`` resamples the same documents for both methods on
each draw, so the comparison keeps the variance reduction that a paired design buys.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BootstrapResult",
    "cluster_bootstrap",
    "cluster_bootstrap_indices",
    "group_by_cluster",
    "paired_cluster_bootstrap",
    "paired_cluster_bootstrap_multi",
]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A point estimate with its resampling distribution summarized."""

    estimate: float
    lower: float
    upper: float
    ci_level: float
    n_resamples: int
    n_clusters: int
    standard_error: float
    degenerate_interval: bool = False
    """True when every resample returned the same value and the reported interval is a
    grafted one-sided bound rather than a percentile interval.

    Marked because otherwise the two are indistinguishable in a CSV column: a
    rule-of-three endpoint and a resampled endpoint sit side by side implying the same
    semantics, and only one of them came from the data."""
    p_value_two_sided: float = float("nan")
    """Bootstrap p-value against ``H0: theta = 0``, from the resample distribution.

    A real p rather than "the interval excluded zero, call it 0.05". That encoding cannot
    survive multiplicity control: four tests all reported at the boundary give Holm
    nothing to order, and its first comparison (0.05 <= 0.05/4) fails for every one of
    them -- so a family of four could never contain a single rejection, whatever the data
    said. That is the inverted-criterion failure in a different costume.

    Computed as ``2 * min(P(draw <= 0), P(draw >= 0))`` with the customary +1 correction,
    so the smallest attainable value is ``2 / (n_resamples + 1)`` and never zero: a
    bootstrap cannot prove a p of zero and should not claim one.
    """

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "ci_level": self.ci_level,
            "n_resamples": self.n_resamples,
            "n_clusters": self.n_clusters,
            "standard_error": self.standard_error,
            "degenerate_interval": self.degenerate_interval,
            "p_value_two_sided": self.p_value_two_sided,
        }

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval excludes zero — for difference estimates."""
        return self.lower > 0.0 or self.upper < 0.0


def group_by_cluster(items: Sequence[T], cluster_of: Callable[[T], str]) -> dict[str, list[T]]:
    """Partition observations by their cluster (normally the document)."""
    groups: dict[str, list[T]] = {}
    for item in items:
        groups.setdefault(cluster_of(item), []).append(item)
    return groups


def _bootstrap_p_value(draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value against ``H0: theta = 0``.

    The +1 in numerator and denominator is the standard correction: a bootstrap of B
    resamples cannot distinguish a p below 1/(B+1) from zero, and reporting zero would
    claim certainty the procedure cannot deliver.
    """
    if draws.size == 0:
        return float("nan")
    n = draws.size
    p_le = (1.0 + float(np.sum(draws <= 0.0))) / (n + 1.0)
    p_ge = (1.0 + float(np.sum(draws >= 0.0))) / (n + 1.0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _widen_degenerate(
    estimate: float, lower: float, upper: float, n_clusters: int, bounds: tuple[float, float] | None
) -> tuple[float, float]:
    """Replace a collapsed percentile interval with an honest one-sided bound.

    Fires when every resample returned the same value, which for a proportion almost always
    means a zero-event statistic: no harmful edit in any document, so no resample can
    produce one either. The rule of three gives the 95% one-sided bound at ``3/n`` on the
    CLUSTER count, documents being the resampling unit.

    Three things this gets right that a naive version does not:

    - it widens toward the **interior** of the statistic's range, so a proportion pinned at
      1.0 yields ``[1 - 3/n, 1]`` rather than an upper bound above 1;
    - it clamps to ``bounds``, so the interval cannot leave the range the statistic lives in;
    - it applies **only** when a range is supplied. The rule of three is a binomial result;
      grafting ``3/n`` onto a degenerate calibration error or a paired difference would be
      dimensionally meaningless, so those callers pass ``None`` and get NaN instead.

    The returned interval is a one-sided bound wearing a two-sided label, which is why
    ``degenerate_interval`` marks it in the artifact rather than leaving it
    indistinguishable from a resampled endpoint.
    """
    if bounds is None:
        return float("nan"), float("nan")
    low_bound, high_bound = bounds
    margin = 3.0 / max(n_clusters, 1)
    if estimate >= high_bound - 1e-12:
        lower, upper = high_bound - margin, high_bound
    else:
        lower, upper = estimate, estimate + margin
    return max(low_bound, lower), min(high_bound, upper)


def cluster_bootstrap(
    items: Sequence[T],
    cluster_of: Callable[[T], str],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
    bounds: tuple[float, float] | None = (0.0, 1.0),
) -> BootstrapResult:
    """Percentile bootstrap over clusters.

    ``bounds`` is the range the statistic lives in, consulted only when the resample
    distribution collapses to a point. Default ``(0, 1)`` because every caller here
    bootstraps a proportion; pass ``None`` for a statistic with no such range, and a
    degenerate draw then yields NaN rather than a fabricated bound.

    Percentile rather than BCa: BCa's acceleration term is itself estimated by jackknife
    over clusters, which is unstable when there are few documents — exactly the regime
    this project's per-fold analyses live in. A wider, honest interval beats a narrower
    one built on an unstable correction.
    """
    groups = group_by_cluster(items, cluster_of)
    cluster_ids = sorted(groups)
    n_clusters = len(cluster_ids)
    estimate = statistic(items)

    if n_clusters < 2:
        # One cluster carries no information about between-document variability; saying so
        # is better than reporting a zero-width interval that implies certainty.
        return BootstrapResult(
            estimate=estimate,
            lower=float("nan"),
            upper=float("nan"),
            ci_level=ci_level,
            n_resamples=0,
            n_clusters=n_clusters,
            standard_error=float("nan"),
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        resample: list[T] = []
        for position in chosen:
            resample.extend(groups[cluster_ids[position]])
        draws[index] = statistic(resample)

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.quantile(draws, alpha))
    upper = float(np.quantile(draws, 1.0 - alpha))

    degenerate = upper - lower <= 0.0
    if degenerate:
        lower, upper = _widen_degenerate(estimate, lower, upper, n_clusters, bounds)

    return BootstrapResult(
        estimate=estimate,
        lower=lower,
        upper=upper,
        ci_level=ci_level,
        n_resamples=n_resamples,
        n_clusters=n_clusters,
        standard_error=float(np.std(draws, ddof=1)),
        degenerate_interval=degenerate,
        p_value_two_sided=_bootstrap_p_value(draws),
    )


def paired_cluster_bootstrap(
    items_a: Sequence[T],
    items_b: Sequence[T],
    cluster_of: Callable[[T], str],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
) -> BootstrapResult:
    """Bootstrap the difference ``statistic(a) - statistic(b)`` on shared clusters.

    Both methods are resampled on the *same* draw of documents. Bootstrapping them
    independently would discard the pairing and inflate the interval, obscuring a real
    difference between two methods evaluated on identical pages.
    """
    groups_a = group_by_cluster(items_a, cluster_of)
    groups_b = group_by_cluster(items_b, cluster_of)
    shared = sorted(set(groups_a) & set(groups_b))
    if len(shared) < 2:
        return BootstrapResult(
            estimate=statistic(items_a) - statistic(items_b),
            lower=float("nan"),
            upper=float("nan"),
            ci_level=ci_level,
            n_resamples=0,
            n_clusters=len(shared),
            standard_error=float("nan"),
        )

    paired_a = [item for cluster in shared for item in groups_a[cluster]]
    paired_b = [item for cluster in shared for item in groups_b[cluster]]
    estimate = statistic(paired_a) - statistic(paired_b)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        chosen = rng.integers(0, len(shared), size=len(shared))
        resample_a: list[T] = []
        resample_b: list[T] = []
        for position in chosen:
            cluster = shared[position]
            resample_a.extend(groups_a[cluster])
            resample_b.extend(groups_b[cluster])
        draws[index] = statistic(resample_a) - statistic(resample_b)

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.quantile(draws, alpha))
    upper = float(np.quantile(draws, 1.0 - alpha))

    if upper - lower <= 0.0:
        # The paired path needs this guard as much as the unpaired ones, and lacked it.
        # A constant-score arm makes the difference algebraically independent of which
        # documents are drawn, so every resample returns the same number: zero sampling
        # variability, a 1e-16 interval, and -- because the bootstrap p-value falls to its
        # floor 2/(B+1) -- MAXIMAL significance. Three published rows were flagged as
        # degraded on exactly that basis.
        #
        # NaN rather than a widened interval. On the unpaired paths the statistic is a
        # proportion and the rule of three is a real bound; a paired DIFFERENCE of two
        # arbitrary statistics has no such bound, and inventing one would be worse than
        # admitting the resampling carries no information here.
        return BootstrapResult(
            estimate=estimate,
            lower=float("nan"),
            upper=float("nan"),
            ci_level=ci_level,
            n_resamples=n_resamples,
            n_clusters=len(shared),
            standard_error=0.0,
            p_value_two_sided=float("nan"),
        )

    return BootstrapResult(
        estimate=estimate,
        lower=lower,
        upper=upper,
        ci_level=ci_level,
        n_resamples=n_resamples,
        n_clusters=len(shared),
        standard_error=float(np.std(draws, ddof=1)),
        p_value_two_sided=_bootstrap_p_value(draws),
    )


def cluster_bootstrap_indices(
    clusters: Sequence[str],
    statistic: Callable[[NDArray[np.intp]], float],
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
    bounds: tuple[float, float] | None = (0.0, 1.0),
) -> BootstrapResult:
    """Cluster bootstrap that hands the statistic **row indices** rather than objects.

    Identical procedure to :func:`cluster_bootstrap` -- same clusters, drawn with
    replacement, whole clusters appended, statistic recomputed per resample -- but the
    resample is a numpy index array instead of a rebuilt list of tuples.

    This exists for one reason: at the pilot's scale the generic path spends nearly all of
    its time unpacking Python tuples. A single (fold, verifier, epsilon) cell resamples
    ~2000 candidates 10 000 times, and rebuilding each resample as a list and then
    re-extracting arrays from it costs tens of millions of interpreter operations, while
    the statistic itself is vectorized. Reducing the resample count instead would have
    meant reporting fewer resamples than the pre-registration promised, which is not a
    performance decision to make quietly.

    ``tests/unit/test_stats.py`` asserts the two paths agree exactly on the same seed.
    """
    by_cluster: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        by_cluster.setdefault(cluster, []).append(index)
    cluster_ids = sorted(by_cluster)
    n_clusters = len(cluster_ids)
    members = [np.asarray(by_cluster[c], dtype=np.intp) for c in cluster_ids]
    everything = np.arange(len(clusters), dtype=np.intp)
    estimate = statistic(everything)

    if n_clusters < 2:
        return BootstrapResult(
            estimate=estimate,
            lower=float("nan"),
            upper=float("nan"),
            ci_level=ci_level,
            n_resamples=0,
            n_clusters=n_clusters,
            standard_error=float("nan"),
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        draws[index] = statistic(np.concatenate([members[position] for position in chosen]))

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.quantile(draws, alpha))
    upper = float(np.quantile(draws, 1.0 - alpha))
    degenerate = upper - lower <= 0.0
    if degenerate:
        lower, upper = _widen_degenerate(estimate, lower, upper, n_clusters, bounds)

    return BootstrapResult(
        estimate=estimate,
        lower=lower,
        upper=upper,
        ci_level=ci_level,
        n_resamples=n_resamples,
        n_clusters=n_clusters,
        standard_error=float(np.std(draws, ddof=1)),
        degenerate_interval=degenerate,
        p_value_two_sided=_bootstrap_p_value(draws),
    )


def paired_cluster_bootstrap_multi(
    items_a: Sequence[T],
    items_b: Sequence[T],
    cluster_of: Callable[[T], str],
    statistic: Callable[[Sequence[T]], Mapping[str, float]],
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
) -> dict[str, BootstrapResult]:
    """Paired cluster bootstrap for several statistics computed together.

    Identical procedure to :func:`paired_cluster_bootstrap` — same clusters, one draw
    applied to both arms, statistic recomputed per resample — but ``statistic`` returns a
    mapping and every key gets its own result from **one** pass.

    This exists because the H1 contrast evaluates five calibration quantities that all come
    out of a single ``calibration_report``. Bootstrapping them separately recomputed that
    report five times per resample and discarded four fifths of it each time, which put the
    full harm-policy sensitivity analysis at roughly three hours. Sharing the pass makes it
    the same numbers in a fifth of the time; ``tests/unit/test_stats.py`` asserts the two
    routes agree exactly on the same seed.
    """
    groups_a = group_by_cluster(items_a, cluster_of)
    groups_b = group_by_cluster(items_b, cluster_of)
    shared = sorted(set(groups_a) & set(groups_b))

    paired_a = [item for cluster in shared for item in groups_a[cluster]]
    paired_b = [item for cluster in shared for item in groups_b[cluster]]
    estimates = {
        key: statistic(paired_a)[key] - value for key, value in statistic(paired_b).items()
    }

    if len(shared) < 2:
        return {
            key: BootstrapResult(
                estimate=estimate,
                lower=float("nan"),
                upper=float("nan"),
                ci_level=ci_level,
                n_resamples=0,
                n_clusters=len(shared),
                standard_error=float("nan"),
            )
            for key, estimate in estimates.items()
        }

    keys = sorted(estimates)
    rng = np.random.default_rng(seed)
    draws = {key: np.empty(n_resamples, dtype=np.float64) for key in keys}
    for index in range(n_resamples):
        chosen = rng.integers(0, len(shared), size=len(shared))
        resample_a: list[T] = []
        resample_b: list[T] = []
        for position in chosen:
            cluster = shared[position]
            resample_a.extend(groups_a[cluster])
            resample_b.extend(groups_b[cluster])
        values_a, values_b = statistic(resample_a), statistic(resample_b)
        for key in keys:
            draws[key][index] = values_a[key] - values_b[key]

    alpha = (1.0 - ci_level) / 2.0
    results: dict[str, BootstrapResult] = {}
    for key in keys:
        column = draws[key]
        lower = float(np.quantile(column, alpha))
        upper = float(np.quantile(column, 1.0 - alpha))
        if upper - lower <= 0.0:
            # Same reasoning as the scalar paired path: a paired difference of two
            # arbitrary statistics has no binomial bound to graft, so NaN rather than a
            # fabricated interval and no p-value from zero sampling variability.
            results[key] = BootstrapResult(
                estimate=estimates[key],
                lower=float("nan"),
                upper=float("nan"),
                ci_level=ci_level,
                n_resamples=n_resamples,
                n_clusters=len(shared),
                standard_error=0.0,
                degenerate_interval=True,
                p_value_two_sided=float("nan"),
            )
            continue
        results[key] = BootstrapResult(
            estimate=estimates[key],
            lower=lower,
            upper=upper,
            ci_level=ci_level,
            n_resamples=n_resamples,
            n_clusters=len(shared),
            standard_error=float(np.std(column, ddof=1)),
            p_value_two_sided=_bootstrap_p_value(column),
        )
    return results
