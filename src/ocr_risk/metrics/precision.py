"""How many accepted edits a risk target actually needs before it means anything.

The H1 pilot pre-registered operating points of 1%, 0.5% and 0.1% harmful accepted edits
and measured coverage of exactly zero at all three. Reading that as "the method is too
conservative" was one interpretation. This module computes the other one, and it does not
need a model or a result: at a finite number of accepted edits, most of those targets
**cannot be certified by any method whatsoever**, because the finite-sample bound the
controller uses never gets below them.

The bound is the repository's own — :func:`ocr_risk.risk.bounds.risk_upper_bound`, the
same LTT/Bentkus construction the controller certifies thresholds with, not a
normal-approximation stand-in. One convention differs: the controller splits its delta
across the threshold grid (``per_test_delta = delta / n_thresholds``) because it certifies
one of many candidate thresholds, while this module inverts the bound at the flat ``delta``
unless ``n_thresholds`` is passed. At flat delta the requirement is therefore a *lower
bound* on what the controller itself would need — at eps=0.05 with harm at eps/2, the
requirement is 245 accepted edits at flat delta 0.1 (first certifiable: 218) against 843
at the controller's per-test delta with the pilot's 200-point grid (first certifiable:
816). Plan with ``n_thresholds`` set to the grid you will actually certify over.

Use it **before** an experiment, to decide which operating points are worth pre-registering,
and after one, to say which reported zeros were informative and which were arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ocr_risk.risk.bounds import risk_upper_bound

__all__ = [
    "PrecisionRequirement",
    "attainable_epsilon",
    "minimum_accepted_edits",
    "precision_table",
]

_SEARCH_CEILING = 2_000_000


def minimum_accepted_edits(
    epsilon: float,
    delta: float = 0.1,
    observed_harm_rate: float = 0.0,
    *,
    n_thresholds: int | None = None,
) -> int | None:
    """A count ``n`` whose upper confidence bound on risk falls at or below ``epsilon``.

    ``observed_harm_rate`` is what the accepted edits actually turn out to contain. At 0 it
    gives the most favourable case — a method that accepted nothing harmful — and is
    therefore a *floor* on the sample size, not an estimate of it. Returns ``None`` when no
    reachable ``n`` suffices, which is the honest answer when the observed rate is at or
    above the target: no amount of data certifies a bound the data violates.

    ``n_thresholds`` selects the controller's grid-split convention: the RiskController
    certifies one threshold out of a grid and spends ``delta / n_thresholds`` per test, so
    passing the grid size inverts the bound at the delta a deployed controller would
    actually certify with. The default keeps the flat-``delta`` convention the readiness
    protocol was frozen under; those frozen numbers are a lower bound on the controller's
    own requirement, never an overstatement of it.

    With ``observed_harm_rate > 0`` the predicate is **not monotone** in ``n``: the harm
    count is ``round(n * rate)``, and each time the rounded count increments the bound
    jumps up, so windows of failing ``n`` exist above the first certifiable count (at
    eps=0.05, harm eps/2, a linear scan first certifies at n=218 while n=261-271 and 293
    fail; this bisection returns 245). The returned number is therefore neither the
    smallest certifiable count nor a threshold above which every count certifies — it is
    the value the bisection converges to under this rounding convention, and it is
    conservative against the first-true count at every published operating point. The
    readiness protocol's amendment records the measured windows.
    """
    if not 0.0 < epsilon < 1.0:
        msg = f"epsilon must lie strictly between 0 and 1; got {epsilon}"
        raise ValueError(msg)
    per_test_delta = delta / n_thresholds if n_thresholds else delta

    def certified(n: int) -> bool:
        harmful = round(n * observed_harm_rate)
        return risk_upper_bound(harmful, n, per_test_delta) <= epsilon

    if not certified(_SEARCH_CEILING):
        return None
    low, high = 1, _SEARCH_CEILING
    while low < high:
        middle = (low + high) // 2
        if certified(middle):
            high = middle
        else:
            low = middle + 1
    return low


def attainable_epsilon(n_accepted: int, delta: float = 0.1, observed_harm: int = 0) -> float:
    """The tightest risk the evidence supports at ``n_accepted`` accepted edits.

    The mirror of :func:`minimum_accepted_edits`, and the number to quote when a target was
    missed: "0.1% was not resolvable here; the data supports 4.8%" says something, whereas
    "coverage was zero" does not distinguish a cautious method from an impossible target.
    """
    if n_accepted <= 0:
        return 1.0
    return risk_upper_bound(observed_harm, n_accepted, delta)


@dataclass(frozen=True, slots=True)
class PrecisionRequirement:
    """One operating point, and whether a given pool could ever reach it."""

    epsilon: float
    delta: float
    n_required_if_clean: int | None
    n_required_at_half_epsilon: int | None
    n_available: int
    resolvable: bool
    """True when the pool can supply the edits the clean case needs. Necessary, not
    sufficient: a real verifier accepts fewer than the ceiling, and any harm at all pushes
    the requirement toward the ``half_epsilon`` column."""

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "n_required_if_clean": self.n_required_if_clean,
            "n_required_at_half_epsilon": self.n_required_at_half_epsilon,
            "n_available": self.n_available,
            "resolvable": self.resolvable,
        }


def precision_table(
    epsilon_grid: Sequence[float], n_available: int, delta: float = 0.1
) -> list[PrecisionRequirement]:
    """Every operating point against the accepted edits a pool could supply."""
    out: list[PrecisionRequirement] = []
    for epsilon in epsilon_grid:
        clean = minimum_accepted_edits(epsilon, delta, 0.0)
        half = minimum_accepted_edits(epsilon, delta, epsilon / 2.0)
        out.append(
            PrecisionRequirement(
                epsilon=epsilon,
                delta=delta,
                n_required_if_clean=clean,
                n_required_at_half_epsilon=half,
                n_available=n_available,
                resolvable=clean is not None and n_available >= clean,
            )
        )
    return out
