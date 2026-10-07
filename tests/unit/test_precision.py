"""Sample-size requirements for a risk target, checked against the bound itself.

These decide which operating points a study may pre-register, so the tests hold the two
properties that make the answer usable: it agrees with the bound it claims to invert, and
it refuses rather than guessing when no sample size would do.
"""

from __future__ import annotations

import pytest

from ocr_risk.metrics.precision import (
    attainable_epsilon,
    minimum_accepted_edits,
    precision_table,
)
from ocr_risk.risk.bounds import risk_upper_bound


@pytest.mark.parametrize("epsilon", [0.10, 0.05, 0.02, 0.01, 0.005])
def test_the_requirement_is_the_exact_inverse_of_the_bound(epsilon: float) -> None:
    """``n`` certifies and ``n - 1`` does not — the definition of "smallest"."""
    n = minimum_accepted_edits(epsilon, delta=0.1)
    assert n is not None
    assert risk_upper_bound(0, n, 0.1) <= epsilon
    assert risk_upper_bound(0, n - 1, 0.1) > epsilon


def test_a_tighter_target_needs_strictly_more_evidence() -> None:
    grid = [0.10, 0.05, 0.02, 0.01, 0.005, 0.001]
    required = [minimum_accepted_edits(e) for e in grid]
    assert all(n is not None for n in required)
    assert required == sorted(required)  # type: ignore[type-var]
    assert required[0] != required[-1]


def test_observed_harm_at_the_target_can_never_be_certified() -> None:
    """No amount of data certifies a bound the data violates. ``None``, not a huge number."""
    assert minimum_accepted_edits(0.01, observed_harm_rate=0.01) is None
    assert minimum_accepted_edits(0.01, observed_harm_rate=0.05) is None


def test_harm_below_the_target_is_certifiable_but_costs_more_data() -> None:
    clean = minimum_accepted_edits(0.01, observed_harm_rate=0.0)
    halfway = minimum_accepted_edits(0.01, observed_harm_rate=0.005)
    assert clean is not None and halfway is not None
    assert halfway > clean


def test_the_attainable_risk_is_what_the_evidence_actually_supports() -> None:
    """The number to quote when a target was missed, instead of reporting zero coverage."""
    assert attainable_epsilon(0) == 1.0
    assert attainable_epsilon(100, observed_harm=0) == pytest.approx(risk_upper_bound(0, 100, 0.1))
    assert attainable_epsilon(1000) < attainable_epsilon(100)


def test_a_pre_registered_grid_is_marked_resolvable_only_where_the_pool_can_reach_it() -> None:
    """The pilot's own grid, against the accepted edits its best pool could supply.

    342 stands for the largest number of accepted edits a thin pool could supply even with
    a *perfect* verifier (the recovery study's best cell holds 595; several hold far less).
    The point of the check is that this is decidable before any model is trained.
    """
    table = {row.epsilon: row for row in precision_table([0.01, 0.005, 0.001], n_available=342)}
    assert not table[0.005].resolvable
    assert not table[0.001].resolvable
    assert table[0.001].n_required_if_clean is not None
    assert table[0.001].n_required_if_clean > 342 * 5


def test_the_controller_convention_is_stricter_than_the_flat_one_the_protocol_froze() -> None:
    """The controller spends delta/n_thresholds per threshold test; the frozen readiness
    table inverted at flat delta. At eps=0.05 the requirement goes 65 -> 168 clean and
    245 -> 843 at harm eps/2, so planning off the flat column understates the requirement
    by roughly 3x (amendment A10.4; R2_h2_readiness_statistics.csv carries both
    conventions). 843 is the bisection's return; the first certifiable count is 816."""
    assert minimum_accepted_edits(0.05, 0.1, 0.0) == 65
    assert minimum_accepted_edits(0.05, 0.1, 0.0, n_thresholds=200) == 168
    assert minimum_accepted_edits(0.05, 0.1, 0.025) == 245
    assert minimum_accepted_edits(0.05, 0.1, 0.025, n_thresholds=200) == 843


def test_the_half_epsilon_column_is_a_rounding_convention_not_a_monotone_threshold() -> None:
    """Pinned from the linear scan behind amendment A10.3: at eps=0.05 with harm at eps/2,
    the first certifiable count is 218 (the bisection returns 245), and the failing counts
    above the published value are exactly 261-271 and 293. The published number is
    conservative against first-true and is a threshold for nothing beyond the window."""
    rate = 0.025

    def certified(n: int) -> bool:
        return risk_upper_bound(round(n * rate), n, 0.1) <= 0.05

    assert not any(certified(n) for n in range(1, 218))
    assert certified(218)
    assert minimum_accepted_edits(0.05, 0.1, rate) == 245
    assert [n for n in range(245, 300) if not certified(n)] == [*range(261, 272), 293]
    assert certified(294)


def test_an_impossible_epsilon_is_rejected_rather_than_searched_for() -> None:
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            minimum_accepted_edits(bad)


def test_the_requirement_serializes_with_the_pool_it_was_judged_against() -> None:
    payload = precision_table([0.01], n_available=50)[0].as_dict()
    assert payload["epsilon"] == 0.01
    assert payload["n_available"] == 50
    assert payload["resolvable"] is False
