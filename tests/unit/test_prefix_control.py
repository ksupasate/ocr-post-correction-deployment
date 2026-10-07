"""Declared prefix families and the two multiplicity rules that run over them.

The interesting claim is the fixed-sequence one: that testing nested prefixes shallow to deep at
the *full* level, and stopping at the first failure, still bounds the chance of deploying an
unsafe prefix by that level. It is a familiar argument, but it depends on the order being fixed
before the data are read and on each marginal test being honest, so the tests below check the
argument's conclusion by simulation rather than asserting that it was implemented.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr_risk.risk.prefix_control import (
    PREFIX_CONTROLS,
    assert_nested,
    prefix_thresholds,
    select_prefix,
)

NINE = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def _accepts(indices: set[int]) -> object:
    """A certifier that certifies exactly the named prefixes, whatever the level."""

    def certifies(index: int, _delta: float) -> bool:
        return index in indices

    return certifies


# ------------------------------------------------------------------------------ the family


def test_an_empty_family_is_not_a_family() -> None:
    with pytest.raises(ValueError, match="at least one depth"):
        assert_nested([])


@pytest.mark.parametrize("depths", [(0.0, 0.5), (0.5, 1.5), (-0.1, 0.5)])
def test_depths_must_be_acceptance_fractions(depths: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
        assert_nested(depths)


@pytest.mark.parametrize("depths", [(0.5, 0.2), (0.2, 0.2), (0.1, 0.3, 0.3)])
def test_the_order_must_be_strict_because_fixed_sequence_depends_on_it(
    depths: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        assert_nested(depths)


def test_a_valid_family_comes_back_as_floats() -> None:
    assert assert_nested([0.1, 0.5, 1]) == (0.1, 0.5, 1.0)


# -------------------------------------------------------------------------- placing the family


def test_thresholds_are_the_quantiles_of_the_scores_they_are_given() -> None:
    placed = prefix_thresholds(np.arange(100.0), (0.1, 0.5))
    assert placed == pytest.approx([89.1, 49.5])


def test_a_deeper_prefix_sits_at_a_lower_threshold() -> None:
    placed = prefix_thresholds(np.arange(100.0), NINE)
    assert np.all(np.diff(placed) < 0.0)


def test_an_empty_score_vector_places_a_family_that_accepts_nothing() -> None:
    placed = prefix_thresholds(np.zeros(0), (0.1, 0.5))
    assert placed.shape == (2,)
    assert np.all(np.isinf(placed))


def test_placing_a_family_validates_it_first() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        prefix_thresholds(np.arange(10.0), (0.5, 0.2))


# ------------------------------------------------------------------------------- the controls


def test_an_unknown_control_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown prefix control"):
        select_prefix(NINE, _accepts(set()), delta=0.05, control="hopeful")  # type: ignore[arg-type]


@pytest.mark.parametrize("control", PREFIX_CONTROLS)
@pytest.mark.parametrize("delta", [0.0, 1.0, -0.5])
def test_every_control_refuses_a_level_outside_the_unit_interval(
    control: str, delta: float
) -> None:
    with pytest.raises(ValueError, match="delta must lie"):
        select_prefix(NINE, _accepts(set()), delta=delta, control=control)


def test_the_union_bound_splits_the_level_across_the_declared_family() -> None:
    seen: list[float] = []

    def certifies(_index: int, level: float) -> bool:
        seen.append(level)
        return True

    decision = select_prefix(NINE, certifies, delta=0.045, control="union")
    assert decision.per_test_delta == pytest.approx(0.005)
    assert seen == [pytest.approx(0.005)] * 9
    assert decision.tested == 9


def test_the_union_bound_deploys_the_deepest_prefix_it_certified() -> None:
    decision = select_prefix(NINE, _accepts({0, 2, 6}), delta=0.05, control="union")
    assert decision.certified == (0, 2, 6)
    assert decision.selected == 6
    assert decision.feasible
    assert decision.stopped_at == -1


def test_the_union_bound_can_certify_a_gap_that_fixed_sequence_cannot_reach() -> None:
    """The power difference between the two rules, stated as a test rather than as prose."""
    scattered = _accepts({0, 5})
    union = select_prefix(NINE, scattered, delta=0.05, control="union")
    sequential = select_prefix(NINE, scattered, delta=0.05, control="fixed_sequence")
    assert union.selected == 5
    assert sequential.selected == 0
    assert sequential.stopped_at == 1


def test_certifying_nothing_is_a_refusal_and_not_a_shallow_deployment() -> None:
    for control in PREFIX_CONTROLS:
        decision = select_prefix(NINE, _accepts(set()), delta=0.05, control=control)
        assert not decision.feasible
        assert decision.selected == -1
        assert decision.certified == ()


def test_fixed_sequence_spends_the_whole_level_on_each_test() -> None:
    seen: list[float] = []

    def certifies(index: int, level: float) -> bool:
        seen.append(level)
        return index < 3

    decision = select_prefix(NINE, certifies, delta=0.05, control="fixed_sequence")
    assert seen == [pytest.approx(0.05)] * 4
    assert decision.per_test_delta == pytest.approx(0.05)
    assert decision.certified == (0, 1, 2)
    assert decision.selected == 2
    assert decision.stopped_at == 3
    assert decision.tested == 4


def test_fixed_sequence_examines_the_whole_family_only_when_it_never_stops() -> None:
    decision = select_prefix(NINE, _accepts(set(range(9))), delta=0.05, control="fixed_sequence")
    assert decision.selected == 8
    assert decision.stopped_at == -1
    assert decision.tested == 9


def test_fixed_sequence_stops_immediately_when_the_shallowest_prefix_fails() -> None:
    decision = select_prefix(NINE, _accepts({1, 2}), delta=0.05, control="fixed_sequence")
    assert decision.tested == 1
    assert decision.stopped_at == 0
    assert not decision.feasible


def test_a_decision_records_everything_needed_to_audit_it() -> None:
    payload = select_prefix(NINE, _accepts({0, 1}), delta=0.05, control="fixed_sequence").as_dict()
    assert payload == {
        "control": "fixed_sequence",
        "family_size": 9,
        "per_test_delta": 0.05,
        "certified": [0, 1],
        "selected": 1,
        "tested": 3,
        "stopped_at": 2,
        "feasible": True,
    }


# ---------------------------------------------------------------- the familywise error rate


def _simulate(control: str, family_size: int, first_unsafe: int, trials: int) -> float:
    """How often a deployed prefix is one the truth says is unsafe.

    Each marginal test is exactly valid: it rejects a true null with probability ``delta`` and
    a false null always. That is the weakest assumption the fixed-sequence argument needs, so a
    simulation under it measures the argument and not the certifier.
    """
    generator = np.random.default_rng(4711)
    delta = 0.05
    failures = 0
    for _ in range(trials):
        draws = generator.random(family_size)

        def certifies(index: int, level: float, draws: np.ndarray = draws) -> bool:
            return True if index < first_unsafe else bool(draws[index] < level)

        decision = select_prefix(
            tuple(0.01 * (i + 1) for i in range(family_size)),
            certifies,
            delta=delta,
            control=control,
        )
        if decision.selected >= first_unsafe:
            failures += 1
    return failures / trials


@pytest.mark.parametrize("control", PREFIX_CONTROLS)
def test_neither_control_deploys_an_unsafe_prefix_more_often_than_its_level(
    control: str,
) -> None:
    assert _simulate(control, 100, 40, 4000) <= 0.05


def test_fixed_sequence_spends_its_whole_level_where_the_union_bound_spends_a_hundredth() -> None:
    """Both are valid; the union bound is valid by being far more conservative than it needs."""
    sequential = _simulate("fixed_sequence", 100, 40, 4000)
    union = _simulate("union", 100, 40, 4000)
    assert union < sequential
    assert sequential > 0.005
