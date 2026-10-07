"""The degenerate inputs every metric will eventually be handed.

``metrics/`` and ``risk/`` are held to 100% because a metric that misbehaves on an empty
or degenerate input does not fail loudly — it returns a number, and the number goes into a
table. Each case here fixes what that number must be, with the reason it is that value and
not the obvious alternative.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr_risk.metrics.calibration import (
    CalibrationReport,
    ReliabilityBin,
    brier_score,
    calibration_report,
    expected_calibration_error,
    murphy_decomposition,
    reliability_bins,
)
from ocr_risk.metrics.edit_accounting import EditDecision, account
from ocr_risk.metrics.selective import (
    RiskCoverageCurve,
    RiskCoveragePoint,
    aurc,
    risk_coverage_curve,
)
from ocr_risk.risk.bounds import bentkus_p_value, hoeffding_p_value, smallest_controlled_risk
from ocr_risk.risk.controller import _candidate_thresholds, select_threshold
from ocr_risk.schemas.enums import HarmPolicy, OutcomeIfAccepted, OutcomeIfRejected

EMPTY = np.zeros(0, dtype=np.float64)


# ------------------------------------------------------------------------- calibration


def test_every_calibration_statistic_of_nothing_is_zero_not_an_exception() -> None:
    """An engine that produced no evaluable candidate must not crash the report builder.

    Zero is defensible only because the accompanying ``n`` is zero; a reader who quotes
    ``brier=0.0`` without it is quoting a count, not a score.
    """
    assert brier_score(EMPTY, EMPTY) == 0.0
    assert reliability_bins(EMPTY, EMPTY, n_bins=10, binning="equal_mass") == []
    assert expected_calibration_error(EMPTY, EMPTY, n_bins=10, binning="equal_width") == (0.0, 0.0)
    assert murphy_decomposition(EMPTY, EMPTY, n_bins=10, binning="equal_mass") == (0.0, 0.0, 0.0)


def _report(binning: str) -> CalibrationReport:
    return CalibrationReport(
        n=4,
        brier=0.25,
        ece_equal_width=0.30,
        ece_equal_mass=0.10,
        max_calibration_error=0.40,
        n_bins=5,
        binning=binning,
        reliability=(ReliabilityBin(0.0, 0.5, 4, 0.25, 0.5),),
        calibration_term=0.06,
        refinement_term=0.19,
        base_rate=0.5,
    )


def test_the_headline_ece_follows_the_configured_binning_scheme() -> None:
    """Equal-width and equal-mass differ by 0.20 here, so picking the wrong one is visible."""
    assert _report("equal_mass").ece == 0.10
    assert _report("equal_width").ece == 0.30


def test_the_serialized_report_carries_the_scheme_that_produced_its_ece() -> None:
    """An ECE without its binning scheme and bin count is not a reportable number."""
    payload = _report("equal_mass").as_dict()
    assert payload["ece"] == payload["ece_equal_mass"] == 0.10
    assert payload["binning"] == "equal_mass"
    assert payload["n_bins"] == 5
    assert payload["reliability"] == [
        {"lower": 0.0, "upper": 0.5, "n": 4, "mean_predicted": 0.25, "mean_observed": 0.5}
    ]


def test_the_report_of_an_empty_evaluation_is_still_well_formed() -> None:
    report = calibration_report(EMPTY, EMPTY, n_bins=10, binning="equal_mass")
    assert report.n == 0
    assert report.reliability == ()


# --------------------------------------------------------------------------- selective


def test_a_curve_exposes_its_axes_in_point_order() -> None:
    curve = RiskCoverageCurve(
        points=(
            RiskCoveragePoint(threshold=0.9, n_considered=10, n_accepted=2, n_harmful=0),
            RiskCoveragePoint(threshold=0.1, n_considered=10, n_accepted=10, n_harmful=3),
        )
    )
    assert curve.coverages == [0.2, 1.0]
    assert curve.risks == [0.0, 0.3]
    payload = curve.as_dict()
    assert payload["joint_harm_rate"] == [0.0, 0.3]
    assert payload["n_harmful"] == [0, 3]


def test_a_frontier_with_no_width_is_summarized_by_its_mean_risk() -> None:
    """Several distinct thresholds that all accept the same set have zero coverage span.

    Integrating over a zero span is a division by zero; the mean of the risks is the only
    value that keeps the answer on the same scale as a real area.
    """
    curve = RiskCoverageCurve(
        points=(
            RiskCoveragePoint(threshold=0.3, n_considered=10, n_accepted=4, n_harmful=1),
            RiskCoveragePoint(threshold=0.2, n_considered=10, n_accepted=4, n_harmful=3),
        )
    )
    assert aurc(curve) == pytest.approx(0.5)


def test_the_threshold_grid_is_thinned_by_quantile_when_scores_outnumber_it() -> None:
    """200 distinct scores into a 5-point grid must still produce a usable sweep."""
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    curve = risk_coverage_curve(scores, np.zeros(200, dtype=bool), n_thresholds=5)
    assert 2 <= len(curve.points) <= 7
    assert curve.points[-1].n_accepted in (0, 200)


def test_a_small_score_set_becomes_the_grid_itself_rather_than_its_quantiles() -> None:
    """Quantiles of 6 scores would invent thresholds between them and blur the sweep."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    grid = _candidate_thresholds(scores, n_grid=200)
    assert list(grid[:-1]) == pytest.approx(list(scores))
    assert grid[-1] > scores.max(), "an abstain-everything point must always be reachable"


# ---------------------------------------------------------------------- edit accounting


def test_an_outcome_with_no_dedicated_counter_still_lands_in_the_totals() -> None:
    """``identity`` has no counter field, so only the generic by-outcome map catches it.

    Identity proposals are filtered from the pool upstream, which is exactly why this path
    goes unexercised in a normal run and why a regression in it would surface as a total
    that quietly stops adding up.
    """
    decision = EditDecision(
        site_id="s1",
        document_id="d1",
        accepted=True,
        d_before=3,
        outcome_if_accepted=OutcomeIfAccepted.IDENTITY,
        outcome_if_rejected=OutcomeIfRejected.MISSED_ERROR,
        delta=0,
        n_candidates=1,
    )
    result = account([decision], harm_policy=HarmPolicy.STRICT_WORSENING)
    assert result.n_accepted == 1
    assert result.by_outcome["identity"] == 1
    assert result.n_lateral_changes == 0
    assert result.n_harmful == 0
    assert result.n_beneficial == 0


# -------------------------------------------------------------------------- risk bounds


@pytest.mark.parametrize("p_value", [hoeffding_p_value, bentkus_p_value])
def test_a_bound_computed_from_no_data_certifies_nothing(p_value: object) -> None:
    """p=1.0, never 0.0. An empty calibration split must not certify every threshold."""
    assert p_value(0, 0.0, 0.01) == 1.0  # type: ignore[operator]
    assert p_value(-1, 0.0, 0.01) == 1.0  # type: ignore[operator]


def test_the_controller_alias_and_the_bound_are_the_same_function() -> None:
    from ocr_risk.risk.bounds import risk_upper_bound

    assert smallest_controlled_risk(2, 100, 0.1) == risk_upper_bound(2, 100, 0.1)


def test_a_threshold_decision_serializes_the_tolerance_it_was_certified_against() -> None:
    """A tau without its epsilon and delta is a number nobody can check later."""
    rng = np.random.default_rng(1)
    scores = rng.random(400)
    harmful = rng.random(400) < 0.02
    payload = select_threshold(scores, harmful, epsilon=0.05, delta=0.1).as_dict()
    assert payload["epsilon"] == 0.05
    assert payload["delta"] == 0.1
    assert set(payload) >= {"tau", "epsilon", "delta", "controller", "feasible"}


def test_the_three_coverages_answer_different_questions() -> None:
    """Repair coverage low with beneficial-edit coverage high is a Q1 failure; both low is
    a Q2 failure. The pilot reported one number and could not tell them apart.

    Four error sites. The generator offered a beneficial candidate at two of them, and the
    verifier accepted one. So: 1/4 sites repaired, 1/2 of what was offered accepted.
    """
    decisions = [
        EditDecision(
            site_id="offered_and_taken",
            document_id="d",
            accepted=True,
            d_before=3,
            outcome_if_accepted=OutcomeIfAccepted.TRUE_CORRECTION,
            outcome_if_rejected=OutcomeIfRejected.MISSED_ERROR,
            delta=3,
            n_candidates=2,
            n_beneficial_available=1,
        ),
        EditDecision(
            site_id="offered_and_declined",
            document_id="d",
            accepted=False,
            d_before=2,
            outcome_if_accepted=None,
            outcome_if_rejected=OutcomeIfRejected.MISSED_ERROR,
            n_candidates=2,
            n_beneficial_available=1,
        ),
        *[
            EditDecision(
                site_id=f"nothing_offered_{i}",
                document_id="d",
                accepted=False,
                d_before=4,
                outcome_if_accepted=None,
                outcome_if_rejected=OutcomeIfRejected.MISSED_ERROR,
                n_candidates=0,
                n_beneficial_available=0,
            )
            for i in range(2)
        ],
    ]
    result = account(decisions, harm_policy=HarmPolicy.STRICT_WORSENING)
    assert result.coverage == pytest.approx(0.25)
    assert result.repair_coverage == pytest.approx(0.25)
    assert result.beneficial_edit_coverage == pytest.approx(0.5)
    payload = result.as_dict()
    assert payload["n_beneficial_available"] == 2


def test_beneficial_edit_coverage_is_zero_when_nothing_was_offered() -> None:
    """Not NaN and not one: a generator that offered nothing did not have its candidates
    accepted, and the denominator is what the diagnostic is about."""
    decision = EditDecision(
        site_id="s",
        document_id="d",
        accepted=False,
        d_before=3,
        outcome_if_accepted=None,
        outcome_if_rejected=OutcomeIfRejected.MISSED_ERROR,
    )
    assert (
        account([decision], harm_policy=HarmPolicy.STRICT_WORSENING).beneficial_edit_coverage == 0.0
    )
