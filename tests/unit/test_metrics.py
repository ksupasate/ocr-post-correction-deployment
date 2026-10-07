"""Metrics, against hand-computed values and an independent reference where one exists.

Self-consistency checks would only prove our function agrees with itself. Where a second
implementation is available (``rapidfuzz`` for edit distance) it is used as an independent
cross-check; everywhere else the expected value is worked out by hand in the test.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from ocr_risk.metrics import (
    EditDecision,
    account,
    aurc,
    brier_score,
    calibration_report,
    cer,
    corpus_cer,
    corpus_wer,
    coverage_at_risk,
    exact_match_rate,
    expected_calibration_error,
    murphy_decomposition,
    per_document_cer,
    risk_coverage_curve,
    wer,
)
from ocr_risk.schemas.enums import (
    CoverageUnit,
    HarmPolicy,
    OutcomeIfAccepted,
    OutcomeIfRejected,
)


# --- text metrics -----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("abc", "abc", 0.0),
        ("abc", "abd", 1 / 3),
        ("abc", "", 1.0),
        ("", "", 0.0),
        ("0.015 mg", "0.15 mg", 1 / 8),
    ],
)
def test_cer_hand_computed(reference: str, hypothesis: str, expected: float) -> None:
    assert cer(reference, hypothesis) == pytest.approx(expected)


def test_cer_of_empty_reference_with_output_is_total_error() -> None:
    """An empty reference with a non-empty hypothesis is 100% error, not a zero-division
    and not 0.0."""
    assert cer("", "spurious") == 1.0


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("the cat sat", "the cat sat", 0.0),
        ("the cat sat", "the dog sat", 1 / 3),
        ("the cat sat", "the cat", 1 / 3),
        ("the cat sat", "", 1.0),
    ],
)
def test_wer_hand_computed(reference: str, hypothesis: str, expected: float) -> None:
    assert wer(reference, hypothesis) == pytest.approx(expected)


def test_corpus_cer_is_not_the_mean_of_document_rates() -> None:
    """Corpus-level aggregation weights by reference length. Averaging per-document rates
    would let a five-character caption outvote a full page."""
    pairs = [("a", "b"), ("a" * 99, "a" * 99)]
    assert corpus_cer(pairs).rate == pytest.approx(1 / 100)
    assert np.mean(per_document_cer(pairs)) == pytest.approx(0.5)


def test_corpus_wer_aggregates_counts() -> None:
    pairs = [("one two", "one three"), ("a b c d", "a b c d")]
    result = corpus_wer(pairs)
    assert (result.errors, result.reference_length) == (1, 6)
    assert result.rate == pytest.approx(1 / 6)


def test_error_rate_addition_accumulates_both_terms() -> None:
    total = corpus_cer([("abc", "abd")]) + corpus_cer([("xy", "xy")])
    assert (total.errors, total.reference_length) == (1, 5)


def test_exact_match_rate() -> None:
    assert exact_match_rate([("a", "a"), ("b", "c")]) == 0.5
    assert exact_match_rate([]) == 0.0


def test_cer_matches_independent_reference_implementation() -> None:
    """Cross-check against rapidfuzz directly, not against our own helper."""
    from rapidfuzz.distance import Levenshtein

    rng = np.random.default_rng(0)
    alphabet = "abc 0.15mg"
    for _ in range(50):
        a = "".join(rng.choice(list(alphabet), size=int(rng.integers(0, 20))))
        b = "".join(rng.choice(list(alphabet), size=int(rng.integers(0, 20))))
        expected = Levenshtein.distance(a, b) / len(a) if a else (0.0 if not b else 1.0)
        assert cer(a, b) == pytest.approx(expected)


# --- edit accounting ----------------------------------------------------------------------
def _decision(
    site: str,
    document: str,
    accepted: bool,
    d_before: int,
    outcome: OutcomeIfAccepted | None,
    delta: int = 0,
) -> EditDecision:
    return EditDecision(
        site_id=site,
        document_id=document,
        accepted=accepted,
        d_before=d_before,
        outcome_if_accepted=outcome,
        outcome_if_rejected=(
            OutcomeIfRejected.PRESERVATION if d_before == 0 else OutcomeIfRejected.MISSED_ERROR
        ),
        delta=delta,
        n_candidates=1,
    )


# 6 sites: 2 accepted-good, 1 accepted-overcorrection, 1 accepted-miscorrection,
# 1 rejected-clean, 1 rejected-with-error.
SCENARIO = [
    _decision("s1", "d1", True, 3, OutcomeIfAccepted.TRUE_CORRECTION, delta=3),
    _decision("s2", "d1", True, 2, OutcomeIfAccepted.PARTIAL_IMPROVEMENT, delta=1),
    _decision("s3", "d1", True, 0, OutcomeIfAccepted.OVERCORRECTION, delta=-2),
    _decision("s4", "d2", True, 4, OutcomeIfAccepted.MISCORRECTION, delta=-1),
    _decision("s5", "d2", False, 0, None),
    _decision("s6", "d2", False, 5, None),
]


def test_edit_accounting_counts_hand_checked() -> None:
    result = account(SCENARIO, HarmPolicy.STRICT_WORSENING)
    assert result.n_sites == 6
    assert result.n_clean_sites == 2  # s3 and s5
    assert result.n_sites_with_error == 4
    assert result.n_accepted == 4
    assert result.n_true_corrections == 1
    assert result.n_partial_improvements == 1
    assert result.n_overcorrections == 1
    assert result.n_miscorrections == 1
    assert result.n_preservations == 1  # s5
    assert result.n_missed_errors == 1  # s6


def test_edit_accounting_rates_hand_checked() -> None:
    result = account(SCENARIO, HarmPolicy.STRICT_WORSENING)
    assert result.coverage == pytest.approx(4 / 6)
    assert result.accepted_edit_risk == pytest.approx(2 / 4)  # over + mis, of 4 accepted
    assert result.joint_harm_rate == pytest.approx(2 / 6)
    assert result.edit_precision == pytest.approx(2 / 4)  # true + partial
    assert result.correction_recall == pytest.approx(2 / 4)  # of 4 error sites
    assert result.preservation_rate == pytest.approx(1 / 2)  # of 2 clean sites
    assert result.overcorrection_rate == pytest.approx(1 / 2)
    assert result.missed_error_rate == pytest.approx(1 / 4)


def test_net_repair_gain_can_be_negative() -> None:
    """A method that breaks more than it fixes must report a loss, not a small gain."""
    destructive = [
        _decision("s1", "d1", True, 1, OutcomeIfAccepted.MISCORRECTION, delta=-5),
        _decision("s2", "d1", True, 1, OutcomeIfAccepted.TRUE_CORRECTION, delta=1),
    ]
    result = account(destructive, HarmPolicy.STRICT_WORSENING)
    assert result.characters_repaired == -4
    assert result.net_repair_gain < 0


def test_harm_policy_changes_the_risk() -> None:
    """The whole reason harm policy is configured rather than fixed."""
    strict = account(SCENARIO, HarmPolicy.STRICT_WORSENING)
    exact = account(SCENARIO, HarmPolicy.EXACT_ONLY)
    assert exact.n_harmful > strict.n_harmful  # partial improvement now counts as harm
    assert exact.accepted_edit_risk > strict.accepted_edit_risk
    assert exact.n_beneficial < strict.n_beneficial


def test_coverage_unit_changes_the_denominator() -> None:
    per_site = account(SCENARIO, HarmPolicy.STRICT_WORSENING, CoverageUnit.SITE)
    per_candidate = account(SCENARIO, HarmPolicy.STRICT_WORSENING, CoverageUnit.CANDIDATE)
    assert per_site.coverage == pytest.approx(4 / 6)
    assert per_candidate.coverage == pytest.approx(4 / 6)  # one candidate per site here


def test_accepted_without_outcome_is_rejected_loudly() -> None:
    bad = [_decision("s", "d", accepted=True, d_before=1, outcome=None)]
    with pytest.raises(ValueError, match="no outcome"):
        account(bad, HarmPolicy.STRICT_WORSENING)


def test_empty_accounting_has_no_nan_rates() -> None:
    result = account([], HarmPolicy.STRICT_WORSENING)
    for value in result.as_dict().values():
        if isinstance(value, float):
            assert not np.isnan(value)


# --- calibration ----------------------------------------------------------------------------
def test_brier_of_perfect_predictions_is_zero() -> None:
    probabilities = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(probabilities, outcomes) == 0.0


def test_brier_of_confidently_wrong_predictions_is_one() -> None:
    assert brier_score(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0


def test_brier_hand_computed() -> None:
    # ((0.8-1)^2 + (0.3-0)^2) / 2 = (0.04 + 0.09) / 2 = 0.065
    assert brier_score(np.array([0.8, 0.3]), np.array([1.0, 0.0])) == pytest.approx(0.065)


def test_perfectly_calibrated_predictions_have_near_zero_ece() -> None:
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0.05, 0.95, size=20_000)
    outcomes = (rng.uniform(size=probabilities.size) < probabilities).astype(float)
    report = calibration_report(probabilities, outcomes, n_bins=10)
    assert report.ece_equal_mass < 0.02
    assert report.ece_equal_width < 0.02


def test_systematic_overconfidence_is_detected() -> None:
    rng = np.random.default_rng(4)
    truth = rng.uniform(0.05, 0.6, size=20_000)
    outcomes = (rng.uniform(size=truth.size) < truth).astype(float)
    inflated = np.clip(truth + 0.3, 0.0, 1.0)
    report = calibration_report(inflated, outcomes, n_bins=10)
    assert report.ece_equal_mass > 0.2
    assert report.calibration_term > 0.05


def test_the_two_binning_schemes_can_disagree() -> None:
    """Exactly why both are reported: a single ECE number is not a reportable result
    without saying which scheme produced it.

    The disagreement is largest with few samples and clustered scores — which is the
    regime a per-fold cross-engine analysis actually lives in, not an exotic corner.
    Uniform miscalibration over many samples makes the schemes agree, so a large-sample
    test here would give false reassurance that the choice does not matter.
    """
    probabilities = np.array([0.05, 0.06, 0.07, 0.90, 0.91, 0.92, 0.93, 0.94])
    outcomes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    width, _ = expected_calibration_error(probabilities, outcomes, 4, "equal_width")
    mass, _ = expected_calibration_error(probabilities, outcomes, 4, "equal_mass")
    assert abs(width - mass) > 0.01, (
        f"binning scheme made no difference (width={width:.4f}, mass={mass:.4f}); "
        "the fixture no longer exercises the disagreement it was built for"
    )


def test_unknown_binning_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown binning scheme"):
        expected_calibration_error(np.array([0.5]), np.array([1.0]), 5, "made_up")


def test_murphy_decomposition_separates_calibration_from_refinement() -> None:
    """A calibrated-but-uninformative predictor should have low calibration error and
    high refinement — the distinction that says whether recalibration would help."""
    rng = np.random.default_rng(6)
    base = 0.3
    outcomes = (rng.uniform(size=10_000) < base).astype(float)
    constant = np.full_like(outcomes, base)
    calibration, refinement, uncertainty = murphy_decomposition(
        constant, outcomes, 10, "equal_mass"
    )
    assert calibration < 0.005
    assert refinement == pytest.approx(uncertainty, abs=0.01)


def test_calibration_report_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        calibration_report(np.array([0.5, 0.5]), np.array([1.0]))


def test_reliability_bins_omit_empty_bins() -> None:
    """An empty bin reported with zero gap would read as perfectly calibrated."""
    report = calibration_report(np.array([0.9, 0.91, 0.92]), np.array([1.0, 1.0, 0.0]), n_bins=10)
    assert all(b.n > 0 for b in report.reliability)


# --- selective / risk-coverage -------------------------------------------------------------
def test_coverage_is_non_increasing_in_the_threshold() -> None:
    rng = np.random.default_rng(1)
    scores = rng.uniform(size=500)
    harmful = rng.uniform(size=500) < 0.2
    curve = risk_coverage_curve(scores, harmful, n_thresholds=50)
    coverages = [p.coverage for p in curve.points]
    assert all(a >= b for a, b in pairwise(coverages))


def test_perfect_ranking_reaches_zero_risk_at_positive_coverage() -> None:
    """A verifier that ranks every harmful edit below every safe one must be able to
    accept all the safe ones at zero risk."""
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    harmful = np.array([False, False, False, True, True])
    curve = risk_coverage_curve(scores, harmful)
    point = coverage_at_risk(curve, 0.0)
    assert point is not None
    assert point.coverage == pytest.approx(3 / 5)
    assert point.risk == 0.0


def test_coverage_at_risk_searches_the_whole_curve() -> None:
    """Selective risk is not monotone in the threshold, so stopping at the first
    crossing would understate what a method can safely do."""
    # The top-scoring edit is harmful, so risk starts at 1.0, dips, then rises again.
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    harmful = np.array([True, False, False, False, False, True])
    curve = risk_coverage_curve(scores, harmful)
    point = coverage_at_risk(curve, 0.25)
    assert point is not None
    assert point.n_accepted >= 4, "gave up before finding the best feasible operating point"


def test_no_feasible_point_returns_none() -> None:
    """When every accepted edit is harmful there is no operating point, and saying so
    beats returning a zero-coverage point that looks safe."""
    curve = risk_coverage_curve(np.array([0.9, 0.8]), np.array([True, True]))
    assert coverage_at_risk(curve, 0.1) is None


def test_joint_harm_rate_is_stable_where_selective_risk_is_not() -> None:
    scores = np.array([0.99, 0.5, 0.4])
    harmful = np.array([True, False, False])
    curve = risk_coverage_curve(scores, harmful)
    top = max(curve.points, key=lambda p: p.threshold if p.n_accepted else -1)
    assert top.risk == 1.0  # one accepted, and it is harmful
    assert top.joint_harm_rate == pytest.approx(1 / 3)


def test_aurc_prefers_a_better_ranking() -> None:
    harmful = np.array([True, True, False, False, False, False])
    good_ranking = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 1.0])  # harmful ranked last
    bad_ranking = np.array([1.0, 0.9, 0.2, 0.3, 0.4, 0.5])  # harmful ranked first
    assert aurc(risk_coverage_curve(good_ranking, harmful)) < aurc(
        risk_coverage_curve(bad_ranking, harmful)
    )


def test_empty_curve_is_handled() -> None:
    """An empty curve has no area, and NaN says that. Zero would be the best possible
    value, which would rank a method that produced nothing above every method that did."""
    curve = risk_coverage_curve(np.array([]), np.array([], dtype=bool))
    assert curve.points == ()
    assert math.isnan(aurc(curve))
    assert coverage_at_risk(curve, 0.1) is None


def test_curve_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        risk_coverage_curve(np.array([0.5, 0.5]), np.array([True]))


def test_aurc_of_a_degenerate_frontier_is_its_own_risk_not_zero() -> None:
    """A constant score yields one operating point, and zero is the BEST possible area.

    The accept-everything reference emits a constant score, so its curve collapses to a
    single point. Returning 0.0 there ranked it as the best method on every fold of the
    real pilot while it is by construction the worst: it accepts every harmful edit. The
    area-average of one point is its own risk, which for accept-everything is exactly the
    base harm rate.
    """
    import numpy as np

    from ocr_risk.metrics import aurc, risk_coverage_curve

    scores = np.full(200, 0.5)
    harmful = np.zeros(200, dtype=bool)
    harmful[:60] = True  # base harm rate 0.30

    value = aurc(risk_coverage_curve(scores, harmful))
    assert value == pytest.approx(0.30), "a single-point frontier reports that point's risk"
