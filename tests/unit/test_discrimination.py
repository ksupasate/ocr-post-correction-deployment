"""Discrimination metrics against values worked out by hand.

Each expected number below is computed on paper in the docstring, not read back from the
function. A metric checked only against its own output cannot catch the class of error
that matters here: a plausible number that is systematically wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr_risk.metrics.discrimination import (
    average_precision,
    discrimination_report,
    resolution,
    roc_auc,
    score_separation,
)


def test_perfect_and_inverted_ranking_are_one_and_zero() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    positive = np.array([1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, positive) == pytest.approx(1.0)
    assert roc_auc(-scores, positive) == pytest.approx(0.0)


def test_roc_auc_matches_the_hand_counted_pair_comparison() -> None:
    """3 positives x 2 negatives = 6 pairs. Positives 0.7, 0.5, 0.2; negatives 0.6, 0.1.

    0.7 beats both. 0.5 beats 0.1, loses to 0.6. 0.2 beats 0.1, loses to 0.6.
    Wins = 2 + 1 + 1 = 4, ties = 0, so AUC = 4/6 = 0.6667.
    """
    scores = np.array([0.7, 0.5, 0.2, 0.6, 0.1])
    positive = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, positive) == pytest.approx(4.0 / 6.0)


def test_a_constant_score_discriminates_at_exactly_chance() -> None:
    """The accept-everything reference emits a constant. Every pair is a tie, so 0.5.

    A threshold sweep with a ``>=`` comparison reports 1.0 here, which would rank the
    worst possible verifier above every real one -- the same failure AURC already had.
    """
    scores = np.full(8, 0.42)
    positive = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, positive) == pytest.approx(0.5)
    assert score_separation(scores, positive) == 0.0


def test_ties_between_the_classes_count_as_half_a_win() -> None:
    """Positives 0.5, 0.5; negatives 0.5, 0.1. Pairs: two ties, two wins → (2*0.5+2)/4."""
    scores = np.array([0.5, 0.5, 0.5, 0.1])
    positive = np.array([1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, positive) == pytest.approx(0.75)


def test_a_single_class_leaves_discrimination_undefined_not_chance() -> None:
    """0.5 would be a measurement. There is nothing to rank, so the answer is NaN."""
    scores = np.array([0.3, 0.6, 0.9])
    assert np.isnan(roc_auc(scores, np.ones(3)))
    assert np.isnan(roc_auc(scores, np.zeros(3)))
    assert np.isnan(average_precision(scores, np.zeros(3)))


def test_average_precision_matches_the_hand_computed_step_sum() -> None:
    """Ranked 0.9(+) 0.8(-) 0.7(+) 0.6(-): precision at recall gains 1/2 and 1/2.

    At rank 1: precision 1/1, recall 1/2. At rank 3: precision 2/3, recall 1.
    AP = 1.0*(1/2) + (2/3)*(1/2) = 0.5 + 0.3333 = 0.8333.
    """
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    positive = np.array([1.0, 0.0, 1.0, 0.0])
    assert average_precision(scores, positive) == pytest.approx(0.5 + 1.0 / 3.0)


def test_average_precision_reads_a_tie_group_at_its_end() -> None:
    """Two positives and two negatives all at 0.5: one operating point, precision 1/2.

    Reading precision partway through the group would report 1.0 for whichever label
    happened to be ordered first, which makes the metric depend on row order.
    """
    scores = np.full(4, 0.5)
    positive = np.array([1.0, 1.0, 0.0, 0.0])
    assert average_precision(scores, positive) == pytest.approx(0.5)
    assert average_precision(scores, positive[::-1]) == pytest.approx(0.5)


def test_score_separation_is_the_pooled_standardized_mean_difference() -> None:
    """Positives {2,4} mean 3 var 2; negatives {0,2} mean 1 var 2. Pooled sd = sqrt(2).

    d = (3 - 1) / sqrt(2) = 1.41421.
    """
    scores = np.array([2.0, 4.0, 0.0, 2.0])
    positive = np.array([1.0, 1.0, 0.0, 0.0])
    assert score_separation(scores, positive) == pytest.approx(2.0 / np.sqrt(2.0))


def test_resolution_is_the_uncertainty_the_scores_removed() -> None:
    """base 0.6 → uncertainty 0.24. Refinement 0.10 leaves resolution 0.14."""
    assert resolution(0.6, 0.10) == pytest.approx(0.14)
    assert resolution(0.6, 0.24) == pytest.approx(0.0), "no sharpness means no resolution"


def test_the_report_bundles_the_endpoints_with_the_n_behind_them() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    positive = np.array([1.0, 1.0, 1.0, 0.0])
    payload = discrimination_report(scores, positive).as_dict()
    assert payload["n"] == 4
    assert payload["n_positive"] == 3
    assert payload["base_rate"] == pytest.approx(0.75)
    assert payload["roc_auc"] == pytest.approx(1.0)


def test_mismatched_lengths_are_rejected_rather_than_broadcast() -> None:
    with pytest.raises(ValueError, match="disagree in length"):
        roc_auc(np.zeros(4), np.zeros(3))


def test_roc_auc_agrees_with_an_independent_implementation() -> None:
    """Cross-checked against scikit-learn, which is in core and not our own code.

    Trusting our own function twice proves consistency, not correctness.
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(20, 200))
        # Coarse quantization on purpose: ties are where the two implementations can
        # legitimately disagree, so they have to be in the comparison.
        scores = np.round(rng.random(n), 2)
        positive = (rng.random(n) < 0.35).astype(np.float64)
        if positive.sum() in (0, n):
            continue
        assert roc_auc(scores, positive) == pytest.approx(
            sklearn_metrics.roc_auc_score(positive, scores)
        )
        assert average_precision(scores, positive) == pytest.approx(
            sklearn_metrics.average_precision_score(positive, scores)
        )
