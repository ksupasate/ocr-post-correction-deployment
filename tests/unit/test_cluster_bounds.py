"""Document-level risk bounds: validity, tightness, degeneracy, and duplication invariance.

The claim these bounds exist to support is that a candidate-weighted harm rate can be bounded
without pretending candidates are independent. Three things have to hold for that claim to mean
anything, and each has a test here that would fail if it did not:

1. the bounds cover their nominal level on data that violates candidate independence;
2. they are *invariant* when a document's candidates are duplicated, because duplication adds no
   independent information -- while the candidate-level Clopper-Pearson bound is not;
3. the degenerate cases behave as the frozen design note says they do, rather than by accident.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import t as student_t

from ocr_risk.risk import clopper_pearson_upper
from ocr_risk.risk.cluster_bounds import (
    BETTING_TRUNCATION,
    CLUSTER_RATIO_METHODS,
    FINITE_SAMPLE_METHODS,
    betting_mean_upper,
    cluster_ratio_bound,
    cluster_ratio_upper,
    cluster_robust_ratio_upper,
    design_effect,
    empirical_bernstein_mean_upper,
    hoeffding_mean_upper,
)

# --------------------------------------------------------------------------- argument validation


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.5])
def test_every_mean_bound_refuses_a_level_outside_the_unit_interval(delta: float) -> None:
    for bound in (hoeffding_mean_upper, empirical_bernstein_mean_upper):
        with pytest.raises(ValueError, match="delta must lie"):
            bound(np.array([0.0, 1.0]), delta, span=1.0)
    with pytest.raises(ValueError, match="delta must lie"):
        betting_mean_upper(np.array([0.0, 1.0]), delta, lower=0.0, span=1.0)


@pytest.mark.parametrize("span", [-1.0, float("nan"), float("inf")])
def test_every_mean_bound_refuses_a_support_it_cannot_use(span: float) -> None:
    for bound in (hoeffding_mean_upper, empirical_bernstein_mean_upper):
        with pytest.raises(ValueError, match="span must be finite"):
            bound(np.array([0.0, 1.0]), 0.05, span=span)
    with pytest.raises(ValueError, match="span must be finite"):
        betting_mean_upper(np.array([0.0, 1.0]), 0.05, lower=0.0, span=span)


# ------------------------------------------------------------------------------- hand-computed


def test_hoeffding_matches_the_closed_form() -> None:
    assert hoeffding_mean_upper([0.0, 1.0], 0.05, span=1.0) == pytest.approx(1.3654091913011426)


def test_empirical_bernstein_matches_the_closed_form() -> None:
    observed = empirical_bernstein_mean_upper([0.0, 1.0, 0.0, 1.0], 0.05, span=1.0)
    assert observed == pytest.approx(4.153228740010524)


def test_empirical_bernstein_keeps_only_its_additive_term_on_constant_data() -> None:
    """Zero sample variance removes the variance term and leaves the O(span / n) price."""
    assert empirical_bernstein_mean_upper([2.0, 2.0, 2.0], 0.05, span=5.0) == pytest.approx(
        23.518463482331295
    )


def test_cluster_robust_matches_the_ratio_estimator_closed_form() -> None:
    observed = cluster_robust_ratio_upper([0.0, 1.0, 0.0, 1.0], [10.0] * 4, 0.05)
    assert observed == pytest.approx(0.11793575062919276)


def test_the_cluster_robust_limit_uses_student_t_and_not_a_normal_quantile() -> None:
    """Four documents is not asymptopia; the small-sample courtesy has to be visible."""
    harmful, accepted = [0.0, 1.0, 0.0, 1.0], [10.0] * 4
    observed = cluster_robust_ratio_upper(harmful, accepted, 0.05)
    residual = np.asarray(harmful) - 0.05 * np.asarray(accepted)
    spread = math.sqrt(float((residual**2).sum()) / (3 * 4 * 100))
    assert observed == pytest.approx(0.05 + float(student_t.ppf(0.95, 3)) * spread)
    assert observed > 0.05 + 1.6448536269514722 * spread


# ---------------------------------------------------------------------------------- degeneracy


def test_no_observations_rules_nothing_out() -> None:
    empty = np.zeros(0, dtype=float)
    assert hoeffding_mean_upper(empty, 0.05, span=1.0) == float("inf")
    assert empirical_bernstein_mean_upper(empty, 0.05, span=1.0) == float("inf")
    assert betting_mean_upper(empty, 0.05, lower=0.0, span=1.0) == float("inf")


def test_a_degenerate_support_returns_the_point_it_collapses_to() -> None:
    assert hoeffding_mean_upper([3.0, 3.0], 0.05, span=0.0) == pytest.approx(3.0)
    assert betting_mean_upper([3.0, 3.0], 0.05, lower=3.0, span=0.0) == pytest.approx(3.0)


def test_empirical_bernstein_falls_back_to_hoeffding_below_two_observations() -> None:
    for values in ([], [0.4]):
        assert empirical_bernstein_mean_upper(values, 0.05, span=1.0) == hoeffding_mean_upper(
            values, 0.05, span=1.0
        )


def test_the_betting_bound_never_leaves_the_declared_support() -> None:
    """Every observation at the ceiling gives no evidence against the ceiling."""
    assert betting_mean_upper([1.0] * 20, 0.05, lower=0.0, span=1.0) == pytest.approx(1.0)
    assert betting_mean_upper([0.0] * 20, 0.05, lower=0.0, span=1.0) < 1.0
    assert betting_mean_upper([0.0] * 20, 0.05, lower=0.0, span=1.0) >= 0.0


def test_the_betting_truncation_is_the_documented_one() -> None:
    assert 0.0 < BETTING_TRUNCATION < 1.0


def test_cluster_robust_refuses_where_it_cannot_be_formed() -> None:
    assert cluster_robust_ratio_upper([1.0], [10.0], 0.05) == 1.0
    assert cluster_robust_ratio_upper([0.0, 0.0], [0.0, 0.0], 0.05) == 1.0


def test_cluster_robust_is_clipped_at_one() -> None:
    assert cluster_robust_ratio_upper([1.0, 1.0, 0.0, 1.0], [1.0] * 4, 0.05) == 1.0


def test_mismatched_cluster_vectors_are_a_hard_error() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        cluster_robust_ratio_upper([1.0, 2.0], [1.0], 0.05)
    with pytest.raises(ValueError, match="shape mismatch"):
        cluster_ratio_bound([1.0, 2.0], [1.0], 0.1, delta=0.05, span=4.0)


def test_an_unknown_cluster_method_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown cluster ratio method"):
        cluster_ratio_bound([1.0], [4.0], 0.1, delta=0.05, span=4.0, method="wishful")


# ------------------------------------------------------------------------------ design effect


def test_the_design_effect_is_undefined_where_it_has_no_meaning() -> None:
    assert math.isnan(design_effect([1.0], [4.0]))
    assert math.isnan(design_effect([0.0, 0.0], [0.0, 0.0]))
    assert math.isnan(design_effect([0.0, 0.0], [4.0, 4.0]))
    assert math.isnan(design_effect([4.0, 4.0], [4.0, 4.0]))


def test_perfectly_homogeneous_documents_carry_no_clustering_penalty() -> None:
    """Every page with the identical harm rate has zero between-document variance."""
    assert design_effect([1.0] * 6, [10.0] * 6) == pytest.approx(0.0)


def test_all_or_nothing_documents_are_the_worst_clustering_case() -> None:
    """Half the pages entirely harmful and half entirely clean: one page, one observation."""
    harmful = [10.0, 10.0, 0.0, 0.0]
    accepted = [10.0] * 4
    assert design_effect(harmful, accepted) > 9.0


# ----------------------------------------------------------------------- the ratio certificate


def test_no_documents_cannot_certify_anything() -> None:
    result = cluster_ratio_bound([], [], 0.1, delta=0.05, span=4.0)
    assert not result.certifies
    assert result.z_upper == float("inf")
    assert result.ratio_upper == 1.0
    assert not result.degenerate_denominator
    assert math.isnan(result.observed_ratio)


def test_a_sample_that_accepted_nothing_has_seen_no_evidence_and_may_not_certify() -> None:
    """The design note's rule 1, and the one place a vacuous "0 <= 0" would be dangerous.

    Every sampled document accepting nothing puts ``Z_d`` at zero, but the *population* can still
    accept up to ``span`` at this prefix. Certifying here would deploy a threshold that no
    observation had constrained.
    """
    result = cluster_ratio_bound([0.0] * 5, [0.0] * 5, 0.1, delta=0.05, span=4.0)
    assert not result.certifies
    assert result.degenerate_denominator
    assert result.z_upper > 0.0
    assert math.isnan(result.observed_ratio)


def test_a_prefix_nothing_anywhere_can_accept_certifies_because_it_deploys_nothing() -> None:
    """The design note's rule 4: zero population support means ``Z`` is identically zero."""
    result = cluster_ratio_bound([0.0] * 5, [0.0] * 5, 0.1, delta=0.05, span=0.0)
    assert result.certifies
    assert result.degenerate_denominator
    assert result.z_upper == pytest.approx(0.0)


def test_the_asymptotic_comparator_also_refuses_an_empty_denominator() -> None:
    result = cluster_ratio_bound(
        [0.0] * 5, [0.0] * 5, 0.1, delta=0.05, span=4.0, method="cluster_robust_asymptotic"
    )
    assert not result.certifies
    assert result.degenerate_denominator
    assert result.ratio_upper == 1.0


@pytest.mark.parametrize("method", CLUSTER_RATIO_METHODS)
def test_every_declared_method_produces_a_complete_record(method: str) -> None:
    result = cluster_ratio_bound(
        [0.0, 1.0, 0.0], [8.0, 9.0, 7.0], 0.1, delta=0.05, span=12.0, method=method
    )
    payload = result.as_dict()
    assert payload["method"] == method
    assert payload["documents"] == 3
    assert payload["accepted"] == 24
    assert payload["harmful"] == 1
    assert payload["observed_ratio"] == pytest.approx(1.0 / 24.0)
    expected = "asymptotic" if method == "cluster_robust_asymptotic" else "finite-sample"
    assert payload["guarantee"] == expected


def test_only_the_asymptotic_method_claims_an_asymptotic_guarantee() -> None:
    for method in (*FINITE_SAMPLE_METHODS, "union_finite"):
        result = cluster_ratio_bound([0.0], [4.0], 0.1, delta=0.05, span=4.0, method=method)
        assert result.guarantee == "finite-sample"
        assert math.isnan(result.ratio_upper)
    asymptotic = cluster_ratio_bound(
        [0.0, 1.0, 0.0],
        [8.0, 9.0, 7.0],
        0.1,
        delta=0.05,
        span=12.0,
        method="cluster_robust_asymptotic",
    )
    assert asymptotic.guarantee == "asymptotic"
    assert math.isnan(asymptotic.z_upper)


def test_the_union_certificate_is_the_best_of_its_three_members_at_a_third_of_the_level() -> None:
    harmful = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 0.0])
    accepted = np.array([20.0, 30.0, 25.0, 40.0, 18.0, 22.0])
    union = cluster_ratio_bound(harmful, accepted, 0.1, delta=0.06, span=40.0)
    members = [
        cluster_ratio_bound(harmful, accepted, 0.1, delta=0.02, span=40.0, method=name).z_upper
        for name in FINITE_SAMPLE_METHODS
    ]
    assert union.z_upper == pytest.approx(min(members))


def test_the_certificate_is_the_sign_of_the_bound_on_the_ratio_null() -> None:
    harmful = np.zeros(200)
    accepted = np.full(200, 10.0)
    result = cluster_ratio_bound(harmful, accepted, 0.1, delta=0.05, span=10.0)
    assert result.z_upper <= 0.0
    assert result.certifies
    unsafe = cluster_ratio_bound(np.full(200, 5.0), accepted, 0.1, delta=0.05, span=10.0)
    assert unsafe.z_upper > 0.0
    assert not unsafe.certifies


# ------------------------------------------------------------- duplication: the whole argument


DUPLICATION_HARM = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0])
DUPLICATION_ACCEPT = np.array([20.0, 30.0, 25.0, 40.0, 18.0, 22.0, 15.0, 31.0])


@pytest.mark.parametrize("method", CLUSTER_RATIO_METHODS)
@pytest.mark.parametrize("copies", [2, 5, 10])
def test_document_aware_certification_is_invariant_under_duplicating_candidates(
    method: str, copies: int
) -> None:
    """Duplicating every candidate inside every document adds no independent document.

    A document-level certificate must therefore return the same decision. All four bounds are
    positively homogeneous in the per-document counts once the support scales with them, so the
    invariance is exact rather than approximate, and this test is the falsification the stage
    brief asks for.
    """
    base = cluster_ratio_bound(
        DUPLICATION_HARM, DUPLICATION_ACCEPT, 0.1, delta=0.05, span=40.0, method=method
    )
    duplicated = cluster_ratio_bound(
        DUPLICATION_HARM * copies,
        DUPLICATION_ACCEPT * copies,
        0.1,
        delta=0.05,
        span=40.0 * copies,
        method=method,
    )
    assert duplicated.certifies == base.certifies
    assert duplicated.observed_ratio == pytest.approx(base.observed_ratio)
    if method == "cluster_robust_asymptotic":
        assert duplicated.ratio_upper == pytest.approx(base.ratio_upper)
    else:
        assert duplicated.z_upper == pytest.approx(base.z_upper * copies)


@pytest.mark.parametrize("copies", [2, 5, 10])
def test_the_candidate_level_bound_becomes_falsely_confident_under_the_same_duplication(
    copies: int,
) -> None:
    """The contrast that makes the invariance test meaningful."""
    harmful = int(DUPLICATION_HARM.sum())
    accepted = int(DUPLICATION_ACCEPT.sum())
    base = clopper_pearson_upper(harmful, accepted, 0.05)
    duplicated = clopper_pearson_upper(harmful * copies, accepted * copies, 0.05)
    assert duplicated < base


# --------------------------------------------------------------------------- validity by Monte Carlo


def _clustered_draw(
    generator: np.random.Generator, documents: int, mean_size: float, rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Documents of unequal size whose harm rates vary, which is the regime the bound is for."""
    sizes = generator.poisson(mean_size, documents) + 1
    per_document = generator.beta(rate * 2.0, (1.0 - rate) * 2.0, documents)
    return generator.binomial(sizes, per_document).astype(float), sizes.astype(float)


@pytest.mark.parametrize("method", (*FINITE_SAMPLE_METHODS, "union_finite"))
def test_the_finite_sample_certificates_hold_their_level_under_strong_clustering(
    method: str,
) -> None:
    """A boundary case: the true candidate-weighted risk is exactly the tolerance.

    A method that certifies here more often than its nominal level is anti-conservative, and the
    frozen design note says it is then demoted to a diagnostic rather than reported as a bound.
    """
    generator = np.random.default_rng(20260907)
    false_certifications = 0
    trials = 600
    for _ in range(trials):
        harmful, accepted = _clustered_draw(generator, 40, 12.0, 0.10)
        if cluster_ratio_bound(
            harmful, accepted, 0.10, delta=0.05, span=60.0, method=method
        ).certifies:
            false_certifications += 1
    assert false_certifications / trials <= 0.05


def test_the_candidate_level_bound_is_the_one_that_fails_that_same_check() -> None:
    """Read on the same clustered draws, the exact Bernoulli interval is not exact."""
    generator = np.random.default_rng(20260907)
    false_certifications = 0
    trials = 600
    for _ in range(trials):
        harmful, accepted = _clustered_draw(generator, 40, 12.0, 0.10)
        upper = clopper_pearson_upper(int(harmful.sum()), int(accepted.sum()), 0.05)
        if upper <= 0.10:
            false_certifications += 1
    assert false_certifications / trials > 0.05


# ------------------------------------------------------------------------------ the inversion


def test_inverting_a_certificate_returns_a_tolerance_it_would_actually_certify() -> None:
    harmful = np.zeros(300)
    accepted = np.full(300, 10.0)
    limit = cluster_ratio_upper(harmful, accepted, delta=0.05, span=10.0, resolution=200)
    assert 0.0 < limit < 1.0
    assert cluster_ratio_bound(harmful, accepted, limit, delta=0.05, span=10.0).certifies


def test_the_inversion_bottoms_out_at_zero_when_every_tolerance_certifies() -> None:
    """A prefix nothing anywhere can accept certifies at every tolerance, including zero."""
    assert cluster_ratio_upper(
        np.zeros(4), np.zeros(4), delta=0.05, span=0.0, resolution=20
    ) == pytest.approx(0.0)


def test_the_inversion_refuses_a_prefix_whose_sample_saw_nothing() -> None:
    assert cluster_ratio_upper(
        np.zeros(4), np.zeros(4), delta=0.05, span=4.0, resolution=20
    ) == pytest.approx(1.0)


def test_the_inversion_reports_one_when_nothing_can_be_certified() -> None:
    assert cluster_ratio_upper(
        np.array([9.0, 9.0]), np.array([10.0, 10.0]), delta=0.05, span=10.0, resolution=20
    ) == pytest.approx(1.0)


def test_a_finer_inversion_grid_never_reports_a_looser_limit() -> None:
    harmful = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0] * 20)
    accepted = np.array([20.0, 30.0, 25.0, 18.0, 40.0, 22.0] * 20)
    coarse = cluster_ratio_upper(harmful, accepted, delta=0.05, span=40.0, resolution=50)
    fine = cluster_ratio_upper(harmful, accepted, delta=0.05, span=40.0, resolution=500)
    assert fine <= coarse + 1e-12


# ---------------------------------------------------------------------------- monotonicity


@pytest.mark.parametrize("method", CLUSTER_RATIO_METHODS)
def test_a_larger_tolerance_is_never_harder_to_certify(method: str) -> None:
    """Certification must be monotone in epsilon, or the deployed prefix is not interpretable."""
    harmful = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0] * 20)
    accepted = np.array([20.0, 30.0, 25.0, 40.0, 18.0, 22.0] * 20)
    decisions = [
        cluster_ratio_bound(
            harmful, accepted, epsilon, delta=0.05, span=40.0, method=method
        ).certifies
        for epsilon in (0.02, 0.05, 0.10, 0.20, 0.50, 0.90)
    ]
    assert decisions == sorted(decisions), decisions


@pytest.mark.parametrize("bound", [hoeffding_mean_upper, empirical_bernstein_mean_upper])
def test_a_tighter_level_gives_a_wider_bound(bound: object) -> None:
    values = np.array([0.1, 0.4, 0.2, 0.8, 0.3, 0.5])
    widths = [bound(values, delta, span=1.0) for delta in (0.2, 0.1, 0.05, 0.01)]  # type: ignore[operator]
    assert widths == sorted(widths)


def test_the_betting_bound_also_widens_as_the_level_tightens() -> None:
    values = np.array([0.1, 0.4, 0.2, 0.8, 0.3, 0.5])
    widths = [
        betting_mean_upper(values, delta, lower=0.0, span=1.0) for delta in (0.2, 0.1, 0.05, 0.01)
    ]
    assert widths == sorted(widths)


@pytest.mark.parametrize("method", (*FINITE_SAMPLE_METHODS, "union_finite"))
def test_more_documents_at_the_same_rate_never_loosen_the_bound(method: str) -> None:
    """The bound must reward evidence: repeating the same pattern on more pages tightens it."""
    pattern_harm = np.array([0.0, 1.0, 0.0, 0.0])
    pattern_accept = np.array([20.0, 30.0, 25.0, 18.0])
    limits = [
        cluster_ratio_bound(
            np.tile(pattern_harm, repeats),
            np.tile(pattern_accept, repeats),
            0.1,
            delta=0.05,
            span=30.0,
            method=method,
        ).z_upper
        / repeats
        for repeats in (1, 2, 5, 20)
    ]
    assert limits == sorted(limits, reverse=True), limits


# ------------------------------------------------------------------------ cluster shapes


def test_one_candidate_per_document_is_the_independent_case() -> None:
    """With one candidate per page the document unit has no dependence left to price.

    It cannot become tighter than the exact Bernoulli interval -- it is distribution-free where
    Clopper-Pearson is exact -- but it must certify the same clearly-safe configuration.
    """
    harmful = np.zeros(400)
    accepted = np.ones(400)
    result = cluster_ratio_bound(harmful, accepted, 0.1, delta=0.05, span=1.0)
    assert result.certifies
    assert result.documents == 400
    assert result.accepted == 400
    assert math.isnan(design_effect(harmful, accepted))


def test_one_candidate_per_document_still_refuses_when_the_rate_is_too_high() -> None:
    generator = np.random.default_rng(31337)
    harmful = generator.binomial(1, 0.3, 400).astype(float)
    assert not cluster_ratio_bound(harmful, np.ones(400), 0.1, delta=0.05, span=1.0).certifies


def test_unequal_cluster_sizes_are_priced_by_the_largest_document() -> None:
    """The support width is the mechanism, so one very large page widens the whole bound.

    Both configurations see the same number of documents, the same total evidence per typical
    page and zero harm; they differ only in that one page is twenty times the size of the others.
    At 400 documents that difference alone is the difference between certifying and refusing.
    """
    documents = 400
    harm = np.zeros(documents)
    even_accept = np.full(documents, 10.0)
    lumpy_accept = even_accept.copy()
    lumpy_accept[0] = 200.0
    even = cluster_ratio_bound(harm, even_accept, 0.1, delta=0.05, span=10.0)
    lumpy = cluster_ratio_bound(harm, lumpy_accept, 0.1, delta=0.05, span=200.0)
    assert even.certifies
    assert not lumpy.certifies
    assert lumpy.z_upper > even.z_upper
