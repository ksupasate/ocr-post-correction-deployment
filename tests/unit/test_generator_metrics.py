"""Generator-quality metrics against hand-computed values.

These are the Q1 endpoints the H2-readiness gate is read from, so each expected number is
worked out in the docstring from the site list above it rather than taken from the
function. A denominator error here would move a go/no-go decision.
"""

from __future__ import annotations

import math

import pytest

from ocr_risk.metrics.generator import (
    SiteProposals,
    beneficial_candidate_rate,
    clean_span_proposal_rate,
    error_repair_opportunity,
    generator_quality,
    harmful_candidate_rate,
    oracle_repair_recall,
    oracle_safe_coverage,
)
from ocr_risk.schemas.enums import HarmPolicy, OutcomeIfAccepted

OUT = OutcomeIfAccepted
STRICT = HarmPolicy.STRICT_WORSENING

# Six sites: two already correct, four with errors.
#   clean_1  d=0, one proposal, overcorrection
#   clean_2  d=0, no proposal
#   err_1    d=4, true correction (+4) and a miscorrection
#   err_2    d=3, partial improvement (+1)
#   err_3    d=2, lateral change only
#   err_4    d=5, no proposal at all
POOL = [
    SiteProposals("clean_1", 0, (OUT.OVERCORRECTION,), (-2,)),
    SiteProposals("clean_2", 0, (), ()),
    SiteProposals("err_1", 4, (OUT.TRUE_CORRECTION, OUT.MISCORRECTION), (4, -1)),
    SiteProposals("err_2", 3, (OUT.PARTIAL_IMPROVEMENT,), (1,)),
    SiteProposals("err_3", 2, (OUT.LATERAL_CHANGE,), (0,)),
    SiteProposals("err_4", 5, (), ()),
]
OUTCOMES = [o for site in POOL for o in site.outcomes]


def test_beneficial_and_harmful_rates_are_over_non_identity_candidates() -> None:
    """5 proposals. Beneficial = true_correction + partial = 2/5. Harmful = over + mis = 2/5.

    Lateral change is neither, so the two rates do not sum to 1 — which is the point of
    reporting the no-change rate beside them.
    """
    assert beneficial_candidate_rate(OUTCOMES, STRICT) == pytest.approx(0.4)
    assert harmful_candidate_rate(OUTCOMES, STRICT) == pytest.approx(0.4)


def test_an_identity_proposal_is_not_counted_as_a_generated_edit() -> None:
    """Otherwise a generator that proposes "leave it alone" everywhere scores well.

    Both rates have to skip it, not only the beneficial one: dropping identities from the
    numerator but keeping them in the denominator would make a quiet generator look safe.
    """
    padded = [*OUTCOMES, OUT.IDENTITY, OUT.IDENTITY, OUT.IDENTITY]
    assert beneficial_candidate_rate(padded, STRICT) == pytest.approx(0.4)
    assert harmful_candidate_rate(padded, STRICT) == pytest.approx(0.4)


def test_the_exact_only_policy_moves_both_rates_the_expected_way() -> None:
    """exact_only counts everything but an exact repair as harmful.

    Harmful becomes over + mis + lateral + partial = 4/5 = 0.8, and beneficial is the lone
    true correction, 1/5 = 0.2. Under strict_worsening the same five candidates give
    0.4/0.4 — the pool did not change, the definition did, which is why the verdict has to
    name the policy it was read under.
    """
    assert beneficial_candidate_rate(OUTCOMES, HarmPolicy.EXACT_ONLY) == pytest.approx(0.2)
    assert harmful_candidate_rate(OUTCOMES, HarmPolicy.EXACT_ONLY) == pytest.approx(0.8)


def test_repair_opportunity_is_over_error_sites_and_needs_only_one_good_candidate() -> None:
    """4 error sites; err_1 and err_2 have a beneficial candidate. 2/4 = 0.5.

    err_1 also has a harmful candidate, which must not disqualify it: the question is
    whether a perfect verifier *could* repair the site, not whether every proposal is good.
    """
    assert error_repair_opportunity(POOL, STRICT) == pytest.approx(0.5)


def test_clean_span_proposal_rate_is_over_clean_sites_only() -> None:
    """2 clean sites, 1 of them proposed on. 1/2 = 0.5."""
    assert clean_span_proposal_rate(POOL) == pytest.approx(0.5)


def test_oracle_safe_coverage_counts_every_site_in_its_denominator() -> None:
    """2 of 6 sites hold a beneficial candidate. 2/6 = 0.3333.

    Clean sites stay in the denominator because the deployed coverage denominator has them
    too; dropping them would report a coverage a deployed system could never reach.
    """
    assert oracle_safe_coverage(POOL, STRICT) == pytest.approx(1 / 3)


def test_oracle_repair_recall_is_weighted_by_characters_not_by_sites() -> None:
    """Total error characters 0+0+4+3+2+5 = 14. Best beneficial gain: err_1 4, err_2 1.

    5/14 = 0.3571. A site with one wrong character and a site with twenty are not the same
    repair, and a site-weighted version would call this 0.5.
    """
    assert oracle_repair_recall(POOL, STRICT) == pytest.approx(5 / 14)


def test_a_pool_with_no_error_characters_leaves_repair_recall_undefined() -> None:
    clean_only = [SiteProposals("a", 0, (OUT.OVERCORRECTION,), (-1,))]
    assert math.isnan(oracle_repair_recall(clean_only, STRICT))
    assert math.isnan(error_repair_opportunity(clean_only, STRICT)), (
        "no error sites means the rate has no denominator; 0.0 would read as total failure"
    )


def test_a_generator_that_proposes_nothing_scores_zero_opportunity_not_nan() -> None:
    """ "Proposed nothing" and "had no error sites" are different states.

    A silent generator must score 0 on repair opportunity — it genuinely repaired nothing —
    while its beneficial rate is undefined, because there are no candidates to take a share
    of. Collapsing either into the other lets a silent generator look accurate.
    """
    silent = [SiteProposals("err", 3, (), ()), SiteProposals("clean", 0, (), ())]
    assert error_repair_opportunity(silent, STRICT) == 0.0
    assert clean_span_proposal_rate(silent) == 0.0
    assert oracle_safe_coverage(silent, STRICT) == 0.0
    assert math.isnan(beneficial_candidate_rate([], STRICT))
    assert math.isnan(harmful_candidate_rate([], STRICT))
    # No sites at all is a third state again: there was nothing to cover.
    assert math.isnan(oracle_safe_coverage([], STRICT))


def test_the_bundled_report_agrees_with_each_metric_computed_alone() -> None:
    quality = generator_quality(POOL, STRICT)
    assert quality.n_sites == 6
    assert quality.n_clean_sites == 2
    assert quality.n_error_sites == 4
    assert quality.n_nonidentity_candidates == 5
    assert quality.beneficial_rate == pytest.approx(beneficial_candidate_rate(OUTCOMES, STRICT))
    assert quality.harmful_rate == pytest.approx(harmful_candidate_rate(OUTCOMES, STRICT))
    assert quality.no_change_rate == pytest.approx(0.2)
    assert quality.exact_correction_rate == pytest.approx(0.2)
    assert quality.partial_improvement_rate == pytest.approx(0.2)
    assert quality.error_repair_opportunity == pytest.approx(0.5)
    assert quality.clean_span_proposal_rate == pytest.approx(0.5)
    assert quality.oracle_safe_coverage == pytest.approx(1 / 3)
    assert quality.oracle_repair_recall == pytest.approx(5 / 14)
    assert set(quality.as_dict()) == {
        "n_sites",
        "n_clean_sites",
        "n_error_sites",
        "n_nonidentity_candidates",
        "beneficial_rate",
        "harmful_rate",
        "no_change_rate",
        "error_repair_opportunity",
        "clean_span_proposal_rate",
        "oracle_safe_coverage",
        "oracle_repair_recall",
        "exact_correction_rate",
        "partial_improvement_rate",
    }


def test_the_oracle_metrics_are_named_so_a_reader_cannot_miss_what_they_are() -> None:
    """A deployable-looking name on an unreachable upper bound is how a ceiling gets
    quoted as a result."""
    for function in (oracle_safe_coverage, oracle_repair_recall):
        assert function.__name__.startswith("oracle_")
        assert "ORACLE" in (function.__doc__ or "")
        assert "NOT DEPLOYABLE" in (function.__doc__ or "")


# ------------------------------------------- deletion-shaped sites and the oracle ceiling

# The OCR invented "243,000" where the ground truth has nothing. d_before = 7. A candidate
# of "23,000" has d_after = 6, so delta = +1 and the taxonomy calls it PARTIAL_IMPROVEMENT
# -- arithmetically right, operationally empty. Only the empty string actually repairs it.
DELETION_POOL = [
    SiteProposals(
        "hallucination",
        7,
        (OUT.PARTIAL_IMPROVEMENT,),
        (1,),
        needs_deletion=True,
    ),
    SiteProposals("real_error", 3, (OUT.TRUE_CORRECTION,), (3,)),
    SiteProposals("clean", 0, (), ()),
]


def test_a_shorter_hallucination_is_not_counted_as_repairable() -> None:
    """It inflated the pilot generator's oracle safe coverage by 55%.

    A verifier cannot be asked to prefer a shorter wrong string over a longer one at a site
    whose correct action is deletion, so the ceiling must not include it. Only the genuine
    repair remains: 1 of 3 sites, and 1 of 2 error sites.
    """
    assert oracle_safe_coverage(DELETION_POOL, STRICT) == pytest.approx(1 / 3)
    assert error_repair_opportunity(DELETION_POOL, STRICT) == pytest.approx(0.5)


def test_the_exact_deletion_is_still_counted(pytestconfig: object = None) -> None:
    """The rule excludes partial shortenings, not deletion itself."""
    deleted = [
        SiteProposals("hallucination", 7, (OUT.TRUE_CORRECTION,), (7,), needs_deletion=True),
    ]
    assert oracle_safe_coverage(deleted, STRICT) == pytest.approx(1.0)
    assert oracle_repair_recall(deleted, STRICT) == pytest.approx(1.0)


def test_the_rule_applies_only_where_the_ground_truth_is_empty() -> None:
    """A partial improvement at an ordinary error site is a real, countable repair."""
    ordinary = [SiteProposals("typo", 4, (OUT.PARTIAL_IMPROVEMENT,), (2,))]
    assert oracle_safe_coverage(ordinary, STRICT) == pytest.approx(1.0)
    assert oracle_repair_recall(ordinary, STRICT) == pytest.approx(0.5)


def test_the_candidate_share_metrics_keep_the_taxonomy_definition() -> None:
    """BCR and HCR must still sum with the no-change rate to 1, or the shares stop adding up.

    The deletion rule is about what a perfect verifier could *reach*, not about relabelling
    outcomes, so it applies to the oracle-facing metrics and to nothing else.
    """
    quality = generator_quality(DELETION_POOL, STRICT)
    assert quality.beneficial_rate == pytest.approx(1.0), "both proposals are beneficial"
    assert quality.oracle_safe_coverage == pytest.approx(1 / 3)
    assert quality.beneficial_rate + quality.harmful_rate + quality.no_change_rate == pytest.approx(
        1.0
    )
