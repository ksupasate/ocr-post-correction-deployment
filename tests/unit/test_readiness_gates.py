"""The three gates, including the verdicts nobody wants.

A gate is only worth having if it can stop the milestone, so every branch is exercised:
READY, NOT READY, PARTIALLY, and the INCONCLUSIVE that fires when a criterion could not be
looked at. The H1 pilot's gate collapsed "we looked and it failed" into "we could not
look", and those lead to different next steps.
"""

from __future__ import annotations

import json

import pytest

from ocr_risk.experiments.readiness import (
    GENERATOR_INCONCLUSIVE,
    GENERATOR_NOT_READY,
    GENERATOR_PARTIALLY_READY,
    GENERATOR_READY,
    H2_INCONCLUSIVE,
    H2_NOT_READY,
    H2_READY,
    RH1_INCONCLUSIVE,
    RH1_NOT_SUPPORTED,
    RH1_PARTIALLY_SUPPORTED,
    RH1_SUPPORTED,
    Criterion,
    EngineCell,
    evaluate_generator_gate,
    evaluate_h2_readiness,
    evaluate_rh1_gate,
    h2_readiness_criteria,
)

ENGINES = ("doctr", "easyocr", "paddleocr", "tesseract")


def _criterion(key: str, met: bool, measurable: bool = True) -> Criterion:
    return Criterion(
        key=key,
        question="q",
        threshold="t",
        observed="o",
        met=met,
        measurable=measurable,
    )


# ------------------------------------------------------------------------ generator gate


@pytest.mark.parametrize(
    ("met", "expected"),
    [
        ((True, True, True), GENERATOR_READY),
        ((True, False, True), GENERATOR_PARTIALLY_READY),
        ((False, False, False), GENERATOR_NOT_READY),
    ],
)
def test_the_generator_gate_returns_each_of_its_verdicts(
    met: tuple[bool, ...], expected: str
) -> None:
    criteria = [_criterion(f"c{i}", value) for i, value in enumerate(met)]
    assert evaluate_generator_gate(criteria).verdict == expected


def test_a_criterion_nobody_could_measure_makes_the_gate_inconclusive() -> None:
    """Not failed. The unseen criterion could have changed the answer either way."""
    criteria = [_criterion("a", True), _criterion("b", True), _criterion("c", False, False)]
    outcome = evaluate_generator_gate(criteria)
    assert outcome.verdict == GENERATOR_INCONCLUSIVE
    assert outcome.n_measurable == 2


# ----------------------------------------------------------------------- h2 readiness


def _cells(oracle_accepted: int, coverage: float, candidates: int = 3000) -> list[EngineCell]:
    return [
        EngineCell(
            engine_id=engine,
            oracle_accepted=oracle_accepted,
            oracle_safe_coverage=coverage,
            n_nonidentity_candidates=candidates,
            n_sites=4500,
            n_clean_sites=2500,
            clean_span_proposal_rate=0.20,
        )
        for engine in ENGINES
    ]


def test_h2_is_ready_only_when_every_frozen_criterion_holds() -> None:
    outcome = evaluate_h2_readiness(
        h2_readiness_criteria(
            per_engine=_cells(oracle_accepted=400, coverage=0.09),
            n_test_documents=61,
            all_natural=True,
            all_pairs_matched=True,
            alignment_sensitivity_reported=True,
        )
    )
    assert outcome.verdict == H2_READY
    assert outcome.n_met == len(outcome.criteria)


def test_h2_has_no_partial_pass() -> None:
    """Three of five conditions does not support three-fifths of an experiment."""
    criteria = h2_readiness_criteria(
        per_engine=_cells(oracle_accepted=100, coverage=0.02),
        n_test_documents=61,
        all_natural=True,
        all_pairs_matched=True,
        alignment_sensitivity_reported=True,
    )
    outcome = evaluate_h2_readiness(criteria)
    assert outcome.verdict == H2_NOT_READY
    assert 0 < outcome.n_met < len(criteria), "this fixture must be a genuine partial"


def test_h2_is_inconclusive_when_the_arms_were_never_certified() -> None:
    outcome = evaluate_h2_readiness(
        h2_readiness_criteria(
            per_engine=_cells(oracle_accepted=400, coverage=0.09),
            n_test_documents=61,
            all_natural=True,
            all_pairs_matched=None,
            alignment_sensitivity_reported=True,
        )
    )
    assert outcome.verdict == H2_INCONCLUSIVE


def test_the_three_of_four_rule_is_applied_per_engine_not_on_an_average() -> None:
    """Two strong engines and two weak ones must not average into a pass.

    The mean of (900, 900, 10, 10) clears 245 comfortably; the criterion is about how many
    engines individually clear it, because engines are fixed environments.
    """
    mixed = [
        EngineCell(
            engine_id=engine,
            oracle_accepted=accepted,
            oracle_safe_coverage=accepted / 4500,
            n_nonidentity_candidates=3000,
            n_sites=4500,
            n_clean_sites=2500,
            clean_span_proposal_rate=0.20,
        )
        for engine, accepted in zip(ENGINES, (900, 900, 10, 10), strict=True)
    ]
    criteria = {
        c.key: c
        for c in h2_readiness_criteria(
            per_engine=mixed,
            n_test_documents=61,
            all_natural=True,
            all_pairs_matched=True,
            alignment_sensitivity_reported=True,
        )
    }
    assert not criteria["A_repair_opportunity"].met
    assert "2/4 engines" in criteria["A_repair_opportunity"].observed


def test_an_unobservable_clean_population_fails_criterion_d() -> None:
    """A generator can win A and B by never touching clean spans, and then the one harm
    the project cares about most stops being measurable."""
    avoidant = [
        EngineCell(
            engine_id=engine,
            oracle_accepted=400,
            oracle_safe_coverage=0.09,
            n_nonidentity_candidates=3000,
            n_sites=4500,
            n_clean_sites=2500,
            clean_span_proposal_rate=0.001,  # 2.5 proposals on clean spans
        )
        for engine in ENGINES
    ]
    criteria = {
        c.key: c
        for c in h2_readiness_criteria(
            per_engine=avoidant,
            n_test_documents=61,
            all_natural=True,
            all_pairs_matched=True,
            alignment_sensitivity_reported=True,
        )
    }
    assert criteria["A_repair_opportunity"].met
    assert criteria["B_oracle_safe_coverage"].met
    assert not criteria["D_overcorrection_observable"].met


def test_a_challenge_polluted_pool_fails_criterion_e() -> None:
    criteria = {
        c.key: c
        for c in h2_readiness_criteria(
            per_engine=_cells(oracle_accepted=400, coverage=0.09),
            n_test_documents=61,
            all_natural=False,
            all_pairs_matched=True,
            alignment_sensitivity_reported=True,
        )
    }
    assert not criteria["E_natural_pool"].met


def test_too_few_document_clusters_fails_criterion_c() -> None:
    """The resampling unit is the document; a percentile interval on 8 clusters is not one."""
    criteria = {
        c.key: c
        for c in h2_readiness_criteria(
            per_engine=_cells(oracle_accepted=400, coverage=0.09),
            n_test_documents=8,
            all_natural=True,
            all_pairs_matched=True,
            alignment_sensitivity_reported=True,
        )
    }
    assert not criteria["C_sample_size"].met


# ------------------------------------------------------------------------------ rh1 gate


@pytest.mark.parametrize(
    ("degraded", "tested", "expected"),
    [
        (4, 4, RH1_SUPPORTED),
        (2, 4, RH1_PARTIALLY_SUPPORTED),
        (0, 4, RH1_NOT_SUPPORTED),
        (0, 0, RH1_INCONCLUSIVE),
    ],
)
def test_the_rh1_gate_returns_each_of_its_verdicts(
    degraded: int, tested: int, expected: str
) -> None:
    outcome = evaluate_rh1_gate(degraded, tested, primary_metric="roc_auc")
    assert outcome.verdict == expected
    assert f"{degraded} of {tested}" in outcome.criteria[0].observed


def test_the_rh1_gate_names_the_metric_the_verdict_was_read_from() -> None:
    """The H1 verdict flipped across harm policies and the report did not say which one."""
    outcome = evaluate_rh1_gate(1, 4, primary_metric="roc_auc")
    assert "roc_auc" in outcome.criteria[0].question


def test_a_gate_outcome_serializes_every_criterion_it_rests_on(tmp_path: object) -> None:
    from pathlib import Path

    outcome = evaluate_rh1_gate(
        2, 4, primary_metric="roc_auc", notes=("harm policy: strict_worsening",)
    )
    path = outcome.save(Path(str(tmp_path)) / "gate.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gate"] == "rh1"
    assert payload["verdict"] == RH1_PARTIALLY_SUPPORTED
    assert payload["criteria"][0]["observed"] == "2 of 4 engines"
    assert payload["notes"] == ["harm policy: strict_worsening"]


def test_disagreement_downgrades_supported_and_not_supported_without_touching_the_counts() -> None:
    """The pre-committed rule says PARTIALLY whatever the counts say. The first
    implementation re-derived the verdict from adjusted counts, which is a no-op from
    SUPPORTED -- exactly the case where the strongest positive verdict would otherwise be
    recorded on the strength of policies that disagree with it."""
    for degraded, expected_observed in ((4, "4 of 4 engines"), (0, "0 of 4 engines")):
        outcome = evaluate_rh1_gate(
            degraded, 4, primary_metric="roc_auc", harm_policy_disagreement=True
        )
        assert outcome.verdict == RH1_PARTIALLY_SUPPORTED
        # The TRUE count survives the downgrade; a fabricated count was the old bug.
        assert outcome.criteria[0].observed == expected_observed
    assert evaluate_rh1_gate(4, 4, primary_metric="roc_auc").verdict == RH1_SUPPORTED
    assert evaluate_rh1_gate(0, 4, primary_metric="roc_auc").verdict == RH1_NOT_SUPPORTED
    already_partial = evaluate_rh1_gate(
        2, 4, primary_metric="roc_auc", harm_policy_disagreement=True
    )
    assert already_partial.verdict == RH1_PARTIALLY_SUPPORTED


def test_an_engine_whose_evidence_was_not_produced_makes_the_gate_inconclusive() -> None:
    """Protocol section 5: a missing or degenerate interval is 'could not look', not
    'looked and found nothing' -- the collapse the project's own pilot was criticized for."""
    outcome = evaluate_rh1_gate(2, 4, primary_metric="roc_auc", n_engines_unmeasurable=1)
    assert outcome.verdict == RH1_INCONCLUSIVE
    assert "1 unmeasurable" in outcome.criteria[0].observed
    assert outcome.criteria[0].measurable is False
