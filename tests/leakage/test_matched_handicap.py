"""The matched-handicap design, verified rather than named.

``in_engine_oracle`` was called a reference condition and treated as one; it was not. It
fitted on four engines against zero-shot's three, saw 23-39% more training candidates, and
could compute a confidence z-score the zero-shot arm structurally cannot. These tests hold
the replacement to the standard the name failed to guarantee: the arms are equal on every
registered dimension, they differ on exactly one, and the certificate says so in a form a
reader can check without rerunning anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ocr_risk.experiments.matched import build_match
from ocr_risk.schemas.documents import SourceDocument
from ocr_risk.schemas.enums import HarmPolicy, SplitRole
from ocr_risk.splits import LeakageError, PartitionSpec, SplitPlan, build_partition, matched_pairs

ENGINES = ("engine_a", "engine_b", "engine_c", "engine_d")
# Deliberately unequal, as the real engines are: docTR emitted 27203 candidates and
# PaddleOCR 21391, so an unmatched design gives one arm a third more data.
PER_ENGINE = {"engine_a": 6, "engine_b": 5, "engine_c": 3, "engine_d": 4}
HARMFUL_SHARE = {"engine_a": 0.5, "engine_b": 0.8, "engine_c": 0.3, "engine_d": 0.6}


def _documents(n: int = 20) -> list[SourceDocument]:
    return [
        SourceDocument(
            document_id=f"doc-{i:03d}",
            dataset_id="synthetic",
            image_path=f"p/{i}.png",
            image_sha256=f"{i:064d}",
            page_index=0,
            width=64,
            height=64,
            gt_text=f"page {i}",
            gt_policy="test",
            has_gt_geometry=False,
            n_gt_tokens=2,
            license_id="synthetic-generated",
        )
        for i in range(n)
    ]


def _frame(documents: list[SourceDocument]) -> pd.DataFrame:
    rows = []
    for document in documents:
        for engine, count in PER_ENGINE.items():
            for k in range(count):
                harmful = (k / count) < HARMFUL_SHARE[engine]
                rows.append(
                    {
                        "candidate_id": f"{document.document_id}:{engine}:{k}",
                        "site_id": f"{document.document_id}:{engine}:s{k}",
                        "document_id": document.document_id,
                        "dataset_id": "synthetic",
                        "engine_id": engine,
                        "outcome_if_accepted": "miscorrection" if harmful else "true_correction",
                        "is_synthetic_hard_negative": False,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def pairs() -> list[object]:
    documents = _documents()
    partition = build_partition(documents, PartitionSpec(seed=7))
    return matched_pairs(ENGINES, partition, _frame(documents))  # type: ignore[return-value]


def test_every_target_engine_is_paired_with_every_donor(pairs: list) -> None:
    """4 targets x 3 donors. A single fixed donor would make the reference arm's fit set
    depend on one arbitrary choice, which on the real corpus moves it by up to 13%."""
    assert len(pairs) == 12
    assert {(p.held_out, p.donor) for p in pairs} == {
        (h, d) for h in ENGINES for d in ENGINES if h != d
    }


def test_both_arms_fit_on_three_engines_and_only_one_of_them_sees_the_target(
    pairs: list,
) -> None:
    for pair in pairs:
        zero_shot, _ = pair.zero_shot.scope_for(SplitRole.FIT)
        reference, _ = pair.reference.scope_for(SplitRole.FIT)
        assert len(zero_shot) == len(reference) == 3
        assert pair.held_out not in zero_shot
        assert pair.held_out in reference
        assert pair.donor not in reference


def test_the_naive_oracle_this_replaces_really_did_fit_on_four(pairs: list) -> None:
    """The comparison that motivates the design, made explicit rather than asserted."""
    documents = _documents()
    partition = build_partition(documents, PartitionSpec(seed=7))
    naive = SplitPlan(
        fold_id="in_engine_oracle:engine_a",
        protocol="in_engine_oracle",
        held_out_engines=frozenset({"engine_a"}),
        all_engines=frozenset(ENGINES),
        partition=partition,
        frame=_frame(documents),
        include_held_out_in_fitting=True,
    )
    assert len(naive.scope_for(SplitRole.FIT)[0]) == 4
    matched = next(p for p in pairs if p.held_out == "engine_a")
    assert len(naive.view(SplitRole.FIT)) > len(matched.zero_shot.view(SplitRole.FIT))


def test_the_matched_sample_equalizes_count_and_class_balance(pairs: list) -> None:
    """Not "adjusted for afterwards": both arms are handed the same counts."""
    for pair in pairs:
        _, _, certificate = build_match(pair, HarmPolicy.STRICT_WORSENING, seed=3)
        left, right = certificate.zero_shot, certificate.reference
        assert left.n_fit_candidates == right.n_fit_candidates
        assert left.n_fit_harmful == right.n_fit_harmful
        assert left.n_calibrate_candidates == right.n_calibrate_candidates
        assert left.n_calibrate_harmful == right.n_calibrate_harmful
        assert certificate.matched, [str(v) for v in certificate.violations]


def test_the_matched_sample_never_touches_the_evaluation_rows(pairs: list) -> None:
    """Both arms must be measured on exactly the same candidates, or nothing is paired."""
    for pair in pairs:
        zero_shot, reference, certificate = build_match(pair, HarmPolicy.STRICT_WORSENING, seed=3)
        evaluate = pair.zero_shot.view(SplitRole.EVALUATE).frame
        assert set(evaluate["candidate_id"]) == set(
            pair.reference.view(SplitRole.EVALUATE).frame["candidate_id"]
        )
        for match in (zero_shot, reference):
            assert len(match.restrict(evaluate, SplitRole.EVALUATE)) == len(evaluate)
        assert certificate.zero_shot.n_evaluate_candidates == len(evaluate)


def test_the_draw_is_reproducible_and_seed_dependent(pairs: list) -> None:
    pair = pairs[0]
    first, _, _ = build_match(pair, HarmPolicy.STRICT_WORSENING, seed=3)
    again, _, _ = build_match(pair, HarmPolicy.STRICT_WORSENING, seed=3)
    other, _, _ = build_match(pair, HarmPolicy.STRICT_WORSENING, seed=4)
    assert first.fit_candidates == again.fit_candidates
    assert first.fit_candidates != other.fit_candidates


def test_the_draw_does_not_depend_on_the_row_order_it_was_handed(pairs: list) -> None:
    """Two arms whose rows arrive differently sorted must still be sampled the same way."""
    documents = _documents()
    partition = build_partition(documents, PartitionSpec(seed=7))
    frame = _frame(documents)
    shuffled = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
    straight = matched_pairs(ENGINES, partition, frame)[0]
    jumbled = matched_pairs(ENGINES, partition, shuffled)[0]
    assert (
        build_match(straight, HarmPolicy.STRICT_WORSENING, seed=3)[0].fit_candidates
        == build_match(jumbled, HarmPolicy.STRICT_WORSENING, seed=3)[0].fit_candidates
    )


def test_the_certificate_fails_when_an_arm_is_not_actually_handicapped(pairs: list) -> None:
    """A reference arm that keeps its z-score is not matched, whatever it is called."""
    from dataclasses import replace

    pair = pairs[0]
    zero_shot, reference, certificate = build_match(pair, HarmPolicy.STRICT_WORSENING, seed=3)
    assert certificate.matched
    from ocr_risk.experiments.matched import _profile

    cheating = replace(reference, ablate_target_engine_zscore=False)
    left = _profile(pair.zero_shot, zero_shot, HarmPolicy.STRICT_WORSENING)
    right = _profile(pair.reference, cheating, HarmPolicy.STRICT_WORSENING)
    assert left.target_engine_zscore_ablated != right.target_engine_zscore_ablated


def test_a_reference_arm_that_does_not_see_its_target_is_a_violation() -> None:
    """The one dimension that must DIFFER. Equality checks alone would pass this."""
    from ocr_risk.splits import MatchedPair

    documents = _documents()
    partition = build_partition(documents, PartitionSpec(seed=7))
    frame = _frame(documents)
    zero_shot = SplitPlan(
        fold_id="loeo_zero_shot:engine_a",
        protocol="loeo_zero_shot",
        held_out_engines=frozenset({"engine_a"}),
        all_engines=frozenset(ENGINES),
        partition=partition,
        frame=frame,
    )
    degenerate = MatchedPair(
        held_out="engine_a", donor="engine_b", zero_shot=zero_shot, reference=zero_shot
    )
    _, _, certificate = build_match(degenerate, HarmPolicy.STRICT_WORSENING, seed=3)
    assert not certificate.matched
    assert any(v.dimension == "target_engine_in_fit_set" for v in certificate.violations)


def test_a_fit_set_containing_the_target_must_declare_in_engine_fitting() -> None:
    """Otherwise a reference arm reads as a zero-shot result in every downstream table."""
    documents = _documents()
    partition = build_partition(documents, PartitionSpec(seed=7))
    with pytest.raises(LeakageError, match="does not declare in-engine fitting"):
        SplitPlan(
            fold_id="sneaky",
            protocol="loeo_zero_shot",
            held_out_engines=frozenset({"engine_a"}),
            all_engines=frozenset(ENGINES),
            partition=partition,
            frame=_frame(documents),
            fit_engines_override=frozenset({"engine_a", "engine_b"}),
        )


def test_the_matched_protocol_is_recognized_as_engine_axis_contamination_by_design(
    pairs: list,
) -> None:
    """It fits on its own target. The audit must classify that as declared, not as a leak."""
    from ocr_risk.splits.audit import _check_descriptor

    for pair in pairs:
        pair.reference.assert_no_leakage()
        findings = _check_descriptor(pair.reference.descriptor(), leaky_run=False)
        assert not [f for f in findings if f.severity == "HIGH"], [str(f) for f in findings]


def test_the_certificate_serializes_to_something_a_reviewer_can_read(pairs: list) -> None:
    import json

    _, _, certificate = build_match(
        pairs[0], HarmPolicy.STRICT_WORSENING, seed=3, shared={"calibrator": "isotonic"}
    )
    payload = json.loads(json.dumps(certificate.as_dict()))
    assert payload["matched"] is True
    assert payload["shared"]["calibrator"] == "isotonic"
    assert payload["zero_shot"]["fit_engines"] != payload["reference"]["fit_engines"]
    assert payload["zero_shot"]["n_fit_candidates"] == payload["reference"]["n_fit_candidates"]


# ------------------------------------------------- the handicap where it is applied


def _bundle(candidate_id: str, engine_id: str, confidence: float) -> object:
    from ocr_risk.schemas.evidence import ConfidenceFeatures, EvidenceBundle, GeometryFeatures

    return EvidenceBundle(
        candidate_id=candidate_id,
        site_id=candidate_id,
        document_id=candidate_id.split(":")[0],
        dataset_id="synthetic",
        engine_id=engine_id,
        original_ocr="addres",
        candidate_text="address",
        context_before="the ",
        context_after=" field",
        conf_features=ConfidenceFeatures(
            native_min=confidence,
            native_mean=confidence,
            conf_normalized=confidence,
            n_spans=1,
            has_native_confidence=True,
        ),
        geom_features=GeometryFeatures(n_spans=1, has_geometry=False),
        available_fields=(),
        masked_fields=(),
    )


def _run_matched_fold(*, with_match: bool) -> object:
    """One fold of each arm, with and without the handicap, on the same rows."""
    from ocr_risk.config.models import EngineConfig, ExperimentConfig, VerifierSpec
    from ocr_risk.experiments.loeo_runner import ConfSample, MethodSpec, run_fold

    documents = _documents()
    partition = build_partition(documents, PartitionSpec(seed=7))
    frame = _frame(documents)
    pair = next(p for p in matched_pairs(ENGINES, partition, frame) if p.held_out == "engine_a")

    bundles = {}
    conf_samples = {}
    for row in frame.itertuples():
        # Engine-specific confidence scales, as the real engines have: PaddleOCR's mean
        # normalized confidence was 0.924 against EasyOCR's 0.519 on the pilot corpus.
        base = {"engine_a": 0.9, "engine_b": 0.5, "engine_c": 0.7, "engine_d": 0.6}[row.engine_id]
        confidence = base + 0.05 * (hash(row.candidate_id) % 3)
        bundles[row.candidate_id] = _bundle(row.candidate_id, row.engine_id, confidence)
        conf_samples[row.candidate_id] = ConfSample(
            engine_id=row.engine_id,
            native_confidences=(confidence,),
            conf_scale="paddleocr_rec_score_0_1",
        )

    config = ExperimentConfig(
        name="matched_probe",
        engines=tuple(EngineConfig(id=e, adapter="replay") for e in ENGINES),
        verifiers=(VerifierSpec(id="v3", kind="feature_logistic", evidence_config="v3"),),
    )
    method = MethodSpec(
        verifier_id="v3",
        kind="feature_logistic",
        evidence_config="v3",
        calibration_method=config.calibration.method,
    )
    match = None
    if with_match:
        _, match, _ = build_match(pair, config.risk.harm_policy, seed=3)
    return run_fold(pair.reference, method, config, bundles, conf_samples=conf_samples, match=match)


def test_the_handicap_shrinks_the_fitting_sample_and_says_so() -> None:
    """Without the match the reference arm keeps its whole fit set, which is the confound."""
    unmatched = _run_matched_fold(with_match=False)
    matched = _run_matched_fold(with_match=True)
    assert matched.diagnostics["matched_handicap"] is True  # type: ignore[attr-defined]
    assert unmatched.diagnostics["matched_handicap"] is False  # type: ignore[attr-defined]
    assert matched.diagnostics["n_fit"] < unmatched.diagnostics["n_fit"]  # type: ignore[attr-defined]
    assert matched.diagnostics["target_engine_zscore_ablated"] is True  # type: ignore[attr-defined]


def test_the_handicap_leaves_the_evaluation_rows_untouched() -> None:
    """Both arms must be scored on the same candidates, or the pairing is not a pairing."""
    unmatched = _run_matched_fold(with_match=False)
    matched = _run_matched_fold(with_match=True)
    assert matched.diagnostics["n_evaluate"] == unmatched.diagnostics["n_evaluate"]  # type: ignore[attr-defined]
    assert {p.candidate_id for p in matched.predictions} == {  # type: ignore[attr-defined]
        p.candidate_id
        for p in unmatched.predictions  # type: ignore[attr-defined]
    }


def test_ablating_the_z_score_changes_the_scores_the_reference_arm_produces() -> None:
    """If it did not, the handicap would be decoration rather than a control.

    The reference arm can compute a z-score for its target engine and the zero-shot arm
    cannot; discarding it is what removes that asymmetry, so it has to actually bite.
    """
    unmatched = _run_matched_fold(with_match=False)
    matched = _run_matched_fold(with_match=True)
    before = {p.candidate_id: p.raw_score for p in unmatched.predictions}  # type: ignore[attr-defined]
    after = {p.candidate_id: p.raw_score for p in matched.predictions}  # type: ignore[attr-defined]
    assert any(before[k] != after[k] for k in before)


# ------------------------------------------------ what the certificate must not miss


def test_every_numeric_profile_field_is_registered_or_deliberately_exempt() -> None:
    """The docstring claimed forgetting to register a field was "visible". It was not.

    ``n_fit_beneficial`` differed by 13% between arms across twelve certificates that all
    read ``matched: true``, because it was recorded and never checked. This test is the
    mechanism that claim needed.
    """
    from dataclasses import fields

    from ocr_risk.experiments.matched import (
        _DELIBERATELY_UNEQUAL,
        _REGISTERED_EQUAL,
        ArmProfile,
    )

    numeric = {
        f.name
        for f in fields(ArmProfile)
        if f.type in {"int", "float", "bool"} or f.name.startswith("n_")
    }
    unaccounted = numeric - set(_REGISTERED_EQUAL) - set(_DELIBERATELY_UNEQUAL)
    assert not unaccounted, f"{sorted(unaccounted)} are measured and then ignored"


def test_a_reference_arm_that_reverts_to_the_four_engine_oracle_is_caught() -> None:
    """The exact regression Amendment 4 exists to retract, red-teamed.

    Drop ``fit_engines_override`` and the reference arm IS ``in_engine_oracle`` again. The
    count matching hides it — the subsample equalizes totals however many engines they came
    from — so the engine cardinality has to be a registered dimension in its own right.
    """
    from ocr_risk.splits import MatchedPair

    documents = _documents()
    partition = build_partition(documents, PartitionSpec(seed=7))
    frame = _frame(documents)
    pair = next(p for p in matched_pairs(ENGINES, partition, frame) if p.held_out == "engine_a")
    reverted = SplitPlan(
        fold_id="in_engine_oracle:engine_a",
        protocol="matched_in_engine",
        held_out_engines=frozenset({"engine_a"}),
        all_engines=frozenset(ENGINES),
        partition=partition,
        frame=frame,
        include_held_out_in_fitting=True,
    )
    degenerate = MatchedPair(
        held_out="engine_a", donor="engine_b", zero_shot=pair.zero_shot, reference=reverted
    )
    _, _, certificate = build_match(degenerate, HarmPolicy.STRICT_WORSENING, seed=3)
    assert not certificate.matched
    assert any(v.dimension == "n_fit_engines" for v in certificate.violations), [
        str(v) for v in certificate.violations
    ]


def test_the_draw_matches_the_natural_sub_pool_not_only_the_mixed_total() -> None:
    """Hard negatives are fitted on but excluded from evaluation and threshold selection.

    Matching the mixed totals left the natural sub-pool differing by up to 9% between arms,
    with a sign set by the target engine's hard-negative rate — a training-opportunity
    difference aligned with the evaluation distribution, which is the confound the arm
    exists to remove.
    """
    documents = _documents()
    partition = build_partition(documents, PartitionSpec(seed=7))
    frame = _frame(documents)
    # Hard-negative rates that differ sharply by engine, as the real pool's do.
    challenge_rate = {"engine_a": 0.7, "engine_b": 0.2, "engine_c": 0.5, "engine_d": 0.3}
    rows = []
    for index, row in enumerate(frame.itertuples()):
        rows.append(
            {
                **row._asdict(),
                "is_synthetic_hard_negative": (index % 10) / 10.0
                < challenge_rate[str(row.engine_id)],
            }
        )
    seeded = pd.DataFrame(rows).drop(columns=["Index"])

    for pair in matched_pairs(ENGINES, partition, seeded):
        _, _, certificate = build_match(pair, HarmPolicy.STRICT_WORSENING, seed=3)
        assert certificate.matched, [str(v) for v in certificate.violations]
        assert certificate.zero_shot.n_fit_natural == certificate.reference.n_fit_natural
        assert (
            certificate.zero_shot.n_fit_natural_harmful
            == certificate.reference.n_fit_natural_harmful
        )
        assert (
            certificate.zero_shot.n_calibrate_natural == certificate.reference.n_calibrate_natural
        )
        assert certificate.zero_shot.n_fit_natural < certificate.zero_shot.n_fit_candidates
