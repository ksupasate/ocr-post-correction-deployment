"""The alignment battery, driven by the declarative fixture cases.

The property this suite protects is not "alignment works" but "alignment refuses to
pretend". Coverage is easy to inflate by forcing weak matches, and every forced match
manufactures a correction site whose ground truth is an artifact — which then corrupts the
harm labels and the headline risk numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from builders import make_document, make_gt_tokens, make_spans

from ocr_risk.align import align_document, summarize
from ocr_risk.config.models import AlignmentConfig
from ocr_risk.schemas.enums import AlignmentRelation, AlignmentStatus

CASES: list[dict[str, Any]] = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "fixtures" / "alignment" / "cases.yaml").read_text()
)["cases"]
CASE_IDS = [case["id"] for case in CASES]


def _run(case: dict[str, Any], config: AlignmentConfig | None = None):  # type: ignore[no-untyped-def]
    tokens, gt_text = make_gt_tokens(case["gt"])
    spans = make_spans(case["ocr"])
    document = make_document(gt_text=gt_text)
    return align_document(document, spans, tokens, config or AlignmentConfig())


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_expected_relations_appear(case: dict[str, Any]) -> None:
    outcome = _run(case)
    relations = {r.relation.value for r in outcome.records}
    for expected in case.get("expect", []):
        assert expected in relations, (
            f"{case['id']}: expected relation {expected!r}; got {sorted(relations)}\n"
            f"why: {case['why']}"
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_forbidden_relations_absent(case: dict[str, Any]) -> None:
    outcome = _run(case)
    relations = {r.relation.value for r in outcome.records}
    for forbidden in case.get("forbid", []):
        assert forbidden not in relations, (
            f"{case['id']}: relation {forbidden!r} must not occur\n"
            f"why: {case['why']}\n"
            + "\n".join(
                f"  {r.relation.value:14} {r.ocr_text!r} -> {r.gt_text!r}" for r in outcome.records
            )
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_status_bounds(case: dict[str, Any]) -> None:
    outcome = _run(case)
    resolved = [r for r in outcome.records if r.status is AlignmentStatus.RESOLVED]
    if "min_resolved" in case:
        assert len(resolved) >= case["min_resolved"], (
            f"{case['id']}: only {len(resolved)} resolved components\n"
            + "\n".join(
                f"  {r.relation.value:14} {r.status.value:11} {r.ocr_text!r} -> {r.gt_text!r}"
                for r in outcome.records
            )
        )
    if "max_resolved_one_to_one" in case:
        forced = [r for r in resolved if r.relation is AlignmentRelation.ONE_TO_ONE]
        assert len(forced) <= case["max_resolved_one_to_one"], (
            f"{case['id']}: aligner forced {len(forced)} matches it should have refused\n"
            + "\n".join(
                f"  {r.ocr_text!r} -> {r.gt_text!r} conf={r.align_confidence:.3f}" for r in forced
            )
        )
    if "min_ambiguous" in case:
        distrusted = [
            r
            for r in outcome.records
            if r.status in (AlignmentStatus.AMBIGUOUS, AlignmentStatus.UNRESOLVED)
        ]
        assert len(distrusted) >= case["min_ambiguous"], (
            f"{case['id']}: expected at least {case['min_ambiguous']} distrusted "
            f"component(s), got {len(distrusted)}\n"
            + "\n".join(
                f"  {r.status.value:11} {r.ocr_text!r} -> {r.gt_text!r} "
                f"conf={r.align_confidence:.3f}"
                for r in outcome.records
            )
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_records_are_internally_consistent(case: dict[str, Any]) -> None:
    outcome = _run(case)
    assert outcome.records, f"{case['id']} produced no alignment records at all"
    for record in outcome.records:
        assert 0.0 <= record.align_confidence <= 1.0
        assert 0.0 <= record.char_agreement <= 1.0
        assert 0.0 <= record.uniqueness_margin <= 1.0
        assert record.edit_distance >= 0
        if record.relation is AlignmentRelation.OCR_INSERTION:
            assert record.ocr_span_ids and not record.gt_token_ids
        if record.relation is AlignmentRelation.OCR_DELETION:
            assert record.gt_token_ids and not record.ocr_span_ids
        if record.status is not AlignmentStatus.RESOLVED:
            assert record.reason_code, "a non-resolved component must say why"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_no_span_or_token_is_claimed_twice(case: dict[str, Any]) -> None:
    """Components partition the spans and tokens; double-claiming would double-count
    every downstream site."""
    outcome = _run(case)
    span_ids = [sid for r in outcome.records for sid in r.ocr_span_ids]
    token_ids = [tid for r in outcome.records for tid in r.gt_token_ids]
    assert len(span_ids) == len(set(span_ids)), "an OCR span appears in two components"
    assert len(token_ids) == len(set(token_ids)), "a GT token appears in two components"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_input_is_accounted_for(case: dict[str, Any]) -> None:
    """Nothing may be silently dropped: every span and token must appear somewhere."""
    outcome = _run(case)
    claimed_spans = {sid for r in outcome.records for sid in r.ocr_span_ids}
    claimed_tokens = {tid for r in outcome.records for tid in r.gt_token_ids}
    tokens, _ = make_gt_tokens(case["gt"])
    spans = make_spans(case["ocr"])
    assert claimed_spans == {s.span_id for s in spans}
    assert claimed_tokens == {t.gt_token_id for t in tokens}


# --- specific properties ------------------------------------------------------------------
def test_split_and_merge_are_mirror_images() -> None:
    """If OCR ['Sm','ith'] vs GT ['Smith'] is a SPLIT, then swapping the roles must give a
    MERGE. An asymmetry here means the relation typing is wrong, not merely different."""
    forward = _run({"gt": ["Smith"], "ocr": ["Sm", "ith"], "id": "f", "why": ""})
    backward = _run({"gt": ["Sm", "ith"], "ocr": ["Smith"], "id": "b", "why": ""})
    assert {r.relation for r in forward.records} == {AlignmentRelation.SPLIT}
    assert {r.relation for r in backward.records} == {AlignmentRelation.MERGE}


def test_lowering_the_floor_cannot_reduce_resolved_count() -> None:
    """Monotonicity: the confidence floor only ever gates components, never creates them."""
    case = {
        "gt": ["alpha", "beta", "gamma"],
        "ocr": ["alpba", "beta", "gemma"],
        "id": "m",
        "why": "",
    }
    strict = _run(case, AlignmentConfig(min_align_confidence=0.95))
    loose = _run(case, AlignmentConfig(min_align_confidence=0.05))
    n_strict = sum(1 for r in strict.records if r.status is AlignmentStatus.RESOLVED)
    n_loose = sum(1 for r in loose.records if r.status is AlignmentStatus.RESOLVED)
    assert n_loose >= n_strict


def test_ambiguous_components_are_retained_not_dropped() -> None:
    """Ambiguity must be counted, not hidden: per-engine ambiguity rate is a confound for
    the cross-engine comparison."""
    case = {"gt": ["alpha", "beta"], "ocr": ["zzzzz", "qqqqq"], "id": "a", "why": ""}
    outcome = _run(case, AlignmentConfig(min_align_confidence=0.99))
    assert outcome.records, "ambiguous components were dropped instead of retained"
    non_resolved = [r for r in outcome.records if r.status is not AlignmentStatus.RESOLVED]
    assert non_resolved
    assert all(r.reason_code for r in non_resolved)


def test_no_geometry_path_renormalizes_weights() -> None:
    """Plain-text ground truth must not be penalized for lacking boxes."""
    tokens, gt_text = make_gt_tokens(["Dose:", "0.015", "mg"], with_geometry=False)
    spans = make_spans(["Dose:", "0.015", "mg"], with_geometry=False)
    outcome = align_document(make_document(gt_text=gt_text), spans, tokens, AlignmentConfig())
    assert not outcome.used_geometry
    assert all(r.geom_score is None for r in outcome.records)
    resolved = [r for r in outcome.records if r.status is AlignmentStatus.RESOLVED]
    assert len(resolved) == 3, "identical text failed to align without geometry"
    assert all(
        r.diagnostics.get("geometry") == "unavailable_weights_renormalized" for r in outcome.records
    )


def test_geometry_improves_confidence_when_boxes_agree() -> None:
    tokens, gt_text = make_gt_tokens(["alpha", "beta"])
    spans = make_spans(["alpha", "beta"])
    document = make_document(gt_text=gt_text)
    with_geom = align_document(document, spans, tokens, AlignmentConfig(use_geometry=True))
    assert all(r.geom_score is not None for r in with_geom.records)
    assert all(r.geom_score > 0.5 for r in with_geom.records if r.geom_score is not None)


def test_summary_reports_ambiguity_rate_per_engine() -> None:
    tokens, gt_text = make_gt_tokens(["alpha", "beta"])
    document = make_document(gt_text=gt_text)
    good = align_document(
        document, make_spans(["alpha", "beta"], engine_id="good"), tokens, AlignmentConfig()
    )
    bad = align_document(
        document, make_spans(["zzzzz", "qqqqq"], engine_id="bad"), tokens, AlignmentConfig()
    )

    stats = summarize([*good.records, *bad.records])
    assert set(stats) == {"good", "bad"}
    assert stats["good"].ambiguity_rate < stats["bad"].ambiguity_rate
    assert stats["good"].mean_char_agreement > stats["bad"].mean_char_agreement


def test_empty_inputs_produce_no_records() -> None:
    document = make_document(gt_text="")
    assert align_document(document, [], [], AlignmentConfig()).records == []
