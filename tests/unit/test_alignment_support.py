"""The common-support control for the per-engine alignment deficit.

EasyOCR left 9 612 components ambiguous against PaddleOCR's 3 334 on the pilot corpus, so
the four engines were measured on different populations. The control has to be
engine-independent and ground-truth-safe or it becomes a second confound, and it must not
quietly drop the engine it was built to check.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ocr_risk.analysis.alignment_support import common_support_tokens, support_sensitivity_table
from ocr_risk.schemas.enums import AlignmentStatus, HarmPolicy

ENGINES = ("engine_a", "engine_b")


def _alignments(rows: list[tuple[str, str, list[str], str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alignment_id": f"a{i}",
                "document_id": document,
                "dataset_id": "funsd",
                "engine_id": engine,
                "gt_token_ids": tokens,
                "status": status,
            }
            for i, (document, engine, tokens, status) in enumerate(rows)
        ]
    )


def test_a_token_only_one_engine_resolved_is_not_common_support() -> None:
    frame = _alignments(
        [
            ("d1", "engine_a", ["t1", "t2"], AlignmentStatus.RESOLVED.value),
            ("d1", "engine_b", ["t1"], AlignmentStatus.RESOLVED.value),
            ("d1", "engine_b", ["t2"], AlignmentStatus.AMBIGUOUS.value),
        ]
    )
    assert common_support_tokens(frame) == {("d1", "t1")}


def test_ambiguity_is_not_treated_as_disagreement_about_the_text() -> None:
    """An ambiguous component contributes to no metric; it is silence, not a wrong answer."""
    frame = _alignments(
        [
            ("d1", "engine_a", ["t1"], AlignmentStatus.AMBIGUOUS.value),
            ("d1", "engine_b", ["t1"], AlignmentStatus.AMBIGUOUS.value),
        ]
    )
    assert common_support_tokens(frame) == set()


def test_the_inclusion_rule_is_the_same_set_of_positions_for_every_engine() -> None:
    """Otherwise the control is a second engine-dependent filter, not a control."""
    frame = _alignments(
        [
            ("d1", "engine_a", ["t1", "t2"], AlignmentStatus.RESOLVED.value),
            ("d1", "engine_b", ["t1", "t2"], AlignmentStatus.RESOLVED.value),
        ]
    )
    support = common_support_tokens(frame)
    assert support == {("d1", "t1"), ("d1", "t2")}


def _sites() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_id": f"{engine}:{token}",
                "document_id": "d1",
                "engine_id": engine,
                "gt_token_ids": [token],
                "d_before": 0 if token == "t1" else 2,
                "evaluable": True,
            }
            for engine in ENGINES
            for token in ("t1", "t2")
        ]
        + [
            {
                "site_id": "engine_a:excluded",
                "document_id": "d1",
                "engine_id": "engine_a",
                "gt_token_ids": ["t3"],
                "d_before": 3,
                "evaluable": False,
            }
        ]
    )


def _pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    labels = []
    for engine in ENGINES:
        for token, outcome in (("t1", "overcorrection"), ("t2", "true_correction")):
            candidate_id = f"{engine}:{token}:c"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "site_id": f"{engine}:{token}",
                    "engine_id": engine,
                    "is_synthetic_hard_negative": False,
                }
            )
            labels.append({"candidate_id": candidate_id, "outcome_if_accepted": outcome})
        adversarial = f"{engine}:t2:hard"
        rows.append(
            {
                "candidate_id": adversarial,
                "site_id": f"{engine}:t2",
                "engine_id": engine,
                "is_synthetic_hard_negative": True,
            }
        )
        labels.append({"candidate_id": adversarial, "outcome_if_accepted": "miscorrection"})
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_both_populations_are_reported_side_by_side() -> None:
    """Nothing is removed from the headline; the two are shown together and compared."""
    alignments = _alignments(
        [
            ("d1", "engine_a", ["t1"], AlignmentStatus.RESOLVED.value),
            ("d1", "engine_a", ["t2"], AlignmentStatus.RESOLVED.value),
            ("d1", "engine_b", ["t1"], AlignmentStatus.RESOLVED.value),
            ("d1", "engine_b", ["t2"], AlignmentStatus.AMBIGUOUS.value),
        ]
    )
    candidates, labels = _pool()
    table = support_sensitivity_table(
        alignments, _sites(), candidates, labels, HarmPolicy.STRICT_WORSENING
    )
    assert set(table["population"]) == {"full", "common_support"}
    assert set(table["engine_id"]) == set(ENGINES)

    full = table[(table["engine_id"] == "engine_a") & (table["population"] == "full")].iloc[0]
    restricted = table[
        (table["engine_id"] == "engine_a") & (table["population"] == "common_support")
    ].iloc[0]
    # t2 is resolved by engine_a but not engine_b, so common support keeps t1 only.
    assert full["n_sites"] == 2
    assert restricted["n_sites"] == 1
    assert restricted["clean_share"] == pytest.approx(1.0)


def test_the_challenge_pool_never_enters_the_sensitivity_rates() -> None:
    """These are natural-pool rates; adversarial rows are ~99% harmful by construction."""
    alignments = _alignments(
        [
            ("d1", engine, [token], AlignmentStatus.RESOLVED.value)
            for engine in ENGINES
            for token in ("t1", "t2")
        ]
    )
    candidates, labels = _pool()
    table = support_sensitivity_table(
        alignments, _sites(), candidates, labels, HarmPolicy.STRICT_WORSENING
    )
    full = table[(table["engine_id"] == "engine_a") & (table["population"] == "full")].iloc[0]
    assert full["n_candidates"] == 2, "the hard negative must be excluded"
    assert full["harmful_rate"] == pytest.approx(0.5)
    assert full["beneficial_rate"] == pytest.approx(0.5)


def test_unevaluable_sites_are_outside_both_populations() -> None:
    """An ambiguous site is carried in the artifact and contributes to no metric."""
    alignments = _alignments(
        [
            ("d1", engine, [token], AlignmentStatus.RESOLVED.value)
            for engine in ENGINES
            for token in ("t1", "t2", "t3")
        ]
    )
    candidates, labels = _pool()
    table = support_sensitivity_table(alignments, _sites(), candidates, labels)
    engine_a = table[(table["engine_id"] == "engine_a") & (table["population"] == "full")].iloc[0]
    assert engine_a["n_sites"] == 2, "the evaluable=False site must not be counted"
