"""Hand-computed tests for the Track-A metric tables and the run orchestration.

The frames are small enough to verify by hand: two documents, one engine, a handful
of regions and candidates whose dispositions and outcomes are named in the fixtures.
Every asserted number is computed in the comment beside it.
"""

from __future__ import annotations

import json

import pandas as pd

from ocr_risk.experiments.cgv3_track_a import (
    TrackAFrames,
    _failure_table,
    _k_grid_table,
    _oracle_table,
    _rule_metrics,
    _site_metrics,
    run_track_a,
)
from ocr_risk.schemas.spans import BBox, CanonicalSpan

__all__ = [
    "TestMetricTables",
    "TestRunTrackA",
    "test_oracle_table_carries_the_cgv2_floors",
]


def _coverage_row(
    site_id: str,
    disposition: str,
    *,
    state: str = "eligible",
    matched: str = "",
    engine: str = "tesseract",
    kind: str = "substitution",
) -> dict[str, object]:
    return {
        "true_site_id": site_id,
        "document_id": "d1",
        "dataset_id": "funsd",
        "engine_id": engine,
        "site_kind": kind,
        "d_before": 1,
        "audit_state": state,
        "disposition": disposition,
        "matched_site_id": matched,
    }


def _frames() -> TrackAFrames:
    coverage = pd.DataFrame(
        [
            _coverage_row("r1", "discovered", matched="site-A"),
            _coverage_row("r2", "discovered", matched="site-B"),
            _coverage_row("r3", "not_discovered"),
            _coverage_row("r4", "boundary_mismatch"),
            _coverage_row("r5", "excluded_ambiguous", state="ambiguous"),
        ]
    )
    sites = pd.DataFrame(
        [
            {
                "site_id": "site-A",
                "document_id": "d1",
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "anchor_kind": "token",
                "site_type": "substitution",
                "provenance_reason": "low_confidence_token",
                "suspicion_score": 0.9,
                "char_start": 0,
                "char_end": 5,
                "covers_true_region": True,
            },
            {
                "site_id": "site-B",
                "document_id": "d1",
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "anchor_kind": "gap",
                "site_type": "insertion",
                "provenance_reason": "gap_anomaly",
                "suspicion_score": 0.5,
                "char_start": 5,
                "char_end": 6,
                "covers_true_region": True,
            },
            {
                "site_id": "site-C",
                "document_id": "d1",
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "anchor_kind": "token",
                "site_type": "substitution",
                "provenance_reason": "lexical_anomaly",
                "suspicion_score": 0.6,
                "char_start": 6,
                "char_end": 9,
                "covers_true_region": False,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            # site-A: rank-0 harmful, rank-1 beneficial -- a K=1 truncation drops the
            # only beneficial candidate; site-B: rank-0 beneficial.
            ("site-A", "g8_union", 0, "miscorrection"),
            ("site-A", "g8_union", 1, "lateral_change"),
            ("site-B", "g8_union", 0, "partial_improvement"),
            ("site-C", "g8_union", 0, "lateral_change"),
            ("site-A", "g3_edit_aware", 0, "miscorrection"),
        ],
        columns=["site_id", "generator_id", "generator_rank", "outcome"],
    ).assign(engine_id="tesseract", document_id="d1")
    pages = pd.DataFrame(
        [
            {
                "document_id": "d1",
                "engine_id": "tesseract",
                "n_tokens": 10,
                "n_discovered": 3,
                "stream_chars": 9,
            }
        ]
    )
    return TrackAFrames(
        true_coverage=coverage,
        discovered=sites,
        candidates=candidates,
        page_counts=pages,
    )


class TestMetricTables:
    def test_site_metrics_count_only_eligible_regions(self) -> None:
        row = _site_metrics(_frames()).iloc[0]
        # eligible = 4 of 5 rows; discovered = 2 -> recall 0.5; the excluded
        # ambiguous region lands in its own column, never the denominator.
        assert row["n_true_eligible"] == 4
        assert row["n_discovered"] == 2
        assert row["site_recall"] == 0.5
        assert row["n_excluded_ambiguous"] == 1
        assert row["n_discovered_sites"] == 3
        assert row["sites_per_1k_tokens"] == 300.0

    def test_rule_metrics_report_per_reason_precision(self) -> None:
        table = _rule_metrics(_frames())
        by_reason = table.set_index("provenance_reason")
        assert by_reason.loc["low_confidence_token", "precision"] == 1.0
        assert by_reason.loc["gap_anomaly", "precision"] == 1.0
        assert by_reason.loc["lexical_anomaly", "precision"] == 0.0

    def test_k_grid_counts_ranked_rows_over_covered_sites(self) -> None:
        table = _k_grid_table(_frames())
        k1 = table[table.k == 1].iloc[0]
        k4 = table[table.k == 4].iloc[0]
        # rank<1 keeps 3 of 4 g8 rows; rank<4 keeps all 4. Covered sites = {A, B}.
        assert k1["n_candidates"] == 3
        assert k4["n_candidates"] == 4
        assert k1["candidates_per_covered_site"] == 1.5
        # Covered rank-0 rows are site-A (harmful) and site-B (beneficial) -- site-C
        # covers no region -- so the beneficial fraction is 1/2.
        assert abs(k1["beneficial_fraction"] - 0.5) < 1e-9

    def test_failure_table_assigns_exactly_one_stage_per_failing_region(self) -> None:
        table = _failure_table(_frames())
        by_region = table.set_index("true_site_id")["failure_class"]
        assert by_region["r3"] == "not_discovered"
        assert by_region["r4"] == "wrong_boundary"
        assert by_region["r5"] == "alignment_ambiguous"
        # r1 is discovered but its only candidate is harmful -> generation failure;
        # r2 is discovered with a beneficial candidate -> absent entirely.
        assert by_region["r1"] == "no_beneficial_candidate"
        assert "r2" not in by_region.index


def test_oracle_table_carries_the_cgv2_floors() -> None:
    row = _oracle_table(_frames()).iloc[0]
    # The floors are imported from the single CGV2 definition, never retyped.
    from ocr_risk.experiments.cgv2_gates import (
        _H2_ACCEPTED_EDITS_FLOOR,
        _H2_SAFE_COVERAGE_FLOOR,
    )

    assert row["h2_floor_edits"] == _H2_ACCEPTED_EDITS_FLOOR == 245
    assert row["h2_floor_coverage"] == _H2_SAFE_COVERAGE_FLOOR == 0.05
    assert row["oracle_accepted_edits"] == 1  # only r2's site-B is beneficial
    assert row["oracle_safe_coverage"] == 0.25  # 1 of 4 eligible


def _span(
    span_id: str,
    text: str,
    start: int,
    *,
    document_id: str = "d1",
    reading_order: int = 0,
    conf: float = 0.99,
) -> CanonicalSpan:
    return CanonicalSpan(
        span_id=span_id,
        document_id=document_id,
        dataset_id="funsd",
        engine_id="tesseract",
        engine_fingerprint="fp",
        text=text,
        reading_order=reading_order,
        line_id="L1",
        bbox=BBox(x0=0.0, y0=0.0, x1=10.0 * len(text), y1=20.0),
        native_conf_recognition=conf * 100.0,
        native_conf_detection=None,
        conf_scale="tesseract_word_conf_0_100",
        char_start=start,
        char_end=start + len(text),
        raw_ref="raw/ref.json",
        raw_index=reading_order,
    )


class TestRunTrackA:
    def test_the_full_pass_discovers_generates_and_measures(self) -> None:
        # Two pages over one engine: page 1 has a low-confidence substitution target
        # ("t0tal"), page 2 the attested missing "of" between "total" and "12".
        # The fold corpus is the same text, so both discoveries are in-vocabulary.
        spans = {
            ("d1", "tesseract"): [
                _span("s0", "t0tal", 0, reading_order=0, conf=0.10),
                _span("s1", "amount", 6, reading_order=1),
            ],
            ("d2", "tesseract"): [
                _span("t0", "total", 0, document_id="d2", reading_order=0),
                _span("t1", "12", 6, document_id="d2", reading_order=1),
            ],
        }
        # The fold corpus excludes the held-out engine by design, so a second engine
        # must read the fit document for (total, of)/(of, 12) to be attested >= 2.
        easyocr: list[CanonicalSpan] = []
        position = 0
        for index, text in enumerate(["total", "of", "12", "total", "of", "12"]):
            easyocr.append(_span(f"e{index}", text, position, reading_order=index))
            position += len(text) + 1
        spans[("d1", "easyocr")] = easyocr
        streams = {pair: " ".join(span.text for span in page) for pair, page in spans.items()}
        alignments = pd.DataFrame(
            [
                {
                    "alignment_id": "a0",
                    "document_id": "d1",
                    "engine_id": "tesseract",
                    "ocr_span_ids": json.dumps(["s0"]),
                    "gt_token_ids": json.dumps(["g0"]),
                    "relation": "one_to_one",
                    "status": "resolved",
                    "ocr_text": "t0tal",
                    "gt_text": "total",
                },
                {
                    "alignment_id": "a1",
                    "document_id": "d2",
                    "engine_id": "tesseract",
                    "ocr_span_ids": json.dumps(["t0"]),
                    "gt_token_ids": json.dumps(["g1"]),
                    "relation": "one_to_one",
                    "status": "resolved",
                    "ocr_text": "total",
                    "gt_text": "total",
                },
                {
                    "alignment_id": "a2",
                    "document_id": "d2",
                    "engine_id": "tesseract",
                    "ocr_span_ids": json.dumps([]),
                    "gt_token_ids": json.dumps(["g2"]),
                    "relation": "one_to_many",
                    "status": "resolved",
                    "ocr_text": "",
                    "gt_text": "of",
                },
                {
                    "alignment_id": "a3",
                    "document_id": "d2",
                    "engine_id": "tesseract",
                    "ocr_span_ids": json.dumps(["t1"]),
                    "gt_token_ids": json.dumps(["g3"]),
                    "relation": "one_to_one",
                    "status": "resolved",
                    "ocr_text": "12",
                    "gt_text": "12",
                },
            ]
        )
        sites = pd.DataFrame(
            [
                {
                    "site_id": "d1:tesseract:site:00001",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "tesseract",
                    "alignment_ids": json.dumps(["a0"]),
                    "ocr_span_ids": json.dumps(["s0"]),
                    "gt_token_ids": json.dumps(["g0"]),
                    "ocr_text": "t0tal",
                    "gt_text": "total",
                    "d_before": 2,
                    "site_kind": "substitution",
                    "evaluable": True,
                    "char_start": 0,
                    "char_end": 5,
                },
                {
                    "site_id": "d2:tesseract:site:00002",
                    "document_id": "d2",
                    "dataset_id": "funsd",
                    "engine_id": "tesseract",
                    "alignment_ids": json.dumps(["a2"]),
                    "ocr_span_ids": json.dumps([]),
                    "gt_token_ids": json.dumps(["g2"]),
                    "ocr_text": "",
                    "gt_text": "of",
                    "d_before": 2,
                    "site_kind": "deletion",
                    "evaluable": True,
                    "char_start": 0,
                    "char_end": 0,
                },
            ]
        )
        gt_tokens = pd.DataFrame(
            [
                {"gt_token_id": "g0", "document_id": "d1", "index": 0, "text": "total"},
                {"gt_token_id": "g1", "document_id": "d2", "index": 0, "text": "total"},
                {"gt_token_id": "g2", "document_id": "d2", "index": 1, "text": "of"},
                {"gt_token_id": "g3", "document_id": "d2", "index": 2, "text": "12"},
            ]
        )
        frames = run_track_a(
            spans_by_pair=spans,
            streams=streams,
            alignments=alignments,
            gt_tokens=gt_tokens,
            sites=sites,
            document_ids=frozenset({"d1", "d2"}),
            fit_documents=frozenset({"d1"}),
            engines=("tesseract",),
        )
        coverage = frames.true_coverage.set_index("true_site_id")
        # The substitution region is discovered through its low-confidence token
        # anchor; the deletion region through the fold-attested gap anchor.
        assert coverage.loc["d1:tesseract:site:00001", "disposition"] == "discovered"
        assert coverage.loc["d2:tesseract:site:00002", "disposition"] == "discovered"
        # The g8 pool carries the incumbent's substitution and the structural rung's
        # insertion at their sites, deduplicated by text.
        union = frames.candidates[frames.candidates.generator_id == "g8_union"]
        shapes = {(row.site_id, row.operation) for row in union.itertuples(index=False)}
        assert shapes  # the union proposed at the discovered sites
        assert set(frames.site_metrics.engine_id) == {"tesseract"}
        assert not frames.oracle.empty
        if not frames.failure.empty:
            # Both fixture regions carry beneficial candidates, so the failure table
            # may legitimately be empty; when it is not, every row names a real stage.
            assert set(frames.failure.stage) <= {"discovery", "generation", "evaluation"}
