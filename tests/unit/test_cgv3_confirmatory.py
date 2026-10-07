"""Hand-computed tests for the confirmatory execution core.

The load-bearing test is ``test_generation_pass_reproduces_run_track_a``: the pre-GT
generation replication must produce, on the same inputs, exactly the generation
columns ``run_track_a`` produces internally -- that equality is what the confirmatory
determinism assertion checks on the fresh data, so it is proven here on a fixture
small enough to verify by hand. The remaining tests cover the inverse freshness
guard, the unlabeled-row semantics, and the equality assertion's red-team paths.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ocr_risk.candidates.base import CandidateProposal
from ocr_risk.discovery.enumerator import DiscoveredSite, DiscoveryProvenance
from ocr_risk.experiments.cgv3_confirmatory import (
    ConfirmatoryDivergenceError,
    _unlabeled_rows,
    assert_confirmatory_selection,
    assert_frames_match_frozen,
    discovered_region_truth,
    discovery_pass,
    fit_fold_resources,
    generation_pass,
    read_frozen_csv,
)
from ocr_risk.experiments.cgv3_study import AlignmentIndex
from ocr_risk.experiments.cgv3_track_a import run_track_a
from ocr_risk.schemas.enums import AnchorKind
from ocr_risk.schemas.spans import BBox, CanonicalSpan

__all__ = [
    "TestAssertConfirmatorySelection",
    "TestDeterminismAssertion",
    "TestPasses",
    "TestReadFrozenCsv",
    "TestUnlabeledRows",
    "_span",
]


def _span(
    span_id: str,
    text: str,
    start: int,
    *,
    document_id: str = "d1",
    reading_order: int = 0,
    conf: float = 0.99,
    engine_id: str = "tesseract",
) -> CanonicalSpan:
    return CanonicalSpan(
        span_id=span_id,
        document_id=document_id,
        dataset_id="funsd",
        engine_id=engine_id,
        engine_fingerprint="fp",
        text=text,
        reading_order=reading_order,
        line_id="L1",
        bbox=BBox(x0=0.0, y0=0.0, x1=10.0 * len(text), y1=20.0),
        native_conf_recognition=conf * 100.0 if engine_id == "tesseract" else conf,
        native_conf_detection=None,
        conf_scale="tesseract_word_conf_0_100" if engine_id == "tesseract" else "unit_conf",
        char_start=start,
        char_end=start + len(text),
        raw_ref="raw/ref.json",
        raw_index=reading_order,
    )


class TestAssertConfirmatorySelection:
    def test_the_exact_reserve_passes(self) -> None:
        from ocr_risk.discovery.freshness import confirmatory_document_ids

        reserve = confirmatory_document_ids()
        assert len(reserve) == 99
        assert assert_confirmatory_selection(reserve, context="test") == reserve

    def test_a_seen_document_mixed_in_raises(self) -> None:
        from ocr_risk.discovery.freshness import confirmatory_document_ids

        poisoned = set(confirmatory_document_ids())
        poisoned.discard(sorted(poisoned)[0])
        poisoned.add("funsd-training_data-0000989556")  # a Track-A calibrate document
        with pytest.raises(ConfirmatoryDivergenceError, match="not exactly"):
            assert_confirmatory_selection(poisoned, context="test")

    def test_one_missing_document_raises(self) -> None:
        from ocr_risk.discovery.freshness import confirmatory_document_ids

        short = set(confirmatory_document_ids())
        short.discard(sorted(short)[0])
        with pytest.raises(ConfirmatoryDivergenceError, match="missing 1"):
            assert_confirmatory_selection(short, context="test")


def _site(
    site_id: str = "d1:tesseract:dsite:00000", kind: AnchorKind = AnchorKind.TOKEN
) -> DiscoveredSite:
    return DiscoveredSite(
        site_id=site_id,
        document_id="d1",
        dataset_id="funsd",
        engine_id="tesseract",
        anchor_kind=kind,
        anchor_ref=f"{kind.value}\0sp0",
        char_start=0,
        char_end=5,
        site_type="substitution",
        suspicion_score=0.9,
        provenance_reason=DiscoveryProvenance.LOW_CONFIDENCE_TOKEN,
        signals={"normalized_confidence": "0.1000"},
    )


class TestReadFrozenCsv:
    def test_anchor_ref_nul_separator_survives_the_roundtrip(self, tmp_path) -> None:
        frame = pd.DataFrame(
            [{"anchor_kind": "token", "anchor_ref": "token\0funsd-d1:tesseract:00000"}]
        )
        path = tmp_path / "sites.csv"
        frame.to_csv(path, index=False)
        back = read_frozen_csv(path)
        assert back.loc[0, "anchor_ref"] == "token\0funsd-d1:tesseract:00000"

    def test_the_default_c_parser_truncates_at_nul(self, tmp_path) -> None:
        """Red team: proves the helper is load-bearing, not decorative (DEFECT-2)."""
        frame = pd.DataFrame([{"anchor_ref": "token\0funsd-d1:tesseract:00000"}])
        path = tmp_path / "sites.csv"
        frame.to_csv(path, index=False)
        back = pd.read_csv(path, keep_default_na=False)
        assert back.loc[0, "anchor_ref"] == "token"

    def test_float_columns_round_trip_bitwise(self, tmp_path) -> None:
        """DEFECT-3: the default parsers shift ~2% of floats by 1 ULP on read."""
        values = [i / 7.0 for i in range(500)] + [0.4892392158508301]
        frame = pd.DataFrame({"score": values})
        path = tmp_path / "scores.csv"
        frame.to_csv(path, index=False)
        back = read_frozen_csv(path)
        assert [float(v) for v in back["score"]] == values

    def test_the_python_engine_alone_mangles_a_float(self, tmp_path) -> None:
        """Red team: string safety alone is not float safety, hence the dual read."""
        values = [i / 7.0 for i in range(500)]
        path = tmp_path / "scores.csv"
        pd.DataFrame({"score": values}).to_csv(path, index=False)
        lossy = pd.read_csv(path, keep_default_na=False, engine="python")
        assert [float(v) for v in lossy["score"]] != values


class TestUnlabeledRows:
    def test_identity_drop_dedup_and_rank_are_hand_computed(self) -> None:
        site = _site()
        proposals = [
            ("g8_union", CandidateProposal(text="total", score=2.0), "substitution"),
            ("g8_union", CandidateProposal(text="total", score=1.0), "substitution"),  # dup text
            ("g8_union", CandidateProposal(text="t0tal", score=3.0), "substitution"),  # identity
            ("g8_union", CandidateProposal(text="tota1", score=1.5), "substitution"),
        ]
        rows = _unlabeled_rows(site, proposals, region_ocr="t0tal")
        # identity dropped, duplicate dropped by text, ranks are 0 and 1
        assert [row["candidate_text"] for row in rows] == ["total", "tota1"]
        assert [row["generator_rank"] for row in rows] == [0, 1]
        assert [row["generator_score"] for row in rows] == [2.0, 1.5]
        assert all("outcome" not in row and "region_gt" not in row for row in rows)


def _equal_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sites = pd.DataFrame(
        [
            {
                "site_id": "d1:tesseract:dsite:00000",
                "document_id": "d1",
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "anchor_kind": "token",
                "anchor_ref": "token\0sp0",
                "char_start": 0,
                "char_end": 5,
                "site_type": "substitution",
                "suspicion_score": 0.9,
                "provenance_reason": "low_confidence_token",
            }
        ]
    )
    # The live frame carries the evaluation-only column; the comparison must ignore it.
    live_sites = sites.copy()
    live_sites["covers_true_region"] = True
    candidates = pd.DataFrame(
        [
            {
                "site_id": "d1:tesseract:dsite:00000",
                "document_id": "d1",
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "generator_id": "g8_union",
                "candidate_text": "total",
                "operation": "substitution",
                "generator_rank": 0,
                "generator_score": 2.0,
                "region_ocr": "t0tal",
            }
        ]
    )
    live_candidates = candidates.copy()
    live_candidates["outcome"] = "true_correction"
    return live_sites, live_candidates, sites, candidates


class TestDeterminismAssertion:
    def test_equal_frames_pass_ignoring_evaluation_columns(self) -> None:
        live_sites, live_candidates, sites, candidates = _equal_frames()
        assert_frames_match_frozen(live_sites, live_candidates, sites, candidates)

    def test_a_tampered_site_coordinate_raises(self) -> None:
        live_sites, live_candidates, sites, candidates = _equal_frames()
        sites = sites.copy()
        sites.loc[0, "char_start"] = 3
        with pytest.raises(ConfirmatoryDivergenceError, match="diverged"):
            assert_frames_match_frozen(live_sites, live_candidates, sites, candidates)

    def test_a_tampered_candidate_text_raises(self) -> None:
        live_sites, live_candidates, sites, candidates = _equal_frames()
        candidates = candidates.copy()
        candidates.loc[0, "candidate_text"] = "totally"
        with pytest.raises(ConfirmatoryDivergenceError, match="diverged"):
            assert_frames_match_frozen(live_sites, live_candidates, sites, candidates)

    def test_a_missing_candidate_row_raises(self) -> None:
        live_sites, live_candidates, sites, candidates = _equal_frames()
        candidates = candidates.iloc[0:0]
        with pytest.raises(ConfirmatoryDivergenceError, match="diverged"):
            assert_frames_match_frozen(live_sites, live_candidates, sites, candidates)


def _fixture() -> dict[str, object]:
    """The two-page fixture from test_cgv3_track_a, reused for the equality proof."""
    spans: dict[tuple[str, str], list[CanonicalSpan]] = {
        ("d1", "tesseract"): [
            _span("s0", "t0tal", 0, reading_order=0, conf=0.10),
            _span("s1", "amount", 6, reading_order=1),
        ],
        ("d2", "tesseract"): [
            _span("t0", "total", 0, document_id="d2", reading_order=0),
            _span("t1", "12", 6, document_id="d2", reading_order=1),
        ],
    }
    easyocr: list[CanonicalSpan] = []
    position = 0
    for index, text in enumerate(["total", "of", "12", "total", "of", "12"]):
        easyocr.append(_span(f"e{index}", text, position, reading_order=index, engine_id="easyocr"))
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
    return {
        "spans": spans,
        "streams": streams,
        "alignments": alignments,
        "sites": sites,
        "gt_tokens": gt_tokens,
    }


class TestPasses:
    def test_generation_pass_reproduces_run_track_a(self) -> None:
        fixture = _fixture()
        spans = fixture["spans"]  # type: ignore[assignment]
        streams = fixture["streams"]  # type: ignore[assignment]
        fit_documents = frozenset({"d1"})
        fitted = fit_fold_resources(streams, fit_documents, ("tesseract",))
        # The pre-GT passes see only the tesseract pages (the fresh selection).
        fresh_spans = {pair: page for pair, page in spans.items() if pair[1] == "tesseract"}
        fresh_streams = {pair: streams[pair] for pair in fresh_spans}
        sites_table, empty = discovery_pass(fresh_spans, fitted, ("tesseract",))
        assert empty == []
        assert len(sites_table) >= 2  # the low-confidence token and the attested gap
        candidates_table = generation_pass(
            fresh_spans, fresh_streams, sites_table, fitted, ("tesseract",)
        )

        frames = run_track_a(
            spans_by_pair=spans,
            streams=streams,
            alignments=fixture["alignments"],  # type: ignore[arg-type]
            gt_tokens=fixture["gt_tokens"],  # type: ignore[arg-type]
            sites=fixture["sites"],  # type: ignore[arg-type]
            document_ids=frozenset({"d1", "d2"}),
            fit_documents=fit_documents,
            engines=("tesseract",),
        )
        # The load-bearing equality: the GT-free tables are what run_track_a scored.
        assert_frames_match_frozen(
            frames.discovered, frames.candidates, sites_table, candidates_table
        )

    def test_discovered_region_truth_computes_d_before_by_hand(self) -> None:
        fixture = _fixture()
        spans = fixture["spans"]  # type: ignore[assignment]
        streams = fixture["streams"]  # type: ignore[assignment]
        fresh_spans = {pair: page for pair, page in spans.items() if pair[1] == "tesseract"}
        fitted = fit_fold_resources(streams, frozenset({"d1"}), ("tesseract",))
        sites_table, _ = discovery_pass(fresh_spans, fitted, ("tesseract",))
        indexes = AlignmentIndex.from_frames(
            fixture["alignments"],  # type: ignore[arg-type]
            fixture["gt_tokens"],  # type: ignore[arg-type]
        )
        truth = discovered_region_truth(fresh_spans, streams, sites_table, indexes).set_index(
            "site_id"
        )
        token_site = truth.loc[next(i for i in truth.index if i.startswith("d1:"))]
        # The d1 token anchor covers "t0tal" against GT "total": one substitution
        # (0 -> o), so the raw-character distance is 1.
        assert token_site["region_ocr"] == "t0tal"
        assert token_site["region_gt"] == "total"
        assert token_site["d_before"] == 1
        gap_site = truth.loc[next(i for i in truth.index if i.startswith("d2:"))]
        # The d2 gap anchor between "total" and "12" carries the missing "of".
        assert gap_site["region_gt"] == "of"
        assert gap_site["d_before"] == 2
        assert bool(truth["index_present"].all())
