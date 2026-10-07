"""Hand-computed tests for the CGV3 offline evaluation: coverage, labels, flanks."""

from __future__ import annotations

import json

import pandas as pd

from ocr_risk.discovery.enumerator import (
    DiscoveredSite,
    DiscoveryResources,
    DiscoveryRules,
    enumerate_sites,
)
from ocr_risk.discovery.views import OcrPageView, OcrTokenView
from ocr_risk.experiments.cgv3_study import (
    AlignmentIndex,
    anchor_stream_text,
    candidates_from_proposals,
    coverage_table,
    true_region_rows,
)
from ocr_risk.schemas.enums import AnchorKind, DiscoveryProvenance


def _token(
    span_id: str, text: str, start: int, *, conf: float = 0.99, x0: float = 0.0
) -> OcrTokenView:
    return OcrTokenView(
        span_id=span_id,
        text=text,
        line_id="L1",
        char_start=start,
        char_end=start + len(text),
        normalized_confidence=conf,
        x0=x0,
        x1=x0 + 10.0 * len(text),
    )


def _page(*tokens: OcrTokenView) -> OcrPageView:
    return OcrPageView(document_id="d1", dataset_id="funsd", engine_id="tesseract", tokens=tokens)


def _alignments_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alignment_id": row["alignment_id"],
                "document_id": "d1",
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "ocr_span_ids": json.dumps(row.get("spans", [])),
                "gt_token_ids": json.dumps(row.get("tokens", [])),
                "relation": row.get("relation", "one_to_one"),
                "status": "resolved",
                "ocr_text": row.get("ocr_text", ""),
                "gt_text": row.get("gt_text", ""),
            }
            for row in rows
        ]
    )


def _gt_tokens_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"gt_token_id": f"t{i}", "document_id": "d1", "index": i, "text": f"w{i}"}
            for i in range(n)
        ]
    )


class TestAlignmentIndex:
    def test_components_are_ordered_by_gt_token_and_flanks_computed(self) -> None:
        alignments = _alignments_frame(
            [
                {
                    "alignment_id": "a0",
                    "spans": ["s0"],
                    "tokens": ["t0"],
                    "ocr_text": "TOTAL",
                    "gt_text": "total",
                },
                {
                    "alignment_id": "a1",
                    "spans": [],
                    "tokens": ["t1"],
                    "ocr_text": "",
                    "gt_text": "of",
                },
                {
                    "alignment_id": "a2",
                    "spans": ["s1"],
                    "tokens": ["t2"],
                    "ocr_text": "12",
                    "gt_text": "12",
                },
            ]
        )
        indexes = AlignmentIndex.from_frames(alignments, _gt_tokens_frame(3))
        index = indexes[("d1", "tesseract")]
        assert [c.alignment_id for c in index.components] == ["a0", "a1", "a2"]
        # The empty component's flanks are the surrounding OCR spans, in GT order.
        assert index.flanks_for_empty_component("a1") == ("s0", "s1")
        # Region truth and gap truth are offline GT couplings.
        assert index.region_truth(["s0"]) == ("TOTAL", "total")
        assert index.gap_truth("s0", "s1") == "of"
        assert index.gap_truth("s0", "s0") == ""

    def test_true_region_rows_derive_flanks_only_for_span_less_sites(self) -> None:
        alignments = _alignments_frame(
            [
                {
                    "alignment_id": "a0",
                    "spans": ["s0"],
                    "tokens": ["t0"],
                    "ocr_text": "TOTAL",
                    "gt_text": "total",
                },
                {
                    "alignment_id": "a1",
                    "spans": [],
                    "tokens": ["t1"],
                    "ocr_text": "",
                    "gt_text": "of",
                },
                {
                    "alignment_id": "a2",
                    "spans": ["s1"],
                    "tokens": ["t2"],
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
                    "alignment_ids": json.dumps(["a1"]),
                    "ocr_span_ids": json.dumps([]),
                    "gt_token_ids": json.dumps(["t1"]),
                    "ocr_text": "",
                    "gt_text": "of",
                    "d_before": 2,
                    "site_kind": "deletion",
                    "evaluable": True,
                    "char_start": 0,
                    "char_end": 0,
                },
                {
                    "site_id": "d1:tesseract:site:00002",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "tesseract",
                    "alignment_ids": json.dumps(["a0"]),
                    "ocr_span_ids": json.dumps(["s0"]),
                    "gt_token_ids": json.dumps(["t0"]),
                    "ocr_text": "TOTAL",
                    "gt_text": "total",
                    "d_before": 2,
                    "site_kind": "substitution",
                    "evaluable": True,
                    "char_start": 0,
                    "char_end": 5,
                },
            ]
        )
        rows = true_region_rows(sites, AlignmentIndex.from_frames(alignments, _gt_tokens_frame(3)))
        deletion = rows[rows["site_kind"] == "deletion"].iloc[0]
        assert deletion["flank_left_span"] == "s0"
        assert deletion["flank_right_span"] == "s1"
        substitution = rows[rows["site_kind"] == "substitution"].iloc[0]
        assert json.loads(substitution["span_ids"]) == ["s0"]


class TestCoverage:
    def _site(self, kind: AnchorKind, *spans: str, start: int = 0, end: int = 1) -> DiscoveredSite:
        return DiscoveredSite(
            site_id=f"d1:tesseract:dsite:{kind.value}:{'-'.join(spans)}",
            document_id="d1",
            dataset_id="funsd",
            engine_id="tesseract",
            anchor_kind=kind,
            anchor_ref="\0".join((kind.value, *spans)),
            char_start=start,
            char_end=end,
            site_type="insertion" if kind is AnchorKind.GAP else "substitution",
            suspicion_score=0.5,
            provenance_reason=DiscoveryProvenance.GAP_ANOMALY
            if kind is AnchorKind.GAP
            else DiscoveryProvenance.LEXICAL_ANOMALY,
            signals={},
        )

    def _true(self, site_id: str, kind: str, spans: list[str], flanks: tuple) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "true_site_id": site_id,
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "tesseract",
                    "site_kind": kind,
                    "d_before": 1,
                    "span_ids": json.dumps(spans),
                    "flank_left_span": flanks[0],
                    "flank_right_span": flanks[1],
                    "char_start": 0,
                    "char_end": 5,
                    "gt_text": "x",
                    "ocr_text": "y",
                }
            ]
        )

    def test_token_anchor_covers_single_span_region(self) -> None:
        true = self._true("r1", "substitution", ["s0"], (None, None))
        coverage, discovered = coverage_table(
            [self._site(AnchorKind.TOKEN, "s0", start=0, end=5)], true
        )
        assert coverage["disposition"].iloc[0] == "discovered"
        assert bool(discovered["covers_true_region"].iloc[0])

    def test_gap_anchor_covers_deletion_region_via_flanks(self) -> None:
        true = self._true("r1", "deletion", [], ("s0", "s1"))
        coverage, _ = coverage_table([self._site(AnchorKind.GAP, "s0", "s1", start=5, end=6)], true)
        assert coverage["disposition"].iloc[0] == "discovered"

    def test_pair_anchor_covers_a_two_span_merge_region(self) -> None:
        true = self._true("r1", "segmentation", ["s0", "s1"], (None, None))
        coverage, _ = coverage_table(
            [self._site(AnchorKind.TOKEN_PAIR, "s0", "s1", start=0, end=11)], true
        )
        assert coverage["disposition"].iloc[0] == "discovered"

    def test_a_gap_anchor_never_covers_a_two_span_region(self) -> None:
        # Amendment A1: an insertion hypothesis cannot host a merge/pair-substitution
        # repair, so a gap anchor sitting between a region's two spans is a boundary
        # mismatch, not a discovery -- even though the anchor's span set matches.
        true = self._true("r1", "segmentation", ["s0", "s1"], (None, None))
        coverage, discovered = coverage_table(
            [self._site(AnchorKind.GAP, "s0", "s1", start=5, end=6)], true
        )
        assert coverage["disposition"].iloc[0] == "boundary_mismatch"
        assert not bool(discovered["covers_true_region"].iloc[0])

    def test_a_token_pair_never_covers_a_flanked_deletion_region(self) -> None:
        true = self._true("r1", "deletion", [], ("s0", "s1"))
        coverage, _ = coverage_table(
            [self._site(AnchorKind.TOKEN_PAIR, "s0", "s1", start=0, end=11)], true
        )
        assert coverage["disposition"].iloc[0] != "discovered"

    def test_an_overlapping_wrong_anchor_is_a_boundary_mismatch_not_a_match(self) -> None:
        true = self._true("r1", "segmentation", ["s0", "s1"], (None, None))
        coverage, discovered = coverage_table(
            [self._site(AnchorKind.TOKEN, "s0", start=0, end=5)], true
        )
        assert coverage["disposition"].iloc[0] == "boundary_mismatch"
        assert not bool(discovered["covers_true_region"].iloc[0])

    def test_audit_states_gate_eligibility(self) -> None:
        true = self._true("r1", "substitution", ["s0"], (None, None))
        states = pd.DataFrame([{"site_id": "r1", "state": "ambiguous"}])
        coverage, _ = coverage_table(
            [self._site(AnchorKind.TOKEN, "s0", start=0, end=5)], true, states
        )
        assert coverage["disposition"].iloc[0] == "excluded_ambiguous"


class TestCandidateLabelling:
    def test_labels_are_recomputed_from_strings_and_identity_is_dropped(self) -> None:
        from ocr_risk.candidates.base import CandidateProposal

        page = _page(_token("s0", "recieve", 0), _token("s1", "total", 8))
        site = DiscoveredSite(
            site_id="d1:tesseract:dsite:00000",
            document_id="d1",
            dataset_id="funsd",
            engine_id="tesseract",
            anchor_kind=AnchorKind.TOKEN,
            anchor_ref="\0".join((AnchorKind.TOKEN.value, "s0")),
            char_start=0,
            char_end=7,
            site_type="substitution",
            suspicion_score=0.6,
            provenance_reason=DiscoveryProvenance.LEXICAL_ANOMALY,
            signals={},
        )
        rows = candidates_from_proposals(
            site,
            page,
            "recieve",
            "receive",
            [
                ("g3_edit_aware", CandidateProposal(text="receive"), "substitution"),
                ("g3_edit_aware", CandidateProposal(text="recieve"), "substitution"),
                ("g7_structural_v2", CandidateProposal(text="receipt"), "substitution"),
            ],
        )
        assert [row.candidate_text for row in rows] == ["receive", "receipt"]
        assert rows[0].outcome.value == "true_correction"
        assert rows[0].d_before == 2 and rows[0].d_after == 0
        # "receipt" is edit-distance-equal to the original relative to "receive":
        # a lateral change, correctly not counted as beneficial.
        assert rows[1].outcome.value == "lateral_change"

    def test_anchor_stream_text_slices_the_canonical_stream(self) -> None:
        site = DiscoveredSite(
            site_id="x",
            document_id="d1",
            dataset_id="funsd",
            engine_id="tesseract",
            anchor_kind=AnchorKind.GAP,
            anchor_ref="\0".join((AnchorKind.GAP.value, "s0", "s1")),
            char_start=5,
            char_end=6,
            site_type="insertion",
            suspicion_score=0.5,
            provenance_reason=DiscoveryProvenance.GAP_ANOMALY,
            signals={},
        )
        assert anchor_stream_text("TOTAL 12 dollars", site) == " "


def test_end_to_end_discovery_through_enumeration_and_coverage() -> None:
    """The full offline loop on one page: enumerate, build true regions, cover."""
    alignments = _alignments_frame(
        [
            {
                "alignment_id": "a0",
                "spans": ["s0"],
                "tokens": ["t0"],
                "ocr_text": "TOTAL",
                "gt_text": "total",
            },
            {"alignment_id": "a1", "spans": [], "tokens": ["t1"], "ocr_text": "", "gt_text": "of"},
            {
                "alignment_id": "a2",
                "spans": ["s1"],
                "tokens": ["t2"],
                "ocr_text": "12",
                "gt_text": "12",
            },
        ]
    )
    sites = pd.DataFrame(
        [
            {
                "site_id": "r-missing",
                "document_id": "d1",
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "alignment_ids": json.dumps(["a1"]),
                "ocr_span_ids": json.dumps([]),
                "gt_token_ids": json.dumps(["t1"]),
                "ocr_text": "",
                "gt_text": "of",
                "d_before": 2,
                "site_kind": "deletion",
                "evaluable": True,
                "char_start": 0,
                "char_end": 0,
            }
        ]
    )
    page = _page(_token("s0", "TOTAL", 0), _token("s1", "12", 6))
    discovered = enumerate_sites(
        page,
        DiscoveryResources(
            lexicon=frozenset({"total", "12", "of"}),
            bigrams=__import__("collections").Counter({("total", "of"): 3, ("of", "12"): 3}),
        ),
        DiscoveryRules(),
    )
    gaps = [site for site in discovered if site.anchor_kind is AnchorKind.GAP]
    assert len(gaps) == 1, "the attested missing 'of' must be discoverable without GT"
    true = true_region_rows(sites, AlignmentIndex.from_frames(alignments, _gt_tokens_frame(3)))
    coverage, _ = coverage_table(discovered, true)
    assert coverage["disposition"].iloc[0] == "discovered"


def test_track_a_metric_tables_count_lowercase_outcomes() -> None:
    """R-82-class guard: outcome enum values are lowercase, and the Track-A metric
    tables must count beneficial/harmful rows correctly against them."""
    import pandas as pd

    from ocr_risk.experiments.cgv3_track_a import TrackAFrames, _harm_table

    frames = TrackAFrames(
        candidates=pd.DataFrame(
            [
                {
                    "generator_id": "g8_union",
                    "engine_id": "e1",
                    "outcome": "true_correction",
                },
                {
                    "generator_id": "g8_union",
                    "engine_id": "e1",
                    "outcome": "miscorrection",
                },
                {
                    "generator_id": "g8_union",
                    "engine_id": "e1",
                    "outcome": "lateral_change",
                },
            ]
        )
    )
    harm = _harm_table(frames)
    row = harm[(harm["generator_id"] == "g8_union")].iloc[0]
    assert row["n_beneficial"] == 1
    assert row["n_harmful"] == 1
    assert row["n_neutral"] == 1


def test_empty_anchor_unresolved_regions_remain_eligible_for_gap_coverage() -> None:
    """Protocol section 8 defines gap-anchor coverage for insertion-kind regions; the
    CGV2 empty-anchor exclusion marked the OCR anchor missing -- CGV3's gap anchor
    exists to satisfy exactly that -- so those regions stay in the denominator."""
    from ocr_risk.discovery.enumerator import DiscoveredSite
    from ocr_risk.experiments.cgv3_study import coverage_table

    def _gap_site() -> DiscoveredSite:
        return DiscoveredSite(
            site_id="d1:tesseract:dsite:00000",
            document_id="d1",
            dataset_id="funsd",
            engine_id="tesseract",
            anchor_kind=AnchorKind.GAP,
            anchor_ref="\0".join((AnchorKind.GAP.value, "s0", "s1")),
            char_start=5,
            char_end=6,
            site_type="insertion",
            suspicion_score=0.5,
            provenance_reason=DiscoveryProvenance.GAP_ANOMALY,
            signals={},
        )

    true = pd.DataFrame(
        [
            {
                "true_site_id": "r1",
                "document_id": "d1",
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "site_kind": "deletion",
                "d_before": 2,
                "span_ids": "[]",
                "flank_left_span": "s0",
                "flank_right_span": "s1",
                "char_start": 0,
                "char_end": 5,
                "gt_text": "of",
                "ocr_text": "",
            }
        ]
    )
    empty_anchor_state = pd.DataFrame(
        [{"site_id": "r1", "state": "unresolved", "reason": "empty_insertion_anchor"}]
    )
    coverage, _ = coverage_table([_gap_site()], true, empty_anchor_state)
    assert coverage["disposition"].iloc[0] == "discovered"
    ambiguous_state = pd.DataFrame([{"site_id": "r1", "state": "ambiguous", "reason": "x"}])
    coverage, _ = coverage_table([_gap_site()], true, ambiguous_state)
    assert coverage["disposition"].iloc[0] == "excluded_ambiguous"
    other_unresolved = pd.DataFrame(
        [{"site_id": "r1", "state": "unresolved", "reason": "missing_stream"}]
    )
    coverage, _ = coverage_table([_gap_site()], true, other_unresolved)
    assert coverage["disposition"].iloc[0] == "excluded_unresolved"


class TestAmendmentA5:
    """Region-unit conditionals and the span-less near-miss rule, pinned by data."""

    def test_conditional_metrics_count_regions_not_sites(self) -> None:
        # One gap anchor covering two consecutive missing tokens is one site but two
        # regions; the conditional table must count regions (amendment A5), and the
        # H4 factorization site_recall x conditional = end-to-end must hold exactly.
        from ocr_risk.experiments.cgv3_track_a import TrackAFrames, _conditional_metrics

        frames = TrackAFrames(
            true_coverage=pd.DataFrame(
                [
                    {
                        "true_site_id": "r1",
                        "document_id": "d1",
                        "dataset_id": "funsd",
                        "engine_id": "tesseract",
                        "site_kind": "deletion",
                        "d_before": 1,
                        "audit_state": "eligible",
                        "disposition": "discovered",
                        "matched_site_id": "site-A",
                    },
                    {
                        "true_site_id": "r2",
                        "document_id": "d1",
                        "dataset_id": "funsd",
                        "engine_id": "tesseract",
                        "site_kind": "deletion",
                        "d_before": 1,
                        "audit_state": "eligible",
                        "disposition": "discovered",
                        "matched_site_id": "site-A",
                    },
                    {
                        "true_site_id": "r3",
                        "document_id": "d1",
                        "dataset_id": "funsd",
                        "engine_id": "tesseract",
                        "site_kind": "deletion",
                        "d_before": 1,
                        "audit_state": "eligible",
                        "disposition": "not_discovered",
                        "matched_site_id": "",
                    },
                ]
            ),
            candidates=pd.DataFrame(
                [
                    {
                        "site_id": "site-A",
                        "generator_id": "g8_union",
                        "generator_rank": 0,
                        "engine_id": "tesseract",
                        "outcome": "true_correction",
                    }
                ]
            ),
        )
        conditional, end_to_end = _conditional_metrics(frames)
        row = conditional[conditional["generator_id"] == "g8_union"].iloc[0]
        assert row["n_covered_regions"] == 2
        assert row["n_regions_with_beneficial"] == 2  # both regions share site-A
        assert row["beneficial_recall_given_discovery"] == 1.0
        e2e = end_to_end[end_to_end["generator_id"] == "g8_union"].iloc[0]
        assert e2e["n_end_to_end_beneficial"] == 2
        site_recall = 2 / 3
        assert (
            abs(
                site_recall * row["beneficial_recall_given_discovery"]
                - e2e["end_to_end_beneficial_opportunity"]
            )
            < 1e-12
        )

    def test_spanless_regions_need_a_flank_sharing_gap_for_boundary_mismatch(self) -> None:
        # The CGV2 [0, 0] char range made the old overlap test degenerate: any site at
        # stream offset 0 counted as a near-miss. A gap anchor elsewhere on the page,
        # sharing no flank with the region, is not a near-miss -- the region is
        # honestly not_discovered even when that anchor sits at offset 0.
        def _gap(first: str, second: str, start: int = 0) -> DiscoveredSite:
            return DiscoveredSite(
                site_id=f"d1:tesseract:dsite:{first}-{second}",
                document_id="d1",
                dataset_id="funsd",
                engine_id="tesseract",
                anchor_kind=AnchorKind.GAP,
                anchor_ref="\0".join((AnchorKind.GAP.value, first, second)),
                char_start=start,
                char_end=start + 1,
                site_type="insertion",
                suspicion_score=0.5,
                provenance_reason=DiscoveryProvenance.GAP_ANOMALY,
                signals={},
            )

        true = pd.DataFrame(
            [
                {
                    "true_site_id": "r1",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "tesseract",
                    "site_kind": "deletion",
                    "d_before": 2,
                    "span_ids": "[]",
                    "flank_left_span": "s0",
                    "flank_right_span": "s2",
                    "char_start": 0,
                    "char_end": 0,
                    "gt_text": "of",
                    "ocr_text": "",
                }
            ]
        )
        coverage, _ = coverage_table([_gap("s5", "s6", start=0)], true)
        assert coverage["disposition"].iloc[0] == "not_discovered"
        coverage, _ = coverage_table([_gap("s0", "s1", start=0)], true)
        assert coverage["disposition"].iloc[0] == "boundary_mismatch"
        coverage, _ = coverage_table([_gap("s2", "s3", start=7)], true)
        assert coverage["disposition"].iloc[0] == "boundary_mismatch"
