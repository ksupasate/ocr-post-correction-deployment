"""Hand-computed tests for the CGV2 study layer: regions, certificates, contrasts."""

from __future__ import annotations

import pytest

from ocr_risk.experiments.cgv2_study import (
    RegionRecord,
    SiteCensusRecord,
    augmented_oracle,
    build_regions,
    engine_contrast,
)
from ocr_risk.experiments.generator_study import ProposalRecord
from ocr_risk.schemas.enums import HarmPolicy, OutcomeIfAccepted


def _census(
    site_id: str,
    *,
    document_id: str = "doc-1",
    engine_id: str = "engine_a",
    kind: str = "substitution",
    d_before: int = 1,
    gt_text: str = "word",
    char_start: int,
    ocr_text: str = "wurd",
    n_spans: int = 1,
) -> SiteCensusRecord:
    return SiteCensusRecord(
        site_id=site_id,
        document_id=document_id,
        dataset_id="funsd",
        engine_id=engine_id,
        site_kind=kind,
        d_before=d_before,
        gt_text=gt_text,
        char_start=char_start,
        char_end=char_start + len(ocr_text),
        n_spans=n_spans,
        ocr_text=ocr_text,
        min_align_confidence=1.0,
    )


def _streams_for(*records: SiteCensusRecord, fill: str = " ") -> dict[tuple[str, str], str]:
    """The exact linearized stream a fixture's offsets describe: texts placed at their
    char offsets in a buffer of separator characters."""
    by_pair: dict[tuple[str, str], list[SiteCensusRecord]] = {}
    for record in records:
        by_pair.setdefault((record.document_id, record.engine_id), []).append(record)
    streams: dict[tuple[str, str], str] = {}
    for key, group in by_pair.items():
        length = max(record.char_end for record in group)
        buffer = [fill] * length
        for record in group:
            for offset, char in enumerate(record.ocr_text):
                buffer[record.char_start + offset] = char
        streams[key] = "".join(buffer)
    return streams


class TestBuildRegions:
    def test_adjacent_pair_with_single_space_gap_is_eligible(self) -> None:
        left = _census(
            "s0", char_start=0, ocr_text="data", gt_text="data", d_before=0, kind="clean"
        )
        right = _census(
            "s1", char_start=5, ocr_text="base", gt_text="base", d_before=0, kind="clean"
        )
        streams = _streams_for(left, right)
        regions, counts = build_regions([left, right], streams)
        assert counts["eligible"] == 1
        assert sum(counts.values()) == 1
        assert len(regions) == 1
        region = regions[0]
        assert region.region_ocr == "data base"
        assert region.region_gt == "data base"
        assert region.gap_chars == 1
        assert (region.left.site_id, region.right.site_id) == ("s0", "s1")

    def test_cross_line_pair_is_excluded_not_joined(self) -> None:
        """The stream separator between lines is a newline; a word merge cannot cross it."""
        left = _census("s0", char_start=0, ocr_text="data")
        right = _census("s1", char_start=5, ocr_text="base")
        streams = {(left.document_id, left.engine_id): "data\nbase"}
        regions, counts = build_regions([left, right], streams)
        assert regions == []
        assert counts["cross_line"] == 1

    def test_intervening_excluded_site_text_is_not_a_gap(self) -> None:
        """A 1-char non-evaluable site between two evaluable ones is real text, and the
        exact slice says so -- offsets alone called this a 3-space gap."""
        left = _census("s0", char_start=0, ocr_text="data")
        right = _census("s1", char_start=7, ocr_text="base")
        streams = {(left.document_id, left.engine_id): "data X base"}
        regions, counts = build_regions([left, right], streams)
        assert regions == []
        assert counts["intervening_text"] == 1

    def test_offset_stream_mismatch_is_named(self) -> None:
        left = _census("s0", char_start=0, ocr_text="data")
        right = _census("s1", char_start=5, ocr_text="base")
        streams = {(left.document_id, left.engine_id): "datA base"}
        regions, counts = build_regions([left, right], streams)
        assert regions == []
        assert counts["text_mismatch"] == 1

    def test_overlapping_ranges_are_counted_not_crashed(self) -> None:
        left = _census("s0", char_start=0, ocr_text="abcdef")
        right = _census("s1", char_start=3, ocr_text="xyzw")
        streams = _streams_for(left, right)
        regions, counts = build_regions([left, right], streams)
        assert regions == []
        assert counts["non_adjacent"] == 1

    def test_empty_ocr_member_is_degenerate(self) -> None:
        left = _census("s0", char_start=0, ocr_text="data")
        right = _census("s1", char_start=5, ocr_text="", kind="deletion", gt_text="missing")
        streams = _streams_for(left, right)
        regions, counts = build_regions([left, right], streams)
        assert regions == []
        assert counts["degenerate"] == 1

    def test_missing_stream_yields_no_regions(self) -> None:
        left = _census("s0", char_start=0, ocr_text="data")
        right = _census("s1", char_start=5, ocr_text="base")
        regions, counts = build_regions([left, right], None)
        assert regions == []
        assert counts["no_stream"] == 1

    def test_regions_never_cross_documents_or_engines(self) -> None:
        a = _census("a0", document_id="doc-a", char_start=0, ocr_text="data")
        b = _census("b0", document_id="doc-b", char_start=5, ocr_text="base")
        regions, _ = build_regions([a, b], _streams_for(a, b))
        assert regions == []

    def test_three_sites_yield_two_ordered_regions(self) -> None:
        sites = [
            _census("s0", char_start=0, ocr_text="aaaa"),
            _census("s1", char_start=5, ocr_text="bbbb"),
            _census("s2", char_start=10, ocr_text="cccc"),
        ]
        regions, counts = build_regions(sites, _streams_for(*sites))
        assert counts["eligible"] == 2
        assert [(r.left.site_id, r.right.site_id) for r in regions] == [("s0", "s1"), ("s1", "s2")]
        assert regions[1].region_ocr == "bbbb cccc"


def _proposal(
    generator_id: str,
    site_id: str,
    *,
    outcome: OutcomeIfAccepted = OutcomeIfAccepted.TRUE_CORRECTION,
    gt_text: str = "word",
    d_before: int = 1,
    delta: int = 1,
    engine_id: str = "engine_a",
    document_id: str = "doc-1",
    site_kind: str = "substitution",
    pool: str = "natural",
) -> ProposalRecord:
    return ProposalRecord(
        generator_id=generator_id,
        site_id=site_id,
        document_id=document_id,
        dataset_id="funsd",
        engine_id=engine_id,
        role="evaluate",
        original_ocr="wurd",
        candidate_text="word",
        gt_text=gt_text,
        d_before=d_before,
        d_after=d_before - delta,
        delta=delta,
        outcome=outcome,
        site_kind=site_kind,
        normalized_confidence=0.5,
        pool=pool,
    )


def _region_record(
    region_id: str,
    left: str,
    right: str,
    *,
    outcome: OutcomeIfAccepted = OutcomeIfAccepted.TRUE_CORRECTION,
    left_kind: str = "substitution",
    right_kind: str = "substitution",
    region_gt: str = "total amount",
    d_before: int = 2,
    delta: int = 2,
    engine_id: str = "engine_a",
) -> RegionRecord:
    return RegionRecord(
        generator_id="g5_structural",
        region_id=region_id,
        document_id="doc-1",
        dataset_id="funsd",
        engine_id=engine_id,
        role="evaluate",
        left_site_id=left,
        right_site_id=right,
        gap_chars=1,
        region_char_start=0,
        region_char_end=9,
        left_kind=left_kind,
        right_kind=right_kind,
        region_ocr="totol amunt",
        region_gt=region_gt,
        candidate_text="total amount",
        d_before=d_before,
        d_after=d_before - delta,
        delta=delta,
        outcome=outcome,
        generator_rank=0,
    )


class TestAugmentedOracle:
    def test_region_acceptance_covers_both_member_sites(self) -> None:
        census = [
            _census("s0", char_start=0, ocr_text="totol"),
            _census("s1", char_start=6, ocr_text="amunt"),
            _census("s2", char_start=12, ocr_text="other"),
        ]
        proposals = [_proposal("g6_union", "s2", outcome=OutcomeIfAccepted.LATERAL_CHANGE)]
        regions = [_region_record("r0", "s0", "s1")]
        result = augmented_oracle(
            census, proposals, regions, "g6_union", HarmPolicy.STRICT_WORSENING
        )
        assert result["engine_a"]["n_evaluable_sites"] == 3
        assert result["engine_a"]["oracle_accepted_edits"] == 2  # s0 + s1 via the region
        assert result["engine_a"]["oracle_safe_coverage"] == 2 / 3

    def test_site_and_region_acceptance_do_not_double_count(self) -> None:
        census = [
            _census("s0", char_start=0, ocr_text="totol"),
            _census("s1", char_start=6, ocr_text="amunt"),
        ]
        proposals = [_proposal("g6_union", "s0")]
        regions = [_region_record("r0", "s0", "s1")]
        result = augmented_oracle(
            census, proposals, regions, "g6_union", HarmPolicy.STRICT_WORSENING
        )
        assert result["engine_a"]["oracle_accepted_edits"] == 2  # both sites, counted once

    def test_insertion_member_region_needs_an_exact_repair(self) -> None:
        """A1/A6: a merged hallucination against a shorter reference is not a repair."""
        census = [
            _census("s0", char_start=0, ocr_text="data"),
            _census("s1", char_start=5, ocr_text="base", gt_text="", d_before=4, kind="insertion"),
        ]
        # region_ocr "data base", region_gt "data": d_before 5, candidate "database" ->
        # d_after 4, delta 1 -- a partial improvement that must NOT count.
        region = _region_record(
            "r0",
            "s0",
            "s1",
            outcome=OutcomeIfAccepted.PARTIAL_IMPROVEMENT,
            right_kind="insertion",
            region_gt="data",
            d_before=5,
            delta=1,
        )
        result = augmented_oracle(census, [], [region], "g6_union", HarmPolicy.STRICT_WORSENING)
        assert result["engine_a"]["oracle_accepted_edits"] == 0

    def test_challenge_candidate_cannot_enter_the_oracle(self) -> None:
        census = [_census("s0", char_start=0)]
        with pytest.raises(ValueError, match="natural pool"):
            augmented_oracle(
                census,
                [_proposal("g6_union", "s0", pool="challenge")],
                [],
                "g6_union",
                HarmPolicy.STRICT_WORSENING,
            )


class TestEngineContrast:
    def test_delta_is_hand_computed_per_engine(self) -> None:
        census = [
            _census("s0", document_id="d1", char_start=0),
            _census("s1", document_id="d1", char_start=8),
            _census("s2", document_id="d2", char_start=0),
        ]
        proposals = [
            _proposal("g6_union", "s0", document_id="d1"),
            _proposal("g6_union", "s2", document_id="d2"),
            _proposal("g3_edit_aware", "s0", document_id="d1"),
        ]
        contrasts = engine_contrast(
            census,
            proposals,
            "g6_union",
            "g3_edit_aware",
            HarmPolicy.STRICT_WORSENING,
            n_resamples=200,
        )
        assert len(contrasts) == 1
        contrast = contrasts[0]
        assert contrast.challenger_value == 2 / 3
        assert contrast.baseline_value == 1 / 3
        assert contrast.delta == pytest.approx(1 / 3)
        assert contrast.n_sites == 3
        assert contrast.n_documents == 2

    def test_stratum_restriction_limits_the_denominator(self) -> None:
        census = [
            _census("s0", document_id="d1", char_start=0, kind="segmentation"),
            _census("s1", document_id="d1", char_start=8, kind="substitution"),
        ]
        proposals = [_proposal("g6_union", "s0", document_id="d1", site_kind="segmentation")]
        contrasts = engine_contrast(
            census,
            proposals,
            "g6_union",
            "g3_edit_aware",
            HarmPolicy.STRICT_WORSENING,
            kinds=frozenset({"segmentation"}),
            n_resamples=100,
        )
        assert contrasts[0].n_sites == 1
        assert contrasts[0].challenger_value == 1.0

    def test_clean_sites_are_excluded_from_the_primary_metric(self) -> None:
        census = [
            _census("s0", document_id="d1", char_start=0, d_before=1),
            _census("s1", document_id="d1", char_start=8, d_before=0, kind="clean"),
        ]
        contrasts = engine_contrast(
            census, [], "g6_union", "g3_edit_aware", HarmPolicy.STRICT_WORSENING, n_resamples=100
        )
        assert contrasts[0].n_sites == 1
