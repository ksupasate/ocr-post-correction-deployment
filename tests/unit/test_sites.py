"""Correction sites and edit application."""

from __future__ import annotations

import pytest

from ocr_risk.config.models import SiteConfig
from ocr_risk.edits import AppliedEdit, apply_edits, build_sites, site_kind_for
from ocr_risk.schemas.alignment import AlignmentRecord
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.enums import AlignmentRelation, AlignmentStatus, SiteKind
from ocr_risk.schemas.spans import CanonicalSpan


def _span(index: int, text: str, start: int) -> CanonicalSpan:
    return CanonicalSpan(
        span_id=f"d:e:{index:05d}",
        document_id="d",
        dataset_id="ds",
        engine_id="e",
        engine_fingerprint="f" * 64,
        text=text,
        reading_order=index,
        bbox=BBox(x0=float(start), y0=0.0, x1=float(start + 10 * len(text)), y1=20.0),
        char_start=start,
        char_end=start + len(text),
        raw_ref="raw.json",
        raw_index=index,
    )


def _alignment(
    index: int,
    ocr_text: str,
    gt_text: str,
    relation: AlignmentRelation = AlignmentRelation.ONE_TO_ONE,
    status: AlignmentStatus = AlignmentStatus.RESOLVED,
    span_ids: tuple[str, ...] = (),
    token_ids: tuple[str, ...] = (),
) -> AlignmentRecord:
    return AlignmentRecord(
        alignment_id=f"d:e:al:{index:05d}",
        document_id="d",
        dataset_id="ds",
        engine_id="e",
        ocr_span_ids=span_ids or (f"d:e:{index:05d}",),
        gt_token_ids=token_ids or (f"d:gt:{index:04d}",),
        relation=relation,
        status=status,
        ocr_text=ocr_text,
        gt_text=gt_text,
        align_confidence=0.9,
        char_agreement=0.9,
        uniqueness_margin=0.9,
        edit_distance=abs(len(ocr_text) - len(gt_text)),
    )


# --- site kinds ---------------------------------------------------------------------------
def test_site_kind_classification() -> None:
    assert site_kind_for(AlignmentRelation.ONE_TO_ONE, 0, True) is SiteKind.CLEAN
    assert site_kind_for(AlignmentRelation.ONE_TO_ONE, 2, True) is SiteKind.SUBSTITUTION
    assert site_kind_for(AlignmentRelation.SPLIT, 2, True) is SiteKind.SEGMENTATION
    assert site_kind_for(AlignmentRelation.MERGE, 2, True) is SiteKind.SEGMENTATION
    assert site_kind_for(AlignmentRelation.OCR_INSERTION, 3, True) is SiteKind.INSERTION
    assert site_kind_for(AlignmentRelation.OCR_DELETION, 3, True) is SiteKind.DELETION


def test_non_evaluable_overrides_every_other_kind() -> None:
    assert site_kind_for(AlignmentRelation.ONE_TO_ONE, 0, False) is SiteKind.EXCLUDED


# --- site construction --------------------------------------------------------------------
def test_clean_sites_are_enumerated_by_default() -> None:
    """Without clean sites the benchmark cannot observe overcorrection at all, which
    would let a reckless method look flawless."""
    spans = [_span(0, "mg", 0)]
    sites = build_sites([_alignment(0, "mg", "mg")], spans, SiteConfig())
    assert len(sites) == 1
    assert sites[0].site_kind is SiteKind.CLEAN
    assert sites[0].d_before == 0


def test_clean_sites_can_be_excluded_for_a_diagnostic() -> None:
    spans = [_span(0, "mg", 0)]
    sites = build_sites([_alignment(0, "mg", "mg")], spans, SiteConfig(include_clean_sites=False))
    assert sites == []


def test_d_before_is_a_raw_character_distance() -> None:
    spans = [_span(0, "rng", 0)]
    sites = build_sites([_alignment(0, "rng", "mg")], spans, SiteConfig())
    assert sites[0].d_before == 2


def test_ambiguous_alignment_yields_a_non_evaluable_site() -> None:
    """Carried so its volume is reportable; excluded from every metric."""
    spans = [_span(0, "zzz", 0)]
    sites = build_sites(
        [_alignment(0, "zzz", "mg", status=AlignmentStatus.AMBIGUOUS)], spans, SiteConfig()
    )
    assert len(sites) == 1
    assert not sites[0].evaluable
    assert sites[0].site_kind is SiteKind.EXCLUDED


def test_oversized_site_is_carried_but_not_evaluated() -> None:
    """An over-long site is usually an alignment failure wearing a site's clothes."""
    long_text = "x" * 200
    spans = [_span(0, long_text, 0)]
    sites = build_sites([_alignment(0, long_text, long_text)], spans, SiteConfig(max_site_chars=64))
    assert len(sites) == 1
    assert not sites[0].evaluable


def test_split_components_merge_into_one_site() -> None:
    """'Sm' + 'ith' -> 'Smith' is a single replacement, not two decisions. Emitting two
    overlapping sites would double-count both the opportunity and the harm."""
    spans = [_span(0, "Sm", 0), _span(1, "ith", 3)]
    alignments = [
        _alignment(
            0,
            "Sm",
            "Smith",
            AlignmentRelation.SPLIT,
            span_ids=("d:e:00000",),
            token_ids=("d:gt:0000",),
        ),
        _alignment(
            1,
            "ith",
            "Smith",
            AlignmentRelation.SPLIT,
            span_ids=("d:e:00001",),
            token_ids=("d:gt:0000",),
        ),
    ]
    sites = build_sites(alignments, spans, SiteConfig(merge_adjacent_components=True))
    assert len(sites) == 1
    assert len(sites[0].alignment_ids) == 2
    assert sites[0].site_kind is SiteKind.SEGMENTATION


def test_unrelated_neighbours_stay_separate() -> None:
    """Merging clean neighbours would fabricate a larger edit than any generator proposes."""
    spans = [_span(0, "alpha", 0), _span(1, "beta", 6)]
    alignments = [_alignment(0, "alpha", "alpha"), _alignment(1, "beta", "beta")]
    sites = build_sites(alignments, spans, SiteConfig(merge_adjacent_components=True))
    assert len(sites) == 2


def test_site_carries_the_weakest_alignment_confidence() -> None:
    spans = [_span(0, "a", 0), _span(1, "b", 2)]
    weak = _alignment(0, "a", "ab", AlignmentRelation.SPLIT, token_ids=("d:gt:0000",)).model_copy(
        update={"align_confidence": 0.42}
    )
    strong = _alignment(1, "b", "ab", AlignmentRelation.SPLIT, token_ids=("d:gt:0000",)).model_copy(
        update={"align_confidence": 0.95}
    )
    sites = build_sites([weak, strong], spans, SiteConfig())
    assert sites[0].min_align_confidence == pytest.approx(0.42)


def test_no_alignments_yields_no_sites() -> None:
    assert build_sites([], [], SiteConfig()) == []


# --- applying edits ------------------------------------------------------------------------
def test_apply_single_edit() -> None:
    assert apply_edits("Dose: rng", [AppliedEdit(6, 9, "mg")]) == "Dose: mg"


def test_apply_multiple_edits_keeps_offsets_valid() -> None:
    """Applied right to left, so earlier replacements do not shift later offsets."""
    text = "aa bb cc"
    edits = [AppliedEdit(0, 2, "XXXX"), AppliedEdit(3, 5, "Y"), AppliedEdit(6, 8, "ZZZ")]
    assert apply_edits(text, edits) == "XXXX Y ZZZ"


def test_apply_edits_is_order_independent() -> None:
    text = "aa bb cc"
    edits = [AppliedEdit(6, 8, "ZZZ"), AppliedEdit(0, 2, "XXXX"), AppliedEdit(3, 5, "Y")]
    assert apply_edits(text, edits) == "XXXX Y ZZZ"


def test_identity_edit_is_a_no_op() -> None:
    assert apply_edits("hello", [AppliedEdit(0, 5, "hello")]) == "hello"


def test_no_edits_returns_the_original() -> None:
    assert apply_edits("unchanged", []) == "unchanged"


def test_overlapping_edits_are_rejected() -> None:
    """Sites do not overlap and at most one candidate is accepted per site, so an overlap
    is a bug — resolving it by precedence would hide that."""
    with pytest.raises(ValueError, match="overlapping edits"):
        apply_edits("abcdef", [AppliedEdit(0, 3, "X"), AppliedEdit(2, 5, "Y")])


def test_out_of_range_edit_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside a text"):
        apply_edits("short", [AppliedEdit(0, 99, "X")])


def test_insertion_at_a_zero_width_range() -> None:
    assert apply_edits("ac", [AppliedEdit(1, 1, "b")]) == "abc"
