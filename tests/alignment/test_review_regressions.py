"""Regressions for defects the independent alignment audit found on real data.

Each test asserts the property the fix established, and would fail if the original
behaviour returned. The audit found all of these while the existing 85-test battery
passed, so these are written against the *semantics* rather than the shapes.
"""

from __future__ import annotations

import pytest

from ocr_risk.align.api import _status_for
from ocr_risk.align.components import Component, orphan_components
from ocr_risk.canonical import CanonicalizationPolicy, canonicalize_response
from ocr_risk.canonical.reading_order import assign_lines
from ocr_risk.config.models import AlignmentConfig
from ocr_risk.engines.base import ParsedSpan
from ocr_risk.layout import group_boxes_into_lines, raster_order
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.enums import AlignmentRelation, AlignmentStatus
from ocr_risk.schemas.spans import RawEngineResponse


def _component(relation: AlignmentRelation, origin: str) -> Component:
    spans = (0,) if relation is AlignmentRelation.OCR_INSERTION else ()
    tokens = () if relation is AlignmentRelation.OCR_INSERTION else (0,)
    return Component(span_indices=spans, token_indices=tokens, relation=relation, origin=origin)


@pytest.mark.parametrize(
    "relation",
    [AlignmentRelation.OCR_INSERTION, AlignmentRelation.OCR_DELETION],
)
def test_never_anchored_items_are_out_of_region_not_findings(
    relation: AlignmentRelation,
) -> None:
    """ "The engine omitted this" and "anchoring lost this" are different claims.

    Orphans previously returned RESOLVED before the confidence floor was consulted, at a
    constant confidence of 0.286 -- well below any floor. On 40 real FUNSD forms that was
    61% of all components, and 38% of the resulting "deletions" had an exact-text
    "insertion" twin on the same page: words the engine read correctly and the aligner
    lost, reported as recognition failures.
    """
    status, reason = _status_for(
        _component(relation, origin="orphan"),
        confidence=0.286,
        config=AlignmentConfig(),
        unresolved_blocks=0,
    )
    assert status is AlignmentStatus.OUT_OF_REGION
    assert reason == "never_anchored"


@pytest.mark.parametrize(
    "relation",
    [AlignmentRelation.OCR_INSERTION, AlignmentRelation.OCR_DELETION],
)
def test_in_region_one_sided_components_remain_findings(relation: AlignmentRelation) -> None:
    """The fix must not throw away real evidence: a DP-found insertion is a hallucination."""
    status, _ = _status_for(
        _component(relation, origin="in_region"),
        confidence=0.286,
        config=AlignmentConfig(),
        unresolved_blocks=0,
    )
    assert status is AlignmentStatus.RESOLVED


def test_refused_block_components_are_unresolved() -> None:
    """A block too large for the DP aligned nothing, so it produced no evidence.

    The refusal branch was previously unreachable: refused blocks emit pure one-sided
    components, which exited early as RESOLVED before the check was reached.
    """
    status, reason = _status_for(
        _component(AlignmentRelation.OCR_INSERTION, origin="refused_block"),
        confidence=0.99,
        config=AlignmentConfig(),
        unresolved_blocks=1,
    )
    assert status is AlignmentStatus.UNRESOLVED
    assert reason == "block_exceeded_dp_limit"


def test_orphan_components_are_tagged_as_orphans() -> None:
    components = orphan_components([0], [1], [5])
    assert {c.origin for c in components} == {"orphan"}


# --- shared line derivation ---------------------------------------------------------------
def test_a_fixed_rounding_grid_would_split_this_line_and_the_shared_grouper_does_not() -> None:
    """Two words on one line, straddling a /12 grid boundary at y=18.

    The FUNSD adapter used round(top / 12.0), which put band boundaries through the
    middle of whatever line happened to straddle them: y0=17.9 landed in band 1 and
    y0=18.1 in band 2. 15.6% of co-linear ground-truth token pairs were separated this
    way, and each one becomes a spurious insert/delete run in the character DP.
    """
    boxes = [(0.0, 17.9, 40.0, 31.9), (50.0, 18.1, 90.0, 32.1)]
    assert round(17.9 / 12.0) != round(18.1 / 12.0), "the boxes must straddle the old grid"
    assert group_boxes_into_lines(boxes) == [0, 0]


def test_raster_order_is_line_by_line_then_left_to_right() -> None:
    boxes = [(50.0, 0.0, 90.0, 14.0), (0.0, 0.0, 40.0, 14.0), (0.0, 40.0, 40.0, 54.0)]
    assert raster_order(boxes) == [1, 0, 2]


def test_geometry_beats_engine_line_ids_so_engines_are_grouped_alike() -> None:
    """Tesseract and docTR report line ids, EasyOCR and PaddleOCR do not.

    The line is the unit region anchoring matches on, so three grouping regimes across
    four engines would be indistinguishable from an engine-transfer effect.
    """
    spans = [
        ParsedSpan(text="a", raw_index=0, bbox=BBox(x0=0, y0=0, x1=10, y1=14), line_id="WRONG"),
        ParsedSpan(text="b", raw_index=1, bbox=BBox(x0=20, y0=0, x1=30, y1=14), line_id="WRONG"),
    ]
    assert assign_lines(spans) == ["line:0000", "line:0000"]


# --- span granularity ---------------------------------------------------------------------
def _raw() -> RawEngineResponse:
    return RawEngineResponse(
        document_id="doc-1",
        dataset_id="ds",
        engine_id="eng",
        engine_fingerprint="f" * 64,
        payload_format="probe",
        payload={},
        payload_sha256="x",
        adapter_version="1",
        started_at_utc="2026-01-01T00:00:00Z",
        duration_seconds=0.1,
        host="h",
        platform="p",
    )


def test_line_level_spans_are_split_into_words() -> None:
    """PaddleOCR detects whole lines; the others detect words.

    Left alone, that moved the correction-site count by 2.4x and the count of clean sites
    -- the only sites where overcorrection is observable -- by 21x, purely from
    tokenization.
    """
    parsed = [ParsedSpan(text="TOTAL DUE 42", raw_index=0, bbox=BBox(x0=0, y0=0, x1=120, y1=14))]
    spans = canonicalize_response(
        raw=_raw(),
        parsed=parsed,
        policy=CanonicalizationPolicy(span_granularity="word"),
        conf_scale_name=None,
        raw_ref="r",
    )
    assert [s.text for s in spans] == ["TOTAL", "DUE", "42"]
    assert all(s.granularity_split for s in spans)
    # Boxes are interpolated left to right and stay inside the parent.
    assert [round(s.bbox.x0) for s in spans if s.bbox] == [0, 60, 100]
    assert all(s.bbox and s.bbox.x1 <= 120.0001 for s in spans)


def test_native_granularity_is_preserved_when_asked() -> None:
    """The normalization must be measurable, not merely assumed harmless."""
    parsed = [ParsedSpan(text="TOTAL DUE 42", raw_index=0, bbox=BBox(x0=0, y0=0, x1=120, y1=14))]
    spans = canonicalize_response(
        raw=_raw(),
        parsed=parsed,
        policy=CanonicalizationPolicy(span_granularity="native"),
        conf_scale_name=None,
        raw_ref="r",
    )
    assert [s.text for s in spans] == ["TOTAL DUE 42"]
    assert not spans[0].granularity_split


def test_a_split_span_carries_the_parent_confidence_and_says_so() -> None:
    """The number is verbatim, but it was measured over the line, not the word."""
    parsed = [
        ParsedSpan(
            text="TOTAL DUE",
            raw_index=0,
            bbox=BBox(x0=0, y0=0, x1=90, y1=14),
            native_conf_recognition=0.87,
        )
    ]
    spans = canonicalize_response(
        raw=_raw(),
        parsed=parsed,
        policy=CanonicalizationPolicy(span_granularity="word"),
        conf_scale_name=None,
        raw_ref="r",
    )
    assert [s.native_conf_recognition for s in spans] == [0.87, 0.87]
    assert all(s.granularity_split for s in spans)
