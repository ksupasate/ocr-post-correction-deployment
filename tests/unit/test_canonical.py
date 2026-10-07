"""Canonicalization: Unicode policy, line grouping, reading order, and stream offsets.

The linearized stream built here is the substrate the aligner walks, so an off-by-one in
offsets or a scrambled reading order manufactures insertions and deletions that look like
recognition errors. These tests pin that behaviour down.
"""

from __future__ import annotations

import pytest

from ocr_risk.canonical import (
    CanonicalizationPolicy,
    apply_policy,
    assign_lines,
    canonicalize_response,
    derive_reading_order,
    get_scale,
    linearize,
    normalize_confidence,
    rebuild_stream,
)
from ocr_risk.canonical.unicode_policy import POLICIES
from ocr_risk.engines.base import ParsedSpan
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.spans import RawEngineResponse


def _span(
    text: str,
    x0: float,
    y0: float,
    *,
    index: int = 0,
    line: str | None = None,
    with_geometry: bool = True,
) -> ParsedSpan:
    return ParsedSpan(
        text=text,
        raw_index=index,
        bbox=BBox(x0=x0, y0=y0, x1=x0 + 10 * max(len(text), 1), y1=y0 + 14)
        if with_geometry
        else None,
        line_id=line,
        native_conf_recognition=90.0,
        reading_order_hint=index,
    )


def _raw(fmt: str = "synthetic_words_v1") -> RawEngineResponse:
    return RawEngineResponse(
        document_id="doc-1",
        dataset_id="synthetic",
        engine_id="eng",
        engine_fingerprint="f" * 64,
        payload_format=fmt,
        payload={},
        payload_sha256="x",
        adapter_version="1",
        started_at_utc="2026-01-01T00:00:00Z",
        duration_seconds=0.1,
        host="h",
        platform="p",
    )


# --- unicode policy ------------------------------------------------------------------
def test_nfc_composes_decomposed_characters() -> None:
    decomposed = "é"  # e + combining acute
    assert apply_policy(decomposed, "nfc") == "é"
    assert len(apply_policy(decomposed, "nfc")) == 1


def test_none_policy_changes_nothing() -> None:
    decomposed = "é"
    assert apply_policy(decomposed, "none") == decomposed


def test_ligature_policy_expands_ligatures() -> None:
    """Left alone, 'ﬁ' vs 'fi' becomes a phantom character error in every alignment."""
    assert apply_policy("ﬁle", "nfc") == "ﬁle"
    assert apply_policy("ﬁle", "nfc_ligatures") == "file"


def test_aggressive_policy_folds_typographic_punctuation() -> None:
    assert apply_policy("don’t — now", "aggressive") == "don't - now"


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown unicode policy"):
        apply_policy("x", "made_up")


@pytest.mark.parametrize("policy", POLICIES)
def test_every_declared_policy_is_implemented(policy: str) -> None:
    assert isinstance(apply_policy("Aﬁé—1", policy), str)


# --- confidence normalization ----------------------------------------------------------
def test_normalization_uses_the_declared_scale() -> None:
    assert normalize_confidence(87.0, "tesseract_word_conf_0_100") == pytest.approx(0.87)
    assert normalize_confidence(0.87, "synthetic_0_1") == pytest.approx(0.87)


def test_normalization_without_a_known_scale_returns_none() -> None:
    """Guessing a scale would invent a measurement."""
    assert normalize_confidence(87.0, "no_such_scale") is None
    assert normalize_confidence(87.0, None) is None


def test_normalization_of_missing_value_stays_missing() -> None:
    assert normalize_confidence(None, "synthetic_0_1") is None


def test_out_of_range_values_are_clipped() -> None:
    assert normalize_confidence(150.0, "tesseract_word_conf_0_100") == 1.0
    assert normalize_confidence(-5.0, "tesseract_word_conf_0_100") == 0.0


def test_builtin_scales_are_registered() -> None:
    for name in ("tesseract_word_conf_0_100", "synthetic_0_1", "synthetic_0_100"):
        assert get_scale(name) is not None


# --- line grouping ----------------------------------------------------------------------
def test_geometry_overrides_engine_line_ids() -> None:
    """All four engines must be grouped by one rule, or grouping confounds transfer.

    Tesseract and docTR report line ids; EasyOCR and PaddleOCR report none. Preferring
    the engine's grouping would give three regimes across four engines, and the line is
    the unit region anchoring matches on — so a grouping difference would be
    indistinguishable from a cross-engine effect.
    """
    spans = [
        _span("a", 0, 0, line="L1"),
        _span("b", 20, 0, line="L1"),
        _span("c", 0, 40, line="L2"),
    ]
    lines = assign_lines(spans)
    assert lines[0] == lines[1] != lines[2], "geometry should still separate the two rows"
    assert lines == [line for line in lines if line.startswith("line:")], (
        "engine-reported ids must not survive when geometry is available"
    )


def test_engine_line_ids_are_the_fallback_only_without_geometry() -> None:
    spans = [
        _span("a", 0, 0, line="L1", with_geometry=False),
        _span("b", 20, 0, line="L1", with_geometry=False),
        _span("c", 0, 40, line="L2", with_geometry=False),
    ]
    assert assign_lines(spans) == ["L1", "L1", "L2"]


def test_lines_are_grouped_by_vertical_overlap_when_ids_are_absent() -> None:
    """A fixed pixel tolerance would fail across the font sizes in one corpus."""
    spans = [_span("a", 0, 0), _span("b", 40, 2), _span("c", 0, 60)]
    lines = assign_lines(spans)
    assert lines[0] == lines[1] != lines[2]


def test_tall_glyph_does_not_split_a_line() -> None:
    tall = ParsedSpan(text="J", raw_index=1, bbox=BBox(x0=40, y0=-4, x1=50, y1=20))
    spans = [_span("a", 0, 0), tall, _span("c", 60, 1)]
    lines = assign_lines(spans)
    assert len(set(lines)) == 1


# --- reading order -------------------------------------------------------------------------
def test_reading_order_is_top_to_bottom_then_left_to_right() -> None:
    spans = [_span("third", 0, 40), _span("second", 40, 0), _span("first", 0, 0)]
    lines = assign_lines(spans)
    order = derive_reading_order(spans, lines)
    ordered = [spans[i].text for i in sorted(range(len(spans)), key=lambda i: order[i])]
    assert ordered == ["first", "second", "third"]


def test_reading_order_is_a_permutation() -> None:
    spans = [_span(str(i), i * 5.0, (i % 3) * 40.0, index=i) for i in range(9)]
    order = derive_reading_order(spans, assign_lines(spans))
    assert sorted(order) == list(range(9))


def test_two_column_page_does_not_interleave() -> None:
    """Columns share vertical bands; ordering by y alone would zip them together."""
    left = [_span(f"L{i}", 0, i * 40.0, index=i) for i in range(3)]
    right = [_span(f"R{i}", 500, i * 40.0, index=10 + i) for i in range(3)]
    spans = [*left, *right]
    lines = [f"col-left:{i}" for i in range(3)] + [f"col-right:{i}" for i in range(3)]
    order = derive_reading_order(spans, lines)
    ordered = [spans[i].text for i in sorted(range(len(spans)), key=lambda i: order[i])]
    # Within each band the left column precedes the right one.
    assert ordered.index("L0") < ordered.index("R0")
    assert ordered.index("L1") < ordered.index("R1")


# --- linearization ---------------------------------------------------------------------------
def test_linearize_joins_with_spaces_and_newlines() -> None:
    linear = linearize(["Dose:", "0.015", "mg"], ["l0", "l0", "l0"])
    assert linear.text == "Dose: 0.015 mg"
    assert linear.starts == (0, 6, 12)
    assert linear.ends == (5, 11, 14)


def test_linearize_separates_lines_with_a_newline() -> None:
    linear = linearize(["a", "b"], ["l0", "l1"])
    assert linear.text == "a\nb"


def test_offsets_index_the_exact_span_text() -> None:
    texts = ["Dose:", "0.015", "mg", "Smith"]
    linear = linearize(texts, ["l0", "l0", "l0", "l1"])
    for text, start, end in zip(texts, linear.starts, linear.ends, strict=True):
        assert linear.text[start:end] == text


def test_span_at_maps_characters_back_to_spans() -> None:
    linear = linearize(["ab", "cd"], ["l0", "l0"])
    assert linear.span_at(0) == 0
    assert linear.span_at(1) == 0
    assert linear.span_at(2) is None  # the separator
    assert linear.span_at(3) == 1


def test_empty_input_linearizes_to_empty() -> None:
    linear = linearize([], [])
    assert linear.text == ""
    assert linear.starts == ()


# --- full canonicalization ---------------------------------------------------------------------
def test_canonicalize_assigns_ids_order_and_offsets() -> None:
    parsed = [_span("world", 60, 0, index=1), _span("hello", 0, 0, index=0)]
    spans = canonicalize_response(
        raw=_raw(),
        parsed=parsed,
        policy=CanonicalizationPolicy(unicode_policy="nfc"),
        conf_scale_name="synthetic_0_1",
        raw_ref="raw/ocr/x.json",
    )
    assert [s.text for s in spans] == ["hello", "world"]
    assert [s.reading_order for s in spans] == [0, 1]
    assert [s.span_id for s in spans] == ["doc-1:eng:00000", "doc-1:eng:00001"]
    assert (spans[0].char_start, spans[0].char_end) == (0, 5)
    assert (spans[1].char_start, spans[1].char_end) == (6, 11)
    # raw_index still points at the original payload element, not the sorted position.
    assert spans[0].raw_index == 0
    assert spans[1].raw_index == 1


def test_canonicalize_preserves_native_confidence_and_scale() -> None:
    spans = canonicalize_response(
        raw=_raw(),
        parsed=[_span("x", 0, 0)],
        policy=CanonicalizationPolicy(),
        conf_scale_name="tesseract_word_conf_0_100",
        raw_ref="r",
    )
    assert spans[0].native_conf_recognition == 90.0
    assert spans[0].conf_scale == "tesseract_word_conf_0_100"


def test_canonicalize_drops_whitespace_only_spans() -> None:
    spans = canonicalize_response(
        raw=_raw(),
        parsed=[_span("real", 0, 0), _span("   ", 50, 0, index=1)],
        policy=CanonicalizationPolicy(drop_empty_spans=True),
        conf_scale_name=None,
        raw_ref="r",
    )
    assert [s.text for s in spans] == ["real"]


def test_canonicalize_of_nothing_is_empty() -> None:
    assert canonicalize_response(_raw(), [], CanonicalizationPolicy(), None, "r") == []


def test_canonicalize_applies_the_unicode_policy() -> None:
    spans = canonicalize_response(
        raw=_raw(),
        parsed=[_span("école", 0, 0)],
        policy=CanonicalizationPolicy(unicode_policy="nfc"),
        conf_scale_name=None,
        raw_ref="r",
    )
    assert spans[0].text == "école"


def test_rebuild_stream_recovers_the_text() -> None:
    spans = canonicalize_response(
        raw=_raw(),
        parsed=[_span("Dose:", 0, 0), _span("mg", 60, 0, index=1), _span("Smith", 0, 40, index=2)],
        policy=CanonicalizationPolicy(),
        conf_scale_name=None,
        raw_ref="r",
    )
    assert rebuild_stream(spans) == "Dose: mg\nSmith"


def test_rebuild_stream_of_nothing_is_empty() -> None:
    assert rebuild_stream([]) == ""
