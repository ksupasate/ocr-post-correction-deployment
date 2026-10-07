"""Turn engine-parsed spans into canonical spans.

This is where cross-engine *policy* is applied — Unicode normalization, reading-order
derivation, line grouping, and character offsets into the page's linearized text stream.
Doing it here rather than in each adapter means every engine is treated identically, and a
policy change re-applies to all of them from preserved raw output.

The linearized stream is the substrate the alignment stage walks, so its construction is
part of the scientific method, not a formatting detail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from ocr_risk.canonical.reading_order import assign_lines, derive_reading_order
from ocr_risk.canonical.unicode_policy import apply_policy
from ocr_risk.engines.base import ParsedSpan
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.spans import CanonicalSpan, RawEngineResponse

__all__ = ["CanonicalizationPolicy", "LinearizedText", "canonicalize_response", "linearize"]

LINE_SEPARATOR = "\n"
TOKEN_SEPARATOR = " "


@dataclass(frozen=True, slots=True)
class CanonicalizationPolicy:
    """How parsed spans become canonical spans. Recorded in the run config."""

    unicode_policy: str = "nfc"
    derive_order_from_geometry: bool = True
    """Derive reading order from boxes rather than trusting the engine. Keeps engines
    comparable; set False only to measure what the derivation itself changes."""
    line_overlap_ratio: float = 0.5
    drop_empty_spans: bool = True
    span_granularity: str = "word"
    """``word`` splits multi-word spans; ``native`` keeps whatever the engine emitted.

    Engines disagree about what a span is. On one real FUNSD page Tesseract emitted 128
    spans, docTR 130, EasyOCR 97, and PaddleOCR 29 — PaddleOCR detects whole text lines.
    Correction sites are built from spans, so on a matched corpus that difference alone
    moved the site count by 2.4x and the count of *clean* sites by 21x. Clean sites are
    the only way overcorrection is observable, so a held-out PaddleOCR fold would have
    looked far safer than it is, for a reason with nothing to do with recognition.

    ``word`` is the default because the cross-engine comparison is the point of the
    benchmark. ``native`` is kept so the effect of the normalization can itself be
    measured rather than assumed harmless."""


@dataclass(frozen=True, slots=True)
class LinearizedText:
    """A page's OCR text as one string, with offsets back to the spans that made it."""

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def span_at(self, char_index: int) -> int | None:
        """Index of the span covering ``char_index``, or ``None`` for separator space."""
        for i, (start, end) in enumerate(zip(self.starts, self.ends, strict=True)):
            if start <= char_index < end:
                return i
        return None


def linearize(texts: Sequence[str], line_ids: Sequence[str]) -> LinearizedText:
    """Join span texts into one stream, recording each span's character interval.

    Spans on the same line are joined with a space and lines with a newline, mirroring how
    ground truth is stored. Separators occupy real offsets so that a merged token in the
    OCR (which lacks the space) shifts against ground truth exactly as it does on the page
    — that displacement is the signal the aligner uses to detect a MERGE.
    """
    parts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    previous_line: str | None = None

    for text, line_id in zip(texts, line_ids, strict=True):
        if previous_line is not None:
            separator = LINE_SEPARATOR if line_id != previous_line else TOKEN_SEPARATOR
            parts.append(separator)
            cursor += len(separator)
        starts.append(cursor)
        parts.append(text)
        cursor += len(text)
        ends.append(cursor)
        previous_line = line_id

    return LinearizedText(text="".join(parts), starts=tuple(starts), ends=tuple(ends))


def _split_to_words(span: ParsedSpan) -> list[ParsedSpan]:
    """Cut a multi-word span into word spans, interpolating boxes by character offset.

    The box is divided horizontally in proportion to character position, which assumes
    left-to-right horizontal text set in a roughly monospaced advance. That is an
    approximation, and a deliberate one: the alternative is leaving engines with
    incomparable span granularity, which corrupts the site population outright. Word
    boxes are only ever used for region anchoring and crop recipes, both tolerant of a
    few pixels; the *text* offsets they carry are exact.

    The parent's confidence is copied to each word and the span is flagged
    ``granularity_split``, because it was measured over the line, not the word.
    """
    text = span.text
    stripped = text.strip()
    if not stripped or " " not in stripped:
        return [span]

    pieces: list[tuple[int, int, str]] = []
    cursor = 0
    for word in text.split(" "):
        if word:
            pieces.append((cursor, cursor + len(word), word))
        cursor += len(word) + 1
    if len(pieces) <= 1:
        return [span]

    total = max(len(text), 1)
    box, poly = span.bbox, span.polygon
    out: list[ParsedSpan] = []
    for start, end, word in pieces:
        sub_box = box
        if box is not None:
            width = box.x1 - box.x0
            sub_box = BBox(
                x0=box.x0 + width * (start / total),
                y0=box.y0,
                x1=box.x0 + width * (end / total),
                y1=box.y1,
            )
        out.append(
            ParsedSpan(
                text=word,
                raw_index=span.raw_index,
                bbox=sub_box,
                # The parent polygon describes the whole line, so carrying it onto a word
                # would assert geometry that was never measured. The derived box is the
                # honest statement of what is known.
                polygon=poly if sub_box is box else None,
                line_id=span.line_id,
                block_id=span.block_id,
                native_conf_recognition=span.native_conf_recognition,
                native_conf_detection=span.native_conf_detection,
                reading_order_hint=span.reading_order_hint,
                granularity_split=True,
            )
        )
    return out


def canonicalize_response(
    raw: RawEngineResponse,
    parsed: Sequence[ParsedSpan],
    policy: CanonicalizationPolicy,
    conf_scale_name: str | None,
    raw_ref: str,
) -> list[CanonicalSpan]:
    """Apply canonicalization policy to one page's parsed spans."""
    spans = [s for s in parsed if not policy.drop_empty_spans or s.text.strip()]
    if policy.span_granularity == "word":
        spans = [word for span in spans for word in _split_to_words(span)]
    if not spans:
        return []

    normalized_texts = [apply_policy(span.text, policy.unicode_policy) for span in spans]
    line_ids = assign_lines(spans, policy.line_overlap_ratio)

    if policy.derive_order_from_geometry:
        order = derive_reading_order(spans, line_ids)
    else:
        order = [
            s.reading_order_hint if s.reading_order_hint is not None else i
            for i, s in enumerate(spans)
        ]

    # Emit in reading order so the linearized stream matches the order a reader would see.
    sequence = sorted(range(len(spans)), key=lambda i: (order[i], i))
    ordered_texts = [normalized_texts[i] for i in sequence]
    ordered_lines = [line_ids[i] for i in sequence]
    linear = linearize(ordered_texts, ordered_lines)

    canonical: list[CanonicalSpan] = []
    for position, source_index in enumerate(sequence):
        span = spans[source_index]
        canonical.append(
            CanonicalSpan(
                span_id=f"{raw.document_id}:{raw.engine_id}:{position:05d}",
                document_id=raw.document_id,
                dataset_id=raw.dataset_id,
                engine_id=raw.engine_id,
                engine_fingerprint=raw.engine_fingerprint,
                text=ordered_texts[position],
                reading_order=position,
                line_id=ordered_lines[position],
                block_id=span.block_id,
                bbox=span.bbox,
                polygon=span.polygon,
                granularity_split=span.granularity_split,
                # Verbatim: canonicalization normalizes text and order, never measurements.
                native_conf_recognition=span.native_conf_recognition,
                native_conf_detection=span.native_conf_detection,
                conf_scale=conf_scale_name,
                char_start=linear.starts[position],
                char_end=linear.ends[position],
                raw_ref=raw_ref,
                raw_index=span.raw_index,
            )
        )
    return canonical


def rebuild_stream(spans: Sequence[CanonicalSpan]) -> str:
    """Reconstruct the linearized text a set of canonical spans came from.

    Uses the stored character offsets rather than re-joining, so the reconstruction is
    exact even if the joining rules later change.
    """
    if not spans:
        return ""
    ordered = sorted(spans, key=lambda s: s.reading_order)
    length = max(s.char_end for s in ordered)
    buffer = [" "] * length
    for span in ordered:
        for offset, char in enumerate(span.text):
            index = span.char_start + offset
            if index < length:
                buffer[index] = char
    # Restore line separators in the gaps between spans on different lines.
    for previous, current in pairwise(ordered):
        if current.line_id != previous.line_id and previous.char_end < len(buffer):
            buffer[previous.char_end] = LINE_SEPARATOR
    return "".join(buffer)
