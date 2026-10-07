"""Reading-order derivation.

Engines disagree about reading order, and some report none at all. Order matters here
because the linearized OCR stream defines character offsets, context windows, and the
sequence the alignment dynamic program walks — a scrambled order produces spurious
insertions and deletions that look like recognition errors.

The order is *derived* from geometry rather than trusted from the engine, so that all
engines are treated identically. The engine's own ordering is kept as a fallback for pages
with no geometry.

Identical treatment is not a nicety in a cross-engine study. Tesseract and docTR report
their own line ids while EasyOCR and PaddleOCR report none, so preferring the engine's
grouping would give three different line-derivation regimes across four engines — and the
line is the unit region anchoring matches on. A grouping difference would then be
indistinguishable from an engine-transfer effect.

The same derivation is applied to **ground truth**, via :func:`group_boxes_into_lines`.
Annotation formats group tokens semantically — a FUNSD entity is a form field spanning
several printed lines, a CORD ``valid_line`` is a receipt item — and matching a semantic
group against a raster line compares incomparable units.
"""

from __future__ import annotations

from collections.abc import Sequence

from ocr_risk.engines.base import ParsedSpan
from ocr_risk.layout import group_boxes_into_lines, raster_order

__all__ = ["assign_lines", "derive_reading_order", "group_boxes_into_lines", "raster_order"]


def assign_lines(spans: Sequence[ParsedSpan], overlap_ratio: float = 0.5) -> list[str]:
    """Group spans into lines from geometry.

    The engine's own line ids are used only when the page has no geometry at all. See the
    module docstring: preferring them would give a different grouping regime per engine,
    and the line is the unit that region anchoring matches on.
    """
    boxes: list[tuple[float, float, float, float] | None] = [
        None if span.bbox is None else (span.bbox.x0, span.bbox.y0, span.bbox.x1, span.bbox.y1)
        for span in spans
    ]
    if all(box is None for box in boxes) and all(span.line_id for span in spans):
        return [str(span.line_id) for span in spans]
    return [f"line:{line:04d}" for line in group_boxes_into_lines(boxes, overlap_ratio)]


def derive_reading_order(spans: Sequence[ParsedSpan], line_ids: Sequence[str]) -> list[int]:
    """Return, for each span, its position in derived reading order.

    Lines are ordered by their topmost edge, then by left edge to keep multi-column pages
    from interleaving; spans within a line are ordered left to right. Spans without
    geometry keep the engine's ordering, since there is nothing better to use.
    """
    if not spans:
        return []

    line_key: dict[str, tuple[float, float]] = {}
    for span, line_id in zip(spans, line_ids, strict=True):
        if span.bbox is None:
            continue
        top, left = span.bbox.y0, span.bbox.x0
        current = line_key.get(line_id)
        line_key[line_id] = (
            (top, left) if current is None else (min(current[0], top), min(current[1], left))
        )

    def sort_key(index: int) -> tuple[float, float, float, int]:
        span = spans[index]
        line_id = line_ids[index]
        top, left = line_key.get(line_id, (float(span.reading_order_hint or index), 0.0))
        within = span.bbox.x0 if span.bbox else float(span.reading_order_hint or index)
        return (top, left, within, index)

    ordered = sorted(range(len(spans)), key=sort_key)
    position = [0] * len(spans)
    for rank, index in enumerate(ordered):
        position[index] = rank
    return position
