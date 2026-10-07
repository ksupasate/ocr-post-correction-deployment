"""Grouping boxes into raster lines, and ordering them.

Shared by the OCR path (via ``canonical.reading_order``) and by every dataset adapter, so
both sides of an alignment are grouped by the same rule. See the package docstring for
why this lives at the bottom of the stack.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["group_boxes_into_lines", "raster_order"]

Box = tuple[float, float, float, float]


def group_boxes_into_lines(
    boxes: Sequence[tuple[float, float, float, float] | None], overlap_ratio: float = 0.5
) -> list[int]:
    """Group ``(x0, y0, x1, y1)`` boxes into raster lines by vertical overlap.

    Two boxes share a line when their vertical extents overlap by at least
    ``overlap_ratio`` of the shorter one, with the running band widened as the line grows
    so a tall glyph does not split it. A fixed pixel tolerance would fail across the font
    sizes in one corpus, and a fixed rounding grid puts a boundary through the middle of
    whatever line happens to straddle it.

    Shared by the OCR and ground-truth paths so both sides of an alignment are grouped by
    the same rule.
    """

    def _key(index: int) -> tuple[float, float]:
        box = boxes[index]
        return (0.0, 0.0) if box is None else (box[1], box[0])

    ordered = sorted(range(len(boxes)), key=_key)
    line_of: dict[int, int] = {}
    current_line = -1
    anchor: tuple[float, float] | None = None

    for index in ordered:
        box = boxes[index]
        if box is None:
            current_line += 1
            line_of[index] = current_line
            anchor = None
            continue
        if anchor is None or not _overlaps(anchor, (box[1], box[3]), overlap_ratio):
            current_line += 1
            anchor = (box[1], box[3])
        else:
            anchor = (min(anchor[0], box[1]), max(anchor[1], box[3]))
        line_of[index] = current_line

    return [line_of[i] for i in range(len(boxes))]


def raster_order(
    boxes: Sequence[tuple[float, float, float, float] | None], overlap_ratio: float = 0.5
) -> list[int]:
    """Indices of ``boxes`` in reading order: line by line, left to right within a line.

    The ground-truth counterpart of :func:`derive_reading_order`. Annotation order is a
    convention of the annotation tool -- FUNSD numbers form fields in logical order, CORD
    numbers receipt items -- and using it as transcription order makes the ground-truth
    stream disagree with every engine's reading for reasons that have nothing to do with
    recognition.
    """
    lines = group_boxes_into_lines(boxes, overlap_ratio)
    tops: dict[int, float] = {}
    lefts: dict[int, float] = {}
    for index, line in enumerate(lines):
        box = boxes[index]
        top = box[1] if box else 0.0
        left = box[0] if box else 0.0
        tops[line] = min(tops.get(line, top), top)
        lefts[line] = min(lefts.get(line, left), left)
    return sorted(
        range(len(boxes)),
        key=lambda i: (
            tops[lines[i]],
            lefts[lines[i]],
            (boxes[i] or (0.0, 0.0, 0.0, 0.0))[0],
            i,
        ),
    )


def _overlaps(a: tuple[float, float], b: tuple[float, float], ratio: float) -> bool:
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    if overlap <= 0:
        return False
    shorter = min(a[1] - a[0], b[1] - b[0])
    return shorter <= 0 or overlap / shorter >= ratio
