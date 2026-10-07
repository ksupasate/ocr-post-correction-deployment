"""Geometric primitives for alignment.

Kept out of :mod:`ocr_risk.schemas` so the data model stays logic-free, and out of the
adapters so no engine can special-case how its boxes are compared.
"""

from __future__ import annotations

from collections.abc import Sequence

from ocr_risk.schemas.base import BBox

__all__ = ["center_distance", "iou", "normalized_center_distance", "union", "vertical_overlap"]


def union(boxes: Sequence[BBox]) -> BBox | None:
    """Smallest box containing all of ``boxes``; ``None`` for an empty sequence."""
    if not boxes:
        return None
    return BBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union.

    Degenerate boxes (a thin glyph with zero width) are legal in the data model, so the
    zero-union case returns 0.0 rather than dividing by zero.
    """
    inter_w = min(a.x1, b.x1) - max(a.x0, b.x0)
    inter_h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    denominator = a.area + b.area - intersection
    return float(intersection / denominator) if denominator > 0 else 0.0


def vertical_overlap(a: BBox, b: BBox) -> float:
    """Vertical overlap as a fraction of the shorter box's height.

    More robust than IoU for deciding whether two spans share a text line, because a short
    word and a long line have tiny IoU even when perfectly co-linear.
    """
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap <= 0:
        return 0.0
    shorter = min(a.height, b.height)
    return 1.0 if shorter <= 0 else min(overlap / shorter, 1.0)


def center_distance(a: BBox, b: BBox) -> float:
    """Euclidean distance between box centers, in pixels."""
    ca, cb = a.center, b.center
    return float(((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2) ** 0.5)


def normalized_center_distance(a: BBox, b: BBox, page_width: float, page_height: float) -> float:
    """Center distance as a fraction of the page diagonal, so it is scale-free."""
    diagonal = float((page_width**2 + page_height**2) ** 0.5)
    return center_distance(a, b) / diagonal if diagonal > 0 else 0.0
