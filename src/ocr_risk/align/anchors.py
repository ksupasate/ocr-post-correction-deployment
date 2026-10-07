"""Stage A: anchor OCR regions to ground-truth regions.

Anchoring bounds the character-level alignment to small, comparable pieces. Without it a
full page is one dynamic program whose cost is quadratic in page length and whose best
path can wander between unrelated lines.

Two modes:

- **Geometric** (ground truth has boxes): lines are matched by a global assignment over a
  cost blending vertical overlap, IoU, and normalized center distance. Assignments above a
  cost ceiling are *dropped*, not kept as weak anchors.
- **Sequential** (no ground-truth geometry): the page is a single region, and the
  character DP is bounded instead by the anchor-and-recurse decomposition in
  :mod:`ocr_risk.align.blocks`.

Leftover lines on either side are reported rather than force-matched: an OCR line with no
ground-truth counterpart is a hallucinated region, and a ground-truth line with no OCR
counterpart is an omitted one. Both are findings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from ocr_risk.align.geometry import iou, normalized_center_distance, union, vertical_overlap
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.documents import GtToken
from ocr_risk.schemas.spans import CanonicalSpan

__all__ = ["Region", "RegionPlan", "build_regions"]


class _Placed(Protocol):
    """The only two attributes anchoring needs from a span or a ground-truth token.

    Declared read-only: the records are frozen, so a protocol with mutable attributes
    would not match them structurally.
    """

    @property
    def line_id(self) -> str | None: ...

    @property
    def bbox(self) -> BBox | None: ...


_UNMATCHED_PENALTY = 10.0


@dataclass(frozen=True, slots=True)
class Region:
    """One anchored pairing of OCR spans with ground-truth tokens."""

    region_id: str
    span_indices: tuple[int, ...]
    token_indices: tuple[int, ...]
    cost: float
    iou: float | None = None


@dataclass(slots=True)
class RegionPlan:
    """Anchored regions plus whatever could not be anchored."""

    regions: list[Region] = field(default_factory=list)
    unmatched_span_indices: tuple[int, ...] = ()
    """OCR lines with no ground-truth counterpart: hallucinated regions."""
    unmatched_token_indices: tuple[int, ...] = ()
    """Ground-truth lines the OCR did not produce: omitted regions."""
    used_geometry: bool = False


def _group_by_line(items: Sequence[_Placed]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        key = item.line_id or "line:unknown"
        groups.setdefault(key, []).append(index)
    return groups


def _line_boxes(indices: Sequence[int], items: Sequence[_Placed]) -> BBox | None:
    boxes = [items[i].bbox for i in indices]
    present = [b for b in boxes if b is not None]
    return union(present) if present else None


def build_regions(
    spans: Sequence[CanonicalSpan],
    tokens: Sequence[GtToken],
    *,
    use_geometry: bool,
    max_anchor_cost: float,
    page_width: float,
    page_height: float,
) -> RegionPlan:
    """Pair OCR lines with ground-truth lines, or fall back to one whole-page region."""
    if not spans and not tokens:
        return RegionPlan()

    geometry_available = (
        use_geometry
        and any(s.bbox is not None for s in spans)
        and any(t.bbox is not None for t in tokens)
    )
    if not geometry_available:
        return RegionPlan(
            regions=[
                Region(
                    region_id="region:page",
                    span_indices=tuple(range(len(spans))),
                    token_indices=tuple(range(len(tokens))),
                    cost=0.0,
                )
            ],
            used_geometry=False,
        )

    ocr_lines = _group_by_line(spans)
    gt_lines = _group_by_line(tokens)
    ocr_keys = sorted(ocr_lines, key=lambda k: min(ocr_lines[k]))
    gt_keys = sorted(gt_lines, key=lambda k: min(gt_lines[k]))

    cost = np.full((len(ocr_keys), len(gt_keys)), _UNMATCHED_PENALTY, dtype=np.float64)
    overlaps = np.zeros_like(cost)
    for i, ocr_key in enumerate(ocr_keys):
        ocr_box = _line_boxes(ocr_lines[ocr_key], spans)
        if ocr_box is None:
            continue
        for j, gt_key in enumerate(gt_keys):
            gt_box = _line_boxes(gt_lines[gt_key], tokens)
            if gt_box is None:
                continue
            overlap = vertical_overlap(ocr_box, gt_box)
            box_iou = iou(ocr_box, gt_box)
            distance = normalized_center_distance(ocr_box, gt_box, page_width, page_height)
            overlaps[i, j] = box_iou
            if overlap <= 0.0:
                # Two lines that do not vertically overlap at all are not the same line,
                # whatever their centres do. Left as the unmatched penalty rather than
                # scored: the weighted form gave every zero-overlap pair a cost in
                # [0.85, 1.0], so the entire "no geometric relationship" region sat just
                # under a 0.92 threshold and 16% of anchors had zero IoU. That is not a
                # threshold that discriminates, it is a cliff.
                continue
            # Vertical overlap dominates: a short word and its long ground-truth line are
            # co-linear but have small IoU, so an IoU-led cost would reject the true pair.
            cost[i, j] = 0.6 * (1.0 - overlap) + 0.25 * (1.0 - box_iou) + 0.15 * min(distance, 1.0)

    rows, cols = linear_sum_assignment(cost)

    regions: list[Region] = []
    matched_ocr: set[int] = set()
    matched_gt: set[int] = set()
    for i, j in zip(rows, cols, strict=True):
        if cost[i, j] > max_anchor_cost:
            continue  # Never keep a weak anchor just to raise coverage.
        matched_ocr.add(i)
        matched_gt.add(j)
        regions.append(
            Region(
                region_id=f"region:{len(regions):04d}",
                span_indices=tuple(ocr_lines[ocr_keys[i]]),
                token_indices=tuple(gt_lines[gt_keys[j]]),
                cost=float(cost[i, j]),
                iou=float(overlaps[i, j]),
            )
        )

    unmatched_spans = tuple(
        index for i, key in enumerate(ocr_keys) if i not in matched_ocr for index in ocr_lines[key]
    )
    unmatched_tokens = tuple(
        index for j, key in enumerate(gt_keys) if j not in matched_gt for index in gt_lines[key]
    )
    regions.sort(key=lambda r: min(r.token_indices) if r.token_indices else 0)
    return RegionPlan(
        regions=regions,
        unmatched_span_indices=unmatched_spans,
        unmatched_token_indices=unmatched_tokens,
        used_geometry=True,
    )
