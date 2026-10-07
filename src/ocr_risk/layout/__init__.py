"""Pure geometric layout primitives, depending on nothing.

This package exists so that the OCR side and the ground-truth side of an alignment can
provably share one line-grouping rule. Reading order lives in ``canonical`` (layer 3),
which dataset adapters (layer 2) may not import; without a lower home, each adapter
invented its own grouping — a fixed rounding grid in one, a semantic annotation group in
another — and the two sides of every alignment were grouped by different rules.

A shared primitive at the bottom of the stack makes that divergence impossible rather
than merely discouraged.
"""

from ocr_risk.layout.lines import group_boxes_into_lines, raster_order

__all__ = ["group_boxes_into_lines", "raster_order"]
