"""Derived spatial features.

Deliberately scale-free: sizes are expressed relative to the page, so a 300 dpi scan and a
600 dpi scan of the same document produce the same features. Absolute pixel sizes would
make a verifier learn the resolution of its training corpus.

No fitting is needed — every feature is a function of one box and the page it sits on — so
there is no population statistic here to leak.
"""

from __future__ import annotations

from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.evidence import GeometryFeatures

__all__ = ["geometry_features"]


def geometry_features(
    bbox: BBox | None, image_width: int, image_height: int, n_spans: int = 1
) -> GeometryFeatures:
    """Spatial features for one site's bounding box."""
    if bbox is None or image_width <= 0 or image_height <= 0:
        return GeometryFeatures(n_spans=n_spans, has_geometry=False)

    width, height = bbox.width, bbox.height
    center = bbox.center
    page_area = float(image_width * image_height)
    return GeometryFeatures(
        width=width / image_width,
        height=height / image_height,
        # A degenerate box is legal (thin glyphs); reporting None beats dividing by zero
        # and beats an invented ratio.
        aspect_ratio=(width / height) if height > 0 else None,
        relative_x=center.x / image_width,
        relative_y=center.y / image_height,
        relative_area=(width * height) / page_area if page_area > 0 else None,
        n_spans=n_spans,
        has_geometry=True,
    )
