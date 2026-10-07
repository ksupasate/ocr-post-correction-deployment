"""Shared value types and model configuration for the canonical data model.

Layer 0: no logic beyond validation and trivially derived properties, and no imports from
sibling packages. Geometric *algorithms* (IoU, assignment) live in :mod:`ocr_risk.align`.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BBox",
    "ConfidenceScale",
    "Point",
    "Polygon",
    "Probability",
    "RecordModel",
    "UnitInterval",
]

# A calibrated or normalized probability. Native engine confidences are NOT this type:
# they keep their own scale (see ConfidenceScale) and are never coerced into [0, 1].
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
UnitInterval = Probability


class RecordModel(BaseModel):
    """Base for every canonical record.

    Frozen so a record cannot be mutated after validation (artifacts are immutable), and
    ``extra="forbid"`` so a typo in a column name fails loudly instead of being silently
    dropped on the way to storage.

    Whitespace is deliberately *not* stripped: leading/trailing spaces in OCR output are
    measurements, not formatting.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=False,
        validate_default=True,
        ser_json_inf_nan="null",
    )


class ConfidenceScale(RecordModel):
    """The native scale a raw engine confidence was reported on.

    Recorded alongside every native confidence so that ``0.87`` from one engine and
    ``87`` from another remain distinguishable and are never silently pooled. Normalizing
    to a common scale is a *derived feature*, produced downstream in the evidence layer.
    """

    name: str
    """Stable identifier, e.g. ``tesseract_word_conf_0_100``."""
    minimum: float
    maximum: float
    higher_is_better: bool = True

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.maximum <= self.minimum:
            msg = f"confidence scale {self.name!r}: maximum must exceed minimum"
            raise ValueError(msg)
        return self


class Point(RecordModel):
    """A pixel coordinate in image space (origin top-left, x right, y down)."""

    x: float
    y: float


class BBox(RecordModel):
    """Axis-aligned bounding box in pixel coordinates.

    Degenerate boxes (zero width or height) are permitted: some engines emit them for
    thin glyphs, and silently dropping or inflating them would falsify the raw evidence.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        if self.x1 < self.x0 or self.y1 < self.y0:
            msg = f"bbox corners out of order: ({self.x0}, {self.y0}) -> ({self.x1}, {self.y1})"
            raise ValueError(msg)
        return self

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point(x=(self.x0 + self.x1) / 2.0, y=(self.y0 + self.y1) / 2.0)


class Polygon(RecordModel):
    """An ordered vertex ring, as produced by detection-based engines.

    Kept distinct from :class:`BBox` because collapsing a polygon to its bounding box
    discards rotation and curvature that some engines actually report.
    """

    points: tuple[Point, ...] = Field(min_length=3)

    @property
    def bbox(self) -> BBox:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    @classmethod
    def from_bbox(cls, box: BBox) -> Polygon:
        """Build the axis-aligned rectangle ring for ``box`` (clockwise from top-left)."""
        return cls(
            points=(
                Point(x=box.x0, y=box.y0),
                Point(x=box.x1, y=box.y0),
                Point(x=box.x1, y=box.y1),
                Point(x=box.x0, y=box.y1),
            )
        )
