"""The deployable view of a page: everything site discovery may see, and nothing else.

This module is the type-level half of the CGV3 deployability invariant (protocol
section 2). :class:`OcrPageView` is a projection of canonical OCR spans in the same
spirit as ``GenerationContext`` -- a deliberately GT-free type, constructed from spans
so that ground truth has no field to enter through. The layering test enforces the
other half: ``discovery`` may not import ``align``, ``edits``, or ``datasets``, the
layers that carry ground truth, so an alignment status or a GT-derived site population
cannot reach the enumerator even by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ocr_risk.canonical.confidence import normalize_confidence
from ocr_risk.schemas.enums import AnchorKind

if TYPE_CHECKING:  # pragma: no cover - import for typing only, no runtime dependency
    from ocr_risk.schemas.spans import CanonicalSpan

__all__ = ["OcrPageView", "OcrTokenView"]

_VIEW_VERSION = "cgv3-discovery-view-v1"


@dataclass(frozen=True, slots=True)
class OcrTokenView:
    """One OCR span as a deployed post-processor sees it."""

    span_id: str
    text: str
    line_id: str | None
    char_start: int
    char_end: int
    normalized_confidence: float | None
    """Native confidence on [0, 1] under its declared scale, or ``None`` when the
    engine reported nothing. Never a sentinel."""
    x0: float | None
    x1: float | None
    """Horizontal box extent in pixels, or ``None`` when the span has no box."""


@dataclass(frozen=True, slots=True)
class OcrPageView:
    """One (document, engine) page: OCR tokens, their order, and nothing else."""

    document_id: str
    dataset_id: str
    engine_id: str
    tokens: tuple[OcrTokenView, ...]
    """In canonical reading order (stream order)."""
    view_version: str = _VIEW_VERSION

    @classmethod
    def from_spans(cls, spans: tuple[CanonicalSpan, ...]) -> OcrPageView:
        """Project canonical spans into the deployable view, dropping everything else.

        Mirrors ``GenerationContext.from_site``: the span record carries provenance the
        enumerator has no business seeing, and the projection makes that explicit
        instead of trusting callers to avert their eyes.
        """
        if not spans:
            raise ValueError("an empty page has no sites to discover")
        documents = {span.document_id for span in spans}
        engines = {span.engine_id for span in spans}
        if len(documents) != 1 or len(engines) != 1:
            raise ValueError("a page view covers exactly one (document, engine) pair")
        ordered = sorted(spans, key=lambda span: span.char_start)
        tokens = tuple(
            OcrTokenView(
                span_id=span.span_id,
                text=span.text,
                line_id=span.line_id,
                char_start=span.char_start,
                char_end=span.char_end,
                normalized_confidence=normalize_confidence(
                    span.native_conf_recognition, span.conf_scale
                ),
                x0=span.bbox.x0 if span.bbox is not None else None,
                x1=span.bbox.x1 if span.bbox is not None else None,
            )
            for span in ordered
        )
        return cls(
            document_id=spans[0].document_id,
            dataset_id=spans[0].dataset_id,
            engine_id=spans[0].engine_id,
            tokens=tokens,
        )

    def adjacent_pairs(self) -> tuple[tuple[OcrTokenView, OcrTokenView], ...]:
        """Adjacent token pairs in stream order, with their line relationship."""
        return tuple(zip(self.tokens[:-1], self.tokens[1:]))

    def anchor_ref(self, kind: AnchorKind, *spans: OcrTokenView) -> str:
        """A deterministic reference for an anchor over one or two tokens."""
        return "\0".join((kind.value, *(span.span_id for span in spans)))
