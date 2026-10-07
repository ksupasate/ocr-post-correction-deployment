"""The OCR engine adapter contract.

Two responsibilities, deliberately separated:

1. :meth:`OCREngineAdapter.recognize` returns the engine's **verbatim** output, wrapped in
   a provenance envelope and stored write-once.
2. :meth:`OCREngineAdapter.canonicalize` turns that stored payload into engine-independent
   spans.

Keeping them apart is what makes the ``replay`` adapter possible: real recognition happens
once, and every later experiment re-derives canonical spans from the preserved bytes. It
also means a canonicalization bug can be fixed and re-applied without re-running OCR, and
that the raw evidence outlives our current opinion about how to read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ocr_risk.io.hashing import canonical_hash
from ocr_risk.schemas.base import BBox, ConfidenceScale, Polygon
from ocr_risk.schemas.spans import EngineFingerprint, RawEngineResponse

__all__ = [
    "Availability",
    "OCREngineAdapter",
    "PageInput",
    "ParsedSpan",
    "make_fingerprint",
]


@dataclass(frozen=True, slots=True)
class ParsedSpan:
    """An engine's own view of one recognized span, before cross-engine normalization.

    Deliberately not a :class:`~ocr_risk.schemas.CanonicalSpan`: character offsets into
    the page's linearized text stream, Unicode policy, and reading-order derivation are
    cross-engine *policy* and belong to :mod:`ocr_risk.canonical`. An adapter that had to
    compute them would be making scientific decisions inside a backend wrapper.
    """

    text: str
    raw_index: int
    bbox: BBox | None = None
    polygon: Polygon | None = None
    line_id: str | None = None
    block_id: str | None = None
    native_conf_recognition: float | None = None
    native_conf_detection: float | None = None
    reading_order_hint: int | None = None
    """The engine's own ordering, when it reports one. ``canonical`` may override it."""
    granularity_split: bool = False
    """Set by ``canonical`` when this span was cut out of a coarser one the engine
    emitted. Never set by an adapter: an adapter reports what its backend produced."""


@dataclass(frozen=True, slots=True)
class Availability:
    """Whether an engine can actually run here, and what to do if not."""

    engine_id: str
    installed: bool
    version: str | None = None
    detail: str = ""
    remediation: str = ""

    def require(self) -> None:
        if not self.installed:
            hint = f"\n  remediation: {self.remediation}" if self.remediation else ""
            msg = f"OCR engine {self.engine_id!r} is not available: {self.detail}{hint}"
            raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class PageInput:
    """One page handed to an engine."""

    document_id: str
    dataset_id: str
    image_path: Path
    image_sha256: str
    width: int
    height: int


def make_fingerprint(
    engine_id: str,
    engine_version: str,
    model_ids: tuple[str, ...],
    config: dict[str, Any],
) -> EngineFingerprint:
    """Build the identity of a specific recognizer configuration.

    Two runs of "the same engine" with different model files or parameters are different
    measurement instruments; the fingerprint keeps them from being pooled by accident.
    """
    config_hash = canonical_hash(config)
    payload = {
        "engine_id": engine_id,
        "engine_version": engine_version,
        "model_ids": sorted(model_ids),
        "config_hash": config_hash,
    }
    return EngineFingerprint(
        engine_id=engine_id,
        engine_version=engine_version,
        model_ids=model_ids,
        config_hash=config_hash,
        fingerprint=canonical_hash(payload),
    )


@runtime_checkable
class OCREngineAdapter(Protocol):
    """Contract every OCR backend satisfies.

    A new engine is added by implementing this and registering it. No experiment,
    alignment, or evaluation code changes.
    """

    engine_id: str
    confidence_scale: ConfidenceScale
    """The native scale this engine reports on. Recorded on every span; never converted
    away."""

    def availability(self) -> Availability: ...

    def fingerprint(self) -> EngineFingerprint: ...

    def recognize(self, page: PageInput) -> RawEngineResponse:
        """Run recognition and return the native payload, unmodified."""
        ...

    def parse(self, raw: RawEngineResponse) -> list[ParsedSpan]:
        """Read a stored raw response into engine-neutral spans.

        Must be a pure function of ``raw``: the same bytes must yield the same spans on
        any machine, with or without the engine installed. That is what lets a
        canonicalization bug be fixed and re-applied without re-running OCR, and what
        makes the ``replay`` adapter and fixture-based tests possible.
        """
        ...
