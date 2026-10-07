"""Raw engine responses and canonical OCR spans.

The split between these two is the provenance backbone of the project: a
:class:`RawEngineResponse` is the engine's own output, stored verbatim and write-once,
while a :class:`CanonicalSpan` is our normalized view of it. Every span points back at the
raw bytes it came from, so any canonicalization decision can be re-examined later.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ocr_risk.schemas.base import BBox, Polygon, RecordModel

__all__ = ["CanonicalSpan", "EngineFingerprint", "RawEngineResponse"]


class EngineFingerprint(RecordModel):
    """Identity of the exact recognizer configuration that produced a response.

    Two runs of "the same engine" with different model files or parameters are different
    measurement instruments. Mixing them in one table would quietly pool incomparable
    observations, so the fingerprint is carried on every span and checked at load time.
    """

    engine_id: str
    engine_version: str
    model_ids: tuple[str, ...] = ()
    """Model/traineddata identifiers, e.g. ``("eng.traineddata@4.1.0",)``."""
    config_hash: str
    """Canonical hash of the adapter's resolved configuration."""
    fingerprint: str
    """Canonical hash over every other field; the value stored on spans."""


class RawEngineResponse(RecordModel):
    """Verbatim engine output plus the metadata needed to reproduce the call.

    Persisted as JSON under ``data/raw/ocr/`` rather than as a table, because ``payload``
    is whatever shape the engine emits. It is wrapped, never rewritten: no key renaming,
    no rounding, no dropping of fields the current canonicalizer happens to ignore.
    """

    document_id: str
    dataset_id: str
    engine_id: str
    engine_fingerprint: str
    payload_format: str
    """Discriminator telling the canonicalizer how to read ``payload``, e.g.
    ``tesseract_tsv_v5`` or ``paddleocr_result_v2``."""
    payload: dict[str, Any]
    payload_sha256: str
    adapter_version: str
    started_at_utc: str
    duration_seconds: float = Field(ge=0.0)
    host: str
    platform: str


class CanonicalSpan(RecordModel):
    """One recognized text span in the engine-independent canonical form.

    Native confidences keep the engine's own scale and are never rescaled, clipped, or
    filled; ``None`` means the engine reported nothing, which is information that a
    sentinel like ``0.0`` would destroy. Normalized confidence is a derived feature and
    lives in the evidence layer instead.
    """

    span_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    engine_fingerprint: str
    text: str
    reading_order: int = Field(ge=0)
    """Index in the canonicalizer's derived reading order for this page."""
    line_id: str | None = None
    block_id: str | None = None
    bbox: BBox | None = None
    polygon: Polygon | None = None
    native_conf_recognition: float | None = None
    native_conf_detection: float | None = None
    conf_scale: str | None = None
    """Name of the :class:`~ocr_risk.schemas.base.ConfidenceScale` the native values use."""
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    """Offsets into the linearized OCR text stream for this (document, engine)."""
    raw_ref: str
    """Relative path of the raw response this span was derived from."""
    raw_index: int = Field(ge=0)
    """Position of the source element within the raw payload."""
    granularity_split: bool = False
    """True when this span was cut out of a coarser one the engine emitted.

    Engines disagree about what a span *is*: Tesseract and docTR emit words, PaddleOCR
    emits whole detected lines. Since correction sites are built from spans, that
    difference alone changes the site population several-fold and nearly eliminates the
    clean sites that make overcorrection observable — a tokenization artifact that would
    be indistinguishable from an engine-transfer effect. Splitting normalizes it.

    The flag matters because a split span's ``native_conf_*`` is the *parent's* value.
    The number is still verbatim engine output, but it was measured over the whole line,
    not this word, and anything reading it as a per-word measurement should know."""
