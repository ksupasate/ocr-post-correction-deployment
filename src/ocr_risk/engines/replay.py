"""Replay adapter: re-derive spans from previously stored raw responses.

This is the payoff of preserving raw engine output. Real recognition is expensive,
sometimes non-deterministic, and often impossible to reproduce on another machine (missing
binary, different model version, no GPU). Replay makes it a one-time cost:

- experiments re-run against preserved bytes instead of re-invoking recognizers;
- CI exercises the real canonicalization path for engines it cannot install;
- a canonicalization fix can be re-applied to historical data without new OCR;
- a collaborator can reproduce an analysis with no OCR stack at all.

It refuses to recognize. Asking it to would silently substitute stale evidence for a fresh
measurement, which is exactly the confusion the raw layer exists to prevent.
"""

from __future__ import annotations

from ocr_risk.engines.base import Availability, OCREngineAdapter, PageInput, ParsedSpan
from ocr_risk.engines.registry import build_engine, register_engine
from ocr_risk.io.raw_store import RawStore
from ocr_risk.schemas.base import ConfidenceScale
from ocr_risk.schemas.spans import EngineFingerprint, RawEngineResponse

__all__ = ["ReplayEngine"]

ADAPTER_NAME = "replay"

# Payload format -> the adapter that knows how to read it. Replay is a dispatcher, not a
# second parser: reimplementing parsing here would let the two drift apart.
_PARSER_ADAPTERS: dict[str, str] = {
    "tesseract_tsv_v5": "tesseract",
    "synthetic_words_v1": "synthetic",
    "paddleocr_result_v2": "paddleocr",
    "easyocr_result_v1": "easyocr",
    "doctr_result_v1": "doctr",
}


@register_engine(ADAPTER_NAME)
class ReplayEngine(OCREngineAdapter):
    """Parses stored responses by delegating to whichever adapter wrote them."""

    confidence_scale = ConfidenceScale(name="replay_passthrough", minimum=0.0, maximum=1.0)

    def __init__(self, engine_id: str = "replay", source_engine_id: str | None = None) -> None:
        self.engine_id = engine_id
        self.source_engine_id = source_engine_id or engine_id
        self.store = RawStore()

    def availability(self) -> Availability:
        return Availability(
            engine_id=self.engine_id,
            installed=True,
            version="1",
            detail="reads preserved raw responses; needs no recognizer",
        )

    def fingerprint(self) -> EngineFingerprint:
        msg = (
            "replay has no fingerprint of its own: spans keep the fingerprint of the "
            "engine that originally produced them. Read it from the stored response."
        )
        raise NotImplementedError(msg)

    def recognize(self, page: PageInput) -> RawEngineResponse:
        msg = (
            f"replay cannot recognize {page.document_id!r}. It re-derives spans from "
            "preserved output; returning stored bytes here would disguise stale evidence "
            "as a fresh measurement."
        )
        raise NotImplementedError(msg)

    def parse(self, raw: RawEngineResponse) -> list[ParsedSpan]:
        adapter_name = _PARSER_ADAPTERS.get(raw.payload_format)
        if adapter_name is None:
            known = ", ".join(sorted(_PARSER_ADAPTERS))
            msg = f"no parser registered for payload format {raw.payload_format!r}; known: {known}"
            raise ValueError(msg)
        adapter = build_engine(adapter_name, engine_id=raw.engine_id)
        return adapter.parse(raw)
