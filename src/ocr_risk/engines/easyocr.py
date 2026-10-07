"""EasyOCR adapter (optional extra ``easyocr``).

Recognition needs ``easyocr`` (which pulls torch); parsing does not. :meth:`parse` is a
pure function of the stored payload, so the canonicalization path stays covered by fixture
tests on machines and CI runners without the backend.

EasyOCR's ``readtext`` returns quadrilaterals with a single recognition confidence per
region and no separate detection score.
"""

from __future__ import annotations

import platform
import socket
import time
from typing import Any

import numpy as np
from PIL import Image

from ocr_risk.engines.base import (
    Availability,
    OCREngineAdapter,
    PageInput,
    ParsedSpan,
    make_fingerprint,
)
from ocr_risk.engines.registry import register_engine
from ocr_risk.io.hashing import canonical_json, sha256_of_bytes
from ocr_risk.schemas.base import ConfidenceScale, Point, Polygon
from ocr_risk.schemas.spans import EngineFingerprint, RawEngineResponse

__all__ = ["EASYOCR_SCALE", "EasyOCREngine"]

ADAPTER_NAME = "easyocr"
ADAPTER_VERSION = "2"
PAYLOAD_FORMAT = "easyocr_result_v1"

EASYOCR_SCALE = ConfidenceScale(
    name="easyocr_conf_0_1", minimum=0.0, maximum=1.0, higher_is_better=True
)


@register_engine(ADAPTER_NAME)
class EasyOCREngine(OCREngineAdapter):
    confidence_scale = EASYOCR_SCALE

    def __init__(
        self,
        engine_id: str = "easyocr",
        languages: tuple[str, ...] = ("en",),
        gpu: bool = False,
        paragraph: bool = False,
    ) -> None:
        self.engine_id = engine_id
        self.languages = tuple(languages)
        self.gpu = gpu
        # Paragraph mode merges words into blocks, which destroys the word-level spans
        # this project aligns on. Off by default and recorded in the fingerprint.
        self.paragraph = paragraph
        self._reader: Any | None = None

    def availability(self) -> Availability:
        try:
            import easyocr
        except ImportError as exc:
            return Availability(
                engine_id=self.engine_id,
                installed=False,
                detail=f"easyocr not importable: {exc}",
                remediation="uv sync --extra easyocr",
            )
        return Availability(
            engine_id=self.engine_id,
            installed=True,
            version=getattr(easyocr, "__version__", "unknown"),
        )

    def fingerprint(self) -> EngineFingerprint:
        availability = self.availability()
        return make_fingerprint(
            engine_id=self.engine_id,
            engine_version=availability.version or "unavailable",
            model_ids=tuple(f"easyocr:{lang}" for lang in self.languages),
            config={
                "languages": list(self.languages),
                "gpu": self.gpu,
                "paragraph": self.paragraph,
            },
        )

    def recognize(self, page: PageInput) -> RawEngineResponse:
        self.availability().require()
        import easyocr

        if self._reader is None:
            self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu, verbose=False)

        # Decode here rather than handing EasyOCR a path. Its own loader goes through
        # scikit-image and then OpenCV, and that route failed on two independent
        # properties of one real volume: LZW compression (needs imagecodecs) and 1-bit
        # bilevel mode (OpenCV refuses it). Four pages were lost to that, and a page
        # missing from one engine breaks the matched-source premise the cross-engine
        # comparison rests on.
        #
        # Converting to 8-bit RGB is also the honest thing for a matched benchmark: every
        # engine should see the same pixels, not whatever its own decoder made of the
        # file. Pillow reads all three of the corpus's formats, and RGB is what EasyOCR
        # converts to internally anyway.
        with Image.open(page.image_path) as image:
            pixels = np.asarray(image.convert("RGB"))

        started = time.monotonic()
        result = self._reader.readtext(pixels, paragraph=self.paragraph)
        payload: dict[str, Any] = {
            "result": [
                {
                    "quad": [[float(p[0]), float(p[1])] for p in entry[0]],
                    "text": str(entry[1]),
                    "conf": float(entry[2]) if len(entry) > 2 else None,
                }
                for entry in result
            ],
            "image_sha256": page.image_sha256,
        }
        return RawEngineResponse(
            document_id=page.document_id,
            dataset_id=page.dataset_id,
            engine_id=self.engine_id,
            engine_fingerprint=self.fingerprint().fingerprint,
            payload_format=PAYLOAD_FORMAT,
            payload=payload,
            payload_sha256=sha256_of_bytes(canonical_json(payload).encode("utf-8")),
            adapter_version=ADAPTER_VERSION,
            started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_seconds=max(0.0, time.monotonic() - started),
            host=socket.gethostname(),
            platform=platform.platform(),
        )

    def parse(self, raw: RawEngineResponse) -> list[ParsedSpan]:
        if raw.payload_format != PAYLOAD_FORMAT:
            msg = f"{self.engine_id}: cannot parse payload format {raw.payload_format!r}"
            raise ValueError(msg)
        spans: list[ParsedSpan] = []
        for index, entry in enumerate(raw.payload.get("result") or []):
            text = str(entry.get("text", ""))
            if not text.strip():
                continue
            polygon = Polygon(
                points=tuple(Point(x=float(p[0]), y=float(p[1])) for p in entry["quad"])
            )
            conf = entry.get("conf")
            spans.append(
                ParsedSpan(
                    text=text,
                    raw_index=index,
                    bbox=polygon.bbox,
                    polygon=polygon,
                    native_conf_recognition=None if conf is None else float(conf),
                    native_conf_detection=None,
                    reading_order_hint=index,
                )
            )
        return spans
