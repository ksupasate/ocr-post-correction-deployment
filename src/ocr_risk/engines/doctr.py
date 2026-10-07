"""docTR adapter (optional extra ``doctr``).

Recognition needs ``python-doctr[torch]``; parsing does not.

docTR is the one backend here that reports geometry in **relative** coordinates (fractions
of page width/height). Those are converted to pixels at parse time using the page size
recorded in the payload, because every other stage — crops, IoU, spatial features —
assumes pixels. The relative values stay in the raw payload, so the conversion is
reversible and auditable.

docTR also reports separate word-level and line-level confidences; both are preserved
rather than collapsed into one.
"""

from __future__ import annotations

import platform
import socket
import time
from typing import Any

from ocr_risk.engines.base import (
    Availability,
    OCREngineAdapter,
    PageInput,
    ParsedSpan,
    make_fingerprint,
)
from ocr_risk.engines.registry import register_engine
from ocr_risk.io.hashing import canonical_json, sha256_of_bytes
from ocr_risk.schemas.base import BBox, ConfidenceScale
from ocr_risk.schemas.spans import EngineFingerprint, RawEngineResponse

__all__ = ["DOCTR_SCALE", "DocTREngine"]

ADAPTER_NAME = "doctr"
ADAPTER_VERSION = "1"
PAYLOAD_FORMAT = "doctr_result_v1"

DOCTR_SCALE = ConfidenceScale(
    name="doctr_word_confidence_0_1", minimum=0.0, maximum=1.0, higher_is_better=True
)


@register_engine(ADAPTER_NAME)
class DocTREngine(OCREngineAdapter):
    confidence_scale = DOCTR_SCALE

    def __init__(
        self,
        engine_id: str = "doctr",
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True,
    ) -> None:
        self.engine_id = engine_id
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained
        self._predictor: Any | None = None

    def availability(self) -> Availability:
        try:
            import doctr
        except ImportError as exc:
            return Availability(
                engine_id=self.engine_id,
                installed=False,
                detail=f"python-doctr not importable: {exc}",
                remediation="uv sync --extra doctr",
            )
        return Availability(
            engine_id=self.engine_id,
            installed=True,
            version=getattr(doctr, "__version__", "unknown"),
        )

    def fingerprint(self) -> EngineFingerprint:
        availability = self.availability()
        return make_fingerprint(
            engine_id=self.engine_id,
            engine_version=availability.version or "unavailable",
            model_ids=(f"det:{self.det_arch}", f"reco:{self.reco_arch}"),
            config={
                "det_arch": self.det_arch,
                "reco_arch": self.reco_arch,
                "pretrained": self.pretrained,
            },
        )

    def recognize(self, page: PageInput) -> RawEngineResponse:
        self.availability().require()
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        if self._predictor is None:
            self._predictor = ocr_predictor(
                det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=self.pretrained
            )

        started = time.monotonic()
        document = DocumentFile.from_images([str(page.image_path)])
        result = self._predictor(document)
        payload: dict[str, Any] = {
            "export": result.export(),
            "image_sha256": page.image_sha256,
            "page_width": page.width,
            "page_height": page.height,
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

        width = float(raw.payload.get("page_width") or 0.0)
        height = float(raw.payload.get("page_height") or 0.0)
        if width <= 0 or height <= 0:
            msg = (
                f"{self.engine_id}: payload lacks page dimensions, so docTR's relative "
                "geometry cannot be converted to pixels"
            )
            raise ValueError(msg)

        spans: list[ParsedSpan] = []
        export = raw.payload.get("export") or {}
        for page in export.get("pages", []):
            for block_index, block in enumerate(page.get("blocks", [])):
                for line_index, line in enumerate(block.get("lines", [])):
                    line_conf = line.get("confidence")
                    for word in line.get("words", []):
                        text = str(word.get("value", ""))
                        if not text.strip():
                            continue
                        # float() on each coordinate rather than arithmetic on whatever
                        # the payload holds. A parser reads a recorded response, and a
                        # recorded response is whatever the storage layer produced;
                        # assuming it is already numeric made this parser fail on its own
                        # stored output while passing on the live objects in tests.
                        (rx0, ry0), (rx1, ry1) = (
                            (float(point[0]), float(point[1])) for point in word["geometry"][:2]
                        )
                        spans.append(
                            ParsedSpan(
                                text=text,
                                raw_index=len(spans),
                                bbox=BBox(
                                    x0=rx0 * width,
                                    y0=ry0 * height,
                                    x1=rx1 * width,
                                    y1=ry1 * height,
                                ),
                                line_id=f"b{block_index}:l{line_index}",
                                block_id=f"b{block_index}",
                                native_conf_recognition=float(word["confidence"]),
                                native_conf_detection=None
                                if line_conf is None
                                else float(line_conf),
                                reading_order_hint=len(spans),
                            )
                        )
        return spans
