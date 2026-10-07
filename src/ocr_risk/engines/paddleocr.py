"""PaddleOCR adapter (optional extra ``paddle``).

Recognition requires ``paddleocr``/``paddlepaddle``, which are heavy and not reliably
installable on every platform. Parsing does not: :meth:`parse` is a pure function of the
stored payload, so the canonicalization path is covered by fixture tests even where the
backend cannot be installed.

PaddleOCR reports one confidence per detected text region, covering both detection and
recognition. It is stored as the recognition confidence with a null detection confidence,
rather than duplicated into both, so downstream code cannot mistake one measurement for two.
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
from ocr_risk.schemas.base import BBox, ConfidenceScale, Point, Polygon
from ocr_risk.schemas.spans import EngineFingerprint, RawEngineResponse

__all__ = ["PADDLE_SCALE", "PaddleOCREngine"]

ADAPTER_NAME = "paddleocr"
ADAPTER_VERSION = "1"
PAYLOAD_FORMAT = "paddleocr_result_v2"

PADDLE_SCALE = ConfidenceScale(
    name="paddleocr_rec_score_0_1", minimum=0.0, maximum=1.0, higher_is_better=True
)


@register_engine(ADAPTER_NAME)
class PaddleOCREngine(OCREngineAdapter):
    confidence_scale = PADDLE_SCALE

    def __init__(
        self,
        engine_id: str = "paddleocr",
        lang: str = "en",
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
    ) -> None:
        self.engine_id = engine_id
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self._ocr: Any | None = None

    def availability(self) -> Availability:
        try:
            import paddleocr
        except ImportError as exc:
            return Availability(
                engine_id=self.engine_id,
                installed=False,
                detail=f"paddleocr not importable: {exc}",
                remediation="uv sync --extra paddle",
            )
        return Availability(
            engine_id=self.engine_id,
            installed=True,
            version=getattr(paddleocr, "__version__", "unknown"),
        )

    def fingerprint(self) -> EngineFingerprint:
        availability = self.availability()
        return make_fingerprint(
            engine_id=self.engine_id,
            engine_version=availability.version or "unavailable",
            model_ids=(f"paddleocr:{self.lang}",),
            config={
                "lang": self.lang,
                "use_angle_cls": self.use_angle_cls,
                "det_db_thresh": self.det_db_thresh,
            },
        )

    def recognize(self, page: PageInput) -> RawEngineResponse:
        self.availability().require()
        from paddleocr import PaddleOCR

        if self._ocr is None:
            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=self.use_angle_cls, show_log=False)

        started = time.monotonic()
        result = self._ocr.ocr(str(page.image_path), cls=self.use_angle_cls)
        payload: dict[str, Any] = {
            "result": _to_jsonable(result),
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
        """Parse a stored PaddleOCR result. Needs no paddle installed.

        Layout: ``[[ [[x,y] x4], [text, score] ], ...]``, optionally wrapped in a
        per-image list.
        """
        if raw.payload_format != PAYLOAD_FORMAT:
            msg = f"{self.engine_id}: cannot parse payload format {raw.payload_format!r}"
            raise ValueError(msg)

        result = raw.payload.get("result") or []
        # Newer paddle versions wrap per-image; older ones return the regions directly.
        if (
            result
            and isinstance(result[0], list)
            and result[0]
            and _looks_like_region(result[0][0])
        ):
            result = result[0]

        spans: list[ParsedSpan] = []
        for index, region in enumerate(result):
            if not _looks_like_region(region):
                continue
            quad, (text, score) = region[0], region[1]
            if not str(text).strip():
                continue
            points = tuple(Point(x=float(p[0]), y=float(p[1])) for p in quad)
            polygon = Polygon(points=points)
            spans.append(
                ParsedSpan(
                    text=str(text),
                    raw_index=index,
                    bbox=polygon.bbox,
                    polygon=polygon,
                    native_conf_recognition=float(score),
                    # Paddle reports a single combined score; leaving detection null keeps
                    # one measurement from being counted as two independent signals.
                    native_conf_detection=None,
                    reading_order_hint=index,
                )
            )
        return spans


def _looks_like_region(item: Any) -> bool:
    return (
        isinstance(item, list | tuple)
        and len(item) == 2
        and isinstance(item[0], list | tuple)
        and len(item[0]) == 4
        and isinstance(item[1], list | tuple)
        and len(item[1]) == 2
    )


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays so the raw payload serializes without loss."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return value


def bbox_of(polygon: Polygon) -> BBox:
    return polygon.bbox
