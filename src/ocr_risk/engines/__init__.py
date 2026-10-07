"""OCR engine adapters. Raw native responses are preserved before canonicalization.

Layer 2. Importing this package registers every adapter. Heavy backends
(paddle/easyocr/doctr) import their dependency lazily inside ``availability()`` and
``recognize()``, so this import is safe with none of them installed — and their
``parse()`` methods remain pure functions of stored payloads, testable from fixtures.
"""

from __future__ import annotations

from ocr_risk.engines import doctr as _doctr  # noqa: F401  (registration side effect)
from ocr_risk.engines import easyocr as _easyocr  # noqa: F401
from ocr_risk.engines import paddleocr as _paddleocr  # noqa: F401
from ocr_risk.engines import replay as _replay  # noqa: F401
from ocr_risk.engines import synthetic_family as _synthetic  # noqa: F401
from ocr_risk.engines import tesseract as _tesseract  # noqa: F401
from ocr_risk.engines.base import (
    Availability,
    OCREngineAdapter,
    PageInput,
    ParsedSpan,
    make_fingerprint,
)
from ocr_risk.engines.registry import (
    available_engines,
    build_engine,
    engine_availability,
    register_engine,
)

__all__ = [
    "Availability",
    "OCREngineAdapter",
    "PageInput",
    "ParsedSpan",
    "available_engines",
    "build_engine",
    "engine_availability",
    "make_fingerprint",
    "register_engine",
]
