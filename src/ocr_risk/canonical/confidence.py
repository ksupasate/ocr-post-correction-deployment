"""Confidence scale registry and normalization.

Three quantities must never be confused, and this module exists to keep them apart:

===========================  =============================================================
quantity                     where it lives
===========================  =============================================================
``native_confidence``        ``CanonicalSpan.native_conf_*``, verbatim on the engine's own
                             scale, never rescaled or filled
``normalized_confidence``    a *derived feature* in the evidence layer; a linear map of
                             the native value onto [0, 1] for cross-engine comparability.
                             Comparable in units, NOT in meaning.
``calibrated_probability``   ``Prediction.calibrated_score``, produced by a calibrator that
                             was fitted on held-out data and frozen
===========================  =============================================================

Normalizing a confidence does **not** make it a probability. Tesseract's 87/100 and
PaddleOCR's 0.87 both normalize to 0.87 while meaning quite different things about how
often that span is correct — which is precisely the cross-engine shift this project
studies, so the distinction has to survive in the data model.
"""

from __future__ import annotations

from ocr_risk.schemas.base import ConfidenceScale

__all__ = ["SCALE_REGISTRY", "get_scale", "normalize_confidence", "register_scale"]

SCALE_REGISTRY: dict[str, ConfidenceScale] = {}


def register_scale(scale: ConfidenceScale) -> ConfidenceScale:
    """Register a native confidence scale, rejecting redefinition under the same name."""
    existing = SCALE_REGISTRY.get(scale.name)
    if existing is not None and existing != scale:
        msg = f"confidence scale {scale.name!r} is already registered with different bounds"
        raise ValueError(msg)
    SCALE_REGISTRY[scale.name] = scale
    return scale


def get_scale(name: str | None) -> ConfidenceScale | None:
    return SCALE_REGISTRY.get(name) if name else None


def normalize_confidence(value: float | None, scale_name: str | None) -> float | None:
    """Map a native confidence onto [0, 1] using its declared scale.

    Returns ``None`` when there is no value or no known scale: guessing a scale would
    silently invent a measurement. Values outside the declared range are clipped, since
    an out-of-range reading is an engine bug rather than information.
    """
    if value is None:
        return None
    scale = get_scale(scale_name)
    if scale is None:
        return None
    span = scale.maximum - scale.minimum
    unit = (value - scale.minimum) / span
    unit = min(max(unit, 0.0), 1.0)
    return unit if scale.higher_is_better else 1.0 - unit


def _register_builtin_scales() -> None:
    """Register the scales of the built-in adapters.

    Imported lazily inside the function so that registering a scale never drags an
    optional backend's module into a core import path.
    """
    from ocr_risk.engines.doctr import DOCTR_SCALE
    from ocr_risk.engines.easyocr import EASYOCR_SCALE
    from ocr_risk.engines.paddleocr import PADDLE_SCALE
    from ocr_risk.engines.synthetic_family import SCALES as SYNTHETIC_SCALES
    from ocr_risk.engines.tesseract import TESSERACT_SCALE

    for scale in (
        TESSERACT_SCALE,
        PADDLE_SCALE,
        EASYOCR_SCALE,
        DOCTR_SCALE,
        *SYNTHETIC_SCALES.values(),
    ):
        register_scale(scale)


_register_builtin_scales()
