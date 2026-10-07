"""Turning masked evidence bundles into feature vectors.

Every featurizer reads **only** what the mask left in the bundle, so an ablation is
honoured mechanically rather than by convention. The image featurizer is the one worth
explaining: it computes cheap statistics over the crop rather than running a network,
because the bootstrap must prove the *architecture* on CPU without torch. A learned
encoder drops in at exactly this seam, and the V3-vs-V6 comparison it enables is the same
comparison either way — only the strength of the image channel changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from rapidfuzz.distance import Levenshtein

from ocr_risk.schemas.enums import EvidenceField
from ocr_risk.schemas.evidence import EvidenceBundle

__all__ = ["FeatureBlock", "confidence_block", "geometry_block", "image_block", "text_block"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    """A named group of features, so a coefficient can be traced back to its channel."""

    names: tuple[str, ...]
    values: FloatArray

    @staticmethod
    def empty() -> FeatureBlock:
        return FeatureBlock(names=(), values=np.zeros(0, dtype=np.float64))


def _missing(value: float | None, default: float = 0.0) -> tuple[float, float]:
    """Return ``(value, is_missing)``.

    A missing measurement is encoded as a value *plus an explicit indicator*, never as a
    sentinel. Imputing a bare 0.0 would let the model read "no confidence reported" as
    "confidence zero", which are different statements about the world.
    """
    return (default, 1.0) if value is None else (float(value), 0.0)


def text_block(bundle: EvidenceBundle) -> FeatureBlock:
    """Lexical relationship between the OCR span and the candidate."""
    original, candidate = bundle.original_ocr, bundle.candidate_text
    if not original and not candidate:
        return FeatureBlock.empty()

    edit_distance = float(Levenshtein.distance(original, candidate))
    longest = max(len(original), len(candidate), 1)
    digits_before = sum(c.isdigit() for c in original)
    digits_after = sum(c.isdigit() for c in candidate)

    values = [
        edit_distance,
        edit_distance / longest,
        float(len(original)),
        float(len(candidate)),
        float(len(candidate) - len(original)),
        float(digits_before),
        float(digits_after - digits_before),
        # A digit change is the signature of the magnitude errors this project cares
        # about most, so it gets its own feature rather than being buried in the distance.
        1.0 if digits_before != digits_after else 0.0,
        1.0 if original.lower() == candidate.lower() else 0.0,
        float(sum(not c.isalnum() for c in original)),
        float(len(bundle.context_before)),
        float(len(bundle.context_after)),
    ]
    names = (
        "text_edit_distance",
        "text_normalized_distance",
        "text_len_original",
        "text_len_candidate",
        "text_len_delta",
        "text_digits_original",
        "text_digit_delta",
        "text_digit_changed",
        "text_case_only_change",
        "text_punct_original",
        "text_context_before_len",
        "text_context_after_len",
    )
    return FeatureBlock(names=names, values=np.asarray(values, dtype=np.float64))


def confidence_block(bundle: EvidenceBundle) -> FeatureBlock:
    """Derived OCR-confidence signals."""
    features = bundle.conf_features
    if not features.has_native_confidence:
        return FeatureBlock(names=("conf_missing",), values=np.asarray([1.0], dtype=np.float64))

    normalized, normalized_missing = _missing(features.conf_normalized)
    zscore, zscore_missing = _missing(features.conf_zscore_within_engine)
    values = [normalized, normalized_missing, zscore, zscore_missing, float(features.n_spans), 0.0]
    names = (
        "conf_normalized",
        "conf_normalized_missing",
        "conf_zscore",
        "conf_zscore_missing",
        "conf_n_spans",
        "conf_missing",
    )
    return FeatureBlock(names=names, values=np.asarray(values, dtype=np.float64))


def geometry_block(bundle: EvidenceBundle) -> FeatureBlock:
    """Scale-free spatial signals."""
    features = bundle.geom_features
    if not features.has_geometry:
        return FeatureBlock(names=("geom_missing",), values=np.asarray([1.0], dtype=np.float64))

    width, _ = _missing(features.width)
    height, _ = _missing(features.height)
    aspect, aspect_missing = _missing(features.aspect_ratio)
    values = [
        width,
        height,
        aspect,
        aspect_missing,
        *_missing(features.relative_x)[:1],
        *_missing(features.relative_y)[:1],
        *_missing(features.relative_area)[:1],
        0.0,
    ]
    names = (
        "geom_width",
        "geom_height",
        "geom_aspect",
        "geom_aspect_missing",
        "geom_rel_x",
        "geom_rel_y",
        "geom_rel_area",
        "geom_missing",
    )
    return FeatureBlock(names=names, values=np.asarray(values, dtype=np.float64))


def image_block(crop: Path | None) -> FeatureBlock:
    """Cheap statistics over the source crop.

    Deliberately not a learned encoder: the bootstrap must prove the architecture on CPU
    with no torch. These statistics do carry real signal — ink coverage and stroke density
    differ measurably between a span that reads ``0.015`` and one that reads ``0.15`` —
    but they are a floor, not a ceiling. Replacing this function with a learned encoder is
    the intended upgrade path, and nothing else in the pipeline changes.
    """
    names = (
        "img_missing",
        "img_mean",
        "img_std",
        "img_ink_fraction",
        "img_row_ink_std",
        "img_col_ink_std",
        "img_edge_density",
        "img_aspect",
    )
    if crop is None or not crop.is_file():
        values = np.zeros(len(names), dtype=np.float64)
        values[0] = 1.0
        return FeatureBlock(names=names, values=values)

    with Image.open(crop) as opened:
        array = np.asarray(opened.convert("L"), dtype=np.float64) / 255.0

    # Ink is dark on light, so invert before measuring coverage.
    ink = 1.0 - array
    threshold = float(np.mean(ink)) + float(np.std(ink))
    binary = (ink > threshold).astype(np.float64)

    values = np.asarray(
        [
            0.0,
            float(np.mean(array)),
            float(np.std(array)),
            float(np.mean(binary)),
            float(np.std(binary.mean(axis=1))) if binary.shape[0] > 1 else 0.0,
            float(np.std(binary.mean(axis=0))) if binary.shape[1] > 1 else 0.0,
            float(np.mean(np.abs(np.diff(array, axis=1)))) if array.shape[1] > 1 else 0.0,
            float(array.shape[1] / max(array.shape[0], 1)),
        ],
        dtype=np.float64,
    )
    return FeatureBlock(names=names, values=values)


def blocks_for(
    bundle: EvidenceBundle, allowed: frozenset[EvidenceField], crop: Path | None
) -> list[FeatureBlock]:
    """Assemble the blocks this ablation permits, in a stable order.

    Order is fixed so a fitted model's coefficient vector keeps its meaning across runs.
    """
    blocks: list[FeatureBlock] = []
    if allowed & {
        EvidenceField.ORIGINAL_OCR,
        EvidenceField.CANDIDATE_TEXT,
        EvidenceField.TEXT_CONTEXT,
    }:
        blocks.append(text_block(bundle))
    if EvidenceField.OCR_CONFIDENCE in allowed:
        blocks.append(confidence_block(bundle))
    if EvidenceField.SPATIAL in allowed:
        blocks.append(geometry_block(bundle))
    if EvidenceField.IMAGE_CROP in allowed:
        blocks.append(image_block(crop))
    return blocks
