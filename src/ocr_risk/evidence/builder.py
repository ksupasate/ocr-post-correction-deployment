"""Building evidence bundles — the ground-truth-blind view of a candidate edit.

The invariant this module exists to enforce: **the verifier sees only what a deployed
system would see.** ``EvidenceBuilder.build`` accepts an
:class:`~ocr_risk.schemas.evidence.ObservationView`, which structurally has no ground-truth
field, so leaking the answer into the features is a type error rather than a discipline
failure (leakage vector L9).

Context windows come from the OCR stream, never from ground truth. That is easy to get
wrong and impossible to detect downstream: a verifier reading correct surrounding text
would learn to trust candidates that merely *fit* text it should not have had.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ocr_risk.evidence.crops import CropPolicy, build_recipe
from ocr_risk.evidence.features_conf import (
    ConfidenceFeaturizer,
    unfitted_confidence_features,
)
from ocr_risk.evidence.features_geom import geometry_features
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.enums import EvidenceField
from ocr_risk.schemas.evidence import ConfidenceFeatures, EvidenceBundle, ObservationView

__all__ = ["EvidenceBuilder", "context_window", "unfitted_confidence_features"]


def context_window(stream: str, char_start: int, char_end: int, width: int) -> tuple[str, str]:
    """Text before and after a span, taken from the **OCR** stream.

    Never from ground truth. Doing so would hand the verifier correct neighbouring text a
    deployed system would not have, and it would learn to accept candidates that fit that
    phantom context.
    """
    if width <= 0:
        return "", ""
    before = stream[max(0, char_start - width) : char_start]
    after = stream[char_end : char_end + width]
    return before, after


@dataclass(slots=True)
class EvidenceBuilder:
    """Turns observations into evidence bundles.

    The confidence featurizer is a *fitted* transformer: population statistics (z-scores)
    come from the fit split only, so no test-split information reaches a feature
    (leakage vector L3).
    """

    crop_policy: CropPolicy
    context_chars: int
    confidence_featurizer: ConfidenceFeaturizer | None = None
    """``None`` yields every confidence feature except the z-score.

    A z-score needs a population, and under leave-one-engine-out the admissible
    population differs per fold: the fit engines of fold *h* include the engine held out
    in fold *h'*. There is therefore no fold-independent z-score, and a bundle built once
    and shared across folds must not carry one. The per-fold value is computed inside the
    fold, where the fit scope is known.
    """

    def build(
        self,
        view: ObservationView,
        *,
        ocr_stream: str,
        char_start: int,
        char_end: int,
        native_confidences: Sequence[float | None],
        conf_scale: str | None,
        page_bbox: BBox | None = None,
    ) -> EvidenceBundle:
        """Assemble one candidate's evidence. ``view`` carries no ground truth."""
        before, after = context_window(ocr_stream, char_start, char_end, self.context_chars)

        conf_features = (
            self.confidence_featurizer.transform(
                native_confidences, conf_scale, engine_id=view.engine_id
            )
            if self.confidence_featurizer is not None
            else unfitted_confidence_features(native_confidences, conf_scale)
        )
        geom_features = geometry_features(
            view.bbox, view.image_width, view.image_height, n_spans=len(view.ocr_span_ids)
        )

        recipe = (
            build_recipe(view.image_sha256, view.bbox, self.crop_policy)
            if view.bbox is not None
            else None
        )

        available = self._available_fields(
            has_crop=recipe is not None,
            has_confidence=conf_features.has_native_confidence,
            has_geometry=geom_features.has_geometry,
            has_context=bool(before or after),
        )

        return EvidenceBundle(
            candidate_id=view.candidate_id,
            site_id=view.site_id,
            document_id=view.document_id,
            dataset_id=view.dataset_id,
            engine_id=view.engine_id,
            original_ocr=view.original_ocr,
            candidate_text=view.candidate_text,
            context_before=before,
            context_after=after,
            crop_recipe_sha256=recipe.recipe_sha256 if recipe else None,
            conf_features=conf_features,
            geom_features=geom_features,
            available_fields=available,
            masked_fields=(),
        )

    @staticmethod
    def _available_fields(
        *, has_crop: bool, has_confidence: bool, has_geometry: bool, has_context: bool
    ) -> tuple[EvidenceField, ...]:
        """Which channels are actually populated for this row.

        Distinguishes "the engine reported no confidence" from "this ablation masked
        confidence away" — a distinction the analysis needs in order to attribute a
        verifier's failure to the right cause.
        """
        fields = [EvidenceField.ORIGINAL_OCR, EvidenceField.CANDIDATE_TEXT]
        if has_crop:
            fields.append(EvidenceField.IMAGE_CROP)
        if has_context:
            fields.append(EvidenceField.TEXT_CONTEXT)
        if has_confidence:
            fields.append(EvidenceField.OCR_CONFIDENCE)
        if has_geometry:
            fields.append(EvidenceField.SPATIAL)
        return tuple(sorted(fields, key=lambda f: f.value))


def empty_confidence_features(n_spans: int) -> ConfidenceFeatures:
    """Confidence features for an engine that reported nothing."""
    return ConfidenceFeatures(n_spans=n_spans, has_native_confidence=False)
