"""Evidence bundles: everything a verifier is allowed to see.

**No type in this module may carry ground truth.** The verifier's job is to judge an edit
from the same information a deployed system would have — OCR text, the candidate, the
source pixels, context, confidence, and geometry. A ground-truth field here would leak the
answer into the features, so ``tests/architecture/test_evidence_has_no_gt.py`` inspects
these classes by reflection and fails the build if one appears.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field

from ocr_risk.schemas.base import BBox, RecordModel
from ocr_risk.schemas.enums import EvidenceField

__all__ = [
    "GT_FORBIDDEN_FIELD_COMPONENTS",
    "GT_FORBIDDEN_FIELD_SUBSTRINGS",
    "ConfidenceFeatures",
    "CropRecipe",
    "EvidenceBundle",
    "GeometryFeatures",
    "ObservationView",
    "field_names_leaking_ground_truth",
]

GT_FORBIDDEN_FIELD_COMPONENTS: frozenset[str] = frozenset({"gt", "groundtruth", "label", "outcome"})
"""Underscore-separated name components that would indicate a ground-truth field.

Matched per component, not as substrings: ``gt_text`` is forbidden, while ``target_height``
(a crop resize parameter) is not. Substring matching produced exactly that false positive,
and a check that cries wolf is a check people learn to silence.
"""

GT_FORBIDDEN_FIELD_SUBSTRINGS: frozenset[str] = frozenset(
    {"ground_truth", "d_before", "d_after", "harmful"}
)
"""Names unambiguous enough to match anywhere in a field name."""


def field_names_leaking_ground_truth(field_names: Iterable[str]) -> list[str]:
    """Return the field names that look like they carry ground truth."""
    offenders: list[str] = []
    for name in field_names:
        lowered = name.lower()
        components = set(lowered.split("_"))
        if components & GT_FORBIDDEN_FIELD_COMPONENTS or any(
            token in lowered for token in GT_FORBIDDEN_FIELD_SUBSTRINGS
        ):
            offenders.append(name)
    return offenders


class CropRecipe(RecordModel):
    """A reproducible description of an image crop — the recipe, not the pixels.

    Storing recipes instead of materialized crops keeps licensed imagery out of the
    artifact tree, keeps the artifacts small, and still reconstructs bit-identical
    evidence years later from the same source image (ADR-005).
    """

    image_sha256: str
    bbox: BBox
    padding_ratio: float = Field(ge=0.0)
    """Padding as a fraction of the box's larger side. Context around a span is often what
    disambiguates an edit, so the amount is an experimental parameter, not a constant."""
    padding_min_px: int = Field(ge=0)
    target_height: int | None = None
    grayscale: bool = False
    recipe_sha256: str
    """Canonical hash of every other field; the cache key and the value stored on rows."""


class ConfidenceFeatures(RecordModel):
    """Derived confidence signals.

    Separate from the native values on ``CanonicalSpan``, which are never overwritten.
    Every statistic here that requires a population (z-scores, ranks) must be computed by
    a transformer fitted on the ``fit`` split only.
    """

    native_min: float | None = None
    native_mean: float | None = None
    conf_normalized: float | None = None
    """Native confidence mapped to [0, 1] using the engine's declared scale."""
    conf_rank_in_line: float | None = None
    conf_zscore_within_engine: float | None = None
    n_spans: int = Field(ge=0)
    has_native_confidence: bool


class GeometryFeatures(RecordModel):
    """Derived spatial signals for the edited region."""

    width: float | None = None
    height: float | None = None
    aspect_ratio: float | None = None
    relative_x: float | None = None
    relative_y: float | None = None
    relative_area: float | None = None
    n_spans: int = Field(ge=0)
    has_geometry: bool


class ObservationView(RecordModel):
    """The ground-truth-free input to evidence construction.

    ``EvidenceBuilder`` accepts only this type. Because it structurally lacks any label
    field, building evidence from ground truth is a type error rather than a discipline
    failure (leakage vector L9).
    """

    site_id: str
    candidate_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    original_ocr: str
    candidate_text: str
    ocr_span_ids: tuple[str, ...]
    image_sha256: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    bbox: BBox | None = None


class EvidenceBundle(RecordModel):
    """Everything a verifier may consume for one candidate edit.

    Ablation configurations V0-V6 are subsets of :class:`~ocr_risk.schemas.enums.
    EvidenceField`; masking removes channels from the bundle before the verifier is
    constructed, so an ablation cannot accidentally read a channel it is meant to exclude.
    """

    candidate_id: str
    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    original_ocr: str
    candidate_text: str
    context_before: str
    context_after: str
    """Surrounding text taken from the OCR stream, never from ground truth."""
    crop_recipe_sha256: str | None = None
    conf_features: ConfidenceFeatures
    geom_features: GeometryFeatures
    available_fields: tuple[EvidenceField, ...]
    """Channels actually populated for this row. Distinguishes "the engine gave no
    confidence" from "this ablation masked confidence away"."""
    masked_fields: tuple[EvidenceField, ...] = ()
