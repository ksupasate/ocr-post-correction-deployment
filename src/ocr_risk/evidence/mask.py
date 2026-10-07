"""Evidence masking: making an ablation structural rather than disciplinary.

An ablation implemented as "the V3 verifier promises not to look at the crop" is a
convention, and conventions decay. Here the excluded channels are *removed from the
bundle* before the verifier is constructed, so V3 cannot read pixels even if a later
featurizer change would happily use them.

Masked channels are blanked, and recorded in ``masked_fields``. Recording matters: it
keeps "this ablation removed confidence" distinguishable from "this engine never reported
confidence", which the analysis needs in order to attribute a verifier's failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from ocr_risk.evidence.fields import EvidenceConfigSpec, get_evidence_config
from ocr_risk.schemas.enums import EvidenceField
from ocr_risk.schemas.evidence import ConfidenceFeatures, EvidenceBundle, GeometryFeatures

__all__ = ["EvidenceMask", "MaskedFieldAccessError"]


class MaskedFieldAccessError(RuntimeError):
    """Raised when code reaches for a channel this ablation removed."""


@dataclass(frozen=True, slots=True)
class EvidenceMask:
    """Applies one ablation configuration to evidence bundles."""

    spec: EvidenceConfigSpec

    @classmethod
    def from_key(cls, key: str) -> EvidenceMask:
        return cls(spec=get_evidence_config(key))

    @property
    def key(self) -> str:
        return self.spec.key

    def allows(self, field: EvidenceField) -> bool:
        return field in self.spec.fields

    def require(self, field: EvidenceField) -> None:
        """Assert a channel is permitted, for code paths that cannot express it in types."""
        if not self.allows(field):
            msg = (
                f"evidence configuration {self.spec.key!r} ({self.spec.label}) excludes "
                f"{field.value!r}; reading it would silently break the ablation"
            )
            raise MaskedFieldAccessError(msg)

    def apply(self, bundle: EvidenceBundle) -> EvidenceBundle:
        """Return a bundle with every excluded channel blanked and recorded."""
        allowed = self.spec.fields
        masked = tuple(
            sorted((f for f in EvidenceField if f not in allowed), key=lambda f: f.value)
        )

        updates: dict[str, object] = {"masked_fields": masked}

        if EvidenceField.ORIGINAL_OCR not in allowed:
            updates["original_ocr"] = ""
        if EvidenceField.CANDIDATE_TEXT not in allowed:
            updates["candidate_text"] = ""
        if EvidenceField.TEXT_CONTEXT not in allowed:
            updates["context_before"] = ""
            updates["context_after"] = ""
        if EvidenceField.IMAGE_CROP not in allowed:
            updates["crop_recipe_sha256"] = None
        if EvidenceField.OCR_CONFIDENCE not in allowed:
            updates["conf_features"] = ConfidenceFeatures(
                n_spans=bundle.conf_features.n_spans, has_native_confidence=False
            )
        if EvidenceField.SPATIAL not in allowed:
            updates["geom_features"] = GeometryFeatures(
                n_spans=bundle.geom_features.n_spans, has_geometry=False
            )

        # available_fields must narrow too, so a downstream reader cannot conclude a
        # channel is present merely because the engine originally provided it.
        updates["available_fields"] = tuple(f for f in bundle.available_fields if f in allowed)
        return bundle.model_copy(update=updates)
