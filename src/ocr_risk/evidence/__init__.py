"""Ground-truth-blind evidence bundles, crop recipes, and ablation masking.

Layer 6. **No type in this package may carry ground truth.** ``EvidenceBuilder`` accepts
an ``ObservationView``, which structurally lacks any label field, so leaking the answer
into the features is a type error rather than a discipline failure. Enforced by
``tests/architecture/test_evidence_has_no_gt.py``.
"""

from __future__ import annotations

from ocr_risk.evidence.builder import EvidenceBuilder, context_window
from ocr_risk.evidence.crops import CropPolicy, build_recipe, crop_path, materialize
from ocr_risk.evidence.features_conf import ConfidenceFeaturizer, NotFittedError
from ocr_risk.evidence.features_geom import geometry_features
from ocr_risk.evidence.fields import EVIDENCE_CONFIGS, EvidenceConfigSpec, get_evidence_config
from ocr_risk.evidence.mask import EvidenceMask, MaskedFieldAccessError

__all__ = [
    "EVIDENCE_CONFIGS",
    "ConfidenceFeaturizer",
    "CropPolicy",
    "EvidenceBuilder",
    "EvidenceConfigSpec",
    "EvidenceMask",
    "MaskedFieldAccessError",
    "NotFittedError",
    "build_recipe",
    "context_window",
    "crop_path",
    "geometry_features",
    "get_evidence_config",
    "materialize",
]
