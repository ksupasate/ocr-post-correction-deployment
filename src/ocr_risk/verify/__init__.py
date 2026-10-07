"""Edit verifiers scoring masked evidence bundles (ablation configurations V0-V6).

Layer 7. Fitting happens only through a ``SplitPlan`` view; this package may not import
``metrics``, so evaluation labels are unreachable from here by construction.
"""

from __future__ import annotations

from ocr_risk.verify import feature_verifier as _feature  # noqa: F401  (registration)
from ocr_risk.verify import torch_slot as _torch  # noqa: F401
from ocr_risk.verify.base import ScoredBatch, VerificationInput, Verifier
from ocr_risk.verify.feature_verifier import (
    AcceptAllVerifier,
    FeatureVerifier,
    NotFittedVerifierError,
)
from ocr_risk.verify.featurizers import FeatureBlock, blocks_for
from ocr_risk.verify.registry import available_verifiers, build_verifier, register_verifier

__all__ = [
    "AcceptAllVerifier",
    "FeatureBlock",
    "FeatureVerifier",
    "NotFittedVerifierError",
    "ScoredBatch",
    "VerificationInput",
    "Verifier",
    "available_verifiers",
    "blocks_for",
    "build_verifier",
    "register_verifier",
]
