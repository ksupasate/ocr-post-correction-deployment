"""Raw engine responses to canonical spans: reading order, confidence, Unicode policy.

Layer 3. This is where cross-engine policy lives, so that no adapter makes a scientific
decision on its own and a policy change re-applies to every engine from preserved raw
output.
"""

from __future__ import annotations

from ocr_risk.canonical.confidence import (
    SCALE_REGISTRY,
    get_scale,
    normalize_confidence,
    register_scale,
)
from ocr_risk.canonical.normalize import (
    CanonicalizationPolicy,
    LinearizedText,
    canonicalize_response,
    linearize,
    rebuild_stream,
)
from ocr_risk.canonical.reading_order import assign_lines, derive_reading_order
from ocr_risk.canonical.unicode_policy import POLICIES, apply_policy, policy_description

__all__ = [
    "POLICIES",
    "SCALE_REGISTRY",
    "CanonicalizationPolicy",
    "LinearizedText",
    "apply_policy",
    "assign_lines",
    "canonicalize_response",
    "derive_reading_order",
    "get_scale",
    "linearize",
    "normalize_confidence",
    "policy_description",
    "rebuild_stream",
    "register_scale",
]
