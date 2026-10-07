"""SGV1 frozen design constants: the verifier ladder and the evaluation frames.

The evidence subsets themselves live where all evidence configurations live,
:mod:`ocr_risk.evidence.fields` (``sgv1_v0`` / ``sgv1_v1`` / ``sgv1_v2``). This module
binds them to their scientific roles and defines the frame vocabulary every SGV1 table
carries, so "which comparison is primary" and "which frame is this row from" are read
from code, not from a document's prose.

Frame vocabulary (docs/sgv1/protocol.md sections 2, 9, 5):

- ``natural``                  -- Frame B and Frame A's source stream; real prevalence.
- ``controlled_degradation``   -- the mechanistic degradation track; never pooled with
  natural into any headline number.
- ``challenge``                -- deliberately constructed or synthetic diagnostic
  material; labelled, never mixed into natural prevalence claims.
"""

from __future__ import annotations

from ocr_risk.evidence.fields import get_evidence_config
from ocr_risk.schemas.enums import EvidenceField

__all__ = [
    "FRAME_CHALLENGE",
    "FRAME_DEGRADATION",
    "FRAME_NATURAL",
    "PRIMARY_CONTRAST",
    "SGV1_LADDER",
    "VALID_FRAMES",
    "validate_ladder",
]

FRAME_NATURAL = "natural"
FRAME_DEGRADATION = "controlled_degradation"
FRAME_CHALLENGE = "challenge"
VALID_FRAMES = frozenset({FRAME_NATURAL, FRAME_DEGRADATION, FRAME_CHALLENGE})

PRIMARY_CONTRAST: tuple[str, str] = ("sgv1_v2", "sgv1_v1")
"""The SGV1-H1 comparison: source-grounded vs strong non-image. Everything else is
secondary or mechanistic."""

SGV1_LADDER: dict[str, str] = {
    "sgv1_v0": "text-only secondary baseline",
    "sgv1_v1": "strong non-image baseline (primary contrast, left)",
    "sgv1_v2": "source-grounded verifier (primary contrast, right)",
}


def validate_ladder() -> None:
    """Fail loudly if the SGV1 subsets stop nesting the way the protocol requires.

    The protocol's causal claim ("the only intended difference is the image channel")
    is only true while ``sgv1_v0 ⊂ sgv1_v1 ⊂ sgv1_v2`` with the image channel absent
    everywhere but ``sgv1_v2``. Called by tests, not at import time, so a bad edit
    fails the build rather than every unrelated import.
    """
    v0 = get_evidence_config("sgv1_v0").fields
    v1 = get_evidence_config("sgv1_v1").fields
    v2 = get_evidence_config("sgv1_v2").fields
    if not v0 < v1 < v2:
        msg = f"SGV1 ladder must nest strictly: v0={sorted(v0)}, v1={sorted(v1)}, v2={sorted(v2)}"
        raise AssertionError(msg)
    image_only_v2 = {
        k: EvidenceField.IMAGE_CROP in get_evidence_config(k).fields for k in SGV1_LADDER
    }
    if image_only_v2 != {"sgv1_v0": False, "sgv1_v1": False, "sgv1_v2": True}:
        msg = f"image channel must appear in sgv1_v2 alone: {image_only_v2}"
        raise AssertionError(msg)
