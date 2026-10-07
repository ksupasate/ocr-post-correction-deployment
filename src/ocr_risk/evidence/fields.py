"""The V0-V6 evidence configurations.

The ablation ladder is defined here, once, as data. Each configuration is a subset of
:class:`~ocr_risk.schemas.enums.EvidenceField`, and masking removes the excluded channels
from the bundle *before* the verifier is constructed — so an ablation cannot read a
channel it is meant to exclude, by oversight or otherwise.

The comparison the ladder exists for is **V3 against V6**: does the source image carry
information about whether an edit is beneficial, beyond what OCR text, context, and
confidence already provide? V4 (image alone) and V5 (image plus the two texts) are
included so a null V3-vs-V6 result is interpretable — without them, "pixels did not help"
could equally mean "pixels were never usable here".
"""

from __future__ import annotations

from dataclasses import dataclass

from ocr_risk.schemas.enums import EvidenceField

__all__ = ["EVIDENCE_CONFIGS", "EvidenceConfigSpec", "get_evidence_config"]

F = EvidenceField


@dataclass(frozen=True, slots=True)
class EvidenceConfigSpec:
    """One rung of the ablation ladder."""

    key: str
    label: str
    fields: frozenset[EvidenceField]
    rationale: str

    @property
    def uses_image(self) -> bool:
        return EvidenceField.IMAGE_CROP in self.fields

    @property
    def uses_text(self) -> bool:
        return bool(
            self.fields
            & {EvidenceField.ORIGINAL_OCR, EvidenceField.CANDIDATE_TEXT, EvidenceField.TEXT_CONTEXT}
        )


EVIDENCE_CONFIGS: dict[str, EvidenceConfigSpec] = {
    spec.key: spec
    for spec in (
        EvidenceConfigSpec(
            key="v0",
            label="accept everything",
            fields=frozenset(),
            rationale=(
                "Degenerate reference point: coverage 1.0 at whatever risk the candidate "
                "pool carries. Anchors every risk-coverage curve."
            ),
        ),
        EvidenceConfigSpec(
            key="v1",
            label="OCR confidence only",
            fields=frozenset({F.OCR_CONFIDENCE}),
            rationale=(
                "The signal an engine already gives away for free. Established prior art, "
                "and the bar any proposed verifier must clear."
            ),
        ),
        EvidenceConfigSpec(
            key="v2",
            label="text and context",
            fields=frozenset({F.ORIGINAL_OCR, F.CANDIDATE_TEXT, F.TEXT_CONTEXT}),
            rationale="What a language-only post-corrector can see.",
        ),
        EvidenceConfigSpec(
            key="v3",
            label="text, context, and confidence",
            fields=frozenset({F.ORIGINAL_OCR, F.CANDIDATE_TEXT, F.TEXT_CONTEXT, F.OCR_CONFIDENCE}),
            rationale=(
                "The strong non-visual baseline, and the left-hand side of the central "
                "comparison. If V6 does not beat this, source pixels add nothing."
            ),
        ),
        EvidenceConfigSpec(
            key="v4",
            label="image crop only",
            fields=frozenset({F.IMAGE_CROP}),
            rationale=(
                "Pixels with no text at all. Included so that a null V3-vs-V6 result can "
                "be attributed: it distinguishes 'pixels do not help' from 'pixels carry "
                "no usable signal in this corpus'."
            ),
        ),
        EvidenceConfigSpec(
            key="v5",
            label="image, original OCR, and candidate",
            fields=frozenset({F.IMAGE_CROP, F.ORIGINAL_OCR, F.CANDIDATE_TEXT}),
            rationale=(
                "The minimal source-grounded verifier: does this crop say O or Y? Isolates "
                "the grounding question from context and confidence."
            ),
        ),
        EvidenceConfigSpec(
            key="v6",
            label="full evidence",
            fields=frozenset(F),
            rationale="Everything a deployed system would have. The right-hand side of H2.",
        ),
        # --- SGV1 ladder (docs/sgv1/protocol.md section 7) ----------------------
        # The dual-frame study's primary comparison is sgv1_v2 vs sgv1_v1: same
        # architecture, same candidates, same fitting procedure -- the only difference
        # is the image channel. sgv1_v0 is the secondary text-only baseline, not a
        # straw man for the headline contrast.
        EvidenceConfigSpec(
            key="sgv1_v0",
            label="SGV1 V0: text only",
            fields=frozenset({F.ORIGINAL_OCR, F.CANDIDATE_TEXT, F.TEXT_CONTEXT}),
            rationale=(
                "Text-only secondary baseline. What a language model with no engine "
                "metadata and no pixels can decide about an edit."
            ),
        ),
        EvidenceConfigSpec(
            key="sgv1_v1",
            label="SGV1 V1: strong non-image",
            fields=frozenset(
                {
                    F.ORIGINAL_OCR,
                    F.CANDIDATE_TEXT,
                    F.TEXT_CONTEXT,
                    F.OCR_CONFIDENCE,
                    F.SPATIAL,
                }
            ),
            rationale=(
                "The strongest verifier allowed to see no source pixels: text, context, "
                "engine confidence, and geometry. The left-hand side of the primary "
                "SGV1 contrast; deliberately every non-image channel there is."
            ),
        ),
        EvidenceConfigSpec(
            key="sgv1_v2",
            label="SGV1 V2: source-grounded",
            fields=frozenset(F),
            rationale=(
                "Exactly sgv1_v1 plus the source-image crop. The right-hand side of the "
                "primary SGV1 contrast; any V2-over-V1 gain is attributable to the image "
                "channel because every other channel is shared."
            ),
        ),
    )
}


def get_evidence_config(key: str) -> EvidenceConfigSpec:
    try:
        return EVIDENCE_CONFIGS[key]
    except KeyError:
        known = ", ".join(sorted(EVIDENCE_CONFIGS))
        msg = f"unknown evidence configuration {key!r}; known: {known}"
        raise KeyError(msg) from None
