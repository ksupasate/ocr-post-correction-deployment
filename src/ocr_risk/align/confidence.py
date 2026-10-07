"""Alignment confidence, and the ambiguity decision that depends on it.

This is the guard against the failure mode that would quietly corrupt every downstream
number: **forcing a low-quality match to raise alignment coverage.** Higher coverage looks
like better infrastructure while manufacturing correction sites whose "ground truth" is an
artifact of a bad pairing — and those sites then pollute the harm labels and therefore the
headline risk.

Confidence blends three signals, because each fails on its own:

- ``char_agreement`` — how much of the text actually matches. Blind to whether a better
  partner existed.
- ``geom_score`` — whether the pixels line up. Unavailable for plain-text ground truth, and
  misleading when an engine's boxes are loose.
- ``uniqueness_margin`` — how much better this pairing is than the runner-up. This is the
  one that catches a confident-looking arbitrary choice between two similar candidates.

When ground truth has no geometry the geometric term is *dropped and the remaining weights
renormalized*, rather than being scored zero. Treating "unmeasurable" as "bad" would push
every plain-text corpus below the ambiguity floor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

__all__ = ["ConfidenceWeights", "component_confidence", "uniqueness_margin"]


@dataclass(frozen=True, slots=True)
class ConfidenceWeights:
    char_agreement: float = 0.5
    geometry: float = 0.3
    uniqueness: float = 0.2


def normalized_distance(a: str, b: str) -> float:
    """Levenshtein distance normalized by the longer string; 0.0 for two empty strings."""
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def uniqueness_margin(ocr_text: str, gt_text: str, alternatives: Sequence[str]) -> float:
    """How much better the chosen ground-truth text is than the next-best candidate.

    ``1.0`` when there is no alternative (unambiguous by construction) and ``0.0`` when an
    alternative fits at least as well — which is precisely the near-tie the aligner must
    not resolve silently.
    """
    if not alternatives:
        return 1.0
    chosen = normalized_distance(ocr_text, gt_text)
    runner_up = min(normalized_distance(ocr_text, alt) for alt in alternatives)
    return max(0.0, min(1.0, runner_up - chosen))


def component_confidence(
    char_agreement: float,
    geom_score: float | None,
    margin: float,
    weights: ConfidenceWeights,
) -> tuple[float, dict[str, str]]:
    """Combine the three signals, renormalizing when geometry is unavailable.

    Returns the confidence and diagnostics recording which terms contributed, so a low
    score can be explained rather than merely observed.
    """
    diagnostics: dict[str, str] = {
        "char_agreement": f"{char_agreement:.4f}",
        "margin": f"{margin:.4f}",
    }

    if geom_score is None:
        total = weights.char_agreement + weights.uniqueness
        diagnostics["geometry"] = "unavailable_weights_renormalized"
        if total <= 0:
            return 0.0, diagnostics
        score = (weights.char_agreement * char_agreement + weights.uniqueness * margin) / total
    else:
        total = weights.char_agreement + weights.geometry + weights.uniqueness
        diagnostics["geometry"] = f"{geom_score:.4f}"
        if total <= 0:
            return 0.0, diagnostics
        score = (
            weights.char_agreement * char_agreement
            + weights.geometry * geom_score
            + weights.uniqueness * margin
        ) / total

    return max(0.0, min(1.0, score)), diagnostics
