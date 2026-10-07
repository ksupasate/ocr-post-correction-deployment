"""Adversarial candidates: linguistically plausible, not source-supported.

This generator exists to create the discrimination the project is about. A text-only
verifier sees a well-formed unit, a common surname, a valid alloy designation — and has
no basis to reject it. A verifier that reads the crop can see that the pixels say
something else.

The families are drawn from the failure modes where a wrong accepted edit actually costs
something:

======================  ============================  ===============================
family                  example                       why it matters
======================  ============================  ===============================
``numeric_magnitude``   ``0.015`` -> ``0.15``         an order of magnitude, in a dose
``unit_substitution``   ``mg`` -> ``ng``              a thousand-fold, and well-formed
``homoglyph``           ``Ti-6Al-4V`` -> ``Ti-6A1-4V``  a different alloy, one glyph away
``plausible_lexical``   ``Smith`` -> ``Smyth``        a real name, the wrong person
``truncation``          ``Ti-6Al-4V`` -> ``Ti-6Al-V``   silently drops a component
======================  ============================  ===============================

Every candidate is tagged ``is_synthetic_hard_negative`` so it stays separable from
naturally generated ones in every count. Because these are derived from the OCR text (not
from ground truth) they are safe to generate anywhere; the leakage risk is in how they are
*used*, and split scoping handles that (leakage vector L6).
"""

from __future__ import annotations

import re

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.registry import register_generator

__all__ = ["HARD_NEGATIVE_FAMILIES", "HardNegativeGenerator"]

HARD_NEGATIVE_FAMILIES = (
    "numeric_magnitude",
    "unit_substitution",
    "homoglyph",
    "plausible_lexical",
    "truncation",
)

# Units that differ by orders of magnitude but a single character.
_UNIT_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "mg": ("ng", "ug", "kg"),
    "ng": ("mg", "ug"),
    "ug": ("mg", "ng"),
    "kg": ("mg", "g"),
    "mL": ("L", "dL"),
    "L": ("mL",),
    "IU": ("U",),
}

# Homoglyph pairs, applied to a single occurrence at a time.
_HOMOGLYPHS: tuple[tuple[str, str], ...] = (
    ("l", "1"),
    ("1", "l"),
    ("O", "0"),
    ("0", "O"),
    ("I", "1"),
    ("S", "5"),
    ("5", "S"),
    ("B", "8"),
    ("Z", "2"),
)

# Surname variants that are themselves real names.
_NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    "Smith": ("Smyth", "Smithe"),
    "Rossi": ("Rossé", "Ross"),
    "Silva": ("Sylva",),
    "Petrov": ("Petroff",),
    "Dubois": ("Dubais",),
    "Andersen": ("Anderson",),
    "Nguyen": ("Ngyuen",),
}

_NUMBER = re.compile(r"\d*\.\d+|\d+")


@register_generator("hard_negative")
class HardNegativeGenerator(BaseGenerator):
    """Synthesizes plausible-but-wrong replacements from the OCR text."""

    generator_id = "hard_negative"
    version = "1"

    def __init__(
        self,
        generator_id: str = "hard_negative",
        families: tuple[str, ...] = HARD_NEGATIVE_FAMILIES,
        max_per_family: int = 1,
    ) -> None:
        unknown = set(families) - set(HARD_NEGATIVE_FAMILIES)
        if unknown:
            msg = f"unknown hard-negative families: {sorted(unknown)}"
            raise ValueError(msg)
        self.generator_id = generator_id
        self.families = tuple(families)
        self.max_per_family = max_per_family

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        original = context.original_ocr
        if not original.strip():
            return []

        proposals: list[CandidateProposal] = []
        for family in self.families:
            produced = 0
            for text in self._for_family(family, original):
                if text == original or any(p.text == text for p in proposals):
                    continue
                proposals.append(
                    CandidateProposal(
                        text=text,
                        score=None,
                        is_synthetic_hard_negative=True,
                        hard_negative_family=family,
                        metadata=self._meta(family=family, source="synthetic_adversarial"),
                    )
                )
                produced += 1
                if produced >= self.max_per_family or len(proposals) >= max_candidates:
                    break
            if len(proposals) >= max_candidates:
                break
        return proposals

    def _for_family(self, family: str, text: str) -> list[str]:
        if family == "numeric_magnitude":
            return _numeric_magnitude(text)
        if family == "unit_substitution":
            return _unit_substitution(text)
        if family == "homoglyph":
            return _homoglyph(text)
        if family == "plausible_lexical":
            return _plausible_lexical(text)
        if family == "truncation":
            return _truncation(text)
        return []


def _numeric_magnitude(text: str) -> list[str]:
    """Shift a decimal point: the same digits, a different quantity."""
    out: list[str] = []
    for match in _NUMBER.finditer(text):
        number = match.group()
        if "." not in number:
            continue
        whole, _, fraction = number.partition(".")
        if fraction:
            # 0.015 -> 0.15 : drop a leading fractional zero
            if fraction.startswith("0") and len(fraction) > 1:
                out.append(text[: match.start()] + f"{whole}.{fraction[1:]}" + text[match.end() :])
            # 0.015 -> 0.0015 : add one
            out.append(text[: match.start()] + f"{whole}.0{fraction}" + text[match.end() :])
    return out


def _unit_substitution(text: str) -> list[str]:
    out: list[str] = []
    for unit, neighbours in _UNIT_NEIGHBOURS.items():
        # Whole-token match only: substituting inside a word would produce nonsense
        # rather than a plausible alternative.
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(unit)}(?![A-Za-z])")
        if not pattern.search(text):
            continue
        out.extend(pattern.sub(neighbour, text, count=1) for neighbour in neighbours)
    return out


def _homoglyph(text: str) -> list[str]:
    out: list[str] = []
    for source, target in _HOMOGLYPHS:
        index = text.find(source)
        if index >= 0:
            out.append(text[:index] + target + text[index + 1 :])
    return out


def _plausible_lexical(text: str) -> list[str]:
    out: list[str] = []
    for name, variants in _NAME_VARIANTS.items():
        if name in text:
            out.extend(text.replace(name, variant, 1) for variant in variants)
    return out


def _truncation(text: str) -> list[str]:
    """Drop a hyphen-delimited component: still well-formed, and a different part."""
    parts = text.split("-")
    if len(parts) < 3:
        return []
    # Remove an interior component; removing the first or last is usually obvious.
    return ["-".join(parts[:index] + parts[index + 1 :]) for index in range(1, len(parts) - 1)]
