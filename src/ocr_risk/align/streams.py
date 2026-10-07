"""Character streams with back-maps to the spans and tokens that produced them.

Alignment happens over characters, but everything downstream — correction sites, edits,
harm labels — is about spans. The back-map is the bridge: every character knows which item
it came from, so a character-level path projects cleanly onto a span-level correspondence.

Separator characters belong to no item and map to ``-1``. That matters: a merged OCR token
lacks the space its ground truth has, and it is exactly that unowned character which lets
the aligner see a MERGE rather than a substitution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ocr_risk.schemas.documents import GtToken
from ocr_risk.schemas.spans import CanonicalSpan

__all__ = ["Stream", "gt_stream", "ocr_stream"]

SEPARATOR = " "
NO_OWNER = -1


@dataclass(frozen=True, slots=True)
class Stream:
    """A character string plus, for each character, the index of its owning item."""

    text: str
    owner: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.text) != len(self.owner):
            msg = f"stream back-map length {len(self.owner)} != text length {len(self.text)}"
            raise ValueError(msg)

    def owners_in(self, start: int, end: int) -> set[int]:
        """Distinct item indices covering the half-open character range."""
        return {o for o in self.owner[start:end] if o != NO_OWNER}


def _build(texts: Sequence[str]) -> Stream:
    parts: list[str] = []
    owner: list[int] = []
    for index, text in enumerate(texts):
        if index:
            parts.append(SEPARATOR)
            owner.append(NO_OWNER)
        parts.append(text)
        owner.extend([index] * len(text))
    return Stream(text="".join(parts), owner=tuple(owner))


def ocr_stream(spans: Sequence[CanonicalSpan]) -> Stream:
    """Character stream over OCR spans, in the order given."""
    return _build([span.text for span in spans])


def gt_stream(tokens: Sequence[GtToken]) -> Stream:
    """Character stream over ground-truth tokens, in the order given."""
    return _build([token.text for token in tokens])
