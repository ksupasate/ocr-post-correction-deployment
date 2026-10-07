"""Applying accepted edits to a transcription.

Needed for the document-level metrics: CER and WER of the *corrected* output cannot be
computed from site-level counts alone, because edits interact through the text they share.
Applying them explicitly, right to left, keeps the offsets of not-yet-applied edits valid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

__all__ = ["AppliedEdit", "apply_edits"]


@dataclass(frozen=True, slots=True)
class AppliedEdit:
    """One accepted replacement over a half-open character range."""

    char_start: int
    char_end: int
    replacement: str
    site_id: str = ""


def apply_edits(text: str, edits: Sequence[AppliedEdit]) -> str:
    """Apply non-overlapping replacements to ``text``.

    Applied in descending start order so each replacement leaves the offsets of the ones
    still to come untouched. Overlapping edits are a programming error, not a data
    condition — the decision policy accepts at most one candidate per site, and sites do
    not overlap — so they raise rather than being silently resolved by precedence.
    """
    if not edits:
        return text

    ordered = sorted(edits, key=lambda e: (e.char_start, e.char_end))
    for previous, current in pairwise(ordered):
        if current.char_start < previous.char_end:
            msg = (
                f"overlapping edits at [{previous.char_start},{previous.char_end}) and "
                f"[{current.char_start},{current.char_end}); sites must be disjoint"
            )
            raise ValueError(msg)

    result = text
    for edit in reversed(ordered):
        if not 0 <= edit.char_start <= edit.char_end <= len(result):
            msg = (
                f"edit range [{edit.char_start},{edit.char_end}) is outside a text of "
                f"length {len(result)}"
            )
            raise ValueError(msg)
        result = result[: edit.char_start] + edit.replacement + result[edit.char_end :]
    return result
