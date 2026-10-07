"""Unicode normalization policy for OCR text.

Engines disagree about Unicode in ways that look cosmetic and are not: one emits
precomposed ``é`` (NFC), another the combining sequence (NFD), a third the ``ﬁ`` ligature
where the ground truth has ``fi``. Left alone, those differences become phantom character
errors in the alignment and phantom "corrections" in the candidate pool — an engine would
appear to make mistakes it never made.

The policy is applied identically to OCR text and ground truth, and its identifier is
recorded, so the choice is auditable rather than implicit.
"""

from __future__ import annotations

import unicodedata

__all__ = ["POLICIES", "apply_policy", "policy_description"]

# Ligatures Unicode's compatibility decomposition would handle, but which we expand even
# under the conservative NFC policy: no transcription convention in these corpora treats
# "ﬁ" and "fi" as different text, so keeping them distinct only manufactures errors.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

# Typographic variants that differ visually but not textually. Dashes and quotes are
# normalized because OCR engines choose between them inconsistently for the same glyph.
_PUNCTUATION = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
    " ": " ",
    " ": " ",
}

POLICIES: tuple[str, ...] = ("none", "nfc", "nfc_ligatures", "nfkc", "aggressive")


def apply_policy(text: str, policy: str) -> str:
    """Normalize ``text`` under a named policy.

    - ``none``           leave bytes untouched (for measuring what the policy costs)
    - ``nfc``            canonical composition only
    - ``nfc_ligatures``  NFC plus ligature expansion
    - ``nfkc``           compatibility composition (also folds superscripts, widths)
    - ``aggressive``     NFKC plus ligature and punctuation folding
    """
    if policy == "none":
        return text
    if policy == "nfc":
        return unicodedata.normalize("NFC", text)
    if policy == "nfc_ligatures":
        return unicodedata.normalize("NFC", _translate(text, _LIGATURES))
    if policy == "nfkc":
        return unicodedata.normalize("NFKC", text)
    if policy == "aggressive":
        folded = _translate(_translate(text, _LIGATURES), _PUNCTUATION)
        return unicodedata.normalize("NFKC", folded)
    known = ", ".join(POLICIES)
    msg = f"unknown unicode policy {policy!r}; known: {known}"
    raise ValueError(msg)


def _translate(text: str, table: dict[str, str]) -> str:
    if not any(ch in text for ch in table):
        return text
    return "".join(table.get(ch, ch) for ch in text)


def policy_description(policy: str) -> str:
    return {
        "none": "no normalization; raw engine bytes preserved",
        "nfc": "Unicode NFC canonical composition",
        "nfc_ligatures": "NFC plus ligature expansion (fi, fl, ffi, ...)",
        "nfkc": "Unicode NFKC compatibility composition",
        "aggressive": "NFKC plus ligature and typographic punctuation folding",
    }[policy]
