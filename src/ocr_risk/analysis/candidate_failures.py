"""Why a proposed edit was harmful: a taxonomy over ``(O, Y, G)`` and the site.

The H1 pilot reported that 73%-88% of the natural candidate pool was harmful and stopped
there. One number cannot tell a fixable generator defect from a structural one, and the
two call for opposite responses: the first says build a better corrector, the second says
this pool cannot be verified into usefulness no matter how good the verifier is.

The classes below are ordered by precedence and assigned by first match, so every proposal
lands in exactly one. Precedence runs structural first — a proposal on an already-correct
span is that, whatever else is also true of it — because the structural classes are the
ones a verifier cannot rescue.

This is **offline analysis**. It reads ground truth, and it runs downstream of generation;
no generator sees any of it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

__all__ = ["FAILURE_CLASSES", "classify_failure", "failure_table", "stratified_failure_table"]

_PUNCTUATION = ".,;:!?()[]{}\"'-" + "\u2013\u2014"  # hyphen, en dash, em dash

_UNITS = frozenset(
    {
        "mg",
        "ng",
        "ug",
        "g",
        "kg",
        "ml",
        "l",
        "cl",
        "dl",
        "mm",
        "cm",
        "m",
        "km",
        "%",
        "usd",
        "idr",
        "rp",
        "eur",
        "gbp",
        "pcs",
        "pc",
        "x",
        "hr",
        "hrs",
        "min",
    }
)
_DIGITS = re.compile(r"\d")
_WORD = re.compile(r"^\w+$", re.UNICODE)

FAILURE_CLASSES: tuple[str, ...] = (
    "beneficial",
    "source_already_correct",
    "site_needs_deletion",
    "site_needs_insertion",
    "numeric_corruption",
    "unit_corruption",
    "capitalized_token_rewrite",
    "punctuation_only",
    "token_merge_or_split",
    "worsened_existing_error",
    "no_improvement",
    "other",
)


def _is_numeric(text: str) -> bool:
    return bool(_DIGITS.search(text))


def _tokens(text: str) -> list[str]:
    return text.split()


def _looks_like_entity(text: str) -> bool:
    """A Capitalized alphabetic token — initial upper, remainder not all upper.

    Deliberately crude and deliberately **not** a named-entity recognizer. Running one here
    would put a third model's error rate inside a diagnostic, and the diagnostic is meant to
    localize a problem, not to be a measurement of entity handling.

    ``not text.isupper()`` is load-bearing on this corpus. Without it every ALL-CAPS token
    qualified, and form and receipt text is full of them: 34% of the class was pairs like
    ``TKE -> DUE`` and ``TUIS -> TBIS``, which are character corruptions and belong in
    ``worsened_existing_error``.
    """
    return (
        bool(text) and text[0].isupper() and not text.isupper() and text.isalpha() and len(text) > 2
    )


def classify_failure(original: str, candidate: str, gt: str, site_kind: str, delta: int) -> str:
    """One class per proposal, by first match in precedence order.

    Structural classes come **first**, before the beneficial check. The docstring said so
    from the start and the code did not: `delta > 0` was tested first, so at a site whose
    ground truth is empty every shorter wrong string — `243,000` proposed as `23,000` —
    scored `beneficial` rather than `site_needs_deletion`, understating the structural
    class by 286 proposals and overstating the beneficial one by the same amount.
    """
    # --- structural: the site makes a substitution the wrong SHAPE of edit --------------
    if not gt:
        # OCR produced text with no ground-truth counterpart; the correct edit is deletion,
        # and only the empty candidate performs it.
        return "beneficial" if candidate == "" else "site_needs_deletion"
    if not original:
        return "site_needs_insertion"
    if original == gt:
        return "source_already_correct"

    if delta > 0:
        return "beneficial"

    # --- content: the OCR was wrong, and the proposal was wrong in a describable way ----
    stripped_original = original.strip(_PUNCTUATION)
    stripped_candidate = candidate.strip(_PUNCTUATION)
    if stripped_original == stripped_candidate and original != candidate:
        return "punctuation_only"

    if len(_tokens(original)) != len(_tokens(candidate)):
        return "token_merge_or_split"

    if _is_numeric(original) or _is_numeric(candidate):
        digits_before = "".join(c for c in original if c.isdigit())
        digits_after = "".join(c for c in candidate if c.isdigit())
        if digits_before != digits_after:
            return "numeric_corruption"

    if original.lower() in _UNITS or candidate.lower() in _UNITS:
        return "unit_corruption"

    if _looks_like_entity(original) and _looks_like_entity(candidate):
        return "capitalized_token_rewrite"

    if delta < 0:
        return "worsened_existing_error"
    return "no_improvement" if _WORD.match(candidate or " ") else "other"


def failure_table(proposals: pd.DataFrame) -> pd.DataFrame:
    """Count and share of every failure class, overall and per generator."""
    frame = proposals.copy()
    frame["failure_class"] = [
        classify_failure(
            str(row.original_ocr or ""),
            str(row.candidate_text or ""),
            str(row.gt_text or ""),
            str(row.site_kind),
            int(str(row.delta)),
        )
        for row in frame.itertuples()
    ]
    counts = frame.groupby(["generator_id", "failure_class"]).size().rename("n").reset_index()
    totals = counts.groupby("generator_id")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    order = {name: index for index, name in enumerate(FAILURE_CLASSES)}
    counts["_order"] = counts["failure_class"].map(order).fillna(len(order))
    return (
        counts.sort_values(["generator_id", "_order"]).drop(columns="_order").reset_index(drop=True)
    )


@dataclass(frozen=True, slots=True)
class Stratum:
    """One way of slicing the proposals, named so the table can be read without the code."""

    name: str
    column: str


def _confidence_band(value: float | None) -> str:
    """Three bands, not a continuous split, because the point is legibility.

    A missing confidence is its own band rather than being folded into "low": an engine
    that reports nothing is a different situation from one that reports low confidence,
    and merging them would attribute the silence to doubt.
    """
    if value is None or pd.isna(value):
        return "not_reported"
    if value < 0.55:
        return "low"
    return "high" if value >= 0.80 else "medium"


def stratified_failure_table(
    proposals: pd.DataFrame, strata: Sequence[str] = ("dataset_id", "engine_id", "site_kind")
) -> pd.DataFrame:
    """Failure classes crossed with the strata the pre-registration named.

    Confidence band is always added, because "did the engine know it was unsure?" is the
    one stratum that separates a detectable failure from an undetectable one — and it is
    the signal a source-grounded verifier would have to beat.
    """
    frame = proposals.copy()
    frame["failure_class"] = [
        classify_failure(
            str(row.original_ocr or ""),
            str(row.candidate_text or ""),
            str(row.gt_text or ""),
            str(row.site_kind),
            int(str(row.delta)),
        )
        for row in frame.itertuples()
    ]
    frame["confidence_band"] = [
        _confidence_band(v) for v in frame.get("normalized_confidence", pd.Series(dtype=float))
    ]
    rows: list[dict[str, object]] = []
    for stratum in [*strata, "confidence_band"]:
        if stratum not in frame.columns:
            continue
        grouped = (
            frame.groupby(["generator_id", stratum, "failure_class"]).size().rename("n")
        ).reset_index()
        totals = grouped.groupby(["generator_id", stratum])["n"].transform("sum")
        grouped["share"] = grouped["n"] / totals
        for row in grouped.itertuples():
            rows.append(
                {
                    "generator_id": row.generator_id,
                    "stratum": stratum,
                    "level": getattr(row, stratum),
                    "failure_class": row.failure_class,
                    "n": row.n,
                    "share": row.share,
                }
            )
    return pd.DataFrame(rows)
