"""The correction-outcome taxonomy: what happens if an edit is accepted.

This module defines the counterfactual at the centre of the project. For a site with OCR
text ``O`` and ground truth ``G``, and a candidate ``Y``::

    d_before = lev(O, G)      d_after = lev(Y, G)      delta = d_before - d_after

Distances are **raw characters, not normalized**. The comparison is ``d_after`` against
``d_before`` for the *same* ``G``, so dividing both by the same constant would only add
noise and would make short spans dominate.

============================================  ========================  ==============
condition                                     outcome_if_accepted       class
============================================  ========================  ==============
``Y == O``                                    ``IDENTITY``              filtered out
``d_before == 0`` and ``d_after > 0``         ``OVERCORRECTION``        **harmful**
``d_before > 0`` and ``d_after == 0``         ``TRUE_CORRECTION``       beneficial
``d_before > 0``, ``0 < d_after < d_before``  ``PARTIAL_IMPROVEMENT``   beneficial
``d_after == d_before`` and ``Y != O``        ``LATERAL_CHANGE``        non-beneficial
``d_before > 0`` and ``d_after > d_before``   ``MISCORRECTION``         **harmful**
============================================  ========================  ==============

Rejecting an edit yields ``PRESERVATION`` when the OCR was already right and
``MISSED_ERROR`` when it was not. A missed error is an *opportunity cost*, not harm: the
system left the transcription no worse than it found it.

Which outcomes count as harmful is a configured
:class:`~ocr_risk.schemas.enums.HarmPolicy`, never a constant here, because the choice
moves the headline numbers and must be reported under all three variants.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from ocr_risk.schemas.enums import (
    HarmPolicy,
    OutcomeIfAccepted,
    OutcomeIfRejected,
    harmful_outcomes,
)

__all__ = [
    "BENEFICIAL_OUTCOMES",
    "classify_accepted",
    "classify_rejected",
    "distance",
    "is_beneficial",
    "is_harmful",
]

BENEFICIAL_OUTCOMES: frozenset[OutcomeIfAccepted] = frozenset(
    {OutcomeIfAccepted.TRUE_CORRECTION, OutcomeIfAccepted.PARTIAL_IMPROVEMENT}
)
"""Outcomes that moved the transcription toward ground truth.

Under ``EXACT_ONLY`` a partial improvement is *also* counted as harmful; the two notions
are separate on purpose, so a candidate can be simultaneously "moved closer" and "not good
enough", and the sensitivity analysis can report both readings.
"""


def distance(a: str, b: str) -> int:
    """Raw character edit distance."""
    return int(Levenshtein.distance(a, b))


def classify_accepted(ocr_text: str, candidate_text: str, gt_text: str) -> OutcomeIfAccepted:
    """Outcome of accepting ``ocr_text -> candidate_text`` against ``gt_text``."""
    if candidate_text == ocr_text:
        return OutcomeIfAccepted.IDENTITY

    d_before = distance(ocr_text, gt_text)
    d_after = distance(candidate_text, gt_text)

    if d_before == 0:
        # The OCR was already correct, so any change can only break it.
        return OutcomeIfAccepted.OVERCORRECTION
    if d_after == 0:
        return OutcomeIfAccepted.TRUE_CORRECTION
    if d_after < d_before:
        return OutcomeIfAccepted.PARTIAL_IMPROVEMENT
    if d_after == d_before:
        return OutcomeIfAccepted.LATERAL_CHANGE
    return OutcomeIfAccepted.MISCORRECTION


def classify_rejected(ocr_text: str, gt_text: str) -> OutcomeIfRejected:
    """Outcome of preserving the OCR at a site."""
    if ocr_text == gt_text:
        return OutcomeIfRejected.PRESERVATION
    return OutcomeIfRejected.MISSED_ERROR


def is_harmful(outcome: OutcomeIfAccepted, policy: HarmPolicy) -> bool:
    """Whether ``policy`` counts ``outcome`` as a harmful accepted edit."""
    return outcome in harmful_outcomes(policy)


def is_beneficial(outcome: OutcomeIfAccepted, policy: HarmPolicy) -> bool:
    """Whether ``policy`` counts ``outcome`` as a beneficial accepted edit.

    Under the strictest policy only an exact repair counts, so a partial improvement is
    neither beneficial nor merely neutral — it is harmful. Deriving both from the same
    policy keeps the two answers consistent.
    """
    if policy is HarmPolicy.EXACT_ONLY:
        return outcome is OutcomeIfAccepted.TRUE_CORRECTION
    return outcome in BENEFICIAL_OUTCOMES
