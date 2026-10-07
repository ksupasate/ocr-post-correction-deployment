"""The correction-outcome taxonomy, exhaustively.

This is the definition the entire risk framing rests on. Every branch is pinned with a
hand-worked example, and the harm/benefit mapping is checked for all three policies —
because switching policy is the sensitivity analysis the paper must report, and a policy
that quietly agreed with another would make that analysis vacuous.
"""

from __future__ import annotations

import pytest

from ocr_risk.edits import classify_accepted, classify_rejected, distance, is_beneficial, is_harmful
from ocr_risk.schemas.enums import (
    HarmPolicy,
    OutcomeIfAccepted,
    OutcomeIfRejected,
    harmful_outcomes,
)

# (ocr, candidate, gt, expected) — every row is a worked example, not a generated case.
TRUTH_TABLE = [
    # Y == O: no edit is proposed at all.
    ("mg", "mg", "mg", OutcomeIfAccepted.IDENTITY),
    ("rng", "rng", "mg", OutcomeIfAccepted.IDENTITY),
    # OCR already correct, candidate changes it -> broke a correct span.
    ("mg", "ng", "mg", OutcomeIfAccepted.OVERCORRECTION),
    ("0.015", "0.15", "0.015", OutcomeIfAccepted.OVERCORRECTION),
    # OCR wrong, candidate exactly right.
    ("rng", "mg", "mg", OutcomeIfAccepted.TRUE_CORRECTION),
    ("Srnith", "Smith", "Smith", OutcomeIfAccepted.TRUE_CORRECTION),
    # OCR wrong, candidate closer but still wrong.  d=2 -> d=1
    ("Ti-6A1-4X", "Ti-6A1-4V", "Ti-6Al-4V", OutcomeIfAccepted.PARTIAL_IMPROVEMENT),
    ("rng", "ug", "mg", OutcomeIfAccepted.PARTIAL_IMPROVEMENT),
    # OCR wrong, candidate equally wrong but different: churn with no gain.  d=1 -> d=1
    ("ng", "ug", "mg", OutcomeIfAccepted.LATERAL_CHANGE),
    ("0.15", "0.0015", "0.015", OutcomeIfAccepted.LATERAL_CHANGE),
    # OCR wrong, candidate further away: made a bad span worse.
    ("rng", "xxxxxx", "mg", OutcomeIfAccepted.MISCORRECTION),
    # A magnitude blowout: dropping the leading zero costs 1, deleting "0." costs 3.
    ("0.15", "15", "0.015", OutcomeIfAccepted.MISCORRECTION),
]


@pytest.mark.parametrize(("ocr", "candidate", "gt", "expected"), TRUTH_TABLE)
def test_accepted_outcomes(ocr: str, candidate: str, gt: str, expected: OutcomeIfAccepted) -> None:
    d_before, d_after = distance(ocr, gt), distance(candidate, gt)
    assert classify_accepted(ocr, candidate, gt) is expected, (
        f"{ocr!r} -> {candidate!r} against {gt!r}: d_before={d_before} d_after={d_after}"
    )


def test_every_accepted_outcome_is_reachable() -> None:
    """A member no example produces is either dead or a gap in the taxonomy."""
    produced = {classify_accepted(o, c, g) for o, c, g, _ in TRUTH_TABLE}
    assert produced == set(OutcomeIfAccepted)


def test_rejected_outcomes() -> None:
    assert classify_rejected("mg", "mg") is OutcomeIfRejected.PRESERVATION
    assert classify_rejected("rng", "mg") is OutcomeIfRejected.MISSED_ERROR


def test_missed_error_is_not_harm() -> None:
    """Leaving an error alone costs an opportunity; it does not make the text worse.
    Conflating the two would make abstention look as bad as breaking the transcription."""
    for policy in HarmPolicy:
        harmful = harmful_outcomes(policy)
        assert OutcomeIfAccepted.IDENTITY not in harmful


# --- harm policies --------------------------------------------------------------------
HARM_EXPECTATIONS = {
    HarmPolicy.STRICT_WORSENING: {
        OutcomeIfAccepted.MISCORRECTION,
        OutcomeIfAccepted.OVERCORRECTION,
    },
    HarmPolicy.NON_IMPROVING: {
        OutcomeIfAccepted.MISCORRECTION,
        OutcomeIfAccepted.OVERCORRECTION,
        OutcomeIfAccepted.LATERAL_CHANGE,
    },
    HarmPolicy.EXACT_ONLY: {
        OutcomeIfAccepted.MISCORRECTION,
        OutcomeIfAccepted.OVERCORRECTION,
        OutcomeIfAccepted.LATERAL_CHANGE,
        OutcomeIfAccepted.PARTIAL_IMPROVEMENT,
    },
}


@pytest.mark.parametrize(("policy", "expected"), HARM_EXPECTATIONS.items())
def test_harmful_sets_per_policy(policy: HarmPolicy, expected: set[OutcomeIfAccepted]) -> None:
    assert harmful_outcomes(policy) == expected


@pytest.mark.parametrize("policy", list(HarmPolicy))
@pytest.mark.parametrize("outcome", list(OutcomeIfAccepted))
def test_is_harmful_matches_the_policy_set(policy: HarmPolicy, outcome: OutcomeIfAccepted) -> None:
    assert is_harmful(outcome, policy) == (outcome in harmful_outcomes(policy))


def test_policies_are_strictly_nested() -> None:
    """Each policy must be strictly stricter than the last. Two policies that agreed
    would make the sensitivity analysis report the same number twice."""
    strict = harmful_outcomes(HarmPolicy.STRICT_WORSENING)
    non_improving = harmful_outcomes(HarmPolicy.NON_IMPROVING)
    exact = harmful_outcomes(HarmPolicy.EXACT_ONLY)
    assert strict < non_improving < exact


@pytest.mark.parametrize("policy", list(HarmPolicy))
def test_nothing_is_both_harmful_and_beneficial(policy: HarmPolicy) -> None:
    for outcome in OutcomeIfAccepted:
        assert not (is_harmful(outcome, policy) and is_beneficial(outcome, policy)), (
            f"{outcome} is both harmful and beneficial under {policy}"
        )


def test_partial_improvement_flips_under_exact_only() -> None:
    """The case that makes the harm policy worth reporting: a partial repair is a gain
    under the default and a harm under the strictest reading."""
    partial = OutcomeIfAccepted.PARTIAL_IMPROVEMENT
    assert is_beneficial(partial, HarmPolicy.STRICT_WORSENING)
    assert not is_harmful(partial, HarmPolicy.STRICT_WORSENING)
    assert not is_beneficial(partial, HarmPolicy.EXACT_ONLY)
    assert is_harmful(partial, HarmPolicy.EXACT_ONLY)


def test_true_correction_is_beneficial_under_every_policy() -> None:
    for policy in HarmPolicy:
        assert is_beneficial(OutcomeIfAccepted.TRUE_CORRECTION, policy)
        assert not is_harmful(OutcomeIfAccepted.TRUE_CORRECTION, policy)


# --- distance ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("mg", "mg", 0),
        ("rng", "mg", 2),  # r->m, n->g ... via one substitution + one deletion
        ("0.015", "0.15", 1),  # delete a zero
        ("Ti-6A1-4V", "Ti-6Al-4V", 1),
        ("abc", "", 3),
    ],
)
def test_distance_against_hand_computed_values(a: str, b: str, expected: int) -> None:
    assert distance(a, b) == expected


def test_distance_is_symmetric() -> None:
    assert distance("Srnith", "Smith") == distance("Smith", "Srnith")
