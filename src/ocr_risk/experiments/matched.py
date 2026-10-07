"""The matched-handicap design: making the reference arm cost the same as the test arm.

The H1 pilot compared ``loeo_zero_shot`` against ``in_engine_oracle`` and read the gap as
the cost of engine transfer. The two arms differed in five ways, not one — the reference
arm fitted on four engines rather than three, saw 23-39% more training candidates, 31-39%
more calibration candidates, and could compute a confidence z-score for the target engine
that the zero-shot arm structurally cannot. Four of those five are training opportunity,
not engine exposure. ``docs/h1_recovery/amendment4_confound.md`` derives it in full.

This module removes all four:

===========================  =========================  ==========================
dimension                    ``in_engine_oracle``       ``matched_in_engine``
===========================  =========================  ==========================
engines in the fit set       4                          **3**, one being the target
fit candidates               +23% to +39%               **equal by construction**
natural fit candidates       +23% to +39%               **equal by construction**
threshold-selection rows     +31% to +38%               **equal by construction**
calibration candidates       +31% to +39%               **equal by construction**
harmful fraction in fitting  differs by up to 0.096     **equal by construction**
target-engine z-score        available                  **ablated in both arms**
===========================  =========================  ==========================

Equality is not asserted, it is measured: :func:`build_match` emits a
:class:`MatchCertificate` per pair recording every registered dimension for both arms, and
the experiment refuses to run if any registered constraint is violated. Names like
``oracle`` and ``matched`` are not evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from ocr_risk.edits.outcome import is_beneficial
from ocr_risk.io.hashing import hash_str
from ocr_risk.schemas.enums import (
    HarmPolicy,
    OutcomeIfAccepted,
    SplitRole,
    harmful_outcomes,
)
from ocr_risk.splits import MatchedPair, SplitPlan

__all__ = [
    "ArmProfile",
    "FoldMatch",
    "MatchCertificate",
    "MatchViolation",
    "build_match",
]


@dataclass(frozen=True, slots=True)
class MatchViolation:
    """A registered constraint that the two arms did not satisfy."""

    dimension: str
    zero_shot: object
    reference: object

    def __str__(self) -> str:
        return f"{self.dimension}: zero_shot={self.zero_shot!r} reference={self.reference!r}"


@dataclass(frozen=True, slots=True)
class ArmProfile:
    """What one arm of a matched pair was actually given, after subsampling."""

    protocol: str
    fold_id: str
    fit_engines: tuple[str, ...]
    calibrate_engines: tuple[str, ...]
    evaluate_engines: tuple[str, ...]
    n_fit_engines: int
    n_fit_documents: int
    n_calibrate_documents: int
    n_evaluate_documents: int
    n_fit_candidates: int
    n_fit_harmful: int
    n_fit_beneficial: int
    n_fit_natural: int
    n_fit_natural_harmful: int
    n_calibrate_candidates: int
    n_calibrate_harmful: int
    n_calibrate_beneficial: int
    n_calibrate_natural: int
    """The rows threshold selection actually runs on: hard negatives are excluded from it,
    so the mixed calibration total can match while this does not."""
    n_calibrate_natural_harmful: int
    n_evaluate_candidates: int
    target_engine_zscore_ablated: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "fold_id": self.fold_id,
            "fit_engines": list(self.fit_engines),
            "calibrate_engines": list(self.calibrate_engines),
            "evaluate_engines": list(self.evaluate_engines),
            "n_fit_engines": self.n_fit_engines,
            "n_fit_documents": self.n_fit_documents,
            "n_calibrate_documents": self.n_calibrate_documents,
            "n_evaluate_documents": self.n_evaluate_documents,
            "n_fit_candidates": self.n_fit_candidates,
            "n_fit_harmful": self.n_fit_harmful,
            "n_fit_beneficial": self.n_fit_beneficial,
            "n_fit_natural": self.n_fit_natural,
            "n_fit_natural_harmful": self.n_fit_natural_harmful,
            "n_calibrate_candidates": self.n_calibrate_candidates,
            "n_calibrate_harmful": self.n_calibrate_harmful,
            "n_calibrate_beneficial": self.n_calibrate_beneficial,
            "n_calibrate_natural": self.n_calibrate_natural,
            "n_calibrate_natural_harmful": self.n_calibrate_natural_harmful,
            "n_evaluate_candidates": self.n_evaluate_candidates,
            "target_engine_zscore_ablated": self.target_engine_zscore_ablated,
        }


@dataclass(frozen=True, slots=True)
class FoldMatch:
    """The subsample and handicap one arm of one pair must apply."""

    fit_candidates: frozenset[str]
    calibrate_candidates: frozenset[str]
    ablate_target_engine_zscore: bool = True

    def restrict(self, frame: pd.DataFrame, role: SplitRole) -> pd.DataFrame:
        """Cut a role's view down to the matched sample. Evaluation is never cut."""
        if role is SplitRole.FIT or role is SplitRole.DEV:
            keep = self.fit_candidates
        elif role is SplitRole.CALIBRATE:
            keep = self.calibrate_candidates
        else:
            # Subsampling the evaluation set would change what is being measured, and the
            # whole point of the design is that both arms are measured on the same rows.
            return frame
        return frame.loc[frame["candidate_id"].isin(keep)].reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class MatchCertificate:
    """Machine-readable proof that a pair of arms is matched, or a list of why not."""

    pair_id: str
    held_out_engine: str
    donor_engine: str
    harm_policy: str
    seed: int
    zero_shot: ArmProfile
    reference: ArmProfile
    shared: dict[str, object] = field(default_factory=dict)
    violations: tuple[MatchViolation, ...] = ()

    @property
    def matched(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "held_out_engine": self.held_out_engine,
            "donor_engine": self.donor_engine,
            "harm_policy": self.harm_policy,
            "seed": self.seed,
            "matched": self.matched,
            "shared": self.shared,
            "zero_shot": self.zero_shot.as_dict(),
            "reference": self.reference.as_dict(),
            "violations": [
                {"dimension": v.dimension, "zero_shot": v.zero_shot, "reference": v.reference}
                for v in self.violations
            ],
        }


# The four strata the draw must match on. Class alone is not enough: the pool is roughly
# half synthesized hard negatives, which are fitted on but excluded from evaluation and
# from threshold selection, so matching the mixed totals leaves the NATURAL sub-pool
# unmatched. Measured on the real pool before this was stratified, the natural fit volume
# differed between arms by up to 9% -- and the sign was a deterministic function of the
# target engine's hard-negative rate, which is exactly the shape of confound the arm exists
# to remove.
_OUTCOME_CLASSES = ("harmful", "beneficial", "neutral")
_STRATA = tuple(
    f"{pool}_{outcome}" for pool in ("natural", "challenge") for outcome in _OUTCOME_CLASSES
)


def _strata(frame: pd.DataFrame, policy: HarmPolicy) -> dict[str, list[str]]:
    """Candidate ids by ``(pool, outcome class)``, the six cells the draw equalizes.

    Three outcome classes, not two. Matching only harmful-versus-safe left the composition
    *inside* the safe class free, and it moved: ``n_fit_beneficial`` differed by up to 13%
    between arms on certificates that read ``matched: true``. That is a feature-distribution
    difference in the negative class, which a ranking model can absolutely learn from.
    """
    if frame.empty:
        return {stratum: [] for stratum in _STRATA}
    harmful_values = {o.value for o in harmful_outcomes(policy)}
    beneficial_values = {o.value for o in OutcomeIfAccepted if is_beneficial(o, policy)}
    outcome = frame["outcome_if_accepted"]
    is_harmful = outcome.isin(harmful_values)
    is_beneficial_row = outcome.isin(beneficial_values)
    is_challenge = (
        frame["is_synthetic_hard_negative"].fillna(False).astype(bool)
        if "is_synthetic_hard_negative" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    ids = frame["candidate_id"].astype(str)
    membership = {
        "harmful": is_harmful,
        "beneficial": is_beneficial_row & ~is_harmful,
        "neutral": ~is_harmful & ~is_beneficial_row,
    }
    return {
        f"{pool}_{name}": sorted(ids[(is_challenge == (pool == "challenge")) & mask])
        for pool in ("natural", "challenge")
        for name, mask in membership.items()
    }


def _counts(frame: pd.DataFrame, policy: HarmPolicy) -> dict[str, int]:
    """Everything a profile reports about one restricted view."""
    if frame.empty:
        return dict.fromkeys(("n", "harmful", "beneficial", "natural", "natural_harmful"), 0)
    harmful_values = {o.value for o in harmful_outcomes(policy)}
    beneficial_values = {o.value for o in OutcomeIfAccepted if is_beneficial(o, policy)}
    outcome = frame["outcome_if_accepted"]
    natural = (
        ~frame["is_synthetic_hard_negative"].fillna(False).astype(bool)
        if "is_synthetic_hard_negative" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    return {
        "n": len(frame),
        "harmful": int(outcome.isin(harmful_values).sum()),
        "beneficial": int(outcome.isin(beneficial_values).sum()),
        "natural": int(natural.sum()),
        # The population threshold selection and every headline metric are computed on.
        "natural_harmful": int((natural & outcome.isin(harmful_values)).sum()),
    }


def _draw(candidate_ids: Sequence[str], k: int, salt: str) -> list[str]:
    """A deterministic sample of ``k`` ids.

    Ordered by a hash of ``(salt, id)`` rather than by a PRNG over positions, so the draw
    does not depend on how the frame happened to be sorted — two arms whose rows arrive in
    different orders must still be sampled the same way.
    """
    ordered = sorted(candidate_ids, key=lambda cid: (hash_str(f"{salt}:{cid}"), cid))
    return ordered[:k]


def _profile(
    plan: SplitPlan,
    match: FoldMatch,
    policy: HarmPolicy,
) -> ArmProfile:
    fit = match.restrict(plan.view(SplitRole.FIT).frame, SplitRole.FIT)
    calibrate = match.restrict(plan.view(SplitRole.CALIBRATE).frame, SplitRole.CALIBRATE)
    evaluate = plan.view(SplitRole.EVALUATE).frame
    fit_counts = _counts(fit, policy)
    cal_counts = _counts(calibrate, policy)
    fit_engines, _ = plan.scope_for(SplitRole.FIT)
    cal_engines, _ = plan.scope_for(SplitRole.CALIBRATE)
    eval_engines, _ = plan.scope_for(SplitRole.EVALUATE)

    def documents(frame: pd.DataFrame) -> int:
        """Documents in the SAMPLE, not in the scope.

        Reading these off ``scope_for`` compared 185 with 185 by construction, so four of
        the registered constraints could not fail. Subsampling can drop a document
        entirely, and a pair of arms fitted on different numbers of pages is not matched.
        """
        return int(frame["document_id"].nunique()) if not frame.empty else 0

    return ArmProfile(
        protocol=plan.protocol,
        fold_id=plan.fold_id,
        fit_engines=tuple(sorted(fit_engines)),
        calibrate_engines=tuple(sorted(cal_engines)),
        evaluate_engines=tuple(sorted(eval_engines)),
        n_fit_engines=len(fit_engines),
        n_fit_documents=documents(fit),
        n_calibrate_documents=documents(calibrate),
        n_evaluate_documents=documents(evaluate),
        n_fit_candidates=fit_counts["n"],
        n_fit_harmful=fit_counts["harmful"],
        n_fit_beneficial=fit_counts["beneficial"],
        n_fit_natural=fit_counts["natural"],
        n_fit_natural_harmful=fit_counts["natural_harmful"],
        n_calibrate_candidates=cal_counts["n"],
        n_calibrate_harmful=cal_counts["harmful"],
        n_calibrate_beneficial=cal_counts["beneficial"],
        n_calibrate_natural=cal_counts["natural"],
        n_calibrate_natural_harmful=cal_counts["natural_harmful"],
        n_evaluate_candidates=len(evaluate),
        target_engine_zscore_ablated=match.ablate_target_engine_zscore,
    )


# The dimensions that must be equal for the contrast to be about engine exposure. Names
# are ArmProfile fields; `_DELIBERATELY_UNEQUAL` below holds the rest, and a test asserts
# every numeric field is in exactly one of the two.
_REGISTERED_EQUAL = (
    "n_fit_engines",
    "n_fit_candidates",
    "n_fit_harmful",
    "n_fit_beneficial",
    "n_fit_natural",
    "n_fit_natural_harmful",
    "n_calibrate_candidates",
    "n_calibrate_harmful",
    "n_calibrate_beneficial",
    "n_calibrate_natural",
    "n_calibrate_natural_harmful",
    "n_evaluate_candidates",
    "evaluate_engines",
    "target_engine_zscore_ablated",
)

# Fields a profile records but which are ALLOWED to differ, each with the reason. Every
# numeric field must appear in one list or the other; `test_every_profile_field_is_
# accounted_for` fails otherwise. The docstring used to claim that forgetting to register
# a field was "visible", and it was not: `n_fit_beneficial` differed by 13% across arms
# on twelve certificates that all read `matched: true`.
_DELIBERATELY_UNEQUAL = {
    "fit_engines": "the identity of the third engine IS the manipulation",
    "calibrate_engines": "as fit_engines -- the donor swap reaches both fitted scopes",
    "n_fit_documents": (
        "documents are not the unit the draw controls -- it equalizes candidate strata, and "
        "which pages survive is incidental. Both arms draw from the same D_fit by "
        "construction, which `assert_no_leakage` checks."
    ),
    "n_calibrate_documents": "as n_fit_documents",
    "n_evaluate_documents": "the evaluation view is never subsampled",
}


def build_match(
    pair: MatchedPair,
    policy: HarmPolicy,
    seed: int,
    shared: dict[str, object] | None = None,
) -> tuple[FoldMatch, FoldMatch, MatchCertificate]:
    """Compute the subsample that makes one pair of arms comparable, and certify it.

    The target is the element-wise minimum of the two arms' counts in each of six strata:
    ``(natural, challenge) x (harmful, beneficial, neutral)``. Taking the minimum of the
    *counts* rather than a fixed fraction means neither arm is ever asked for rows it does
    not have.

    Stratifying on the **pool** as well as the class is not decoration. Roughly half the
    rows are synthesized hard negatives, which are fitted on but excluded from evaluation
    and from threshold selection; matching only the mixed totals left the natural sub-pool
    differing by up to 9% between arms, with a sign determined by the target engine's
    hard-negative rate. That is a training-opportunity difference aligned with the
    evaluation distribution — the exact shape of confound the arm exists to remove.
    """
    profiles: dict[str, FoldMatch] = {}
    per_arm: dict[str, dict[SplitRole, dict[str, list[str]]]] = {}
    for name, plan in (("zero_shot", pair.zero_shot), ("reference", pair.reference)):
        per_arm[name] = {
            role: _strata(plan.view(role).frame, policy)
            for role in (SplitRole.FIT, SplitRole.CALIBRATE)
        }

    targets: dict[SplitRole, dict[str, int]] = {
        role: {
            stratum: min(len(per_arm[arm][role][stratum]) for arm in per_arm) for stratum in _STRATA
        }
        for role in (SplitRole.FIT, SplitRole.CALIBRATE)
    }

    for name in ("zero_shot", "reference"):
        drawn: dict[SplitRole, frozenset[str]] = {}
        for role in (SplitRole.FIT, SplitRole.CALIBRATE):
            salt = f"{seed}:{pair.pair_id}:{name}:{role.value}"
            drawn[role] = frozenset(
                candidate_id
                for stratum in _STRATA
                for candidate_id in _draw(
                    per_arm[name][role][stratum], targets[role][stratum], f"{salt}:{stratum}"
                )
            )
        profiles[name] = FoldMatch(
            fit_candidates=drawn[SplitRole.FIT],
            calibrate_candidates=drawn[SplitRole.CALIBRATE],
            ablate_target_engine_zscore=True,
        )

    zero_shot_profile = _profile(pair.zero_shot, profiles["zero_shot"], policy)
    reference_profile = _profile(pair.reference, profiles["reference"], policy)
    violations = tuple(
        MatchViolation(
            dimension=dimension,
            zero_shot=getattr(zero_shot_profile, dimension),
            reference=getattr(reference_profile, dimension),
        )
        for dimension in _REGISTERED_EQUAL
        if getattr(zero_shot_profile, dimension) != getattr(reference_profile, dimension)
    )
    # The one dimension that must DIFFER. A "matched" pair whose arms are identical in
    # target-engine exposure measures nothing, and would pass every equality check above.
    # Checking only the target's half left a reference arm that quietly KEPT its donor --
    # a four-engine fit in a three-engine costume -- certifiable as matched, so the swap
    # is verified in both directions: the target out of the zero-shot fit set and into the
    # reference's, the donor along the reverse path.
    if (pair.held_out in zero_shot_profile.fit_engines) or (
        pair.held_out not in reference_profile.fit_engines
    ):
        violations = (
            *violations,
            MatchViolation(
                dimension="target_engine_in_fit_set",
                zero_shot=pair.held_out in zero_shot_profile.fit_engines,
                reference=pair.held_out in reference_profile.fit_engines,
            ),
        )
    if (pair.donor not in zero_shot_profile.fit_engines) or (
        pair.donor in reference_profile.fit_engines
    ):
        violations = (
            *violations,
            MatchViolation(
                dimension="donor_engine_in_fit_set",
                zero_shot=pair.donor in zero_shot_profile.fit_engines,
                reference=pair.donor in reference_profile.fit_engines,
            ),
        )

    certificate = MatchCertificate(
        pair_id=pair.pair_id,
        held_out_engine=pair.held_out,
        donor_engine=pair.donor,
        harm_policy=policy.value,
        seed=seed,
        zero_shot=zero_shot_profile,
        reference=reference_profile,
        shared=dict(shared or {}),
        violations=violations,
    )
    return profiles["zero_shot"], profiles["reference"], certificate
