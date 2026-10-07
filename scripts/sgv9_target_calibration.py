#!/usr/bin/env python3
"""SGV9: can a target engine's harm probabilities be recovered without target labels?

SGV8 ended by separating two things that had been travelling together for eight stages. The
ordering a frozen risk model puts on candidates survives a change of OCR engine -- SGV8's
deployment classifier discriminated safe from unsafe deployments on engines it had never seen,
with an AUC of 0.66 to 0.81 against a shuffled control of 0.31 to 0.63. What did not survive
was the *level*: its mean predicted probability of a safe deployment was 0.017 on docTR where
the realised rate was 0.485. Ranking transferred; calibration did not.

That diagnosis was made one level up, about a controller. This stage asks the same question one
level down, about the thing the whole platform is built on -- the candidate's own probability of
being a harmful accepted edit -- and it asks it because that probability is what places the
deployment cut. The SGV5 model's harm head is isotonic-calibrated on the *source* engines'
CALIBRATION rows. Applied to an unseen engine and read literally, it says the following (all
four numbers are the identity map under this stage's own cut rule, and all four appear in
`risk_coverage_table.json`):

    engine      cut placed at a predicted harm of 0.05      realised harm
    docTR                                                        0.139
    EasyOCR                                                      0.198
    PaddleOCR                                                    0.090
    Tesseract                                                    0.018

Three engines are over-confident by a factor of 1.8 to 4.0 and pay for it in broken bounds; the
fourth is under-confident by a factor of 2.8 and pays for it in coverage -- 0.298 repair recall
where the achievable frontier at the same bound is 0.460. Both failures are the same failure,
and neither is a failure of the ranking.

    SGV9-T1: the absolute harm probability of a frozen candidate-risk model can be recovered on
    an unseen OCR engine from that engine's *unlabelled* candidate scores, closely enough to
    place a risk-bounded deployment cut better than transferring either the source threshold or
    the source probabilities, and without disturbing the ranking that already transfers.

**The hypothesis is recorded as SGV9-T1, not H1.** `docs/sgv1/protocol.md` binds SGV1-H1
through SGV1-H4 and forbids reusing a frozen identifier for a different claim; every stage since
has opened its own family. The brief's H1/H2/H3 are carried as sub-claims T1a (a target-free map
improves absolute harm estimation while preserving ranking), T1b (a few target labels interpolate
smoothly between the target-free map and supervised recalibration) and T1c (better calibration
means fewer unsafe deployments at a fixed bound), all under T1.

Ten decisions fix what the numbers below can mean. Nine were made before any SGV9 result was
read; the tenth is dated, says exactly what had been seen when it was made, and is the only one
that is not part of the pre-registration. `design_record.json` carries all ten with the
prediction they imply.

**1. Only one thing is transformed, and it is not the ranking.** The frozen SGV5 model, its
features, its candidate pool and its decision score `benefit - lambda * harm` are rebuilt bit for
bit and never touched. A calibration map acts on the harm head alone, and the decision score that
orders candidates is SGV5's, unchanged, for every arm in this stage. A map therefore cannot move
a single candidate past another one in the accept order; it can only move the *cut*.

**2. Every map is monotone in the score it transforms, and the constraint is enforced, not
assumed.** The brief requires `p_i < p_j => T(p_i) <= T(p_j)`. Each fitted map is checked against
its own knots and against the realised transform on every block it is applied to, and a violation
is a hard failure. The two arms that are deliberately non-monotone are named as ablations and are
excluded from every claim.

**3. The cut is placed on the unlabelled target pool, never on the evaluation block.**
`.claude/rules/experiment-leakage.md` names threshold selection as one of the things a held-out
engine's test documents may not touch, and it does not carve out an exception for reading only
their scores. So the map is fitted on the source calibration rows and the held-out engine's
*TRAIN* rows, the cut is placed on those same TRAIN rows, and the resulting threshold meets the
DEVELOPMENT rows exactly once, to be measured. That costs something real -- on Tesseract the pool
is 49.1% harmful and the evaluation block is 38.8%, and no pool-fitted map can know that -- and
the cost is reported rather than removed.

**4. The rule change and the calibration are separated.** Placing a cut by predicted harm rather
than by a transferred source threshold is itself a change, and it can win or lose on its own. So
the identity map under the new rule is carried as a full arm (ablation A), and the published SGV5
deployed point is carried beside it. A gain that appears under the identity map is a gain from
the rule, not from calibration, and the table is laid out so that cannot be missed.

**5. A calibrated point estimate does not deliver a bound.** If a map were perfect, the deepest
prefix whose *expected* harm equals epsilon has realised harm above epsilon about half the time.
That is arithmetic, not a defect, and it is visible in this stage's own oracle: a map fitted on
the evaluation labels themselves still breaks the bound in 11 of the 12 engine-by-epsilon cells,
and one fitted on the target engine's own TRAIN labels breaks it in all 12.
Method D therefore does not sharpen the estimate, it bounds it, and it is the only arm from which
a safety claim is read.

**6. The reference for "how wrong will this be on an engine I have never seen" is other engines,
not other rows.** Method D's inflation is a quantile over a leave-one-source-engine-out replay of
the entire pipeline: for each fit engine in turn, the map is refitted from the remaining fit
engines and its error is measured on the held-out fit engine. With three source engines the
distribution-free coverage that construction can support is 3/4, not 1 - alpha, and this stage
takes the maximum of the three rather than pretending otherwise.

**7. Structure and difficulty enter as strata, fitted on the source and frozen.** Method C's
transport is stratified by the edit's operation class and by a difficulty tercile whose cut
points come from the source calibration rows and are then applied unchanged to the target. A
quantile computed over a pooled frame would be a leak of exactly the kind
`.claude/rules/experiment-leakage.md` names.

**8. The few-shot arms buy their labels from the pool, on SGV6's and SGV7's terms.** The budget
grid is the brief's -- 0, 5, 10, 25, 50, 100 -- drawn by random acquisition from the held-out
engine's TRAIN rows over 50 seeds, which is the same pool, the same rule and the same seed count
those two stages used.

**9. Two oracles, and neither is a method.** `oracle_pool` fits the map on every label in the
adaptation pool; `oracle_eval` fits it on the evaluation block itself. The first is the ceiling
of what target labels could buy without touching the evaluation documents, the second is the
ceiling of the map class outright. Both are upper bounds, both are excluded from selection, and
the achievable frontier is reported beside them as the ceiling of the ranking.

**10. One arm was added mid-stage, and it is labelled.** Methods B and C cannot represent a
change in prevalence -- a rank transport reproduces the source's marginal on the target whatever
the target's own prevalence is -- and Method A2 can move the level but not decide which rows
move. Composing them (`mC2_transport_prior`) is what the two mechanisms jointly permit, and the
arm was added after the fitted maps' marginals were read off the four folds and BEFORE any
endpoint, any calibration error or any deployed point had been computed. It is reported as a
composition of two pre-specified arms, it is never the primary, and the record says all of this
rather than leaving a reader to infer it. Method C remains the primary because the brief names it.

    --calibrate    rebuild the frozen model per fold, fit every map, write the per-row table
    --mapping      the fitted maps themselves, their knots and their monotonicity checks
    --results      calibration error, ranking preservation and the paired intervals
    --reliability  the reliability diagrams and the prefix-calibration curves
    --coverage     the risk-coverage table: every map, every epsilon, every cut rule
    --efficiency   the few-shot budget curves
    --oracle       the gap to the two oracles and to the achievable frontier
    --ablation     the brief's six ablations
    --negative     the five negative tests the brief requires
    --figures      the required figures
    --decide       the machine-readable finding
    --record       provenance for every artifact

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; no map, no threshold and no selection
reads a label from the held-out engine's DEVELOPMENT rows.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_candidate_conditioned as cc
import sgv1_domain_generalization as dg
import sgv1_risk_policy as policy
import sgv1_verifier_pilot as pilot
import sgv2_reliability_layer as rl
import sgv3_self_aware as sa
import sgv5_candidate_reliability as c5
import sgv6_target_engine_adaptation as s6
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.calibration import calibration_report
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.risk.controller import select_threshold

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv9"

SCORES = OUT / "calibrated_scores.parquet"
DESIGN_RECORD = OUT / "design_record.json"
CALIBRATION_MAPPING = OUT / "calibration_mapping.json"
CALIBRATION_RESULTS = OUT / "calibration_results.json"
RELIABILITY_METRICS = OUT / "reliability_metrics.json"
RISK_COVERAGE = OUT / "risk_coverage_table.json"
LABEL_EFFICIENCY = OUT / "label_efficiency.json"
ORACLE_GAP = OUT / "oracle_gap.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
DECISION = OUT / "research_decision.json"
PROVENANCE = OUT / "provenance_manifest.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

# The endpoint, its frontier and its interval have one implementation in this repository.
# Nine stages now import them rather than restating them; `tests/leakage` asserts the identity.
achievable_repair_recall = rl.achievable_repair_recall
paired_repair_recall_delta = sa._paired_repair_recall_delta
deployed_point = c5._deployed
_stable_seed = s6._stable_seed
_logit = s6._logit
_sigmoid = s6._sigmoid
build_fold = rl.build_fold
_spearman = rl._spearman

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
PRIMARY_KEY = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = pilot.BOOTSTRAP_SEED
CROSS_ENGINE = s6.CROSS_ENGINE
PROB_FLOOR = s6.PROB_FLOOR

# The brief's budget grid, drawn by SGV6's random acquisition from SGV6's pool over SGV7's
# seed count, so a few-shot cell here is the same experiment those stages ran.
BUDGETS = (0, 5, 10, 25, 50, 100)
NONZERO_BUDGETS = tuple(n for n in BUDGETS if n > 0)
SEED_REPEATS = 50
SHRINKAGE_PRIOR = 50.0
"""Labels at which a few-shot fit is weighted equally with the target-free map. Fixed before
any result was read: at N = 50 the direct fit sees ~50 * 0.6 = 30 harmful rows, which is the
point at which an empirical harm rate has a standard error near 0.07 -- comparable to the
transfer error the target-free maps are trying to remove."""

N_BINS = 15
"""Bins for ECE and for the Murphy decomposition. Both binning schemes are reported; the
project's rule is that a single ECE without its scheme and bin count is not a reportable
number, so `calibration_report` carries all of them."""

STRATUM_SHRINKAGE = 100.0
"""Rows at which a stratum's own source reliability curve is weighted equally with the pooled
one. A stratum with a handful of source rows would otherwise fit noise and export it."""

CONFORMAL_ALPHA = 0.10
N_SOURCE_ENGINES = 3
CONFORMAL_COVERAGE = N_SOURCE_ENGINES / (N_SOURCE_ENGINES + 1)
"""What a distribution-free quantile over three exchangeable calibration engines can support.
Taking the maximum of three leave-one-source-engine-out errors is the most conservative choice
the construction admits, and it guarantees 3/4, not 1 - CONFORMAL_ALPHA. Reported as such."""


class PhaseError(RuntimeError):
    """A fold, map, monotonicity or identity invariant failed."""


# ------------------------------------------------------------------ the map registry

IDENTITY = "m0_identity"
SOURCE_ISOTONIC = "mD_source_only"
TEMPERATURE = "mA_temperature"
PRIOR_SHIFT = "mA2_prior_shift"
RANK_TRANSPORT = "mB_rank_transport"
QUANTILE_TRANSPORT = "mC_quantile_transport"
TRANSPORT_PRIOR = "mC2_transport_prior"
UNPROJECTED = "mC_unprojected"
NO_STRATA = "mE_no_strata"
CONFORMAL = "mD_conformal"
PERMUTED = "ablationB_permuted"
SCRAMBLED = "ablationC_scrambled"
ORACLE_POOL = "oracle_pool"
ORACLE_EVAL = "oracle_eval"

PRIMARY_MAP = QUANTILE_TRANSPORT

# The label-free family. Every one of these is fitted from the source calibration rows and the
# held-out engine's unlabelled TRAIN scores, and from nothing else.
TARGET_FREE = (
    IDENTITY,
    SOURCE_ISOTONIC,
    TEMPERATURE,
    PRIOR_SHIFT,
    RANK_TRANSPORT,
    QUANTILE_TRANSPORT,
    TRANSPORT_PRIOR,
    CONFORMAL,
)
CONTROLS = (PERMUTED, SCRAMBLED, UNPROJECTED, NO_STRATA)
ORACLES = (ORACLE_POOL, ORACLE_EVAL)
FITTED_MAPS = (*TARGET_FREE, SCRAMBLED, UNPROJECTED, NO_STRATA, *ORACLES)
"""Arms that are a function of the score and therefore have a fitted map object. The
permutation ablation is not one: it acts on rows, not on scores, and is materialised as a
column of the score table rather than as a map."""
ALL_ARMS = (*FITTED_MAPS, PERMUTED)

FEW_SHOT_DIRECT = "mE_few_shot_direct"
FEW_SHOT_SHRUNK = "mE_few_shot_shrunk"
FEW_SHOT_PRIOR = "mE_few_shot_prior"
FEW_SHOT_COMPOSED = "mE_few_shot_composed"
FEW_SHOT = (FEW_SHOT_DIRECT, FEW_SHOT_SHRUNK, FEW_SHOT_PRIOR, FEW_SHOT_COMPOSED)
PRIMARY_FEW_SHOT = FEW_SHOT_SHRUNK

MAP_NOTES: dict[str, str] = {
    IDENTITY: (
        "Baseline 0 and ablation A. SGV5's harm head as published, read as a probability on the "
        "unseen engine. Under this stage's cut rule it is also the control that separates the "
        "rule change from the calibration."
    ),
    SOURCE_ISOTONIC: (
        "Ablation D -- remove target distribution information. An isotonic recalibration fitted "
        "on the source calibration rows and nothing else. SGV5's harm head is ALREADY isotonic "
        "on exactly that block, and isotonic regression is a projection, so this arm is the "
        "identity up to floating point. That is a proof, not a measurement, and the residual is "
        "recorded to show the code agrees with the proof."
    ),
    TEMPERATURE: (
        "Method A. p' = sigmoid(logit(p) / T) with T = sd(logit p on the target pool) / "
        "sd(logit p on the source calibration rows). Uses target scores, never target labels. "
        "It can rescale confidence but cannot move the mean independently of the spread."
    ),
    PRIOR_SHIFT: (
        "Method A2. A binned black-box shift estimate of the target harm prevalence from the "
        "unlabelled target score histogram and the source confusion between bins and labels, "
        "followed by the standard prior correction. Monotone in p for any positive weights. "
        "Identified only under label shift -- P(score | harmful) equal in source and target -- "
        "and negative test 2 measures how far that assumption is from true on each engine."
    ),
    RANK_TRANSPORT: (
        "Method B. T(p) = R_s(F_s^-1(F_t(p))): read the target row's rank in the unlabelled "
        "target score distribution, take the source score at that rank, and report the harm "
        "rate the source labels attach to it. Assumes rank, not level, carries the risk -- which "
        "is what SGV8 found transfers. It cannot represent a change in prevalence: its mean "
        "predicted harm on the target is the source's mean by construction."
    ),
    QUANTILE_TRANSPORT: (
        "Method C, the primary. Rank transport computed WITHIN strata of edit operation class "
        "and difficulty tercile, with each stratum's source reliability curve shrunk toward the "
        "pooled one, then projected onto the monotone-in-p cone so the brief's ordering "
        "constraint holds exactly."
    ),
    UNPROJECTED: (
        "Ablation C. Method C without the monotone projection. A real, motivated, non-monotone "
        "calibration: it is what the strata actually say before the constraint is imposed. "
        "Excluded from every claim; its purpose is to price the constraint."
    ),
    NO_STRATA: (
        "Ablation E -- remove structural statistics. Method C with a single stratum. It equals "
        "Method B on every score level the isotonic projection was fitted at, because a marginal "
        "rank transport is already non-decreasing and the projection has nothing to remove. It "
        "does NOT equal Method B everywhere: the transport is a step function of the score and "
        "the projection interpolates linearly across those steps, so the two differ on rows "
        "whose score falls in a gap the pool does not contain. That difference is measured per "
        "block rather than assumed away, and it is the only thing the projection does when "
        "there is one stratum."
    ),
    TRANSPORT_PRIOR: (
        "Method C composed with Method A2's prior correction: transport the ranks, then move the "
        "LEVEL until the map's mean over the unlabelled target pool equals the label-free "
        "prevalence estimate. Both steps are non-decreasing, so the composition is. It exists "
        "because Methods B and C cannot represent a change in prevalence at all -- a rank "
        "transport carries the source's marginal onto the target unchanged -- and Method A2 can "
        "move only the level. WHEN THIS ARM WAS ADDED: after the fitted maps' own marginals were "
        "read off the four folds and before any endpoint, any calibration error and any deployed "
        "point had been computed. It is reported as a pre-specified composition of two arms that "
        "were specified before the run, and never as the primary; Method C remains the primary "
        "because the brief names it."
    ),
    CONFORMAL: (
        "Method D. Method C's estimate plus an inflation equal to the largest error the same "
        "pipeline makes when it is replayed leave-one-source-engine-out. The inflated estimate "
        "is not a better point estimate and is not scored as one; it is the only arm from which "
        "a safety claim is read."
    ),
    PERMUTED: (
        "Ablation B -- random permutation control. Method C's outputs permuted across rows, so "
        "the marginal distribution of the estimates is exactly preserved and their association "
        "with the score is destroyed. Any metric this arm also improves was improved by the "
        "marginal, not by the mapping."
    ),
    SCRAMBLED: (
        "Ablation C, second form. Method C's knots shuffled, so the map is a non-monotone "
        "function of p that still takes the same values. Together with the unprojected arm it "
        "separates 'non-monotone because the strata disagree' from 'non-monotone at random'."
    ),
    ORACLE_POOL: (
        "Ablation F. Isotonic recalibration fitted on every label in the adaptation pool. The "
        "ceiling of what target labels could buy WITHOUT touching the evaluation documents. "
        "Upper bound only; never selected on, never deployed from."
    ),
    ORACLE_EVAL: (
        "Ablation F, absolute form. Isotonic recalibration fitted on the evaluation block "
        "itself. The ceiling of the monotone map class outright. Evaluation only."
    ),
    FEW_SHOT_DIRECT: (
        "Method E, direct. Isotonic recalibration fitted on the N purchased target labels alone."
    ),
    FEW_SHOT_SHRUNK: (
        "Method E, the primary few-shot arm. A convex blend of Method C and the direct fit with "
        "weight N / (N + 50) on the direct fit, so N = 0 is exactly Method C and the transition "
        "the brief's H2 asks for is continuous by construction rather than by luck."
    ),
    FEW_SHOT_COMPOSED: (
        "Method E over the composed map: the same convex blend as the shrunk arm, but anchored "
        "at the composition rather than at Method C, so its zero-label point is the best "
        "target-free map this stage has rather than the primary one. Added at the same moment as "
        "the composed map itself and for the same reason."
    ),
    FEW_SHOT_PRIOR: (
        "Method E, prevalence-anchored. The prior correction of Method A2 with its label-free "
        "prevalence estimate replaced by a shrunk empirical prevalence from the N labels. Aimed "
        "at the level failure specifically, and the cheapest possible use of a label."
    ),
}

# ------------------------------------------------------------------ the cut rules

SOURCE_THRESHOLD = "source_threshold"
QUANTILE_THRESHOLD = "quantile_matched_threshold"
POINT_CUT = "predicted_harm_point"
BOUND_CUT = "predicted_harm_bound"
RULES = (SOURCE_THRESHOLD, QUANTILE_THRESHOLD, POINT_CUT, BOUND_CUT)
PRIMARY_RULE = BOUND_CUT

RULE_NOTES: dict[str, str] = {
    SOURCE_THRESHOLD: (
        "SGV5's published deployed rule, unchanged: the threshold that holds the bound on the "
        "source engines' calibration rows, met with the unseen engine. It does not read the "
        "calibration map at all, so it appears once rather than once per map."
    ),
    QUANTILE_THRESHOLD: (
        "The brief's Method B written on the threshold instead of on the score: F_t^-1(F_s(tau)) "
        "on the decision scale. Take the source threshold's quantile among the source "
        "calibration scores and use the target pool's score at the same quantile. Label-free, "
        "and like the source threshold it does not read the map."
    ),
    POINT_CUT: (
        "The deepest prefix of the target pool, ordered by SGV5's decision score, whose MEAN "
        "CALIBRATED harm estimate is at most epsilon. The threshold is the decision score at "
        "that prefix and it is applied to the evaluation block unchanged."
    ),
    BOUND_CUT: (
        "The same prefix search against an UPPER BOUND on the prefix's harm rather than its "
        "point estimate: the calibrated mean, plus the leave-one-source-engine-out transfer "
        "inflation, plus a one-sided binomial allowance for the size of the accepted set."
    ),
}

# ------------------------------------------------------------------ the strata

OPERATION_COLUMNS = (
    "prov_operation_substitution",
    "prov_operation_pair_substitution",
    "prov_operation_insertion",
    "prov_operation_deletion",
    "prov_operation_split",
    "prov_operation_merge",
)
OPERATION_CLASSES = ("substitution", "insertion", "deletion", "structural")
DIFFICULTY_COLUMN = "prov_site_candidate_count"
DIFFICULTY_TERCILES = (1.0 / 3.0, 2.0 / 3.0)

STRATUM_NOTE = (
    "Method C's strata. The operation class is the brief's 'correction structure statistics' -- "
    "what kind of edit is being proposed -- and the difficulty tercile is its 'candidate "
    "difficulty distribution', taken as how many candidates compete at the same site. Both are "
    "ground-truth-blind and both are available on the target without a single label. The "
    "tercile cut points are computed on the SOURCE calibration rows and frozen; computing them "
    "over a pooled frame would be the leak `.claude/rules/experiment-leakage.md` names."
)


# ------------------------------------------------------------------ the folds


def _calibration_half(document: str) -> int:
    """A deterministic two-way split of the CALIBRATION pages, derived from content.

    `hash()` moves with PYTHONHASHSEED and would make the split differ between processes; a
    sha256 of the document identifier does not. The device is SGV8's, with SGV9's own salt, so
    the two stages' inner folds are independent draws of the same construction rather than the
    same draw reused under a second name.
    """
    digest = hashlib.sha256(f"sgv9-calibration-half|{document}".encode()).digest()
    return digest[0] % 2


def inner_fold(base: dg.Design, held_out: str, target: str) -> rl.Fold:
    """One leave-one-source-engine-out replay, used only to price the transfer error.

    For the outer fold that holds out E, each fit engine F takes a turn as a pseudo-target: the
    map is refitted from the remaining fit engines and its error is measured on F. E appears
    nowhere. No DEVELOPMENT page appears anywhere -- those are the same scans the outer
    evaluation block is read from, and this corpus is matched-source, so a DEVELOPMENT page read
    through F carries the outcome of a DEVELOPMENT page read through E on the same scan.

    The CALIBRATION pages are split in half by content hash: the inner map calibrates on the
    first half through the source engines and is scored on the second half through F. Without
    that halving the inner map would be calibrated on the very pages its error is read from --
    different engine, same scan -- and every inflation this stage computes would be optimistic.
    """
    if target == held_out:
        raise PhaseError(f"the pseudo-target {target!r} is the outer held-out engine")
    role = base.meta["role"].to_numpy(str)
    engine = base.meta["engine_id"].to_numpy(str)
    documents = base.documents
    half = np.array([_calibration_half(name) for name in documents.tolist()])
    sources = tuple(e for e in base.engines if e not in (held_out, target))
    if not sources:
        raise PhaseError(f"{held_out}/{target}: no source engine is left to calibrate from")
    source = np.isin(engine, list(sources))
    fold = rl.Fold(
        held_out=target,
        train_engines=sources,
        fit=np.flatnonzero((role == "TRAIN") & source),
        source_cal=np.flatnonzero((role == "CALIBRATION") & source & (half == 0)),
        seen_eval=np.array([], dtype=int),
        eval=np.flatnonzero((role == "CALIBRATION") & (engine == target) & (half == 1)),
        unlabeled_target=np.flatnonzero((role == "TRAIN") & (engine == target)),
    )
    outcome_documents = set(documents[fold.eval].tolist())
    for name, block in (
        ("fit", fold.fit),
        ("source_cal", fold.source_cal),
        ("unlabeled_target", fold.unlabeled_target),
    ):
        if set(documents[block].tolist()) & outcome_documents:
            raise PhaseError(f"{held_out}/{target}: {name} shares pages with the inner outcome")
        if held_out in set(engine[block].tolist()):
            raise PhaseError(f"{held_out}/{target}: the outer held-out engine reached {name}")
    if held_out in set(engine[fold.eval].tolist()):
        raise PhaseError(f"{held_out}/{target}: the outer held-out engine reached the outcome")
    roles_touched = set(role[np.concatenate([fold.fit, fold.source_cal, fold.eval])].tolist())
    if "DEVELOPMENT" in roles_touched:
        raise PhaseError(f"{held_out}/{target}: a DEVELOPMENT page reached an inner fold")
    if fold.eval.size == 0 or fold.unlabeled_target.size == 0:
        raise PhaseError(f"{held_out}/{target}: the inner fold is empty")
    return fold


@dataclass(slots=True)
class Block:
    """One index block with everything a map may read from it, and the labels kept apart.

    `harmful` and `beneficial` are present because the OUTCOME of a block has to be measurable,
    but no map fitted on a label-free block is given this object -- the fitting functions take
    `SourceView` and `TargetView`, and `tests/leakage` walks their syntax trees.
    """

    index: np.ndarray
    candidate_id: np.ndarray
    documents: np.ndarray
    p_harm: np.ndarray
    p_benefit: np.ndarray
    utility: np.ndarray
    stratum: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray

    @property
    def size(self) -> int:
        return int(self.index.size)


@dataclass(frozen=True, slots=True)
class SourceView:
    """What a map may read from the source engines: their scores, strata AND their labels."""

    p_harm: np.ndarray
    stratum: np.ndarray
    harmful: np.ndarray
    documents: np.ndarray


@dataclass(frozen=True, slots=True)
class TargetView:
    """What a map may read from the unseen engine: scores and strata, and nothing else.

    There is deliberately no `harmful` field and no `beneficial` field. A target-free map that
    wanted a target label could not reach one through this object, and the leakage suite asserts
    that every target-free fitting function names no other source of rows.
    """

    p_harm: np.ndarray
    stratum: np.ndarray


@dataclass(slots=True)
class FoldState:
    """One fold's frozen SGV5 model plus the three blocks this stage is allowed to read."""

    held_out: str
    selected: dict[str, Any]
    lambda_: float
    calibration: Block
    pool: Block
    evaluation: Block
    strata: Strata
    source_thresholds: dict[str, float]
    source_threshold_feasible: dict[str, bool]
    identity: dict[str, Any]
    diagnostics: dict[str, Any]


# ------------------------------------------------------------------ the strata


@dataclass(frozen=True, slots=True)
class Strata:
    """The frozen stratification: how a row is assigned, and what the cells are.

    Fitted on the source calibration rows only. The difficulty cut points are the transformer
    statistics `.claude/rules/experiment-leakage.md` requires to be frozen from the fit side --
    computing them over the target, or over both, would let the target's own distribution decide
    which cell a target row lands in.
    """

    cuts: tuple[float, float]
    names: tuple[str, ...]
    fitted_on: str

    @property
    def size(self) -> int:
        return len(self.names)


def _operation_class(matrix: np.ndarray, names: tuple[str, ...], index: np.ndarray) -> np.ndarray:
    """Which of four edit-operation classes each row proposes.

    The six one-hot columns SGV1 froze are collapsed to four because split and merge together
    account for a small share of the pool and a stratum has to be large enough to carry its own
    reliability curve. A row with no operation flag set -- which the generator does produce --
    falls to `substitution`, the modal class, rather than to a fifth cell of its own.
    """
    position = {name: i for i, name in enumerate(names)}
    missing = [name for name in OPERATION_COLUMNS if name not in position]
    if missing:
        raise PhaseError(f"the design matrix is missing the operation columns {missing}")
    block = matrix[np.ix_(index, [position[name] for name in OPERATION_COLUMNS])]
    assignment = np.zeros(index.size, dtype=int)
    assignment[block[:, 2] > 0.5] = 1  # insertion
    assignment[block[:, 3] > 0.5] = 2  # deletion
    assignment[(block[:, 4] > 0.5) | (block[:, 5] > 0.5)] = 3  # split or merge
    return assignment


def _difficulty(matrix: np.ndarray, names: tuple[str, ...], index: np.ndarray) -> np.ndarray:
    position = {name: i for i, name in enumerate(names)}
    if DIFFICULTY_COLUMN not in position:
        raise PhaseError(f"the design matrix is missing {DIFFICULTY_COLUMN}")
    return np.asarray(matrix[index, position[DIFFICULTY_COLUMN]], dtype=float)


def fit_strata(matrix: np.ndarray, names: tuple[str, ...], source: np.ndarray) -> Strata:
    """Freeze the difficulty tercile cut points on the source calibration rows."""
    values = _difficulty(matrix, names, source)
    cuts = tuple(float(q) for q in np.quantile(values, DIFFICULTY_TERCILES))
    labels = tuple(
        f"{operation}|{tier}"
        for operation in OPERATION_CLASSES
        for tier in ("easy", "medium", "hard")
    )
    return Strata(cuts=(cuts[0], cuts[1]), names=labels, fitted_on="source calibration rows")


def assign_strata(
    strata: Strata, matrix: np.ndarray, names: tuple[str, ...], index: np.ndarray
) -> np.ndarray:
    """Cell index for each row, using the frozen cut points and nothing else."""
    operation = _operation_class(matrix, names, index)
    difficulty = _difficulty(matrix, names, index)
    tier = np.digitize(difficulty, np.asarray(strata.cuts, dtype=float), right=False)
    return operation * 3 + np.clip(tier, 0, 2)


# ------------------------------------------------------------------ rebuilding the frozen model


def _blocks_from(
    design: dg.Design,
    extended: c5.FoldDesign,
    model: c5.Reliability,
    lambda_: float,
    strata: Strata,
    index: np.ndarray,
) -> Block:
    probabilities = model.probabilities(design, extended.block(index))
    harm = np.asarray(probabilities["harm"], dtype=float)
    benefit = np.asarray(probabilities["benefit"], dtype=float)
    return Block(
        index=index,
        candidate_id=design.meta["candidate_id"].astype(str).to_numpy()[index],
        documents=design.documents[index],
        p_harm=harm,
        p_benefit=benefit,
        utility=benefit - lambda_ * harm,
        stratum=assign_strata(strata, design.matrix, design.names, index),
        harmful=design.harmful[index],
        beneficial=design.beneficial[index],
    )


def build_state(
    base: dg.Design,
    fold: rl.Fold,
    selection: dict[str, Any],
    signatures: np.ndarray,
    retrieval_columns: list[int],
    held_out: str,
    frozen: pd.DataFrame | None,
) -> FoldState:
    """Rebuild SGV5's selected model on one fold and materialise the three blocks SGV9 reads.

    `frozen` is SGV5's published score table and is supplied only for the OUTER folds, where the
    rebuilt decision score must equal SGV5's published vector bit for bit. The inner folds have
    no published counterpart -- SGV5 never fitted them -- so they carry no such check, and the
    representation, model class and harm-aversion weight they use are the ones SGV5 froze for the
    outer fold rather than anything SGV9 chooses.
    """
    extended_raw, retrieval = c5.extend_fold(base, fold, signatures, retrieval_columns)
    extended = s6._fill_target_block(
        base, extended_raw, retrieval, fold, signatures, retrieval_columns
    )
    design = extended.design
    representation, model_kind = selection["representation"], selection["model"]
    if model_kind == "ranker":  # pragma: no cover - no fold selected the ranker
        raise PhaseError(
            "a calibration map acts on a probability and the ranker exposes none; a fold that "
            "selected it would need its own construction"
        )
    lambda_ = float(selection["lambda"])
    model = c5.fit_reliability(
        design, fold, extended.columns(representation), model_kind, representation
    )
    strata = fit_strata(design.matrix, design.names, fold.source_cal)

    calibration = _blocks_from(design, extended, model, lambda_, strata, fold.source_cal)
    pool = _blocks_from(design, extended, model, lambda_, strata, fold.unlabeled_target)
    evaluation = _blocks_from(design, extended, model, lambda_, strata, fold.eval)

    identity: dict[str, Any] = {"checked": False}
    if frozen is not None:
        identifiers = design.meta["candidate_id"].astype(str).to_numpy()
        reference = frozen[frozen["held_out_engine"] == held_out].set_index("candidate_id")
        published = reference.loc[identifiers[fold.eval], s6.SGV5_ARM].to_numpy(dtype=float)
        difference = float(np.abs(evaluation.utility - published).max())
        if difference != 0.0:
            raise PhaseError(
                f"{held_out}: the rebuilt SGV5 decision score differs from the published one by "
                f"{difference:g}; every SGV9 number is measured against that score and a drifted "
                "baseline invalidates all of them"
            )
        identity = {"checked": True, "max_abs_difference": difference, "rows": int(fold.eval.size)}

    evaluation_documents = set(evaluation.documents.tolist())
    for name, block in (("calibration", calibration), ("pool", pool)):
        if set(block.documents.tolist()) & evaluation_documents:
            raise PhaseError(f"{held_out}: the {name} block shares pages with the evaluation")

    # One threshold per epsilon, not one at the primary epsilon read against all three. SGV5's
    # rule is "the deepest cut the source calibration rows say holds THIS bound", and comparing a
    # cut chosen for 0.10 against a bound of 0.05 would penalise the baseline for a question it
    # was never asked.
    thresholds: dict[str, float] = {}
    feasible: dict[str, bool] = {}
    for epsilon in EPSILONS:
        key = f"epsilon_{int(epsilon * 100)}"
        decision = select_threshold(
            calibration.utility,
            calibration.harmful,
            epsilon,
            delta=DELTA,
            controller="empirical",
        )
        feasible[key] = bool(decision.feasible)
        thresholds[key] = float(decision.tau) if decision.feasible else float(np.inf)

    diagnostics = {
        "rows": {
            "source_calibration": calibration.size,
            "adaptation_pool": pool.size,
            "evaluation": evaluation.size,
        },
        "documents": {
            "source_calibration": len(set(calibration.documents.tolist())),
            "adaptation_pool": len(set(pool.documents.tolist())),
            "evaluation": len(evaluation_documents),
        },
        "harm_prevalence": {
            "source_calibration": float(calibration.harmful.mean()),
            "adaptation_pool": float(pool.harmful.mean()),
            "evaluation": float(evaluation.harmful.mean()),
            "note": (
                "the pool and the evaluation block are the same engine on different pages. Where "
                "they differ, no pool-fitted map can know it, and the difference is a floor on "
                "this stage's calibration error rather than a defect of any one method."
            ),
        },
        "mean_predicted_harm": {
            "source_calibration": float(calibration.p_harm.mean()),
            "adaptation_pool": float(pool.p_harm.mean()),
            "evaluation": float(evaluation.p_harm.mean()),
        },
        "beneficial_rows": {
            "adaptation_pool": int(pool.beneficial.sum()),
            "evaluation": int(evaluation.beneficial.sum()),
        },
        "strata": {
            "difficulty_cuts": list(strata.cuts),
            "cells": strata.size,
            "source_counts": np.bincount(calibration.stratum, minlength=strata.size).tolist(),
            "pool_counts": np.bincount(pool.stratum, minlength=strata.size).tolist(),
            "evaluation_counts": np.bincount(evaluation.stratum, minlength=strata.size).tolist(),
            "note": STRATUM_NOTE,
        },
    }
    return FoldState(
        held_out=held_out,
        selected=dict(selection),
        lambda_=lambda_,
        calibration=calibration,
        pool=pool,
        evaluation=evaluation,
        strata=strata,
        source_thresholds=thresholds,
        source_threshold_feasible=feasible,
        identity=identity,
        diagnostics=diagnostics,
    )


# ------------------------------------------------------------------ the calibration maps


@dataclass(slots=True)
class Fitted:
    """One fitted calibration map: how to apply it, and enough of it to publish.

    `evaluate` is the map itself. `grid` and `values` are the same map tabulated at the score
    levels it will actually meet, which is what `calibration_mapping.json` carries and what the
    monotonicity check reads. Nothing else in this stage may reconstruct a map from the table --
    the table is a record, not the implementation.
    """

    name: str
    kind: str
    parameters: dict[str, float]
    grid: np.ndarray
    values: np.ndarray
    diagnostics: dict[str, Any]
    evaluate: Any = field(compare=False, repr=False, default=None)
    monotone_required: bool = True

    def apply(self, p_harm: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        out = np.asarray(self.evaluate(np.asarray(p_harm, dtype=float), stratum), dtype=float)
        return np.clip(out, 0.0, 1.0)


def _isotonic(x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> Any:
    """A monotone probability calibrator. One import, one configuration, used everywhere here."""
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True)
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), sample_weight=weight)
    return model


def _mid_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The mid-rank empirical CDF of `values` within `reference`.

    Mid-rank rather than left- or right-continuous because SGV5's harm head is isotonic and
    therefore takes a small number of distinct levels: a one-sided CDF would send an entire
    level to the bottom or the top of its own tie block and make the transport discontinuous
    exactly where most of the mass is.
    """
    ordered = np.sort(np.asarray(reference, dtype=float))
    n = ordered.size
    if n == 0:
        return np.full(np.asarray(values).shape, 0.5)
    low = np.searchsorted(ordered, values, side="left")
    high = np.searchsorted(ordered, values, side="right")
    return (low + high) / (2.0 * n)


def _quantile_of(reference: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The empirical quantile function of `reference`, evaluated at probabilities `u`."""
    ordered = np.sort(np.asarray(reference, dtype=float))
    if ordered.size == 0:
        return np.zeros(np.asarray(u).shape)
    position = np.clip(u * ordered.size - 0.5, 0.0, ordered.size - 1.0)
    return np.asarray(np.interp(position, np.arange(ordered.size, dtype=float), ordered))


def _score_grid(state: FoldState) -> np.ndarray:
    """The score levels a map will be tabulated at: the ones the FITTING blocks contain.

    The evaluation block's own levels are deliberately excluded. Tabulating a map at a score the
    evaluation block happens to contain would not move a fitted value, but it would make the
    published record of the map depend on the held-out engine's DEVELOPMENT rows, and this stage
    keeps that dependency at exactly zero.
    """
    return np.unique(np.concatenate([state.calibration.p_harm, state.pool.p_harm]))


def _source_view(state: FoldState) -> SourceView:
    return SourceView(
        p_harm=state.calibration.p_harm,
        stratum=state.calibration.stratum,
        harmful=state.calibration.harmful.astype(float),
        documents=state.calibration.documents,
    )


def _target_view(state: FoldState) -> TargetView:
    return TargetView(p_harm=state.pool.p_harm, stratum=state.pool.stratum)


# --- Baseline 0 and ablation A ------------------------------------------------------------


def fit_identity(source: SourceView, target: TargetView, grid: np.ndarray) -> Fitted:
    """No calibration. SGV5's harm head read as a probability on an engine it never saw."""

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return p

    return Fitted(
        name=IDENTITY,
        kind="identity",
        parameters={},
        grid=grid,
        values=grid.copy(),
        diagnostics={"note": MAP_NOTES[IDENTITY]},
        evaluate=evaluate,
    )


# --- ablation D: source statistics only ---------------------------------------------------


def fit_source_isotonic(source: SourceView, target: TargetView, grid: np.ndarray) -> Fitted:
    """Recalibrate on the source calibration rows and nothing else.

    SGV5 fitted its harm head's calibrator by isotonic regression on exactly these rows, and
    isotonic regression is the projection onto the monotone cone -- projecting twice is
    projecting once. So this arm is provably the identity, and the residual below is the code
    agreeing with the proof rather than an experimental finding.
    """
    model = _isotonic(source.p_harm, source.harmful)

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(p), dtype=float)

    values = evaluate(grid, np.zeros(grid.size, dtype=int))
    return Fitted(
        name=SOURCE_ISOTONIC,
        kind="table",
        parameters={},
        grid=grid,
        values=values,
        diagnostics={
            "max_abs_deviation_from_identity_on_the_source_grid": float(
                np.abs(values - grid).max()
            ),
            "note": MAP_NOTES[SOURCE_ISOTONIC],
        },
        evaluate=evaluate,
    )


# --- Method A: temperature ------------------------------------------------------------------


def fit_temperature(source: SourceView, target: TargetView, grid: np.ndarray) -> Fitted:
    """p' = sigmoid(logit(p) / T), with T matching the target's logit spread to the source's.

    The brief asks for a temperature estimated from source calibration and target UNLABELLED
    statistics, and the only statistic of the target available without a label is the score
    distribution. Its dispersion in logit space is the natural match: a target whose logits are
    more spread out than the source's is stating more confidence than the calibrator was fitted
    to support, and T > 1 removes exactly that much of it. A temperature cannot move the mean
    independently of the spread, which is the limitation this arm exists to demonstrate.
    """
    source_logit = _logit(np.clip(source.p_harm, PROB_FLOOR, 1.0 - PROB_FLOOR))
    target_logit = _logit(np.clip(target.p_harm, PROB_FLOOR, 1.0 - PROB_FLOOR))
    source_sd = float(np.std(source_logit))
    target_sd = float(np.std(target_logit))
    degenerate = not (np.isfinite(source_sd) and np.isfinite(target_sd) and source_sd > 0.0)
    temperature = 1.0 if degenerate else max(target_sd / source_sd, 1e-3)

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        clipped = np.clip(p, PROB_FLOOR, 1.0 - PROB_FLOOR)
        return np.asarray(_sigmoid(_logit(clipped) / temperature), dtype=float)

    return Fitted(
        name=TEMPERATURE,
        kind="temperature",
        parameters={"temperature": temperature},
        grid=grid,
        values=evaluate(grid, np.zeros(grid.size, dtype=int)),
        diagnostics={
            "source_logit_sd": source_sd,
            "target_logit_sd": target_sd,
            "degenerate": bool(degenerate),
            "note": MAP_NOTES[TEMPERATURE],
        },
        evaluate=evaluate,
    )


# --- Method A2: label-free prior correction --------------------------------------------------

BBSE_BINS = 10


def _prior_correction(p: np.ndarray, odds_ratio: float) -> np.ndarray:
    """The standard prior-shift correction, written on the odds scale.

    p' = r p / (r p + 1 - p) with r the ratio of target to source prior odds. Its derivative is
    r / (r p + 1 - p)^2, strictly positive for r > 0, so the map is strictly increasing and the
    brief's ordering constraint holds for every positive r without needing to be enforced.
    """
    clipped = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    numerator = odds_ratio * clipped
    return np.asarray(numerator / np.maximum(numerator + (1.0 - clipped), 1e-12), dtype=float)


def estimate_target_prevalence(source: SourceView, target: TargetView) -> dict[str, Any]:
    """A black-box shift estimate of the target harm prevalence, from scores alone.

    Bin the score by source quantiles; the source's joint distribution over (bin, label) and the
    target's marginal distribution over bins determine the target's label prior, PROVIDED the
    conditional score distribution given the label is the same on both engines. That proviso is
    the label-shift assumption, it is not testable without target labels, and negative test 2
    measures how far from true it is on each engine by comparing this estimate with the pool's
    actual prevalence -- a comparison that is evaluation-only and reaches no fitted quantity.
    """
    from scipy.optimize import nnls

    edges = np.unique(np.quantile(source.p_harm, np.linspace(0.0, 1.0, BBSE_BINS + 1))[1:-1])
    source_bin = np.digitize(source.p_harm, edges, right=False)
    target_bin = np.digitize(target.p_harm, edges, right=False)
    n_bins = edges.size + 1

    joint = np.zeros((n_bins, 2), dtype=float)
    for label in (0, 1):
        mask = source.harmful > 0.5 if label == 1 else source.harmful <= 0.5
        joint[:, label] = np.bincount(source_bin[mask], minlength=n_bins) / max(
            source.p_harm.size, 1
        )
    marginal = np.bincount(target_bin, minlength=n_bins) / max(target.p_harm.size, 1)

    weights, residual = nnls(joint, marginal)
    source_prior = float(source.harmful.mean())
    raw = np.array([weights[0] * (1.0 - source_prior), weights[1] * source_prior], dtype=float)
    total = float(raw.sum())
    degenerate = not np.isfinite(total) or total <= 0.0
    estimate = source_prior if degenerate else float(raw[1] / total)
    clipped = float(np.clip(estimate, 0.01, 0.99))
    return {
        "estimate": clipped,
        "unclipped_estimate": estimate,
        "clipped": bool(clipped != estimate),
        "degenerate": bool(degenerate),
        "source_prevalence": source_prior,
        "bins": int(n_bins),
        "residual": float(residual),
        "weights": [float(weights[0]), float(weights[1])],
    }


def _odds(prevalence: float) -> float:
    bounded = float(np.clip(prevalence, 1e-6, 1.0 - 1e-6))
    return bounded / (1.0 - bounded)


def fit_prior_shift(source: SourceView, target: TargetView, grid: np.ndarray) -> Fitted:
    """Method A2: estimate the target's harm prevalence from its scores, then correct the prior."""
    shift = estimate_target_prevalence(source, target)
    ratio = _odds(shift["estimate"]) / _odds(shift["source_prevalence"])

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return _prior_correction(p, ratio)

    return Fitted(
        name=PRIOR_SHIFT,
        kind="prior",
        parameters={"odds_ratio": float(ratio), "estimated_prevalence": shift["estimate"]},
        grid=grid,
        values=evaluate(grid, np.zeros(grid.size, dtype=int)),
        diagnostics={**shift, "note": MAP_NOTES[PRIOR_SHIFT]},
        evaluate=evaluate,
    )


# --- Method B: marginal rank transport -------------------------------------------------------


def _reliability_curve(p_harm: np.ndarray, harmful: np.ndarray) -> Any:
    """The source's own reliability: what fraction of rows at this score were actually harmful."""
    return _isotonic(p_harm, harmful)


def _transport(source_scores: np.ndarray, target_scores: np.ndarray, curve: Any) -> Any:
    """R_s(F_s^-1(F_t(p))): the target row's rank, the source's score at that rank, its risk.

    Every one of the three steps is non-decreasing, so the composition is non-decreasing and the
    brief's ordering constraint holds by construction rather than by enforcement.
    """

    def evaluate(p: np.ndarray) -> np.ndarray:
        u = _mid_cdf(target_scores, p)
        matched = _quantile_of(source_scores, u)
        return np.asarray(curve.predict(matched), dtype=float)

    return evaluate


def fit_rank_transport(source: SourceView, target: TargetView, grid: np.ndarray) -> Fitted:
    """Method B. Assumes rank, not level, carries the risk across engines.

    Its blind spot is stated rather than discovered: the mean of this map over the target pool is
    the source's mean harm rate, whatever the target's actually is, because a rank transport
    carries the source's marginal onto the target unchanged. Where prevalence really did shift --
    and on this corpus it shifts from 0.49 to 0.80 across the four folds -- Method B cannot see it.
    """
    curve = _reliability_curve(source.p_harm, source.harmful)
    transported = _transport(source.p_harm, target.p_harm, curve)

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return transported(p)

    values = evaluate(grid, np.zeros(grid.size, dtype=int))
    return Fitted(
        name=RANK_TRANSPORT,
        kind="table",
        parameters={},
        grid=grid,
        values=values,
        diagnostics={
            "mean_on_the_target_pool": float(evaluate(target.p_harm, target.stratum).mean()),
            "source_prevalence": float(source.harmful.mean()),
            "note": MAP_NOTES[RANK_TRANSPORT],
        },
        evaluate=evaluate,
    )


# --- Method C: stratified quantile risk transport --------------------------------------------


def _stratified_raw(
    source: SourceView, target: TargetView, n_strata: int
) -> tuple[Any, dict[str, Any]]:
    """The transport computed inside each stratum, with each stratum's curve shrunk to the pool.

    A stratum with a handful of source rows would otherwise fit its own noise and export it to
    the target as if it were structure. The shrinkage weight is fixed at STRATUM_SHRINKAGE rows
    before any result is read, and the per-stratum counts are published so a reader can see which
    cells are carried by their own data and which by the pooled curve.
    """
    pooled_curve = _reliability_curve(source.p_harm, source.harmful)
    per_stratum: list[Any] = []
    counts: list[dict[str, int]] = []
    for cell in range(n_strata):
        source_mask = source.stratum == cell
        target_mask = target.stratum == cell
        n_source = int(source_mask.sum())
        n_target = int(target_mask.sum())
        counts.append({"source": n_source, "target": n_target})
        if n_source < 2 or n_target < 2:
            per_stratum.append(None)
            continue
        cell_curve = _reliability_curve(source.p_harm[source_mask], source.harmful[source_mask])
        weight = n_source / (n_source + STRATUM_SHRINKAGE)
        cell_transport = _transport(
            source.p_harm[source_mask], target.p_harm[target_mask], cell_curve
        )
        pooled_transport = _transport(
            source.p_harm[source_mask], target.p_harm[target_mask], pooled_curve
        )
        per_stratum.append((weight, cell_transport, pooled_transport))
    fallback = _transport(source.p_harm, target.p_harm, pooled_curve)

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        out = np.empty(np.asarray(p).shape, dtype=float)
        for cell in range(n_strata):
            mask = np.asarray(stratum) == cell
            if not mask.any():
                continue
            entry = per_stratum[cell]
            if entry is None:
                out[mask] = fallback(np.asarray(p)[mask])
                continue
            weight, cell_transport, pooled_transport = entry
            values = np.asarray(p)[mask]
            out[mask] = weight * cell_transport(values) + (1.0 - weight) * pooled_transport(values)
        return out

    return evaluate, {
        "counts": counts,
        "cells_with_their_own_curve": int(sum(1 for e in per_stratum if e is not None)),
        "shrinkage_rows": STRATUM_SHRINKAGE,
    }


def _monotone_violation(grid: np.ndarray, values: np.ndarray) -> float:
    """How far the tabulated map falls below its own running maximum. Zero iff non-decreasing."""
    order = np.argsort(grid, kind="stable")
    ordered = np.asarray(values, dtype=float)[order]
    return float(np.max(np.maximum.accumulate(ordered) - ordered)) if ordered.size else 0.0


def fit_quantile_transport(
    source: SourceView,
    target: TargetView,
    grid: np.ndarray,
    n_strata: int,
    project: bool = True,
) -> Fitted:
    """Method C, the primary map, and its unprojected form.

    The stratified transport is monotone in p inside every stratum but not across them: two rows
    with the same score in different strata get different estimates, and the ordering the brief
    requires is a statement about scores, not about scores-within-a-cell. The projection is an
    isotonic regression of the stratified estimate on the score, fitted over the target pool, so
    the constraint is imposed where the map will actually be used and weighted by how often each
    score occurs there.
    """
    raw, strata_record = _stratified_raw(source, target, n_strata)
    raw_on_pool = raw(target.p_harm, target.stratum)
    violation = _monotone_violation(target.p_harm, raw_on_pool)

    if not project:

        def unprojected(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
            return raw(p, stratum)

        return Fitted(
            name=UNPROJECTED,
            kind="stratified_table",
            parameters={},
            grid=grid,
            values=unprojected(grid, np.zeros(grid.size, dtype=int)),
            diagnostics={
                **strata_record,
                "monotone_violation_on_the_pool": violation,
                "note": MAP_NOTES[UNPROJECTED],
            },
            evaluate=unprojected,
            monotone_required=False,
        )

    projection = _isotonic(target.p_harm, raw_on_pool)

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return np.asarray(projection.predict(p), dtype=float)

    projected_on_pool = evaluate(target.p_harm, target.stratum)
    return Fitted(
        name=QUANTILE_TRANSPORT,
        kind="table",
        parameters={},
        grid=grid,
        values=evaluate(grid, np.zeros(grid.size, dtype=int)),
        diagnostics={
            **strata_record,
            "monotone_violation_before_projection": violation,
            "mean_abs_change_from_the_projection": float(
                np.abs(projected_on_pool - raw_on_pool).mean()
            ),
            "mean_on_the_target_pool": float(projected_on_pool.mean()),
            "note": MAP_NOTES[QUANTILE_TRANSPORT],
        },
        evaluate=evaluate,
    )


def fit_no_strata(source: SourceView, target: TargetView, grid: np.ndarray) -> Fitted:
    """Ablation E: Method C with one stratum, so the only thing left of Method C is its projection.

    A marginal rank transport is already non-decreasing, so the isotonic projection has no
    ordering to repair and the arm has no degrees of freedom of its own beyond that. It is NOT
    identical to Method B: the transport is a step function of the score, the projection is
    fitted at the pool's score levels and interpolates linearly between them, so the two arms
    agree exactly wherever the projection saw a score and can differ by up to one step where it
    did not. `--ablation` measures that difference on every block rather than asserting it away.
    """
    single = TargetView(p_harm=target.p_harm, stratum=np.zeros(target.p_harm.size, dtype=int))
    single_source = SourceView(
        p_harm=source.p_harm,
        stratum=np.zeros(source.p_harm.size, dtype=int),
        harmful=source.harmful,
        documents=source.documents,
    )
    fitted = fit_quantile_transport(single_source, single, grid, n_strata=1, project=True)
    return Fitted(
        name=NO_STRATA,
        kind=fitted.kind,
        parameters=fitted.parameters,
        grid=fitted.grid,
        values=fitted.values,
        diagnostics={**fitted.diagnostics, "note": MAP_NOTES[NO_STRATA]},
        evaluate=fitted.evaluate,
    )


# --- ablations B and C: the destroyed maps ---------------------------------------------------


def fit_scrambled(primary: Fitted, seed: int) -> Fitted:
    """Ablation C, second form: Method C's values shuffled across its own knots.

    The map still takes exactly the values Method C takes, in exactly the same proportions, and
    is a deterministic function of the score. What it no longer is, is monotone. Any metric this
    arm preserves was never carried by the ordering.
    """
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(primary.values, dtype=float).copy()
    rng.shuffle(shuffled)
    grid = np.asarray(primary.grid, dtype=float)

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return np.asarray(np.interp(np.asarray(p, dtype=float), grid, shuffled), dtype=float)

    return Fitted(
        name=SCRAMBLED,
        kind="table",
        parameters={},
        grid=grid,
        values=shuffled,
        diagnostics={
            "monotone_violation": _monotone_violation(grid, shuffled),
            "note": MAP_NOTES[SCRAMBLED],
        },
        evaluate=evaluate,
        monotone_required=False,
    )


def permuted_estimates(estimates: np.ndarray, seed: int) -> np.ndarray:
    """Ablation B: Method C's estimates permuted across rows of one block.

    This is not a function of the score and so has no map object. Its marginal distribution is
    exactly Method C's, which is the point: any calibration metric it also improves was improved
    by the shape of the marginal and not by which row got which value.
    """
    rng = np.random.default_rng(seed)
    return np.asarray(estimates, dtype=float)[rng.permutation(estimates.size)]


# --- ablation F: the two oracles --------------------------------------------------------------


def fit_oracle(name: str, p_harm: np.ndarray, harmful: np.ndarray, grid: np.ndarray) -> Fitted:
    """An isotonic recalibration fitted on the labels of the block it will be scored on.

    Upper bound only. `oracle_pool` uses the adaptation pool's labels and so is the ceiling of
    what target supervision could buy without touching the evaluation documents; `oracle_eval`
    uses the evaluation block's own labels and so is the ceiling of the monotone map class
    outright. Neither is a method, neither is selected on, and both are excluded from every
    criterion.
    """
    model = _isotonic(p_harm, harmful.astype(float))

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(p), dtype=float)

    return Fitted(
        name=name,
        kind="table",
        parameters={},
        grid=grid,
        values=evaluate(grid, np.zeros(grid.size, dtype=int)),
        diagnostics={"fitted_rows": int(p_harm.size), "note": MAP_NOTES[name]},
        evaluate=evaluate,
    )


# --- Method E: few-shot -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BudgetView:
    """The N purchased target labels: the only labelled target rows a Method E arm may read.

    Drawn by SGV6's random acquisition, from SGV6's pool, under SGV6's seed formula, so a cell
    at budget N here contains the same rows SGV6 and SGV7 bought at budget N with the same seed.
    `tests/leakage` asserts that identity rather than trusting the coincidence.
    """

    positions: np.ndarray
    p_harm: np.ndarray
    stratum: np.ndarray
    harmful: np.ndarray
    documents: np.ndarray

    @property
    def size(self) -> int:
        return int(self.positions.size)


def draw_budget(pool: Block, held_out: str, size: int, seed: int) -> BudgetView:
    order = np.asarray(
        np.random.default_rng(_stable_seed(held_out, s6.RANDOM, str(seed))).permutation(pool.size),
        dtype=int,
    )
    positions = order[:size]
    return BudgetView(
        positions=positions,
        p_harm=pool.p_harm[positions],
        stratum=pool.stratum[positions],
        harmful=pool.harmful[positions].astype(float),
        documents=pool.documents[positions],
    )


def _anchor_odds_ratio(values: np.ndarray, target_mean: float) -> float:
    """The odds ratio that moves a probability vector's MEAN onto a stated value.

    The mean of the prior-corrected vector is strictly increasing in the odds ratio -- each row
    is -- so a bisection on log r converges to the unique solution wherever one exists, and
    where none does (a target mean outside the vector's own reachable range) the search returns
    the nearest endpoint and the caller records that it was clamped.
    """
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if values.size == 0 or not np.isfinite(target_mean):
        return 1.0
    low, high = -12.0, 12.0
    if float(_prior_correction(values, float(np.exp(high))).mean()) < target_mean:
        return float(np.exp(high))
    if float(_prior_correction(values, float(np.exp(low))).mean()) > target_mean:
        return float(np.exp(low))
    for _ in range(80):
        middle = 0.5 * (low + high)
        if float(_prior_correction(values, float(np.exp(middle))).mean()) < target_mean:
            low = middle
        else:
            high = middle
    return float(np.exp(0.5 * (low + high)))


def fit_transport_prior(
    primary: Fitted, target: TargetView, zero_label: dict[str, Any], grid: np.ndarray
) -> Fitted:
    """Method C, then Method A2's level correction. Transport the ranks, then move the level.

    The two target-free mechanisms this stage has are complementary and neither is sufficient.
    A rank transport reproduces the source's marginal on the target whatever the target's own
    prevalence is; a prior correction moves the marginal but knows nothing about which rows
    should move. Composing them uses each for what it can do: the transport decides the shape,
    the anchor decides the level. Both factors are non-decreasing in the score, so the ordering
    constraint survives the composition.
    """
    transported = primary.apply(target.p_harm, target.stratum)
    wanted = float(zero_label["estimate"])
    ratio = _anchor_odds_ratio(transported, wanted)
    achieved = float(_prior_correction(transported, ratio).mean())

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return _prior_correction(primary.apply(p, stratum), ratio)

    return Fitted(
        name=TRANSPORT_PRIOR,
        kind="composed",
        parameters={"odds_ratio": float(ratio), "anchored_prevalence": wanted},
        grid=grid,
        values=evaluate(grid, np.zeros(grid.size, dtype=int)),
        diagnostics={
            "mean_before_the_anchor": float(transported.mean()),
            "mean_after_the_anchor": achieved,
            "clamped": bool(abs(achieved - wanted) > 1e-6),
            "label_free_prevalence": wanted,
            "note": MAP_NOTES[TRANSPORT_PRIOR],
        },
        evaluate=evaluate,
    )


def fit_few_shot_direct(budget: BudgetView, grid: np.ndarray) -> Fitted:
    """Isotonic recalibration on the purchased labels alone, and nothing else.

    At N = 5 this is close to degenerate and is expected to be: five Bernoulli draws cannot
    resolve a probability to better than about 0.2, and the arm is here to show what a naive
    use of a small budget costs rather than to win.
    """
    if budget.size < 2 or budget.harmful.min() == budget.harmful.max():

        def degenerate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
            if budget.size == 0:
                return np.asarray(p, dtype=float)
            return np.full(np.asarray(p).shape, float(budget.harmful.mean()))

        return Fitted(
            name=FEW_SHOT_DIRECT,
            kind="constant" if budget.size else "identity",
            parameters={"n_labels": float(budget.size)},
            grid=grid,
            values=degenerate(grid, np.zeros(grid.size, dtype=int)),
            diagnostics={"degenerate": True, "note": MAP_NOTES[FEW_SHOT_DIRECT]},
            evaluate=degenerate,
        )
    model = _isotonic(budget.p_harm, budget.harmful)

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(p), dtype=float)

    return Fitted(
        name=FEW_SHOT_DIRECT,
        kind="table",
        parameters={"n_labels": float(budget.size)},
        grid=grid,
        values=evaluate(grid, np.zeros(grid.size, dtype=int)),
        diagnostics={"degenerate": False, "note": MAP_NOTES[FEW_SHOT_DIRECT]},
        evaluate=evaluate,
    )


def fit_few_shot_shrunk(
    target_free: Fitted,
    direct: Fitted,
    budget: BudgetView,
    grid: np.ndarray,
    name: str = FEW_SHOT_SHRUNK,
) -> Fitted:
    """A convex blend of Method C and the direct fit, weight N / (N + 50) on the direct fit.

    Convexity is what makes the brief's H2 transition continuous by construction: at N = 0 the
    arm IS Method C, at large N it is the direct fit, and a convex combination of two
    non-decreasing maps is non-decreasing, so the ordering constraint survives the blend. The
    blend weight was fixed before any result was read and is not tuned per engine or per budget.
    """
    weight = budget.size / (budget.size + SHRINKAGE_PRIOR)

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return (1.0 - weight) * target_free.apply(p, stratum) + weight * direct.apply(p, stratum)

    return Fitted(
        name=name,
        kind="blend",
        parameters={"n_labels": float(budget.size), "weight_on_the_direct_fit": float(weight)},
        grid=grid,
        values=evaluate(grid, np.zeros(grid.size, dtype=int)),
        diagnostics={"anchored_at": target_free.name, "note": MAP_NOTES[name]},
        evaluate=evaluate,
    )


def fit_few_shot_prior(
    source: SourceView, zero_label: dict[str, Any], budget: BudgetView, grid: np.ndarray
) -> Fitted:
    """The prior correction of Method A2 with the prevalence shrunk toward the purchased labels.

    The cheapest possible use of a label: N labels estimate one number, the target's harm
    prevalence, and that number is exactly what a rank transport cannot see and a temperature
    cannot move on its own. At N = 0 the arm is Method A2 exactly.
    """
    label_free = float(zero_label["estimate"])
    weight = budget.size / (budget.size + SHRINKAGE_PRIOR)
    empirical = float(budget.harmful.mean()) if budget.size else label_free
    prevalence = float(np.clip((1.0 - weight) * label_free + weight * empirical, 0.01, 0.99))
    ratio = _odds(prevalence) / _odds(float(zero_label["source_prevalence"]))

    def evaluate(p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        return _prior_correction(p, ratio)

    return Fitted(
        name=FEW_SHOT_PRIOR,
        kind="prior",
        parameters={
            "n_labels": float(budget.size),
            "odds_ratio": float(ratio),
            "estimated_prevalence": prevalence,
            "weight_on_the_purchased_labels": float(weight),
        },
        grid=grid,
        values=evaluate(grid, np.zeros(grid.size, dtype=int)),
        diagnostics={
            "label_free_prevalence": label_free,
            "empirical_prevalence": empirical,
            "note": MAP_NOTES[FEW_SHOT_PRIOR],
        },
        evaluate=evaluate,
    )


# ------------------------------------------------------------------ the cut rules


@dataclass(frozen=True, slots=True)
class PrefixCurves:
    """Everything a cut rule needs, computed once per (block, estimate) pair.

    Ordered by SGV5's decision score, descending, with `kind="stable"` so ties resolve by row
    order identically in every run. `effective_n` is the Kish effective sample size of the
    prefix's DOCUMENTS -- the resampling unit this project uses everywhere -- because a prefix of
    400 candidates drawn from 30 receipts carries nothing like 400 independent observations.
    """

    order: np.ndarray
    utility: np.ndarray
    predicted: np.ndarray
    realized: np.ndarray
    repair_recall: np.ndarray
    effective_n: np.ndarray
    total_beneficial: int


def _occurrence_index(codes: np.ndarray) -> np.ndarray:
    """For each position, how many earlier positions share its document.

    The k-th candidate from a document raises that document's squared count by 2(k-1)+1, so a
    cumulative sum of this array is the running sum of squared cluster sizes -- the denominator
    of the Kish effective sample size -- without a Python loop over the block.
    """
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    ordered = np.asarray(codes)[order]
    starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))
    lengths = np.diff(np.concatenate([starts, [ordered.size]]))
    within = np.arange(ordered.size, dtype=np.int64) - np.repeat(starts, lengths)
    out = np.empty(codes.size, dtype=np.int64)
    out[order] = within
    return out


def prefix_curves(block: Block, estimates: np.ndarray) -> PrefixCurves:
    order = np.argsort(-block.utility, kind="stable")
    counts = np.arange(1, block.size + 1, dtype=float)
    harmful = block.harmful[order].astype(float)
    beneficial = block.beneficial[order].astype(float)
    total = int(block.beneficial.sum())

    codes = pd.factorize(block.documents[order])[0]
    sum_of_squares = np.cumsum(2.0 * _occurrence_index(codes) + 1.0)

    return PrefixCurves(
        order=order,
        utility=block.utility[order],
        predicted=np.cumsum(np.asarray(estimates, dtype=float)[order]) / counts,
        realized=np.cumsum(harmful) / counts,
        repair_recall=np.cumsum(beneficial) / max(total, 1),
        effective_n=counts**2 / np.maximum(sum_of_squares, 1.0),
        total_beneficial=total,
    )


def _deepest(rate: np.ndarray, epsilon: float) -> int:
    """The most permissive prefix whose rate is within epsilon; -1 if none is.

    The search takes the LAST feasible rank rather than stopping at the first infeasible one:
    the rate is not monotone in the rank, and stopping early would silently pick a shallower cut
    than the rule specifies. SGV7 made and documented the same choice for the same reason.
    """
    feasible = np.flatnonzero(np.asarray(rate, dtype=float) <= epsilon)
    return int(feasible[-1]) if feasible.size else -1


def harm_upper_bound(curves: PrefixCurves, inflation: float, delta: float = DELTA) -> np.ndarray:
    """An upper bound on a prefix's realised harm rate, from three terms named separately.

    The point estimate is the calibrated mean. The transfer term is how wrong the same pipeline
    was when it was replayed onto a source engine it had not calibrated from, which is the only
    honest reference for an engine nobody has labels for. The sampling term is a one-sided
    normal allowance at the prefix's effective document count, because the realised rate on the
    batch that is actually deployed on is a finite sample even when the estimate is right.
    """
    from ocr_risk.stats.multiplicity import _normal_quantile

    z = _normal_quantile(1.0 - delta)
    point = np.clip(curves.predicted, 0.0, 1.0)
    sampling = z * np.sqrt(
        np.maximum(point * (1.0 - point), 0.0) / np.maximum(curves.effective_n, 1.0)
    )
    return point + float(inflation) + sampling


def _deployed_at(block: Block, tau: float, epsilon: float) -> dict[str, Any]:
    """What the threshold does on the evaluation block, at one epsilon.

    Coverage, realised harm and repair recall come from `c5._deployed`, which is the one
    implementation of the deployed operating point in this repository. Only the comparison
    against epsilon is local, because that function reports it against the primary epsilon and
    this stage reports all three.
    """
    point = dict(deployed_point(block.utility, block.harmful, block.beneficial, tau))
    realized = float(point["realized_harm_rate"])
    holds = True if point["accepts_nothing"] else bool(realized <= epsilon)
    point["epsilon"] = float(epsilon)
    point["bound_violation"] = (
        0.0 if point["accepts_nothing"] else float(max(0.0, realized - epsilon))
    )
    point["holds_bound"] = holds
    point["tau"] = float(tau)
    return point


def place_cut(
    pool: Block,
    estimates: np.ndarray,
    epsilon: float,
    rule: str,
    inflation: float,
    state: FoldState,
) -> dict[str, Any]:
    """Choose a deployment threshold. Reads target SCORES, target strata, and no target label.

    Four rules, three of which are label-free on the target and one of which -- the source
    threshold -- does not read the target at all. None of them reaches the evaluation block: the
    prefix search runs over the adaptation pool and only the resulting scalar leaves this
    function. `tests/leakage` walks this function's syntax tree and fails if it names an outcome
    array, an evaluation block or an oracle.
    """
    key = f"epsilon_{int(epsilon * 100)}"
    source_tau = state.source_thresholds[key]
    if rule == SOURCE_THRESHOLD:
        return {
            "tau": source_tau,
            "feasible": state.source_threshold_feasible[key],
            "pool_rank": -1,
            "pool_predicted_harm": float("nan"),
            "pool_bound": float("nan"),
        }
    curves = prefix_curves(pool, estimates)
    if rule == QUANTILE_THRESHOLD:
        if not state.source_threshold_feasible[key]:
            return {
                "tau": source_tau,
                "feasible": False,
                "pool_rank": -1,
                "pool_predicted_harm": float("nan"),
                "pool_bound": float("nan"),
            }
        quantile = float(np.mean(state.calibration.utility <= source_tau))
        tau = float(np.quantile(pool.utility, quantile))
        rank = int(np.count_nonzero(pool.utility >= tau))
        return {
            "tau": tau,
            "feasible": True,
            "pool_rank": rank,
            "pool_predicted_harm": float(
                curves.predicted[min(max(rank, 1), curves.predicted.size) - 1]
            ),
            "pool_bound": float("nan"),
            "source_threshold_quantile": quantile,
        }
    criterion = curves.predicted if rule == POINT_CUT else harm_upper_bound(curves, inflation)
    index = _deepest(criterion, epsilon)
    if index < 0:
        return {
            "tau": float(np.inf),
            "feasible": False,
            "pool_rank": 0,
            "pool_predicted_harm": float("nan"),
            "pool_bound": float("nan"),
        }
    return {
        "tau": float(curves.utility[index]),
        "feasible": True,
        "pool_rank": index + 1,
        "pool_predicted_harm": float(curves.predicted[index]),
        "pool_bound": float(criterion[index]),
        "pool_effective_n": float(curves.effective_n[index]),
    }


# ------------------------------------------------------------------ Method D: the transfer bound


def _transfer_error(state: FoldState, epsilon: float) -> dict[str, Any]:
    """Replay the whole procedure on one pseudo-target and record how wrong the estimate was.

    The quantity is `realised harm minus calibrated estimate` over the rows the procedure would
    actually have accepted -- not a per-row residual. A per-row residual on a Bernoulli outcome
    is dominated by the outcome's own variance and says nothing about whether an accept SET's
    harm rate was estimated correctly, which is the only thing the bound has to get right.
    """
    grid = _score_grid(state)
    fitted = fit_quantile_transport(
        _source_view(state), _target_view(state), grid, state.strata.size
    )
    pool_estimates = fitted.apply(state.pool.p_harm, state.pool.stratum)
    placed = place_cut(
        state.pool,
        pool_estimates,
        epsilon,
        POINT_CUT,
        0.0,
        state,
    )
    if not placed["feasible"]:
        return {"pseudo_target": state.held_out, "error": float("nan"), "n_accepted": 0}
    outcome = state.evaluation
    accepted = np.isfinite(outcome.utility) & (outcome.utility >= placed["tau"])
    if not accepted.any():
        return {"pseudo_target": state.held_out, "error": float("nan"), "n_accepted": 0}
    estimates = fitted.apply(outcome.p_harm, outcome.stratum)
    realized = float(outcome.harmful[accepted].astype(float).mean())
    predicted = float(estimates[accepted].mean())
    return {
        "pseudo_target": state.held_out,
        "error": realized - predicted,
        "realized_harm_rate": realized,
        "predicted_harm_rate": predicted,
        "n_accepted": int(accepted.sum()),
        "pool_rank": placed["pool_rank"],
        "tau": placed["tau"],
    }


def transfer_inflation(
    base: dg.Design,
    held_out: str,
    selection: dict[str, Any],
    signatures: np.ndarray,
    retrieval_columns: list[int],
) -> dict[str, Any]:
    """How wrong Method C is on an engine it did not calibrate from, priced on source engines.

    Each fit engine takes a turn as a pseudo-target and the entire pipeline is refitted from the
    remaining fit engines. Three replays, so three errors per epsilon, and the inflation is the
    MAXIMUM of the three rather than a smoothed quantile: with three exchangeable calibration
    points the distribution-free coverage a quantile can support is 3/4, and taking the maximum
    is the only choice that attains it. The stage reports 3/4 and does not claim 1 - alpha.

    The replays are also structurally harder than the fold they are used on -- each is calibrated
    from two engines where the outer fold has three -- so the inflation is biased upward. For a
    bound that is the safe direction, and it is recorded rather than corrected.
    """
    sources = tuple(engine for engine in base.engines if engine != held_out)
    per_epsilon: dict[str, Any] = {}
    replays: dict[str, list[dict[str, Any]]] = {}
    for pseudo_target in sources:
        fold = inner_fold(base, held_out, pseudo_target)
        state = build_state(
            base, fold, selection, signatures, retrieval_columns, pseudo_target, None
        )
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            replays.setdefault(key, []).append(_transfer_error(state, epsilon))
    for key, rows in replays.items():
        errors = [row["error"] for row in rows if np.isfinite(row["error"])]
        per_epsilon[key] = {
            "inflation": float(max(errors)) if errors else 0.0,
            "errors": {row["pseudo_target"]: row["error"] for row in rows},
            "replays": rows,
            "usable_replays": len(errors),
        }
    pooled = [
        row["error"] for rows in replays.values() for row in rows if np.isfinite(row["error"])
    ]
    return {
        "per_epsilon": per_epsilon,
        "pooled_inflation": float(max(pooled)) if pooled else 0.0,
        "source_engines": list(sources),
        "distribution_free_coverage": CONFORMAL_COVERAGE,
        "nominal_alpha": CONFORMAL_ALPHA,
        "note": (
            "the inflation is the largest error the pipeline made across a leave-one-"
            "source-engine-out replay. With three exchangeable calibration engines the "
            "guaranteed coverage is 3/4; SGV9 states that and does not state 1 - alpha."
        ),
    }


def fit_conformal(primary: Fitted, inflation: dict[str, Any]) -> Fitted:
    """Method D. Method C's estimate, carried with the transfer error it is entitled to expect.

    The map itself is Method C's -- Method D adds no sharpening and is not scored as a point
    estimate. What it adds is the inflation the cut rule uses, and the three-way low / uncertain
    / high classification the brief asks for, both of which live on the epsilon axis rather than
    on the score axis and so cannot be folded into a single transformed probability.
    """
    return Fitted(
        name=CONFORMAL,
        kind=primary.kind,
        parameters=dict(primary.parameters),
        grid=primary.grid,
        values=primary.values,
        diagnostics={
            **primary.diagnostics,
            "inflation_per_epsilon": {
                key: cell["inflation"] for key, cell in inflation["per_epsilon"].items()
            },
            "distribution_free_coverage": CONFORMAL_COVERAGE,
            "note": MAP_NOTES[CONFORMAL],
        },
        evaluate=primary.evaluate,
    )


def risk_sets(estimates: np.ndarray, epsilon: float, inflation: float) -> dict[str, Any]:
    """The brief's low / uncertain / high partition, on the inflated interval.

    A row is `low` when even the inflated estimate is within epsilon, `high` when even the
    deflated estimate is above it, and `uncertain` when the interval straddles epsilon. The
    interval is the same one the bound cut uses, so the partition and the deployment decision
    cannot disagree.
    """
    upper = np.clip(np.asarray(estimates, dtype=float) + inflation, 0.0, 1.0)
    lower = np.clip(np.asarray(estimates, dtype=float) - inflation, 0.0, 1.0)
    low = upper <= epsilon
    high = lower > epsilon
    uncertain = ~(low | high)
    return {
        "low_risk": int(low.sum()),
        "uncertain": int(uncertain.sum()),
        "high_risk": int(high.sum()),
        "share_low_risk": float(low.mean()) if low.size else float("nan"),
        "share_uncertain": float(uncertain.mean()) if uncertain.size else float("nan"),
        "share_high_risk": float(high.mean()) if high.size else float("nan"),
        "inflation": float(inflation),
    }


# ------------------------------------------------------------------ fitting the whole family


def fit_family(state: FoldState, inflation: dict[str, Any]) -> dict[str, Fitted]:
    """Every deterministic map for one fold, fitted in one place so the inputs cannot diverge.

    The label-free arms see `SourceView` and `TargetView` and nothing else. The two oracles are
    constructed here as well, from labels this stage is explicitly allowed to read for an upper
    bound, and they are kept in the same dictionary so that a phase which forgets to exclude
    them fails a test rather than publishing a number.
    """
    source, target = _source_view(state), _target_view(state)
    grid = _score_grid(state)
    family: dict[str, Fitted] = {
        IDENTITY: fit_identity(source, target, grid),
        SOURCE_ISOTONIC: fit_source_isotonic(source, target, grid),
        TEMPERATURE: fit_temperature(source, target, grid),
        PRIOR_SHIFT: fit_prior_shift(source, target, grid),
        RANK_TRANSPORT: fit_rank_transport(source, target, grid),
        QUANTILE_TRANSPORT: fit_quantile_transport(source, target, grid, state.strata.size),
        NO_STRATA: fit_no_strata(source, target, grid),
    }
    family[TRANSPORT_PRIOR] = fit_transport_prior(
        family[QUANTILE_TRANSPORT], target, family[PRIOR_SHIFT].diagnostics, grid
    )
    family[UNPROJECTED] = fit_quantile_transport(
        source, target, grid, state.strata.size, project=False
    )
    family[CONFORMAL] = fit_conformal(family[QUANTILE_TRANSPORT], inflation)
    family[SCRAMBLED] = fit_scrambled(
        family[QUANTILE_TRANSPORT], _stable_seed(state.held_out, SCRAMBLED)
    )
    family[ORACLE_POOL] = fit_oracle(ORACLE_POOL, state.pool.p_harm, state.pool.harmful, grid)
    family[ORACLE_EVAL] = fit_oracle(
        ORACLE_EVAL, state.evaluation.p_harm, state.evaluation.harmful, grid
    )

    for name, fitted in family.items():
        if not fitted.monotone_required:
            continue
        violation = _monotone_violation(fitted.grid, fitted.values)
        if violation > 1e-9:
            raise PhaseError(
                f"{state.held_out}/{name}: the map falls by {violation:g} against its own score "
                "ordering; the brief requires T(p_i) <= T(p_j) whenever p_i < p_j and every arm "
                "except the two named ablations is constructed to satisfy it"
            )
    return family


# ------------------------------------------------------------------ ranking preservation


def _kendall_tau(left: np.ndarray, right: np.ndarray, cap: int = 2000) -> float:
    """Rank agreement between two score vectors, sub-sampled at a fixed cap by an even stride.

    The exact statistic is quadratic in the worst case and these blocks run past four thousand
    rows. A stride needs no seed, so the number cannot move with a random draw, and the blocks
    are in candidate order rather than score order, so a stride is not a selection on the
    quantity being measured. The construction is SGV8's, for the same reason.
    """
    from scipy.stats import kendalltau

    size = int(np.asarray(left).size)
    if size < 3:
        return float("nan")
    if size > cap:
        index = np.linspace(0, size - 1, cap).astype(int)
        left, right = np.asarray(left)[index], np.asarray(right)[index]
    value = kendalltau(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def ranking_report(estimates: np.ndarray, block: Block) -> dict[str, float]:
    """Whether the map moved the ordering, and whether it moved the discrimination.

    Three numbers with three different jobs. Kendall tau and Spearman against the ORIGINAL score
    say whether the map preserved the ordering it was given -- a strictly increasing map scores
    1.0 on both, and anything below that is ties the map created by collapsing distinct scores
    onto one value. AUROC against the realised outcome says whether that collapse cost anything:
    a strictly increasing map cannot change it at all, so a drop here is exactly the information
    the ties destroyed.
    """
    original = block.p_harm
    harmful = block.harmful.astype(float)
    return {
        "kendall_tau_vs_original_score": _kendall_tau(original, estimates),
        "spearman_vs_original_score": _spearman(original, estimates),
        "auroc_vs_harmful": roc_auc(np.asarray(estimates, dtype=float), harmful),
        "auroc_of_the_original_score": roc_auc(np.asarray(original, dtype=float), harmful),
        "distinct_values": float(np.unique(np.asarray(estimates, dtype=float)).size),
        "distinct_values_of_the_original_score": float(np.unique(original).size),
    }


def calibration_metrics(estimates: np.ndarray, block: Block) -> dict[str, Any]:
    """The brief's Metric 1, from the one implementation of it in this repository.

    `ocr_risk.metrics.calibration.calibration_report` carries Brier, both ECE binning schemes,
    the maximum calibration error, the Murphy decomposition and the reliability bins with their
    counts. The project's rule is that an ECE without its scheme and bin count is not a
    reportable number, so all of them travel together and none is recomputed here.
    """
    report = calibration_report(
        np.asarray(estimates, dtype=float),
        block.harmful.astype(float),
        n_bins=N_BINS,
        binning="equal_mass",
    )
    payload = report.as_dict()
    payload["mean_predicted"] = float(np.mean(estimates)) if np.size(estimates) else float("nan")
    payload["mean_realized"] = float(block.harmful.mean()) if block.size else float("nan")
    payload["signed_level_error"] = payload["mean_predicted"] - payload["mean_realized"]
    return payload


def prefix_calibration(curves: PrefixCurves, epsilons: tuple[float, ...]) -> dict[str, Any]:
    """Calibration where the decision is actually made: on the prefixes a cut can select.

    A global ECE is dominated by the bulk of the score distribution, and a risk-bounded
    deployment never accepts the bulk -- it accepts a prefix of the ranking whose harm rate is
    near epsilon. So the decision-relevant question is not "is this probability right on
    average" but "is the MEAN of this probability right on the set the cut would choose", and
    that is what these numbers are. They are reported as a diagnostic of where the calibration
    error lives, not as a second endpoint.
    """
    out: dict[str, Any] = {}
    for epsilon in epsilons:
        key = f"epsilon_{int(epsilon * 100)}"
        index = _deepest(curves.realized, epsilon)
        if index < 0:
            out[key] = {"frontier_rank": 0, "signed_error_at_the_frontier": float("nan")}
            continue
        out[key] = {
            "frontier_rank": index + 1,
            "predicted_at_the_frontier": float(curves.predicted[index]),
            "realized_at_the_frontier": float(curves.realized[index]),
            "signed_error_at_the_frontier": float(curves.predicted[index] - curves.realized[index]),
            "repair_recall_at_the_frontier": float(curves.repair_recall[index]),
        }
    deployable = curves.realized <= 0.30
    if deployable.any():
        error = curves.predicted[deployable] - curves.realized[deployable]
        out["deployable_region"] = {
            "prefixes": int(deployable.sum()),
            "mean_absolute_error": float(np.abs(error).mean()),
            "mean_signed_error": float(error.mean()),
            "note": (
                "over every prefix whose realised harm rate is at most 0.30, which is the whole "
                "region any epsilon in this project's grid could select from."
            ),
        }
    else:
        out["deployable_region"] = {"prefixes": 0, "mean_absolute_error": float("nan")}
    return out


# ------------------------------------------------------------------ phase: --calibrate

REPORT_GRID = np.linspace(0.0, 1.0, 201)
"""The score levels every published map is tabulated at. Even rather than quantile-spaced so
the record of a map does not depend on any engine's own score distribution, and small enough to
live in a JSON artifact a reviewer can read."""

FEW_SHOT_CALIBRATION = OUT / "few_shot_calibration.parquet"
FEW_SHOT_DEPLOYMENT = OUT / "few_shot_deployment.parquet"


def _published_map(fitted: Fitted) -> dict[str, Any]:
    """A map as it goes into `calibration_mapping.json`: tabulated, not pickled.

    A fitted sklearn object is not a research artifact -- it cannot be read, diffed or checked
    by anyone without this exact library version. A monotone function on [0, 1] tabulated at 201
    points can be, and `--mapping` recomputes the monotonicity claim from this table rather than
    trusting the assertion the fitting code already made.
    """
    values = fitted.apply(REPORT_GRID, np.zeros(REPORT_GRID.size, dtype=int))
    return {
        "kind": fitted.kind,
        "parameters": {name: float(value) for name, value in fitted.parameters.items()},
        "monotone_required": bool(fitted.monotone_required),
        "grid": [round(float(x), 6) for x in REPORT_GRID],
        "values": [round(float(x), 8) for x in values],
        "monotone_violation_on_the_published_grid": _monotone_violation(REPORT_GRID, values),
        "diagnostics": fitted.diagnostics,
    }


def _block_frame(engine: str, block_name: str, block: Block, family: dict[str, Fitted]) -> Any:
    frame = pd.DataFrame(
        {
            "held_out_engine": engine,
            "block": block_name,
            "candidate_id": block.candidate_id,
            "document_id": block.documents,
            "stratum": block.stratum,
            "utility": block.utility,
            "p_harm": block.p_harm,
            "p_benefit": block.p_benefit,
            "is_harmful": block.harmful.astype(bool),
            "beneficial": block.beneficial.astype(bool),
        }
    )
    for name in FITTED_MAPS:
        frame[f"est__{name}"] = family[name].apply(block.p_harm, block.stratum)
    frame[f"est__{PERMUTED}"] = permuted_estimates(
        frame[f"est__{QUANTILE_TRANSPORT}"].to_numpy(dtype=float),
        _stable_seed(engine, PERMUTED, block_name),
    )
    return frame


def _few_shot_cells(
    state: FoldState, family: dict[str, Fitted], inflation: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Method E over the brief's budget grid, on SGV6's pool and SGV6's random acquisition.

    The bound cut for a few-shot arm reuses METHOD C's transfer inflation. Refitting the
    inflation per budget would mean replaying the inner folds three thousand times, and the
    approximation runs in the safe direction: a few-shot map that is better calibrated than
    Method C is inflated by more than it needs and the arm becomes conservative, never optimistic.
    """
    source = _source_view(state)
    zero_label = family[PRIOR_SHIFT].diagnostics
    calibration_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    for size in BUDGETS:
        seeds = (0,) if size == 0 else tuple(range(SEED_REPEATS))
        for seed in seeds:
            budget = draw_budget(state.pool, state.held_out, size, seed)
            direct = fit_few_shot_direct(budget, REPORT_GRID)
            arms = {
                FEW_SHOT_DIRECT: direct,
                FEW_SHOT_SHRUNK: fit_few_shot_shrunk(
                    family[QUANTILE_TRANSPORT], direct, budget, REPORT_GRID
                ),
                FEW_SHOT_PRIOR: fit_few_shot_prior(source, zero_label, budget, REPORT_GRID),
                FEW_SHOT_COMPOSED: fit_few_shot_shrunk(
                    family[TRANSPORT_PRIOR], direct, budget, REPORT_GRID, FEW_SHOT_COMPOSED
                ),
            }
            for name, fitted in arms.items():
                pool_estimates = fitted.apply(state.pool.p_harm, state.pool.stratum)
                eval_estimates = fitted.apply(state.evaluation.p_harm, state.evaluation.stratum)
                report = calibration_metrics(eval_estimates, state.evaluation)
                ranking = ranking_report(eval_estimates, state.evaluation)
                calibration_rows.append(
                    {
                        "held_out_engine": state.held_out,
                        "arm": name,
                        "budget": size,
                        "seed": seed,
                        "n_labels": budget.size,
                        "n_documents": int(np.unique(budget.documents).size) if budget.size else 0,
                        "harmful_labels": float(budget.harmful.sum()),
                        "brier": report["brier"],
                        "ece_equal_mass": report["ece_equal_mass"],
                        "ece_equal_width": report["ece_equal_width"],
                        "max_calibration_error": report["max_calibration_error"],
                        "calibration_term": report["calibration_term"],
                        "mean_predicted": report["mean_predicted"],
                        "signed_level_error": report["signed_level_error"],
                        "auroc_vs_harmful": ranking["auroc_vs_harmful"],
                        "kendall_tau_vs_original_score": ranking["kendall_tau_vs_original_score"],
                    }
                )
                for epsilon in EPSILONS:
                    key = f"epsilon_{int(epsilon * 100)}"
                    lift = float(inflation["per_epsilon"][key]["inflation"])
                    for rule in (POINT_CUT, BOUND_CUT):
                        placed = place_cut(
                            state.pool,
                            pool_estimates,
                            epsilon,
                            rule,
                            lift,
                            state,
                        )
                        outcome = _deployed_at(state.evaluation, placed["tau"], epsilon)
                        deployment_rows.append(
                            {
                                "held_out_engine": state.held_out,
                                "arm": name,
                                "budget": size,
                                "seed": seed,
                                "epsilon": epsilon,
                                "rule": rule,
                                "tau": placed["tau"],
                                "cut_feasible": bool(placed["feasible"]),
                                "pool_rank": int(placed["pool_rank"]),
                                "coverage": outcome["coverage"],
                                "n_accepted": outcome["n_accepted"],
                                "realized_harm_rate": outcome["realized_harm_rate"],
                                "repair_recall": outcome["repair_recall"],
                                "holds_bound": bool(outcome["holds_bound"]),
                                "bound_violation": outcome["bound_violation"],
                                "accepts_nothing": bool(outcome["accepts_nothing"]),
                            }
                        )
    return calibration_rows, deployment_rows


def run_calibrate() -> int:
    """Rebuild the frozen model per fold, fit every map, and write the per-row tables.

    This is the only phase that touches the model. Everything downstream reads
    `calibrated_scores.parquet` and the two few-shot tables, which is what makes a re-analysis
    cheap and what makes "every number traces to an artifact" checkable rather than aspirational.
    """
    started = time.monotonic()
    base, signatures, retrieval_columns = s6.load_base()
    selection = s6.frozen_selection()
    frozen = s6.load_frozen_scores()
    engines = sorted(base.engines)

    frames: list[Any] = []
    calibration_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    folds: dict[str, Any] = {}
    mapping: dict[str, Any] = {}

    for engine in engines:
        fold = build_fold(base, engine)
        inflation = transfer_inflation(
            base, engine, selection[engine], signatures, retrieval_columns
        )
        state = build_state(
            base, fold, selection[engine], signatures, retrieval_columns, engine, frozen
        )
        family = fit_family(state, inflation)
        for block_name, block in (
            ("calibration", state.calibration),
            ("pool", state.pool),
            ("evaluation", state.evaluation),
        ):
            frames.append(_block_frame(engine, block_name, block, family))
        cells, deployments = _few_shot_cells(state, family, inflation)
        calibration_rows.extend(cells)
        deployment_rows.extend(deployments)
        mapping[engine] = {
            "selected_by_sgv5": dict(state.selected),
            "strata": {
                "difficulty_cuts": list(state.strata.cuts),
                "cells": state.strata.size,
                "names": list(state.strata.names),
                "fitted_on": state.strata.fitted_on,
            },
            "transfer_inflation": inflation,
            "maps": {name: _published_map(family[name]) for name in FITTED_MAPS},
            "risk_sets": {
                f"epsilon_{int(epsilon * 100)}": risk_sets(
                    family[QUANTILE_TRANSPORT].apply(state.pool.p_harm, state.pool.stratum),
                    epsilon,
                    float(inflation["per_epsilon"][f"epsilon_{int(epsilon * 100)}"]["inflation"]),
                )
                for epsilon in EPSILONS
            },
        }
        folds[engine] = {
            "identity_against_published_sgv5": state.identity,
            "source_thresholds": state.source_thresholds,
            "source_threshold_feasible": state.source_threshold_feasible,
            "diagnostics": state.diagnostics,
        }
        print(
            f"  {engine}: {state.evaluation.size} evaluation rows, {state.pool.size} pool rows, "
            f"inflation {inflation['per_epsilon'][PRIMARY_KEY]['inflation']:+.4f} at "
            f"epsilon={PRIMARY_EPSILON}"
        )

    pilot._write_parquet_once(SCORES, pd.concat(frames, ignore_index=True))
    pilot._write_parquet_once(FEW_SHOT_CALIBRATION, pd.DataFrame(calibration_rows))
    pilot._write_parquet_once(FEW_SHOT_DEPLOYMENT, pd.DataFrame(deployment_rows))
    cc._write_json_once(
        DESIGN_RECORD,
        {
            "schema_version": "sgv9-design-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "stage": "SGV9 -- target-free risk calibration",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "pre_registration": {
                "claim": (
                    "SGV9-T1: the absolute harm probability of the frozen SGV5 candidate-risk "
                    "model can be recovered on an unseen OCR engine from that engine's "
                    "UNLABELLED candidate scores, closely enough to place a risk-bounded "
                    "deployment cut better than transferring either the source threshold or the "
                    "source probabilities, and without disturbing the ranking that transfers."
                ),
                "sub_claims": {
                    "T1a": "a target-free map improves absolute harm estimation while "
                    "preserving ranking (the brief's H1)",
                    "T1b": "a few target labels interpolate smoothly between the target-free "
                    "map and supervised recalibration (H2)",
                    "T1c": "better calibration means fewer unsafe deployments at a fixed "
                    "bound (H3)",
                },
                "primary_map": PRIMARY_MAP,
                "primary_rule": PRIMARY_RULE,
                "primary_few_shot_arm": PRIMARY_FEW_SHOT,
                "target_free_family": list(TARGET_FREE),
                "controls": list(CONTROLS),
                "oracles": list(ORACLES),
                "few_shot_arms": list(FEW_SHOT),
                "cut_rules": list(RULES),
                "epsilons": list(EPSILONS),
                "budgets": list(BUDGETS),
                "seed_repeats": SEED_REPEATS,
                "arm_added_mid_stage": {
                    "arm": TRANSPORT_PRIOR,
                    "few_shot_arm": FEW_SHOT_COMPOSED,
                    "what_had_been_seen": (
                        "the fitted maps' own parameters and their mean over each fold's "
                        "unlabelled pool, plus the pool's harm prevalence. No calibration error, "
                        "no endpoint and no deployed point had been computed for any arm."
                    ),
                    "why": (
                        "Methods B and C carry the source's marginal onto the target unchanged "
                        "and Method A2 moves the level without deciding which rows move; the "
                        "composition is what the two mechanisms jointly permit, and without it "
                        "the stage would test two half-methods and could not answer whether "
                        "calibration is the missing component."
                    ),
                    "status": "not the primary; reported as a composition of two pre-specified "
                    "arms and excluded from nothing else",
                },
                "prediction_made_before_any_result_was_read": (
                    "The four folds exhibit two different failures and the same map cannot "
                    "repair both. Three engines are over-confident in the low-risk tail and one "
                    "is under-confident, so a map that only rescales dispersion (Method A) or "
                    "only carries ranks (Method B) should help at most half of them; a map that "
                    "can move the level (Method A2, and Method C through its strata) should help "
                    "the under-confident engine most. EasyOCR's achievable frontier at "
                    "epsilon = 0.05 is itself near zero, which is a failure of the ORDERING, and "
                    "no monotone map can repair a failure of the ordering. The expected outcome "
                    "is therefore a safety gain that is broad and a recall gain that is not, and "
                    "the criteria are written so that outcome reads as NOT SUPPORTED."
                ),
            },
            "leakage_boundary": {
                "map_fitted_on": "source engines' CALIBRATION rows, and the held-out engine's "
                "TRAIN scores; no held-out-engine label of any kind",
                "threshold_placed_on": "the held-out engine's TRAIN rows, by predicted harm; "
                "the evaluation block is never read before the threshold is fixed",
                "evaluation_block": "held-out engine, DEVELOPMENT documents; read once, to be "
                "measured",
                "oracles": "upper bounds only; excluded from every criterion and from every "
                "selection",
                "few_shot_labels": "purchased from the adaptation pool by SGV6's random "
                "acquisition under SGV6's seed formula",
            },
            "map_notes": MAP_NOTES,
            "rule_notes": RULE_NOTES,
            "folds": folds,
            "rows": {
                "calibrated_scores": int(sum(len(frame) for frame in frames)),
                "few_shot_calibration": len(calibration_rows),
                "few_shot_deployment": len(deployment_rows),
            },
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    cc._write_json_once(
        CALIBRATION_MAPPING,
        {
            "schema_version": "sgv9-mapping-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "report_grid": "201 evenly spaced probabilities on [0, 1]",
            "folds": mapping,
        },
    )
    print(
        f"calibrate: {len(engines)} folds, {sum(len(f) for f in frames)} rows -> "
        f"{cc._relative(SCORES)} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ reading the artifacts back


@dataclass(slots=True)
class LoadedFold:
    """One fold read back from the write-once score table, in the shape the phases want."""

    held_out: str
    calibration: Block
    pool: Block
    evaluation: Block
    estimates: dict[str, dict[str, np.ndarray]]
    source_thresholds: dict[str, float]
    source_threshold_feasible: dict[str, bool]
    inflation: dict[str, float]
    diagnostics: dict[str, Any]


def _block_from_frame(frame: Any) -> Block:
    return Block(
        index=np.arange(len(frame), dtype=int),
        candidate_id=frame["candidate_id"].astype(str).to_numpy(),
        documents=frame["document_id"].astype(str).to_numpy(),
        p_harm=frame["p_harm"].to_numpy(dtype=float),
        p_benefit=frame["p_benefit"].to_numpy(dtype=float),
        utility=frame["utility"].to_numpy(dtype=float),
        stratum=frame["stratum"].to_numpy(dtype=int),
        harmful=frame["is_harmful"].to_numpy(dtype=bool),
        beneficial=frame["beneficial"].to_numpy(dtype=bool),
    )


def load_folds() -> tuple[dict[str, LoadedFold], dict[str, Any]]:
    """Read the score table and the design record, and check they describe the same run.

    `.claude/rules/` requires evaluation to read immutable artifacts rather than recompute
    upstream, so every phase after `--calibrate` comes through here. The row-count check is what
    turns "these files are in the same directory" into "these files are from the same run".
    """
    if not SCORES.is_file():
        raise PhaseError(f"{cc._relative(SCORES)} is missing; run --calibrate first")
    record = cc._read_json(DESIGN_RECORD)
    frame = pd.read_parquet(SCORES)
    if len(frame) != int(record["rows"]["calibrated_scores"]):
        raise PhaseError(
            f"{cc._relative(SCORES)} has {len(frame)} rows and the design record describes "
            f"{record['rows']['calibrated_scores']}; the two are not from the same run"
        )
    folds: dict[str, LoadedFold] = {}
    for engine, group in frame.groupby("held_out_engine", sort=True):
        blocks: dict[str, Any] = {
            name: part.reset_index(drop=True) for name, part in group.groupby("block", sort=True)
        }
        missing = {"calibration", "pool", "evaluation"} - set(blocks)
        if missing:
            raise PhaseError(f"{engine}: the score table is missing the {sorted(missing)} block")
        estimates = {
            arm: {name: part[f"est__{arm}"].to_numpy(dtype=float) for name, part in blocks.items()}
            for arm in ALL_ARMS
        }
        fold_record = record["folds"][str(engine)]
        folds[str(engine)] = LoadedFold(
            held_out=str(engine),
            calibration=_block_from_frame(blocks["calibration"]),
            pool=_block_from_frame(blocks["pool"]),
            evaluation=_block_from_frame(blocks["evaluation"]),
            estimates=estimates,
            source_thresholds={
                key: float(value) for key, value in fold_record["source_thresholds"].items()
            },
            source_threshold_feasible={
                key: bool(value) for key, value in fold_record["source_threshold_feasible"].items()
            },
            inflation={
                key: float(cell["inflation"])
                for key, cell in cc._read_json(CALIBRATION_MAPPING)["folds"][str(engine)][
                    "transfer_inflation"
                ]["per_epsilon"].items()
            },
            diagnostics=fold_record["diagnostics"],
        )
    return folds, record


def _as_state(fold: LoadedFold) -> FoldState:
    """A `FoldState` carrying only what a cut rule reads, rebuilt from the artifacts.

    The model, the strata object and the fitted maps are deliberately absent: after
    `--calibrate` nothing may refit anything, and a phase that tried would find no model to
    refit rather than quietly building a second one.
    """
    return FoldState(
        held_out=fold.held_out,
        selected={},
        lambda_=float("nan"),
        calibration=fold.calibration,
        pool=fold.pool,
        evaluation=fold.evaluation,
        strata=Strata(cuts=(float("nan"), float("nan")), names=(), fitted_on="read back"),
        source_thresholds=fold.source_thresholds,
        source_threshold_feasible=fold.source_threshold_feasible,
        identity={},
        diagnostics=fold.diagnostics,
    )


# ------------------------------------------------------------------ the resampling engine


def document_resamples(documents: np.ndarray, seed: int, n_resamples: int) -> list[np.ndarray]:
    """Document-clustered resample index arrays, built exactly as the library builds them.

    `cluster_bootstrap_indices` draws whole clusters with replacement from a SORTED list of
    cluster identifiers using `rng.integers(0, n_clusters, size=n_clusters)`. This reproduces
    that construction so several statistics can share one set of draws instead of each paying
    for its own; `tests/leakage` asserts a confidence interval computed from these draws matches
    the library's for the same seed, so the duplication is checked rather than trusted.
    """
    by_cluster: dict[str, list[int]] = {}
    for index, name in enumerate(documents.tolist()):
        by_cluster.setdefault(str(name), []).append(index)
    cluster_ids = sorted(by_cluster)
    members = [np.asarray(by_cluster[name], dtype=np.intp) for name in cluster_ids]
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        draws.append(np.concatenate([members[position] for position in chosen]))
    return draws


@dataclass(slots=True)
class ResampleCurves:
    """Cumulative accepted / harmful / beneficial counts by rank, one row per resample.

    Built once per fold and shared by every arm, every epsilon and every cut rule. A threshold
    is a rank in the fold's own descending decision-score order, so a cell's whole resample
    distribution is one column of these matrices -- which is what makes 672 cells of 2,000
    document-clustered resamples affordable. The device is SGV7's `RankCurves`.
    """

    accepted: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    total_beneficial: np.ndarray
    order: np.ndarray
    utility_sorted: np.ndarray

    @property
    def n_resamples(self) -> int:
        return int(self.accepted.shape[0])


def resample_curves(block: Block, seed: int, n_resamples: int) -> ResampleCurves:
    order = np.argsort(-block.utility, kind="stable")
    rank = np.empty(block.size, dtype=np.intp)
    rank[order] = np.arange(block.size, dtype=np.intp)
    harmful = block.harmful.astype(np.int64)
    beneficial = block.beneficial.astype(np.int64)

    draws = document_resamples(block.documents, seed, n_resamples)
    accepted = np.zeros((n_resamples, block.size), dtype=np.int32)
    harm = np.zeros((n_resamples, block.size), dtype=np.int32)
    benefit = np.zeros((n_resamples, block.size), dtype=np.int32)
    total = np.zeros(n_resamples, dtype=np.int32)
    for index, draw in enumerate(draws):
        ranks = rank[draw]
        accepted[index] = np.cumsum(np.bincount(ranks, minlength=block.size))
        harm[index] = np.cumsum(np.bincount(ranks, weights=harmful[draw], minlength=block.size))
        benefit[index] = np.cumsum(
            np.bincount(ranks, weights=beneficial[draw], minlength=block.size)
        )
        total[index] = int(beneficial[draw].sum())
    return ResampleCurves(
        accepted=accepted,
        harmful=harm,
        beneficial=benefit,
        total_beneficial=total,
        order=order,
        utility_sorted=block.utility[order],
    )


def _rank_of(curves: ResampleCurves, tau: float) -> int:
    """How many of the block's rows a threshold accepts, in the block's own score order."""
    if not np.isfinite(tau):
        return 0
    return int(np.count_nonzero(curves.utility_sorted >= tau))


def resampled_deployment(curves: ResampleCurves, tau: float, epsilon: float) -> dict[str, Any]:
    """The deployed operating point's resample distribution at a FIXED threshold.

    The threshold is not re-placed inside the resample: it was chosen on the adaptation pool and
    an operator would carry that one number to the next batch of pages. What varies here is the
    batch, which is exactly the question a violation rate has to answer -- if this threshold met
    a different sample of this engine's pages, how often would the bound break?
    """
    rank = _rank_of(curves, tau)
    if rank == 0:
        return {
            "n_accepted_mean": 0.0,
            "realized_harm_mean": float("nan"),
            "realized_harm_ci": [float("nan"), float("nan")],
            "repair_recall_mean": 0.0,
            "repair_recall_ci": [0.0, 0.0],
            "violation_rate": 0.0,
            "usable_resamples": 0,
            "accepts_nothing": True,
        }
    column = rank - 1
    accepted = curves.accepted[:, column].astype(float)
    harmful = curves.harmful[:, column].astype(float)
    beneficial = curves.beneficial[:, column].astype(float)
    total = curves.total_beneficial.astype(float)
    usable = accepted > 0
    harm_rate = np.full(curves.n_resamples, np.nan)
    harm_rate[usable] = harmful[usable] / accepted[usable]
    recall = np.divide(beneficial, np.maximum(total, 1.0))
    finite = np.isfinite(harm_rate)
    return {
        "n_accepted_mean": float(accepted.mean()),
        "realized_harm_mean": float(np.nanmean(harm_rate)) if finite.any() else float("nan"),
        "realized_harm_ci": [
            float(np.nanquantile(harm_rate, 0.025)) if finite.any() else float("nan"),
            float(np.nanquantile(harm_rate, 0.975)) if finite.any() else float("nan"),
        ],
        "repair_recall_mean": float(recall.mean()),
        "repair_recall_ci": [
            float(np.quantile(recall, 0.025)),
            float(np.quantile(recall, 0.975)),
        ],
        "violation_rate": float(np.mean(harm_rate[finite] > epsilon)) if finite.any() else 0.0,
        "usable_resamples": int(finite.sum()),
        "accepts_nothing": False,
        "recall_draws": recall,
    }


def _interval(draws: np.ndarray) -> dict[str, float]:
    """A percentile interval over paired resample draws, plus the two-sided bootstrap p."""
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return {
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "draws": 0,
        }
    below = float(np.mean(finite <= 0.0))
    above = float(np.mean(finite >= 0.0))
    return {
        "estimate": float(finite.mean()),
        "ci_lower": float(np.quantile(finite, 0.025)),
        "ci_upper": float(np.quantile(finite, 0.975)),
        "p_value": float(min(1.0, 2.0 * min(below, above))),
        "draws": int(finite.size),
    }


# ------------------------------------------------------------------ phase: --mapping


def run_mapping() -> int:
    """Re-verify every published map from its published table, not from the code that fitted it.

    `--calibrate` already asserted monotonicity on the dense grid of score levels each map will
    meet. This phase asks a different question: does the map as PUBLISHED -- the 201-point table
    a reviewer can read -- satisfy the constraint the stage claims for it? A map that passed the
    first check and fails this one would mean the record does not describe the method.
    """
    started = time.monotonic()
    mapping = cc._read_json(CALIBRATION_MAPPING)
    verified: dict[str, Any] = {}
    failures: list[str] = []
    for engine, cell in sorted(mapping["folds"].items()):
        per_map: dict[str, Any] = {}
        for name, published in sorted(cell["maps"].items()):
            grid = np.asarray(published["grid"], dtype=float)
            values = np.asarray(published["values"], dtype=float)
            violation = _monotone_violation(grid, values)
            required = bool(published["monotone_required"])
            if required and violation > 1e-6:
                failures.append(f"{engine}/{name}: {violation:g}")
            per_map[name] = {
                "monotone_required": required,
                "monotone_violation": violation,
                "satisfies_the_ordering_constraint": bool(not required or violation <= 1e-6),
                "range": [float(values.min()), float(values.max())],
                "distinct_values": int(np.unique(values).size),
                "mean_over_the_published_grid": float(values.mean()),
                "identity_distance": float(np.abs(values - grid).mean()),
            }
        verified[engine] = per_map
    if failures:
        raise PhaseError(
            "the published table of a map that claims to be monotone is not: " + "; ".join(failures)
        )
    cc._write_json_once(
        OUT / "mapping_verification.json",
        {
            "schema_version": "sgv9-mapping-verification-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "checked": (
                "every map's PUBLISHED 201-point table, independently of the fitting code's own "
                "assertion on the dense grid"
            ),
            "arms_required_to_be_monotone": [
                name for name in FITTED_MAPS if name not in (UNPROJECTED, SCRAMBLED)
            ],
            "arms_deliberately_not_monotone": [UNPROJECTED, SCRAMBLED],
            "folds": verified,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"mapping: {len(verified)} folds verified, 0 ordering violations")
    return 0


# ------------------------------------------------------------------ phase: --results


CALIBRATION_FAMILY = "4 engines x the target-free family, on Brier"


def _calibration_delta(
    fold: LoadedFold, arm: str, reference: str, statistic: str
) -> dict[str, float]:
    """Paired document-clustered bootstrap of a calibration statistic against a reference arm.

    One index draw is applied to both arms, so the interval is on the DIFFERENCE and the pairing
    -- the two arms score the same pages through the same engine -- is not thrown away. Brier is
    the primary because it is a strictly proper scoring rule and needs no binning; the ECE delta
    is reported beside it under both schemes, with the bin count recorded, because a single ECE
    without its scheme is not a reportable number in this project.
    """
    from ocr_risk.metrics.calibration import brier_score, expected_calibration_error

    block = fold.evaluation
    outcomes = block.harmful.astype(float)
    mine = fold.estimates[arm]["evaluation"]
    theirs = fold.estimates[reference]["evaluation"]

    if statistic == "brier":

        def score(index: np.ndarray) -> float:
            return brier_score(mine[index], outcomes[index]) - brier_score(
                theirs[index], outcomes[index]
            )

    elif statistic in ("ece_equal_mass", "ece_equal_width"):
        binning = "equal_mass" if statistic.endswith("mass") else "equal_width"

        def score(index: np.ndarray) -> float:
            return (
                expected_calibration_error(mine[index], outcomes[index], N_BINS, binning)[0]
                - expected_calibration_error(theirs[index], outcomes[index], N_BINS, binning)[0]
            )

    else:  # pragma: no cover - the statistic table is closed
        raise PhaseError(f"unknown calibration statistic {statistic!r}")

    from ocr_risk.stats.bootstrap import cluster_bootstrap_indices

    result = cluster_bootstrap_indices(
        list(block.documents),
        score,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        bounds=None,
    )
    return {
        "estimate": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "standard_error": result.standard_error,
        "p_value": result.p_value_two_sided,
        "n_documents": result.n_clusters,
        "improves": bool(np.isfinite(result.upper) and result.upper < 0.0),
    }


def run_results() -> int:
    """Metric 1 and Metric 4: calibration error, ranking preservation, and the paired intervals."""
    started = time.monotonic()
    folds, _ = load_folds()
    from ocr_risk.stats.multiplicity import holm_bonferroni

    per_fold: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for engine, fold in sorted(folds.items()):
        arms: dict[str, Any] = {}
        for arm in ALL_ARMS:
            estimates = fold.estimates[arm]["evaluation"]
            arms[arm] = {
                "calibration": calibration_metrics(estimates, fold.evaluation),
                "ranking": ranking_report(estimates, fold.evaluation),
                "prefix": prefix_calibration(prefix_curves(fold.evaluation, estimates), EPSILONS),
            }
            if arm != IDENTITY:
                arms[arm]["vs_identity"] = {
                    statistic: _calibration_delta(fold, arm, IDENTITY, statistic)
                    for statistic in ("brier", "ece_equal_mass", "ece_equal_width")
                }
                if arm in TARGET_FREE:
                    p_values[f"{engine}|{arm}"] = float(
                        arms[arm]["vs_identity"]["brier"]["p_value"]
                    )
        per_fold[engine] = {
            "arms": arms,
            "rows": fold.evaluation.size,
            "documents": len(set(fold.evaluation.documents.tolist())),
            "harm_prevalence": float(fold.evaluation.harmful.mean()),
            "diagnostics": fold.diagnostics,
        }

    usable = {key: value for key, value in p_values.items() if np.isfinite(value)}
    adjusted = holm_bonferroni(usable) if usable else []
    cc._write_json_once(
        CALIBRATION_RESULTS,
        {
            "schema_version": "sgv9-calibration-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "primary_statistic": "brier",
            "primary_statistic_note": (
                "Brier is primary because it is a strictly proper scoring rule that needs no "
                "binning and decomposes into a calibration and a refinement term. ECE is "
                "reported under both binning schemes with n_bins recorded; a single ECE without "
                "its scheme is not a reportable number in this project."
            ),
            "n_bins": N_BINS,
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "unit": "document",
                "paired": True,
                "seed": BOOTSTRAP_SEED,
            },
            "multiplicity": {
                "family": CALIBRATION_FAMILY,
                "method": "holm",
                "size": len(usable),
                "tests": [
                    {
                        "cell": test.label,
                        "p_value": test.p_value,
                        "adjusted_p_value": test.adjusted_p_value,
                        "significant": test.significant,
                    }
                    for test in adjusted
                ],
                "survivors": [test.label for test in adjusted if test.significant],
            },
            "folds": per_fold,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    surviving = [test.label for test in adjusted if test.significant]
    print(f"results: {len(usable)} tests, {len(surviving)} surviving Holm")
    return 0


# ------------------------------------------------------------------ phase: --reliability


def run_reliability() -> int:
    """The reliability diagrams, and the prefix-calibration curves the decision actually reads.

    Two views of the same failure. The reliability diagram is the standard one: bin the estimate,
    compare with the realised rate. The prefix curve is the one this stage needs: a risk-bounded
    deployment never accepts a bin, it accepts a PREFIX of the ranking, and what decides whether
    the bound holds is whether the mean estimate over that prefix is right. A map can flatten the
    diagram and still misplace every cut, and a map can leave the diagram alone and fix them.
    """
    started = time.monotonic()
    folds, _ = load_folds()
    per_fold: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        curves_by_arm: dict[str, Any] = {}
        for arm in ALL_ARMS:
            estimates = fold.estimates[arm]["evaluation"]
            report = calibration_metrics(estimates, fold.evaluation)
            curves = prefix_curves(fold.evaluation, estimates)
            marks = [10, 25, 50, 100, 200, 400, 800, 1600]
            curves_by_arm[arm] = {
                "reliability_equal_mass": report["reliability"],
                "brier": report["brier"],
                "calibration_term": report["calibration_term"],
                "refinement_term": report["refinement_term"],
                "base_rate": report["base_rate"],
                "mean_predicted": report["mean_predicted"],
                "mean_realized": report["mean_realized"],
                "prefix": {
                    str(k): {
                        "predicted": float(curves.predicted[k - 1]),
                        "realized": float(curves.realized[k - 1]),
                        "ratio": float(curves.realized[k - 1] / max(curves.predicted[k - 1], 1e-9)),
                        "repair_recall": float(curves.repair_recall[k - 1]),
                        "effective_documents": float(curves.effective_n[k - 1]),
                    }
                    for k in marks
                    if k <= fold.evaluation.size
                },
                "prefix_calibration": prefix_calibration(curves, EPSILONS),
            }
        per_fold[engine] = {
            "arms": curves_by_arm,
            "rows": fold.evaluation.size,
            "note": (
                "the ratio column is realised harm divided by predicted harm over the same "
                "prefix. It is the multiplicative form of the same error the additive transfer "
                "inflation measures, and it is the more interpretable of the two at small "
                "epsilon, where an additive error of 0.15 and a bound of 0.05 are not "
                "commensurable quantities."
            ),
        }
    cc._write_json_once(
        RELIABILITY_METRICS,
        {
            "schema_version": "sgv9-reliability-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "n_bins": N_BINS,
            "binning": "equal_mass",
            "folds": per_fold,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"reliability: {len(per_fold)} folds x {len(ALL_ARMS)} arms")
    return 0


# ------------------------------------------------------------------ phase: --coverage


def _max_feasible_inflation(curves: PrefixCurves, epsilon: float) -> float:
    """The largest transfer inflation at which the bound cut would still accept something.

    Compared with the inflation actually estimated, this is the single number that says how far
    a certified deployment is from being possible on this engine at this bound. A positive gap
    means the bound is affordable; a negative one means the engine-to-engine calibration error
    is larger than the bound itself, and no amount of sharpening the point estimate would help.
    """
    from ocr_risk.stats.multiplicity import _normal_quantile

    z = _normal_quantile(1.0 - DELTA)
    point = np.clip(curves.predicted, 0.0, 1.0)
    sampling = z * np.sqrt(
        np.maximum(point * (1.0 - point), 0.0) / np.maximum(curves.effective_n, 1.0)
    )
    return float(epsilon - np.min(point + sampling))


def run_coverage() -> int:
    """Metrics 2 and 3: risk-controlled repair recall and harm-bound violations, every rule.

    The layout is deliberate. `source_threshold` and `quantile_matched_threshold` do not read a
    calibration map at all, so they appear once per fold rather than once per arm; every other
    cell is one (arm, epsilon, rule) triple. The identity arm under the predicted-harm rules is
    the control that separates the change of RULE from the change of CALIBRATION, and a gain
    that appears there is not a gain from calibration.
    """
    started = time.monotonic()
    folds, _ = load_folds()
    per_fold: dict[str, Any] = {}

    for engine, fold in sorted(folds.items()):
        state = _as_state(fold)
        curves_cache = resample_curves(
            fold.evaluation, _stable_seed(engine, "coverage"), BOOTSTRAP_RESAMPLES
        )
        frontier = achievable_repair_recall(
            fold.evaluation.utility, fold.evaluation.harmful, fold.evaluation.beneficial
        )
        rule_free: dict[str, Any] = {}
        for rule in (SOURCE_THRESHOLD, QUANTILE_THRESHOLD):
            placed = place_cut(
                fold.pool,
                fold.estimates[IDENTITY]["pool"],
                PRIMARY_EPSILON,
                rule,
                0.0,
                state,
            )
            per_epsilon = {}
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                outcome = _deployed_at(fold.evaluation, placed["tau"], epsilon)
                resampled = resampled_deployment(curves_cache, placed["tau"], epsilon)
                resampled.pop("recall_draws", None)
                per_epsilon[key] = {**outcome, "resampled": resampled}
            rule_free[rule] = {"placement": placed, "per_epsilon": per_epsilon}

        arms: dict[str, Any] = {}
        baseline_draws: dict[str, np.ndarray] = {}
        for arm in ALL_ARMS:
            pool_estimates = fold.estimates[arm]["pool"]
            pool_curves = prefix_curves(fold.pool, pool_estimates)
            per_rule: dict[str, Any] = {}
            for rule in (POINT_CUT, BOUND_CUT):
                per_epsilon = {}
                for epsilon in EPSILONS:
                    key = f"epsilon_{int(epsilon * 100)}"
                    lift = fold.inflation[key] if rule == BOUND_CUT else 0.0
                    placed = place_cut(
                        fold.pool,
                        pool_estimates,
                        epsilon,
                        rule,
                        lift,
                        state,
                    )
                    outcome = _deployed_at(fold.evaluation, placed["tau"], epsilon)
                    resampled = resampled_deployment(curves_cache, placed["tau"], epsilon)
                    draws = resampled.pop("recall_draws", None)
                    if arm == IDENTITY and draws is not None:
                        baseline_draws[f"{rule}|{key}"] = draws
                    per_epsilon[key] = {
                        **outcome,
                        "placement": placed,
                        "inflation": float(lift),
                        "max_inflation_that_still_deploys": _max_feasible_inflation(
                            pool_curves, epsilon
                        ),
                        "resampled": resampled,
                        "_draws": draws,
                    }
                per_rule[rule] = per_epsilon
            arms[arm] = per_rule

        for arm in ALL_ARMS:
            for rule in (POINT_CUT, BOUND_CUT):
                for key in (f"epsilon_{int(e * 100)}" for e in EPSILONS):
                    cell = arms[arm][rule][key]
                    draws = cell.pop("_draws")
                    reference = baseline_draws.get(f"{rule}|{key}")
                    cell["repair_recall_delta_vs_identity"] = (
                        _interval(np.asarray(draws) - np.asarray(reference))
                        if draws is not None and reference is not None
                        else {"estimate": float("nan")}
                    )

        per_fold[engine] = {
            "rules_without_a_map": rule_free,
            "arms": arms,
            "achievable_frontier": {
                f"epsilon_{int(e * 100)}": frontier[f"epsilon_{int(e * 100)}"] for e in EPSILONS
            },
            "transfer_inflation": fold.inflation,
            "rows": fold.evaluation.size,
            "documents": len(set(fold.evaluation.documents.tolist())),
            "beneficial_rows": int(fold.evaluation.beneficial.sum()),
        }

    cc._write_json_once(
        RISK_COVERAGE,
        {
            "schema_version": "sgv9-coverage-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "epsilons": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "primary_rule": PRIMARY_RULE,
            "rule_notes": RULE_NOTES,
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "unit": "document",
                "threshold_is_fixed_inside_the_resample": True,
                "note": (
                    "the threshold was chosen on the adaptation pool and is carried to the next "
                    "batch unchanged, so the resample varies the batch and not the decision. "
                    "That is the question a violation rate has to answer."
                ),
            },
            "folds": per_fold,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"coverage: {len(per_fold)} folds x {len(ALL_ARMS)} arms x {len(EPSILONS)} epsilons")
    return 0


# ------------------------------------------------------------------ phase: --efficiency


def _aggregate(frame: Any, columns: tuple[str, ...]) -> dict[str, Any]:
    """Mean and a percentile interval over the seeds of one cell, per column."""
    out: dict[str, Any] = {"seeds": len(frame)}
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[column] = {"mean": float("nan"), "ci": [float("nan"), float("nan")]}
            continue
        out[column] = {
            "mean": float(finite.mean()),
            "ci": [
                float(np.quantile(finite, 0.025)),
                float(np.quantile(finite, 0.975)),
            ],
            "n": int(finite.size),
        }
    return out


def run_efficiency() -> int:
    """Metric 5 and the brief's H2: what a target label buys, and whether the transition is smooth.

    Four few-shot arms over the brief's budget grid, fifty seeds at every non-zero budget and one
    at zero because every arm is deterministic there. The zero-label point of each arm is exactly
    the target-free map it is anchored at, so the curve starts where the target-free family ends
    and H2's "smooth transition" is a property to be measured rather than a shape to be drawn.
    """
    started = time.monotonic()
    if not FEW_SHOT_CALIBRATION.is_file():
        raise PhaseError("the few-shot tables are missing; run --calibrate first")
    calibration = pd.read_parquet(FEW_SHOT_CALIBRATION)
    deployment = pd.read_parquet(FEW_SHOT_DEPLOYMENT)
    folds, _ = load_folds()

    per_fold: dict[str, Any] = {}
    for engine in sorted(folds):
        arms: dict[str, Any] = {}
        for arm in FEW_SHOT:
            budgets: dict[str, Any] = {}
            for size in BUDGETS:
                cell = calibration[
                    (calibration["held_out_engine"] == engine)
                    & (calibration["arm"] == arm)
                    & (calibration["budget"] == size)
                ]
                if cell.empty:
                    continue
                entry = {
                    "calibration": _aggregate(
                        cell,
                        (
                            "brier",
                            "ece_equal_mass",
                            "ece_equal_width",
                            "max_calibration_error",
                            "mean_predicted",
                            "signed_level_error",
                            "auroc_vs_harmful",
                            "kendall_tau_vs_original_score",
                        ),
                    ),
                    "harmful_labels_bought": float(cell["harmful_labels"].mean()),
                    "documents_touched": float(cell["n_documents"].mean()),
                }
                for epsilon in EPSILONS:
                    key = f"epsilon_{int(epsilon * 100)}"
                    for rule in (POINT_CUT, BOUND_CUT):
                        block = deployment[
                            (deployment["held_out_engine"] == engine)
                            & (deployment["arm"] == arm)
                            & (deployment["budget"] == size)
                            & (deployment["epsilon"] == epsilon)
                            & (deployment["rule"] == rule)
                        ]
                        if block.empty:
                            continue
                        entry[f"{rule}|{key}"] = {
                            **_aggregate(
                                block,
                                ("repair_recall", "realized_harm_rate", "coverage", "tau"),
                            ),
                            "violation_rate": float(1.0 - block["holds_bound"].mean()),
                            "accepts_nothing_rate": float(block["accepts_nothing"].mean()),
                        }
                budgets[str(size)] = entry
            arms[arm] = budgets
        per_fold[engine] = {"arms": arms}

    # What one label is worth, as a slope on the endpoint between adjacent budgets.
    slopes: dict[str, Any] = {}
    for engine, block in per_fold.items():
        per_arm: dict[str, Any] = {}
        for arm, budgets in block["arms"].items():
            key = f"{POINT_CUT}|{PRIMARY_KEY}"
            points = [
                (int(size), budgets[size][key]["repair_recall"]["mean"])
                for size in sorted(budgets, key=int)
                if key in budgets[size]
            ]
            per_arm[arm] = {
                "budget_curve": [{"n_labels": n, "repair_recall": r} for n, r in points],
                "gain_per_label": [
                    {
                        "from": points[i][0],
                        "to": points[i + 1][0],
                        "per_label": (points[i + 1][1] - points[i][1])
                        / max(points[i + 1][0] - points[i][0], 1),
                    }
                    for i in range(len(points) - 1)
                ],
                "total_gain_from_zero_labels": (
                    points[-1][1] - points[0][1] if len(points) >= 2 else float("nan")
                ),
                "monotone_in_the_budget": bool(
                    all(points[i + 1][1] >= points[i][1] - 1e-12 for i in range(len(points) - 1))
                ),
            }
        slopes[engine] = per_arm

    cc._write_json_once(
        LABEL_EFFICIENCY,
        {
            "schema_version": "sgv9-efficiency-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "budgets": list(BUDGETS),
            "seed_repeats": SEED_REPEATS,
            "acquisition": "random, from the held-out engine's TRAIN rows, under SGV6's "
            "seed formula",
            "arms": {name: MAP_NOTES[name] for name in FEW_SHOT},
            "zero_label_anchor": {
                FEW_SHOT_DIRECT: IDENTITY,
                FEW_SHOT_SHRUNK: QUANTILE_TRANSPORT,
                FEW_SHOT_PRIOR: PRIOR_SHIFT,
                FEW_SHOT_COMPOSED: TRANSPORT_PRIOR,
            },
            "bound_cut_caveat": (
                "the bound cut for a few-shot arm reuses Method C's transfer inflation rather "
                "than refitting it per budget, which would mean replaying the inner folds for "
                "every budget and seed. The approximation runs in the safe direction only: a "
                "better-calibrated map is inflated by more than it needs and becomes "
                "conservative, never optimistic."
            ),
            "folds": per_fold,
            "label_value": slopes,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"efficiency: {len(per_fold)} folds x {len(FEW_SHOT)} arms x {len(BUDGETS)} budgets")
    return 0


# ------------------------------------------------------------------ phase: --oracle


def run_oracle() -> int:
    """The two ceilings, and how much of the gap to each a target-free map closes.

    Three references, each answering a different question. The ACHIEVABLE FRONTIER is what the
    ranking could deliver if the cut were placed with hindsight -- the ceiling of the ordering,
    which no calibration can pass. `oracle_pool` is what target labels could buy without touching
    the evaluation documents. `oracle_eval` recalibrates on the evaluation block itself and is
    the ceiling of the monotone map class outright; where it still breaks the bound, the failure
    is not one a better map could repair.
    """
    started = time.monotonic()
    coverage = cc._read_json(RISK_COVERAGE)
    calibration = cc._read_json(CALIBRATION_RESULTS)
    per_fold: dict[str, Any] = {}
    for engine, block in sorted(coverage["folds"].items()):
        frontier = block["achievable_frontier"]
        per_epsilon: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            ceiling = float(frontier[key]["repair_recall"])
            baseline = float(
                block["rules_without_a_map"][SOURCE_THRESHOLD]["per_epsilon"][key]["repair_recall"]
            )
            entry: dict[str, Any] = {
                "achievable_frontier_repair_recall": ceiling,
                "published_sgv5_deployed_repair_recall": baseline,
                "gap_to_close": ceiling - baseline,
                "arms": {},
            }
            for arm in ALL_ARMS:
                cell = block["arms"][arm][POINT_CUT][key]
                recall = float(cell["repair_recall"])
                entry["arms"][arm] = {
                    "repair_recall": recall,
                    "realized_harm_rate": cell["realized_harm_rate"],
                    "holds_bound": cell["holds_bound"],
                    "share_of_the_gap_closed": (
                        (recall - baseline) / (ceiling - baseline)
                        if abs(ceiling - baseline) > 1e-12
                        else float("nan")
                    ),
                }
            entry["oracle_gap"] = {
                arm: entry["arms"][ORACLE_EVAL]["repair_recall"]
                - entry["arms"][arm]["repair_recall"]
                for arm in TARGET_FREE
            }
            per_epsilon[key] = entry
        brier = calibration["folds"][engine]["arms"]
        per_fold[engine] = {
            "per_epsilon": per_epsilon,
            "brier": {arm: brier[arm]["calibration"]["brier"] for arm in ALL_ARMS},
            "brier_gap_to_the_evaluation_oracle": {
                arm: brier[arm]["calibration"]["brier"] - brier[ORACLE_EVAL]["calibration"]["brier"]
                for arm in TARGET_FREE
            },
            "brier_share_of_the_oracle_gap_closed": {
                arm: (
                    (brier[IDENTITY]["calibration"]["brier"] - brier[arm]["calibration"]["brier"])
                    / max(
                        brier[IDENTITY]["calibration"]["brier"]
                        - brier[ORACLE_EVAL]["calibration"]["brier"],
                        1e-12,
                    )
                )
                for arm in TARGET_FREE
            },
        }
    cc._write_json_once(
        ORACLE_GAP,
        {
            "schema_version": "sgv9-oracle-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "references": {
                "achievable_frontier": "the cut placed with hindsight on the evaluation rows. "
                "The ceiling of the ORDERING; a monotone recalibration cannot pass it.",
                ORACLE_POOL: MAP_NOTES[ORACLE_POOL],
                ORACLE_EVAL: MAP_NOTES[ORACLE_EVAL],
            },
            "folds": per_fold,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"oracle: {len(per_fold)} folds")
    return 0


# ------------------------------------------------------------------ phase: --ablation

ABLATIONS = {
    "A_identity_mapping": IDENTITY,
    "B_random_permutation": PERMUTED,
    "C_non_monotonic": UNPROJECTED,
    "C_scrambled": SCRAMBLED,
    "D_source_statistics_only": SOURCE_ISOTONIC,
    "E_no_structural_statistics": NO_STRATA,
    "F_oracle_calibration": ORACLE_EVAL,
}


def run_ablation() -> int:
    """The brief's six ablations, each as a difference against the primary map on the same rows.

    Two of them answer a question the code can settle rather than an experiment. Ablation D
    recalibrates on the source rows SGV5 already isotonically calibrated on, and isotonic
    regression is idempotent, so the arm is the identity by construction and the residual is
    reported as such. Ablation E is Method C with one stratum, which reduces to Method B, and the
    equality is asserted here rather than left as a coincidence for a reader to notice.
    """
    started = time.monotonic()
    folds, _ = load_folds()
    calibration = cc._read_json(CALIBRATION_RESULTS)
    coverage = cc._read_json(RISK_COVERAGE)

    per_fold: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        primary = calibration["folds"][engine]["arms"][PRIMARY_MAP]
        rows: dict[str, Any] = {}
        for label, arm in ABLATIONS.items():
            arm_block = calibration["folds"][engine]["arms"][arm]
            deployed = {
                f"epsilon_{int(e * 100)}": {
                    "repair_recall": coverage["folds"][engine]["arms"][arm][POINT_CUT][
                        f"epsilon_{int(e * 100)}"
                    ]["repair_recall"],
                    "delta_vs_primary": coverage["folds"][engine]["arms"][arm][POINT_CUT][
                        f"epsilon_{int(e * 100)}"
                    ]["repair_recall"]
                    - coverage["folds"][engine]["arms"][PRIMARY_MAP][POINT_CUT][
                        f"epsilon_{int(e * 100)}"
                    ]["repair_recall"],
                    "holds_bound": coverage["folds"][engine]["arms"][arm][POINT_CUT][
                        f"epsilon_{int(e * 100)}"
                    ]["holds_bound"],
                }
                for e in EPSILONS
            }
            rows[label] = {
                "arm": arm,
                "note": MAP_NOTES[arm],
                "brier": arm_block["calibration"]["brier"],
                "delta_brier_vs_primary": arm_block["calibration"]["brier"]
                - primary["calibration"]["brier"],
                "ece_equal_mass": arm_block["calibration"]["ece_equal_mass"],
                "delta_ece_vs_primary": arm_block["calibration"]["ece_equal_mass"]
                - primary["calibration"]["ece_equal_mass"],
                "auroc_vs_harmful": arm_block["ranking"]["auroc_vs_harmful"],
                "delta_auroc_vs_primary": arm_block["ranking"]["auroc_vs_harmful"]
                - primary["ranking"]["auroc_vs_harmful"],
                "kendall_tau_vs_original_score": arm_block["ranking"][
                    "kendall_tau_vs_original_score"
                ],
                "deployed": deployed,
            }
        per_fold[engine] = rows

        identity_residual = float(
            np.abs(
                fold.estimates[SOURCE_ISOTONIC]["evaluation"]
                - fold.estimates[IDENTITY]["evaluation"]
            ).max()
        )
        strata_residual = {
            name: float(
                np.abs(fold.estimates[NO_STRATA][name] - fold.estimates[RANK_TRANSPORT][name]).max()
            )
            for name in ("calibration", "pool", "evaluation")
        }
        conformal_residual = float(
            np.abs(
                fold.estimates[CONFORMAL]["evaluation"]
                - fold.estimates[QUANTILE_TRANSPORT]["evaluation"]
            ).max()
        )
        identities[engine] = {
            "ablation_d_equals_the_identity_map": {
                "max_abs_difference": identity_residual,
                "holds": bool(identity_residual <= 1e-6),
                "why": (
                    "SGV5 fitted its harm head's calibrator by isotonic regression on exactly "
                    "the rows this ablation refits on, and isotonic regression is the projection "
                    "onto the monotone cone. Projecting twice is projecting once, so the arm is "
                    "the identity as a matter of arithmetic and not of measurement."
                ),
            },
            "ablation_e_equals_method_b": {
                "max_abs_difference_per_block": strata_residual,
                "holds_where_the_projection_was_fitted": bool(strata_residual["pool"] <= 1e-9),
                "max_abs_difference_where_it_interpolates": strata_residual["evaluation"],
                "why": (
                    "a marginal rank transport is already non-decreasing, so the isotonic "
                    "projection has no ordering to repair and the two arms agree exactly at "
                    "every score level the projection was fitted at -- the whole adaptation "
                    "pool. They differ only on rows whose score falls between two pool levels, "
                    "where the projection interpolates across a step of the transport. The "
                    "claim that the two are identical EVERYWHERE was made in an earlier draft "
                    "of this stage and is false; the leakage suite caught it and it is recorded "
                    "here rather than repaired by loosening the comparison."
                ),
            },
            "method_d_shares_method_c_point_estimate": {
                "max_abs_difference": conformal_residual,
                "holds": bool(conformal_residual == 0.0),
                "why": (
                    "Method D adds an inflation to the CUT, not a sharpening to the estimate. "
                    "The two arms differ only where a threshold is placed."
                ),
            },
        }

    permutation_check = {
        engine: {
            "marginal_preserved": bool(
                np.allclose(
                    np.sort(folds[engine].estimates[PERMUTED]["evaluation"]),
                    np.sort(folds[engine].estimates[QUANTILE_TRANSPORT]["evaluation"]),
                )
            ),
            "auroc_of_the_permutation": calibration["folds"][engine]["arms"][PERMUTED]["ranking"][
                "auroc_vs_harmful"
            ],
            "auroc_of_the_primary": calibration["folds"][engine]["arms"][PRIMARY_MAP]["ranking"][
                "auroc_vs_harmful"
            ],
        }
        for engine in sorted(folds)
    }
    cc._write_json_once(
        ABLATION_RESULTS,
        {
            "schema_version": "sgv9-ablation-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "primary_map": PRIMARY_MAP,
            "ablations": {label: MAP_NOTES[arm] for label, arm in ABLATIONS.items()},
            "folds": per_fold,
            "structural_identities": identities,
            "permutation_control": permutation_check,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"ablation: {len(ABLATIONS)} ablations x {len(per_fold)} folds")
    return 0


# ------------------------------------------------------------------ phase: --negative


def run_negative() -> int:
    """The five failure tests the brief requires, each written so it can come back positive."""
    started = time.monotonic()
    folds, _ = load_folds()
    calibration = cc._read_json(CALIBRATION_RESULTS)
    coverage = cc._read_json(RISK_COVERAGE)
    mapping = cc._read_json(CALIBRATION_MAPPING)
    efficiency = cc._read_json(LABEL_EFFICIENCY)

    # 1. Does calibration only rescale scores without improving deployment?
    # Run over the whole target-free family rather than the primary alone. The question is only
    # answerable on an arm that DOES improve calibration, and if the primary is not one of those
    # the test would come back False by vacuity and say nothing.
    rescale: dict[str, Any] = {}
    for engine in sorted(folds):
        base = calibration["folds"][engine]["arms"][IDENTITY]
        per_arm: dict[str, Any] = {}
        for arm in TARGET_FREE:
            block = calibration["folds"][engine]["arms"][arm]
            deployed = {
                f"epsilon_{int(e * 100)}": coverage["folds"][engine]["arms"][arm][POINT_CUT][
                    f"epsilon_{int(e * 100)}"
                ]["repair_recall"]
                - coverage["folds"][engine]["arms"][IDENTITY][POINT_CUT][f"epsilon_{int(e * 100)}"][
                    "repair_recall"
                ]
                for e in EPSILONS
            }
            safer = {
                f"epsilon_{int(e * 100)}": coverage["folds"][engine]["arms"][IDENTITY][POINT_CUT][
                    f"epsilon_{int(e * 100)}"
                ]["resampled"]["violation_rate"]
                - coverage["folds"][engine]["arms"][arm][POINT_CUT][f"epsilon_{int(e * 100)}"][
                    "resampled"
                ]["violation_rate"]
                for e in EPSILONS
            }
            improves_calibration = block["calibration"]["brier"] < base["calibration"]["brier"]
            improves_deployment = any(value > 0.0 for value in deployed.values()) or any(
                value > 0.0 for value in safer.values()
            )
            per_arm[arm] = {
                "delta_brier": block["calibration"]["brier"] - base["calibration"]["brier"],
                "delta_ece_equal_mass": block["calibration"]["ece_equal_mass"]
                - base["calibration"]["ece_equal_mass"],
                "delta_repair_recall": deployed,
                "reduction_in_violation_rate": safer,
                "improves_calibration": bool(improves_calibration),
                "improves_deployment_or_safety": bool(improves_deployment),
                "rescales_without_deploying_better": bool(
                    improves_calibration and not improves_deployment
                ),
            }
        rescale[engine] = {
            "per_arm": per_arm,
            "arms_that_improve_calibration": [
                arm for arm, cell in per_arm.items() if cell["improves_calibration"]
            ],
            "arms_that_rescale_without_deploying_better": [
                arm for arm, cell in per_arm.items() if cell["rescales_without_deploying_better"]
            ],
            **per_arm[PRIMARY_MAP],
        }

    # 2. Does distribution matching fail when engines have different error mechanisms?
    mechanism: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        shift = mapping["folds"][engine]["maps"][PRIOR_SHIFT]["diagnostics"]
        actual = float(fold.pool.harmful.mean())
        source = float(shift["source_prevalence"])
        estimated = float(shift["estimate"])
        transported = float(
            mapping["folds"][engine]["maps"][QUANTILE_TRANSPORT]["diagnostics"][
                "mean_on_the_target_pool"
            ]
        )
        mechanism[engine] = {
            "source_prevalence": source,
            "actual_pool_prevalence": actual,
            "label_free_estimate": estimated,
            "error_of_the_label_free_estimate": estimated - actual,
            "error_of_assuming_no_shift": source - actual,
            "estimate_beats_assuming_no_shift": bool(
                abs(estimated - actual) < abs(source - actual)
            ),
            "mean_of_the_rank_transport_on_the_pool": transported,
            "error_of_the_rank_transport": transported - actual,
            "note": (
                "the label-shift assumption is not testable without target labels; this "
                "comparison uses the pool's realised prevalence and is EVALUATION ONLY. It "
                "reaches no fitted quantity and no threshold."
            ),
        }

    # 3. Does calibration improve metrics but worsen actual harm?
    harm: dict[str, Any] = {}
    for engine in sorted(folds):
        cells = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            mine = coverage["folds"][engine]["arms"][PRIMARY_MAP][POINT_CUT][key]
            base = coverage["folds"][engine]["arms"][IDENTITY][POINT_CUT][key]
            cells[key] = {
                "realized_harm_rate": mine["realized_harm_rate"],
                "identity_realized_harm_rate": base["realized_harm_rate"],
                "delta": mine["realized_harm_rate"] - base["realized_harm_rate"],
                "violation_rate": mine["resampled"]["violation_rate"],
                "identity_violation_rate": base["resampled"]["violation_rate"],
                "worse_harm_despite_better_calibration": bool(
                    calibration["folds"][engine]["arms"][PRIMARY_MAP]["calibration"]["brier"]
                    < calibration["folds"][engine]["arms"][IDENTITY]["calibration"]["brier"]
                    and np.isfinite(mine["realized_harm_rate"])
                    and np.isfinite(base["realized_harm_rate"])
                    and mine["realized_harm_rate"] > base["realized_harm_rate"]
                ),
            }
        harm[engine] = cells

    # 4. Does few-shot calibration outperform the zero-label methods?
    few_shot: dict[str, Any] = {}
    for engine in sorted(folds):
        arms: dict[str, Any] = {}
        for arm in FEW_SHOT:
            budgets = efficiency["folds"][engine]["arms"][arm]
            zero = budgets["0"]["calibration"]["brier"]["mean"] if "0" in budgets else float("nan")
            best_budget, best_value = None, float("inf")
            for size in sorted(budgets, key=int):
                value = budgets[size]["calibration"]["brier"]["mean"]
                if np.isfinite(value) and value < best_value:
                    best_budget, best_value = int(size), value
            arms[arm] = {
                "brier_at_zero_labels": zero,
                "best_budget_by_brier": best_budget,
                "best_brier": best_value,
                "labels_help": bool(
                    best_budget is not None and best_budget > 0 and best_value < zero - 1e-12
                ),
                "brier_at_100_labels": budgets.get("100", {})
                .get("calibration", {})
                .get("brier", {})
                .get("mean", float("nan")),
            }
        few_shot[engine] = arms

    # 5. Does the method collapse on EasyOCR?
    collapse: dict[str, Any] = {}
    for engine in sorted(folds):
        cells = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            mine = coverage["folds"][engine]["arms"][PRIMARY_MAP][POINT_CUT][key]
            frontier = coverage["folds"][engine]["achievable_frontier"][key]["repair_recall"]
            share = mine["repair_recall"] / frontier if frontier > 1e-12 else float("nan")
            cells[key] = {
                "repair_recall": mine["repair_recall"],
                "achievable_frontier": frontier,
                "share_of_the_frontier": share,
                "holds_bound": bool(mine["holds_bound"]),
                "exceeds_the_frontier_by_violating": bool(
                    np.isfinite(share) and share > 1.0 and not mine["holds_bound"]
                ),
                "accepts_nothing": bool(mine["accepts_nothing"]),
                "the_frontier_itself_is_near_zero": bool(frontier < 0.05),
            }
        collapse[engine] = cells

    cc._write_json_once(
        NEGATIVE_TESTS,
        {
            "schema_version": "sgv9-negative-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "development_only": True,
            "synthetic": False,
            "primary_map": PRIMARY_MAP,
            "test_1_rescales_without_improving_deployment": rescale,
            "test_2_different_error_mechanisms": mechanism,
            "test_3_better_metrics_worse_harm": harm,
            "test_4_few_shot_beats_zero_label": few_shot,
            "test_5_collapse_on_easyocr": {
                "per_engine": collapse,
                "note": (
                    "a low repair recall is only a collapse if the ranking could have supported "
                    "more. The achievable frontier is the ceiling of the ordering on the same "
                    "rows, so the share of the frontier separates 'the map failed' from 'there "
                    "was nothing there to take'. A share ABOVE one is not success: the frontier "
                    "is the deepest cut whose realised harm holds the bound, so an arm can only "
                    "pass it by breaking the bound, and `exceeds_the_frontier_by_violating` says "
                    "when that is what happened."
                ),
            },
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print("negative: 5 tests written")
    return 0


# ------------------------------------------------------------------ phase: --figures


def run_figures() -> int:
    """Four figures, each carrying the development-only annotation the project's rules require."""
    started = time.monotonic()
    calibration = cc._read_json(CALIBRATION_RESULTS)
    coverage = cc._read_json(RISK_COVERAGE)
    reliability = cc._read_json(RELIABILITY_METRICS)
    efficiency = cc._read_json(LABEL_EFFICIENCY)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV9 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(coverage["folds"])
    written: list[Path] = []

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    shown = {
        IDENTITY: ("#444", "x", "baseline 0: no calibration"),
        TEMPERATURE: ("#3a5f9e", "o", "A: temperature"),
        PRIOR_SHIFT: ("#2e7d5b", "^", "A2: prior shift"),
        RANK_TRANSPORT: ("#8a5fa3", "s", "B: rank transport"),
        QUANTILE_TRANSPORT: ("#a33", "D", "C: quantile transport"),
        TRANSPORT_PRIOR: ("#c87a2b", "v", "C2: transport + prior"),
        ORACLE_EVAL: ("#777", "*", "oracle (evaluation labels)"),
    }

    # --- reliability_diagrams.png ---------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.2))
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        panel.plot([0, 1], [0, 1], "--", color="#bbb", linewidth=1)
        for arm, (colour, marker, label) in shown.items():
            bins = reliability["folds"][engine]["arms"][arm]["reliability_equal_mass"]
            if not bins:
                continue
            panel.plot(
                [b["mean_predicted"] for b in bins],
                [b["mean_observed"] for b in bins],
                marker=marker,
                color=colour,
                markersize=4,
                linewidth=1.2,
                label=label,
            )
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("predicted harm probability")
        panel.set_ylabel("realised harm rate")
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)
    np.atleast_1d(panels)[0].legend(fontsize=6, loc="upper left")
    finish(
        figure,
        FIGURE_DIR / "reliability_diagrams.png",
        "Reliability on the unseen engine, equal-mass bins",
    )

    # --- prefix_calibration.png -----------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.2))
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        for arm, (colour, marker, label) in shown.items():
            prefix = reliability["folds"][engine]["arms"][arm]["prefix"]
            ranks = sorted(int(k) for k in prefix)
            panel.plot(
                ranks,
                [prefix[str(k)]["predicted"] for k in ranks],
                marker=marker,
                color=colour,
                markersize=3,
                linewidth=1.2,
                label=label,
            )
        prefix = reliability["folds"][engine]["arms"][IDENTITY]["prefix"]
        ranks = sorted(int(k) for k in prefix)
        panel.plot(
            ranks,
            [prefix[str(k)]["realized"] for k in ranks],
            color="#000",
            linewidth=2.0,
            label="realised",
        )
        for epsilon in EPSILONS:
            panel.axhline(epsilon, color="#d33", linewidth=0.6, linestyle=":")
        panel.set_xscale("log")
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("accepted prefix (candidates)")
        panel.set_ylabel("harm rate over the prefix")
    np.atleast_1d(panels)[0].legend(fontsize=6, loc="upper left")
    finish(
        figure,
        FIGURE_DIR / "prefix_calibration.png",
        "Predicted versus realised harm over the prefixes a cut can select",
    )

    # --- risk_controlled_recall.png -------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.2))
    width = 0.11
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        block = coverage["folds"][engine]
        positions = np.arange(len(EPSILONS), dtype=float)
        panel.plot(
            positions,
            [
                block["achievable_frontier"][f"epsilon_{int(e * 100)}"]["repair_recall"]
                for e in EPSILONS
            ],
            "k_",
            markersize=26,
            label="achievable frontier",
        )
        panel.plot(
            positions,
            [
                block["rules_without_a_map"][SOURCE_THRESHOLD]["per_epsilon"][
                    f"epsilon_{int(e * 100)}"
                ]["repair_recall"]
                for e in EPSILONS
            ],
            "P",
            color="#000",
            markersize=7,
            label="SGV5 source threshold",
        )
        for offset, (arm, (colour, _, label)) in enumerate(shown.items()):
            values = [
                block["arms"][arm][POINT_CUT][f"epsilon_{int(e * 100)}"]["repair_recall"]
                for e in EPSILONS
            ]
            holds = [
                block["arms"][arm][POINT_CUT][f"epsilon_{int(e * 100)}"]["holds_bound"]
                for e in EPSILONS
            ]
            bars = panel.bar(
                positions + (offset - 3) * width, values, width * 0.9, color=colour, label=label
            )
            for bar, ok in zip(bars, holds, strict=True):
                if not ok:
                    bar.set_hatch("///")
                    bar.set_edgecolor("#000")
        panel.set_xticks(positions)
        panel.set_xticklabels([f"eps={e}" for e in EPSILONS])
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_ylabel("repair recall at the deployed cut")
    np.atleast_1d(panels)[0].legend(fontsize=6, loc="upper left")
    finish(
        figure,
        FIGURE_DIR / "risk_controlled_recall.png",
        "Deployed repair recall by cut rule. Hatched bars broke the bound",
    )

    # --- label_efficiency.png -------------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.2))
    palette = {
        FEW_SHOT_DIRECT: "#444",
        FEW_SHOT_SHRUNK: "#a33",
        FEW_SHOT_PRIOR: "#2e7d5b",
        FEW_SHOT_COMPOSED: "#c87a2b",
    }
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        for arm, colour in palette.items():
            budgets = efficiency["folds"][engine]["arms"][arm]
            sizes = sorted((int(k) for k in budgets), key=int)
            values = [budgets[str(n)]["calibration"]["brier"]["mean"] for n in sizes]
            lower = [budgets[str(n)]["calibration"]["brier"]["ci"][0] for n in sizes]
            upper = [budgets[str(n)]["calibration"]["brier"]["ci"][1] for n in sizes]
            panel.plot(sizes, values, "-o", color=colour, markersize=3, label=arm)
            panel.fill_between(sizes, lower, upper, color=colour, alpha=0.15)
        oracle = calibration["folds"][engine]["arms"][ORACLE_EVAL]["calibration"]["brier"]
        panel.axhline(oracle, color="#777", linestyle="--", linewidth=1, label="oracle (eval)")
        panel.set_xscale("symlog", linthresh=5)
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("target labels purchased")
        panel.set_ylabel("Brier on the evaluation block")
    np.atleast_1d(panels)[0].legend(fontsize=6)
    finish(
        figure,
        FIGURE_DIR / "label_efficiency.png",
        "Calibration error against target-label budget, 50 seeds",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv9-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "synthetic": False,
            "development_only": True,
            "sources": {
                cc._relative(path): file_sha256(path)
                for path in (
                    CALIBRATION_RESULTS,
                    RISK_COVERAGE,
                    RELIABILITY_METRICS,
                    LABEL_EFFICIENCY,
                )
                if path.is_file()
            },
            "figures": {cc._relative(path): file_sha256(path) for path in written},
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ phase: --decide


def _criterion_1(results: dict[str, Any], arm: str) -> dict[str, Any]:
    """Calibration error improves significantly over SGV5.

    Brier is the statistic, the interval is a paired document-clustered bootstrap against the
    identity map on the same rows, and the family is every target-free arm on every engine under
    Holm. "Significantly" is read as: the Holm-adjusted test survives AND the paired interval
    excludes zero on the improving side.
    """
    survivors = set(results["multiplicity"]["survivors"])
    per_engine: dict[str, Any] = {}
    for engine, block in sorted(results["folds"].items()):
        cell = block["arms"][arm]["vs_identity"]["brier"]
        per_engine[engine] = {
            "delta_brier": cell["estimate"],
            "ci": [cell["ci_lower"], cell["ci_upper"]],
            "p_value": cell["p_value"],
            "survives_holm": bool(f"{engine}|{arm}" in survivors),
            "improves": bool(cell["improves"]),
            "met": bool(cell["improves"] and f"{engine}|{arm}" in survivors),
        }
    met = [engine for engine, cell in per_engine.items() if cell["met"]]
    return {
        "met": len(met) >= 3,
        "engines_met": met,
        "rule": "a Holm-surviving paired improvement in Brier on at least 3 of 4 engines",
        "per_engine": per_engine,
    }


def _criterion_2(coverage: dict[str, Any], arm: str, rule: str) -> dict[str, Any]:
    """Risk-controlled repair recall improves on at least 3 of 4 unseen engines.

    Measured against SGV5's published deployed point -- the source threshold met with the unseen
    engine -- because that is what this stage proposes to replace. An arm that accepts nothing
    cannot meet this criterion at any epsilon, which is the intended behaviour: refusing is safe
    and is not an improvement in recall.
    """
    per_engine: dict[str, Any] = {}
    for engine, block in sorted(coverage["folds"].items()):
        cells = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            mine = block["arms"][arm][rule][key]
            base = block["rules_without_a_map"][SOURCE_THRESHOLD]["per_epsilon"][key]
            cells[key] = {
                "repair_recall": mine["repair_recall"],
                "baseline_repair_recall": base["repair_recall"],
                "delta": mine["repair_recall"] - base["repair_recall"],
                "improves": bool(mine["repair_recall"] > base["repair_recall"] + 1e-12),
                "accepts_nothing": bool(mine["accepts_nothing"]),
            }
        per_engine[engine] = {
            "per_epsilon": cells,
            "improves_at_the_primary_epsilon": cells[PRIMARY_KEY]["improves"],
            "improves_at_every_epsilon": all(cell["improves"] for cell in cells.values()),
        }
    met = [e for e, cell in per_engine.items() if cell["improves_at_the_primary_epsilon"]]
    return {
        "met": len(met) >= 3,
        "engines_met": met,
        "rule": "deployed repair recall above SGV5's published deployed point at "
        f"epsilon={PRIMARY_EPSILON}, on at least 3 of 4 engines",
        "per_engine": per_engine,
    }


def _criterion_3(coverage: dict[str, Any], arm: str, rule: str) -> dict[str, Any]:
    """Harm-bound violations decrease.

    Counted over the whole 4 x 3 grid of engines and epsilons, against SGV5's published deployed
    point, and read from the document-clustered resample distribution rather than from the single
    realised outcome so the number is a rate and not a coin flip.

    A cell in which the arm accepts NOTHING is excluded from the credit. An arm that refuses
    every deployment violates no bound and is worthless, and letting that count would make this
    criterion pass for the empty policy. That restriction makes the criterion strictly harder
    than the brief's wording, not easier.
    """
    mine_total = 0.0
    base_total = 0.0
    vacuous = 0
    cells: dict[str, Any] = {}
    for engine, block in sorted(coverage["folds"].items()):
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            mine = block["arms"][arm][rule][key]
            base = block["rules_without_a_map"][SOURCE_THRESHOLD]["per_epsilon"][key]
            empty = bool(mine["accepts_nothing"])
            vacuous += int(empty)
            mine_rate = float(mine["resampled"]["violation_rate"])
            base_rate = float(base["resampled"]["violation_rate"])
            mine_total += mine_rate
            base_total += base_rate
            cells[f"{engine}|{key}"] = {
                "violation_rate": mine_rate,
                "baseline_violation_rate": base_rate,
                "delta": mine_rate - base_rate,
                "accepts_nothing": empty,
            }
    n = len(cells)
    return {
        "met": bool(mine_total < base_total - 1e-12 and vacuous == 0),
        "mean_violation_rate": mine_total / max(n, 1),
        "baseline_mean_violation_rate": base_total / max(n, 1),
        "cells": n,
        "cells_in_which_the_arm_accepts_nothing": vacuous,
        "rule": "a lower mean violation rate than SGV5's deployed point across all 12 cells, "
        "AND no cell in which the arm accepts nothing",
        "per_cell": cells,
    }


def _criterion_4(results: dict[str, Any], arm: str) -> dict[str, Any]:
    """Ranking performance is preserved.

    A strictly increasing map leaves AUROC exactly unchanged, so any drop is precisely the
    information the map destroyed by collapsing distinct scores onto one value. The tolerance is
    zero to three decimal places rather than a hand-picked epsilon: a map that cannot hold AUROC
    to a thousandth has not preserved the ranking in any sense a deployment would care about.
    """
    per_engine: dict[str, Any] = {}
    for engine, block in sorted(results["folds"].items()):
        ranking = block["arms"][arm]["ranking"]
        drop = float(ranking["auroc_of_the_original_score"]) - float(ranking["auroc_vs_harmful"])
        per_engine[engine] = {
            "auroc": ranking["auroc_vs_harmful"],
            "auroc_of_the_original_score": ranking["auroc_of_the_original_score"],
            "auroc_drop": drop,
            "kendall_tau_vs_original_score": ranking["kendall_tau_vs_original_score"],
            "spearman_vs_original_score": ranking["spearman_vs_original_score"],
            "met": bool(drop <= 1e-3),
            "met_at_a_hundredth": bool(drop <= 1e-2),
        }
    met = [engine for engine, cell in per_engine.items() if cell["met"]]
    loose = [engine for engine, cell in per_engine.items() if cell["met_at_a_hundredth"]]
    return {
        "met": len(met) == len(per_engine) and bool(per_engine),
        "engines_met": met,
        "rule": "AUROC against the realised outcome no more than 0.001 below the original "
        "score's, on every engine",
        "engines_met_at_a_looser_hundredth": loose,
        "second_reading_note": (
            "the 0.001 bar is the pre-registered one and is what the verdict uses. A second "
            "reading at 0.01 is reported alongside it because the brief's word is 'destroying' "
            "and a reader is entitled to see how far the arm is from either bar. Neither reading "
            "can change this stage's verdict, which turns on criterion 1."
        ),
        "per_engine": per_engine,
    }


def _criterion_5(efficiency: dict[str, Any], results: dict[str, Any], arm: str) -> dict[str, Any]:
    """A zero-label or few-label method approaches the supervised oracle.

    "Approaches" is read on Brier, as the share of the identity-to-oracle gap the arm closes,
    with the oracle being the isotonic recalibration fitted on the evaluation block itself. Half
    the gap is the bar, fixed before the numbers were read, and it is required on at least three
    engines at a budget of at most 100 labels -- the largest the brief's grid contains.
    """
    per_engine: dict[str, Any] = {}
    for engine, block in sorted(results["folds"].items()):
        identity = float(block["arms"][IDENTITY]["calibration"]["brier"])
        oracle = float(block["arms"][ORACLE_EVAL]["calibration"]["brier"])
        span = identity - oracle
        budgets = efficiency["folds"][engine]["arms"][arm]
        shares = {
            size: (identity - budgets[size]["calibration"]["brier"]["mean"]) / span
            if abs(span) > 1e-12
            else float("nan")
            for size in sorted(budgets, key=int)
        }
        best = max((value for value in shares.values() if np.isfinite(value)), default=float("nan"))
        # Whether there is anything to close. The paired interval on the oracle's own Brier
        # improvement over the identity says whether a measurable calibration error exists at
        # all; where it does not, a "share of the gap closed" is a ratio with a denominator
        # inside its own noise and the criterion is asking for something unmeasurable.
        oracle_delta = block["arms"][ORACLE_EVAL]["vs_identity"]["brier"]
        per_engine[engine] = {
            "identity_brier": identity,
            "oracle_brier": oracle,
            "gap": span,
            "gap_is_measurable": bool(oracle_delta["improves"]),
            "oracle_delta_ci": [oracle_delta["ci_lower"], oracle_delta["ci_upper"]],
            "share_of_the_gap_closed_by_budget": shares,
            "best_share": best,
            "best_absolute_brier_improvement": float(
                identity
                - min(
                    (budgets[size]["calibration"]["brier"]["mean"] for size in budgets),
                    default=identity,
                )
            ),
            "met": bool(np.isfinite(best) and best >= 0.5),
        }
    met = [engine for engine, cell in per_engine.items() if cell["met"]]
    return {
        "met": len(met) >= 3,
        "engines_met": met,
        "rule": "at least half the identity-to-oracle Brier gap closed at a budget of at most "
        "100 labels, on at least 3 of 4 engines",
        "engines_with_a_measurable_gap_to_close": [
            engine for engine, cell in per_engine.items() if cell["gap_is_measurable"]
        ],
        "gap_note": (
            "where the oracle's own paired improvement over the identity does not exclude zero, "
            "there is no measurable calibration error for a method to remove and the share is a "
            "ratio with a denominator inside its own noise. That is reported rather than used to "
            "excuse a failure: the criterion stands as written."
        ),
        "per_engine": per_engine,
    }


def _criteria_for(
    arm: str,
    few_shot_arm: str,
    rule: str,
    results: dict[str, Any],
    coverage: dict[str, Any],
    efficiency: dict[str, Any],
) -> dict[str, Any]:
    return {
        "criterion_1_calibration_error_improves": _criterion_1(results, arm),
        "criterion_2_risk_controlled_recall_improves": _criterion_2(coverage, arm, rule),
        "criterion_3_harm_bound_violations_decrease": _criterion_3(coverage, arm, rule),
        "criterion_4_ranking_preserved": _criterion_4(results, arm),
        "criterion_5_approaches_the_oracle": _criterion_5(efficiency, results, few_shot_arm),
    }


def run_decide() -> int:
    """The machine-readable finding, with the brief's five criteria applied as written."""
    started = time.monotonic()
    results = cc._read_json(CALIBRATION_RESULTS)
    coverage = cc._read_json(RISK_COVERAGE)
    efficiency = cc._read_json(LABEL_EFFICIENCY)
    negative = cc._read_json(NEGATIVE_TESTS)
    oracle = cc._read_json(ORACLE_GAP)

    family = {
        arm: _criteria_for(arm, few_shot, rule, results, coverage, efficiency)
        for arm, few_shot, rule in (
            (QUANTILE_TRANSPORT, FEW_SHOT_SHRUNK, POINT_CUT),
            (TRANSPORT_PRIOR, FEW_SHOT_COMPOSED, POINT_CUT),
            (PRIOR_SHIFT, FEW_SHOT_PRIOR, POINT_CUT),
            (TEMPERATURE, FEW_SHOT_DIRECT, POINT_CUT),
            (RANK_TRANSPORT, FEW_SHOT_SHRUNK, POINT_CUT),
            (CONFORMAL, FEW_SHOT_SHRUNK, BOUND_CUT),
        )
    }
    primary = family[PRIMARY_MAP]
    met = [name for name, cell in sorted(primary.items()) if cell["met"]]
    supported = len(met) == 5

    bottleneck = {
        engine: {
            "transfer_inflation_at_the_primary_epsilon": coverage["folds"][engine][
                "transfer_inflation"
            ][PRIMARY_KEY],
            "largest_inflation_the_bound_could_afford": coverage["folds"][engine]["arms"][
                QUANTILE_TRANSPORT
            ][BOUND_CUT][PRIMARY_KEY]["max_inflation_that_still_deploys"],
            "achievable_frontier": coverage["folds"][engine]["achievable_frontier"][PRIMARY_KEY][
                "repair_recall"
            ],
            "evaluation_oracle_repair_recall": coverage["folds"][engine]["arms"][ORACLE_EVAL][
                POINT_CUT
            ][PRIMARY_KEY]["repair_recall"],
            "pool_versus_evaluation_prevalence": {
                "pool": coverage["folds"][engine].get("harm_prevalence"),
                "note": "see design_record.json -> folds -> diagnostics -> harm_prevalence",
            },
            "share_of_the_oracle_brier_gap_closed": oracle["folds"][engine][
                "brier_share_of_the_oracle_gap_closed"
            ][PRIMARY_MAP],
        }
        for engine in sorted(coverage["folds"])
    }

    cc._write_json_once(
        DECISION,
        {
            "schema_version": "sgv9-decision-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "stage": "SGV9 -- target-free risk calibration",
            "primary_map": PRIMARY_MAP,
            "primary_rule": PRIMARY_RULE,
            "primary_few_shot_arm": PRIMARY_FEW_SHOT,
            "epsilon": PRIMARY_EPSILON,
            "verdict": "SUPPORTED" if supported else "NOT SUPPORTED",
            "criteria_met": met,
            "criteria": primary,
            "pre_registration_note": (
                "Method C is the primary map, named in the module docstring and in "
                "design_record.json before any SGV9 number was read, because the brief names it "
                "as the main proposed method. The composed arm mC2_transport_prior was added "
                "after the fitted maps' marginals were read and before any endpoint was "
                "computed; design_record.json records exactly what had been seen at that point. "
                "Every arm below carries the multiplicity of a family of six."
            ),
            "exploratory_family": family,
            "negative_tests": {
                "test_1_rescales_without_improving_deployment": {
                    engine: cell["rescales_without_deploying_better"]
                    for engine, cell in negative[
                        "test_1_rescales_without_improving_deployment"
                    ].items()
                },
                "test_2_label_free_prevalence_beats_assuming_no_shift": {
                    engine: cell["estimate_beats_assuming_no_shift"]
                    for engine, cell in negative["test_2_different_error_mechanisms"].items()
                },
                "test_5_share_of_the_frontier_at_the_primary_epsilon": {
                    engine: cell[PRIMARY_KEY]["share_of_the_frontier"]
                    for engine, cell in negative["test_5_collapse_on_easyocr"]["per_engine"].items()
                },
            },
            "remaining_bottleneck": {
                "note": (
                    "the transfer inflation is the largest error the same pipeline made when it "
                    "was replayed onto a source engine it had not calibrated from. Where it "
                    "exceeds what the bound could afford, no sharpening of the point estimate "
                    "would make a certified deployment possible at that epsilon; the binding "
                    "constraint is the engine-to-engine variability of the calibration itself."
                ),
                "per_engine": bottleneck,
            },
            "claims_not_made": [
                "SGV9 does not claim to solve OCR correction.",
                "SGV9 does not claim universal adaptation.",
                "SGV9 does not claim domain generalization.",
                "SGV9 does not claim a distribution-free guarantee at 1 - alpha: with three "
                "source engines the coverage a leave-one-engine-out quantile can support is 3/4.",
                "No result here is confirmatory: the reserve is locked and was not accessed.",
            ],
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: {'SUPPORTED' if supported else 'NOT SUPPORTED'} ({len(met)}/5 criteria)")
    return 0


# ------------------------------------------------------------------ provenance


def run_record() -> int:
    started = time.monotonic()
    produced = [
        path
        for path in (
            SCORES,
            FEW_SHOT_CALIBRATION,
            FEW_SHOT_DEPLOYMENT,
            DESIGN_RECORD,
            CALIBRATION_MAPPING,
            OUT / "mapping_verification.json",
            CALIBRATION_RESULTS,
            RELIABILITY_METRICS,
            RISK_COVERAGE,
            LABEL_EFFICIENCY,
            ORACLE_GAP,
            ABLATION_RESULTS,
            NEGATIVE_TESTS,
            DECISION,
            FIGURE_MANIFEST,
        )
        if path.is_file()
    ]
    figures = sorted(FIGURE_DIR.glob("*.png")) if FIGURE_DIR.is_dir() else []
    inputs = [
        dg.DESIGN_MATRIX,
        c5.FEATURES,
        c5.PREDICTIONS,
        c5.POLICY_SELECTION,
        c5.FIT_RECORD,
    ]
    cc._write_json_once(
        PROVENANCE,
        {
            "schema_version": "sgv9-provenance-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV9-T1",
            "stage": "SGV9 -- target-free risk calibration",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "inputs": {cc._relative(p): file_sha256(p) for p in inputs if p.is_file()},
            "artifacts": {cc._relative(p): file_sha256(p) for p in (*produced, *figures)},
            "maps": MAP_NOTES,
            "cut_rules": RULE_NOTES,
            "frozen_components": (
                "the SGV5 model, its representation, its harm-aversion weight, the candidate "
                "features, the candidate pool and the decision score are rebuilt from SGV5's own "
                "record and asserted equal to SGV5's published score vector bit for bit on every "
                "fold. SGV9 changes the harm PROBABILITY attached to a candidate and the "
                "threshold that probability implies, and nothing else."
            ),
            "strata": STRATUM_NOTE,
            "inner_folds": (
                "for the outer fold that holds out E, each fit engine takes a turn as a "
                "pseudo-target and the whole pipeline is refitted from the remaining fit "
                "engines. E appears nowhere in an inner fold and no DEVELOPMENT page appears "
                "anywhere in one; the inner outcome is read on the half of the CALIBRATION pages "
                "the inner map was not calibrated on."
            ),
            "budget": {
                "grid": list(BUDGETS),
                "seed_repeats": SEED_REPEATS,
                "acquisition": "random, SGV6's ordering under SGV6's seed formula",
                "shrinkage_prior_labels": SHRINKAGE_PRIOR,
            },
            "statistics": {
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "unit": "document",
                "paired": True,
                "multiplicity": "holm",
                "primary_calibration_statistic": "brier",
                "ece_bins": N_BINS,
                "delta": DELTA,
                "distribution_free_coverage_of_the_transfer_bound": CONFORMAL_COVERAGE,
            },
            "invariants": {
                "held_out_engine_and_held_out_document": True,
                "map_fitted_without_any_target_label": True,
                "threshold_placed_without_reading_the_evaluation_block": True,
                "oracles_used_only_as_upper_bounds": True,
                "engine_identity_used_as_a_feature": False,
                "confirmatory_accessed": False,
            },
            "commands": [
                "uv run python scripts/sgv9_target_calibration.py --calibrate",
                "uv run python scripts/sgv9_target_calibration.py --mapping --results "
                "--reliability --coverage --efficiency --oracle --ablation --negative "
                "--figures --decide --record",
            ],
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"record: {len(produced) + len(figures)} artifacts -> {cc._relative(PROVENANCE)}")
    return 0


def main() -> int:
    stages = {
        "--calibrate": run_calibrate,
        "--mapping": run_mapping,
        "--results": run_results,
        "--reliability": run_reliability,
        "--coverage": run_coverage,
        "--efficiency": run_efficiency,
        "--oracle": run_oracle,
        "--ablation": run_ablation,
        "--negative": run_negative,
        "--figures": run_figures,
        "--decide": run_decide,
        "--record": run_record,
    }
    requested = [flag for flag in sys.argv[1:] if flag in stages]
    unknown = [flag for flag in sys.argv[1:] if flag not in stages]
    if unknown or not requested:
        print(f"usage: {Path(__file__).name} [{' | '.join(stages)}]")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    for flag in stages:
        if flag in requested:
            code = stages[flag]()
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
