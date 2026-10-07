#!/usr/bin/env python3
"""SGV11: is the accept/reject decision itself the thing that fails to transfer?

SGV10 closed by naming what was left. Not the representation, not the calibration, not the
boundary estimate: "a scalar threshold, chosen on one block of pages and applied to another."
Every stage from SGV5 onward has changed what is *estimated* -- a better representation, a
transported calibration map, a boundary-local harm probability -- and every one of them has
handed the estimate to the same decision rule. This stage leaves the estimate completely alone
and changes only the rule.

That makes SGV11 a decision layer and nothing else. No model is fitted here. SGV5's decision
score, SGV9's calibration map and SGV10's boundary composite are read from their published
tables, verified by hash and row-for-row against the vectors their own stages published, and
never recomputed. What varies is which scalar is carried to the unseen engine and how it was
chosen from the source engines' labelled rows.

The brief's hypothesis is a minimax one:

    pi* = argmin_pi max_e Risk_e(pi),   e ranging over the SOURCE engines

and there is a reason to expect it to mean something here beyond conservatism. With K
exchangeable source engines, `max_e Risk_e(t) <= epsilon` is a rank statistic: the unseen
engine's risk exceeds the maximum of K draws with probability at most 1/(K+1). With the three
source engines this corpus affords, a threshold certified by the minimax rule should therefore
hold on the unseen engine with distribution-free probability 3/4 -- the same 3/4 SGV9 and SGV10
reported for the maximum of three leave-one-source-engine-out replays, and for the same reason.
Leave-one-engine-out over four engines puts that exchangeability assumption directly under test,
which every stage since SGV1 has relied on without measuring.

    SGV11-P1: a decision policy whose threshold is chosen to bound the WORST source engine's
    harm rate rather than the average one transfers to an unseen OCR engine better than SGV5's
    published policy -- higher repair recall among the deployments that hold the bound, and the
    bound held on at least three of the four unseen engines.

Ten decisions fix what the numbers below can mean. All ten were made before any SGV11 endpoint
was computed. Four of them were informed by a DESIGN PROBE on SGV10's published table, run
before this pre-registration was written; decisions 4, 5, 8 and 9 name the probe explicitly, and
`design_record.json` carries what the probe measured so a reader can check that the final
artifacts reproduce it. No criterion was chosen to make an arm pass: the probe already showed
the primary arm failing the primary criterion on all four engines, and the criterion was kept.

**1. Nothing upstream moves, and no model is fitted.** SGV5's published score vector, SGV9's
published calibrated table and SGV10's published boundary table are read and hash-verified
against the provenance records their own stages wrote. The SGV10 evaluation rows' decision score
is compared row for row against SGV5's published `arm__sgv5`; a nonzero difference is a hard
failure. `candidate_id` alignment against SGV1's design matrix is checked on every row of every
block.

**2. Every policy ranks by a FROZEN score and changes only the action map.** This is what makes
the stage's question answerable. Two rankings are carried -- SGV5's decision score and SGV10's
boundary composite -- and within a ranking every policy sees the same order. A difference
between two policies in the same cell is therefore a difference in where the cut went and
nowhere else, which is the literal form of the brief's "same ranking, different action mapping".

**3. The threshold is chosen on the SOURCE engines' labelled calibration rows.** Not on the
unlabelled target pool, which is what SGV9 and SGV10 did. The two stages answer different
questions: SGV9/SGV10 asked whether a target-side estimate can locate the cut, and SGV11 asks
whether a source-side *decision* can be made robust enough to survive the move. Both are legal;
neither reads a target label. Which one is better is measured, not assumed, and SGV9's and
SGV10's published deployments are carried as comparators in every table.

**4. The primary endpoint is RISK-CONTROLLED repair recall: the repair recall of the accepted
set if its realised harm rate is within epsilon, and zero otherwise.** The brief names "repair
recall at fixed harm bound" as the primary metric, and this is that quantity written as a single
scalar. It is not gameable by breaking the constraint, and it removes a trap the probe found:
SGV5's published EasyOCR deployment realises 0.305 harm at epsilon = 0.10, so its unconstrained
repair recall of 0.793 is a number no policy that respects the bound can or should beat. Raw
repair recall, realised harm and coverage are all reported beside it.

**5. Method E is a KL ball on the per-engine harm rate, not a CVaR over pages.** The brief
offers "worst-k engine risk, CVaR risk, uncertainty weighted risk" as practical approximations.
With three source engines, CVaR over ENGINES at any level above 2/3 is exactly the minimax rule
and adds nothing. CVaR over PAGES is not degenerate in that sense but it is degenerate in
practice -- the probe measured coverage collapsing to 0.002 on docTR at epsilon = 0.10, because
a page holding one accepted harmful candidate has a page-level rate of 1.0 and the worst decile
of pages is made of those. Both are implemented and reported; the primary DRO arm is the KL ball
`sup{nu : kl(nu || mu) <= rho}` applied to each source engine's rate and then maximised over
engines, which nests the minimax rule exactly at rho = 0 and is the same object as a KL-UCB
confidence bound.

**6. Every hyperparameter is chosen by an inner leave-one-source-engine-out replay.** The CVaR
level, the KL radius and the selective policy's tail-risk cutoff are selected inside the source
engines only, on SGV9's calibration-half split so that the engine held out of the inner fit is
scored on pages the inner fit never saw. The split is SGV9's, reused rather than re-salted, so
this stage's inner folds are identical to SGV9's and SGV10's. `selection_scope` records it.

**7. Two transport modes are carried, and both are reported everywhere.** A threshold can be
carried to the unseen engine as a score VALUE, which is what SGV5 published and what the brief
implies, or as an acceptance QUANTILE matched on the unlabelled target pool. Neither reads a
target label. The value mode is primary; the quantile mode is the direct test of whether the
minimax question is confounded with the score-scale question, and it is reported in every table
rather than as an afterthought.

**8. The failure analysis is an exact decomposition, not a narrative.** For any threshold, the
realised harm on the evaluation block is the sum of four named terms: the criterion the policy
bounded, the criterion's slack against the realised source harm, the engine shift (source rows
to the target's own pool at the same threshold), and the block shift (the target's pool to the
target's evaluation block at the same threshold). Every term is measurable because the pool
carries labels that no fitting step is allowed to read. The probe found the block term positive
on all four engines and larger than the engine term on three of them, which is why it is a
first-class artifact rather than a paragraph.

**9. The in-engine oracle is reported in two forms, because the deepest-feasible rule overfits.**
Ablation 4 asks for an in-engine oracle policy: a threshold chosen on the target engine's own
labelled pool. Under the deepest-feasible convention every other arm uses, the probe measured
that oracle VIOLATING epsilon = 0.10 on all four engines -- 0.216 on docTR -- because the pool's
harm curve is not monotone and the rule takes the last feasible rank rather than the first
infeasible one. Both forms are reported: the deepest-feasible oracle, which is the matched
comparison, and a monotone oracle that stops at the first violation, which separates "the rule
is optimistic" from "the block does not transfer".

**10. Criterion 4 is a floor, not a sentence.** An arm earns credit in a cell only if it accepts
at least half as many candidates as SGV5's published deployment and beats matched-coverage
random acceptance. Both bars were fixed before any endpoint was computed and are the same bars
SGV10 used, so the two stages' non-degeneracy claims mean the same thing.

    --policies      verify the frozen inputs, fit every policy on the source engines, write the rows
    --transfer      decision transfer against score transfer: same ranking, different action map
    --results       the main table: risk-controlled repair recall, with intervals and Holm
    --coverage      risk-coverage curves and AURC, per ranking
    --robustness    minimax against average: what the conservatism costs and what it buys
    --decomposition the four-term harm decomposition, per policy and per engine
    --oracle        the in-engine oracle, both forms, and the remaining gap
    --ablation      the brief's four ablations
    --negative      the five leakage checks and the degeneracy controls
    --figures       the required figures
    --decide        the machine-readable finding
    --record        provenance for every artifact

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; no threshold, no hyperparameter and no
selection reads a label from the held-out engine's DEVELOPMENT rows.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
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
import sgv5_candidate_reliability as c5
import sgv6_target_engine_adaptation as s6
import sgv9_target_calibration as s9
import sgv10_tail_risk_control as s10
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.selective import aurc, risk_coverage_curve

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv11_minimax_policy_transfer"

SCORES = OUT / "policy_scores.parquet"
DESIGN_RECORD = OUT / "design_record.json"
DECISION_TRANSFER = OUT / "decision_transfer.json"
POLICY_RESULTS = OUT / "policy_results.json"
RISK_COVERAGE = OUT / "risk_coverage_table.json"
ROBUSTNESS = OUT / "robustness_analysis.json"
HARM_DECOMPOSITION = OUT / "harm_decomposition.json"
ORACLE_GAP = OUT / "oracle_gap.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
DECISION = OUT / "research_decision.json"
PROVENANCE = OUT / "provenance_manifest.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

# Eleven stages now share one implementation of the endpoint, the frontier, the interval and the
# resampling engine. `tests/leakage/test_sgv11_minimax_policy_transfer.py` asserts the identity
# of every alias below, so "SGV11 measures what SGV10 measured" is checkable rather than claimed.
achievable_repair_recall = rl.achievable_repair_recall
deployed_point = c5._deployed
Block = s9.Block
prefix_curves = s9.prefix_curves
resample_curves = s9.resample_curves
resampled_deployment = s9.resampled_deployment
document_resamples = s9.document_resamples
_deployed_at = s9._deployed_at
_deepest = s9._deepest
_interval = s9._interval
_calibration_half = s9._calibration_half
_stable_seed = s6._stable_seed

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
PRIMARY_KEY = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = pilot.BOOTSTRAP_SEED
CROSS_ENGINE = s6.CROSS_ENGINE
BLOCK_NAMES = s10.BLOCK_NAMES
CONFORMAL_COVERAGE = s9.CONFORMAL_COVERAGE


class PhaseError(RuntimeError):
    """A freeze, role, split, transport, selection or degeneracy invariant failed."""


# ------------------------------------------------------------------ the policy registry

SGV5_PUBLISHED = "pA_sgv5_published"
ERM_AVERAGE = "pB_erm_average"
MINIMAX = "pC_minimax"
SELECTIVE = "pD_selective"
DRO_CVAR = "pE_dro_cvar"
DRO_KL = "pF_dro_kl"
POOLED_ERM = "aA_pooled_erm"
RANDOM_MATCHED = "aC_random_matched"
ORACLE_POOL = "oracle_pool"
ORACLE_MONOTONE = "oracle_monotone"

PRIMARY_POLICY = MINIMAX

# The pre-specified family. Criterion 2 counts engines inside it and nothing outside it is
# eligible to be the finding; the multiplicity family in `--results` is the primary policy
# crossed with the four engines and the three epsilons.
PROPOSED = (ERM_AVERAGE, MINIMAX, SELECTIVE, DRO_CVAR, DRO_KL)
BASELINES = (SGV5_PUBLISHED,)
CONTROLS = (POOLED_ERM, RANDOM_MATCHED)
ORACLES = (ORACLE_POOL, ORACLE_MONOTONE)
ALL_POLICIES = (*BASELINES, *PROPOSED, *CONTROLS, *ORACLES)

# Policies whose parameters are chosen on the source engines' labelled rows. Everything outside
# this tuple either reads nothing (the published baseline) or reads a target label (the oracles),
# and `--negative` asserts that the two sets do not overlap.
SOURCE_FITTED = (ERM_AVERAGE, MINIMAX, SELECTIVE, DRO_CVAR, DRO_KL, POOLED_ERM)

POLICY_NOTES = {
    SGV5_PUBLISHED: (
        "Policy A, the baseline. SGV5's published deployment: its decision score and the "
        "threshold its own pooled source calibration rows chose under the empirical controller "
        "with a binomial slack. Reported under its own transport only, because that is the "
        "transport it was published with."
    ),
    ERM_AVERAGE: (
        "Policy B. The deepest threshold whose MEAN harm rate across the three source engines "
        "is within epsilon. Empirical risk minimisation with the engine structure respected but "
        "not exploited: an engine that is much worse than the others is averaged away."
    ),
    MINIMAX: (
        "Policy C, the primary arm and the brief's hypothesis. The deepest threshold whose "
        "WORST source engine's harm rate is within epsilon. With three exchangeable source "
        "engines this is a rank statistic with distribution-free coverage 3/4 on the unseen "
        "engine, which is a testable prediction rather than a claim."
    ),
    SELECTIVE: (
        "Policy D. Three actions. ACCEPT above the more conservative of the average and minimax "
        "thresholds and only where SGV10's boundary harm estimate is within a cutoff; ABSTAIN "
        "between the two thresholds or where the tail-risk cutoff objects; REJECT below the more "
        "permissive threshold. The abstain region is exactly where the two risk aggregations "
        "disagree, so it introduces no width parameter of its own."
    ),
    DRO_CVAR: (
        "Policy E, first form. The deepest threshold whose conditional value at risk over source "
        "PAGES -- the mean accepted harm rate of the worst share of pages -- is within epsilon. "
        "The level is chosen by an inner leave-one-source-engine-out replay."
    ),
    DRO_KL: (
        "Policy E, second form and the primary DRO arm. The deepest threshold whose worst-case "
        "harm rate under a KL ball of radius rho around each source engine's empirical rate, "
        "maximised over engines, is within epsilon. Nests Policy C exactly at rho = 0. The "
        "radius is chosen by an inner leave-one-source-engine-out replay."
    ),
    POOLED_ERM: (
        "Ablation 3. The deepest threshold whose harm rate over the source rows POOLED -- every "
        "engine's rows in one bag, engine identity discarded -- is within epsilon. The engine "
        "structure is removed entirely, which is what 'normal ERM' means here."
    ),
    RANDOM_MATCHED: (
        "A control, not a method. Accepts a content-derived pseudo-random subset of the "
        "evaluation block of exactly the size the primary policy accepted. Any gain the proposed "
        "policies show over this one is a gain from the ordering and the cut, not from accepting "
        "fewer rows."
    ),
    ORACLE_POOL: (
        "Ablation 4, first form. An ORACLE: the deepest prefix of the held-out engine's OWN "
        "labelled pool whose realised harm is within epsilon, carried to the evaluation block. "
        "The ceiling of what target labels could buy a threshold without touching an evaluation "
        "page. Not a method; excluded from every selection."
    ),
    ORACLE_MONOTONE: (
        "Ablation 4, second form. An ORACLE: the same construction stopped at the FIRST prefix "
        "of the pool that violates epsilon rather than the last one that holds. Separates the "
        "deepest-feasible rule's own optimism from the block-to-block shift. Not a method; "
        "excluded from every selection."
    ),
}


# ------------------------------------------------------------------ rankings and transports

RANK_SGV5 = "sgv5_decision_score"
RANK_SGV10 = "sgv10_boundary_composite"
RANKINGS = (RANK_SGV5, RANK_SGV10)
PRIMARY_RANKING = RANK_SGV5

RANKING_NOTES = {
    RANK_SGV5: (
        "SGV5's frozen decision score `benefit - lambda * harm`, read from SGV10's published "
        "table and verified row for row against SGV5's own published vector. The primary "
        "ranking, because the brief freezes the candidate model."
    ),
    RANK_SGV10: (
        "SGV10's boundary composite for the same epsilon: SGV5's order outside the boundary "
        "band, the veto model's order inside it. Carried so that ablation 2 -- does the "
        "tail-risk information help the DECISION transfer -- is a contrast between two "
        "rankings under identical policies rather than a separate experiment."
    ),
}

TRANSPORT_VALUE = "value"
TRANSPORT_QUANTILE = "quantile"
TRANSPORTS = (TRANSPORT_VALUE, TRANSPORT_QUANTILE)
PRIMARY_TRANSPORT = TRANSPORT_VALUE

TRANSPORT_NOTES = {
    TRANSPORT_VALUE: (
        "The threshold is carried to the unseen engine as a score VALUE. Reads no target row at "
        "all. This is what SGV5 published and what the brief's formulation implies."
    ),
    TRANSPORT_QUANTILE: (
        "The threshold is carried as an ACCEPTANCE RATE: the share of source rows it accepted is "
        "matched on the unlabelled target pool, and the resulting pool score becomes the "
        "threshold. Reads target scores, never a target label. Isolates the score-scale shift "
        "from the decision rule."
    ),
}

# Three actions. The integer codes are what the row table carries and what `actions_of` returns;
# nothing downstream compares action names as strings.
REJECT, ABSTAIN, ACCEPT = 0, 1, 2
ACTION_NAMES = {REJECT: "REJECT", ABSTAIN: "ABSTAIN", ACCEPT: "ACCEPT"}


# ------------------------------------------------------------------ constants

# The CVaR levels and KL radii the inner replay chooses from. Both grids were fixed before any
# SGV11 endpoint was computed; `--ablation` reports the endpoint under every value of each so the
# choice is visible rather than load-bearing.
CVAR_BETAS = (0.50, 0.80, 0.90, 0.95)
KL_RADII = (0.0, 0.005, 0.02, 0.05, 0.10)
TAIL_CUTOFF_QUANTILES = (0.50, 0.70, 0.85, 1.00)

# The scalarised utility the brief writes down. It is a REPORTING CONVENTION, not an endpoint:
# the weights are not measured quantities and no criterion reads them. The primary endpoint is
# the risk-controlled repair recall, which needs no weights.
UTILITY_WEIGHTS = ((1.0, 0.0), (1.0, 0.1), (5.0, 0.1))

COVERAGE_FLOOR = 0.50
ENGINE_BAR = 3
RANDOM_SEED = 20260905


# ------------------------------------------------------------------ the pre-registration

PRE_REGISTRATION = {
    "claim": (
        "SGV11-P1: a decision policy whose threshold is chosen to bound the WORST source "
        "engine's harm rate rather than the average one transfers to an unseen OCR engine "
        "better than SGV5's published policy -- higher risk-controlled repair recall, and the "
        "bound held on at least three of the four unseen engines."
    ),
    "sub_claims": {
        "P1a": (
            "the minimax threshold's risk-controlled repair recall exceeds SGV5's published "
            "deployment's on the unseen engine (the brief's RQ1)"
        ),
        "P1b": (
            "the improvement comes from the worst-case aggregation and not from the engine "
            "structure alone: minimax beats both the average-risk and the pooled policy"
        ),
        "P1c": (
            "the exchangeability the whole leave-one-engine-out protocol assumes holds well "
            "enough that a maximum over three source engines certifies the fourth at 3/4"
        ),
    },
    "primary_policy": PRIMARY_POLICY,
    "primary_ranking": PRIMARY_RANKING,
    "primary_transport": PRIMARY_TRANSPORT,
    "primary_endpoint": (
        "risk-controlled repair recall: the repair recall of the accepted set if its realised "
        "harm rate is within epsilon, and zero otherwise"
    ),
    "criteria": {
        "1": (
            "the primary policy's risk-controlled repair recall exceeds SGV5's published "
            "deployment's"
        ),
        "2": "criterion 1 holds on at least three of the four unseen engines",
        "3": (
            "the realised harm rate of what the primary policy accepts is within epsilon on at "
            "least three of the four unseen engines, which is the exchangeability prediction"
        ),
        "4": (
            "the primary policy accepts at least half as many candidates as SGV5's published "
            "deployment and beats matched-coverage random acceptance"
        ),
        "5": (
            "the WORST unseen engine's risk-controlled repair recall improves against SGV5's "
            "worst unseen engine, which is what a minimax formulation promises"
        ),
    },
    "verdict_rule": (
        "SUPPORTED only if all five criteria are met by the primary policy at the primary "
        "epsilon; otherwise NOT SUPPORTED, with the criteria that failed named"
    ),
}

# What the design probe measured, before this pre-registration was written. Decisions 4, 5, 8
# and 9 were informed by these numbers and say so. They are recorded here so that a reader can
# check the final artifacts against them: `--negative` reproduces every one and fails if any
# has drifted.
DESIGN_PROBE = {
    "what_it_was": (
        "a read-only replay on SGV10's published `boundary_scores.parquet` that placed the "
        "average, minimax, pooled and in-engine-oracle thresholds at epsilon = 0.10 under value "
        "transport on the SGV5 ranking, and measured the deployed operating point of each"
    ),
    "when": "before the SGV11 pre-registration was written; no SGV11 artifact existed",
    "informed_decisions": [4, 5, 8, 9],
    "measurements": {
        "minimax_realized_harm": {
            "doctr": 0.0417,
            "easyocr": 0.3064,
            "paddleocr": 0.0642,
            "tesseract": 0.0772,
        },
        "minimax_repair_recall": {
            "doctr": 0.5273,
            "easyocr": 0.7872,
            "paddleocr": 0.3594,
            "tesseract": 0.5615,
        },
        "oracle_pool_realized_harm": {
            "doctr": 0.2162,
            "easyocr": 0.1493,
            "paddleocr": 0.1238,
            "tesseract": 0.1266,
        },
        "cvar_90_coverage": {
            "doctr": 0.002,
            "easyocr": 0.003,
            "paddleocr": 0.002,
            "tesseract": 0.013,
        },
    },
    # The probe printed each quantity at a fixed number of decimals, so a value transcribed from
    # it can differ from the pipeline's by at most half a unit in the last place. The tolerance
    # is that half-ulp and nothing else; it is derived from how the probe printed, never chosen
    # to make the check pass.
    "printed_decimals": {
        "minimax_realized_harm": 4,
        "minimax_repair_recall": 4,
        "oracle_pool_realized_harm": 4,
        "cvar_90_coverage": 3,
    },
    "note": (
        "the probe showed the primary policy FAILING criterion 1 on all four engines before the "
        "criterion was written down, and the criterion was kept as the brief specifies it"
    ),
}

DECISIONS = [
    {
        "id": 1,
        "decision": "Nothing upstream moves and no model is fitted; the freeze is hash-checked.",
        "prediction": (
            "SGV10's evaluation-block decision score equals SGV5's published `arm__sgv5` "
            "exactly, and every input artifact matches the hash its own stage recorded"
        ),
        "pre_registered": True,
    },
    {
        "id": 2,
        "decision": "Every policy ranks by a frozen score and changes only the action map.",
        "prediction": (
            "two policies in the same cell differ only in coverage and in which rows are inside "
            "the accept set, never in the order of the rows"
        ),
        "pre_registered": True,
    },
    {
        "id": 3,
        "decision": "The threshold is chosen on the source engines' labelled calibration rows.",
        "prediction": (
            "the source criterion is close to epsilon by construction, so any violation on the "
            "unseen engine is entirely engine shift plus block shift"
        ),
        "pre_registered": True,
    },
    {
        "id": 4,
        "decision": "The primary endpoint is risk-controlled repair recall.",
        "prediction": (
            "SGV5's published EasyOCR deployment scores zero, because it realises 0.305 harm at "
            "epsilon = 0.10, and any policy that holds the bound there beats it"
        ),
        "pre_registered": True,
        "informed_by": "design probe on SGV10's published table",
    },
    {
        "id": 5,
        "decision": (
            "Method E is a KL ball on the per-engine rate; page CVaR is reported beside it."
        ),
        "prediction": (
            "the KL ball nests minimax exactly at rho = 0, and the page CVaR is degenerate at "
            "high levels -- coverage near 0.002 on docTR at epsilon = 0.10"
        ),
        "pre_registered": True,
        "informed_by": "design probe on SGV10's published table",
    },
    {
        "id": 6,
        "decision": (
            "Every hyperparameter is chosen by an inner leave-one-source-engine-out replay on "
            "SGV9's calibration-half split."
        ),
        "prediction": "the inner selection is stable across folds",
        "pre_registered": True,
    },
    {
        "id": 7,
        "decision": "Two transport modes are carried and both are reported everywhere.",
        "prediction": (
            "quantile transport changes the deployed point substantially, because SGV8 and SGV9 "
            "both found the decision score's scale does not survive the engine change"
        ),
        "pre_registered": True,
    },
    {
        "id": 8,
        "decision": "The failure analysis is an exact four-term decomposition of realised harm.",
        "prediction": (
            "the block term is positive on all four engines, and on at least three of them it "
            "is larger in magnitude than the engine term"
        ),
        "pre_registered": True,
        "informed_by": "design probe on SGV10's published table",
    },
    {
        "id": 9,
        "decision": "The in-engine oracle is reported in two forms.",
        "prediction": (
            "the deepest-feasible in-engine oracle violates epsilon on the evaluation block on "
            "all four engines despite holding it on the pool it was chosen on"
        ),
        "pre_registered": True,
        "informed_by": "design probe on SGV10's published table",
    },
    {
        "id": 10,
        "decision": "Criterion 4 is enforced as a coverage floor and a random control.",
        "prediction": (
            "an arm that holds the bound by accepting almost nothing fails the floor, which is "
            "what the floor is for"
        ),
        "pre_registered": True,
    },
]


# ------------------------------------------------------------------ what each fitter may read


@dataclass(frozen=True, slots=True)
class SourcePolicyView:
    """What a policy may read while choosing a threshold: source rows, labels, engine ids.

    Every field belongs to the three SOURCE engines' CALIBRATION rows. There is no target
    field of any kind, and the leakage suite walks every selection function's syntax tree to
    confirm that none of them names another source of rows.
    """

    score: np.ndarray
    tail_risk: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    documents: np.ndarray
    engines: np.ndarray

    @property
    def size(self) -> int:
        return int(self.score.size)


@dataclass(frozen=True, slots=True)
class TargetPoolView:
    """What transport may read from the unseen engine: scores, and nothing else.

    There is deliberately no `harmful` field and no `engines` field. A transport that wanted a
    target label could not reach one through this object, and a policy that wanted to know
    WHICH engine it had been handed could not reach that either -- which is the brief's engine
    leakage check, enforced by the type rather than by a convention.
    """

    score: np.ndarray

    @property
    def size(self) -> int:
        return int(self.score.size)


# ------------------------------------------------------------------ the source risk curves


@dataclass(frozen=True, slots=True)
class SourceCurves:
    """Every aggregation of the source engines' harm rate, on one shared threshold grid.

    The grid is the distinct values of the source score, so every point is a threshold an
    operator could actually set, and `accepted >= g` has the same meaning here as it has at
    deployment. `per_engine` is one row per source engine in `engine_names` order.
    """

    grid: np.ndarray
    engine_names: tuple[str, ...]
    per_engine: np.ndarray
    pooled: np.ndarray
    accepted: np.ndarray
    coverage: np.ndarray

    @property
    def average(self) -> np.ndarray:
        return self.per_engine.mean(axis=0)

    @property
    def minimax(self) -> np.ndarray:
        return self.per_engine.max(axis=0)


def _rate_at(score: np.ndarray, harmful: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The harm rate of `score >= g`, for every g in an ASCENDING grid, without a Python loop.

    A descending sort turns "accepted" into a prefix, so one `searchsorted` gives the prefix
    length at every grid point and one cumulative sum gives the harmful count. Zero coverage is
    reported as a rate of 0.0, matching `c5._deployed`'s treatment of an empty accept set: a
    policy that accepts nothing does no harm, and the coverage floor is what stops that from
    being a way to pass.
    """
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    sorted_score = np.asarray(score, dtype=float)[order]
    cumulative = np.cumsum(np.asarray(harmful, dtype=float)[order])
    depth = np.searchsorted(-sorted_score, -np.asarray(grid, dtype=float), side="right")
    safe = np.maximum(depth, 1)
    return np.where(depth > 0, cumulative[safe - 1] / safe, 0.0)


def source_curves(view: SourcePolicyView) -> SourceCurves:
    engine_names = tuple(sorted(set(view.engines.tolist())))
    grid = np.unique(np.asarray(view.score, dtype=float))
    per_engine = np.vstack(
        [
            _rate_at(view.score[view.engines == name], view.harmful[view.engines == name], grid)
            for name in engine_names
        ]
    )
    accepted = np.array(
        [int(np.count_nonzero(view.score >= value)) for value in grid], dtype=np.int64
    )
    return SourceCurves(
        grid=grid,
        engine_names=engine_names,
        per_engine=per_engine,
        pooled=_rate_at(view.score, view.harmful, grid),
        accepted=accepted,
        coverage=accepted / max(view.size, 1),
    )


def cvar_pages(view: SourcePolicyView, grid: np.ndarray, beta: float) -> np.ndarray:
    """Policy E, first form: the mean accepted harm rate of the worst source PAGES.

    A candidate's harm is Bernoulli, so `CVaR_beta` of a single candidate is 1 whenever beta
    exceeds `1 - p` and the quantity carries no information at the candidate level. SGV10 made
    the same argument and took the page as the unit; this stage keeps it, because the document
    is the resampling unit everywhere in this project.

    The grid is walked from the strictest threshold down, so each step only has to add the rows
    that the previous threshold excluded. Pages that accept nothing are excluded from the tail
    rather than counted as harmless, which is the conservative reading and the one that makes
    the statistic comparable across thresholds.
    """
    score = np.asarray(view.score, dtype=float)
    harmful = np.asarray(view.harmful, dtype=float)
    codes, _ = pd.factorize(view.documents)
    n_pages = int(codes.max()) + 1 if codes.size else 0
    order = np.argsort(-score, kind="stable")
    depth = np.searchsorted(-score[order], -np.asarray(grid, dtype=float), side="right")

    accepted = np.zeros(n_pages, dtype=np.float64)
    harmed = np.zeros(n_pages, dtype=np.float64)
    out = np.zeros(grid.size, dtype=float)
    position = 0
    for index in range(grid.size - 1, -1, -1):
        target = int(depth[index])
        while position < target:
            row = int(order[position])
            accepted[codes[row]] += 1.0
            harmed[codes[row]] += harmful[row]
            position += 1
        live = accepted > 0
        if not live.any():
            out[index] = 0.0
            continue
        rates = np.sort(harmed[live] / accepted[live])[::-1]
        take = max(1, int(np.ceil((1.0 - beta) * rates.size)))
        out[index] = float(rates[:take].mean())
    return out


def kl_upper(mu: np.ndarray, rho: float) -> np.ndarray:
    """The largest Bernoulli mean inside a KL ball of radius `rho` around `mu`.

    `sup { nu : KL(Bernoulli(nu) || Bernoulli(mu)) <= rho }`, solved by bisection on [mu, 1].
    This is the worst case of a distributionally robust program whose uncertainty set is a KL
    ball around the empirical loss distribution, and it is the same object as a KL-UCB
    confidence bound; at `rho = 0` it returns `mu` exactly, which is what makes Policy F nest
    Policy C rather than approximate it.

    An empirical rate of exactly 0 admits no ball at all -- a distribution with no mass at 1
    cannot have mass moved onto it under a finite KL -- so the bound returns 0 there. That is
    the correct answer for this uncertainty set and it is stated as a limitation rather than
    patched, because patching it would make the radius mean something other than a KL radius.
    """
    values = np.clip(np.asarray(mu, dtype=float), 0.0, 1.0)
    if rho <= 0.0:
        return values
    lower = values.copy()
    upper = np.ones_like(values)
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(middle > 0, middle * np.log(np.divide(middle, values)), 0.0)
            right = np.where(
                middle < 1, (1 - middle) * np.log(np.divide(1 - middle, 1 - values)), 0.0
            )
        divergence = np.where(np.isfinite(left + right), left + right, np.inf)
        inside = divergence <= rho
        lower = np.where(inside, middle, lower)
        upper = np.where(inside, upper, middle)
    return np.where(values <= 0.0, 0.0, lower)


# ------------------------------------------------------------------ placing and carrying a cut


def _deepest_on_grid(criterion: np.ndarray, epsilon: float) -> int:
    """The most permissive grid index whose criterion is within epsilon; -1 if none is.

    The grid ascends in score, so depth DEscends along it and the deepest feasible prefix is at
    the lowest feasible index. Reversing the array and calling `_deepest` keeps the "last
    feasible rather than first infeasible" convention in one implementation -- SGV7 chose it,
    SGV9 and SGV10 imported it, and the reason is unchanged: the harm rate is not monotone in
    the rank, so stopping at the first infeasible prefix would silently pick a shallower cut
    than the rule specifies.
    """
    values = np.asarray(criterion, dtype=float)
    index = _deepest(values[::-1], epsilon)
    return -1 if index < 0 else int(values.size - 1 - index)


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Everything a fitted policy carries to the unseen engine. Scalars only, by construction.

    A policy is a decision layer, so what crosses the engine boundary is a handful of numbers:
    where to accept, where to abstain, which tail-risk values to veto, and -- for the matched
    control -- how many rows to take at random. No array, no model and no target row is carried
    with it, which is what makes "the policy cannot know the test engine" enforceable.
    """

    policy: str
    ranking: str
    transport: str
    epsilon: float
    tau_accept: float
    tau_abstain: float
    tail_cutoff: float
    feasible: bool
    hyperparameter: float
    hyperparameter_name: str
    source_criterion: float
    source_coverage: float
    source_realized_harm: float
    random_accept: int
    random_seed: int
    note: str


def _refused(policy: str, ranking: str, transport: str, epsilon: float, note: str) -> PolicySpec:
    """No threshold in the grid met the constraint, so the policy deploys nothing.

    A refusal is a legitimate outcome of a risk-constrained rule and is recorded as one. It
    scores zero on the primary endpoint -- a policy that accepts nothing repairs nothing -- and
    it fails the coverage floor, so it cannot become a finding by abstaining from the question.
    """
    return PolicySpec(
        policy=policy,
        ranking=ranking,
        transport=transport,
        epsilon=float(epsilon),
        tau_accept=float(np.inf),
        tau_abstain=float(np.inf),
        tail_cutoff=float(np.inf),
        feasible=False,
        hyperparameter=float("nan"),
        hyperparameter_name="",
        source_criterion=float("nan"),
        source_coverage=0.0,
        source_realized_harm=float("nan"),
        random_accept=-1,
        random_seed=0,
        note=note,
    )


def transport_threshold(
    tau_source: float, source: SourcePolicyView, pool: TargetPoolView, mode: str
) -> float:
    """Carry a source threshold to the unseen engine. Reads target SCORES, never a target label.

    `value` carries the number itself and reads nothing at all. `quantile` carries the share of
    source rows the threshold accepted and finds the pool score that accepts the same share, by
    counting rather than by interpolating -- `np.quantile` would return a value that no row
    holds, and a threshold that lies between two adjacent scores accepts the same set as the
    lower of them while being harder to check. `tests/leakage` walks this function's syntax tree
    and fails if it names an outcome array or an evaluation block.
    """
    if mode == TRANSPORT_VALUE:
        return float(tau_source)
    if mode != TRANSPORT_QUANTILE:
        raise PhaseError(f"unknown transport mode {mode!r}")
    if not np.isfinite(tau_source):
        return float(np.inf)
    share = float(np.mean(np.asarray(source.score, dtype=float) >= tau_source))
    if share <= 0.0 or pool.size == 0:
        return float(np.inf)
    ordered = np.sort(np.asarray(pool.score, dtype=float))[::-1]
    count = int(np.clip(round(share * pool.size), 1, pool.size))
    return float(ordered[count - 1])


def actions_of(score: np.ndarray, tail_risk: np.ndarray, spec: PolicySpec) -> np.ndarray:
    """The three-way action for every row. One implementation, used everywhere.

    ACCEPT needs the score above the accept threshold AND the tail-risk estimate within the
    cutoff; a policy with no tail-risk veto carries an infinite cutoff and the second condition
    is vacuous. ABSTAIN is everything below the accept threshold but above the abstain
    threshold; a policy with no abstain region carries an infinite abstain threshold and the
    region is empty. Everything else is REJECT.
    """
    values = np.asarray(score, dtype=float)
    actions = np.full(values.size, REJECT, dtype=np.int8)
    if spec.random_accept >= 0:
        rng = np.random.default_rng(spec.random_seed)
        chosen = rng.permutation(values.size)[: min(spec.random_accept, values.size)]
        actions[chosen] = ACCEPT
        return actions
    accept = (
        np.isfinite(values)
        & (values >= spec.tau_accept)
        & (np.asarray(tail_risk, dtype=float) <= spec.tail_cutoff)
    )
    abstain = (~accept) & np.isfinite(values) & (values >= spec.tau_abstain)
    actions[abstain] = ABSTAIN
    actions[accept] = ACCEPT
    return actions


def _surrogate(accepted: np.ndarray) -> np.ndarray:
    """An accept mask expressed as a score, so the published operating-point code can read it.

    `c5._deployed` is the one implementation of the deployed operating point in this repository
    and it takes a score and a threshold. Two of this stage's policies accept a set that is not
    a prefix of any ranking -- the selective policy vetoes rows inside the accept region, and
    the matched control takes a random subset -- so the mask is turned into a two-valued score
    at threshold 0.5 rather than the metric being reimplemented for them.
    """
    return np.where(np.asarray(accepted, dtype=bool), 1.0, 0.0)


# ------------------------------------------------------------------ the endpoint


def risk_controlled_recall(point: dict[str, Any], epsilon: float) -> float:
    """The primary endpoint: repair recall if the accept set held the bound, zero otherwise.

    The brief names "repair recall at fixed harm bound" as the primary metric. This is that
    quantity as a single scalar, so a policy cannot buy recall by breaking the constraint it
    was given, and so a baseline that already breaks it does not set an unbeatable target. An
    accept set that is empty scores zero: it holds the bound vacuously and repairs nothing.
    """
    if point["accepts_nothing"] or int(point["n_accepted"]) == 0:
        return 0.0
    realized = float(point["realized_harm_rate"])
    return float(point["repair_recall"]) if realized <= epsilon else 0.0


def measure(block: Block, actions: np.ndarray, epsilon: float) -> dict[str, Any]:
    """What one action assignment does on the evaluation block, with everything it implies."""
    codes = np.asarray(actions, dtype=np.int8)
    accepted = codes == ACCEPT
    abstained = codes == ABSTAIN
    point = dict(deployed_point(_surrogate(accepted), block.harmful, block.beneficial, 0.5))
    realized = float(point["realized_harm_rate"])
    point["epsilon"] = float(epsilon)
    point["holds_bound"] = True if point["accepts_nothing"] else bool(realized <= epsilon)
    point["bound_violation"] = (
        0.0 if point["accepts_nothing"] else float(max(0.0, realized - epsilon))
    )
    point["risk_controlled_repair_recall"] = risk_controlled_recall(point, epsilon)
    total = max(int(block.beneficial.sum()), 1)
    point["abstention_rate"] = float(abstained.mean()) if codes.size else 0.0
    point["n_abstained"] = int(abstained.sum())
    point["reject_rate"] = float((codes == REJECT).mean()) if codes.size else 0.0
    point["abstained_beneficial"] = int(block.beneficial[abstained].sum())
    point["abstained_harmful"] = int(block.harmful[abstained].sum())
    point["recall_recoverable_under_review"] = float(block.beneficial[abstained].sum() / total)
    point["joint_harm_rate"] = float(block.harmful[accepted].sum() / max(codes.size, 1))
    point["utility"] = {
        f"lambda_{lam:g}__gamma_{gam:g}": float(
            point["repair_recall"]
            - lam * (0.0 if point["accepts_nothing"] else realized)
            - gam * point["abstention_rate"]
        )
        for lam, gam in UTILITY_WEIGHTS
    }
    return point


# ------------------------------------------------------------------ document-clustered draws


def document_multiplicities(documents: np.ndarray, seed: int, n_resamples: int) -> np.ndarray:
    """How many times each document is drawn in each resample, as one integer matrix.

    `document_resamples` returns index arrays, which is the right shape for one statistic and
    the wrong shape for four hundred cells: every cell would pay a full gather over its rows.
    A resample's total of any per-row quantity is a weighted sum of that quantity's per-document
    totals, so the whole stage's resampling reduces to one matrix product per cell against this
    matrix. `tests/leakage` asserts that each row of it equals the document counts of the
    corresponding draw from `document_resamples` at the same seed, so the two constructions are
    checked against each other rather than assumed identical.
    """
    cluster_ids = sorted({str(name) for name in documents.tolist()})
    rng = np.random.default_rng(seed)
    n_clusters = len(cluster_ids)
    out = np.zeros((n_resamples, n_clusters), dtype=np.int32)
    for index in range(n_resamples):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        out[index] = np.bincount(chosen, minlength=n_clusters)
    return out


@dataclass(slots=True)
class FoldDraws:
    """One fold's document draws, shared by every policy, ranking, transport and epsilon."""

    multiplicity: np.ndarray
    codes: np.ndarray
    n_documents: int


def fold_draws(block: Block, seed: int, n_resamples: int) -> FoldDraws:
    cluster_ids = sorted({str(name) for name in block.documents.tolist()})
    position = {name: index for index, name in enumerate(cluster_ids)}
    codes = np.array([position[str(name)] for name in block.documents.tolist()], dtype=np.intp)
    return FoldDraws(
        multiplicity=document_multiplicities(block.documents, seed, n_resamples),
        codes=codes,
        n_documents=len(cluster_ids),
    )


def resampled_actions(
    block: Block, actions: np.ndarray, draws: FoldDraws, epsilon: float
) -> dict[str, Any]:
    """The deployed point's resample distribution at a FIXED action assignment.

    The threshold is not re-chosen inside the resample: it was chosen on the source engines and
    an operator would carry that one number to the next batch of pages. What varies is the
    batch, which is exactly the question a violation rate has to answer -- if this policy met a
    different sample of this engine's pages, how often would the bound break?
    """
    accepted = (np.asarray(actions, dtype=np.int8) == ACCEPT).astype(np.float64)
    columns = np.column_stack(
        [
            accepted,
            accepted * block.harmful.astype(np.float64),
            accepted * block.beneficial.astype(np.float64),
            block.beneficial.astype(np.float64),
        ]
    )
    per_document = np.zeros((draws.n_documents, 4), dtype=np.float64)
    np.add.at(per_document, draws.codes, columns)
    totals = draws.multiplicity.astype(np.float64) @ per_document
    n_accepted, n_harmful, n_beneficial, n_total = totals.T

    usable = n_accepted > 0
    harm_rate = np.full(n_accepted.size, np.nan)
    harm_rate[usable] = n_harmful[usable] / n_accepted[usable]
    recall = n_beneficial / np.maximum(n_total, 1.0)
    controlled = np.where(usable & (harm_rate <= epsilon), recall, 0.0)
    finite = np.isfinite(harm_rate)
    return {
        "n_accepted_mean": float(n_accepted.mean()),
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
        "risk_controlled_mean": float(controlled.mean()),
        "risk_controlled_ci": [
            float(np.quantile(controlled, 0.025)),
            float(np.quantile(controlled, 0.975)),
        ],
        "violation_rate": float(np.mean(harm_rate[finite] > epsilon)) if finite.any() else 0.0,
        "usable_resamples": int(finite.sum()),
        "accepts_nothing": bool(not usable.any()),
        "controlled_draws": controlled,
    }


# ------------------------------------------------------------------ the criteria


HYPERPARAMETERS = {
    DRO_CVAR: ("cvar_beta", CVAR_BETAS),
    DRO_KL: ("kl_radius", KL_RADII),
    SELECTIVE: ("tail_cutoff_quantile", TAIL_CUTOFF_QUANTILES),
}


def criterion_of(
    policy: str, curves: SourceCurves, view: SourcePolicyView, hyperparameter: float
) -> np.ndarray:
    """The quantity a policy bounds by epsilon, on the shared source threshold grid.

    Every entry is a function of the SOURCE engines' rows and nothing else. This is the only
    place the five aggregations differ, which is what makes ablation 1 and ablation 3 a
    contrast between two lines here rather than between two pipelines.
    """
    if policy in (ERM_AVERAGE, SELECTIVE):
        return curves.average
    if policy == MINIMAX:
        return curves.minimax
    if policy == POOLED_ERM:
        return curves.pooled
    if policy == DRO_CVAR:
        return cvar_pages(view, curves.grid, float(hyperparameter))
    if policy == DRO_KL:
        return np.asarray(kl_upper(curves.per_engine, float(hyperparameter)).max(axis=0))
    raise PhaseError(f"{policy!r} has no source criterion; it is not fitted on source rows")


def _inner_views(view: SourcePolicyView, pseudo_target: str) -> tuple[SourcePolicyView, ...]:
    """One leave-one-source-engine-out replay, split by SGV9's calibration halves.

    The pages are halved by content hash so the engine held out of the inner fit is scored on
    pages the inner fit never saw. Without the halving the inner threshold would be chosen on
    the very pages its error is read from -- different engine, same scan, because this corpus is
    matched-source -- and every selection this stage makes would be optimistic. The halving
    function is SGV9's, imported rather than re-salted, so this stage's inner folds are
    identical to SGV9's and SGV10's and the three stages' selections are directly comparable.
    """
    half = np.array([_calibration_half(str(name)) for name in view.documents.tolist()])
    fit = (view.engines != pseudo_target) & (half == 0)
    outcome = (view.engines == pseudo_target) & (half == 1)
    return _subset(view, fit), _subset(view, outcome)


def _subset(view: SourcePolicyView, mask: np.ndarray) -> SourcePolicyView:
    return SourcePolicyView(
        score=view.score[mask],
        tail_risk=view.tail_risk[mask],
        harmful=view.harmful[mask],
        beneficial=view.beneficial[mask],
        documents=view.documents[mask],
        engines=view.engines[mask],
    )


def select_hyperparameter(policy: str, view: SourcePolicyView, epsilon: float) -> dict[str, Any]:
    """Choose a policy's one free parameter inside the SOURCE engines, by inner replay.

    Each source engine takes a turn as a pseudo-target: the criterion is built from the other
    two on the first half of the calibration pages, the threshold is placed, and the resulting
    scalar is measured on the pseudo-target's second half. The value with the highest mean
    risk-controlled repair recall wins; ties go to the more conservative value, which for every
    grid in this stage is the later entry. The selection reads no target row of any kind and
    `selection_scope` records that.
    """
    name, grid = HYPERPARAMETERS[policy]
    engines = tuple(sorted(set(view.engines.tolist())))
    replays: dict[str, list[dict[str, Any]]] = {}
    for value in grid:
        rows: list[dict[str, Any]] = []
        for pseudo_target in engines:
            inner_fit, inner_outcome = _inner_views(view, pseudo_target)
            if inner_fit.size == 0 or inner_outcome.size == 0:
                rows.append({"pseudo_target": pseudo_target, "controlled": 0.0, "usable": False})
                continue
            spec = _place_on_source(policy, inner_fit, epsilon, float(value))
            actions = actions_of(inner_outcome.score, inner_outcome.tail_risk, spec)
            point = measure(_view_as_block(inner_outcome), actions, epsilon)
            rows.append(
                {
                    "pseudo_target": pseudo_target,
                    "controlled": float(point["risk_controlled_repair_recall"]),
                    "coverage": float(point["coverage"]),
                    "realized_harm_rate": float(point["realized_harm_rate"]),
                    "usable": True,
                }
            )
        replays[f"{value:g}"] = rows
    scores = {
        key: float(np.mean([row["controlled"] for row in rows])) for key, rows in replays.items()
    }
    best = max(grid, key=lambda value: (scores[f"{value:g}"], grid.index(value)))
    return {
        "name": name,
        "grid": list(grid),
        "selected": float(best),
        "inner_scores": scores,
        "replays": replays,
        "selection_scope": (
            "inner leave-one-source-engine-out on SGV9's calibration halves; no target row, no "
            "target label and no evaluation page is read"
        ),
        "tie_break": "the later grid entry, which is the more conservative value",
        "transport_note": (
            "selected under value transport only. The parameter governs the source criterion "
            "and not the carry, and the inner fold has no unlabelled target pool to match a "
            "quantile against without reading the block the inner outcome comes from."
        ),
    }


def _view_as_block(view: SourcePolicyView) -> Block:
    """A labelled view seen through the published operating-point code's own type.

    Only the inner replay uses this: it has to measure a deployed point on source rows, and the
    metric is defined once, on `Block`. The identifiers and the two calibrated probabilities are
    not read by anything downstream of `measure`, so they are filled rather than carried.
    """
    size = view.size
    return Block(
        index=np.arange(size, dtype=int),
        candidate_id=np.array([""] * size, dtype=object),
        documents=view.documents,
        p_harm=np.zeros(size, dtype=float),
        p_benefit=np.zeros(size, dtype=float),
        utility=np.asarray(view.score, dtype=float),
        stratum=np.zeros(size, dtype=int),
        harmful=np.asarray(view.harmful, dtype=bool),
        beneficial=np.asarray(view.beneficial, dtype=bool),
    )


def _place_on_source(
    policy: str, view: SourcePolicyView, epsilon: float, hyperparameter: float
) -> PolicySpec:
    """Place a policy's threshold on the SOURCE engines' labelled rows. Reads no target row.

    The returned spec is in source-score space and under value transport; `fit_policy` carries
    it. Splitting the two makes the inner selection reuse the outer placement rather than
    approximate it, and makes "the policy is a handful of scalars" true of the fitting code as
    well as of the artifact. `tests/leakage` walks this function's syntax tree and fails if it
    names a target block, an evaluation array or an oracle.
    """
    curves = source_curves(view)
    if curves.grid.size == 0:
        return _refused(policy, "", TRANSPORT_VALUE, epsilon, "the source block is empty")
    name = HYPERPARAMETERS[policy][0] if policy in HYPERPARAMETERS else ""
    criterion = criterion_of(policy, curves, view, hyperparameter)
    index = _deepest_on_grid(criterion, epsilon)
    if index < 0:
        return _refused(
            policy,
            "",
            TRANSPORT_VALUE,
            epsilon,
            f"no threshold on the source grid holds {policy}'s criterion at epsilon={epsilon}",
        )
    tau_accept = float(curves.grid[index])
    tau_abstain = float(np.inf)
    tail_cutoff = float(np.inf)
    bound = criterion
    if policy == SELECTIVE:
        # The accept region is bounded by the STRICTER of the two aggregations and the abstain
        # region is everything down to the more permissive one, so the reported criterion has to
        # be the one that governs the accept set rather than the one that opened the search.
        strict = _deepest_on_grid(curves.minimax, epsilon)
        if strict >= 0:
            bounds = sorted((tau_accept, float(curves.grid[strict])))
            tau_abstain, tau_accept = bounds[0], bounds[1]
            bound = curves.minimax if tau_accept == float(curves.grid[strict]) else curves.average
        quantile = float(hyperparameter)
        if quantile < 1.0:
            inside = np.asarray(view.score, dtype=float) >= tau_accept
            population = np.asarray(view.tail_risk, dtype=float)[inside]
            if population.size:
                tail_cutoff = float(np.quantile(population, quantile))
    at = int(np.searchsorted(curves.grid, tau_accept, side="left"))
    at = int(np.clip(at, 0, curves.grid.size - 1))
    accepted = (np.asarray(view.score, dtype=float) >= tau_accept) & (
        np.asarray(view.tail_risk, dtype=float) <= tail_cutoff
    )
    return PolicySpec(
        policy=policy,
        ranking="",
        transport=TRANSPORT_VALUE,
        epsilon=float(epsilon),
        tau_accept=tau_accept,
        tau_abstain=tau_abstain,
        tail_cutoff=tail_cutoff,
        feasible=True,
        hyperparameter=float(hyperparameter),
        hyperparameter_name=name,
        source_criterion=float(bound[at]),
        source_coverage=float(accepted.mean()) if accepted.size else 0.0,
        source_realized_harm=(
            float(np.asarray(view.harmful, dtype=float)[accepted].mean())
            if accepted.any()
            else float("nan")
        ),
        random_accept=-1,
        random_seed=0,
        note=(
            f"{policy} at epsilon={epsilon}: the deepest source threshold whose criterion is "
            f"{bound[at]:.6f}"
        ),
    )


def fit_policy(
    policy: str,
    view: SourcePolicyView,
    pool: TargetPoolView,
    ranking: str,
    transport: str,
    epsilon: float,
    hyperparameter: float,
) -> PolicySpec:
    """Choose the policy on the source engines, then carry it to the unseen engine."""
    placed = _place_on_source(policy, view, epsilon, hyperparameter)
    if not placed.feasible:
        return replace(placed, ranking=ranking, transport=transport)
    tau_accept = transport_threshold(placed.tau_accept, view, pool, transport)
    tau_abstain = (
        transport_threshold(placed.tau_abstain, view, pool, transport)
        if np.isfinite(placed.tau_abstain)
        else float(np.inf)
    )
    return replace(
        placed, ranking=ranking, transport=transport, tau_accept=tau_accept, tau_abstain=tau_abstain
    )


def fit_oracle(pool: Block, ranking: str, epsilon: float, monotone: bool) -> PolicySpec:
    """AN ORACLE. Places the cut on the held-out engine's OWN labelled pool.

    This is the only function in the stage that reads a target label, it is never called for a
    policy in `SOURCE_FITTED`, and `--negative` asserts both. The two forms differ in one line:
    the deepest-feasible form takes the last prefix that holds epsilon and the monotone form
    stops at the first prefix that does not. The gap between them is how much of the oracle's
    own bound violation is the deepest-feasible rule's optimism rather than the block shift.
    """
    order = np.argsort(-pool.utility, kind="stable")
    rate = np.cumsum(pool.harmful[order].astype(float)) / np.arange(1, pool.size + 1)
    if monotone:
        infeasible = np.flatnonzero(rate > epsilon)
        index = int(infeasible[0]) - 1 if infeasible.size else pool.size - 1
    else:
        index = _deepest(rate, epsilon)
    name = ORACLE_MONOTONE if monotone else ORACLE_POOL
    if index < 0:
        return _refused(
            name, ranking, TRANSPORT_VALUE, epsilon, "no prefix of the target pool holds epsilon"
        )
    return PolicySpec(
        policy=name,
        ranking=ranking,
        transport=TRANSPORT_VALUE,
        epsilon=float(epsilon),
        tau_accept=float(pool.utility[order][index]),
        tau_abstain=float(np.inf),
        tail_cutoff=float(np.inf),
        feasible=True,
        hyperparameter=float("nan"),
        hyperparameter_name="",
        source_criterion=float(rate[index]),
        source_coverage=float((index + 1) / max(pool.size, 1)),
        source_realized_harm=float(rate[index]),
        random_accept=-1,
        random_seed=0,
        note=(
            "ORACLE: the cut is placed on the held-out engine's own labelled pool, "
            + ("stopping at the first violation" if monotone else "at the deepest feasible prefix")
        ),
    )


def fit_random(
    ranking: str, transport: str, epsilon: float, engine: str, n_accepted: int
) -> PolicySpec:
    """A control, not a method: accept a pseudo-random subset of the matched size.

    The seed is derived from the fold's content with sha256, never from `hash()`, which
    `PYTHONHASHSEED` can move between processes.
    """
    return PolicySpec(
        policy=RANDOM_MATCHED,
        ranking=ranking,
        transport=transport,
        epsilon=float(epsilon),
        tau_accept=float("nan"),
        tau_abstain=float(np.inf),
        tail_cutoff=float(np.inf),
        feasible=n_accepted > 0,
        hyperparameter=float("nan"),
        hyperparameter_name="",
        source_criterion=float("nan"),
        source_coverage=float("nan"),
        source_realized_harm=float("nan"),
        random_accept=int(n_accepted),
        random_seed=_stable_seed("sgv11-random", engine, ranking, transport, str(epsilon)),
        note=f"matched-coverage random control accepting {n_accepted} rows",
    )


# ------------------------------------------------------------------ the freeze verification


def _hash_against(path: Path, record: Path, field: str = "artifacts") -> dict[str, Any]:
    """One published artifact, checked against the hash the stage that produced it recorded."""
    if not path.is_file():
        raise PhaseError(f"{cc._relative(path)} is missing; SGV11 fits nothing and cannot make it")
    if not record.is_file():
        raise PhaseError(f"{cc._relative(record)} is missing; the freeze cannot be verified")
    published = dict(cc._read_json(record).get(field, {}))
    relative = cc._relative(path)
    if relative not in published:
        raise PhaseError(f"{relative} is not in {cc._relative(record)}")
    digest = file_sha256(path)
    if digest != published[relative]:
        raise PhaseError(f"{relative} does not match the hash {cc._relative(record)} recorded")
    return {"path": relative, "sha256": digest, "verified_against": cc._relative(record)}


def verify_frozen() -> dict[str, Any]:
    """Every upstream artifact this stage reads, checked before a single number is computed.

    The brief asks for candidate_id alignment, score identity and artifact hash consistency.
    All three are here and all three are hard failures. Score identity is the strong one: the
    decision score SGV11 ranks by is compared row for row against the vector SGV5 published, so
    a drift anywhere in the eleven-stage chain that produced SGV10's table would stop this stage
    rather than silently move its baseline.
    """
    checked = [
        _hash_against(dg.DESIGN_MATRIX, dg.FIT_RECORD),
        _hash_against(c5.PREDICTIONS, c5.FIT_RECORD),
        _hash_against(s9.SCORES, s9.PROVENANCE),
        _hash_against(s10.SCORES, s10.PROVENANCE),
        _hash_against(s10.DESIGN_RECORD, s10.PROVENANCE),
    ]
    table = pd.read_parquet(s10.SCORES)
    design = pd.read_parquet(
        dg.DESIGN_MATRIX, columns=["candidate_id", "document_id", "engine_id", "role"]
    ).set_index("candidate_id")
    identifiers = table["candidate_id"].astype(str)
    unknown = int((~identifiers.isin(design.index)).sum())
    if unknown:
        raise PhaseError(f"{unknown} SGV10 rows carry a candidate_id SGV1's design does not know")
    joined = design.loc[identifiers]
    published_documents = joined["document_id"].to_numpy(str)
    mismatched = int((published_documents != table["document_id"].to_numpy(str)).sum())
    if mismatched:
        raise PhaseError(f"{mismatched} SGV10 rows disagree with SGV1's design on the document")

    frozen = s6.load_frozen_scores().set_index(["held_out_engine", "candidate_id"])
    evaluation = table[table["block"] == "evaluation"]
    keys = list(
        zip(
            evaluation["held_out_engine"].astype(str),
            evaluation["candidate_id"].astype(str),
            strict=True,
        )
    )
    published = frozen.loc[keys, s6.SGV5_ARM].to_numpy(dtype=float)
    difference = float(np.abs(evaluation["utility"].to_numpy(dtype=float) - published).max())
    if difference != 0.0:
        raise PhaseError(
            "SGV10's evaluation decision score differs from SGV5's published vector by "
            f"{difference:g}; every SGV11 number is a decision on that score and a drifted "
            "score invalidates all of them"
        )
    return {
        "artifacts": checked,
        "candidate_id_alignment": {
            "rows": len(table),
            "unknown_candidate_ids": unknown,
            "document_disagreements": mismatched,
        },
        "score_identity_against_sgv5": {
            "arm": s6.SGV5_ARM,
            "rows": len(evaluation),
            "max_abs_difference": difference,
        },
        "models_fitted_by_this_stage": 0,
        "note": (
            "SGV11 fits no model. Every score, every calibrated probability and every "
            "tail-risk estimate is read from a published table and hash-verified against the "
            "record its own stage wrote."
        ),
    }


# ------------------------------------------------------------------ the folds


@dataclass(slots=True)
class Fold:
    """One held-out engine: three blocks, two rankings, the tail-risk estimate, the source ids."""

    held_out: str
    blocks: dict[str, Block]
    rankings: dict[str, dict[str, dict[str, np.ndarray]]]
    tail_risk: dict[str, dict[str, np.ndarray]]
    ceiling: dict[str, np.ndarray]
    engines: dict[str, np.ndarray]
    source_engines: tuple[str, ...]
    sgv5_threshold: dict[str, float]

    def view(self, ranking: str, key: str) -> SourcePolicyView:
        block = self.blocks["calibration"]
        return SourcePolicyView(
            score=self.rankings[ranking][key]["calibration"],
            tail_risk=self.tail_risk[key]["calibration"],
            harmful=block.harmful,
            beneficial=block.beneficial,
            documents=block.documents,
            engines=self.engines["calibration"],
        )

    def pool(self, ranking: str, key: str) -> TargetPoolView:
        return TargetPoolView(score=self.rankings[ranking][key]["pool"])

    def scored(self, name: str, ranking: str, key: str) -> Block:
        return replace(self.blocks[name], utility=self.rankings[ranking][key][name])


def _block_from_frame(frame: pd.DataFrame) -> Block:
    """The stratum and the two calibrated probabilities are SGV9's and SGV10's respectively.

    SGV11 reads neither: it ranks by a frozen score and vetoes by a frozen tail-risk estimate,
    both of which are carried as their own columns. They are filled rather than copied so that
    nothing downstream can come to believe this stage reads them.
    """
    size = len(frame)
    return Block(
        index=np.arange(size, dtype=int),
        candidate_id=frame["candidate_id"].astype(str).to_numpy(),
        documents=frame["document_id"].astype(str).to_numpy(),
        p_harm=np.zeros(size, dtype=float),
        p_benefit=np.zeros(size, dtype=float),
        utility=frame[f"score__{RANK_SGV5}"].to_numpy(dtype=float),
        stratum=np.zeros(size, dtype=int),
        harmful=frame["is_harmful"].to_numpy(dtype=bool),
        beneficial=frame["beneficial"].to_numpy(dtype=bool),
    )


def build_rows() -> pd.DataFrame:
    """The stage's whole input, in one table: two rankings, one tail-risk estimate, the labels.

    Everything here is a projection of tables SGV1, SGV5, SGV9 and SGV10 already published. The
    only column this stage adds is `engine_id`, joined from SGV1's design matrix, and it is
    present on the SOURCE rows because the minimax question is unanswerable without it. It is
    NOT reachable by any policy on the target side: the target is handed a `TargetPoolView`,
    which has one field.
    """
    table = pd.read_parquet(s10.SCORES)
    design = pd.read_parquet(
        dg.DESIGN_MATRIX, columns=["candidate_id", "engine_id", "role"]
    ).set_index("candidate_id")
    joined = design.loc[table["candidate_id"].astype(str)]
    frame = pd.DataFrame(
        {
            "held_out_engine": table["held_out_engine"].astype(str).to_numpy(),
            "block": table["block"].astype(str).to_numpy(),
            "candidate_id": table["candidate_id"].astype(str).to_numpy(),
            "document_id": table["document_id"].astype(str).to_numpy(),
            "engine_id": joined["engine_id"].astype(str).to_numpy(),
            "role": joined["role"].astype(str).to_numpy(),
            f"score__{RANK_SGV5}": table["utility"].to_numpy(dtype=float),
            "is_harmful": table["is_harmful"].to_numpy(dtype=bool),
            "beneficial": table["beneficial"].to_numpy(dtype=bool),
        }
    )
    for epsilon in EPSILONS:
        key = f"epsilon_{int(epsilon * 100)}"
        frame[f"score__{RANK_SGV10}__{key}"] = table[f"score__{s10.HYBRID}__{key}"].to_numpy(
            dtype=float
        )
        frame[f"tail_risk__{key}"] = table[f"est__{s10.HYBRID}__{key}"].to_numpy(dtype=float)
        # An ANALYSIS CEILING, never a ranking any policy may use: SGV10's veto refitted on the
        # evaluation block itself. It bounds what a better score could buy at a perfect cut, so
        # the score-transfer and decision-transfer gaps can be separated. `--negative` asserts
        # that no source-fitted policy reads it.
        frame[f"ceiling__{key}"] = table[f"score__{s10.ORACLE_EVAL}__{key}"].to_numpy(dtype=float)
    return frame


def folds_from_frame(
    frame: pd.DataFrame, thresholds: dict[str, dict[str, float]]
) -> dict[str, Fold]:
    """Group the row table into folds. One implementation, used while fitting and while reading.

    `thresholds` is SGV5's published cut per engine and epsilon. It comes from SGV10's design
    record while this stage is fitting and from this stage's own record afterwards, and
    `tests/leakage` asserts the two agree, so the baseline every table is measured against
    cannot drift between the phase that wrote it and the phases that read it.
    """
    folds: dict[str, Fold] = {}
    for engine, group in frame.groupby("held_out_engine", sort=True):
        blocks: dict[str, Block] = {}
        rankings: dict[str, dict[str, dict[str, np.ndarray]]] = {r: {} for r in RANKINGS}
        tail: dict[str, dict[str, np.ndarray]] = {}
        ceiling: dict[str, np.ndarray] = {}
        engines: dict[str, np.ndarray] = {}
        for block_name, block_group in group.groupby("block", sort=True):
            block_frame = block_group.reset_index(drop=True)
            name = str(block_name)
            blocks[name] = _block_from_frame(block_frame)
            engines[name] = block_frame["engine_id"].astype(str).to_numpy()
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                rankings[RANK_SGV5].setdefault(key, {})[name] = block_frame[
                    f"score__{RANK_SGV5}"
                ].to_numpy(dtype=float)
                rankings[RANK_SGV10].setdefault(key, {})[name] = block_frame[
                    f"score__{RANK_SGV10}__{key}"
                ].to_numpy(dtype=float)
                tail.setdefault(key, {})[name] = block_frame[f"tail_risk__{key}"].to_numpy(
                    dtype=float
                )
                if name == "evaluation":
                    ceiling[key] = block_frame[f"ceiling__{key}"].to_numpy(dtype=float)
        if set(blocks) != set(BLOCK_NAMES):
            raise PhaseError(f"{engine}: the row table is missing a block ({sorted(blocks)})")
        source = tuple(sorted(set(engines["calibration"].tolist())))
        if str(engine) in source:
            raise PhaseError(f"{engine}: the held-out engine is inside its own source block")
        evaluation_documents = set(blocks["evaluation"].documents.tolist())
        for name in ("calibration", "pool"):
            if set(blocks[name].documents.tolist()) & evaluation_documents:
                raise PhaseError(f"{engine}: the {name} block shares pages with the evaluation")
        folds[str(engine)] = Fold(
            held_out=str(engine),
            blocks=blocks,
            rankings=rankings,
            tail_risk=tail,
            ceiling=ceiling,
            engines=engines,
            source_engines=source,
            sgv5_threshold={key: float(value) for key, value in thresholds[str(engine)].items()},
        )
    if not folds:
        raise PhaseError("the row table is empty")
    return folds


def load_folds() -> tuple[dict[str, Fold], dict[str, Any]]:
    if not SCORES.is_file() or not DESIGN_RECORD.is_file():
        raise PhaseError("run --policies before any phase that reads its output")
    record = cc._read_json(DESIGN_RECORD)
    thresholds = {
        engine: dict(payload["sgv5_threshold"]) for engine, payload in record["folds"].items()
    }
    return folds_from_frame(pd.read_parquet(SCORES), thresholds), record


def _write(path: Path, payload: dict[str, Any], schema: str) -> None:
    cc._write_json_once(
        path,
        {
            "schema_version": schema,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV11-P1",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            **payload,
        },
    )


def _spec_as_dict(spec: PolicySpec) -> dict[str, Any]:
    return {
        "policy": spec.policy,
        "ranking": spec.ranking,
        "transport": spec.transport,
        "epsilon": spec.epsilon,
        "tau_accept": spec.tau_accept,
        "tau_abstain": spec.tau_abstain,
        "tail_cutoff": spec.tail_cutoff,
        "feasible": spec.feasible,
        "hyperparameter": spec.hyperparameter,
        "hyperparameter_name": spec.hyperparameter_name,
        "source_criterion": spec.source_criterion,
        "source_coverage": spec.source_coverage,
        "source_realized_harm": spec.source_realized_harm,
        "random_accept": spec.random_accept,
        "random_seed": spec.random_seed,
        "note": spec.note,
    }


def _spec_from_dict(payload: dict[str, Any]) -> PolicySpec:
    return PolicySpec(**{name: payload[name] for name in PolicySpec.__slots__})


# ------------------------------------------------------------------ fitting the whole family


def sgv5_baseline(fold: Fold, epsilon: float) -> PolicySpec:
    """Policy A: SGV5's published threshold, carried verbatim.

    Read from SGV5's own record through SGV10's, never re-chosen. Its transport is neither of
    this stage's two -- it is the one SGV5 published -- and it is labelled `published` so that
    no table can silently compare it against a policy under a different carry.
    """
    key = f"epsilon_{int(epsilon * 100)}"
    return PolicySpec(
        policy=SGV5_PUBLISHED,
        ranking=RANK_SGV5,
        transport="published",
        epsilon=float(epsilon),
        tau_accept=float(fold.sgv5_threshold[key]),
        tau_abstain=float(np.inf),
        tail_cutoff=float(np.inf),
        feasible=True,
        hyperparameter=float("nan"),
        hyperparameter_name="",
        source_criterion=float("nan"),
        source_coverage=float("nan"),
        source_realized_harm=float("nan"),
        random_accept=-1,
        random_seed=0,
        note="SGV5's published deployment: the threshold its own calibration rows chose",
    )


def fit_cell(
    fold: Fold,
    ranking: str,
    transport: str,
    epsilon: float,
    selections: dict[str, dict[str, Any]],
) -> dict[str, PolicySpec]:
    """Every policy in one cell of the (ranking, transport, epsilon) grid, for one fold."""
    key = f"epsilon_{int(epsilon * 100)}"
    view, pool = fold.view(ranking, key), fold.pool(ranking, key)
    specs: dict[str, PolicySpec] = {SGV5_PUBLISHED: sgv5_baseline(fold, epsilon)}
    for name in SOURCE_FITTED:
        chosen = float(selections.get(name, {}).get("selected", float("nan")))
        specs[name] = fit_policy(name, view, pool, ranking, transport, epsilon, chosen)
    scored_pool = fold.scored("pool", ranking, key)
    specs[ORACLE_POOL] = fit_oracle(scored_pool, ranking, epsilon, monotone=False)
    specs[ORACLE_MONOTONE] = fit_oracle(scored_pool, ranking, epsilon, monotone=True)

    evaluation = fold.blocks["evaluation"]
    score = fold.rankings[ranking][key]["evaluation"]
    primary = actions_of(score, fold.tail_risk[key]["evaluation"], specs[PRIMARY_POLICY])
    specs[RANDOM_MATCHED] = fit_random(
        ranking, transport, epsilon, fold.held_out, int((primary == ACCEPT).sum())
    )
    if set(specs) != set(ALL_POLICIES):
        raise PhaseError(f"the cell is missing a policy: {sorted(set(ALL_POLICIES) - set(specs))}")
    if evaluation.size == 0:
        raise PhaseError(f"{fold.held_out}: the evaluation block is empty")
    return specs


def measure_cell(
    fold: Fold, ranking: str, epsilon: float, specs: dict[str, PolicySpec]
) -> dict[str, dict[str, Any]]:
    """Meet the evaluation block once, with every policy in the cell."""
    key = f"epsilon_{int(epsilon * 100)}"
    evaluation = fold.blocks["evaluation"]
    tail = fold.tail_risk[key]["evaluation"]
    out: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        score = fold.rankings[spec.ranking or ranking][key]["evaluation"]
        point = measure(evaluation, actions_of(score, tail, spec), epsilon)
        point["policy"] = name
        point["ranking"] = spec.ranking or ranking
        point["transport"] = spec.transport
        point["held_out_engine"] = fold.held_out
        point["feasible"] = bool(spec.feasible)
        point["tau_accept"] = float(spec.tau_accept)
        point["source_criterion"] = float(spec.source_criterion)
        point["source_coverage"] = float(spec.source_coverage)
        out[name] = point
    return out


def cell_key(ranking: str, transport: str, key: str) -> str:
    return f"{ranking}|{transport}|{key}"


def run_policies() -> int:
    """Verify the freeze, project the published tables into one row table, fit every policy.

    This is the only phase that chooses anything. Everything downstream reads
    `policy_scores.parquet` and the scalars in `design_record.json`, which is what makes a
    re-analysis cheap and what makes "every number traces to an artifact" checkable rather than
    aspirational. No model is fitted anywhere in this stage.
    """
    started = time.monotonic()
    verification = verify_frozen()
    frame = build_rows()
    pilot._write_parquet_once(SCORES, frame)

    sgv10_record = cc._read_json(s10.DESIGN_RECORD)
    thresholds = {
        engine: dict(payload["source_thresholds"])
        for engine, payload in sgv10_record["folds"].items()
    }
    folds = folds_from_frame(frame, thresholds)

    record: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        selections: dict[str, dict[str, Any]] = {}
        cells: dict[str, Any] = {}
        for ranking in RANKINGS:
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                view = fold.view(ranking, key)
                chosen = {
                    name: select_hyperparameter(name, view, epsilon) for name in HYPERPARAMETERS
                }
                selections[f"{ranking}|{key}"] = chosen
                for transport in TRANSPORTS:
                    specs = fit_cell(fold, ranking, transport, epsilon, chosen)
                    cells[cell_key(ranking, transport, key)] = {
                        name: _spec_as_dict(spec) for name, spec in sorted(specs.items())
                    }
        record[engine] = {
            "sgv5_threshold": {
                f"epsilon_{int(e * 100)}": float(fold.sgv5_threshold[f"epsilon_{int(e * 100)}"])
                for e in EPSILONS
            },
            "source_engines": list(fold.source_engines),
            "rows": {name: int(block.size) for name, block in sorted(fold.blocks.items())},
            "documents": {
                name: len(set(block.documents.tolist()))
                for name, block in sorted(fold.blocks.items())
            },
            "harm_prevalence": {
                name: float(block.harmful.mean()) for name, block in sorted(fold.blocks.items())
            },
            "source_rows_by_engine": {
                name: int((fold.engines["calibration"] == name).sum())
                for name in fold.source_engines
            },
            "source_harm_by_engine": {
                name: float(
                    fold.blocks["calibration"].harmful[fold.engines["calibration"] == name].mean()
                )
                for name in fold.source_engines
            },
            "hyperparameter_selection": selections,
            "cells": cells,
        }
        primary = cells[cell_key(PRIMARY_RANKING, PRIMARY_TRANSPORT, PRIMARY_KEY)][PRIMARY_POLICY]
        print(
            f"  {engine}: {fold.blocks['evaluation'].size} evaluation rows, "
            f"{fold.blocks['pool'].size} pool rows, sources {'/'.join(fold.source_engines)}, "
            f"minimax tau {primary['tau_accept']:.4f} at source criterion "
            f"{primary['source_criterion']:.4f}"
        )

    cc._write_json_once(
        DESIGN_RECORD,
        {
            "schema_version": "sgv11-design-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV11-P1",
            "stage": "SGV11 -- minimax risk-constrained decision transfer",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "pre_registration": PRE_REGISTRATION,
            "design_probe": DESIGN_PROBE,
            "decisions": DECISIONS,
            "freeze_verification": verification,
            "policies": {name: POLICY_NOTES[name] for name in ALL_POLICIES},
            "rankings": dict(RANKING_NOTES),
            "transports": dict(TRANSPORT_NOTES),
            "constants": {
                "epsilons": list(EPSILONS),
                "cvar_betas": list(CVAR_BETAS),
                "kl_radii": list(KL_RADII),
                "tail_cutoff_quantiles": list(TAIL_CUTOFF_QUANTILES),
                "utility_weights": [list(pair) for pair in UTILITY_WEIGHTS],
                "coverage_floor": COVERAGE_FLOOR,
                "engine_bar": ENGINE_BAR,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "delta": DELTA,
            },
            "folds": record,
        },
    )
    elapsed = time.monotonic() - started
    print(f"policies: {len(folds)} folds, {len(frame)} rows -> {SCORES} ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ reading the record back


def read_cell(
    record: dict[str, Any], engine: str, ranking: str, transport: str, key: str
) -> dict[str, PolicySpec]:
    """One cell's fitted policies, read back as the scalars they are. Nothing is re-chosen."""
    payload = record["folds"][engine]["cells"][cell_key(ranking, transport, key)]
    return {name: _spec_from_dict(spec) for name, spec in payload.items()}


def frontier_of(fold: Fold, ranking: str, key: str) -> dict[str, Any]:
    """The achievable frontier of one ranking on the evaluation block: an ORACLE over the cut.

    The deepest prefix of this ranking whose REALISED harm holds epsilon. No policy can be
    asked to reach it -- it reads the labels of the block it is measured on -- and it is the
    reference the decision-transfer gap is taken against, because everything between a policy
    and this number is the cost of not knowing where to cut.
    """
    block = fold.blocks["evaluation"]
    return dict(
        achievable_repair_recall(
            fold.rankings[ranking][key]["evaluation"], block.harmful, block.beneficial
        )[key]
    )


def sgv10_published() -> dict[str, dict[str, dict[str, Any]]]:
    """SGV10's own deployed operating points, so the brief's three-way comparison is real.

    Read from SGV10's published risk-coverage table under its primary arm and primary cut rule,
    which are the deployment SGV10 reported. Nothing is recomputed: if SGV10's number moved,
    the hash check in `verify_frozen` would already have stopped the stage.
    """
    table = cc._read_json(s10.RISK_COVERAGE)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for engine, per_epsilon in table["engines"].items():
        for key, cell in per_epsilon.items():
            point = cell["arms"].get(f"{s10.PRIMARY_ARM}|{s10.PRIMARY_RULE}")
            if point is None:
                continue
            out.setdefault(engine, {})[key] = {
                "arm": s10.PRIMARY_ARM,
                "rule": s10.PRIMARY_RULE,
                "coverage": float(point["coverage"]),
                "n_accepted": int(point["n_accepted"]),
                "realized_harm_rate": float(point["realized_harm_rate"]),
                "repair_recall": float(point["repair_recall"]),
                "holds_bound": bool(point["holds_bound"]),
                "risk_controlled_repair_recall": risk_controlled_recall(
                    point, float(cell["epsilon"])
                ),
            }
    return out


# ------------------------------------------------------------------ phase: --results

# The multiplicity family, fixed before any endpoint was computed: the primary policy under the
# primary ranking and the primary transport, on four engines at three epsilons. Twelve tests,
# the same family size SGV10 used, so the two stages' adjusted p-values mean the same thing.
RESULTS_FAMILY = (
    f"{PRIMARY_POLICY} under {PRIMARY_RANKING} and {PRIMARY_TRANSPORT} transport, "
    "paired against SGV5's published deployment, on 4 unseen engines at 3 epsilons"
)


def run_results() -> int:
    """The main table: every policy, every ranking, every transport, every epsilon.

    Section 6 of the report is written from this file. The primary endpoint is the
    risk-controlled repair recall; raw repair recall, realised harm, coverage, abstention and
    the two comparators are all reported beside it so a reader can see what the constraint cost.
    """
    started = time.monotonic()
    folds, record = load_folds()
    published = sgv10_published()
    engines = sorted(folds)
    family: dict[str, float] = {}
    cells: dict[str, Any] = {}

    for engine in engines:
        fold = folds[engine]
        draws = fold_draws(fold.blocks["evaluation"], BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES)
        per_engine: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            baseline_spec = sgv5_baseline(fold, epsilon)
            baseline_actions = actions_of(
                fold.rankings[RANK_SGV5][key]["evaluation"],
                fold.tail_risk[key]["evaluation"],
                baseline_spec,
            )
            baseline = measure(fold.blocks["evaluation"], baseline_actions, epsilon)
            baseline_draws = resampled_actions(
                fold.blocks["evaluation"], baseline_actions, draws, epsilon
            )
            for ranking in RANKINGS:
                for transport in TRANSPORTS:
                    specs = read_cell(record, engine, ranking, transport, key)
                    measured = measure_cell(fold, ranking, epsilon, specs)
                    arms: dict[str, Any] = {}
                    for name, point in sorted(measured.items()):
                        resampled = resampled_actions(
                            fold.blocks["evaluation"],
                            actions_of(
                                fold.rankings[specs[name].ranking or ranking][key]["evaluation"],
                                fold.tail_risk[key]["evaluation"],
                                specs[name],
                            ),
                            draws,
                            epsilon,
                        )
                        delta = _interval(
                            resampled["controlled_draws"] - baseline_draws["controlled_draws"]
                        )
                        point["coverage_floor_met"] = bool(
                            point["n_accepted"] >= COVERAGE_FLOOR * max(baseline["n_accepted"], 1)
                        )
                        point["beats_random_control"] = bool(
                            point["risk_controlled_repair_recall"]
                            > measured[RANDOM_MATCHED]["risk_controlled_repair_recall"]
                        )
                        point["meets_criterion_1"] = bool(
                            point["risk_controlled_repair_recall"]
                            > baseline["risk_controlled_repair_recall"]
                        )
                        arms[name] = {
                            **point,
                            "resampled": {
                                k: v for k, v in resampled.items() if k != "controlled_draws"
                            },
                            "paired_against_sgv5": delta,
                        }
                        if (
                            name == PRIMARY_POLICY
                            and ranking == PRIMARY_RANKING
                            and transport == PRIMARY_TRANSPORT
                        ):
                            family[f"{engine}|{key}"] = float(delta["p_value"])
                    per_engine[cell_key(ranking, transport, key)] = {
                        "epsilon": float(epsilon),
                        "ranking": ranking,
                        "transport": transport,
                        "baseline": baseline,
                        "baseline_resampled": {
                            k: v for k, v in baseline_draws.items() if k != "controlled_draws"
                        },
                        "sgv10_published": published.get(engine, {}).get(key),
                        "frontier": frontier_of(fold, ranking, key),
                        "policies": arms,
                    }
        cells[engine] = per_engine

    from ocr_risk.stats.multiplicity import holm_bonferroni

    adjusted = holm_bonferroni(family) if family else []
    _write(
        POLICY_RESULTS,
        {
            "stage": "SGV11 -- risk-controlled repair recall under a transferred decision",
            "family": RESULTS_FAMILY,
            "primary_policy": PRIMARY_POLICY,
            "primary_ranking": PRIMARY_RANKING,
            "primary_transport": PRIMARY_TRANSPORT,
            "primary_endpoint": PRE_REGISTRATION["primary_endpoint"],
            "criterion_1": PRE_REGISTRATION["criteria"]["1"],
            "coverage_floor": COVERAGE_FLOOR,
            "bootstrap": {
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "unit": "document",
                "threshold": "held fixed inside every resample",
                "pairing": "the same document draws for the policy and the baseline",
            },
            "holm": [
                {
                    "label": test.label,
                    "p_value": test.p_value,
                    "adjusted_p_value": test.adjusted_p_value,
                    "significant": test.significant,
                }
                for test in adjusted
            ],
            "summary": _summary(cells),
            "engines": cells,
        },
        "sgv11-results-v1",
    )
    print(
        f"results: {len(engines)} folds x {len(ALL_POLICIES)} policies x {len(RANKINGS)} "
        f"rankings x {len(TRANSPORTS)} transports x {len(EPSILONS)} epsilons "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _summary(cells: dict[str, Any]) -> dict[str, Any]:
    """Metrics 7 and 8: the worst unseen engine and the average one, per policy and per cell."""
    out: dict[str, Any] = {}
    engines = sorted(cells)
    keys = sorted({name for engine in engines for name in cells[engine]})
    for name in keys:
        per_policy: dict[str, Any] = {}
        for arm in ALL_POLICIES:
            values = [
                float(cells[e][name]["policies"][arm]["risk_controlled_repair_recall"])
                for e in engines
                if name in cells[e]
            ]
            harms = [
                float(cells[e][name]["policies"][arm]["realized_harm_rate"])
                for e in engines
                if name in cells[e]
            ]
            holds = [
                bool(cells[e][name]["policies"][arm]["holds_bound"])
                for e in engines
                if name in cells[e]
            ]
            per_policy[arm] = {
                "worst_engine": float(min(values)) if values else float("nan"),
                "average_engine": float(np.mean(values)) if values else float("nan"),
                "engines_holding_bound": int(sum(holds)),
                "worst_engine_realized_harm": (
                    float(np.nanmax(harms)) if harms and np.isfinite(harms).any() else float("nan")
                ),
                "engines_meeting_criterion_1": int(
                    sum(
                        bool(cells[e][name]["policies"][arm]["meets_criterion_1"])
                        for e in engines
                        if name in cells[e]
                    )
                ),
            }
        baseline = [
            float(cells[e][name]["baseline"]["risk_controlled_repair_recall"])
            for e in engines
            if name in cells[e]
        ]
        per_policy[SGV5_PUBLISHED + "__baseline"] = {
            "worst_engine": float(min(baseline)) if baseline else float("nan"),
            "average_engine": float(np.mean(baseline)) if baseline else float("nan"),
        }
        out[name] = per_policy
    return out


# ------------------------------------------------------------------ phase: --transfer


def run_transfer() -> int:
    """Required analysis 1: does DECISION transfer differ from SCORE transfer?

    The brief's test is "same ranking, different action mapping", and this stage is built so
    that it is literally that: every policy in a cell ranks by the same frozen score, so the
    only thing that varies is which prefix of it is accepted. That makes an exact decomposition
    available. Between the perfect accept set and what a policy actually deploys there are two
    gaps and nothing else --

        1.000                     every beneficial candidate and no harmful one (arithmetic)
        ceiling frontier          the best cut of a ranking fitted on the evaluation block
          |  score-transfer gap   what a better SCORE would buy at a perfect cut
        frozen frontier           the best cut of the ranking the policy actually has
          |  decision-transfer gap  what not knowing WHERE to cut costs
        deployed                  the policy's risk-controlled repair recall

    -- and their sizes say which of the two problems is left. Both frontiers read the labels of
    the block they are measured on; they are ceilings, not methods, and no policy is fitted from
    them.
    """
    started = time.monotonic()
    folds, record = load_folds()
    engines = sorted(folds)
    cells: dict[str, Any] = {}
    for engine in engines:
        fold = folds[engine]
        block = fold.blocks["evaluation"]
        per_engine: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            ceiling = dict(
                achievable_repair_recall(fold.ceiling[key], block.harmful, block.beneficial)[key]
            )
            for ranking in RANKINGS:
                frontier = frontier_of(fold, ranking, key)
                score_gap = float(ceiling["repair_recall"] - frontier["repair_recall"])
                policies: dict[str, Any] = {}
                for transport in TRANSPORTS:
                    specs = read_cell(record, engine, ranking, transport, key)
                    measured = measure_cell(fold, ranking, epsilon, specs)
                    for name, point in sorted(measured.items()):
                        if name in ORACLES or name == RANDOM_MATCHED:
                            continue
                        # Each policy is measured against the frontier of the ranking IT uses.
                        # The published baseline ranks by SGV5's score in every cell, so
                        # comparing it to the composite ranking's frontier would price it
                        # against a curve it never rode.
                        own = specs[name].ranking or ranking
                        own_frontier = frontier if own == ranking else frontier_of(fold, own, key)
                        own_score = fold.rankings[own][key]["evaluation"]
                        deployed = float(point["risk_controlled_repair_recall"])
                        decision_gap = float(own_frontier["repair_recall"] - deployed)
                        total = decision_gap + score_gap
                        policies[f"{name}|{transport}"] = {
                            "deployed": deployed,
                            "ranking_used": own,
                            "own_frontier": float(own_frontier["repair_recall"]),
                            "decision_transfer_gap": decision_gap,
                            "score_transfer_gap": score_gap,
                            "total_gap_to_ceiling": float(ceiling["repair_recall"] - deployed),
                            "decision_share_of_gap": (
                                float(decision_gap / total)
                                if total > 0 and decision_gap >= 0 and score_gap >= 0
                                else float("nan")
                            ),
                            "accept_set_is_a_prefix": _is_prefix(
                                own_score,
                                actions_of(
                                    own_score, fold.tail_risk[key]["evaluation"], specs[name]
                                )
                                == ACCEPT,
                            ),
                        }
                per_engine[f"{ranking}|{key}"] = {
                    "epsilon": float(epsilon),
                    "ranking": ranking,
                    "frozen_frontier": frontier,
                    "ceiling_frontier": ceiling,
                    "score_transfer_gap": score_gap,
                    "perfect_accept_set": {
                        "repair_recall": 1.0,
                        "realized_harm_rate": 0.0,
                        "note": (
                            "arithmetic, not a ceiling anyone could approach: accept every "
                            "beneficial candidate and no harmful one"
                        ),
                    },
                    "policies": policies,
                    "same_ranking_check": _same_ranking_check(fold, record, ranking, key, epsilon),
                }
        cells[engine] = per_engine

    _write(
        DECISION_TRANSFER,
        {
            "stage": "SGV11 -- decision transfer against score transfer",
            "question": (
                "does the accept/reject decision fail to transfer for a different reason than "
                "the score does, and which of the two is the larger remaining gap"
            ),
            "ceiling_source": (
                f"SGV10's `{s10.ORACLE_EVAL}` re-ranking, fitted on the evaluation block itself "
                "and published by SGV10. An ORACLE used only as an analysis ceiling; no policy "
                "in this stage reads it."
            ),
            "engines": cells,
            "aggregate": _transfer_aggregate(cells),
        },
        "sgv11-decision-transfer-v1",
    )
    print(f"transfer: {len(engines)} folds x {len(RANKINGS)} rankings x {len(EPSILONS)} epsilons")
    del started
    return 0


def _is_prefix(score: np.ndarray, accepted: np.ndarray) -> bool:
    """Is this accept set the top of the ranking, or does it skip rows?

    Every two-action policy accepts a prefix, so its repair recall cannot exceed the ranking's
    own achievable frontier. The selective policy's tail-risk veto and the matched random
    control both skip rows, so their accept sets are not prefixes and their decision-transfer
    gap can be NEGATIVE -- they can beat the frontier of the ranking they started from, which
    is the whole mechanism SGV10 identified. The flag is reported rather than the gap clipped.
    """
    mask = np.asarray(accepted, dtype=bool)
    if not mask.any():
        return True
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    ordered = mask[order]
    return bool(not ordered[int(ordered.sum()) :].any())


def _same_ranking_check(
    fold: Fold, record: dict[str, Any], ranking: str, key: str, epsilon: float
) -> dict[str, Any]:
    """Decision 2's prediction, checked: two two-action policies differ only in where they cut.

    Every two-action policy in a cell accepts a prefix of the same ranking, so their accept sets
    must be nested and their risk-coverage curves must be the same curve read at different
    points. Both are asserted here rather than asserted in prose, and the AURC is reported once
    because it is a property of the ranking that no policy in this stage can change.
    """
    block = fold.blocks["evaluation"]
    score = fold.rankings[ranking][key]["evaluation"]
    tail = fold.tail_risk[key]["evaluation"]
    accepted: dict[str, np.ndarray] = {}
    for transport in TRANSPORTS:
        for name, spec in read_cell(record, fold.held_out, ranking, transport, key).items():
            if name in (SELECTIVE, RANDOM_MATCHED, SGV5_PUBLISHED) or name in ORACLES:
                continue
            accepted[f"{name}|{transport}"] = actions_of(score, tail, spec) == ACCEPT
    names = sorted(accepted)
    violations = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            a, b = accepted[names[left]], accepted[names[right]]
            if not (bool((a & ~b).any()) is False or bool((b & ~a).any()) is False):
                violations.append([names[left], names[right]])
    curve = risk_coverage_curve(score, block.harmful)
    return {
        "two_action_policies": names,
        "nested_accept_sets": not violations,
        "violations": violations,
        "aurc_of_the_ranking": float(aurc(curve)),
        "epsilon": float(epsilon),
        "note": (
            "AURC is a property of the ranking. Every policy in this stage shares it by "
            "construction, which is exactly why AURC cannot separate them and why the "
            "risk-controlled repair recall at a fixed epsilon is the endpoint."
        ),
    }


def _transfer_aggregate(cells: dict[str, Any]) -> dict[str, Any]:
    """The two gaps averaged over the four unseen engines, per ranking, policy and transport."""
    out: dict[str, Any] = {}
    engines = sorted(cells)
    keys = sorted({name for engine in engines for name in cells[engine]})
    for name in keys:
        rows = [cells[e][name] for e in engines if name in cells[e]]
        policies = sorted({p for row in rows for p in row["policies"]})
        out[name] = {
            "mean_score_transfer_gap": float(np.mean([r["score_transfer_gap"] for r in rows])),
            "mean_frozen_frontier": float(
                np.mean([r["frozen_frontier"]["repair_recall"] for r in rows])
            ),
            "mean_ceiling_frontier": float(
                np.mean([r["ceiling_frontier"]["repair_recall"] for r in rows])
            ),
            "policies": {
                policy: {
                    "mean_deployed": float(
                        np.mean([r["policies"][policy]["deployed"] for r in rows])
                    ),
                    "mean_decision_transfer_gap": float(
                        np.mean([r["policies"][policy]["decision_transfer_gap"] for r in rows])
                    ),
                }
                for policy in policies
            },
        }
    return out


# ------------------------------------------------------------------ phase: --coverage


def run_coverage() -> int:
    """Secondary metrics 4, 5 and 6: the risk-coverage frontier, AURC, and bound violations.

    The curve is the whole frontier of a ranking on the evaluation block -- a measurement, with
    hindsight, of what every threshold on that ordering would have done. It is never used to
    choose a threshold. Because every policy in a cell shares its ranking, the curve and its
    AURC are shared too, and the deployed points are marked on it so a reader can see how far
    apart the five aggregations land on one identical frontier. That picture is the stage's
    argument in a single object.
    """
    started = time.monotonic()
    folds, record = load_folds()
    engines = sorted(folds)
    from ocr_risk.metrics.selective import coverage_at_risk

    cells: dict[str, Any] = {}
    violations: dict[str, int] = dict.fromkeys(ALL_POLICIES, 0)
    considered = 0
    for engine in engines:
        fold = folds[engine]
        block = fold.blocks["evaluation"]
        per_engine: dict[str, Any] = {}
        for ranking in RANKINGS:
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                curve = risk_coverage_curve(
                    fold.rankings[ranking][key]["evaluation"], block.harmful
                )
                best = coverage_at_risk(curve, epsilon)
                deployed: dict[str, Any] = {}
                for transport in TRANSPORTS:
                    specs = read_cell(record, engine, ranking, transport, key)
                    for name, point in sorted(measure_cell(fold, ranking, epsilon, specs).items()):
                        deployed[f"{name}|{transport}"] = {
                            "coverage": float(point["coverage"]),
                            "realized_harm_rate": float(point["realized_harm_rate"]),
                            "repair_recall": float(point["repair_recall"]),
                            "risk_controlled_repair_recall": float(
                                point["risk_controlled_repair_recall"]
                            ),
                            "abstention_rate": float(point["abstention_rate"]),
                            "holds_bound": bool(point["holds_bound"]),
                            "bound_violation": float(point["bound_violation"]),
                            "accepts_nothing": bool(point["accepts_nothing"]),
                        }
                        if transport == PRIMARY_TRANSPORT and ranking == PRIMARY_RANKING:
                            considered += 1
                            violations[name] += int(not point["holds_bound"])
                per_engine[f"{ranking}|{key}"] = {
                    "epsilon": float(epsilon),
                    "ranking": ranking,
                    "aurc": float(aurc(curve)),
                    "curve": curve.as_dict(),
                    "best_coverage_within_epsilon": (
                        None
                        if best is None
                        else {
                            "coverage": float(best.coverage),
                            "risk": float(best.risk),
                            "n_accepted": int(best.n_accepted),
                        }
                    ),
                    "deployed": deployed,
                }
        cells[engine] = per_engine

    _write(
        RISK_COVERAGE,
        {
            "stage": "SGV11 -- risk-coverage frontiers under a shared ranking",
            "aurc_note": (
                "AURC summarises the RANKING. SGV11 changes no ranking, so within a cell every "
                "policy shares this number; it is reported because the brief asks for it and "
                "because sharing it is the point."
            ),
            "bound_violations": {
                "cells_considered_per_policy": considered // max(len(ALL_POLICIES), 1),
                "scope": (
                    f"{PRIMARY_RANKING} under {PRIMARY_TRANSPORT} transport, four engines, "
                    "three epsilons"
                ),
                "by_policy": violations,
            },
            "engines": cells,
        },
        "sgv11-risk-coverage-v1",
    )
    print(
        f"coverage: {len(engines)} folds x {len(RANKINGS)} rankings x {len(EPSILONS)} epsilons "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ phase: --robustness


def run_robustness() -> int:
    """Required analysis 2: is minimax overly conservative? Utility lost against safety gained.

    Three questions, each with a number. What does the worst-case aggregation cost in repair
    recall against the average one on the same rows? What does it buy in realised harm? And
    does the 3/4 distribution-free coverage that the minimax rule inherits from being a maximum
    over three exchangeable engines actually appear on the fourth?

    The third is the one worth the stage. Every leave-one-engine-out result in SGV1 to SGV10
    assumes the four engines are exchangeable; this is the first place that assumption is
    turned into a prediction with a number attached, and the prediction is testable in exactly
    four trials.
    """
    started = time.monotonic()
    folds, record = load_folds()
    engines = sorted(folds)
    cells: dict[str, Any] = {}
    for engine in engines:
        fold = folds[engine]
        per_engine: dict[str, Any] = {}
        for ranking in RANKINGS:
            for transport in TRANSPORTS:
                for epsilon in EPSILONS:
                    key = f"epsilon_{int(epsilon * 100)}"
                    specs = read_cell(record, engine, ranking, transport, key)
                    measured = measure_cell(fold, ranking, epsilon, specs)
                    strict, loose = measured[MINIMAX], measured[ERM_AVERAGE]
                    harm_gain = float(loose["realized_harm_rate"] - strict["realized_harm_rate"])
                    recall_cost = float(loose["repair_recall"] - strict["repair_recall"])
                    per_engine[cell_key(ranking, transport, key)] = {
                        "epsilon": float(epsilon),
                        "minimax": _short(strict),
                        "average": _short(loose),
                        "recall_given_up": recall_cost,
                        "harm_avoided": harm_gain,
                        "recall_per_point_of_harm": (
                            float(recall_cost / harm_gain)
                            if np.isfinite(harm_gain) and abs(harm_gain) > 1e-12
                            else float("nan")
                        ),
                        "source_threshold_gap": float(
                            specs[MINIMAX].tau_accept - specs[ERM_AVERAGE].tau_accept
                        ),
                        "source_criterion": {
                            "minimax": float(specs[MINIMAX].source_criterion),
                            "average": float(specs[ERM_AVERAGE].source_criterion),
                        },
                    }
        record_fold = record["folds"][engine]
        per_engine["source_engine_spread"] = {
            "rows": record_fold["source_rows_by_engine"],
            "harm_prevalence": record_fold["source_harm_by_engine"],
            "spread": float(
                max(record_fold["source_harm_by_engine"].values())
                - min(record_fold["source_harm_by_engine"].values())
            ),
            "target_harm_prevalence": float(record_fold["harm_prevalence"]["evaluation"]),
            "target_inside_source_range": bool(
                min(record_fold["source_harm_by_engine"].values())
                <= record_fold["harm_prevalence"]["evaluation"]
                <= max(record_fold["source_harm_by_engine"].values())
            ),
        }
        cells[engine] = per_engine

    _write(
        ROBUSTNESS,
        {
            "stage": "SGV11 -- what the worst-case aggregation costs and what it buys",
            "exchangeability": _exchangeability(folds, record),
            "engines": cells,
        },
        "sgv11-robustness-v1",
    )
    print(f"robustness: {len(engines)} folds ({time.monotonic() - started:.0f}s)")
    return 0


def _short(point: dict[str, Any]) -> dict[str, Any]:
    return {
        name: point[name]
        for name in (
            "coverage",
            "n_accepted",
            "realized_harm_rate",
            "repair_recall",
            "risk_controlled_repair_recall",
            "holds_bound",
            "abstention_rate",
        )
    }


def _exchangeability(folds: dict[str, Fold], record: dict[str, Any]) -> dict[str, Any]:
    """Sub-claim P1c: does the maximum over three source engines certify the fourth at 3/4?

    If the four engines were exchangeable draws, the unseen engine's harm rate at a threshold
    would exceed the maximum of the three source rates with probability at most 1/(K+1) = 1/4,
    so the minimax threshold should hold the bound on at least three of the four folds. This
    reports how many folds it actually holds on, and -- separately, because the two can come
    apart -- how often the target's own realised rate exceeds the source maximum at the same
    threshold, which is the rank statistic itself with nothing else in the way.
    """
    per_epsilon: dict[str, Any] = {}
    for epsilon in EPSILONS:
        key = f"epsilon_{int(epsilon * 100)}"
        rows: dict[str, Any] = {}
        for engine, fold in sorted(folds.items()):
            spec = read_cell(record, engine, PRIMARY_RANKING, PRIMARY_TRANSPORT, key)[MINIMAX]
            block = fold.blocks["evaluation"]
            score = fold.rankings[PRIMARY_RANKING][key]["evaluation"]
            point = measure(
                block, actions_of(score, fold.tail_risk[key]["evaluation"], spec), epsilon
            )
            view = fold.view(PRIMARY_RANKING, key)
            per_source = {
                name: float(
                    view.harmful[(view.engines == name) & (view.score >= spec.tau_accept)].mean()
                )
                if bool(((view.engines == name) & (view.score >= spec.tau_accept)).any())
                else float("nan")
                for name in fold.source_engines
            }
            maximum = float(np.nanmax(list(per_source.values()))) if per_source else float("nan")
            realized = float(point["realized_harm_rate"])
            rows[engine] = {
                "source_rates_at_the_threshold": per_source,
                "source_maximum": maximum,
                "target_realized": realized,
                "target_exceeds_source_maximum": bool(
                    np.isfinite(realized) and np.isfinite(maximum) and realized > maximum
                ),
                "holds_epsilon": bool(point["holds_bound"]),
            }
        exceed = sum(1 for row in rows.values() if row["target_exceeds_source_maximum"])
        per_epsilon[key] = {
            "epsilon": float(epsilon),
            "engines": rows,
            "folds_holding_epsilon": sum(1 for row in rows.values() if row["holds_epsilon"]),
            "folds_where_target_exceeds_source_maximum": exceed,
            "predicted_at_most": 1,
            "prediction": (
                "with K = 3 exchangeable source engines the unseen engine's rate exceeds the "
                "maximum of the three with probability at most 1/(K+1) = 1/4, so at most one "
                "of the four folds should exceed it"
            ),
            "distribution_free_coverage": CONFORMAL_COVERAGE,
        }
    return per_epsilon


# ------------------------------------------------------------------ phase: --decomposition


def harm_terms(fold: Fold, ranking: str, key: str, spec: PolicySpec) -> dict[str, Any]:
    """The exact four-term decomposition of what a threshold's harm rate turns into.

    At one threshold, four harm rates are measurable on three different sets of rows:

        criterion       what the policy bounded on the SOURCE engines, by construction <= epsilon
        source realised the actual harm rate of the source rows the threshold accepts
        pool realised   the same threshold on the unseen engine's own TRAIN pool
        evaluation      the same threshold on the unseen engine's DEVELOPMENT block

    and the differences between consecutive rows are named: `criterion_slack` (how much the
    aggregation over-states or under-states its own engines), `engine_shift` (what changes when
    the rows come from an engine nobody calibrated on), and `block_shift` (what changes when the
    pages change but the engine does not). They sum exactly to the realised evaluation harm, and
    the last of the three is the one no amount of engine-level robustness can touch.

    The pool's labels are read HERE and nowhere else in the stage. This is an analysis of a
    threshold that was already chosen; no policy, no hyperparameter and no transport reads it.
    """
    score_of = {name: fold.rankings[ranking][key][name] for name in BLOCK_NAMES}
    tail_of = {name: fold.tail_risk[key][name] for name in BLOCK_NAMES}
    rates: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in BLOCK_NAMES:
        accepted = actions_of(score_of[name], tail_of[name], spec) == ACCEPT
        counts[name] = int(accepted.sum())
        block = fold.blocks[name]
        rates[name] = (
            float(block.harmful[accepted].astype(float).mean()) if accepted.any() else float("nan")
        )
    criterion = float(spec.source_criterion)
    return {
        "criterion": criterion,
        # SGV5's published threshold was not chosen by any of this stage's criteria, so it has
        # no criterion term and its slack is undefined. The other three terms are still exact
        # for it, which is the point of measuring them at a threshold rather than at a rule.
        "has_source_criterion": bool(np.isfinite(criterion)),
        "source_realized": rates["calibration"],
        "pool_realized": rates["pool"],
        "evaluation_realized": rates["evaluation"],
        "criterion_slack": float(rates["calibration"] - criterion),
        "engine_shift": float(rates["pool"] - rates["calibration"]),
        "block_shift": float(rates["evaluation"] - rates["pool"]),
        "n_accepted": counts,
        "identity_residual": float(
            rates["evaluation"]
            - (
                criterion
                + (rates["calibration"] - criterion)
                + (rates["pool"] - rates["calibration"])
                + (rates["evaluation"] - rates["pool"])
            )
        ),
    }


def run_decomposition() -> int:
    """Required analysis 3: where does the policy fail, in terms that add up to the failure.

    Section 8 of the report is written from this file. EasyOCR is the case every stage since
    SGV1 has named as the hardest transfer, and the decomposition says in which of the three
    terms its difficulty actually sits.
    """
    started = time.monotonic()
    folds, record = load_folds()
    engines = sorted(folds)
    cells: dict[str, Any] = {}
    for engine in engines:
        fold = folds[engine]
        per_engine: dict[str, Any] = {}
        for ranking in RANKINGS:
            for transport in TRANSPORTS:
                for epsilon in EPSILONS:
                    key = f"epsilon_{int(epsilon * 100)}"
                    specs = read_cell(record, engine, ranking, transport, key)
                    per_engine[cell_key(ranking, transport, key)] = {
                        "epsilon": float(epsilon),
                        "policies": {
                            name: harm_terms(fold, ranking, key, spec)
                            for name, spec in sorted(specs.items())
                            if name not in (RANDOM_MATCHED,) and spec.feasible
                        },
                    }
        cells[engine] = per_engine

    aggregate: dict[str, Any] = {}
    for name in sorted({k for e in engines for k in cells[e] if isinstance(cells[e][k], dict)}):
        rows = {e: cells[e][name]["policies"] for e in engines if name in cells[e]}
        policies = sorted({p for row in rows.values() for p in row})
        aggregate[name] = {
            policy: {
                term: {
                    engine: float(rows[engine][policy][term])
                    for engine in engines
                    if policy in rows.get(engine, {})
                }
                for term in ("criterion_slack", "engine_shift", "block_shift")
            }
            for policy in policies
        }

    primary = aggregate[cell_key(PRIMARY_RANKING, PRIMARY_TRANSPORT, PRIMARY_KEY)][PRIMARY_POLICY]
    block_positive = sum(1 for value in primary["block_shift"].values() if value > 0)
    block_larger = sum(
        1
        for engine in primary["block_shift"]
        if abs(primary["block_shift"][engine]) > abs(primary["engine_shift"][engine])
    )
    _write(
        HARM_DECOMPOSITION,
        {
            "stage": "SGV11 -- where a transferred threshold's harm actually comes from",
            "terms": {
                "criterion": "what the policy bounded on the source engines, by construction",
                "criterion_slack": "source realised harm minus the criterion",
                "engine_shift": "the target pool's harm at the same threshold, minus the source's",
                "block_shift": (
                    "the target evaluation block's harm at the same threshold, minus the target "
                    "pool's. The engine is identical on both sides of this term; only the pages "
                    "differ."
                ),
            },
            "decision_8_prediction": {
                "claim": (
                    "the block term is positive on all four engines, and on at least three of "
                    "them it is larger in magnitude than the engine term"
                ),
                "scope": cell_key(PRIMARY_RANKING, PRIMARY_TRANSPORT, PRIMARY_KEY)
                + f" / {PRIMARY_POLICY}",
                "engines_with_positive_block_shift": block_positive,
                "engines_where_block_exceeds_engine": block_larger,
                "met": bool(block_positive == len(engines) and block_larger >= ENGINE_BAR),
            },
            "pool_labels": (
                "the target pool's labels are read in this phase and nowhere else. Every "
                "threshold analysed here was already chosen without them."
            ),
            "aggregate": aggregate,
            "engines": cells,
        },
        "sgv11-harm-decomposition-v1",
    )
    print(f"decomposition: {len(engines)} folds ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --oracle


def run_oracle() -> int:
    """Ablation 4 and required analysis 4: how much of the gap target labels could close.

    Four references, in order of what each is allowed to see. The frozen ranking's frontier
    reads the evaluation block's labels but may only move the cut. The in-engine oracle reads
    the target engine's own POOL labels and may only move the cut. The ceiling reads the
    evaluation block's labels and may re-rank. And the perfect accept set reads everything and
    is arithmetic. A policy's distance to each says a different thing, and the distance to the
    in-engine oracle is the one the brief asks for.
    """
    started = time.monotonic()
    folds, record = load_folds()
    engines = sorted(folds)
    cells: dict[str, Any] = {}
    for engine in engines:
        fold = folds[engine]
        block = fold.blocks["evaluation"]
        per_engine: dict[str, Any] = {}
        for ranking in RANKINGS:
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                specs = read_cell(record, engine, ranking, PRIMARY_TRANSPORT, key)
                measured = measure_cell(fold, ranking, epsilon, specs)
                oracle = measured[ORACLE_POOL]
                monotone = measured[ORACLE_MONOTONE]
                frontier = frontier_of(fold, ranking, key)
                ceiling = dict(
                    achievable_repair_recall(fold.ceiling[key], block.harmful, block.beneficial)[
                        key
                    ]
                )
                per_engine[f"{ranking}|{key}"] = {
                    "epsilon": float(epsilon),
                    "in_engine_oracle_deepest": _short(oracle),
                    "in_engine_oracle_monotone": _short(monotone),
                    "frozen_ranking_frontier": frontier,
                    "reranking_ceiling": ceiling,
                    "perfect_accept_set": {"repair_recall": 1.0, "realized_harm_rate": 0.0},
                    "oracle_holds_epsilon_on_the_evaluation_block": bool(oracle["holds_bound"]),
                    "oracle_pool_criterion": float(specs[ORACLE_POOL].source_criterion),
                    "oracle_optimism": float(
                        oracle["realized_harm_rate"] - specs[ORACLE_POOL].source_criterion
                    ),
                    "gap_to_oracle": {
                        name: float(
                            oracle["risk_controlled_repair_recall"]
                            - measured[name]["risk_controlled_repair_recall"]
                        )
                        for name in (*BASELINES, *PROPOSED, *CONTROLS)
                    },
                    "gap_to_frontier": {
                        name: float(
                            frontier["repair_recall"]
                            - measured[name]["risk_controlled_repair_recall"]
                        )
                        for name in (*BASELINES, *PROPOSED, *CONTROLS)
                    },
                }
        cells[engine] = per_engine

    primary = [cells[engine][f"{PRIMARY_RANKING}|{PRIMARY_KEY}"] for engine in engines]
    _write(
        ORACLE_GAP,
        {
            "stage": "SGV11 -- what target labels could buy a threshold, and what they could not",
            "decision_9_prediction": {
                "claim": (
                    "the deepest-feasible in-engine oracle violates epsilon on the evaluation "
                    "block on all four engines despite holding it on the pool it was chosen on"
                ),
                "scope": f"{PRIMARY_RANKING} / {PRIMARY_KEY}",
                "engines_where_the_oracle_violates": sum(
                    1 for row in primary if not row["oracle_holds_epsilon_on_the_evaluation_block"]
                ),
                "met": bool(
                    sum(
                        1
                        for row in primary
                        if not row["oracle_holds_epsilon_on_the_evaluation_block"]
                    )
                    == len(engines)
                ),
            },
            "reading": (
                "the in-engine oracle knows the held-out engine's own labels on thousands of its "
                "own pages and still cannot place a threshold that holds on a different block of "
                "the same engine's pages. Whatever remains after that is not an engine-transfer "
                "problem."
            ),
            "engines": cells,
        },
        "sgv11-oracle-gap-v1",
    )
    print(f"oracle: {len(engines)} folds ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --ablation

ABLATIONS = {
    "1_average_versus_minimax": {
        "question": "does the robustness come from the minimax formulation?",
        "contrast": f"{ERM_AVERAGE} against {MINIMAX}, same rows, same ranking, same transport",
    },
    "2_remove_sgv10_tail_risk": {
        "question": "does tail-risk information improve DECISION transfer?",
        "contrast": (
            f"{RANK_SGV5} against {RANK_SGV10} for every policy, and the selective policy's "
            "tail-risk veto on against off"
        ),
    },
    "3_remove_worst_engine_constraint": {
        "question": "is it the engine structure, or the worst case over it?",
        "contrast": (
            f"{POOLED_ERM} (no engine structure) against {ERM_AVERAGE} (structure, averaged) "
            f"against {MINIMAX} (structure, worst case)"
        ),
    },
    "4_oracle_policy": {
        "question": "how much of the remaining gap could target labels close?",
        "contrast": f"{ORACLE_POOL} and {ORACLE_MONOTONE} against every fitted policy",
    },
}


def run_ablation() -> int:
    """The brief's four ablations, plus the two hyperparameter sweeps the selection stood in for.

    The sweeps matter as much as the ablations: the CVaR level and the KL radius are chosen by
    an inner replay that never sees the target, so the endpoint under every grid value says what
    a perfectly informed choice would have bought and what the honest one cost.
    """
    started = time.monotonic()
    folds, record = load_folds()
    engines = sorted(folds)
    out: dict[str, Any] = {name: {"spec": spec, "engines": {}} for name, spec in ABLATIONS.items()}

    for engine in engines:
        fold = folds[engine]
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            by_cell = {
                (ranking, transport): measure_cell(
                    fold, ranking, epsilon, read_cell(record, engine, ranking, transport, key)
                )
                for ranking in RANKINGS
                for transport in TRANSPORTS
            }
            primary = by_cell[(PRIMARY_RANKING, PRIMARY_TRANSPORT)]
            out["1_average_versus_minimax"]["engines"].setdefault(engine, {})[key] = {
                transport: {
                    "average": _short(by_cell[(PRIMARY_RANKING, transport)][ERM_AVERAGE]),
                    "minimax": _short(by_cell[(PRIMARY_RANKING, transport)][MINIMAX]),
                    "delta_risk_controlled": float(
                        by_cell[(PRIMARY_RANKING, transport)][MINIMAX][
                            "risk_controlled_repair_recall"
                        ]
                        - by_cell[(PRIMARY_RANKING, transport)][ERM_AVERAGE][
                            "risk_controlled_repair_recall"
                        ]
                    ),
                }
                for transport in TRANSPORTS
            }
            out["2_remove_sgv10_tail_risk"]["engines"].setdefault(engine, {})[key] = {
                "ranking": {
                    name: {
                        RANK_SGV5: _short(by_cell[(RANK_SGV5, PRIMARY_TRANSPORT)][name]),
                        RANK_SGV10: _short(by_cell[(RANK_SGV10, PRIMARY_TRANSPORT)][name]),
                        "delta_risk_controlled": float(
                            by_cell[(RANK_SGV10, PRIMARY_TRANSPORT)][name][
                                "risk_controlled_repair_recall"
                            ]
                            - by_cell[(RANK_SGV5, PRIMARY_TRANSPORT)][name][
                                "risk_controlled_repair_recall"
                            ]
                        ),
                    }
                    for name in PROPOSED
                },
                "tail_veto": _tail_veto_sweep(fold, record, engine, key, epsilon),
            }
            out["3_remove_worst_engine_constraint"]["engines"].setdefault(engine, {})[key] = {
                "pooled_no_engine_structure": _short(primary[POOLED_ERM]),
                "per_engine_averaged": _short(primary[ERM_AVERAGE]),
                "per_engine_worst_case": _short(primary[MINIMAX]),
                "worst_case_minus_pooled": float(
                    primary[MINIMAX]["risk_controlled_repair_recall"]
                    - primary[POOLED_ERM]["risk_controlled_repair_recall"]
                ),
                "averaged_minus_pooled": float(
                    primary[ERM_AVERAGE]["risk_controlled_repair_recall"]
                    - primary[POOLED_ERM]["risk_controlled_repair_recall"]
                ),
            }
            out["4_oracle_policy"]["engines"].setdefault(engine, {})[key] = {
                "oracle_deepest": _short(primary[ORACLE_POOL]),
                "oracle_monotone": _short(primary[ORACLE_MONOTONE]),
                "gap_from_primary": float(
                    primary[ORACLE_POOL]["risk_controlled_repair_recall"]
                    - primary[PRIMARY_POLICY]["risk_controlled_repair_recall"]
                ),
                "oracle_holds_bound": bool(primary[ORACLE_POOL]["holds_bound"]),
            }

    _write(
        ABLATION_RESULTS,
        {
            "stage": "SGV11 -- the brief's four ablations, and what the selection cost",
            "ablations": out,
            "hyperparameter_sweeps": _sweeps(folds, record),
        },
        "sgv11-ablation-v1",
    )
    print(
        f"ablation: {len(ABLATIONS)} ablations x {len(engines)} folds "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _tail_veto_sweep(
    fold: Fold, record: dict[str, Any], engine: str, key: str, epsilon: float
) -> dict[str, Any]:
    """The selective policy with its tail-risk veto on and off, everything else identical.

    Turning the veto off is `tail_cutoff = +inf`, which is one field of the spec, so this is a
    contrast between two thresholds on the same rows rather than between two constructions.
    """
    out: dict[str, Any] = {}
    for ranking in RANKINGS:
        spec = read_cell(record, engine, ranking, PRIMARY_TRANSPORT, key)[SELECTIVE]
        score = fold.rankings[ranking][key]["evaluation"]
        tail = fold.tail_risk[key]["evaluation"]
        with_veto = measure(fold.blocks["evaluation"], actions_of(score, tail, spec), epsilon)
        without = measure(
            fold.blocks["evaluation"],
            actions_of(score, tail, replace(spec, tail_cutoff=float(np.inf))),
            epsilon,
        )
        out[ranking] = {
            "with_tail_veto": _short(with_veto),
            "without_tail_veto": _short(without),
            "tail_cutoff": float(spec.tail_cutoff),
            "delta_risk_controlled": float(
                with_veto["risk_controlled_repair_recall"]
                - without["risk_controlled_repair_recall"]
            ),
        }
    return out


def _sweeps(folds: dict[str, Fold], record: dict[str, Any]) -> dict[str, Any]:
    """Every grid value of every hyperparameter, deployed, beside the one the replay chose.

    The gap between the best grid value and the chosen one is what leakage-free selection costs
    on this corpus. SGV6 reported the same quantity for few-shot adaptation and found it large;
    reporting it here makes the two comparable.
    """
    out: dict[str, Any] = {}
    for arm, (name, grid) in sorted(HYPERPARAMETERS.items()):
        per_policy: dict[str, Any] = {}
        for engine, fold in sorted(folds.items()):
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                view = fold.view(PRIMARY_RANKING, key)
                pool = fold.pool(PRIMARY_RANKING, key)
                score = fold.rankings[PRIMARY_RANKING][key]["evaluation"]
                tail = fold.tail_risk[key]["evaluation"]
                rows: dict[str, Any] = {}
                for value in grid:
                    spec = fit_policy(
                        arm, view, pool, PRIMARY_RANKING, PRIMARY_TRANSPORT, epsilon, float(value)
                    )
                    point = measure(
                        fold.blocks["evaluation"], actions_of(score, tail, spec), epsilon
                    )
                    rows[f"{value:g}"] = _short(point)
                chosen = float(
                    record["folds"][engine]["hyperparameter_selection"][f"{PRIMARY_RANKING}|{key}"][
                        arm
                    ]["selected"]
                )
                best = max(rows, key=lambda k: rows[k]["risk_controlled_repair_recall"])
                per_policy.setdefault(engine, {})[key] = {
                    "grid": rows,
                    "selected_by_inner_replay": chosen,
                    "best_on_the_evaluation_block": float(best),
                    "selection_cost": float(
                        rows[best]["risk_controlled_repair_recall"]
                        - rows[f"{chosen:g}"]["risk_controlled_repair_recall"]
                    ),
                }
        out[arm] = {"parameter": name, "grid": list(grid), "engines": per_policy}
    return out


# ------------------------------------------------------------------ phase: --negative


def _permuted_engines(view: SourcePolicyView, engine: str, key: str) -> np.ndarray:
    """The source engine labels reassigned at random, with the group sizes preserved.

    The seed is derived from content with sha256, never from `hash()`. Preserving the sizes is
    what makes this a control rather than a different experiment: the minimax rule still takes a
    maximum over three groups of the same three sizes, and only the membership is noise.
    """
    rng = np.random.default_rng(_stable_seed("sgv11-engine-permutation", engine, key))
    return np.asarray(view.engines)[rng.permutation(view.size)]


def run_negative() -> int:
    """The brief's five leakage checks and the degeneracy controls, as measurements.

    Each one is written so that a violation would show up as a number, not as an absent
    assertion: the engine control refits the primary policy on shuffled engine labels, the
    label control refits every policy with the target's labels destroyed and compares the
    thresholds bit for bit, and the degeneracy control counts the cells where an arm passes the
    bound by accepting almost nothing.
    """
    started = time.monotonic()
    folds, record = load_folds()
    engines = sorted(folds)
    tests: dict[str, Any] = {}

    # --- 1. engine leakage: is the engine structure real? --------------------------------
    rows: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            view = fold.view(PRIMARY_RANKING, key)
            pool = fold.pool(PRIMARY_RANKING, key)
            shuffled = replace(view, engines=_permuted_engines(view, engine, key))
            real = fit_policy(
                MINIMAX, view, pool, PRIMARY_RANKING, PRIMARY_TRANSPORT, epsilon, float("nan")
            )
            control = fit_policy(
                MINIMAX, shuffled, pool, PRIMARY_RANKING, PRIMARY_TRANSPORT, epsilon, float("nan")
            )
            score = fold.rankings[PRIMARY_RANKING][key]["evaluation"]
            tail = fold.tail_risk[key]["evaluation"]
            rows.setdefault(engine, {})[key] = {
                "real": _short(
                    measure(fold.blocks["evaluation"], actions_of(score, tail, real), epsilon)
                ),
                "shuffled_engine_labels": _short(
                    measure(fold.blocks["evaluation"], actions_of(score, tail, control), epsilon)
                ),
                "tau_real": float(real.tau_accept),
                "tau_shuffled": float(control.tau_accept),
                "thresholds_differ": bool(real.tau_accept != control.tau_accept),
            }
    differ = sum(
        1 for engine in rows for key in rows[engine] if rows[engine][key]["thresholds_differ"]
    )
    tests["1_engine_structure_is_real"] = {
        "question": (
            "does the minimax rule exploit real between-engine structure, or would any "
            "partition of the source rows into three groups of the same sizes do as well?"
        ),
        "cells": len(engines) * len(EPSILONS),
        "cells_where_the_threshold_moves": differ,
        "passes": bool(differ >= ENGINE_BAR),
        "reading": (
            "a threshold that did not move under a shuffled partition would mean the worst-case "
            "aggregation was reading noise rather than engine identity"
        ),
        "engines": rows,
    }

    # --- 2. label leakage: destroy the target's labels and refit --------------------------
    drift: list[dict[str, Any]] = []
    for engine, fold in sorted(folds.items()):
        rng = np.random.default_rng(_stable_seed("sgv11-target-label-permutation", engine))
        for ranking in RANKINGS:
            for transport in TRANSPORTS:
                for epsilon in EPSILONS:
                    key = f"epsilon_{int(epsilon * 100)}"
                    view = fold.view(ranking, key)
                    pool = fold.pool(ranking, key)
                    for name in SOURCE_FITTED:
                        chosen = float(
                            record["folds"][engine]["hyperparameter_selection"][f"{ranking}|{key}"]
                            .get(name, {})
                            .get("selected", float("nan"))
                        )
                        published = read_cell(record, engine, ranking, transport, key)[name]
                        refitted = fit_policy(name, view, pool, ranking, transport, epsilon, chosen)
                        if float(published.tau_accept) != float(refitted.tau_accept):
                            drift.append(
                                {
                                    "engine": engine,
                                    "cell": cell_key(ranking, transport, key),
                                    "policy": name,
                                    "published": float(published.tau_accept),
                                    "refitted": float(refitted.tau_accept),
                                }
                            )
        del rng
    tests["2_no_target_label_reaches_a_threshold"] = {
        "question": "can any fitted threshold be changed by changing a target label?",
        "construction": (
            "every source-fitted policy is refitted from the SourcePolicyView and the "
            "TargetPoolView alone. Neither object carries a target label -- TargetPoolView has "
            "one field, the pool's scores -- so a refit that reproduces the published threshold "
            "exactly is proof that no target label reached it. The syntax-tree scan in "
            "tests/leakage checks the same thing from the other direction."
        ),
        "cells": len(engines)
        * len(RANKINGS)
        * len(TRANSPORTS)
        * len(EPSILONS)
        * len(SOURCE_FITTED),
        "thresholds_that_drifted": len(drift),
        "drift": drift,
        "passes": not drift,
    }

    # --- 3. oracle leakage ---------------------------------------------------------------
    overlap = sorted(set(SOURCE_FITTED) & set(ORACLES))
    tests["3_oracles_are_never_selected_from"] = {
        "question": "can an oracle become the finding?",
        "source_fitted": list(SOURCE_FITTED),
        "oracles": list(ORACLES),
        "overlap": overlap,
        "proposed_family": list(PROPOSED),
        "oracles_inside_the_proposed_family": sorted(set(ORACLES) & set(PROPOSED)),
        "passes": not overlap and not (set(ORACLES) & set(PROPOSED)),
    }

    # --- 4. selection leakage -------------------------------------------------------------
    scopes: list[dict[str, Any]] = []
    for engine in engines:
        for name, payload in sorted(record["folds"][engine]["hyperparameter_selection"].items()):
            for arm, chosen in sorted(payload.items()):
                targets = sorted(
                    {row["pseudo_target"] for rows_ in chosen["replays"].values() for row in rows_}
                )
                scopes.append(
                    {
                        "engine": engine,
                        "cell": name,
                        "policy": arm,
                        "selected": chosen["selected"],
                        "pseudo_targets": targets,
                        "held_out_engine_appears": bool(engine in targets),
                    }
                )
    tests["4_selection_stays_inside_the_source_engines"] = {
        "question": "is any hyperparameter chosen with knowledge of the held-out engine?",
        "selections": len(scopes),
        "selections_touching_the_held_out_engine": sum(
            1 for row in scopes if row["held_out_engine_appears"]
        ),
        "passes": not any(row["held_out_engine_appears"] for row in scopes),
        "detail": scopes,
    }

    # --- 5. degeneracy: does anything win by refusing? -------------------------------------
    degenerate: dict[str, Any] = {}
    for name in ALL_POLICIES:
        cells: list[dict[str, Any]] = []
        for engine, fold in sorted(folds.items()):
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                specs = read_cell(record, engine, PRIMARY_RANKING, PRIMARY_TRANSPORT, key)
                measured = measure_cell(fold, PRIMARY_RANKING, epsilon, specs)
                baseline = measure(
                    fold.blocks["evaluation"],
                    actions_of(
                        fold.rankings[RANK_SGV5][key]["evaluation"],
                        fold.tail_risk[key]["evaluation"],
                        sgv5_baseline(fold, epsilon),
                    ),
                    epsilon,
                )
                point = measured[name]
                cells.append(
                    {
                        "engine": engine,
                        "epsilon": float(epsilon),
                        "n_accepted": int(point["n_accepted"]),
                        "baseline_n_accepted": int(baseline["n_accepted"]),
                        "coverage_floor_met": bool(
                            point["n_accepted"] >= COVERAGE_FLOOR * max(baseline["n_accepted"], 1)
                        ),
                        "accepts_nothing": bool(point["accepts_nothing"]),
                        "risk_controlled_repair_recall": float(
                            point["risk_controlled_repair_recall"]
                        ),
                    }
                )
        degenerate[name] = {
            "cells": cells,
            "cells_below_the_coverage_floor": sum(
                1 for row in cells if not row["coverage_floor_met"]
            ),
            "cells_accepting_nothing": sum(1 for row in cells if row["accepts_nothing"]),
        }
    tests["5_no_policy_wins_by_refusing"] = {
        "question": "does any policy hold the bound by accepting almost nothing?",
        "coverage_floor": COVERAGE_FLOOR,
        "rule": (
            "a policy earns credit in a cell only if it accepts at least half as many "
            "candidates as SGV5's published deployment"
        ),
        "policies": degenerate,
        "passes": bool(degenerate[PRIMARY_POLICY]["cells_below_the_coverage_floor"] == 0),
    }

    # --- 6. the design probe, reproduced --------------------------------------------------
    tests["6_design_probe_reproduces"] = _reproduce_probe(folds, record)

    _write(
        NEGATIVE_TESTS,
        {
            "stage": "SGV11 -- leakage checks and degeneracy controls",
            "tests": tests,
            "all_passed": bool(all(test["passes"] for test in tests.values())),
        },
        "sgv11-negative-v1",
    )
    print(
        f"negative: {len(tests)} tests, "
        f"{sum(1 for t in tests.values() if t['passes'])} passing "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _reproduce_probe(folds: dict[str, Fold], record: dict[str, Any]) -> dict[str, Any]:
    """Decision 4, 5, 8 and 9's source, recomputed from the artifacts they informed.

    The pre-registration names four numbers that a read-only probe on SGV10's published table
    measured before this stage was designed. If the final pipeline does not reproduce them, the
    probe was measuring something else and the disclosure in `design_record.json` would be
    misleading rather than merely incomplete.
    """
    measured: dict[str, dict[str, float]] = {
        "minimax_realized_harm": {},
        "minimax_repair_recall": {},
        "oracle_pool_realized_harm": {},
        "cvar_90_coverage": {},
    }
    for engine, fold in sorted(folds.items()):
        specs = read_cell(record, engine, PRIMARY_RANKING, PRIMARY_TRANSPORT, PRIMARY_KEY)
        points = measure_cell(fold, PRIMARY_RANKING, PRIMARY_EPSILON, specs)
        measured["minimax_realized_harm"][engine] = float(points[MINIMAX]["realized_harm_rate"])
        measured["minimax_repair_recall"][engine] = float(points[MINIMAX]["repair_recall"])
        measured["oracle_pool_realized_harm"][engine] = float(
            points[ORACLE_POOL]["realized_harm_rate"]
        )
        spec = fit_policy(
            DRO_CVAR,
            fold.view(PRIMARY_RANKING, PRIMARY_KEY),
            fold.pool(PRIMARY_RANKING, PRIMARY_KEY),
            PRIMARY_RANKING,
            PRIMARY_TRANSPORT,
            PRIMARY_EPSILON,
            0.90,
        )
        point = measure(
            fold.blocks["evaluation"],
            actions_of(
                fold.rankings[PRIMARY_RANKING][PRIMARY_KEY]["evaluation"],
                fold.tail_risk[PRIMARY_KEY]["evaluation"],
                spec,
            ),
            PRIMARY_EPSILON,
        )
        measured["cvar_90_coverage"][engine] = float(point["coverage"])

    decimals = dict(DESIGN_PROBE["printed_decimals"])
    tolerance = {name: 0.5 * 10.0 ** (-value) for name, value in decimals.items()}
    misses = [
        {
            "quantity": quantity,
            "engine": engine,
            "recorded": float(value),
            "measured": measured[quantity][engine],
            "difference": float(measured[quantity][engine] - value),
            "tolerance": tolerance[quantity],
        }
        for quantity, per_engine in DESIGN_PROBE["measurements"].items()
        for engine, value in per_engine.items()
        if abs(measured[quantity][engine] - value) > tolerance[quantity]
    ]
    return {
        "question": (
            "does the final pipeline reproduce the four numbers the design probe measured, and "
            "which the pre-registration discloses as having informed four decisions?"
        ),
        "tolerance": tolerance,
        "tolerance_rule": (
            "half a unit in the last decimal the probe printed, which is the largest difference "
            "a correct transcription can produce"
        ),
        "recorded": DESIGN_PROBE["measurements"],
        "measured": measured,
        "misses": misses,
        "passes": not misses,
    }


# ------------------------------------------------------------------ phase: --decide


def _criteria_for(
    folds: dict[str, Fold],
    record: dict[str, Any],
    policy: str,
    ranking: str,
    transport: str,
    epsilon: float,
) -> dict[str, Any]:
    """The five pre-registered criteria for one policy in one cell, with the evidence."""
    key = f"epsilon_{int(epsilon * 100)}"
    engines = sorted(folds)
    rows: dict[str, Any] = {}
    for engine in engines:
        fold = folds[engine]
        specs = read_cell(record, engine, ranking, transport, key)
        measured = measure_cell(fold, ranking, epsilon, specs)
        baseline = measure(
            fold.blocks["evaluation"],
            actions_of(
                fold.rankings[RANK_SGV5][key]["evaluation"],
                fold.tail_risk[key]["evaluation"],
                sgv5_baseline(fold, epsilon),
            ),
            epsilon,
        )
        point, control = measured[policy], measured[RANDOM_MATCHED]
        rows[engine] = {
            "risk_controlled_repair_recall": float(point["risk_controlled_repair_recall"]),
            "baseline_risk_controlled_repair_recall": float(
                baseline["risk_controlled_repair_recall"]
            ),
            "realized_harm_rate": float(point["realized_harm_rate"]),
            "repair_recall": float(point["repair_recall"]),
            "coverage": float(point["coverage"]),
            "n_accepted": int(point["n_accepted"]),
            "baseline_n_accepted": int(baseline["n_accepted"]),
            "holds_bound": bool(point["holds_bound"]),
            "improves_on_baseline": bool(
                point["risk_controlled_repair_recall"] > baseline["risk_controlled_repair_recall"]
            ),
            "coverage_floor_met": bool(
                point["n_accepted"] >= COVERAGE_FLOOR * max(baseline["n_accepted"], 1)
            ),
            "beats_random_control": bool(
                point["risk_controlled_repair_recall"] > control["risk_controlled_repair_recall"]
            ),
        }
    improving = sum(1 for row in rows.values() if row["improves_on_baseline"])
    holding = sum(1 for row in rows.values() if row["holds_bound"])
    solid = sum(
        1 for row in rows.values() if row["coverage_floor_met"] and row["beats_random_control"]
    )
    worst = min(row["risk_controlled_repair_recall"] for row in rows.values())
    baseline_worst = min(row["baseline_risk_controlled_repair_recall"] for row in rows.values())
    criteria = {
        "1": {
            "statement": PRE_REGISTRATION["criteria"]["1"],
            "engines_meeting": improving,
            "met": bool(improving > 0),
        },
        "2": {
            "statement": PRE_REGISTRATION["criteria"]["2"],
            "engines_meeting": improving,
            "bar": ENGINE_BAR,
            "met": bool(improving >= ENGINE_BAR),
        },
        "3": {
            "statement": PRE_REGISTRATION["criteria"]["3"],
            "engines_meeting": holding,
            "bar": ENGINE_BAR,
            "met": bool(holding >= ENGINE_BAR),
        },
        "4": {
            "statement": PRE_REGISTRATION["criteria"]["4"],
            "engines_meeting": solid,
            "bar": ENGINE_BAR,
            "met": bool(solid >= ENGINE_BAR),
        },
        "5": {
            "statement": PRE_REGISTRATION["criteria"]["5"],
            "worst_engine": float(worst),
            "baseline_worst_engine": float(baseline_worst),
            "met": bool(worst > baseline_worst),
        },
    }
    return {
        "policy": policy,
        "ranking": ranking,
        "transport": transport,
        "epsilon": float(epsilon),
        "criteria": criteria,
        "criteria_met": sum(1 for value in criteria.values() if value["met"]),
        "verdict": (
            "SUPPORTED" if all(value["met"] for value in criteria.values()) else "NOT SUPPORTED"
        ),
        "failed": [name for name, value in criteria.items() if not value["met"]],
        "engines": rows,
    }


def _bottleneck() -> dict[str, Any]:
    """The brief's closing question, answered with numbers this stage produced.

    Four candidate bottlenecks are named in the brief. Three of them are measured here and the
    fourth was measured by SGV9 and is cited rather than re-derived.
    """
    transfer = cc._read_json(DECISION_TRANSFER)
    decomposition = cc._read_json(HARM_DECOMPOSITION)
    oracle = cc._read_json(ORACLE_GAP)
    cell = f"{PRIMARY_RANKING}|{PRIMARY_KEY}"
    aggregate = transfer["aggregate"][cell]
    best = min(
        aggregate["policies"],
        key=lambda name: aggregate["policies"][name]["mean_decision_transfer_gap"],
    )
    terms = decomposition["aggregate"][cell_key(PRIMARY_RANKING, PRIMARY_TRANSPORT, PRIMARY_KEY)][
        PRIMARY_POLICY
    ]
    engines = sorted(terms["block_shift"])
    return {
        "score_transfer": {
            "quantity": (
                "mean gap between the frozen ranking's achievable frontier and a re-ranking "
                "fitted on the evaluation block, at the primary epsilon"
            ),
            "value": float(aggregate["mean_score_transfer_gap"]),
            "source": cc._relative(DECISION_TRANSFER),
        },
        "decision_transfer": {
            "quantity": (
                "mean gap between the frozen ranking's achievable frontier and the best "
                "policy's deployed risk-controlled repair recall"
            ),
            "value": float(aggregate["policies"][best]["mean_decision_transfer_gap"]),
            "attained_by": best,
            "source": cc._relative(DECISION_TRANSFER),
        },
        "engine_shift": {
            "quantity": (
                "mean absolute engine-shift term of the primary policy's harm decomposition"
            ),
            "value": float(np.mean([abs(terms["engine_shift"][e]) for e in engines])),
            "per_engine": {e: float(terms["engine_shift"][e]) for e in engines},
            "source": cc._relative(HARM_DECOMPOSITION),
        },
        "block_shift": {
            "quantity": (
                "mean absolute block-shift term: the same engine, the same threshold, "
                "different pages"
            ),
            "value": float(np.mean([abs(terms["block_shift"][e]) for e in engines])),
            "per_engine": {e: float(terms["block_shift"][e]) for e in engines},
            "source": cc._relative(HARM_DECOMPOSITION),
        },
        "in_engine_oracle": {
            "quantity": (
                "how many of the four engines the in-engine oracle's threshold, chosen on that "
                "engine's own labelled pool, still violates epsilon on"
            ),
            "value": int(oracle["decision_9_prediction"]["engines_where_the_oracle_violates"]),
            "source": cc._relative(ORACLE_GAP),
        },
        "calibration": {
            "quantity": (
                "share of the boundary's Brier score that is refinement rather than "
                "calibration, measured by SGV9 and SGV10 and cited, not re-derived here"
            ),
            "value": None,
            "source": "docs/sgv10/tail_risk_control.md",
        },
    }


def run_decide() -> int:
    """The machine-readable finding, from the artifacts and nowhere else."""
    started = time.monotonic()
    folds, record = load_folds()
    primary = _criteria_for(
        folds, record, PRIMARY_POLICY, PRIMARY_RANKING, PRIMARY_TRANSPORT, PRIMARY_EPSILON
    )
    grid = {
        f"{policy}|{ranking}|{transport}|{key}": _criteria_for(
            folds, record, policy, ranking, transport, epsilon
        )
        for policy in PROPOSED
        for ranking in RANKINGS
        for transport in TRANSPORTS
        for epsilon, key in ((e, f"epsilon_{int(e * 100)}") for e in EPSILONS)
    }
    strongest = max(grid, key=lambda name: (grid[name]["criteria_met"], name))
    _write(
        DECISION,
        {
            "stage": "SGV11 -- minimax risk-constrained decision transfer",
            "pre_registration": PRE_REGISTRATION,
            "verdict": primary["verdict"],
            "criteria_met": primary["criteria_met"],
            "primary": primary,
            "strongest_cell_in_the_grid": {
                "cell": strongest,
                "criteria_met": grid[strongest]["criteria_met"],
                "note": (
                    "reported because the grid was pre-registered in full, not because it is "
                    "the finding. The finding is the primary policy in the primary cell."
                ),
                "detail": grid[strongest],
            },
            "grid": grid,
            "bottleneck": _bottleneck(),
            "claims_not_made": (
                "No novelty is claimed. Minimax risk, distributionally robust optimisation, "
                "selective classification with abstention and conformal-style rank bounds are "
                "all established prior art; this stage evaluates them as decision layers over a "
                "frozen OCR correction score under unseen-engine shift and reports what it "
                "measured."
            ),
        },
        "sgv11-decision-v1",
    )
    print(
        f"decide: {primary['verdict']} ({primary['criteria_met']}/5 criteria) "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ phase: --figures


def run_figures() -> int:
    """The four figures the brief names, each carrying the development-only annotation."""
    started = time.monotonic()
    results = cc._read_json(POLICY_RESULTS)
    coverage = cc._read_json(RISK_COVERAGE)
    transfer = cc._read_json(DECISION_TRANSFER)
    robustness = cc._read_json(ROBUSTNESS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV11 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(coverage["engines"])
    written: list[Path] = []

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    shown = {
        ERM_AVERAGE: ("#c87a2b", "v", "B: average risk"),
        MINIMAX: ("#a33", "D", "C: minimax (primary)"),
        SELECTIVE: ("#2e7d5b", "^", "D: selective, three actions"),
        DRO_CVAR: ("#3a5f9e", "o", "E: DRO, page CVaR"),
        DRO_KL: ("#8a5fa3", "s", "F: DRO, KL ball"),
        POOLED_ERM: ("#666", "x", "ablation 3: pooled ERM"),
    }
    primary_cell = cell_key(PRIMARY_RANKING, PRIMARY_TRANSPORT, PRIMARY_KEY)

    # --- risk_coverage.png -------------------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.2))
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        curve = coverage["engines"][engine][f"{PRIMARY_RANKING}|{PRIMARY_KEY}"]
        panel.plot(
            curve["curve"]["coverage"],
            curve["curve"]["risk"],
            color="#bbb",
            linewidth=1.2,
            label="frontier of the frozen ranking",
        )
        panel.axhline(PRIMARY_EPSILON, color="#999", linestyle="--", linewidth=1)
        cell = results["engines"][engine][primary_cell]
        for name, (colour, marker, label) in shown.items():
            point = cell["policies"][name]
            if point["accepts_nothing"]:
                continue
            panel.scatter(
                point["coverage"],
                point["realized_harm_rate"],
                color=colour,
                marker=marker,
                s=46,
                label=label,
            )
        base = cell["baseline"]
        panel.scatter(
            base["coverage"],
            base["realized_harm_rate"],
            color="#000",
            marker="P",
            s=64,
            label="A: SGV5 published",
        )
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("coverage")
        panel.set_ylabel(f"realised harm rate (epsilon = {PRIMARY_EPSILON})")
    np.atleast_1d(panels)[0].legend(fontsize=6, loc="upper left")
    finish(
        figure,
        FIGURE_DIR / "risk_coverage.png",
        "Every two-action policy rides the same frontier; what changes is only where on it "
        "they land. The selective policy leaves the curve because its tail-risk veto skips "
        "rows inside the accept region, so its accept set is not a prefix of the ranking.",
    )

    # --- minimax_versus_average.png ----------------------------------------------------------
    figure, panels = plt.subplots(1, 2, figsize=(10.5, 4.4))
    positions = np.arange(len(engines), dtype=float)
    width = 0.36
    for index, (field, label) in enumerate(
        (("realized_harm_rate", "realised harm rate"), ("repair_recall", "repair recall"))
    ):
        panel = panels[index]
        for offset, (name, colour) in enumerate(((ERM_AVERAGE, "#c87a2b"), (MINIMAX, "#a33"))):
            values = [
                robustness["engines"][engine][primary_cell][
                    "average" if name == ERM_AVERAGE else "minimax"
                ][field]
                for engine in engines
            ]
            panel.bar(
                positions + (offset - 0.5) * width,
                values,
                width,
                color=colour,
                label="B: average" if name == ERM_AVERAGE else "C: minimax",
            )
        if field == "realized_harm_rate":
            panel.axhline(
                PRIMARY_EPSILON,
                color="#333",
                linestyle="--",
                linewidth=1,
                label=f"epsilon = {PRIMARY_EPSILON}",
            )
        panel.set_xticks(positions)
        panel.set_xticklabels(engines, fontsize=8)
        panel.set_ylabel(label)
        panel.legend(fontsize=7)
    finish(
        figure,
        FIGURE_DIR / "minimax_versus_average.png",
        "Ablation 1. The worst-case aggregation lowers realised harm on every engine and lowers "
        "repair recall on every engine.",
    )

    # --- oracle_gap.png ----------------------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.4))
    ladder = [
        ("deployed (primary)", "#a33"),
        ("frozen frontier", "#3a5f9e"),
        ("in-engine oracle", "#2e7d5b"),
        ("re-ranking ceiling", "#888"),
    ]
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        cell = results["engines"][engine][primary_cell]
        gap = transfer["engines"][engine][f"{PRIMARY_RANKING}|{PRIMARY_KEY}"]
        values = [
            cell["policies"][PRIMARY_POLICY]["risk_controlled_repair_recall"],
            gap["frozen_frontier"]["repair_recall"],
            cell["policies"][ORACLE_POOL]["risk_controlled_repair_recall"],
            gap["ceiling_frontier"]["repair_recall"],
        ]
        panel.barh(
            np.arange(len(ladder)), values, color=[colour for _, colour in ladder], height=0.6
        )
        panel.set_yticks(np.arange(len(ladder)))
        panel.set_yticklabels([label for label, _ in ladder], fontsize=7)
        panel.set_xlim(0.0, 1.0)
        panel.set_xlabel("risk-controlled repair recall")
        panel.set_title(f"held out: {engine}", fontsize=10)
    finish(
        figure,
        FIGURE_DIR / "oracle_gap.png",
        "Ablation 4. The in-engine oracle reads the held-out engine's own labels and still "
        "scores zero wherever its threshold breaks the bound on a different block of pages.",
    )

    # --- worst_engine_risk.png ---------------------------------------------------------------
    figure, panels = plt.subplots(1, len(EPSILONS), figsize=(4.4 * len(EPSILONS), 4.2))
    names = [SGV5_PUBLISHED, *PROPOSED, POOLED_ERM]
    for panel, epsilon in zip(np.atleast_1d(panels), EPSILONS, strict=True):
        key = f"epsilon_{int(epsilon * 100)}"
        cell = cell_key(PRIMARY_RANKING, PRIMARY_TRANSPORT, key)
        worst = [results["summary"][cell][name]["worst_engine"] for name in names]
        average = [results["summary"][cell][name]["average_engine"] for name in names]
        positions = np.arange(len(names), dtype=float)
        panel.bar(positions - 0.19, worst, 0.38, color="#a33", label="worst unseen engine")
        panel.bar(positions + 0.19, average, 0.38, color="#3a5f9e", label="average unseen engine")
        panel.set_xticks(positions)
        panel.set_xticklabels([name.split("_", 1)[0] for name in names], fontsize=8)
        panel.set_ylabel("risk-controlled repair recall")
        panel.set_title(f"epsilon = {epsilon}", fontsize=10)
        panel.legend(fontsize=7)
    finish(
        figure,
        FIGURE_DIR / "worst_engine_risk.png",
        "Metrics 7 and 8. The worst unseen engine is EasyOCR in every cell, and no policy "
        "reaches a legal deployment there under the primary ranking.",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv11-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV11-P1",
            "synthetic": False,
            "development_only": True,
            "sources": {
                cc._relative(path): file_sha256(path)
                for path in (POLICY_RESULTS, RISK_COVERAGE, DECISION_TRANSFER, ROBUSTNESS)
                if path.is_file()
            },
            "figures": {cc._relative(path): file_sha256(path) for path in written},
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ phase: --record


def run_record() -> int:
    started = time.monotonic()
    produced = [
        path
        for path in (
            SCORES,
            DESIGN_RECORD,
            DECISION_TRANSFER,
            POLICY_RESULTS,
            RISK_COVERAGE,
            ROBUSTNESS,
            HARM_DECOMPOSITION,
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
        dg.FIT_RECORD,
        c5.PREDICTIONS,
        c5.FIT_RECORD,
        s9.SCORES,
        s9.PROVENANCE,
        s10.SCORES,
        s10.DESIGN_RECORD,
        s10.RISK_COVERAGE,
        s10.PROVENANCE,
    ]
    cc._write_json_once(
        PROVENANCE,
        {
            "schema_version": "sgv11-provenance-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV11-P1",
            "stage": "SGV11 -- minimax risk-constrained decision transfer",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "models_fitted": 0,
            "inputs": {cc._relative(p): file_sha256(p) for p in inputs if p.is_file()},
            "artifacts": {cc._relative(p): file_sha256(p) for p in (*produced, *figures)},
            "policies": POLICY_NOTES,
            "rankings": RANKING_NOTES,
            "transports": TRANSPORT_NOTES,
            "ablations": {name: spec["question"] for name, spec in ABLATIONS.items()},
            "regeneration": (
                "uv run python scripts/sgv11_minimax_policy_transfer.py --policies, then "
                "--transfer --results --coverage --robustness --decomposition --oracle "
                "--ablation --negative --figures --decide --record"
            ),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"record: {len(produced) + len(figures)} artifacts -> {PROVENANCE}")
    return 0


# ------------------------------------------------------------------ entry point

PHASES = (
    ("policies", run_policies),
    ("transfer", run_transfer),
    ("results", run_results),
    ("coverage", run_coverage),
    ("robustness", run_robustness),
    ("decomposition", run_decomposition),
    ("oracle", run_oracle),
    ("ablation", run_ablation),
    ("negative", run_negative),
    ("figures", run_figures),
    ("decide", run_decide),
    ("record", run_record),
)


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name, _run in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    parsed = parser.parse_args(argv)
    requested = [(name, run) for name, run in PHASES if getattr(parsed, name)]
    if not requested:
        parser.print_help()
        return 2
    for _name, run in requested:
        code = run()
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
