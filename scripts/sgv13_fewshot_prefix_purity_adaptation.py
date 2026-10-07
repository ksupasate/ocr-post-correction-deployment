#!/usr/bin/env python3
"""SGV13: how many target-engine labels does the accepted PREFIX actually cost?

SGV12 ended by locating the loss. Between SGV5's frozen ranking and SGV10's re-ranking ceiling
the mean held-out AUROC moves 0.9357 -> 0.9483, which is almost nothing, while the achievable
risk-controlled repair frontier moves 0.5059 -> 0.8124, which is almost everything. At matched
coverage on docTR the ceiling's accepted set holds 0.0034 harmful rows against the baseline's
0.0835 and captures 0.9636 of the repairs against 0.5818, and the two accepted sets OVERLAP by
0.9199. The gap is not a disagreement about the global order. It is a disagreement about roughly
a tenth of the accepted set.

SGV12 also closed the source-only route to it: every source-fitted ranking objective made the
held-out ordering significantly WORSE, the hypothesis class bound before the objective did, and
shuffling the environment labels beat the real partition. What SGV12 did not test is the one
thing SGV6 showed can move: labels from the engine being deployed on.

    SGV13-P1: a small, strategically acquired budget of target-engine labels, spent on the
    candidates that decide the accepted prefix, recovers a materially larger share of SGV12's
    actionable frontier gap than the same budget spent on generic few-shot adaptation.

**The hypothesis is recorded as SGV13-P1.** `docs/sgv1/protocol.md` binds SGV1-H1..H4 and says a
frozen ID is never reused for a different claim; every stage since SGV2 has opened its own
family for the same reason. The brief's RQ1..RQ4 are carried as sub-claims P1a (target labels
recover prefix purity), P1b (prefix-focused acquisition beats generic acquisition at matched
budget), P1c (the mechanism differs by engine) and P1d (the label cost of each recovery
milestone is measurable), all under P1.

Twelve decisions fix what the numbers below can mean.

**1. Nothing upstream moves and no feature is added.** SGV1's design matrix, SGV5's frozen
per-fold selection, SGV6's adaptation pool, SGV9's inner folds, SGV10's ceiling and SGV11's and
SGV12's gap ladders are read and hash-verified against the records their own stages wrote.
SGV5's selected model is rebuilt through SGV12's `build_state`, which asserts its decision score
equals the published `arm__sgv5` vector bit for bit. `features_added` is 0 and the leakage suite
walks this file's syntax tree to confirm no column is constructed here.

**2. The budget pays for everything.** A target label spent on fitting, on calibrating, on
picking a shrinkage, on picking an arm or on placing a cut is the same label and is counted
once. Every choice any arm makes on the target side is made inside the drawn budget by grouped
cross-validation over the budget's own DOCUMENTS. `budget_inventory.json` records rows and
documents for every draw, and the leakage suite asserts no arm reads a pool row outside its
budget.

**3. Acquisition is outcome-free by construction, not by inspection.** The object an acquisition
rule is handed is a `PoolView`, whose `__slots__` contain features, frozen scores, documents,
sites and candidate ids and contain NO outcome field. The labels exist only on the `Budget` that
`purchase` returns, after the rows have been chosen. The leakage suite asserts the slot list and
walks every acquisition function's syntax tree for `harmful`, `beneficial` and `purchase`.

**4. At N = 0 every arm IS SGV5, bit for bit.** Not approximately, not up to a rounding. Each
arm's construction contains the frozen model exactly -- a zero correction, an empty pair set, an
unfittable calibrator -- and `--adapt` asserts a maximum absolute difference of 0.0 against the
frozen score and an identical accepted set at every epsilon, on every fold, for every arm and
every acquisition rule. That is what makes a measured gain a statement about the labels rather
than about having more knobs.

**5. The cut rule is a declared axis, not a hidden one.** Every arm is deployed under two
pre-registered cut rules: `frozen_source`, the published `select_threshold` call on the SOURCE
engines' calibration rows that produced SGV5's own threshold, which isolates the ranking; and
`target_budget`, the same call on the budget's own labelled rows, which is what a deployment
with N labels can actually do. The PRIMARY endpoint uses `target_budget`, because decision 2
says the budget pays for the threshold too. Both are computed for every arm, which is what makes
the mechanism decomposition a set of controlled counterfactuals rather than an argument from the
arm's name.

**6. The prefix is defined from source-side quantities and frozen scores only.** The acceptance
SHARE comes from the fraction of SOURCE calibration rows the published threshold accepts at the
primary epsilon -- SGV12's `_relevant_prefix`, imported rather than restated -- and only which
rows carry the mark depends on the block. The adjacent band is a multiple of that share, chosen
on source-side simulation and fixed before any outer target row is read.

**7. The proposed arm is a correction, so with no target evidence it corrects nothing.** A5's
objective is a source pairwise term plus a target pairwise term plus a prefix-swap term over a
correction beta added to the FROZEN score. The source term is an anchor on the correction, not
a refit of the source model, so it is present only when a target term is: at N = 0 the
correction is exactly zero and A5 is SGV5. This is a stated rule and `--negative` asserts it.

**8. Two arms are allowed to move the hypothesis class, and they are expensive, so they carry
fewer draws.** A4 refits SGV5's own model class jointly on source rows plus the weighted budget;
A6 refits it at raised capacity, because SGV12's ablation A found capacity binding before
objective. They cost 13.1 s and 18.0 s per cell against 0.09 s for the proposed arm, and the SGV6
comparison costs up to about 25 s at N = 1000, so all three run at JOINT_DRAWS draws on the
random rule and the rule the design froze rather than DRAWS on seven. The joint refit the design
does NOT select is reduced further, to the random rule alone on a four-point ladder; the one it
DOES select carries every budget, because criterion 4 is a label-efficiency criterion and cannot
be read on four points against a baseline measured on nine. Every reduced budget is also a
full-ladder budget, so every comparison is at matched N. The reduction is declared here, recorded
in the design record and the budget inventory, and their intervals are reported wider rather than
being compared as if they were not.

**9. The SGV6 comparison is SGV6's own code, on the same rows.** `sgv6_generic` calls
`s6.fit_arm` and `s6.select_arm` over `s6.ADAPTER_ARMS`, on an `s6.Budget` drawn from an
`s6.TargetPool` with THIS stage's acquisition order, so the two procedures see the identical
labelled candidates at every budget. A difference between them cannot be a difference in which
rows were bought.

**10. Everything is selected on source-side simulation before any outer target row is read.**
`--preregister` replays the whole few-shot protocol with each SOURCE engine playing target
inside SGV9's inner folds, and that is the only place the band multiple, the target weight, the
pair weights, the ridge, the acquisition rule, the arm and the minimum coverage floor are
chosen. The outer evaluation is then run once. The design record is written before `--adapt` and
its hash is carried into every downstream artifact.

**11. Six criteria, all required, and the label-reduction bar is 25%.** Fixed before any outer
number existed: operational improvement over the SGV6-style baseline at epsilon = 0.10;
favourable on at least 3 of 4 engines; the realised harm bound held on at least 3 of 4;
50% gap recovery at at least 25% fewer labels than the generic baseline; a measurable prefix
purity improvement; and a non-degenerate accepted set. The primary test family is twenty tests,
Holm-corrected, document-clustered and paired.

**12. The gap ladder is SGV12's, not a new one.** The denominator of every recovery fraction is
`Frontier_ceiling - Frontier_baseline` read from SGV12's `gap_decomposition.json`, which read it
from SGV11's, which read the ceiling from SGV10's published `oracle_eval` re-ranking. Three
stages, one ladder. `--gap` re-derives it and asserts it reproduces.

    --preregister  the source-side simulation, and the design frozen from it
    --adapt        every arm at every budget under every acquisition rule; the score table
    --results      the operational endpoints: repair recall, harm, coverage, accepted sets
    --prefix       accepted-prefix purity, overlap, and the harmful-to-beneficial swaps
    --gap          gap recovery against SGV12's ladder, and the label cost of each milestone
    --mechanism    which of score scale, ranking, representation and cut is doing the work
    --acquisition  the six acquisition rules against each other and against random
    --sgv6         SGV6's own procedure, on the same rows, at the same budgets
    --oracle       the three diagnostic upper bounds; never used for selection
    --easyocr      the pre-specified EasyOCR mechanism analysis
    --negative     the brief's fifteen falsification tests
    --stats        the pre-registered family: document-clustered paired bootstrap and Holm
    --figures      the six required figures
    --decide       the machine-readable finding
    --record       provenance, the dependency audit, and the traceability index

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents. The target labels this stage spends come
from TRAIN documents of the held-out engine; no evaluation document contributes a label to any
fit, any calibration, any threshold, any shrinkage, any arm choice or any acquisition ordering.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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
import sgv11_minimax_policy_transfer as s11
import sgv12_robust_ranking_transfer as s12
from ocr_risk.io.hashing import file_sha256

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv13_fewshot_prefix_purity_adaptation"

SCORES = OUT / "adaptation_scores.parquet"
COUNTS = OUT / "deployment_counts.parquet"
DESIGN_RECORD = OUT / "design_record.json"
FROZEN_IDENTITY = OUT / "frozen_identity.json"
BUDGET_INVENTORY = OUT / "budget_inventory.json"
ADAPTATION_RESULTS = OUT / "adaptation_results.json"
PREFIX_PURITY = OUT / "prefix_purity.json"
GAP_RECOVERY = OUT / "gap_recovery.json"
MECHANISM = OUT / "mechanism_decomposition.json"
ACQUISITION_RESULTS = OUT / "acquisition_results.json"
SGV6_COMPARISON = OUT / "sgv6_comparison.json"
ORACLE_ANALYSIS = OUT / "oracle_analysis.json"
EASYOCR_ANALYSIS = OUT / "easyocr_analysis.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
MODEL_SELECTION = OUT / "model_selection.json"
DECISION = OUT / "research_decision.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

SCHEMA_VERSION = 1
STAGE = "sgv13_fewshot_prefix_purity_adaptation"
HYPOTHESIS = "SGV13-P1"

# ------------------------------------------------------------------ imported, never restated
#
# Every quantity below has exactly one implementation in this repository. Seven stages now
# import these rather than restating them and `tests/leakage` asserts the identity by `is`.

achievable_repair_recall = policy.repair_recall_at_bounded_harm
deployed_point = c5._deployed
risk_controlled_recall = s11.risk_controlled_recall
document_resamples = s9.document_resamples
document_multiplicities = s11.document_multiplicities
_interval = s9._interval
_stable_seed = s6._stable_seed
inner_fold = s9.inner_fold
build_fold = rl.build_fold
relevant_prefix = s12._relevant_prefix
place_cut = s12.place_cut
deployed = s12.deployed
frontier = s12.frontier
ranking_quality = s12.ranking_quality
epsilon_key = s12.epsilon_key
build_state = s12.build_state
fit_weighted = s6._fit_weighted
logit = s6._logit
sigmoid = s6._sigmoid

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
PRIMARY_KEY = epsilon_key(PRIMARY_EPSILON)
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = pilot.BOOTSTRAP_SEED

FIT = s12.FIT
CALIBRATION = s12.CALIBRATION
EVALUATION = s12.EVALUATION
POOL = s12.POOL


class PhaseError(RuntimeError):
    """A guard this stage refuses to run past."""


# ------------------------------------------------------------------ the pre-registered grid

# The label ladder. 0 is the frozen identity; 1000 is the bridge to SGV6, whose ladder was
# (0, 10, 25, 50, 100, 250, 500, 1000), so every SGV6 budget is also an SGV13 budget and the
# comparison in `--sgv6` is a comparison at matched N rather than an interpolation.
BUDGETS = (0, 5, 10, 25, 50, 100, 250, 500, 1000)
NONZERO_BUDGETS = tuple(n for n in BUDGETS if n > 0)
SGV6_BRIDGE = 1000

# Repeated draws. Common random seeds across every arm and every acquisition rule, so a paired
# comparison at (engine, budget, draw) is a comparison of two procedures on the same opportunity.
DRAWS = 20
JOINT_DRAWS = 5
# The joint arms refit SGV5's boosted model class on the source rows plus the budget, which
# feasibility probe 3 timed at 13.1 s (A4) and 18.0 s (A6) per cell against 0.09 s for the
# proposed arm. At those costs the full factorial would add about an hour to the run for two
# CONTROL arms, so they run on a coarser ladder under the random acquisition only. Every point on
# the ladder is also a point on the full one, so the comparison with the frozen-feature arms is
# at matched N and never an interpolation. Declared in decision 8.
JOINT_BUDGETS = (0, 10, 250, 1000)
DRAW_SEED = 20260906

# ------------------------------------------------------------------ acquisition rules

Q_RANDOM = "q0_random"
Q_DIVERSITY = "q1_diversity"
Q_UNCERTAINTY = "q2_uncertainty"
Q_DISAGREEMENT = "q3_ranking_disagreement"
Q_PREFIX_SWAP = "q4_prefix_swap"
Q_HYBRID = "q5_diversity_prefix"
Q_RANDOM_DOCUMENT = "q0d_random_document"
ACQUISITIONS = (
    Q_RANDOM,
    Q_RANDOM_DOCUMENT,
    Q_DIVERSITY,
    Q_UNCERTAINTY,
    Q_DISAGREEMENT,
    Q_PREFIX_SWAP,
    Q_HYBRID,
)
STOCHASTIC_ACQUISITIONS = (Q_RANDOM, Q_RANDOM_DOCUMENT)
# The two rules the proposed acquisition must beat to earn a claim about targeting, per the
# brief's section 22: matched random, and random matched on the DOCUMENTS it touches.
RANDOM_CONTROLS = (Q_RANDOM, Q_RANDOM_DOCUMENT)

ACQUISITION_NOTES = {
    Q_RANDOM: (
        "document-grouped random permutation of the pool. The acquisition baseline and the "
        "only rule whose ordering depends on the draw seed; `s6.acquisition_order` verbatim."
    ),
    Q_RANDOM_DOCUMENT: (
        "a random permutation of the pool's DOCUMENTS, then a random permutation of the rows "
        "inside each. The brief's document-matched random control: it spends the same budget "
        "while touching documents in the same profile a document-diverse rule would, so a gain "
        "for Q5 over it cannot be a gain from document spread alone."
    ),
    Q_DIVERSITY: (
        "greedy farthest-point traversal of the frozen adapter representation, started at the "
        "row nearest the pool centroid. `s6.acquisition_order` verbatim, so the diversity rule "
        "SGV6 measured and the diversity rule SGV13 measures are one function."
    ),
    Q_UNCERTAINTY: (
        "distance from the frozen decision score to the deployed threshold, ascending. "
        "`s6.acquisition_order` verbatim. Carried because SGV6's evidence for it was weak and a "
        "baseline that prior evidence disfavours is still a baseline."
    ),
    Q_DISAGREEMENT: (
        "the spread of a row's normalised rank across five SOURCE-fitted rankings SGV12 already "
        "published on this exact pool block. The signal is disagreement about ORDER, not "
        "entropy; no oracle or ceiling column is read and `--negative` asserts it."
    ),
    Q_PREFIX_SWAP: (
        "expected information about a deployment-critical swap: a kernel on rank distance from "
        "the source-transported cut, multiplied by the number of same-site candidates on the "
        "opposite side of that cut. Reads the frozen score and the site structure and nothing "
        "else; it is outcome-free before the label is purchased."
    ),
    Q_HYBRID: (
        "the prefix-swap value taken in document round-robin order, so a page with fifty "
        "near-duplicate candidates contributes its best row before it contributes its second."
    ),
}

# ------------------------------------------------------------------ adaptation arms

A0_FROZEN = "a0_frozen"
A1_CALIBRATION = "a1_target_calibration"
A2_CUT = "a2_target_cut"
A3_HEAD = "a3_target_head"
A4_JOINT = "a4_joint_refit"
A5_PREFIX = "a5_prefix_focused"
A6_CAPACITY = "a6_capacity_refit"
SGV6_GENERIC = "b1_sgv6_generic"
A5_PERMUTED = "c1_prefix_permuted_labels"
A3_PERMUTED = "c2_head_permuted_labels"
A4_PERMUTED = "c3_joint_permuted_labels"
SOURCE_REFIT = "c4_source_only_refit"
ORACLE_CUT = "o1_oracle_cut"
ORACLE_RERANK = "o2_oracle_rerank"

CHEAP_ARMS = (A0_FROZEN, A1_CALIBRATION, A2_CUT, A3_HEAD, A5_PREFIX, SGV6_GENERIC)
JOINT_ARMS = (A4_JOINT, A6_CAPACITY)
CONTROL_ARMS = (A5_PERMUTED, A3_PERMUTED, A4_PERMUTED, SOURCE_REFIT)
ORACLE_ARMS = (ORACLE_CUT, ORACLE_RERANK)
ADAPTATION_ARMS = CHEAP_ARMS + JOINT_ARMS
ALL_ARMS = ADAPTATION_ARMS + CONTROL_ARMS + ORACLE_ARMS
# The arms `--preregister` may choose between as the primary SGV13 method. The oracles, the
# permutation controls and the SGV6 baseline are not candidates for it.
SELECTABLE_ARMS = (A1_CALIBRATION, A2_CUT, A3_HEAD, A4_JOINT, A5_PREFIX, A6_CAPACITY)

ARM_NOTES = {
    A0_FROZEN: (
        "SGV5's frozen model and SGV5's published threshold. Spends no target label at any "
        "budget, so its curve is flat by construction and every other arm's gain is read "
        "against it."
    ),
    A1_CALIBRATION: (
        "ranking frozen; the budget buys a monotone recalibration of the frozen decision score "
        "to a target harm probability, and the cut becomes the absolute calibrated one "
        "`p_hat(harm) <= epsilon`. Isolates mechanism M1, score scale. Under the two shared cut "
        "rules a monotone map commutes with the threshold and this arm degenerates to A0 or A2; "
        "`--negative` asserts the degeneracy rather than leaving it implicit."
    ),
    A2_CUT: (
        "score and ranking frozen; the budget buys only the deployment cut, through the same "
        "`select_threshold` call every other arm uses, on the budget's own labelled rows. "
        "Isolates mechanism M4, decision cut."
    ),
    A3_HEAD: (
        "representation frozen; the budget buys an offset-parameterised ridge correction "
        "`s + w . x` on SGV5's own selected columns, its shrinkage chosen by grouped "
        "cross-validation inside the budget. Isolates a target-specific remap of existing "
        "features; the residual over it is mechanism M3."
    ),
    A4_JOINT: (
        "SGV5's own model class refitted on the source rows plus the weighted budget, with the "
        "target row weight taken from the source-side simulation. The construction is "
        "`s6._fit_weighted`'s and is asserted bit-identical to it; the one difference is that "
        "the outcome labels are passed in rather than read from the design matrix, so that "
        "section 21's permutation control reaches the refit. The arm SGV6's diagnostic evidence "
        "said should be strong."
    ),
    A5_PREFIX: (
        "the proposed method. An offset-parameterised linear correction to the frozen score, "
        "fitted by a pairwise objective with three terms: source pairs as an anchor, target "
        "pairs from the budget, and prefix-swap pairs formed from budget rows that a deployment "
        "would have to exchange. Zero correction at N = 0 by construction."
    ),
    A6_CAPACITY: (
        "A4 with the boosted model's capacity raised, because SGV12's ablation A found the "
        "hypothesis class binding before the objective. Same features, same rows, same weight."
    ),
    SGV6_GENERIC: (
        "SGV6's generic few-shot procedure, called through SGV6's own `fit_arm` and "
        "`select_arm` over its nine adapter families, on an `s6.Budget` carrying THIS stage's "
        "acquisition order. The comparison of record for criterion 4."
    ),
    A5_PERMUTED: (
        "A5 with the budget's outcome labels permuted within the budget before fitting. The "
        "rows, the documents, the acquisition and the class balance are identical; only the "
        "label-to-row assignment is destroyed."
    ),
    A3_PERMUTED: "A3 under the same permutation, so the control is not specific to one arm.",
    SOURCE_REFIT: (
        "the joint refit's construction on the SOURCE rows alone, at unit weight and zero target "
        "labels. It separates two things a joint refit changes at once: the fitting path -- a "
        "weighted re-estimation rather than SGV5's own call -- and the target rows. Whatever "
        "this arm moves is the path; whatever it does not move is the data."
    ),
    A4_PERMUTED: (
        "the joint refit under the same permutation. Section 21 asks for a label-permutation "
        "control on the arm whose gain is being claimed, and which arm that is depends on what "
        "the source-side simulation selects, so all three adaptation families carry one. It "
        "runs on the joint arms' reduced grid because it is a refit and costs what a refit "
        "costs."
    ),
    ORACLE_CUT: (
        "diagnostic only. The frozen ranking with the deepest cut whose realised harm holds "
        "epsilon ON THE EVALUATION BLOCK. This is SGV12's `frontier` and SGV11's rung."
    ),
    ORACLE_RERANK: (
        "diagnostic only. SGV10's published `oracle_eval` re-ranking, the ceiling of the ladder "
        "SGV11 and SGV12 measured against. Never fitted here and never used for selection."
    ),
}

# ------------------------------------------------------------------ cut rules

CUT_FROZEN = "frozen_source"
CUT_BUDGET_EMPIRICAL = "target_budget_empirical"
CUT_BUDGET_LTT = "target_budget_conservative"
CUT_CALIBRATED = "calibrated_absolute"
CUT_ORACLE = "oracle_evaluation"
CUT_RULES = (CUT_FROZEN, CUT_BUDGET_EMPIRICAL, CUT_BUDGET_LTT)
BUDGET_CUTS = (CUT_BUDGET_EMPIRICAL, CUT_BUDGET_LTT)

CUT_NOTES = {
    CUT_FROZEN: (
        "`select_threshold(source calibration score, source calibration harm, epsilon, "
        "delta=DELTA, controller='empirical')` -- the published call that produced SGV5's own "
        "threshold, on the arm's own score. Reads no target row. Isolates the ranking."
    ),
    CUT_BUDGET_EMPIRICAL: (
        "the identical call on the budget's own labelled target rows, under the empirical "
        "controller. What a deployment holding N target labels can do if it treats the budget as "
        "a calibration sample. At N = 0 there is no target row and it falls back to "
        "`frozen_source` exactly."
    ),
    CUT_BUDGET_LTT: (
        "the same, under the Bentkus learn-then-test controller, which certifies the bound at "
        "finite N rather than estimating it. Section 7 asks for conservative estimation where "
        "appropriate; whether it is appropriate at these budgets is measured, not assumed."
    ),
    CUT_CALIBRATED: (
        "accept where the target-calibrated harm probability is at most epsilon. Defined only "
        "for A1, which is the arm whose mechanism is the calibrated level."
    ),
    CUT_ORACLE: "diagnostic only: the deepest evaluation-block prefix whose realised harm holds.",
}

BUDGET_CONTROLLERS = {CUT_BUDGET_EMPIRICAL: "empirical", CUT_BUDGET_LTT: "ltt_bentkus"}
FROZEN_CONTROLLER = "empirical"

# ------------------------------------------------------------------ hyperparameter grids
#
# Every grid below is swept on SOURCE-SIDE SIMULATION only, inside `--preregister`, and the
# choice is frozen into `design_record.json` before `--adapt` reads an outer target row.

BAND_MULTIPLES = (0.5, 1.0, 2.0)
TARGET_WEIGHTS = (1.0, 5.0, 10.0, 25.0)
LAMBDA_TARGET = (1.0, 5.0)
LAMBDA_PREFIX = (1.0, 5.0, 25.0)
RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0)
HEAD_SHRINKAGE = (float("inf"), 1e4, 1e3, 1e2, 1e1, 1.0)
SOURCE_PAIR_BUDGET = 20000
SOURCE_PAIR_FAMILY = s12.P0_GLOBAL
LBFGS_MAXITER = 300
INNER_FOLDS = s6.INNER_FOLDS

CAPACITY_ITERATIONS = 2 * c5.BOOSTING_ITERATIONS
CAPACITY_LEAVES = 63

# The five SOURCE-fitted SGV12 rankings the disagreement rule reads. Every one of them was
# fitted without a single label from the engine it is being read on; the oracle and ceiling
# columns SGV12 also published are deliberately absent and `--negative` asserts their absence.
DISAGREEMENT_COLUMNS = (
    "score__mA_sgv5_frozen",
    "score__mB_pair_linear",
    "score__mC_pair_boosted",
    "score__mD_listwise",
    "score__cB_clf_boosted",
)

# ------------------------------------------------------------------ decision constants

RECOVERY_MILESTONES = (0.25, 0.50, 0.75)
# Section 9's secondary annotation-cost model, reported descriptively and never as an endpoint.
# The per-document coefficient is a grid rather than a number because the repository's protocol
# fixes no annotation cost and inventing one would put a made-up constant into a comparison.
DOCUMENT_COST_GRID = (0.0, 1.0, 5.0)
MILESTONE_FOR_CRITERION_4 = 0.50
LABEL_REDUCTION_BAR = 0.25
ENGINE_BAR = 3
N_ENGINES = 4
PRIMARY_FAMILY_SIZE = 20
MIN_COVERAGE_FLOOR_QUANTILE = 0.5
PROB_FLOOR = s6.PROB_FLOOR


PRE_REGISTRATION = {
    "hypothesis_id": HYPOTHESIS,
    "hypothesis": (
        "a small, strategically acquired budget of target-engine labels, spent on the "
        "candidates that decide the accepted prefix, recovers a materially larger share of "
        "SGV12's actionable frontier gap than the same budget spent on generic few-shot "
        "adaptation"
    ),
    "sub_claims": {
        "P1a": "target labels recover accepted-prefix purity on the held-out engine",
        "P1b": "prefix-focused acquisition beats generic acquisition at matched budget",
        "P1c": "the mechanism that recovers the gap differs between engines",
        "P1d": "the label cost of each recovery milestone is measurable",
    },
    "primary_endpoint": (
        "risk-controlled repair recall on the held-out engine's evaluation block at "
        "epsilon = 0.10 under the `target_budget` cut rule, credited only where the realised "
        "harm bound holds"
    ),
    "secondary_epsilons": [0.05, 0.2],
    "primary_comparison": (
        "the source-selected SGV13 arm against `b1_sgv6_generic` at matched budget, matched "
        "acquisition order, matched adaptation pool and matched evaluation documents"
    ),
    "criteria": {
        "1_operational_improvement": (
            "at epsilon = 0.10 the selected method's mean risk-controlled repair recall over "
            "draws exceeds `b1_sgv6_generic`'s at the selected operating budget"
        ),
        "2_cross_engine_breadth": f"criterion 1 holds on at least {ENGINE_BAR} of {N_ENGINES}",
        "3_safety": (
            f"the realised accepted-set harm rate is at most epsilon on at least {ENGINE_BAR} of "
            f"{N_ENGINES} engines at the selected operating budget, with no credit for an arm "
            "that accepts nothing"
        ),
        "4_label_efficiency": (
            f"the selected method reaches {MILESTONE_FOR_CRITERION_4:.0%} recovery of SGV12's "
            f"actionable gap with at least {LABEL_REDUCTION_BAR:.0%} fewer target labels than "
            "`b1_sgv6_generic` needs for the same milestone"
        ),
        "5_prefix_mechanism": (
            "the improvement is accompanied by a fall in the accepted-prefix harm rate or a "
            f"positive count of harmful-to-beneficial swaps on at least {ENGINE_BAR} of "
            f"{N_ENGINES} engines"
        ),
        "6_non_degenerate": (
            "the selected method's accepted coverage is at least the floor fixed by the "
            "source-side simulation, on at least "
            f"{ENGINE_BAR} of {N_ENGINES} engines"
        ),
    },
    "criteria_total": 6,
    "verdict_rule": "SUPPORTED requires all six criteria; otherwise NOT SUPPORTED",
    "primary_family_size": PRIMARY_FAMILY_SIZE,
    "primary_family": (
        "the selected method against `b1_sgv6_generic` on risk-controlled repair recall at "
        "three epsilons on four engines (12), against `a0_frozen` on risk-controlled repair "
        "recall at the primary epsilon on four engines (4), and against `a0_frozen` on the "
        "accepted-prefix harm rate on four engines (4)"
    ),
    "multiplicity": "Holm at 0.05 over the whole family; document-clustered paired bootstrap",
    "label_reduction_bar": LABEL_REDUCTION_BAR,
    "recovery_milestones": list(RECOVERY_MILESTONES),
    "budgets": list(BUDGETS),
    "draws": DRAWS,
    "joint_draws": JOINT_DRAWS,
    "frozen_before_outer_evaluation": [
        "the adaptation arm families and their construction",
        "the six acquisition rules",
        "the two shared cut rules and which is primary",
        "the prefix band multiple",
        "the source:target weight for the joint arms",
        "the pair weights and ridge for the proposed arm",
        "the head shrinkage grid",
        "the minimum deployment coverage floor",
        "the model-selection strategy",
        "the six criteria and the label-reduction bar",
        "the primary test family and its size",
    ],
}

FEASIBILITY_PROBE = {
    "1_cut_rule_feasibility": {
        "question": "can a budget of N target labels place a deployment cut at all?",
        "what_was_run": (
            "every frozen-feature arm was fitted on the docTR OUTER fold at N in {0, 25, 250} "
            "under the prefix-swap acquisition and deployed under a target-budget cut using the "
            "Bentkus controller. THIS PROBE READ A HELD-OUT ENGINE'S OUTCOME and is disclosed in "
            "full for that reason."
        ),
        "what_was_seen": {
            "n_0": {"coverage": 0.25, "risk_controlled_repair_recall": 0.5818},
            "n_25": {
                "coverage": 0.0,
                "risk_controlled_repair_recall": 0.0,
                "budget_harmful": 23,
                "budget_beneficial": 1,
            },
            "n_250": {
                "coverage": 0.0,
                "risk_controlled_repair_recall": 0.0,
                "budget_harmful": 232,
                "budget_beneficial": 11,
            },
        },
        "why": (
            "a prefix-targeted acquisition is 93% harmful by construction -- it is drawn from "
            "the region a deployment is about to accept -- so it cannot certify a 10% harm bound "
            "at any budget, and every arm's deployment collapsed to accepting nothing. The "
            "targeted budget is a good fitting sample and a bad calibration sample, and that is "
            "a property of the acquisition rather than of any arm."
        ),
        "what_it_changed": (
            "one thing: the cut rule the PRIMARY endpoint is read under stopped being fixed to "
            "the budget rule and became a pre-registered choice made on source-side simulation "
            "in `--preregister` stage 0, from the FROZEN baseline's behaviour on inner folds. "
            "The three rules themselves, every arm, every criterion, every epsilon, the "
            "hypothesis and the label ladder are unchanged, and all three rules are still "
            "computed and published for every cell."
        ),
    },
    "4_the_permutation_control_was_inoperative": {
        "question": "does permuting a budget's labels change what a joint refit sees?",
        "what_was_run": (
            "the first full outer run fitted `c3_joint_permuted_labels` through "
            "`s6._fit_weighted`, which reads the outcome labels for its fitting rows out of the "
            "DESIGN MATRIX. THIS PROBE READ HELD-OUT ENGINE OUTCOMES, because it is a comparison "
            "of two arms on the outer folds, and it is disclosed in full for that reason."
        ),
        "what_was_seen": {
            "max_abs_difference_permuted_minus_unpermuted": 0.0,
            "metrics_compared": [
                "risk_controlled_repair_recall",
                "auroc_safe",
                "pair_accuracy",
                "prefix_harm_rate",
                "coverage",
            ],
            "engines": 4,
            "budgets_checked": [10, 250, 1000],
            "note": (
                "bit-identical on every metric, every engine and every budget. The permutation "
                "changes the `Budget` object and not the design matrix, so a refit that reads "
                "the design sees the true labels either way and the control returns the arm it "
                "is meant to control."
            ),
        },
        "why": (
            "SGV6's function is correct for SGV6 -- its budget rows are exactly the rows whose "
            "labels it may read, so the accounting never breaks. What breaks is the CONTROL: a "
            "null result from it would have looked like evidence that the labels do not matter, "
            "when in fact the labels were never delivered."
        ),
        "what_it_changed": (
            "the joint refits now go through this stage's own `_refit`, which takes the outcome "
            "labels explicitly -- the design's for the source rows, the BUDGET's for the target "
            "rows. `tests/leakage` asserts `_refit` reproduces `s6._fit_weighted` bit for bit "
            "when handed the design's own labels, so no arm's published numbers move; only the "
            "permutation control becomes operative. No criterion, epsilon, arm or hypothesis "
            "changed, and the source-side simulation reproduced identically afterwards."
        ),
    },
    "2_isotonic_ties": {
        "question": "can A1 use isotonic regression and still leave the ranking unchanged?",
        "what_was_run": "the same probe fitted A1 with an isotonic member on docTR at N = 25",
        "what_was_seen": {"auroc_frozen": 0.9898, "auroc_isotonic_member": 0.9063},
        "why": (
            "isotonic regression is monotone but only weakly, so it collapses score intervals "
            "to a single value. The ordering it returns is not the frozen ordering and the "
            "held-out AUROC falls purely from ties, which would make A1 -- the arm whose whole "
            "contract is that the ranking does not move -- a ranking change."
        ),
        "what_it_changed": (
            "A1 registers the strictly monotone Platt member only. Isotonic is recorded as "
            "considered and excluded, not silently dropped."
        ),
    },
    "3_cost": {
        "question": "what does each arm cost per cell, and what grid does that afford?",
        "what_was_run": "one fit of each arm on the docTR fold at N in {0, 25, 250}",
        "what_was_seen": {
            "a1_target_calibration_seconds": 0.02,
            "a3_target_head_seconds": 0.05,
            "a5_prefix_focused_seconds": 0.09,
            "b1_sgv6_generic_seconds": 8.19,
            "a4_joint_refit_seconds": 13.1,
            "a6_capacity_refit_seconds": 18.0,
        },
        "why_it_matters": (
            "decision 8's reduced ladder and draw count for the joint arms comes from these "
            "numbers and from nothing about their performance, which the probe did not read."
        ),
        "what_it_changed": "the joint arms' budget ladder and draw count. No criterion.",
    },
}


DECISIONS = {
    "1_nothing_upstream_moves": (
        "SGV1's matrix, SGV5's selection, SGV6's pool, SGV9's inner folds, SGV10's ceiling and "
        "SGV11's and SGV12's ladders are read and hash-verified. SGV5's decision score is "
        "rebuilt and asserted bit-identical to the published vector. features_added is 0."
    ),
    "2_the_budget_pays_for_everything": (
        "fitting, calibration, shrinkage selection, arm selection and threshold placement all "
        "spend the same labels and are all confined to the drawn budget, by grouped "
        "cross-validation over the budget's own documents."
    ),
    "3_acquisition_is_outcome_free_by_construction": (
        "acquisition rules are handed a `PoolView` with no outcome slot; labels appear only on "
        "the `Budget` that `purchase` returns."
    ),
    "4_n_zero_is_sgv5_bit_for_bit": (
        "every arm contains the frozen model exactly and `--adapt` asserts a maximum absolute "
        "score difference of 0.0 and an identical accepted set at every epsilon."
    ),
    "5_the_cut_rule_is_a_declared_axis": (
        "every arm is deployed under all three cut rules and which one the PRIMARY endpoint is "
        "read under is chosen in `--preregister` from the frozen baseline's behaviour on inner "
        "folds, because feasibility probe 1 found a targeted budget is not a calibration sample. "
        "All three are recorded for every cell, which is what makes the mechanism decomposition "
        "a set of controlled counterfactuals rather than an argument from the arm's name."
    ),
    "6_the_prefix_comes_from_source_quantities": (
        "the acceptance share is the fraction of SOURCE calibration rows the published "
        "threshold accepts at the primary epsilon (SGV12's `_relevant_prefix`); the band is a "
        "multiple of it chosen on source-side simulation."
    ),
    "7_the_proposed_arm_corrects_nothing_without_evidence": (
        "A5's source pairwise term anchors a correction rather than refitting a model, so it "
        "is present only when a target term is. With an empty budget the correction is exactly "
        "zero."
    ),
    "8_the_expensive_arms_carry_fewer_draws": (
        f"A4 at 13.1 s and A6 at 18.0 s per cell, and the SGV6 comparison at up to about 25 s at "
        f"N = 1000, against 0.09 s for the proposed arm. All three therefore run at {JOINT_DRAWS} "
        f"draws -- the replication SGV6 itself ran -- under the random rule and the rule the "
        f"design froze, where the frozen-feature arms run at {DRAWS} draws on "
        f"{len(ACQUISITIONS)} rules. The joint refit the design did NOT select is reduced "
        f"further, to the random rule alone on the {len(JOINT_BUDGETS)}-point ladder "
        f"{JOINT_BUDGETS}; the one it DID select carries all {len(BUDGETS)} budgets, because "
        "criterion 4 is a label-efficiency criterion and cannot be read on four points against a "
        "baseline measured on nine. Every reduced budget is also a full-ladder budget, so every "
        "comparison is at matched N. Their intervals are wider and are reported as such."
    ),
    "9_the_sgv6_comparison_is_sgv6_code": (
        "`b1_sgv6_generic` calls `s6.fit_arm` and `s6.select_arm` on an `s6.Budget` built from "
        "this stage's acquisition order, so the two procedures buy the identical rows."
    ),
    "10_everything_is_selected_on_source_simulation": (
        "`--preregister` replays the protocol with each source engine playing target inside "
        "SGV9's inner folds; the outer evaluation runs once afterwards."
    ),
    "11_six_criteria_all_required": (
        "operational improvement, cross-engine breadth, safety, label efficiency at a 25% "
        "reduction bar, a prefix-purity mechanism, and a non-degenerate accepted set."
    ),
    "12_the_gap_ladder_is_sgv12s": (
        "the recovery denominator is SGV12's `Frontier_ceiling - Frontier_baseline`, which came "
        "from SGV11's, which read the ceiling from SGV10's published `oracle_eval`."
    ),
}


# ------------------------------------------------------------------ the freeze


def verify_frozen() -> dict[str, Any]:
    """Every upstream artifact this stage reads, checked before a single model is fitted.

    SGV12's own verification is the base -- the same nine artifacts, checked against the same
    records -- and SGV13 adds the three it depends on that SGV12 did not: SGV6's adaptation
    scores and its provenance manifest, and SGV12's own score table and gap decomposition.
    Nothing is recomputed from a number typed into this file.
    """
    base = s12.verify_frozen()
    extra = [
        s12._hash_against(s6.SCORES, s6.PROVENANCE),
        s12._hash_against(s6.ADAPTATION_RESULTS, s6.PROVENANCE),
        s12._hash_against(s12.SCORES, s12.PROVENANCE),
        s12._hash_against(s12.GAP_DECOMPOSITION, s12.PROVENANCE),
        s12._hash_against(s12.DESIGN_RECORD, s12.PROVENANCE),
    ]
    ladder = cc_read_json(s12.GAP_DECOMPOSITION)["aggregate"][PRIMARY_KEY]
    base["artifacts"] = list(base["artifacts"]) + extra
    base["sgv12_published_ladder"] = {
        "baseline_frontier": float(ladder["mean_baseline_frozen_frontier"]),
        "ceiling_frontier": float(ladder["mean_ceiling_frontier"]),
        "actionable_gap": float(ladder["mean_score_transfer_gap"]),
        "note": (
            "the denominator of every recovery fraction in this stage, read from SGV12's "
            "artifact. SGV12 read it from SGV11's and SGV11 read the ceiling from SGV10's "
            "published `oracle_eval` re-ranking; the three stages share one ladder."
        ),
    }
    # Every SGV12 number this stage's report quotes is read here rather than typed there, so a
    # decimal in the motivation traces to an SGV13 artifact exactly like a decimal in the results.
    ranking = cc_read_json(s12.RANKING_METRICS)["aggregate"]
    prefix = cc_read_json(s12.NEGATIVE_TESTS)["failure_tests"]["3_global_versus_prefix"][
        "per_engine"
    ]
    ceiling_key = f"{s12.CEILING}__{PRIMARY_KEY}"
    base["sgv12_published_ranking"] = {
        "frozen_mean_auroc_unseen_engine": float(ranking[s12.SGV5_FROZEN]["mean_auroc_setting_c"]),
        "ceiling_mean_auroc_unseen_engine": float(
            cc_read_json(s12.RANKING_METRICS)["oracles"][ceiling_key]["mean_auroc_setting_c"]
        ),
        "ceiling_mean_pair_accuracy_unseen_engine": float(
            cc_read_json(s12.RANKING_METRICS)["oracles"][ceiling_key][
                "mean_pair_accuracy_setting_c"
            ]
        ),
        "selected_mean_auroc_unseen_engine": float(ranking[s12.SELECTED]["mean_auroc_setting_c"]),
        "engines_where_the_selected_ranking_improved_auroc": int(
            ranking[s12.SELECTED]["engines_improving_auroc"]
        ),
        "note": (
            "SGV12's held-out ranking quality on unseen engines, read from its artifact. The "
            "ceiling entries are SGV10's published `oracle_eval` re-ranking, which is the same "
            "object SGV11 and SGV12 measured their ladders against."
        ),
    }
    base["sgv12_published_prefix_purity"] = {
        engine: {
            "baseline_prefix_harm_rate": float(row["arms"][s12.SGV5_FROZEN]["harm_rate"]),
            "ceiling_prefix_harm_rate": float(row["arms"][ceiling_key]["harm_rate"]),
            "baseline_repair_capture": float(row["arms"][s12.SGV5_FROZEN]["beneficial_recall"]),
            "ceiling_repair_capture": float(row["arms"][ceiling_key]["beneficial_recall"]),
            "baseline_overlap_with_ceiling": float(
                row["arms"][s12.SGV5_FROZEN]["prefix_overlap_with_ceiling"]
            ),
            "matched_coverage": float(row["arms"][s12.SGV5_FROZEN]["coverage"]),
        }
        for engine, row in prefix.items()
    }
    base["note"] = (
        "SGV13 adds no feature and changes no label. What it buys is a small number of outcome "
        "labels from the engine being deployed on, and what it fits are corrections to SGV5's "
        "frozen score over the columns SGV5's own selected model already used."
    )
    return base


def cc_read_json(path: Path) -> dict[str, Any]:
    """SGV1's reader, so a malformed artifact fails the same way it fails everywhere else."""
    return dict(cc._read_json(path))


# ------------------------------------------------------------------ the leakage boundary


@dataclass(frozen=True, slots=True)
class PoolView:
    """The held-out engine's adaptation pool as an acquisition rule is allowed to see it.

    There is no outcome field, and that is the whole point of the type. An acquisition rule
    receives this object and returns an ordering; the labels come into existence only on the
    `Budget` that `purchase` builds from a prefix of that ordering. The leakage suite asserts
    the slot list against this docstring and walks every acquisition function's syntax tree.
    """

    index: np.ndarray
    features: np.ndarray
    frozen_score: np.ndarray
    adapter_input: np.ndarray
    documents: np.ndarray
    sites: np.ndarray
    candidates: np.ndarray
    disagreement: np.ndarray
    in_prefix: np.ndarray
    in_band: np.ndarray
    accepted_share: float
    frozen_threshold: float

    @property
    def size(self) -> int:
        return int(self.index.size)


@dataclass(frozen=True, slots=True)
class Budget:
    """One purchased budget: which rows, what they cost, and how they split for inner validation.

    `folds` is a grouped k-fold over the budget's own DOCUMENTS, `s6.draw_budget`'s construction.
    Grouping matters more here than anywhere else in the project: a budget of fifty rows can hold
    eight candidates from one receipt, and an ungrouped split would let a page sit on both sides
    of the inner validation at exactly the budgets where the claim is most fragile.
    """

    positions: np.ndarray
    index: np.ndarray
    features: np.ndarray
    frozen_score: np.ndarray
    documents: np.ndarray
    sites: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    in_prefix: np.ndarray
    in_band: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    n_documents: int

    @property
    def size(self) -> int:
        return int(self.positions.size)


@dataclass(frozen=True, slots=True)
class Block:
    """One scored block of rows -- evaluation, calibration or pool -- with its outcomes."""

    name: str
    index: np.ndarray
    features: np.ndarray
    frozen_score: np.ndarray
    documents: np.ndarray
    sites: np.ndarray
    candidates: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray

    @property
    def size(self) -> int:
        return int(self.index.size)


@dataclass(frozen=True, slots=True)
class FoldSetup:
    """One leave-one-engine-out fold with SGV5 rebuilt, SGV6's adapter space, and the prefix.

    `state` is SGV12's `build_state`, which asserts the rebuilt decision score equals SGV5's
    published vector bit for bit. `base` is the same fold expressed as SGV6's own `TargetPool`
    type, so SGV6's unmodified `fit_arm` and `select_arm` can run on exactly these rows; `--adapt`
    re-runs `s6.build_base_fold` on every outer fold and asserts the two constructions agree to
    0.0 rather than this file claiming they do.
    """

    held_out: str
    state: s12.FoldState
    base: s6.TargetPool
    columns: tuple[int, ...]
    lambda_: float
    evaluation: Block
    calibration: Block
    view: PoolView
    band_size: int
    prefix_size: int
    outcomes: dict[str, np.ndarray]
    identity: dict[str, Any]
    projection: Any
    scaler: Any


def _block(setup_state: s12.FoldState, name: str, index: np.ndarray) -> Block:
    meta = setup_state.design.meta
    return Block(
        name=name,
        index=index,
        features=setup_state.matrix(index),
        frozen_score=setup_state.frozen_score(index),
        documents=setup_state.design.documents[index],
        sites=meta["site_id"].astype(str).to_numpy()[index],
        candidates=meta["candidate_id"].astype(str).to_numpy()[index],
        harmful=setup_state.design.harmful[index],
        beneficial=setup_state.design.beneficial[index],
    )


def prefix_masks(score: np.ndarray, share: float, band_multiple: float) -> tuple[np.ndarray, ...]:
    """The core accepted prefix and the adjacent outside band of one block, by rank.

    Both are defined on the FROZEN score at the acceptance share the SOURCE cut takes, so the
    region a target label is bought for is fixed before any target label exists. The band is the
    next `band_multiple * take` rows below the cut: the candidates a deployment would have to
    promote if it were going to change its accepted set at all.
    """
    size = int(np.asarray(score).size)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    take = int(np.clip(round(share * size), 0, size))
    reach = int(np.clip(take + round(band_multiple * take), 0, size))
    core = np.zeros(size, dtype=bool)
    band = np.zeros(size, dtype=bool)
    core[order[:take]] = True
    band[order[take:reach]] = True
    return core, band


def disagreement_signal(held_out: str, candidates: np.ndarray) -> np.ndarray:
    """The spread of a row's normalised rank across five SOURCE-fitted SGV12 rankings.

    Every column read here was fitted without a single label from the engine it is being read
    on, so the signal is available to an operator before annotating anything. The oracle and
    ceiling columns SGV12 also published on this block are deliberately absent; `--negative`
    asserts that `DISAGREEMENT_COLUMNS` contains no name carrying `oracle` or `ceiling`.
    """
    table = pd.read_parquet(
        s12.SCORES, columns=["held_out_engine", "block", "candidate_id", *DISAGREEMENT_COLUMNS]
    )
    table = table[(table["held_out_engine"] == held_out) & (table["block"] == POOL)]
    if table.empty:
        raise PhaseError(f"{held_out}: SGV12 published no pool rows to read disagreement from")
    frame = table.set_index(table["candidate_id"].astype(str))
    missing = int(np.sum(~np.isin(candidates, frame.index.to_numpy())))
    if missing:
        raise PhaseError(f"{held_out}: {missing} pool rows are absent from SGV12's score table")
    frame = frame.loc[candidates]
    size = len(frame)
    normalised = np.empty((size, len(DISAGREEMENT_COLUMNS)), dtype=float)
    for position, column in enumerate(DISAGREEMENT_COLUMNS):
        values = frame[column].to_numpy(dtype=float)
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        normalised[:, position] = order / max(size - 1, 1)
    return np.asarray(normalised.std(axis=1), dtype=float)


def adapter_space(state: s12.FoldState) -> tuple[Any, Any]:
    """SGV6's frozen adapter input space, rebuilt from the same frozen model on the same rows.

    The projection and the standardiser are fitted on the FIT engines' training rows and frozen
    before any target row is touched, so the space an acquisition rule measures diversity in
    carries no target statistic -- not a mean, not a variance, not a principal direction. The
    construction is `s6.build_base_fold`'s; `--adapt` re-runs SGV6's own function on every outer
    fold and asserts the two agree exactly, rather than this file claiming they do.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    model, design, fold = state.frozen_model, state.design, state.fold
    fit_scaled = model._scaled(design, fold.fit)
    components = min(s6.ADAPTER_COMPONENTS, fit_scaled.shape[1], fit_scaled.shape[0])
    projection = PCA(n_components=components, random_state=pilot.FIT_SEED).fit(fit_scaled)
    probabilities = model.probabilities(design, state.extended.block(fold.fit))
    scaler = StandardScaler().fit(
        np.column_stack(
            [
                logit(probabilities["harm"]),
                logit(probabilities["benefit"]),
                projection.transform(fit_scaled),
            ]
        )
    )
    return projection, scaler


def target_pool(
    state: s12.FoldState, projection: Any, scaler: Any, lambda_: float
) -> s6.TargetPool:
    """The held-out engine's TRAIN rows as SGV6's own pool type, so SGV6's code can run on them."""
    index = state.index(POOL)
    design = state.design
    probabilities = state.frozen_model.probabilities(design, state.extended.block(index))
    harm, benefit = probabilities["harm"], probabilities["benefit"]
    return s6.TargetPool(
        index=index,
        documents=design.documents[index],
        harmful=design.harmful[index],
        beneficial=design.beneficial[index],
        p_harm=harm,
        p_benefit=benefit,
        utility=benefit - lambda_ * harm,
        adapter_input=s6._adapter_input(
            state.frozen_model,
            design,
            state.extended.block(index),
            projection,
            scaler,
            harm,
            benefit,
        ),
    )


def build_setup(
    base: dg.Design,
    fold: rl.Fold,
    held_out: str,
    signatures: np.ndarray,
    retrieval_columns: list[int],
    selection: dict[str, Any],
    frozen: pd.DataFrame | None,
    band_multiple: float,
    disagreement: bool = True,
) -> FoldSetup:
    """Rebuild SGV5 on one fold and resolve every block, every label gate and the prefix region.

    One construction path serves the outer folds and SGV9's inner replay. `frozen` is SGV5's
    published score table on an outer fold, where `build_state` asserts the rebuilt decision score
    equals it bit for bit, and is None on an inner fold, which has no published counterpart --
    what an inner fold does carry is SGV5's frozen choice of representation, model class and
    lambda for the OUTER fold it belongs to, so the replay replays the frozen pipeline rather
    than a pipeline that re-chose itself.
    """
    state = build_state(base, fold, selection, signatures, retrieval_columns, held_out, frozen)
    lambda_ = float(selection["lambda"])
    projection, scaler = adapter_space(state)
    pool = target_pool(state, projection, scaler, lambda_)
    pool_index = state.index(POOL)
    evaluation = _block(state, EVALUATION, state.index(EVALUATION))
    calibration = _block(state, CALIBRATION, state.index(CALIBRATION))
    _, share = relevant_prefix(state, pool_index)
    pool_score = state.frozen_score(pool_index)
    core, band = prefix_masks(pool_score, share, band_multiple)
    meta = state.design.meta
    candidates = meta["candidate_id"].astype(str).to_numpy()[pool_index]
    signal = (
        disagreement_signal(held_out, candidates)
        if disagreement
        else np.zeros(pool_index.size, dtype=float)
    )
    view = PoolView(
        index=pool_index,
        features=state.matrix(pool_index),
        frozen_score=pool_score,
        adapter_input=pool.adapter_input,
        documents=state.design.documents[pool_index],
        sites=meta["site_id"].astype(str).to_numpy()[pool_index],
        candidates=candidates,
        disagreement=signal,
        in_prefix=core,
        in_band=band,
        accepted_share=float(share),
        frozen_threshold=float(state.source_threshold[PRIMARY_KEY]),
    )
    evaluation_documents = set(evaluation.documents.tolist())
    if set(view.documents.tolist()) & evaluation_documents:
        raise PhaseError(f"{held_out}: the adaptation pool shares documents with the evaluation")
    if set(view.sites.tolist()) & set(evaluation.sites.tolist()):
        raise PhaseError(f"{held_out}: the adaptation pool shares correction sites with the eval")
    if set(calibration.documents.tolist()) & evaluation_documents:
        raise PhaseError(f"{held_out}: the source calibration block reaches evaluation documents")
    identity = {
        "rebuild_max_abs_difference_from_published_sgv5": float(
            state.identity.get("max_abs_difference", float("nan"))
        ),
        "identity_checked": bool(state.identity.get("checked", False)),
        "evaluation_rows": int(evaluation.size),
        "evaluation_documents": len(evaluation_documents),
        "pool_rows": int(view.size),
        "pool_documents": len(set(view.documents.tolist())),
        "pool_beneficial_rate": float(state.design.beneficial[pool_index].mean()),
        "pool_harmful_rate": float(state.design.harmful[pool_index].mean()),
        "accepted_share": float(share),
        "prefix_rows": int(core.sum()),
        "band_rows": int(band.sum()),
    }
    return FoldSetup(
        held_out=held_out,
        state=state,
        base=pool,
        columns=tuple(state.columns),
        lambda_=lambda_,
        evaluation=evaluation,
        calibration=calibration,
        view=view,
        band_size=int(band.sum()),
        prefix_size=int(core.sum()),
        outcomes={
            "harmful": state.design.harmful[pool_index],
            "beneficial": state.design.beneficial[pool_index],
        },
        identity=identity,
        projection=projection,
        scaler=scaler,
    )


# ------------------------------------------------------------------ acquisition
#
# Every function below is handed a `PoolView` and returns an ordering. None of them may read an
# outcome: the type carries none, and `tests/leakage` walks each of their syntax trees for the
# names `harmful`, `beneficial`, `outcomes` and `purchase` and fails if one appears.


def _document_round_robin(documents: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Rows in document round-robin order, each document internally sorted by `value`.

    A page with fifty near-duplicate candidates contributes its best row before it contributes
    its second, so a budget of fifty rows touches up to fifty documents rather than one. The
    document order is by each document's best value, and every tie in either key falls back to
    the row position, so the traversal is a pure function of the pool.
    """
    order = np.lexsort((np.arange(value.size), -np.asarray(value, dtype=float)))
    per_document: dict[str, list[int]] = {}
    for position in order.tolist():
        per_document.setdefault(str(documents[position]), []).append(position)
    ranked = sorted(per_document, key=lambda name: (-float(value[per_document[name][0]]), name))
    out: list[int] = []
    depth = 0
    while len(out) < value.size:
        for name in ranked:
            rows = per_document[name]
            if depth < len(rows):
                out.append(rows[depth])
        depth += 1
    return np.asarray(out, dtype=int)


def _prefix_swap_value(view: PoolView) -> np.ndarray:
    """How much a label on this row would say about a deployment-critical swap.

    Two factors, each doing one job. The kernel on rank distance from the transported cut says
    how close the row is to the decision at all -- a row deep inside the accepted set or far
    below the band is one no plausible relabelling moves across. The partner count says whether
    exchanging it is even possible: how many candidates at the SAME correction site sit on the
    other side of the cut, which is the pair whose ordering decides what the accepted set
    contains. Both read the frozen score and the site structure; neither reads an outcome.
    """
    size = view.size
    if size == 0:
        return np.zeros(0, dtype=float)
    order = np.argsort(-view.frozen_score, kind="stable")
    rank = np.empty(size, dtype=float)
    rank[order] = np.arange(size, dtype=float)
    cut = float(np.clip(view.accepted_share * size, 0.0, float(size)))
    width = max(float(view.in_band.sum()), 1.0)
    kernel = np.exp(-np.abs(rank - cut) / width)
    inside = view.in_prefix
    partners = np.zeros(size, dtype=float)
    by_site: dict[str, list[int]] = {}
    for position in range(size):
        by_site.setdefault(str(view.sites[position]), []).append(position)
    for rows in by_site.values():
        block = np.asarray(rows, dtype=int)
        n_inside = float(inside[block].sum())
        n_outside = float(block.size - n_inside)
        partners[block] = np.where(inside[block], n_outside, n_inside)
    return kernel * (1.0 + partners)


def acquisition_order(view: PoolView, rule: str, pool: s6.TargetPool, seed: int) -> np.ndarray:
    """Positions in the pool, in the order a human would be asked to label them.

    Every rule returns one complete ordering and every budget is a PREFIX of it, so the label
    ladder is a pure budget increase and the N = 250 experiment contains the N = 100 experiment.
    The ordering is computed once from frozen quantities and is never revised as labels arrive,
    so these curves are a lower bound on what a sequential loop could deliver rather than an
    estimate of it -- the same restriction SGV6 recorded, kept so the two are comparable.
    """
    if rule in (Q_RANDOM, Q_DIVERSITY, Q_UNCERTAINTY):
        legacy = {Q_RANDOM: s6.RANDOM, Q_DIVERSITY: s6.DIVERSITY, Q_UNCERTAINTY: s6.UNCERTAINTY}
        return s6.acquisition_order(pool, legacy[rule], view.frozen_threshold, seed)
    if rule == Q_RANDOM_DOCUMENT:
        generator = np.random.default_rng(seed)
        names = sorted(set(view.documents.tolist()))
        shuffled = list(generator.permutation(len(names)))
        rank = {names[int(position)]: order for order, position in enumerate(shuffled)}
        jitter = generator.permutation(view.size)
        key = np.array([rank[str(name)] for name in view.documents.tolist()], dtype=float)
        return np.asarray(np.lexsort((jitter, key)), dtype=int)
    if rule == Q_DISAGREEMENT:
        return np.asarray(np.lexsort((np.arange(view.size), -view.disagreement)), dtype=int)
    if rule == Q_PREFIX_SWAP:
        return np.asarray(np.lexsort((np.arange(view.size), -_prefix_swap_value(view))), dtype=int)
    if rule == Q_HYBRID:
        return _document_round_robin(view.documents, _prefix_swap_value(view))
    raise PhaseError(f"unknown acquisition rule {rule!r}")


# ------------------------------------------------------------------ purchasing labels
#
# This is the ONLY function in the stage that reads `FoldSetup.outcomes`. Everything upstream of
# it -- every acquisition rule, every ordering, every band definition -- is outcome-free, and
# `tests/leakage` asserts that this function has exactly the call sites listed in its docstring.


def purchase(
    setup: FoldSetup, order: np.ndarray, size: int, permute_seed: int | None = None
) -> Budget:
    """Buy the first `size` rows of an ordering and attach their labels.

    Call sites: `_fit_all_arms` and `run_preregister`, both of which pass an ordering produced
    without reading an outcome. `permute_seed` is the label-permutation control of section 21:
    the rows, the documents, the acquisition and the class balance are held exactly and only the
    label-to-row assignment is destroyed, so an arm that gains under it is gaining from training
    mechanics rather than from target supervision.
    """
    positions = np.asarray(order[:size], dtype=int)
    view = setup.view
    harmful = setup.outcomes["harmful"][positions]
    beneficial = setup.outcomes["beneficial"][positions]
    if permute_seed is not None:
        shuffle = np.random.default_rng(permute_seed).permutation(positions.size)
        harmful, beneficial = harmful[shuffle], beneficial[shuffle]
    documents = view.documents[positions]
    groups = sorted(set(documents.tolist()))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    count = min(INNER_FOLDS, len(groups))
    if count >= 2:
        assignment = {name: i % count for i, name in enumerate(groups)}
        block = np.array([assignment[name] for name in documents.tolist()])
        for i in range(count):
            held = np.flatnonzero(block == i)
            rest = np.flatnonzero(block != i)
            if held.size and rest.size:
                folds.append((rest, held))
    return Budget(
        positions=positions,
        index=view.index[positions],
        features=view.features[positions],
        frozen_score=view.frozen_score[positions],
        documents=documents,
        sites=view.sites[positions],
        harmful=harmful,
        beneficial=beneficial,
        in_prefix=view.in_prefix[positions],
        in_band=view.in_band[positions],
        folds=tuple(folds),
        n_documents=len(groups),
    )


# ------------------------------------------------------------------ the arms

FROZEN = "frozen"
CORRECTION = "linear_correction"
REFIT = "refit"
CALIBRATED = "calibrated_probability"
SGV6_KIND = "sgv6_adapter"


@dataclass(frozen=True, slots=True)
class Adapted:
    """One arm fitted at one budget. `kind` says how a block is scored, never the arm's name."""

    arm: str
    kind: str
    weights: np.ndarray | None
    model: c5.Reliability | None
    calibrator: Any | None
    setting: float
    note: dict[str, Any]
    scorer: Any = None
    """Only the SGV6 comparison arm sets this: SGV6's own `Adapted.score` bound to a block."""

    def score(self, setup: FoldSetup, block: Block) -> np.ndarray:
        if self.kind == FROZEN:
            return np.asarray(block.frozen_score, dtype=float)
        if self.kind == CORRECTION:
            assert self.weights is not None
            if not np.any(self.weights):
                # An exactly zero correction returns the frozen vector itself rather than
                # `s + 0 . x`. The two agree to about 1e-16 and that is not the same thing as
                # exact: decision 4 says N = 0 IS SGV5, so the identity has to be an identity.
                return np.asarray(block.frozen_score, dtype=float)
            return np.asarray(block.frozen_score + block.features @ self.weights, dtype=float)
        if self.kind == REFIT:
            assert self.model is not None
            return np.asarray(
                self.model.utility(
                    setup.state.design, setup.state.extended.block(block.index), setup.lambda_
                ),
                dtype=float,
            )
        if self.kind == CALIBRATED:
            if self.calibrator is None:
                return np.asarray(block.frozen_score, dtype=float)
            return -np.asarray(self.probability(block), dtype=float)
        if self.kind == SGV6_KIND:
            assert self.scorer is not None
            return np.asarray(self.scorer(block), dtype=float)
        raise PhaseError(f"unknown adapted kind {self.kind!r}")

    def probability(self, block: Block) -> np.ndarray:
        """The target-calibrated harm probability. Defined only for the calibrated kind."""
        if self.kind != CALIBRATED or self.calibrator is None:
            raise PhaseError(f"{self.arm} exposes no target-calibrated harm probability")
        return np.clip(
            np.asarray(self.calibrator.predict(-np.asarray(block.frozen_score, dtype=float))),
            PROB_FLOOR,
            1.0 - PROB_FLOOR,
        )


def frozen_arm() -> Adapted:
    """A0. No target label is read at any budget, so its curve is flat by construction."""
    return Adapted(
        arm=A0_FROZEN,
        kind=FROZEN,
        weights=None,
        model=None,
        calibrator=None,
        setting=0.0,
        note={"target_labels_used": 0},
    )


@dataclass(frozen=True, slots=True)
class MonotoneCalibrator:
    """A calibrated harm probability that is NON-INCREASING in the frozen score, by construction.

    A1's contract is that the ranking does not move, so the map from score to probability has to
    be monotone and the direction has to be fixed rather than fitted. Both members take `-s` as
    their input and are monotone non-decreasing in it: the Platt member has its slope clamped at
    zero, the isotonic member is monotone by definition. A budget whose labels say the frozen
    ordering is backwards therefore produces a flat map -- A1 declines to invert the ranking --
    and `clamped` records how often that happened.
    """

    kind: str
    intercept: float
    slope: float
    clamped: bool

    def predict(self, transformed: np.ndarray) -> np.ndarray:
        return np.asarray(
            sigmoid(self.intercept + self.slope * np.asarray(transformed, dtype=float)),
            dtype=float,
        )


def _fit_calibrator(kind: str, transformed: np.ndarray, harmful: np.ndarray) -> Any:
    if np.unique(harmful).size < 2:
        return None
    if kind != "platt":  # pragma: no cover - A1 registers one member; see its docstring
        raise PhaseError(f"A1 is a strictly monotone family and {kind!r} is not one")
    weights, _ = s6._penalised_logistic(
        transformed[:, None], np.asarray(harmful, dtype=float), np.zeros(transformed.size), 1e-6
    )
    slope = float(weights[1])
    return MonotoneCalibrator("platt", float(weights[0]), max(slope, 0.0), slope < 0.0)


def fit_calibration_arm(budget: Budget) -> Adapted:
    """A1. The budget buys a monotone recalibration of the frozen score and nothing else.

    The map is Platt and only Platt, and that is a restriction rather than an omission. A1's
    contract is that the ranking does not move; isotonic regression is monotone but only WEAKLY,
    so it collapses whole score intervals to one value and changes the ordering by creating ties.
    A feasibility probe measured the cost on docTR at N = 25 -- held-out AUROC fell from 0.9898
    to 0.9063 under an isotonic member, purely from ties -- and it is disclosed in the design
    record. A two-parameter logistic map with its slope clamped at zero is strictly monotone
    wherever the slope is positive and is the identity on the ordering, which is what "modify
    only the level" has to mean here. The budget's own grouped out-of-fold Brier score is still
    computed and published, so the quality of the calibration is reported rather than assumed.
    """
    from ocr_risk.metrics.calibration import brier_score

    note: dict[str, Any] = {"target_labels_used": int(budget.size), "candidates": {}}
    if budget.size == 0 or np.unique(budget.harmful).size < 2:
        note["fittable"] = False
        note["reason"] = "the budget carries a single harm class" if budget.size else "empty"
        return Adapted(A1_CALIBRATION, FROZEN, None, None, None, 0.0, note)
    transformed = -budget.frozen_score
    chosen = "platt"
    out_of_fold = np.full(budget.size, np.nan)
    for train, held in budget.folds:
        fitted = _fit_calibrator(chosen, transformed[train], budget.harmful[train])
        if fitted is not None:
            out_of_fold[held] = fitted.predict(transformed[held])
    covered = np.isfinite(out_of_fold)
    note["candidates"][chosen] = {
        "out_of_fold_brier": (
            float(brier_score(out_of_fold[covered], budget.harmful[covered].astype(float)))
            if covered.sum() >= 2
            else float("nan")
        ),
        "rows_scored": int(covered.sum()),
    }
    calibrator = _fit_calibrator(chosen, transformed, budget.harmful)
    if calibrator is None:  # pragma: no cover - guarded by the single-class check above
        note["fittable"] = False
        return Adapted(A1_CALIBRATION, FROZEN, None, None, None, 0.0, note)
    note.update(
        {
            "fittable": True,
            "selected": chosen,
            "slope_clamped_to_preserve_the_ranking": bool(calibrator.clamped),
        }
    )
    return Adapted(A1_CALIBRATION, CALIBRATED, None, None, calibrator, 0.0, note)


def cut_arm() -> Adapted:
    """A2. The score and the ranking are the frozen ones; only the cut rule differs.

    This arm is deliberately identical to A0 under `frozen_source` and differs from it only
    under `target_budget`. That is the point: it makes "what a target label buys when it is
    spent entirely on the threshold" a measured quantity rather than an argument, and
    `--negative` asserts the two arms' `frozen_source` deployments agree exactly.
    """
    return Adapted(A2_CUT, FROZEN, None, None, None, 0.0, {"target_labels_used": 0})


def fit_head_arm(budget: Budget, arm: str = A3_HEAD) -> Adapted:
    """A3. An offset-parameterised ridge correction on SGV5's own selected columns.

    The frozen decision score enters as an OFFSET with its coefficient fixed at one, so the
    budget is spent entirely on the correction and the prior mean is "no correction" rather than
    "no information". At infinite precision the correction is exactly zero and the arm returns
    SGV5; the grid contains that element and the tie-break prefers it, so a budget carrying no
    signal returns the frozen model rather than noise.

    Whether the frozen score is well scaled to act as a log-odds offset is a real question and
    it is not answered here. It is answered by A1, which is the arm whose whole mechanism is the
    level, and the mechanism decomposition reads the two together rather than assuming either.
    """
    note: dict[str, Any] = {"target_labels_used": int(budget.size), "grid": {}}
    if budget.size == 0 or np.unique(budget.harmful).size < 2 or not budget.folds:
        note["fittable"] = False
        return Adapted(arm, CORRECTION, np.zeros(0), None, None, float("inf"), note)
    offset = -budget.frozen_score
    target = budget.harmful.astype(float)
    scores: dict[float, tuple[float, float]] = {}
    for tau in HEAD_SHRINKAGE:
        out_of_fold = np.full(budget.size, np.nan)
        for train, held in budget.folds:
            weights, _ = s6._penalised_logistic(
                budget.features[train], target[train], offset[train], tau
            )
            out_of_fold[held] = offset[held] + s6._design_row(budget.features[held]) @ weights
        covered = np.isfinite(out_of_fold)
        scores[tau] = (
            s6._inner_score(
                -out_of_fold[covered], budget.harmful[covered], budget.beneficial[covered]
            )
            if covered.sum() >= 2
            else (float("-inf"), float("-inf"))
        )
        note["grid"][f"tau_{tau:g}"] = {
            "inner_repair_recall": scores[tau][0],
            "inner_auc_beneficial": scores[tau][1],
            "rows_scored": int(covered.sum()),
        }

    def rank(tau: float) -> tuple[float, float, float]:
        endpoint, auc = scores[tau]
        return (endpoint, -1.0 if np.isnan(auc) else auc, -s6.Family().amount(tau))

    # The tie-break runs toward LESS adaptation: an infinite precision is the frozen model and
    # wins every tie, so a budget whose grid is flat returns SGV5 rather than a fit. The keys are
    # SGV6's own `_inner_score`, imported, so the two stages select on the same criterion.
    chosen = max(HEAD_SHRINKAGE, key=rank)
    weights, _ = s6._penalised_logistic(budget.features, target, offset, chosen)
    # The intercept is dropped: it translates every score by one constant, which no ranking and
    # no re-derived threshold can see. Keeping it would put an unidentifiable offset into the
    # score table and make two arms differ by a number that changes no decision.
    correction = -np.asarray(weights[1:], dtype=float)
    note.update({"fittable": True, "selected_precision": float(chosen)})
    return Adapted(arm, CORRECTION, correction, None, None, float(chosen), note)


# ------------------------------------------------------------------ A5: the proposed arm

TARGET_PAIR_BUDGET = 20000


@dataclass(frozen=True, slots=True)
class PairBlock:
    """One set of ranking constraints as a difference matrix and a frozen-score margin."""

    name: str
    differences: np.ndarray
    margin: np.ndarray
    eligible: int

    @property
    def size(self) -> int:
        return int(self.margin.size)


def _pairs_from(
    features: np.ndarray,
    score: np.ndarray,
    good: np.ndarray,
    bad: np.ndarray,
    name: str,
    seed: int,
    cap: int,
) -> PairBlock:
    """Every (good, bad) pair, or a deterministic stratified draw of `cap` of them.

    The draw is stratified by the GOOD row: each beneficial candidate contributes the same
    number of constraints up to the remainder, so a budget with one beneficial row on a busy
    page cannot dominate the objective. `eligible` records what the cap cost, which is what
    section 29 asks a pair budget to publish.
    """
    good_rows = np.flatnonzero(good)
    bad_rows = np.flatnonzero(bad)
    eligible = int(good_rows.size * bad_rows.size)
    if eligible == 0:
        width = int(features.shape[1]) if features.ndim == 2 else 0
        return PairBlock(name, np.zeros((0, width)), np.zeros(0), 0)
    if eligible <= cap:
        left = np.repeat(good_rows, bad_rows.size)
        right = np.tile(bad_rows, good_rows.size)
    else:
        generator = np.random.default_rng(seed)
        per_good = max(cap // int(good_rows.size), 1)
        left_parts, right_parts = [], []
        for row in good_rows.tolist():
            take = min(per_good, int(bad_rows.size))
            picked = generator.choice(bad_rows, size=take, replace=False)
            left_parts.append(np.full(take, row, dtype=int))
            right_parts.append(np.sort(picked))
        left = np.concatenate(left_parts)
        right = np.concatenate(right_parts)
    return PairBlock(
        name=name,
        differences=features[left] - features[right],
        margin=np.asarray(score[left] - score[right], dtype=float),
        eligible=eligible,
    )


def _prefix_objective(
    weights: np.ndarray, blocks: tuple[tuple[PairBlock, float], ...], ridge: float
) -> tuple[float, np.ndarray]:
    """The three-term pairwise correction objective and its exact gradient.

    Every term is the SAME RankNet loss over a different constraint set, so the gradient
    collapses to one weighted matvec per block. The frozen score enters as a fixed margin, which
    is what makes `beta = 0` mean "SGV5" rather than "an unfitted linear model".
    """
    value = 0.5 * ridge * float(weights @ weights)
    gradient = ridge * np.asarray(weights, dtype=float)
    for block, factor in blocks:
        if block.size == 0 or factor == 0.0:
            continue
        margin = block.margin + block.differences @ weights
        value += factor * float(np.mean(s12._softplus(-margin)))
        share = factor / float(block.size)
        gradient -= share * (block.differences.T @ s12._sigmoid(-margin))
    return value, gradient


def fit_prefix_arm(
    budget: Budget,
    source: PairBlock,
    lambda_target: float,
    lambda_prefix: float,
    ridge: float,
    seed: int,
    arm: str = A5_PREFIX,
) -> Adapted:
    """A5. A correction to the frozen score fitted on source, target and prefix-swap pairs.

    The prefix-swap block is the mechanism the arm is named for: constraints of the form
    "this beneficial candidate the deployment would REJECT must outrank that harmful candidate
    the deployment would ACCEPT". They are the pairs whose ordering decides the accepted set, and
    SGV12 measured that changing about a tenth of the accepted set is the whole actionable gap.

    Decision 7 lives here. The source block anchors the correction rather than refitting the
    source model, so it is present only when a target block is: an empty budget yields an
    exactly zero correction and the arm returns SGV5. That is asserted, not assumed.
    """
    note: dict[str, Any] = {"target_labels_used": int(budget.size)}
    width = int(budget.features.shape[1]) if budget.size else int(source.differences.shape[1])
    if budget.size == 0 or not budget.beneficial.any() or not budget.harmful.any():
        note.update(
            {
                "fittable": False,
                "reason": (
                    "the budget carries no beneficial-harmful pair, so there is no target "
                    "constraint and decision 7 makes the correction exactly zero"
                ),
                "budget_beneficial": int(budget.beneficial.sum()),
                "budget_harmful": int(budget.harmful.sum()),
            }
        )
        return Adapted(arm, CORRECTION, np.zeros(width), None, None, 0.0, note)
    from scipy.optimize import minimize

    target = _pairs_from(
        budget.features,
        budget.frozen_score,
        budget.beneficial,
        budget.harmful,
        "target",
        seed,
        TARGET_PAIR_BUDGET,
    )
    swap = _pairs_from(
        budget.features,
        budget.frozen_score,
        budget.beneficial & ~budget.in_prefix,
        budget.harmful & budget.in_prefix,
        "prefix_swap",
        seed + 1,
        TARGET_PAIR_BUDGET,
    )
    blocks = ((source, 1.0), (target, lambda_target), (swap, lambda_prefix))
    result = minimize(
        lambda w: _prefix_objective(w, blocks, ridge),
        np.zeros(width),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER},
    )
    note.update(
        {
            "fittable": True,
            "lambda_target": float(lambda_target),
            "lambda_prefix": float(lambda_prefix),
            "ridge": float(ridge),
            "pairs": {
                "source": {"drawn": source.size, "eligible": source.eligible},
                "target": {"drawn": target.size, "eligible": target.eligible},
                "prefix_swap": {"drawn": swap.size, "eligible": swap.eligible},
            },
            "objective": float(result.fun),
            "converged": bool(result.success),
        }
    )
    return Adapted(arm, CORRECTION, np.asarray(result.x, dtype=float), None, None, 0.0, note)


# ------------------------------------------------------------------ A4 and A6: joint refits


def fit_joint_arm(
    setup: FoldSetup, budget: Budget, weight: float, capacity: bool, arm: str
) -> Adapted:
    """A4, A6 and the permutation control. SGV5's two-head construction refitted jointly.

    The source rows carry the design's own labels and the target rows carry the BUDGET's, which
    is what makes section 21's control operative: a permuted budget hands permuted labels to the
    refit. `_refit` reproduces `s6._fit_weighted` exactly when handed the design's labels and
    `tests/leakage` asserts that, so this is SGV6's construction with the label source made
    explicit rather than a second construction.

    A6 raises the boosted model's capacity, because SGV12's ablation A found the hypothesis class
    binding before the objective did: the same classification objective fell from 0.9356 to
    0.8945 held-out AUROC moving from boosted trees to a linear scorer, while changing the
    objective at matched linear capacity moved it by 0.0077. If capacity is the constraint, an
    arm that is allowed to raise it should be the one that moves.
    """
    note: dict[str, Any] = {"target_labels_used": int(budget.size), "target_row_weight": weight}
    if budget.size == 0:
        note["fittable"] = False
        note["reason"] = "an empty budget leaves SGV5's own fitting set, so the arm IS SGV5"
        return Adapted(arm, FROZEN, None, None, None, 0.0, note)
    state = setup.state
    fit = state.index(FIT)
    rows = np.concatenate([fit, budget.index])
    weights = np.ones(rows.size)
    weights[fit.size :] = weight
    harmful = np.concatenate([state.design.harmful[fit], budget.harmful])
    beneficial = np.concatenate([state.design.beneficial[fit], budget.beneficial])
    model = _refit(setup, rows, weights, harmful, beneficial, capacity)
    note.update(
        {
            "fittable": True,
            "fit_rows": int(rows.size),
            "source_rows": int(fit.size),
            "capacity_raised": bool(capacity),
            "target_labels_come_from": "the purchased budget, not the design matrix",
            "calibrated_on": "the source engines' calibration documents, as SGV5 does",
        }
    )
    return Adapted(arm, REFIT, None, model, None, float(weight), note)


def _refit(
    setup: FoldSetup,
    rows: np.ndarray,
    weights: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    capacity: bool,
) -> c5.Reliability:
    """SGV5's two-head construction refitted on `rows`, with the outcome labels PASSED IN.

    `s6._fit_weighted` reads the labels for its fitting rows out of the design matrix. That is
    correct for SGV6, whose budget rows are exactly the rows whose labels it may read, but it
    makes section 21's permutation control inoperative here: permuting a budget changes the
    `Budget` object and not the design, so a refit reading the design would see the true labels
    either way and the control would return the arm it is meant to control. Feasibility probe 4
    caught exactly that -- the permuted and unpermuted refits came out bit-identical on every
    metric, every engine and every budget -- and this function is the fix.

    Everything else is `_fit_weighted`'s: the standardiser, the two heads, the model class and
    hyperparameters SGV5 selected for this fold, the three-class outcome encoding, and the
    calibrators fitted on the SOURCE engines' calibration documents. `tests/leakage` asserts
    that it reproduces `s6._fit_weighted` bit for bit when handed the design's own labels, so
    this is one construction with the label source made explicit and not a second one.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    from ocr_risk.calibrate.calibrators import build_calibrator

    state = setup.state
    kind = str(state.selection["model"])
    if kind == "logistic":  # pragma: no cover - every fold selected a boosted model
        raise PhaseError("the joint refits are defined for the boosted model class only")
    design = state.design
    columns = list(setup.columns)
    block = design.matrix[np.ix_(rows, columns)]
    scaler = StandardScaler().fit(block)
    scaled = np.asarray(scaler.transform(block), dtype=float)
    outcome = np.full(rows.size, policy.CLASS_NEUTRAL, dtype=np.int64)
    outcome[harmful] = policy.CLASS_HARM
    outcome[beneficial] = policy.CLASS_BENEFIT
    balance = "balanced" if kind == c5.BOOSTED_BALANCED else None
    rounds = CAPACITY_ITERATIONS if capacity else c5.BOOSTING_ITERATIONS
    leaves = {"max_leaf_nodes": CAPACITY_LEAVES} if capacity else {}

    def make() -> Any:
        return HistGradientBoostingClassifier(
            max_iter=rounds, class_weight=balance, random_state=pilot.FIT_SEED, **leaves
        )

    fitted = c5.Reliability(
        kind=kind,
        representation=str(state.selection["representation"]),
        columns=tuple(columns),
        scaler=scaler,
        harm=make().fit(scaled, harmful.astype(int), sample_weight=weights),
        outcome=make().fit(scaled, outcome, sample_weight=weights),
        harm_calibrator=None,
        benefit_calibrator=None,
        weights=None,
    )
    calibration = state.index(CALIBRATION)
    raw = fitted.raw(design, calibration)
    harm_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    benefit_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    harm_calibrator.fit(raw["harm_raw"], design.harmful[calibration].astype(float))
    benefit_calibrator.fit(raw["benefit_raw"], design.beneficial[calibration].astype(float))
    fitted.harm_calibrator = harm_calibrator
    fitted.benefit_calibrator = benefit_calibrator
    return fitted


# ------------------------------------------------------------------ the SGV6 comparison arm


def fit_sgv6_arm(setup: FoldSetup, order: np.ndarray, size: int) -> tuple[Adapted, dict[str, Any]]:
    """SGV6's generic few-shot procedure, called through SGV6's own functions on the same rows.

    `s6.draw_budget` is handed THIS stage's acquisition order, so the SGV6 arm and every SGV13
    arm at the same (engine, budget, acquisition, draw) buy the identical labelled candidates. A
    difference between them is a difference in what the labels were spent on and cannot be a
    difference in which labels were bought. `s6.fit_arm` and `s6.select_arm` are unmodified, so
    the nine adapter families, their grids, their identity elements and the tie-break that
    resolves toward not adapting at all are SGV6's, not a restatement of them.
    """
    budget = s6.draw_budget(setup.base, order, size)
    lambda_ = setup.lambda_
    fitted = {arm: s6.fit_arm(arm, budget, lambda_) for arm in s6.ADAPTER_ARMS}
    name, record = s6.select_arm(fitted, budget)
    chosen = fitted[name]

    def block_score(block: Block) -> np.ndarray:
        state = setup.state
        rows = state.extended.block(block.index)
        probabilities = state.frozen_model.probabilities(state.design, rows)
        z = s6._adapter_input(
            state.frozen_model,
            state.design,
            rows,
            setup.projection,
            setup.scaler,
            probabilities["harm"],
            probabilities["benefit"],
        )
        return np.asarray(
            chosen.score(probabilities["harm"], probabilities["benefit"], z), dtype=float
        )

    return (
        Adapted(
            arm=SGV6_GENERIC,
            kind=SGV6_KIND,
            weights=None,
            model=None,
            calibrator=None,
            setting=float(chosen.setting),
            note={
                "target_labels_used": int(budget.size),
                "selected_family": name,
                "selected_setting": float(chosen.setting),
                "lambda": float(chosen.lambda_),
                "budget_documents": int(budget.n_documents),
            },
            scorer=block_score,
        ),
        record,
    )


# ------------------------------------------------------------------ deployment


def budget_block(setup: FoldSetup, budget: Budget) -> Block:
    """The budget's own rows dressed as a block, so an arm can be scored on what it was fitted on.

    This is the only block whose outcomes come from a purchase rather than from the corpus, and
    it exists so `target_budget` can place a cut through exactly the same `select_threshold` call
    every other cut rule uses. Its documents are pool documents and never evaluation documents.
    """
    return Block(
        name="budget",
        index=budget.index,
        features=budget.features,
        frozen_score=budget.frozen_score,
        documents=budget.documents,
        sites=budget.sites,
        candidates=setup.view.candidates[budget.positions],
        harmful=budget.harmful,
        beneficial=budget.beneficial,
    )


def thresholds_for(
    setup: FoldSetup, adapted: Adapted, budget: Budget, rule: str
) -> dict[str, dict[str, Any]]:
    """Where the cut goes for one arm under one rule, at every epsilon.

    `frozen_source` is the published call on the SOURCE engines' calibration rows -- the identical
    call that produced SGV5's threshold, reading no target row of any kind. The two budget rules
    are the identical call on the budget's own labelled target rows, under the empirical and the
    Bentkus controller respectively, and at N = 0 both ARE `frozen_source`, because there is no
    target row to read.
    """
    out: dict[str, dict[str, Any]] = {}
    if rule == CUT_FROZEN or budget.size == 0:
        score = adapted.score(setup, setup.calibration)
        for epsilon in EPSILONS:
            out[epsilon_key(epsilon)] = place_cut(
                score, setup.calibration.harmful, epsilon, FROZEN_CONTROLLER
            )
        return out
    if rule in BUDGET_CUTS:
        block = budget_block(setup, budget)
        score = adapted.score(setup, block)
        for epsilon in EPSILONS:
            out[epsilon_key(epsilon)] = place_cut(
                score, block.harmful, epsilon, BUDGET_CONTROLLERS[rule]
            )
        return out
    if rule == CUT_CALIBRATED:
        if adapted.kind != CALIBRATED:
            # A budget too small or too one-sided to fit the map leaves the arm as the frozen
            # model, whose score is a utility and not a probability. Comparing it to epsilon
            # would be comparing two different units, so the absolute cut falls back to the
            # frozen one -- which is what "these labels do not yet say anything" means.
            return thresholds_for(setup, adapted, budget, CUT_FROZEN)
        for epsilon in EPSILONS:
            out[epsilon_key(epsilon)] = {
                "tau": -float(epsilon),
                "feasible": True,
                "calibration_coverage": float("nan"),
                "calibration_risk": float("nan"),
                "controller": CUT_CALIBRATED,
            }
        return out
    raise PhaseError(f"unknown cut rule {rule!r}")


def _accepted_at_share(score: np.ndarray, share: float) -> np.ndarray:
    """The top `share` of a block by score. Used only for matched-coverage purity comparisons."""
    size = int(np.asarray(score).size)
    take = int(np.clip(round(share * size), 0, size))
    mask = np.zeros(size, dtype=bool)
    if take:
        mask[np.argsort(-np.asarray(score, dtype=float), kind="stable")[:take]] = True
    return mask


def prefix_statistics(
    block: Block, accepted: np.ndarray, baseline: np.ndarray, ceiling: np.ndarray
) -> dict[str, Any]:
    """What an accepted set contains, and how it differs from the baseline's and the ceiling's.

    SGV12 measured the actionable gap as a purity difference at matched coverage rather than a
    concordance difference, and found the baseline's prefix and the ceiling's overlapping by
    about 0.90 while their harm rates differed by a factor of twenty-five. These are the same
    statistics on the same construction, so the two stages' prefix numbers are comparable.

    `swaps_recovered` counts the exchange the mechanism is named for: a harmful candidate the
    baseline accepted that this arm rejects, paired with a beneficial candidate the baseline
    rejected that this arm accepts. It is reported as two counts and their minimum rather than as
    a single number, because an arm that only drops rows is not swapping anything.
    """
    harmful = block.harmful
    beneficial = block.beneficial
    size = int(accepted.sum())
    union = int((accepted | baseline).sum())
    dropped_harmful = int((baseline & harmful & ~accepted).sum())
    gained_beneficial = int((~baseline & beneficial & accepted).sum())
    return {
        "n_accepted": size,
        "coverage": float(accepted.mean()) if accepted.size else 0.0,
        "prefix_harm_rate": float(harmful[accepted].mean()) if size else 0.0,
        "prefix_repair_recall": (
            float(beneficial[accepted].sum() / beneficial.sum()) if beneficial.any() else 0.0
        ),
        "beneficial_accepted": int(beneficial[accepted].sum()),
        "harmful_accepted": int(harmful[accepted].sum()),
        "overlap_with_baseline": float((accepted & baseline).sum() / union) if union else 1.0,
        "overlap_with_ceiling": (
            float((accepted & ceiling).sum() / (accepted | ceiling).sum())
            if int((accepted | ceiling).sum())
            else 1.0
        ),
        "harmful_dropped_vs_baseline": dropped_harmful,
        "beneficial_gained_vs_baseline": gained_beneficial,
        "swaps_recovered": min(dropped_harmful, gained_beneficial),
    }


def per_document_counts(
    block: Block, accepted: np.ndarray, documents: list[str]
) -> dict[str, np.ndarray]:
    """One cell's accepted, harmful-accepted, beneficial-accepted and beneficial totals per page.

    Every endpoint this stage tests is a ratio of sums over rows, so a document-clustered
    resample of it is a weighted sum of these four vectors and nothing else has to be stored.
    The order is the sorted document order `document_multiplicities` uses, so the resampling is
    one matrix product per cell rather than a gather over rows.
    """
    codes = {name: position for position, name in enumerate(documents)}
    index = np.array([codes[str(name)] for name in block.documents.tolist()], dtype=int)
    size = len(documents)
    return {
        "accepted": np.bincount(index, weights=accepted.astype(float), minlength=size),
        "harmful_accepted": np.bincount(
            index, weights=(accepted & block.harmful).astype(float), minlength=size
        ),
        "beneficial_accepted": np.bincount(
            index, weights=(accepted & block.beneficial).astype(float), minlength=size
        ),
        "beneficial_total": np.bincount(
            index, weights=block.beneficial.astype(float), minlength=size
        ),
        "rows": np.bincount(index, minlength=size).astype(float),
    }


def cut_rules_for(arm: str) -> tuple[str, ...]:
    """Which cut rules an arm is deployed under. Two for every arm, three for the calibrated one."""
    return (*CUT_RULES, CUT_CALIBRATED) if arm == A1_CALIBRATION else CUT_RULES


@dataclass(frozen=True, slots=True)
class Reference:
    """Everything a cell is measured against, computed once per fold and never per cell."""

    documents: tuple[str, ...]
    baseline_score: np.ndarray
    baseline_accepted: np.ndarray
    baseline_depth: int
    matched_share: float
    ceiling_score: np.ndarray
    ceiling_accepted: np.ndarray
    frozen_thresholds: dict[str, dict[str, Any]]


def build_reference(setup: FoldSetup) -> Reference:
    """The frozen baseline's deployment and SGV10's published ceiling, on the evaluation block.

    The matched coverage every purity comparison uses is the baseline's OWN deployed coverage at
    the primary epsilon. Fixing it to the baseline rather than to a round number is what makes
    "this arm's accepted set is purer" a statement about composition instead of about depth.
    """
    block = setup.evaluation
    frozen = frozen_arm()
    empty = purchase(setup, np.zeros(0, dtype=int), 0)
    thresholds = thresholds_for(setup, frozen, empty, CUT_FROZEN)
    tau = float(thresholds[PRIMARY_KEY]["tau"])
    accepted = (
        np.asarray(block.frozen_score >= tau) if np.isfinite(tau) else np.zeros(block.size, bool)
    )
    ceiling = s12._ceiling_columns(setup.held_out, block.candidates)[s12.ceiling_arm(PRIMARY_KEY)]
    share = float(accepted.mean()) if block.size else 0.0
    return Reference(
        documents=tuple(sorted({str(name) for name in block.documents.tolist()})),
        baseline_score=np.asarray(block.frozen_score, dtype=float),
        baseline_accepted=accepted,
        baseline_depth=int(accepted.sum()),
        matched_share=share,
        ceiling_score=np.asarray(ceiling, dtype=float),
        ceiling_accepted=_accepted_at_share(ceiling, share),
        frozen_thresholds=thresholds,
    )


@dataclass(slots=True)
class CountSink:
    """Per-document deployment counts, accumulated columnar rather than as rows.

    The grid is about ten thousand cells and each cell decomposes into a hundred-odd documents,
    so a row-per-document accumulation would be millions of Python dicts and gigabytes of memory
    for a table that is five float columns wide. Each cell contributes one small array here and
    the frame is assembled once per fold, which is the same data at a fraction of the cost.

    Only the cut rules the analysis actually RESAMPLES are persisted -- the primary one and the
    oracle one. Every cut rule's scalar deployment is in the cell table for every arm; what is
    not kept for the others is the per-document decomposition, which is a resampling detail and
    not a result.
    """

    keys: list[tuple[str, str, str, int, int, str, float]]
    blocks: list[np.ndarray]
    documents: list[str]

    def add(
        self,
        context: dict[str, Any],
        arm: str,
        rule: str,
        epsilon: float,
        counts: dict[str, np.ndarray],
    ) -> None:
        self.keys.append(
            (
                str(context["held_out_engine"]),
                arm,
                str(context["acquisition"]),
                int(context["budget"]),
                int(context["draw"]),
                rule,
                float(epsilon),
            )
        )
        self.blocks.append(
            np.vstack(
                [
                    counts["accepted"],
                    counts["harmful_accepted"],
                    counts["beneficial_accepted"],
                    counts["beneficial_total"],
                    counts["rows"],
                ]
            )
        )

    def frame(self) -> pd.DataFrame:
        if not self.keys:
            return pd.DataFrame()
        width = len(self.documents)
        stacked = np.concatenate(self.blocks, axis=1)
        repeat = np.repeat(np.arange(len(self.keys)), width)
        columns = list(zip(*self.keys, strict=True))
        return pd.DataFrame(
            {
                "held_out_engine": pd.Categorical(np.asarray(columns[0])[repeat]),
                "arm": pd.Categorical(np.asarray(columns[1])[repeat]),
                "acquisition": pd.Categorical(np.asarray(columns[2])[repeat]),
                "budget": np.asarray(columns[3], dtype=np.int32)[repeat],
                "draw": np.asarray(columns[4], dtype=np.int16)[repeat],
                "cut_rule": pd.Categorical(np.asarray(columns[5])[repeat]),
                "epsilon": np.asarray(columns[6], dtype=np.float32)[repeat],
                "document_id": pd.Categorical(np.tile(self.documents, len(self.keys))),
                "accepted": stacked[0].astype(np.float32),
                "harmful_accepted": stacked[1].astype(np.float32),
                "beneficial_accepted": stacked[2].astype(np.float32),
                "beneficial_total": stacked[3].astype(np.float32),
                "rows": stacked[4].astype(np.float32),
            }
        )


def evaluate_cell(
    setup: FoldSetup,
    adapted: Adapted,
    budget: Budget,
    reference: Reference,
    context: dict[str, Any],
    sink: CountSink,
    counted: tuple[str, ...],
) -> dict[str, Any]:
    """One (engine, arm, acquisition, budget, draw) cell: every metric, and the per-page counts.

    The deployment is computed under every cut rule the arm carries, at every epsilon, through
    the same `deployed` function SGV12 used, so a difference between two arms in the same cell is
    a difference in the score or in where the cut went and never in how either was measured.
    """
    block = setup.evaluation
    score = adapted.score(setup, block)
    row: dict[str, Any] = dict(context)
    row["arm"] = adapted.arm
    row["setting"] = float(adapted.setting)
    row["budget_rows"] = int(budget.size)
    row["budget_documents"] = int(budget.n_documents)
    row["budget_beneficial"] = int(budget.beneficial.sum())
    row["budget_harmful"] = int(budget.harmful.sum())
    row["budget_in_prefix"] = int(budget.in_prefix.sum())
    row["budget_in_band"] = int(budget.in_band.sum())
    row["identical_to_frozen"] = bool(np.array_equal(score, reference.baseline_score))
    # The proposed arm's three constraint sets, carried into the table rather than left in a
    # fitting note. Whether the prefix-swap term existed at all is the single most important
    # thing to know when reading this arm on an engine whose pool is 3% beneficial.
    pairs = dict(adapted.note.get("pairs", {}))
    for name in ("source", "target", "prefix_swap"):
        row[f"pairs__{name}"] = int(pairs.get(name, {}).get("drawn", 0))
        row[f"pairs__{name}__eligible"] = int(pairs.get(name, {}).get("eligible", 0))
    quality = ranking_quality(score, block.harmful, block.beneficial)
    for key in ("auroc_safe", "pair_accuracy", "average_precision_beneficial", "aurc"):
        row[f"rank__{key}"] = float(quality[key])
    # The same ordering read IN SAMPLE, on the rows the budget bought. The distance between this
    # and the held-out number is what failure test 6 reads: an arm that orders its own budget
    # perfectly and the evaluation block no better than the frozen model has memorised the
    # budget, and no amount of held-out averaging makes that visible on its own.
    if budget.size:
        own = budget_block(setup, budget)
        in_sample = ranking_quality(adapted.score(setup, own), own.harmful, own.beneficial)
        row["budget__auroc_safe"] = float(in_sample["auroc_safe"])
        row["budget__pair_accuracy"] = float(in_sample["pair_accuracy"])
    else:
        row["budget__auroc_safe"] = float("nan")
        row["budget__pair_accuracy"] = float("nan")
    achievable = frontier(score, block.harmful, block.beneficial)
    for epsilon in EPSILONS:
        key = epsilon_key(epsilon)
        row[f"frontier__{key}"] = float(achievable[key]["repair_recall"])
    matched = _accepted_at_share(score, reference.matched_share)
    for name, value in prefix_statistics(
        block, matched, reference.baseline_accepted, reference.ceiling_accepted
    ).items():
        row[f"matched__{name}"] = value

    for rule in cut_rules_for(adapted.arm):
        thresholds = thresholds_for(setup, adapted, budget, rule)
        for epsilon in EPSILONS:
            key = epsilon_key(epsilon)
            tau = float(thresholds[key]["tau"])
            point = deployed(score, block.harmful, block.beneficial, tau, epsilon)
            accepted = (
                np.asarray(score >= tau) if np.isfinite(tau) else np.zeros(block.size, dtype=bool)
            )
            stem = f"dep__{rule}__{key}"
            row[f"{stem}__tau"] = tau
            row[f"{stem}__feasible"] = bool(thresholds[key]["feasible"])
            row[f"{stem}__coverage"] = float(point["coverage"])
            row[f"{stem}__realized_harm_rate"] = float(point["realized_harm_rate"])
            row[f"{stem}__holds_bound"] = bool(point["holds_bound"])
            row[f"{stem}__repair_recall"] = float(point["repair_recall"])
            row[f"{stem}__risk_controlled"] = float(point["risk_controlled_repair_recall"])
            row[f"{stem}__n_accepted"] = int(point["n_accepted"])
            row[f"{stem}__repairs_captured"] = int(point["repairs_captured"])
            row[f"{stem}__joint_harm_rate"] = float(point["joint_harm_rate"])
            if epsilon == PRIMARY_EPSILON:
                # Computed for every cut rule rather than only the primary one, so that which
                # rule the design froze cannot change what this table contains.
                for name, value in prefix_statistics(
                    block, accepted, reference.baseline_accepted, reference.ceiling_accepted
                ).items():
                    row[f"deployed_prefix__{rule}__{name}"] = value
                for name, value in _counterfactuals(
                    block, score, accepted, reference, epsilon
                ).items():
                    row[f"{name}__{rule}"] = value
            if rule in counted:
                sink.add(
                    context,
                    adapted.arm,
                    rule,
                    epsilon,
                    per_document_counts(block, accepted, list(reference.documents)),
                )
    return row


def _counterfactuals(
    block: Block,
    score: np.ndarray,
    accepted: np.ndarray,
    reference: Reference,
    epsilon: float,
) -> dict[str, Any]:
    """The two transplants the mechanism decomposition reads, computed where the vectors live.

    `frozen_rank_new_depth` takes this arm's accepted COUNT and gives it to the frozen ranking;
    `new_rank_frozen_depth` takes the baseline's count and gives it to this arm's ranking. The
    difference between the cell and the first is what the ranking bought at fixed depth; the
    difference between the cell and the second is what the depth bought at fixed ranking. Neither
    is a claim about a mechanism on its own -- section 17 asks for controlled counterfactuals and
    these are the controls.
    """
    depth = int(accepted.sum())
    frozen_at_new_depth = _accepted_at_share(
        reference.baseline_score, depth / block.size if block.size else 0.0
    )
    new_at_frozen_depth = _accepted_at_share(
        score, reference.baseline_depth / block.size if block.size else 0.0
    )

    def endpoint(mask: np.ndarray) -> tuple[float, float]:
        if not mask.any():
            return 0.0, 0.0
        harm = float(block.harmful[mask].mean())
        recall = (
            float(block.beneficial[mask].sum() / block.beneficial.sum())
            if block.beneficial.any()
            else 0.0
        )
        return (recall if harm <= epsilon else 0.0), harm

    recall_a, harm_a = endpoint(frozen_at_new_depth)
    recall_b, harm_b = endpoint(new_at_frozen_depth)
    return {
        "cf__frozen_rank_new_depth__risk_controlled": recall_a,
        "cf__frozen_rank_new_depth__harm": harm_a,
        "cf__new_rank_frozen_depth__risk_controlled": recall_b,
        "cf__new_rank_frozen_depth__harm": harm_b,
        "cf__depth": depth,
        "cf__baseline_depth": reference.baseline_depth,
    }


# ------------------------------------------------------------------ the driver


def load_base() -> tuple[dg.Design, np.ndarray, list[int]]:
    """SGV1's design matrix and SGV5's retrieval configuration, through SGV6's own loader."""
    return s6.load_base()


EXPENSIVE_ARMS = (SGV6_GENERIC, A4_JOINT, A6_CAPACITY, A4_PERMUTED, SOURCE_REFIT)


def draws_for(rule: str, arm: str) -> int:
    """How many repeated budget draws a cell gets, and why it is not always `DRAWS`.

    Only two acquisition rules are stochastic; the other five are deterministic functions of the
    frozen score, the site structure and SGV12's published pool rankings, so a second draw of
    them would return the identical rows and the identical fit. Running one draw there is not a
    reduction in evidence, it is a refusal to publish the same number twenty times.

    Two exceptions, both declared. The label-permutation controls carry the full count under
    every rule, because their randomness is in the permutation rather than in the acquisition.
    The three EXPENSIVE arms -- the two joint refits at 13-18 s a cell and the SGV6 comparison at
    up to 27 s a cell at N = 1000 -- carry `JOINT_DRAWS`, which is the replication SGV6 itself
    ran, and are restricted to the random rule plus whichever rule the design froze.
    """
    if arm in EXPENSIVE_ARMS:
        return JOINT_DRAWS if rule in STOCHASTIC_ACQUISITIONS else 1
    if arm in CONTROL_ARMS:
        return DRAWS
    return DRAWS if rule in STOCHASTIC_ACQUISITIONS else 1


def budgets_for(arm: str, design: Design) -> tuple[int, ...]:
    """Which budgets an arm is fitted at.

    The joint refits run on the coarse ladder because each cell is a full model refit -- unless
    one of them is the arm the source-side simulation SELECTED, in which case it gets the full
    ladder. Criterion 4 is a label-efficiency criterion and it cannot be read on four points when
    the baseline it is compared against has nine.
    """
    if arm == SOURCE_REFIT:
        return (0,)
    if arm == A4_PERMUTED:
        return JOINT_BUDGETS
    if arm in JOINT_ARMS and arm != design.arm:
        return JOINT_BUDGETS
    return BUDGETS


def acquisitions_for(arm: str, design: Design) -> tuple[str, ...]:
    """Which acquisition rules an arm is run under.

    The frozen-feature arms run under all seven, which is what makes the acquisition comparison a
    comparison. The SGV6 baseline and the permutation controls run under the random rule and the
    rule the design froze, which is the contrast every claim about targeting is read from. The
    joint refit that was NOT selected runs under the random rule alone: it is the representation
    control of negative test 9, the question it answers is whether more capacity beats less at
    matched budget, and which rows were bought is not that question. Running it under seven rules
    would cost hours and answer nothing the cheap arms do not.
    """
    if arm == SOURCE_REFIT:
        return (Q_RANDOM,)
    if arm in JOINT_ARMS and arm != design.arm:
        return (Q_RANDOM,)
    if arm in EXPENSIVE_ARMS or arm in CONTROL_ARMS:
        return tuple(dict.fromkeys((Q_RANDOM, design.acquisition)))
    return ACQUISITIONS


def source_pair_block(state: s12.FoldState, seed: int) -> PairBlock:
    """A5's source anchor: SGV12's own pair construction on the source engines' training rows.

    `s12.build_pairs` is called rather than reimplemented, so the eligibility rule, the strata,
    the largest-remainder allocation and the content-derived draw are the ones SGV12 published
    and swept. The budget is smaller than SGV12's because this block is an anchor evaluated on
    every one of several thousand cells rather than a fit performed four times; the sensitivity
    of the choice is measured on source-side simulation in `--preregister` and nowhere else.
    """
    view = s12.source_view(state)
    pairs = s12.build_pairs(SOURCE_PAIR_FAMILY, view, s12.ENGINE_SHARD, seed, SOURCE_PAIR_BUDGET)
    index = state.index(FIT)
    score = state.frozen_score(index)
    return PairBlock(
        name="source",
        differences=view.features[pairs.positive] - view.features[pairs.negative],
        margin=np.asarray(score[pairs.positive] - score[pairs.negative], dtype=float),
        eligible=int(pairs.eligible),
    )


@dataclass(frozen=True, slots=True)
class Design:
    """The configuration `--preregister` froze. Every field is read, none is chosen at run time."""

    band_multiple: float
    lambda_target: float
    lambda_prefix: float
    ridge: float
    target_weight: float
    acquisition: str
    arm: str
    primary_cut: str
    operating_budget: int
    coverage_floor: float


def load_design() -> Design:
    """Read the frozen design, and refuse to run without it.

    `--adapt` cannot be run before `--preregister`, and that ordering is the point: it is what
    makes the outer evaluation a single pass over a configuration nothing in it chose.
    """
    if not DESIGN_RECORD.is_file():
        raise PhaseError(
            "design_record.json is missing; run --preregister before --adapt. The design has to "
            "be frozen from source-side simulation before an outer target row is read"
        )
    record = cc_read_json(DESIGN_RECORD)["frozen_design"]
    return Design(
        band_multiple=float(record["band_multiple"]),
        lambda_target=float(record["lambda_target"]),
        lambda_prefix=float(record["lambda_prefix"]),
        ridge=float(record["ridge"]),
        target_weight=float(record["target_weight"]),
        acquisition=str(record["acquisition"]),
        arm=str(record["arm"]),
        primary_cut=str(record["primary_cut"]),
        operating_budget=int(record["operating_budget"]),
        coverage_floor=float(record["coverage_floor"]),
    )


def primary_rule(design: Design) -> str:
    """The acquisition rule the SELECTED arm actually ran under.

    Every arm runs under the rule the design froze -- the frozen-feature arms because they run
    under all seven, the SGV6 baseline and the selected arm because they are given it explicitly.
    This exists so that the answer is stated once rather than re-derived in every analysis phase.
    """
    return design.acquisition


def fit_one(
    setup: FoldSetup,
    arm: str,
    budget: Budget,
    order: np.ndarray,
    design: Design,
    source: PairBlock,
    seed: int,
) -> tuple[Adapted, dict[str, Any]]:
    """Fit one arm at one budget. The single dispatch point, so no arm can acquire a side door."""
    if arm == A0_FROZEN:
        return frozen_arm(), {}
    if arm == A2_CUT:
        return cut_arm(), {}
    if arm == A1_CALIBRATION:
        return fit_calibration_arm(budget), {}
    if arm in (A3_HEAD, A3_PERMUTED):
        return fit_head_arm(budget, arm), {}
    if arm in (A5_PREFIX, A5_PERMUTED):
        return (
            fit_prefix_arm(
                budget,
                source,
                design.lambda_target,
                design.lambda_prefix,
                design.ridge,
                seed,
                arm,
            ),
            {},
        )
    if arm == SOURCE_REFIT:
        fit = setup.state.index(FIT)
        return (
            Adapted(
                arm,
                REFIT,
                None,
                _refit(
                    setup,
                    fit,
                    np.ones(fit.size),
                    setup.state.design.harmful[fit],
                    setup.state.design.beneficial[fit],
                    capacity=False,
                ),
                None,
                0.0,
                {"target_labels_used": 0, "fit_rows": int(fit.size), "fittable": True},
            ),
            {},
        )
    if arm in (A4_JOINT, A4_PERMUTED):
        return fit_joint_arm(setup, budget, design.target_weight, False, arm), {}
    if arm == A6_CAPACITY:
        return fit_joint_arm(setup, budget, design.target_weight, True, arm), {}
    if arm == SGV6_GENERIC:
        return fit_sgv6_arm(setup, order, budget.size)
    raise PhaseError(f"unknown arm {arm!r}")


Q_ORACLE = "q_oracle_swap_labels"


def oracle_acquisition_order(setup: FoldSetup) -> np.ndarray:
    """DIAGNOSTIC ONLY. Which pool rows an omniscient annotator would have labelled first.

    This function reads `FoldSetup.outcomes` directly and is therefore not an acquisition rule:
    it is the upper bound section 23 asks for, and it exists to separate two failures that look
    identical from the outside -- adaptation that cannot use labels, and acquisition that cannot
    find useful ones. It orders the pool by whether a row is a swap constraint at all (a harmful
    candidate inside the prefix, or a beneficial candidate outside it) and then by closeness to
    the cut. `tests/leakage` asserts that its only call site produces cells whose arm name begins
    with the oracle prefix and that no selectable arm is ever fitted on its ordering.
    """
    view = setup.view
    harmful = setup.outcomes["harmful"]
    beneficial = setup.outcomes["beneficial"]
    decisive = (harmful & view.in_prefix) | (beneficial & ~view.in_prefix)
    return np.asarray(
        np.lexsort((np.arange(view.size), -_prefix_swap_value(view), -decisive.astype(float))),
        dtype=int,
    )


def _oracle_cells(
    setup: FoldSetup, reference: Reference, context: dict[str, Any], sink: CountSink
) -> list[dict[str, Any]]:
    """O1 and O2: the two evaluation-label upper bounds, deployed under the oracle cut.

    Neither is a method and neither may enter a selection, a criterion or a headline. They are
    the rungs of SGV11's and SGV12's ladder, recomputed here on the same block so that this
    stage's recovery fractions have the same denominator those stages published.
    """
    block = setup.evaluation
    rows: list[dict[str, Any]] = []
    for arm, score in (
        (ORACLE_CUT, reference.baseline_score),
        (ORACLE_RERANK, reference.ceiling_score),
    ):
        achievable = frontier(score, block.harmful, block.beneficial)
        row: dict[str, Any] = dict(context)
        row.update(
            {
                "arm": arm,
                "setting": 0.0,
                "budget_rows": 0,
                "budget_documents": 0,
                "budget_beneficial": 0,
                "budget_harmful": 0,
                "budget_in_prefix": 0,
                "budget_in_band": 0,
                "identical_to_frozen": bool(np.array_equal(score, reference.baseline_score)),
            }
        )
        quality = ranking_quality(score, block.harmful, block.beneficial)
        for key in ("auroc_safe", "pair_accuracy", "average_precision_beneficial", "aurc"):
            row[f"rank__{key}"] = float(quality[key])
        matched = _accepted_at_share(score, reference.matched_share)
        for name, value in prefix_statistics(
            block, matched, reference.baseline_accepted, reference.ceiling_accepted
        ).items():
            row[f"matched__{name}"] = value
        order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
        for epsilon in EPSILONS:
            key = epsilon_key(epsilon)
            cell = achievable[key]
            row[f"frontier__{key}"] = float(cell["repair_recall"])
            accepted = np.zeros(block.size, dtype=bool)
            accepted[order[: int(cell["n_accepted"])]] = True
            stem = f"dep__{CUT_ORACLE}__{key}"
            row[f"{stem}__tau"] = float("nan")
            row[f"{stem}__feasible"] = True
            row[f"{stem}__coverage"] = float(cell["coverage"])
            row[f"{stem}__realized_harm_rate"] = float(cell["realized_harm_rate"])
            row[f"{stem}__holds_bound"] = True
            row[f"{stem}__repair_recall"] = float(cell["repair_recall"])
            row[f"{stem}__risk_controlled"] = float(cell["repair_recall"])
            row[f"{stem}__n_accepted"] = int(cell["n_accepted"])
            row[f"{stem}__repairs_captured"] = int(block.beneficial[accepted].sum())
            row[f"{stem}__joint_harm_rate"] = (
                float(block.harmful[accepted].sum() / block.size) if block.size else 0.0
            )
            if epsilon == PRIMARY_EPSILON:
                for name, value in prefix_statistics(
                    block, accepted, reference.baseline_accepted, reference.ceiling_accepted
                ).items():
                    row[f"deployed_prefix__{CUT_ORACLE}__{name}"] = value
            sink.add(
                context,
                arm,
                CUT_ORACLE,
                epsilon,
                per_document_counts(block, accepted, list(reference.documents)),
            )
        rows.append(row)
    return rows


def _write(path: Path, payload: dict[str, Any], schema: str) -> None:
    """One artifact, with the six facts every artifact in this project has to carry."""
    body = {
        "schema_version": f"sgv13-{schema}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "artifact": schema,
        "hypothesis_id": HYPOTHESIS,
        "development_only": True,
        "confirmatory_accessed": False,
        "synthetic": False,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    cc._write_json_once(path, body)


def _orders(setup: FoldSetup, rule: str, count: int) -> dict[int, np.ndarray]:
    """The acquisition orderings for one rule, one per draw where the rule is stochastic.

    A deterministic rule is computed once and reused, not recomputed per draw: the farthest-point
    traversal is quadratic in the pool and recomputing it twenty times to get twenty copies of the
    same permutation would be pure waste.
    """
    if rule not in STOCHASTIC_ACQUISITIONS:
        single = acquisition_order(
            setup.view, rule, setup.base, _stable_seed("sgv13-acquisition", setup.held_out, rule)
        )
        return dict.fromkeys(range(max(count, 1)), single)
    return {
        draw: acquisition_order(
            setup.view,
            rule,
            setup.base,
            _stable_seed("sgv13-acquisition", setup.held_out, rule, str(draw)),
        )
        for draw in range(count)
    }


def run_adapt() -> int:
    """Fit every arm at every budget under every acquisition rule, and write the cell tables.

    This is the only phase that reads a target label, and it reads them exactly once: the design
    it runs under is `--preregister`'s, loaded from disk and never recomputed. Everything the
    analysis phases need is computed here, where the score vectors live, and persisted; nothing
    downstream refits a model or rescores a block.
    """
    design = load_design()
    base, signatures, retrieval_columns = load_base()
    selection = s6.frozen_selection()
    frozen = s6.load_frozen_scores()
    started = time.time()
    cells: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    identity: dict[str, Any] = {}
    inventory: dict[str, Any] = {}
    # The cut rules whose per-document decomposition is persisted: the one the primary endpoint
    # is read under, and the oracle one. See `CountSink`.
    counted = (design.primary_cut, CUT_ORACLE)

    for engine in sorted(base.engines):
        fold = build_fold(base, engine)
        setup = build_setup(
            base,
            fold,
            engine,
            signatures,
            retrieval_columns,
            selection[engine],
            frozen,
            design.band_multiple,
        )
        reference = build_reference(setup)
        sink = CountSink(keys=[], blocks=[], documents=list(reference.documents))
        source = source_pair_block(
            setup.state, _stable_seed("sgv13-source-pairs", engine, SOURCE_PAIR_FAMILY)
        )
        # SGV6's own reconstruction of the same fold, run once here as an independent check that
        # the pool this stage adapts on is the pool SGV6 adapted on, to the last bit.
        sgv6_base = s6.build_base_fold(
            base, engine, signatures, retrieval_columns, selection[engine], frozen
        )
        utility_gap = float(np.abs(sgv6_base.pool.utility - setup.view.frozen_score).max())
        adapter_gap = float(np.abs(sgv6_base.pool.adapter_input - setup.view.adapter_input).max())
        if utility_gap != 0.0 or adapter_gap != 0.0:
            raise PhaseError(
                f"{engine}: this stage's reconstruction of the adaptation pool differs from "
                f"SGV6's by {utility_gap:g} on the score and {adapter_gap:g} on the adapter "
                "input; the SGV6 comparison would not be a comparison on the same rows"
            )
        # The distribution diagnostics the EasyOCR analysis is pre-specified to read. SGV12 found
        # the median evaluation-minus-calibration score shift to be -1.1485 on docTR and +0.2074
        # on EasyOCR, which is why a threshold transported by VALUE accepts a different share on
        # the two engines; these are the same three quantities on the same blocks.
        calibration_score = setup.calibration.frozen_score
        evaluation_score = setup.evaluation.frozen_score
        pool_score = setup.view.frozen_score
        # The frozen cut this stage places for the baseline must be the one `build_state` derived
        # from SGV5's own published call. They are two paths to the same `select_threshold`
        # invocation and a difference between them would mean the shared decision rule is not
        # shared; SGV12 asserted the same identity against SGV10's published table.
        cut_gap = max(
            abs(
                float(reference.frozen_thresholds[epsilon_key(epsilon)]["tau"])
                - float(setup.state.source_threshold[epsilon_key(epsilon)])
            )
            for epsilon in EPSILONS
        )
        if cut_gap != 0.0:
            raise PhaseError(
                f"{engine}: the frozen cut placed by this stage differs from the one "
                f"`build_state` derived by {cut_gap:g}; the decision rule below the score is "
                "supposed to be one call and this says it is two"
            )
        identity[engine] = {
            **setup.identity,
            "frozen_thresholds": {
                epsilon_key(epsilon): float(setup.state.source_threshold[epsilon_key(epsilon)])
                for epsilon in EPSILONS
            },
            "frozen_cut_max_abs_difference": cut_gap,
            "median_score_shift_evaluation_minus_calibration": float(
                np.median(evaluation_score) - np.median(calibration_score)
            ),
            "median_score_shift_pool_minus_calibration": float(
                np.median(pool_score) - np.median(calibration_score)
            ),
            "evaluation_harmful_rate": float(setup.evaluation.harmful.mean()),
            "evaluation_beneficial_rate": float(setup.evaluation.beneficial.mean()),
            "evaluation_beneficial_rows": int(setup.evaluation.beneficial.sum()),
            "sites_offering_both_outcomes": int(
                sum(
                    1
                    for site in set(setup.evaluation.sites.tolist())
                    for mask in [setup.evaluation.sites == site]
                    if setup.evaluation.harmful[mask].any()
                    and setup.evaluation.beneficial[mask].any()
                )
            ),
            "sgv6_pool_score_max_abs_difference": utility_gap,
            "sgv6_adapter_input_max_abs_difference": adapter_gap,
            "baseline_deployed_coverage_at_primary_epsilon": reference.matched_share,
            "baseline_accepted_rows": reference.baseline_depth,
            "source_anchor_pairs": int(source.size),
            "source_anchor_eligible_pairs": int(source.eligible),
        }
        orders: dict[str, dict[int, np.ndarray]] = {}
        for arm in ALL_ARMS:
            if arm in ORACLE_ARMS:
                continue
            arm_started = time.time()
            before = len(cells)
            for rule in acquisitions_for(arm, design):
                count = draws_for(rule, arm)
                if rule not in orders or len(orders[rule]) < count:
                    orders[rule] = _orders(setup, rule, count)
                for size in budgets_for(arm, design):
                    for draw in range(count if size else 1):
                        order = orders[rule][draw if rule in STOCHASTIC_ACQUISITIONS else 0]
                        seed = _stable_seed("sgv13-fit", engine, arm, rule, str(size), str(draw))
                        permutation = (
                            _stable_seed("sgv13-permute", engine, rule, str(size), str(draw))
                            if arm in CONTROL_ARMS and size
                            else None
                        )
                        budget = purchase(setup, order, size, permutation)
                        adapted, extra = fit_one(setup, arm, budget, order, design, source, seed)
                        context = {
                            "held_out_engine": engine,
                            "acquisition": rule,
                            "budget": int(size),
                            "draw": int(draw),
                        }
                        row = evaluate_cell(
                            setup, adapted, budget, reference, context, sink, counted
                        )
                        row["selected_family"] = str(adapted.note.get("selected_family", ""))
                        row["fittable"] = bool(adapted.note.get("fittable", True))
                        cells.append(row)
                        if extra:
                            inventory.setdefault("sgv6_selection", []).append(
                                {**context, "selected_arm": extra["selected_arm"]}
                            )
            print(
                f"    {engine}/{arm}: {len(cells) - before} cells in "
                f"{round(time.time() - arm_started)}s",
                flush=True,
            )
        cells.extend(
            _oracle_cells(
                setup,
                reference,
                {"held_out_engine": engine, "acquisition": Q_ORACLE, "budget": 0, "draw": 0},
                sink,
            )
        )
        # O3's acquisition arm: the proposed method fitted on an omniscient ordering. Diagnostic.
        oracle_order = oracle_acquisition_order(setup)
        for size in NONZERO_BUDGETS:
            budget = purchase(setup, oracle_order, size)
            # O3 fits the SELECTED arm on an omniscient ordering. Fitting a different arm here
            # and comparing it to the selected one would report the difference between two arms
            # as if it were the value of the acquisition.
            adapted, _ = fit_one(
                setup,
                design.arm,
                budget,
                oracle_order,
                design,
                source,
                _stable_seed("sgv13-oracle-fit", engine, str(size)),
            )
            row = evaluate_cell(
                setup,
                Adapted(
                    arm=f"o3_oracle_acquisition__{adapted.arm}",
                    kind=adapted.kind,
                    weights=adapted.weights,
                    model=adapted.model,
                    calibrator=adapted.calibrator,
                    setting=adapted.setting,
                    note=adapted.note,
                    scorer=adapted.scorer,
                ),
                budget,
                reference,
                {
                    "held_out_engine": engine,
                    "acquisition": Q_ORACLE,
                    "budget": int(size),
                    "draw": 0,
                },
                sink,
                counted,
            )
            cells.append(row)
        frames.append(sink.frame())
        print(
            f"  {engine}: {len(cells)} cells, pool {setup.view.size} rows / "
            f"{setup.identity['pool_documents']} documents, prefix {setup.prefix_size}, "
            f"band {setup.band_size}",
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(cells)
    # Decision 4, asserted here rather than discovered in `--negative`: at budget 0 every
    # adaptation arm returns SGV5's score bit for bit, under every acquisition rule.
    # `c4_source_only_refit` is a REFIT, not the frozen model, so it is exempt by name and its
    # distance from the frozen score is published instead -- that distance is the whole point of
    # the arm, because it is how much the fitting path moves on its own.
    exempt = [*ORACLE_ARMS, SOURCE_REFIT]
    baseline_cells = frame[(frame["budget"] == 0) & (~frame["arm"].isin(exempt))]
    if not bool(baseline_cells["identical_to_frozen"].all()):
        broken = sorted(
            {
                str(name)
                for name in baseline_cells[~baseline_cells["identical_to_frozen"]]["arm"].tolist()
            }
        )
        raise PhaseError(
            f"these arms did not reproduce the frozen score exactly at budget 0: {broken}. "
            "Decision 4 says N = 0 IS SGV5, and a measured gain is only a statement about the "
            "labels if that holds"
        )
    frame.to_parquet(SCORES, index=False)
    counts = pd.concat(frames, ignore_index=True)
    counts.to_parquet(COUNTS, index=False)
    _write(
        FROZEN_IDENTITY,
        {
            "per_engine": identity,
            "frozen_upstream": verify_frozen(),
            "note": (
                "the rebuilt SGV5 decision score against the published one, and this stage's "
                "reconstruction of the adaptation pool against SGV6's. Both are maximum absolute "
                "differences over every row of the block and both must be exactly 0.0."
            ),
        },
        "frozen_identity",
    )
    _write(
        BUDGET_INVENTORY,
        {
            "design": _design_dict(design),
            "budgets": list(BUDGETS),
            "joint_budgets": list(JOINT_BUDGETS),
            "draws": {
                arm: {rule: draws_for(rule, arm) for rule in acquisitions_for(arm, design)}
                for arm in ADAPTATION_ARMS + CONTROL_ARMS
            },
            "cells": len(frame),
            "document_count_rows": len(counts),
            "per_engine": identity,
            "sgv6_selection": inventory.get("sgv6_selection", []),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "budget_inventory",
    )
    print(
        f"adapt: {len(frame)} cells, {len(counts)} document rows -> {cc._relative(SCORES)} "
        f"({round(time.time() - started)}s)"
    )
    return 0


def _design_dict(design: Design) -> dict[str, Any]:
    return {
        "band_multiple": design.band_multiple,
        "lambda_target": design.lambda_target,
        "lambda_prefix": design.lambda_prefix,
        "ridge": design.ridge,
        "target_weight": design.target_weight,
        "acquisition": design.acquisition,
        "arm": design.arm,
        "primary_cut": design.primary_cut,
        "operating_budget": design.operating_budget,
        "coverage_floor": design.coverage_floor,
    }


# ------------------------------------------------------------------ source-side pre-registration
#
# Q3 is absent from the selectable set. SGV12's published rankings on an inner fold's pool were
# fitted with the OUTER held-out engine in their source set, so using them to choose a design
# that is then applied to that engine would let its labels reach the choice. The rule is still
# measured on the outer folds and reported; it is simply not eligible to be selected, and
# `--negative` asserts that the selected acquisition is never Q3.

SELECTABLE_ACQUISITIONS = tuple(rule for rule in ACQUISITIONS if rule != Q_DISAGREEMENT)
# The inner ladder covers the upper half of the outer one. A shorter inner ladder
# would cap the operating budget the criteria are read at, and a method that only
# works at 500 labels would then fail a criterion for a reason that is an artefact
# of the simulation rather than a property of the method.
INNER_BUDGETS = (25, 100, 250, 500, 1000)
INNER_DRAWS = 3
INNER_JOINT_BUDGET = 250
COVERAGE_FLOOR_FRACTION = 0.25


def with_band(setup: FoldSetup, multiple: float) -> FoldSetup:
    """The same fold with a different adjacent-band width. Recomputes masks, refits nothing."""
    core, band = prefix_masks(setup.view.frozen_score, setup.view.accepted_share, multiple)
    view = PoolView(
        index=setup.view.index,
        features=setup.view.features,
        frozen_score=setup.view.frozen_score,
        adapter_input=setup.view.adapter_input,
        documents=setup.view.documents,
        sites=setup.view.sites,
        candidates=setup.view.candidates,
        disagreement=setup.view.disagreement,
        in_prefix=core,
        in_band=band,
        accepted_share=setup.view.accepted_share,
        frozen_threshold=setup.view.frozen_threshold,
    )
    return FoldSetup(
        held_out=setup.held_out,
        state=setup.state,
        base=setup.base,
        columns=setup.columns,
        lambda_=setup.lambda_,
        evaluation=setup.evaluation,
        calibration=setup.calibration,
        view=view,
        band_size=int(band.sum()),
        prefix_size=int(core.sum()),
        outcomes=setup.outcomes,
        identity=setup.identity,
        projection=setup.projection,
        scaler=setup.scaler,
    )


def inner_endpoint(
    setup: FoldSetup,
    arm: str,
    design: Design,
    source: PairBlock,
    rule: str,
    sizes: tuple[int, ...],
    draws: int,
) -> dict[str, float]:
    """One configuration's risk-controlled repair recall on one inner fold, averaged over draws.

    The endpoint is the outer stage's primary endpoint computed on the inner block under the
    PRIMARY cut rule, so what the simulation optimises and what the outer evaluation reports are
    the same quantity. Nothing about the outer held-out engine is visible here: SGV9's inner fold
    excludes it from every block and this function reads no artifact keyed by it.
    """
    block = setup.evaluation
    per_size: dict[int, list[float]] = {size: [] for size in sizes}
    count = draws if rule in STOCHASTIC_ACQUISITIONS else 1
    orders = _orders(setup, rule, count)
    for size in sizes:
        for draw in range(count):
            order = orders[draw]
            budget = purchase(setup, order, size)
            adapted, _ = fit_one(
                setup,
                arm,
                budget,
                order,
                design,
                source,
                _stable_seed("sgv13-inner", setup.held_out, arm, rule, str(size), str(draw)),
            )
            score = adapted.score(setup, block)
            thresholds = thresholds_for(setup, adapted, budget, design.primary_cut)
            tau = float(thresholds[PRIMARY_KEY]["tau"])
            point = deployed(score, block.harmful, block.beneficial, tau, PRIMARY_EPSILON)
            per_size[size].append(float(point["risk_controlled_repair_recall"]))
    out = {f"n_{size}": float(np.mean(per_size[size])) for size in sizes}
    out["mean"] = float(np.mean(list(out.values())))
    return out


def _mean_over(rows: list[dict[str, float]]) -> float:
    values = [row["mean"] for row in rows if np.isfinite(row["mean"])]
    return float(np.mean(values)) if values else float("-inf")


def _default_design(band: float) -> Design:
    """The midpoint configuration every sweep starts from, so no stage reads its own answer."""
    return Design(
        band_multiple=band,
        lambda_target=LAMBDA_TARGET[0],
        lambda_prefix=LAMBDA_PREFIX[1],
        ridge=RIDGE_GRID[1],
        target_weight=TARGET_WEIGHTS[1],
        acquisition=Q_RANDOM,
        arm=A5_PREFIX,
        primary_cut=CUT_FROZEN,
        operating_budget=INNER_BUDGETS[-1],
        coverage_floor=0.0,
    )


def run_preregister() -> int:
    """Choose the whole design on source-side simulation, then freeze it.

    Each source engine takes a turn as a pseudo-target inside SGV9's inner folds, and the entire
    few-shot protocol runs there: the same pool construction, the same acquisition rules, the
    same arms, the same cut rules and the same endpoint. The outer held-out engine appears in no
    inner block. Five stages run in a declared order and each reads only the choices the stages
    before it made, so no stage can be tuned on its own answer:

        1  the adjacent-band width          4  the acquisition rule
        2  the proposed arm's pair weights  5  the arm, and the operating budget it is read at
        3  the joint arms' target weight     6  the cut rule the primary endpoint is read under

    Stages 1 to 5 all read the endpoint under `frozen_source`, which reads no target row, so the
    arm and the acquisition are chosen on what the RANKING buys and cannot be contaminated by
    cut-placement noise. The cut rule is chosen afterwards, for the design that was selected,
    because feasibility probe 1 found that whether a target cut can be placed at all depends on
    the shape of the budget the acquisition produces.

    The coverage floor of criterion 6 is set last, from the frozen baseline's own inner
    deployment, so "accepts a meaningful number of corrections" is calibrated to what this corpus
    supports rather than to a number chosen to be passable.
    """
    started = time.time()
    base, signatures, retrieval_columns = load_base()
    selection = s6.frozen_selection()
    setups: list[FoldSetup] = []
    for held_out in sorted(base.engines):
        for target in sorted(e for e in base.engines if e != held_out):
            fold = inner_fold(base, held_out, target)
            setups.append(
                build_setup(
                    base,
                    fold,
                    target,
                    signatures,
                    retrieval_columns,
                    selection[held_out],
                    None,
                    BAND_MULTIPLES[1],
                    disagreement=False,
                )
            )
    sources = {
        id(setup): source_pair_block(
            setup.state, _stable_seed("sgv13-source-pairs", setup.held_out, SOURCE_PAIR_FAMILY)
        )
        for setup in setups
    }
    print(f"  {len(setups)} inner folds built ({round(time.time() - started)}s)", flush=True)

    record: dict[str, Any] = {}

    # 1 -- the adjacent-band width.
    band_scores: dict[str, float] = {}
    for multiple in BAND_MULTIPLES:
        rows = [
            inner_endpoint(
                with_band(setup, multiple),
                A5_PREFIX,
                _default_design(multiple),
                sources[id(setup)],
                Q_RANDOM,
                INNER_BUDGETS,
                INNER_DRAWS,
            )
            for setup in setups
        ]
        band_scores[f"band_{multiple:g}"] = _mean_over(rows)
    band = max(BAND_MULTIPLES, key=lambda m: (band_scores[f"band_{m:g}"], -m))
    record["1_band_multiple"] = {"grid": band_scores, "selected": float(band)}
    banded = [with_band(setup, band) for setup in setups]
    print(f"  band multiple {band:g} ({round(time.time() - started)}s)", flush=True)

    # 2 -- the proposed arm's pair weights and ridge.
    weight_scores: dict[str, float] = {}
    for lambda_target in LAMBDA_TARGET:
        for lambda_prefix in LAMBDA_PREFIX:
            for ridge in RIDGE_GRID:
                candidate = Design(
                    band_multiple=band,
                    lambda_target=lambda_target,
                    lambda_prefix=lambda_prefix,
                    ridge=ridge,
                    target_weight=TARGET_WEIGHTS[1],
                    acquisition=Q_RANDOM,
                    arm=A5_PREFIX,
                    primary_cut=CUT_FROZEN,
                    operating_budget=INNER_BUDGETS[-1],
                    coverage_floor=0.0,
                )
                key = f"lt_{lambda_target:g}__lp_{lambda_prefix:g}__ridge_{ridge:g}"
                weight_scores[key] = _mean_over(
                    [
                        inner_endpoint(
                            setup,
                            A5_PREFIX,
                            candidate,
                            sources[id(original)],
                            Q_RANDOM,
                            INNER_BUDGETS,
                            INNER_DRAWS,
                        )
                        for setup, original in zip(banded, setups, strict=True)
                    ]
                )
    best_key = max(weight_scores, key=lambda k: (weight_scores[k], k))
    parts = dict(part.split("_", 1) for part in best_key.split("__"))
    lambda_target, lambda_prefix, ridge = (
        float(parts["lt"]),
        float(parts["lp"]),
        float(parts["ridge"]),
    )
    record["2_pair_weights"] = {
        "grid": weight_scores,
        "selected": {
            "lambda_target": lambda_target,
            "lambda_prefix": lambda_prefix,
            "ridge": ridge,
        },
    }
    print(
        f"  lambda_target {lambda_target:g} lambda_prefix {lambda_prefix:g} ridge {ridge:g} "
        f"({round(time.time() - started)}s)",
        flush=True,
    )

    # 3 -- the joint arms' target weight, at one budget because each cell is a full refit.
    joint_scores: dict[str, float] = {}
    for weight in TARGET_WEIGHTS:
        candidate = Design(
            band_multiple=band,
            lambda_target=lambda_target,
            lambda_prefix=lambda_prefix,
            ridge=ridge,
            target_weight=weight,
            acquisition=Q_RANDOM,
            arm=A4_JOINT,
            primary_cut=CUT_FROZEN,
            operating_budget=INNER_JOINT_BUDGET,
            coverage_floor=0.0,
        )
        joint_scores[f"weight_{weight:g}"] = _mean_over(
            [
                inner_endpoint(
                    setup,
                    A4_JOINT,
                    candidate,
                    sources[id(original)],
                    Q_RANDOM,
                    (INNER_JOINT_BUDGET,),
                    1,
                )
                for setup, original in zip(banded, setups, strict=True)
            ]
        )
    target_weight = max(TARGET_WEIGHTS, key=lambda w: (joint_scores[f"weight_{w:g}"], -w))
    record["3_target_weight"] = {"grid": joint_scores, "selected": float(target_weight)}
    print(f"  target weight {target_weight:g} ({round(time.time() - started)}s)", flush=True)

    frozen_design = Design(
        band_multiple=band,
        lambda_target=lambda_target,
        lambda_prefix=lambda_prefix,
        ridge=ridge,
        target_weight=target_weight,
        acquisition=Q_RANDOM,
        arm=A5_PREFIX,
        primary_cut=CUT_FROZEN,
        operating_budget=INNER_BUDGETS[-1],
        coverage_floor=0.0,
    )

    # 4 -- the acquisition rule.
    acquisition_scores: dict[str, float] = {}
    for rule in SELECTABLE_ACQUISITIONS:
        acquisition_scores[rule] = _mean_over(
            [
                inner_endpoint(
                    setup,
                    A5_PREFIX,
                    frozen_design,
                    sources[id(original)],
                    rule,
                    INNER_BUDGETS,
                    INNER_DRAWS,
                )
                for setup, original in zip(banded, setups, strict=True)
            ]
        )
    acquisition = max(
        SELECTABLE_ACQUISITIONS,
        key=lambda rule: (acquisition_scores[rule], -SELECTABLE_ACQUISITIONS.index(rule)),
    )
    record["4_acquisition"] = {
        "grid": acquisition_scores,
        "selected": acquisition,
        "excluded": {
            Q_DISAGREEMENT: (
                "SGV12's published rankings on an inner pool were fitted with the outer "
                "held-out engine in their source set, so selecting on them would let that "
                "engine's labels reach the design. The rule is measured on the outer folds and "
                "reported; it is not eligible to be selected."
            )
        },
    }
    print(f"  acquisition {acquisition} ({round(time.time() - started)}s)", flush=True)

    # 5 -- the arm, and the budget it is read at.
    arm_curves: dict[str, dict[str, float]] = {}
    for arm in SELECTABLE_ARMS:
        sizes = (INNER_JOINT_BUDGET,) if arm in JOINT_ARMS else INNER_BUDGETS
        rows = [
            inner_endpoint(
                setup,
                arm,
                Design(
                    band_multiple=band,
                    lambda_target=lambda_target,
                    lambda_prefix=lambda_prefix,
                    ridge=ridge,
                    target_weight=target_weight,
                    acquisition=acquisition,
                    arm=arm,
                    primary_cut=CUT_FROZEN,
                    operating_budget=INNER_BUDGETS[-1],
                    coverage_floor=0.0,
                ),
                sources[id(original)],
                acquisition,
                sizes,
                1 if arm in JOINT_ARMS else INNER_DRAWS,
            )
            for setup, original in zip(banded, setups, strict=True)
        ]
        arm_curves[arm] = {
            key: float(np.mean([row[key] for row in rows if np.isfinite(row[key])]))
            for key in rows[0]
        }
    arm = max(
        SELECTABLE_ARMS,
        key=lambda name: (arm_curves[name]["mean"], -SELECTABLE_ARMS.index(name)),
    )
    curve = arm_curves[arm]
    ladder = [size for size in INNER_BUDGETS if f"n_{size}" in curve]
    ceiling = max(curve[f"n_{size}"] for size in ladder) if ladder else 0.0
    operating = next(
        (size for size in ladder if curve[f"n_{size}"] >= 0.95 * ceiling),
        ladder[-1] if ladder else INNER_BUDGETS[-1],
    )
    if operating not in BUDGETS:
        raise PhaseError(
            f"the operating budget {operating} is not on the ladder {arm} runs -- the criteria "
            "would be read at a budget with no cell. The inner ladder and the arm's outer ladder "
            "must agree before the design can be frozen"
        )
    record["5_arm"] = {
        "grid": arm_curves,
        "selected": arm,
        "operating_budget": int(operating),
        "operating_budget_rule": (
            "the smallest inner budget whose mean inner endpoint reaches 95% of the arm's own "
            "inner maximum, so the operating point is where the curve flattens rather than where "
            "it happens to peak on three inner points"
        ),
    }

    chosen_design = Design(
        band_multiple=band,
        lambda_target=lambda_target,
        lambda_prefix=lambda_prefix,
        ridge=ridge,
        target_weight=target_weight,
        acquisition=acquisition,
        arm=arm,
        primary_cut=CUT_FROZEN,
        operating_budget=int(operating),
        coverage_floor=0.0,
    )

    # 6 -- the cut rule the primary endpoint is read under, given the arm and the rule.
    #
    # This runs LAST, not first, and the ordering is the point. Where the cut goes depends on
    # what the budget looks like, and what the budget looks like depends on the acquisition rule
    # -- feasibility probe 1 found a prefix-targeted budget is 93% harmful and can certify
    # nothing. So the band, the weights, the acquisition and the arm are all chosen under
    # `frozen_source`, which reads no target row and therefore isolates the ranking, and the cut
    # rule is chosen afterwards for the design that was actually selected. A rule that certifies
    # a bound by accepting nothing is not a deployment, so a rule whose mean inner coverage is
    # zero is rejected however safe it looks.
    cut_rows: dict[str, dict[str, float]] = {}
    for rule in CUT_RULES:
        coverage: list[float] = []
        endpoint: list[float] = []
        holds: list[float] = []
        sizes = (INNER_JOINT_BUDGET,) if arm in JOINT_ARMS else INNER_BUDGETS
        repeats = 1 if arm in JOINT_ARMS else INNER_DRAWS
        for setup, original in zip(banded, setups, strict=True):
            source_block = sources[id(original)]
            candidate = Design(**{**_design_dict(chosen_design), "primary_cut": rule})
            orders = _orders(setup, acquisition, repeats)
            for size in sizes:
                for draw in range(repeats if acquisition in STOCHASTIC_ACQUISITIONS else 1):
                    budget = purchase(setup, orders[draw], size)
                    adapted, _ = fit_one(
                        setup,
                        arm,
                        budget,
                        orders[draw],
                        candidate,
                        source_block,
                        _stable_seed("sgv13-cut", setup.held_out, rule, str(size), str(draw)),
                    )
                    block = setup.evaluation
                    score = adapted.score(setup, block)
                    tau = float(thresholds_for(setup, adapted, budget, rule)[PRIMARY_KEY]["tau"])
                    point = deployed(score, block.harmful, block.beneficial, tau, PRIMARY_EPSILON)
                    coverage.append(float(point["coverage"]))
                    endpoint.append(float(point["risk_controlled_repair_recall"]))
                    holds.append(float(bool(point["holds_bound"])))
        cut_rows[rule] = {
            "mean_coverage": float(np.mean(coverage)),
            "mean_risk_controlled_repair_recall": float(np.mean(endpoint)),
            "share_holding_the_bound": float(np.mean(holds)),
            "share_accepting_nothing": float(np.mean(np.asarray(coverage) == 0.0)),
        }
    feasible = [rule for rule in CUT_RULES if cut_rows[rule]["mean_coverage"] > 0.0]
    primary_cut = max(
        feasible or list(CUT_RULES),
        key=lambda rule: (
            cut_rows[rule]["mean_risk_controlled_repair_recall"],
            -CUT_RULES.index(rule),
        ),
    )
    record["6_primary_cut_rule"] = {
        "grid": cut_rows,
        "selected": primary_cut,
        "rule": (
            "the highest mean inner risk-controlled repair recall among the rules whose mean "
            "inner coverage is above zero, with `frozen_source` first in the tie-break. A rule "
            "that holds the bound by accepting nothing is not a deployment and cannot win here. "
            "Swept for the SELECTED arm under the SELECTED acquisition, because the shape of the "
            "budget is what decides whether a target cut can be placed at all."
        ),
        "swept_for": {"arm": arm, "acquisition": acquisition},
        "prompted_by": "feasibility_probe_1",
    }
    print(f"  primary cut {primary_cut} ({round(time.time() - started)}s)", flush=True)

    # 7 -- the coverage floor of criterion 6, from the frozen baseline's own inner deployment.
    coverages: list[float] = []
    for setup in banded:
        empty = purchase(setup, np.zeros(0, dtype=int), 0)
        thresholds = thresholds_for(setup, frozen_arm(), empty, CUT_FROZEN)
        tau = float(thresholds[PRIMARY_KEY]["tau"])
        block = setup.evaluation
        coverages.append(float(np.mean(block.frozen_score >= tau)) if np.isfinite(tau) else 0.0)
    floor = float(COVERAGE_FLOOR_FRACTION * float(np.median(coverages)))
    record["7_coverage_floor"] = {
        "inner_baseline_coverages": coverages,
        "median": float(np.median(coverages)),
        "fraction": COVERAGE_FLOOR_FRACTION,
        "selected": floor,
        "rule": (
            f"{COVERAGE_FLOOR_FRACTION:g} of the median inner deployed coverage of the FROZEN "
            "baseline. A method accepting less than this is treated as degenerate for criterion "
            "6, so 'safe because nothing is deployed' cannot count as a pass."
        ),
    }
    print(f"  arm {arm} at n={operating}, coverage floor {floor:.4f}", flush=True)

    final = Design(
        band_multiple=band,
        lambda_target=lambda_target,
        lambda_prefix=lambda_prefix,
        ridge=ridge,
        target_weight=target_weight,
        acquisition=acquisition,
        arm=arm,
        primary_cut=primary_cut,
        operating_budget=int(operating),
        coverage_floor=floor,
    )
    _write(
        DESIGN_RECORD,
        {
            "pre_registration": PRE_REGISTRATION,
            "decisions": DECISIONS,
            "feasibility_probe": FEASIBILITY_PROBE,
            "arms": ARM_NOTES,
            "acquisitions": ACQUISITION_NOTES,
            "cut_rules": CUT_NOTES,
            "frozen_design": _design_dict(final),
            "source_side_simulation": record,
            "inner_folds": [
                {"pseudo_target": setup.held_out, **setup.identity} for setup in setups
            ],
            "features_added": 0,
            "runtime_seconds": round(time.time() - started, 1),
            "note": (
                "every field of `frozen_design` was chosen here, on source engines only, before "
                "any outer target row was read. `--adapt` loads this record and cannot run "
                "without it."
            ),
        },
        "design_record",
    )
    print(
        f"preregister: {arm} on {acquisition}, band {band:g}, weights "
        f"({lambda_target:g}, {lambda_prefix:g}, {ridge:g}), target weight {target_weight:g} "
        f"({round(time.time() - started)}s)"
    )
    return 0


# ------------------------------------------------------------------ reading the cell tables


def load_cells() -> pd.DataFrame:
    if not SCORES.is_file():
        raise PhaseError(f"{cc._relative(SCORES)} is missing; run --adapt first")
    return pd.read_parquet(SCORES)


def load_counts() -> pd.DataFrame:
    if not COUNTS.is_file():
        raise PhaseError(f"{cc._relative(COUNTS)} is missing; run --adapt first")
    return pd.read_parquet(COUNTS)


CELL_KEYS = ("held_out_engine", "arm", "acquisition", "budget")


def summarise(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """One column over the draws of one cell: the mean, its spread, and how many draws it had.

    The spread across draws and a document-clustered interval answer different questions and
    disagree about the size of the uncertainty at small budgets. Both are reported, and where
    they disagree the larger one is the honest one -- SGV6's rule, kept.
    """
    values = frame[column].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "draws": int(finite.size),
    }


def cell_curve(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    """One arm's curve over the budget ladder for one engine and one acquisition rule."""
    return {
        f"n_{int(size)}": summarise(block, column)
        for size, block in sorted(frame.groupby("budget"), key=lambda item: int(item[0]))
    }


def endpoint_column(cut: str, epsilon: float) -> str:
    return f"dep__{cut}__{epsilon_key(epsilon)}__risk_controlled"


def run_results() -> int:
    """The operational endpoints: what each arm deploys, under each cut rule, at each epsilon."""
    started = time.time()
    design = load_design()
    cells = load_cells()
    payload: dict[str, Any] = {"frozen_design": _design_dict(design), "per_engine": {}}
    for engine, block in sorted(cells.groupby("held_out_engine")):
        engine_rows: dict[str, Any] = {}
        for arm, arm_block in sorted(block.groupby("arm")):
            per_acquisition: dict[str, Any] = {}
            for rule, rule_block in sorted(arm_block.groupby("acquisition")):
                cut_rows: dict[str, Any] = {}
                for cut in (*CUT_RULES, CUT_CALIBRATED, CUT_ORACLE):
                    column = endpoint_column(cut, PRIMARY_EPSILON)
                    if column not in rule_block.columns:
                        continue
                    if rule_block[column].isna().all():
                        continue
                    stem = f"dep__{cut}__"
                    cut_rows[cut] = {
                        "risk_controlled_repair_recall": cell_curve(rule_block, column),
                        "realized_harm_rate": cell_curve(
                            rule_block, f"{stem}{PRIMARY_KEY}__realized_harm_rate"
                        ),
                        "coverage": cell_curve(rule_block, f"{stem}{PRIMARY_KEY}__coverage"),
                        "n_accepted": cell_curve(rule_block, f"{stem}{PRIMARY_KEY}__n_accepted"),
                        "repairs_captured": cell_curve(
                            rule_block, f"{stem}{PRIMARY_KEY}__repairs_captured"
                        ),
                        "holds_bound_share": cell_curve(
                            rule_block, f"{stem}{PRIMARY_KEY}__holds_bound"
                        ),
                        "secondary_epsilons": {
                            epsilon_key(epsilon): cell_curve(
                                rule_block, endpoint_column(cut, epsilon)
                            )
                            for epsilon in EPSILONS
                            if epsilon != PRIMARY_EPSILON
                        },
                    }
                per_acquisition[rule] = {
                    "cuts": cut_rows,
                    "ranking": {
                        "auroc_safe": cell_curve(rule_block, "rank__auroc_safe"),
                        "pair_accuracy": cell_curve(rule_block, "rank__pair_accuracy"),
                        "average_precision_beneficial": cell_curve(
                            rule_block, "rank__average_precision_beneficial"
                        ),
                    },
                    "achievable_frontier": cell_curve(rule_block, f"frontier__{PRIMARY_KEY}"),
                    "identical_to_frozen_share": cell_curve(rule_block, "identical_to_frozen"),
                }
            engine_rows[arm] = per_acquisition
        payload["per_engine"][engine] = engine_rows
    payload["note"] = (
        "every number is a mean over the draws of one cell, with the spread across draws beside "
        "it. Global ranking metrics are reported for completeness and are NOT the primary "
        "endpoint: SGV12 measured a 0.0126 mean AUROC difference sitting on top of a 0.3065 "
        "frontier difference, so a ranking metric read alone would be actively misleading here."
    )
    payload["runtime_seconds"] = round(time.time() - started, 1)
    _write(ADAPTATION_RESULTS, payload, "adaptation_results")
    print(f"results: {len(cells)} cells -> {cc._relative(ADAPTATION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the gap ladder


def published_ladder() -> dict[str, dict[str, float]]:
    """SGV12's per-engine ladder, read from its artifact and never typed into this file.

    The ceiling is `frozen_frontier + score_transfer_gap` because that is how SGV12 recorded it:
    the gap IS the ceiling minus the frozen frontier. Reconstructing it this way rather than
    reading a `ceiling` field keeps SGV13's denominator identical to SGV12's by construction, and
    `--gap` asserts the four engines' means reproduce SGV12's published aggregate.
    """
    record = cc_read_json(s12.GAP_DECOMPOSITION)
    out: dict[str, dict[str, float]] = {}
    for engine, block in record["engines"].items():
        row = block[PRIMARY_KEY]["arms"][s12.SGV5_FROZEN]
        baseline = float(row["frozen_frontier"])
        gap = float(row["score_transfer_gap"])
        out[str(engine)] = {
            "baseline_frontier": baseline,
            "actionable_gap": gap,
            "ceiling_frontier": baseline + gap,
            "baseline_deployed": float(row["deployed"]),
            "decision_transfer_gap": float(row["decision_transfer_gap"]),
        }
    return out


def recovery(value: float, baseline: float, gap: float) -> float:
    """Section 14's recovery fraction, with the one case it is not defined for left undefined.

    Tesseract's actionable gap is exactly 0.0 in SGV12's published ladder -- the frozen ranking
    already reaches the ceiling frontier there -- so the fraction is 0/0. SGV12 recorded `nan`
    for it and so does this stage. Filling it with a zero or a one would put a number that means
    "no evidence" into an average that reads it as evidence.
    """
    if not np.isfinite(gap) or gap == 0.0:
        return float("nan")
    return float((value - baseline) / gap)


def _milestones(curve: dict[int, float]) -> dict[str, Any]:
    """The smallest budget on the ladder reaching each recovery milestone, or none."""
    ladder = sorted(curve)
    out: dict[str, Any] = {}
    for milestone in RECOVERY_MILESTONES:
        reached = next((size for size in ladder if curve[size] >= milestone), None)
        out[f"labels_to_{int(milestone * 100)}pc"] = reached
    return out


def run_gap() -> int:
    """Recompute SGV12's ladder, place every arm on it, and price each milestone in labels."""
    started = time.time()
    design = load_design()
    cells = load_cells()
    published = published_ladder()
    frozen_rows = cells[(cells["arm"] == A0_FROZEN) & (cells["budget"] == 0)]
    ceiling_rows = cells[cells["arm"] == ORACLE_RERANK]
    reproduction: dict[str, Any] = {}
    for engine in sorted(published):
        rebuilt_baseline = float(
            frozen_rows[frozen_rows["held_out_engine"] == engine][f"frontier__{PRIMARY_KEY}"].iloc[
                0
            ]
        )
        rebuilt_ceiling = float(
            ceiling_rows[ceiling_rows["held_out_engine"] == engine][
                f"frontier__{PRIMARY_KEY}"
            ].iloc[0]
        )
        rebuilt_deployed = float(
            frozen_rows[frozen_rows["held_out_engine"] == engine][
                endpoint_column(CUT_FROZEN, PRIMARY_EPSILON)
            ].mean()
        )
        reproduction[engine] = {
            "sgv12_baseline_deployed": published[engine]["baseline_deployed"],
            "sgv13_baseline_deployed": rebuilt_deployed,
            "deployed_difference": abs(rebuilt_deployed - published[engine]["baseline_deployed"]),
            "sgv12_baseline_frontier": published[engine]["baseline_frontier"],
            "sgv13_baseline_frontier": rebuilt_baseline,
            "baseline_difference": abs(rebuilt_baseline - published[engine]["baseline_frontier"]),
            "sgv12_ceiling_frontier": published[engine]["ceiling_frontier"],
            "sgv13_ceiling_frontier": rebuilt_ceiling,
            "ceiling_difference": abs(rebuilt_ceiling - published[engine]["ceiling_frontier"]),
        }
    worst = max(
        max(row["baseline_difference"], row["ceiling_difference"]) for row in reproduction.values()
    )
    if worst > 1e-9:
        raise PhaseError(
            f"this stage's rebuilt gap ladder differs from SGV12's published one by {worst:g}; "
            "every recovery fraction here has SGV12's denominator, every criterion compares "
            "against SGV12's published baseline deployment, and a drift in either invalidates "
            "all of them"
        )

    per_engine: dict[str, Any] = {}
    for engine, block in sorted(cells.groupby("held_out_engine")):
        reference = published[engine]
        arms: dict[str, Any] = {}
        for arm, arm_block in sorted(block.groupby("arm")):
            per_acquisition: dict[str, Any] = {}
            for rule, rule_block in sorted(arm_block.groupby("acquisition")):
                achievable = {
                    int(size): float(piece[f"frontier__{PRIMARY_KEY}"].to_numpy(dtype=float).mean())
                    for size, piece in rule_block.groupby("budget")
                }
                # A local name: the oracle arms are deployed under their own cut rule and the
                # rebinding must not leak into the next arm's loop iteration.
                endpoint = endpoint_column(design.primary_cut, PRIMARY_EPSILON)
                if endpoint not in rule_block.columns or rule_block[endpoint].isna().all():
                    endpoint = endpoint_column(CUT_ORACLE, PRIMARY_EPSILON)
                deployed_curve = {
                    int(size): float(piece[endpoint].to_numpy(dtype=float).mean())
                    for size, piece in rule_block.groupby("budget")
                    if endpoint in piece.columns and not piece[endpoint].isna().all()
                }
                achievable_recovery = {
                    size: recovery(
                        value, reference["baseline_frontier"], reference["actionable_gap"]
                    )
                    for size, value in achievable.items()
                }
                deployed_recovery = {
                    size: recovery(
                        value, reference["baseline_deployed"], reference["actionable_gap"]
                    )
                    for size, value in deployed_curve.items()
                }
                documents = {
                    int(size): float(piece["budget_documents"].to_numpy(dtype=float).mean())
                    for size, piece in rule_block.groupby("budget")
                }
                per_acquisition[rule] = {
                    "achievable_frontier": {f"n_{k}": v for k, v in sorted(achievable.items())},
                    "achievable_recovery": {
                        f"n_{k}": v for k, v in sorted(achievable_recovery.items())
                    },
                    "achievable_milestones": _milestones(achievable_recovery),
                    "deployed_endpoint": {f"n_{k}": v for k, v in sorted(deployed_curve.items())},
                    "deployed_recovery": {
                        f"n_{k}": v for k, v in sorted(deployed_recovery.items())
                    },
                    "deployed_milestones": _milestones(deployed_recovery),
                    "labels_per_unit_recall": {
                        f"n_{k}": (
                            float((v - deployed_curve.get(0, 0.0)) / k) if k else float("nan")
                        )
                        for k, v in sorted(deployed_curve.items())
                    },
                    "labels_to_half_oracle_headroom": _milestones(deployed_recovery).get(
                        "labels_to_50pc"
                    ),
                    "budget_documents": {f"n_{k}": v for k, v in sorted(documents.items())},
                    "annotation_cost": {
                        f"c_document_{cost:g}": {
                            f"n_{k}": float(k + cost * documents.get(k, 0.0))
                            for k in sorted(deployed_curve)
                        }
                        for cost in DOCUMENT_COST_GRID
                    },
                }
            arms[arm] = per_acquisition
        per_engine[engine] = {"reference": reference, "arms": arms}

    _write(
        GAP_RECOVERY,
        {
            "frozen_design": _design_dict(design),
            "sgv12_reproduction": reproduction,
            "max_abs_difference_from_sgv12": worst,
            "aggregate": {
                "mean_baseline_frontier": float(
                    np.mean([r["baseline_frontier"] for r in published.values()])
                ),
                "mean_ceiling_frontier": float(
                    np.mean([r["ceiling_frontier"] for r in published.values()])
                ),
                "mean_actionable_gap": float(
                    np.mean([r["actionable_gap"] for r in published.values()])
                ),
            },
            "per_engine": per_engine,
            "recovery_definition": (
                "(Frontier_SGV13 - Frontier_baseline) / (Frontier_ceiling - Frontier_baseline), "
                "with the numerator's frontier read on the same block and by the same function "
                "SGV11 and SGV12 used. The achievable ladder places the arm's RANKING on SGV12's "
                "rung; the deployed ladder places what the arm actually deploys against the "
                "baseline's deployment. Both are reported because they answer different "
                "questions and SGV12 showed they can disagree by a factor of twenty-four."
            ),
            "label_efficiency_definitions": {
                "labels_to_25pc / 50pc / 75pc": (
                    "the smallest budget on the ladder whose recovery fraction reaches the "
                    "milestone. `None` means the ladder never reached it."
                ),
                "labels_to_half_oracle_headroom": (
                    "SGV6's continuity metric. Half the headroom between the baseline and the "
                    "ceiling IS the 50% recovery milestone under this stage's definition, so the "
                    "two are the same number and are reported under both names rather than "
                    "computed twice from different denominators."
                ),
                "labels_per_unit_recall": (
                    "descriptive only. At N = 5 it divides by five and is not an estimate of "
                    "anything; section 15 says so and this stage does not overinterpret it."
                ),
                "annotation_cost": (
                    "section 9's secondary model, `N_labels + c_document * N_documents`, over a "
                    "grid of per-document coefficients because this repository's protocol fixes "
                    "no annotation cost. Descriptive; no criterion reads it."
                ),
            },
            "undefined_engines": [
                engine
                for engine, row in published.items()
                if not np.isfinite(row["actionable_gap"]) or row["actionable_gap"] == 0.0
            ],
            "runtime_seconds": round(time.time() - started, 1),
        },
        "gap_recovery",
    )
    print(f"gap: ladder reproduced to {worst:g} -> {cc._relative(GAP_RECOVERY)}")
    return 0


# ------------------------------------------------------------------ prefix purity


def run_prefix() -> int:
    """The mechanism SGV12 named: what the accepted set CONTAINS, at matched coverage and as
    deployed.

    Two views of the same question. The matched-coverage view fixes the accepted count to the
    frozen baseline's own deployed count and asks what changed inside it, which is the statistic
    SGV12 used to show that the actionable gap is a purity difference of about a tenth of the
    accepted set rather than a concordance difference. The deployed view lets the count move,
    which is what a real deployment does, and is reported for every cut rule so that a purity
    gain bought by accepting fewer rows is visible as one.
    """
    started = time.time()
    design = load_design()
    cells = load_cells()
    per_engine: dict[str, Any] = {}
    for engine, block in sorted(cells.groupby("held_out_engine")):
        arms: dict[str, Any] = {}
        for arm, arm_block in sorted(block.groupby("arm")):
            per_acquisition: dict[str, Any] = {}
            for rule, rule_block in sorted(arm_block.groupby("acquisition")):
                matched = {
                    name: cell_curve(rule_block, f"matched__{name}")
                    for name in (
                        "prefix_harm_rate",
                        "prefix_repair_recall",
                        "overlap_with_baseline",
                        "overlap_with_ceiling",
                        "harmful_dropped_vs_baseline",
                        "beneficial_gained_vs_baseline",
                        "swaps_recovered",
                        "n_accepted",
                    )
                }
                deployed_views: dict[str, Any] = {}
                for cut in (*CUT_RULES, CUT_CALIBRATED, CUT_ORACLE):
                    column = f"deployed_prefix__{cut}__prefix_harm_rate"
                    if column not in rule_block.columns or rule_block[column].isna().all():
                        continue
                    deployed_views[cut] = {
                        name: cell_curve(rule_block, f"deployed_prefix__{cut}__{name}")
                        for name in (
                            "prefix_harm_rate",
                            "prefix_repair_recall",
                            "coverage",
                            "overlap_with_baseline",
                            "overlap_with_ceiling",
                            "swaps_recovered",
                            "harmful_dropped_vs_baseline",
                            "beneficial_gained_vs_baseline",
                        )
                    }
                per_acquisition[rule] = {"matched_coverage": matched, "deployed": deployed_views}
            arms[arm] = per_acquisition
        per_engine[engine] = arms
    _write(
        PREFIX_PURITY,
        {
            "frozen_design": _design_dict(design),
            "matched_coverage_rule": (
                "the frozen baseline's OWN deployed coverage at the primary epsilon on that "
                "fold, so a purity comparison is a comparison of composition and never of depth"
            ),
            "per_engine": per_engine,
            "swap_definition": (
                "a harmful candidate the baseline accepted and this arm rejects, paired with a "
                "beneficial candidate the baseline rejected and this arm accepts. Reported as "
                "two counts and their minimum, because an arm that only drops rows is not "
                "swapping anything and should not be credited as if it were."
            ),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "prefix_purity",
    )
    print(f"prefix: {len(cells)} cells -> {cc._relative(PREFIX_PURITY)}")
    return 0


# ------------------------------------------------------------------ mechanism decomposition


def run_mechanism() -> int:
    """Which of score scale, ranking, representation and cut is actually doing the work.

    Section 17 forbids assigning a mechanism from an arm's name, so every number here is a
    controlled counterfactual computed on the same block from the same score vectors:

        M1 score scale     A1's gain. Ranking provably unchanged; only the level and the
                           calibrated cut move.
        M2 ranking         this arm's ranking at the BASELINE's accepted depth, against the
                           baseline. Depth held fixed, so only the order can move it.
        M3 representation  the part of the arm's gain that A3 -- a target-specific head on the
                           frozen features -- does not reproduce. A residual, and labelled one.
        M4 decision cut    the FROZEN ranking at this arm's accepted depth, against the
                           baseline. Order held fixed, so only the depth can move it.

    M2 and M4 are transplants of one quantity into the other's setting and do not have to sum to
    the total: the two effects interact, and the interaction is reported rather than absorbed.
    """
    started = time.time()
    design = load_design()
    cells = load_cells()
    cut = design.primary_cut
    per_engine: dict[str, Any] = {}
    for engine, block in sorted(cells.groupby("held_out_engine")):
        baseline_rows = block[(block["arm"] == A0_FROZEN) & (block["budget"] == 0)]
        baseline = float(baseline_rows[endpoint_column(cut, PRIMARY_EPSILON)].iloc[0])
        head_curve = {
            int(size): float(piece[endpoint_column(cut, PRIMARY_EPSILON)].mean())
            for size, piece in block[
                (block["arm"] == A3_HEAD) & (block["acquisition"] == primary_rule(design))
            ].groupby("budget")
        }
        calibration_curve = {
            int(size): float(piece[endpoint_column(CUT_CALIBRATED, PRIMARY_EPSILON)].mean())
            for size, piece in block[
                (block["arm"] == A1_CALIBRATION) & (block["acquisition"] == primary_rule(design))
            ].groupby("budget")
            if endpoint_column(CUT_CALIBRATED, PRIMARY_EPSILON) in piece.columns
        }
        arms: dict[str, Any] = {}
        for arm, arm_block in sorted(block.groupby("arm")):
            rule_block = arm_block[arm_block["acquisition"] == primary_rule(design)]
            if rule_block.empty:
                rule_block = arm_block[arm_block["acquisition"] == Q_RANDOM]
            if rule_block.empty:
                continue
            rows: dict[str, Any] = {}
            for size, piece in sorted(rule_block.groupby("budget"), key=lambda x: int(x[0])):
                total = float(piece[endpoint_column(cut, PRIMARY_EPSILON)].mean()) - baseline
                ranking_only = (
                    float(piece[f"cf__new_rank_frozen_depth__risk_controlled__{cut}"].mean())
                    - baseline
                )
                depth_only = (
                    float(piece[f"cf__frozen_rank_new_depth__risk_controlled__{cut}"].mean())
                    - baseline
                )
                head = head_curve.get(int(size), baseline) - baseline
                scale = calibration_curve.get(int(size), baseline) - baseline
                rows[f"n_{int(size)}"] = {
                    "total_effect": total,
                    "m1_score_scale": scale,
                    "m2_ranking_at_fixed_depth": ranking_only,
                    "m3_representation_residual": total - head,
                    "m4_depth_at_fixed_ranking": depth_only,
                    "interaction": total - ranking_only - depth_only,
                    "accepted_rows": float(piece[f"cf__depth__{cut}"].mean()),
                    "baseline_accepted_rows": float(piece[f"cf__baseline_depth__{cut}"].mean()),
                }
            arms[arm] = rows
        per_engine[engine] = {"baseline_endpoint": baseline, "arms": arms}
    _write(
        MECHANISM,
        {
            "frozen_design": _design_dict(design),
            "cut_rule": cut,
            "definitions": {
                "m1_score_scale": (
                    "A1's own effect under its calibrated absolute cut. A1's ranking is provably "
                    "the frozen one, so anything it moves is level and nothing else."
                ),
                "m2_ranking_at_fixed_depth": (
                    "this arm's ranking accepting exactly as many rows as the baseline did"
                ),
                "m3_representation_residual": (
                    "the total effect minus A3's, where A3 is a target head on the FROZEN "
                    "features. A residual: it is what a frozen-feature remap did not reproduce, "
                    "not a direct measurement of a representation change."
                ),
                "m4_depth_at_fixed_ranking": (
                    "the frozen ranking accepting exactly as many rows as this arm did"
                ),
                "interaction": (
                    "total minus M2 minus M4. The two transplants do not have to sum to the "
                    "total and this is what is left over; it is reported rather than absorbed."
                ),
            },
            "per_engine": per_engine,
            "runtime_seconds": round(time.time() - started, 1),
        },
        "mechanism_decomposition",
    )
    print(f"mechanism: 4 mechanisms, {len(per_engine)} engines -> {cc._relative(MECHANISM)}")
    return 0


# ------------------------------------------------------------------ acquisition and SGV6


def _curve_for(
    cells: pd.DataFrame, engine: str, arm: str, rule: str, column: str
) -> dict[int, float]:
    block = cells[
        (cells["held_out_engine"] == engine)
        & (cells["arm"] == arm)
        & (cells["acquisition"] == rule)
    ]
    if block.empty or column not in block.columns:
        return {}
    return {
        int(size): float(piece[column].to_numpy(dtype=float).mean())
        for size, piece in block.groupby("budget")
        if not piece[column].isna().all()
    }


def run_acquisition() -> int:
    """Every acquisition rule against every other, and against the two random controls.

    Section 22 asks the proposed rule to beat matched random AND document-matched random before
    anything is claimed about targeting, and it asks for the failure to be reported if it does
    not. Both controls are here, the comparison is at matched budget on the same arm, and the
    verdict is written down whichever way it falls.
    """
    started = time.time()
    design = load_design()
    cells = load_cells()
    column = endpoint_column(design.primary_cut, PRIMARY_EPSILON)
    per_engine: dict[str, Any] = {}
    for engine in sorted(cells["held_out_engine"].unique()):
        arms: dict[str, Any] = {}
        for arm in (A5_PREFIX, A3_HEAD, A1_CALIBRATION, A2_CUT):
            rules: dict[str, Any] = {}
            for rule in ACQUISITIONS:
                curve = _curve_for(cells, engine, arm, rule, column)
                purity = _curve_for(
                    cells,
                    engine,
                    arm,
                    rule,
                    f"deployed_prefix__{design.primary_cut}__prefix_harm_rate",
                )
                composition = _curve_for(cells, engine, arm, rule, "budget_beneficial")
                documents = _curve_for(cells, engine, arm, rule, "budget_documents")
                in_prefix = _curve_for(cells, engine, arm, rule, "budget_in_prefix")
                swap_pairs = _curve_for(cells, engine, arm, rule, "pairs__prefix_swap")
                target_pairs = _curve_for(cells, engine, arm, rule, "pairs__target")
                if not curve:
                    continue
                rules[rule] = {
                    "risk_controlled_repair_recall": {
                        f"n_{k}": v for k, v in sorted(curve.items())
                    },
                    "prefix_harm_rate": {f"n_{k}": v for k, v in sorted(purity.items())},
                    "budget_beneficial": {f"n_{k}": v for k, v in sorted(composition.items())},
                    "budget_documents": {f"n_{k}": v for k, v in sorted(documents.items())},
                    "budget_in_prefix": {f"n_{k}": v for k, v in sorted(in_prefix.items())},
                    "prefix_swap_pairs": {f"n_{k}": v for k, v in sorted(swap_pairs.items())},
                    "target_pairs": {f"n_{k}": v for k, v in sorted(target_pairs.items())},
                    "mean_over_budgets": float(
                        np.mean([v for k, v in curve.items() if k > 0]) if len(curve) > 1 else 0.0
                    ),
                }
            arms[arm] = rules
        per_engine[engine] = arms

    proposed = (Q_PREFIX_SWAP, Q_HYBRID)
    verdict: dict[str, Any] = {}
    for rule in proposed:
        wins: dict[str, dict[str, Any]] = {}
        for control in RANDOM_CONTROLS:
            better = 0
            deltas: dict[str, float] = {}
            for engine in sorted(cells["held_out_engine"].unique()):
                left = per_engine[engine][A5_PREFIX].get(rule, {}).get("mean_over_budgets")
                right = per_engine[engine][A5_PREFIX].get(control, {}).get("mean_over_budgets")
                if left is None or right is None:
                    continue
                deltas[engine] = float(left - right)
                better += int(left > right)
            wins[control] = {
                "engines_favourable": better,
                "per_engine_delta": deltas,
                "meets_engine_bar": bool(better >= ENGINE_BAR),
            }
        verdict[rule] = wins
    _write(
        ACQUISITION_RESULTS,
        {
            "frozen_design": _design_dict(design),
            "acquisitions": ACQUISITION_NOTES,
            "per_engine": per_engine,
            "proposed_versus_random": verdict,
            "reading": (
                "an acquisition rule earns a claim only by beating BOTH random controls on at "
                f"least {ENGINE_BAR} of {N_ENGINES} engines. If adaptation works and acquisition "
                "does not, the two findings are reported separately and neither is used to "
                "support the other."
            ),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "acquisition_results",
    )
    print(f"acquisition: {len(ACQUISITIONS)} rules -> {cc._relative(ACQUISITION_RESULTS)}")
    return 0


def run_sgv6() -> int:
    """The comparison of record: SGV13's selected arm against SGV6's own procedure.

    Matched on everything a comparison can be matched on -- the same adaptation pool, the same
    acquisition ordering, the same purchased rows, the same evaluation documents, the same harm
    constraint and the same cut rule -- because `b1_sgv6_generic` is SGV6's `fit_arm` and
    `select_arm` called on an `s6.Budget` built from this stage's order. What differs is what the
    labels were spent on, which is the only thing section 19 is asking about.
    """
    started = time.time()
    design = load_design()
    cells = load_cells()
    published = published_ladder()
    column = endpoint_column(design.primary_cut, PRIMARY_EPSILON)
    per_engine: dict[str, Any] = {}
    for engine in sorted(cells["held_out_engine"].unique()):
        reference = published[engine]
        rows: dict[str, Any] = {}
        for rule in tuple(dict.fromkeys((Q_RANDOM, primary_rule(design)))):
            proposed = _curve_for(cells, engine, design.arm, rule, column)
            generic = _curve_for(cells, engine, SGV6_GENERIC, rule, column)
            shared = sorted(set(proposed) & set(generic))
            proposed_recovery = {
                size: recovery(
                    proposed[size], reference["baseline_deployed"], reference["actionable_gap"]
                )
                for size in shared
            }
            generic_recovery = {
                size: recovery(
                    generic[size], reference["baseline_deployed"], reference["actionable_gap"]
                )
                for size in shared
            }
            rows[rule] = {
                "sgv13_endpoint": {f"n_{k}": proposed[k] for k in shared},
                "sgv6_endpoint": {f"n_{k}": generic[k] for k in shared},
                "paired_delta": {f"n_{k}": proposed[k] - generic[k] for k in shared},
                "sgv13_recovery": {f"n_{k}": proposed_recovery[k] for k in shared},
                "sgv6_recovery": {f"n_{k}": generic_recovery[k] for k in shared},
                "sgv13_milestones": _milestones(proposed_recovery),
                "sgv6_milestones": _milestones(generic_recovery),
                "sgv6_selected_families": sorted(
                    {
                        str(name)
                        for name in cells[
                            (cells["held_out_engine"] == engine)
                            & (cells["arm"] == SGV6_GENERIC)
                            & (cells["acquisition"] == rule)
                        ]["selected_family"].tolist()
                        if str(name)
                    }
                ),
            }
        per_engine[engine] = rows
    _write(
        SGV6_COMPARISON,
        {
            "frozen_design": _design_dict(design),
            "matched_on": [
                "the adaptation pool (the held-out engine's TRAIN documents)",
                "the acquisition ordering and therefore the purchased rows",
                "the label budget",
                "the evaluation documents",
                "the harm constraint and the epsilon grid",
                "the cut rule",
            ],
            "per_engine": per_engine,
            "criterion_4_rule": (
                f"the selected method must reach {MILESTONE_FOR_CRITERION_4:.0%} recovery with "
                f"at least {LABEL_REDUCTION_BAR:.0%} fewer labels than `b1_sgv6_generic`. A "
                "milestone neither reaches is not a pass for either."
            ),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "sgv6_comparison",
    )
    print(f"sgv6: {len(per_engine)} engines -> {cc._relative(SGV6_COMPARISON)}")
    return 0


# ------------------------------------------------------------------ oracles and EasyOCR


def run_oracle() -> int:
    """The three diagnostic upper bounds. None of them may enter a selection or a criterion.

    O1 and O2 are the rungs SGV11 and SGV12 published, recomputed on this block. O3 is the one
    that separates the two failures that look identical from the outside: an omniscient
    acquisition, spending the same budget on the rows a labelled oracle would have chosen. If O3
    is far above every real acquisition rule, the limit is which labels are bought; if it is not,
    the limit is what the adaptation can do with them.
    """
    started = time.time()
    design = load_design()
    cells = load_cells()
    column = endpoint_column(design.primary_cut, PRIMARY_EPSILON)
    published = published_ladder()
    per_engine: dict[str, Any] = {}
    for engine in sorted(cells["held_out_engine"].unique()):
        block = cells[cells["held_out_engine"] == engine]
        oracle_arms = sorted(
            {str(name) for name in block["arm"].tolist() if str(name).startswith("o3_")}
        )
        real = _curve_for(cells, engine, design.arm, primary_rule(design), column)
        oracle_curves = {
            arm: {
                int(size): float(piece[column].to_numpy(dtype=float).mean())
                for size, piece in block[block["arm"] == arm].groupby("budget")
            }
            for arm in oracle_arms
        }
        per_engine[engine] = {
            "o1_oracle_cut": {
                "frontier": float(
                    block[block["arm"] == ORACLE_CUT][f"frontier__{PRIMARY_KEY}"].iloc[0]
                ),
                "prefix_harm_rate": float(
                    block[block["arm"] == ORACLE_CUT][
                        f"deployed_prefix__{CUT_ORACLE}__prefix_harm_rate"
                    ].iloc[0]
                ),
            },
            "o2_oracle_rerank": {
                "frontier": float(
                    block[block["arm"] == ORACLE_RERANK][f"frontier__{PRIMARY_KEY}"].iloc[0]
                ),
                "prefix_harm_rate": float(
                    block[block["arm"] == ORACLE_RERANK][
                        f"deployed_prefix__{CUT_ORACLE}__prefix_harm_rate"
                    ].iloc[0]
                ),
                "overlap_with_baseline": float(
                    block[block["arm"] == ORACLE_RERANK][
                        f"deployed_prefix__{CUT_ORACLE}__overlap_with_baseline"
                    ].iloc[0]
                ),
            },
            "o3_oracle_acquisition": {
                arm: {f"n_{k}": v for k, v in sorted(curve.items())}
                for arm, curve in oracle_curves.items()
            },
            "selected_method_under_its_own_acquisition": {
                f"n_{k}": v for k, v in sorted(real.items())
            },
            "oracle_acquisition_headroom": {
                f"n_{size}": float(
                    max(
                        (curve.get(size, float("nan")) for curve in oracle_curves.values()),
                        default=float("nan"),
                    )
                    - real.get(size, float("nan"))
                )
                for size in sorted(real)
                if size > 0
            },
            "reference": published[engine],
        }
    _write(
        ORACLE_ANALYSIS,
        {
            "frozen_design": _design_dict(design),
            "diagnostic_only": True,
            "per_engine": per_engine,
            "reading": (
                "a large O3 headroom means the budget contains the information and the "
                "acquisition rule is not finding it; a small one means the adaptation cannot "
                "use the labels however they are chosen. Neither number may be used to select "
                "a method, and `--negative` asserts that no oracle arm appears in the design "
                "record or in any criterion."
            ),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "oracle_analysis",
    )
    print(f"oracle: 3 bounds, {len(per_engine)} engines -> {cc._relative(ORACLE_ANALYSIS)}")
    return 0


def run_easyocr() -> int:
    """The pre-specified EasyOCR analysis. Five questions, fixed before any outer row was read.

    SGV12 left EasyOCR as the one engine whose ceiling accepted the IDENTICAL candidate set as
    the baseline -- prefix overlap 1.0000 -- while its median evaluation-minus-calibration score
    shift was +0.2074 against -1.1485 on docTR. Re-ranking the frozen representation could not
    reach its failure, and a value-transported threshold accepted far too much. This asks whether
    target labels reach it, and which kind.
    """
    started = time.time()
    design = load_design()
    cells = load_cells()
    identity = cc_read_json(FROZEN_IDENTITY)["per_engine"]
    column = endpoint_column(design.primary_cut, PRIMARY_EPSILON)
    answers: dict[str, Any] = {}
    for engine in sorted(cells["held_out_engine"].unique()):
        rule = primary_rule(design)
        calibration_column = endpoint_column(CUT_CALIBRATED, PRIMARY_EPSILON)
        budget_column = endpoint_column(CUT_BUDGET_EMPIRICAL, PRIMARY_EPSILON)
        baseline = _curve_for(cells, engine, A0_FROZEN, rule, column).get(0, float("nan"))
        usable: int | None = None
        for size in NONZERO_BUDGETS:
            block = cells[
                (cells["held_out_engine"] == engine)
                & (cells["arm"] == design.arm)
                & (cells["acquisition"] == rule)
                & (cells["budget"] == size)
            ]
            if block.empty:
                continue
            recall = float(block[column].mean())
            coverage = float(block[f"dep__{design.primary_cut}__{PRIMARY_KEY}__coverage"].mean())
            holds = float(block[f"dep__{design.primary_cut}__{PRIMARY_KEY}__holds_bound"].mean())
            if recall > 0.0 and coverage >= design.coverage_floor and holds >= 1.0:
                usable = int(size)
                break
        answers[engine] = {
            "1_does_target_calibration_recover": {
                f"n_{k}": v
                for k, v in sorted(
                    _curve_for(cells, engine, A1_CALIBRATION, rule, calibration_column).items()
                )
            },
            "2_does_target_cut_selection_recover": {
                f"n_{k}": v
                for k, v in sorted(_curve_for(cells, engine, A2_CUT, rule, budget_column).items())
            },
            "3_does_target_reranking_change_anything": {
                "endpoint": {
                    f"n_{k}": v
                    for k, v in sorted(_curve_for(cells, engine, A5_PREFIX, rule, column).items())
                },
                "prefix_overlap_with_baseline": {
                    f"n_{k}": v
                    for k, v in sorted(
                        _curve_for(
                            cells,
                            engine,
                            A5_PREFIX,
                            rule,
                            f"deployed_prefix__{design.primary_cut}__overlap_with_baseline",
                        ).items()
                    )
                },
            },
            "4_does_representation_refit_help_beyond_scale": {
                "joint": {
                    f"n_{k}": v
                    for k, v in sorted(
                        _curve_for(cells, engine, A4_JOINT, Q_RANDOM, column).items()
                    )
                },
                "capacity": {
                    f"n_{k}": v
                    for k, v in sorted(
                        _curve_for(cells, engine, A6_CAPACITY, Q_RANDOM, column).items()
                    )
                },
            },
            "5_smallest_operationally_usable_budget": usable,
            "baseline_endpoint": baseline,
            "distribution": {
                key: identity[engine][key]
                for key in (
                    "median_score_shift_evaluation_minus_calibration",
                    "median_score_shift_pool_minus_calibration",
                    "evaluation_harmful_rate",
                    "evaluation_beneficial_rate",
                    "evaluation_beneficial_rows",
                    "sites_offering_both_outcomes",
                    "pool_beneficial_rate",
                    "accepted_share",
                )
            },
        }
    _write(
        EASYOCR_ANALYSIS,
        {
            "frozen_design": _design_dict(design),
            "pre_specified": True,
            "per_engine": answers,
            "note": (
                "every engine is analysed under the identical five questions so that EasyOCR's "
                "answers are read against three comparisons rather than on their own. No arm was "
                "special-cased for any engine at training time and `--negative` asserts it."
            ),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "easyocr_analysis",
    )
    print(f"easyocr: 5 questions x {len(answers)} engines -> {cc._relative(EASYOCR_ANALYSIS)}")
    return 0


# ------------------------------------------------------------------ inference

RECALL = "risk_controlled_repair_recall"
HARM = "prefix_harm_rate"


@dataclass(frozen=True, slots=True)
class CountBlock:
    """One cell's per-document counts, one row per draw, in the sorted document order."""

    documents: tuple[str, ...]
    accepted: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    total: np.ndarray


def count_block(
    counts: pd.DataFrame, engine: str, arm: str, rule: str, size: int, cut: str, epsilon: float
) -> CountBlock:
    """Assemble the per-document count matrices for one cell, one matrix row per draw."""
    block = counts[
        (counts["held_out_engine"] == engine)
        & (counts["arm"] == arm)
        & (counts["acquisition"] == rule)
        & (counts["budget"] == size)
        & (counts["cut_rule"] == cut)
        & (np.isclose(counts["epsilon"], epsilon))
    ]
    if block.empty:
        raise PhaseError(f"no counts for {engine}/{arm}/{rule}/n={size}/{cut}/eps={epsilon}")
    documents = tuple(sorted({str(name) for name in block["document_id"].tolist()}))
    draws = sorted({int(value) for value in block["draw"].tolist()})
    shape = (len(draws), len(documents))
    matrices = {
        name: np.zeros(shape, dtype=float)
        for name in ("accepted", "harmful_accepted", "beneficial_accepted", "beneficial_total")
    }
    position = {name: index for index, name in enumerate(documents)}
    row_of = {draw: index for index, draw in enumerate(draws)}
    rows = block["draw"].to_numpy(dtype=int)
    columns = np.array([position[str(name)] for name in block["document_id"].tolist()], dtype=int)
    index = np.array([row_of[int(value)] for value in rows.tolist()], dtype=int)
    for name in matrices:
        matrices[name][index, columns] = block[name].to_numpy(dtype=float)
    return CountBlock(
        documents=documents,
        accepted=matrices["accepted"],
        harmful=matrices["harmful_accepted"],
        beneficial=matrices["beneficial_accepted"],
        total=matrices["beneficial_total"],
    )


def statistic_draws(
    block: CountBlock, multiplicity: np.ndarray, name: str, eps: float
) -> np.ndarray:
    """One statistic over every resample and every draw, as a matrix product and nothing else.

    Both endpoints are ratios of sums over rows, so a document-clustered resample of either is a
    weighted sum of the per-document totals. That turns a 2,000-draw bootstrap of a cell into
    four matrix products rather than 2,000 re-derivations of an accept set. `tests/leakage`
    checks the result against a literal recount on drawn resamples rather than trusting it.
    """
    accepted = multiplicity @ block.accepted.T
    harmful = multiplicity @ block.harmful.T
    if name == HARM:
        return np.where(accepted > 0.0, harmful / np.maximum(accepted, 1e-12), 0.0)
    beneficial = multiplicity @ block.beneficial.T
    total = multiplicity @ block.total.T
    recall = np.where(total > 0.0, beneficial / np.maximum(total, 1e-12), 0.0)
    rate = np.where(accepted > 0.0, harmful / np.maximum(accepted, 1e-12), 0.0)
    return np.where((accepted > 0.0) & (rate <= eps), recall, 0.0)


def paired_delta(
    left: CountBlock, right: CountBlock, name: str, epsilon: float, seed: int
) -> dict[str, float]:
    """A document-clustered PAIRED bootstrap of one statistic's difference between two cells.

    The same resampled documents are given to both cells, which is what makes it paired: the two
    arms are read on the same pages in every resample, so the page-to-page variation that they
    share cancels instead of being counted twice. The unit is the document and never the row --
    candidates inside one receipt are strongly correlated and a row-level interval would be
    dishonestly narrow.
    """
    if left.documents != right.documents:
        raise PhaseError("a paired bootstrap needs the two cells to span the same documents")
    multiplicity = document_multiplicities(
        np.asarray(left.documents), seed, BOOTSTRAP_RESAMPLES
    ).astype(float)
    identity = np.ones((1, len(left.documents)), dtype=float)
    point_left = float(statistic_draws(left, identity, name, epsilon).mean())
    point_right = float(statistic_draws(right, identity, name, epsilon).mean())
    deltas = statistic_draws(left, multiplicity, name, epsilon).mean(axis=1) - statistic_draws(
        right, multiplicity, name, epsilon
    ).mean(axis=1)
    # SGV9's interval and its two-sided bootstrap p, imported rather than restated, so that the
    # percentile convention and the tie handling are the ones five earlier stages published.
    interval = _interval(deltas)
    return {
        "left": point_left,
        "right": point_right,
        "delta": point_left - point_right,
        "bootstrap_mean_delta": float(interval["estimate"]),
        "ci_lower": float(interval["ci_lower"]),
        "ci_upper": float(interval["ci_upper"]),
        "p_value": float(interval["p_value"]),
        "documents": len(left.documents),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def holm(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni over the pre-registered family, whose size was fixed before any outcome."""
    order = sorted(rows, key=lambda name: (rows[name]["p_value"], name))
    size = len(order)
    if size != PRIMARY_FAMILY_SIZE:
        raise PhaseError(
            f"the primary family was pre-registered at {PRIMARY_FAMILY_SIZE} tests and {size} "
            "were built; the correction must be applied to the family that was declared"
        )
    running = 0.0
    out: dict[str, dict[str, float]] = {}
    for position, name in enumerate(order):
        adjusted = min(1.0, (size - position) * rows[name]["p_value"])
        running = max(running, adjusted)
        out[name] = {**rows[name], "p_holm": running, "survives_holm_at_0_05": running < 0.05}
    return out


def run_stats() -> int:
    """The pre-registered family of twenty tests, document-clustered, paired, Holm-corrected.

    The family and its size were fixed before any outer target row was read, and the size is
    asserted here rather than counted from what happened to be computable: twelve comparisons
    against the SGV6 baseline at three epsilons on four engines, four against the frozen baseline
    on the primary endpoint, and four against the frozen baseline on the accepted-prefix harm
    rate. Every test is at the operating budget the source-side simulation chose, under the cut
    rule it chose, on the acquisition rule it chose.
    """
    started = time.time()
    design = load_design()
    counts = load_counts()
    engines = sorted({str(name) for name in counts["held_out_engine"].tolist()})
    size = design.operating_budget
    rule = primary_rule(design)
    cut = design.primary_cut
    rows: dict[str, dict[str, float]] = {}
    labels: dict[str, dict[str, Any]] = {}

    def add(name: str, meta: dict[str, Any], result: dict[str, float]) -> None:
        rows[name] = result
        labels[name] = meta

    for engine in engines:
        selected = count_block(counts, engine, design.arm, rule, size, cut, PRIMARY_EPSILON)
        # At budget 0 every cut rule falls back to `frozen_source`, so the baseline is read
        # under the primary rule's label -- which is the only one whose per-document
        # decomposition is persisted -- and carries the frozen cut's numbers.
        frozen = count_block(counts, engine, A0_FROZEN, rule, 0, cut, PRIMARY_EPSILON)
        for epsilon in EPSILONS:
            left = count_block(counts, engine, design.arm, rule, size, cut, epsilon)
            right = count_block(counts, engine, SGV6_GENERIC, rule, size, cut, epsilon)
            name = f"{RECALL}|sgv6|{engine}|{epsilon_key(epsilon)}"
            add(
                name,
                {
                    "comparison": f"{design.arm} vs {SGV6_GENERIC}",
                    "statistic": RECALL,
                    "engine": engine,
                    "epsilon": float(epsilon),
                },
                paired_delta(
                    left,
                    right,
                    RECALL,
                    epsilon,
                    _stable_seed("sgv13-boot", engine, "sgv6", epsilon_key(epsilon)),
                ),
            )
        add(
            f"{RECALL}|frozen|{engine}|{PRIMARY_KEY}",
            {
                "comparison": f"{design.arm} vs {A0_FROZEN}",
                "statistic": RECALL,
                "engine": engine,
                "epsilon": float(PRIMARY_EPSILON),
            },
            paired_delta(
                selected,
                frozen,
                RECALL,
                PRIMARY_EPSILON,
                _stable_seed("sgv13-boot", engine, "frozen", PRIMARY_KEY),
            ),
        )
        add(
            f"{HARM}|frozen|{engine}|{PRIMARY_KEY}",
            {
                "comparison": f"{design.arm} vs {A0_FROZEN}",
                "statistic": HARM,
                "engine": engine,
                "epsilon": float(PRIMARY_EPSILON),
                "direction": "a NEGATIVE delta is the favourable direction for a harm rate",
            },
            paired_delta(
                selected,
                frozen,
                HARM,
                PRIMARY_EPSILON,
                _stable_seed("sgv13-boot", engine, "harm", PRIMARY_KEY),
            ),
        )
    corrected = holm(rows)
    surviving = [name for name, row in corrected.items() if row["survives_holm_at_0_05"]]
    favourable = [
        name
        for name in surviving
        if (labels[name]["statistic"] == HARM) == (corrected[name]["delta"] < 0.0)
    ]
    _write(
        STATISTICAL_TESTS,
        {
            "frozen_design": _design_dict(design),
            "family_size": PRIMARY_FAMILY_SIZE,
            "family_definition": PRE_REGISTRATION["primary_family"],
            "operating_budget": size,
            "acquisition": rule,
            "cut_rule": cut,
            "tests": {name: {**labels[name], **corrected[name]} for name in corrected},
            "surviving_holm": surviving,
            "surviving_and_favourable": favourable,
            "surviving_and_unfavourable": sorted(set(surviving) - set(favourable)),
            "method": (
                "document-clustered PAIRED bootstrap with "
                f"{BOOTSTRAP_RESAMPLES} resamples, the same resampled documents given to both "
                "arms in every resample, averaged over the draws of each cell; Holm at 0.05 over "
                "the whole family. The resampling unit is the document and never the row."
            ),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "statistical_tests",
    )
    print(
        f"stats: {len(corrected)} tests, {len(surviving)} survive Holm "
        f"({len(favourable)} favourable) -> {cc._relative(STATISTICAL_TESTS)}"
    )
    return 0


# ------------------------------------------------------------------ falsification


def _spearman(values: list[float]) -> float:
    """Rank correlation of a curve with its own budget order. `s6._spearman`'s construction."""
    if len(values) < 3:
        return float("nan")
    return float(s6._spearman(list(range(len(values))), values))


def _shard_endpoints(
    counts: pd.DataFrame, engine: str, arm: str, rule: str, size: int, cut: str
) -> dict[str, float]:
    """The primary endpoint recomputed inside each document shard of the evaluation block.

    The shard is `sha256(document_id) mod 4`, SGV12's partition, imported rather than re-salted,
    so "a different set of documents from the same engine" means the same thing in both stages.
    """
    block = counts[
        (counts["held_out_engine"] == engine)
        & (counts["arm"] == arm)
        & (counts["acquisition"] == rule)
        & (counts["budget"] == size)
        & (counts["cut_rule"] == cut)
        & (np.isclose(counts["epsilon"], PRIMARY_EPSILON))
    ]
    out: dict[str, float] = {}
    if block.empty:
        return out
    shards = np.array(
        [s12.document_shard(str(name)) for name in block["document_id"].tolist()], dtype=int
    )
    for shard in sorted(set(shards.tolist())):
        piece = block[shards == shard]
        accepted = float(piece["accepted"].sum())
        harmful = float(piece["harmful_accepted"].sum())
        beneficial = float(piece["beneficial_accepted"].sum())
        total = float(piece["beneficial_total"].sum())
        rate = harmful / accepted if accepted else 0.0
        recall = beneficial / total if total else 0.0
        out[f"shard_{shard}"] = float(recall if accepted and rate <= PRIMARY_EPSILON else 0.0)
    return out


def _document_influence(
    counts: pd.DataFrame, engine: str, arm: str, rule: str, size: int, cut: str
) -> dict[str, float]:
    """How much of the endpoint one document carries, by leaving each one out in turn."""
    block = count_block(counts, engine, arm, rule, size, cut, PRIMARY_EPSILON)
    identity = np.ones((1, len(block.documents)), dtype=float)
    full = float(statistic_draws(block, identity, RECALL, PRIMARY_EPSILON).mean())
    shifts: dict[str, float] = {}
    for position, name in enumerate(block.documents):
        mask = np.ones((1, len(block.documents)), dtype=float)
        mask[0, position] = 0.0
        shifts[name] = full - float(statistic_draws(block, mask, RECALL, PRIMARY_EPSILON).mean())
    worst = max(shifts, key=lambda name: abs(shifts[name])) if shifts else ""
    return {
        "endpoint": full,
        "largest_single_document_shift": float(shifts.get(worst, 0.0)),
        "document": worst,
        "documents": float(len(block.documents)),
    }


def _sgv6_family_counts(cells: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Which adapter family SGV6's own `select_arm` chose, per engine and budget."""
    block = cells[cells["arm"] == SGV6_GENERIC]
    out: dict[str, dict[str, int]] = {}
    for engine, piece in sorted(block.groupby("held_out_engine")):
        counts: dict[str, int] = {}
        for name in piece["selected_family"].tolist():
            key = str(name) or "(none)"
            counts[key] = counts.get(key, 0) + 1
        out[str(engine)] = dict(sorted(counts.items()))
    return out


def run_negative() -> int:
    """The brief's fifteen falsification tests, plus the five leakage assertions.

    Every one of these is an attempt to make the stage's claim false. Where a control wins it is
    recorded as a control that won; nothing here is promoted to the method afterwards.
    """
    started = time.time()
    design = load_design()
    cells = load_cells()
    counts = load_counts()
    engines = sorted({str(name) for name in cells["held_out_engine"].tolist()})
    cut = design.primary_cut
    rule = primary_rule(design)
    size = design.operating_budget
    column = endpoint_column(cut, PRIMARY_EPSILON)
    tests: dict[str, Any] = {}

    def curve(arm: str, acquisition: str) -> dict[str, dict[int, float]]:
        return {engine: _curve_for(cells, engine, arm, acquisition, column) for engine in engines}

    selected = curve(design.arm, rule)
    frozen = curve(A0_FROZEN, rule)

    def compare(other: dict[str, dict[int, float]], at: int) -> dict[str, Any]:
        deltas = {
            engine: float(
                selected[engine].get(at, float("nan")) - other[engine].get(at, float("nan"))
            )
            for engine in engines
        }
        wins = sum(1 for value in deltas.values() if np.isfinite(value) and value > 0.0)
        return {
            "per_engine_delta": deltas,
            "engines_where_the_selected_method_is_higher": wins,
            "selected_method_wins": bool(wins >= ENGINE_BAR),
        }

    def rank_rules(arm: str, at: int) -> dict[str, Any]:
        """Every acquisition rule's endpoint for one arm, ordered, with the missing ones named.

        An arm that did not run under a rule has no cell there, and reading that as a loss for
        the rule would be reading absence as evidence. Rules the arm never ran under are listed
        separately rather than scored.
        """
        per_rule: dict[str, Any] = {}
        for other in ACQUISITIONS:
            found = [
                _curve_for(cells, engine, arm, other, column).get(at, float("nan"))
                for engine in engines
            ]
            finite = [v for v in found if np.isfinite(v)]
            per_rule[other] = {
                "per_engine": dict(zip(engines, found, strict=True)),
                "mean": float(np.mean(finite)) if finite else float("nan"),
                "engines_measured": len(finite),
            }
        measured = [r for r, row in per_rule.items() if row["engines_measured"] == len(engines)]
        ordered = sorted(measured, key=lambda r: -per_rule[r]["mean"])
        return {
            "arm": arm,
            "budget": at,
            "per_rule": per_rule,
            "ranking_over_fully_measured_rules": ordered,
            "rules_not_run_for_this_arm": [r for r in ACQUISITIONS if r not in measured],
        }

    tests["1_random_labelling_does_as_well"] = {
        "question": "does the selected acquisition beat a matched random one at the same budget?",
        "on_the_selected_arm": {
            "arm": design.arm,
            "versus_random": compare(curve(design.arm, Q_RANDOM), size),
        },
        "on_a_frozen_feature_arm": {
            "arm": A5_PREFIX,
            "note": (
                "the selected arm runs under the random rule and the selected rule only, so the "
                "document-matched random control is read on an arm that spans all seven rules"
            ),
            "ranking": rank_rules(A5_PREFIX, size),
        },
    }
    tests["2_calibration_only_explains_the_gain"] = {
        "question": "does A1, which cannot change the ranking, reach the same endpoint?",
        "comparison": compare(
            {
                engine: _curve_for(
                    cells,
                    engine,
                    A1_CALIBRATION,
                    rule,
                    endpoint_column(CUT_CALIBRATED, PRIMARY_EPSILON),
                )
                for engine in engines
            },
            size,
        ),
    }
    tests["3_target_cut_selection_explains_the_gain"] = {
        "question": "does A2, which changes only where the cut goes, reach the same endpoint?",
        "comparison": compare(curve(A2_CUT, rule), size),
    }
    tests["4_ranking_improves_but_the_frontier_does_not"] = {
        "question": "is a global ranking gain buying anything operationally?",
        "per_engine": {
            engine: {
                "delta_auroc": float(
                    _curve_for(cells, engine, design.arm, rule, "rank__auroc_safe").get(
                        size, float("nan")
                    )
                    - _curve_for(cells, engine, A0_FROZEN, rule, "rank__auroc_safe").get(
                        0, float("nan")
                    )
                ),
                "delta_frontier": float(
                    _curve_for(cells, engine, design.arm, rule, f"frontier__{PRIMARY_KEY}").get(
                        size, float("nan")
                    )
                    - _curve_for(cells, engine, A0_FROZEN, rule, f"frontier__{PRIMARY_KEY}").get(
                        0, float("nan")
                    )
                ),
                "delta_deployed": float(
                    selected[engine].get(size, float("nan")) - frozen[engine].get(0, float("nan"))
                ),
            }
            for engine in engines
        },
    }
    coverage_column = f"dep__{cut}__{PRIMARY_KEY}__coverage"
    tests["5_it_merely_lowers_coverage"] = {
        "question": "is the harm bound being held by accepting less rather than accepting better?",
        "per_engine": {
            engine: {
                "selected_coverage": float(
                    _curve_for(cells, engine, design.arm, rule, coverage_column).get(
                        size, float("nan")
                    )
                ),
                "baseline_coverage": float(
                    _curve_for(cells, engine, A0_FROZEN, rule, coverage_column).get(0, float("nan"))
                ),
                "coverage_floor": design.coverage_floor,
            }
            for engine in engines
        },
    }
    tests["6_it_overfits_the_adaptation_documents"] = {
        "question": "does the arm order its own budget far better than the evaluation block?",
        "per_engine": {
            engine: {
                "in_sample_pair_accuracy": float(
                    _curve_for(cells, engine, design.arm, rule, "budget__pair_accuracy").get(
                        size, float("nan")
                    )
                ),
                "held_out_pair_accuracy": float(
                    _curve_for(cells, engine, design.arm, rule, "rank__pair_accuracy").get(
                        size, float("nan")
                    )
                ),
            }
            for engine in engines
        },
    }
    tests["7_it_fails_on_new_documents_from_the_same_engine"] = {
        "question": "is the endpoint stable across document shards of the evaluation block?",
        "per_engine": {
            engine: {
                "selected": _shard_endpoints(counts, engine, design.arm, rule, size, cut),
                "baseline": _shard_endpoints(counts, engine, A0_FROZEN, rule, 0, cut),
            }
            for engine in engines
        },
    }
    tests["8_selection_collapses_at_low_beneficial_prevalence"] = {
        "question": "does the budget contain enough beneficial rows to select anything?",
        "per_engine": {
            engine: {
                "budget_beneficial": float(
                    _curve_for(cells, engine, design.arm, rule, "budget_beneficial").get(
                        size, float("nan")
                    )
                ),
                "budget_harmful": float(
                    _curve_for(cells, engine, design.arm, rule, "budget_harmful").get(
                        size, float("nan")
                    )
                ),
                "pool_beneficial_rate": float(
                    cc_read_json(FROZEN_IDENTITY)["per_engine"][engine]["pool_beneficial_rate"]
                ),
                "delta_versus_frozen": float(
                    selected[engine].get(size, float("nan")) - frozen[engine].get(0, float("nan"))
                ),
            }
            for engine in engines
        },
        "sgv6_families_the_budget_selected": _sgv6_family_counts(cells),
        "reading": (
            "SGV6's own selection collapsed at low beneficial prevalence, which is why this "
            "stage fixes one arm from source-side simulation. The family counts show what a "
            "budget-driven selection actually picks here: a distribution concentrated on the "
            "unadapted model is a selection declining to adapt, not a selection succeeding."
        ),
    }
    joint_sizes = [n for n in JOINT_BUDGETS if n > 0]
    prefix_curve = curve(A5_PREFIX, Q_RANDOM)

    def against_prefix(other: dict[str, dict[int, float]], at: int) -> dict[str, Any]:
        deltas = {
            engine: float(
                other[engine].get(at, float("nan")) - prefix_curve[engine].get(at, float("nan"))
            )
            for engine in engines
        }
        wins = sum(1 for value in deltas.values() if np.isfinite(value) and value > 0.0)
        return {
            "per_engine_delta_joint_minus_prefix": deltas,
            "engines_where_the_joint_refit_is_higher": wins,
            "joint_refit_wins": bool(wins >= ENGINE_BAR),
        }

    tests["9_joint_refit_outperforms_prefix_adaptation"] = {
        "question": (
            "does moving the hypothesis class beat correcting the frozen one at matched budget?"
        ),
        "per_budget": {
            f"n_{n}": {
                "a4_joint_versus_a5_prefix": against_prefix(curve(A4_JOINT, Q_RANDOM), n),
                "a6_capacity_versus_a5_prefix": against_prefix(curve(A6_CAPACITY, Q_RANDOM), n),
            }
            for n in joint_sizes
        },
        "note": (
            "read against the RANDOM acquisition, which is the rule every joint arm ran under, so "
            "the contrast is the hypothesis class and not the acquisition. The comparison is "
            "stated as joint minus prefix whichever arm the design selected."
        ),
    }
    tests["10_prefix_acquisition_beats_diversity_and_random"] = {
        "question": "does targeting the prefix buy better labels than covering the space?",
        "note": (
            "read on the frozen-feature arms, which span all seven rules. The selected arm runs "
            "under two of them, so an acquisition comparison on it would be reading absence as "
            "evidence about the five it never saw."
        ),
        "per_arm": {arm: rank_rules(arm, size) for arm in (A5_PREFIX, A3_HEAD, A1_CALIBRATION)},
        "source_side_ranking": cc_read_json(DESIGN_RECORD)["source_side_simulation"][
            "4_acquisition"
        ]["grid"],
    }
    permutation_budget = size if size in JOINT_BUDGETS else max(n for n in JOINT_BUDGETS)
    tests["11_label_permutation_reproduces_the_gain"] = {
        "question": "does the arm gain when the budget's labels are shuffled within it?",
        "on_the_selected_arms_own_family": {
            "arm": A4_PERMUTED,
            "budget": permutation_budget,
            "comparison": compare(curve(A4_PERMUTED, rule), permutation_budget),
            "permuted_versus_frozen": {
                engine: float(
                    _curve_for(cells, engine, A4_PERMUTED, rule, column).get(
                        permutation_budget, float("nan")
                    )
                    - frozen[engine].get(0, float("nan"))
                )
                for engine in engines
            },
            "note": (
                "the joint refit with the budget's outcome labels permuted within the budget. "
                "The rows, the documents, the acquisition, the class balance and the source "
                "block are identical; only the label-to-row assignment is destroyed. It runs on "
                "the joint arms' reduced ladder, so it is read at the nearest budget on that "
                "ladder to the operating point."
            ),
        },
        "on_the_proposed_arm": {
            "arm": A5_PERMUTED,
            "budget": size,
            "comparison": compare(curve(A5_PERMUTED, rule), size),
            "permuted_versus_frozen": {
                engine: float(
                    _curve_for(cells, engine, A5_PERMUTED, rule, column).get(size, float("nan"))
                    - frozen[engine].get(0, float("nan"))
                )
                for engine in engines
            },
        },
    }
    mechanism = cc_read_json(MECHANISM)["per_engine"]
    dominant: dict[str, Any] = {}
    for engine in engines:
        row = mechanism[engine]["arms"].get(design.arm, {}).get(f"n_{size}", {})
        effects = {
            key: float(row.get(key, float("nan")))
            for key in (
                "m1_score_scale",
                "m2_ranking_at_fixed_depth",
                "m3_representation_residual",
                "m4_depth_at_fixed_ranking",
            )
        }
        finite = {k: v for k, v in effects.items() if np.isfinite(v)}
        dominant[engine] = {
            "effects": effects,
            "largest": max(finite, key=lambda k: abs(finite[k])) if finite else None,
        }
    labels = {row["largest"] for row in dominant.values() if row["largest"]}
    easyocr_label = dominant.get("easyocr", {}).get("largest")
    others = {
        row["largest"] for name, row in dominant.items() if name != "easyocr" and row["largest"]
    }
    tests["12_easyocr_needs_a_different_mechanism"] = {
        "question": "is the dominant mechanism the same on every engine?",
        "per_engine": dominant,
        "distinct_dominant_mechanisms": sorted(labels),
        "easyocr_dominant_mechanism": easyocr_label,
        "easyocr_differs_from_every_other_engine": bool(
            easyocr_label is not None and easyocr_label not in others
        ),
        "reading": (
            "a single dominant mechanism across four engines supports a universal adapter; more "
            "than one supports the heterogeneous reading section 34 calls outcome D. The label "
            "is the largest ABSOLUTE effect, so a mechanism that is dominant and harmful is "
            "still reported as dominant."
        ),
    }
    tests["13_one_document_carries_the_result"] = {
        "question": "how much of the endpoint does the single most influential page carry?",
        "per_engine": {
            engine: _document_influence(counts, engine, design.arm, rule, size, cut)
            for engine in engines
        },
    }
    monotone: dict[str, Any] = {}
    for engine in engines:
        ladder = sorted(selected[engine])
        values = [selected[engine][n] for n in ladder]
        steps = [
            (ladder[i], ladder[i + 1], values[i + 1] - values[i]) for i in range(len(values) - 1)
        ]
        monotone[engine] = {
            "rank_correlation_with_the_budget": _spearman(values),
            "non_increasing_steps": [
                {"from": int(a), "to": int(b), "delta": float(d)} for a, b, d in steps if d < 0.0
            ],
            "steps": len(steps),
        }
    tests["14_more_labels_monotonically_improve"] = {
        "question": "does the curve rise with the budget?",
        "per_engine": monotone,
    }
    tests["15_more_labels_make_it_worse"] = {
        "question": "are there budgets where buying more labels costs endpoint?",
        "per_engine": {engine: monotone[engine]["non_increasing_steps"] for engine in engines},
        "note": "reported as it is. A non-monotone curve is a result, not noise to be smoothed.",
    }

    identity = cc_read_json(FROZEN_IDENTITY)["per_engine"]
    # The oracle re-ranking is SGV10's published ceiling and is NOT the frozen score; it is
    # excluded by name rather than by a tolerance, so a real identity failure cannot hide in it.
    zero = cells[(cells["budget"] == 0) & (~cells["arm"].isin(ORACLE_ARMS))]
    assertions = {
        "n_zero_is_the_frozen_model_on_every_arm": {
            "cells": len(zero),
            "all_identical_to_frozen": bool(zero["identical_to_frozen"].all()),
            "arms": sorted({str(name) for name in zero["arm"].tolist()}),
            "excluded": list(ORACLE_ARMS),
            "excluded_reason": (
                "`o2_oracle_rerank` carries SGV10's published ceiling ranking, which is not the "
                "frozen score and must not be; `o1_oracle_cut` does carry the frozen ranking and "
                "differs only in where its cut goes"
            ),
        },
        "rebuilt_sgv5_score_matches_the_published_one": {
            engine: identity[engine]["rebuild_max_abs_difference_from_published_sgv5"]
            for engine in engines
        },
        "adaptation_pool_matches_sgv6s": {
            engine: identity[engine]["sgv6_pool_score_max_abs_difference"] for engine in engines
        },
        "no_oracle_arm_reached_the_design": {
            "design": _design_dict(design),
            "oracle_names": [*ORACLE_ARMS, Q_ORACLE],
            "clean": bool(
                not any(
                    str(value).startswith(("o1_", "o2_", "o3_")) or str(value) == Q_ORACLE
                    for value in _design_dict(design).values()
                )
            ),
        },
        "the_disagreement_rule_was_not_selectable": {
            "selected_acquisition": design.acquisition,
            "excluded": Q_DISAGREEMENT,
            "clean": bool(design.acquisition != Q_DISAGREEMENT),
        },
    }
    for name, block in assertions.items():
        if name.endswith("_the_published_one") or name == "adaptation_pool_matches_sgv6s":
            worst = max(float(value) for value in block.values())
            if worst != 0.0:
                raise PhaseError(f"{name}: maximum difference {worst:g} is not 0.0")
    if not assertions["n_zero_is_the_frozen_model_on_every_arm"]["all_identical_to_frozen"]:
        raise PhaseError("an arm at budget 0 did not reproduce the frozen score exactly")

    _write(
        NEGATIVE_TESTS,
        {
            "frozen_design": _design_dict(design),
            "intent": (
                "each test is an attempt to make this stage's claim false. Where a control wins "
                "it is recorded as a control that won; nothing here is promoted afterwards."
            ),
            "operating_budget": size,
            "failure_tests": tests,
            "leakage_assertions": assertions,
            "runtime_seconds": round(time.time() - started, 1),
        },
        "negative_tests",
    )
    print(f"negative: {len(tests)} failure tests, {len(assertions)} assertions")
    return 0


# ------------------------------------------------------------------ figures

FIGURE_ENGINE_ORDER = ("doctr", "easyocr", "paddleocr", "tesseract")
FIGURE_ARMS = (A0_FROZEN, A1_CALIBRATION, A2_CUT, A3_HEAD, A5_PREFIX, SGV6_GENERIC)


def run_figures() -> int:
    """The six figures the brief names, each carrying the development-only annotation."""
    started = time.time()
    design = load_design()
    cells = load_cells()
    results = cc_read_json(GAP_RECOVERY)
    mechanism = cc_read_json(MECHANISM)
    acquisition = cc_read_json(ACQUISITION_RESULTS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV13 DEVELOPMENT -- not a confirmatory result"
    engines = [e for e in FIGURE_ENGINE_ORDER if e in set(cells["held_out_engine"].tolist())]
    written: list[Path] = []
    column = endpoint_column(design.primary_cut, PRIMARY_EPSILON)

    def rule_for(arm: str) -> str:
        # Every arm ran under the rule the design froze except the joint refit the design did NOT
        # select, which ran under the random rule alone. Plotting that one under the frozen rule
        # would draw an empty line.
        allowed = acquisitions_for(arm, design)
        return design.acquisition if design.acquisition in allowed else Q_RANDOM

    def column_for(arm: str) -> str:
        # A1's mechanism IS its calibrated absolute cut; under the two shared rules it is A0 or
        # A2 by construction, so plotting it under the shared column would draw a line that says
        # nothing about the arm. Every other arm is plotted under the primary rule.
        return endpoint_column(CUT_CALIBRATED, PRIMARY_EPSILON) if arm == A1_CALIBRATION else column

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    def ladder(arm: str, engine: str, rule: str, name: str) -> tuple[list[int], list[float]]:
        curve = _curve_for(cells, engine, arm, rule, name)
        keys = sorted(curve)
        return keys, [curve[k] for k in keys]

    figure, panels = plt.subplots(1, len(engines), figsize=(4.0 * len(engines), 4.0), sharey=True)
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        for arm in FIGURE_ARMS:
            keys, values = ladder(arm, engine, rule_for(arm), column_for(arm))
            if keys:
                panel.plot(keys, values, marker="o", markersize=3, label=arm)
        panel.set_title(engine, fontsize=9)
        panel.set_xscale("symlog", linthresh=5)
        panel.set_xlabel("target labels")
    np.atleast_1d(panels)[0].set_ylabel(f"repair recall at harm <= {PRIMARY_EPSILON:g}")
    np.atleast_1d(panels)[-1].legend(fontsize=6, loc="best")
    finish(figure, FIGURE_DIR / "repair_recall_vs_labels.png", "Risk-controlled repair recall")

    figure, panels = plt.subplots(1, len(engines), figsize=(4.0 * len(engines), 4.0), sharey=True)
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        block = results["per_engine"][engine]["arms"]
        for arm in (design.arm, SGV6_GENERIC, A3_HEAD, A1_CALIBRATION):
            curve = block.get(arm, {}).get(rule_for(arm), {}).get("deployed_recovery", {})
            keys = sorted(int(k[2:]) for k in curve)
            if keys:
                panel.plot(
                    keys, [curve[f"n_{k}"] for k in keys], marker="o", markersize=3, label=arm
                )
        for milestone in RECOVERY_MILESTONES:
            panel.axhline(milestone, color="grey", linewidth=0.5, linestyle=":")
        panel.set_title(engine, fontsize=9)
        panel.set_xscale("symlog", linthresh=5)
        panel.set_xlabel("target labels")
    np.atleast_1d(panels)[0].set_ylabel("recovery of SGV12's actionable gap")
    np.atleast_1d(panels)[-1].legend(fontsize=6, loc="best")
    finish(figure, FIGURE_DIR / "gap_recovery_vs_labels.png", "Gap recovery against SGV12")

    harm_column = f"deployed_prefix__{design.primary_cut}__prefix_harm_rate"
    figure, panels = plt.subplots(1, len(engines), figsize=(4.0 * len(engines), 4.0), sharey=True)
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        for arm in FIGURE_ARMS:
            stem = (
                f"deployed_prefix__{CUT_CALIBRATED}__prefix_harm_rate"
                if arm == A1_CALIBRATION
                else harm_column
            )
            keys, values = ladder(arm, engine, rule_for(arm), stem)
            if keys:
                panel.plot(keys, values, marker="o", markersize=3, label=arm)
        panel.axhline(PRIMARY_EPSILON, color="black", linewidth=0.7)
        panel.set_title(engine, fontsize=9)
        panel.set_xscale("symlog", linthresh=5)
        panel.set_xlabel("target labels")
    np.atleast_1d(panels)[0].set_ylabel("accepted-prefix harm rate")
    np.atleast_1d(panels)[-1].legend(fontsize=6, loc="best")
    finish(figure, FIGURE_DIR / "prefix_harm_vs_labels.png", "Accepted-prefix purity")

    figure, panel = plt.subplots(figsize=(9.0, 4.6))
    positions = np.arange(len(engines), dtype=float)
    width = 0.8 / max(len(ACQUISITIONS), 1)
    for offset, rule in enumerate(ACQUISITIONS):
        values = [
            acquisition["per_engine"][engine][A5_PREFIX].get(rule, {}).get("mean_over_budgets", 0.0)
            for engine in engines
        ]
        panel.bar(
            positions + (offset - len(ACQUISITIONS) / 2 + 0.5) * width, values, width, label=rule
        )
    panel.set_xticks(positions)
    panel.set_xticklabels(engines, fontsize=8)
    panel.set_ylabel("mean risk-controlled repair recall over the ladder")
    panel.legend(fontsize=6, ncol=2)
    finish(figure, FIGURE_DIR / "acquisition_comparison.png", "Acquisition rules")

    figure, panels = plt.subplots(1, len(engines), figsize=(4.0 * len(engines), 4.0), sharey=True)
    keys = (
        "m1_score_scale",
        "m2_ranking_at_fixed_depth",
        "m3_representation_residual",
        "m4_depth_at_fixed_ranking",
    )
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        block = mechanism["per_engine"][engine]["arms"].get(design.arm, {})
        row = block.get(f"n_{design.operating_budget}", {})
        panel.bar(np.arange(len(keys)), [row.get(k, 0.0) for k in keys])
        panel.set_xticks(np.arange(len(keys)))
        panel.set_xticklabels([k.split("_", 1)[0] for k in keys], fontsize=8)
        panel.axhline(0.0, color="black", linewidth=0.6)
        panel.set_title(f"{engine} (n={design.operating_budget})", fontsize=9)
    np.atleast_1d(panels)[0].set_ylabel("effect on the endpoint vs the frozen baseline")
    finish(figure, FIGURE_DIR / "mechanism_decomposition.png", "Mechanism decomposition")

    figure, panel = plt.subplots(figsize=(9.0, 4.6))
    width = 0.8 / max(len(RECOVERY_MILESTONES), 1)
    for offset, milestone in enumerate(RECOVERY_MILESTONES):
        key = f"labels_to_{int(milestone * 100)}pc"
        values = []
        for engine in engines:
            block = results["per_engine"][engine]["arms"].get(design.arm, {})
            reached = block.get(primary_rule(design), {}).get("deployed_milestones", {}).get(key)
            values.append(float(reached) if reached else 0.0)
        panel.bar(
            positions + (offset - len(RECOVERY_MILESTONES) / 2 + 0.5) * width,
            values,
            width,
            label=f"{int(milestone * 100)}% recovery",
        )
    panel.set_xticks(positions)
    panel.set_xticklabels(engines, fontsize=8)
    panel.set_ylabel("target labels needed (0 = milestone not reached on the ladder)")
    panel.legend(fontsize=7)
    finish(figure, FIGURE_DIR / "engine_label_efficiency.png", "Label cost of each milestone")

    _write(
        FIGURE_MANIFEST,
        {
            "figures": [
                {
                    "path": cc._relative(path),
                    "sha256": file_sha256(path),
                    "derived_from": [
                        cc._relative(SCORES),
                        cc._relative(GAP_RECOVERY),
                        cc._relative(MECHANISM),
                        cc._relative(ACQUISITION_RESULTS),
                    ],
                }
                for path in written
            ],
            "annotation": note,
            "runtime_seconds": round(time.time() - started, 1),
        },
        "figure_manifest",
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ the decision


def _favourable(delta: float) -> bool:
    return bool(np.isfinite(delta) and delta > 0.0)


def run_decide() -> int:
    """The six pre-registered criteria, evaluated exactly as they were written down.

    Nothing here is computed for the first time: every input is read from an artifact an earlier
    phase wrote. The criteria were fixed in `--preregister` before any outer target row was read
    and the verdict is whatever they say.
    """
    started = time.time()
    design = load_design()
    cells = load_cells()
    gap = cc_read_json(GAP_RECOVERY)
    comparison = cc_read_json(SGV6_COMPARISON)
    prefix = cc_read_json(PREFIX_PURITY)
    stats = cc_read_json(STATISTICAL_TESTS)
    negative = cc_read_json(NEGATIVE_TESTS)
    easyocr = cc_read_json(EASYOCR_ANALYSIS)
    engines = sorted({str(name) for name in cells["held_out_engine"].tolist()})
    size = design.operating_budget
    rule = primary_rule(design)
    cut = design.primary_cut
    column = endpoint_column(cut, PRIMARY_EPSILON)

    per_engine: dict[str, Any] = {}
    for engine in engines:
        block = cells[
            (cells["held_out_engine"] == engine)
            & (cells["arm"] == design.arm)
            & (cells["acquisition"] == rule)
            & (cells["budget"] == size)
        ]
        generic = cells[
            (cells["held_out_engine"] == engine)
            & (cells["arm"] == SGV6_GENERIC)
            & (cells["acquisition"] == rule)
            & (cells["budget"] == size)
        ]
        baseline = cells[
            (cells["held_out_engine"] == engine)
            & (cells["arm"] == A0_FROZEN)
            & (cells["budget"] == 0)
        ]
        harm_column = f"deployed_prefix__{cut}__prefix_harm_rate"
        swap_column = f"deployed_prefix__{cut}__swaps_recovered"
        per_engine[engine] = {
            "selected_endpoint": float(block[column].mean()) if not block.empty else float("nan"),
            "sgv6_endpoint": (float(generic[column].mean()) if not generic.empty else float("nan")),
            "baseline_endpoint": float(baseline[column].mean()),
            "realized_harm_rate": float(
                block[f"dep__{cut}__{PRIMARY_KEY}__realized_harm_rate"].mean()
            ),
            "holds_bound_share": float(block[f"dep__{cut}__{PRIMARY_KEY}__holds_bound"].mean()),
            "coverage": float(block[f"dep__{cut}__{PRIMARY_KEY}__coverage"].mean()),
            "accepted_rows": float(block[f"dep__{cut}__{PRIMARY_KEY}__n_accepted"].mean()),
            "prefix_harm_rate": float(block[harm_column].mean()),
            "baseline_prefix_harm_rate": float(baseline[harm_column].mean()),
            "swaps_recovered": float(block[swap_column].mean()),
            "deployed_recovery": gap["per_engine"][engine]["arms"][design.arm][rule][
                "deployed_recovery"
            ].get(f"n_{size}", float("nan")),
            "achievable_recovery": gap["per_engine"][engine]["arms"][design.arm][rule][
                "achievable_recovery"
            ].get(f"n_{size}", float("nan")),
        }

    criterion_1 = {
        name: _favourable(row["selected_endpoint"] - row["sgv6_endpoint"])
        for name, row in per_engine.items()
    }
    criterion_3 = {
        name: bool(row["holds_bound_share"] >= 1.0 and row["accepted_rows"] > 0.0)
        for name, row in per_engine.items()
    }
    criterion_5 = {
        name: bool(
            row["prefix_harm_rate"] < row["baseline_prefix_harm_rate"]
            or row["swaps_recovered"] > 0.0
        )
        for name, row in per_engine.items()
    }
    criterion_6 = {
        name: bool(row["coverage"] >= design.coverage_floor and row["coverage"] > 0.0)
        for name, row in per_engine.items()
    }
    milestone_key = f"labels_to_{int(MILESTONE_FOR_CRITERION_4 * 100)}pc"
    efficiency: dict[str, Any] = {}
    for engine in engines:
        block = comparison["per_engine"][engine].get(rule, {})
        ours = block.get("sgv13_milestones", {}).get(milestone_key)
        theirs = block.get("sgv6_milestones", {}).get(milestone_key)
        efficiency[engine] = {
            "sgv13_labels": ours,
            "sgv6_labels": theirs,
            "reduction": (float(1.0 - ours / theirs) if ours and theirs else None),
            "meets_bar": bool(
                ours is not None
                and theirs is not None
                and ours > 0
                and theirs > 0
                and (1.0 - ours / theirs) >= LABEL_REDUCTION_BAR
            ),
        }

    criteria = {
        "1_operational_improvement": {
            "holds": bool(sum(criterion_1.values()) >= ENGINE_BAR),
            "per_engine": criterion_1,
            "rule": PRE_REGISTRATION["criteria"]["1_operational_improvement"],
            "engines_favourable": int(sum(criterion_1.values())),
        },
        "2_cross_engine_breadth": {
            "holds": bool(sum(criterion_1.values()) >= ENGINE_BAR),
            "engines_favourable": int(sum(criterion_1.values())),
            "bar": ENGINE_BAR,
            "rule": PRE_REGISTRATION["criteria"]["2_cross_engine_breadth"],
        },
        "3_safety": {
            "holds": bool(sum(criterion_3.values()) >= ENGINE_BAR),
            "per_engine": criterion_3,
            "engines_holding": int(sum(criterion_3.values())),
            "rule": PRE_REGISTRATION["criteria"]["3_safety"],
        },
        "4_label_efficiency": {
            "holds": bool(sum(1 for row in efficiency.values() if row["meets_bar"]) >= ENGINE_BAR),
            "per_engine": efficiency,
            "milestone": MILESTONE_FOR_CRITERION_4,
            "bar": LABEL_REDUCTION_BAR,
            "rule": PRE_REGISTRATION["criteria"]["4_label_efficiency"],
        },
        "5_prefix_mechanism": {
            "holds": bool(sum(criterion_5.values()) >= ENGINE_BAR),
            "per_engine": criterion_5,
            "rule": PRE_REGISTRATION["criteria"]["5_prefix_mechanism"],
        },
        "6_non_degenerate": {
            "holds": bool(sum(criterion_6.values()) >= ENGINE_BAR),
            "per_engine": criterion_6,
            "coverage_floor": design.coverage_floor,
            "rule": PRE_REGISTRATION["criteria"]["6_non_degenerate"],
        },
    }
    met = int(sum(1 for row in criteria.values() if row["holds"]))
    verdict = "SUPPORTED" if met == len(criteria) else "NOT SUPPORTED"

    # The mechanism label is computed once, in `--negative`, and read here. Two computations of
    # the same argmax is two chances to disagree about what the stage found.
    dominant = negative["failure_tests"]["12_easyocr_needs_a_different_mechanism"]["per_engine"]

    # SGV13-P1 names a mechanism -- adaptation focused on the deployment prefix -- and the
    # registered selection rule (S0) may or may not choose the arm that implements it. The
    # criteria are read on the source-selected arm, as registered. The proposed arm's own
    # comparison is reported beside them whichever way the selection fell, so the named
    # hypothesis is answered directly rather than by whatever the simulation happened to pick.
    permutation_budget = size if size in JOINT_BUDGETS else max(n for n in JOINT_BUDGETS)
    permuted_control = {
        engine: {
            "permuted": _curve_for(cells, engine, A4_PERMUTED, rule, column).get(
                permutation_budget, float("nan")
            ),
            "unpermuted": _curve_for(cells, engine, design.arm, rule, column).get(
                permutation_budget, float("nan")
            ),
            "baseline": _curve_for(cells, engine, A0_FROZEN, rule, column).get(0, float("nan")),
        }
        for engine in engines
    }
    proposed_rule = design.acquisition if A5_PREFIX not in JOINT_ARMS else Q_RANDOM
    proposed_direct: dict[str, Any] = {}
    for engine in engines:
        selected_curve = _curve_for(cells, engine, A5_PREFIX, proposed_rule, column)
        generic_curve = _curve_for(cells, engine, SGV6_GENERIC, rule, column)
        frozen_curve = _curve_for(cells, engine, A0_FROZEN, rule, column)
        proposed_direct[engine] = {
            "a5_prefix_focused": selected_curve.get(size, float("nan")),
            "b1_sgv6_generic": generic_curve.get(size, float("nan")),
            "a0_frozen": frozen_curve.get(0, float("nan")),
            "delta_versus_sgv6": float(
                selected_curve.get(size, float("nan")) - generic_curve.get(size, float("nan"))
            ),
            "delta_versus_frozen": float(
                selected_curve.get(size, float("nan")) - frozen_curve.get(0, float("nan"))
            ),
            "prefix_swap_pairs_available": float(
                _curve_for(cells, engine, A5_PREFIX, proposed_rule, "pairs__prefix_swap").get(
                    size, float("nan")
                )
            ),
        }
    _write(
        DECISION,
        {
            "verdict": verdict,
            "label_permutation_control_on_the_selected_family": {
                "arm": A4_PERMUTED,
                "budget": permutation_budget,
                "per_engine": permuted_control,
                "note": (
                    "section 21's control, on the family the design selected. The rows, the "
                    "documents, the acquisition, the class balance and the source block are "
                    "identical to the selected arm's; only the label-to-row assignment inside "
                    "the budget is destroyed. A gain that survives this is a gain from training "
                    "mechanics rather than from target supervision."
                ),
            },
            "proposed_mechanism_direct_comparison": {
                "arm": A5_PREFIX,
                "acquisition": proposed_rule,
                "is_the_selected_method": bool(design.arm == A5_PREFIX),
                "per_engine": proposed_direct,
                "note": (
                    "SGV13-P1 names prefix-focused adaptation. The six criteria are read on the "
                    "arm the source-side simulation selected, which is the registered rule; this "
                    "block reports the named arm's own numbers whether or not it was the one "
                    "selected, so the hypothesis is answered directly. If it was not selected, "
                    "these numbers are SECONDARY and NON-CONFIRMATORY by construction."
                ),
            },
            "criteria_met": met,
            "criteria_total": len(criteria),
            "criteria": criteria,
            "selected_adaptation_method": design.arm,
            "selected_acquisition_method": design.acquisition,
            "selected_cut_rule": design.primary_cut,
            "target_label_budgets": list(BUDGETS),
            "operating_budget": size,
            "primary_epsilon": PRIMARY_EPSILON,
            "results_by_engine": per_engine,
            "safety_by_engine": criterion_3,
            "deployment_rate_by_engine": {
                name: row["coverage"] for name, row in per_engine.items()
            },
            "gap_recovery_by_budget": {
                engine: gap["per_engine"][engine]["arms"][design.arm][rule]["deployed_recovery"]
                for engine in engines
            },
            "labels_to_milestones": {
                engine: gap["per_engine"][engine]["arms"][design.arm][rule]["deployed_milestones"]
                for engine in engines
            },
            "sgv6_comparison": efficiency,
            "prefix_purity_change": {
                engine: {
                    "baseline": per_engine[engine]["baseline_prefix_harm_rate"],
                    "selected": per_engine[engine]["prefix_harm_rate"],
                    "swaps_recovered": per_engine[engine]["swaps_recovered"],
                }
                for engine in engines
            },
            "mechanism_decomposition": dominant,
            "statistical_significance": {
                "family_size": stats["family_size"],
                "surviving_holm": stats["surviving_holm"],
                "surviving_and_favourable": stats["surviving_and_favourable"],
                "surviving_and_unfavourable": stats["surviving_and_unfavourable"],
            },
            "holm_family_definition": stats["family_definition"],
            "negative_test_outcomes": {
                name: block.get("question", "") for name, block in negative["failure_tests"].items()
            },
            "easyocr_mechanism_outcome": easyocr["per_engine"].get("easyocr", {}),
            "post_hoc_diagnostics_separated": {
                "these_are_diagnostic_and_entered_no_criterion": [
                    ORACLE_CUT,
                    ORACLE_RERANK,
                    "o3_oracle_acquisition",
                    Q_ORACLE,
                ],
                "note": (
                    "the oracle arms read the evaluation block's own labels. They bound what is "
                    "achievable and are reported for that; none of them appears in a criterion, "
                    "in the design record, or in a headline claim."
                ),
            },
            "matched_coverage_prefix": {
                engine: prefix["per_engine"][engine][design.arm][rule]["matched_coverage"][
                    "prefix_harm_rate"
                ].get(f"n_{size}", {})
                for engine in engines
            },
            "limitations": [
                "four OCR engines and one receipt corpus. Four folds cannot establish that a "
                "budget which fails here fails in general, and section 24's rule against "
                "inferring impossibility from four engines applies to this stage too.",
                "the adaptation pool is the held-out engine's TRAIN documents, so every budget "
                "is drawn from pages a real deployment would also have to annotate; the cost "
                "model is labels and documents and not annotator minutes.",
                "the acquisition orderings are computed once from the frozen model and never "
                "revised as labels arrive, so these curves are a lower bound on what a "
                "sequential loop could deliver rather than an estimate of it.",
                "the two joint-refit arms and the SGV6 comparison run at fewer draws and, for "
                "the joint arms, on a coarser budget ladder; their intervals are wider and are "
                "reported as such.",
                "the ranking-disagreement acquisition could not be validated on source-side "
                "simulation, because SGV12's published rankings on an inner pool were fitted "
                "with the outer held-out engine in their source set. It is measured and "
                "reported but was never eligible to be selected.",
                "beneficial candidates are rare on two of the four engines -- the docTR and "
                "PaddleOCR adaptation pools are about 3% beneficial -- so a nominal budget of N "
                "rows is a very different experiment on different engines, and the curves are "
                "reported per engine and never pooled.",
            ],
            "runtime_seconds": round(time.time() - started, 1),
        },
        "research_decision",
    )
    print(f"decide: {verdict} -- {met} of {len(criteria)} criteria")
    return 0


# ------------------------------------------------------------------ provenance

DEPENDENCY_SCRIPTS = (
    "scripts/sgv1_verifier_pilot.py",
    "scripts/sgv1_candidate_conditioned.py",
    "scripts/sgv1_risk_policy.py",
    "scripts/sgv1_domain_generalization.py",
    "scripts/sgv2_reliability_layer.py",
    "scripts/sgv3_self_aware.py",
    "scripts/sgv4_environment_aware.py",
    "scripts/sgv5_candidate_reliability.py",
    "scripts/sgv6_target_engine_adaptation.py",
    "scripts/sgv9_target_calibration.py",
    "scripts/sgv10_tail_risk_control.py",
    "scripts/sgv11_minimax_policy_transfer.py",
    "scripts/sgv12_robust_ranking_transfer.py",
    "scripts/sgv13_fewshot_prefix_purity_adaptation.py",
)

UPSTREAM = (
    dg.DESIGN_MATRIX,
    dg.FIT_RECORD,
    c5.FEATURES,
    c5.PREDICTIONS,
    c5.POLICY_SELECTION,
    c5.FIT_RECORD,
    s6.SCORES,
    s6.ADAPTATION_RESULTS,
    s6.PROVENANCE,
    s9.SCORES,
    s9.PROVENANCE,
    s10.SCORES,
    s10.DESIGN_RECORD,
    s10.PROVENANCE,
    s11.DECISION_TRANSFER,
    s11.SCORES,
    s11.PROVENANCE,
    s12.SCORES,
    s12.GAP_DECOMPOSITION,
    s12.DESIGN_RECORD,
    s12.PROVENANCE,
)

PRODUCED = (
    DESIGN_RECORD,
    FROZEN_IDENTITY,
    BUDGET_INVENTORY,
    SCORES,
    COUNTS,
    ADAPTATION_RESULTS,
    PREFIX_PURITY,
    GAP_RECOVERY,
    MECHANISM,
    ACQUISITION_RESULTS,
    SGV6_COMPARISON,
    ORACLE_ANALYSIS,
    EASYOCR_ANALYSIS,
    NEGATIVE_TESTS,
    STATISTICAL_TESTS,
    MODEL_SELECTION,
    DECISION,
    FIGURE_MANIFEST,
)


def _tracked(path: Path) -> str:
    """Is this path in the index, deliberately ignored, or neither. Asked of git, not guessed."""
    import subprocess

    relative = cc._relative(path)
    listed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode == 0:
        return "tracked"
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", relative], cwd=REPO, capture_output=True, check=False
    )
    return "gitignored" if ignored.returncode == 0 else "untracked"


def run_record() -> int:
    """Provenance, the dependency audit section 36 asks for, and the traceability index.

    The chain is now fourteen scripts deep and twelve of them are uncommitted. What a reader
    needs is not that the situation is untidy but exactly which files a clean clone would be
    missing and which of them are regenerable, and that is a question with a checkable answer.
    """
    started = time.time()
    produced = [path for path in PRODUCED if path.is_file()]
    figures = sorted(FIGURE_DIR.glob("*.png")) if FIGURE_DIR.is_dir() else []
    scripts = {name: _tracked(REPO / name) for name in DEPENDENCY_SCRIPTS}
    artifacts = {cc._relative(path): _tracked(path) for path in UPSTREAM}
    outputs = {cc._relative(path): _tracked(path) for path in (*produced, *figures)}
    design = load_design()
    _write(
        MODEL_SELECTION,
        {
            "strategy": "S0 -- a single design chosen entirely from source-side simulation",
            "why": (
                "SGV6 showed that model selection itself collapses when the target sample "
                "carries almost no beneficial rows: on docTR its adaptation pool is 3.0% "
                "beneficial, so a budget of 100 rows holds three of them and an unconstrained "
                "'pick the best arm on the target sample' has nothing to pick on. This stage "
                "therefore fixes ONE arm, one acquisition rule and one cut rule from source-side "
                "simulation and applies it unchanged to every engine. What the budget still "
                "chooses inside itself is a shrinkage or a calibration map, by grouped "
                "cross-validation over its own documents, and every such family contains the "
                "frozen model exactly so a budget carrying no signal returns SGV5."
            ),
            "frozen_design": _design_dict(design),
            "source_side_simulation": cc_read_json(DESIGN_RECORD)["source_side_simulation"],
            "not_used_for_selection": [
                "any evaluation-block label",
                "any oracle arm or oracle acquisition",
                "the ranking-disagreement acquisition, which could not be validated without the "
                "outer held-out engine appearing in the rankings it reads",
            ],
            "runtime_seconds": round(time.time() - started, 1),
        },
        "model_selection",
    )
    _write(
        PROVENANCE,
        {
            "stage_title": "SGV13 -- few-shot target-supervised prefix-purity adaptation",
            "features_added": 0,
            "models_fitted_note": (
                "SGV13 fits adaptation arms on target labels, which is the point of the stage. "
                "It fits no feature extractor and no candidate generator, and it retrains "
                "nothing upstream: SGV5's selected model is rebuilt and asserted identical to "
                "its published decision score before any SGV13 arm is fitted, and the "
                "adaptation pool is asserted identical to SGV6's."
            ),
            "inputs": {cc._relative(p): file_sha256(p) for p in UPSTREAM if p.is_file()},
            "artifacts": {cc._relative(p): file_sha256(p) for p in (*produced, *figures)},
            "dependency_audit": {
                "scripts": scripts,
                "upstream_artifacts": artifacts,
                "outputs": outputs,
                "clean_clone": {
                    "scripts_missing_from_a_clean_clone": sorted(
                        name for name, state in scripts.items() if state != "tracked"
                    ),
                    "upstream_artifacts_missing_from_a_clean_clone": sorted(
                        name for name, state in artifacts.items() if state != "tracked"
                    ),
                    "outputs_gitignored_but_regenerable": sorted(
                        name for name, state in outputs.items() if state == "gitignored"
                    ),
                    "non_regenerable_dependencies": [
                        "data/raw/** -- the recorded OCR responses. Write-once and not committed "
                        "by policy; the download scripts, checksums and license records are what "
                        "the repository ships."
                    ],
                    "known_repository_hygiene_issue": (
                        "tests/leakage/test_sgv7_policy_selection.py and "
                        "tests/leakage/test_sgv8_evidence_aware_deployment.py assert that every "
                        "file named in their provenance manifest exists, including gitignored "
                        "parquet intermediates, so those two suites would fail on a clean clone. "
                        "SGV9's, SGV10's, SGV11's, SGV12's and SGV13's equivalents ask git "
                        "whether a missing file is deliberately ignored and are clean. This is "
                        "recorded, not fixed: repairing SGV7 or SGV8 here would mean editing a "
                        "frozen stage, which section 36 forbids without explicit authorisation."
                    ),
                    "committed": (
                        "nothing. The working tree carries this stage and twelve untracked "
                        "predecessors; what to commit and in what order is not a decision this "
                        "stage may take."
                    ),
                },
            },
            "runtime_seconds": round(time.time() - started, 1),
        },
        "provenance",
    )
    _write(
        TRACEABILITY,
        {
            "phases": {
                "--preregister": [cc._relative(DESIGN_RECORD)],
                "--adapt": [
                    cc._relative(SCORES),
                    cc._relative(COUNTS),
                    cc._relative(FROZEN_IDENTITY),
                    cc._relative(BUDGET_INVENTORY),
                ],
                "--results": [cc._relative(ADAPTATION_RESULTS)],
                "--prefix": [cc._relative(PREFIX_PURITY)],
                "--gap": [cc._relative(GAP_RECOVERY)],
                "--mechanism": [cc._relative(MECHANISM)],
                "--acquisition": [cc._relative(ACQUISITION_RESULTS)],
                "--sgv6": [cc._relative(SGV6_COMPARISON)],
                "--oracle": [cc._relative(ORACLE_ANALYSIS)],
                "--easyocr": [cc._relative(EASYOCR_ANALYSIS)],
                "--negative": [cc._relative(NEGATIVE_TESTS)],
                "--stats": [cc._relative(STATISTICAL_TESTS)],
                "--figures": [cc._relative(FIGURE_MANIFEST)],
                "--decide": [cc._relative(DECISION)],
                "--record": [
                    cc._relative(PROVENANCE),
                    cc._relative(TRACEABILITY),
                    cc._relative(MODEL_SELECTION),
                ],
            },
            "report_sections": {
                "1. Motivation": [cc._relative(GAP_RECOVERY)],
                "2. Frozen evidence from SGV1-SGV12": [cc._relative(FROZEN_IDENTITY)],
                "3. Research questions": [cc._relative(DESIGN_RECORD)],
                "4. Few-shot protocol": [cc._relative(DESIGN_RECORD)],
                "5. Target label budget accounting": [cc._relative(BUDGET_INVENTORY)],
                "6. Adaptation methods": [cc._relative(DESIGN_RECORD)],
                "7. Acquisition methods": [cc._relative(ACQUISITION_RESULTS)],
                "8. Source-side pre-registration simulation": [
                    cc._relative(DESIGN_RECORD),
                    cc._relative(MODEL_SELECTION),
                ],
                "9. Main operational results": [cc._relative(ADAPTATION_RESULTS)],
                "10. Prefix-purity analysis": [cc._relative(PREFIX_PURITY)],
                "11. Gap-recovery analysis": [cc._relative(GAP_RECOVERY)],
                "12. Comparison with SGV6": [cc._relative(SGV6_COMPARISON)],
                "13. Mechanism decomposition": [cc._relative(MECHANISM)],
                "14. EasyOCR analysis": [cc._relative(EASYOCR_ANALYSIS)],
                "15. Acquisition ablations": [cc._relative(ACQUISITION_RESULTS)],
                "16. Negative tests": [
                    cc._relative(NEGATIVE_TESTS),
                    cc._relative(ORACLE_ANALYSIS),
                ],
                "17. Limitations": [cc._relative(DECISION)],
                "18. Research decision": [
                    cc._relative(DECISION),
                    cc._relative(STATISTICAL_TESTS),
                ],
            },
            "note": (
                "every number in `docs/sgv13/fewshot_prefix_purity_adaptation.md` comes from one "
                "of these files. A decimal that cannot be found in one of them does not belong "
                "in the report."
            ),
            "runtime_seconds": round(time.time() - started, 1),
        },
        "traceability",
    )
    print(f"record: {len(produced) + len(figures)} artifacts -> {cc._relative(PROVENANCE)}")
    return 0


PHASES = {
    "preregister": run_preregister,
    "adapt": run_adapt,
    "results": run_results,
    "prefix": run_prefix,
    "gap": run_gap,
    "mechanism": run_mechanism,
    "acquisition": run_acquisition,
    "sgv6": run_sgv6,
    "oracle": run_oracle,
    "easyocr": run_easyocr,
    "negative": run_negative,
    "stats": run_stats,
    "figures": run_figures,
    "decide": run_decide,
    "record": run_record,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    chosen = [name for name in PHASES if getattr(args, name)]
    if not chosen:
        parser.print_help()
        return 1
    for name in chosen:
        status = PHASES[name]()
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
