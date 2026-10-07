#!/usr/bin/env python3
"""SGV12: is the correction-risk RANKING the thing that fails to transfer?

SGV11 closed by measuring where the loss actually sits. Between a perfect accept set and what a
policy deploys on an unseen engine there are exactly two gaps, and SGV11 published both: with
the cut placed by an oracle on the block it is measured on, a re-ranking fitted on that block
reaches 0.8124 mean repair recall while the frozen SGV5 ranking reaches 0.5059, and the best
deployed policy reaches 0.4279. The score-transfer gap is 0.3065 and the decision-transfer gap
is 0.0779: the ranking is 3.9x the larger of the two. Every stage from SGV5 to SGV11 changed
what happens BELOW the score. This stage changes the score's objective and nothing else.

That makes SGV12 a ranking experiment and nothing else. The candidate universe, the labels, the
document partition, the fold construction and the feature representation are SGV5's, rebuilt
from its published record and asserted bit-for-bit identical to its published decision score
before anything is fitted. No feature is added. The decision rule below the score is frozen and
is the same call for every ranking in the stage -- the empirical controller on the source
engines' calibration rows, which is the call that produced SGV5's own published threshold -- so
a difference between two rankings in the same cell cannot be a difference in where the cut went.

The brief's hypothesis is that the objective is the problem: that pooled binary risk
classification learns an ordering which is stable on the engines it was fitted on and not on a
new one, and that an objective which optimises the ORDER, and optimises it against the worst
source environment rather than the average one, transfers further.

    SGV12-R1: a ranking objective fitted on source OCR engines with an explicit
    cross-environment robustness term produces a better held-out ordering of correction
    candidates than SGV5's pooled classification objective on unseen OCR engines, and that
    better ordering raises the achievable risk-controlled repair frontier.

The chain has three links and the stage is built so that each is measured separately, because
the interesting failure is the middle one: a ranking can improve and buy nothing.

    better held-out ranking  ->  higher frozen-ranking frontier  ->  better deployed repair

Eleven decisions fix what the numbers below can mean. Three FEASIBILITY PROBES informed them and
all three are disclosed in `design_record.json` under `feasibility_probe`: one rebuilt SGV5's
selected model on all four folds and measured the cost and the identity, one measured the
wall-clock cost of a pooled pairwise fit and a boosted rank fit on docTR's source rows, and one
DID read a held-out engine's outcome -- it fitted each objective once on the docTR fold and
reported held-out AUROC and pair accuracy. That third probe changed exactly one thing, the
boosted arm's capacity, which was below the classification control it is contrasted against in
ablation A. It changed no criterion, no epsilon, no hypothesis and no other constant, and it
already showed every proposed arm BELOW the frozen baseline on that fold; the criteria were kept
as written. `--rank` refits the probe cell and asserts it reproduces.

**1. Nothing upstream moves, and no feature is added.** SGV1's design matrix, SGV5's structural
block, SGV5's frozen per-fold selection, SGV10's published boundary table and SGV11's published
gap decomposition are read and hash-verified against the records their own stages wrote. SGV5's
selected model is rebuilt through SGV5's own `fit_reliability` and its decision score on the
evaluation rows is compared to the published `arm__sgv5` vector; a nonzero difference is a hard
failure. `features_added` is 0 in the design record and the leakage suite walks this file's
syntax tree to confirm that no column is constructed here.

**2. Every SGV12 model sees exactly the columns SGV5's selected model saw.** The representation
is fold-dependent because SGV5's selection is -- docTR and Tesseract selected the 96-column
Phase 3 matrix, EasyOCR and PaddleOCR the 128-column extended one -- and using each fold's own
selected column set is what makes "only the objective changed" true within a fold.

**3. The decision rule below the score is frozen and shared.** For every ranking, at every
epsilon, the threshold is `select_threshold(source calibration score, source calibration
harm, epsilon, controller='empirical')` -- the published controller, on the source engines'
labelled calibration rows, reading no target row. That is the identical call that produced the
threshold SGV5 published and SGV10 and SGV11 carried, so this stage's baseline arm reproduces
SGV5's published deployment exactly, and `--negative` asserts it does.

**4. Higher means safer.** Fixed once. SGV5's decision score is `benefit - lambda * harm` and
every objective here is written so that a beneficial candidate should score above a harmful one.
The leakage suite asserts the orientation on every arm rather than trusting the sign.

**5. The environments are `(engine, document shard)`.** The shard is `sha256(document_id) mod
4`: one global deterministic partition of the 710 documents, identical across engines and roles,
so a document is never split and a group always holds many documents. Group sizes are published
in `group_inventory.json`, and a group whose outcome variation is too small to define a ranking
metric is flagged there rather than dropped silently.

**6. A document IS a page in this corpus, so the brief's within-page family is implemented as
within-SITE.** CORD receipts are one image per document, so a "within-page" pair family would be
the within-document family under a second name. The finer local control this corpus actually
supports is the correction SITE -- the competing candidates proposed at one span, which is the
comparison an operator faces -- and SGV5's own ranker already used it. This is a deviation from
the brief's wording, taken because the repository's structure decides it, and it makes P2 a
strictly stronger control than the brief asked for rather than a weaker one.

**7. The pair budget is 60,000, allocated across strata in proportion to eligible pairs.** All
of the fit rows would afford 76 million beneficial-harmful pairs on some folds, so a budget is
required rather than optional. 60,000 is SGV5's own `RANKER_PAIRS`, so the pooled linear arm at
the global family is SGV5's published ranker construction with only the pairing rule changed.
The allocation is deterministic largest-remainder over strata and the draw is seeded from
content; `pair_inventory.json` records the eligible count, the drawn count and the coverage of
every family, and the budget is swept on SOURCE VALIDATION only.

**8. The pair family and the objective are selected by an inner leave-one-source-engine-out
replay, on the MEAN inner score and not the worst-case one.** The replay is SGV9's
`inner_fold`, reused rather than re-salted, so the inner folds are the same inner folds SGV9,
SGV10 and SGV11 used: the outer held-out engine appears nowhere, no DEVELOPMENT page appears
anywhere, and the CALIBRATION pages are split in half by content hash. Selecting on the mean
rather than the worst case is deliberate: a worst-case selection rule would make "the robust
objective wins" and "the robust selection rule wins" the same measurement. What a worst-case
selection rule would have chosen is reported as an ablation.

**9. Ablations E and F are orthogonal contrasts at fixed scope.** Locality is P1 and P2 against
P0 with the eligibility rule unrestricted; decision relevance is P4 against P0 with the scope
global in both. Running them as one axis would confound "local pairs help" with "operationally
weighted pairs help", which are different claims about different mechanisms.

**10. The ceiling is SGV10's published `oracle_eval` re-ranking.** Not a new one. It is the same
object SGV11 measured its ladder against, so the recovery fraction's denominator is exactly the
0.3065 SGV11 published and the two stages' decompositions are the same decomposition. SGV12
additionally fits its own oracle re-ranker on the evaluation block, reported beside it as a
second ceiling and used for nothing else.

**11. The primary family is twenty tests and the recovery threshold is 0.20.** Both were fixed
before any held-out SGV12 number existed: the selected method against SGV5's frozen ranking on
four unseen engines, on held-out AUROC, pairwise ranking accuracy, and risk-controlled repair
recall at three epsilons. Holm over the whole family, document-clustered paired bootstrap.

    --rank        verify the freeze, build pairs and groups, fit every ranking, write the scores
    --ranking     settings A and C: held-out ranking quality, and what it costs to change engine
    --blocks      settings B and D: within-engine block shift, and the hardest pre-defined block
    --frontier    the frozen decision rule: risk-controlled repair recall, coverage, harm
    --gap         the SGV11 ladder recomputed, and the score-transfer gap recovery fraction
    --ablation    the brief's eight ablations
    --negative    the brief's ten failure tests
    --easyocr     the pre-specified EasyOCR diagnostic
    --stats       the pre-registered family: document-clustered bootstrap and Holm
    --figures     the five required figures
    --decide      the machine-readable finding
    --record      provenance, the dependency audit, and the traceability index

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; no model, no threshold, no
hyperparameter and no selection reads a label from the held-out engine.
"""

from __future__ import annotations

import argparse
import hashlib
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
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.discrimination import average_precision, roc_auc
from ocr_risk.metrics.selective import aurc, risk_coverage_curve
from ocr_risk.risk.controller import select_threshold
from ocr_risk.stats.multiplicity import holm_bonferroni

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv12_robust_ranking_transfer"

SCORES = OUT / "ranking_scores.parquet"
DESIGN_RECORD = OUT / "design_record.json"
PAIR_INVENTORY = OUT / "pair_inventory.json"
GROUP_INVENTORY = OUT / "group_inventory.json"
MODEL_SELECTION = OUT / "model_selection.json"
RANKING_METRICS = OUT / "ranking_metrics.json"
BLOCK_SHIFT = OUT / "block_shift_metrics.json"
FRONTIER_RESULTS = OUT / "frontier_results.json"
GAP_DECOMPOSITION = OUT / "gap_decomposition.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
EASYOCR_ANALYSIS = OUT / "easyocr_analysis.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
DECISION = OUT / "research_decision.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

# Twelve stages now share one implementation of the endpoint, the frontier, the resampling
# engine and the interval. `tests/leakage/test_sgv12_robust_ranking_transfer.py` asserts the
# identity of every alias below, so "SGV12 measures what SGV11 measured" is checkable.
achievable_repair_recall = policy.repair_recall_at_bounded_harm
deployed_point = c5._deployed
risk_controlled_recall = s11.risk_controlled_recall
document_resamples = s9.document_resamples
document_multiplicities = s11.document_multiplicities
_interval = s9._interval
_calibration_half = s9._calibration_half
_stable_seed = s6._stable_seed
inner_fold = s9.inner_fold
build_fold = rl.build_fold

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
PRIMARY_KEY = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = pilot.BOOTSTRAP_SEED


class PhaseError(RuntimeError):
    """A freeze, representation, split, pairing, group or orientation invariant failed."""


def epsilon_key(epsilon: float) -> str:
    return f"epsilon_{int(epsilon * 100)}"


# ------------------------------------------------------------------ the five blocks

# SGV5's fold, named once. `fit` and `calibration` are the source engines; `pool` and
# `evaluation` are the held-out engine; `seen_eval` is the source engines on the same
# DEVELOPMENT pages the evaluation block is read from, which is what makes setting A a
# same-pages, different-engine contrast rather than a different-corpus one.
FIT = "fit"
CALIBRATION = "calibration"
SEEN_EVAL = "seen_eval"
EVALUATION = "evaluation"
POOL = "pool"
BLOCK_NAMES = (FIT, CALIBRATION, SEEN_EVAL, EVALUATION, POOL)

BLOCK_NOTES = {
    FIT: "source engines, TRAIN documents. Every model in this stage is fitted here, only here.",
    CALIBRATION: "source engines, CALIBRATION documents. Every threshold comes from here.",
    SEEN_EVAL: "source engines, DEVELOPMENT documents. Setting A: a seen engine on unseen pages.",
    EVALUATION: "held-out engine, DEVELOPMENT documents. Setting C, and the only headline block.",
    POOL: "held-out engine, TRAIN documents. Labels read for analysis only, never for fitting.",
}

# The four evaluation settings the brief separates, mapped onto the blocks that realise them.
SETTING_A = "A_in_distribution_documents"
SETTING_B = "B_within_engine_block_shift"
SETTING_C = "C_unseen_engine"
SETTING_D = "D_unseen_engine_hardest_block"
SETTINGS = (SETTING_A, SETTING_B, SETTING_C, SETTING_D)

SETTING_NOTES = {
    SETTING_A: (
        "Seen engines, unseen documents. The source engines' DEVELOPMENT rows. Answers whether "
        "the ranker generalises to new pages when the OCR process is one it was fitted on."
    ),
    SETTING_B: (
        "One engine, one document shard at a time. Measured on every engine, and reported as the "
        "worst shard and as the fit-block-to-held-out-block drop. For a source engine the fit "
        "block is in-sample by construction and the drop prices that optimism; for the held-out "
        "engine both blocks are out-of-sample and the drop is pure page composition."
    ),
    SETTING_C: (
        "Held-out engine, unseen documents. The primary cross-engine setting and the block every "
        "headline is read from."
    ),
    SETTING_D: (
        "Held-out engine, its hardest pre-defined document shard. Difficulty is fixed by the "
        "highest mean published SGV5 `p_harm` over the shard's evaluation rows: a frozen model "
        "output, computed before any SGV12 score existed, reading no label and no SGV12 result."
    ),
}


# ------------------------------------------------------------------ the method registry

SGV5_FROZEN = "mA_sgv5_frozen"
PAIR_LINEAR = "mB_pair_linear"
PAIR_BOOSTED = "mC_pair_boosted"
LISTWISE = "mD_listwise"
GROUP_DRO = "mE_group_dro"
CVAR_RANK = "mF_cvar_rank"
VARIANCE_REG = "mG_variance_penalty"

SELECTED = "sel_source_validated"

CLF_LINEAR = "cA_clf_linear"
CLF_BOOSTED = "cB_clf_boosted"
DRO_ENGINE_ONLY = "cC_dro_engine_only"
DRO_BLOCK_ONLY = "cD_dro_block_only"
DRO_SHUFFLED = "cG_dro_shuffled_groups"
ORACLE_RERANK = "oH_oracle_rerank"
CEILING = "ceiling_sgv10_oracle_eval"

# The brief's five proposed model families. Model A is the frozen baseline and is not proposed.
PROPOSED = (PAIR_LINEAR, PAIR_BOOSTED, LISTWISE, GROUP_DRO, CVAR_RANK, VARIANCE_REG)
BASELINES = (SGV5_FROZEN,)
CONTROLS = (CLF_LINEAR, CLF_BOOSTED, DRO_ENGINE_ONLY, DRO_BLOCK_ONLY, DRO_SHUFFLED)
ORACLES = (ORACLE_RERANK,)

METHOD_NOTES = {
    SGV5_FROZEN: (
        "Model A, the baseline and the reference every criterion is read against. SGV5's "
        "published decision score `benefit - lambda * harm` under the model class, "
        "representation and lambda SGV5's own inner selection froze. Rebuilt through SGV5's "
        "code and asserted equal to its published vector to the last bit. Not retrained."
    ),
    PAIR_LINEAR: (
        "Model B. A linear scorer trained on the RankNet pairwise logistic loss over "
        "beneficial-versus-harmful pairs. Same columns, same rows, same scaler as Model A; the "
        "objective is the only thing that differs from `cA_clf_linear`, which is what makes "
        "ablation A a contrast rather than a comparison of two different experiments."
    ),
    PAIR_BOOSTED: (
        "Model C. The same pairwise loss, descended by gradient boosting over depth-limited "
        "regression trees instead of solved over a linear scorer. Paired with `cB_clf_boosted`, "
        "which is the same tree capacity under the classification objective, so the objective "
        "effect and the capacity effect are separable."
    ),
    LISTWISE: (
        "Model D. ListNet top-one: a softmax over each document's candidate list, fitted against "
        "the target distribution the outcome labels induce. The ranking unit is the whole "
        "document rather than a pair, which is the listwise question the brief asks for."
    ),
    GROUP_DRO: (
        "Model E. The pairwise loss aggregated over environments by a temperature-controlled "
        "soft maximum instead of an average. Its gradient is exactly an exponential reweighting "
        "of the per-group gradients, and it nests the exact minimax objective as the temperature "
        "goes to zero. The temperature is chosen by the inner replay."
    ),
    CVAR_RANK: (
        "Model F. The pairwise loss aggregated by the conditional value at risk of the "
        "per-environment losses at level alpha, in its convex dual form. Nests the group mean "
        "exactly at alpha = 1. The level is chosen by the inner replay from the brief's grid."
    ),
    VARIANCE_REG: (
        "Model G, the optional one. The mean pairwise loss plus a penalty on the VARIANCE of the "
        "per-environment losses. Not generic domain invariance: the penalised quantity is the "
        "spread of the RANKING loss across environments, which is the stage's hypothesis written "
        "as a regulariser. Non-convex; started from the pooled solution and recorded as a local "
        "optimum, with the weight chosen by the inner replay."
    ),
    SELECTED: (
        "Not a model: the source-validated SELECTION PROCEDURE, carried as an arm so that every "
        "table reads one column. Per fold it is whichever objective and pair family the inner "
        "leave-one-source-engine-out replay chose, which is the object an operator could deploy. "
        "The per-fold choices are published in `model_selection.json`."
    ),
    CLF_LINEAR: (
        "Ablation A, linear half. SGV5's classification objective at linear capacity: the two "
        "logistic heads, isotonic calibration and the fold's own lambda, fitted through SGV5's "
        "`fit_reliability` on the identical rows and columns Model B sees."
    ),
    CLF_BOOSTED: (
        "Ablation A, boosted half. The same, with SGV5's gradient-boosted heads. For three of "
        "four folds this is SGV5's selected model class, so the contrast with Model C is the "
        "objective at the capacity the published baseline actually uses."
    ),
    DRO_ENGINE_ONLY: (
        "Ablation C. Model E with the environments collapsed to the three source ENGINES: does "
        "engine identity alone carry the robustness, or is the document shard doing work?"
    ),
    DRO_BLOCK_ONLY: (
        "Ablation D. Model E with the environments collapsed to the four document SHARDS, engine "
        "identity discarded. The direct test of whether SGV11's within-engine block instability "
        "is a thing a training objective can address."
    ),
    DRO_SHUFFLED: (
        "Ablation G, a control and not a method. Model E with the group labels permuted while "
        "the group sizes are preserved. If it matches Model E, the environment structure is "
        "decorative and the gain is a regularisation effect."
    ),
    ORACLE_RERANK: (
        "Ablation H, an ORACLE. The selected objective fitted on the EVALUATION block's own "
        "labels. Analysis only: it appears in no selection, no criterion and no headline, and "
        "`--negative` asserts that no deployed arm reads it."
    ),
    CEILING: (
        "The ceiling of the gap ladder, read from SGV10's published table rather than refitted: "
        "SGV10's `oracle_eval` re-ranking of the evaluation block. The same object SGV11 used, "
        "so the recovery fraction's denominator is exactly the gap SGV11 published."
    ),
}


# ------------------------------------------------------------------ the pair families

P0_GLOBAL = "p0_global"
P1_DOCUMENT = "p1_document"
P2_SITE = "p2_site"
P3_MATCHED = "p3_matched_type"
P4_DECISION = "p4_decision_relevant"
PAIR_FAMILIES = (P0_GLOBAL, P1_DOCUMENT, P2_SITE, P3_MATCHED, P4_DECISION)

# P0 is the brief's weak baseline and the fixed scope both orthogonal ablations are read
# against. The selection in `--rank` chooses among the four structured families only.
SELECTABLE_FAMILIES = (P1_DOCUMENT, P2_SITE, P3_MATCHED, P4_DECISION)

PAIR_NOTES = {
    P0_GLOBAL: (
        "Family P0, the weak baseline. A beneficial row and a harmful row drawn from anywhere in "
        "the fit block, across documents and across engines. Vulnerable by construction to any "
        "shortcut that separates documents or engines rather than candidates."
    ),
    P1_DOCUMENT: (
        "Family P1. Both members read from the same document by the same engine, so no "
        "document-level difficulty and no engine identity can separate a pair."
    ),
    P2_SITE: (
        "Family P2. Both members proposed at the same correction SITE. In this corpus a document "
        "is a page, so this -- not a within-page family -- is the finer local control, and it is "
        "the comparison an operator actually faces: of the edits proposed HERE, which is safe."
    ),
    P3_MATCHED: (
        "Family P3. Both members share an observable structure bucket -- edit operation, original "
        "span-length band and normalised-confidence band -- drawn globally within the bucket. The "
        "buckets are fixed cut points on frozen columns, never quantiles of the fit sample, so "
        "the bucketing is not itself a fitted statistic."
    ),
    P4_DECISION: (
        "Family P4. Global scope, restricted to pairs with at least one member inside the SOURCE "
        "deployment-relevant prefix -- the top of the frozen SGV5 ranking at the acceptance share "
        "the published threshold takes on the source calibration rows. Reads no target row."
    ),
}


# ------------------------------------------------------------------ the environments

ENGINE_SHARD = "engine_shard"
ENGINE_ONLY = "engine_only"
SHARD_ONLY = "shard_only"
SHUFFLED = "shuffled_engine_shard"
GROUPINGS = (ENGINE_SHARD, ENGINE_ONLY, SHARD_ONLY, SHUFFLED)

N_SHARDS = 4
SHARD_SALT = "sgv12-document-shard"

GROUP_NOTES = {
    ENGINE_SHARD: "The primary environment: the pair of an OCR engine and a document shard.",
    ENGINE_ONLY: "Ablation C: the OCR engine alone, four environments per corpus, three per fold.",
    SHARD_ONLY: "Ablation D: the document shard alone, engine identity discarded.",
    SHUFFLED: "Ablation G: the primary labels permuted with the group sizes preserved.",
}


# ------------------------------------------------------------------ constants

# The pair budget. 60,000 is SGV5's own RANKER_PAIRS, so the pooled linear arm at the global
# family is SGV5's published ranker construction with only the pairing rule changed. The sweep
# is run on SOURCE VALIDATION only and is reported in `--ablation`.
PAIR_BUDGET = 60000
PAIR_BUDGET_SWEEP = (15000, 30000, 60000, 120000)

# The linear scorer's ridge coefficient, applied identically to every linear objective so that
# no arm is regularised differently from another. `tests/leakage` cross-checks the pooled
# pairwise solver against sklearn's LogisticRegression at the matching penalty.
L2_PENALTY = 1e-4
LBFGS_MAXITER = 500

# Model C's capacity, fixed rather than selected, and MATCHED to the classification control it
# is contrasted against. SGV5 fits `HistGradientBoostingClassifier(max_iter=200)`, whose defaults
# are a learning rate of 0.1, at most 31 leaves and at least 20 samples in a leaf. Model C is
# 200 rounds of a depth-5 regression tree -- at most 32 leaves -- at the same rate and the same
# leaf floor, which is as close as sklearn's two tree implementations come to each other. The
# first capacity tried was 100 rounds at depth 4; the feasibility probe showed it was strictly
# below the control's, which would have made ablation A a capacity contrast rather than an
# objective contrast, and the record discloses both.
BOOST_ROUNDS = 200
BOOST_RATE = 0.1
BOOST_DEPTH = 5
BOOST_MIN_LEAF = 20
PROBE_BOOST_ROUNDS = 100
PROBE_BOOST_DEPTH = 4

# The hyperparameter grids the inner replay chooses from. All three were fixed before any SGV12
# held-out number was computed, and `--ablation` reports the endpoint under every value.
DRO_TEMPERATURES = (0.01, 0.05, 0.20)
CVAR_ALPHAS = (0.10, 0.20, 0.30)
VARIANCE_WEIGHTS = (0.1, 1.0)

HYPERPARAMETERS: dict[str, tuple[str, tuple[float, ...]]] = {
    GROUP_DRO: ("dro_temperature", DRO_TEMPERATURES),
    CVAR_RANK: ("cvar_alpha", CVAR_ALPHAS),
    VARIANCE_REG: ("variance_weight", VARIANCE_WEIGHTS),
}

# P3's buckets. Fixed cut points on frozen columns, never quantiles of the sample.
LENGTH_EDGES = (0.0, 3.0, 6.0, 10.0)
CONFIDENCE_EDGES = (0.25, 0.5, 0.75)

# A group must hold this many rows of each outcome class before a ranking metric is read from
# it. Below the bar the metric is reported as unavailable and the group is flagged, never
# dropped without a record.
MIN_GROUP_ROWS = 30
MIN_GROUP_CLASS = 5

# The pre-registered recovery threshold for criterion 5, fixed before any held-out number.
RECOVERY_THRESHOLD = 0.20
# Criteria 2, 3 and 4 each require this many of the four unseen engines.
ENGINE_BAR = 3

RANDOM_SEED = 20260905


# ------------------------------------------------------------------ the pre-registration

PRE_REGISTRATION = {
    "hypothesis_id": "SGV12-R1",
    "statement": (
        "A ranking objective fitted on source OCR engines with an explicit cross-environment "
        "robustness term produces a better held-out ordering of correction candidates than "
        "SGV5's pooled classification objective on unseen OCR engines, and that better ordering "
        "raises the achievable risk-controlled repair frontier."
    ),
    "research_questions": {
        "SGV12-RQ1": (
            "Can direct ranking optimisation improve correction-risk ordering on unseen OCR "
            "engines compared with the SGV5 binary risk model?"
        ),
        "SGV12-RQ2": (
            "Can group-robust ranking reduce ranking degradation across document blocks within "
            "the same OCR engine?"
        ),
        "SGV12-RQ3": (
            "Does improved held-out ranking actually raise the achievable risk-controlled repair "
            "frontier? The chain is: better held-out ranking -> higher frozen-ranking frontier "
            "-> better risk-controlled repair. If the ranking improves and the frontier does "
            "not, the hypothesis is not supported."
        ),
    },
    "sub_hypotheses": {
        "SGV12-H1": (
            "A pairwise or listwise risk-ranking objective outperforms SGV5's binary "
            "classification objective on unseen-engine ranking quality."
        ),
        "SGV12-H2": (
            "A group-aware ranking objective over (engine, document block) reduces worst-group "
            "ranking degradation relative to pooled training."
        ),
        "SGV12-H3": (
            "Ranking improvement translates into a higher achievable repair-recall frontier "
            "under fixed harmful-edit constraints."
        ),
        "SGV12-H4": (
            "The improvement occurs across multiple unseen engines and is not explained by one "
            "engine, one document block, or a trivial global ordering shortcut."
        ),
    },
    "primary_endpoint": (
        "Two, and the verdict needs both. Mechanistic: held-out AUROC and pairwise ranking "
        "accuracy of the selected method against SGV5's frozen ranking on the unseen engine. "
        "Operational: risk-controlled repair recall -- the repair recall of the accepted set if "
        "its realised harm rate is within epsilon and zero otherwise -- at the frozen decision "
        "rule, at epsilon = 0.10."
    ),
    "primary_family": (
        "The source-validated selected method paired against SGV5's frozen ranking on 4 unseen "
        "engines, on held-out AUROC (4 tests), pairwise ranking accuracy (4 tests) and "
        "risk-controlled repair recall at epsilon in {0.05, 0.10, 0.20} (12 tests): 20 tests, "
        "Holm-adjusted over the whole family, document-clustered paired bootstrap."
    ),
    "criteria": {
        "1_ranking_improvement": (
            "The selected method significantly improves held-out ranking quality over SGV5 "
            "after the pre-registered Holm correction."
        ),
        "2_cross_engine_breadth": (
            f"Held-out ranking quality improves in the favourable direction on at least "
            f"{ENGINE_BAR} of 4 unseen engines."
        ),
        "3_operational_frontier": (
            f"Risk-controlled repair recall improves over SGV5 on at least {ENGINE_BAR} of 4 "
            f"unseen engines at epsilon = {PRIMARY_EPSILON}, without exceeding the realised harm "
            "bound -- which the risk-controlled endpoint enforces by construction."
        ),
        "4_worst_group_robustness": (
            f"Worst-document-shard ranking or frontier improves relative to the pooled ranking "
            f"on at least {ENGINE_BAR} of 4 engines."
        ),
        "5_gap_recovery": (
            f"The selected method recovers at least {RECOVERY_THRESHOLD:.2f} of SGV11's "
            "score-transfer gap on the pooled primary analysis."
        ),
    },
    "verdict_rule": (
        "SUPPORTED only if all five criteria hold. Fewer than five is NOT SUPPORTED. The rule "
        "and the recovery threshold were fixed before any held-out SGV12 number was computed and "
        "are not revised afterwards."
    ),
    "what_is_not_claimed": (
        "This stage does not build an OCR correction model and claims no priority over the prior "
        "art. It is an "
        "evaluation of whether the remaining cross-engine safety gap is a ranking-objective "
        "problem, and a measurement of how much of it a source-only robust ranking objective can "
        "close."
    ),
}

DECISIONS = {
    "1_freeze": (
        "Nothing upstream moves and no feature is added. SGV5's selected model is rebuilt "
        "through SGV5's own code and its decision score is asserted equal to the published "
        "`arm__sgv5` vector on every evaluation row of every fold; a nonzero difference is a "
        "hard failure. `features_added` is 0 and the leakage suite walks this file's syntax tree "
        "to confirm no column is constructed here."
    ),
    "2_representation": (
        "Every SGV12 model sees exactly the columns SGV5's selected model saw on that fold: the "
        "96-column Phase 3 matrix for docTR and Tesseract, the 128-column extended matrix for "
        "EasyOCR and PaddleOCR. Within a fold, only the objective differs."
    ),
    "3_frozen_decision_rule": (
        "One decision rule for every ranking at every epsilon: the empirical controller on the "
        "source engines' labelled calibration rows. It is the identical call that produced "
        "SGV5's published threshold, so the baseline arm reproduces SGV5's published deployment "
        "and `--negative` asserts that it does. The Learn-then-Test controller is reported "
        "beside it for every ranking, never for one."
    ),
    "4_orientation": (
        "Higher means safer, fixed once. Every objective is written so a beneficial candidate "
        "should outrank a harmful one, and the leakage suite asserts the orientation per arm."
    ),
    "5_environments": (
        f"The environment is (engine, document shard) with the shard a sha256 of the document "
        f"identifier modulo {N_SHARDS}: one global partition, identical across engines and roles, "
        "so a document is never split across environments. Sizes are published and a group with "
        "too little outcome variation to define a ranking metric is flagged, never dropped."
    ),
    "6_within_page_is_within_site": (
        "A document is a page in this corpus, so the brief's within-page pair family would "
        "duplicate the within-document one. P2 is implemented as within-SITE instead -- the "
        "competing candidates at one span -- which is a strictly stronger local control and the "
        "unit SGV5's own ranker used. Recorded as a deviation from the brief's wording."
    ),
    "7_pair_budget": (
        f"The pair budget is {PAIR_BUDGET}, SGV5's own RANKER_PAIRS, allocated across strata by "
        "deterministic largest-remainder in proportion to eligible pairs and drawn from a "
        "content-derived seed. Eligible, drawn and coverage counts are published per family, and "
        "the budget is swept on source validation only."
    ),
    "8_selection_on_the_mean": (
        "The pair family and the objective are chosen by SGV9's inner leave-one-source-engine-out "
        "replay -- the outer held-out engine appears nowhere, no DEVELOPMENT page appears "
        "anywhere -- on the MEAN inner pairwise accuracy. Not the worst case: a worst-case "
        "selection rule would make 'the robust objective wins' and 'the robust selection rule "
        "wins' the same measurement. What a worst-case rule would have chosen is an ablation."
    ),
    "9_orthogonal_ablations": (
        "Locality (E) and decision relevance (F) are separate axes read at a fixed scope: E is "
        "P1 and P2 against P0, F is P4 against P0. Running them as one axis would confound two "
        "different claims about two different mechanisms."
    ),
    "10_ceiling_is_sgv10s": (
        "The gap ladder's ceiling is SGV10's published `oracle_eval` re-ranking, read and not "
        "refitted, so the recovery fraction's denominator is exactly the 0.3065 SGV11 published "
        "and the two stages' ladders are the same ladder. SGV12's own oracle re-ranker is "
        "reported beside it and used for nothing else."
    ),
    "11_family_and_threshold": (
        f"The primary family is 20 tests and the recovery threshold is {RECOVERY_THRESHOLD:.2f}. "
        "Both were fixed before any held-out SGV12 number existed and neither is revised."
    ),
}

FEASIBILITY_PROBE = {
    "purpose": (
        "Two probes were run before this pre-registration was written, to establish that the "
        "stage is buildable at all. Neither read a held-out engine's outcome and neither "
        "computed any endpoint this stage reports. They are disclosed because they informed "
        "decisions 2 and 7."
    ),
    "probe_1_rebuild": {
        "what": (
            "Rebuilt SGV5's selected model on all four folds through `c5.fit_reliability` and "
            "compared the decision score on the evaluation rows to the published `arm__sgv5`."
        ),
        "measured": {
            "max_abs_difference_per_fold": 0.0,
            "selected_representation": {
                "doctr": "phase3",
                "easyocr": "extended",
                "paddleocr": "extended",
                "tesseract": "phase3",
            },
            "wall_clock_seconds_total": 61,
        },
        "informed": "decision 2, that the representation is SGV5's per-fold selected column set",
    },
    "probe_2_cost": {
        "what": (
            "One pooled pairwise linear fit and one 50-round boosted rank fit on docTR's SOURCE "
            "fit rows, to price the model roster. Source-side training loss only."
        ),
        "measured": {
            "fit_rows": 29251,
            "linear_lbfgs_seconds": 2.8,
            "boosted_50_rounds_seconds": 9.5,
            "pooled_pairwise_training_loss": 0.1004,
        },
        "informed": (
            "decision 7, that a pair budget is required rather than optional, and the size of "
            "the model roster the stage can afford to fit under an inner replay"
        ),
    },
    "probe_3_capacity": {
        "what": (
            "One fit of each objective at pair family P1 on the docTR fold, scored on the docTR "
            "EVALUATION block. This probe DID read a held-out engine's outcome, on one of the "
            "four folds, and is disclosed in full for that reason. It was run to check that the "
            "boosted arm's capacity matched the classification control it is contrasted against "
            "in ablation A, and it showed that it did not."
        ),
        "measured_auroc_safe": {
            "mA_sgv5_frozen": 0.9898,
            "mB_pair_linear": 0.9684,
            "mC_pair_boosted_at_100_rounds_depth_4": 0.8725,
            "mD_listwise": 0.8891,
            "mE_group_dro_at_0.05": 0.9714,
            "mF_cvar_rank_at_0.2": 0.9596,
            "mG_variance_penalty_at_1.0": 0.9731,
        },
        "measured_pair_accuracy": {
            "mA_sgv5_frozen": 0.9578,
            "mB_pair_linear": 0.9058,
            "mC_pair_boosted_at_100_rounds_depth_4": 0.8648,
            "mD_listwise": 0.8933,
            "mE_group_dro_at_0.05": 0.9104,
            "mF_cvar_rank_at_0.2": 0.8907,
            "mG_variance_penalty_at_1.0": 0.9156,
        },
        "printed_decimals": 4,
        "informed": (
            "the boosted arm's capacity only. 100 rounds of a depth-4 tree is strictly below "
            "`HistGradientBoostingClassifier(max_iter=200)`, so ablation A would have contrasted "
            "a weaker hypothesis class against a stronger one and called the difference an "
            "objective effect. The capacity was raised to 200 rounds of a depth-5 tree with a "
            "leaf floor of 20 to match the control. No criterion, no hypothesis, no epsilon and "
            "no other constant was changed -- and the probe already shows every proposed arm "
            "BELOW the frozen baseline on this fold, so the criteria were kept as written."
        ),
        "reproduction": (
            "`--rank` refits the probe cell on the docTR fold and records whether every number "
            "above comes back to the last printed decimal. A miss is a hard failure for every "
            "arm whose implementation has not changed since the probe ran. Two refits are "
            "deliberately NOT the shipping configuration, because the probe was not run with "
            "it: the boosted arm is refitted at the probe's discarded capacity, and the "
            "variance-penalised arm from a cold start rather than from the pooled solution the "
            "shipping arm warm-starts from. That objective is not convex and the two starts "
            "reach different local optima -- 0.9156 cold against 0.9155 warm on this cell -- "
            "which is itself the reason the shipping arm fixes its start."
        ),
        "arms_whose_implementation_changed_after_the_probe": {
            "mC_pair_boosted_at_100_rounds_depth_4": (
                "capacity only, and the probe capacity is refitted here, so the disclosed number "
                "is still checked exactly."
            ),
            "mF_cvar_rank_at_0.2": (
                "the CVaR dual's inner variable. The probe ran with eta set to the empirical "
                "(1 - alpha) quantile of the environment losses, which minimises the UNSMOOTHED "
                "dual; the analytic gradient the solver was given assumes the eta that minimises "
                "the SMOOTHED one, and the leakage suite's finite-difference check found the two "
                "disagreeing in the third decimal. The shipping solver drives eta to the smoothed "
                "optimum by bisection. The probe's number therefore CANNOT be reproduced without "
                "shipping the superseded solver, so it is recorded, compared, and not asserted -- "
                "the gap between the two is published below rather than hidden by a tolerance."
            ),
        },
    },
}


# ------------------------------------------------------------------ verifying the freeze


def _hash_against(path: Path, record: Path, field: str = "artifacts") -> dict[str, Any]:
    """One published artifact, checked against the hash the stage that produced it recorded."""
    if not path.is_file():
        raise PhaseError(f"{cc._relative(path)} is missing; SGV12 cannot make an upstream table")
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
    """Every upstream artifact this stage reads, checked before a single model is fitted.

    Hash consistency is the cheap half. The expensive half is that the candidate universe, the
    labels and the document partition SGV12 fits on are asserted to be the ones SGV5 fitted on:
    the identifiers align with SGV1's design matrix row for row, and the rebuilt SGV5 model's
    decision score is compared to SGV5's published vector in `--rank`, per fold, to the last bit.
    """
    checked = [
        _hash_against(dg.DESIGN_MATRIX, dg.FIT_RECORD),
        _hash_against(c5.FEATURES, c5.FIT_RECORD),
        _hash_against(c5.PREDICTIONS, c5.FIT_RECORD),
        _hash_against(c5.POLICY_SELECTION, c5.FIT_RECORD),
        _hash_against(s9.SCORES, s9.PROVENANCE),
        _hash_against(s10.SCORES, s10.PROVENANCE),
        _hash_against(s10.DESIGN_RECORD, s10.PROVENANCE),
        _hash_against(s11.DECISION_TRANSFER, s11.PROVENANCE),
        _hash_against(s11.SCORES, s11.PROVENANCE),
    ]
    design = pd.read_parquet(
        dg.DESIGN_MATRIX, columns=["candidate_id", "document_id", "engine_id", "role"]
    )
    structural = pd.read_parquet(c5.FEATURES, columns=["candidate_id"])
    if list(structural["candidate_id"].astype(str)) != list(design["candidate_id"].astype(str)):
        raise PhaseError("SGV5's structural block is not row-aligned with SGV1's design matrix")

    table = pd.read_parquet(s10.SCORES, columns=["candidate_id", "document_id", "block"])
    known = design.set_index("candidate_id")
    identifiers = table["candidate_id"].astype(str)
    unknown = int((~identifiers.isin(known.index)).sum())
    if unknown:
        raise PhaseError(f"{unknown} SGV10 rows carry a candidate_id SGV1's design does not know")
    joined = known.loc[identifiers]
    mismatched = int(
        (joined["document_id"].to_numpy(str) != table["document_id"].to_numpy(str)).sum()
    )
    if mismatched:
        raise PhaseError(f"{mismatched} SGV10 rows disagree with SGV1's design on the document")

    published_ladder = cc._read_json(s11.DECISION_TRANSFER)["aggregate"][
        f"{s11.RANK_SGV5}|{PRIMARY_KEY}"
    ]
    return {
        "artifacts": checked,
        "candidate_id_alignment": {
            "design_rows": len(design),
            "structural_rows": len(structural),
            "sgv10_rows": len(table),
            "unknown_candidate_ids": unknown,
            "document_disagreements": mismatched,
        },
        "sgv11_published_ladder": {
            "ceiling_frontier": float(published_ladder["mean_ceiling_frontier"]),
            "frozen_frontier": float(published_ladder["mean_frozen_frontier"]),
            "score_transfer_gap": float(published_ladder["mean_score_transfer_gap"]),
            "note": (
                "the denominator of this stage's recovery fraction, read from SGV11's own "
                "artifact rather than recomputed, and re-derived independently in `--gap`"
            ),
        },
        "features_added_by_this_stage": 0,
        "note": (
            "SGV12 adds no feature and changes no label. What it fits is a scoring function "
            "over the columns SGV5's selected model already used, on the rows SGV5 already "
            "fitted on, under a different objective."
        ),
    }


# ------------------------------------------------------------------ shards and environments


def document_shard(document: str) -> int:
    """A deterministic shard for one document, derived from content and never from `hash()`.

    `hash()` moves with PYTHONHASHSEED, so a group built from it would differ between processes
    and the group inventory would not be reproducible. The salt is this stage's, so the
    partition is a fresh draw of the same construction SGV8 and SGV9 used rather than a reuse
    of their split under a second name.
    """
    digest = hashlib.sha256(f"{SHARD_SALT}|{document}".encode()).digest()
    return digest[0] % N_SHARDS


def shard_of(documents: np.ndarray) -> np.ndarray:
    return np.array([document_shard(str(name)) for name in documents.tolist()], dtype=np.int8)


def group_labels(engines: np.ndarray, shards: np.ndarray, grouping: str) -> np.ndarray:
    """The environment label of every row under one grouping. Strings, so a record can read them.

    The shuffled grouping is built in `permuted_groups` rather than here: it needs a seed and a
    size-preserving permutation, and mixing that into the definition would make the control look
    like a fifth environment definition rather than the same one with its labels destroyed.
    """
    if grouping == ENGINE_SHARD:
        return np.array(
            [f"{e}|s{int(s)}" for e, s in zip(engines.tolist(), shards.tolist())], dtype=object
        )
    if grouping == ENGINE_ONLY:
        return np.asarray(engines, dtype=object)
    if grouping == SHARD_ONLY:
        return np.array([f"s{int(s)}" for s in shards.tolist()], dtype=object)
    raise PhaseError(f"unknown grouping {grouping!r}")


def permuted_groups(labels: np.ndarray, seed: int) -> np.ndarray:
    """Ablation G: the same group sizes, assigned to different rows.

    A permutation of the label vector preserves every group's size exactly while destroying the
    correspondence between a row and its environment, which is the only thing the control has to
    do. Reassigning rows to groups by a fresh random draw would also change the size profile and
    would confound 'the structure is decorative' with 'the group sizes changed'.
    """
    rng = np.random.default_rng(seed)
    return np.asarray(labels, dtype=object)[rng.permutation(labels.size)]


# ------------------------------------------------------------------ what a fit may read


@dataclass(frozen=True, slots=True)
class SourceFitView:
    """Everything a ranking objective is allowed to read while it is being fitted.

    Every field belongs to the SOURCE engines' TRAIN rows. There is no target field of any kind
    and no evaluation field of any kind, and the leakage suite walks every fitting function's
    syntax tree to confirm that none of them names another source of rows. `decision_relevant`
    is a source-side mask -- the top of the frozen SGV5 ranking at the acceptance share the
    published threshold takes on the SOURCE calibration rows -- and reads no target row either.
    """

    features: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    documents: np.ndarray
    sites: np.ndarray
    engines: np.ndarray
    shards: np.ndarray
    bucket: np.ndarray
    decision_relevant: np.ndarray

    @property
    def size(self) -> int:
        return int(self.features.shape[0])

    @property
    def width(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True, slots=True)
class Pairs:
    """One drawn pair set: which rows, which environment, and what it cost to draw them."""

    family: str
    positive: np.ndarray
    negative: np.ndarray
    group: np.ndarray
    eligible: int
    drawn: int
    strata: int
    strata_used: int
    cross_group: int
    exhaustive: bool

    @property
    def size(self) -> int:
        return int(self.positive.size)

    def inventory(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "eligible_pairs": int(self.eligible),
            "drawn_pairs": int(self.drawn),
            "coverage": float(self.drawn / self.eligible) if self.eligible else 0.0,
            "strata": int(self.strata),
            "strata_contributing": int(self.strata_used),
            "cross_environment_pairs": int(self.cross_group),
            "enumerated_exhaustively": bool(self.exhaustive),
        }


def matched_bucket(names: tuple[str, ...], matrix: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Family P3's observable structure bucket, read from frozen columns and never fitted.

    Three coordinates, all of them already in SGV1's design matrix: which edit operation the
    generator proposed, how long the original span was, and where the engine's own normalised
    confidence sat. The bands are FIXED cut points, not quantiles of this sample -- a quantile
    would be a statistic estimated on the fit rows and would make the bucketing itself a fitted
    transformer, which `.claude/rules/experiment-leakage.md` treats as a leak when it moves.
    """
    position = {name: i for i, name in enumerate(names)}
    operations = sorted(n for n in names if n.startswith("prov_operation_"))
    if not operations:
        raise PhaseError("the design matrix carries no prov_operation_* column")
    block = matrix[np.ix_(index, [position[name] for name in operations])]
    operation = np.argmax(block, axis=1)
    length = np.digitize(matrix[index, position["text_len_original"]], np.array(LENGTH_EDGES))
    confidence = np.digitize(matrix[index, position["conf_normalized"]], np.array(CONFIDENCE_EDGES))
    missing = matrix[index, position["conf_normalized_missing"]] > 0.5
    confidence = np.where(missing, len(CONFIDENCE_EDGES) + 1, confidence)
    return np.asarray(
        operation * 100 + length * 10 + confidence,
        dtype=np.int64,
    )


def _strata_for(family: str, view: SourceFitView) -> list[tuple[np.ndarray, np.ndarray]]:
    """The (beneficial pool, harmful pool) blocks a family may draw a pair from.

    A family is entirely a statement about which rows may be compared, so it is entirely a
    statement about how the eligible set is partitioned. Everything downstream -- the budget
    allocation, the draw, the environment assignment -- is shared, which is what makes the five
    families comparable rather than five separate constructions.
    """
    beneficial = np.flatnonzero(view.beneficial)
    harmful = np.flatnonzero(view.harmful)
    if beneficial.size == 0 or harmful.size == 0:
        return []
    if family == P0_GLOBAL:
        return [(beneficial, harmful)]
    if family == P4_DECISION:
        relevant = np.asarray(view.decision_relevant, dtype=bool)
        positive_relevant = beneficial[relevant[beneficial]]
        negative_relevant = harmful[relevant[harmful]]
        positive_other = beneficial[~relevant[beneficial]]
        blocks = [(positive_relevant, harmful), (positive_other, negative_relevant)]
        return [(p, n) for p, n in blocks if p.size and n.size]
    if family == P1_DOCUMENT:
        keys = np.array(
            [
                f"{e}|{d}"
                for e, d in zip(view.engines.tolist(), view.documents.tolist(), strict=True)
            ],
            dtype=object,
        )
    elif family == P2_SITE:
        keys = np.asarray(view.sites, dtype=object)
    elif family == P3_MATCHED:
        keys = np.asarray(view.bucket, dtype=object)
    else:
        raise PhaseError(f"unknown pair family {family!r}")
    positives: dict[Any, list[int]] = {}
    negatives: dict[Any, list[int]] = {}
    for row in beneficial.tolist():
        positives.setdefault(keys[row], []).append(row)
    for row in harmful.tolist():
        negatives.setdefault(keys[row], []).append(row)
    shared = sorted(set(positives) & set(negatives), key=str)
    return [
        (np.array(positives[key], dtype=int), np.array(negatives[key], dtype=int)) for key in shared
    ]


def _allocate(eligible: np.ndarray, budget: int) -> np.ndarray:
    """Largest-remainder allocation of a budget across strata, capped at what each can supply.

    Proportional-to-eligible is the allocation that leaves the drawn set closest to a uniform
    sample of the eligible set, which is what makes the budget a sampling decision rather than a
    reweighting one. Largest remainder rather than rounding, so the allocation is deterministic
    and sums to the budget exactly; the cap and one redistribution pass stop a small stratum
    from being handed more pairs than it contains.
    """
    counts = np.asarray(eligible, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(counts.size, dtype=np.int64)
    if total <= budget:
        return counts.copy()
    exact = counts.astype(float) * (budget / total)
    allocation = np.floor(exact).astype(np.int64)
    remainder = budget - int(allocation.sum())
    if remainder > 0:
        order = np.lexsort((np.arange(counts.size), -(exact - allocation)))
        allocation[order[:remainder]] += 1
    allocation = np.minimum(allocation, counts)
    leftover = budget - int(allocation.sum())
    if leftover > 0:
        room = counts - allocation
        order = np.lexsort((np.arange(counts.size), -room))
        for index in order:
            if leftover <= 0:
                break
            take = int(min(leftover, room[index]))
            allocation[index] += take
            leftover -= take
    return allocation


def build_pairs(family: str, view: SourceFitView, grouping: str, seed: int, budget: int) -> Pairs:
    """Draw one family's pairs under a fixed budget. Deterministic in the seed and the data.

    A stratum whose whole product fits inside its allocation is ENUMERATED rather than sampled,
    so the small local families -- and P2 in particular, where this corpus offers only a few
    thousand competing-candidate comparisons -- are exact rather than a noisy draw of themselves.
    Larger strata are sampled with replacement from the product, which is uniform over the
    stratum's eligible pairs.
    """
    blocks = _strata_for(family, view)
    if not blocks:
        empty = np.array([], dtype=int)
        return Pairs(family, empty, empty, np.array([], dtype=object), 0, 0, 0, 0, 0, True)
    eligible = np.array([p.size * n.size for p, n in blocks], dtype=np.int64)
    allocation = _allocate(eligible, budget)
    rng = np.random.default_rng(seed)
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    exhaustive = True
    for (pool_p, pool_n), take, whole in zip(blocks, allocation, eligible, strict=True):
        if take <= 0:
            continue
        if take >= whole:
            grid_p, grid_n = np.meshgrid(pool_p, pool_n, indexing="ij")
            positives.append(grid_p.ravel())
            negatives.append(grid_n.ravel())
            continue
        exhaustive = False
        positives.append(rng.choice(pool_p, size=int(take), replace=True))
        negatives.append(rng.choice(pool_n, size=int(take), replace=True))
    if not positives:
        empty = np.array([], dtype=int)
        return Pairs(
            family,
            empty,
            empty,
            np.array([], dtype=object),
            int(eligible.sum()),
            0,
            int(eligible.size),
            0,
            0,
            True,
        )
    positive = np.concatenate(positives)
    negative = np.concatenate(negatives)
    labels = group_labels(view.engines, view.shards, grouping)
    group = labels[positive]
    return Pairs(
        family=family,
        positive=positive,
        negative=negative,
        group=group,
        eligible=int(eligible.sum()),
        drawn=int(positive.size),
        strata=int(eligible.size),
        strata_used=int((allocation > 0).sum()),
        cross_group=int((labels[positive] != labels[negative]).sum()),
        exhaustive=exhaustive,
    )


# ------------------------------------------------------------------ the ranking objectives

POOLED = "pooled"
GROUP_MEAN = "group_mean"
DRO_SOFTMAX = "dro_softmax"
CVAR = "cvar"
VARIANCE = "variance"

# The hinge in the CVaR dual is smoothed at this width so the objective is once differentiable
# and L-BFGS is solving the function it is given rather than a subgradient of a kink. The width
# is three orders of magnitude below a typical per-environment loss, so the smoothing moves the
# value far less than the optimiser's own tolerance.
CVAR_SMOOTHING = 1e-3


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """The logistic function without an overflow.

    Branching with `np.where` would evaluate `exp(x)` on the positive side too, and the CVaR
    dual divides by a smoothing width of 1e-3, so the argument routinely reaches four figures.
    Masked assignment evaluates each branch only where it is finite.
    """
    values = np.asarray(x, dtype=float)
    out = np.empty(values.shape, dtype=float)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    lower = np.exp(values[~positive])
    out[~positive] = lower / (1.0 + lower)
    return out


@dataclass(frozen=True, slots=True)
class Environments:
    """Pairs bucketed into environments, with everything the aggregators need precomputed."""

    codes: np.ndarray
    counts: np.ndarray
    names: tuple[str, ...]

    @property
    def size(self) -> int:
        return int(self.counts.size)


def environments_of(pairs: Pairs, pooled: bool) -> Environments:
    """One environment holding every pair, or one per distinct group label.

    The pooled aggregator is not a special case in the code: it is this function returning a
    single environment, so `mB_pair_linear` and `mE_group_dro` run through the same loss, the
    same gradient and the same solver, and the only difference between them is how the
    per-environment losses are combined.
    """
    if pooled or pairs.size == 0:
        return Environments(
            codes=np.zeros(pairs.size, dtype=np.intp),
            counts=np.array([max(pairs.size, 1)], dtype=np.int64),
            names=(POOLED,),
        )
    names = tuple(sorted({str(name) for name in pairs.group.tolist()}))
    position = {name: index for index, name in enumerate(names)}
    codes = np.array([position[str(name)] for name in pairs.group.tolist()], dtype=np.intp)
    counts = np.bincount(codes, minlength=len(names)).astype(np.int64)
    return Environments(codes=codes, counts=np.maximum(counts, 1), names=names)


def _aggregate(losses: np.ndarray, aggregator: str, parameter: float) -> tuple[float, np.ndarray]:
    """Combine per-environment losses into one number, and say how each of them moved it.

    Every aggregator here is a different answer to the same question -- how much does the worst
    environment count -- and each nests a simpler one at the edge of its parameter: the soft
    maximum is the exact minimax as the temperature goes to zero and the environment mean as it
    grows, and the conditional value at risk is the environment mean at alpha = 1. The leakage
    suite asserts both nestings numerically rather than taking the algebra on trust.
    """
    values = np.asarray(losses, dtype=float)
    size = values.size
    if aggregator in (POOLED, GROUP_MEAN):
        return float(values.mean()), np.full(size, 1.0 / size)
    if aggregator == DRO_SOFTMAX:
        temperature = max(float(parameter), 1e-12)
        shifted = values / temperature
        peak = float(shifted.max())
        weights = np.exp(shifted - peak)
        total = float(weights.sum())
        value = temperature * (peak + np.log(total / size))
        return float(value), weights / total
    if aggregator == CVAR:
        # Rockafellar-Uryasev's dual, with the hinge smoothed and eta driven to the SMOOTHED
        # optimum rather than set to the empirical quantile. The distinction is not cosmetic:
        # the envelope theorem gives `dA/dL_g = sigma((L_g - eta*)/s) / (alpha * G)` only at the
        # eta that the smoothed inner problem minimises, and at the quantile -- which minimises
        # the UNsmoothed one -- the omitted d(eta)/d(L) term is large enough that the analytic
        # gradient and a finite difference disagree in the third decimal. The leakage suite
        # checks them against each other, which is how that was found.
        alpha = min(max(float(parameter), 1e-6), 1.0)
        if alpha >= 1.0:
            return float(values.mean()), np.full(size, 1.0 / size)
        lower = float(values.min()) - 40.0 * CVAR_SMOOTHING
        upper = float(values.max()) + 40.0 * CVAR_SMOOTHING
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            slope = 1.0 - float(_sigmoid((values - middle) / CVAR_SMOOTHING).mean()) / alpha
            lower, upper = (lower, middle) if slope > 0.0 else (middle, upper)
        eta = 0.5 * (lower + upper)
        excess = (values - eta) / CVAR_SMOOTHING
        value = eta + CVAR_SMOOTHING * float(_softplus(excess).mean()) / alpha
        return float(value), _sigmoid(excess) / (alpha * size)
    if aggregator == VARIANCE:
        weight = float(parameter)
        mean = float(values.mean())
        centred = values - mean
        value = mean + weight * float((centred**2).mean())
        return float(value), (1.0 + 2.0 * weight * centred) / size
    raise PhaseError(f"unknown aggregator {aggregator!r}")


def _pair_objective(
    weights: np.ndarray,
    differences: np.ndarray,
    environments: Environments,
    aggregator: str,
    parameter: float,
) -> tuple[float, np.ndarray]:
    """The RankNet loss aggregated over environments, and its gradient. One implementation.

    The chain rule collapses to a single matrix-vector product. The aggregate's derivative with
    respect to environment g is a scalar, the environment's own loss is a mean of per-pair
    losses, so the gradient is `-D' u` for a per-pair weight `u` that depends on the pair's
    environment and on its own margin. Nothing here is quadratic in the number of environments
    and nothing loops over pairs.
    """
    margin = differences @ weights
    per_pair = _softplus(-margin)
    sums = np.bincount(environments.codes, weights=per_pair, minlength=environments.size)
    losses = sums / environments.counts
    value, share = _aggregate(losses, aggregator, parameter)
    scale = (share / environments.counts)[environments.codes]
    gradient = -(differences.T @ (scale * _sigmoid(-margin)))
    return (
        float(value + 0.5 * L2_PENALTY * float(weights @ weights)),
        gradient + L2_PENALTY * weights,
    )


def fit_linear_ranker(
    view: SourceFitView,
    pairs: Pairs,
    aggregator: str,
    parameter: float,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Solve one linear ranking objective with L-BFGS. Deterministic, full batch, no step size.

    Full batch rather than stochastic on purpose: a minibatch schedule would put the result at
    the mercy of a shuffle order, and every number in this stage has to survive being recomputed
    on another machine. The pooled objective is convex, and so are the group mean, the soft
    maximum and the conditional value at risk; the variance-penalised one is not, and is started
    from the pooled solution and recorded as a local optimum.
    """
    from scipy.optimize import minimize

    if pairs.size == 0:
        raise PhaseError(
            f"{pairs.family}: no pair was drawn on a block of {view.size} rows with "
            f"{int(view.beneficial.sum())} beneficial, {int(view.harmful.sum())} harmful and "
            f"{int(view.decision_relevant.sum())} inside the deployment prefix, so no ranker "
            "can be fitted"
        )
    differences = view.features[pairs.positive] - view.features[pairs.negative]
    environments = environments_of(pairs, aggregator == POOLED)
    initial = np.zeros(view.width) if start is None else np.asarray(start, dtype=float).copy()
    result = minimize(
        lambda w: _pair_objective(w, differences, environments, aggregator, parameter),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER},
    )
    return np.asarray(result.x, dtype=float)


def fit_boosted_ranker(
    view: SourceFitView,
    pairs: Pairs,
    rounds: int = BOOST_ROUNDS,
    depth: int = BOOST_DEPTH,
) -> list[Any]:
    """Model C: the same pairwise loss, descended by gradient boosting instead of solved.

    Functional gradient descent on the RankNet loss -- the objective is identical to Model B's
    pooled form, and what changes is the hypothesis class the descent happens in. No subsampling
    of rows, of columns or of pairs, so the sequence of trees is a deterministic function of the
    data and the drawn pairs. `rounds` and `depth` are arguments only so that the feasibility
    probe's discarded capacity can be reproduced from the artifacts; every fitted arm uses the
    module constants.
    """
    from sklearn.tree import DecisionTreeRegressor

    if pairs.size == 0:
        raise PhaseError(f"{pairs.family}: no pair was drawn, so no ranker can be fitted")
    scores = np.zeros(view.size, dtype=float)
    trees: list[Any] = []
    for _ in range(rounds):
        margin = scores[pairs.positive] - scores[pairs.negative]
        weight = _sigmoid(-margin) / pairs.size
        gradient = np.zeros(view.size, dtype=float)
        np.add.at(gradient, pairs.positive, -weight)
        np.add.at(gradient, pairs.negative, weight)
        tree = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=BOOST_MIN_LEAF, random_state=RANDOM_SEED
        ).fit(view.features, -gradient)
        scores += BOOST_RATE * np.asarray(tree.predict(view.features), dtype=float)
        trees.append(tree)
    return trees


@dataclass(frozen=True, slots=True)
class Lists:
    """Per-document candidate lists for the listwise objective, laid out contiguously."""

    order: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    target: np.ndarray

    @property
    def size(self) -> int:
        return int(self.starts.size)


def build_lists(view: SourceFitView) -> Lists:
    """Model D's ranking units: one list per (engine, document), with the label distribution.

    ListNet's top-one target is a softmax over a relevance score, and the relevance here is the
    outcome the stage is about: a beneficial candidate should be first, a harmful one last, a
    neutral one between. Lists whose candidates all carry the same outcome contribute a constant
    and are dropped, which keeps the objective's scale independent of how many uninformative
    pages the corpus happens to contain.
    """
    keys = np.array(
        [f"{e}|{d}" for e, d in zip(view.engines.tolist(), view.documents.tolist(), strict=True)],
        dtype=object,
    )
    relevance = np.where(view.beneficial, 1.0, np.where(view.harmful, -1.0, 0.0))
    order = np.lexsort((np.arange(view.size), np.array([str(k) for k in keys.tolist()])))
    ordered_keys = np.array([str(keys[i]) for i in order.tolist()])
    boundary = np.flatnonzero(np.r_[True, ordered_keys[1:] != ordered_keys[:-1]])
    lengths = np.diff(np.r_[boundary, ordered_keys.size])
    keep = np.array(
        [
            length >= 2 and float(np.ptp(relevance[order[start : start + length]])) > 0.0
            for start, length in zip(boundary.tolist(), lengths.tolist(), strict=True)
        ]
    )
    if not keep.any():
        raise PhaseError("no document offers a list with more than one distinct outcome")
    kept = np.concatenate(
        [
            order[start : start + length]
            for start, length, take in zip(
                boundary.tolist(), lengths.tolist(), keep.tolist(), strict=True
            )
            if take
        ]
    )
    kept_lengths = lengths[keep]
    starts = np.r_[0, np.cumsum(kept_lengths)[:-1]]
    scores = relevance[kept]
    shifted = scores - np.repeat(np.maximum.reduceat(scores, starts), kept_lengths)
    weights = np.exp(shifted)
    totals = np.add.reduceat(weights, starts)
    return Lists(
        order=kept,
        starts=starts.astype(np.intp),
        lengths=kept_lengths.astype(np.intp),
        target=weights / np.repeat(totals, kept_lengths),
    )


def _list_objective(
    weights: np.ndarray, features: np.ndarray, lists: Lists
) -> tuple[float, np.ndarray]:
    """ListNet's top-one cross-entropy over the document lists, and its gradient."""
    scores = features @ weights
    peak = np.repeat(np.maximum.reduceat(scores, lists.starts), lists.lengths)
    exponentials = np.exp(scores - peak)
    totals = np.add.reduceat(exponentials, lists.starts)
    log_partition = np.log(totals) + np.maximum.reduceat(scores, lists.starts)
    cross = log_partition - np.add.reduceat(lists.target * scores, lists.starts)
    probabilities = exponentials / np.repeat(totals, lists.lengths)
    gradient = features.T @ ((probabilities - lists.target) / lists.size)
    return (
        float(cross.mean() + 0.5 * L2_PENALTY * float(weights @ weights)),
        gradient + L2_PENALTY * weights,
    )


def fit_listwise(view: SourceFitView, lists: Lists) -> np.ndarray:
    from scipy.optimize import minimize

    features = view.features[lists.order]
    result = minimize(
        lambda w: _list_objective(w, features, lists),
        np.zeros(view.width),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER},
    )
    return np.asarray(result.x, dtype=float)


# ------------------------------------------------------------------ one fold, fully materialised


@dataclass(slots=True)
class FoldState:
    """One leave-one-engine-out fold with SGV5's model rebuilt on it and every block resolved."""

    held_out: str
    fold: rl.Fold
    design: dg.Design
    extended: c5.FoldDesign
    columns: tuple[int, ...]
    selection: dict[str, Any]
    scaler: Any
    frozen_model: c5.Reliability
    identity: dict[str, Any]
    source_threshold: dict[str, float]

    def index(self, block: str) -> np.ndarray:
        blocks = {
            FIT: self.fold.fit,
            CALIBRATION: self.fold.source_cal,
            SEEN_EVAL: self.fold.seen_eval,
            EVALUATION: self.fold.eval,
            POOL: self.fold.unlabeled_target,
        }
        if block not in blocks:
            raise PhaseError(f"unknown block {block!r}")
        return blocks[block]

    def matrix(self, index: np.ndarray) -> np.ndarray:
        block = self.design.matrix[np.ix_(self.extended.block(index), list(self.columns))]
        return np.asarray(self.scaler.transform(block), dtype=float)

    def frozen_score(self, index: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.frozen_model.utility(
                self.design, self.extended.block(index), float(self.selection["lambda"])
            ),
            dtype=float,
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
    """Rebuild SGV5's selected model on one fold and freeze the scaler every SGV12 model shares.

    The construction is SGV10's, imported rather than restated, and the identity check is the
    same one: on an OUTER fold the rebuilt decision score must equal SGV5's published vector bit
    for bit, because every comparison in this stage is against that baseline. Inner folds have
    no published counterpart and carry no such check; what they do carry is SGV5's frozen choice
    of representation, model class and lambda for the OUTER fold, so an inner replay is a replay
    of the frozen pipeline and not of a pipeline that re-chose itself.
    """
    from sklearn.preprocessing import StandardScaler

    extended_raw, retrieval = c5.extend_fold(base, fold, signatures, retrieval_columns)
    extended = s6._fill_target_block(
        base, extended_raw, retrieval, fold, signatures, retrieval_columns
    )
    design = extended.design
    representation, model_kind = str(selection["representation"]), str(selection["model"])
    if model_kind == "ranker":  # pragma: no cover - no fold selected the ranker
        raise PhaseError(
            "SGV5's ranker exposes no calibrated harm probability and the ablation-A controls "
            "are built from the two heads; a fold that selected it would need its own control"
        )
    columns = extended.columns(representation)
    model = c5.fit_reliability(design, fold, columns, model_kind, representation)
    scaler = StandardScaler().fit(design.matrix[np.ix_(fold.fit, columns)])

    identity: dict[str, Any] = {"checked": False}
    if frozen is not None:
        identifiers = design.meta["candidate_id"].astype(str).to_numpy()
        reference = frozen[frozen["held_out_engine"] == held_out].set_index("candidate_id")
        published = reference.loc[identifiers[fold.eval], s6.SGV5_ARM].to_numpy(dtype=float)
        rebuilt = np.asarray(
            model.utility(design, extended.block(fold.eval), float(selection["lambda"])),
            dtype=float,
        )
        difference = float(np.abs(rebuilt - published).max())
        if difference != 0.0:
            raise PhaseError(
                f"{held_out}: the rebuilt SGV5 decision score differs from the published one by "
                f"{difference:g}; every SGV12 comparison is against that score and a drifted "
                "baseline invalidates all of them"
            )
        identity = {"checked": True, "max_abs_difference": difference, "rows": int(fold.eval.size)}

    state = FoldState(
        held_out=held_out,
        fold=fold,
        design=design,
        extended=extended,
        columns=tuple(columns),
        selection=dict(selection),
        scaler=scaler,
        frozen_model=model,
        identity=identity,
        source_threshold={},
    )
    calibration = state.index(CALIBRATION)
    scores = state.frozen_score(calibration)
    harmful = design.harmful[calibration]
    for epsilon in EPSILONS:
        decision = select_threshold(scores, harmful, epsilon, delta=DELTA, controller="empirical")
        state.source_threshold[epsilon_key(epsilon)] = (
            float(decision.tau) if decision.feasible else float(np.inf)
        )
    return state


def _relevant_prefix(state: FoldState, index: np.ndarray) -> tuple[np.ndarray, float]:
    """The deployment-relevant prefix of a block, at the acceptance share the SOURCE cut takes.

    Family P4 needs a notion of "inside the deployment prefix" on whichever rows an objective is
    being fitted on. The SHARE always comes from the source engines' calibration rows -- the
    fraction of them that SGV5's published threshold accepts at the primary epsilon -- and only
    which rows carry the mark depends on the block. That keeps the operational weighting a
    source-side quantity even when the block is the oracle's.
    """
    calibration = state.index(CALIBRATION)
    tau = state.source_threshold[PRIMARY_KEY]
    share = float(np.mean(state.frozen_score(calibration) >= tau)) if np.isfinite(tau) else 0.0
    scores = state.frozen_score(index)
    take = int(np.clip(round(share * index.size), 0, index.size))
    mask = np.zeros(index.size, dtype=bool)
    if take:
        mask[np.argsort(-scores, kind="stable")[:take]] = True
    return mask, share


def source_view(state: FoldState) -> SourceFitView:
    """The SOURCE engines' TRAIN rows, standardised, with the environments and the pair keys.

    This is the only object any fitting function in this stage is handed. It carries no target
    row, no evaluation row and no held-out engine identity; `decision_relevant` is derived from
    the frozen SGV5 ranking and the acceptance share the published threshold takes on the
    SOURCE calibration rows, so even the operational weighting reads nothing from the engine it
    will be deployed on.
    """
    index = state.index(FIT)
    meta = state.design.meta
    relevant, _ = _relevant_prefix(state, index)
    documents = state.design.documents[index]
    return SourceFitView(
        features=state.matrix(index),
        harmful=state.design.harmful[index],
        beneficial=state.design.beneficial[index],
        documents=documents,
        sites=meta["site_id"].astype(str).to_numpy()[index],
        engines=meta["engine_id"].astype(str).to_numpy()[index],
        shards=shard_of(documents),
        bucket=matched_bucket(state.design.names, state.design.matrix, index),
        decision_relevant=relevant,
    )


def oracle_view(state: FoldState) -> SourceFitView:
    """Ablation H only: the EVALUATION block dressed as a fitting view.

    This is the one function in the stage that hands a fitting objective the held-out engine's
    own labelled rows, and it exists so that the oracle re-ranking upper bound can be measured.
    It is named separately from `source_view` precisely so that the leakage suite can assert
    that exactly one call site reaches it and that the arm it produces appears in no selection,
    no criterion and no headline.
    """
    index = state.index(EVALUATION)
    meta = state.design.meta
    documents = state.design.documents[index]
    relevant, _ = _relevant_prefix(state, index)
    return SourceFitView(
        features=state.matrix(index),
        harmful=state.design.harmful[index],
        beneficial=state.design.beneficial[index],
        documents=documents,
        sites=meta["site_id"].astype(str).to_numpy()[index],
        engines=meta["engine_id"].astype(str).to_numpy()[index],
        shards=shard_of(documents),
        bucket=matched_bucket(state.design.names, state.design.matrix, index),
        decision_relevant=relevant,
    )


# ------------------------------------------------------------------ methods and scorers

LINEAR = "linear"
BOOSTED = "boosted"
RELIABILITY = "reliability"
FROZEN = "frozen"


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """What one arm is: a hypothesis class, an aggregator, an environment definition."""

    kind: str
    aggregator: str
    grouping: str
    hyperparameter: str
    model_class: str


METHOD_SPECS: dict[str, MethodSpec] = {
    SGV5_FROZEN: MethodSpec(FROZEN, "", "", "", ""),
    PAIR_LINEAR: MethodSpec(LINEAR, POOLED, ENGINE_SHARD, "", ""),
    PAIR_BOOSTED: MethodSpec(BOOSTED, POOLED, ENGINE_SHARD, "", ""),
    LISTWISE: MethodSpec(LINEAR, "", "", "", ""),
    GROUP_DRO: MethodSpec(LINEAR, DRO_SOFTMAX, ENGINE_SHARD, "dro_temperature", ""),
    CVAR_RANK: MethodSpec(LINEAR, CVAR, ENGINE_SHARD, "cvar_alpha", ""),
    VARIANCE_REG: MethodSpec(LINEAR, VARIANCE, ENGINE_SHARD, "variance_weight", ""),
    CLF_LINEAR: MethodSpec(RELIABILITY, "", "", "", "logistic"),
    CLF_BOOSTED: MethodSpec(RELIABILITY, "", "", "", "boosted"),
    DRO_ENGINE_ONLY: MethodSpec(LINEAR, DRO_SOFTMAX, ENGINE_ONLY, "dro_temperature", ""),
    DRO_BLOCK_ONLY: MethodSpec(LINEAR, DRO_SOFTMAX, SHARD_ONLY, "dro_temperature", ""),
    DRO_SHUFFLED: MethodSpec(LINEAR, DRO_SOFTMAX, SHUFFLED, "dro_temperature", ""),
}

FITTED_METHODS = tuple(
    name for name, spec in METHOD_SPECS.items() if spec.kind in (LINEAR, BOOSTED, RELIABILITY)
)


@dataclass(frozen=True, slots=True)
class Ranker:
    """A fitted scoring function, and the three facts that identify which one it is."""

    method: str
    family: str
    kind: str
    parameter: float
    weights: np.ndarray | None = None
    trees: tuple[Any, ...] | None = None
    reliability: Any | None = None

    def score(self, state: FoldState, index: np.ndarray) -> np.ndarray:
        if self.kind == FROZEN:
            return state.frozen_score(index)
        if self.kind == RELIABILITY:
            assert self.reliability is not None
            return np.asarray(
                self.reliability.utility(
                    state.design, state.extended.block(index), float(state.selection["lambda"])
                ),
                dtype=float,
            )
        matrix = state.matrix(index)
        if self.kind == LINEAR:
            assert self.weights is not None
            return np.asarray(matrix @ self.weights, dtype=float)
        assert self.trees is not None
        total = np.zeros(matrix.shape[0], dtype=float)
        for tree in self.trees:
            total += BOOST_RATE * np.asarray(tree.predict(matrix), dtype=float)
        return total


def relabel(pairs: Pairs, view: SourceFitView, grouping: str, seed: int) -> Pairs:
    """The same drawn pairs under a different environment definition.

    Ablations C, D and G change what an environment IS, not which comparisons the objective
    sees. Redrawing the pairs for each of them would confound the two, so the pair set is drawn
    once per family and only its labels move.
    """
    if grouping == SHUFFLED:
        labels = permuted_groups(group_labels(view.engines, view.shards, ENGINE_SHARD), seed)
    else:
        labels = group_labels(view.engines, view.shards, grouping)
    return Pairs(
        family=pairs.family,
        positive=pairs.positive,
        negative=pairs.negative,
        group=labels[pairs.positive],
        eligible=pairs.eligible,
        drawn=pairs.drawn,
        strata=pairs.strata,
        strata_used=pairs.strata_used,
        cross_group=int((labels[pairs.positive] != labels[pairs.negative]).sum()),
        exhaustive=pairs.exhaustive,
    )


def fit_ranker(
    state: FoldState,
    view: SourceFitView,
    method: str,
    family: str,
    parameter: float,
    pairs: Pairs,
    pooled_start: np.ndarray | None = None,
) -> Ranker:
    """Fit one arm. Every branch reads `view` and `pairs` and nothing else.

    The classification controls go through SGV5's own `fit_reliability`, so ablation A compares
    two objectives run by two implementations that were not written to agree -- the published
    one and this stage's -- rather than one implementation parameterised two ways.
    """
    spec = METHOD_SPECS[method]
    if spec.kind == RELIABILITY:
        model = c5.fit_reliability(
            state.design,
            state.fold,
            list(state.columns),
            spec.model_class,
            str(state.selection["representation"]),
        )
        return Ranker(
            method=method,
            family=family,
            kind=RELIABILITY,
            parameter=float("nan"),
            reliability=model,
        )
    if spec.kind == BOOSTED:
        return Ranker(
            method=method,
            family=family,
            kind=BOOSTED,
            parameter=float("nan"),
            trees=tuple(fit_boosted_ranker(view, pairs)),
        )
    if method == LISTWISE:
        # Built from the view that was handed in, never cached on the fold. A cached list layout
        # would silently be the SOURCE block's when the same fold later fits the oracle arm on
        # the evaluation block, which is the one place in this stage where a stale cache would
        # be a leak rather than a slowdown.
        return Ranker(
            method=method,
            family="",
            kind=LINEAR,
            parameter=float("nan"),
            weights=fit_listwise(view, build_lists(view)),
        )
    grouped = (
        pairs
        if spec.grouping == ENGINE_SHARD and method != DRO_SHUFFLED
        else relabel(
            pairs, view, spec.grouping, _stable_seed("sgv12-shuffle", state.held_out, family)
        )
    )
    start = pooled_start if spec.aggregator == VARIANCE else None
    weights = fit_linear_ranker(view, grouped, spec.aggregator, parameter, start)
    return Ranker(
        method=method, family=family, kind=LINEAR, parameter=float(parameter), weights=weights
    )


# ------------------------------------------------------------------ the ranking metrics


def ranking_quality(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray
) -> dict[str, Any]:
    """What one ordering is worth on one block. Every quantity comes from `metrics/`.

    `auroc_safe` is the brief's held-out AUROC written in this stage's orientation: the
    probability that a random NOT-harmful candidate outranks a random harmful one. `pair_accuracy`
    is the brief's pairwise ranking accuracy, which is the same statistic restricted to rows that
    are unambiguously beneficial or unambiguously harmful -- the pairs a ranking objective is
    actually trained on, with the neutral rows that neither helps nor hurts removed.
    """
    values = np.asarray(score, dtype=float)
    harm = np.asarray(harmful, dtype=bool)
    benefit = np.asarray(beneficial, dtype=bool)
    decided = harm | benefit
    return {
        "n": int(values.size),
        "n_harmful": int(harm.sum()),
        "n_beneficial": int(benefit.sum()),
        "n_decided": int(decided.sum()),
        "auroc_safe": roc_auc(values, (~harm).astype(float)),
        "pair_accuracy": (
            roc_auc(values[decided], benefit[decided].astype(float))
            if int(decided.sum()) > 0
            else float("nan")
        ),
        "average_precision_safe": average_precision(values, (~harm).astype(float)),
        "average_precision_beneficial": average_precision(values, benefit.astype(float)),
        "aurc": float(aurc(risk_coverage_curve(values, harm))) if values.size else float("nan"),
    }


def _by_key(keys: np.ndarray) -> dict[str, np.ndarray]:
    """Row indices grouped by key, in one pass.

    Every per-group statistic in this stage would otherwise cost a full-length comparison per
    group, which is quadratic in the number of keys and turns the site-level tables into minutes
    of work for a quantity that is one dictionary pass.
    """
    members: dict[str, list[int]] = {}
    for index, name in enumerate(keys.tolist()):
        members.setdefault(str(name), []).append(index)
    return {name: np.asarray(rows, dtype=np.intp) for name, rows in members.items()}


def group_quality(
    score: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """The same quantities per environment, plus the worst one and the spread across them.

    A group that cannot define the statistic -- too few rows, or one outcome class missing -- is
    FLAGGED and excluded from the worst-case, never silently counted as a perfect score or as a
    zero. The count of flagged groups is reported beside the worst case so a worst-group number
    can never be read without knowing how many groups were eligible to produce it.
    """
    members = _by_key(labels)
    names = sorted(members)
    per_group: dict[str, Any] = {}
    eligible: list[str] = []
    flagged: list[str] = []
    for name in names:
        rows = members[name]
        harm = np.asarray(harmful, dtype=bool)[rows]
        benefit = np.asarray(beneficial, dtype=bool)[rows]
        usable = (
            rows.size >= MIN_GROUP_ROWS
            and int(harm.sum()) >= MIN_GROUP_CLASS
            and int((~harm).sum()) >= MIN_GROUP_CLASS
        )
        block = ranking_quality(np.asarray(score, dtype=float)[rows], harm, benefit)
        block["eligible"] = bool(usable)
        per_group[name] = block
        (eligible if usable else flagged).append(name)
    values = [per_group[name]["auroc_safe"] for name in eligible]
    pairs = [
        per_group[name]["pair_accuracy"]
        for name in eligible
        if np.isfinite(per_group[name]["pair_accuracy"])
    ]
    return {
        "groups": per_group,
        "n_groups": len(names),
        "n_eligible": len(eligible),
        "flagged": flagged,
        "worst_auroc": float(min(values)) if values else float("nan"),
        "worst_group": (
            min(eligible, key=lambda name: per_group[name]["auroc_safe"]) if values else ""
        ),
        "mean_auroc": float(np.mean(values)) if values else float("nan"),
        "spread_auroc": float(max(values) - min(values)) if values else float("nan"),
        "worst_pair_accuracy": float(min(pairs)) if pairs else float("nan"),
        "mean_pair_accuracy": float(np.mean(pairs)) if pairs else float("nan"),
    }


# ------------------------------------------------------------------ the frozen decision rule

EMPIRICAL = "empirical"
LTT = "ltt_bentkus"
CONTROLLER_RULES = (EMPIRICAL, LTT)
PRIMARY_CONTROLLER = EMPIRICAL


def place_cut(
    calibration_score: np.ndarray,
    calibration_harmful: np.ndarray,
    epsilon: float,
    controller: str,
) -> dict[str, Any]:
    """The one cut-placement procedure this stage uses, for every ranking without exception.

    The published controller, on the SOURCE engines' labelled calibration rows, reading no
    target row of any kind. This is the identical call that produced the threshold SGV5
    published and SGV10 and SGV11 carried, which is what makes SGV12's baseline arm reproduce
    SGV5's published deployment rather than approximate it.
    """
    decision = select_threshold(
        np.asarray(calibration_score, dtype=float),
        np.asarray(calibration_harmful, dtype=bool),
        epsilon,
        delta=DELTA,
        controller=controller,
    )
    return {
        "tau": float(decision.tau) if decision.feasible else float(np.inf),
        "feasible": bool(decision.feasible),
        "calibration_coverage": float(decision.coverage),
        "calibration_risk": float(decision.observed_risk),
        "controller": controller,
    }


def deployed(
    score: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    tau: float,
    epsilon: float,
) -> dict[str, Any]:
    """What the cut does on the evaluation block, with the risk-controlled endpoint attached."""
    point = dict(deployed_point(np.asarray(score, dtype=float), harmful, beneficial, tau))
    realized = float(point["realized_harm_rate"])
    point["epsilon"] = float(epsilon)
    point["tau"] = float(tau)
    point["holds_bound"] = True if point["accepts_nothing"] else bool(realized <= epsilon)
    point["bound_violation"] = (
        0.0 if point["accepts_nothing"] else float(max(0.0, realized - epsilon))
    )
    point["risk_controlled_repair_recall"] = risk_controlled_recall(point, epsilon)
    point["repairs_captured"] = (
        0
        if point["accepts_nothing"]
        else int(np.asarray(beneficial, dtype=bool)[np.asarray(score, dtype=float) >= tau].sum())
    )
    point["joint_harm_rate"] = (
        0.0
        if point["accepts_nothing"]
        else float(
            np.asarray(harmful, dtype=bool)[np.asarray(score, dtype=float) >= tau].sum()
            / max(int(np.asarray(score).size), 1)
        )
    )
    return point


def frontier(score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray) -> dict[str, Any]:
    """The achievable frontier of one ranking: an ORACLE over the cut, not a method.

    The deepest prefix of this ranking whose realised harm holds epsilon, read on the block it
    is measured on. Identical in construction to SGV11's `frontier_of`, because the whole point
    of the gap ladder is that the two stages measure the same rungs.
    """
    return dict(achievable_repair_recall(np.asarray(score, dtype=float), harmful, beneficial))


# ------------------------------------------------------------------ source-only selection


def _inner_states(
    base: dg.Design,
    held_out: str,
    selection: dict[str, Any],
    signatures: np.ndarray,
    retrieval_columns: list[int],
) -> dict[str, tuple[FoldState, SourceFitView]]:
    """SGV9's inner leave-one-source-engine-out replay, one state per pseudo-target.

    The outer held-out engine appears in no inner block, no DEVELOPMENT page appears anywhere,
    and the CALIBRATION pages are split in half by content hash so the inner outcome is read on
    pages the inner fit never saw. All three are asserted inside `s9.inner_fold`, which is
    imported rather than restated so that SGV9's, SGV10's, SGV11's and SGV12's inner folds are
    the same folds.
    """
    out: dict[str, tuple[FoldState, SourceFitView]] = {}
    for target in sorted(e for e in base.engines if e != held_out):
        fold = inner_fold(base, held_out, target)
        state = build_state(base, fold, selection, signatures, retrieval_columns, target, None)
        out[target] = (state, source_view(state))
    return out


def _inner_score(
    inner: dict[str, tuple[FoldState, SourceFitView]],
    method: str,
    family: str,
    parameter: float,
    budget: int,
) -> dict[str, Any]:
    """Fit one configuration on every inner fold and read its pairwise accuracy on the outcome.

    The reported number is the MEAN over inner folds, which is decision 8: selecting on the
    worst inner fold would make 'the robust objective wins' and 'the robust selection rule wins'
    the same measurement. The worst inner fold is recorded beside the mean so the ablation that
    asks what a worst-case rule would have chosen can read it without refitting anything.
    """
    per_target: dict[str, float] = {}
    for target, (state, view) in sorted(inner.items()):
        pairs = build_pairs(
            family, view, ENGINE_SHARD, _stable_seed("sgv12-pairs", target, family), budget
        )
        ranker = fit_ranker(state, view, method, family, parameter, pairs)
        index = state.index(EVALUATION)
        quality = ranking_quality(
            ranker.score(state, index),
            state.design.harmful[index],
            state.design.beneficial[index],
        )
        per_target[target] = float(quality["pair_accuracy"])
    values = [v for v in per_target.values() if np.isfinite(v)]
    return {
        "method": method,
        "pair_family": family,
        "parameter": float(parameter),
        "pair_budget": int(budget),
        "per_inner_target": per_target,
        "mean_pair_accuracy": float(np.mean(values)) if values else float("nan"),
        "worst_pair_accuracy": float(np.min(values)) if values else float("nan"),
    }


def _best(rows: dict[str, dict[str, Any]], key: str, order: tuple[str, ...]) -> str:
    """Argmax with a deterministic tie-break: the earliest name in the pre-registered order."""
    ranked = sorted(
        order,
        key=lambda name: (
            -(rows[name][key] if np.isfinite(rows[name][key]) else -np.inf),
            order.index(name),
        ),
    )
    return ranked[0]


def select_configuration(
    inner: dict[str, tuple[FoldState, SourceFitView]],
) -> dict[str, Any]:
    """Choose the pair family, then the objective and its hyperparameter. Source engines only.

    Two stages rather than one grid, because the pair family is a property of the OBJECTIVE's
    input and the hyperparameter is a property of its aggregation: crossing them would multiply
    the inner fits by five for a choice the cheapest arm can already make. The family is chosen
    with the pooled linear ranker, which is the arm whose objective is defined by the pairs and
    nothing else, and the chosen family is then held fixed while the objective is chosen.
    """
    families = {
        family: _inner_score(inner, PAIR_LINEAR, family, float("nan"), PAIR_BUDGET)
        for family in SELECTABLE_FAMILIES
    }
    family = _best(families, "mean_pair_accuracy", SELECTABLE_FAMILIES)

    per_method: dict[str, dict[str, Any]] = {}
    sweeps: dict[str, dict[str, Any]] = {}
    for method in PROPOSED:
        name, grid = HYPERPARAMETERS.get(method, ("", (float("nan"),)))
        rows = {
            f"{value:g}": _inner_score(inner, method, family, value, PAIR_BUDGET) for value in grid
        }
        labels = [f"{value:g}" for value in grid]
        chosen = _best(rows, "mean_pair_accuracy", tuple(labels))
        sweeps[method] = {"hyperparameter": name, "per_value": rows, "selected": chosen}
        per_method[method] = dict(rows[chosen])
        per_method[method]["hyperparameter_name"] = name
        per_method[method]["hyperparameter"] = float(grid[labels.index(chosen)])
    method = _best(per_method, "mean_pair_accuracy", PROPOSED)

    budgets = {
        f"{budget}": _inner_score(inner, PAIR_LINEAR, family, float("nan"), budget)
        for budget in PAIR_BUDGET_SWEEP
    }
    return {
        "selection_scope": (
            "inner leave-one-source-engine-out replay over the source engines only; the outer "
            "held-out engine appears in no inner block and no DEVELOPMENT page appears anywhere"
        ),
        "selection_objective": "mean pairwise ranking accuracy over the inner pseudo-targets",
        "pair_family_stage": {"per_family": families, "selected": family},
        "objective_stage": {"per_method": per_method, "sweeps": sweeps, "selected": method},
        "pair_budget_sensitivity": budgets,
        "worst_case_alternative": {
            "family": _best(families, "worst_pair_accuracy", SELECTABLE_FAMILIES),
            "method": _best(per_method, "worst_pair_accuracy", PROPOSED),
            "note": (
                "what a worst-inner-fold selection rule would have chosen. Reported, never used: "
                "decision 8 selects on the mean so that the objective and the selection rule are "
                "not the same measurement."
            ),
        },
        "selected": {
            "pair_family": family,
            "method": method,
            "hyperparameter_name": str(per_method[method]["hyperparameter_name"]),
            "hyperparameter": float(per_method[method]["hyperparameter"]),
        },
        "hyperparameters": {name: float(per_method[name]["hyperparameter"]) for name in PROPOSED},
    }


# ------------------------------------------------------------------ arms and their columns


def score_column(arm: str) -> str:
    return f"score__{arm}"


def family_arm(family: str) -> str:
    return f"fam__{family}"


def sweep_arm(method: str, value: float) -> str:
    return f"sweep__{method}__{value:g}"


def budget_arm(budget: int) -> str:
    return f"budget__{budget}"


def ceiling_arm(key: str) -> str:
    return f"{CEILING}__{key}"


PRIMARY_ARMS = (SGV5_FROZEN, *PROPOSED, SELECTED)
ABLATION_ARMS = (CLF_LINEAR, CLF_BOOSTED, DRO_ENGINE_ONLY, DRO_BLOCK_ONLY, DRO_SHUFFLED)
FAMILY_ARMS = tuple(family_arm(family) for family in PAIR_FAMILIES)
SWEEP_ARMS = tuple(
    sweep_arm(method, value) for method, (_, grid) in HYPERPARAMETERS.items() for value in grid
)
BUDGET_ARMS = tuple(budget_arm(budget) for budget in PAIR_BUDGET_SWEEP)
CEILING_ARMS = tuple(ceiling_arm(epsilon_key(epsilon)) for epsilon in EPSILONS)
ALL_ARMS = (
    *PRIMARY_ARMS,
    *ABLATION_ARMS,
    *FAMILY_ARMS,
    *SWEEP_ARMS,
    *BUDGET_ARMS,
    ORACLE_RERANK,
)
# Arms a criterion, a headline or a selection may read. The oracle re-ranker and the ceiling
# are excluded by construction; `--negative` asserts the exclusion rather than assuming it.
DEPLOYABLE_ARMS = (*PRIMARY_ARMS, *ABLATION_ARMS, *FAMILY_ARMS, *SWEEP_ARMS, *BUDGET_ARMS)


def fit_all(
    state: FoldState,
    view: SourceFitView,
    pairs: dict[str, Pairs],
    chosen: dict[str, Any],
) -> dict[str, Ranker]:
    """Fit every arm this fold contributes. The only place in the stage that fits anything.

    The order matters in one place only: the pooled linear solution is fitted first because the
    variance-penalised objective is non-convex and starts from it, which makes Model G a
    documented refinement of Model B rather than a different random restart each time it runs.
    """
    family = str(chosen["selected"]["pair_family"])
    method = str(chosen["selected"]["method"])
    hyper = {name: float(value) for name, value in chosen["hyperparameters"].items()}
    rankers: dict[str, Ranker] = {
        SGV5_FROZEN: Ranker(method=SGV5_FROZEN, family="", kind=FROZEN, parameter=float("nan"))
    }
    pooled = fit_ranker(state, view, PAIR_LINEAR, family, float("nan"), pairs[family])
    rankers[PAIR_LINEAR] = pooled
    for name in PROPOSED:
        if name == PAIR_LINEAR:
            continue
        rankers[name] = fit_ranker(
            state, view, name, family, hyper.get(name, float("nan")), pairs[family], pooled.weights
        )
    rankers[SELECTED] = rankers[method]
    for name in ABLATION_ARMS:
        rankers[name] = fit_ranker(
            state,
            view,
            name,
            family,
            hyper.get(GROUP_DRO, float("nan")),
            pairs[family],
            pooled.weights,
        )
    # Ablations E and F read the SELECTED objective under every pair family. A listwise
    # selection makes that axis vacuous -- the objective never sees a pair -- so the sweep falls
    # back to the pooled linear ranker and the design record says which of the two it ran.
    sweep_method = PAIR_LINEAR if method == LISTWISE else method
    for name in PAIR_FAMILIES:
        rankers[family_arm(name)] = fit_ranker(
            state,
            view,
            sweep_method,
            name,
            hyper.get(sweep_method, float("nan")),
            pairs[name],
            pooled.weights,
        )
    for name, (_, grid) in HYPERPARAMETERS.items():
        for value in grid:
            rankers[sweep_arm(name, value)] = fit_ranker(
                state, view, name, family, value, pairs[family], pooled.weights
            )
    for budget in PAIR_BUDGET_SWEEP:
        drawn = build_pairs(
            family,
            view,
            ENGINE_SHARD,
            _stable_seed("sgv12-pairs", state.held_out, family),
            budget,
        )
        rankers[budget_arm(budget)] = fit_ranker(
            state, view, PAIR_LINEAR, family, float("nan"), drawn
        )
    # Ablation H uses the SELECTED objective, listwise included: the oracle is meant to bound
    # what this stage's own hypothesis class could reach with the answers, so substituting a
    # different objective for it would price a different ceiling.
    oracle = oracle_view(state)
    oracle_pairs = build_pairs(
        family,
        oracle,
        ENGINE_SHARD,
        _stable_seed("sgv12-oracle", state.held_out, family),
        PAIR_BUDGET,
    )
    rankers[ORACLE_RERANK] = fit_ranker(
        state, oracle, method, family, hyper.get(method, float("nan")), oracle_pairs
    )
    missing = set(ALL_ARMS) - set(rankers)
    if missing:
        raise PhaseError(f"{state.held_out}: arms were not fitted: {sorted(missing)}")
    return rankers


def _ceiling_columns(engine: str, candidates: np.ndarray) -> dict[str, np.ndarray]:
    """SGV10's published `oracle_eval` re-ranking for one fold's evaluation rows, read not made."""
    table = pd.read_parquet(s10.SCORES)
    block = table[(table["held_out_engine"] == engine) & (table["block"] == "evaluation")]
    indexed = block.set_index(block["candidate_id"].astype(str))
    ordered = indexed.loc[[str(name) for name in candidates.tolist()]]
    return {
        ceiling_arm(epsilon_key(epsilon)): ordered[
            f"score__{s10.ORACLE_EVAL}__{epsilon_key(epsilon)}"
        ].to_numpy(dtype=float)
        for epsilon in EPSILONS
    }


def _fold_frame(state: FoldState, rankers: dict[str, Ranker]) -> pd.DataFrame:
    """One fold's five blocks, every arm scored on every row, in one table."""
    meta = state.design.meta
    frames: list[pd.DataFrame] = []
    for block in BLOCK_NAMES:
        index = state.index(block)
        documents = state.design.documents[index]
        frame = pd.DataFrame(
            {
                "held_out_engine": state.held_out,
                "block": block,
                "candidate_id": meta["candidate_id"].astype(str).to_numpy()[index],
                "document_id": documents,
                "site_id": meta["site_id"].astype(str).to_numpy()[index],
                "engine_id": meta["engine_id"].astype(str).to_numpy()[index],
                "shard": shard_of(documents),
                "is_harmful": state.design.harmful[index],
                "beneficial": state.design.beneficial[index],
            }
        )
        for arm in ALL_ARMS:
            frame[score_column(arm)] = rankers[arm].score(state, index)
        for name in CEILING_ARMS:
            frame[score_column(name)] = np.nan
        if block == EVALUATION:
            ceiling = _ceiling_columns(state.held_out, frame["candidate_id"].to_numpy())
            for name, values in ceiling.items():
                frame[score_column(name)] = values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ writing an artifact


def _write(path: Path, payload: dict[str, Any], schema: str) -> None:
    """One artifact, with the six facts every artifact in this project has to carry."""
    body = {
        "schema_version": schema,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "hypothesis_id": PRE_REGISTRATION["hypothesis_id"],
        "development_only": True,
        "synthetic": False,
        "confirmatory_accessed": False,
        **payload,
    }
    cc._write_json_once(path, body)


def _group_inventory(state: FoldState, view: SourceFitView) -> dict[str, Any]:
    """Every environment this fold defines, with its size and whether a metric can be read."""
    out: dict[str, Any] = {}
    for grouping in (ENGINE_SHARD, ENGINE_ONLY, SHARD_ONLY):
        labels = group_labels(view.engines, view.shards, grouping)
        names = sorted({str(name) for name in labels.tolist()})
        sizes: dict[str, Any] = {}
        for name in names:
            mask = np.asarray([str(value) == name for value in labels.tolist()], dtype=bool)
            harm = view.harmful[mask]
            sizes[name] = {
                "rows": int(mask.sum()),
                "documents": len(set(view.documents[mask].tolist())),
                "harmful": int(harm.sum()),
                "beneficial": int(view.beneficial[mask].sum()),
                "harm_prevalence": float(harm.mean()) if int(mask.sum()) else float("nan"),
                "eligible_for_a_ranking_metric": bool(
                    int(mask.sum()) >= MIN_GROUP_ROWS
                    and int(harm.sum()) >= MIN_GROUP_CLASS
                    and int((~harm).sum()) >= MIN_GROUP_CLASS
                ),
            }
        out[grouping] = {
            "note": GROUP_NOTES[grouping],
            "n_groups": len(names),
            "n_flagged": sum(
                1 for row in sizes.values() if not row["eligible_for_a_ranking_metric"]
            ),
            "groups": sizes,
        }
    return out


PROBE_ENGINE = "doctr"
PROBE_FAMILY = P1_DOCUMENT
PROBE_PARAMETERS = {
    PAIR_LINEAR: float("nan"),
    PAIR_BOOSTED: float("nan"),
    LISTWISE: float("nan"),
    GROUP_DRO: 0.05,
    CVAR_RANK: 0.2,
    VARIANCE_REG: 1.0,
}
PROBE_LABELS = {
    SGV5_FROZEN: "mA_sgv5_frozen",
    PAIR_LINEAR: "mB_pair_linear",
    PAIR_BOOSTED: "mC_pair_boosted_at_100_rounds_depth_4",
    LISTWISE: "mD_listwise",
    GROUP_DRO: "mE_group_dro_at_0.05",
    CVAR_RANK: "mF_cvar_rank_at_0.2",
    VARIANCE_REG: "mG_variance_penalty_at_1.0",
}


def reproduce_probe(state: FoldState, view: SourceFitView, pairs: Pairs) -> dict[str, Any]:
    """Refit the disclosed probe cell and check it comes back. A miss is a hard failure.

    Refitted exactly as the probe ran it, which is not in every respect how the stage ships. The
    boosted arm is refitted at the capacity the probe used and this stage discarded, because that
    is the only way its disclosed number can be checked at all. The variance-penalised arm is
    refitted from a COLD start, because the probe ran it from one; the shipping arm warm-starts
    from the pooled solution, the objective is not convex, and the two land in different local
    optima. Both facts are disclosed rather than smoothed over by widening a tolerance.
    """
    tolerance = 0.5 * 10.0 ** -int(FEASIBILITY_PROBE["probe_3_capacity"]["printed_decimals"])
    index = state.index(EVALUATION)
    harmful, beneficial = state.design.harmful[index], state.design.beneficial[index]
    observed: dict[str, dict[str, float]] = {}
    for method, label in PROBE_LABELS.items():
        if method == SGV5_FROZEN:
            scores = state.frozen_score(index)
        elif method == PAIR_BOOSTED:
            trees = fit_boosted_ranker(view, pairs, PROBE_BOOST_ROUNDS, PROBE_BOOST_DEPTH)
            ranker = Ranker(method, PROBE_FAMILY, BOOSTED, float("nan"), trees=tuple(trees))
            scores = ranker.score(state, index)
        else:
            ranker = fit_ranker(
                state, view, method, PROBE_FAMILY, PROBE_PARAMETERS[method], pairs, None
            )
            scores = ranker.score(state, index)
        quality = ranking_quality(scores, harmful, beneficial)
        observed[label] = {
            "auroc_safe": float(quality["auroc_safe"]),
            "pair_accuracy": float(quality["pair_accuracy"]),
        }
    changed = set(
        FEASIBILITY_PROBE["probe_3_capacity"]["arms_whose_implementation_changed_after_the_probe"]
    )
    misses: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    for field in ("auroc_safe", "pair_accuracy"):
        disclosed = FEASIBILITY_PROBE["probe_3_capacity"][
            "measured_auroc_safe" if field == "auroc_safe" else "measured_pair_accuracy"
        ]
        for label, value in disclosed.items():
            gap = abs(observed[label][field] - float(value))
            row = {
                "arm": label,
                "field": field,
                "disclosed": float(value),
                "reproduced": observed[label][field],
                "gap": gap,
            }
            if gap <= tolerance:
                continue
            # An arm whose implementation was superseded after the probe cannot reproduce it
            # without shipping the superseded implementation. Its gap is published, not asserted.
            (superseded if label in changed else misses).append(row)
    if misses:
        raise PhaseError(f"the disclosed probe does not reproduce: {misses}")
    return {
        "engine": PROBE_ENGINE,
        "pair_family": PROBE_FAMILY,
        "tolerance": tolerance,
        "reproduced": observed,
        "misses": misses,
        "superseded_implementations": superseded,
        "matches": True,
    }


def run_rank() -> int:
    """Verify the freeze, select on the source engines, fit every arm, write the score table.

    This is the only phase that fits anything. Everything downstream reads
    `ranking_scores.parquet` and the records written here, which is what makes a re-analysis
    cheap and what makes "every number traces to an artifact" checkable rather than aspirational.
    """
    started = time.monotonic()
    verification = verify_frozen()
    base, signatures, retrieval_columns = s6.load_base()
    selection = s6.frozen_selection()
    frozen = s6.load_frozen_scores()
    published_thresholds = cc._read_json(s10.DESIGN_RECORD)["folds"]

    frames: list[pd.DataFrame] = []
    probe_reproduction: dict[str, Any] = {}
    folds: dict[str, Any] = {}
    pairs_record: dict[str, Any] = {}
    groups_record: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for engine in sorted(base.engines):
        fold = build_fold(base, engine)
        state = build_state(
            base, fold, selection[engine], signatures, retrieval_columns, engine, frozen
        )
        recorded = dict(published_thresholds[engine]["source_thresholds"])
        for key, tau in sorted(state.source_threshold.items()):
            if float(recorded[key]) != tau:
                raise PhaseError(
                    f"{engine}: this stage's frozen decision rule places SGV5's cut at {tau:g} "
                    f"but SGV10 published {float(recorded[key]):g} for {key}; the rule the "
                    "baseline is measured under is not the rule the baseline was published with"
                )
        view = source_view(state)
        pairs = {
            family: build_pairs(
                family,
                view,
                ENGINE_SHARD,
                _stable_seed("sgv12-pairs", engine, family),
                PAIR_BUDGET,
            )
            for family in PAIR_FAMILIES
        }
        inner = _inner_states(base, engine, selection[engine], signatures, retrieval_columns)
        chosen = select_configuration(inner)
        inner_rows = {
            target: {
                "fit_rows": int(inner_state.index(FIT).size),
                "outcome_rows": int(inner_state.index(EVALUATION).size),
                "source_engines": list(inner_state.fold.train_engines),
            }
            for target, (inner_state, _) in sorted(inner.items())
        }
        del inner
        rankers = fit_all(state, view, pairs, chosen)
        frames.append(_fold_frame(state, rankers))
        if engine == PROBE_ENGINE:
            probe_reproduction = reproduce_probe(state, view, pairs[PROBE_FAMILY])

        selections[engine] = {**chosen, "inner_folds": inner_rows}
        pairs_record[engine] = {family: pairs[family].inventory() for family in PAIR_FAMILIES}
        groups_record[engine] = _group_inventory(state, view)
        folds[engine] = {
            "identity_against_published_sgv5": state.identity,
            "selected_by_sgv5": dict(state.selection),
            "representation_columns": len(state.columns),
            "source_thresholds": dict(state.source_threshold),
            "source_thresholds_match_sgv10": True,
            "source_engines": list(fold.train_engines),
            "rows": {name: int(state.index(name).size) for name in BLOCK_NAMES},
            "documents": {
                name: len(set(state.design.documents[state.index(name)].tolist()))
                for name in BLOCK_NAMES
            },
            "harm_prevalence": {
                name: float(state.design.harmful[state.index(name)].mean()) for name in BLOCK_NAMES
            },
            "decision_relevant_fit_rows": int(view.decision_relevant.sum()),
            "arms_fitted": len(ALL_ARMS),
        }
        print(
            f"  {engine}: {state.index(EVALUATION).size} evaluation rows, "
            f"{state.index(FIT).size} fit rows, {len(state.columns)} columns, selected "
            f"{chosen['selected']['method']} on {chosen['selected']['pair_family']}"
        )
        del state, view, pairs, rankers

    frame = pd.concat(frames, ignore_index=True)
    pilot._write_parquet_once(SCORES, frame)
    _write(
        DESIGN_RECORD,
        {
            "stage": "SGV12 -- robust cross-engine risk ranking",
            "pre_registration": PRE_REGISTRATION,
            "decisions": DECISIONS,
            "feasibility_probe": {**FEASIBILITY_PROBE, "probe_3_reproduction": probe_reproduction},
            "freeze_verification": verification,
            "methods": {name: METHOD_NOTES[name] for name in METHOD_NOTES},
            "pair_families": dict(PAIR_NOTES),
            "groupings": dict(GROUP_NOTES),
            "blocks": dict(BLOCK_NOTES),
            "settings": dict(SETTING_NOTES),
            "constants": {
                "epsilons": list(EPSILONS),
                "primary_epsilon": PRIMARY_EPSILON,
                "n_shards": N_SHARDS,
                "pair_budget": PAIR_BUDGET,
                "pair_budget_sweep": list(PAIR_BUDGET_SWEEP),
                "l2_penalty": L2_PENALTY,
                "lbfgs_maxiter": LBFGS_MAXITER,
                "boost_rounds": BOOST_ROUNDS,
                "boost_rate": BOOST_RATE,
                "boost_depth": BOOST_DEPTH,
                "boost_min_samples_leaf": BOOST_MIN_LEAF,
                "dro_temperatures": list(DRO_TEMPERATURES),
                "cvar_alphas": list(CVAR_ALPHAS),
                "variance_weights": list(VARIANCE_WEIGHTS),
                "cvar_smoothing": CVAR_SMOOTHING,
                "min_group_rows": MIN_GROUP_ROWS,
                "min_group_class": MIN_GROUP_CLASS,
                "recovery_threshold": RECOVERY_THRESHOLD,
                "engine_bar": ENGINE_BAR,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "delta": DELTA,
            },
            "decision_rule": {
                "controller": PRIMARY_CONTROLLER,
                "reported_beside_it": LTT,
                "fitted_on": "source engines, CALIBRATION documents",
                "note": (
                    "identical for every ranking in the stage. Asserted per fold to reproduce "
                    "the threshold SGV10 published for SGV5's frozen score, so the baseline arm "
                    "is SGV5's published deployment and not an approximation of it."
                ),
            },
            "folds": folds,
        },
        "sgv12-design-v1",
    )
    _write(
        PAIR_INVENTORY,
        {
            "stage": "SGV12 -- what each pair family could draw and what it drew",
            "families": dict(PAIR_NOTES),
            "budget": PAIR_BUDGET,
            "engines": pairs_record,
        },
        "sgv12-pairs-v1",
    )
    _write(
        GROUP_INVENTORY,
        {
            "stage": "SGV12 -- the environments, their sizes, and which can carry a metric",
            "shard_construction": (
                f"sha256('{SHARD_SALT}|' + document_id)[0] mod {N_SHARDS}: one global partition "
                "of the 710 documents, identical across engines and roles"
            ),
            "engines": groups_record,
        },
        "sgv12-groups-v1",
    )
    _write(
        MODEL_SELECTION,
        {
            "stage": "SGV12 -- what the source engines chose, and what they would have chosen",
            "engines": selections,
        },
        "sgv12-selection-v1",
    )
    elapsed = time.monotonic() - started
    print(f"rank: {len(folds)} folds, {len(frame)} rows -> {SCORES} ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ reading the score table


def load_rows() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not SCORES.is_file() or not DESIGN_RECORD.is_file():
        raise PhaseError("run --rank before any phase that reads its output")
    return pd.read_parquet(SCORES), cc._read_json(DESIGN_RECORD)


def block_rows(frame: pd.DataFrame, engine: str, block: str) -> pd.DataFrame:
    return frame[(frame["held_out_engine"] == engine) & (frame["block"] == block)].reset_index(
        drop=True
    )


def _arm(frame: pd.DataFrame, arm: str) -> np.ndarray:
    column = score_column(arm)
    if column not in frame.columns:
        raise PhaseError(f"the score table carries no column for {arm}")
    return frame[column].to_numpy(dtype=float)


def _labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame["is_harmful"].to_numpy(dtype=bool),
        frame["beneficial"].to_numpy(dtype=bool),
    )


def _engines(record: dict[str, Any]) -> list[str]:
    return sorted(record["folds"])


def selected_of(record: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """What the source engines chose, per fold, read from the record rather than re-chosen."""
    del record
    if not MODEL_SELECTION.is_file():
        raise PhaseError("run --rank before reading the selection")
    payload = cc._read_json(MODEL_SELECTION)
    return {engine: dict(block["selected"]) for engine, block in payload["engines"].items()}


# ------------------------------------------------------------------ phase: --ranking

# The arms every table reports. The oracle re-ranker and the ceiling are reported only in the
# phases that exist to price them, and never beside a deployable arm without their label.
REPORTED_ARMS = (*PRIMARY_ARMS, *ABLATION_ARMS, *FAMILY_ARMS)


def run_ranking() -> int:
    """Settings A and C: what each ordering is worth on a seen engine and on an unseen one.

    The two are read on the SAME DEVELOPMENT pages -- this corpus is matched-source, so the
    source engines' `seen_eval` block and the held-out engine's `evaluation` block are the same
    scans read by different engines. That is what makes the difference between them an engine
    effect and not a page effect, and it is the reason setting A is defined on those pages rather
    than on a held-out slice of the fit documents.
    """
    started = time.monotonic()
    frame, record = load_rows()
    engines = _engines(record)
    cells: dict[str, Any] = {}
    for engine in engines:
        blocks: dict[str, Any] = {}
        for block in (CALIBRATION, SEEN_EVAL, EVALUATION, POOL):
            rows = block_rows(frame, engine, block)
            harmful, beneficial = _labels(rows)
            blocks[block] = {
                "setting": SETTING_A
                if block == SEEN_EVAL
                else (SETTING_C if block == EVALUATION else ""),
                "rows": len(rows),
                "documents": int(rows["document_id"].nunique()),
                "harm_prevalence": float(harmful.mean()) if len(rows) else float("nan"),
                "arms": {
                    arm: ranking_quality(_arm(rows, arm), harmful, beneficial)
                    for arm in REPORTED_ARMS
                },
            }
        # Per-engine detail inside setting A: the source engines are not interchangeable, and a
        # pooled seen-engine number can hide one of them carrying the whole thing.
        seen = block_rows(frame, engine, SEEN_EVAL)
        harmful, beneficial = _labels(seen)
        per_source = {
            source: {
                arm: ranking_quality(
                    _arm(seen, arm)[seen["engine_id"].to_numpy(str) == source],
                    harmful[seen["engine_id"].to_numpy(str) == source],
                    beneficial[seen["engine_id"].to_numpy(str) == source],
                )
                for arm in REPORTED_ARMS
            }
            for source in sorted(seen["engine_id"].astype(str).unique())
        }
        degradation = {
            arm: {
                "source_validation_auroc": blocks[CALIBRATION]["arms"][arm]["auroc_safe"],
                "held_out_engine_auroc": blocks[EVALUATION]["arms"][arm]["auroc_safe"],
                "delta_auroc": float(
                    blocks[CALIBRATION]["arms"][arm]["auroc_safe"]
                    - blocks[EVALUATION]["arms"][arm]["auroc_safe"]
                ),
                "source_validation_pair_accuracy": blocks[CALIBRATION]["arms"][arm][
                    "pair_accuracy"
                ],
                "held_out_engine_pair_accuracy": blocks[EVALUATION]["arms"][arm]["pair_accuracy"],
                "delta_pair_accuracy": float(
                    blocks[CALIBRATION]["arms"][arm]["pair_accuracy"]
                    - blocks[EVALUATION]["arms"][arm]["pair_accuracy"]
                ),
            }
            for arm in REPORTED_ARMS
        }
        evaluation = block_rows(frame, engine, EVALUATION)
        eval_harmful, eval_beneficial = _labels(evaluation)
        cells[engine] = {
            "blocks": blocks,
            "setting_a_by_source_engine": per_source,
            "ranking_degradation": degradation,
            # Reported apart from every deployable arm and labelled, because both of these read
            # the evaluation block's own labels and neither is a method.
            "oracle_arms": {
                name: ranking_quality(_arm(evaluation, name), eval_harmful, eval_beneficial)
                for name in (ORACLE_RERANK, ceiling_arm(PRIMARY_KEY))
            },
        }

    aggregate = {
        arm: {
            "mean_auroc_setting_a": float(
                np.mean([cells[e]["blocks"][SEEN_EVAL]["arms"][arm]["auroc_safe"] for e in engines])
            ),
            "mean_auroc_setting_c": float(
                np.mean(
                    [cells[e]["blocks"][EVALUATION]["arms"][arm]["auroc_safe"] for e in engines]
                )
            ),
            "mean_pair_accuracy_setting_c": float(
                np.mean(
                    [cells[e]["blocks"][EVALUATION]["arms"][arm]["pair_accuracy"] for e in engines]
                )
            ),
            "mean_average_precision_safe_setting_c": float(
                np.mean(
                    [
                        cells[e]["blocks"][EVALUATION]["arms"][arm]["average_precision_safe"]
                        for e in engines
                    ]
                )
            ),
            "mean_delta_auroc_vs_sgv5": float(
                np.mean(
                    [
                        cells[e]["blocks"][EVALUATION]["arms"][arm]["auroc_safe"]
                        - cells[e]["blocks"][EVALUATION]["arms"][SGV5_FROZEN]["auroc_safe"]
                        for e in engines
                    ]
                )
            ),
            "engines_improving_auroc": int(
                sum(
                    cells[e]["blocks"][EVALUATION]["arms"][arm]["auroc_safe"]
                    > cells[e]["blocks"][EVALUATION]["arms"][SGV5_FROZEN]["auroc_safe"]
                    for e in engines
                )
            ),
            "engines_improving_pair_accuracy": int(
                sum(
                    cells[e]["blocks"][EVALUATION]["arms"][arm]["pair_accuracy"]
                    > cells[e]["blocks"][EVALUATION]["arms"][SGV5_FROZEN]["pair_accuracy"]
                    for e in engines
                )
            ),
            "mean_ranking_degradation": float(
                np.mean([cells[e]["ranking_degradation"][arm]["delta_auroc"] for e in engines])
            ),
        }
        for arm in REPORTED_ARMS
    }
    oracles = {
        name: {
            "mean_auroc_setting_c": float(
                np.mean([cells[e]["oracle_arms"][name]["auroc_safe"] for e in engines])
            ),
            "mean_pair_accuracy_setting_c": float(
                np.mean([cells[e]["oracle_arms"][name]["pair_accuracy"] for e in engines])
            ),
            "note": METHOD_NOTES.get(name, METHOD_NOTES[CEILING]),
        }
        for name in (ORACLE_RERANK, ceiling_arm(PRIMARY_KEY))
    }
    _write(
        RANKING_METRICS,
        {
            "stage": "SGV12 -- held-out ranking quality, settings A and C",
            "question": (
                "does a robust ranking objective order correction candidates better than SGV5's "
                "classification objective on an OCR engine it was never fitted on"
            ),
            "orientation": (
                "higher score means safer; AUROC is the probability that a non-harmful row "
                "outranks a harmful one"
            ),
            "settings": {SETTING_A: SETTING_NOTES[SETTING_A], SETTING_C: SETTING_NOTES[SETTING_C]},
            "selected_per_fold": selected_of(),
            "engines": cells,
            "aggregate": aggregate,
            "oracles": oracles,
        },
        "sgv12-ranking-v1",
    )
    elapsed = time.monotonic() - started
    print(f"ranking: {len(engines)} folds x {len(REPORTED_ARMS)} arms ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --blocks


def _published_p_harm(engine: str, candidates: np.ndarray) -> np.ndarray:
    """SGV5's published calibrated harm probability for one fold's evaluation rows.

    Read, never recomputed, and used for exactly one thing: fixing which document shard setting
    D calls the hardest, before any SGV12 score exists. It is a frozen model output, so the
    definition reads no label and nothing this stage produced.
    """
    frame = s6.load_frozen_scores()
    block = frame[frame["held_out_engine"] == engine]
    indexed = block.set_index(block["candidate_id"].astype(str))
    return indexed.loc[[str(name) for name in candidates.tolist()], "p_harm"].to_numpy(dtype=float)


def hardest_shard(rows: pd.DataFrame, engine: str) -> dict[str, Any]:
    """Setting D's block, fixed by a frozen model output and never by an SGV12 result."""
    predicted = _published_p_harm(engine, rows["candidate_id"].to_numpy())
    shards = rows["shard"].to_numpy(dtype=int)
    per_shard = {
        int(shard): float(predicted[shards == shard].mean())
        for shard in sorted(set(shards.tolist()))
    }
    chosen = max(per_shard, key=lambda key: (per_shard[key], -key))
    return {
        "definition": (
            "the document shard with the highest mean published SGV5 p_harm over this engine's "
            "evaluation rows: a frozen model output, no label, no SGV12 quantity"
        ),
        "mean_predicted_harm_by_shard": per_shard,
        "hardest_shard": int(chosen),
        "rows": int((shards == chosen).sum()),
    }


def run_blocks() -> int:
    """Settings B and D: does the ordering survive a change of page composition, engine held fixed.

    SGV11 found a threshold chosen on an engine's own labelled pool breaking its bound on that
    same engine's evaluation pages, which is a statement about page composition and not about
    engines. This phase asks the ranking-level version of that question, and it asks it on both
    sides of the fold: for a source engine the fit block is in-sample, so the drop prices the
    model's own optimism, and for the held-out engine both blocks are out-of-sample, so the drop
    is page composition with nothing else moving.
    """
    started = time.monotonic()
    frame, record = load_rows()
    engines = _engines(record)
    cells: dict[str, Any] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        pool = block_rows(frame, engine, POOL)
        fit = block_rows(frame, engine, FIT)
        seen = block_rows(frame, engine, SEEN_EVAL)
        harmful, beneficial = _labels(evaluation)
        shard_labels = np.array([f"s{int(s)}" for s in evaluation["shard"].tolist()], dtype=object)
        seen_labels = np.array(
            [
                f"{e}|s{int(s)}"
                for e, s in zip(seen["engine_id"].tolist(), seen["shard"].tolist(), strict=True)
            ],
            dtype=object,
        )
        hardest = hardest_shard(evaluation, engine)
        mask = evaluation["shard"].to_numpy(dtype=int) == hardest["hardest_shard"]
        per_arm: dict[str, Any] = {}
        for arm in REPORTED_ARMS:
            scores = _arm(evaluation, arm)
            within = group_quality(scores, harmful, beneficial, shard_labels)
            pool_harm, pool_ben = _labels(pool)
            fit_harm, fit_ben = _labels(fit)
            seen_harm, seen_ben = _labels(seen)
            held_out_pool = ranking_quality(_arm(pool, arm), pool_harm, pool_ben)
            held_out_eval = ranking_quality(scores, harmful, beneficial)
            source_fit = ranking_quality(_arm(fit, arm), fit_harm, fit_ben)
            source_seen = ranking_quality(_arm(seen, arm), seen_harm, seen_ben)
            per_arm[arm] = {
                "setting_b_within_held_out_engine": within,
                "setting_b_within_source_engines": group_quality(
                    _arm(seen, arm), seen_harm, seen_ben, seen_labels
                ),
                "cross_block_degradation": {
                    "held_out_engine": {
                        "train_pages_auroc": held_out_pool["auroc_safe"],
                        "development_pages_auroc": held_out_eval["auroc_safe"],
                        "delta_auroc": float(
                            held_out_pool["auroc_safe"] - held_out_eval["auroc_safe"]
                        ),
                        "note": "both blocks out of sample; the drop is page composition alone",
                    },
                    "source_engines": {
                        "fit_pages_auroc": source_fit["auroc_safe"],
                        "development_pages_auroc": source_seen["auroc_safe"],
                        "delta_auroc": float(source_fit["auroc_safe"] - source_seen["auroc_safe"]),
                        "note": "the fit block is in sample; the drop prices the model's optimism",
                    },
                },
                "setting_d_hardest_shard": ranking_quality(
                    scores[mask], harmful[mask], beneficial[mask]
                ),
            }
        cells[engine] = {"hardest_shard": hardest, "arms": per_arm}

    aggregate = {
        arm: {
            "mean_worst_shard_auroc": float(
                np.mean(
                    [
                        cells[e]["arms"][arm]["setting_b_within_held_out_engine"]["worst_auroc"]
                        for e in engines
                    ]
                )
            ),
            "mean_shard_spread": float(
                np.mean(
                    [
                        cells[e]["arms"][arm]["setting_b_within_held_out_engine"]["spread_auroc"]
                        for e in engines
                    ]
                )
            ),
            "engines_improving_worst_shard_vs_sgv5": int(
                sum(
                    cells[e]["arms"][arm]["setting_b_within_held_out_engine"]["worst_auroc"]
                    > cells[e]["arms"][SGV5_FROZEN]["setting_b_within_held_out_engine"][
                        "worst_auroc"
                    ]
                    for e in engines
                )
            ),
            "engines_improving_worst_shard_vs_pooled": int(
                sum(
                    cells[e]["arms"][arm]["setting_b_within_held_out_engine"]["worst_auroc"]
                    > cells[e]["arms"][PAIR_LINEAR]["setting_b_within_held_out_engine"][
                        "worst_auroc"
                    ]
                    for e in engines
                )
            ),
            "mean_hardest_shard_auroc": float(
                np.mean(
                    [
                        cells[e]["arms"][arm]["setting_d_hardest_shard"]["auroc_safe"]
                        for e in engines
                    ]
                )
            ),
            "mean_block_shift_held_out": float(
                np.mean(
                    [
                        cells[e]["arms"][arm]["cross_block_degradation"]["held_out_engine"][
                            "delta_auroc"
                        ]
                        for e in engines
                    ]
                )
            ),
        }
        for arm in REPORTED_ARMS
    }
    _write(
        BLOCK_SHIFT,
        {
            "stage": "SGV12 -- within-engine block shift, settings B and D",
            "question": (
                "does a ranking that survives the engine boundary also survive a change of page "
                "composition inside one engine, which is where SGV11 saw a bound break"
            ),
            "settings": {SETTING_B: SETTING_NOTES[SETTING_B], SETTING_D: SETTING_NOTES[SETTING_D]},
            "engines": cells,
            "aggregate": aggregate,
        },
        "sgv12-blocks-v1",
    )
    elapsed = time.monotonic() - started
    print(f"blocks: {len(engines)} folds, {N_SHARDS} shards ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --frontier

MINIMAX_RULE = "sgv11_minimax"
DEPLOY_RULES = (EMPIRICAL, LTT, MINIMAX_RULE)

RULE_NOTES = {
    EMPIRICAL: (
        "The PRIMARY rule and the only one any criterion reads. SGV5's published controller on "
        "the source engines' calibration rows: the identical call that placed the threshold "
        "SGV5 published, applied to every ranking in this stage without exception."
    ),
    LTT: (
        "Learn-then-Test with a Bentkus bound at delta, the conservative controller the same "
        "published module offers. Reported for every ranking, never for one, so that a "
        "difference between two rankings can never be a difference in which controller they got."
    ),
    MINIMAX_RULE: (
        "SGV11's minimax cut: the deepest source threshold whose WORST source engine's harm rate "
        "is within epsilon, through SGV11's own implementation. Reported for every ranking "
        "identically. It is a decision-layer rule and is carried here only so that the "
        "ranking-versus-decision question can be read at two different rules rather than one."
    ),
}


def minimax_cut(
    score: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    documents: np.ndarray,
    engines: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    """SGV11's minimax threshold, through SGV11's implementation and not a second one."""
    view = s11.SourcePolicyView(
        score=np.asarray(score, dtype=float),
        tail_risk=np.zeros(np.asarray(score).size, dtype=float),
        harmful=np.asarray(harmful, dtype=bool),
        beneficial=np.asarray(beneficial, dtype=bool),
        documents=np.asarray(documents, dtype=object),
        engines=np.asarray(engines, dtype=object),
    )
    curves = s11.source_curves(view)
    index = s11._deepest_on_grid(curves.minimax, epsilon)
    if index < 0:
        return {
            "tau": float(np.inf),
            "feasible": False,
            "calibration_coverage": 0.0,
            "calibration_risk": float("nan"),
            "controller": MINIMAX_RULE,
        }
    return {
        "tau": float(curves.grid[index]),
        "feasible": True,
        "calibration_coverage": float(curves.coverage[index]),
        "calibration_risk": float(curves.minimax[index]),
        "controller": MINIMAX_RULE,
    }


def _cut_for(rule: str, calibration: pd.DataFrame, arm: str, epsilon: float) -> dict[str, Any]:
    harmful, beneficial = _labels(calibration)
    scores = _arm(calibration, arm)
    if rule == MINIMAX_RULE:
        return minimax_cut(
            scores,
            harmful,
            beneficial,
            calibration["document_id"].to_numpy(str),
            calibration["engine_id"].to_numpy(str),
            epsilon,
        )
    return place_cut(scores, harmful, epsilon, rule)


def run_frontier() -> int:
    """The operational half: what each ordering is worth once a cut has to be placed on it.

    Every ranking meets the same rule, on the same rows, at the same epsilon. That is decision 3
    and it is the whole reason this phase can be read as a statement about rankings: a
    difference here cannot be a difference in the decision layer, because there is only one
    decision layer and every arm is handed it.
    """
    started = time.monotonic()
    frame, record = load_rows()
    engines = _engines(record)
    cells: dict[str, Any] = {}
    for engine in engines:
        calibration = block_rows(frame, engine, CALIBRATION)
        evaluation = block_rows(frame, engine, EVALUATION)
        harmful, beneficial = _labels(evaluation)
        per_epsilon: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = epsilon_key(epsilon)
            arms: dict[str, Any] = {}
            for arm in REPORTED_ARMS:
                scores = _arm(evaluation, arm)
                curve = risk_coverage_curve(scores, harmful)
                block: dict[str, Any] = {
                    "achievable_frontier": frontier(scores, harmful, beneficial)[key],
                    "aurc": float(aurc(curve)),
                    "rules": {},
                }
                for rule in DEPLOY_RULES:
                    cut = _cut_for(rule, calibration, arm, epsilon)
                    point = deployed(scores, harmful, beneficial, cut["tau"], epsilon)
                    block["rules"][rule] = {**cut, **point}
                arms[arm] = block
            per_epsilon[key] = {"epsilon": float(epsilon), "arms": arms}
        cells[engine] = {
            "evaluation_rows": len(evaluation),
            "evaluation_documents": int(evaluation["document_id"].nunique()),
            "total_beneficial": int(beneficial.sum()),
            "per_epsilon": per_epsilon,
        }

    aggregate: dict[str, Any] = {}
    for epsilon in EPSILONS:
        key = epsilon_key(epsilon)
        aggregate[key] = {
            arm: {
                "mean_risk_controlled_repair_recall": float(
                    np.mean(
                        [
                            cells[e]["per_epsilon"][key]["arms"][arm]["rules"][PRIMARY_CONTROLLER][
                                "risk_controlled_repair_recall"
                            ]
                            for e in engines
                        ]
                    )
                ),
                "mean_achievable_frontier": float(
                    np.mean(
                        [
                            cells[e]["per_epsilon"][key]["arms"][arm]["achievable_frontier"][
                                "repair_recall"
                            ]
                            for e in engines
                        ]
                    )
                ),
                "engines_holding_bound": int(
                    sum(
                        cells[e]["per_epsilon"][key]["arms"][arm]["rules"][PRIMARY_CONTROLLER][
                            "holds_bound"
                        ]
                        for e in engines
                    )
                ),
                "engines_improving_vs_sgv5": int(
                    sum(
                        cells[e]["per_epsilon"][key]["arms"][arm]["rules"][PRIMARY_CONTROLLER][
                            "risk_controlled_repair_recall"
                        ]
                        > cells[e]["per_epsilon"][key]["arms"][SGV5_FROZEN]["rules"][
                            PRIMARY_CONTROLLER
                        ]["risk_controlled_repair_recall"]
                        for e in engines
                    )
                ),
                "mean_aurc": float(
                    np.mean([cells[e]["per_epsilon"][key]["arms"][arm]["aurc"] for e in engines])
                ),
            }
            for arm in REPORTED_ARMS
        }
    _write(
        FRONTIER_RESULTS,
        {
            "stage": "SGV12 -- the risk-controlled frontier under one frozen decision rule",
            "question": (
                "does a better held-out ordering buy anything once the cut has to be placed "
                "without target labels"
            ),
            "primary_rule": PRIMARY_CONTROLLER,
            "rules": dict(RULE_NOTES),
            "endpoint": (
                "risk-controlled repair recall: the repair recall of the accepted set if its "
                "realised harm rate is within epsilon, and zero otherwise. Imported from SGV11 "
                "so the two stages' primary numbers are the same quantity."
            ),
            "engines": cells,
            "aggregate": aggregate,
        },
        "sgv12-frontier-v1",
    )
    elapsed = time.monotonic() - started
    print(f"frontier: {len(engines)} folds x {len(EPSILONS)} epsilons ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --gap


def run_gap() -> int:
    """SGV11's ladder, recomputed rung by rung with a different score on the middle rung.

        1.000                     every beneficial candidate and no harmful one (arithmetic)
        ceiling frontier          the best cut of a re-ranking fitted on the evaluation block
          |  score-transfer gap   what a better SCORE would buy at a perfect cut
        frozen frontier           the best cut of the ranking a policy actually has
          |  decision-transfer gap  what not knowing WHERE to cut costs
        deployed                  the arm's risk-controlled repair recall

    SGV11 measured the two gaps at 0.3065 and 0.0779 with SGV5's ranking on the middle rung, and
    that is the whole reason this stage exists. The recovery fraction is how much of the upper
    gap a new ranking closes, with the ceiling and the baseline frontier both unchanged, so a
    number here is directly comparable to the number SGV11 published.
    """
    started = time.monotonic()
    frame, record = load_rows()
    engines = _engines(record)
    published = record["freeze_verification"]["sgv11_published_ladder"]
    cells: dict[str, Any] = {}
    for engine in engines:
        calibration = block_rows(frame, engine, CALIBRATION)
        evaluation = block_rows(frame, engine, EVALUATION)
        harmful, beneficial = _labels(evaluation)
        per_epsilon: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = epsilon_key(epsilon)
            ceiling = frontier(_arm(evaluation, ceiling_arm(key)), harmful, beneficial)[key]
            own_ceiling = frontier(_arm(evaluation, ORACLE_RERANK), harmful, beneficial)[key]
            baseline = frontier(_arm(evaluation, SGV5_FROZEN), harmful, beneficial)[key]
            headroom = float(ceiling["repair_recall"] - baseline["repair_recall"])
            arms: dict[str, Any] = {}
            for arm in REPORTED_ARMS:
                scores = _arm(evaluation, arm)
                own = frontier(scores, harmful, beneficial)[key]
                cut = _cut_for(PRIMARY_CONTROLLER, calibration, arm, epsilon)
                point = deployed(scores, harmful, beneficial, cut["tau"], epsilon)
                deployed_value = float(point["risk_controlled_repair_recall"])
                arms[arm] = {
                    "frozen_frontier": float(own["repair_recall"]),
                    "deployed": deployed_value,
                    "score_transfer_gap": float(ceiling["repair_recall"] - own["repair_recall"]),
                    "decision_transfer_gap": float(own["repair_recall"] - deployed_value),
                    "total_gap_to_ceiling": float(ceiling["repair_recall"] - deployed_value),
                    "frontier_gain_vs_sgv5": float(
                        own["repair_recall"] - baseline["repair_recall"]
                    ),
                    "recovery_fraction": (
                        float((own["repair_recall"] - baseline["repair_recall"]) / headroom)
                        if headroom > 0
                        else float("nan")
                    ),
                }
            per_epsilon[key] = {
                "epsilon": float(epsilon),
                "ceiling_frontier": ceiling,
                "sgv12_oracle_rerank_frontier": own_ceiling,
                "baseline_frozen_frontier": baseline,
                "headroom": headroom,
                "arms": arms,
            }
        cells[engine] = per_epsilon

    aggregate: dict[str, Any] = {}
    for epsilon in EPSILONS:
        key = epsilon_key(epsilon)
        ceiling_mean = float(
            np.mean([cells[e][key]["ceiling_frontier"]["repair_recall"] for e in engines])
        )
        baseline_mean = float(
            np.mean([cells[e][key]["baseline_frozen_frontier"]["repair_recall"] for e in engines])
        )
        headroom = ceiling_mean - baseline_mean
        aggregate[key] = {
            "epsilon": float(epsilon),
            "mean_ceiling_frontier": ceiling_mean,
            "mean_baseline_frozen_frontier": baseline_mean,
            "mean_score_transfer_gap": headroom,
            "arms": {
                arm: {
                    "mean_frozen_frontier": float(
                        np.mean([cells[e][key]["arms"][arm]["frozen_frontier"] for e in engines])
                    ),
                    "mean_score_transfer_gap": float(
                        np.mean([cells[e][key]["arms"][arm]["score_transfer_gap"] for e in engines])
                    ),
                    "mean_deployed": float(
                        np.mean([cells[e][key]["arms"][arm]["deployed"] for e in engines])
                    ),
                    "mean_decision_transfer_gap": float(
                        np.mean(
                            [cells[e][key]["arms"][arm]["decision_transfer_gap"] for e in engines]
                        )
                    ),
                    "pooled_recovery_fraction": (
                        float(
                            (
                                float(
                                    np.mean(
                                        [
                                            cells[e][key]["arms"][arm]["frozen_frontier"]
                                            for e in engines
                                        ]
                                    )
                                )
                                - baseline_mean
                            )
                            / headroom
                        )
                        if headroom > 0
                        else float("nan")
                    ),
                    "engines_with_a_higher_frontier": int(
                        sum(
                            cells[e][key]["arms"][arm]["frontier_gain_vs_sgv5"] > 0 for e in engines
                        )
                    ),
                }
                for arm in REPORTED_ARMS
            },
        }

    # SGV11's own ladder, read from its artifact so the report's motivation section can cite the
    # deployed rung and the decision gap without either number being retyped.
    sgv11_aggregate = cc._read_json(s11.DECISION_TRANSFER)["aggregate"][
        f"{s11.RANK_SGV5}|{PRIMARY_KEY}"
    ]["policies"]
    best_deployed = max(
        sgv11_aggregate, key=lambda name: float(sgv11_aggregate[name]["mean_deployed"])
    )
    reproduction = {
        "sgv11_published": published,
        "sgv11_published_best_deployed": {
            "policy": best_deployed,
            "mean_deployed": float(sgv11_aggregate[best_deployed]["mean_deployed"]),
            "mean_decision_transfer_gap": float(
                sgv11_aggregate[best_deployed]["mean_decision_transfer_gap"]
            ),
            "note": (
                "the best of SGV11's own arms at the primary epsilon under its primary ranking "
                "and transport, read from its artifact and not recomputed here"
            ),
        },
        "sgv12_recomputed": {
            "ceiling_frontier": aggregate[PRIMARY_KEY]["mean_ceiling_frontier"],
            "frozen_frontier": aggregate[PRIMARY_KEY]["mean_baseline_frozen_frontier"],
            "score_transfer_gap": aggregate[PRIMARY_KEY]["mean_score_transfer_gap"],
        },
        "max_abs_difference": max(
            abs(
                float(published["ceiling_frontier"])
                - aggregate[PRIMARY_KEY]["mean_ceiling_frontier"]
            ),
            abs(
                float(published["frozen_frontier"])
                - aggregate[PRIMARY_KEY]["mean_baseline_frozen_frontier"]
            ),
        ),
    }
    if reproduction["max_abs_difference"] > 1e-12:
        raise PhaseError(
            "SGV12's recomputed ladder does not reproduce SGV11's published one "
            f"(max difference {reproduction['max_abs_difference']:g}); the recovery fraction's "
            "denominator would not be the gap SGV11 reported"
        )
    _write(
        GAP_DECOMPOSITION,
        {
            "stage": "SGV12 -- the SGV11 gap ladder with a new ranking on the middle rung",
            "recovery_fraction_definition": (
                "(frontier of this arm - frontier of SGV5's frozen ranking) / (ceiling frontier "
                "- frontier of SGV5's frozen ranking), all three measured by the same oracle-cut "
                "construction on the same evaluation block. The denominator is SGV11's published "
                "score-transfer gap and is asserted to reproduce it exactly."
            ),
            "ceiling_source": METHOD_NOTES[CEILING],
            "sgv11_reproduction": reproduction,
            "pre_registered_threshold": RECOVERY_THRESHOLD,
            "engines": cells,
            "aggregate": aggregate,
        },
        "sgv12-gap-v1",
    )
    elapsed = time.monotonic() - started
    print(f"gap: ladder reproduced to {reproduction['max_abs_difference']:g} ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ contrasts and ablations


def contrast(frame: pd.DataFrame, engines: list[str], arm: str, reference: str) -> dict[str, Any]:
    """One arm against one reference, on the four unseen engines, mechanically and operationally.

    Every ablation in this stage is this function called with two arm names. Writing them that
    way rather than eight bespoke comparisons is what makes them commensurable: the same blocks,
    the same rule, the same endpoint, the same direction convention in all eight.
    """
    per_engine: dict[str, Any] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        calibration = block_rows(frame, engine, CALIBRATION)
        harmful, beneficial = _labels(evaluation)
        left = ranking_quality(_arm(evaluation, arm), harmful, beneficial)
        right = ranking_quality(_arm(evaluation, reference), harmful, beneficial)
        shard_labels = np.array([f"s{int(s)}" for s in evaluation["shard"].tolist()], dtype=object)
        row: dict[str, Any] = {
            "delta_auroc": float(left["auroc_safe"] - right["auroc_safe"]),
            "delta_pair_accuracy": float(left["pair_accuracy"] - right["pair_accuracy"]),
            "delta_worst_shard_auroc": float(
                group_quality(_arm(evaluation, arm), harmful, beneficial, shard_labels)[
                    "worst_auroc"
                ]
                - group_quality(_arm(evaluation, reference), harmful, beneficial, shard_labels)[
                    "worst_auroc"
                ]
            ),
            "arm_auroc": float(left["auroc_safe"]),
            "reference_auroc": float(right["auroc_safe"]),
        }
        for epsilon in EPSILONS:
            key = epsilon_key(epsilon)
            values = []
            for name in (arm, reference):
                scores = _arm(evaluation, name)
                cut = _cut_for(PRIMARY_CONTROLLER, calibration, name, epsilon)
                point = deployed(scores, harmful, beneficial, cut["tau"], epsilon)
                values.append(float(point["risk_controlled_repair_recall"]))
            row[f"delta_risk_controlled__{key}"] = float(values[0] - values[1])
            row[f"arm_risk_controlled__{key}"] = values[0]
            row[f"reference_risk_controlled__{key}"] = values[1]
        per_engine[engine] = row
    fields = sorted({name for row in per_engine.values() for name in row})
    return {
        "arm": arm,
        "reference": reference,
        "per_engine": per_engine,
        "mean": {name: float(np.mean([per_engine[e][name] for e in engines])) for name in fields},
        "engines_favouring_the_arm": {
            name: int(sum(per_engine[e][name] > 0 for e in engines))
            for name in fields
            if name.startswith("delta_")
        },
    }


def run_ablation() -> int:
    """The brief's eight ablations, each one a contrast between two arms and nothing else.

    Ablation H is the only one that is not a method comparison: the oracle re-ranker reads the
    evaluation block's own labels and exists to bound what any re-ranking of this representation
    could reach. It appears here, in `--gap` as a second ceiling, and nowhere else.
    """
    started = time.monotonic()
    frame, record = load_rows()
    engines = _engines(record)
    selection = cc._read_json(MODEL_SELECTION)["engines"]
    ablations = {
        "A_classification_vs_ranking": {
            "question": "does the OBJECTIVE matter, with the hypothesis class held fixed",
            "linear": contrast(frame, engines, PAIR_LINEAR, CLF_LINEAR),
            "boosted": contrast(frame, engines, PAIR_BOOSTED, CLF_BOOSTED),
            "note": (
                "the two halves hold capacity fixed on each side. The boosted half is matched to "
                "SGV5's own HistGradientBoostingClassifier configuration, which is what the "
                "feasibility probe's disclosed capacity change was for."
            ),
        },
        "B_pooled_vs_group_dro": {
            "question": "does aggregating the loss over environments rather than over pairs help",
            "contrast": contrast(frame, engines, GROUP_DRO, PAIR_LINEAR),
        },
        "C_engine_groups_only": {
            "question": "is engine identity alone enough to define the environment",
            "contrast": contrast(frame, engines, DRO_ENGINE_ONLY, GROUP_DRO),
        },
        "D_document_blocks_only": {
            "question": (
                "is the document shard alone enough -- the direct test of whether SGV11's "
                "within-engine instability is addressable by a training objective"
            ),
            "contrast": contrast(frame, engines, DRO_BLOCK_ONLY, GROUP_DRO),
        },
        "E_global_pairing": {
            "question": "does local pair construction matter, at a fixed eligibility rule",
            "within_document": contrast(
                frame, engines, family_arm(P1_DOCUMENT), family_arm(P0_GLOBAL)
            ),
            "within_site": contrast(frame, engines, family_arm(P2_SITE), family_arm(P0_GLOBAL)),
            "matched_type": contrast(frame, engines, family_arm(P3_MATCHED), family_arm(P0_GLOBAL)),
        },
        "F_no_decision_relevant_weighting": {
            "question": "does restricting pairs to the deployment prefix matter, at a fixed scope",
            "contrast": contrast(frame, engines, family_arm(P4_DECISION), family_arm(P0_GLOBAL)),
        },
        "G_shuffled_group_labels": {
            "question": "is the environment structure real, or is the gain a regularisation effect",
            "contrast": contrast(frame, engines, DRO_SHUFFLED, GROUP_DRO),
        },
        "H_oracle_reranker": {
            "question": "what could a re-ranking of this representation reach with the answers",
            "contrast": contrast(frame, engines, ORACLE_RERANK, SGV5_FROZEN),
            "note": (
                "an ORACLE, fitted on the evaluation block's own labels. Analysis only: it feeds "
                "no criterion, no selection and no headline, and `--negative` asserts it."
            ),
        },
    }
    sweeps = {
        method: {
            "hyperparameter": HYPERPARAMETERS[method][0],
            "source_validated_choice": {
                engine: selection[engine]["hyperparameters"][method] for engine in engines
            },
            "per_value": {
                f"{value:g}": contrast(frame, engines, sweep_arm(method, value), SGV5_FROZEN)
                for value in HYPERPARAMETERS[method][1]
            },
        }
        for method in HYPERPARAMETERS
    }
    budgets = {
        f"{budget}": contrast(frame, engines, budget_arm(budget), SGV5_FROZEN)
        for budget in PAIR_BUDGET_SWEEP
    }
    families = {
        family: contrast(frame, engines, family_arm(family), SGV5_FROZEN)
        for family in PAIR_FAMILIES
    }
    _write(
        ABLATION_RESULTS,
        {
            "stage": "SGV12 -- the eight ablations",
            "direction_convention": (
                "every delta is arm minus reference, so a positive delta favours the arm named "
                "first. `engines_favouring_the_arm` counts strictly positive deltas out of four."
            ),
            "ablations": ablations,
            "hyperparameter_sweeps": sweeps,
            "pair_budget_sweep": budgets,
            "pair_family_sweep": families,
        },
        "sgv12-ablation-v1",
    )
    elapsed = time.monotonic() - started
    print(f"ablation: {len(ablations)} ablations, {len(engines)} folds ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --negative


def _within_unit_accuracy(frame: pd.DataFrame, arm: str, unit: str) -> dict[str, Any]:
    """Pairwise accuracy restricted to comparisons inside one document or one site.

    A ranking that wins only because it separates easy pages from hard ones cannot win here:
    every comparison is between two candidates the operator sees at the same time. This is
    failure test 2, and it is the evaluation-side counterpart of pair families P1 and P2.
    """
    harmful, beneficial = _labels(frame)
    scores = _arm(frame, arm)
    members = _by_key(frame[unit].astype(str).to_numpy())
    concordant = 0.0
    total = 0
    for rows in members.values():
        good = rows[beneficial[rows]]
        bad = rows[harmful[rows]]
        if good.size == 0 or bad.size == 0:
            continue
        difference = scores[good][:, None] - scores[bad][None, :]
        concordant += float((difference > 0).sum()) + 0.5 * float((difference == 0).sum())
        total += int(good.size * bad.size)
    return {
        "unit": unit,
        "comparable_pairs": total,
        "accuracy": float(concordant / total) if total else float("nan"),
    }


def _prefix_accuracy(frame: pd.DataFrame, arm: str, coverage: float) -> dict[str, Any]:
    """Pairwise accuracy inside an arm's OWN top-coverage prefix.

    Failure test 3. A ranking can improve globally and be no better where the cut actually
    lands, and the whole operational question lives in the prefix. The prefix is a fixed SHARE
    of the block rather than a fixed threshold, so two arms on different score scales are
    compared on the same number of rows.

    Three quantities, and the last two are the ones to read. Pairwise accuracy INSIDE the prefix
    is confounded by the prefix's own purity: a ranking good enough to keep almost every harmful
    row out admits only the few that sit closest to the boundary, and those few then outrank many
    beneficial rows within the prefix, so a BETTER ranking can score worse on it. The prefix's
    harm rate and its repair recall are not confounded that way, and they are what the frontier
    is a function of.
    """
    harmful, beneficial = _labels(frame)
    scores = _arm(frame, arm)
    take = int(np.clip(round(coverage * scores.size), 1, scores.size))
    keep = np.argsort(-scores, kind="stable")[:take]
    inside = harmful[keep] | beneficial[keep]
    return {
        "coverage": float(coverage),
        "rows": take,
        "decided_rows": int(inside.sum()),
        "prefix_ids": frozenset(frame["candidate_id"].to_numpy()[keep].tolist()),
        "harm_rate": float(harmful[keep].mean()),
        "beneficial_captured": int(beneficial[keep].sum()),
        "beneficial_recall": float(beneficial[keep].sum() / max(int(beneficial.sum()), 1)),
        "accuracy": (
            roc_auc(scores[keep][inside], beneficial[keep][inside].astype(float))
            if int(inside.sum()) > 0
            else float("nan")
        ),
    }


def _prefix_row(
    frame: pd.DataFrame,
    arm: str,
    coverage: float,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    ceiling_prefix: frozenset[str],
) -> dict[str, Any]:
    """One arm's prefix, with how much of it the ceiling would have chosen differently.

    The overlap is the sharpest form of the measurement: it says how many rows have to change
    hands for the whole score-transfer gap to close. The prefix identifiers themselves are used
    to compute it and are not written to the artifact -- they would be tens of thousands of
    strings for a quantity that is one number.
    """
    block = _prefix_accuracy(frame, arm, coverage)
    prefix = block.pop("prefix_ids")
    block["global_pair_accuracy"] = ranking_quality(_arm(frame, arm), harmful, beneficial)[
        "pair_accuracy"
    ]
    block["prefix_overlap_with_ceiling"] = (
        float(len(prefix & ceiling_prefix) / max(len(prefix), 1)) if ceiling_prefix else 1.0
    )
    return block


def run_negative() -> int:
    """The brief's ten failure tests, plus the five leakage assertions the project requires.

    Every one of them is written so that a PASS is the interesting outcome only if the method
    also won somewhere: a control that matches the method is reported as a control that matched
    the method, never re-labelled as a robustness check that the method survived.
    """
    started = time.monotonic()
    frame, record = load_rows()
    engines = _engines(record)
    selection = cc._read_json(MODEL_SELECTION)["engines"]
    published = cc._read_json(s11.DECISION_TRANSFER)["engines"]
    tests: dict[str, Any] = {}

    # 1. Does the arm's score encode which engine produced the row?
    identity: dict[str, Any] = {}
    for engine in engines:
        seen = block_rows(frame, engine, SEEN_EVAL)
        sources = sorted(seen["engine_id"].astype(str).unique())
        identity[engine] = {
            arm: {
                source: roc_auc(
                    _arm(seen, arm), (seen["engine_id"].to_numpy(str) == source).astype(float)
                )
                for source in sources
            }
            for arm in REPORTED_ARMS
        }
    tests["1_engine_identity_shortcut"] = {
        "question": "does the score separate one source engine from the others",
        "measure": "AUROC of the arm's score for 'this row came from engine X', on setting A rows",
        "per_engine": identity,
        "max_engine_separation": {
            arm: float(np.max([max(identity[e][arm].values()) for e in engines]))
            for arm in REPORTED_ARMS
        },
        # An AUROC of 0.2 carries exactly as much engine information as one of 0.8 -- it puts the
        # engine at the bottom instead of the top -- so the summary that answers the question is
        # the distance from a half, not the raw maximum.
        "max_distance_from_half": {
            arm: float(
                np.max(
                    [
                        max(abs(float(value) - 0.5) for value in identity[e][arm].values())
                        for e in engines
                    ]
                )
            )
            for arm in REPORTED_ARMS
        },
        "reading": (
            "0.5 is no engine information and 0.0 distance from a half is none either. A large "
            "distance says the ordering is partly an ordering of engines, which is the shortcut "
            "families P1 and P2 exist to block."
        ),
    }

    # 2. Does any advantage survive being restricted to comparisons inside one page or one site?
    matched: dict[str, Any] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        matched[engine] = {
            arm: {
                "document": _within_unit_accuracy(evaluation, arm, "document_id"),
                "site": _within_unit_accuracy(evaluation, arm, "site_id"),
            }
            for arm in REPORTED_ARMS
        }
    tests["2_matched_unit_accuracy"] = {
        "question": "does the improvement disappear when every comparison is local",
        "per_engine": matched,
        "mean_document_accuracy": {
            arm: float(np.mean([matched[e][arm]["document"]["accuracy"] for e in engines]))
            for arm in REPORTED_ARMS
        },
        "mean_site_accuracy": {
            arm: float(np.nanmean([matched[e][arm]["site"]["accuracy"] for e in engines]))
            for arm in REPORTED_ARMS
        },
    }

    # 3. Global order against order inside the prefix the cut actually lands in.
    prefix: dict[str, Any] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        calibration = block_rows(frame, engine, CALIBRATION)
        harmful, beneficial = _labels(evaluation)
        cut = _cut_for(PRIMARY_CONTROLLER, calibration, SGV5_FROZEN, PRIMARY_EPSILON)
        point = deployed(
            _arm(evaluation, SGV5_FROZEN), harmful, beneficial, cut["tau"], PRIMARY_EPSILON
        )
        coverage = float(point["coverage"]) if not point["accepts_nothing"] else 0.1
        ceiling_prefix = _prefix_accuracy(evaluation, ceiling_arm(PRIMARY_KEY), coverage)[
            "prefix_ids"
        ]
        prefix[engine] = {
            "reference_coverage": coverage,
            "arms": {
                arm: _prefix_row(evaluation, arm, coverage, harmful, beneficial, ceiling_prefix)
                # The CEILING is included here and nowhere else in this phase. It is the one
                # ranking known to close the score-transfer gap, so what its prefix looks like is
                # the measurement that says what closing the gap consists of.
                for arm in (*REPORTED_ARMS, ceiling_arm(PRIMARY_KEY))
            },
        }
    tests["3_global_versus_prefix"] = {
        "question": "does the ordering improve where the cut lands, or only far from it",
        "prefix_definition": (
            "each arm's own top rows, at the acceptance share SGV5's published deployment takes "
            "on the same block at the primary epsilon"
        ),
        "reading": (
            "read `harm_rate` and `beneficial_recall`, not `accuracy`. Pairwise accuracy inside "
            "the prefix falls as the prefix gets purer, because the few harmful rows that remain "
            "are the ones ranked highest among harmful; the ceiling arm makes that plain. "
            "`prefix_overlap_with_ceiling` is the share of an arm's accepted rows that the "
            "ceiling would also accept at the same coverage: it says how few rows have to change "
            "hands for the whole score-transfer gap to close."
        ),
        "per_engine": prefix,
    }

    # 4 and 5. What group robustness costs the groups that were already fine, and whether the
    # worst group only improved because the arm accepted less.
    sacrifice: dict[str, Any] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        calibration = block_rows(frame, engine, CALIBRATION)
        harmful, beneficial = _labels(evaluation)
        labels = np.array([f"s{int(s)}" for s in evaluation["shard"].tolist()], dtype=object)
        robust = group_quality(_arm(evaluation, GROUP_DRO), harmful, beneficial, labels)
        pooled = group_quality(_arm(evaluation, PAIR_LINEAR), harmful, beneficial, labels)
        shared = sorted(set(robust["groups"]) & set(pooled["groups"]))
        deltas = {
            name: float(robust["groups"][name]["auroc_safe"] - pooled["groups"][name]["auroc_safe"])
            for name in shared
        }
        coverage = {}
        for name, arm in (("group_dro", GROUP_DRO), ("pooled", PAIR_LINEAR)):
            cut = _cut_for(PRIMARY_CONTROLLER, calibration, arm, PRIMARY_EPSILON)
            point = deployed(
                _arm(evaluation, arm), harmful, beneficial, cut["tau"], PRIMARY_EPSILON
            )
            coverage[name] = float(point["coverage"])
        sacrifice[engine] = {
            "per_shard_delta_auroc": deltas,
            "best_shard_delta": float(min(deltas.values())) if deltas else float("nan"),
            "worst_shard_delta": float(robust["worst_auroc"] - pooled["worst_auroc"]),
            "coverage": coverage,
            "delta_coverage": float(coverage["group_dro"] - coverage["pooled"]),
        }
    tests["4_group_dro_sacrifices_easy_groups"] = {
        "question": "does the robust objective buy the worst shard by giving up the others",
        "per_engine": sacrifice,
        "shards_made_worse": {
            engine: sum(
                1 for value in sacrifice[engine]["per_shard_delta_auroc"].values() if value < 0
            )
            for engine in engines
        },
    }
    tests["5_worst_group_gain_from_coverage"] = {
        "question": "is the worst-shard gain just a smaller accept set",
        "per_engine": {
            engine: {
                "worst_shard_delta_auroc": sacrifice[engine]["worst_shard_delta"],
                "delta_coverage": sacrifice[engine]["delta_coverage"],
            }
            for engine in engines
        },
        "note": (
            "the ranking metric does not depend on the cut at all, so a worst-shard AUROC gain "
            "cannot be bought with coverage. The coverage delta is reported beside it so a "
            "reader can see that the two moved independently."
        ),
    }

    # 6, 7 and 8: the outlier, the selection, and the block shift.
    tests["6_easyocr_outlier"] = {
        "question": "is EasyOCR still the engine nothing transfers to",
        "per_arm_auroc": {
            arm: {
                engine: ranking_quality(
                    _arm(block_rows(frame, engine, EVALUATION), arm),
                    *_labels(block_rows(frame, engine, EVALUATION)),
                )["auroc_safe"]
                for engine in engines
            }
            for arm in (SGV5_FROZEN, SELECTED, GROUP_DRO)
        },
        "detail": "the full diagnostic is in easyocr_analysis.json",
    }
    oracle_selection: dict[str, Any] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        harmful, beneficial = _labels(evaluation)
        scored = {
            arm: float(ranking_quality(_arm(evaluation, arm), harmful, beneficial)["pair_accuracy"])
            for arm in PROPOSED
        }
        best = max(scored, key=lambda name: (scored[name], -PROPOSED.index(name)))
        chosen = str(selection[engine]["selected"]["method"])
        oracle_selection[engine] = {
            "source_validated": chosen,
            "best_on_the_evaluation_block": best,
            "agree": bool(chosen == best),
            "pair_accuracy_selected": scored[chosen],
            "pair_accuracy_best": scored[best],
            "cost_of_selecting_blind": float(scored[best] - scored[chosen]),
        }
    tests["7_source_validation_picks_the_wrong_method"] = {
        "question": "does a leakage-free selection rule choose the method the target would have",
        "per_engine": oracle_selection,
        "folds_agreeing": int(sum(row["agree"] for row in oracle_selection.values())),
        "note": (
            "the best-on-evaluation column is an ORACLE and is reported for exactly this "
            "diagnostic. It selects nothing."
        ),
    }
    collapse: dict[str, Any] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        harmful, beneficial = _labels(evaluation)
        labels = np.array([f"s{int(s)}" for s in evaluation["shard"].tolist()], dtype=object)
        collapse[engine] = {
            arm: {
                "pooled_auroc": ranking_quality(_arm(evaluation, arm), harmful, beneficial)[
                    "auroc_safe"
                ],
                "worst_shard_auroc": group_quality(
                    _arm(evaluation, arm), harmful, beneficial, labels
                )["worst_auroc"],
            }
            for arm in REPORTED_ARMS
        }
    tests["8_within_engine_block_collapse"] = {
        "question": "does the ordering hold up on the worst page composition inside one engine",
        "per_engine": collapse,
        "mean_pooled_minus_worst": {
            arm: float(
                np.mean(
                    [
                        collapse[e][arm]["pooled_auroc"] - collapse[e][arm]["worst_shard_auroc"]
                        for e in engines
                    ]
                )
            )
            for arm in REPORTED_ARMS
        },
    }

    # 9. The rule is fixed by construction; what is measured is whether the SIGN of every
    # comparison survives being read under the other two rules.
    consistency: dict[str, Any] = {}
    for arm in REPORTED_ARMS:
        signs: dict[str, Any] = {}
        for rule in DEPLOY_RULES:
            deltas = []
            for engine in engines:
                evaluation = block_rows(frame, engine, EVALUATION)
                calibration = block_rows(frame, engine, CALIBRATION)
                harmful, beneficial = _labels(evaluation)
                values = []
                for name in (arm, SGV5_FROZEN):
                    cut = _cut_for(rule, calibration, name, PRIMARY_EPSILON)
                    point = deployed(
                        _arm(evaluation, name), harmful, beneficial, cut["tau"], PRIMARY_EPSILON
                    )
                    values.append(float(point["risk_controlled_repair_recall"]))
                deltas.append(values[0] - values[1])
            signs[rule] = {
                "mean_delta": float(np.mean(deltas)),
                "engines_improving": int(sum(value > 0 for value in deltas)),
            }
        consistency[arm] = signs
    tests["9_survives_a_different_decision_rule"] = {
        "question": "does the comparison hold under the other two cut rules",
        "note": (
            "the primary rule is identical for every arm by construction, so this test cannot "
            "detect a decision-layer advantage; what it can detect is a comparison that only "
            "exists at one controller."
        ),
        "per_arm": consistency,
    }

    tests["10_shuffled_group_control"] = {
        "question": "does destroying the environment structure change anything",
        "contrast": contrast(frame, engines, DRO_SHUFFLED, GROUP_DRO),
    }

    # The five leakage assertions.
    leakage: dict[str, Any] = {}
    excluded = [name for name in (ORACLE_RERANK, *CEILING_ARMS) if name in DEPLOYABLE_ARMS]
    if excluded:
        raise PhaseError(f"an oracle arm is inside the deployable set: {excluded}")
    orientation = {}
    for engine in engines:
        fit = block_rows(frame, engine, FIT)
        harmful, beneficial = _labels(fit)
        for arm in REPORTED_ARMS:
            value = roc_auc(_arm(fit, arm), (~harmful).astype(float))
            orientation[f"{engine}|{arm}"] = float(value)
            if not (value > 0.5):
                raise PhaseError(
                    f"{engine}/{arm}: the fitted score does not put safe rows above harmful ones "
                    f"on its own fit block (AUROC {value:g}); the orientation convention is broken"
                )
    isolation = {}
    for engine in engines:
        evaluation = set(block_rows(frame, engine, EVALUATION)["document_id"].astype(str))
        for block in (FIT, CALIBRATION, POOL):
            overlap = evaluation & set(block_rows(frame, engine, block)["document_id"].astype(str))
            isolation[f"{engine}|{block}"] = len(overlap)
            if overlap:
                raise PhaseError(f"{engine}: the {block} block shares pages with the evaluation")
    metadata = {
        "candidate_id",
        "site_id",
        "document_id",
        "engine_id",
        "role",
        "outcome",
        "is_harmful",
        "beneficial",
    }
    engine_columns = [
        name
        for name in pd.read_parquet(dg.DESIGN_MATRIX).columns
        if name not in metadata and "engine" in name
    ]
    if engine_columns:
        raise PhaseError(f"the representation carries an engine column: {engine_columns}")
    baseline_match = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        calibration = block_rows(frame, engine, CALIBRATION)
        harmful, beneficial = _labels(evaluation)
        for epsilon in EPSILONS:
            key = epsilon_key(epsilon)
            cut = _cut_for(PRIMARY_CONTROLLER, calibration, SGV5_FROZEN, epsilon)
            point = deployed(
                _arm(evaluation, SGV5_FROZEN), harmful, beneficial, cut["tau"], epsilon
            )
            reference = published[engine][f"{s11.RANK_SGV5}|{key}"]["policies"][
                f"{s11.SGV5_PUBLISHED}|{s11.TRANSPORT_VALUE}"
            ]["deployed"]
            difference = abs(float(point["risk_controlled_repair_recall"]) - float(reference))
            baseline_match[f"{engine}|{key}"] = difference
            if difference > 1e-12:
                raise PhaseError(
                    f"{engine}/{key}: this stage's baseline deployment does not reproduce SGV11's "
                    f"published one (difference {difference:g})"
                )
    leakage = {
        "oracle_arms_excluded_from_the_deployable_set": True,
        "score_orientation_on_the_fit_block": orientation,
        "evaluation_documents_shared_with_a_fitting_block": isolation,
        "engine_identity_columns_in_the_record": engine_columns,
        "baseline_reproduces_sgv11_published_deployment": baseline_match,
    }
    _write(
        NEGATIVE_TESTS,
        {
            "stage": "SGV12 -- the ten failure tests and the five leakage assertions",
            "intent": (
                "these are attempts to falsify the mechanism, not confirmations of it. A control "
                "that matches the method is reported as a control that matched the method."
            ),
            "failure_tests": tests,
            "leakage_assertions": leakage,
        },
        "sgv12-negative-v1",
    )
    elapsed = time.monotonic() - started
    print(f"negative: {len(tests)} failure tests, {len(leakage)} assertions ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --easyocr

EASYOCR = "easyocr"


def _operation_of(candidates: np.ndarray) -> np.ndarray:
    """Which edit operation the generator proposed, from SGV1's frozen one-hot columns."""
    design = pd.read_parquet(dg.DESIGN_MATRIX)
    names = [name for name in design.columns if name.startswith("prov_operation_")]
    indexed = design.set_index(design["candidate_id"].astype(str))
    block = indexed.loc[[str(name) for name in candidates.tolist()], names].to_numpy(dtype=float)
    chosen = np.argmax(block, axis=1)
    return np.array(
        [names[i].removeprefix("prov_operation_") for i in chosen.tolist()], dtype=object
    )


def _neighbourhood(engine: str) -> dict[str, Any]:
    """SGV5's frozen retrieval block: how far the target rows sit from the source neighbourhood.

    Read from SGV5's published predictions rather than recomputed. These four columns are the
    only description of "source-neighbourhood composition" this project has, and they were
    fitted on the source engines' rows, so a large distance on the target side is exactly the
    statement that the target engine proposes edits the source engines did not.
    """
    frame = pd.read_parquet(c5.PREDICTIONS)
    block = frame[frame["held_out_engine"] == engine]
    out: dict[str, Any] = {}
    for mode, label in (
        (s6.CROSS_ENGINE, "held_out_engine_evaluation"),
        (c5.SOURCE_CALIBRATION, "source_calibration"),
    ):
        rows = block[block["evaluation_mode"] == mode]
        out[label] = {
            "rows": len(rows),
            "mean_knn_distance": float(rows["retr_knn_distance"].mean()),
            "mean_knn_agreement": float(rows["retr_knn_agreement"].mean()),
            "mean_knn_harmful_rate": float(rows["retr_knn_harmful_rate"].mean()),
            "share_with_a_seen_signature": float(rows["retr_signature_seen"].mean()),
        }
    out["distance_inflation"] = float(
        out["held_out_engine_evaluation"]["mean_knn_distance"]
        - out["source_calibration"]["mean_knn_distance"]
    )
    return out


def _inversions(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    """How many beneficial-versus-harmful comparisons this ordering gets backwards."""
    harmful, beneficial = _labels(frame)
    scores = _arm(frame, arm)
    good, bad = scores[beneficial], scores[harmful]
    if good.size == 0 or bad.size == 0:
        return {"pairs": 0, "inversions": 0, "inversion_rate": float("nan")}
    difference = good[:, None] - bad[None, :]
    inverted = int((difference < 0).sum())
    tied = int((difference == 0).sum())
    total = int(good.size * bad.size)
    return {
        "pairs": total,
        "inversions": inverted,
        "ties": tied,
        "inversion_rate": float((inverted + 0.5 * tied) / total),
    }


def _per_document_accuracy(frame: pd.DataFrame, arm: str) -> dict[str, float]:
    harmful, beneficial = _labels(frame)
    scores = _arm(frame, arm)
    members = _by_key(frame["document_id"].astype(str).to_numpy())
    out: dict[str, float] = {}
    for name, rows in sorted(members.items()):
        good, bad = scores[rows[beneficial[rows]]], scores[rows[harmful[rows]]]
        if good.size == 0 or bad.size == 0:
            continue
        difference = good[:, None] - bad[None, :]
        out[name] = float(
            (float((difference > 0).sum()) + 0.5 * float((difference == 0).sum()))
            / float(good.size * bad.size)
        )
    return out


def run_easyocr() -> int:
    """The pre-specified per-engine diagnostic, computed for all four and read for EasyOCR.

    Computing it for one engine would make every number unreadable: 0.66 harm prevalence is only
    interesting beside docTR's 0.75 and Tesseract's 0.39. The section names EasyOCR because
    every stage since SGV1 has found it the engine nothing transfers to, and the question is
    which of the four candidate mechanisms that failure actually is.
    """
    from scipy.stats import spearmanr

    started = time.monotonic()
    frame, record = load_rows()
    engines = _engines(record)
    cells: dict[str, Any] = {}
    per_document: dict[str, dict[str, float]] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        calibration = block_rows(frame, engine, CALIBRATION)
        harmful, beneficial = _labels(evaluation)
        sites = evaluation["site_id"].astype(str).to_numpy()
        members = _by_key(sites)
        both = sum(
            1
            for rows in members.values()
            if bool(beneficial[rows].any()) and bool(harmful[rows].any())
        )
        operations = _operation_of(evaluation["candidate_id"].to_numpy())
        by_operation = {
            str(operation): {
                arm: ranking_quality(
                    _arm(evaluation, arm)[operations == operation],
                    harmful[operations == operation],
                    beneficial[operations == operation],
                )["pair_accuracy"]
                for arm in (SGV5_FROZEN, SELECTED, GROUP_DRO)
            }
            | {"rows": int((operations == operation).sum())}
            for operation in sorted(set(operations.tolist()))
        }
        distribution = {
            arm: {
                "calibration_quantiles": [
                    float(q) for q in np.quantile(_arm(calibration, arm), [0.1, 0.5, 0.9])
                ],
                "evaluation_quantiles": [
                    float(q) for q in np.quantile(_arm(evaluation, arm), [0.1, 0.5, 0.9])
                ],
                "median_shift": float(
                    np.median(_arm(evaluation, arm)) - np.median(_arm(calibration, arm))
                ),
            }
            for arm in (SGV5_FROZEN, SELECTED, GROUP_DRO)
        }
        per_document[engine] = _per_document_accuracy(evaluation, SGV5_FROZEN)
        cells[engine] = {
            "class_prevalence": {
                block: {
                    "rows": len(block_rows(frame, engine, block)),
                    "harm": float(_labels(block_rows(frame, engine, block))[0].mean()),
                    "beneficial": float(_labels(block_rows(frame, engine, block))[1].mean()),
                }
                for block in (CALIBRATION, POOL, EVALUATION)
            },
            "site_ambiguity": {
                "sites": len(members),
                "sites_offering_both_outcomes": both,
                "share": float(both / max(len(members), 1)),
                "within_site_accuracy": {
                    arm: _within_unit_accuracy(evaluation, arm, "site_id")["accuracy"]
                    for arm in (SGV5_FROZEN, SELECTED, GROUP_DRO)
                },
            },
            "source_neighbourhood": _neighbourhood(engine),
            "ranking_inversions": {
                arm: _inversions(evaluation, arm) for arm in (SGV5_FROZEN, SELECTED, GROUP_DRO)
            },
            "by_candidate_type": by_operation,
            "score_distribution": distribution,
        }

    shared: dict[str, Any] = {}
    for left in engines:
        for right in engines:
            if left >= right:
                continue
            common = sorted(set(per_document[left]) & set(per_document[right]))
            if len(common) < 3:
                continue
            a = [per_document[left][name] for name in common]
            b = [per_document[right][name] for name in common]
            correlation = spearmanr(a, b)
            shared[f"{left}|{right}"] = {
                "shared_documents": len(common),
                "spearman": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
                "mean_difference": float(np.mean(a) - np.mean(b)),
            }
    _write(
        EASYOCR_ANALYSIS,
        {
            "stage": "SGV12 -- the pre-specified per-engine diagnostic",
            "focus": EASYOCR,
            "question": (
                "is EasyOCR's failure a rank reversal, a class-mechanism mismatch, an "
                "insufficient representation, a block composition effect, or a genuine unseen "
                "engine mechanism shift"
            ),
            "note": (
                "computed identically for all four engines. Nothing here was used to train, "
                "select or threshold anything: it reads the published score table and SGV5's "
                "frozen retrieval block after every arm was already fitted."
            ),
            "engines": cells,
            "cross_engine_document_agreement": {
                "definition": (
                    "per-document within-document pairwise accuracy of SGV5's frozen ranking, "
                    "correlated across engine pairs on the DEVELOPMENT pages both engines read"
                ),
                "pairs": shared,
            },
        },
        "sgv12-easyocr-v1",
    )
    elapsed = time.monotonic() - started
    print(f"easyocr: {len(engines)} engines, {len(shared)} engine pairs ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --stats


def _document_codes(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    names = sorted({str(name) for name in frame["document_id"].tolist()})
    position = {name: index for index, name in enumerate(names)}
    codes = np.array([position[str(name)] for name in frame["document_id"].tolist()], dtype=np.intp)
    return codes, len(names)


def _segment_sum(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of `values` into `n_groups` buckets given by `codes`.

    `np.add.at` would do this in one line and is unbuffered, which on the concordance block --
    a few thousand rows by a few thousand columns -- costs minutes rather than seconds. Sorting
    once and using `reduceat` gives the same answer at the speed of a contiguous reduction.
    """
    block = np.asarray(values, dtype=float)
    out = np.zeros((n_groups, block.shape[1]), dtype=float)
    if block.shape[0] == 0:
        return out
    order = np.argsort(codes, kind="stable")
    ordered = codes[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    out[ordered[starts]] = np.add.reduceat(block[order], starts, axis=0)
    return out


def auc_form(
    scores: np.ndarray, positive: np.ndarray, codes: np.ndarray, n_documents: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Mann-Whitney statistic as a quadratic form in the per-document multiplicities.

    A document-clustered bootstrap of an AUROC has to recompute a rank statistic on every
    resample, and doing that two thousand times per cell by re-sorting the block is minutes of
    work for a number that is exactly `m' C m / (m'p)(m'q)`: the concordance count decomposes
    over the pair of documents the two rows come from, so one 133-by-133 matrix computed once
    answers every resample with a matrix product. `tests/leakage` checks the identity against
    `metrics.discrimination.roc_auc` on the all-ones multiplicity and against a direct
    recomputation on drawn resamples.
    """
    values = np.asarray(scores, dtype=float)
    mask = np.asarray(positive, dtype=bool)
    good, bad = np.flatnonzero(mask), np.flatnonzero(~mask)
    counts_good = np.bincount(codes[good], minlength=n_documents).astype(float)
    counts_bad = np.bincount(codes[bad], minlength=n_documents).astype(float)
    if good.size == 0 or bad.size == 0:
        return np.zeros((n_documents, n_documents)), counts_good, counts_bad
    difference = values[good][:, None] - values[bad][None, :]
    concordant = (difference > 0).astype(float) + 0.5 * (difference == 0).astype(float)
    partial = _segment_sum(concordant, codes[good], n_documents)
    matrix = _segment_sum(partial.T, codes[bad], n_documents).T
    return matrix, counts_good, counts_bad


def auc_draws(
    form: tuple[np.ndarray, np.ndarray, np.ndarray], multiplicity: np.ndarray
) -> np.ndarray:
    matrix, counts_good, counts_bad = form
    weights = multiplicity.astype(float)
    numerator = np.einsum("rd,de,re->r", weights, matrix, weights)
    denominator = (weights @ counts_good) * (weights @ counts_bad)
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), np.nan)


def deployed_draws(
    accepted: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    codes: np.ndarray,
    n_documents: int,
    multiplicity: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """The risk-controlled repair recall on every resample, at a FIXED accept set.

    The threshold is not re-chosen inside the resample: it came from the source engines and an
    operator would carry that one number to the next batch of pages. Identical in construction
    to SGV11's `resampled_actions`, so the two stages' intervals mean the same thing.
    """
    take = np.asarray(accepted, dtype=float)
    columns = np.column_stack(
        [
            take,
            take * np.asarray(harmful, dtype=float),
            take * np.asarray(beneficial, dtype=float),
            np.asarray(beneficial, dtype=float),
        ]
    )
    per_document = np.zeros((n_documents, 4), dtype=float)
    np.add.at(per_document, codes, columns)
    totals = multiplicity.astype(float) @ per_document
    n_accepted, n_harmful, n_beneficial, n_total = totals.T
    usable = n_accepted > 0
    harm_rate = np.full(n_accepted.size, np.inf)
    harm_rate[usable] = n_harmful[usable] / n_accepted[usable]
    recall = n_beneficial / np.maximum(n_total, 1.0)
    return np.where(usable & (harm_rate <= epsilon), recall, 0.0)


PRIMARY_FAMILY_SIZE = 20


def run_stats() -> int:
    """The pre-registered family: twenty paired document-clustered tests, Holm over all of them.

    The family was fixed before any held-out number existed and is not chosen after the fact:
    the source-validated method against SGV5's frozen ranking, on four unseen engines, on
    held-out AUROC, pairwise ranking accuracy and risk-controlled repair recall at three
    epsilons. Every test is PAIRED -- the same documents, the same draws, the two arms measured
    on the same resample -- which is the comparison the project's own statistics rules require.
    """
    started = time.monotonic()
    frame, record = load_rows()
    engines = _engines(record)
    raw: dict[str, float] = {}
    detail: dict[str, Any] = {}
    for engine in engines:
        evaluation = block_rows(frame, engine, EVALUATION)
        calibration = block_rows(frame, engine, CALIBRATION)
        harmful, beneficial = _labels(evaluation)
        codes, n_documents = _document_codes(evaluation)
        multiplicity = document_multiplicities(
            evaluation["document_id"].to_numpy(str), BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES
        )
        decided = harmful | beneficial
        decided_codes = codes[decided]
        for label, arm_positive, subset in (
            ("auroc", ~harmful, np.ones(len(evaluation), dtype=bool)),
            ("pair_accuracy", beneficial, decided),
        ):
            draws = []
            for arm in (SELECTED, SGV5_FROZEN):
                form = auc_form(
                    _arm(evaluation, arm)[subset],
                    arm_positive[subset],
                    decided_codes if label == "pair_accuracy" else codes,
                    n_documents,
                )
                draws.append(auc_draws(form, multiplicity))
            interval = _interval(draws[0] - draws[1])
            name = f"{label}|{engine}"
            raw[name] = float(interval["p_value"])
            detail[name] = {
                "engine": engine,
                "quantity": label,
                "arm": SELECTED,
                "reference": SGV5_FROZEN,
                "point_estimate": float(np.nanmean(draws[0]) - np.nanmean(draws[1])),
                **interval,
            }
        for epsilon in EPSILONS:
            key = epsilon_key(epsilon)
            draws = []
            for arm in (SELECTED, SGV5_FROZEN):
                cut = _cut_for(PRIMARY_CONTROLLER, calibration, arm, epsilon)
                accepted = _arm(evaluation, arm) >= cut["tau"]
                draws.append(
                    deployed_draws(
                        accepted, harmful, beneficial, codes, n_documents, multiplicity, epsilon
                    )
                )
            interval = _interval(draws[0] - draws[1])
            name = f"risk_controlled|{engine}|{key}"
            raw[name] = float(interval["p_value"])
            detail[name] = {
                "engine": engine,
                "quantity": "risk_controlled_repair_recall",
                "epsilon": float(epsilon),
                "arm": SELECTED,
                "reference": SGV5_FROZEN,
                "point_estimate": float(draws[0].mean() - draws[1].mean()),
                **interval,
            }
    if len(raw) != PRIMARY_FAMILY_SIZE:
        raise PhaseError(
            f"the pre-registered family is {PRIMARY_FAMILY_SIZE} tests and {len(raw)} were run"
        )
    adjusted = holm_bonferroni({name: value for name, value in raw.items() if np.isfinite(value)})
    _write(
        STATISTICAL_TESTS,
        {
            "stage": "SGV12 -- the pre-registered primary family",
            "family": PRE_REGISTRATION["primary_family"],
            "family_size": PRIMARY_FAMILY_SIZE,
            "resampling": {
                "unit": "document",
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "paired": True,
                "note": (
                    "the two arms are measured on the same resample of the same documents, and "
                    "the accept set is held fixed inside the resample because the threshold came "
                    "from the source engines and does not move with the batch"
                ),
            },
            "tests": detail,
            "holm": [
                {
                    "label": row.label,
                    "p_value": row.p_value,
                    "adjusted_p_value": row.adjusted_p_value,
                    "significant": row.significant,
                }
                for row in adjusted
            ],
            "any_significant": bool(any(row.significant for row in adjusted)),
            "smallest_adjusted_p": (
                float(min(row.adjusted_p_value for row in adjusted)) if adjusted else float("nan")
            ),
        },
        "sgv12-stats-v1",
    )
    elapsed = time.monotonic() - started
    print(f"stats: {len(raw)} tests, Holm over the family ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --figures

FIGURE_ARMS = (
    SGV5_FROZEN,
    PAIR_LINEAR,
    PAIR_BOOSTED,
    LISTWISE,
    GROUP_DRO,
    CVAR_RANK,
    VARIANCE_REG,
    SELECTED,
)


def run_figures() -> int:
    """The five figures the brief names, each carrying the development-only annotation."""
    started = time.monotonic()
    ranking = cc._read_json(RANKING_METRICS)
    blocks = cc._read_json(BLOCK_SHIFT)
    frontier_results = cc._read_json(FRONTIER_RESULTS)
    gap = cc._read_json(GAP_DECOMPOSITION)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV12 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(ranking["engines"])
    written: list[Path] = []

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    positions = np.arange(len(engines), dtype=float)
    width = 0.8 / len(FIGURE_ARMS)

    figure, panels = plt.subplots(1, 2, figsize=(13, 4.6))
    for offset, arm in enumerate(FIGURE_ARMS):
        values = [
            ranking["engines"][e]["blocks"][EVALUATION]["arms"][arm]["auroc_safe"] for e in engines
        ]
        panels[0].bar(
            positions + (offset - len(FIGURE_ARMS) / 2 + 0.5) * width,
            values,
            width,
            label=arm.split("_", 1)[0],
        )
        pairs = [
            ranking["engines"][e]["blocks"][EVALUATION]["arms"][arm]["pair_accuracy"]
            for e in engines
        ]
        panels[1].bar(
            positions + (offset - len(FIGURE_ARMS) / 2 + 0.5) * width,
            pairs,
            width,
            label=arm.split("_", 1)[0],
        )
    for panel, label in zip(
        panels, ("held-out AUROC (safe over harmful)", "pairwise ranking accuracy"), strict=True
    ):
        panel.set_xticks(positions)
        panel.set_xticklabels(engines, fontsize=8)
        panel.set_ylabel(label)
        panel.set_ylim(0.5, 1.0)
        panel.axhline(0.5, color="#888", linewidth=0.8)
        panel.legend(fontsize=6, ncol=2)
    finish(
        figure,
        FIGURE_DIR / "heldout_auc.png",
        "Setting C. Held-out ranking quality on each unseen engine, against SGV5's frozen ranking.",
    )

    figure, panel = plt.subplots(figsize=(9, 4.6))
    for offset, arm in enumerate(FIGURE_ARMS):
        values = [
            blocks["engines"][e]["arms"][arm]["setting_b_within_held_out_engine"]["worst_auroc"]
            for e in engines
        ]
        panel.bar(
            positions + (offset - len(FIGURE_ARMS) / 2 + 0.5) * width,
            values,
            width,
            label=arm.split("_", 1)[0],
        )
    panel.set_xticks(positions)
    panel.set_xticklabels(engines, fontsize=8)
    panel.set_ylabel("worst document-shard AUROC")
    panel.set_ylim(0.5, 1.0)
    panel.legend(fontsize=6, ncol=2)
    finish(
        figure,
        FIGURE_DIR / "worst_group_auc.png",
        "Setting B. The worst of four document shards inside each unseen engine.",
    )

    figure, panels = plt.subplots(1, len(EPSILONS), figsize=(14, 4.4), sharey=True)
    for panel, epsilon in zip(np.atleast_1d(panels), EPSILONS, strict=True):
        key = epsilon_key(epsilon)
        for offset, arm in enumerate(FIGURE_ARMS):
            values = [
                frontier_results["engines"][e]["per_epsilon"][key]["arms"][arm]["rules"][
                    PRIMARY_CONTROLLER
                ]["risk_controlled_repair_recall"]
                for e in engines
            ]
            panel.bar(
                positions + (offset - len(FIGURE_ARMS) / 2 + 0.5) * width,
                values,
                width,
                label=arm.split("_", 1)[0],
            )
        panel.set_xticks(positions)
        panel.set_xticklabels(engines, fontsize=8)
        panel.set_title(f"epsilon = {epsilon}", fontsize=10)
        panel.set_ylabel("risk-controlled repair recall")
    np.atleast_1d(panels)[0].legend(fontsize=6, ncol=2)
    finish(
        figure,
        FIGURE_DIR / "risk_coverage_frontier.png",
        "The operational endpoint under one frozen decision rule, identical for every ranking.",
    )

    figure, panel = plt.subplots(figsize=(9.5, 4.8))
    key = PRIMARY_KEY
    ceiling = gap["aggregate"][key]["mean_ceiling_frontier"]
    baseline = gap["aggregate"][key]["mean_baseline_frozen_frontier"]
    names = [SGV5_FROZEN, *PROPOSED, SELECTED]
    frontiers = [gap["aggregate"][key]["arms"][name]["mean_frozen_frontier"] for name in names]
    deployments = [gap["aggregate"][key]["arms"][name]["mean_deployed"] for name in names]
    spots = np.arange(len(names), dtype=float)
    panel.bar(spots - 0.2, frontiers, 0.4, color="#3a5f9e", label="frozen-ranking frontier")
    panel.bar(spots + 0.2, deployments, 0.4, color="#a33", label="deployed")
    panel.axhline(
        ceiling,
        color="#2a7",
        linestyle="--",
        linewidth=1.2,
        label=f"re-ranking ceiling {ceiling:.4f}",
    )
    panel.axhline(
        baseline,
        color="#888",
        linestyle=":",
        linewidth=1.0,
        label=f"SGV5 frozen frontier {baseline:.4f}",
    )
    panel.set_xticks(spots)
    panel.set_xticklabels([name.split("_", 1)[0] for name in names], fontsize=8)
    panel.set_ylabel("repair recall, mean over four unseen engines")
    panel.legend(fontsize=7)
    finish(
        figure,
        FIGURE_DIR / "gap_decomposition.png",
        f"SGV11's ladder at epsilon = {PRIMARY_EPSILON}. The distance to the dashed line is "
        "the score-transfer gap this stage set out to close.",
    )

    figure, panel = plt.subplots(figsize=(9.5, 4.6))
    for offset, arm in enumerate(FIGURE_ARMS):
        values = [
            blocks["engines"][e]["arms"][arm]["cross_block_degradation"]["held_out_engine"][
                "delta_auroc"
            ]
            for e in engines
        ]
        panel.bar(
            positions + (offset - len(FIGURE_ARMS) / 2 + 0.5) * width,
            values,
            width,
            label=arm.split("_", 1)[0],
        )
    panel.axhline(0.0, color="#333", linewidth=0.8)
    panel.set_xticks(positions)
    panel.set_xticklabels(engines, fontsize=8)
    panel.set_ylabel("AUROC on TRAIN pages minus AUROC on DEVELOPMENT pages")
    panel.legend(fontsize=6, ncol=2)
    finish(
        figure,
        FIGURE_DIR / "block_shift.png",
        "Within-engine block shift. Both blocks are out of sample, so a nonzero bar is page "
        "composition alone.",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv12-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": PRE_REGISTRATION["hypothesis_id"],
            "synthetic": False,
            "development_only": True,
            "sources": {
                cc._relative(path): file_sha256(path)
                for path in (RANKING_METRICS, BLOCK_SHIFT, FRONTIER_RESULTS, GAP_DECOMPOSITION)
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


def _finite(value: Any) -> float:
    """A comparison key that treats an undefined statistic as the worst possible one.

    An arm whose metric is undefined on some fold must never win a post-hoc `max`; NaN compares
    false against everything and would otherwise be returned by whichever order the loop happened
    to take.
    """
    number = float(value)
    return number if np.isfinite(number) else float("-inf")


def run_decide() -> int:
    """The five pre-registered criteria, read from the artifacts, and the verdict they imply.

    Nothing is decided here. Every number comes from a file another phase wrote, the criteria
    are the ones the design record fixed before any held-out number existed, and the verdict is
    the arithmetic of counting how many of them hold.
    """
    started = time.monotonic()
    ranking = cc._read_json(RANKING_METRICS)
    blocks = cc._read_json(BLOCK_SHIFT)
    frontier_results = cc._read_json(FRONTIER_RESULTS)
    gap = cc._read_json(GAP_DECOMPOSITION)
    stats = cc._read_json(STATISTICAL_TESTS)
    negative = cc._read_json(NEGATIVE_TESTS)
    selection = cc._read_json(MODEL_SELECTION)["engines"]
    engines = sorted(ranking["engines"])
    key = PRIMARY_KEY

    ranking_tests = [
        row for row in stats["holm"] if row["label"].startswith(("auroc|", "pair_accuracy|"))
    ]
    favourable = [
        row
        for row in ranking_tests
        if row["significant"] and stats["tests"][row["label"]]["point_estimate"] > 0
    ]
    criterion_1 = {
        "statement": PRE_REGISTRATION["criteria"]["1_ranking_improvement"],
        "met": bool(favourable),
        "significant_and_favourable": [row["label"] for row in favourable],
        "smallest_adjusted_p_among_ranking_tests": (
            float(min(row["adjusted_p_value"] for row in ranking_tests))
            if ranking_tests
            else float("nan")
        ),
    }
    aggregate_ranking = ranking["aggregate"][SELECTED]
    criterion_2 = {
        "statement": PRE_REGISTRATION["criteria"]["2_cross_engine_breadth"],
        "met": bool(aggregate_ranking["engines_improving_auroc"] >= ENGINE_BAR),
        "engines_improving_auroc": int(aggregate_ranking["engines_improving_auroc"]),
        "engines_improving_pair_accuracy": int(
            aggregate_ranking["engines_improving_pair_accuracy"]
        ),
        "required": ENGINE_BAR,
    }
    operational = frontier_results["aggregate"][key][SELECTED]
    criterion_3 = {
        "statement": PRE_REGISTRATION["criteria"]["3_operational_frontier"],
        "met": bool(operational["engines_improving_vs_sgv5"] >= ENGINE_BAR),
        "engines_improving": int(operational["engines_improving_vs_sgv5"]),
        "engines_holding_bound": int(operational["engines_holding_bound"]),
        "mean_risk_controlled_repair_recall": float(
            operational["mean_risk_controlled_repair_recall"]
        ),
        "baseline_mean": float(
            frontier_results["aggregate"][key][SGV5_FROZEN]["mean_risk_controlled_repair_recall"]
        ),
        "required": ENGINE_BAR,
    }
    worst_group = blocks["aggregate"][SELECTED]
    criterion_4 = {
        "statement": PRE_REGISTRATION["criteria"]["4_worst_group_robustness"],
        "met": bool(worst_group["engines_improving_worst_shard_vs_pooled"] >= ENGINE_BAR),
        "engines_improving_worst_shard_vs_pooled": int(
            worst_group["engines_improving_worst_shard_vs_pooled"]
        ),
        "engines_improving_worst_shard_vs_sgv5": int(
            worst_group["engines_improving_worst_shard_vs_sgv5"]
        ),
        "mean_worst_shard_auroc": float(worst_group["mean_worst_shard_auroc"]),
        "pooled_reference": PAIR_LINEAR,
        "required": ENGINE_BAR,
        "note": (
            "the reference is the POOLED ranking, as the brief specifies. When the source "
            "validation selects the pooled ranker itself the comparison is an arm against "
            "itself and cannot be met; the count against SGV5 is reported beside it."
        ),
    }
    recovery = float(gap["aggregate"][key]["arms"][SELECTED]["pooled_recovery_fraction"])
    criterion_5 = {
        "statement": PRE_REGISTRATION["criteria"]["5_gap_recovery"],
        "met": bool(np.isfinite(recovery) and recovery >= RECOVERY_THRESHOLD),
        "pooled_recovery_fraction": recovery,
        "threshold": RECOVERY_THRESHOLD,
        "mean_frozen_frontier": float(
            gap["aggregate"][key]["arms"][SELECTED]["mean_frozen_frontier"]
        ),
        "mean_baseline_frozen_frontier": float(
            gap["aggregate"][key]["mean_baseline_frozen_frontier"]
        ),
        "mean_ceiling_frontier": float(gap["aggregate"][key]["mean_ceiling_frontier"]),
        "engines_with_a_higher_frontier": int(
            gap["aggregate"][key]["arms"][SELECTED]["engines_with_a_higher_frontier"]
        ),
    }
    criteria = {
        "1_ranking_improvement": criterion_1,
        "2_cross_engine_breadth": criterion_2,
        "3_operational_frontier": criterion_3,
        "4_worst_group_robustness": criterion_4,
        "5_gap_recovery": criterion_5,
    }
    met = sum(1 for block in criteria.values() if block["met"])
    verdict = "SUPPORTED" if met == len(criteria) else "NOT SUPPORTED"

    ranking_improved = criterion_2["engines_improving_auroc"] >= ENGINE_BAR
    frontier_improved = criterion_3["engines_improving"] >= ENGINE_BAR
    if ranking_improved and not frontier_improved:
        reading = (
            "ranking quality improved, but the improvement was not operationally actionable "
            "under the registered harm constraint."
        )
    elif not ranking_improved:
        reading = (
            "source-only robust ranking was insufficient; the remaining gap likely requires "
            "target-specific supervision or richer cross-environment representation."
        )
    else:
        reading = (
            "both the ordering and the risk-controlled frontier moved in the favourable "
            "direction on the majority of unseen engines."
        )

    posthoc = {
        "note": (
            "everything in this block is POST HOC and feeds no criterion. It is reported so a "
            "reader can see what the stage would have concluded under a different, unregistered "
            "choice, which is exactly the choice that was not made."
        ),
        "best_arm_by_mean_held_out_auroc": max(
            REPORTED_ARMS,
            key=lambda name: _finite(ranking["aggregate"][name]["mean_auroc_setting_c"]),
        ),
        "best_arm_by_mean_risk_controlled_recall": max(
            REPORTED_ARMS,
            key=lambda name: _finite(
                frontier_results["aggregate"][key][name]["mean_risk_controlled_repair_recall"]
            ),
        ),
        "best_arm_by_pooled_recovery_fraction": max(
            REPORTED_ARMS,
            key=lambda name: _finite(
                gap["aggregate"][key]["arms"][name]["pooled_recovery_fraction"]
            ),
        ),
        "source_validation_agreement": negative["failure_tests"][
            "7_source_validation_picks_the_wrong_method"
        ]["folds_agreeing"],
        "oracle_rerank_mean_auroc": float(
            ranking["oracles"][ORACLE_RERANK]["mean_auroc_setting_c"]
        ),
        "sgv10_ceiling_mean_auroc": float(
            ranking["oracles"][ceiling_arm(PRIMARY_KEY)]["mean_auroc_setting_c"]
        ),
    }

    _write(
        DECISION,
        {
            "stage": "SGV12 -- robust cross-engine risk ranking",
            "verdict": verdict,
            "criteria_met": met,
            "criteria_total": len(criteria),
            "criteria": criteria,
            "selected_method": {engine: selection[engine]["selected"] for engine in engines},
            "primary_epsilon": PRIMARY_EPSILON,
            "primary_controller": PRIMARY_CONTROLLER,
            "ranking_effects_by_engine": {
                engine: {
                    "sgv5_auroc": ranking["engines"][engine]["blocks"][EVALUATION]["arms"][
                        SGV5_FROZEN
                    ]["auroc_safe"],
                    "selected_auroc": ranking["engines"][engine]["blocks"][EVALUATION]["arms"][
                        SELECTED
                    ]["auroc_safe"],
                    "delta_auroc": float(
                        ranking["engines"][engine]["blocks"][EVALUATION]["arms"][SELECTED][
                            "auroc_safe"
                        ]
                        - ranking["engines"][engine]["blocks"][EVALUATION]["arms"][SGV5_FROZEN][
                            "auroc_safe"
                        ]
                    ),
                    "delta_pair_accuracy": float(
                        ranking["engines"][engine]["blocks"][EVALUATION]["arms"][SELECTED][
                            "pair_accuracy"
                        ]
                        - ranking["engines"][engine]["blocks"][EVALUATION]["arms"][SGV5_FROZEN][
                            "pair_accuracy"
                        ]
                    ),
                }
                for engine in engines
            },
            "frontier_effects_by_engine": {
                engine: {
                    "sgv5": frontier_results["engines"][engine]["per_epsilon"][key]["arms"][
                        SGV5_FROZEN
                    ]["rules"][PRIMARY_CONTROLLER],
                    "selected": frontier_results["engines"][engine]["per_epsilon"][key]["arms"][
                        SELECTED
                    ]["rules"][PRIMARY_CONTROLLER],
                }
                for engine in engines
            },
            "worst_group_effects": {
                engine: {
                    "sgv5": blocks["engines"][engine]["arms"][SGV5_FROZEN][
                        "setting_b_within_held_out_engine"
                    ]["worst_auroc"],
                    "pooled": blocks["engines"][engine]["arms"][PAIR_LINEAR][
                        "setting_b_within_held_out_engine"
                    ]["worst_auroc"],
                    "selected": blocks["engines"][engine]["arms"][SELECTED][
                        "setting_b_within_held_out_engine"
                    ]["worst_auroc"],
                }
                for engine in engines
            },
            "score_gap_recovery_fraction": {
                epsilon_key(epsilon): float(
                    gap["aggregate"][epsilon_key(epsilon)]["arms"][SELECTED][
                        "pooled_recovery_fraction"
                    ]
                )
                for epsilon in EPSILONS
            },
            "statistical_significance": {
                "family": stats["family"],
                "family_size": stats["family_size"],
                "any_significant_after_holm": bool(stats["any_significant"]),
                "smallest_adjusted_p": stats["smallest_adjusted_p"],
            },
            "reading": reading,
            "bottleneck": (
                "SGV11 measured the score-transfer gap at 0.3065 and the decision-transfer gap "
                "at 0.0779. What this stage measures is how much of the larger one a source-only "
                "robust ranking objective can close, with the decision layer frozen and "
                "identical for every ranking. The recovery fraction above is that number."
            ),
            "limitations": [
                "Four engines and one corpus. A leave-one-engine-out design with four folds "
                "cannot separate an engine effect from an engine-specific corpus effect, and no "
                "impossibility can be inferred from four engines.",
                "The representation is frozen by construction, so this stage bounds what a "
                "better OBJECTIVE can do on SGV5's features and says nothing about what a "
                "better representation could do.",
                "The pair budget is 60,000 of up to 88 million eligible comparisons for the "
                "global family. The budget is swept on source validation only, so its effect on "
                "the held-out numbers is bounded by that sweep rather than measured directly.",
                "The variance-penalised objective is not convex. It is started from the pooled "
                "solution and is a local optimum; a different start reaches a different one, and "
                "the feasibility probe's disclosed reproduction shows the size of that gap.",
                "The oracle re-ranker and the ceiling read the evaluation block's own labels. "
                "They bound what is reachable and are not methods.",
                "DEVELOPMENT rows only. The confirmatory reserve is untouched, so nothing here "
                "is a confirmatory result.",
            ],
            "post_hoc_diagnostics": posthoc,
            "what_is_not_claimed": PRE_REGISTRATION["what_is_not_claimed"],
        },
        "sgv12-decision-v1",
    )
    elapsed = time.monotonic() - started
    print(f"decide: {verdict} -- {met} of {len(criteria)} criteria ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --record

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
)

PRODUCED = (
    SCORES,
    DESIGN_RECORD,
    PAIR_INVENTORY,
    GROUP_INVENTORY,
    MODEL_SELECTION,
    RANKING_METRICS,
    BLOCK_SHIFT,
    FRONTIER_RESULTS,
    GAP_DECOMPOSITION,
    ABLATION_RESULTS,
    NEGATIVE_TESTS,
    EASYOCR_ANALYSIS,
    STATISTICAL_TESTS,
    DECISION,
    FIGURE_MANIFEST,
)

UPSTREAM = (
    dg.DESIGN_MATRIX,
    dg.FIT_RECORD,
    c5.FEATURES,
    c5.PREDICTIONS,
    c5.POLICY_SELECTION,
    c5.FIT_RECORD,
    s9.SCORES,
    s9.PROVENANCE,
    s10.SCORES,
    s10.DESIGN_RECORD,
    s10.PROVENANCE,
    s11.SCORES,
    s11.DECISION_TRANSFER,
    s11.PROVENANCE,
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
        ["git", "check-ignore", "-q", relative],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return "gitignored" if ignored.returncode == 0 else "untracked"


def run_record() -> int:
    """Provenance, the dependency audit the brief asks for, and the traceability index.

    The audit is here rather than in a note because the chain is now thirteen scripts deep and
    none of them is committed. What a reader needs to know is not that the situation is untidy
    but exactly which files a clean clone would be missing and which of them are regenerable,
    and that is a question with a checkable answer.
    """
    started = time.monotonic()
    produced = [path for path in PRODUCED if path.is_file()]
    figures = sorted(FIGURE_DIR.glob("*.png")) if FIGURE_DIR.is_dir() else []
    scripts = {name: _tracked(REPO / name) for name in DEPENDENCY_SCRIPTS}
    artifacts = {cc._relative(path): _tracked(path) for path in UPSTREAM}
    outputs = {cc._relative(path): _tracked(path) for path in (*produced, *figures)}
    clean_clone = {
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
            "data/raw/** -- the recorded OCR responses. Write-once and not committed by policy; "
            "the download scripts, checksums and license records are what the repository ships.",
        ],
        "known_repository_hygiene_issue": (
            "tests/leakage/test_sgv7_policy_selection.py and "
            "tests/leakage/test_sgv8_evidence_aware_deployment.py assert that every file named "
            "in their provenance manifest exists, including gitignored parquet intermediates, so "
            "those two suites would fail on a clean clone. SGV9's, SGV10's, SGV11's and SGV12's "
            "equivalents ask git whether a missing file is deliberately ignored and are clean. "
            "This is recorded, not fixed: changing SGV7 or SGV8 during SGV12 would mean editing "
            "a frozen stage, and the fix belongs to whoever re-verifies those stages."
        ),
    }
    cc._write_json_once(
        PROVENANCE,
        {
            "schema_version": "sgv12-provenance-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": PRE_REGISTRATION["hypothesis_id"],
            "stage": "SGV12 -- robust cross-engine risk ranking",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "features_added": 0,
            "models_fitted_note": (
                "SGV12 fits ranking models, which is the point of the stage. It fits no feature "
                "extractor, no calibrator and no candidate generator, and it retrains nothing "
                "upstream: SGV5's selected model is rebuilt and asserted identical to its "
                "published score before any SGV12 model is fitted."
            ),
            "inputs": {cc._relative(p): file_sha256(p) for p in UPSTREAM if p.is_file()},
            "artifacts": {cc._relative(p): file_sha256(p) for p in (*produced, *figures)},
            "dependency_audit": {
                "scripts": scripts,
                "upstream_artifacts": artifacts,
                "outputs": outputs,
                "clean_clone": clean_clone,
            },
            "regeneration": (
                "uv run python scripts/sgv12_robust_ranking_transfer.py --rank, then "
                "--ranking --blocks --frontier --gap --ablation --negative --easyocr --stats "
                "--figures --decide --record"
            ),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    cc._write_json_once(
        TRACEABILITY,
        {
            "schema_version": "sgv12-traceability-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": PRE_REGISTRATION["hypothesis_id"],
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "note": (
                "which phase writes which artifact, and which report section is written from "
                "it. A number in the report that cannot be reached through this index does not "
                "belong in the report."
            ),
            "phases": {
                "--rank": [
                    cc._relative(SCORES),
                    cc._relative(DESIGN_RECORD),
                    cc._relative(PAIR_INVENTORY),
                    cc._relative(GROUP_INVENTORY),
                    cc._relative(MODEL_SELECTION),
                ],
                "--ranking": [cc._relative(RANKING_METRICS)],
                "--blocks": [cc._relative(BLOCK_SHIFT)],
                "--frontier": [cc._relative(FRONTIER_RESULTS)],
                "--gap": [cc._relative(GAP_DECOMPOSITION)],
                "--ablation": [cc._relative(ABLATION_RESULTS)],
                "--negative": [cc._relative(NEGATIVE_TESTS)],
                "--easyocr": [cc._relative(EASYOCR_ANALYSIS)],
                "--stats": [cc._relative(STATISTICAL_TESTS)],
                "--figures": [cc._relative(FIGURE_MANIFEST)],
                "--decide": [cc._relative(DECISION)],
                "--record": [cc._relative(PROVENANCE), cc._relative(TRACEABILITY)],
            },
            "report_sections": {
                "1. Motivation": [cc._relative(GAP_DECOMPOSITION)],
                "2. Frozen prior evidence": [cc._relative(DESIGN_RECORD)],
                "3. Research questions and hypotheses": [cc._relative(DESIGN_RECORD)],
                "4. Pair construction": [cc._relative(PAIR_INVENTORY)],
                "5. Group construction": [cc._relative(GROUP_INVENTORY)],
                "6. Ranking methods": [cc._relative(DESIGN_RECORD)],
                "7. Nested evaluation protocol": [
                    cc._relative(DESIGN_RECORD),
                    cc._relative(MODEL_SELECTION),
                ],
                "8. Held-out ranking results": [cc._relative(RANKING_METRICS)],
                "9. Within-engine block-shift results": [cc._relative(BLOCK_SHIFT)],
                "10. Risk-controlled frontier results": [cc._relative(FRONTIER_RESULTS)],
                "11. Gap decomposition": [cc._relative(GAP_DECOMPOSITION)],
                "12. Ablations": [cc._relative(ABLATION_RESULTS)],
                "13. EasyOCR failure analysis": [cc._relative(EASYOCR_ANALYSIS)],
                "14. Negative tests": [cc._relative(NEGATIVE_TESTS)],
                "15. Limitations": [cc._relative(DECISION)],
                "16. Research decision": [
                    cc._relative(DECISION),
                    cc._relative(STATISTICAL_TESTS),
                ],
            },
            "artifacts": {cc._relative(p): file_sha256(p) for p in produced},
        },
    )
    print(f"record: {len(produced) + len(figures)} artifacts -> {cc._relative(PROVENANCE)}")
    return 0


# ------------------------------------------------------------------ entry point

PHASES = (
    ("rank", run_rank),
    ("ranking", run_ranking),
    ("blocks", run_blocks),
    ("frontier", run_frontier),
    ("gap", run_gap),
    ("ablation", run_ablation),
    ("negative", run_negative),
    ("easyocr", run_easyocr),
    ("stats", run_stats),
    ("figures", run_figures),
    ("decide", run_decide),
    ("record", run_record),
)


def main(argv: list[str]) -> int:
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
