#!/usr/bin/env python3
"""SGV-RK2: how many target labels does a new correction generator cost before the ranker deploys?

SGV-RK1 closed with outcome B. Its document-level LambdaMART ranker selected OCR corrections better
than SGV-DS1's reliability score on the generators it was fitted on, and failed on a generator it
never saw: mean leave-one-generator-out harm AUROC 0.5583 against a floor of 0.75. RK1 recorded the
mechanism -- evidence that marks a good edit for one generator marks a bad one for the other -- and
named this stage. RK2 measures the label-cost frontier of adapting that ranker to an unseen
generator.

**1. Nothing upstream moves.** The candidate universe, the labels, the R3 features, RK1's ranker and
its frozen hyperparameters, and SGV-RL2's sealed target splits and label-purchase draws are read
from their own artifacts and hash-verified. The purchased labels at every shared budget are RL2's,
row for row, so the two stages differ only in the model being adapted.

**2. Every target label is counted.** A deployable policy needs a threshold as well as a ranking,
and the threshold also costs labels. The budget N is therefore the whole target-label cost: one
purchased site in three is held back to choose the threshold, by a label-blind hash of the site id,
and the ranker is fitted on the rest. No unbudgeted target label reaches any arm.

**3. The test block is sealed first.** The target generator's test rows are RL2's evaluation rows,
on RL1's document folds 3 and 4. They are frozen, with their digest, before any model is fitted, and
they are never purchased, fitted on or used to choose a threshold.

**4. Generator identity never enters a feature vector.** It defines the directions and the splits
and nothing else, in every arm.

    --reconstruct   RK1, RL2 and DS1 re-read from their own artifacts
    --freeze        upstream hashes and the frozen inputs this stage consumes
    --preregister   arms, budgets, the within-budget threshold rule, criteria, outcome rules
    --splits        the sealed target test block and every adaptation block, per direction
    --draws         RL2's purchase draws, reproduced, and the fit/calibration split of each
    --reproduce     RK1's transfer and in-distribution ranker scores, reproduced exactly
    --adapt         every arm x direction x budget x draw, scored on the sealed test block
    --curves        ranking recovery: harm AUROC, benefit AUROC, Top-1, NDCG, gap recovery
    --deployment    the within-budget policy on the sealed test block, and the label-cost frontier
    --compare       source data (joint vs target-only) and adaptation without full retraining
    --controls      the label-permutation control
    --stats         document-clustered paired bootstrap, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --diagnose      POST HOC: why the within-budget threshold misses (feeds no criterion)
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen upstream inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / ADAPTATION ONLY. No candidate is generated, no OCR output is regenerated, no model
family or hyperparameter is searched, nothing is certified and nothing is production-ready. The
confirmatory reserve stays LOCKED and `ready_for_external_confirmation` is false by construction.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_rk1_learning_to_rank as rk1
from ocr_risk.io.hashing import canonical_hash, file_sha256

ds1 = rk1.ds1
rl2 = rk1.rl2
rl1 = rk1.rl1
hy1 = rk1.hy1
gen1 = rk1.gen1
xr1 = rk1.xr1
xc1 = rk1.xc1
s15 = rk1.s15

REPO = rk1.REPO
OUT = REPO / "results/generated/sgv_rk2_fewshot_ranker_adaptation"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"

SPLIT_REGISTRY = OUT / "split_registry.json"
DRAW_REGISTRY = OUT / "draw_registry.json"
LABEL_COMPOSITION = OUT / "label_composition.json"
REPRODUCTION = OUT / "reproduction.json"

ADAPTED_SCORES = OUT / "adapted_scores.parquet"
ADAPTATION_REGISTRY = OUT / "adaptation_registry.json"

RANKING_CURVES = OUT / "ranking_curves.json"
DEPLOYMENT_CURVES = OUT / "deployment_curves.json"
FRONTIER = OUT / "label_cost_frontier.json"
ARM_COMPARISON = OUT / "arm_comparison.json"
THRESHOLD_DIAGNOSTIC = OUT / "threshold_diagnostic.json"
CONTROL_RESULTS = OUT / "control_results.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_rk2/fewshot_ranker_adaptation.md"

SCHEMA_VERSION = 1
STAGE = "sgv_rk2_fewshot_ranker_adaptation"
HYPOTHESIS = "SGV-RK2-D1"
STAGE_KIND = "DEVELOPMENT / ADAPTATION"

PhaseError = rk1.PhaseError
_relative = rk1._relative
_git = rk1._git
_write_json_once = rk1._write_json_once
_write_parquet_once = rk1._write_parquet_once
cc_read_json = rk1.cc_read_json
_require = rk1._require
_forbid = rk1._forbid
auroc = rk1.auroc

BOOTSTRAP_RESAMPLES = rk1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rk1.BOOTSTRAP_SEED
ALPHA = rk1.ALPHA
FIT_SEED = rk1.FIT_SEED

# ------------------------------------------------------------------ the frozen design

C0 = rk1.C0
C1 = rk1.C1
D_G2Q = rl2.D_G2Q
D_Q2G = rl2.D_Q2G
DIRECTIONS = rl2.DIRECTIONS
SOURCE_OF = rl2.SOURCE_OF
TARGET_OF = rl2.TARGET_OF
DIRECTION_LABEL = rl2.DIRECTION_LABEL
DIRECTION_SHORT = {D_G2Q: "current to Qwen", D_Q2G: "Qwen to current"}

# Folds. The source generator keeps DS1's fit and threshold blocks, so the zero-shot arm is RK1's
# own protocol on one generator. The target generator keeps RL2's adaptation and evaluation folds,
# so the purchased labels are RL2's and the test block is RL2's sealed evaluation pool.
SOURCE_FIT_FOLDS = rk1.FIT_FOLDS
SOURCE_CALIBRATION_FOLDS = rk1.THRESHOLD_FOLDS
ADAPTATION_FOLDS = rl2.ADAPTATION_FOLDS
TEST_FOLDS = rl2.EVALUATION_FOLDS

BUDGETS = (0, 25, 50, 100, 250, 500)
DECISION_BUDGET = 100
SATURATION_BUDGET = 500
DRAWS = rl2.DRAWS
INFERENCE_DRAW = rl2.INFERENCE_DRAW

# Within-budget calibration: one purchased site in three chooses the threshold. The assignment
# hashes the site id alone, so it reads no label, and it is the same in every draw and budget, so
# the fit and calibration parts both nest as the budget grows.
CALIBRATION_SEED = "sgv-rk2-within-budget-calibration-v1:2026-09-21"
CALIBRATION_MODULUS = 3

A0 = "a0_zero_shot"
A1 = "a1_target_only"
A2 = "a2_joint"
A3 = "a3_leaf_refit"
ARMS = (A0, A1, A2, A3)
ADAPTED = (A1, A2, A3)
PRIMARY_ARM = A1
ARM_LABEL = {
    A0: "zero shot: RK1's ranker fitted on the source generator alone, no target label",
    A1: "target-only: RK1's ranker fitted on the purchased target labels alone",
    A2: "joint: RK1's ranker fitted on the source rows plus the purchased target labels",
    A3: (
        "leaf refit: the zero-shot ranker's trees kept frozen, only their leaf values "
        "re-estimated on the purchased target labels"
    ),
}

# Deployment, reused from RK1 and DS1 unchanged.
EPSILONS = rk1.EPSILONS
PRIMARY_EPSILON = rk1.PRIMARY_EPSILON
HARM_TOLERANCE = ds1.HARM_TOLERANCE

# Criteria, frozen before any RK2 endpoint. The harm floor is RK1's criterion-B floor.
HARM_FLOOR = rk1.TRANSFER_HARM_FLOOR
DRAW_MAJORITY = 0.5
MIN_RECOVERY_DENOMINATOR = rl2.MIN_RECOVERY_DENOMINATOR
NEAR_RANDOM_BAND = (1.0 - rl1.NEAR_RANDOM, rl1.NEAR_RANDOM)
PERMUTATION_SEED = 20260927

QUESTIONS = {
    "Q1": (
        "how many labelled target candidates are needed before an unseen generator is deployable?"
    ),
    "Q2": "does adding source-generator data help, or cause negative transfer?",
    "Q3": "can adaptation recover ranking performance without full retraining?",
}

OUTCOME_TAXONOMY = {
    "A": "few-shot adaptation makes an unseen generator deployable within the decision budget",
    "B": "an unseen generator becomes deployable in both directions, but only beyond the decision "
    "budget",
    "C": "adaptation makes an unseen generator deployable in one direction only",
    "D": "no tested label budget makes an unseen generator deployable",
}
NEXT_STAGE = {
    "A": "NEXT: TARGETED EXTERNAL CONFIRMATION OF FEW-SHOT GENERATOR ONBOARDING",
    "B": "NEXT: LABEL-EFFICIENT ACQUISITION FOR GENERATOR ONBOARDING",
    "C": "NEXT: DIRECTION-SPECIFIC DIAGNOSIS OF THE GENERATOR THAT DOES NOT ADAPT",
    "D_threshold": "NEXT: WITHIN-BUDGET THRESHOLD CALIBRATION FOR A NEW GENERATOR",
    "D_ranking": "NEXT: RICHER CANDIDATE VERIFICATION EVIDENCE BEFORE MORE LABELS",
}

PRIMARY_FAMILY = (
    "P1_target_only_vs_zero_shot_harm_auroc",
    "P2_target_only_vs_zero_shot_repair_recall",
    "P3_joint_vs_target_only_harm_auroc",
    "P4_leaf_refit_vs_target_only_harm_auroc",
)

UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_rk1": {
        "outcome": "B",
        "recommended_next_stage": "NEXT: GENERATOR ADAPTATION OF THE RANKER UNDER A LABEL BUDGET",
        "production_ready": False,
        "ready_for_external_confirmation": False,
        "confirmatory_reserve_consumed": False,
        "falsification_tests_passed": 26,
        "falsification_tests_total": 26,
        "primary_ranker": rk1.PRIMARY_RANKER,
    },
    "sgv_rl2": {"outcome": "B", "confirmatory_reserve_consumed": False},
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_rk2-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "stage_kind": STAGE_KIND,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str) -> dict[str, Any]:
    return {
        **_envelope(artifact),
        "uses_ground_truth": True,
        "production_ready": False,
        "certified": False,
        "confirmatory_reserve_consumed": False,
    }


def _num(value: Any) -> float | None:
    """A JSON-safe number: None for anything undefined, never NaN."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def recovery(
    adapted: float | None, zero_shot: float | None, reference: float | None
) -> float | None:
    """Share of the zero-shot-to-reference gap that adaptation closes, RL2's definition.

    Undefined, and never a success, when the reference is barely above the zero-shot result: a
    ratio whose denominator is noise would turn "nothing to recover" into "everything recovered".
    """
    if adapted is None or zero_shot is None or reference is None:
        return None
    gap = reference - zero_shot
    if gap < MIN_RECOVERY_DENOMINATOR:
        return None
    return float((adapted - zero_shot) / gap)


def minimal_budget(flags: dict[int, bool]) -> int | None:
    """The smallest tested budget from which a property holds at every larger tested budget.

    A property that holds at 50, fails at 100 and holds again at 250 has a minimal budget of 250:
    a single lucky budget is not a frontier.
    """
    ordered = sorted(flags)
    for position, budget in enumerate(ordered):
        if all(flags[b] for b in ordered[position:]):
            return budget
    return None


def assign_outcome(n_star: dict[str, int | None]) -> str:
    """Exactly one outcome for the primary arm, in the frozen precedence D, C, A, B."""
    reached = [name for name, budget in n_star.items() if budget is not None]
    if not reached:
        return "D"
    if len(reached) < len(n_star):
        return "C"
    if all(budget is not None and budget <= DECISION_BUDGET for budget in n_star.values()):
        return "A"
    return "B"


def next_stage(outcome: str, ranking_ready_both: bool) -> str:
    """D splits by the part that failed: a ranking that recovers but never deploys."""
    if outcome != "D":
        return NEXT_STAGE[outcome]
    return NEXT_STAGE["D_threshold"] if ranking_ready_both else NEXT_STAGE["D_ranking"]


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    files = list(rk1._upstream_files())
    files.append(REPO / "scripts/sgv_rk1_learning_to_rank.py")
    for stage_out in (rk1.OUT, rl2.OUT):
        for pattern in ("*.json", "*.parquet"):
            files.extend(sorted(stage_out.glob(pattern)))
    files.append(rk1.REPORT)
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = xr1._check
    checks = list(rk1.upstream_checks())
    decisions = {"sgv_rk1": cc_read_json(rk1.DECISION), "sgv_rl2": cc_read_json(rl2.DECISION)}
    for stage, expected in UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))
    checks.append(
        _check("sgv_rk1.determinism", cc_read_json(rk1.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_rk1.report_traceability",
            cc_read_json(rk1.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
            0,
        )
    )
    checks.append(
        _check(
            "sgv_rk1.transfer_below_floor",
            bool(cc_read_json(rk1.TRANSFER)["primary_ranker"]["meets_the_floor"]),
            False,
        )
    )
    checks.append(
        _check("sgv_rl2.determinism", cc_read_json(rl2.DETERMINISM)["all_runs_identical"], True)
    )
    return checks


def run_reconstruct() -> int:
    started = time.monotonic()
    _forbid(RESEARCH_FREEZE)
    checks = upstream_checks()
    failed = [c for c in checks if not c["agrees"]]
    if failed:
        raise PhaseError(
            f"{len(failed)} upstream checks failed: {[c['check'] for c in failed][:5]}"
        )
    print(f"reconstruct: {len(checks)} upstream checks pass ({time.monotonic() - started:.0f}s)")
    return 0


def _labels_exist() -> bool:
    return any(p.exists() for p in (ADAPTED_SCORES, RANKING_CURVES, DEPLOYMENT_CURVES, DECISION))


def run_freeze() -> int:
    started = time.monotonic()
    for path in (RESEARCH_FREEZE, FROZEN_CONFIGURATION):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an RK2 endpoint already exists; the freeze must precede every one")
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {_relative(p): file_sha256(p) for p in _upstream_files()}
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "upstream_sha256": hashes,
            "upstream_file_count": len(hashes),
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status": _git("status", "--short"),
                "diff_stat": _git("diff", "--stat"),
            },
            "regenerates_a_candidate": False,
            "regenerates_an_ocr_output": False,
            "changes_the_feature_schema": False,
            "changes_an_upstream_outcome": False,
            "uses_ground_truth": False,
        },
    )
    rk1_design = cc_read_json(rk1.DESIGN_RECORD)
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "ranker": rk1.PRIMARY_RANKER,
            "ranker_label": rk1.MODEL_LABEL[rk1.PRIMARY_RANKER],
            "lambdamart": rk1_design["lambdamart"],
            "relevance_grades": rk1_design["relevance_grades"],
            "representation": rk1.REPRESENTATION,
            "representation_columns": len(rk1.model_columns(rk1.PRIMARY_RANKER)),
            "feature_matrix": _relative(rk1.FEATURE_MATRIX),
            "population": _relative(rl1.CANDIDATE_POPULATION),
            "document_folds": _relative(rl1.GROUP_REGISTRY),
            "rl2_draw_seed": rl2.DRAW_SEED,
            "rl2_draws": rl2.DRAWS,
            "rl2_budgets": list(rl2.BUDGETS),
            "rl2_draw_registry": _relative(rl2.DRAW_REGISTRY),
            "rk1_transfer": cc_read_json(rk1.TRANSFER)["primary_ranker"],
            "rk1_logo_by_split": {
                split: cc_read_json(rk1.TRANSFER)["by_split"][split][rk1.PRIMARY_RANKER]
                for split in rk1.LOGO_SPLITS
            },
            "uses_ground_truth": False,
        },
    )
    print(f"freeze: {len(hashes)} upstream files hashed ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the pre-registration


def run_preregister() -> int:
    """Every choice that could otherwise be made after seeing an endpoint."""
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("an RK2 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "research_question": (
                "how much labelled data is required to adapt a learning-to-rank model to a new "
                "OCR correction generator?"
            ),
            "questions": QUESTIONS,
            "primary_hypothesis": (
                "RK1's ranker carries evidence that transfers once the target generator's own "
                "labels re-fit its decision surface, and a budget of the order of a hundred "
                "labelled target candidates makes the target generator deployable"
            ),
            "frozen_inputs": {
                "population": (
                    "SGV-RL1's primary candidate universe, loaded by SGV-RL2's own loader with "
                    "RL1's document folds attached"
                ),
                "features": (
                    "RK1's feature matrix, restricted to R3's 93 columns; RK1 verified it equal to "
                    "RL1's on every row of this universe"
                ),
                "ranker": (
                    "RK1's primary document-level LambdaMART with its frozen hyperparameters: no "
                    "search, no tuning, the same fitting function"
                ),
                "relevance_grades": "exact 3, partial improvement 2, neutral 1, harmful 0",
            },
            "directions": {
                name: {
                    "source_generator": SOURCE_OF[name],
                    "target_generator": TARGET_OF[name],
                    "label": DIRECTION_LABEL[name],
                }
                for name in DIRECTIONS
            },
            "blocks": {
                "source_fit": f"source rows in RL1 folds {list(SOURCE_FIT_FOLDS)}",
                "source_calibration": (
                    f"source rows in RL1 fold {list(SOURCE_CALIBRATION_FOLDS)}, used only to "
                    "choose the zero-shot arm's threshold"
                ),
                "target_adaptation_pool": (
                    f"target rows in RL1 folds {list(ADAPTATION_FOLDS)}: SGV-RL2's adaptation "
                    "pool, exactly"
                ),
                "target_test": (
                    f"target rows in RL1 folds {list(TEST_FOLDS)}: SGV-RL2's sealed evaluation "
                    "pool, exactly, and the same test folds as SGV-DS1 and SGV-RK1"
                ),
                "site_rule": (
                    "RL1's leave-one-generator-out rule, inherited through RL2: a source row is "
                    "kept only if its site carries no target-generator candidate and a target row "
                    "only if its site carries no source-generator candidate"
                ),
                "sealed": (
                    "the target test rows are frozen, with their digest, before any fit, and are "
                    "never purchased, fitted on or used to choose a threshold"
                ),
            },
            "label_budgets": list(BUDGETS),
            "decision_budget": DECISION_BUDGET,
            "saturation_budget": SATURATION_BUDGET,
            "draws_per_budget": DRAWS,
            "draw_seed": rl2.DRAW_SEED,
            "acquisition_rule": (
                "SGV-RL2's rule and RL2's own seed, unchanged: documents in a seeded hash order, "
                "candidates inside a document by frozen candidate id, labels bought in that order "
                "until the budget is met. The rows bought at every budget shared with RL2 are "
                "therefore RL2's rows, which the draws phase verifies against RL2's digests"
            ),
            "budgets_are_nested": True,
            "within_budget_calibration": {
                "rule": (
                    "a purchased site is a calibration site when a seeded hash of its site id is "
                    f"divisible by {CALIBRATION_MODULUS}; the ranker is fitted on the other "
                    "purchased sites and the threshold is chosen on the calibration sites"
                ),
                "seed": CALIBRATION_SEED,
                "modulus": CALIBRATION_MODULUS,
                "why": (
                    "a deployable policy needs a threshold as well as a ranking, and a threshold "
                    "chosen on labels outside the budget would hide part of the label cost. N is "
                    "therefore the whole target-label cost of deploying"
                ),
                "why_sites_not_documents": (
                    "the smallest budgets buy one or two documents, which cannot be divided by "
                    "document. Fit and calibration sites are site-disjoint but can share a page, "
                    "so the within-budget threshold may be optimistic; the sealed test block, "
                    "which shares no page with either, measures whether it holds"
                ),
                "label_blind": True,
            },
            "arms": ARM_LABEL,
            "primary_arm": PRIMARY_ARM,
            "primary_arm_reason": (
                "SGV-RL2 found that fitting on the target labels alone beat the joint refit at "
                "most budgets, and SGV-RK1 found that the two generators' evidence points in "
                "opposite directions. Both are prior-stage findings; none of this stage's "
                "endpoints informed the choice"
            ),
            "joint_weighting": (
                "equal observation weights. RK1's LambdaMART takes Newton-step leaf values, and "
                "in a leaf holding one generator's rows alone a row weight cancels out of the "
                "step, so a down-weighted arm would only partly down-weight. It is not run"
            ),
            "leaf_refit": (
                "the zero-shot ranker's regression trees are kept, split for split. Their leaf "
                "values are re-estimated from zero on the purchased target labels, tree by tree, "
                "with RK1's own Newton step, so only the leaf values -- at most eight per tree -- "
                "change. A leaf no fitting row reaches contributes nothing. Refitting the leaves "
                "on the source's own training rows reproduces the source ranker exactly"
            ),
            "frozen_hyperparameter_consequence": (
                f"RK1's minimum of {rk1.LAMBDAMART_MIN_LEAF} rows per leaf means a tree needs "
                f"{2 * rk1.LAMBDAMART_MIN_LEAF} fitting rows to split. A target-only fit on fewer "
                "cannot learn anything; its scores are constant and it is recorded as "
                "untrainable. This follows from the frozen configuration and is declared here "
                "rather than tuned around"
            ),
            "threshold_rule": (
                "RK1's rule unchanged: every distinct score on the calibration decisions is a "
                "candidate cutoff, and the loosest cutoff whose calibration harm meets the target "
                "is kept. One decision per site, the ranker's top candidate. The zero-shot arm, "
                "which owns no target label, chooses its cutoff on the source calibration fold"
            ),
            "abstention": (
                "an arm abstains -- accepts nothing and sends every site to review -- when no "
                "cutoff meets the target, when no calibration site was bought, or when its ranker "
                "is untrainable. A constant score is not a ranking and never deploys"
            ),
            "metrics": {
                "harm_auroc": "harm AUROC of the ranker score on the sealed test block",
                "benefit_auroc": "benefit AUROC of the ranker score on the sealed test block",
                "top1": (
                    "share of test choice sites whose top-ranked candidate is exact. Defined only "
                    "where the target generator offers a choice; the image corrector proposes "
                    "one candidate per site, so in that direction Top-1 is undefined, not chance"
                ),
                "repair_recall": (
                    "exact-repair recall over the target-reachable error sites: SGV-HY1's "
                    "evaluable OCR error sites in the test block that LP1's frozen links connect "
                    "to a target test decision site, repaired by an accepted exact edit"
                ),
                "review_rate": "share of the target test decision sites sent to a human",
                "selective_harm": "harmful accepted edits over accepted edits",
                "oracle_threshold": (
                    "analysis only: the loosest cutoff whose TEST harm meets the target. It reads "
                    "test labels, is never deployable, and exists to separate ranking quality from "
                    "threshold estimation"
                ),
                "aggregation": (
                    "median over draws, with quartiles; an untrainable draw is counted, never "
                    "silently dropped"
                ),
            },
            "epsilons": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "deployable": {
                "draw": (
                    f"a draw is deployable when its ranker's test harm AUROC is at least "
                    f"{HARM_FLOOR} and its within-budget policy at harm <= {PRIMARY_EPSILON} "
                    "repairs at least one error site while its realized test harm meets the target"
                ),
                "cell": (
                    f"an arm is deployable at a budget and direction when at least "
                    f"{DRAW_MAJORITY} of its draws are deployable"
                ),
                "minimal_budget": (
                    "N* is the smallest tested budget from which the arm is deployable at every "
                    "larger tested budget; none if it is not deployable at the largest"
                ),
                "ranking_ready_budget": (
                    f"the same, for the ranking condition alone: at least {DRAW_MAJORITY} of "
                    f"draws with test harm AUROC of {HARM_FLOOR} or more"
                ),
                "floor_source": "the harm floor is RK1's criterion-B transfer floor",
            },
            "recovery": (
                "(A_N - A_0) / (A_ref - A_0) on harm AUROC, where A_ref is RK1's in-distribution "
                "primary ranker -- fitted on both generators' fit block -- restricted to exactly "
                "the same test rows. A denominator below "
                f"{MIN_RECOVERY_DENOMINATOR} is unidentifiable. The reference saw target labels "
                "and is not a ceiling"
            ),
            "outcome_rule": OUTCOME_TAXONOMY,
            "outcome_precedence": (
                "on the primary arm's N* in the two directions: D if neither direction reaches "
                "N*, C if exactly one does, A if both reach it at or below the decision budget, "
                "otherwise B"
            ),
            "next_stage": NEXT_STAGE,
            "q2_reading": (
                "negative transfer when, in both directions, target-only beats the joint arm on "
                "median test harm AUROC at a majority of the budgets where both are defined; "
                "source data helps when the joint arm wins a majority in both; otherwise mixed"
            ),
            "q3_reading": (
                "adaptation recovers ranking without full retraining when the leaf-refit arm has "
                "a ranking-ready budget in both directions"
            ),
            "statistical_family": list(PRIMARY_FAMILY),
            "statistical_plan": {
                "unit": "document, resampled within environment",
                "paired": True,
                "budget": DECISION_BUDGET,
                "draw": INFERENCE_DRAW,
                "auroc_statistic": (
                    "the mean over the two directions of each direction's test harm AUROC; one "
                    "document draw serves both directions, which share their test pages"
                ),
                "recall_statistic": (
                    "the pooled share over both directions' target-reachable error sites, "
                    "declared here so that no pooled-versus-averaged choice is made later"
                ),
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "multiplicity": "Holm within the frozen family",
                "undefined_comparison": "recorded with p = 1.0 and never counted as support",
                "budget_curves_are_descriptive": True,
            },
            "controls": {
                "label_permutation": (
                    f"at the saturation budget, the target-only ranker refitted on grades "
                    "shuffled inside each purchased document query; its median test harm AUROC "
                    f"must fall inside {list(NEAR_RANDOM_BAND)}"
                ),
                "seed": PERMUTATION_SEED,
            },
            "disclosure_a_structural_probe_was_run": (
                "before this record was written, a probe read label-blind structure only: block "
                "sizes, candidates per site and per document, and LambdaMART fit times on "
                "adaptation rows. It showed that the image corrector proposes one candidate per "
                "site, so Top-1 is undefined when it is the target. No test-block metric was "
                "computed. RK1's and RL2's published results were read, as prior stages"
            ),
            "non_goals": [
                "no candidate is generated and no OCR output is regenerated",
                "no feature is added to or removed from R3, and no generator identity enters any "
                "feature vector",
                "no model family, hyperparameter or weighting is searched",
                "no certification, finite-sample guarantee or production claim",
                "no source-retention or mixed-stream analysis: the adapted ranker is evaluated on "
                "the target generator it was adapted to",
                "the confirmatory reserve is never unlocked",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the population and blocks


def load_population() -> pd.DataFrame:
    """RL1's primary universe with RL1's folds, RK1's relevance grades and a site key."""
    population = rl2.load_population()
    population["grade"] = rk1.relevance(population)
    population["site_key"] = population["site_group"].astype(str)
    return population


@dataclass(slots=True)
class Blocks:
    """One direction's frozen row blocks, as index arrays into the population."""

    direction: str
    source: str
    target: str
    source_fit: np.ndarray
    source_calibration: np.ndarray
    target_adapt: np.ndarray
    target_test: np.ndarray
    frozen: Any


def build_blocks(name: str, population: pd.DataFrame) -> Blocks:
    """RL2's direction, with its source rows divided into DS1's fit and threshold blocks."""
    frozen = rl2.build_direction(name, population)
    fold = population["fold"].to_numpy(dtype=np.int64)
    source = frozen.source_train
    return Blocks(
        direction=name,
        source=frozen.source,
        target=frozen.target,
        source_fit=source[np.isin(fold[source], SOURCE_FIT_FOLDS)],
        source_calibration=source[np.isin(fold[source], SOURCE_CALIBRATION_FOLDS)],
        target_adapt=frozen.target_adapt,
        target_test=frozen.target_eval,
        frozen=frozen,
    )


def all_blocks(population: pd.DataFrame) -> dict[str, Blocks]:
    return {name: build_blocks(name, population) for name in DIRECTIONS}


def _is_calibration_site(site: str) -> bool:
    return (
        int(canonical_hash({"seed": CALIBRATION_SEED, "site": site})[:8], 16) % CALIBRATION_MODULUS
        == 0
    )


def calibration_mask(sites: Sequence[str]) -> np.ndarray:
    """Which rows sit on a calibration site. It reads the site id and nothing else."""
    verdict = {site: _is_calibration_site(site) for site in sorted(set(sites))}
    return np.asarray([verdict[site] for site in sites], dtype=bool)


def split_purchase(bought: np.ndarray, calibration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The purchased rows, in purchase order, divided into the fit part and the calibration part."""
    marked = calibration[bought]
    return bought[~marked], bought[marked]


def _digest(population: pd.DataFrame, index: np.ndarray) -> str:
    return str(canonical_hash(population.iloc[index]["candidate_id"].astype(str).tolist()))


def _block_summary(population: pd.DataFrame, index: np.ndarray) -> dict[str, Any]:
    block = population.iloc[index]
    return {
        "candidates": int(index.size),
        "documents": int(block["document_id"].nunique()),
        "sites": int(block["site_group"].nunique()),
        "environments": int(block["environment"].nunique()),
        "folds": sorted(int(f) for f in set(block["fold"].tolist())),
    }


def run_splits() -> int:
    """The sealed target test block and every adaptation block, frozen before any fit."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    _forbid(SPLIT_REGISTRY)
    population = load_population()
    calibration = calibration_mask(population["site_group"].astype(str).tolist())
    registry: dict[str, Any] = {}
    leakage: dict[str, Any] = {}
    for name, blocks in all_blocks(population).items():
        parts = {
            "source_fit": blocks.source_fit,
            "source_calibration": blocks.source_calibration,
            "target_adaptation_pool": blocks.target_adapt,
            "target_test": blocks.target_test,
        }
        test = population.iloc[blocks.target_test]
        registry[name] = {
            "source_generator": blocks.source,
            "target_generator": blocks.target,
            **{key: _block_summary(population, index) for key, index in parts.items()},
            "target_test_digest": _digest(population, blocks.target_test),
            "target_test_sites_with_several_candidates": int(
                (test.groupby("site_group").size() >= 2).sum()
            ),
            "adaptation_pool_calibration_sites": int(
                population.iloc[blocks.target_adapt][calibration[blocks.target_adapt]][
                    "site_group"
                ].nunique()
            ),
            "adaptation_pool_calibration_rows": int(calibration[blocks.target_adapt].sum()),
            "equals_rl2_adaptation_pool": bool(
                np.array_equal(blocks.target_adapt, blocks.frozen.target_adapt)
            ),
            "equals_rl2_evaluation_pool": bool(
                np.array_equal(blocks.target_test, blocks.frozen.target_eval)
            ),
        }
        training = np.concatenate(
            [blocks.source_fit, blocks.source_calibration, blocks.target_adapt]
        )

        def crossing(column: str, left: np.ndarray, right: np.ndarray) -> int:
            return len(
                set(population.iloc[left][column].astype(str))
                & set(population.iloc[right][column].astype(str))
            )

        leakage[name] = {
            "documents_training_vs_test": crossing("document_id", training, blocks.target_test),
            "sites_training_vs_test": crossing("site_group", training, blocks.target_test),
            "edit_groups_training_vs_test": crossing("edit_group", training, blocks.target_test),
            "candidates_training_vs_test": crossing("candidate_id", training, blocks.target_test),
            "sites_source_vs_target": crossing(
                "site_group",
                np.concatenate([blocks.source_fit, blocks.source_calibration]),
                np.concatenate([blocks.target_adapt, blocks.target_test]),
            ),
            "documents_source_fit_vs_source_calibration": crossing(
                "document_id", blocks.source_fit, blocks.source_calibration
            ),
        }
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_analysis_envelope("split_registry"),
            "directions": registry,
            "leakage": leakage,
            "nothing_crosses": bool(
                all(value == 0 for block in leakage.values() for value in block.values())
            ),
            "calibration_rule": cc_read_json(DESIGN_RECORD)["within_budget_calibration"]["rule"],
            "test_block_is_sealed": cc_read_json(DESIGN_RECORD)["blocks"]["sealed"],
            "frozen_before_any_fit": not ADAPTED_SCORES.exists(),
        },
    )
    print(
        "splits: "
        + ", ".join(
            f"{name} test {registry[name]['target_test']['candidates']}" for name in DIRECTIONS
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the draws


def run_draws() -> int:
    """RL2's purchase draws, reproduced row for row, and what each budget's split bought."""
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    for path in (DRAW_REGISTRY, LABEL_COMPOSITION):
        _forbid(path)
    population = load_population()
    calibration = calibration_mask(population["site_group"].astype(str).tolist())
    rl2_rows = {
        (row["direction"], int(row["draw"]), int(row["budget"])): row["candidate_digest"]
        for row in cc_read_json(rl2.DRAW_REGISTRY)["rows"]
    }
    rows: list[dict[str, Any]] = []
    for name, blocks in all_blocks(population).items():
        test = set(blocks.target_test.tolist())
        for draw in range(DRAWS):
            order = rl2.draw_order(population, blocks.frozen, draw)
            for budget in BUDGETS[1:]:
                bought = rl2.purchased(order, budget)
                fit, cal = split_purchase(bought, calibration)
                frame = population.iloc[bought]
                digest = _digest(population, bought)
                fit_frame = population.iloc[fit]
                rows.append(
                    {
                        "direction": name,
                        "draw": draw,
                        "budget": budget,
                        "purchased": int(bought.size),
                        "fit_rows": int(fit.size),
                        "calibration_rows": int(cal.size),
                        "calibration_sites": int(population.iloc[cal]["site_group"].nunique()),
                        "documents": int(frame["document_id"].nunique()),
                        "fit_graded_document_queries": int(
                            sum(
                                1
                                for _k, group in fit_frame.groupby("document_id")
                                if group["grade"].nunique() >= 2
                            )
                        ),
                        "harmful": int(frame["is_harmful"].sum()),
                        "beneficial": int(frame["beneficial"].sum()),
                        "exact": int(frame["exact"].sum()),
                        "candidate_digest": digest,
                        "reproduces_rl2": bool(rl2_rows.get((name, draw, budget)) == digest),
                        "in_rl2_grid": bool((name, draw, budget) in rl2_rows),
                        "touches_test": bool(set(bought.tolist()) & test),
                    }
                )
    table = pd.DataFrame(rows)
    shared = table[table["in_rl2_grid"]]
    _write_json_once(
        DRAW_REGISTRY,
        {
            **_analysis_envelope("draw_registry"),
            "seed": rl2.DRAW_SEED,
            "draws": DRAWS,
            "budgets": list(BUDGETS),
            "rows": rows,
            "count": len(rows),
            "cells_shared_with_rl2": len(shared),
            "reproduces_rl2_everywhere": bool(
                len(shared) == len(table) and shared["reproduces_rl2"].all()
            ),
            "touches_test_anywhere": bool(table["touches_test"].any()),
            "acquisition_rule": cc_read_json(DESIGN_RECORD)["acquisition_rule"],
            "selected_without_evaluation_labels": True,
        },
    )
    composition: dict[str, Any] = {}
    for name in DIRECTIONS:
        composition[name] = {}
        for budget in BUDGETS[1:]:
            cell = table[(table["direction"] == name) & (table["budget"] == budget)]
            composition[name][str(budget)] = {
                "median_fit_rows": float(cell["fit_rows"].median()),
                "median_calibration_rows": float(cell["calibration_rows"].median()),
                "median_calibration_sites": float(cell["calibration_sites"].median()),
                "median_documents": float(cell["documents"].median()),
                "median_harmful": float(cell["harmful"].median()),
                "median_beneficial": float(cell["beneficial"].median()),
                "median_exact": float(cell["exact"].median()),
                "draws_with_fewer_fit_rows_than_a_split_needs": int(
                    (cell["fit_rows"] < 2 * rk1.LAMBDAMART_MIN_LEAF).sum()
                ),
                "draws_without_a_calibration_site": int((cell["calibration_rows"] == 0).sum()),
            }
    _write_json_once(
        LABEL_COMPOSITION,
        {
            **_analysis_envelope("label_composition"),
            "by_direction": composition,
            "rows_a_tree_needs_to_split": 2 * rk1.LAMBDAMART_MIN_LEAF,
            "why": (
                "what a budget buys decides what an arm can learn from it: the fit part must hold "
                "enough rows for a tree to split, and the calibration part must hold at least one "
                "site for a threshold to exist"
            ),
        },
    )
    if not (len(shared) == len(table) and shared["reproduces_rl2"].all()):
        raise PhaseError("RL2's purchase draws did not reproduce; RK2 stops")
    if table["touches_test"].any():
        raise PhaseError("a purchase reached the sealed test block; RK2 stops")
    print(
        f"draws: {len(rows)} draw-budget cells, all reproduce RL2 "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ reproduction gates


def run_reproduce() -> int:
    """RK1's transfer scores and its in-distribution scores, reproduced exactly, before any arm."""
    started = time.monotonic()
    _require(DRAW_REGISTRY, "draws")
    _forbid(REPRODUCTION)
    population = pd.read_parquet(rk1.RANKING_POPULATION)
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    published = pd.read_parquet(rk1.RANKING_SCORES)
    splits = {split.name: split for split in rk1.build_splits(population)}
    columns = rk1.model_columns(rk1.PRIMARY_RANKER)
    out: dict[str, Any] = {}
    for name in (*rk1.LOGO_SPLITS, rk1.SPLIT_BLOCKS):
        split = splits[name]
        frame = rk1.population_frame(population, split.population)
        rank_score, _safety = rk1.score_model(
            rk1.PRIMARY_RANKER, frame, matrix, split.train, split.test, columns
        )
        rebuilt = pd.Series(rank_score, index=frame.iloc[split.test]["candidate_id"].to_numpy())
        cell = published[
            (published["split"] == name)
            & (published["model"] == rk1.PRIMARY_RANKER)
            & (published["variant"] == rk1.V_FULL)
        ].set_index("candidate_id")["rank_score"]
        shared = cell.index.intersection(rebuilt.index)
        difference = float(np.abs(cell.loc[shared] - rebuilt.loc[shared]).max())
        out[name] = {
            "rows": len(shared),
            "published_rows": len(cell),
            "max_absolute_difference": difference,
            "identical": bool(difference == 0.0 and len(shared) == len(cell)),
        }
    rk1_transfer = cc_read_json(rk1.TRANSFER)
    _write_json_once(
        REPRODUCTION,
        {
            **_analysis_envelope("reproduction"),
            "ranker": rk1.PRIMARY_RANKER,
            "by_split": out,
            "all_identical": bool(all(row["identical"] for row in out.values())),
            "rk1_logo_harm_auroc": {
                split: rk1_transfer["by_split"][split][rk1.PRIMARY_RANKER]["harm_auroc"]
                for split in rk1.LOGO_SPLITS
            },
            "rk1_logo_mean_harm_auroc": rk1_transfer["primary_ranker"]["mean_logo_harm_auroc"],
            "note": (
                "this reproduces RK1 on RK1's own splits, whose training rows share pages with "
                "their test rows by design. RK2's zero-shot arm is the same ranker fitted on the "
                "source fit block only and scored on the sealed test block, a stricter quantity"
            ),
        },
    )
    if not all(row["identical"] for row in out.values()):
        raise PhaseError(f"RK1's ranker did not reproduce: {out}")
    print(
        f"reproduce: RK1's ranker reproduces on {len(out)} splits "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the arms


def leaf_refit(source: Any, design: np.ndarray, grades: np.ndarray, groups: np.ndarray) -> Any:
    """The source ranker's trees, split for split, with leaf values re-estimated on new rows.

    This is RK1's own boosting loop with the tree-growing step replaced by the source's frozen
    trees: at each round the LambdaRank gradients are taken at the current scores and every leaf
    the rows reach gets RK1's Newton step. Refitting on the rows the source was trained on
    therefore reproduces the source exactly, and a leaf no row reaches contributes nothing.
    """
    keys = pd.Series(groups).astype(str)
    queries = [
        np.asarray(index, dtype=np.int64)
        for _key, index in sorted(keys.groupby(keys).groups.items())
        if len(set(grades[np.asarray(index)].tolist())) >= 2
    ]
    if not queries or not source.trees:
        return rk1.LambdaMART(trees=[], leaf_values=[], learning_rate=source.learning_rate)
    used = np.sort(np.concatenate(queries))
    scores = np.zeros(design.shape[0], dtype=np.float64)
    leaf_values: list[np.ndarray] = []
    for tree in source.trees:
        lambdas, weights = rk1._lambda_gradients(scores, grades, queries)
        leaves = tree.apply(design[used])
        values = np.zeros(tree.tree_.node_count, dtype=np.float64)
        for leaf in np.unique(leaves):
            member = leaves == leaf
            values[leaf] = float(lambdas[used][member].sum()) / (
                float(weights[used][member].sum()) + 1e-9
            )
        scores += source.learning_rate * values[tree.apply(design)]
        leaf_values.append(values)
    return rk1.LambdaMART(
        trees=list(source.trees), leaf_values=leaf_values, learning_rate=source.learning_rate
    )


def fit_arm(
    arm: str,
    design: np.ndarray,
    grades: np.ndarray,
    groups: np.ndarray,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    source_model: Any,
) -> Any:
    """One arm's ranker: the purchased fit rows, plus the source rows for the joint arm."""

    def fit(rows: np.ndarray) -> Any:
        return rk1.fit_lambdamart(design[rows], grades[rows], groups[rows])

    if arm == A0:
        return fit(source_rows)
    if arm == A1:
        return fit(target_rows)
    if arm == A2:
        return fit(np.concatenate([source_rows, target_rows]))
    if arm == A3:
        return leaf_refit(
            source_model, design[target_rows], grades[target_rows], groups[target_rows]
        )
    raise PhaseError(f"unknown arm {arm}")


def _constant(values: np.ndarray) -> bool:
    return bool(values.size == 0 or np.unique(values).size <= 1)


def run_adapt() -> int:
    """Every arm x direction x budget x draw: test scores and calibration scores, nothing else."""
    started = time.monotonic()
    _require(REPRODUCTION, "reproduce")
    for path in (ADAPTED_SCORES, ADAPTATION_REGISTRY):
        _forbid(path)
    population = load_population()
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    columns = rk1.model_columns(rk1.PRIMARY_RANKER)
    design = rk1._design(population, matrix, columns)
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = population["document_id"].astype(str).to_numpy()
    calibration = calibration_mask(population["site_group"].astype(str).tolist())
    ids = population["candidate_id"].astype(str).to_numpy()
    sealed = {
        name: row["target_test_digest"]
        for name, row in cc_read_json(SPLIT_REGISTRY)["directions"].items()
    }
    frames: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []

    def emit(
        name: str, arm: str, budget: int, draw: int, kind: str, rows: np.ndarray, model: Any
    ) -> None:
        frames.append(
            pd.DataFrame(
                {
                    "direction": name,
                    "arm": arm,
                    "budget": int(budget),
                    "draw": int(draw),
                    "evaluation_set": kind,
                    "candidate_id": ids[rows],
                    "score": model.score(design[rows]) if rows.size else np.empty(0),
                }
            )
        )

    for name, blocks in all_blocks(population).items():
        if _digest(population, blocks.target_test) != sealed[name]:
            raise PhaseError(f"{name}: the test block differs from the one sealed in the registry")
        began = time.monotonic()
        source_model = fit_arm(
            A0, design, grades, groups, blocks.source_fit, blocks.source_fit[:0], None
        )
        emit(name, A0, 0, 0, "test", blocks.target_test, source_model)
        emit(name, A0, 0, 0, "calibration", blocks.source_calibration, source_model)
        registry.append(
            {
                "direction": name,
                "arm": A0,
                "budget": 0,
                "draw": 0,
                "source_rows": int(blocks.source_fit.size),
                "target_fit_rows": 0,
                "calibration_rows": int(blocks.source_calibration.size),
                "calibration_generator": blocks.source,
                "trees": len(source_model.trees),
                "constant_on_test": _constant(source_model.score(design[blocks.target_test])),
            }
        )
        for draw in range(DRAWS):
            order = rl2.draw_order(population, blocks.frozen, draw)
            for budget in BUDGETS[1:]:
                bought = rl2.purchased(order, budget)
                fit_rows, cal_rows = split_purchase(bought, calibration)
                for arm in ADAPTED:
                    t0 = time.monotonic()
                    model = fit_arm(
                        arm, design, grades, groups, blocks.source_fit, fit_rows, source_model
                    )
                    emit(name, arm, budget, draw, "test", blocks.target_test, model)
                    emit(name, arm, budget, draw, "calibration", cal_rows, model)
                    registry.append(
                        {
                            "direction": name,
                            "arm": arm,
                            "budget": int(budget),
                            "draw": int(draw),
                            "source_rows": int(blocks.source_fit.size) if arm == A2 else 0,
                            "target_fit_rows": int(fit_rows.size),
                            "calibration_rows": int(cal_rows.size),
                            "calibration_generator": blocks.target,
                            "trees": len(model.trees),
                            "frozen_source_trees": arm == A3,
                            "constant_on_test": _constant(model.score(design[blocks.target_test])),
                            "runtime_seconds": round(time.monotonic() - t0, 3),
                        }
                    )
        print(f"  {name}: {time.monotonic() - began:.0f}s", flush=True)
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(
        ["direction", "arm", "budget", "draw", "evaluation_set", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    _write_parquet_once(ADAPTED_SCORES, table)
    untrainable = [row for row in registry if row["constant_on_test"]]
    _write_json_once(
        ADAPTATION_REGISTRY,
        {
            **_analysis_envelope("adaptation_registry"),
            "arms": ARM_LABEL,
            "primary_arm": PRIMARY_ARM,
            "cells": registry,
            "fits": len(registry),
            "scored_rows": len(table),
            "feature_columns": len(columns),
            "carries_source_columns": bool([c for c in columns if c.startswith(rl1.FAM_SOURCE)]),
            "untrainable_cells": len(untrainable),
            "untrainable_by_arm_and_budget": {
                f"{row['direction']}|{row['arm']}|{row['budget']}": sum(
                    1
                    for other in untrainable
                    if (other["direction"], other["arm"], other["budget"])
                    == (row["direction"], row["arm"], row["budget"])
                )
                for row in untrainable
            },
            "zero_shot_recorded_once": (
                "the zero-shot arm reads no target label, so it is fitted once per direction and "
                "stands for every arm at the zero budget"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"adapt: {len(registry)} fits, {len(table)} scored rows, {len(untrainable)} untrainable "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ shared evaluation helpers

LABELS = (
    "site_group",
    "site_key",
    "document_id",
    "environment",
    "is_harmful",
    "beneficial",
    "exact",
    "grade",
)


def load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The adapted scores and the labelled population, keyed by candidate id."""
    return pd.read_parquet(ADAPTED_SCORES), load_population().set_index("candidate_id")


def labelled(cell: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """A scored cell with its labels, as RK1's metric and policy functions read it."""
    frame = cell[["candidate_id", "score"]].join(
        population[list(LABELS)], on="candidate_id", how="inner"
    )
    return frame.assign(rank_score=frame["score"], safety=frame["score"]).reset_index(drop=True)


def cell_key(direction: str, arm: str, budget: int, draw: int) -> tuple[str, str, int, int]:
    """Budget zero is the zero-shot arm for every arm, and the zero-shot arm is budget zero."""
    if budget == 0 or arm == A0:
        return direction, A0, 0, 0
    return direction, arm, budget, draw


def draws_at(budget: int) -> range:
    return range(1) if budget == 0 else range(DRAWS)


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    defined = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {
        "draws": len(values),
        "defined": len(defined),
        "median": float(np.median(defined)) if defined else None,
        "q1": float(np.percentile(defined, 25)) if defined else None,
        "q3": float(np.percentile(defined, 75)) if defined else None,
    }


def _index_cells(scores: pd.DataFrame) -> dict[tuple[str, str, int, int, str], pd.DataFrame]:
    return {
        (str(d), str(a), int(b), int(r), str(k)): group
        for (d, a, b, r, k), group in scores.groupby(
            ["direction", "arm", "budget", "draw", "evaluation_set"], sort=True
        )
    }


# ------------------------------------------------------------------ ranking recovery


def reference_frame(population: pd.DataFrame, direction: str, blocks: Blocks) -> pd.DataFrame:
    """RK1's in-distribution primary ranker, read from RK1's scores, on exactly the test rows."""
    published = pd.read_parquet(rk1.RANKING_SCORES)
    cell = published[
        (published["split"] == rk1.SPLIT_BLOCKS)
        & (published["model"] == rk1.PRIMARY_RANKER)
        & (published["variant"] == rk1.V_FULL)
    ]
    test_ids = population.reset_index().iloc[blocks.target_test]["candidate_id"].astype(str)
    cell = cell[cell["candidate_id"].isin(set(test_ids))].rename(columns={"rank_score": "score"})
    if len(cell) != len(test_ids):
        raise PhaseError(f"{direction}: RK1's reference does not cover the sealed test block")
    return labelled(cell, population)


def run_curves() -> int:
    """Harm AUROC, benefit AUROC, Top-1 and NDCG@3 for every cell, and the recovery of the gap."""
    started = time.monotonic()
    _require(ADAPTATION_REGISTRY, "adapt")
    _forbid(RANKING_CURVES)
    scores, population = load_scores()
    cells = _index_cells(scores)
    blocks = all_blocks(population.reset_index())
    per_draw: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    references: dict[str, Any] = {}
    for name in DIRECTIONS:
        reference = rk1.ranking_metrics(reference_frame(population, name, blocks[name]))
        references[name] = {
            "harm_auroc": reference["harm_auroc"],
            "benefit_auroc": reference["benefit_auroc"],
            "top1": reference["top1"],
            "choice_sites": reference["choice_sites"],
        }
        curves[name] = {}
        for arm in ARMS:
            curves[name][arm] = {}
            for budget in BUDGETS:
                if arm == A0 and budget > 0:
                    continue
                values: dict[str, list[float | None]] = {
                    k: [] for k in ("harm_auroc", "benefit_auroc", "top1", "ndcg_at_3")
                }
                choice_sites = 0
                for draw in draws_at(budget):
                    key = cell_key(name, arm, budget, draw)
                    metrics = rk1.ranking_metrics(labelled(cells[(*key, "test")], population))
                    row = {
                        "harm_auroc": metrics["harm_auroc"],
                        "benefit_auroc": metrics["benefit_auroc"],
                        "top1": metrics["top1"],
                        "ndcg_at_3": metrics["ndcg_at_k"]["3"],
                    }
                    choice_sites = int(metrics["choice_sites"])
                    for metric, value in row.items():
                        values[metric].append(value)
                    per_draw.append(
                        {
                            "direction": name,
                            "arm": arm,
                            "budget": budget,
                            "draw": draw,
                            **row,
                            "constant_score": bool(metrics["constant_score"]),
                            "choice_sites": choice_sites,
                        }
                    )
                curves[name][arm][str(budget)] = {
                    metric: _summary(v) for metric, v in values.items()
                } | {
                    "choice_sites": choice_sites,
                    "untrainable_draws": sum(1 for v in values["harm_auroc"] if v is None),
                    "draws_at_or_above_the_harm_floor": sum(
                        1 for v in values["harm_auroc"] if v is not None and v >= HARM_FLOOR
                    ),
                }
    for name in DIRECTIONS:
        zero = curves[name][A0]["0"]["harm_auroc"]["median"]
        ref = references[name]["harm_auroc"]
        references[name]["zero_shot_harm_auroc"] = zero
        references[name]["gap"] = _num(None if zero is None or ref is None else ref - zero)
        for arm in ADAPTED:
            for budget in BUDGETS[1:]:
                row = curves[name][arm][str(budget)]
                row["harm_recovery"] = recovery(row["harm_auroc"]["median"], zero, ref)
    _write_json_once(
        RANKING_CURVES,
        {
            **_analysis_envelope("ranking_curves"),
            "block": "sealed target test block",
            "by_direction": curves,
            "per_draw": per_draw,
            "seen_generator_reference": references,
            "reference_note": cc_read_json(DESIGN_RECORD)["recovery"],
            "harm_floor": HARM_FLOOR,
            "top1_note": cc_read_json(DESIGN_RECORD)["metrics"]["top1"],
            "rk1_logo_harm_auroc": cc_read_json(REPRODUCTION)["rk1_logo_harm_auroc"],
        },
    )
    print(
        "curves: "
        + ", ".join(
            f"{name} zero-shot {curves[name][A0]['0']['harm_auroc']['median']:.4f} -> "
            f"target-only@{SATURATION_BUDGET} "
            f"{curves[name][A1][str(SATURATION_BUDGET)]['harm_auroc']['median']:.4f}"
            for name in DIRECTIONS
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ deployment under a budget


def decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """One decision per site: the ranker's top candidate, the candidate id breaking ties."""
    return ds1.top_per_site(frame, "rank_score", False).reset_index(drop=True)


def reachable_error_keys(
    test_sites: set[str], errors: pd.DataFrame, mapping: dict[str, set[str]]
) -> set[str]:
    """Test-block error sites that LP1's frozen links connect to a target test decision site."""
    keys = set(errors[errors["block"] == "test"]["error_key"].astype(str))
    linked: set[str] = set()
    for site in test_sites:
        linked |= mapping.get(site, set())
    return linked & keys


def gate(
    calibration: pd.DataFrame, test: pd.DataFrame, epsilon: float, trainable: bool
) -> tuple[float | None, pd.DataFrame]:
    """The within-budget policy: a cutoff chosen on calibration decisions, applied to test ones."""
    cutoff = rk1.choose_threshold(calibration, epsilon) if trainable else None
    return cutoff, (test[test["safety"] >= cutoff] if cutoff is not None else test.head(0))


def oracle_accepted(test: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """ANALYSIS ONLY: the loosest cutoff meeting the target on the test labels themselves."""
    cutoff = rk1.choose_threshold(test, epsilon)
    return test[test["safety"] >= cutoff] if cutoff is not None else test.head(0)


def policy_outcome(
    test: pd.DataFrame,
    accepted: pd.DataFrame,
    keys: set[str],
    mapping: dict[str, set[str]],
    epsilon: float,
) -> dict[str, Any]:
    row = ds1._as_dict(
        ds1.evaluate_policy(test, accepted, keys, mapping, int(test["document_id"].nunique()))
    )
    row["holds_its_target"] = bool(
        len(accepted) == 0
        or (
            row["selective_harm_rate"] is not None
            and row["selective_harm_rate"] <= epsilon + HARM_TOLERANCE
        )
    )
    row["working_safe_gate"] = bool(
        len(accepted) > 0 and row["repaired_error_sites"] >= 1 and row["holds_its_target"]
    )
    return row


def run_deployment() -> int:
    """The within-budget policy on the sealed test block, and the label-cost frontier."""
    started = time.monotonic()
    _require(RANKING_CURVES, "curves")
    for path in (DEPLOYMENT_CURVES, FRONTIER):
        _forbid(path)
    scores, population = load_scores()
    cells = _index_cells(scores)
    errors = ds1.load_error_population()
    mapping = ds1.site_to_errors(errors)
    blocks = all_blocks(population.reset_index())
    ranking = cc_read_json(RANKING_CURVES)
    harm_by_draw = {
        (row["direction"], row["arm"], row["budget"], row["draw"]): row["harm_auroc"]
        for row in ranking["per_draw"]
    }
    per_draw: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    ceilings: dict[str, Any] = {}
    references: dict[str, Any] = {}
    rk1_thresholds = cc_read_json(rk1.DEPLOYMENT)["by_model"][rk1.PRIMARY_RANKER]
    for name in DIRECTIONS:
        test_ids = population.reset_index().iloc[blocks[name].target_test]["candidate_id"]
        test_sites = set(population.loc[test_ids, "site_key"].astype(str))
        keys = reachable_error_keys(test_sites, errors, mapping)
        truth = labelled(pd.DataFrame({"candidate_id": test_ids, "score": 0.0}), population)
        perfect = truth.sort_values(["exact", "candidate_id"], ascending=[False, True])
        perfect = perfect.groupby("site_key", sort=False).head(1)
        ceilings[name] = {
            "reachable_error_sites": len(keys),
            "decision_sites": len(test_sites),
            "perfect_selection_repair_recall": ds1._as_dict(
                ds1.evaluate_policy(decisions(truth), perfect[perfect["exact"]], keys, mapping, 1)
            )["repair_recall"],
            "analysis_only": True,
        }
        reference = decisions(reference_frame(population, name, blocks[name]))
        references[name] = {}
        for epsilon in EPSILONS:
            tau = rk1_thresholds[rl1.epsilon_key(epsilon)]["threshold"]
            accepted = (
                reference[reference["safety"] >= tau] if tau is not None else reference.head(0)
            )
            references[name][rl1.epsilon_key(epsilon)] = policy_outcome(
                reference, accepted, keys, mapping, epsilon
            ) | {"threshold": tau}
        curves[name] = {}
        for arm in ARMS:
            curves[name][arm] = {}
            for budget in BUDGETS:
                if arm == A0 and budget > 0:
                    continue
                rows: list[dict[str, Any]] = []
                for draw in draws_at(budget):
                    key = cell_key(name, arm, budget, draw)
                    test = decisions(labelled(cells[(*key, "test")], population))
                    cal_cell = cells.get((*key, "calibration"))
                    calibration = (
                        decisions(labelled(cal_cell, population))
                        if cal_cell is not None and len(cal_cell)
                        else test.head(0)
                    )
                    trainable = harm_by_draw[key] is not None
                    record: dict[str, Any] = {
                        "direction": name,
                        "arm": arm,
                        "budget": budget,
                        "draw": draw,
                        "trainable": trainable,
                        "calibration_decisions": len(calibration),
                        "harm_auroc": harm_by_draw[key],
                    }
                    for epsilon in EPSILONS:
                        cutoff, accepted = gate(calibration, test, epsilon, trainable)
                        outcome = policy_outcome(test, accepted, keys, mapping, epsilon)
                        outcome["threshold"] = cutoff
                        outcome["abstains"] = cutoff is None
                        oracle = oracle_accepted(test, epsilon) if trainable else test.head(0)
                        outcome["oracle_repair_recall_analysis_only"] = policy_outcome(
                            test, oracle, keys, mapping, epsilon
                        )["repair_recall"]
                        record[rl1.epsilon_key(epsilon)] = outcome
                    primary = record[rl1.epsilon_key(PRIMARY_EPSILON)]
                    record["deployable_draw"] = bool(
                        record["harm_auroc"] is not None
                        and record["harm_auroc"] >= HARM_FLOOR
                        and primary["working_safe_gate"]
                    )
                    rows.append(record)
                    per_draw.append(record)
                curves[name][arm][str(budget)] = aggregate_deployment(rows)
    frontier = label_cost_frontier(curves)
    _write_json_once(
        DEPLOYMENT_CURVES,
        {
            **_analysis_envelope("deployment_curves"),
            "block": "sealed target test block",
            "by_direction": curves,
            "per_draw": per_draw,
            "perfect_selection_ceiling": ceilings,
            "seen_generator_reference": references,
            "seen_generator_reference_note": (
                "RK1's in-distribution ranker at RK1's own thresholds, which were chosen on both "
                "generators' threshold block, restricted to these test rows. It saw target "
                "labels; it is a reference, not a within-budget policy"
            ),
            "threshold_rule": cc_read_json(DESIGN_RECORD)["threshold_rule"],
            "recall_denominator": cc_read_json(DESIGN_RECORD)["metrics"]["repair_recall"],
            "oracle_note": cc_read_json(DESIGN_RECORD)["metrics"]["oracle_threshold"],
            "primary_epsilon": PRIMARY_EPSILON,
        },
    )
    _write_json_once(
        FRONTIER,
        {
            **_analysis_envelope("label_cost_frontier"),
            **frontier,
            "definition": cc_read_json(DESIGN_RECORD)["deployable"],
            "primary_arm": PRIMARY_ARM,
            "harm_floor": HARM_FLOOR,
            "primary_epsilon": PRIMARY_EPSILON,
            "draw_majority": DRAW_MAJORITY,
        },
    )
    primary = frontier["minimal_deployable_budget"][PRIMARY_ARM]
    print(
        f"deployment: primary N* {primary} ({time.monotonic() - started:.0f}s)",
    )
    return 0


def aggregate_deployment(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Across draws: medians of what the policy did, and the share of draws where it worked."""
    out: dict[str, Any] = {"draws": len(rows)}
    for epsilon in EPSILONS:
        key = rl1.epsilon_key(epsilon)
        points = [row[key] for row in rows]
        out[key] = {
            "repair_recall": _summary([p["repair_recall"] for p in points]),
            "review_rate": _summary([p["review_rate"] for p in points]),
            "coverage": _summary([p["coverage"] for p in points]),
            "selective_harm_rate": _summary(
                [p["selective_harm_rate"] for p in points if p["accepted"] > 0]
            ),
            "oracle_repair_recall_analysis_only": _summary(
                [p["oracle_repair_recall_analysis_only"] for p in points]
            ),
            "share_abstaining": float(np.mean([p["abstains"] for p in points])),
            "share_accepting_nothing": float(np.mean([p["accepted"] == 0 for p in points])),
            "share_holding_the_target": float(np.mean([p["holds_its_target"] for p in points])),
            "share_accepting_and_exceeding_the_target": float(
                np.mean([p["accepted"] > 0 and not p["holds_its_target"] for p in points])
            ),
            "share_working_safe_gate": float(np.mean([p["working_safe_gate"] for p in points])),
        }
    out["share_ranking_ready"] = float(
        np.mean([r["harm_auroc"] is not None and r["harm_auroc"] >= HARM_FLOOR for r in rows])
    )
    out["share_deployable"] = float(np.mean([r["deployable_draw"] for r in rows]))
    out["deployable"] = bool(out["share_deployable"] >= DRAW_MAJORITY)
    out["ranking_ready"] = bool(out["share_ranking_ready"] >= DRAW_MAJORITY)
    return out


def label_cost_frontier(curves: dict[str, Any]) -> dict[str, Any]:
    """N* and the ranking-ready budget for every arm and direction, from the per-budget cells."""
    deployable: dict[str, Any] = {}
    ready: dict[str, Any] = {}
    flags: dict[str, Any] = {}
    for arm in ARMS:
        deployable[arm] = {}
        ready[arm] = {}
        flags[arm] = {}
        for name in DIRECTIONS:

            def cell(budget: int, arm: str = arm, name: str = name) -> dict[str, Any]:
                return curves[name][A0 if budget == 0 else arm][str(budget)]

            budgets = (0,) if arm == A0 else BUDGETS
            dep = {b: bool(cell(b)["deployable"]) for b in budgets}
            rank = {b: bool(cell(b)["ranking_ready"]) for b in budgets}
            deployable[arm][name] = minimal_budget(dep)
            ready[arm][name] = minimal_budget(rank)
            flags[arm][name] = {
                str(b): {
                    "deployable": dep[b],
                    "ranking_ready": rank[b],
                    "share_deployable": cell(b)["share_deployable"],
                    "share_ranking_ready": cell(b)["share_ranking_ready"],
                    "share_working_safe_gate": cell(b)[rl1.epsilon_key(PRIMARY_EPSILON)][
                        "share_working_safe_gate"
                    ],
                }
                for b in budgets
            }
    return {
        "minimal_deployable_budget": deployable,
        "ranking_ready_budget": ready,
        "by_budget": flags,
    }


# ------------------------------------------------------------------ Q2 and Q3


def run_compare() -> int:
    """Q2: does source data help or hurt? Q3: does refitting leaves recover without retraining?"""
    started = time.monotonic()
    _require(FRONTIER, "deployment")
    _forbid(ARM_COMPARISON)
    curves = cc_read_json(RANKING_CURVES)["by_direction"]
    deployment = cc_read_json(DEPLOYMENT_CURVES)["by_direction"]
    frontier = cc_read_json(FRONTIER)
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    table: dict[str, Any] = {}
    wins: dict[str, dict[str, Any]] = {}
    for name in DIRECTIONS:
        table[name] = {}
        wins[name] = {}
        for budget in BUDGETS[1:]:

            def median(
                arm: str, metric: str, budget: int = budget, name: str = name
            ) -> float | None:
                value = curves[name][arm][str(budget)][metric]["median"]
                return None if value is None else float(value)

            def recall(arm: str, budget: int = budget, name: str = name) -> float | None:
                value = deployment[name][arm][str(budget)][key]["repair_recall"]["median"]
                return None if value is None else float(value)

            table[name][str(budget)] = {
                "harm_auroc": {arm: median(arm, "harm_auroc") for arm in ADAPTED},
                "benefit_auroc": {arm: median(arm, "benefit_auroc") for arm in ADAPTED},
                "repair_recall_at_primary_target": {arm: recall(arm) for arm in ADAPTED},
                "joint_minus_target_only_harm_auroc": rk1._delta(
                    median(A2, "harm_auroc"), median(A1, "harm_auroc")
                ),
                "leaf_refit_minus_target_only_harm_auroc": rk1._delta(
                    median(A3, "harm_auroc"), median(A1, "harm_auroc")
                ),
                "leaf_refit_minus_joint_harm_auroc": rk1._delta(
                    median(A3, "harm_auroc"), median(A2, "harm_auroc")
                ),
                "untrainable_draws": {
                    arm: curves[name][arm][str(budget)]["untrainable_draws"] for arm in ADAPTED
                },
            }
        defined = [
            b
            for b in BUDGETS[1:]
            if table[name][str(b)]["joint_minus_target_only_harm_auroc"] is not None
        ]
        target_wins = sum(
            1 for b in defined if table[name][str(b)]["joint_minus_target_only_harm_auroc"] < 0
        )
        joint_wins = sum(
            1 for b in defined if table[name][str(b)]["joint_minus_target_only_harm_auroc"] > 0
        )
        only_joint = [
            b
            for b in BUDGETS[1:]
            if table[name][str(b)]["harm_auroc"][A1] is None
            and table[name][str(b)]["harm_auroc"][A2] is not None
        ]
        wins[name] = {
            "budgets_compared": defined,
            "target_only_wins": target_wins,
            "joint_wins": joint_wins,
            "budgets_where_only_the_joint_arm_can_be_fitted": only_joint,
        }
    majority = {name: len(wins[name]["budgets_compared"]) / 2.0 for name in DIRECTIONS}
    if all(wins[n]["target_only_wins"] > majority[n] for n in DIRECTIONS):
        q2 = "negative_transfer"
    elif all(wins[n]["joint_wins"] > majority[n] for n in DIRECTIONS):
        q2 = "source_data_helps"
    else:
        q2 = "mixed"
    leaf_ready = frontier["ranking_ready_budget"][A3]
    q3 = bool(all(leaf_ready[name] is not None for name in DIRECTIONS))
    _write_json_once(
        ARM_COMPARISON,
        {
            **_analysis_envelope("arm_comparison"),
            "by_direction": table,
            "q2_wins": wins,
            "q2_reading": q2,
            "q2_rule": cc_read_json(DESIGN_RECORD)["q2_reading"],
            "q3_leaf_refit_ranking_ready_budget": leaf_ready,
            "q3_leaf_refit_minimal_deployable_budget": frontier["minimal_deployable_budget"][A3],
            "q3_recovers_without_full_retraining": q3,
            "q3_rule": cc_read_json(DESIGN_RECORD)["q3_reading"],
            "descriptive": (
                "the per-budget comparisons are medians over draws and are descriptive; the "
                "inferential comparisons at the decision budget are in the statistical family"
            ),
        },
    )
    print(f"compare: Q2 {q2}, Q3 {q3} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ controls


def permutation_control(population: pd.DataFrame, matrix: pd.DataFrame) -> dict[str, Any]:
    """The target-only ranker at the saturation budget, refitted on grades shuffled per query.

    Shuffling inside each purchased document keeps every query's grade mix -- all the LambdaMART
    loss sees of a query besides its features -- and destroys only the link between a
    candidate's evidence and its grade. If adaptation came from anything but that link, the
    shuffled ranker would recover as much.
    """
    design = rk1._design(population, matrix, rk1.model_columns(rk1.PRIMARY_RANKER))
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = population["document_id"].astype(str).to_numpy()
    harmful = population["is_harmful"].to_numpy(dtype=bool)
    calibration = calibration_mask(population["site_group"].astype(str).tolist())
    out: dict[str, Any] = {}
    for name, blocks in all_blocks(population).items():
        values: list[float | None] = []
        for draw in range(DRAWS):
            order = rl2.draw_order(population, blocks.frozen, draw)
            fit_rows, _cal = split_purchase(rl2.purchased(order, SATURATION_BUDGET), calibration)
            generator = np.random.default_rng(PERMUTATION_SEED + draw)
            shuffled = grades[fit_rows].copy()
            fit_groups = groups[fit_rows]
            for key in sorted(set(fit_groups.tolist())):
                members = np.flatnonzero(fit_groups == key)
                shuffled[members] = shuffled[members][generator.permutation(members.size)]
            model = rk1.fit_lambdamart(design[fit_rows], shuffled, fit_groups)
            score = model.score(design[blocks.target_test])
            values.append(None if _constant(score) else auroc(-score, harmful[blocks.target_test]))
        out[name] = {"harm_auroc_by_draw": values, **_summary(values)}
    medians = [out[name]["median"] for name in DIRECTIONS]
    low, high = NEAR_RANDOM_BAND
    return {
        "by_direction": out,
        "budget": SATURATION_BUDGET,
        "arm": PRIMARY_ARM,
        "seed": PERMUTATION_SEED,
        "near_random_band": list(NEAR_RANDOM_BAND),
        "near_random": bool(all(m is not None and low <= m <= high for m in medians)),
    }


def run_controls() -> int:
    started = time.monotonic()
    _require(ARM_COMPARISON, "compare")
    _forbid(CONTROL_RESULTS)
    population = load_population()
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    control = permutation_control(population, matrix)
    honest = cc_read_json(RANKING_CURVES)["by_direction"]
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "label_permutation": control,
            "honest_labels_same_budget": {
                name: honest[name][PRIMARY_ARM][str(SATURATION_BUDGET)]["harm_auroc"]
                for name in DIRECTIONS
            },
            "rule": cc_read_json(DESIGN_RECORD)["controls"]["label_permutation"],
        },
    )
    print(
        f"controls: permutation near random {control['near_random']} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ statistics


def _inference_folds(
    scores: pd.DataFrame, population: pd.DataFrame, arms: Sequence[str]
) -> list[pd.DataFrame]:
    """Each direction's test rows with every named arm's score side by side."""
    cells = _index_cells(scores)
    folds = []
    for name in DIRECTIONS:
        base: pd.DataFrame | None = None
        for arm in arms:
            key = cell_key(name, arm, DECISION_BUDGET, INFERENCE_DRAW)
            frame = labelled(cells[(*key, "test")], population)
            column = frame[["candidate_id", "score"]].rename(columns={"score": arm})
            if base is None:
                base = frame[
                    ["candidate_id", "environment", "document_id", "is_harmful", "beneficial"]
                ].merge(column, on="candidate_id", validate="one_to_one")
            else:
                base = base.merge(column, on="candidate_id", validate="one_to_one")
        if base is None:
            raise PhaseError("no arm named for an inference fold")
        folds.append(base.sort_values("candidate_id", kind="stable").reset_index(drop=True))
    return folds


def auroc_comparison(
    name: str,
    left: str,
    right: str,
    effect: dict[str, float],
    draws: dict[str, np.ndarray],
    folds: Sequence[pd.DataFrame],
) -> dict[str, Any]:
    """left minus right, as the mean over directions of per-direction test harm AUROC."""
    from ocr_risk.stats.bootstrap import _bootstrap_p_value

    constant = [
        column for column in (left, right) for fold in folds if _constant(fold[column].to_numpy())
    ]
    base: dict[str, Any] = {
        "comparison": name,
        "family": "primary",
        "left": left,
        "right": right,
        "units": int(sum(len(f) for f in folds)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "pooled": False,
    }
    if constant:
        return base | {
            "defined": False,
            "undefined_because": f"constant scores in {sorted(set(constant))}",
            "effect": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": 1.0,
            "environments_improved": 0,
            "environments_worsened": 0,
        }
    difference = draws[left] - draws[right]
    finite = difference[np.isfinite(difference)]
    low, high = rk1._interval(difference)
    per_environment: dict[str, float] = {}
    for environment in sorted(set().union(*(set(f["environment"].astype(str)) for f in folds))):
        values = []
        for fold in folds:
            rows = fold[fold["environment"] == environment]
            harmful = rows["is_harmful"].to_numpy(bool)
            values.append(
                auroc(-rows[left].to_numpy(np.float64), harmful)
                - auroc(-rows[right].to_numpy(np.float64), harmful)
            )
        finite_values = [v for v in values if np.isfinite(v)]
        if finite_values:
            per_environment[environment] = float(np.mean(finite_values))
    return base | {
        "defined": True,
        "effect": float(effect[left] - effect[right]),
        "ci_low": low,
        "ci_high": high,
        "p_value": float(_bootstrap_p_value(finite)),
        "environments_improved": sum(1 for v in per_environment.values() if v > 0),
        "environments_worsened": sum(1 for v in per_environment.values() if v < 0),
    }


def recall_frame(
    scores: pd.DataFrame, population: pd.DataFrame, left: str, right: str
) -> pd.DataFrame:
    """Both directions' target-reachable error sites; a and b are the two arms' repairs."""
    cells = _index_cells(scores)
    errors = ds1.load_error_population()
    mapping = ds1.site_to_errors(errors)
    blocks = all_blocks(population.reset_index())
    ranking = {
        (row["direction"], row["arm"], row["budget"], row["draw"]): row["harm_auroc"]
        for row in cc_read_json(RANKING_CURVES)["per_draw"]
    }
    frames = []
    for name in DIRECTIONS:
        test_ids = population.reset_index().iloc[blocks[name].target_test]["candidate_id"]
        keys = reachable_error_keys(
            set(population.loc[test_ids, "site_key"].astype(str)), errors, mapping
        )
        repaired = {}
        for arm in (left, right):
            key = cell_key(name, arm, DECISION_BUDGET, INFERENCE_DRAW)
            test = decisions(labelled(cells[(*key, "test")], population))
            cal_cell = cells.get((*key, "calibration"))
            calibration = (
                decisions(labelled(cal_cell, population))
                if cal_cell is not None and len(cal_cell)
                else test.head(0)
            )
            _cutoff, accepted = gate(calibration, test, PRIMARY_EPSILON, ranking[key] is not None)
            repaired[arm] = ds1._repaired_flags(
                test, accepted, errors[errors["error_key"].isin(keys)], mapping
            )
        block = errors[errors["error_key"].isin(keys)].sort_values("error_key", kind="stable")
        frame = ds1._recall_frame(block, repaired[left], repaired[right], "error_site")
        frames.append(frame.assign(direction=name))
    return pd.concat(frames, ignore_index=True)


def inference_draw_policies() -> dict[str, Any]:
    """What each compared arm's policy did at the decision budget on the pre-registered draw."""
    rows = cc_read_json(DEPLOYMENT_CURVES)["per_draw"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    out: dict[str, Any] = {}
    for name in DIRECTIONS:
        out[name] = {}
        for arm in ARMS:
            wanted = cell_key(name, arm, DECISION_BUDGET, INFERENCE_DRAW)
            row = next(
                r
                for r in rows
                if (r["direction"], r["arm"], r["budget"], r["draw"]) == wanted
                and (arm == A0 or r["arm"] == arm)
            )
            point = row[key]
            out[name][arm] = {
                "harm_auroc": row["harm_auroc"],
                "accepted": point["accepted"],
                "decision_sites": point["sites"],
                "repair_recall": point["repair_recall"],
                "selective_harm_rate": point["selective_harm_rate"],
                "holds_its_target": point["holds_its_target"],
                "working_safe_gate": point["working_safe_gate"],
            }
    return out


def run_stats() -> int:
    """The frozen family of four, document-clustered and paired, Holm-corrected."""
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    scores, population = load_scores()
    arms = (A0, A1, A2, A3)
    folds = _inference_folds(scores, population, arms)
    effect, draws = rk1.fold_mean_auroc_draws(folds, arms)
    recall = recall_frame(scores, population, A1, A0)
    family = {
        PRIMARY_FAMILY[0]: auroc_comparison(PRIMARY_FAMILY[0], A1, A0, effect, draws, folds),
        PRIMARY_FAMILY[1]: xr1.comparison(recall, PRIMARY_FAMILY[1], "primary", pooled=True)
        | {"left": A1, "right": A0, "defined": True},
        PRIMARY_FAMILY[2]: auroc_comparison(PRIMARY_FAMILY[2], A2, A1, effect, draws, folds),
        PRIMARY_FAMILY[3]: auroc_comparison(PRIMARY_FAMILY[3], A3, A1, effect, draws, folds),
    }
    for row in family.values():
        if row["p_value"] is None or not np.isfinite(row["p_value"]):
            row["p_value"] = 1.0
            row["defined"] = False
    adjusted = s15.holm(family)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "budget": DECISION_BUDGET,
            "draw": INFERENCE_DRAW,
            "epsilon": PRIMARY_EPSILON,
            "bootstrap": {
                "unit": "document, resampled within environment",
                "paired": True,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "alpha": ALPHA,
            },
            "multiplicity": "Holm-Bonferroni inside the frozen primary family",
            "primary_family": adjusted,
            "family_size": len(PRIMARY_FAMILY),
            "declared_before_any_endpoint": True,
            "harm_auroc_by_arm_at_the_inference_draw": {arm: effect[arm] for arm in arms},
            "units": {
                PRIMARY_FAMILY[
                    0
                ]: "test candidates of both directions; mean of per-direction AUROC",
                PRIMARY_FAMILY[1]: (
                    "target-reachable error sites of both directions, pooled; a and b are "
                    "'repaired by an accepted exact edit'"
                ),
                PRIMARY_FAMILY[
                    2
                ]: "test candidates of both directions; mean of per-direction AUROC",
                PRIMARY_FAMILY[
                    3
                ]: "test candidates of both directions; mean of per-direction AUROC",
            },
            "direction": dict.fromkeys(PRIMARY_FAMILY, "positive favours the left arm"),
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "inference_draw_policies": inference_draw_policies(),
            "recall_is_not_safety": (
                "P2 compares repair recall and nothing else. Whether each policy held its harm "
                "target on the same draw is recorded beside it, so a recall gain bought with "
                "harm above the target cannot be read as a safe gain"
            ),
            "draw_spread_is_descriptive": (
                "few-shot draw variability and clustered evaluation uncertainty are never pooled "
                "into one interval: the test fixes the pre-registered draw and the quartiles in "
                "the curve artifacts carry the draw spread"
            ),
        },
    )
    print(
        f"stats: {sum(1 for v in adjusted.values() if v['survives_holm'])}/{len(adjusted)} "
        f"survive Holm ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ falsification


def _module_tree() -> ast.Module:
    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def _functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _names_in(function_name: str) -> set[str]:
    """Every name a function mentions, following the calls it makes inside this module."""
    functions = _functions()
    seen: set[str] = set()
    names: set[str] = set()
    stack = [function_name]
    while stack:
        current = stack.pop()
        if current in seen or current not in functions:
            continue
        seen.add(current)
        for node in ast.walk(functions[current]):
            if isinstance(node, ast.Name):
                names.add(node.id)
                stack.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names


def _called_names() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run_negative":
            continue
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _gate_threshold_argument() -> list[str]:
    """Which decisions frame `gate` hands to the threshold chooser, read from the AST.

    The chooser's first argument is the decisions it reads labels from; the second is the risk
    target. Only the first can carry a leak.
    """
    found: list[str] = []
    for node in ast.walk(_functions()["gate"]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "choose_threshold"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            found.append(node.args[0].id)
    return found


def run_negative() -> int:
    """Twenty-five tests, each of which would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    population = load_population()
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    scores = pd.read_parquet(ADAPTED_SCORES)
    splits = cc_read_json(SPLIT_REGISTRY)
    draws = cc_read_json(DRAW_REGISTRY)
    registry = cc_read_json(ADAPTATION_REGISTRY)
    curves = cc_read_json(RANKING_CURVES)
    blocks = all_blocks(population)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    columns = rk1.model_columns(rk1.PRIMARY_RANKER)
    identity = [c for c in columns if c.startswith(rl1.FAM_SOURCE)]
    record("t01_no_generator_identity_column_in_any_arm", not identity, identity)
    leaked = sorted(set(columns) & rk1.LABEL_FIELDS)
    record("t02_no_label_or_oracle_field_is_a_design_column", not leaked, leaked)
    sealed = {
        name: blocks[name].target_test.size == row["target_test"]["candidates"]
        and _digest(population, blocks[name].target_test) == row["target_test_digest"]
        and row["equals_rl2_evaluation_pool"]
        for name, row in splits["directions"].items()
    }
    record(
        "t03_the_test_block_is_rl2s_evaluation_pool_sealed_before_any_fit",
        all(sealed.values())
        and splits["frozen_before_any_fit"]
        and splits["issued_utc"] <= registry["issued_utc"],
        sealed,
    )
    test_ids = {
        name: set(population.iloc[blocks[name].target_test]["candidate_id"].astype(str))
        for name in DIRECTIONS
    }
    in_calibration = {
        name: int(
            scores[(scores["direction"] == name) & (scores["evaluation_set"] == "calibration")][
                "candidate_id"
            ]
            .isin(test_ids[name])
            .sum()
        )
        for name in DIRECTIONS
    }
    record(
        "t04_no_test_row_is_purchased_or_used_to_choose_a_threshold",
        not draws["touches_test_anywhere"] and not any(in_calibration.values()),
        in_calibration,
    )
    record(
        "t05_no_test_document_site_or_edit_group_reaches_any_training_row",
        bool(splits["nothing_crosses"]),
        splits["leakage"],
    )
    record(
        "t06_the_purchased_rows_are_rl2s_at_every_shared_budget",
        bool(draws["reproduces_rl2_everywhere"]),
        {"cells": draws["cells_shared_with_rl2"]},
    )
    nested = True
    for _name, block in blocks.items():
        for draw in range(DRAWS):
            order = rl2.draw_order(population, block.frozen, draw)
            previous = np.empty(0, dtype=np.int64)
            for budget in BUDGETS[1:]:
                bought = rl2.purchased(order, budget)
                nested &= bool(np.array_equal(bought[: previous.size], previous))
                previous = bought
    record("t07_budgets_nest_inside_every_draw", nested, {"draws": DRAWS})
    calibration = calibration_mask(population["site_group"].astype(str).tolist())
    reshuffled = population.sample(frac=1.0, random_state=FIT_SEED)
    again = calibration_mask(reshuffled["site_group"].astype(str).tolist())
    consistent = bool(
        np.array_equal(again, calibration[population.index.get_indexer(reshuffled.index)])
    )
    overlap = 0
    for _name, block in blocks.items():
        order = rl2.draw_order(population, block.frozen, INFERENCE_DRAW)
        fit, cal = split_purchase(rl2.purchased(order, SATURATION_BUDGET), calibration)
        overlap += len(
            set(population.iloc[fit]["site_group"]) & set(population.iloc[cal]["site_group"])
        )
    record(
        "t08_fit_and_calibration_parts_are_site_disjoint_and_label_blind",
        consistent and overlap == 0,
        {"shared_sites": overlap, "row_order_invariant": consistent},
    )
    record(
        "t09_thresholds_are_chosen_on_calibration_decisions_only",
        _gate_threshold_argument() == ["calibration"],
        _gate_threshold_argument(),
    )
    cells = registry["cells"]
    record(
        "t10_the_target_only_arm_contains_no_source_row",
        all(row["source_rows"] == 0 for row in cells if row["arm"] == A1),
        sorted({row["source_rows"] for row in cells if row["arm"] == A1}),
    )
    zero = [row for row in cells if row["arm"] == A0]
    record(
        "t11_the_zero_shot_arm_reads_no_target_label",
        len(zero) == len(DIRECTIONS)
        and all(row["target_fit_rows"] == 0 and row["budget"] == 0 for row in zero)
        and all(row["calibration_generator"] == SOURCE_OF[row["direction"]] for row in zero),
        zero,
    )
    design = rk1._design(population, matrix, columns)
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = population["document_id"].astype(str).to_numpy()
    kept: list[bool] = []
    exact_self: list[float] = []
    for _name, block in blocks.items():
        source = rk1.fit_lambdamart(
            design[block.source_fit], grades[block.source_fit], groups[block.source_fit]
        )
        order = rl2.draw_order(population, block.frozen, INFERENCE_DRAW)
        fit, _cal = split_purchase(rl2.purchased(order, SATURATION_BUDGET), calibration)
        refit = leaf_refit(source, design[fit], grades[fit], groups[fit])
        kept.append(
            all(
                a is b
                and np.array_equal(a.tree_.feature, b.tree_.feature)
                and np.array_equal(a.tree_.threshold, b.tree_.threshold)
                for a, b in zip(source.trees, refit.trees, strict=True)
            )
        )
        itself = leaf_refit(
            source,
            design[block.source_fit],
            grades[block.source_fit],
            groups[block.source_fit],
        )
        exact_self.append(
            float(
                np.abs(
                    itself.score(design[block.target_test])
                    - source.score(design[block.target_test])
                ).max()
            )
        )
    record("t12_leaf_refit_keeps_every_source_split", all(kept), kept)
    record(
        "t13_leaf_refit_on_the_sources_own_rows_reproduces_the_source_exactly",
        all(value == 0.0 for value in exact_self),
        exact_self,
    )
    reproduction = cc_read_json(REPRODUCTION)
    record(
        "t14_rk1s_transfer_and_in_distribution_scores_reproduce_exactly",
        bool(reproduction["all_identical"]),
        {k: v["max_absolute_difference"] for k, v in reproduction["by_split"].items()},
    )
    deployment = cc_read_json(DEPLOYMENT_CURVES)["per_draw"]
    untrainable_deploys = [
        (row["direction"], row["arm"], row["budget"], row["draw"])
        for row in deployment
        if not row["trainable"] and any(row[rl1.epsilon_key(e)]["accepted"] > 0 for e in EPSILONS)
    ]
    untrainable_scored = [
        row
        for row in curves["per_draw"]
        if row["constant_score"] and (row["harm_auroc"] is not None or row["top1"] is not None)
    ]
    record(
        "t15_an_untrainable_ranker_is_undefined_and_never_deploys",
        not untrainable_deploys and not untrainable_scored,
        {"deploying": untrainable_deploys[:5], "scored": len(untrainable_scored)},
    )
    control = cc_read_json(CONTROL_RESULTS)["label_permutation"]
    record(
        "t16_a_ranker_fitted_on_shuffled_grades_is_near_random",
        bool(control["near_random"]),
        {name: control["by_direction"][name]["median"] for name in DIRECTIONS},
    )
    deploying = _names_in("run_deployment") - {"oracle_accepted"}
    record(
        "t17_the_oracle_threshold_never_reaches_a_deployed_policy",
        "oracle_accepted" not in _names_in("gate")
        and "oracle_accepted" not in _names_in("policy_outcome")
        and "choose_threshold" in deploying,
        sorted(_names_in("gate") & {"oracle_accepted"}),
    )
    epsilon_leak = (_names_in("fit_arm") | _names_in("leaf_refit")) & {
        "EPSILONS",
        "PRIMARY_EPSILON",
        "epsilon_key",
        "choose_threshold",
    }
    record("t18_no_risk_target_reaches_a_fitted_ranker", not epsilon_leak, sorted(epsilon_leak))
    forbidden = sorted(
        _called_names()
        & {"select_threshold", "RiskController", "Calibrator", "clopper_pearson", "conformal"}
    )
    record("t19_no_certification_machinery_is_invoked", not forbidden, forbidden)
    generated = set(pd.read_parquet(hy1.CANDIDATES)["candidate_id"].astype(str))
    novel = sorted(set(population["candidate_id"].astype(str)) - generated)
    record(
        "t20_every_candidate_is_one_hy1_already_generated",
        not novel,
        {"candidates": int(population["candidate_id"].nunique()), "not_in_hy1": len(novel)},
    )
    design_issued = cc_read_json(DESIGN_RECORD)["issued_utc"]
    record(
        "t21_the_design_record_predates_every_endpoint",
        all(
            design_issued <= cc_read_json(path)["issued_utc"]
            for path in (SPLIT_REGISTRY, ADAPTATION_REGISTRY, RANKING_CURVES, DEPLOYMENT_CURVES)
        ),
        design_issued,
    )
    image_target = [
        row
        for row in curves["per_draw"]
        if row["direction"] == D_G2Q and (row["choice_sites"] != 0 or row["top1"] is not None)
    ]
    record(
        "t22_top1_is_undefined_not_chance_where_the_target_offers_no_choice",
        not image_target
        and splits["directions"][D_G2Q]["target_test_sites_with_several_candidates"] == 0,
        {"cells_with_a_top1": len(image_target)},
    )
    # A hand-built check that leaf refit can reverse what the source learned: the source ranks
    # a feature upward, the new rows grade it downward, and the refit must follow the new rows.
    toy_x = np.tile(np.arange(4.0), 30).reshape(-1, 1)
    toy_group = np.repeat(np.arange(30), 4).astype(str)
    toy_source = rk1.fit_lambdamart(toy_x, np.tile(np.array([0, 1, 2, 3]), 30), toy_group)
    toy_refit = leaf_refit(toy_source, toy_x, np.tile(np.array([3, 2, 1, 0]), 30), toy_group)
    probe = np.arange(4.0).reshape(-1, 1)
    record(
        "t23_leaf_refit_relearns_a_reversed_hand_built_ordering",
        bool(np.all(np.diff(toy_source.score(probe)) > 0))
        and bool(np.all(np.diff(toy_refit.score(probe)) < 0)),
        {"source": toy_source.score(probe).tolist(), "refit": toy_refit.score(probe).tolist()},
    )
    flipped: list[bool] = []
    frame_scores = _index_cells(scores)
    labels = population.set_index("candidate_id")
    for name in DIRECTIONS:
        key = cell_key(name, PRIMARY_ARM, DECISION_BUDGET, INFERENCE_DRAW)
        cal = decisions(labelled(frame_scores[(*key, "calibration")], labels))
        test = decisions(labelled(frame_scores[(*key, "test")], labels))
        inverted = test.assign(is_harmful=~test["is_harmful"], exact=~test["exact"])
        for epsilon in EPSILONS:
            flipped.append(
                gate(cal, test, epsilon, True)[0] == gate(cal, inverted, epsilon, True)[0]
            )
    record(
        "t24_inverting_every_test_label_leaves_every_threshold_unchanged",
        all(flipped),
        {"checked": len(flipped)},
    )
    record(
        "t25_every_cell_carries_every_draw",
        all(
            int(
                scores[
                    (scores["direction"] == name)
                    & (scores["arm"] == arm)
                    & (scores["budget"] == budget)
                    & (scores["evaluation_set"] == "test")
                ]["draw"].nunique()
            )
            == DRAWS
            for name in DIRECTIONS
            for arm in ADAPTED
            for budget in BUDGETS[1:]
        ),
        {"draws": DRAWS},
    )
    passed = sum(1 for test in tests if test["passed"])
    _write_json_once(
        FALSIFICATION,
        {
            **_analysis_envelope("falsification_tests"),
            "tests": tests,
            "passed": passed,
            "total": len(tests),
            "all_passed": passed == len(tests),
        },
    )
    failing = [test["test"] for test in tests if not test["passed"]]
    print(
        f"negative: {passed}/{len(tests)} pass ({time.monotonic() - started:.0f}s)"
        + (f", failing {failing}" if failing else "")
    )
    if failing:
        raise PhaseError(f"{len(failing)} falsification tests failed: {failing}")
    return 0


# ------------------------------------------------------------------ the decision


def criteria_from_artifacts() -> dict[str, Any]:
    """Every criterion, re-derived from the persisted artifacts and nothing else."""
    frontier = cc_read_json(FRONTIER)
    comparison = cc_read_json(ARM_COMPARISON)
    family = cc_read_json(STATISTICAL_TESTS)["primary_family"]
    n_star = {name: frontier["minimal_deployable_budget"][PRIMARY_ARM][name] for name in DIRECTIONS}
    ready = {name: frontier["ranking_ready_budget"][PRIMARY_ARM][name] for name in DIRECTIONS}
    outcome = assign_outcome(n_star)
    ranking_ready_both = bool(all(value is not None for value in ready.values()))
    supported = {
        name: bool(row["defined"] and row["survives_holm"]) for name, row in family.items()
    }
    return {
        "n_star": n_star,
        "ranking_ready_budget": ready,
        "ranking_ready_both": ranking_ready_both,
        "outcome": outcome,
        "next_stage": next_stage(outcome, ranking_ready_both),
        "frontier": frontier["minimal_deployable_budget"],
        "ranking_ready_frontier": frontier["ranking_ready_budget"],
        "q2": comparison["q2_reading"],
        "q3": comparison["q3_recovers_without_full_retraining"],
        "supported": supported,
        "family": family,
    }


def outcome_label(state: dict[str, Any]) -> str:
    parts = []
    for name in DIRECTIONS:
        budget = state["n_star"][name]
        shown = budget if budget is not None else "not reached"
        parts.append(f"{DIRECTION_SHORT[name]} N* {shown}")
    return f"{state['outcome']}: {OUTCOME_TAXONOMY[state['outcome']]} -- target-only " + "; ".join(
        parts
    )


def run_decide() -> int:
    """The frozen outcome rule, applied to the persisted artifacts."""
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    negative = cc_read_json(FALSIFICATION)
    curves = cc_read_json(RANKING_CURVES)
    deployment = cc_read_json(DEPLOYMENT_CURVES)["by_direction"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    headline = {
        name: {
            "zero_shot_harm_auroc": curves["by_direction"][name][A0]["0"]["harm_auroc"]["median"],
            "target_only_harm_auroc_at_saturation": curves["by_direction"][name][PRIMARY_ARM][
                str(SATURATION_BUDGET)
            ]["harm_auroc"]["median"],
            "seen_generator_reference_harm_auroc": curves["seen_generator_reference"][name][
                "harm_auroc"
            ],
            "target_only_recall_at_saturation": deployment[name][PRIMARY_ARM][
                str(SATURATION_BUDGET)
            ][key]["repair_recall"]["median"],
            "target_only_review_rate_at_saturation": deployment[name][PRIMARY_ARM][
                str(SATURATION_BUDGET)
            ][key]["review_rate"]["median"],
        }
        for name in DIRECTIONS
    }
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "primary_arm": PRIMARY_ARM,
            "primary_epsilon": PRIMARY_EPSILON,
            "harm_floor": HARM_FLOOR,
            "decision_budget": DECISION_BUDGET,
            "n_star_primary": state["n_star"],
            "ranking_ready_budget_primary": state["ranking_ready_budget"],
            "ranking_ready_in_both_directions": state["ranking_ready_both"],
            "label_cost_frontier": state["frontier"],
            "ranking_ready_frontier": state["ranking_ready_frontier"],
            "headline": headline,
            "q2_source_data": state["q2"],
            "q3_recovers_without_full_retraining": state["q3"],
            "statistically_supported": state["supported"],
            "statistical_family_surviving": cc_read_json(STATISTICAL_TESTS)["surviving"],
            "outcome": state["outcome"],
            "outcome_label": outcome_label(state),
            "recommended_next_stage": state["next_stage"],
            "production_ready": False,
            "certified": False,
            "deployable_claim": (
                "'deployable' is the pre-registered minimal definition on development "
                "environments: harm ranking at the floor and a working, target-holding gate in "
                "most draws. It is not a certification and not a production claim"
            ),
            "ready_for_external_confirmation": False,
            "confirmatory_reserve_consumed": False,
            "falsification_tests_passed": int(negative["passed"]),
            "falsification_tests_total": int(negative["total"]),
            "hypothesis_id": HYPOTHESIS,
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: {outcome_label(state)}")
    return 0


# ------------------------------------------------------------------ post-hoc threshold diagnostic


def run_diagnose() -> int:
    """POST HOC: why the within-budget threshold misses its target on the sealed test block.

    Added after the deployment results were read, and computed from the stored scores alone. It
    compares, draw by draw, what the calibration sites said at the chosen cutoff with what the
    test block then showed, and the ranker's harm AUROC on the two. It feeds no criterion.
    """
    started = time.monotonic()
    _require(DECISION, "decide")
    _forbid(THRESHOLD_DIAGNOSTIC)
    scores, population = load_scores()
    cells = _index_cells(scores)
    ranking = {
        (row["direction"], row["arm"], row["budget"], row["draw"]): row["harm_auroc"]
        for row in cc_read_json(RANKING_CURVES)["per_draw"]
    }
    out: dict[str, Any] = {}
    for name in DIRECTIONS:
        out[name] = {}
        for arm in ARMS:
            out[name][arm] = {}
            for budget in (0,) if arm == A0 else BUDGETS[1:]:
                rows: list[dict[str, Any]] = []
                for draw in draws_at(budget):
                    key = cell_key(name, arm, budget, draw)
                    if ranking[key] is None:
                        continue
                    cal = labelled(cells[(*key, "calibration")], population)
                    test = labelled(cells[(*key, "test")], population)
                    cal_decisions = decisions(cal)
                    test_decisions = decisions(test)
                    cutoff, accepted = gate(cal_decisions, test_decisions, PRIMARY_EPSILON, True)
                    believed = (
                        cal_decisions[cal_decisions["safety"] >= cutoff]
                        if cutoff is not None
                        else cal_decisions.head(0)
                    )
                    rows.append(
                        {
                            "calibration_harm_auroc": _num(
                                auroc(
                                    -cal["score"].to_numpy(np.float64),
                                    cal["is_harmful"].to_numpy(bool),
                                )
                            ),
                            "test_harm_auroc": ranking[key],
                            "abstains": cutoff is None,
                            "calibration_accepted": len(believed),
                            "calibration_harm_at_cutoff": rk1._share(
                                int(believed["is_harmful"].sum()), len(believed)
                            ),
                            "test_accepted": len(accepted),
                            "test_harm_at_cutoff": rk1._share(
                                int(accepted["is_harmful"].sum()), len(accepted)
                            ),
                        }
                    )
                gated = [r for r in rows if not r["abstains"] and r["test_accepted"] > 0]
                out[name][arm][str(budget)] = {
                    "draws_with_a_trainable_ranker": len(rows),
                    "draws_with_a_cutoff_accepting_on_test": len(gated),
                    "calibration_harm_auroc": _summary([r["calibration_harm_auroc"] for r in rows]),
                    "test_harm_auroc": _summary([r["test_harm_auroc"] for r in rows]),
                    "calibration_accepted": _summary([r["calibration_accepted"] for r in gated]),
                    "calibration_harm_at_cutoff": _summary(
                        [r["calibration_harm_at_cutoff"] for r in gated]
                    ),
                    "test_harm_at_cutoff": _summary([r["test_harm_at_cutoff"] for r in gated]),
                    "share_of_cutoffs_resting_on_ten_or_fewer_calibration_decisions": (
                        float(np.mean([r["calibration_accepted"] <= 10 for r in gated]))
                        if gated
                        else None
                    ),
                    "share_of_cutoffs_whose_test_harm_exceeds_the_target": (
                        float(
                            np.mean(
                                [
                                    r["test_harm_at_cutoff"] is not None
                                    and r["test_harm_at_cutoff"] > PRIMARY_EPSILON + HARM_TOLERANCE
                                    for r in gated
                                ]
                            )
                        )
                        if gated
                        else None
                    ),
                }
    _write_json_once(
        THRESHOLD_DIAGNOSTIC,
        {
            **_analysis_envelope("threshold_diagnostic"),
            "post_hoc": True,
            "added_after": "the deployment results were read",
            "feeds_no_criterion": True,
            "epsilon": PRIMARY_EPSILON,
            "by_direction": out,
            "what_it_compares": (
                "for each draw with a trainable ranker, the harm the calibration sites showed at "
                "the chosen cutoff against the harm the sealed test block showed at the same "
                "cutoff, and the ranker's harm AUROC on the calibration sites against the test "
                "block. The zero-shot arm's calibration sites are the source generator's"
            ),
            "two_readings": {
                "small_accept_sets": (
                    "a cutoff chosen as the loosest one meeting the target can rest on a handful "
                    "of calibration decisions, whose harm is then an optimistic estimate"
                ),
                "shared_pages": (
                    "calibration sites share pages with the fitting sites, while the test block "
                    "shares none, so the ranker can look better on calibration than on test"
                ),
            },
        },
    )
    print(f"diagnose: post-hoc threshold diagnostic written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = rk1.FIGURE_NOTE
SURFACE = rk1.SURFACE
INK = rk1.INK
INK_SECONDARY = rk1.INK_SECONDARY
GRID = rk1.GRID
# Categorical slots 1-3 of the reference palette, validated as a set on this surface. The third
# sits below 3:1 contrast, so every line also carries a marker and a direct label.
ARM_COLOURS = {A1: "#2a78d6", A2: "#eb6834", A3: "#1baf7a"}
ARM_MARKERS = {A1: "o", A2: "s", A3: "^"}
ARM_SHORT = {A0: "zero shot", A1: "target-only", A2: "joint", A3: "leaf refit"}
FIGURES = (
    "harm_auroc_by_budget.png",
    "benefit_and_top1_by_budget.png",
    "deployment_by_budget.png",
    "threshold_decomposition.png",
)


def run_figures() -> int:
    """Four figures, minimal academic style, every value read from a persisted artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    _require(THRESHOLD_DIAGNOSTIC, "diagnose")
    for path in (FIGURE_DIR, FIGURE_MANIFEST):
        _forbid(path)
    FIGURE_DIR.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "svg.hashsalt": "rk2",
        }
    )
    manifest: dict[str, Any] = {}

    def source_key(path: Path) -> str:
        stage = FIGURE_DIR.parent
        return str(path.relative_to(stage)) if path.is_relative_to(stage) else _relative(path)

    def save(fig: Any, name: str, sources: Sequence[Path], caption: str) -> None:
        fig.text(0.01, 0.005, FIGURE_NOTE, fontsize=5.5, color=INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = FIGURE_DIR / name
        fig.savefig(
            path, dpi=200, bbox_inches="tight", pad_inches=0.08, metadata={"Software": None}
        )
        plt.close(fig)
        manifest[name] = {
            "path": f"{FIGURE_DIR.name}/{name}",
            "caption": caption,
            "sources": {source_key(s): xr1._signature(s) for s in sources},
        }

    positions = np.arange(len(BUDGETS))
    ranking = cc_read_json(RANKING_CURVES)
    deployment = cc_read_json(DEPLOYMENT_CURVES)
    key = rl1.epsilon_key(PRIMARY_EPSILON)

    def series(
        rows: dict[str, Any], arm: str, pick: Callable[[dict[str, Any]], Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        middle, low, high = [], [], []
        for budget in BUDGETS:
            cell = rows[A0 if budget == 0 else arm][str(budget)]
            value = pick(cell)
            # A median resting on fewer than half the draws is not drawn: at the smallest
            # budgets the target-only ranker is trainable in one or two draws out of twenty.
            if isinstance(value, dict) and value["defined"] < value["draws"] * DRAW_MAJORITY:
                middle.append(np.nan)
                low.append(np.nan)
                high.append(np.nan)
            elif isinstance(value, dict):
                middle.append(np.nan if value["median"] is None else value["median"])
                low.append(np.nan if value["q1"] is None else value["q1"])
                high.append(np.nan if value["q3"] is None else value["q3"])
            else:
                middle.append(np.nan if value is None else float(value))
                low.append(np.nan)
                high.append(np.nan)
        return np.asarray(middle), np.asarray(low), np.asarray(high)

    def place_labels(ax: Any, ends: list[tuple[float, float, str]], gap: float) -> None:
        """Direct labels at the line ends, nudged apart so that no two overlap."""
        placed: list[float] = []
        for x, y, text in sorted(ends, key=lambda end: end[1]):
            y = max([y, *[p + gap for p in placed]])
            placed.append(y)
            ax.annotate(
                text,
                (x, y),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=6.5,
                color=INK_SECONDARY,
                va="center",
            )

    def arm_lines(
        ax: Any,
        rows: dict[str, Any],
        pick: Callable[[dict[str, Any]], Any],
        label: bool,
        gap: float = 0.03,
    ) -> None:
        ends: list[tuple[float, float, str]] = []
        for arm in ADAPTED:
            middle, low, high = series(rows, arm, pick)
            ax.fill_between(positions, low, high, color=ARM_COLOURS[arm], alpha=0.12, lw=0)
            ax.plot(
                positions,
                middle,
                color=ARM_COLOURS[arm],
                marker=ARM_MARKERS[arm],
                markersize=4,
                linewidth=1.6,
                label=ARM_SHORT[arm],
            )
            finite = np.flatnonzero(np.isfinite(middle))
            if label and finite.size:
                last = int(finite[-1])
                ends.append((float(positions[last]), float(middle[last]), ARM_SHORT[arm]))
        if ends:
            place_labels(ax, ends, gap)
        ax.set_xticks(positions, [str(b) for b in BUDGETS])
        ax.set_xlim(-0.3, len(BUDGETS) - 0.1)

    # Figure 1 -- harm AUROC against the label budget.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
    for ax, name in zip(axes, DIRECTIONS, strict=True):
        rows = ranking["by_direction"][name]
        arm_lines(ax, rows, lambda cell: cell["harm_auroc"], label=True)
        ax.axhline(HARM_FLOOR, color=INK, linewidth=0.9, linestyle="--")
        reference = ranking["seen_generator_reference"][name]["harm_auroc"]
        ax.axhline(reference, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
        ax.axhline(0.5, color=GRID, linewidth=0.8)
        ax.set_title(f"{DIRECTION_SHORT[name]} (target: {TARGET_OF[name]})", fontsize=8.5)
        ax.set_xlabel("labelled target candidates bought (N)")
    axes[0].set_ylabel("test harm AUROC, median over draws")
    axes[0].set_ylim(0.3, 1.0)
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle(
        "Harm ranking against the label budget (dashed: 0.75 floor; dotted: in-distribution "
        "ranker; no point where under half the draws are trainable)",
        fontsize=9,
    )
    save(fig, FIGURES[0], [RANKING_CURVES], "median test harm AUROC by arm and label budget")

    # Figure 2 -- benefit AUROC in both directions, Top-1 where it is defined.
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for ax, name in zip(axes[:2], DIRECTIONS, strict=True):
        rows = ranking["by_direction"][name]
        arm_lines(ax, rows, lambda cell: cell["benefit_auroc"], label=False)
        ax.axhline(
            ranking["seen_generator_reference"][name]["benefit_auroc"],
            color=INK_SECONDARY,
            linewidth=0.8,
            linestyle=":",
        )
        ax.axhline(0.5, color=GRID, linewidth=0.8)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"benefit AUROC, {DIRECTION_SHORT[name]}", fontsize=8.5)
        ax.set_xlabel("N")
    rows = ranking["by_direction"][D_Q2G]
    arm_lines(axes[2], rows, lambda cell: cell["top1"], label=False)
    axes[2].axhline(
        ranking["seen_generator_reference"][D_Q2G]["top1"],
        color=INK_SECONDARY,
        linewidth=0.8,
        linestyle=":",
    )
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title(f"Top-1 on choice sites, {DIRECTION_SHORT[D_Q2G]}", fontsize=8.5)
    axes[2].set_xlabel("N")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle(
        "Benefit ranking and within-site choice (Top-1 is undefined when the image corrector "
        "is the target: one candidate per site)",
        fontsize=8.5,
    )
    save(
        fig,
        FIGURES[1],
        [RANKING_CURVES],
        "median benefit AUROC by arm and budget, and Top-1 where the target offers a choice",
    )

    # Figure 3 -- the within-budget policy at the primary target.
    panels = (
        (lambda cell: cell[key]["repair_recall"], "exact-repair recall"),
        (lambda cell: cell[key]["review_rate"], "human review rate"),
        (lambda cell: cell["share_deployable"], "share of draws deployable"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 7.2), sharex=True)
    for column, name in enumerate(DIRECTIONS):
        rows = deployment["by_direction"][name]
        for row, (pick, title) in enumerate(panels):
            ax = axes[row, column]
            arm_lines(ax, rows, pick, label=row == 0 and column == 0, gap=0.05)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel(title if column == 0 else "")
            if row == 0:
                ax.set_title(f"{DIRECTION_SHORT[name]} (target: {TARGET_OF[name]})", fontsize=8.5)
            if row == 2:
                ax.axhline(DRAW_MAJORITY, color=INK, linewidth=0.9, linestyle="--")
                ax.set_xlabel("labelled target candidates bought (N)")
    axes[0, 1].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle(
        f"The within-budget policy at harm <= {PRIMARY_EPSILON} on the sealed test block "
        "(medians over draws; dashed: the deployable majority)",
        fontsize=9,
    )
    save(
        fig,
        FIGURES[2],
        [DEPLOYMENT_CURVES],
        "repair recall, review rate and deployable share of draws by arm and budget",
    )

    # Figure 4 -- POST HOC: what the calibration sites promised and what the test block delivered.
    diagnostic = cc_read_json(THRESHOLD_DIAGNOSTIC)["by_direction"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for column, name in enumerate(DIRECTIONS):
        rows = diagnostic[name][PRIMARY_ARM]
        ax = axes[0, column]
        for field, colour, marker, style, label in (
            ("calibration_harm_at_cutoff", INK_SECONDARY, "D", "--", "harm on calibration sites"),
            ("test_harm_at_cutoff", ARM_COLOURS[PRIMARY_ARM], "o", "-", "harm on the test block"),
        ):
            middle = np.asarray(
                [
                    np.nan
                    if budget == 0
                    or rows[str(budget)]["draws_with_a_trainable_ranker"] < DRAWS * DRAW_MAJORITY
                    or rows[str(budget)][field]["median"] is None
                    else rows[str(budget)][field]["median"]
                    for budget in BUDGETS
                ]
            )
            ax.plot(
                positions,
                middle,
                color=colour,
                marker=marker,
                markersize=4,
                linewidth=1.5,
                linestyle=style,
                label=label,
            )
        ax.axhline(PRIMARY_EPSILON, color=INK, linewidth=0.9, linestyle=":")
        ax.set_ylim(-0.02, 0.5)
        ax.set_title(f"target-only, {DIRECTION_SHORT[name]}", fontsize=8.5)
        ax.set_ylabel("selective harm at the cutoff" if column == 0 else "")
        rows = deployment["by_direction"][name]
        ax = axes[1, column]
        within, _l, _h = series(rows, PRIMARY_ARM, lambda cell: cell[key]["repair_recall"])
        oracle, _l, _h = series(
            rows, PRIMARY_ARM, lambda cell: cell[key]["oracle_repair_recall_analysis_only"]
        )
        ax.plot(
            positions,
            within,
            color=ARM_COLOURS[PRIMARY_ARM],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label="within-budget threshold",
        )
        ax.plot(
            positions,
            oracle,
            color=INK_SECONDARY,
            marker="D",
            markersize=4,
            linewidth=1.2,
            linestyle="--",
            label="oracle threshold (analysis only)",
        )
        ceiling = deployment["perfect_selection_ceiling"][name]["perfect_selection_repair_recall"]
        ax.axhline(ceiling, color=INK, linewidth=0.8, linestyle=":")
        ax.set_ylim(-0.01, 0.3)
        ax.set_xticks(positions, [str(b) for b in BUDGETS])
        ax.set_xlabel("labelled target candidates bought (N)")
        ax.set_ylabel("exact-repair recall" if column == 0 else "")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper right")
    axes[1, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle(
        f"POST HOC: the within-budget cutoff at harm <= {PRIMARY_EPSILON} (dotted top: the "
        "target; dotted bottom: perfect selection)",
        fontsize=8.5,
    )
    save(
        fig,
        FIGURES[3],
        [THRESHOLD_DIAGNOSTIC, DEPLOYMENT_CURVES],
        "post hoc: calibration against test harm at the chosen cutoff, and within-budget "
        "against oracle-threshold recall, for the primary arm",
    )

    missing = [name for name in FIGURES if name not in manifest]
    if missing:
        raise PhaseError(f"{len(missing)} figures were not produced: {missing}")
    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_analysis_envelope("figure_manifest"),
            "figures": manifest,
            "count": len(manifest),
            "note": FIGURE_NOTE,
            "synthetic": False,
            "style": "matplotlib only, minimal academic",
            "every_value_read_from_an_artifact": True,
        },
    )
    print(f"figures: {len(manifest)} written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ determinism and record

DERIVED_OUTPUTS = (
    "SPLIT_REGISTRY",
    "DRAW_REGISTRY",
    "LABEL_COMPOSITION",
    "REPRODUCTION",
    "ADAPTED_SCORES",
    "ADAPTATION_REGISTRY",
    "RANKING_CURVES",
    "DEPLOYMENT_CURVES",
    "FRONTIER",
    "ARM_COMPARISON",
    "CONTROL_RESULTS",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "DECISION",
    "THRESHOLD_DIAGNOSTIC",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("splits", run_splits),
        ("draws", run_draws),
        ("reproduce", run_reproduce),
        ("adapt", run_adapt),
        ("curves", run_curves),
        ("deployment", run_deployment),
        ("compare", run_compare),
        ("controls", run_controls),
        ("stats", run_stats),
        ("negative", run_negative),
        ("decide", run_decide),
        ("diagnose", run_diagnose),
        ("figures", run_figures),
    )


def _current_signatures() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in DERIVED_OUTPUTS:
        path = globals()[name]
        if isinstance(path, Path) and path.is_file():
            out[path.name] = xr1._signature(path)
        elif isinstance(path, Path) and path.is_dir():
            for figure in sorted(path.glob("*.png")):
                out[f"figures/{figure.name}"] = file_sha256(figure)
    return out


def _sandboxed(directory: Path) -> dict[str, str]:
    """Every derived phase, re-run from the frozen upstream inputs into a directory of its own."""
    import shutil

    saved = {name: globals()[name] for name in DERIVED_OUTPUTS}
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    try:
        for name in DERIVED_OUTPUTS:
            globals()[name] = directory / saved[name].name
        for _name, run in _derived_phases():
            run()
        return _current_signatures()
    finally:
        for name, path in saved.items():
            globals()[name] = path


def run_determinism() -> int:
    """Two independent regenerations from the frozen upstream inputs."""
    started = time.monotonic()
    _require(FIGURE_MANIFEST, "figures")
    _forbid(DETERMINISM)
    published = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(
        name for name in published if any(run.get(name) != published[name] for run in runs)
    )
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "derived_artifacts_compared": len(published),
            "derived_regenerations": len(runs),
            "differing_artifacts": differing,
            "all_derived_artifacts_identical": not differing,
            "all_runs_identical": not differing,
            "seeds": {
                "fit": FIT_SEED,
                "draws": rl2.DRAW_SEED,
                "calibration": CALIBRATION_SEED,
                "bootstrap": BOOTSTRAP_SEED,
                "permutation": PERMUTATION_SEED,
            },
            "outcome": cc_read_json(DECISION)["outcome"],
            "uses_ground_truth": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"determinism: {len(published)} artifacts x {len(runs)} regenerations, "
        f"identical {not differing} ({time.monotonic() - started:.0f}s)"
    )
    return 0


PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    DESIGN_RECORD,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE, FROZEN_CONFIGURATION, REPRODUCTION),
    "2. Research Questions and Pre-Registration": (DESIGN_RECORD, FRONTIER),
    "3. Directions and the Sealed Test Block": (SPLIT_REGISTRY, DESIGN_RECORD),
    "4. Label Acquisition and the Within-Budget Threshold": (
        DRAW_REGISTRY,
        LABEL_COMPOSITION,
        DESIGN_RECORD,
        SPLIT_REGISTRY,
    ),
    "5. Adaptation Arms": (
        DESIGN_RECORD,
        FROZEN_CONFIGURATION,
        ADAPTATION_REGISTRY,
        LABEL_COMPOSITION,
    ),
    "6. Reproduction Gates": (REPRODUCTION, DRAW_REGISTRY, SPLIT_REGISTRY, FALSIFICATION),
    "7. Ranking Recovery": (RANKING_CURVES, DESIGN_RECORD, ADAPTATION_REGISTRY),
    "8. Deployment Under a Label Budget": (DEPLOYMENT_CURVES, FRONTIER, DESIGN_RECORD),
    "9. The Label-Cost Frontier": (FRONTIER, DEPLOYMENT_CURVES, RANKING_CURVES, DESIGN_RECORD),
    "10. Why the Within-Budget Threshold Fails (Post Hoc)": (
        THRESHOLD_DIAGNOSTIC,
        DEPLOYMENT_CURVES,
        DESIGN_RECORD,
    ),
    "11. Source Data: Help or Negative Transfer": (
        ARM_COMPARISON,
        RANKING_CURVES,
        DEPLOYMENT_CURVES,
        STATISTICAL_TESTS,
        DESIGN_RECORD,
    ),
    "12. Adaptation Without Full Retraining": (
        ARM_COMPARISON,
        RANKING_CURVES,
        FRONTIER,
        STATISTICAL_TESTS,
        DESIGN_RECORD,
    ),
    "13. Statistical Tests": (STATISTICAL_TESTS, DESIGN_RECORD, RANKING_CURVES),
    "14. Controls and Falsification": (CONTROL_RESULTS, FALSIFICATION, RANKING_CURVES),
    "15. Limitations": (
        DESIGN_RECORD,
        SPLIT_REGISTRY,
        LABEL_COMPOSITION,
        RANKING_CURVES,
        DEPLOYMENT_CURVES,
        STATISTICAL_TESTS,
    ),
    "16. Research Decision": (DECISION, FRONTIER, STATISTICAL_TESTS, FALSIFICATION, DESIGN_RECORD),
    "17. Next Research Direction": (
        DECISION,
        FRONTIER,
        DEPLOYMENT_CURVES,
        RANKING_CURVES,
        ARM_COMPARISON,
        THRESHOLD_DIAGNOSTIC,
    ),
}


def run_record() -> int:
    """Provenance, the upstream re-hash, the decision re-derivation and the traceability index."""
    started = time.monotonic()
    for path in (PROVENANCE, TRACEABILITY):
        _forbid(path)
    missing = [p for p in PRODUCED if not p.exists() and p not in (PROVENANCE, TRACEABILITY)]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {[_relative(p) for p in missing][:5]}")
    if not cc_read_json(DETERMINISM)["all_runs_identical"]:
        raise PhaseError("determinism did not pass; the record will not be issued")
    freeze = cc_read_json(RESEARCH_FREEZE)
    moved = sorted(
        path for path, sha in freeze["upstream_sha256"].items() if file_sha256(REPO / path) != sha
    )
    if moved:
        raise PhaseError(f"{len(moved)} upstream files moved since the freeze: {moved[:4]}")
    global DECISION
    saved = DECISION
    rederived: list[bool] = []
    for run in range(2):
        directory = CACHE / "record" / f"run_{run}"
        directory.mkdir(parents=True, exist_ok=True)
        sandbox = directory / saved.name
        sandbox.unlink(missing_ok=True)
        try:
            DECISION = sandbox
            run_decide()
        finally:
            DECISION = saved
        again = cc_read_json(sandbox)
        published = cc_read_json(saved)
        rederived.append(
            all(
                again[key] == published[key]
                for key in published
                if key not in ("issued_utc", "runtime_seconds")
            )
        )
    if not all(rederived):
        raise PhaseError("the decision did not re-derive identically")
    audit = gen1.audit_report(REPORT, REPORT_SECTIONS)
    if audit["untraceable_numeric_claims"]:
        raise PhaseError(
            f"{audit['untraceable_numeric_claims']} untraceable claims, "
            f"e.g. {audit['untraceable'][:3]}"
        )
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "artifacts": {_relative(p): file_sha256(p) for p in PRODUCED if p.is_file()},
            "artifact_count": sum(1 for p in PRODUCED if p.is_file()),
            "figures": {
                f"figures/{p.name}": file_sha256(p) for p in sorted(FIGURE_DIR.glob("*.png"))
            },
            "upstream_files_rehashed": len(freeze["upstream_sha256"]),
            "upstream_files_moved": moved,
            "decision_rederived_identically": rederived,
            "regenerates_a_candidate": False,
            "regenerates_an_ocr_output": False,
            "changes_an_upstream_outcome": False,
            "issued_head": _git("rev-parse", "HEAD"),
            "report": {"path": _relative(REPORT), "sha256": file_sha256(REPORT)},
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        TRACEABILITY,
        {
            **_envelope("traceability"),
            "report": _relative(REPORT),
            "sections": {
                name: [_relative(p) for p in paths] for name, paths in REPORT_SECTIONS.items()
            },
            "audit": audit,
            "untraceable": audit["untraceable"],
            "rule": (
                "every number in the report is read from one of the artifacts its section names, "
                "and from the field that means what the sentence says"
            ),
        },
    )
    print(
        f"record: {sum(1 for p in PRODUCED if p.is_file())} artifacts, "
        f"report claims {audit['numeric_claims']}, untraceable "
        f"{audit['untraceable_numeric_claims']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    phases: dict[str, Callable[[], int]] = {
        "reconstruct": run_reconstruct,
        "freeze": run_freeze,
        "preregister": run_preregister,
        "splits": run_splits,
        "draws": run_draws,
        "reproduce": run_reproduce,
        "adapt": run_adapt,
        "curves": run_curves,
        "deployment": run_deployment,
        "compare": run_compare,
        "controls": run_controls,
        "stats": run_stats,
        "negative": run_negative,
        "decide": run_decide,
        "diagnose": run_diagnose,
        "figures": run_figures,
        "determinism": run_determinism,
        "record": run_record,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    for name in phases:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(argv)
    chosen = [name for name in phases if getattr(args, name)]
    if not chosen:
        parser.error("choose at least one phase")
    for name in chosen:
        began = time.monotonic()
        code = phases[name]()
        if code != 0:
            return code
        print(f"  [{name} {time.monotonic() - began:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
