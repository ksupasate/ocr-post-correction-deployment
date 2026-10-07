#!/usr/bin/env python3
"""SGV-TH1: can a within-budget decision boundary deploy an unseen correction generator safely?

SGV-RK2 closed with outcome D. About 250 labelled target candidates recovered the adapted ranker's
harm ordering in both transfer directions, and no budget up to 500 made the target generator
deployable, because the cutoff chosen on the within-budget calibration sites overshot its harm
target on the sealed test block in most draws. RK2's post-hoc diagnostic named two readings:
cutoffs resting on a handful of calibration decisions, and calibration sites sharing pages with the
fitting sites. This stage holds the ranking fixed and varies only the decision boundary.

**1. The ranking is RK2's.** Every strategy that uses a single ranker uses RK2's target-only ranker
(and, as a declared secondary, RK2's joint ranker), refitted with RK2's own function on RK2's own
rows and verified bit for bit against RK2's stored scores. The only variable is the boundary.

**2. Every label is counted and the test block is RK2's.** The purchased labels are RK2's -- which
are RL2's -- at every budget and draw. The sealed test block is RK2's, frozen by digest before any
boundary is computed, and no boundary reads a test label.

**3. The prior findings shape the design and are disclosed.** SGV15 and SGV15b showed on this
project that a candidate-level exact bound over-certifies when harm clusters by page, that a valid
fixed-sequence gate over a fine family refuses at these sample sizes, and that a distribution-free
document-level certificate needs on the order of 800 target documents. The bound-based strategies
here are therefore labelled as what they are, and the sealed test block across 20 draws is what
decides whether any of them is safe.

    --reconstruct   RK2, RK1, RL2 and the certification stages re-read from their own artifacts
    --freeze        upstream hashes, including the risk-bound code this stage calls
    --preregister   strategies, rankers, budgets, bounds, criteria, outcome rules
    --splits        the sealed test block, verified against RK2's, and the cross-fitting folds
    --score         RK2's rankers refitted and reproduced, source-calibration and cross-fit scores
    --calibrate     every strategy x ranker x direction x budget x draw x risk target
    --curves        violation, usefulness, coverage, calibration error, worst environment
    --frontier      the deployable frontier and the boundary-gap reading
    --controls      the random-ranking control
    --stats         paired exact sign tests over draws, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen upstream inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / DEPLOYMENT METHODOLOGY ONLY. No candidate is generated, no ranker is searched, no
bound is a certificate unless it is labelled one, nothing is production-ready. The confirmatory
reserve stays LOCKED and `ready_for_external_confirmation` is false by construction.
"""

from __future__ import annotations

import argparse
import ast
import functools
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_rk2_fewshot_ranker_adaptation as rk2
from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.risk.bounds import clopper_pearson_upper
from ocr_risk.risk.cluster_bounds import design_effect

rk1 = rk2.rk1
ds1 = rk2.ds1
rl2 = rk2.rl2
rl1 = rk2.rl1
hy1 = rk2.hy1
gen1 = rk2.gen1
xr1 = rk2.xr1
s15 = rk2.s15

REPO = rk2.REPO
OUT = REPO / "results/generated/sgv_th1_threshold_calibration"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"

SPLIT_REGISTRY = OUT / "split_registry.json"
CELL_SCORES = OUT / "cell_scores.parquet"
SCORE_REGISTRY = OUT / "score_registry.json"
REPRODUCTION = OUT / "reproduction.json"

POLICY_OUTCOMES = OUT / "policy_outcomes.parquet"
CALIBRATION_REGISTRY = OUT / "calibration_registry.json"
STRATEGY_CURVES = OUT / "strategy_curves.json"
FRONTIER = OUT / "deployable_frontier.json"
CONTROL_RESULTS = OUT / "control_results.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_th1/threshold_calibration.md"

SCHEMA_VERSION = 1
STAGE = "sgv_th1_threshold_calibration"
HYPOTHESIS = "SGV-TH1-D1"
STAGE_KIND = "DEVELOPMENT / DEPLOYMENT METHODOLOGY"

PhaseError = rk2.PhaseError
_relative = rk2._relative
_git = rk2._git
_write_json_once = rk2._write_json_once
_write_parquet_once = rk2._write_parquet_once
cc_read_json = rk2.cc_read_json
_require = rk2._require
_forbid = rk2._forbid
auroc = rk2.auroc
_summary = rk2._summary
_num = rk2._num

BOOTSTRAP_RESAMPLES = rk2.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rk2.BOOTSTRAP_SEED
ALPHA = rk2.ALPHA
FIT_SEED = rk2.FIT_SEED

# ------------------------------------------------------------------ the frozen design

DIRECTIONS = rk2.DIRECTIONS
D_G2Q = rk2.D_G2Q
D_Q2G = rk2.D_Q2G
SOURCE_OF = rk2.SOURCE_OF
TARGET_OF = rk2.TARGET_OF
DIRECTION_SHORT = rk2.DIRECTION_SHORT

BUDGETS = (25, 50, 100, 250, 500)
DECISION_BUDGET = 250
SATURATION_BUDGET = 500
DRAWS = rk2.DRAWS

# Rankers. Both are RK2's, refitted with RK2's function on RK2's rows. The joint ranker is the only
# one that can be fitted at the two smallest budgets, which is why it is carried at all.
R_TARGET = "target_only"
R_JOINT = "joint"
RANKERS = (R_TARGET, R_JOINT)
PRIMARY_RANKER = R_TARGET
RANKER_ARM = {R_TARGET: rk2.A1, R_JOINT: rk2.A2}

# Cross-fitting. Fold 0 is exactly RK2's held-back calibration sites, so RK2's ranker is member 0.
FOLDS = rk2.CALIBRATION_MODULUS
FOLD_SEED = rk2.CALIBRATION_SEED

S1 = "s1_source_transfer"
S2 = "s2_naive"
S3 = "s3_conservative"
S4 = "s4_adaptive"
S5 = "s5_abstention_aware"
STRATEGIES = (S1, S2, S3, S4, S5)
PRIMARY_STRATEGY = S5
SINGLE_RANKER_STRATEGIES = (S1, S2, S3)
ENSEMBLE_STRATEGIES = (S4, S5)
BOUND_STRATEGIES = (S3, S4, S5)
STRATEGY_LABEL = {
    S1: (
        "source threshold transfer: RK2's ranker, cut at the loosest cutoff meeting the target "
        "on the SOURCE generator's calibration fold. No target label chooses the cutoff"
    ),
    S2: (
        "naive calibration: RK2's rule exactly -- the loosest cutoff whose empirical harm meets "
        "the target on the held-back third of the purchased sites"
    ),
    S3: (
        "conservative calibration: the same held-back sites, cut at the loosest cutoff whose "
        "exact one-sided Clopper-Pearson upper bound on harm meets the target"
    ),
    S4: (
        "adaptive calibration: ranking and cutoff fitted jointly from every purchased label by "
        "three-fold cross-fitting. Scores are put on one scale by their quantile among unlabelled "
        "adaptation rows; the cutoff is the conservative rule on the out-of-fold decisions; a "
        "test candidate is accepted only if all three fitted rankers place it above the cutoff"
    ),
    S5: (
        "abstention-aware policy: the adaptive strategy, with the bound's sample size shrunk by "
        "the page design effect of the out-of-fold evidence, and no automation in a test "
        "environment that no purchased page came from. Coverage is the largest the two "
        "constraints allow; anything else is sent to review"
    ),
}

EPSILONS = rk2.EPSILONS
PRIMARY_EPSILON = rk2.PRIMARY_EPSILON
HARM_TOLERANCE = rk2.HARM_TOLERANCE
DELTA = 0.10

# Criteria, frozen before any TH1 endpoint. A boundary is safe at a budget when at most one draw
# in ten accepts edits whose realized test harm exceeds the target -- the same 0.1 the bound is
# declared at -- and useful when at least half its draws repair something while holding it.
VIOLATION_CEILING = 0.10
WORKING_FLOOR = 0.5
WORST_ENVIRONMENT_MIN_ACCEPTED = 5
CONTROL_SEED = 20260928

QUESTIONS = {
    "Q1": "can a within-budget decision boundary deploy an unseen generator safely?",
    "Q2": "which calibration strategy holds the harm target, and what coverage does it keep?",
    "Q3": "is the remaining bottleneck the decision boundary rather than candidate quality?",
}
OUTCOME_TAXONOMY = {
    "A": "a within-budget boundary deploys an unseen generator safely at or below the decision "
    "budget in both directions",
    "B": "a within-budget boundary deploys safely in both directions, but only beyond the "
    "decision budget",
    "C": "a within-budget boundary deploys safely in one direction only",
    "D": "no tested label budget yields a boundary that is both safe and useful",
}
NEXT_STAGE = {
    "A": "NEXT: TARGETED EXTERNAL CONFIRMATION OF THE GENERATOR-ONBOARDING PROTOCOL",
    "B": "NEXT: LABEL-EFFICIENT ACQUISITION FOR GENERATOR ONBOARDING",
    "C": "NEXT: DIRECTION-SPECIFIC DIAGNOSIS OF THE GENERATOR THAT CANNOT BE ONBOARDED",
    "D_safe": "NEXT: PAGE-BUDGET FRONTIER FOR A SAFE AND USEFUL BOUNDARY",
    "D_unsafe": "NEXT: STOP BOUNDARY HEURISTICS; RE-EXAMINE TOP-OF-LIST PURITY",
}
PRIMARY_FAMILY = (
    "P1_abstention_aware_vs_naive_violation",
    "P2_conservative_vs_naive_violation",
    "P3_abstention_aware_vs_naive_repair_recall",
    "P4_adaptive_vs_conservative_repair_recall",
)
FAMILY_PAIRS = {
    PRIMARY_FAMILY[0]: (S5, S2, "violation"),
    PRIMARY_FAMILY[1]: (S3, S2, "violation"),
    PRIMARY_FAMILY[2]: (S5, S2, "repair_recall"),
    PRIMARY_FAMILY[3]: (S4, S3, "repair_recall"),
}

UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_rk2": {
        "outcome": "D",
        "recommended_next_stage": "NEXT: WITHIN-BUDGET THRESHOLD CALIBRATION FOR A NEW GENERATOR",
        "production_ready": False,
        "ready_for_external_confirmation": False,
        "confirmatory_reserve_consumed": False,
        "falsification_tests_passed": 25,
        "falsification_tests_total": 25,
        "ranking_ready_in_both_directions": True,
    },
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_th1-{artifact}-v{SCHEMA_VERSION}",
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


def assign_outcome(n_star: dict[str, int | None]) -> str:
    """Exactly one outcome for the primary strategy, in the frozen precedence D, C, A, B."""
    reached = [name for name, budget in n_star.items() if budget is not None]
    if not reached:
        return "D"
    if len(reached) < len(n_star):
        return "C"
    if all(budget is not None and budget <= DECISION_BUDGET for budget in n_star.values()):
        return "A"
    return "B"


def next_stage(outcome: str, safe_everywhere: bool) -> str:
    """D splits by how it failed: safe but never useful, or not even safe."""
    if outcome != "D":
        return NEXT_STAGE[outcome]
    return NEXT_STAGE["D_safe"] if safe_everywhere else NEXT_STAGE["D_unsafe"]


# ------------------------------------------------------------------ section 0: frozen state

RISK_CODE = (
    REPO / "src/ocr_risk/risk/bounds.py",
    REPO / "src/ocr_risk/risk/cluster_bounds.py",
)


def _upstream_files() -> list[Path]:
    files = list(rk2._upstream_files())
    files.append(REPO / "scripts/sgv_rk2_fewshot_ranker_adaptation.py")
    for pattern in ("*.json", "*.parquet"):
        files.extend(sorted(rk2.OUT.glob(pattern)))
    files.append(rk2.REPORT)
    files.extend(RISK_CODE)
    for stage in ("sgv15_target_risk_certification", "sgv15b_document_level_certification"):
        files.append(REPO / "results/generated" / stage / "research_decision.json")
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = xr1._check
    checks = list(rk2.upstream_checks())
    decision = cc_read_json(rk2.DECISION)
    for key, value in UPSTREAM_EXPECTED["sgv_rk2"].items():
        checks.append(_check(f"sgv_rk2.{key}", decision.get(key), value))
    checks.append(
        _check("sgv_rk2.determinism", cc_read_json(rk2.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_rk2.report_traceability",
            cc_read_json(rk2.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
            0,
        )
    )
    for stage in ("sgv15_target_risk_certification", "sgv15b_document_level_certification"):
        prior = cc_read_json(REPO / "results/generated" / stage / "research_decision.json")
        checks.append(_check(f"{stage}.outcome", prior["outcome"]["label"], "D"))
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
    return any(p.exists() for p in (CELL_SCORES, POLICY_OUTCOMES, STRATEGY_CURVES, DECISION))


def run_freeze() -> int:
    started = time.monotonic()
    for path in (RESEARCH_FREEZE, FROZEN_CONFIGURATION):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("a TH1 endpoint already exists; the freeze must precede every one")
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {_relative(p): file_sha256(p) for p in _upstream_files()}
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "upstream_sha256": hashes,
            "upstream_file_count": len(hashes),
            "risk_code_hashed": [_relative(p) for p in RISK_CODE],
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status": _git("status", "--short"),
                "diff_stat": _git("diff", "--stat"),
            },
            "regenerates_a_candidate": False,
            "changes_a_ranker": False,
            "changes_an_upstream_outcome": False,
            "uses_ground_truth": False,
        },
    )
    rk2_decision = cc_read_json(rk2.DECISION)
    sgv15b = cc_read_json(
        REPO / "results/generated/sgv15b_document_level_certification/research_decision.json"
    )
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "rankers": {ranker: RANKER_ARM[ranker] for ranker in RANKERS},
            "rk2_outcome": rk2_decision["outcome"],
            "rk2_ranking_ready_budget": rk2_decision["ranking_ready_budget_primary"],
            "rk2_headline": rk2_decision["headline"],
            "rk2_calibration_seed": rk2.CALIBRATION_SEED,
            "rk2_calibration_modulus": rk2.CALIBRATION_MODULUS,
            "rl2_draw_seed": rl2.DRAW_SEED,
            "draws": DRAWS,
            "sgv15b_outcome": sgv15b["outcome"]["label"],
            "sgv15b_finding": (
                "a distribution-free document-level certificate for candidate-weighted harm needs "
                "on the order of 800 independent target documents; a fine family under "
                "fixed-sequence gatekeeping refused on every cell; the candidate-level exact bound "
                "over-certified under page clustering"
            ),
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
        raise PhaseError("a TH1 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "research_question": (
                "can a threshold calibration strategy enable safe deployment of an unseen OCR "
                "correction generator under a limited label budget?"
            ),
            "questions": QUESTIONS,
            "main_hypothesis": (
                "after ranking adaptation, the remaining bottleneck is not candidate quality but "
                "a reliable decision boundary"
            ),
            "held_fixed": (
                "the ranking: every single-ranker strategy uses RK2's ranker, refitted with RK2's "
                "function on RK2's rows and verified bit for bit against RK2's stored scores"
            ),
            "rankers": {
                R_TARGET: "RK2's target-only LambdaMART -- primary, as in RK2",
                R_JOINT: (
                    "RK2's joint LambdaMART -- secondary; the only one that can be fitted at the "
                    "two smallest budgets"
                ),
            },
            "primary_ranker": PRIMARY_RANKER,
            "strategies": STRATEGY_LABEL,
            "primary_strategy": PRIMARY_STRATEGY,
            "primary_strategy_reason": (
                "the brief asks for a deployment methodology; the abstention-aware policy is the "
                "complete one, and the other four are its baselines and its ablations"
            ),
            "reading_of_adaptive": (
                "the brief's 'joint ranking plus uncertainty-aware threshold' is read as ranking "
                "and cutoff fitted jointly from the same labels by cross-fitting, with the "
                "disagreement of the cross-fitted rankers as the uncertainty. It is not the joint "
                "source-plus-target ranker, which is carried separately as a secondary ranker"
            ),
            "cross_fitting": {
                "folds": FOLDS,
                "fold_rule": (
                    "a purchased site's fold is the same seeded hash of its site id that RK2 used, "
                    "modulo 3. Fold 0 is exactly RK2's held-back calibration sites, so the member "
                    "that holds out fold 0 is RK2's own ranker"
                ),
                "seed": FOLD_SEED,
                "members": (
                    "one ranker per fold, fitted on the purchased sites of the other two folds "
                    "(and, for the joint ranker, the source fit block)"
                ),
                "out_of_fold": (
                    "every purchased site is scored by the one member that never saw it"
                ),
                "common_scale": (
                    "each member's score is replaced by its mid-rank quantile among the member's "
                    "scores on the target adaptation rows NOT purchased in this draw and budget. "
                    "Those rows carry no label anywhere in this stage, and they are never test "
                    "rows"
                ),
                "test_score": (
                    "the mean of the three members' quantiles chooses the candidate at a site; the "
                    "minimum of the three decides acceptance, so a candidate is accepted only when "
                    "every member places it above the cutoff"
                ),
                "untrainable_member": "an ensemble with any untrainable member does not deploy",
            },
            "source_transfer": (
                "the cutoff is chosen by RK2's naive rule on the source generator's calibration "
                "fold scored by the same adapted ranker; the held-back target sites are simply "
                "unused, so the ranking stays identical to the other single-ranker strategies"
            ),
            "bound": {
                "family": "exact one-sided Clopper-Pearson, `risk.bounds.clopper_pearson_upper`",
                "delta": DELTA,
                "rule": (
                    "every distinct calibration score is a candidate cutoff; the bound is taken at "
                    "each, and the loosest cutoff whose bound meets the target is kept"
                ),
                "what_it_is_not": (
                    "not a simultaneous guarantee: multiplicity across cutoffs is not corrected, "
                    "because SGV15b found the valid fixed-sequence gate over a fine family refused "
                    "on every cell at these sample sizes. It also assumes candidates are "
                    "independent, which page clustering violates. Its coverage is measured on the "
                    "sealed test block, not assumed"
                ),
                "page_design_effect": (
                    "for the abstention-aware policy only: the design effect of the harm rate "
                    "across purchased pages, `risk.cluster_bounds.design_effect`, computed once "
                    "per cell over all out-of-fold decisions and floored at 1. The bound is taken "
                    "on the candidate counts divided by it. With fewer than two pages, or an "
                    "undefined design effect, the policy abstains. This is an effective-sample "
                    "correction, not a finite-sample certificate"
                ),
            },
            "environment_abstention": (
                "the abstention-aware policy sends to review every test decision in an "
                "environment that none of the purchased pages came from. The environment is "
                "operational metadata -- the OCR engine and corpus -- not a label"
            ),
            "label_budgets": list(BUDGETS),
            "decision_budget": DECISION_BUDGET,
            "decision_budget_reason": (
                "RK2's target-only ranker is ranking-ready at 250 labels in both directions, so "
                "250 is where the boundary alone is being tested"
            ),
            "saturation_budget": SATURATION_BUDGET,
            "draws": DRAWS,
            "epsilons": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "metrics": {
                "harmful_accepted_rate": "selective harm: harmful accepted over accepted",
                "repair_recall": (
                    "exact-repair recall over RK2's target-reachable error sites of the sealed "
                    "test block"
                ),
                "coverage": "accepted over target test decision sites",
                "review_rate": "one minus coverage",
                "violation": (
                    "the draw accepts something and its realized test harm exceeds the target"
                ),
                "working_safe_gate": (
                    "the draw accepts something, repairs at least one error site and holds the "
                    "target"
                ),
                "calibration_error": (
                    "the realized test harm at the chosen cutoff minus the harm the calibration "
                    "evidence showed at that cutoff; positive means the calibration was "
                    "optimistic. For bound strategies, whether the realized harm stayed under the "
                    "claimed bound is recorded too"
                ),
                "worst_environment": (
                    "across the test environments, the largest harmful accepted rate among those "
                    f"with at least {WORST_ENVIRONMENT_MIN_ACCEPTED} accepted decisions, and the "
                    "number of environments whose accepted edits exceed the target"
                ),
                "oracle": (
                    "analysis only: the loosest cutoff meeting the target on the test labels, "
                    "for the same ranking. It is never deployable and says what a perfect "
                    "boundary would deliver"
                ),
            },
            "deployable": {
                "safe": (
                    f"at most {VIOLATION_CEILING} of the draws violate the target at the primary "
                    "risk target"
                ),
                "useful": f"at least {WORKING_FLOOR} of the draws are a working safe gate",
                "cell": "safe and useful",
                "minimal_budget": (
                    "N* is the smallest tested budget from which the strategy is deployable at "
                    "every larger tested budget"
                ),
            },
            "outcome_rule": OUTCOME_TAXONOMY,
            "outcome_precedence": (
                "on the primary strategy with the primary ranker: D if neither direction reaches "
                "N*, C if one does, A if both reach it at or below the decision budget, else B"
            ),
            "next_stage": NEXT_STAGE,
            "next_stage_rule": (
                "D maps to D_safe when the primary strategy is safe at every budget in both "
                "directions -- it failed by abstaining -- and to D_unsafe otherwise"
            ),
            "hypothesis_reading": (
                "per direction, on the primary ranker at the saturation budget: a boundary gap "
                "exists when the oracle boundary is a working safe gate in at least half the draws "
                "while the naive boundary is not safe. The gap is closed when the primary strategy "
                "is deployable there. No oracle gap means the ranking, not the boundary, binds"
            ),
            "statistical_family": list(PRIMARY_FAMILY),
            "statistical_plan": {
                "unit": (
                    "the draw. The quantity under test is a property of a calibration procedure "
                    "across calibration samples, so a draw -- one random purchase of target "
                    "pages -- "
                    "is the replicate. The sealed test block is fixed, so every inference is "
                    "conditional on it, and draws that share purchased pages are not independent; "
                    "both limits are stated"
                ),
                "test": "exact two-sided sign test on the paired draws of both directions",
                "effect": (
                    "for violation, the difference in the share of draws violating; for recall, "
                    "the median paired difference"
                ),
                "interval": (
                    "percentile interval from resampling draws with replacement inside each "
                    "direction"
                ),
                "budget": DECISION_BUDGET,
                "ranker": PRIMARY_RANKER,
                "epsilon": PRIMARY_EPSILON,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "multiplicity": "Holm within the frozen family",
            },
            "controls": {
                "random_ranking": (
                    "every strategy that uses a bound, re-run with seeded uniform random scores in "
                    "place of the ranker's. A boundary that is safe on a real ranking but "
                    "certifies "
                    "a random one is not protecting anything; the bound strategies must violate "
                    f"in at most {VIOLATION_CEILING} of draws at every budget with random scores"
                ),
                "seed": CONTROL_SEED,
            },
            "disclosure_prior_findings": (
                "SGV15 and SGV15b, earlier stages of this project on the engine-shift axis, found "
                "that conservative certification controls false certification but refuses at "
                "practical budgets, that the candidate-level exact bound over-certifies under page "
                "clustering with design effects between 7 and 38, and that a distribution-free "
                "document-level certificate needs on the order of 800 target documents. SGV-RK2's "
                "post-hoc diagnostic found small calibration accept sets and shared pages. These "
                "findings shaped the strategies, and the bound strategies are expected to refuse "
                "at the smallest budgets"
            ),
            "non_goals": [
                "no candidate is generated, no ranker is re-selected or tuned",
                "no strategy parameter -- delta, the fold count, the design-effect floor, the "
                "environment rule -- is chosen after an endpoint",
                "no bound is called a certificate unless it is one",
                "no production claim and no external confirmation",
                "the confirmatory reserve is never unlocked",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ splits and folds


def fold_of(sites: Sequence[str]) -> np.ndarray:
    """Each site's cross-fitting fold: RK2's hash of the site id, modulo the fold count."""
    verdict = {
        site: int(canonical_hash({"seed": FOLD_SEED, "site": site})[:8], 16) % FOLDS
        for site in sorted(set(sites))
    }
    return np.asarray([verdict[site] for site in sites], dtype=np.int64)


def run_splits() -> int:
    """The sealed test block, verified against RK2's, frozen before any boundary exists."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    _forbid(SPLIT_REGISTRY)
    population = rk2.load_population()
    folds = fold_of(population["site_group"].astype(str).tolist())
    rk2_splits = cc_read_json(rk2.SPLIT_REGISTRY)["directions"]
    registry: dict[str, Any] = {}
    for name, blocks in rk2.all_blocks(population).items():
        pool = population.iloc[blocks.target_adapt]
        registry[name] = {
            "target_test_digest": rk2._digest(population, blocks.target_test),
            "target_test_candidates": int(blocks.target_test.size),
            "equals_rk2_test_block": bool(
                rk2._digest(population, blocks.target_test)
                == rk2_splits[name]["target_test_digest"]
            ),
            "adaptation_pool_candidates": int(blocks.target_adapt.size),
            "adaptation_pool_sites_by_fold": {
                str(k): int(pool[folds[blocks.target_adapt] == k]["site_group"].nunique())
                for k in range(FOLDS)
            },
            "fold_zero_is_rk2s_calibration": bool(
                np.array_equal(
                    folds[blocks.target_adapt] == 0,
                    rk2.calibration_mask(pool["site_group"].astype(str).tolist()),
                )
            ),
            "test_environments": sorted(
                set(population.iloc[blocks.target_test]["environment"].astype(str))
            ),
        }
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_analysis_envelope("split_registry"),
            "directions": registry,
            "rk2_leakage": cc_read_json(rk2.SPLIT_REGISTRY)["leakage"],
            "rk2_nothing_crosses": cc_read_json(rk2.SPLIT_REGISTRY)["nothing_crosses"],
            "every_test_block_is_rk2s": bool(
                all(row["equals_rk2_test_block"] for row in registry.values())
            ),
            "frozen_before_any_boundary": not POLICY_OUTCOMES.exists(),
        },
    )
    if not all(row["equals_rk2_test_block"] for row in registry.values()):
        raise PhaseError("the test block differs from RK2's; TH1 stops")
    print(f"splits: test blocks sealed and equal to RK2's ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ scoring


def quantiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mid-rank quantile of each value among the reference scores; ties share the middle."""
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    if ordered.size == 0:
        return np.full(len(values), np.nan)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return np.asarray((left + right) / (2.0 * ordered.size), dtype=np.float64)


def cross_fit(
    ranker: str,
    design: np.ndarray,
    grades: np.ndarray,
    groups: np.ndarray,
    source_fit: np.ndarray,
    bought: np.ndarray,
    folds: np.ndarray,
) -> list[Any]:
    """One member per fold, each fitted by RK2's own function on the other folds' sites."""
    return [
        rk2.fit_arm(
            RANKER_ARM[ranker],
            design,
            grades,
            groups,
            source_fit,
            bought[folds[bought] != k],
            None,
        )
        for k in range(FOLDS)
    ]


def run_score() -> int:
    """RK2's rankers refitted and reproduced; source-calibration, test and out-of-fold scores."""
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    for path in (CELL_SCORES, SCORE_REGISTRY, REPRODUCTION):
        _forbid(path)
    population = rk2.load_population()
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    design = rk1._design(population, matrix, rk1.model_columns(rk1.PRIMARY_RANKER))
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = population["document_id"].astype(str).to_numpy()
    folds = fold_of(population["site_group"].astype(str).tolist())
    ids = population["candidate_id"].astype(str).to_numpy()
    environments = population["environment"].astype(str).to_numpy()
    documents = population["document_id"].astype(str).to_numpy()
    sealed = {
        name: row["target_test_digest"]
        for name, row in cc_read_json(SPLIT_REGISTRY)["directions"].items()
    }
    rk2_scores = pd.read_parquet(rk2.ADAPTED_SCORES)
    rk2_cells = {
        key: group.set_index("candidate_id")["score"]
        for key, group in rk2_scores[rk2_scores["arm"].isin(set(RANKER_ARM.values()))].groupby(
            ["arm", "direction", "budget", "draw", "evaluation_set"], sort=True
        )
    }
    frames: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []
    reproduction: dict[str, dict[str, float]] = {
        ranker: {"test": 0.0, "calibration": 0.0, "cells": 0} for ranker in RANKERS
    }
    for ranker in RANKERS:
        for name, blocks in rk2.all_blocks(population).items():
            if rk2._digest(population, blocks.target_test) != sealed[name]:
                raise PhaseError(f"{name}: the test block differs from the sealed one")
            began = time.monotonic()
            for draw in range(DRAWS):
                order = rl2.draw_order(population, blocks.frozen, draw)
                for budget in BUDGETS:
                    bought = rl2.purchased(order, budget)
                    members = cross_fit(
                        ranker, design, grades, groups, blocks.source_fit, bought, folds
                    )
                    reference = np.setdiff1d(blocks.target_adapt, bought)
                    test = blocks.target_test
                    raw_test = [m.score(design[test]) for m in members]
                    raw_reference = [m.score(design[reference]) for m in members]
                    trainable = [not rk2._constant(values) for values in raw_test]
                    q_test = [
                        quantiles(ref, values)
                        for ref, values in zip(raw_reference, raw_test, strict=True)
                    ]
                    base = {"ranker": ranker, "direction": name, "budget": budget, "draw": draw}
                    frames.append(
                        pd.DataFrame(
                            {
                                **base,
                                "evaluation_set": "test",
                                "candidate_id": ids[test],
                                "fold": -1,
                                "raw": raw_test[0],
                                "q_mean": np.mean(q_test, axis=0),
                                "q_min": np.min(q_test, axis=0),
                                "q_oof": np.nan,
                            }
                        )
                    )
                    source_cal = blocks.source_calibration
                    frames.append(
                        pd.DataFrame(
                            {
                                **base,
                                "evaluation_set": "source_calibration",
                                "candidate_id": ids[source_cal],
                                "fold": -1,
                                "raw": members[0].score(design[source_cal]),
                                "q_mean": np.nan,
                                "q_min": np.nan,
                                "q_oof": np.nan,
                            }
                        )
                    )
                    for k, member in enumerate(members):
                        held = bought[folds[bought] == k]
                        raw_held = member.score(design[held]) if held.size else np.empty(0)
                        frames.append(
                            pd.DataFrame(
                                {
                                    **base,
                                    "evaluation_set": "oof",
                                    "candidate_id": ids[held],
                                    "fold": k,
                                    "raw": raw_held,
                                    "q_mean": np.nan,
                                    "q_min": np.nan,
                                    "q_oof": quantiles(raw_reference[k], raw_held)
                                    if held.size
                                    else np.empty(0),
                                }
                            )
                        )
                    arm = RANKER_ARM[ranker]
                    stored_test = rk2_cells[(arm, name, budget, draw, "test")]
                    reproduction[ranker]["test"] = max(
                        reproduction[ranker]["test"],
                        float(
                            np.abs(
                                stored_test.loc[ids[test]].to_numpy(np.float64) - raw_test[0]
                            ).max()
                        ),
                    )
                    held0 = bought[folds[bought] == 0]
                    if held0.size:
                        stored_cal = rk2_cells[(arm, name, budget, draw, "calibration")]
                        if len(stored_cal) != held0.size:
                            raise PhaseError("RK2's calibration rows are not fold 0")
                        reproduction[ranker]["calibration"] = max(
                            reproduction[ranker]["calibration"],
                            float(
                                np.abs(
                                    stored_cal.loc[ids[held0]].to_numpy(np.float64)
                                    - members[0].score(design[held0])
                                ).max()
                            ),
                        )
                    reproduction[ranker]["cells"] += 1
                    registry.append(
                        {
                            **base,
                            "members_trainable": trainable,
                            "member_target_rows": [
                                int((folds[bought] != k).sum()) for k in range(FOLDS)
                            ],
                            "out_of_fold_rows": int(bought.size),
                            "reference_rows": int(reference.size),
                            "reference_digest": str(canonical_hash(ids[reference].tolist())),
                            "reference_touches_purchase_or_test": bool(
                                np.intersect1d(reference, bought).size
                                or np.intersect1d(reference, test).size
                            ),
                            "purchased_documents": sorted(set(documents[bought].tolist())),
                            "purchased_environments": sorted(set(environments[bought].tolist())),
                        }
                    )
            print(f"  {ranker} {name}: {time.monotonic() - began:.0f}s", flush=True)
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(
        ["ranker", "direction", "budget", "draw", "evaluation_set", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    _write_parquet_once(CELL_SCORES, table)
    identical = all(
        row["test"] == 0.0 and row["calibration"] == 0.0 for row in reproduction.values()
    )
    _write_json_once(
        REPRODUCTION,
        {
            **_analysis_envelope("reproduction"),
            "by_ranker": {
                ranker: {
                    "rk2_arm": RANKER_ARM[ranker],
                    "cells": int(row["cells"]),
                    "max_absolute_test_difference": row["test"],
                    "max_absolute_calibration_difference": row["calibration"],
                }
                for ranker, row in reproduction.items()
            },
            "identical": identical,
            "note": (
                "member 0 of every cross-fit is RK2's ranker: refitted here with RK2's function on "
                "RK2's fit rows and compared with RK2's stored test and calibration scores"
            ),
        },
    )
    untrainable = sum(1 for row in registry if not all(row["members_trainable"]))
    _write_json_once(
        SCORE_REGISTRY,
        {
            **_analysis_envelope("score_registry"),
            "cells": registry,
            "cell_count": len(registry),
            "fits": len(registry) * FOLDS,
            "scored_rows": len(table),
            "cells_with_an_untrainable_member": untrainable,
            "cells_whose_rk2_ranker_is_untrainable": sum(
                1 for row in registry if not row["members_trainable"][0]
            ),
            "reference_rule": cc_read_json(DESIGN_RECORD)["cross_fitting"]["common_scale"],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if not identical:
        raise PhaseError(f"RK2's rankers did not reproduce: {reproduction}")
    print(
        f"score: {len(registry)} cells, {len(registry) * FOLDS} fits, RK2 reproduced exactly "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the boundaries

LABELS = rk2.LABELS


@functools.cache
def _upper(harmful: float, total: float, delta: float) -> float:
    return float(clopper_pearson_upper(harmful, total, delta))  # type: ignore[arg-type]


def bound_threshold(
    calibration: pd.DataFrame, epsilon: float, delta: float, deff: float = 1.0
) -> tuple[float | None, float | None]:
    """The loosest cutoff whose Clopper-Pearson upper bound on calibration harm meets the target.

    Every distinct calibration score is a candidate cutoff and accept sets are "score at or above
    the cutoff", exactly as in RK1's empirical rule. The bound is taken on the counts divided by
    `deff`, so a design effect above one shrinks the sample the bound believes it has. The rule is
    pointwise: multiplicity across cutoffs is not corrected, and the design record says why.
    """
    if calibration.empty:
        return None, None
    safety = calibration["safety"].to_numpy(dtype=np.float64)
    harmful = calibration["is_harmful"].to_numpy(dtype=bool)
    order = np.argsort(-safety, kind="mergesort")
    ordered = safety[order]
    cumulative_harm = np.cumsum(harmful[order])
    last_of_tie = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    chosen: float | None = None
    bound: float | None = None
    for position in last_of_tie:
        total = float(position + 1)
        upper = _upper(float(cumulative_harm[position]) / deff, total / deff, delta)
        if upper <= epsilon + HARM_TOLERANCE:
            chosen, bound = float(ordered[position]), upper
    return chosen, bound


def page_design_effect(calibration: pd.DataFrame) -> float | None:
    """The design effect of the harm rate across purchased pages, floored at one.

    None -- and the abstention-aware policy then abstains -- when fewer than two pages carry
    calibration decisions or the design effect is undefined.
    """
    if calibration.empty:
        return None
    per_page = calibration.groupby("document_id")["is_harmful"].agg(["sum", "size"])
    if len(per_page) < 2:
        return None
    value = design_effect(
        per_page["sum"].to_numpy(dtype=np.float64), per_page["size"].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(value):
        return None
    return float(max(1.0, value))


def _share(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def environment_harm(accepted: pd.DataFrame, epsilon: float) -> dict[str, Any]:
    """Harm by test environment among the accepted edits."""
    if accepted.empty:
        return {
            "environments_accepting": 0,
            "environments_exceeding": 0,
            "worst_environment_harm": None,
        }
    per = accepted.groupby("environment")["is_harmful"].agg(["sum", "size"])
    rate = per["sum"] / per["size"]
    eligible = rate[per["size"] >= WORST_ENVIRONMENT_MIN_ACCEPTED]
    return {
        "environments_accepting": len(per),
        "environments_exceeding": int((rate > epsilon + HARM_TOLERANCE).sum()),
        "worst_environment_harm": float(eligible.max()) if len(eligible) else None,
    }


def evaluate(
    strategy: str,
    epsilon: float,
    frames: dict[str, pd.DataFrame],
    trainable: dict[str, bool],
    environments: set[str],
    keys: set[str],
    mapping: dict[str, set[str]],
) -> dict[str, Any]:
    """One strategy's boundary on one cell: chosen without test labels, then read on the test."""
    if strategy in SINGLE_RANKER_STRATEGIES:
        test = frames["test_single"]
        calibration = frames["source" if strategy == S1 else "held_back"]
        usable = trainable["single"]
    else:
        test = frames["test_ensemble"]
        calibration = frames["out_of_fold"]
        usable = trainable["ensemble"]
    cutoff: float | None = None
    bound: float | None = None
    deff: float | None = None
    if usable:
        if strategy in (S1, S2):
            cutoff = rk1.choose_threshold(calibration, epsilon)
        elif strategy in (S3, S4):
            cutoff, bound = bound_threshold(calibration, epsilon, DELTA)
        else:
            deff = page_design_effect(calibration)
            if deff is not None:
                cutoff, bound = bound_threshold(calibration, epsilon, DELTA, deff)
    accepted = test[test["safety"] >= cutoff] if cutoff is not None else test.head(0)
    abstained_environments = 0
    if strategy == S5 and not accepted.empty:
        outside = ~accepted["environment"].astype(str).isin(environments)
        abstained_environments = int(accepted.loc[outside, "environment"].nunique())
        accepted = accepted[~outside]
    row = rk2.policy_outcome(test, accepted, keys, mapping, epsilon)
    believed = (
        calibration[calibration["safety"] >= cutoff] if cutoff is not None else calibration.head(0)
    )
    calibration_harm = _share(float(believed["is_harmful"].sum()), len(believed))
    realized = row["selective_harm_rate"]
    oracle = rk2.oracle_accepted(test, epsilon) if usable else test.head(0)
    oracle_row = rk2.policy_outcome(test, oracle, keys, mapping, epsilon)
    return {
        "strategy": strategy,
        "epsilon": epsilon,
        "trainable": bool(usable),
        "cutoff": cutoff,
        "abstains": cutoff is None,
        "calibration_decisions": len(calibration),
        "calibration_accepted": len(believed),
        "calibration_harm": calibration_harm,
        "claimed_bound": bound,
        "design_effect": deff,
        "accepted": row["accepted"],
        "sites": row["sites"],
        "coverage": row["coverage"],
        "review_rate": row["review_rate"],
        "selective_harm_rate": realized,
        "joint_harm_rate": row["joint_harm_rate"],
        "repair_recall": row["repair_recall"],
        "repaired_error_sites": row["repaired_error_sites"],
        "holds": row["holds_its_target"],
        "violation": bool(row["accepted"] > 0 and not row["holds_its_target"]),
        "working_safe_gate": row["working_safe_gate"],
        "harm_estimation_error": (
            None if realized is None or calibration_harm is None else realized - calibration_harm
        ),
        "bound_covers": (
            None if bound is None or realized is None else bool(realized <= bound + HARM_TOLERANCE)
        ),
        "abstained_environments": abstained_environments,
        "accepted_environments": ";".join(sorted(set(accepted["environment"].astype(str)))),
        **environment_harm(accepted, epsilon),
        "oracle_repair_recall": oracle_row["repair_recall"],
        "oracle_working": oracle_row["working_safe_gate"],
    }


def cell_frames(
    cell: dict[str, pd.DataFrame], labels: pd.DataFrame, replace: Callable[..., np.ndarray] | None
) -> dict[str, pd.DataFrame]:
    """The labelled decision frames every strategy reads for one cell.

    `replace`, when given, swaps every score for a seeded random one: the random-ranking control.
    """

    def join(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.join(labels[list(LABELS)], on="candidate_id", how="inner").reset_index(
            drop=True
        )

    def swap(values: pd.Series, tag: str) -> np.ndarray:
        array = values.to_numpy(dtype=np.float64)
        return replace(tag, array.size) if replace is not None else array

    test = join(cell["test"])
    single = test.assign(rank_score=swap(test["raw"], "t"), safety=swap(test["raw"], "t"))
    mean_q = swap(test["q_mean"], "m")
    ensemble = test.assign(
        rank_score=mean_q,
        safety=swap(test["q_min"], "m") if replace is not None else test["q_min"],
    )
    oof = join(cell["oof"])
    held = oof[oof["fold"] == 0]
    source = join(cell["source_calibration"])
    return {
        "test_single": rk2.decisions(single),
        "test_ensemble": rk2.decisions(ensemble),
        "held_back": rk2.decisions(
            held.assign(rank_score=swap(held["raw"], "h"), safety=swap(held["raw"], "h"))
        ),
        "out_of_fold": rk2.decisions(
            oof.assign(rank_score=swap(oof["q_oof"], "o"), safety=swap(oof["q_oof"], "o"))
        ),
        "source": rk2.decisions(
            source.assign(rank_score=swap(source["raw"], "s"), safety=swap(source["raw"], "s"))
        ),
    }


def _cells(scores: pd.DataFrame) -> dict[tuple[str, str, int, int], dict[str, pd.DataFrame]]:
    out: dict[tuple[str, str, int, int], dict[str, pd.DataFrame]] = {}
    for (ranker, name, budget, draw, kind), group in scores.groupby(
        ["ranker", "direction", "budget", "draw", "evaluation_set"], sort=True
    ):
        out.setdefault((str(ranker), str(name), int(budget), int(draw)), {})[str(kind)] = group
    return out


def _test_context(
    population: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    errors = ds1.load_error_population()
    mapping = ds1.site_to_errors(errors)
    keys: dict[str, set[str]] = {}
    for name, blocks in rk2.all_blocks(population).items():
        sites = set(population.iloc[blocks.target_test]["site_key"].astype(str))
        keys[name] = rk2.reachable_error_keys(sites, errors, mapping)
    return keys, mapping


def run_strategies(
    replace_for: Callable[[tuple[str, str, int, int]], Callable[..., np.ndarray] | None],
    strategies: Sequence[str],
    rankers: Sequence[str],
    force_trainable: bool = False,
) -> list[dict[str, Any]]:
    """Every named strategy on every cell of the named rankers, at every risk target."""
    population = rk2.load_population()
    labels = population.set_index("candidate_id")
    keys, mapping = _test_context(population)
    registry = {
        (row["ranker"], row["direction"], row["budget"], row["draw"]): row
        for row in cc_read_json(SCORE_REGISTRY)["cells"]
    }
    rows: list[dict[str, Any]] = []
    for key, cell in _cells(pd.read_parquet(CELL_SCORES)).items():
        if key[0] not in rankers:
            continue
        ranker, name, budget, draw = key
        info = registry[key]
        frames = cell_frames(cell, labels, replace_for(key))
        trainable = {
            "single": force_trainable or bool(info["members_trainable"][0]),
            "ensemble": force_trainable or bool(all(info["members_trainable"])),
        }
        environments = set(info["purchased_environments"])
        for strategy in strategies:
            for epsilon in EPSILONS:
                rows.append(
                    {
                        "ranker": ranker,
                        "direction": name,
                        "budget": budget,
                        "draw": draw,
                        **evaluate(
                            strategy,
                            epsilon,
                            frames,
                            trainable,
                            environments,
                            keys[name],
                            mapping,
                        ),
                    }
                )
    return rows


def run_calibrate() -> int:
    """Every strategy x ranker x direction x budget x draw x risk target, on the sealed test."""
    started = time.monotonic()
    _require(REPRODUCTION, "score")
    for path in (POLICY_OUTCOMES, CALIBRATION_REGISTRY):
        _forbid(path)
    rows = run_strategies(lambda _key: None, STRATEGIES, RANKERS)
    table = pd.DataFrame(rows).sort_values(
        ["ranker", "strategy", "direction", "budget", "draw", "epsilon"], kind="stable"
    )
    _write_parquet_once(POLICY_OUTCOMES, table.reset_index(drop=True))
    _write_json_once(
        CALIBRATION_REGISTRY,
        {
            **_analysis_envelope("calibration_registry"),
            "strategies": STRATEGY_LABEL,
            "rows": len(table),
            "rankers": list(RANKERS),
            "delta": DELTA,
            "epsilons": list(EPSILONS),
            "cutoffs_chosen_without_test_labels": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"calibrate: {len(table)} policy outcomes ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ curves and frontier


def _values(frame: pd.DataFrame, column: str) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in frame[column].tolist()]


def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    """Across draws: how often a boundary violates, works or abstains, and what it delivers."""
    accepting = frame[frame["accepted"] > 0]
    return {
        "draws": len(frame),
        "violation_share": float(frame["violation"].mean()),
        "working_share": float(frame["working_safe_gate"].mean()),
        "abstain_share": float(frame["abstains"].mean()),
        "accept_nothing_share": float((frame["accepted"] == 0).mean()),
        "untrainable_draws": int((~frame["trainable"]).sum()),
        "repair_recall": _summary(_values(frame, "repair_recall")),
        "coverage": _summary(_values(frame, "coverage")),
        "review_rate": _summary(_values(frame, "review_rate")),
        "selective_harm_rate": _summary(_values(accepting, "selective_harm_rate")),
        "calibration_accepted": _summary(_values(frame, "calibration_accepted")),
        "calibration_harm": _summary(_values(accepting, "calibration_harm")),
        "harm_estimation_error": _summary(_values(frame, "harm_estimation_error")),
        "mean_harm_estimation_error": _num(
            frame["harm_estimation_error"].dropna().astype(float).mean()
            if frame["harm_estimation_error"].notna().any()
            else None
        ),
        "bound_coverage_share": _num(
            frame["bound_covers"].dropna().astype(bool).mean()
            if frame["bound_covers"].notna().any()
            else None
        ),
        "design_effect": _summary(_values(frame, "design_effect")),
        "worst_environment_harm": _summary(_values(accepting, "worst_environment_harm")),
        "median_environments_exceeding": float(frame["environments_exceeding"].median()),
        "draws_with_an_environment_exceeding": int((frame["environments_exceeding"] > 0).sum()),
        "oracle_repair_recall": _summary(_values(frame, "oracle_repair_recall")),
        "oracle_working_share": float(frame["oracle_working"].mean()),
    }


def run_curves() -> int:
    started = time.monotonic()
    _require(CALIBRATION_REGISTRY, "calibrate")
    _forbid(STRATEGY_CURVES)
    table = pd.read_parquet(POLICY_OUTCOMES)
    scores = pd.read_parquet(CELL_SCORES)
    labels = rk2.load_population().set_index("candidate_id")
    curves: dict[str, Any] = {}
    for (ranker, strategy, name, budget, epsilon), frame in table.groupby(
        ["ranker", "strategy", "direction", "budget", "epsilon"], sort=True
    ):
        curves.setdefault(str(ranker), {}).setdefault(str(strategy), {}).setdefault(
            str(name), {}
        ).setdefault(str(int(budget)), {})[rl1.epsilon_key(float(epsilon))] = aggregate(frame)
    ranking: dict[str, Any] = {}
    test = scores[scores["evaluation_set"] == "test"]
    for (ranker, name, budget), frame in test.groupby(["ranker", "direction", "budget"]):
        single: list[float | None] = []
        ensemble: list[float | None] = []
        for _draw, cell in frame.groupby("draw"):
            harmful = labels.loc[cell["candidate_id"], "is_harmful"].to_numpy(dtype=bool)
            raw = cell["raw"].to_numpy(dtype=np.float64)
            mean_q = cell["q_mean"].to_numpy(dtype=np.float64)
            single.append(None if rk2._constant(raw) else auroc(-raw, harmful))
            ensemble.append(None if rk2._constant(mean_q) else auroc(-mean_q, harmful))
        ranking.setdefault(str(ranker), {}).setdefault(str(name), {})[str(int(budget))] = {
            "rk2_ranker_harm_auroc": _summary(single),
            "cross_fit_ensemble_harm_auroc": _summary(ensemble),
        }
    _write_json_once(
        STRATEGY_CURVES,
        {
            **_analysis_envelope("strategy_curves"),
            "by_ranker": curves,
            "ranking_quality": ranking,
            "block": "sealed target test block",
            "violation_ceiling": VIOLATION_CEILING,
            "working_floor": WORKING_FLOOR,
            "delta": DELTA,
            "metrics": cc_read_json(DESIGN_RECORD)["metrics"],
        },
    )
    print(f"curves: {len(table)} outcomes aggregated ({time.monotonic() - started:.0f}s)")
    return 0


def deployable(cell: dict[str, Any]) -> dict[str, bool]:
    safe = bool(cell["violation_share"] <= VIOLATION_CEILING)
    useful = bool(cell["working_share"] >= WORKING_FLOOR)
    return {"safe": safe, "useful": useful, "deployable": safe and useful}


def run_frontier() -> int:
    """N* for every strategy and ranker, and the pre-registered boundary-gap reading."""
    started = time.monotonic()
    _require(STRATEGY_CURVES, "curves")
    _forbid(FRONTIER)
    curves = cc_read_json(STRATEGY_CURVES)["by_ranker"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    frontier: dict[str, Any] = {}
    flags: dict[str, Any] = {}
    for ranker in RANKERS:
        for strategy in STRATEGIES:
            for name in DIRECTIONS:
                per_budget = {
                    b: deployable(curves[ranker][strategy][name][str(b)][key]) for b in BUDGETS
                }
                flags.setdefault(ranker, {}).setdefault(strategy, {})[name] = {
                    str(b): v for b, v in per_budget.items()
                }
                frontier.setdefault(ranker, {}).setdefault(strategy, {})[name] = {
                    "n_star": rk2.minimal_budget(
                        {b: v["deployable"] for b, v in per_budget.items()}
                    ),
                    "safe_at_every_budget": all(v["safe"] for v in per_budget.values()),
                    "safe_budgets": [b for b, v in per_budget.items() if v["safe"]],
                    "useful_budgets": [b for b, v in per_budget.items() if v["useful"]],
                }
    gap: dict[str, Any] = {}
    for name in DIRECTIONS:
        saturation = str(SATURATION_BUDGET)
        naive = curves[PRIMARY_RANKER][S2][name][saturation][key]
        primary = curves[PRIMARY_RANKER][PRIMARY_STRATEGY][name][saturation][key]
        oracle_useful = bool(naive["oracle_working_share"] >= WORKING_FLOOR)
        naive_unsafe = bool(naive["violation_share"] > VIOLATION_CEILING)
        closed = deployable(primary)["deployable"]
        if not oracle_useful:
            reading = "no boundary gap: even an oracle boundary is not useful, so the ranking binds"
        elif not naive_unsafe:
            reading = "no boundary gap: the naive boundary is already safe"
        elif closed:
            reading = "the boundary is the bottleneck, and the primary strategy closes it"
        else:
            reading = "the boundary is the bottleneck, and no within-budget boundary closed it"
        gap[name] = {
            "oracle_working_share": naive["oracle_working_share"],
            "naive_violation_share": naive["violation_share"],
            "primary_violation_share": primary["violation_share"],
            "primary_working_share": primary["working_share"],
            "boundary_gap": bool(oracle_useful and naive_unsafe),
            "closed": bool(closed),
            "reading": reading,
        }
    _write_json_once(
        FRONTIER,
        {
            **_analysis_envelope("deployable_frontier"),
            "frontier": frontier,
            "by_budget": flags,
            "hypothesis_reading": gap,
            "definition": cc_read_json(DESIGN_RECORD)["deployable"],
            "primary_strategy": PRIMARY_STRATEGY,
            "primary_ranker": PRIMARY_RANKER,
            "primary_epsilon": PRIMARY_EPSILON,
            "violation_ceiling": VIOLATION_CEILING,
            "working_floor": WORKING_FLOOR,
        },
    )
    primary = {n: frontier[PRIMARY_RANKER][PRIMARY_STRATEGY][n]["n_star"] for n in DIRECTIONS}
    print(f"frontier: primary N* {primary} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ controls


def random_scores(key: tuple[str, str, int, int]) -> Callable[[str, int], np.ndarray]:
    """Seeded uniform scores for one cell, drawn per frame so each frame is its own draw."""

    def draw(tag: str, size: int) -> np.ndarray:
        seed = int(canonical_hash({"seed": CONTROL_SEED, "cell": list(key), "frame": tag})[:8], 16)
        return np.random.default_rng(seed).random(size)

    return draw


def run_controls() -> int:
    """The bound strategies must not certify a random ranking."""
    started = time.monotonic()
    _require(FRONTIER, "frontier")
    _forbid(CONTROL_RESULTS)
    # A random ranking is always a ranking, so the control scores it even at the budgets where
    # the real ranker cannot be fitted; otherwise the control would pass there by abstaining.
    rows = run_strategies(
        random_scores, (S2, *BOUND_STRATEGIES), (PRIMARY_RANKER,), force_trainable=True
    )
    table = pd.DataFrame(rows)
    table = table[table["epsilon"] == PRIMARY_EPSILON]
    out: dict[str, Any] = {}
    for (strategy, name, budget), frame in table.groupby(["strategy", "direction", "budget"]):
        out.setdefault(str(strategy), {}).setdefault(str(name), {})[str(int(budget))] = {
            "violation_share": float(frame["violation"].mean()),
            "working_share": float(frame["working_safe_gate"].mean()),
            "abstain_share": float(frame["abstains"].mean()),
        }
    worst = {
        strategy: max(
            cell["violation_share"]
            for by_budget in out[strategy].values()
            for cell in by_budget.values()
        )
        for strategy in out
    }
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "random_ranking": out,
            "worst_violation_share": worst,
            "bound_strategies_protect": bool(
                all(worst[s] <= VIOLATION_CEILING for s in BOUND_STRATEGIES)
            ),
            "ranker": PRIMARY_RANKER,
            "epsilon": PRIMARY_EPSILON,
            "seed": CONTROL_SEED,
            "rule": cc_read_json(DESIGN_RECORD)["controls"]["random_ranking"],
        },
    )
    print(f"controls: worst random-ranking violation {worst} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ statistics


def sign_test(differences: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided sign test on paired differences; ties carry no information."""
    from scipy.stats import binomtest

    positive = int((differences > 0).sum())
    negative = int((differences < 0).sum())
    total = positive + negative
    p = float(binomtest(positive, total, 0.5).pvalue) if total else 1.0
    return p, positive, negative


def paired_draws(
    table: pd.DataFrame, left: str, right: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two strategies' values on the same draws of both directions, with each draw's direction."""
    frame = table[
        (table["ranker"] == PRIMARY_RANKER)
        & (table["budget"] == DECISION_BUDGET)
        & (table["epsilon"] == PRIMARY_EPSILON)
    ]
    wide = frame.pivot_table(
        index=["direction", "draw"], columns="strategy", values=metric, aggfunc="first"
    ).sort_index()
    return (
        wide[left].to_numpy(dtype=np.float64),
        wide[right].to_numpy(dtype=np.float64),
        wide.index.get_level_values("direction").to_numpy(),
    )


def draw_comparison(
    name: str, table: pd.DataFrame, left: str, right: str, metric: str
) -> dict[str, Any]:
    a, b, direction = paired_draws(table, left, right, metric)
    difference = a - b

    def effect(values_a: np.ndarray, values_b: np.ndarray) -> float:
        if metric == "violation":
            return float(values_a.mean() - values_b.mean())
        return float(np.median(values_a - values_b))

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero(direction == d) for d in sorted(set(direction.tolist()))]
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for resample in range(BOOTSTRAP_RESAMPLES):
        index = np.concatenate([generator.choice(s, s.size, replace=True) for s in strata])
        draws[resample] = effect(a[index], b[index])
    p, positive, negative = sign_test(difference)
    return {
        "comparison": name,
        "left": left,
        "right": right,
        "metric": metric,
        "effect": effect(a, b),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_value": p,
        "draws_left_higher": positive,
        "draws_right_higher": negative,
        "ties": int(difference.size - positive - negative),
        "units": int(difference.size),
        "left_value": float(a.mean()) if metric == "violation" else float(np.median(a)),
        "right_value": float(b.mean()) if metric == "violation" else float(np.median(b)),
    }


def run_stats() -> int:
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    table = pd.read_parquet(POLICY_OUTCOMES)
    table["violation"] = table["violation"].astype(float)
    family = {
        name: draw_comparison(name, table, left, right, metric)
        for name, (left, right, metric) in FAMILY_PAIRS.items()
    }
    adjusted = s15.holm(family)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "primary_family": adjusted,
            "family_size": len(PRIMARY_FAMILY),
            "declared_before_any_endpoint": True,
            "budget": DECISION_BUDGET,
            "ranker": PRIMARY_RANKER,
            "epsilon": PRIMARY_EPSILON,
            "plan": cc_read_json(DESIGN_RECORD)["statistical_plan"],
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "direction_of_effects": (
                "left minus right: a negative violation effect favours the left strategy; a "
                "negative recall effect is what the left strategy's safety costs"
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


def run_negative() -> int:
    """Each test would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    population = rk2.load_population()
    labels = population.set_index("candidate_id")
    scores = pd.read_parquet(CELL_SCORES)
    table = pd.read_parquet(POLICY_OUTCOMES)
    splits = cc_read_json(SPLIT_REGISTRY)
    registry = cc_read_json(SCORE_REGISTRY)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    columns = rk1.model_columns(rk1.PRIMARY_RANKER)
    identity = [c for c in columns if c.startswith(rl1.FAM_SOURCE)]
    leaked = sorted(set(columns) & rk1.LABEL_FIELDS)
    record(
        "t01_no_generator_identity_or_label_field_is_a_design_column",
        not identity and not leaked,
        {"identity": identity, "label": leaked},
    )
    record(
        "t02_the_test_block_is_rk2s_and_was_sealed_before_any_boundary",
        bool(splits["every_test_block_is_rk2s"])
        and bool(splits["frozen_before_any_boundary"])
        and splits["issued_utc"] <= cc_read_json(CALIBRATION_REGISTRY)["issued_utc"],
        {name: row["equals_rk2_test_block"] for name, row in splits["directions"].items()},
    )
    blocks = rk2.all_blocks(population)
    test_ids = {
        name: set(population.iloc[b.target_test]["candidate_id"].astype(str))
        for name, b in blocks.items()
    }
    crossing = {
        name: int(
            scores[(scores["direction"] == name) & (scores["evaluation_set"] != "test")][
                "candidate_id"
            ]
            .isin(test_ids[name])
            .sum()
        )
        for name in DIRECTIONS
    }
    record(
        "t03_no_test_row_is_purchased_calibrated_on_or_used_as_the_quantile_reference",
        not any(crossing.values())
        and not any(row["reference_touches_purchase_or_test"] for row in registry["cells"]),
        crossing,
    )
    record(
        "t04_no_document_site_or_edit_group_crosses_into_the_test_block",
        bool(splits["rk2_nothing_crosses"]),
        splits["rk2_leakage"],
    )
    reproduction = cc_read_json(REPRODUCTION)
    record(
        "t05_rk2s_rankers_reproduce_exactly",
        bool(reproduction["identical"]),
        reproduction["by_ranker"],
    )
    rk2_outcomes = pd.DataFrame(cc_read_json(rk2.DEPLOYMENT_CURVES)["per_draw"])
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    mismatches = 0
    compared = 0
    naive = table[(table["strategy"] == S2) & (table["epsilon"] == PRIMARY_EPSILON)]
    for ranker in RANKERS:
        arm = RANKER_ARM[ranker]
        ours = naive[naive["ranker"] == ranker].set_index(["direction", "budget", "draw"])
        theirs = rk2_outcomes[rk2_outcomes["arm"] == arm]
        for _i, row in theirs.iterrows():
            if int(row["budget"]) not in BUDGETS or int(row["draw"]) >= DRAWS:
                continue
            mine = ours.loc[(row["direction"], int(row["budget"]), int(row["draw"]))]
            compared += 1
            same = (
                int(mine["accepted"]) == int(row[key]["accepted"])
                and float(mine["repair_recall"]) == float(row[key]["repair_recall"])
                and (mine["cutoff"] if pd.notna(mine["cutoff"]) else None) == row[key]["threshold"]
            )
            mismatches += 0 if same else 1
    record(
        "t06_naive_calibration_reproduces_rk2s_deployment_exactly",
        mismatches == 0 and compared == len(RANKERS) * len(DIRECTIONS) * len(BUDGETS) * DRAWS,
        {"compared": compared, "mismatches": mismatches},
    )
    folds = fold_of(population["site_group"].astype(str).tolist())
    fold_by_id = dict(zip(population["candidate_id"].astype(str), folds.tolist(), strict=True))
    oof = scores[scores["evaluation_set"] == "oof"]
    wrong_fold = int((oof["candidate_id"].map(fold_by_id) != oof["fold"]).sum())
    record(
        "t07_every_out_of_fold_score_comes_from_the_member_that_never_saw_the_site",
        wrong_fold == 0 and bool(splits["directions"][D_G2Q]["fold_zero_is_rk2s_calibration"]),
        {"rows": len(oof), "wrong_fold": wrong_fold},
    )
    # Inverting every test label must leave every cutoff where it was.
    cells = _cells(scores)
    info = {(r["ranker"], r["direction"], r["budget"], r["draw"]): r for r in registry["cells"]}
    keys, mapping = _test_context(population)
    flipped = labels.copy()
    for name in DIRECTIONS:
        index = population.iloc[blocks[name].target_test]["candidate_id"]
        flipped.loc[index, "is_harmful"] = ~flipped.loc[index, "is_harmful"]
        flipped.loc[index, "exact"] = ~flipped.loc[index, "exact"]
    unchanged: list[bool] = []
    for cell_key_ in sorted(cells)[:: max(1, len(cells) // 40)]:
        entry = info[cell_key_]
        trainable = {
            "single": bool(entry["members_trainable"][0]),
            "ensemble": bool(all(entry["members_trainable"])),
        }
        honest = cell_frames(cells[cell_key_], labels, None)
        inverted = cell_frames(cells[cell_key_], flipped, None)
        for strategy in STRATEGIES:
            a = evaluate(
                strategy,
                PRIMARY_EPSILON,
                honest,
                trainable,
                set(entry["purchased_environments"]),
                keys[cell_key_[1]],
                mapping,
            )
            b = evaluate(
                strategy,
                PRIMARY_EPSILON,
                inverted,
                trainable,
                set(entry["purchased_environments"]),
                keys[cell_key_[1]],
                mapping,
            )
            unchanged.append(a["cutoff"] == b["cutoff"])
    record(
        "t08_inverting_every_test_label_leaves_every_cutoff_unchanged",
        all(unchanged),
        {"checked": len(unchanged)},
    )
    s1 = table[(table["strategy"] == S1) & table["trainable"]]
    source_sites = {
        name: int(population.iloc[blocks[name].source_calibration]["site_group"].nunique())
        for name in DIRECTIONS
    }
    record(
        "t09_the_source_transfer_cutoff_reads_the_source_calibration_fold_only",
        bool(
            all(
                int(row["calibration_decisions"]) == source_sites[row["direction"]]
                for _i, row in s1.iterrows()
            )
        ),
        {"rows": len(s1), "source_calibration_sites": source_sites},
    )
    bounded = table[table["strategy"].isin(BOUND_STRATEGIES) & table["claimed_bound"].notna()]
    record(
        "t10_no_bound_strategy_deploys_a_cutoff_whose_bound_exceeds_the_target",
        bool((bounded["claimed_bound"] <= bounded["epsilon"] + HARM_TOLERANCE).all())
        and bool(
            table[table["strategy"].isin(BOUND_STRATEGIES) & table["cutoff"].notna()][
                "claimed_bound"
            ]
            .notna()
            .all()
        ),
        {"rows": len(bounded)},
    )
    # Twenty-two clean decisions are the fewest at which the bound can reach 0.1 at delta 0.1:
    # 1 - 0.1 ** (1 / 22) is 0.0994, and with twenty-one it is 0.1038.
    record(
        "t11_the_clopper_pearson_bound_matches_hand_arithmetic",
        abs(_upper(0.0, 22.0, DELTA) - (1.0 - DELTA ** (1.0 / 22.0))) < 1e-12
        and _upper(0.0, 22.0, DELTA) <= 0.1 < _upper(0.0, 21.0, DELTA),
        {"n22": _upper(0.0, 22.0, DELTA), "n21": _upper(0.0, 21.0, DELTA)},
    )
    s5 = table[(table["strategy"] == S5) & (table["accepted"] > 0)]
    outside = 0
    for _i, row in s5.iterrows():
        entry = info[(row["ranker"], row["direction"], int(row["budget"]), int(row["draw"]))]
        automated = set(str(row["accepted_environments"]).split(";"))
        outside += 0 if automated <= set(entry["purchased_environments"]) else 1
    record(
        "t12_the_abstention_aware_policy_never_automates_an_unpurchased_environment",
        outside == 0,
        {"rows": len(s5), "violations": outside},
    )
    untrainable = table[~table["trainable"]]
    record(
        "t13_an_untrainable_ranker_never_deploys",
        bool((untrainable["accepted"] == 0).all()),
        {"rows": len(untrainable)},
    )
    test = scores[scores["evaluation_set"] == "test"]
    record(
        "t14_the_unanimity_score_never_exceeds_the_mean_score",
        bool((test["q_min"] <= test["q_mean"] + 1e-12).all()),
        {"rows": len(test)},
    )
    deploying = _names_in("evaluate") - {"oracle_accepted"}
    record(
        "t15_the_oracle_boundary_never_reaches_a_deployed_cutoff",
        "oracle_accepted" not in _names_in("bound_threshold")
        and "oracle_accepted" not in _names_in("page_design_effect")
        and "choose_threshold" in deploying,
        sorted(_names_in("bound_threshold") & {"oracle_accepted"}),
    )
    epsilon_leak = _names_in("cross_fit") & {"EPSILONS", "PRIMARY_EPSILON", "choose_threshold"}
    record("t16_no_risk_target_reaches_a_fitted_ranker", not epsilon_leak, sorted(epsilon_leak))
    generated = set(pd.read_parquet(hy1.CANDIDATES)["candidate_id"].astype(str))
    novel = sorted(set(population["candidate_id"].astype(str)) - generated)
    record("t17_every_candidate_is_one_hy1_already_generated", not novel, {"novel": len(novel)})
    design_issued = cc_read_json(DESIGN_RECORD)["issued_utc"]
    record(
        "t18_the_design_record_predates_every_endpoint",
        all(
            design_issued <= cc_read_json(path)["issued_utc"]
            for path in (SPLIT_REGISTRY, SCORE_REGISTRY, CALIBRATION_REGISTRY, STRATEGY_CURVES)
        ),
        design_issued,
    )
    control = cc_read_json(CONTROL_RESULTS)
    record(
        "t19_no_bound_strategy_certifies_a_random_ranking",
        bool(control["bound_strategies_protect"]),
        control["worst_violation_share"],
    )
    # Mid-rank quantiles by hand: against [1, 2, 2, 3], 2 sits at (1 + 3) / 8 and 0 at 0.
    hand = quantiles(np.array([1.0, 2.0, 2.0, 3.0]), np.array([2.0, 0.0, 4.0]))
    record(
        "t20_quantiles_match_hand_arithmetic",
        bool(np.allclose(hand, [0.5, 0.0, 1.0])),
        hand.tolist(),
    )
    # A design effect of four turns 3 harms in 40 into 0.75 in 10, whose bound no longer meets 0.1.
    frame = pd.DataFrame(
        {"safety": np.linspace(1.0, 0.0, 40), "is_harmful": [False] * 37 + [True] * 3}
    )
    plain, _ = bound_threshold(frame, 0.2, DELTA, 1.0)
    shrunk, _ = bound_threshold(frame, 0.2, DELTA, 4.0)
    record(
        "t21_a_design_effect_above_one_makes_the_bound_stricter",
        plain is not None and (shrunk is None or shrunk > plain),
        {"plain": plain, "shrunk": shrunk},
    )
    forbidden = sorted(
        {"select_threshold", "RiskController", "Calibrator", "conformal"}
        & (_names_in("evaluate") | _names_in("bound_threshold"))
    )
    record("t22_no_other_threshold_machinery_is_invoked", not forbidden, forbidden)
    budgets_ok = sorted(set(table["budget"].astype(int))) == list(BUDGETS) and all(
        int(group["draw"].nunique()) == DRAWS
        for _k, group in table.groupby(["ranker", "strategy", "direction", "budget"])
    )
    record("t23_every_cell_carries_every_draw_and_budget", budgets_ok, {"draws": DRAWS})
    record(
        "t24_every_outcome_uses_the_declared_delta_and_risk_targets",
        sorted(set(table["epsilon"].astype(float))) == sorted(EPSILONS)
        and cc_read_json(CALIBRATION_REGISTRY)["delta"] == DELTA,
        {"epsilons": sorted(set(table["epsilon"].astype(float)))},
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
    family = cc_read_json(STATISTICAL_TESTS)["primary_family"]
    primary = frontier["frontier"][PRIMARY_RANKER][PRIMARY_STRATEGY]
    n_star = {name: primary[name]["n_star"] for name in DIRECTIONS}
    safe_everywhere = bool(all(primary[name]["safe_at_every_budget"] for name in DIRECTIONS))
    outcome = assign_outcome(n_star)
    return {
        "n_star": n_star,
        "safe_everywhere": safe_everywhere,
        "outcome": outcome,
        "next_stage": next_stage(outcome, safe_everywhere),
        "frontier": {
            ranker: {
                strategy: {name: row["n_star"] for name, row in by_direction.items()}
                for strategy, by_direction in by_strategy.items()
            }
            for ranker, by_strategy in frontier["frontier"].items()
        },
        "hypothesis_reading": frontier["hypothesis_reading"],
        "supported": {name: bool(row["survives_holm"]) for name, row in family.items()},
    }


def outcome_label(state: dict[str, Any]) -> str:
    parts = []
    for name in DIRECTIONS:
        budget = state["n_star"][name]
        shown = budget if budget is not None else "not reached"
        parts.append(f"{DIRECTION_SHORT[name]} N* {shown}")
    return (
        f"{state['outcome']}: {OUTCOME_TAXONOMY[state['outcome']]} -- abstention-aware policy "
        + "; ".join(parts)
    )


def run_decide() -> int:
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    negative = cc_read_json(FALSIFICATION)
    curves = cc_read_json(STRATEGY_CURVES)["by_ranker"][PRIMARY_RANKER]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    headline = {
        strategy: {
            name: {
                str(b): {
                    "violation_share": curves[strategy][name][str(b)][key]["violation_share"],
                    "working_share": curves[strategy][name][str(b)][key]["working_share"],
                    "median_repair_recall": curves[strategy][name][str(b)][key]["repair_recall"][
                        "median"
                    ],
                }
                for b in (DECISION_BUDGET, SATURATION_BUDGET)
            }
            for name in DIRECTIONS
        }
        for strategy in STRATEGIES
    }
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "primary_strategy": PRIMARY_STRATEGY,
            "primary_ranker": PRIMARY_RANKER,
            "primary_epsilon": PRIMARY_EPSILON,
            "delta": DELTA,
            "n_star_primary": state["n_star"],
            "primary_safe_at_every_budget": state["safe_everywhere"],
            "frontier": state["frontier"],
            "hypothesis_reading": state["hypothesis_reading"],
            "headline": headline,
            "statistically_supported": state["supported"],
            "statistical_family_surviving": cc_read_json(STATISTICAL_TESTS)["surviving"],
            "outcome": state["outcome"],
            "outcome_label": outcome_label(state),
            "recommended_next_stage": state["next_stage"],
            "production_ready": False,
            "certified": False,
            "no_bound_here_is_a_certificate": True,
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


# ------------------------------------------------------------------ figures

FIGURE_NOTE = rk2.FIGURE_NOTE
SURFACE = rk2.SURFACE
INK = rk2.INK
INK_SECONDARY = rk2.INK_SECONDARY
GRID = rk2.GRID
# Categorical slots 1-5 of the reference palette, validated as a set on this surface. Three sit
# below 3:1 contrast, so every series also carries its own marker and appears in the legend.
STRATEGY_COLOURS = {
    S1: "#2a78d6",
    S2: "#eb6834",
    S3: "#1baf7a",
    S4: "#eda100",
    S5: "#e87ba4",
}
STRATEGY_MARKERS = {S1: "o", S2: "s", S3: "^", S4: "D", S5: "v"}
STRATEGY_SHORT = {
    S1: "source transfer",
    S2: "naive",
    S3: "conservative",
    S4: "adaptive",
    S5: "abstention-aware",
}
FIGURES = (
    "violation_by_budget.png",
    "working_by_budget.png",
    "recall_and_review_by_budget.png",
    "calibration_error.png",
)


def run_figures() -> int:
    """Four figures, minimal academic style, every value read from a persisted artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    _require(DECISION, "decide")
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
            "svg.hashsalt": "th1",
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

    curves = cc_read_json(STRATEGY_CURVES)["by_ranker"][PRIMARY_RANKER]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    positions = np.arange(len(BUDGETS))

    def line(ax: Any, strategy: str, values: Sequence[float | None], label: bool) -> None:
        ax.plot(
            positions,
            [np.nan if v is None else v for v in values],
            color=STRATEGY_COLOURS[strategy],
            marker=STRATEGY_MARKERS[strategy],
            markersize=4,
            linewidth=1.5,
            label=STRATEGY_SHORT[strategy] if label else None,
        )

    def share_figure(field: str, reference: float, title: str, ylabel: str, name: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
        for index, (ax, direction) in enumerate(zip(axes, DIRECTIONS, strict=True)):
            for strategy in STRATEGIES:
                values = [curves[strategy][direction][str(b)][key][field] for b in BUDGETS]
                line(ax, strategy, values, label=index == 0)
            ax.axhline(reference, color=INK, linewidth=0.9, linestyle="--")
            ax.set_xticks(positions, [str(b) for b in BUDGETS])
            ax.set_xlabel("labelled target candidates bought (N)")
            ax.set_title(
                f"{DIRECTION_SHORT[direction]} (target: {TARGET_OF[direction]})", fontsize=8.5
            )
            ax.set_ylim(-0.03, 1.03)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
        fig.suptitle(title, fontsize=9)
        save(fig, name, [STRATEGY_CURVES], title)

    share_figure(
        "violation_share",
        VIOLATION_CEILING,
        f"Share of draws whose accepted edits exceed harm {PRIMARY_EPSILON} on the sealed test "
        f"block (dashed: the {VIOLATION_CEILING} safety ceiling)",
        "share of draws violating the target",
        FIGURES[0],
    )
    share_figure(
        "working_share",
        WORKING_FLOOR,
        f"Share of draws that repair something and hold harm {PRIMARY_EPSILON} (dashed: the "
        f"{WORKING_FLOOR} usefulness floor)",
        "share of draws with a working safe gate",
        FIGURES[1],
    )

    # Figure 3 -- what each boundary delivers, and what a perfect one would.
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
    for column, direction in enumerate(DIRECTIONS):
        for row, (field, ylabel) in enumerate(
            (("repair_recall", "exact-repair recall"), ("review_rate", "human review rate"))
        ):
            ax = axes[row, column]
            for strategy in STRATEGIES:
                values = [
                    curves[strategy][direction][str(b)][key][field]["median"] for b in BUDGETS
                ]
                line(ax, strategy, values, label=row == 0 and column == 0)
            if field == "repair_recall":
                oracle = [
                    curves[S2][direction][str(b)][key]["oracle_repair_recall"]["median"]
                    for b in BUDGETS
                ]
                ax.plot(
                    positions,
                    [np.nan if v is None else v for v in oracle],
                    color=INK_SECONDARY,
                    linestyle=":",
                    linewidth=1.2,
                    label="oracle boundary (analysis only)" if column == 0 else None,
                )
                ax.set_ylim(-0.01, 0.3)
            else:
                ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(positions, [str(b) for b in BUDGETS])
            ax.set_ylabel(ylabel if column == 0 else "")
            if row == 0:
                ax.set_title(
                    f"{DIRECTION_SHORT[direction]} (target: {TARGET_OF[direction]})", fontsize=8.5
                )
            else:
                ax.set_xlabel("labelled target candidates bought (N)")
    axes[0, 0].legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.suptitle(
        f"What each boundary delivers at harm <= {PRIMARY_EPSILON}, medians over draws",
        fontsize=9,
    )
    save(fig, FIGURES[2], [STRATEGY_CURVES], "median recall and review rate by strategy and budget")

    # Figure 4 -- what the calibration evidence claimed against what the test block showed.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        for index, strategy in enumerate(STRATEGIES):
            cell = curves[strategy][direction][str(SATURATION_BUDGET)][key]
            realized = cell["selective_harm_rate"]["median"]
            claimed = cell["calibration_harm"]["median"]
            if realized is None or claimed is None:
                ax.text(index, 0.02, "abstains", ha="center", fontsize=6, color=INK_SECONDARY)
                continue
            ax.plot([index, index], [claimed, realized], color=GRID, linewidth=2.0, zorder=1)
            ax.scatter(index, claimed, marker="D", s=22, color=INK_SECONDARY, zorder=2)
            ax.scatter(
                index,
                realized,
                marker=STRATEGY_MARKERS[strategy],
                s=30,
                color=STRATEGY_COLOURS[strategy],
                zorder=3,
            )
        ax.axhline(PRIMARY_EPSILON, color=INK, linewidth=0.9, linestyle="--")
        ax.set_xticks(
            np.arange(len(STRATEGIES)), [STRATEGY_SHORT[s] for s in STRATEGIES], rotation=20
        )
        ax.set_title(f"{DIRECTION_SHORT[direction]}, N = {SATURATION_BUDGET}", fontsize=8.5)
        ax.set_xlim(-0.5, len(STRATEGIES) - 0.5)
        ax.set_ylim(-0.02, 0.45)
    axes[0].set_ylabel("harmful accepted rate at the chosen cutoff")
    axes[0].scatter([], [], marker="D", color=INK_SECONDARY, label="on the calibration evidence")
    axes[0].scatter([], [], marker="o", color=INK_SECONDARY, label="on the sealed test block")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.suptitle(
        "Calibration error: the harm each boundary saw against the harm it delivered "
        "(medians over accepting draws; dashed: the target)",
        fontsize=8.5,
    )
    save(
        fig,
        FIGURES[3],
        [STRATEGY_CURVES],
        "median calibration harm against realized test harm at the chosen cutoff",
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
    "CELL_SCORES",
    "SCORE_REGISTRY",
    "REPRODUCTION",
    "POLICY_OUTCOMES",
    "CALIBRATION_REGISTRY",
    "STRATEGY_CURVES",
    "FRONTIER",
    "CONTROL_RESULTS",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "DECISION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("splits", run_splits),
        ("score", run_score),
        ("calibrate", run_calibrate),
        ("curves", run_curves),
        ("frontier", run_frontier),
        ("controls", run_controls),
        ("stats", run_stats),
        ("negative", run_negative),
        ("decide", run_decide),
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
                "folds": FOLD_SEED,
                "bootstrap": BOOTSTRAP_SEED,
                "control": CONTROL_SEED,
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
    "1. Motivation": (RESEARCH_FREEZE, FROZEN_CONFIGURATION),
    "2. Research Questions and Prior Findings": (DESIGN_RECORD, FROZEN_CONFIGURATION),
    "3. The Held-Fixed Ranking and the Sealed Test Block": (
        REPRODUCTION,
        SPLIT_REGISTRY,
        SCORE_REGISTRY,
        DESIGN_RECORD,
    ),
    "4. Five Boundary Strategies": (DESIGN_RECORD, FALSIFICATION),
    "5. Reproduction Gates": (REPRODUCTION, FALSIFICATION, SPLIT_REGISTRY),
    "6. Safety: How Often Each Boundary Violates the Target": (STRATEGY_CURVES, DESIGN_RECORD),
    "7. Usefulness: Recall, Coverage and Review": (STRATEGY_CURVES, DESIGN_RECORD),
    "8. Calibration Error": (STRATEGY_CURVES, DESIGN_RECORD),
    "9. Worst-Case Environments": (STRATEGY_CURVES, DESIGN_RECORD),
    "10. The Deployable Frontier": (FRONTIER, STRATEGY_CURVES, DESIGN_RECORD),
    "11. Is the Boundary the Bottleneck?": (FRONTIER, STRATEGY_CURVES, DESIGN_RECORD),
    "12. The Joint Ranker": (STRATEGY_CURVES, FRONTIER, DESIGN_RECORD),
    "13. Statistical Tests": (STATISTICAL_TESTS, DESIGN_RECORD),
    "14. Controls and Falsification": (CONTROL_RESULTS, FALSIFICATION, DESIGN_RECORD),
    "15. Limitations": (DESIGN_RECORD, STRATEGY_CURVES, SCORE_REGISTRY, FRONTIER),
    "16. Research Decision": (
        DECISION,
        FRONTIER,
        STATISTICAL_TESTS,
        FALSIFICATION,
        DESIGN_RECORD,
    ),
    "17. Next Research Direction": (DECISION, FRONTIER, STRATEGY_CURVES),
}


def run_record() -> int:
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
            "changes_a_ranker": False,
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
        "score": run_score,
        "calibrate": run_calibrate,
        "curves": run_curves,
        "frontier": run_frontier,
        "controls": run_controls,
        "stats": run_stats,
        "negative": run_negative,
        "decide": run_decide,
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
