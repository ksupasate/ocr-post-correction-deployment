#!/usr/bin/env python3
"""SGV-TH2: how many annotated pages does a safe and useful OCR-correction boundary cost?

SGV-TH1 closed with outcome D on its safe branch. Every within-budget boundary it tested either
exceeded the harm target or abstained, and the policy that corrected its bound for page
clustering never violated and never automated: harm clusters by page, with a design effect of four
to five, so candidate labels overstate how much a budget knows. This stage prices labels in pages.

**1. A page is the unit that is bought.** A budget of N pages buys every target-generator candidate
on N whole pages of the adaptation pool. The ranking and the boundary are both fitted from those
pages, so N is the whole annotation cost of deploying.

**2. The measurable range is small, and that is stated, not hidden.** Every candidate these two
generators produced lies on 49 pages; each direction's adaptation pool holds 31 or 30, and the
sealed test block 18 or 16 more. Budgets of 10 and 25 pages and the full pool are measured. Budgets
of 50, 100 and 250 pages cannot be, and are reported as a PROJECTION: pool pages resampled to those
sizes, which says how many pages a page-level boundary would need if further pages looked like
these, and nothing about a deployment that was never run.

**3. The test pages are isolated and the page sampling is frozen first.** The sealed test block is
RK2's and TH1's, never bought, never used as a quantile reference, never resampled. Every page
selection of every arm is written to a registry, with its digest, before any ranker is fitted.

    --reconstruct   TH1, RK2 and the certification stages re-read from their own artifacts
    --freeze        upstream hashes, including the risk-bound code this stage calls
    --preregister   arms, page budgets, bounds, projection, criteria, outcome rules
    --splits        the sealed test pages, verified against TH1's
    --sample        every page selection of every arm, frozen before any fit
    --score         rankers fitted on the bought pages; test, calibration and out-of-fold scores
    --calibrate     every arm x direction x budget x draw x risk target, on the sealed test pages
    --curves        violation, usefulness, recall, coverage, worst-page harm, calibration error
    --frontier      the measured deployable frontier
    --project       PROJECTION to 50-1000 pages by resampling pool pages
    --controls      the random-ranking control
    --stats         paired exact sign tests over draws, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule and the model-versus-calibration reading
    --diagnose      POST HOC: out-of-fold purity and pool-budget realizations (no criterion)
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen upstream inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / DEPLOYMENT METHODOLOGY ONLY. No candidate is generated and no OCR output is
regenerated. No bound is a certificate unless it is labelled one, the projection is never a
measurement, nothing is production-ready, and the confirmatory reserve stays LOCKED.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_th1_threshold_calibration as th1
from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.risk.cluster_bounds import cluster_ratio_bound
from ocr_risk.risk.prefix_control import prefix_thresholds

rk2 = th1.rk2
rk1 = th1.rk1
ds1 = th1.ds1
rl2 = th1.rl2
rl1 = th1.rl1
hy1 = th1.hy1
gen1 = th1.gen1
xr1 = th1.xr1
s15 = th1.s15

REPO = th1.REPO
OUT = REPO / "results/generated/sgv_th2_page_budget_frontier"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"

SPLIT_REGISTRY = OUT / "split_registry.json"
PAGE_SELECTIONS = OUT / "page_selection_registry.json"
CELL_SCORES = OUT / "cell_scores.parquet"
SCORE_REGISTRY = OUT / "score_registry.json"

POLICY_OUTCOMES = OUT / "policy_outcomes.parquet"
CALIBRATION_REGISTRY = OUT / "calibration_registry.json"
ARM_CURVES = OUT / "arm_curves.json"
FRONTIER = OUT / "page_budget_frontier.json"
PROJECTION = OUT / "page_projection.json"
PURITY_DIAGNOSTIC = OUT / "purity_diagnostic.json"
CONTROL_RESULTS = OUT / "control_results.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_th2/page_budget_frontier.md"

SCHEMA_VERSION = 1
STAGE = "sgv_th2_page_budget_frontier"
HYPOTHESIS = "SGV-TH2-D1"
STAGE_KIND = "DEVELOPMENT / DEPLOYMENT METHODOLOGY"

PhaseError = th1.PhaseError
_relative = th1._relative
_git = th1._git
_write_json_once = th1._write_json_once
_write_parquet_once = th1._write_parquet_once
cc_read_json = th1.cc_read_json
_require = th1._require
_forbid = th1._forbid
auroc = th1.auroc
_summary = th1._summary
_num = th1._num
_share = th1._share

BOOTSTRAP_RESAMPLES = th1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = th1.BOOTSTRAP_SEED
ALPHA = th1.ALPHA
FIT_SEED = th1.FIT_SEED

# ------------------------------------------------------------------ the frozen design

DIRECTIONS = th1.DIRECTIONS
D_G2Q = th1.D_G2Q
D_Q2G = th1.D_Q2G
TARGET_OF = th1.TARGET_OF
DIRECTION_SHORT = th1.DIRECTION_SHORT

# Page budgets. "all" is the whole adaptation pool of the direction, 31 or 30 pages.
POOL = "all"
MEASURED = ("10", "25", POOL)
DECISION_BUDGET = "10"
DRAWS = th1.DRAWS
PAGE_SEED = "sgv-th2-page-selection-v1:2026-09-22"
FOLD_COUNT = 3

# The projection: pool pages resampled with replacement to sizes the pool does not contain. The
# brief's 50, 100 and 250, and two larger sizes declared to locate the frontier if it lies beyond.
PROJECTED = (50, 100, 250, 500, 1000)
PROJECTION_RESAMPLES = 20
PROJECTION_SEED = "sgv-th2-projection-v1:2026-09-22"
PROJECTION_LADDER = tuple(round(0.005 * k, 3) for k in range(1, 101))

CL = "cl_candidate_level"
PL = "pl_page_level"
ST = "st_stratified_pages"
DV = "dv_diversity_pages"
CF = "cf_cross_fitted_pages"
SP = "sp_spread_labels"
ARMS = (CL, PL, ST, DV, CF, SP)
PRIMARY_ARM = CF
SINGLE_RANKER_ARMS = (CL, PL, ST, DV)
ENSEMBLE_ARMS = (CF, SP)
PAGE_BOUND_ARMS = (PL, ST, DV, CF, SP)
SELECTION_OF = {CL: "random", PL: "random", ST: "stratified", DV: "diversity", CF: "random"}
ARM_LABEL = {
    CL: (
        "candidate-level calibration: random pages; within them a third of the sites, by RK2's "
        "site hash, are held back; the cutoff is the loosest whose candidate-level "
        "Clopper-Pearson bound meets the target"
    ),
    PL: (
        "page-level calibration: random pages; every third bought page is held back whole; the "
        "bound is taken on counts divided by the page design effect of the held-back pages"
    ),
    ST: (
        "stratified page sampling: pages bought round-robin over the test-relevant environments, "
        "then the page-level calibration of PL"
    ),
    DV: (
        "diversity-aware page selection: pages bought by greedy farthest-point selection on "
        "label-free page feature means, then the page-level calibration of PL"
    ),
    CF: (
        "cross-fitted page calibration: random pages in three page-level folds; each ranker is "
        "fitted on two folds and scores the third; every bought page is calibration evidence; "
        "a candidate is accepted only when all three rankers agree; page-corrected bound"
    ),
    SP: (
        "DIAGNOSTIC -- label-matched spread: the same number of candidate labels as the random "
        "purchase, scattered over every pool page, then CF's procedure. It holds the label count "
        "fixed and varies how many pages the labels come from"
    ),
}

EPSILONS = th1.EPSILONS
PRIMARY_EPSILON = th1.PRIMARY_EPSILON
HARM_TOLERANCE = th1.HARM_TOLERANCE
DELTA = th1.DELTA

VIOLATION_CEILING = th1.VIOLATION_CEILING
WORKING_FLOOR = th1.WORKING_FLOOR
WORST_PAGE_MIN_ACCEPTED = 5
CONTROL_SEED = 20260929

QUESTIONS = {
    "Q1": "how many annotated pages yield a safe and useful deployment boundary?",
    "Q2": "is page-level diversity the limit, rather than the number of candidate labels?",
    "Q3": "is safe automation limited by model capability or by page-level calibration data?",
}
OUTCOME_TAXONOMY = {
    "A": "a safe and useful boundary within 25 annotated pages in both directions",
    "B": "a safe and useful boundary in both directions, but only with the full page pool",
    "C": "a safe and useful boundary in one direction only, within the measured pages",
    "D": "no measured page budget yields a safe and useful boundary",
}
NEXT_STAGE = {
    "A": "NEXT: TARGETED EXTERNAL CONFIRMATION OF PAGE-BUDGETED GENERATOR ONBOARDING",
    "B": "NEXT: PAGE-EFFICIENT ACQUISITION FOR GENERATOR ONBOARDING",
    "C": "NEXT: DIRECTION-SPECIFIC DIAGNOSIS OF THE GENERATOR THAT CANNOT BE ONBOARDED",
    "D_calibration": "NEXT: EXPAND THE LABELLED PAGE POOL BEFORE ANY FURTHER BOUNDARY WORK",
    "D_model": "NEXT: TOP-OF-LIST PURITY OF THE ADAPTED RANKER BEFORE ANY BOUNDARY WORK",
}
PRIMARY_FAMILY = (
    "P1_page_level_vs_candidate_level_violation",
    "P2_cross_fitted_vs_page_level_working",
    "P3_stratified_vs_random_pages_working",
    "P4_diversity_vs_random_pages_working",
    "P5_spread_labels_vs_whole_pages_working",
)
FAMILY_PAIRS = {
    PRIMARY_FAMILY[0]: (PL, CL, "violation"),
    PRIMARY_FAMILY[1]: (CF, PL, "working_safe_gate"),
    PRIMARY_FAMILY[2]: (ST, PL, "working_safe_gate"),
    PRIMARY_FAMILY[3]: (DV, PL, "working_safe_gate"),
    PRIMARY_FAMILY[4]: (SP, CF, "working_safe_gate"),
}

UPSTREAM_EXPECTED: dict[str, Any] = {
    "outcome": "D",
    "recommended_next_stage": "NEXT: PAGE-BUDGET FRONTIER FOR A SAFE AND USEFUL BOUNDARY",
    "primary_safe_at_every_budget": True,
    "production_ready": False,
    "ready_for_external_confirmation": False,
    "confirmatory_reserve_consumed": False,
    "falsification_tests_passed": 24,
    "falsification_tests_total": 24,
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_th2-{artifact}-v{SCHEMA_VERSION}",
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


def budget_pages(label: str, pool_pages: int) -> int:
    return pool_pages if label == POOL else int(label)


def minimal_budget(flags: dict[str, bool]) -> str | None:
    """The smallest measured budget from which a property holds at every larger measured one."""
    ordered = [label for label in MEASURED if label in flags]
    for position, label in enumerate(ordered):
        if all(flags[b] for b in ordered[position:]):
            return label
    return None


def assign_outcome(n_star: dict[str, str | None]) -> str:
    """Exactly one outcome for the primary arm, in the frozen precedence D, C, A, B."""
    reached = [name for name, budget in n_star.items() if budget is not None]
    if not reached:
        return "D"
    if len(reached) < len(n_star):
        return "C"
    if all(budget in ("10", "25") for budget in n_star.values()):
        return "A"
    return "B"


def limit_reading(model_limited: bool, projected_frontier: int | None, measured: bool) -> str:
    """What limits safe automation in one direction, by the frozen rule."""
    if measured:
        return "not limited at the measured page budgets"
    if model_limited:
        return "model-limited: even an oracle boundary is not a working safe gate"
    if projected_frontier is not None and projected_frontier <= 250:
        return "calibration-limited, within the brief's page range by projection"
    if projected_frontier is not None:
        return "calibration-limited, beyond the brief's page range by projection"
    return "calibration-limited, beyond every projected page count"


def next_stage(outcome: str, model_limited_everywhere: bool) -> str:
    if outcome != "D":
        return NEXT_STAGE[outcome]
    return NEXT_STAGE["D_model"] if model_limited_everywhere else NEXT_STAGE["D_calibration"]


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    files = list(th1._upstream_files())
    files.append(REPO / "scripts/sgv_th1_threshold_calibration.py")
    for pattern in ("*.json", "*.parquet"):
        files.extend(sorted(th1.OUT.glob(pattern)))
    files.append(th1.REPORT)
    files.append(REPO / "src/ocr_risk/risk/prefix_control.py")
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    _check = xr1._check
    checks = list(th1.upstream_checks())
    decision = cc_read_json(th1.DECISION)
    for key, value in UPSTREAM_EXPECTED.items():
        checks.append(_check(f"sgv_th1.{key}", decision.get(key), value))
    checks.append(
        _check("sgv_th1.determinism", cc_read_json(th1.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_th1.report_traceability",
            cc_read_json(th1.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
            0,
        )
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
    return any(p.exists() for p in (CELL_SCORES, POLICY_OUTCOMES, ARM_CURVES, DECISION))


def run_freeze() -> int:
    started = time.monotonic()
    for path in (RESEARCH_FREEZE, FROZEN_CONFIGURATION):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("a TH2 endpoint already exists; the freeze must precede every one")
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
            "changes_an_upstream_outcome": False,
            "uses_ground_truth": False,
        },
    )
    decision = cc_read_json(th1.DECISION)
    curves = cc_read_json(th1.STRATEGY_CURVES)["by_ranker"][th1.PRIMARY_RANKER]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "th1_outcome": decision["outcome"],
            "th1_next_stage": decision["recommended_next_stage"],
            "th1_design_effect_at_500_labels": {
                name: curves[th1.S5][name]["500"][key]["design_effect"]["median"]
                for name in DIRECTIONS
            },
            "th1_adaptive_at_500_labels": {
                name: {
                    "violation_share": curves[th1.S4][name]["500"][key]["violation_share"],
                    "working_share": curves[th1.S4][name]["500"][key]["working_share"],
                }
                for name in DIRECTIONS
            },
            "ranker": "RK2's target-only LambdaMART, fitted with RK2's own function",
            "rl2_draw_seed": rl2.DRAW_SEED,
            "uses_ground_truth": False,
        },
    )
    print(f"freeze: {len(hashes)} upstream files hashed ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the pre-registration


def run_preregister() -> int:
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("a TH2 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "research_question": (
                "how many page-level annotations are required to obtain a safe and useful OCR "
                "correction deployment boundary?"
            ),
            "questions": QUESTIONS,
            "hypothesis": (
                "page-level diversity is the limiting factor, not the number of individual "
                "candidate labels"
            ),
            "unit": (
                "the page: a document_id. A page read by two OCR engines appears under two "
                "environments and is bought, split and held out as one page, as RL1's folds do"
            ),
            "budgets": {
                "measured": list(MEASURED),
                "pool": (
                    "every page of the direction's adaptation pool: 31 pages when the image "
                    "corrector is the target and 30 when the frozen generator is"
                ),
                "projected": list(PROJECTED),
                "why_projected": (
                    "every candidate of these two generators lies on 49 pages; 50, 100 and 250 "
                    "pages exceed the pool and, for 100 and 250, the whole universe. The user "
                    "chose "
                    "to measure what the data supports and to project the rest"
                ),
            },
            "decision_budget": DECISION_BUDGET,
            "decision_budget_reason": (
                "10 pages is the largest budget at which the arms' page sets differ substantially; "
                "at 25 pages every draw buys most of the pool"
            ),
            "draws": DRAWS,
            "page_selection": {
                "random": (
                    "RL2's own draw order, read at the page level: the first N pages in the order "
                    "RL2 purchases them. Nothing about it reads a label"
                ),
                "stratified": (
                    "round-robin over the environments of the pool, in a seeded order per draw; at "
                    "each turn the environment's next unbought page in RL2's order is bought. A "
                    "page bought for one environment leaves every environment's queue"
                ),
                "diversity": (
                    "greedy farthest-point selection on each page's mean R3 feature vector over "
                    "its "
                    "target candidates, standardized on the pool's pages; the first page is RL2's "
                    "first page of the draw; ties by page id. Features carry no label"
                ),
                "spread": (
                    "the number of candidates the random purchase of the same budget buys, drawn "
                    "from every pool candidate in a seeded order. It holds the label count fixed "
                    "and spreads it over every pool page"
                ),
                "frozen_before_any_fit": True,
                "seed": PAGE_SEED,
            },
            "arms": ARM_LABEL,
            "primary_arm": PRIMARY_ARM,
            "primary_arm_reason": (
                "SGV-TH1 found cross-fitting the only boundary that was both nearly calibrated and "
                "occasionally useful; its page-level form is the candidate methodology"
            ),
            "ranker": (
                "RK2's target-only LambdaMART with its frozen hyperparameters, fitted with RK2's "
                "own function on the bought pages' fitting rows"
            ),
            "splits_within_the_budget": {
                CL: "sites, by RK2's seeded site hash: one site in three is held back",
                PL: (
                    "pages: the page at every third position of the purchase order, starting at "
                    "the third, is held back whole"
                ),
                CF: (
                    "pages: a bought page's fold is its purchase position modulo 3; each ranker "
                    "holds out one fold"
                ),
                SP: "pages: a page's fold is a seeded hash of its id modulo 3",
            },
            "common_scale": (
                "for the cross-fitted arms, each ranker's score is replaced by its mid-rank "
                "quantile among its own scores on every row of the target adaptation pool. The "
                "pool's features carry no label anywhere in this stage and include no test page"
            ),
            "bound": {
                "family": "exact one-sided Clopper-Pearson, `risk.bounds.clopper_pearson_upper`",
                "delta": DELTA,
                "rule": (
                    "TH1's pointwise rule: the loosest cutoff whose bound meets the target. It is "
                    "not a simultaneous guarantee"
                ),
                "page_correction": (
                    "for every arm but CL, candidate counts are divided by the design effect of "
                    "the "
                    "harm rate across the calibration pages, `risk.cluster_bounds.design_effect`, "
                    "floored at 1; fewer than two calibration pages or an undefined design effect "
                    "means abstention"
                ),
            },
            "epsilons": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "metrics": {
                "harmful_accepted_rate": "selective harm on the sealed test pages",
                "repair_recall": "exact-repair recall over RK2's target-reachable error sites",
                "coverage": "accepted over target test decision sites",
                "review_rate": "one minus coverage",
                "worst_page_harm": (
                    "the largest harmful accepted rate over the test pages with at least "
                    f"{WORST_PAGE_MIN_ACCEPTED} accepted decisions, and the number of test pages "
                    "whose accepted edits exceed the target"
                ),
                "calibration_error": (
                    "the realized test harm at the chosen cutoff minus the calibration harm at it"
                ),
                "oracle": "analysis only: the loosest cutoff meeting the target on the test labels",
            },
            "deployable": {
                "safe": f"at most {VIOLATION_CEILING} of draws violate the primary risk target",
                "useful": f"at least {WORKING_FLOOR} of draws are a working safe gate",
                "n_star": (
                    "the smallest measured budget from which the arm is safe and useful at every "
                    "larger measured budget"
                ),
            },
            "projection": {
                "what": (
                    "the primary arm's procedure with the ranking fixed: one cross-fit over the "
                    "whole pool, with page folds by a seeded hash of the page id. For each "
                    "projected size, pool pages are drawn with replacement and each drawn copy "
                    "counts as its own page; the page-corrected bound, and the project's "
                    "finite-sample document-level certificate, choose a cutoff on those pages' "
                    "out-of-fold decisions; the cutoff is applied to the sealed test pages"
                ),
                "sizes": list(PROJECTED),
                "resamples": PROJECTION_RESAMPLES,
                "seed": PROJECTION_SEED,
                "cutoff_ladder": (
                    "100 acceptance depths from 0.005 to 0.5, placed by "
                    "`risk.prefix_control.prefix_thresholds` on the pool's unlabelled out-of-fold "
                    "scores"
                ),
                "finite_sample_certificate": (
                    "`risk.cluster_bounds.cluster_ratio_bound` with its declared `union_finite` "
                    "method; the span at each cutoff is the largest per-page accepted count over "
                    "the pool's pages, read from unlabelled scores"
                ),
                "what_it_is_not": (
                    "not a measurement. Resampling cannot add diversity: every projected page is a "
                    "copy of one of 31 or 30 real pages, and the ranking never improves. It says "
                    "how many pages a boundary would need if further pages looked like these"
                ),
                "projected_frontier": (
                    "the smallest projected size at which the page-corrected boundary is safe and "
                    "useful on the sealed test pages across the resamples"
                ),
            },
            "limit_reading": (
                "per direction: 'not limited' if the primary arm is deployable at a measured "
                "budget; 'model-limited' if the oracle boundary on the full-pool primary ranking "
                "is not a working safe gate in at least half the draws; otherwise "
                "calibration-limited, "
                "within the brief's page range if the projected frontier is at most 250 pages, "
                "beyond it otherwise"
            ),
            "outcome_rule": OUTCOME_TAXONOMY,
            "outcome_precedence": (
                "on the primary arm: D if neither direction reaches a measured N*, C if one does, "
                "A if both reach it within 25 pages, otherwise B"
            ),
            "next_stage": NEXT_STAGE,
            "next_stage_rule": (
                "D maps to D_model when both directions are model-limited, to D_calibration "
                "otherwise"
            ),
            "statistical_family": list(PRIMARY_FAMILY),
            "statistical_plan": {
                "unit": (
                    "the draw, as in SGV-TH1: a calibration procedure's behaviour across purchases "
                    "of target pages. Inference is conditional on the sealed test pages"
                ),
                "test": "exact two-sided sign test on the paired draws of both directions",
                "budget": DECISION_BUDGET,
                "epsilon": PRIMARY_EPSILON,
                "interval": "percentile, resampling draws within direction",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "multiplicity": "Holm within the frozen family",
            },
            "controls": {
                "random_ranking": (
                    "the page-level and cross-fitted arms re-run with seeded uniform scores; they "
                    f"must violate in at most {VIOLATION_CEILING} of draws at every measured budget"
                ),
                "seed": CONTROL_SEED,
            },
            "disclosure": (
                "SGV-TH1's results, and SGV15b's finding that a finite-sample document-level "
                "certificate needed on the order of eight hundred documents, were read before this "
                "record. The page-structure probe read label-blind counts only: pages per pool, "
                "per environment and candidates per page"
            ),
            "non_goals": [
                "no candidate is generated, no page is added, no OCR output is regenerated",
                "no ranker is searched or tuned",
                "no bound is called a certificate unless it is one",
                "the projection is never read by a criterion as a measurement",
                "the confirmatory reserve is never unlocked",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ pages


def load_population() -> pd.DataFrame:
    return rk2.load_population()


def pool_pages(population: pd.DataFrame, blocks: Any) -> list[str]:
    return sorted(set(population.iloc[blocks.target_adapt]["document_id"].astype(str)))


def page_order(population: pd.DataFrame, blocks: Any, draw: int) -> list[str]:
    """RL2's purchase order for a draw, read at the page level: pages by first appearance."""
    order = rl2.draw_order(population, blocks.frozen, draw)
    pages = population.iloc[order]["document_id"].astype(str).tolist()
    return list(dict.fromkeys(pages))


def stratified_pages(population: pd.DataFrame, blocks: Any, draw: int, budget: int) -> list[str]:
    """Round-robin over environments; a page bought for one leaves every queue."""
    pool = population.iloc[blocks.target_adapt]
    order = page_order(population, blocks, draw)
    position = {page: i for i, page in enumerate(order)}
    environments = sorted(
        set(pool["environment"].astype(str)),
        key=lambda env: (canonical_hash({"seed": PAGE_SEED, "draw": draw, "env": env}), env),
    )
    queues = {
        env: sorted(
            set(pool[pool["environment"] == env]["document_id"].astype(str)),
            key=lambda page: position[page],
        )
        for env in environments
    }
    chosen: list[str] = []
    taken: set[str] = set()
    while len(chosen) < budget:
        progressed = False
        for env in environments:
            while queues[env] and queues[env][0] in taken:
                queues[env].pop(0)
            if queues[env] and len(chosen) < budget:
                page = queues[env].pop(0)
                chosen.append(page)
                taken.add(page)
                progressed = True
        if not progressed:
            break
    return chosen


def page_embeddings(population: pd.DataFrame, blocks: Any, matrix: pd.DataFrame) -> pd.DataFrame:
    """Each pool page's mean R3 feature vector, standardized over the pool's pages. No label."""
    pool = population.iloc[blocks.target_adapt].reset_index(drop=True)
    design = rk1._design(pool, matrix, rk1.model_columns(rk1.PRIMARY_RANKER))
    frame = pd.DataFrame(design).assign(page=pool["document_id"].astype(str).to_numpy())
    means = frame.groupby("page", sort=True).mean()
    spread = means.std(ddof=0).replace(0.0, 1.0)
    return (means - means.mean()) / spread


def diversity_pages(embeddings: pd.DataFrame, first: str, budget: int) -> list[str]:
    """Greedy farthest-point selection from a given first page; ties go to the smaller page id."""
    names = embeddings.index.to_list()
    vectors = embeddings.to_numpy(dtype=np.float64)
    index = {name: i for i, name in enumerate(names)}
    chosen = [first]
    distance = np.linalg.norm(vectors - vectors[index[first]], axis=1)
    while len(chosen) < min(budget, len(names)):
        open_pages = [i for i in range(len(names)) if names[i] not in chosen]
        top = max(distance[i] for i in open_pages)
        best = min((i for i in open_pages if distance[i] == top), key=lambda i: names[i])
        chosen.append(names[best])
        distance = np.minimum(distance, np.linalg.norm(vectors - vectors[best], axis=1))
    return chosen


def spread_rows(population: pd.DataFrame, blocks: Any, draw: int, count: int) -> np.ndarray:
    """`count` pool candidates in a seeded order over every pool page; reads no label."""
    pool = blocks.target_adapt
    ids = population.iloc[pool]["candidate_id"].astype(str).tolist()
    ranked = sorted(
        range(len(ids)),
        key=lambda i: (canonical_hash({"seed": PAGE_SEED, "draw": draw, "id": ids[i]}), ids[i]),
    )
    return np.asarray(pool[ranked[:count]], dtype=np.int64)


def rows_on_pages(population: pd.DataFrame, blocks: Any, pages: Sequence[str]) -> np.ndarray:
    """Every pool candidate on the given pages, in page order and then by candidate id."""
    pool = population.iloc[blocks.target_adapt]
    position = {page: i for i, page in enumerate(pages)}
    member = pool[pool["document_id"].astype(str).isin(position)]
    ordered = member.assign(_page=member["document_id"].astype(str).map(position)).sort_values(
        ["_page", "candidate_id"], kind="stable"
    )
    return np.asarray(ordered.index.to_numpy(), dtype=np.int64)


def select(
    selection: str,
    population: pd.DataFrame,
    blocks: Any,
    draw: int,
    pages: int,
    embeddings: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """One arm's purchase for one draw and budget: the ordered pages and the rows they carry."""
    order = page_order(population, blocks, draw)
    if selection == "random":
        chosen = order[:pages]
    elif selection == "stratified":
        chosen = stratified_pages(population, blocks, draw, pages)
    elif selection == "diversity":
        chosen = diversity_pages(embeddings, order[0], pages)
    elif selection == "spread":
        count = int(rows_on_pages(population, blocks, order[:pages]).size)
        rows = spread_rows(population, blocks, draw, count)
        touched = list(dict.fromkeys(population.iloc[rows]["document_id"].astype(str)))
        return touched, rows
    else:
        raise PhaseError(f"unknown selection {selection}")
    return list(chosen), rows_on_pages(population, blocks, chosen)


SELECTIONS = ("random", "stratified", "diversity", "spread")


def run_splits() -> int:
    """The sealed test pages, verified against TH1's, and the isolation of the page pools."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    _forbid(SPLIT_REGISTRY)
    population = load_population()
    th1_splits = cc_read_json(th1.SPLIT_REGISTRY)["directions"]
    registry: dict[str, Any] = {}
    for name, blocks in rk2.all_blocks(population).items():
        pool = pool_pages(population, blocks)
        test = sorted(set(population.iloc[blocks.target_test]["document_id"].astype(str)))
        registry[name] = {
            "pool_pages": len(pool),
            "test_pages": len(test),
            "pool_candidates": int(blocks.target_adapt.size),
            "test_candidates": int(blocks.target_test.size),
            "pages_shared_by_pool_and_test": len(set(pool) & set(test)),
            "test_digest": rk2._digest(population, blocks.target_test),
            "equals_th1_test_block": bool(
                rk2._digest(population, blocks.target_test)
                == th1_splits[name]["target_test_digest"]
            ),
            "pool_environments": int(population.iloc[blocks.target_adapt]["environment"].nunique()),
            "test_environments": int(population.iloc[blocks.target_test]["environment"].nunique()),
            "candidates_per_pool_page": _summary(
                population.iloc[blocks.target_adapt]
                .groupby("document_id")
                .size()
                .astype(float)
                .tolist()
            ),
        }
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_analysis_envelope("split_registry"),
            "directions": registry,
            "universe_pages": int(population["document_id"].nunique()),
            "test_pages_isolated": bool(
                all(row["pages_shared_by_pool_and_test"] == 0 for row in registry.values())
            ),
            "every_test_block_is_th1s": bool(
                all(row["equals_th1_test_block"] for row in registry.values())
            ),
            "frozen_before_any_fit": not CELL_SCORES.exists(),
        },
    )
    if not all(row["equals_th1_test_block"] for row in registry.values()):
        raise PhaseError("the test block differs from TH1's; TH2 stops")
    print(
        "splits: "
        + ", ".join(
            f"{n} pool {r['pool_pages']} test {r['test_pages']}" for n, r in registry.items()
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


def run_sample() -> int:
    """Every page selection of every arm, frozen with its digest before any ranker is fitted."""
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    _forbid(PAGE_SELECTIONS)
    population = load_population()
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    rows_out: list[dict[str, Any]] = []
    for name, blocks in rk2.all_blocks(population).items():
        embeddings = page_embeddings(population, blocks, matrix)
        pool = pool_pages(population, blocks)
        for label in MEASURED:
            pages = budget_pages(label, len(pool))
            for draw in range(DRAWS):
                for selection in SELECTIONS:
                    chosen, rows = select(selection, population, blocks, draw, pages, embeddings)
                    frame = population.iloc[rows]
                    rows_out.append(
                        {
                            "direction": name,
                            "selection": selection,
                            "budget": label,
                            "draw": draw,
                            "pages": chosen,
                            "page_count": len(chosen),
                            "candidates": int(rows.size),
                            "environments": int(frame["environment"].nunique()),
                            "candidate_digest": str(
                                canonical_hash(frame["candidate_id"].astype(str).tolist())
                            ),
                            "touches_test": bool(
                                set(chosen)
                                & set(
                                    population.iloc[blocks.target_test]["document_id"].astype(str)
                                )
                            ),
                        }
                    )
    table = pd.DataFrame(rows_out)
    matched = table.pivot_table(
        index=["direction", "budget", "draw"],
        columns="selection",
        values="candidates",
        aggfunc="first",
    )
    _write_json_once(
        PAGE_SELECTIONS,
        {
            **_analysis_envelope("page_selection_registry"),
            "rows": rows_out,
            "count": len(rows_out),
            "seed": PAGE_SEED,
            "rules": cc_read_json(DESIGN_RECORD)["page_selection"],
            "spread_matches_random_label_count": bool(
                (matched["spread"] == matched["random"]).all()
            ),
            "touches_test_anywhere": bool(table["touches_test"].any()),
            "median_environments_at_the_decision_budget": {
                selection: float(
                    table[(table["selection"] == selection) & (table["budget"] == DECISION_BUDGET)][
                        "environments"
                    ].median()
                )
                for selection in SELECTIONS
            },
            "median_pages_touched_by_spread": {
                label: float(
                    table[(table["selection"] == "spread") & (table["budget"] == label)][
                        "page_count"
                    ].median()
                )
                for label in MEASURED
            },
            "median_candidates_bought": {
                label: {
                    name: float(
                        table[
                            (table["selection"] == "random")
                            & (table["budget"] == label)
                            & (table["direction"] == name)
                        ]["candidates"].median()
                    )
                    for name in DIRECTIONS
                }
                for label in MEASURED
            },
            "frozen_before_any_fit": not CELL_SCORES.exists(),
            "selected_without_labels": True,
        },
    )
    if table["touches_test"].any():
        raise PhaseError("a page selection reached the sealed test pages; TH2 stops")
    print(f"sample: {len(rows_out)} page selections frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ scoring


def page_folds(arm: str, pages: Sequence[str]) -> dict[str, int]:
    """Which fold or held-back part each bought page belongs to."""
    if arm in (PL, ST, DV):
        return {
            page: (1 if position % FOLD_COUNT == 2 else 0) for position, page in enumerate(pages)
        }
    if arm == CF:
        return {page: position % FOLD_COUNT for position, page in enumerate(pages)}
    if arm == SP:
        return {
            page: int(canonical_hash({"seed": PAGE_SEED, "page": page})[:8], 16) % FOLD_COUNT
            for page in pages
        }
    raise PhaseError(f"{arm} has no page folds")


def run_score() -> int:
    """Rankers fitted on the bought pages; test, calibration and out-of-fold scores."""
    started = time.monotonic()
    _require(PAGE_SELECTIONS, "sample")
    for path in (CELL_SCORES, SCORE_REGISTRY):
        _forbid(path)
    population = load_population()
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    design = rk1._design(population, matrix, rk1.model_columns(rk1.PRIMARY_RANKER))
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = population["document_id"].astype(str).to_numpy()
    ids = population["candidate_id"].astype(str).to_numpy()
    documents = population["document_id"].astype(str).to_numpy()
    calibration_site = rk2.calibration_mask(population["site_group"].astype(str).tolist())
    frozen = {
        (row["direction"], row["selection"], row["budget"], row["draw"]): row
        for row in cc_read_json(PAGE_SELECTIONS)["rows"]
    }
    frames: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []

    def fit(rows: np.ndarray) -> Any:
        return rk2.fit_arm(rk2.A1, design, grades, groups, rows[:0], rows, None)

    for name, blocks in rk2.all_blocks(population).items():
        began = time.monotonic()
        embeddings = page_embeddings(population, blocks, pd.read_parquet(rk1.FEATURE_MATRIX))
        pool = blocks.target_adapt
        test = blocks.target_test
        for label in MEASURED:
            pages_n = budget_pages(label, len(pool_pages(population, blocks)))
            for draw in range(DRAWS):
                for arm in ARMS:
                    selection = "spread" if arm == SP else SELECTION_OF[arm]
                    pages, rows = select(selection, population, blocks, draw, pages_n, embeddings)
                    record = frozen[(name, selection, label, draw)]
                    digest = str(canonical_hash(ids[rows].tolist()))
                    if digest != record["candidate_digest"] or pages != record["pages"]:
                        raise PhaseError(
                            f"{name} {arm} {label} {draw}: selection is not the frozen one"
                        )
                    base = {"arm": arm, "direction": name, "budget": label, "draw": draw}
                    if arm in SINGLE_RANKER_ARMS:
                        if arm == CL:
                            held = calibration_site[rows]
                        else:
                            folds = page_folds(arm, pages)
                            held = np.asarray([folds[d] == 1 for d in documents[rows]], dtype=bool)
                        model = fit(rows[~held])
                        cal = rows[held]
                        raw_test = model.score(design[test])
                        frames.append(
                            pd.DataFrame(
                                {
                                    **base,
                                    "evaluation_set": "test",
                                    "candidate_id": ids[test],
                                    "fold": -1,
                                    "raw": raw_test,
                                    "q_mean": np.nan,
                                    "q_min": np.nan,
                                    "q_oof": np.nan,
                                }
                            )
                        )
                        frames.append(
                            pd.DataFrame(
                                {
                                    **base,
                                    "evaluation_set": "calibration",
                                    "candidate_id": ids[cal],
                                    "fold": -1,
                                    "raw": model.score(design[cal]) if cal.size else np.empty(0),
                                    "q_mean": np.nan,
                                    "q_min": np.nan,
                                    "q_oof": np.nan,
                                }
                            )
                        )
                        trainable = [not rk2._constant(raw_test)]
                        fit_rows = [int((~held).sum())]
                        calibration_pages = len(set(documents[cal].tolist()))
                    else:
                        folds = page_folds(arm, pages)
                        fold = np.asarray([folds[d] for d in documents[rows]], dtype=np.int64)
                        members = [fit(rows[fold != k]) for k in range(FOLD_COUNT)]
                        raw_test = [m.score(design[test]) for m in members]
                        reference = [m.score(design[pool]) for m in members]
                        q_test = [
                            th1.quantiles(ref, values)
                            for ref, values in zip(reference, raw_test, strict=True)
                        ]
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
                        for k, member in enumerate(members):
                            held = rows[fold == k]
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
                                        "q_oof": th1.quantiles(reference[k], raw_held)
                                        if held.size
                                        else np.empty(0),
                                    }
                                )
                            )
                        trainable = [not rk2._constant(values) for values in raw_test]
                        fit_rows = [int((fold != k).sum()) for k in range(FOLD_COUNT)]
                        calibration_pages = len(pages)
                    registry.append(
                        {
                            **base,
                            "pages": len(pages),
                            "candidates": int(rows.size),
                            "fit_rows": fit_rows,
                            "calibration_pages": calibration_pages,
                            "trainable": trainable,
                            "selection": selection,
                        }
                    )
        print(f"  {name}: {time.monotonic() - began:.0f}s", flush=True)
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(
        ["arm", "direction", "budget", "draw", "evaluation_set", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    _write_parquet_once(CELL_SCORES, table)
    _write_json_once(
        SCORE_REGISTRY,
        {
            **_analysis_envelope("score_registry"),
            "cells": registry,
            "cell_count": len(registry),
            "fits": int(sum(len(row["fit_rows"]) for row in registry)),
            "scored_rows": len(table),
            "untrainable_cells": sum(1 for row in registry if not all(row["trainable"])),
            "selections_verified_against_the_frozen_registry": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"score: {len(registry)} cells, {sum(len(r['fit_rows']) for r in registry)} fits "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the boundaries

LABELS = rk2.LABELS


def page_harm(accepted: pd.DataFrame, epsilon: float) -> dict[str, Any]:
    """Harm by test page among the accepted edits."""
    if accepted.empty:
        return {"pages_accepting": 0, "pages_exceeding": 0, "worst_page_harm": None}
    per = accepted.groupby("document_id")["is_harmful"].agg(["sum", "size"])
    rate = per["sum"] / per["size"]
    eligible = rate[per["size"] >= WORST_PAGE_MIN_ACCEPTED]
    return {
        "pages_accepting": len(per),
        "pages_exceeding": int((rate > epsilon + HARM_TOLERANCE).sum()),
        "worst_page_harm": float(eligible.max()) if len(eligible) else None,
    }


def boundary(
    arm: str, calibration: pd.DataFrame, epsilon: float, usable: bool
) -> tuple[float | None, float | None, float | None]:
    """The arm's cutoff, the bound it claimed and the page design effect it used, if any."""
    if not usable:
        return None, None, None
    if arm == CL:
        cutoff, bound = th1.bound_threshold(calibration, epsilon, DELTA, 1.0)
        return cutoff, bound, None
    deff = th1.page_design_effect(calibration)
    if deff is None:
        return None, None, None
    cutoff, bound = th1.bound_threshold(calibration, epsilon, DELTA, deff)
    return cutoff, bound, deff


def evaluate_arm(
    arm: str,
    epsilon: float,
    frames: dict[str, pd.DataFrame],
    usable: bool,
    keys: set[str],
    mapping: dict[str, set[str]],
) -> dict[str, Any]:
    """One arm's boundary on one cell: chosen without test labels, then read on the test pages."""
    test = frames["test"]
    calibration = frames["calibration"]
    cutoff, bound, deff = boundary(arm, calibration, epsilon, usable)
    accepted = test[test["safety"] >= cutoff] if cutoff is not None else test.head(0)
    row = rk2.policy_outcome(test, accepted, keys, mapping, epsilon)
    believed = (
        calibration[calibration["safety"] >= cutoff] if cutoff is not None else calibration.head(0)
    )
    calibration_harm = _share(float(believed["is_harmful"].sum()), len(believed))
    realized = row["selective_harm_rate"]
    oracle = rk2.oracle_accepted(test, epsilon) if usable else test.head(0)
    oracle_row = rk2.policy_outcome(test, oracle, keys, mapping, epsilon)
    return {
        "arm": arm,
        "epsilon": epsilon,
        "trainable": bool(usable),
        "cutoff": cutoff,
        "abstains": cutoff is None,
        "calibration_decisions": len(calibration),
        "calibration_pages": int(calibration["document_id"].nunique()) if len(calibration) else 0,
        "calibration_accepted": len(believed),
        "calibration_harm": calibration_harm,
        "claimed_bound": bound,
        "design_effect": deff,
        "accepted": row["accepted"],
        "sites": row["sites"],
        "coverage": row["coverage"],
        "review_rate": row["review_rate"],
        "selective_harm_rate": realized,
        "repair_recall": row["repair_recall"],
        "repaired_error_sites": row["repaired_error_sites"],
        "holds": row["holds_its_target"],
        "violation": bool(row["accepted"] > 0 and not row["holds_its_target"]),
        "working_safe_gate": row["working_safe_gate"],
        "harm_estimation_error": (
            None if realized is None or calibration_harm is None else realized - calibration_harm
        ),
        **page_harm(accepted, epsilon),
        "oracle_repair_recall": oracle_row["repair_recall"],
        "oracle_working": oracle_row["working_safe_gate"],
    }


def cell_frames(
    arm: str,
    cell: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    replace: Callable[[str, int], np.ndarray] | None,
) -> dict[str, pd.DataFrame]:
    """The labelled decision frames an arm reads; `replace` swaps every score for a random one."""

    def join(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.join(labels[list(LABELS)], on="candidate_id", how="inner").reset_index(
            drop=True
        )

    def scores(values: pd.Series, tag: str) -> np.ndarray:
        array = values.to_numpy(dtype=np.float64)
        return replace(tag, array.size) if replace is not None else array

    test = join(cell["test"])
    if arm in SINGLE_RANKER_ARMS:
        cal = join(cell.get("calibration", cell["test"].head(0)))
        raw = scores(test["raw"], "t")
        return {
            "test": rk2.decisions(test.assign(rank_score=raw, safety=raw)),
            "calibration": rk2.decisions(
                cal.assign(rank_score=scores(cal["raw"], "c"), safety=scores(cal["raw"], "c"))
            ),
        }
    oof = join(cell.get("oof", cell["test"].head(0)))
    mean_q = scores(test["q_mean"], "m")
    min_q = scores(test["q_min"], "m") if replace is not None else test["q_min"].to_numpy()
    q_oof = scores(oof["q_oof"], "o")
    return {
        "test": rk2.decisions(test.assign(rank_score=mean_q, safety=min_q)),
        "calibration": rk2.decisions(oof.assign(rank_score=q_oof, safety=q_oof)),
    }


def _cells(scores: pd.DataFrame) -> dict[tuple[str, str, str, int], dict[str, pd.DataFrame]]:
    out: dict[tuple[str, str, str, int], dict[str, pd.DataFrame]] = {}
    for (arm, name, budget, draw, kind), group in scores.groupby(
        ["arm", "direction", "budget", "draw", "evaluation_set"], sort=True
    ):
        out.setdefault((str(arm), str(name), str(budget), int(draw)), {})[str(kind)] = group
    return out


def run_arms(
    arms: Sequence[str],
    replace_for: Callable[[tuple[str, str, str, int]], Callable[[str, int], np.ndarray] | None],
    force_trainable: bool = False,
) -> list[dict[str, Any]]:
    population = load_population()
    labels = population.set_index("candidate_id")
    keys, mapping = th1._test_context(population)
    registry = {
        (row["arm"], row["direction"], row["budget"], row["draw"]): row
        for row in cc_read_json(SCORE_REGISTRY)["cells"]
    }
    out: list[dict[str, Any]] = []
    for key, cell in _cells(pd.read_parquet(CELL_SCORES)).items():
        arm, name, budget, draw = key
        if arm not in arms:
            continue
        usable = force_trainable or bool(all(registry[key]["trainable"]))
        frames = cell_frames(arm, cell, labels, replace_for(key))
        for epsilon in EPSILONS:
            out.append(
                {
                    "direction": name,
                    "budget": budget,
                    "draw": draw,
                    "pages": registry[key]["pages"],
                    "candidates": registry[key]["candidates"],
                    **evaluate_arm(arm, epsilon, frames, usable, keys[name], mapping),
                }
            )
    return out


def run_calibrate() -> int:
    started = time.monotonic()
    _require(SCORE_REGISTRY, "score")
    for path in (POLICY_OUTCOMES, CALIBRATION_REGISTRY):
        _forbid(path)
    table = pd.DataFrame(run_arms(ARMS, lambda _key: None)).sort_values(
        ["arm", "direction", "budget", "draw", "epsilon"], kind="stable"
    )
    _write_parquet_once(POLICY_OUTCOMES, table.reset_index(drop=True))
    _write_json_once(
        CALIBRATION_REGISTRY,
        {
            **_analysis_envelope("calibration_registry"),
            "arms": ARM_LABEL,
            "rows": len(table),
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
    accepting = frame[frame["accepted"] > 0]
    return {
        "draws": len(frame),
        "violation_share": float(frame["violation"].mean()),
        "working_share": float(frame["working_safe_gate"].mean()),
        "abstain_share": float(frame["abstains"].mean()),
        "untrainable_draws": int((~frame["trainable"]).sum()),
        "pages": _summary(_values(frame, "pages")),
        "candidates": _summary(_values(frame, "candidates")),
        "calibration_pages": _summary(_values(frame, "calibration_pages")),
        "repair_recall": _summary(_values(frame, "repair_recall")),
        "coverage": _summary(_values(frame, "coverage")),
        "review_rate": _summary(_values(frame, "review_rate")),
        "selective_harm_rate": _summary(_values(accepting, "selective_harm_rate")),
        "calibration_harm": _summary(_values(accepting, "calibration_harm")),
        "harm_estimation_error": _summary(_values(frame, "harm_estimation_error")),
        "design_effect": _summary(_values(frame, "design_effect")),
        "worst_page_harm": _summary(_values(accepting, "worst_page_harm")),
        "draws_with_a_page_exceeding": int((frame["pages_exceeding"] > 0).sum()),
        "oracle_repair_recall": _summary(_values(frame, "oracle_repair_recall")),
        "oracle_working_share": float(frame["oracle_working"].mean()),
    }


def run_curves() -> int:
    started = time.monotonic()
    _require(CALIBRATION_REGISTRY, "calibrate")
    _forbid(ARM_CURVES)
    table = pd.read_parquet(POLICY_OUTCOMES)
    curves: dict[str, Any] = {}
    for (arm, name, budget, epsilon), frame in table.groupby(
        ["arm", "direction", "budget", "epsilon"], sort=True
    ):
        curves.setdefault(str(arm), {}).setdefault(str(name), {}).setdefault(str(budget), {})[
            rl1.epsilon_key(float(epsilon))
        ] = aggregate(frame)
    _write_json_once(
        ARM_CURVES,
        {
            **_analysis_envelope("arm_curves"),
            "by_arm": curves,
            "block": "sealed target test pages",
            "violation_ceiling": VIOLATION_CEILING,
            "working_floor": WORKING_FLOOR,
            "worst_page_min_accepted": WORST_PAGE_MIN_ACCEPTED,
            "metrics": cc_read_json(DESIGN_RECORD)["metrics"],
        },
    )
    print(f"curves: {len(table)} outcomes aggregated ({time.monotonic() - started:.0f}s)")
    return 0


def run_frontier() -> int:
    started = time.monotonic()
    _require(ARM_CURVES, "curves")
    _forbid(FRONTIER)
    curves = cc_read_json(ARM_CURVES)["by_arm"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    frontier: dict[str, Any] = {}
    for arm in ARMS:
        for name in DIRECTIONS:
            flags = {b: th1.deployable(curves[arm][name][b][key]) for b in MEASURED}
            frontier.setdefault(arm, {})[name] = {
                "n_star": minimal_budget({b: v["deployable"] for b, v in flags.items()}),
                "by_budget": flags,
                "safe_at_every_budget": all(v["safe"] for v in flags.values()),
            }
    oracle = {
        name: curves[PRIMARY_ARM][name][POOL][key]["oracle_working_share"] for name in DIRECTIONS
    }
    _write_json_once(
        FRONTIER,
        {
            **_analysis_envelope("page_budget_frontier"),
            "frontier": frontier,
            "primary_arm": PRIMARY_ARM,
            "oracle_working_share_at_the_pool": oracle,
            "model_limited": {name: bool(v < WORKING_FLOOR) for name, v in oracle.items()},
            "definition": cc_read_json(DESIGN_RECORD)["deployable"],
            "violation_ceiling": VIOLATION_CEILING,
            "working_floor": WORKING_FLOOR,
        },
    )
    print(
        "frontier: primary N* "
        + str({n: frontier[PRIMARY_ARM][n]["n_star"] for n in DIRECTIONS})
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the projection


def _projection_fit(
    population: pd.DataFrame,
    blocks: Any,
    design: np.ndarray,
    grades: np.ndarray,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One cross-fit over the whole pool: out-of-fold decisions and the test decisions."""
    labels = population.set_index("candidate_id")
    pool = blocks.target_adapt
    test = blocks.target_test
    documents = population["document_id"].astype(str).to_numpy()
    ids = population["candidate_id"].astype(str).to_numpy()
    fold = np.asarray(
        [
            int(canonical_hash({"seed": PROJECTION_SEED, "page": d})[:8], 16) % FOLD_COUNT
            for d in documents[pool]
        ],
        dtype=np.int64,
    )
    members = [
        rk2.fit_arm(rk2.A1, design, grades, groups, pool[:0], pool[fold != k], None)
        for k in range(FOLD_COUNT)
    ]
    reference = [m.score(design[pool]) for m in members]
    q_test = [
        th1.quantiles(r, m.score(design[test])) for m, r in zip(members, reference, strict=True)
    ]
    test_frame = pd.DataFrame(
        {
            "candidate_id": ids[test],
            "q_mean": np.mean(q_test, axis=0),
            "q_min": np.min(q_test, axis=0),
        }
    ).join(labels[list(LABELS)], on="candidate_id")
    oof_parts = []
    for k, member in enumerate(members):
        held = pool[fold == k]
        oof_parts.append(
            pd.DataFrame(
                {
                    "candidate_id": ids[held],
                    "q": th1.quantiles(reference[k], member.score(design[held])),
                }
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True).join(labels[list(LABELS)], on="candidate_id")
    return (
        rk2.decisions(oof.assign(rank_score=oof["q"], safety=oof["q"])),
        rk2.decisions(
            test_frame.assign(rank_score=test_frame["q_mean"], safety=test_frame["q_min"])
        ),
    )


def _certified_cutoff(
    calibration: pd.DataFrame, cutoffs: np.ndarray, spans: np.ndarray, epsilon: float, method: str
) -> float | None:
    """The loosest ladder cutoff the named bound certifies on these (resampled) pages."""
    pages = calibration["document_id"].to_numpy()
    order = np.argsort(pages, kind="stable")
    page_ids, inverse = np.unique(pages[order], return_inverse=True)
    safety = calibration["safety"].to_numpy(dtype=np.float64)[order]
    harmful = calibration["is_harmful"].to_numpy(dtype=bool)[order]
    deff = th1.page_design_effect(calibration) if method == "page_corrected" else None
    if method == "page_corrected" and deff is None:
        return None
    chosen: float | None = None
    for cutoff, span in zip(cutoffs, spans, strict=True):
        accept = safety >= cutoff
        accepted = np.bincount(inverse, weights=accept, minlength=page_ids.size)
        harm = np.bincount(inverse, weights=accept & harmful, minlength=page_ids.size)
        if method == "page_corrected":
            assert deff is not None
            upper = th1._upper(float(harm.sum()) / deff, float(accepted.sum()) / deff, DELTA)
            certified = accepted.sum() > 0 and upper <= epsilon + HARM_TOLERANCE
        else:
            certified = bool(
                accepted.sum() > 0
                and cluster_ratio_bound(
                    harm, accepted, epsilon, delta=DELTA, span=float(span)
                ).certifies
            )
        if certified and (chosen is None or cutoff < chosen):
            chosen = float(cutoff)
    return chosen


PROJECTION_METHODS = ("page_corrected", "finite_sample_certificate")


def run_project() -> int:
    """PROJECTION: pool pages resampled to sizes the pool does not contain. Never a measurement."""
    started = time.monotonic()
    _require(FRONTIER, "frontier")
    _forbid(PROJECTION)
    population = load_population()
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    design = rk1._design(population, matrix, rk1.model_columns(rk1.PRIMARY_RANKER))
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = population["document_id"].astype(str).to_numpy()
    keys, mapping = th1._test_context(population)
    out: dict[str, Any] = {}
    for name, blocks in rk2.all_blocks(population).items():
        began = time.monotonic()
        oof, test = _projection_fit(population, blocks, design, grades, groups)
        pages = sorted(set(oof["document_id"].astype(str)))
        by_page = dict(iter(oof.groupby("document_id", sort=True)))
        cutoffs = prefix_thresholds(oof["safety"].to_numpy(dtype=np.float64), PROJECTION_LADDER)
        spans = np.asarray(
            [
                float(oof[oof["safety"] >= c].groupby("document_id").size().max())
                if (oof["safety"] >= c).any()
                else 0.0
                for c in cutoffs
            ]
        )
        sizes = (len(pages), *PROJECTED)
        out[name] = {"pool_pages": len(pages), "by_size": {}}
        for size in sizes:
            rows: dict[str, list[dict[str, Any]]] = {m: [] for m in PROJECTION_METHODS}
            for resample in range(PROJECTION_RESAMPLES):
                seed = int(
                    canonical_hash(
                        {"seed": PROJECTION_SEED, "direction": name, "size": size, "r": resample}
                    )[:8],
                    16,
                )
                drawn = np.random.default_rng(seed).choice(len(pages), size=size, replace=True)
                calibration = pd.concat(
                    [
                        by_page[pages[p]].assign(document_id=f"{pages[p]}#{i:05d}")
                        for i, p in enumerate(drawn)
                    ],
                    ignore_index=True,
                )
                for method in PROJECTION_METHODS:
                    cutoff = _certified_cutoff(calibration, cutoffs, spans, PRIMARY_EPSILON, method)
                    accepted = (
                        test[test["safety"] >= cutoff] if cutoff is not None else test.head(0)
                    )
                    row = rk2.policy_outcome(test, accepted, keys[name], mapping, PRIMARY_EPSILON)
                    rows[method].append(
                        {
                            "certifies": cutoff is not None,
                            "violation": bool(row["accepted"] > 0 and not row["holds_its_target"]),
                            "working": bool(row["working_safe_gate"]),
                            "repair_recall": row["repair_recall"],
                            "coverage": row["coverage"],
                        }
                    )
            out[name]["by_size"][str(size)] = {
                method: {
                    "resamples": len(values),
                    "certifying_share": float(np.mean([v["certifies"] for v in values])),
                    "violation_share": float(np.mean([v["violation"] for v in values])),
                    "working_share": float(np.mean([v["working"] for v in values])),
                    "repair_recall": _summary([v["repair_recall"] for v in values]),
                    "coverage": _summary([v["coverage"] for v in values]),
                }
                for method, values in rows.items()
            }
        oracle = rk2.oracle_accepted(test, PRIMARY_EPSILON)
        out[name]["oracle_repair_recall"] = rk2.policy_outcome(
            test, oracle, keys[name], mapping, PRIMARY_EPSILON
        )["repair_recall"]
        for method in PROJECTION_METHODS:
            flags = {
                size: th1.deployable(out[name]["by_size"][str(size)][method])["deployable"]
                for size in PROJECTED
            }
            frontier = None
            for position, size in enumerate(PROJECTED):
                if all(flags[s] for s in PROJECTED[position:]):
                    frontier = size
                    break
            out[name][f"projected_frontier_{method}"] = frontier
        print(f"  {name}: {time.monotonic() - began:.0f}s", flush=True)
    _write_json_once(
        PROJECTION,
        {
            **_analysis_envelope("page_projection"),
            "PROJECTION_NOT_A_MEASUREMENT": True,
            "by_direction": out,
            "what_it_is": cc_read_json(DESIGN_RECORD)["projection"],
            "sizes": list(PROJECTED),
            "resamples": PROJECTION_RESAMPLES,
            "ladder_depths": len(PROJECTION_LADDER),
            "epsilon": PRIMARY_EPSILON,
            "delta": DELTA,
        },
    )
    print(
        "project: "
        + ", ".join(
            f"{n} page-corrected frontier {out[n]['projected_frontier_page_corrected']}, "
            f"certificate {out[n]['projected_frontier_finite_sample_certificate']}"
            for n in DIRECTIONS
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ controls and statistics


def random_scores(key: tuple[str, str, str, int]) -> Callable[[str, int], np.ndarray]:
    def draw(tag: str, size: int) -> np.ndarray:
        seed = int(canonical_hash({"seed": CONTROL_SEED, "cell": list(key), "frame": tag})[:8], 16)
        return np.random.default_rng(seed).random(size)

    return draw


def run_controls() -> int:
    started = time.monotonic()
    _require(PROJECTION, "project")
    _forbid(CONTROL_RESULTS)
    table = pd.DataFrame(run_arms((PL, CF), random_scores, force_trainable=True))
    table = table[table["epsilon"] == PRIMARY_EPSILON]
    out: dict[str, Any] = {}
    for (arm, name, budget), frame in table.groupby(["arm", "direction", "budget"]):
        out.setdefault(str(arm), {}).setdefault(str(name), {})[str(budget)] = {
            "violation_share": float(frame["violation"].mean()),
            "working_share": float(frame["working_safe_gate"].mean()),
        }
    worst = {
        arm: max(c["violation_share"] for b in out[arm].values() for c in b.values()) for arm in out
    }
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "random_ranking": out,
            "worst_violation_share": worst,
            "page_bounds_protect": bool(all(v <= VIOLATION_CEILING for v in worst.values())),
            "rule": cc_read_json(DESIGN_RECORD)["controls"]["random_ranking"],
        },
    )
    print(f"controls: worst random violation {worst} ({time.monotonic() - started:.0f}s)")
    return 0


def paired(table: pd.DataFrame, left: str, right: str, metric: str) -> tuple[np.ndarray, ...]:
    frame = table[(table["budget"] == DECISION_BUDGET) & (table["epsilon"] == PRIMARY_EPSILON)]
    wide = frame.pivot_table(
        index=["direction", "draw"], columns="arm", values=metric, aggfunc="first"
    ).sort_index()
    return (
        wide[left].to_numpy(dtype=np.float64),
        wide[right].to_numpy(dtype=np.float64),
        wide.index.get_level_values("direction").to_numpy(),
    )


def comparison(
    name: str, table: pd.DataFrame, left: str, right: str, metric: str
) -> dict[str, Any]:
    a, b, direction = paired(table, left, right, metric)
    difference = a - b
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero(direction == d) for d in sorted(set(direction.tolist()))]
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for resample in range(BOOTSTRAP_RESAMPLES):
        index = np.concatenate([generator.choice(s, s.size, replace=True) for s in strata])
        draws[resample] = float(a[index].mean() - b[index].mean())
    p, positive, negative = th1.sign_test(difference)
    return {
        "comparison": name,
        "left": left,
        "right": right,
        "metric": metric,
        "effect": float(a.mean() - b.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_value": p,
        "draws_left_higher": positive,
        "draws_right_higher": negative,
        "ties": int(difference.size - positive - negative),
        "units": int(difference.size),
        "left_share": float(a.mean()),
        "right_share": float(b.mean()),
    }


def run_stats() -> int:
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    table = pd.read_parquet(POLICY_OUTCOMES)
    for column in ("violation", "working_safe_gate"):
        table[column] = table[column].astype(float)
    family = {
        name: comparison(name, table, left, right, metric)
        for name, (left, right, metric) in FAMILY_PAIRS.items()
    }
    adjusted = s15.holm(family)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "primary_family": adjusted,
            "family_size": len(PRIMARY_FAMILY),
            "budget": DECISION_BUDGET,
            "epsilon": PRIMARY_EPSILON,
            "plan": cc_read_json(DESIGN_RECORD)["statistical_plan"],
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "effect_is_a_difference_in_shares_of_draws": True,
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
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    population = load_population()
    labels = population.set_index("candidate_id")
    scores = pd.read_parquet(CELL_SCORES)
    table = pd.read_parquet(POLICY_OUTCOMES)
    splits = cc_read_json(SPLIT_REGISTRY)
    selections = cc_read_json(PAGE_SELECTIONS)
    registry = cc_read_json(SCORE_REGISTRY)
    blocks = rk2.all_blocks(population)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    columns = rk1.model_columns(rk1.PRIMARY_RANKER)
    record(
        "t01_no_generator_identity_or_label_field_is_a_design_column",
        not [c for c in columns if c.startswith(rl1.FAM_SOURCE)]
        and not set(columns) & rk1.LABEL_FIELDS,
    )
    record(
        "t02_the_test_pages_are_th1s_and_share_no_page_with_the_pool",
        bool(splits["every_test_block_is_th1s"]) and bool(splits["test_pages_isolated"]),
        {n: r["pages_shared_by_pool_and_test"] for n, r in splits["directions"].items()},
    )
    record(
        "t03_every_page_selection_was_frozen_before_any_fit",
        bool(selections["frozen_before_any_fit"])
        and bool(splits["frozen_before_any_fit"])
        and selections["issued_utc"] <= registry["issued_utc"]
        and bool(registry["selections_verified_against_the_frozen_registry"]),
        {"selections": selections["count"]},
    )
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
        "t04_no_test_row_is_bought_calibrated_on_or_used_as_a_reference",
        not any(crossing.values()) and not selections["touches_test_anywhere"],
        crossing,
    )
    # The selections read no label: rebuilding them with every label inverted changes nothing.
    flipped = population.copy()
    flipped["is_harmful"] = ~flipped["is_harmful"]
    flipped["exact"] = ~flipped["exact"]
    flipped["grade"] = 3 - flipped["grade"]
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    same = True
    for name, block in blocks.items():
        embeddings = page_embeddings(flipped, block, matrix)
        pool_n = len(pool_pages(flipped, block))
        for selection in SELECTIONS:
            for label in MEASURED:
                again, _rows = select(
                    selection, flipped, block, 0, budget_pages(label, pool_n), embeddings
                )
                frozen = next(
                    r
                    for r in selections["rows"]
                    if (r["direction"], r["selection"], r["budget"], r["draw"])
                    == (name, selection, label, 0)
                )
                same &= again == frozen["pages"]
    record("t05_page_selections_read_no_label", same)
    record(
        "t06_the_spread_arm_buys_as_many_labels_as_the_random_arm",
        bool(selections["spread_matches_random_label_count"]),
    )
    disjoint = True
    for block in blocks.values():
        order = page_order(population, block, 0)[:10]
        folds = page_folds(PL, order)
        held = {p for p, f in folds.items() if f == 1}
        disjoint &= held == {order[2], order[5], order[8]}
    record("t07_page_level_calibration_holds_back_whole_pages", disjoint)
    oof = scores[scores["evaluation_set"] == "oof"]
    documents = dict(
        zip(
            population["candidate_id"].astype(str),
            population["document_id"].astype(str),
            strict=True,
        )
    )
    wrong = 0
    for (arm, name, budget, draw), group in oof.groupby(["arm", "direction", "budget", "draw"]):
        pages = next(
            r["pages"]
            for r in selections["rows"]
            if (r["direction"], r["selection"], r["budget"], r["draw"])
            == (name, "spread" if arm == SP else "random", budget, draw)
        )
        folds = page_folds(str(arm), pages)
        wrong += int((group["candidate_id"].map(documents).map(folds) != group["fold"]).sum())
    record(
        "t08_every_out_of_fold_score_comes_from_the_ranker_that_never_saw_the_page",
        wrong == 0,
        {"rows": len(oof), "wrong": wrong},
    )
    keys, mapping = th1._test_context(population)
    cells = _cells(scores)
    info = {(r["arm"], r["direction"], r["budget"], r["draw"]): r for r in registry["cells"]}
    inverted = labels.copy()
    for block in blocks.values():
        index = population.iloc[block.target_test]["candidate_id"]
        inverted.loc[index, "is_harmful"] = ~inverted.loc[index, "is_harmful"]
        inverted.loc[index, "exact"] = ~inverted.loc[index, "exact"]
    unchanged: list[bool] = []
    for key in sorted(cells)[:: max(1, len(cells) // 60)]:
        usable = bool(all(info[key]["trainable"]))
        honest = evaluate_arm(
            key[0],
            PRIMARY_EPSILON,
            cell_frames(key[0], cells[key], labels, None),
            usable,
            keys[key[1]],
            mapping,
        )
        flip = evaluate_arm(
            key[0],
            PRIMARY_EPSILON,
            cell_frames(key[0], cells[key], inverted, None),
            usable,
            keys[key[1]],
            mapping,
        )
        unchanged.append(honest["cutoff"] == flip["cutoff"])
    record(
        "t09_inverting_every_test_label_leaves_every_cutoff_unchanged",
        all(unchanged),
        {"checked": len(unchanged)},
    )
    bounded = table[table["cutoff"].notna()]
    record(
        "t10_no_arm_deploys_a_cutoff_whose_bound_exceeds_the_target",
        bool((bounded["claimed_bound"] <= bounded["epsilon"] + HARM_TOLERANCE).all()),
        {"rows": len(bounded)},
    )
    page_level = table[table["arm"].isin(PAGE_BOUND_ARMS) & table["cutoff"].notna()]
    record(
        "t11_every_page_level_cutoff_used_a_design_effect_of_at_least_one",
        bool((page_level["design_effect"] >= 1.0).all()),
        {"rows": len(page_level)},
    )
    record(
        "t12_an_untrainable_ranker_never_deploys",
        bool((table[~table["trainable"]]["accepted"] == 0).all()),
    )
    record(
        "t13_the_oracle_never_reaches_a_deployed_cutoff",
        "oracle_accepted" not in _names_in("boundary"),
        sorted(_names_in("boundary") & {"oracle_accepted"}),
    )
    record(
        "t14_no_risk_target_reaches_a_fitted_ranker",
        not (_names_in("page_folds") | _names_in("select")) & {"EPSILONS", "PRIMARY_EPSILON"},
    )
    projection = cc_read_json(PROJECTION)
    record(
        "t15_the_projection_is_labelled_and_resamples_only_pool_pages",
        bool(projection["PROJECTION_NOT_A_MEASUREMENT"])
        and "target_test" not in _names_in("_certified_cutoff"),
        {"sizes": projection["sizes"]},
    )
    control = cc_read_json(CONTROL_RESULTS)
    record(
        "t16_no_page_level_bound_certifies_a_random_ranking",
        bool(control["page_bounds_protect"]),
        control["worst_violation_share"],
    )
    # Stratified sampling by hand: two environments, alternate turns, a shared page taken once.
    # Pages at 0, 1, 10 and 11 on one axis, starting from 0: 11 is farthest; then 1 and 10 tie
    # at a distance of 1 from the chosen set, and the tie goes to the smaller page id.
    toy = pd.DataFrame({"x": [0.0, 1.0, 10.0, 11.0]}, index=["a", "b", "c", "d"])
    record(
        "t17_the_greedy_diversity_rule_matches_hand_arithmetic",
        diversity_pages(toy, "a", 3) == ["a", "d", "b"],
        diversity_pages(toy, "a", 3),
    )
    generated = set(pd.read_parquet(hy1.CANDIDATES)["candidate_id"].astype(str))
    record(
        "t18_every_candidate_is_one_hy1_already_generated",
        not set(population["candidate_id"].astype(str)) - generated,
    )
    issued = cc_read_json(DESIGN_RECORD)["issued_utc"]
    record(
        "t19_the_design_record_predates_every_endpoint",
        all(
            issued <= cc_read_json(p)["issued_utc"]
            for p in (PAGE_SELECTIONS, SCORE_REGISTRY, CALIBRATION_REGISTRY, ARM_CURVES, PROJECTION)
        ),
    )
    record(
        "t20_every_cell_carries_every_draw",
        all(
            int(g["draw"].nunique()) == DRAWS
            for _k, g in table.groupby(["arm", "direction", "budget"])
        ),
    )
    record(
        "t21_every_outcome_uses_the_declared_delta_and_risk_targets",
        sorted(set(table["epsilon"].astype(float))) == sorted(EPSILONS)
        and cc_read_json(CALIBRATION_REGISTRY)["delta"] == DELTA,
    )
    record(
        "t22_budgets_nest_for_random_pages",
        all(
            page_order(population, block, d)[:10] == page_order(population, block, d)[:25][:10]
            for block in blocks.values()
            for d in range(DRAWS)
        ),
    )
    passed = sum(1 for t in tests if t["passed"])
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
    failing = [t["test"] for t in tests if not t["passed"]]
    print(
        f"negative: {passed}/{len(tests)} pass ({time.monotonic() - started:.0f}s)"
        + (f", failing {failing}" if failing else "")
    )
    if failing:
        raise PhaseError(f"{len(failing)} falsification tests failed: {failing}")
    return 0


# ------------------------------------------------------------------ the decision


def criteria_from_artifacts() -> dict[str, Any]:
    frontier = cc_read_json(FRONTIER)
    projection = cc_read_json(PROJECTION)["by_direction"]
    family = cc_read_json(STATISTICAL_TESTS)["primary_family"]
    primary = frontier["frontier"][PRIMARY_ARM]
    n_star = {name: primary[name]["n_star"] for name in DIRECTIONS}
    outcome = assign_outcome(n_star)
    limits = {
        name: limit_reading(
            frontier["model_limited"][name],
            projection[name]["projected_frontier_page_corrected"],
            n_star[name] is not None,
        )
        for name in DIRECTIONS
    }
    model_everywhere = all(frontier["model_limited"][n] for n in DIRECTIONS)
    return {
        "n_star": n_star,
        "outcome": outcome,
        "next_stage": next_stage(outcome, model_everywhere),
        "limits": limits,
        "frontier": {
            arm: {n: v["n_star"] for n, v in row.items()}
            for arm, row in frontier["frontier"].items()
        },
        "projected_frontier": {
            name: {
                method: projection[name][f"projected_frontier_{method}"]
                for method in PROJECTION_METHODS
            }
            for name in DIRECTIONS
        },
        "supported": {name: bool(row["survives_holm"]) for name, row in family.items()},
    }


def shown(budget: str | None) -> str:
    return budget if budget is not None else "not reached"


def outcome_label(state: dict[str, Any]) -> str:
    parts = [f"{DIRECTION_SHORT[n]} N* {shown(state['n_star'][n])}" for n in DIRECTIONS]
    return (
        f"{state['outcome']}: {OUTCOME_TAXONOMY[state['outcome']]} -- cross-fitted pages "
        + "; ".join(parts)
    )


def run_decide() -> int:
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    negative = cc_read_json(FALSIFICATION)
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "primary_arm": PRIMARY_ARM,
            "primary_epsilon": PRIMARY_EPSILON,
            "delta": DELTA,
            "n_star_primary": state["n_star"],
            "frontier": state["frontier"],
            "limit_reading": state["limits"],
            "projected_frontier": state["projected_frontier"],
            "projection_is_not_a_measurement": True,
            "statistically_supported": state["supported"],
            "statistical_family_surviving": cc_read_json(STATISTICAL_TESTS)["surviving"],
            "outcome": state["outcome"],
            "outcome_label": outcome_label(state),
            "recommended_next_stage": state["next_stage"],
            "production_ready": False,
            "certified": False,
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


# ------------------------------------------------------------------ post-hoc diagnostic


def run_diagnose() -> int:
    """POST HOC: why one direction never certifies, and how many realizations the pool budget has.

    Added after the results were read and computed from the same fits and stored outcomes. It
    feeds no criterion.
    """
    started = time.monotonic()
    _require(DECISION, "decide")
    _forbid(PURITY_DIAGNOSTIC)
    population = load_population()
    matrix = pd.read_parquet(rk1.FEATURE_MATRIX)
    design = rk1._design(population, matrix, rk1.model_columns(rk1.PRIMARY_RANKER))
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = population["document_id"].astype(str).to_numpy()
    purity: dict[str, Any] = {}
    for name, blocks in rk2.all_blocks(population).items():
        oof, _test = _projection_fit(population, blocks, design, grades, groups)
        cutoffs = prefix_thresholds(oof["safety"].to_numpy(dtype=np.float64), PROJECTION_LADDER)
        rows = []
        for depth, cutoff in zip(PROJECTION_LADDER, cutoffs, strict=True):
            accepted = oof[oof["safety"] >= cutoff]
            rows.append(
                {
                    "depth": depth,
                    "accepted": len(accepted),
                    "harm": _share(float(accepted["is_harmful"].sum()), len(accepted)),
                }
            )
        harms = [r["harm"] for r in rows if r["harm"] is not None]
        best = min(rows, key=lambda r: (r["harm"] if r["harm"] is not None else 2.0, r["depth"]))
        purity[name] = {
            "out_of_fold_decisions": len(oof),
            "harm_by_depth": rows,
            "minimum_harm_over_the_ladder": float(min(harms)),
            "depth_of_the_minimum": best["depth"],
            "any_depth_below_the_target": bool(min(harms) <= PRIMARY_EPSILON + HARM_TOLERANCE),
        }
    table = pd.read_parquet(POLICY_OUTCOMES)
    pool = table[(table["budget"] == POOL) & (table["epsilon"] == PRIMARY_EPSILON)]
    realizations = {
        str(arm): {
            str(name): int(
                group[["cutoff", "accepted", "repaired_error_sites"]]
                .astype(str)
                .drop_duplicates()
                .shape[0]
            )
            for name, group in frame.groupby("direction")
        }
        for arm, frame in pool.groupby("arm")
    }
    _write_json_once(
        PURITY_DIAGNOSTIC,
        {
            **_analysis_envelope("purity_diagnostic"),
            "post_hoc": True,
            "added_after": "the measured and projected results were read",
            "feeds_no_criterion": True,
            "out_of_fold_purity": purity,
            "distinct_realizations_at_the_pool_budget": realizations,
            "draws_per_cell": DRAWS,
            "what_it_explains": (
                "a page-level bound can certify a cutoff only where the out-of-fold harm of the "
                "pool's decisions sits below the target; if it never does at any depth, pages "
                "that look like these cannot help, however many are bought. And at the pool "
                "budget every draw buys the same pages, so a rule that does not depend on the "
                "purchase order has one realization, not twenty"
            ),
        },
    )
    print(
        "diagnose: minimum out-of-fold harm "
        + str({n: round(v["minimum_harm_over_the_ladder"], 4) for n, v in purity.items()})
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = th1.FIGURE_NOTE
SURFACE = th1.SURFACE
INK = th1.INK
INK_SECONDARY = th1.INK_SECONDARY
GRID = th1.GRID
# Categorical slots 1-6 of the reference palette, validated as a set on this surface. Three sit
# below 3:1 contrast, so every series also carries its own marker and a legend entry.
ARM_COLOURS = {
    CL: "#2a78d6",
    PL: "#eb6834",
    ST: "#1baf7a",
    DV: "#eda100",
    CF: "#e87ba4",
    SP: "#008300",
}
ARM_MARKERS = {CL: "o", PL: "s", ST: "^", DV: "D", CF: "v", SP: "P"}
ARM_SHORT = {
    CL: "candidate-level",
    PL: "page-level",
    ST: "stratified pages",
    DV: "diversity pages",
    CF: "cross-fitted pages",
    SP: "spread labels",
}
METHOD_SHORT = {
    "page_corrected": "page-corrected bound",
    "finite_sample_certificate": "finite-sample page certificate",
}
FIGURES = (
    "safety_and_usefulness_by_budget.png",
    "recall_and_design_effect_by_budget.png",
    "page_projection.png",
    "out_of_fold_purity.png",
)


def run_figures() -> int:
    """Four figures, minimal academic style, every value read from a persisted artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    _require(PURITY_DIAGNOSTIC, "diagnose")
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
            "svg.hashsalt": "th2",
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

    curves = cc_read_json(ARM_CURVES)["by_arm"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    positions = np.arange(len(MEASURED))
    ticks = ["10 pages", "25 pages", "full pool"]

    def arm_line(ax: Any, arm: str, values: Sequence[float | None], label: bool) -> None:
        ax.plot(
            positions,
            [np.nan if v is None else v for v in values],
            color=ARM_COLOURS[arm],
            marker=ARM_MARKERS[arm],
            markersize=4,
            linewidth=1.5,
            label=ARM_SHORT[arm] if label else None,
        )

    # Figure 1 -- safety and usefulness at the measured page budgets.
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for column, name in enumerate(DIRECTIONS):
        for row, (field, ylabel, reference) in enumerate(
            (
                ("violation_share", "share of draws violating harm 0.1", VIOLATION_CEILING),
                ("working_share", "share of draws with a working safe gate", WORKING_FLOOR),
            )
        ):
            ax = axes[row, column]
            for arm in ARMS:
                arm_line(
                    ax,
                    arm,
                    [curves[arm][name][b][key][field] for b in MEASURED],
                    label=row == 0 and column == 0,
                )
            ax.axhline(reference, color=INK, linewidth=0.9, linestyle="--")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(positions, ticks)
            ax.set_ylabel(ylabel if column == 0 else "")
            if row == 0:
                ax.set_title(f"{DIRECTION_SHORT[name]} (target: {TARGET_OF[name]})", fontsize=8.5)
    axes[0, 0].legend(frameon=False, fontsize=6.3, loc="upper right")
    fig.suptitle(
        "Measured page budgets: safety (top, dashed: the 0.1 ceiling) and usefulness (bottom, "
        "dashed: the 0.5 floor)",
        fontsize=8.5,
    )
    save(fig, FIGURES[0], [ARM_CURVES], "violation and working shares by arm and page budget")

    # Figure 2 -- what is recalled, and how clustered the calibration evidence is.
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for column, name in enumerate(DIRECTIONS):
        ax = axes[0, column]
        for arm in ARMS:
            arm_line(
                ax,
                arm,
                [curves[arm][name][b][key]["repair_recall"]["median"] for b in MEASURED],
                label=column == 0,
            )
        ax.plot(
            positions,
            [curves[CF][name][b][key]["oracle_repair_recall"]["median"] for b in MEASURED],
            color=INK_SECONDARY,
            linestyle=":",
            linewidth=1.2,
            label="oracle boundary, cross-fitted (analysis only)" if column == 0 else None,
        )
        ax.set_ylim(-0.01, 0.25)
        ax.set_ylabel("median exact-repair recall" if column == 0 else "")
        ax.set_title(f"{DIRECTION_SHORT[name]} (target: {TARGET_OF[name]})", fontsize=8.5)
        ax = axes[1, column]
        for arm in PAGE_BOUND_ARMS:
            arm_line(
                ax,
                arm,
                [curves[arm][name][b][key]["design_effect"]["median"] for b in MEASURED],
                label=False,
            )
        ax.axhline(1.0, color=GRID, linewidth=0.8)
        ax.set_ylim(0.0, 7.0)
        ax.set_xticks(positions, ticks)
        ax.set_ylabel("median page design effect" if column == 0 else "")
    axes[0, 1].legend(
        *axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=6.0, loc="upper right"
    )
    fig.suptitle(
        f"What each arm recalls at harm <= {PRIMARY_EPSILON} (top) and how clustered its "
        "calibration evidence is (bottom)",
        fontsize=8.5,
    )
    save(fig, FIGURES[1], [ARM_CURVES], "median recall and page design effect by arm and budget")

    # Figure 3 -- PROJECTION.
    projection = cc_read_json(PROJECTION)["by_direction"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for column, name in enumerate(DIRECTIONS):
        sizes = [int(s) for s in projection[name]["by_size"]]
        sizes.sort()
        for method, colour, marker in (
            ("page_corrected", ARM_COLOURS[CF], "v"),
            ("finite_sample_certificate", INK, "s"),
        ):
            cells = [projection[name]["by_size"][str(s)][method] for s in sizes]
            axes[0, column].plot(
                sizes,
                [c["working_share"] for c in cells],
                color=colour,
                marker=marker,
                markersize=4,
                linewidth=1.5,
                label=METHOD_SHORT[method] if column == 0 else None,
            )
            axes[1, column].plot(
                sizes,
                [
                    np.nan if c["repair_recall"]["median"] is None else c["repair_recall"]["median"]
                    for c in cells
                ],
                color=colour,
                marker=marker,
                markersize=4,
                linewidth=1.5,
            )
        axes[0, column].axhline(WORKING_FLOOR, color=INK, linewidth=0.9, linestyle="--")
        axes[0, column].axvline(projection[name]["pool_pages"], color=GRID, linewidth=1.0)
        axes[1, column].axhline(
            projection[name]["oracle_repair_recall"],
            color=INK_SECONDARY,
            linewidth=0.9,
            linestyle=":",
        )
        axes[0, column].set_ylim(-0.03, 1.03)
        axes[1, column].set_ylim(-0.01, 0.25)
        axes[0, column].set_title(f"{DIRECTION_SHORT[name]}", fontsize=8.5)
        axes[1, column].set_xscale("log")
        axes[1, column].set_xticks(sizes, [str(s) for s in sizes])
        axes[1, column].set_xlabel("pages (grey line: the real pool; beyond it, resampled copies)")
    axes[0, 0].set_ylabel("share of resamples with a working safe gate")
    axes[1, 0].set_ylabel("median exact-repair recall")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle(
        "PROJECTION, not a measurement: pool pages resampled to larger page counts (dotted: the "
        "oracle boundary)",
        fontsize=8.5,
    )
    save(fig, FIGURES[2], [PROJECTION], "projected working share and recall by page count")

    # Figure 4 -- POST HOC: the out-of-fold harm of the pool's decisions by acceptance depth.
    purity = cc_read_json(PURITY_DIAGNOSTIC)["out_of_fold_purity"]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for name, colour, style in ((D_G2Q, ARM_COLOURS[CL], "-"), (D_Q2G, ARM_COLOURS[PL], "--")):
        rows = purity[name]["harm_by_depth"]
        ax.plot(
            [r["depth"] for r in rows],
            [np.nan if r["harm"] is None else r["harm"] for r in rows],
            color=colour,
            linewidth=1.4,
            linestyle=style,
            label=DIRECTION_SHORT[name],
        )
    ax.axhline(PRIMARY_EPSILON, color=INK, linewidth=0.9, linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 0.45)
    ax.set_xlabel("acceptance depth: share of the pool's decisions accepted")
    ax.set_ylabel("out-of-fold harm among the accepted")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(
        "POST HOC: how pure the top of the adapted ranking is (dashed: the target)", fontsize=8.5
    )
    save(fig, FIGURES[3], [PURITY_DIAGNOSTIC], "post hoc: out-of-fold harm by acceptance depth")

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
            "projection_figure_is_not_a_measurement": FIGURES[2],
            "style": "matplotlib only, minimal academic",
            "every_value_read_from_an_artifact": True,
        },
    )
    print(f"figures: {len(manifest)} written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ determinism and record

DERIVED_OUTPUTS = (
    "SPLIT_REGISTRY",
    "PAGE_SELECTIONS",
    "CELL_SCORES",
    "SCORE_REGISTRY",
    "POLICY_OUTCOMES",
    "CALIBRATION_REGISTRY",
    "ARM_CURVES",
    "FRONTIER",
    "PROJECTION",
    "CONTROL_RESULTS",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "DECISION",
    "PURITY_DIAGNOSTIC",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("splits", run_splits),
        ("sample", run_sample),
        ("score", run_score),
        ("calibrate", run_calibrate),
        ("curves", run_curves),
        ("frontier", run_frontier),
        ("project", run_project),
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
    differing = sorted(n for n in published if any(r.get(n) != published[n] for r in runs))
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
                "pages": PAGE_SEED,
                "projection": PROJECTION_SEED,
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
    "2. Research Questions and Hypothesis": (DESIGN_RECORD, ARM_CURVES),
    "3. What the Data Can Measure": (SPLIT_REGISTRY, PAGE_SELECTIONS, DESIGN_RECORD),
    "4. Six Arms": (PAGE_SELECTIONS, SCORE_REGISTRY, DESIGN_RECORD),
    "5. Safety and Usefulness at the Measured Budgets": (
        ARM_CURVES,
        PURITY_DIAGNOSTIC,
        PAGE_SELECTIONS,
    ),
    "6. Recall, Review and Worst-Page Harm": (ARM_CURVES, PAGE_SELECTIONS),
    "7. Calibration Error and the Page Design Effect": (ARM_CURVES, PAGE_SELECTIONS),
    "8. Does Page Diversity Matter?": (STATISTICAL_TESTS, ARM_CURVES, PAGE_SELECTIONS),
    "9. The Measured Frontier": (FRONTIER, ARM_CURVES),
    "10. Projection Beyond the Pool": (PROJECTION, DESIGN_RECORD),
    "11. Model Capability or Calibration Data?": (
        DECISION,
        FRONTIER,
        PROJECTION,
        PURITY_DIAGNOSTIC,
    ),
    "12. Statistical Tests": (STATISTICAL_TESTS, DESIGN_RECORD, PAGE_SELECTIONS),
    "13. Controls and Falsification": (CONTROL_RESULTS, FALSIFICATION),
    "14. Limitations": (SPLIT_REGISTRY, PURITY_DIAGNOSTIC, STATISTICAL_TESTS, DESIGN_RECORD),
    "15. Research Decision": (DECISION, FRONTIER, PROJECTION, STATISTICAL_TESTS, FALSIFICATION),
    "16. Next Research Direction": (
        DECISION,
        PROJECTION,
        PURITY_DIAGNOSTIC,
        SPLIT_REGISTRY,
        PAGE_SELECTIONS,
    ),
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
        p for p, sha in freeze["upstream_sha256"].items() if file_sha256(REPO / p) != sha
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
                again[k] == published[k]
                for k in published
                if k not in ("issued_utc", "runtime_seconds")
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
            "sections": {n: [_relative(p) for p in ps] for n, ps in REPORT_SECTIONS.items()},
            "audit": audit,
            "untraceable": audit["untraceable"],
            "rule": (
                "every number in the report is read from one of the artifacts its section names, "
                "and from the field that means what the sentence says"
            ),
        },
    )
    print(
        f"record: {sum(1 for p in PRODUCED if p.is_file())} artifacts, report claims "
        f"{audit['numeric_claims']}, untraceable {audit['untraceable_numeric_claims']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    phases: dict[str, Callable[[], int]] = {
        "reconstruct": run_reconstruct,
        "freeze": run_freeze,
        "preregister": run_preregister,
        "splits": run_splits,
        "sample": run_sample,
        "score": run_score,
        "calibrate": run_calibrate,
        "curves": run_curves,
        "frontier": run_frontier,
        "project": run_project,
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
