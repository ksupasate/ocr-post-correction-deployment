#!/usr/bin/env python3
"""SGV-RK4: generator-adaptive risk-aware ranking with page-level calibration.

SGV-RK3's risk-aware LambdaRank selects OCR corrections well on generators it was trained on and
fails on one it never saw: a mean held-out harm AUROC of 0.69, and a source-chosen cutoff that is
unsafe in one direction. SGV-RK2 showed that target labels recover a ranking, and SGV-TH1 and TH2
that a cutoff chosen on few pages is optimistic because OCR harm clusters by page. This stage
adapts RK3 to the new generator with a budget of labelled PAGES and chooses its cutoff with the
page structure in view.

**1. Page budgets, measured only.** 0, 5, 10 and 25 pages and the whole adaptation pool (31 and 30
pages). The brief's 50 to 500 pages exceed the pool and are recorded as not measurable here; no
projection is made, because a ranker refitted on resampled copies of 30 pages is not evidence.

**2. RK3's primary transfer protocol.** Source fit, source cutoff block and the sealed target test
block are RK3's generator-and-document held-out blocks exactly, so budget 0 is RK3's zero-shot
result. The adaptation pool is the target generator's pages in the same folds as the source rows.

**3. Four adaptation arms and one baseline.** Full refitting on source plus target pages (M1),
leaf values only (M2), a target calibration layer a*S + b (M3), a page-risk term added in log-odds
(M4), and RK2's target-only ranker on the same pages.

**4. Three cutoff rules on held-back pages.** Purchased pages are split into fit and calibration
pages, never sharing a page. The cutoff is RK1's plug-in rule, a candidate-level Clopper-Pearson
bound, or the same bound with the page design effect.

    --reconstruct   section 0: DS1, RK1, RK2, TH1, TH2 and RK3 re-read from their own artifacts
    --freeze        upstream hashes and the frozen inputs this stage consumes
    --preregister   budgets, arms, cutoff rules, deployable definition, criteria, outcome rule
    --splits        the four blocks per direction and the adaptation pool's pages
    --sample        every draw's page order, purchases, fit pages and calibration pages
    --reproduce     DS1, RK1, RK2 and RK3 rebuilt exactly before any adaptation
    --adapt         every arm x direction x budget x draw, scored on test and calibration rows
    --policies      every cutoff rule at every harm target, read on the sealed test pages
    --curves        label-efficiency curves over draws
    --frontier      the deployable frontier and the ranking-ready budgets
    --transfer      generator transfer matrix and the harm-recall frontier
    --calibration   the page calibration effect
    --ablate        feature-family ablations of the primary arm
    --controls      target-grade permutation and random ranking
    --stats         exact sign tests over paired draws, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen upstream inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / ADAPTATION ONLY. No candidate is generated, no OCR output is regenerated, no
generator identity enters a model, nothing is certified and nothing is production-ready. The
confirmatory reserve stays LOCKED and `ready_for_external_confirmation` is false by construction.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_rk3_risk_aware_ranking as rk3
import sgv_th2_page_budget_frontier as th2
from ocr_risk.io.hashing import canonical_hash, file_sha256

th1 = th2.th1
rk2 = rk3.rk2
rk1 = rk3.rk1
ds1 = rk3.ds1
rl2 = rk3.rl2
rl1 = rk3.rl1
s15 = rk3.s15

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/generated/sgv_rk4"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"

SPLIT_REGISTRY = OUT / "split_registry.json"
PURCHASE_REGISTRY = OUT / "purchase_registry.json"
REPRODUCTION = OUT / "reproduction.json"
CELL_SCORES = OUT / "cell_scores.parquet"
ADAPTATION_REGISTRY = OUT / "adaptation_registry.json"

POLICY_OUTCOMES = OUT / "policy_outcomes.parquet"
RANKING_OUTCOMES = OUT / "ranking_outcomes.parquet"
POLICY_REGISTRY = OUT / "policy_registry.json"
CURVES = OUT / "label_efficiency_curves.json"
FRONTIER = OUT / "deployable_frontier.json"
TRANSFER = OUT / "generator_transfer.json"
CALIBRATION_EFFECT = OUT / "page_calibration_effect.json"
ABLATION = OUT / "feature_ablation.json"
CONTROL_RESULTS = OUT / "control_results.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_rk4/generator_adaptive_ranking.md"

SCHEMA_VERSION = 1
STAGE = "sgv_rk4_generator_adaptive_ranking"
HYPOTHESIS = "SGV-RK4-D1"
STAGE_KIND = "DEVELOPMENT / ADAPTATION"

PhaseError = rk3.PhaseError
_relative = rk3._relative
_git = rk3._git
_write_json_once = rk3._write_json_once
_write_parquet_once = rk3._write_parquet_once
cc_read_json = rk3.cc_read_json
_require = rk3._require
_forbid = rk3._forbid
auroc = rk3.auroc
_share = rk3._share

BOOTSTRAP_RESAMPLES = rk3.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rk3.BOOTSTRAP_SEED
FIT_SEED = rk3.FIT_SEED
PERMUTATION_SEED = 20260931

# ------------------------------------------------------------------ the frozen design

C0 = rk3.C0
C1 = rk3.C1
GENERATORS = rk3.GENERATORS
D_G2Q = rk3.D_G2Q
D_Q2G = rk3.D_Q2G
DIRECTIONS = rk3.DIRECTIONS
SOURCE_OF = rk3.SOURCE_OF
TARGET_OF = rk3.TARGET_OF
DIRECTION_SHORT = rk3.DIRECTION_SHORT
OTHER = {D_G2Q: D_Q2G, D_Q2G: D_G2Q}

POOL_FOLDS = (0, 1, 2)
TEST_FOLDS = (3, 4)

EPSILONS = rk3.EPSILONS
PRIMARY_EPSILON = rk3.PRIMARY_EPSILON
HARM_TOLERANCE = rk3.HARM_TOLERANCE
DELTA = th1.DELTA
VIOLATION_CEILING = th1.VIOLATION_CEILING
WORKING_FLOOR = th1.WORKING_FLOOR
HARM_FLOOR = rk3.TRANSFER_HARM_FLOOR
WORST_PAGE_MIN_ACCEPTED = th2.WORST_PAGE_MIN_ACCEPTED

POOL = "pool"
BUDGETS = ("0", "5", "10", "25", POOL)
ADAPTED_BUDGETS = BUDGETS[1:]
DECISION_BUDGET = "10"
BRIEF_BUDGETS_NOT_MEASURABLE = (50, 100, 250, 500)
DRAWS = rk2.DRAWS
PAGE_SEED = "sgv-rk4-page-order-v1:2026-09-22"
CALIBRATION_EVERY = 3  # the page at every third purchase position is held back to choose cutoffs
ABLATION_DRAWS = 5

A0 = "a0_zero_shot"
M1 = "m1_full_adaptation"
M2 = "m2_leaf_adaptation"
M3 = "m3_calibration_layer"
M4 = "m4_page_risk"
B2 = "b2_rk2_target_only"
ARMS = (A0, M1, M2, M3, M4, B2)
ADAPTED_ARMS = (M1, M2, M3, M4, B2)
PRIMARY_ARM = M1
TREE_ARMS = (A0, M1, M2, B2)
ARM_LABEL = {
    A0: "RK3's risk-aware LambdaRank on R5, fitted on the source generator's fit rows alone",
    M1: "full adaptation: risk-aware LambdaRank on R5 refitted on the source fit rows plus the "
    "purchased target fit pages",
    M2: "leaf adaptation: A0's trees split for split, leaf values re-estimated on the target fit "
    "pages with the risk-aware gradient",
    M3: "target calibration layer: S_new = a * S_A0 + b, fitted by logistic regression of harm on "
    "A0's score over the target fit pages",
    M4: "page-aware risk adjustment: M3's log-odds plus a page-risk term fitted on label-free page "
    "features and the page's mean A0 score",
    B2: "SGV-RK2's primary arm on the same pages: RK1's LambdaMART on R3, target fit pages only",
}
ARM_SHORT = {
    A0: "zero shot",
    M1: "M1 full",
    M2: "M2 leaf",
    M3: "M3 calibration",
    M4: "M4 page risk",
    B2: "RK2 target-only",
}

R_NAIVE = "naive"
R_CANDIDATE = "candidate_bound"
R_PAGE = "page_bound"
RULES = (R_NAIVE, R_CANDIDATE, R_PAGE)
PRIMARY_RULE = R_PAGE
RULE_LABEL = {
    R_NAIVE: "RK1's plug-in rule: the loosest cutoff whose calibration harm meets the target",
    R_CANDIDATE: (
        f"the loosest cutoff whose one-sided Clopper-Pearson bound at delta {DELTA} on "
        "calibration harm meets the target, counting decisions as independent"
    ),
    R_PAGE: (
        "the same bound on counts divided by the page design effect of the calibration pages; "
        "no cutoff when fewer than two calibration pages carry decisions"
    ),
}

PAGE_FEATURES = (
    "rx_page_low_confidence_share",
    "rx_page_oov_share",
    "rx_page_lm_per_char",
    "rx_page_site_density",
)
PAGE_MEAN_SCORE = "page_mean_zero_shot_score"

OUTCOME_TAXONOMY = {
    "A": "reliable automation within the measured page budgets in both directions",
    "B": "the ranking recovers, but no measured page budget yields reliable automation",
    "C": "reliable automation within the measured budgets in one direction only",
    "D": "neither the ranking recovers nor reliable automation is reached",
}
NEXT_STAGE = {
    "A": "NEXT: CONFIRM THE ADAPTED RANKER AND ITS PAGE-LEVEL CUTOFF ON NEW PAGES",
    "B": "NEXT: EXPAND THE LABELLED PAGE POOL FOR THE ADAPTED RANKER",
    "C": "NEXT: DIRECTION-SPECIFIC ONBOARDING FOR THE GENERATOR THAT DID NOT REACH A SAFE CUTOFF",
    "D": "NEXT: TARGET-GENERATOR EVIDENCE BEYOND LABEL ADAPTATION",
}

PRIMARY_FAMILY = (
    "P1_harm_auroc_m1_vs_zero_shot_at_pool",
    "P2_harm_auroc_m1_vs_rk2_target_only_at_10_pages",
    "P3_harm_auroc_m2_vs_m1_at_10_pages",
    "P4_violation_page_vs_candidate_bound_m1_at_pool",
    "P5_harm_auroc_m4_vs_m3_at_pool",
)
FAMILY_SPEC: dict[str, dict[str, str]] = {
    PRIMARY_FAMILY[0]: {"left": M1, "right": A0, "metric": "harm_auroc", "budget": POOL},
    PRIMARY_FAMILY[1]: {"left": M1, "right": B2, "metric": "harm_auroc", "budget": "10"},
    PRIMARY_FAMILY[2]: {"left": M2, "right": M1, "metric": "harm_auroc", "budget": "10"},
    PRIMARY_FAMILY[3]: {
        "left": R_PAGE,
        "right": R_CANDIDATE,
        "metric": "violation",
        "budget": POOL,
        "arm": M1,
    },
    PRIMARY_FAMILY[4]: {"left": M4, "right": M3, "metric": "harm_auroc", "budget": POOL},
}

UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_rk3": {
        "outcome": "B",
        "recommended_next_stage": (
            "NEXT: TARGET-GENERATOR ADAPTATION OF THE RISK-AWARE RANKER WITH PAGE-LEVEL CUTOFFS"
        ),
        "ready_for_external_confirmation": False,
        "deployment_claim_permitted": False,
    },
    "sgv_th2": {"outcome": "D", "ready_for_external_confirmation": False},
    "sgv_th1": {"outcome": "D", "ready_for_external_confirmation": False},
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_rk4-{artifact}-v{SCHEMA_VERSION}",
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
    """The smallest measured budget from which the flag holds at every larger measured budget."""
    chosen: str | None = None
    for label in reversed(BUDGETS):
        if label not in flags:
            continue
        if not flags[label]:
            break
        chosen = label
    return chosen


def assign_outcome(n_star: dict[str, str | None], ranking_ready: dict[str, str | None]) -> str:
    """A: both directions deployable; C: one; B: ranking recovers in both; D: otherwise."""
    reached = [d for d in DIRECTIONS if n_star[d] is not None]
    if len(reached) == len(DIRECTIONS):
        return "A"
    if reached:
        return "C"
    if all(ranking_ready[d] is not None for d in DIRECTIONS):
        return "B"
    return "D"


def deployable(violation_share: float | None, working_share: float | None) -> bool:
    return bool(
        violation_share is not None
        and working_share is not None
        and violation_share <= VIOLATION_CEILING
        and working_share >= WORKING_FLOOR
    )


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    files = list(rk3._upstream_files())
    files.append(REPO / "scripts/sgv_rk3_risk_aware_ranking.py")
    files.append(REPO / "scripts/sgv_th1_threshold_calibration.py")
    files.append(REPO / "scripts/sgv_th2_page_budget_frontier.py")
    for module in (rk3, th1, th2):
        for pattern in ("*.json", "*.parquet"):
            files.extend(sorted(module.OUT.glob(pattern)))
        files.append(module.REPORT)
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = rk3.xr1._check
    checks = list(rk3.upstream_checks())
    decisions = {
        "sgv_rk3": cc_read_json(rk3.DECISION),
        "sgv_th2": cc_read_json(th2.DECISION),
        "sgv_th1": cc_read_json(th1.DECISION),
    }
    for stage, expected in UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))
    for stage, module in (("sgv_rk3", rk3), ("sgv_th2", th2), ("sgv_th1", th1)):
        checks.append(
            _check(
                f"{stage}.determinism",
                cc_read_json(module.DETERMINISM)["all_runs_identical"],
                True,
            )
        )
        checks.append(
            _check(
                f"{stage}.report_traceability",
                cc_read_json(module.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
                0,
            )
        )
        checks.append(
            _check(
                f"{stage}.confirmatory_reserve_consumed",
                decisions[stage].get("confirmatory_reserve_consumed"),
                False,
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
    return any(p.exists() for p in (CELL_SCORES, POLICY_OUTCOMES, DECISION))


def run_freeze() -> int:
    started = time.monotonic()
    for path in (RESEARCH_FREEZE, FROZEN_CONFIGURATION):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an RK4 endpoint already exists; the freeze must precede every one")
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
            "regenerates_ocr": False,
            "changes_an_upstream_outcome": False,
            "uses_ground_truth": False,
        },
    )
    rk3_transfer = cc_read_json(rk3.TRANSFER)
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "rk3_feature_matrix": _relative(rk3.FEATURE_MATRIX),
            "rk3_ranking_scores": _relative(rk3.RANKING_SCORES),
            "rk3_utility": cc_read_json(rk3.DESIGN_RECORD)["utility"],
            "rk3_boosting": cc_read_json(rk3.DESIGN_RECORD)["boosting"],
            "rk3_outcome": cc_read_json(rk3.DECISION)["outcome"],
            "rk3_held_out_harm_auroc": {
                d: rk3_transfer["tdoc"][d][rk3.M_RK3]["harm_auroc"] for d in DIRECTIONS
            },
            "rk3_held_out_mean_harm_auroc": rk3_transfer["tdoc_mean_harm_auroc"][rk3.M_RK3],
            "rk3_in_distribution_reference": {
                d: rk3_transfer["matrix"][rk3.M_RK3][
                    f"train_both|eval_{rk3.GENERATOR_SHORT[TARGET_OF[d]]}"
                ]["harm_auroc"]
                for d in DIRECTIONS
            },
            "th1_deployable_definition": {
                "violation_ceiling": VIOLATION_CEILING,
                "working_floor": WORKING_FLOOR,
                "delta": DELTA,
            },
            "th2_outcome": cc_read_json(th2.DECISION)["outcome"],
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
        raise PhaseError("an RK4 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "research_questions": {
                "RQ1": "can few-shot target-generator adaptation recover ranking performance?",
                "RQ2": "can page-level calibration improve safety under page-clustered OCR errors?",
                "RQ3": (
                    "how many labelled pages are required before reliable automation becomes "
                    "possible?"
                ),
            },
            "primary_hypothesis": (
                "adapting RK3 with the target generator's labelled pages recovers its harm "
                "ranking, and a cutoff chosen with the page design effect in view holds the harm "
                "target where a candidate-level cutoff does not, so that some measured page "
                "budget yields reliable automation"
            ),
            "budgets": {
                "measured_pages": list(BUDGETS),
                "pool": "every page of the target adaptation pool, 31 and 30 pages",
                "not_measurable": list(BRIEF_BUDGETS_NOT_MEASURABLE),
                "why_not_measurable": (
                    "the brief's 50 to 500 pages exceed the 31 and 30 pages the adaptation pools "
                    "hold. They are recorded as not measurable on this data, and no projection is "
                    "made: a ranker refitted on resampled copies of about 30 pages would not be "
                    "evidence of what 100 distinct pages buy. The user chose this before any fit"
                ),
                "decision_budget": DECISION_BUDGET,
                "draws": DRAWS,
            },
            "blocks": {
                "source_fit": "RK3's held-out protocol training rows: source-only rows, folds 0-1",
                "source_calibration": "RK3's held-out cutoff rows: source-only rows, fold 2",
                "target_pool": (
                    "every row the target generator proposed on folds 0-2; its pages are the "
                    "adaptation pool"
                ),
                "target_test": (
                    "RK3's held-out evaluation rows exactly: every row the target generator "
                    "proposed on folds 3-4. Sealed before any fit"
                ),
                "source_test": (
                    "every row the source generator proposed on folds 3-4, read only to measure "
                    "whether adaptation keeps the source generator's ranking"
                ),
                "page": (
                    "a document: labelling a page labels every engine's reading of it, so the "
                    "page is the unit of labelling cost"
                ),
            },
            "purchase": {
                "order": (
                    "pages sorted by a seeded hash of the draw index and the page id; a budget is "
                    "a prefix of the order, so budgets nest inside a draw"
                ),
                "calibration_pages": (
                    f"the page at every purchase position p with p mod {CALIBRATION_EVERY} = 2 is "
                    "held back to choose cutoffs; the others fit. Fit and calibration pages never "
                    "share a page, and both nest with the budget"
                ),
                "reads_labels": False,
                "seed": PAGE_SEED,
            },
            "arms": ARM_LABEL,
            "primary_arm": PRIMARY_ARM,
            "primary_arm_reason": (
                "the brief's first method. Source data is what RK3 was fitted on, and SGV-RK2 "
                "found joint refitting direction-dependent, so RK2's target-only arm runs beside "
                "it as the baseline"
            ),
            "page_risk_features": {
                "page_features": list(PAGE_FEATURES),
                "page_mean_score": (
                    "the mean zero-shot score over every target candidate on the same reading of "
                    "the page, labelled or not"
                ),
                "why": (
                    "a document-query ranker never trains the offsets between pages, so a global "
                    "cutoff inherits them untrained; M4 learns a page-level offset from label-free "
                    "page evidence"
                ),
            },
            "calibration_layer_note": (
                "an affine correction with a positive slope cannot reorder candidates, and every "
                "cutoff rule reads the arm's own calibration scores, so M3 measures what the "
                "zero-shot ranking achieves with a target-chosen cutoff. A negative slope would "
                "reverse the ranking"
            ),
            "rules": RULE_LABEL,
            "primary_rule": PRIMARY_RULE,
            "zero_budget_rule": (
                "at budget 0 no target page is labelled: the zero-shot ranker's cutoff is chosen "
                "on the source calibration block by the same rule"
            ),
            "deployable": {
                "draw_violates": (
                    "it accepts something on test and its realized harm exceeds the target"
                ),
                "draw_works": "it accepts something, holds the target and repairs an error site",
                "cell_deployable": (
                    f"violation share at most {VIOLATION_CEILING} and working share at least "
                    f"{WORKING_FLOOR} over the draws, TH1's definition"
                ),
                "n_star": (
                    "the smallest measured budget from which the cell is deployable at every "
                    "larger measured budget"
                ),
                "ranking_ready": (
                    f"the smallest measured budget from which the median test harm AUROC is at "
                    f"least {HARM_FLOOR} at every larger measured budget"
                ),
            },
            "criteria": {
                "R1": (
                    f"RQ1: {PRIMARY_ARM} is ranking-ready within the measured budgets in both "
                    "directions"
                ),
                "R2": (
                    f"RQ2: at the pool, for {PRIMARY_ARM}, in every direction where the "
                    f"candidate-level bound violates in more than {VIOLATION_CEILING} of draws, "
                    f"the page-level bound violates in at most {VIOLATION_CEILING}; and at least "
                    "one direction has such a candidate-level violation share"
                ),
                "R3": (
                    f"RQ3: {PRIMARY_ARM} under the {PRIMARY_RULE} rule has an N* within the "
                    "measured budgets in both directions"
                ),
                "deployment_claim": (
                    "permitted only when R3 holds, and then only as development evidence on the "
                    "measured pages"
                ),
            },
            "outcome_rule": OUTCOME_TAXONOMY,
            "outcome_precedence": "A, then C, then B, then D; the outcome reads R1 and R3",
            "next_stage_rule": NEXT_STAGE,
            "recall_denominator": (
                "every evaluable error site of the test block, as RK3's held-out protocol"
            ),
            "statistical_family": {name: FAMILY_SPEC[name] for name in PRIMARY_FAMILY},
            "statistical_plan": {
                "unit": "draw, paired across arms or rules, both directions pooled",
                "test": "exact two-sided sign test; tied draws carry no information",
                "effect": "median paired difference, or the difference in violation shares",
                "interval": (
                    f"{BOOTSTRAP_RESAMPLES} draw resamples stratified by direction, seed "
                    f"{BOOTSTRAP_SEED}"
                ),
                "multiplicity": "Holm within the frozen family of five",
                "scope": (
                    "draw-level inference conditional on the sealed test pages; test-page "
                    "sampling uncertainty is not in these intervals"
                ),
            },
            "controls": {
                "permutation": (
                    f"{B2} refitted at the pool with target grades shuffled inside each purchased "
                    "page, every draw; mean test harm AUROC must fall in "
                    f"{list(rk3.PERMUTATION_BAND)}"
                ),
                "permutation_m1": (
                    f"{PRIMARY_ARM} refitted the same way, reported descriptively: its source rows "
                    "keep their grades, so it should fall back toward the zero-shot ranking"
                ),
                "random": "seeded random scores through every cutoff rule",
            },
            "ablations": {
                "arm": PRIMARY_ARM,
                "budget": POOL,
                "draws": ABLATION_DRAWS,
                "drops": {
                    name: list(drop)
                    for name, drop in rk3.ABLATIONS.items()
                    if name.startswith("minus_new_")
                },
                "also": "R5 without every new family, which is R3",
            },
            "non_goals": [
                "no candidate is generated and no OCR output is regenerated",
                "no generator identity enters any model",
                "no hyperparameter is searched",
                "no page budget beyond the pool is projected",
                "the confirmatory reserve is not touched",
                "no certification, production claim or external confirmation",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the blocks


@dataclass(slots=True)
class Blocks:
    """One direction's frozen row blocks, as index arrays into RK3's population order."""

    direction: str
    source: str
    target: str
    source_fit: np.ndarray
    source_calibration: np.ndarray
    pool: np.ndarray
    test: np.ndarray
    source_test: np.ndarray
    pages: list[str]


def load_population() -> pd.DataFrame:
    return rk3.load_population()


def build_blocks(population: pd.DataFrame) -> dict[str, Blocks]:
    """RK3's held-out protocol plus the target adaptation pool. No label is read."""
    protocols = rk3.build_protocols(
        {rk3.ORDER_RK1: population, rk3.ORDER_RK2: rk3.strict_population()}
    )
    has = rk3.membership(population)
    fold = population["fold"].to_numpy(dtype=np.int64)
    documents = population["document_id"].astype(str).to_numpy()
    out: dict[str, Blocks] = {}
    for direction in DIRECTIONS:
        held = protocols[rk3.TDOC[direction]]
        target = TARGET_OF[direction]
        pool = np.flatnonzero(has[target] & np.isin(fold, POOL_FOLDS))
        out[direction] = Blocks(
            direction=direction,
            source=SOURCE_OF[direction],
            target=target,
            source_fit=held.train,
            source_calibration=held.threshold,
            pool=pool,
            test=held.test,
            source_test=protocols[rk3.TDOC[OTHER[direction]]].test,
            pages=sorted(set(documents[pool].tolist())),
        )
    return out


def page_order(pages: Sequence[str], draw: int) -> list[str]:
    """A draw's purchase order: pages sorted by a seeded hash. It reads the page id alone."""
    return sorted(
        pages,
        key=lambda page: (
            canonical_hash({"seed": PAGE_SEED, "draw": int(draw), "page": page}),
            page,
        ),
    )


def purchase(pages: Sequence[str], draw: int, label: str) -> tuple[list[str], list[str]]:
    """The fit pages and the held-back calibration pages a budget buys, in purchase order."""
    bought = page_order(pages, draw)[: budget_pages(label, len(pages))]
    fit = [p for i, p in enumerate(bought) if i % CALIBRATION_EVERY != 2]
    calibration = [p for i, p in enumerate(bought) if i % CALIBRATION_EVERY == 2]
    return fit, calibration


def rows_on(population: pd.DataFrame, rows: np.ndarray, pages: Sequence[str]) -> np.ndarray:
    documents = population["document_id"].astype(str).to_numpy()
    return rows[np.isin(documents[rows], list(pages))]


def _digest(population: pd.DataFrame, index: np.ndarray) -> str:
    return str(canonical_hash(population.iloc[index]["candidate_id"].astype(str).tolist()))


def run_splits() -> int:
    """The four blocks per direction and the adaptation pool, frozen before any fit."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    _forbid(SPLIT_REGISTRY)
    population = load_population()
    blocks = build_blocks(population)
    rk3_splits = cc_read_json(rk3.SPLIT_REGISTRY)["protocols"]
    documents = population["document_id"].astype(str).to_numpy()
    rows: dict[str, Any] = {}
    for direction, b in blocks.items():
        parts = {
            "source_fit": b.source_fit,
            "source_calibration": b.source_calibration,
            "target_pool": b.pool,
            "target_test": b.test,
            "source_test": b.source_test,
        }
        training = np.concatenate([b.source_fit, b.source_calibration, b.pool])

        def shared(left: np.ndarray, right: np.ndarray) -> int:
            return len(set(documents[left].tolist()) & set(documents[right].tolist()))

        rk3_row = rk3_splits[rk3.TDOC[direction]]
        rows[direction] = {
            "source_generator": b.source,
            "target_generator": b.target,
            **{
                name: {
                    "rows": int(index.size),
                    "pages": len(set(documents[index].tolist())),
                    "sites": int(population.iloc[index]["site_group"].nunique()),
                }
                for name, index in parts.items()
            },
            "pool_pages": len(b.pages),
            "test_digest": _digest(population, b.test),
            "equals_rk3_held_out_protocol": bool(
                b.source_fit.size == rk3_row["train"]["rows"]
                and b.source_calibration.size == rk3_row["threshold"]["rows"]
                and b.test.size == rk3_row["test"]["rows"]
            ),
            "isolation": {
                "pages_training_vs_test": shared(training, b.test),
                "pages_pool_vs_test": shared(b.pool, b.test),
                "candidates_training_vs_test": len(
                    set(population.iloc[training]["candidate_id"])
                    & set(population.iloc[b.test]["candidate_id"])
                ),
                "sites_training_vs_test": len(
                    set(population.iloc[training]["site_group"])
                    & set(population.iloc[b.test]["site_group"])
                ),
            },
            "brief_budgets_not_measurable": [
                n for n in BRIEF_BUDGETS_NOT_MEASURABLE if n > len(b.pages)
            ],
        }
    isolated = all(v == 0 for row in rows.values() for v in row["isolation"].values())
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_analysis_envelope("split_registry"),
            "directions": rows,
            "isolated": isolated,
            "blocks": cc_read_json(DESIGN_RECORD)["blocks"],
            "frozen_before_any_fit": not CELL_SCORES.exists(),
        },
    )
    if not isolated:
        raise PhaseError("a training page, site or candidate reaches the test block")
    print(
        "splits: "
        + ", ".join(f"{d} pool {r['pool_pages']} pages" for d, r in rows.items())
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


def run_sample() -> int:
    """Every draw's purchase, frozen before any fit. It reads page ids and nothing else."""
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    _forbid(PURCHASE_REGISTRY)
    population = load_population()
    rows = []
    for direction, b in build_blocks(population).items():
        for label in ADAPTED_BUDGETS:
            for draw in range(DRAWS):
                fit, calibration = purchase(b.pages, draw, label)
                fit_rows = rows_on(population, b.pool, fit)
                cal_rows = rows_on(population, b.pool, calibration)
                rows.append(
                    {
                        "direction": direction,
                        "budget": label,
                        "draw": draw,
                        "pages": len(fit) + len(calibration),
                        "fit_pages": fit,
                        "calibration_pages": calibration,
                        "fit_rows": int(fit_rows.size),
                        "calibration_rows": int(cal_rows.size),
                        "fit_digest": _digest(population, fit_rows),
                        "calibration_digest": _digest(population, cal_rows),
                    }
                )
    distinct = {
        f"{d}|{label}": len(
            {tuple(r["fit_pages"]) for r in rows if r["direction"] == d and r["budget"] == label}
        )
        for d in DIRECTIONS
        for label in ADAPTED_BUDGETS
    }
    _write_json_once(
        PURCHASE_REGISTRY,
        {
            **_analysis_envelope("purchase_registry"),
            "rows": rows,
            "draws": DRAWS,
            "budgets": list(ADAPTED_BUDGETS),
            "distinct_fit_page_sets": distinct,
            "rule": cc_read_json(DESIGN_RECORD)["purchase"],
            "frozen_before_any_fit": not CELL_SCORES.exists(),
        },
    )
    print(f"sample: {len(rows)} purchases ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the arms

LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000


def risk_leaf_refit(
    source: Any, design: np.ndarray, utility: np.ndarray, groups: np.ndarray
) -> Any:
    """The source ranker's trees, split for split, with leaf values re-estimated on new rows.

    RK3's boosting loop with the tree-growing step replaced by the source's frozen trees: at each
    round the risk-aware LambdaRank gradients are taken at the current scores and every leaf the
    rows reach gets the Newton step. On the source's own rows it reproduces the source exactly.
    """
    queries = rk3._queries(groups, utility)
    if not queries or not source.trees:
        return rk1.LambdaMART(trees=[], leaf_values=[], learning_rate=source.learning_rate)
    used = np.sort(np.concatenate(queries))
    scores = np.zeros(design.shape[0], dtype=np.float64)
    leaf_values: list[np.ndarray] = []
    for tree in source.trees:
        lambdas, curvature = rk3.ranking_gradients(scores, utility, queries, True, "range")
        leaves = tree.apply(design[used])
        values = np.zeros(tree.tree_.node_count, dtype=np.float64)
        for leaf in np.unique(leaves):
            member = leaves == leaf
            values[leaf] = float(lambdas[used][member].sum()) / (
                float(curvature[used][member].sum()) + 1e-9
            )
        scores += source.learning_rate * values[tree.apply(design)]
        leaf_values.append(values)
    return rk1.LambdaMART(
        trees=list(source.trees), leaf_values=leaf_values, learning_rate=source.learning_rate
    )


@dataclass(slots=True)
class Logistic:
    """An unweighted logistic model on standardized inputs; None when one class is missing."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def harm(self, inputs: np.ndarray) -> np.ndarray:
        z = ((inputs - self.mean) / self.scale) @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def fit_logistic(inputs: np.ndarray, harmful: np.ndarray) -> Logistic | None:
    from sklearn.linear_model import LogisticRegression

    if inputs.shape[0] == 0 or len(set(harmful.tolist())) < 2:
        return None
    mean = inputs.mean(axis=0)
    scale = inputs.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    model = LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, random_state=FIT_SEED)
    model.fit((inputs - mean) / scale, harmful.astype(int))
    return Logistic(
        mean=mean,
        scale=scale,
        coefficients=np.asarray(model.coef_[0], dtype=np.float64),
        intercept=float(model.intercept_[0]),
    )


def page_mean_scores(population: pd.DataFrame, rows: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Each row's page mean of the zero-shot score over the given rows, per engine reading.

    The rows are every target candidate on the pool and test pages, labelled or not, so the mean
    reads scores only.
    """
    keys = (
        population["environment"].astype(str) + "|" + population["document_id"].astype(str)
    ).to_numpy()
    frame = pd.DataFrame({"key": keys[rows], "score": scores[rows]})
    means = frame.groupby("key")["score"].mean()
    out = np.full(len(population), np.nan)
    out[rows] = means.reindex(keys[rows]).to_numpy(dtype=np.float64)
    return out


def page_inputs(
    population: pd.DataFrame, matrix: pd.DataFrame, zero_shot: np.ndarray, page_mean: np.ndarray
) -> np.ndarray:
    """M4's inputs per row: the zero-shot score, its page mean and four label-free page features."""
    page = rl1._matrix_for(population, matrix, list(PAGE_FEATURES))
    return np.column_stack([zero_shot, page_mean, page])


@dataclass(slots=True)
class Fitted:
    """One arm's fitted scorer and what the registry records about it."""

    score: Callable[[np.ndarray], np.ndarray]
    trainable: bool
    detail: dict[str, Any]


def fit_arm(
    arm: str,
    fit_rows: np.ndarray,
    context: dict[str, Any],
) -> Fitted:
    """One arm fitted on the purchased target fit rows (plus source rows for M1)."""
    design5 = context["design5"]
    design3 = context["design3"]
    utility = context["utility"]
    grades = context["grades"]
    groups = context["groups"]
    harmful = context["harmful"]
    zero_shot_model = context["zero_shot_model"]
    if arm == M1:
        rows = np.concatenate([context["source_fit"], fit_rows])
        model = rk3.fit_boosted(design5[rows], utility[rows], groups[rows], rk3.OBJ_RISK)
        return Fitted(
            lambda r, m=model: m.score(design5[r]),
            bool(model.trees),
            {"trees": len(model.trees), "rows": int(rows.size)},
        )
    if arm == M2:
        model = risk_leaf_refit(
            zero_shot_model, design5[fit_rows], utility[fit_rows], groups[fit_rows]
        )
        return Fitted(
            lambda r, m=model: m.score(design5[r]),
            bool(model.trees),
            {"trees": len(model.trees), "rows": int(fit_rows.size)},
        )
    if arm == B2:
        model = rk1.fit_lambdamart(design3[fit_rows], grades[fit_rows], groups[fit_rows])
        return Fitted(
            lambda r, m=model: m.score(design3[r]),
            bool(model.trees),
            {"trees": len(model.trees), "rows": int(fit_rows.size)},
        )
    if arm in (M3, M4):
        inputs = context["m3_inputs"] if arm == M3 else context["m4_inputs"]
        logistic = fit_logistic(inputs[fit_rows], harmful[fit_rows])
        if logistic is None:
            return Fitted(lambda r: np.full(r.size, np.nan), False, {"rows": int(fit_rows.size)})
        return Fitted(
            lambda r, m=logistic, x=inputs: 1.0 - m.harm(x[r]),
            True,
            {
                "rows": int(fit_rows.size),
                "standardized_coefficients": m_list(logistic.coefficients),
                "intercept": logistic.intercept,
            },
        )
    raise PhaseError(f"{arm} is not an adapted arm")


def m_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in values]


def arm_context(population: pd.DataFrame, matrix: pd.DataFrame, b: Blocks) -> dict[str, Any]:
    """Everything the arms of one direction read, built once."""
    design5 = rl1._matrix_for(population, matrix, rk3.columns_for(rk3.R5))
    design3 = rl1._matrix_for(population, matrix, rk3.columns_for(rk3.R3))
    grades = population["grade"].to_numpy(dtype=np.int64)
    utility = rk3.utility_of(grades, rk3.RISK_UTILITY)
    groups = population["document_id"].astype(str).to_numpy()
    zero_shot_model = rk3.fit_boosted(
        design5[b.source_fit], utility[b.source_fit], groups[b.source_fit], rk3.OBJ_RISK
    )
    zero_shot = zero_shot_model.score(design5)
    target_rows = np.concatenate([b.pool, b.test])
    page_mean = page_mean_scores(population, target_rows, zero_shot)
    return {
        "design5": design5,
        "design3": design3,
        "grades": grades,
        "utility": utility,
        "groups": groups,
        "harmful": population["is_harmful"].to_numpy(dtype=bool),
        "zero_shot_model": zero_shot_model,
        "zero_shot": zero_shot,
        "source_fit": b.source_fit,
        "m3_inputs": zero_shot[:, None],
        "m4_inputs": page_inputs(population, matrix, zero_shot, page_mean),
    }


# ------------------------------------------------------------------ reproduction gates

REPRODUCED_BUDGETS = rk3.REPRODUCED_BUDGETS


def run_reproduce() -> int:
    """DS1's score, RK1's ranker, RK2's arms and RK3's zero-shot ranker, rebuilt exactly."""
    started = time.monotonic()
    _require(PURCHASE_REGISTRY, "sample")
    _forbid(REPRODUCTION)
    populations = rk3.frames()
    matrix = pd.read_parquet(rk3.FEATURE_MATRIX)
    protocols = rk3.build_protocols(populations)
    frame = populations[rk3.ORDER_RK1]
    base = protocols[rk3.P_IN]
    out: dict[str, Any] = {}
    published_ds1 = pd.read_parquet(ds1.DEPLOYMENT_SCORES).set_index("candidate_id")
    rows, rank_score, safety, _trees = rk3.score_job(rk3.M_DS1, frame, matrix, base)
    ids = frame.iloc[rows]["candidate_id"].astype(str).to_numpy()
    count, rank_gap = rk3._max_difference(
        pd.Series(rank_score, index=ids), published_ds1["score_benefit"]
    )
    _count, safety_gap = rk3._max_difference(pd.Series(safety, index=ids), published_ds1["safety"])
    out["ds1_reliability_score"] = {
        "rows": count,
        "max_absolute_difference": max(rank_gap, safety_gap),
        "identical": bool(rank_gap == 0.0 and safety_gap == 0.0 and count == len(ids)),
    }
    rows, rank_score, _safety, _trees = rk3.score_job(rk3.M_RK1, frame, matrix, base)
    rk1_scores = pd.read_parquet(rk1.RANKING_SCORES)
    cell = rk1_scores[
        (rk1_scores["split"] == rk1.SPLIT_BLOCKS)
        & (rk1_scores["model"] == rk1.PRIMARY_RANKER)
        & (rk1_scores["variant"] == rk1.V_FULL)
    ].set_index("candidate_id")["rank_score"]
    count, gap = rk3._max_difference(pd.Series(rank_score, index=ids), cell)
    out["rk1_primary_ranker"] = {
        "rows": count,
        "max_absolute_difference": gap,
        "identical": bool(gap == 0.0 and count == len(ids) == len(cell)),
    }
    strict = populations[rk3.ORDER_RK2]
    design = rl1._matrix_for(strict, matrix, rk3.r3_columns())
    grades = strict["grade"].to_numpy(dtype=np.int64)
    groups = strict["document_id"].astype(str).to_numpy()
    calibration = rk2.calibration_mask(strict["site_group"].astype(str).tolist())
    strict_ids = strict["candidate_id"].astype(str).to_numpy()
    adapted = pd.read_parquet(rk2.ADAPTED_SCORES)
    cells: dict[str, Any] = {}

    def compare(
        direction: str, arm: str, budget: int, kind: str, index: np.ndarray, model: Any
    ) -> dict[str, Any]:
        stored = adapted[
            (adapted["direction"] == direction)
            & (adapted["arm"] == arm)
            & (adapted["budget"] == budget)
            & (adapted["draw"] == 0)
            & (adapted["evaluation_set"] == kind)
        ].set_index("candidate_id")["score"]
        rebuilt = pd.Series(model.score(design[index]), index=strict_ids[index])
        count, gap = rk3._max_difference(rebuilt, stored)
        return {
            "rows": count,
            "max_absolute_difference": gap,
            "identical": bool(gap == 0.0 and count == len(stored) == index.size),
        }

    for direction, blocks in rk2.all_blocks(strict).items():
        source_model = rk2.fit_arm(
            rk2.A0, design, grades, groups, blocks.source_fit, blocks.source_fit[:0], None
        )
        cells[f"{direction}|{rk2.A0}|0|test"] = compare(
            direction, rk2.A0, 0, "test", blocks.target_test, source_model
        )
        order = rl2.draw_order(strict, blocks.frozen, 0)
        for budget in REPRODUCED_BUDGETS:
            fit_rows, cal_rows = rk2.split_purchase(rl2.purchased(order, budget), calibration)
            model = rk2.fit_arm(
                rk2.A1, design, grades, groups, blocks.source_fit, fit_rows, source_model
            )
            cells[f"{direction}|{rk2.A1}|{budget}|test"] = compare(
                direction, rk2.A1, budget, "test", blocks.target_test, model
            )
            cells[f"{direction}|{rk2.A1}|{budget}|calibration"] = compare(
                direction, rk2.A1, budget, "calibration", cal_rows, model
            )
    out["rk2_arms"] = {"cells": cells, "identical": all(c["identical"] for c in cells.values())}
    population = load_population()
    rk3_scores = pd.read_parquet(rk3.RANKING_SCORES)
    zero_shot: dict[str, Any] = {}
    for direction, b in build_blocks(population).items():
        context = arm_context(population, matrix, b)
        published = rk3_scores[
            (rk3_scores["protocol"] == rk3.TDOC[direction])
            & (rk3_scores["model"] == rk3.M_RK3)
            & (rk3_scores["variant"] == rk3.V_FULL)
        ].set_index("candidate_id")["safety"]
        index = np.concatenate([b.source_calibration, b.test])
        rebuilt = pd.Series(
            context["zero_shot"][index],
            index=population.iloc[index]["candidate_id"].astype(str).to_numpy(),
        )
        count, gap = rk3._max_difference(rebuilt, published)
        zero_shot[direction] = {
            "rows": count,
            "max_absolute_difference": gap,
            "identical": bool(gap == 0.0 and count == len(published) == index.size),
        }
    out["rk3_zero_shot"] = {
        "directions": zero_shot,
        "identical": all(row["identical"] for row in zero_shot.values()),
    }
    identical = all(block["identical"] for block in out.values())
    _write_json_once(
        REPRODUCTION,
        {
            **_analysis_envelope("reproduction"),
            "by_baseline": out,
            "all_identical": identical,
            "note": (
                "DS1 and RK1 are refitted on DS1's fit block; RK2's zero-shot and target-only "
                "arms are refitted on RK2's own blocks and purchases; RK3's risk-aware ranker is "
                "refitted on its held-out protocol's source rows and compared on every cutoff and "
                "test row it scored"
            ),
        },
    )
    if not identical:
        raise PhaseError(
            f"a baseline did not reproduce: { {k: v['identical'] for k, v in out.items()} }"
        )
    print(
        f"reproduce: DS1, RK1, {len(cells)} RK2 cells and RK3's zero-shot ranker reproduce "
        f"exactly ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ adaptation


def run_adapt() -> int:
    """Every arm x direction x budget x draw: test, calibration and source-test scores."""
    started = time.monotonic()
    _require(REPRODUCTION, "reproduce")
    for path in (CELL_SCORES, ADAPTATION_REGISTRY):
        _forbid(path)
    population = load_population()
    matrix = pd.read_parquet(rk3.FEATURE_MATRIX)
    ids = population["candidate_id"].astype(str).to_numpy()
    frozen = {
        (row["direction"], row["budget"], row["draw"]): row
        for row in cc_read_json(PURCHASE_REGISTRY)["rows"]
    }
    sealed = {
        d: row["test_digest"] for d, row in cc_read_json(SPLIT_REGISTRY)["directions"].items()
    }
    frames: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []

    def emit(
        direction: str,
        arm: str,
        label: str,
        draw: int,
        kind: str,
        rows: np.ndarray,
        values: np.ndarray,
    ) -> None:
        frames.append(
            pd.DataFrame(
                {
                    "direction": direction,
                    "arm": arm,
                    "budget": label,
                    "draw": int(draw),
                    "evaluation_set": kind,
                    "candidate_id": ids[rows],
                    "score": values,
                }
            )
        )

    for direction, b in build_blocks(population).items():
        began = time.monotonic()
        if _digest(population, b.test) != sealed[direction]:
            raise PhaseError(f"{direction}: the test block differs from the sealed one")
        context = arm_context(population, matrix, b)
        zero_shot = context["zero_shot"]
        emit(direction, A0, "0", 0, "test", b.test, zero_shot[b.test])
        emit(
            direction,
            A0,
            "0",
            0,
            "calibration",
            b.source_calibration,
            zero_shot[b.source_calibration],
        )
        emit(direction, A0, "0", 0, "source_test", b.source_test, zero_shot[b.source_test])
        registry.append(
            {
                "direction": direction,
                "arm": A0,
                "budget": "0",
                "draw": 0,
                "trainable": bool(context["zero_shot_model"].trees),
                "calibration": "source",
                "fit_pages": 0,
                "calibration_pages": int(
                    population.iloc[b.source_calibration]["document_id"].nunique()
                ),
            }
        )
        for label in ADAPTED_BUDGETS:
            for draw in range(DRAWS):
                fit_pages, cal_pages = purchase(b.pages, draw, label)
                fit_rows = rows_on(population, b.pool, fit_pages)
                cal_rows = rows_on(population, b.pool, cal_pages)
                record = frozen[(direction, label, draw)]
                if (
                    _digest(population, fit_rows) != record["fit_digest"]
                    or _digest(population, cal_rows) != record["calibration_digest"]
                ):
                    raise PhaseError(f"{direction} {label} {draw}: not the frozen purchase")
                for arm in ADAPTED_ARMS:
                    fitted = fit_arm(arm, fit_rows, context)
                    test_scores = fitted.score(b.test)
                    trainable = fitted.trainable and not rk2._constant(test_scores)
                    emit(direction, arm, label, draw, "test", b.test, test_scores)
                    emit(
                        direction,
                        arm,
                        label,
                        draw,
                        "calibration",
                        cal_rows,
                        fitted.score(cal_rows) if cal_rows.size else np.empty(0),
                    )
                    if label == POOL and arm in TREE_ARMS:
                        emit(
                            direction,
                            arm,
                            label,
                            draw,
                            "source_test",
                            b.source_test,
                            fitted.score(b.source_test),
                        )
                    registry.append(
                        {
                            "direction": direction,
                            "arm": arm,
                            "budget": label,
                            "draw": draw,
                            "trainable": trainable,
                            "calibration": "target",
                            "fit_pages": len(fit_pages),
                            "calibration_pages": len(cal_pages),
                            "fit_rows": int(fit_rows.size),
                            "calibration_rows": int(cal_rows.size),
                            **fitted.detail,
                        }
                    )
        print(f"  {direction}: {time.monotonic() - began:.0f}s", flush=True)
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(
        ["direction", "arm", "budget", "draw", "evaluation_set", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    _write_parquet_once(CELL_SCORES, table)
    _write_json_once(
        ADAPTATION_REGISTRY,
        {
            **_analysis_envelope("adaptation_registry"),
            "arms": ARM_LABEL,
            "primary_arm": PRIMARY_ARM,
            "cells": registry,
            "fits": len(registry),
            "scored_rows": len(table),
            "untrainable": [
                f"{r['direction']}|{r['arm']}|{r['budget']}|{r['draw']}"
                for r in registry
                if not r["trainable"]
            ],
            "carries_source_columns": bool(
                [c for c in rk3.columns_for(rk3.R5) if c.startswith(rl1.FAM_SOURCE)]
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"adapt: {len(registry)} fits, {len(table)} scored rows ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ cutoff rules and outcomes

LABELS = (
    "site_group",
    "site_key",
    "document_id",
    "environment",
    "corrector_source",
    "corrector_sources",
    "is_harmful",
    "beneficial",
    "exact",
    "grade",
)


def choose_cutoff(
    rule: str, calibration: pd.DataFrame, epsilon: float, usable: bool
) -> tuple[float | None, float | None, float | None]:
    """A rule's cutoff on calibration decisions: (cutoff, claimed bound, design effect)."""
    if not usable or calibration.empty:
        return None, None, None
    if rule == R_NAIVE:
        return rk1.choose_threshold(calibration, epsilon), None, None
    if rule == R_CANDIDATE:
        cutoff, bound = th1.bound_threshold(calibration, epsilon, DELTA, 1.0)
        return cutoff, bound, None
    if rule == R_PAGE:
        deff = th1.page_design_effect(calibration)
        if deff is None:
            return None, None, None
        cutoff, bound = th1.bound_threshold(calibration, epsilon, DELTA, deff)
        return cutoff, bound, deff
    raise PhaseError(f"unknown rule {rule}")


def evaluate(
    rule: str,
    epsilon: float,
    test: pd.DataFrame,
    calibration: pd.DataFrame,
    usable: bool,
    keys: set[str],
    mapping: dict[str, set[str]],
) -> dict[str, Any]:
    """One rule's cutoff, chosen without test labels, read on the sealed test decisions."""
    cutoff, bound, deff = choose_cutoff(rule, calibration, epsilon, usable)
    accepted = test[test["safety"] >= cutoff] if cutoff is not None else test.head(0)
    row = rk2.policy_outcome(test, accepted, keys, mapping, epsilon)
    believed = (
        calibration[calibration["safety"] >= cutoff] if cutoff is not None else calibration.head(0)
    )
    calibration_harm = _share(float(believed["is_harmful"].sum()), len(believed))
    realized = row["selective_harm_rate"]
    oracle = rk2.oracle_accepted(test, epsilon) if usable else test.head(0)
    return {
        "rule": rule,
        "epsilon": epsilon,
        "usable": bool(usable),
        "cutoff": cutoff,
        "abstains": cutoff is None,
        "claimed_bound": bound,
        "design_effect": deff,
        "calibration_decisions": len(calibration),
        "calibration_pages": int(calibration["document_id"].nunique()) if len(calibration) else 0,
        "calibration_accepted": len(believed),
        "calibration_harm": calibration_harm,
        "accepted": row["accepted"],
        "sites": row["sites"],
        "coverage": row["coverage"],
        "review_rate": row["review_rate"],
        "selective_harm_rate": realized,
        "repair_recall": row["repair_recall"],
        "repaired_error_sites": row["repaired_error_sites"],
        "holds": row["holds_its_target"],
        "violation": bool(row["accepted"] > 0 and not row["holds_its_target"]),
        "working": row["working_safe_gate"],
        "harm_estimation_error": (
            None if realized is None or calibration_harm is None else realized - calibration_harm
        ),
        **th2.page_harm(accepted, epsilon),
        "oracle_repair_recall": rk2.policy_outcome(test, oracle, keys, mapping, epsilon)[
            "repair_recall"
        ],
    }


@dataclass(slots=True)
class Evaluation:
    """Everything an evaluation phase reads, loaded once."""

    population: pd.DataFrame
    labels: pd.DataFrame
    blocks: dict[str, Blocks]
    cells: dict[tuple[str, str, str, int], dict[str, pd.DataFrame]]
    registry: dict[tuple[str, str, str, int], dict[str, Any]]
    keys: set[str]
    mapping: dict[str, set[str]]


def load_evaluation() -> Evaluation:
    population = load_population()
    labels = population.set_index("candidate_id")[list(LABELS)]
    cells: dict[tuple[str, str, str, int], dict[str, pd.DataFrame]] = {}
    for key, group in pd.read_parquet(CELL_SCORES).groupby(
        ["direction", "arm", "budget", "draw", "evaluation_set"], sort=True
    ):
        direction, arm, budget, draw, kind = key
        joined = group[["candidate_id", "score"]].join(labels, on="candidate_id", how="inner")
        if len(joined) != len(group):
            raise PhaseError(f"{key}: a scored candidate has no label row")
        cells.setdefault((str(direction), str(arm), str(budget), int(draw)), {})[str(kind)] = (
            joined.assign(rank_score=joined["score"], safety=joined["score"]).reset_index(drop=True)
        )
    registry = {
        (r["direction"], r["arm"], r["budget"], r["draw"]): r
        for r in cc_read_json(ADAPTATION_REGISTRY)["cells"]
    }
    errors = ds1.load_error_population()
    return Evaluation(
        population=population,
        labels=labels,
        blocks=build_blocks(population),
        cells=cells,
        registry=registry,
        keys=set(errors[errors["block"] == "test"]["error_key"].astype(str)),
        mapping=ds1.site_to_errors(errors),
    )


def cell_key(direction: str, arm: str, budget: str, draw: int) -> tuple[str, str, str, int]:
    """Budget zero is the zero-shot ranker for every arm, and the zero-shot ranker is budget 0."""
    if budget == "0" or arm == A0:
        return direction, A0, "0", 0
    return direction, arm, budget, draw


def draws_at(budget: str) -> range:
    return range(1) if budget == "0" else range(DRAWS)


def cell_outcomes(
    evaluation: Evaluation, key: tuple[str, str, str, int], rules: Sequence[str] = RULES
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One cell's ranking metrics and every rule's outcome at every harm target."""
    cell = evaluation.cells[key]
    test = cell["test"]
    usable = bool(evaluation.registry[key]["trainable"])
    metrics = rk1.ranking_metrics(test) if usable else {"rows": len(test)}
    ranking = {
        "harm_auroc": metrics.get("harm_auroc"),
        "benefit_auroc": metrics.get("benefit_auroc"),
        "top1": metrics.get("top1"),
        "choice_sites": metrics.get("choice_sites", 0),
        "trainable": usable,
        "retention_harm_auroc": (
            harm_auroc_of(cell["source_test"]) if "source_test" in cell and usable else None
        ),
    }
    tested = decisions(test)
    calibration = decisions(cell.get("calibration", test.head(0)))
    policies = [
        evaluate(rule, epsilon, tested, calibration, usable, evaluation.keys, evaluation.mapping)
        for rule in rules
        for epsilon in EPSILONS
    ]
    return ranking, policies


def decisions(frame: pd.DataFrame) -> pd.DataFrame:
    return rk2.decisions(frame)


def harm_auroc_of(frame: pd.DataFrame) -> float:
    return auroc(-frame["safety"].to_numpy(np.float64), frame["is_harmful"].to_numpy(bool))


def run_policies() -> int:
    """Every cell's ranking metrics and every cutoff rule's test outcome."""
    started = time.monotonic()
    _require(ADAPTATION_REGISTRY, "adapt")
    for path in (POLICY_OUTCOMES, RANKING_OUTCOMES, POLICY_REGISTRY):
        _forbid(path)
    evaluation = load_evaluation()
    ranking_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for key in sorted(evaluation.cells):
        direction, arm, budget, draw = key
        base = {"direction": direction, "arm": arm, "budget": budget, "draw": draw}
        ranking, policies = cell_outcomes(evaluation, key)
        ranking_rows.append({**base, **ranking})
        policy_rows.extend({**base, **row} for row in policies)
    ranking_table = pd.DataFrame(ranking_rows).sort_values(
        ["direction", "arm", "budget", "draw"], kind="stable"
    )
    policy_table = pd.DataFrame(policy_rows).sort_values(
        ["direction", "arm", "budget", "draw", "rule", "epsilon"], kind="stable"
    )
    _write_parquet_once(RANKING_OUTCOMES, ranking_table.reset_index(drop=True))
    _write_parquet_once(POLICY_OUTCOMES, policy_table.reset_index(drop=True))
    _write_json_once(
        POLICY_REGISTRY,
        {
            **_analysis_envelope("policy_registry"),
            "rules": RULE_LABEL,
            "primary_rule": PRIMARY_RULE,
            "cells": len(ranking_table),
            "policy_rows": len(policy_table),
            "epsilons": list(EPSILONS),
            "delta": DELTA,
            "recall_denominator": len(evaluation.keys),
            "cutoffs_chosen_without_test_labels": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"policies: {len(ranking_table)} cells, {len(policy_table)} rule outcomes "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ curves and frontier


def _values(frame: pd.DataFrame, column: str) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in frame[column].tolist()]


def _median(values: Sequence[Any]) -> float | None:
    finite = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.median(finite)) if finite else None


def ranking_rows(table: pd.DataFrame, direction: str, arm: str, budget: str) -> pd.DataFrame:
    key = cell_key(direction, arm, budget, 0)
    return table[
        (table["direction"] == direction) & (table["arm"] == key[1]) & (table["budget"] == key[2])
    ]


def policy_rows(
    table: pd.DataFrame, direction: str, arm: str, budget: str, rule: str, epsilon: float
) -> pd.DataFrame:
    key = cell_key(direction, arm, budget, 0)
    return table[
        (table["direction"] == direction)
        & (table["arm"] == key[1])
        & (table["budget"] == key[2])
        & (table["rule"] == rule)
        & (table["epsilon"] == epsilon)
    ]


def policy_summary(rows: pd.DataFrame) -> dict[str, Any]:
    accepting = rows[rows["accepted"] > 0]
    return {
        "draws": len(rows),
        "violation_share": float(rows["violation"].mean()) if len(rows) else None,
        "working_share": float(rows["working"].mean()) if len(rows) else None,
        "abstain_share": float(rows["abstains"].mean()) if len(rows) else None,
        "accepting_draws": len(accepting),
        "median_recall": _median(_values(rows, "repair_recall")),
        "median_coverage": _median(_values(rows, "coverage")),
        "median_review_rate": _median(_values(rows, "review_rate")),
        "median_selective_harm": _median(_values(accepting, "selective_harm_rate")),
        "median_calibration_harm": _median(_values(accepting, "calibration_harm")),
        "median_harm_estimation_error": _median(_values(accepting, "harm_estimation_error")),
        "median_design_effect": _median(_values(rows, "design_effect")),
        "median_worst_page_harm": _median(_values(accepting, "worst_page_harm")),
        "median_oracle_recall": _median(_values(rows, "oracle_repair_recall")),
    }


def run_curves() -> int:
    """Label-efficiency curves: medians over draws at every measured page budget."""
    started = time.monotonic()
    _require(POLICY_REGISTRY, "policies")
    _forbid(CURVES)
    ranking = pd.read_parquet(RANKING_OUTCOMES)
    policies = pd.read_parquet(POLICY_OUTCOMES)
    ranking_curves: dict[str, Any] = {}
    policy_curves: dict[str, Any] = {}
    for direction in DIRECTIONS:
        for arm in ARMS:
            for budget in BUDGETS:
                if arm == A0 and budget != "0":
                    continue
                rows = ranking_rows(ranking, direction, arm, budget)
                ranking_curves.setdefault(direction, {}).setdefault(arm, {})[budget] = {
                    "harm_auroc": rk2._summary(_values(rows, "harm_auroc")),
                    "benefit_auroc": rk2._summary(_values(rows, "benefit_auroc")),
                    "top1": rk2._summary(_values(rows, "top1")),
                    "retention_harm_auroc": rk2._summary(_values(rows, "retention_harm_auroc")),
                    "draws_at_or_above_the_floor": int(
                        sum(
                            1
                            for v in _values(rows, "harm_auroc")
                            if v is not None and v >= HARM_FLOOR
                        )
                    ),
                    "untrainable_draws": int((~rows["trainable"].astype(bool)).sum()),
                }
                for rule in RULES:
                    for epsilon in EPSILONS:
                        summary = policy_summary(
                            policy_rows(policies, direction, arm, budget, rule, epsilon)
                        )
                        policy_curves.setdefault(direction, {}).setdefault(arm, {}).setdefault(
                            rule, {}
                        ).setdefault(rl1.epsilon_key(epsilon), {})[budget] = summary
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    pool_m1 = {d: ranking_curves[d][PRIMARY_ARM][POOL]["harm_auroc"]["median"] for d in DIRECTIONS}
    _write_json_once(
        CURVES,
        {
            **_analysis_envelope("label_efficiency_curves"),
            "ranking": ranking_curves,
            "policies": policy_curves,
            "budgets": list(BUDGETS),
            "pool_pages": {
                d: row["pool_pages"]
                for d, row in cc_read_json(SPLIT_REGISTRY)["directions"].items()
            },
            "primary_epsilon_key": key,
            "zero_budget_note": "budget 0 is the zero-shot ranker for every arm",
            "median_note": "a median resting on fewer than half the draws is not reported",
        },
    )
    print(
        "curves: M1 median harm AUROC at the pool "
        + ", ".join(f"{d} {v}" for d, v in pool_m1.items())
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


def run_frontier() -> int:
    """The deployable frontier, N* per arm and rule, and the ranking-ready budgets."""
    started = time.monotonic()
    _require(CURVES, "curves")
    _forbid(FRONTIER)
    curves = cc_read_json(CURVES)
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    table: dict[str, Any] = {}
    n_star: dict[str, Any] = {}
    ready: dict[str, Any] = {}
    for direction in DIRECTIONS:
        for arm in ADAPTED_ARMS:
            for rule in RULES:
                flags = {}
                for budget in BUDGETS:
                    source_arm = A0 if budget == "0" else arm
                    row = curves["policies"][direction][source_arm][rule][key][budget]
                    flags[budget] = deployable(row["violation_share"], row["working_share"])
                    table.setdefault(direction, {}).setdefault(arm, {}).setdefault(rule, {})[
                        budget
                    ] = {
                        "violation_share": row["violation_share"],
                        "working_share": row["working_share"],
                        "deployable": flags[budget],
                    }
                n_star.setdefault(direction, {}).setdefault(arm, {})[rule] = minimal_budget(flags)
            harm = {
                budget: curves["ranking"][direction][A0 if budget == "0" else arm][budget][
                    "harm_auroc"
                ]["median"]
                for budget in BUDGETS
            }
            ready.setdefault(direction, {})[arm] = minimal_budget(
                {b: v is not None and v >= HARM_FLOOR for b, v in harm.items()}
            )
    _write_json_once(
        FRONTIER,
        {
            **_analysis_envelope("deployable_frontier"),
            "table": table,
            "n_star": n_star,
            "ranking_ready": ready,
            "primary": {
                "arm": PRIMARY_ARM,
                "rule": PRIMARY_RULE,
                "n_star": {d: n_star[d][PRIMARY_ARM][PRIMARY_RULE] for d in DIRECTIONS},
                "ranking_ready": {d: ready[d][PRIMARY_ARM] for d in DIRECTIONS},
            },
            "definition": cc_read_json(DESIGN_RECORD)["deployable"],
            "measured_budgets": list(BUDGETS),
            "not_measurable": list(BRIEF_BUDGETS_NOT_MEASURABLE),
        },
    )
    primary_n_star = {d: n_star[d][PRIMARY_ARM][PRIMARY_RULE] for d in DIRECTIONS}
    print(
        f"frontier: primary N* {primary_n_star}, "
        f"ranking-ready { ({d: ready[d][PRIMARY_ARM] for d in DIRECTIONS}) } "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ transfer and frontier curves

MATRIX_ARMS = (A0, M1, M2, B2)
SWEEP_ARMS = (A0, M1, M2, M4, B2)
SWEEP_BUDGETS = ("10", POOL)


def run_transfer() -> int:
    """The generator transfer matrix and the ANALYSIS-ONLY harm-recall frontier over draws."""
    started = time.monotonic()
    _require(FRONTIER, "frontier")
    _forbid(TRANSFER)
    curves = cc_read_json(CURVES)["ranking"]
    frozen = cc_read_json(FROZEN_CONFIGURATION)
    matrix: dict[str, Any] = {}
    for arm in MATRIX_ARMS:
        budget = "0" if arm == A0 else POOL
        for direction in DIRECTIONS:
            row = curves[direction][arm][budget]
            source, target = (
                rk3.GENERATOR_SHORT[SOURCE_OF[direction]],
                rk3.GENERATOR_SHORT[TARGET_OF[direction]],
            )
            matrix.setdefault(arm, {})[f"adapted_to_{target}|eval_{target}"] = {
                "median_harm_auroc": row["harm_auroc"]["median"],
                "median_benefit_auroc": row["benefit_auroc"]["median"],
                "draws": row["harm_auroc"]["draws"],
            }
            matrix[arm][f"adapted_to_{target}|eval_{source}"] = {
                "median_harm_auroc": row["retention_harm_auroc"]["median"],
                "draws": row["retention_harm_auroc"]["draws"],
            }
    evaluation = load_evaluation()
    sweeps: dict[str, Any] = {}
    for direction in DIRECTIONS:
        for arm in SWEEP_ARMS:
            for budget in ("0",) if arm == A0 else SWEEP_BUDGETS:
                frontiers = []
                for draw in draws_at(budget):
                    key = cell_key(direction, arm, budget, draw)
                    if not evaluation.registry[key]["trainable"]:
                        continue
                    test = evaluation.cells[key]["test"]
                    points = rk3.sweep(decisions(test), evaluation.keys, evaluation.mapping)
                    frontiers.append(rk3.curve_summary(points)["frontier"])
                sweeps.setdefault(direction, {}).setdefault(arm, {})[budget] = {
                    "draws": len(frontiers),
                    "median_frontier": [
                        {
                            "harm": level,
                            "recall": _median([f[i]["recall"] for f in frontiers]),
                        }
                        for i, level in enumerate(rk3.HARM_GRID)
                    ],
                }
    _write_json_once(
        TRANSFER,
        {
            **_analysis_envelope("generator_transfer"),
            "matrix": matrix,
            "matrix_note": (
                "each arm is adapted to one generator; it is read on that generator's sealed test "
                "rows and on the source generator's test rows, which no arm trains on. Adapted "
                "arms "
                "are at the pool, medians over draws; the zero-shot ranker is budget 0"
            ),
            "rk3_in_distribution_reference": frozen["rk3_in_distribution_reference"],
            "rk3_held_out_reference": frozen["rk3_held_out_harm_auroc"],
            "frontier": sweeps,
            "frontier_note": (
                "ANALYSIS ONLY: each draw's test decisions swept over every distinct cutoff; the "
                "highest recall at each realized harm, median over draws. No criterion reads it"
            ),
            "harm_grid": list(rk3.HARM_GRID),
        },
    )
    print(f"transfer: matrix for {len(matrix)} arms ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ page calibration effect


def run_calibration() -> int:
    """RQ2: what page-level cutoffs and the page-risk term change, budget by budget."""
    started = time.monotonic()
    _require(TRANSFER, "transfer")
    _forbid(CALIBRATION_EFFECT)
    curves = cc_read_json(CURVES)
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    registry = cc_read_json(ADAPTATION_REGISTRY)["cells"]
    rules = {
        direction: {
            rule: {
                budget: curves["policies"][direction][A0 if budget == "0" else PRIMARY_ARM][rule][
                    key
                ][budget]
                for budget in BUDGETS
            }
            for rule in RULES
        }
        for direction in DIRECTIONS
    }
    page_terms = {}
    for direction in DIRECTIONS:
        for arm in (M3, M4):
            for budget in ADAPTED_BUDGETS:
                cells = [
                    r
                    for r in registry
                    if r["direction"] == direction
                    and r["arm"] == arm
                    and r["budget"] == budget
                    and r.get("standardized_coefficients")
                ]
                coefficients = np.asarray(
                    [r["standardized_coefficients"] for r in cells], dtype=np.float64
                )
                page_terms.setdefault(direction, {}).setdefault(arm, {})[budget] = {
                    "fits": len(cells),
                    "median_coefficients": (
                        [float(v) for v in np.median(coefficients, axis=0)] if len(cells) else None
                    ),
                    "fits_reversing_the_zero_shot_ranking": int((coefficients[:, 0] > 0).sum())
                    if len(cells)
                    else 0,
                }
    inputs = ["zero_shot_score", PAGE_MEAN_SCORE, *PAGE_FEATURES]
    m4_vs_m3 = {
        direction: {
            budget: {
                arm: {
                    "median_harm_auroc": curves["ranking"][direction][arm][budget]["harm_auroc"][
                        "median"
                    ],
                    **{
                        rule: {
                            k: curves["policies"][direction][arm][rule][key][budget][k]
                            for k in (
                                "violation_share",
                                "working_share",
                                "median_recall",
                                "median_worst_page_harm",
                            )
                        }
                        for rule in RULES
                    },
                }
                for arm in (M3, M4)
            }
            for budget in ADAPTED_BUDGETS
        }
        for direction in DIRECTIONS
    }
    _write_json_once(
        CALIBRATION_EFFECT,
        {
            **_analysis_envelope("page_calibration_effect"),
            "primary_arm_by_rule": rules,
            "m4_vs_m3": m4_vs_m3,
            "logistic_terms": page_terms,
            "logistic_inputs": {M3: inputs[:1], M4: inputs},
            "slope_note": (
                "the coefficients are on the predicted probability of harm. A positive score "
                "coefficient means a higher zero-shot score predicts more harm, which reverses the "
                "zero-shot ranking; such fits are counted"
            ),
        },
    )
    print(f"calibration: rules and page terms recorded ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ ablations


def run_ablate() -> int:
    """Feature-family ablations of the primary arm at the pool, over the first draws."""
    started = time.monotonic()
    _require(CALIBRATION_EFFECT, "calibration")
    _forbid(ABLATION)
    population = load_population()
    matrix = pd.read_parquet(rk3.FEATURE_MATRIX)
    errors = ds1.load_error_population()
    keys = set(errors[errors["block"] == "test"]["error_key"].astype(str))
    mapping = ds1.site_to_errors(errors)
    labels = population.set_index("candidate_id")[list(LABELS)]
    variants = {name: drop for name, drop in rk3.ABLATIONS.items() if name.startswith("minus_new_")}
    variants["r3_only"] = rk3.NEW_FAMILIES
    grades = population["grade"].to_numpy(dtype=np.int64)
    utility = rk3.utility_of(grades, rk3.RISK_UTILITY)
    groups = population["document_id"].astype(str).to_numpy()
    ids = population["candidate_id"].astype(str).to_numpy()
    out: dict[str, Any] = {}
    for direction, b in build_blocks(population).items():
        for name, drop in {"full": (), **variants}.items():
            design = rl1._matrix_for(population, matrix, rk3.columns_for(rk3.R5, drop))
            rows_out = []
            for draw in range(ABLATION_DRAWS):
                fit_pages, cal_pages = purchase(b.pages, draw, POOL)
                fit_rows = rows_on(population, b.pool, fit_pages)
                cal_rows = rows_on(population, b.pool, cal_pages)
                rows = np.concatenate([b.source_fit, fit_rows])
                model = rk3.fit_boosted(design[rows], utility[rows], groups[rows], rk3.OBJ_RISK)

                def frame(
                    index: np.ndarray, m: Any = model, x: np.ndarray = design
                ) -> pd.DataFrame:
                    score = m.score(x[index])
                    base = labels.loc[ids[index]].reset_index()
                    return base.assign(rank_score=score, safety=score)

                test, calibration = frame(b.test), frame(cal_rows)
                outcomes = {
                    rule: evaluate(
                        rule,
                        PRIMARY_EPSILON,
                        decisions(test),
                        decisions(calibration),
                        True,
                        keys,
                        mapping,
                    )
                    for rule in (R_NAIVE, R_PAGE)
                }
                rows_out.append(
                    {
                        "harm_auroc": harm_auroc_of(test),
                        "naive_recall": outcomes[R_NAIVE]["repair_recall"],
                        "naive_violation": outcomes[R_NAIVE]["violation"],
                        "page_recall": outcomes[R_PAGE]["repair_recall"],
                        "page_working": outcomes[R_PAGE]["working"],
                    }
                )
            out.setdefault(direction, {})[name] = {
                "median_harm_auroc": _median([r["harm_auroc"] for r in rows_out]),
                "median_naive_recall": _median([r["naive_recall"] for r in rows_out]),
                "naive_violation_share": float(np.mean([r["naive_violation"] for r in rows_out])),
                "median_page_recall": _median([r["page_recall"] for r in rows_out]),
                "page_working_share": float(np.mean([r["page_working"] for r in rows_out])),
                "columns": len(rk3.columns_for(rk3.R5, drop)),
            }
        full = out[direction]["full"]
        for name in variants:
            row = out[direction][name]
            row["delta_harm_auroc"] = rk1._delta(
                row["median_harm_auroc"], full["median_harm_auroc"]
            )
            row["delta_naive_recall"] = rk1._delta(
                row["median_naive_recall"], full["median_naive_recall"]
            )
    _write_json_once(
        ABLATION,
        {
            **_analysis_envelope("feature_ablation"),
            "arm": PRIMARY_ARM,
            "budget": POOL,
            "draws": ABLATION_DRAWS,
            "by_direction": out,
            "variants": {name: list(drop) for name, drop in variants.items()},
            "descriptive": "no ablation is in the statistical family",
        },
    )
    print(
        f"ablate: {len(variants)} variants x {len(out)} directions "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ controls


def shuffled_within_pages(
    grades: np.ndarray, documents: np.ndarray, rows: np.ndarray, draw: int
) -> np.ndarray:
    """Grades of `rows` permuted inside each page, seeded per draw and page; others untouched."""
    out = grades.copy()
    for page in sorted(set(documents[rows].tolist())):
        index = rows[documents[rows] == page]
        generator = np.random.default_rng(
            int(canonical_hash({"seed": PERMUTATION_SEED, "draw": draw, "page": page})[:8], 16)
        )
        out[index] = grades[index][generator.permutation(index.size)]
    return out


def run_controls() -> int:
    """Target-grade permutation for RK2's target-only arm and M1, and random scores."""
    started = time.monotonic()
    _require(ABLATION, "ablate")
    _forbid(CONTROL_RESULTS)
    population = load_population()
    matrix = pd.read_parquet(rk3.FEATURE_MATRIX)
    documents = population["document_id"].astype(str).to_numpy()
    harmful = population["is_harmful"].to_numpy(dtype=bool)
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = documents
    permutation: dict[str, Any] = {}
    evaluation = load_evaluation()
    random_rows: list[dict[str, Any]] = []
    for direction, b in build_blocks(population).items():
        context = arm_context(population, matrix, b)
        values: dict[str, list[float]] = {B2: [], M1: []}
        for draw in range(DRAWS):
            fit_pages, _calibration_pages = purchase(b.pages, draw, POOL)
            fit_rows = rows_on(population, b.pool, fit_pages)
            permuted = shuffled_within_pages(grades, documents, fit_rows, draw)
            b2 = rk1.fit_lambdamart(
                context["design3"][fit_rows], permuted[fit_rows], groups[fit_rows]
            )
            values[B2].append(auroc(-b2.score(context["design3"][b.test]), harmful[b.test]))
            rows = np.concatenate([b.source_fit, fit_rows])
            m1 = rk3.fit_boosted(
                context["design5"][rows],
                rk3.utility_of(permuted[rows], rk3.RISK_UTILITY),
                groups[rows],
                rk3.OBJ_RISK,
            )
            values[M1].append(auroc(-m1.score(context["design5"][b.test]), harmful[b.test]))
            key = cell_key(direction, M1, POOL, draw)
            generator = np.random.default_rng(
                int(canonical_hash({"seed": PERMUTATION_SEED, "random": key})[:8], 16)
            )
            test = evaluation.cells[key]["test"]
            calibration = evaluation.cells[key]["calibration"]
            test = test.assign(rank_score=generator.random(len(test)))
            test = test.assign(safety=test["rank_score"])
            calibration = calibration.assign(rank_score=generator.random(len(calibration)))
            calibration = calibration.assign(safety=calibration["rank_score"])
            for rule in RULES:
                random_rows.append(
                    evaluate(
                        rule,
                        PRIMARY_EPSILON,
                        decisions(test),
                        decisions(calibration),
                        True,
                        evaluation.keys,
                        evaluation.mapping,
                    )
                    | {"direction": direction, "draw": draw}
                )
        permutation[direction] = {
            arm: {"mean_harm_auroc": float(np.mean(v)), "values": [float(x) for x in v]}
            for arm, v in values.items()
        }
    b2_mean = float(np.mean([v for d in permutation.values() for v in d[B2]["values"]]))
    random_table = pd.DataFrame(random_rows)
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "permutation": {
                "by_direction": permutation,
                "b2_mean_harm_auroc": b2_mean,
                "band": list(rk3.PERMUTATION_BAND),
                "b2_inside_the_band": bool(
                    rk3.PERMUTATION_BAND[0] <= b2_mean <= rk3.PERMUTATION_BAND[1]
                ),
                "rule": cc_read_json(DESIGN_RECORD)["controls"],
            },
            "random": {
                rule: {
                    "violation_share": float(part["violation"].mean()),
                    "working_share": float(part["working"].mean()),
                    "abstain_share": float(part["abstains"].mean()),
                }
                for rule, part in random_table.groupby("rule", sort=True)
            },
        },
    )
    print(
        f"controls: permuted target-only harm AUROC {b2_mean:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ statistics


def paired_values(
    ranking: pd.DataFrame, policies: pd.DataFrame, spec: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two arms' (or rules') values on the same draws of both directions, with each direction."""
    a: list[float] = []
    b: list[float] = []
    direction_of: list[str] = []
    budget = spec["budget"]
    for direction in DIRECTIONS:
        for draw in range(DRAWS):
            if spec["metric"] == "violation":
                rows = policies[
                    (policies["direction"] == direction)
                    & (policies["arm"] == spec["arm"])
                    & (policies["budget"] == budget)
                    & (policies["draw"] == draw)
                    & (policies["epsilon"] == PRIMARY_EPSILON)
                ].set_index("rule")["violation"]
                left, right = float(rows[spec["left"]]), float(rows[spec["right"]])
            else:
                values = []
                for arm in (spec["left"], spec["right"]):
                    key = cell_key(direction, arm, budget, draw)
                    cell = ranking[
                        (ranking["direction"] == key[0])
                        & (ranking["arm"] == key[1])
                        & (ranking["budget"] == key[2])
                        & (ranking["draw"] == key[3])
                    ][spec["metric"]]
                    values.append(
                        float(cell.iloc[0]) if len(cell) and not pd.isna(cell.iloc[0]) else np.nan
                    )
                left, right = values
            if np.isfinite(left) and np.isfinite(right):
                a.append(left)
                b.append(right)
                direction_of.append(direction)
    return np.asarray(a), np.asarray(b), np.asarray(direction_of)


def draw_comparison(
    name: str, spec: dict[str, str], ranking: pd.DataFrame, policies: pd.DataFrame
) -> dict[str, Any]:
    a, b, direction = paired_values(ranking, policies, spec)
    violation = spec["metric"] == "violation"

    def effect(left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0:
            return float("nan")
        return float(left.mean() - right.mean()) if violation else float(np.median(left - right))

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero(direction == d) for d in sorted(set(direction.tolist()))]
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for resample in range(BOOTSTRAP_RESAMPLES):
        index = (
            np.concatenate([generator.choice(s, s.size, replace=True) for s in strata])
            if strata
            else np.empty(0, dtype=np.int64)
        )
        draws[resample] = effect(a[index], b[index])
    p, positive, negative = th1.sign_test(a - b)
    finite = draws[np.isfinite(draws)]
    return {
        "comparison": name,
        **spec,
        "units": int(a.size),
        "effect": effect(a, b) if a.size else None,
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else None,
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else None,
        "p_value": p,
        "draws_left_higher": positive,
        "draws_right_higher": negative,
        "ties": int(a.size - positive - negative),
        "left_value": (float(a.mean()) if violation else float(np.median(a))) if a.size else None,
        "right_value": (float(b.mean()) if violation else float(np.median(b))) if b.size else None,
    }


def run_stats() -> int:
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    ranking = pd.read_parquet(RANKING_OUTCOMES)
    policies = pd.read_parquet(POLICY_OUTCOMES)
    family = {
        name: draw_comparison(name, FAMILY_SPEC[name], ranking, policies) for name in PRIMARY_FAMILY
    }
    adjusted = s15.holm(family)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "family": adjusted,
            "family_size": len(adjusted),
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "plan": cc_read_json(DESIGN_RECORD)["statistical_plan"],
        },
    )
    print(
        f"stats: {sum(1 for v in adjusted.values() if v['survives_holm'])}/{len(adjusted)} "
        f"survive Holm ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ falsification

PURCHASE_FUNCTIONS = ("page_order", "purchase", "rows_on", "page_mean_scores", "build_blocks")


def _module_tree() -> ast.Module:
    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def _functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _mentions(function_name: str) -> set[str]:
    """Every name, attribute and string a function mentions, following calls in this module."""
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
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
    return names


def _cutoff_first_arguments() -> list[str]:
    """What `evaluate` hands to `choose_cutoff` as the decisions a cutoff is chosen on."""
    found: list[str] = []
    for node in ast.walk(_functions()["evaluate"]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "choose_cutoff"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
        ):
            found.append(node.args[1].id)
    return found


def run_negative() -> int:
    """Every claim the stage makes, with a test that would fail if the claim were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    tests: list[dict[str, Any]] = []

    def record(name: str, claim: str, passed: bool, evidence: Any) -> None:
        tests.append({"test": name, "claim": claim, "passed": bool(passed), "evidence": evidence})

    population = load_population()
    blocks = build_blocks(population)
    splits = cc_read_json(SPLIT_REGISTRY)
    purchases = cc_read_json(PURCHASE_REGISTRY)["rows"]
    reproduction = cc_read_json(REPRODUCTION)["by_baseline"]
    registry = cc_read_json(ADAPTATION_REGISTRY)
    policies = pd.read_parquet(POLICY_OUTCOMES)
    scores = pd.read_parquet(CELL_SCORES)
    matrix = pd.read_parquet(rk3.FEATURE_MATRIX)
    label_fields = rk3.LABEL_FIELDS

    columns = [*rk3.columns_for(rk3.R5), *PAGE_FEATURES]
    record(
        "t01_no_generator_identity_in_any_model",
        "no design column or page feature names a generator, and no fit carries a source column",
        not [c for c in columns if c.startswith(rl1.FAM_SOURCE) or "corrector" in c]
        and not registry["carries_source_columns"],
        {"columns": len(columns)},
    )
    pool_pages = {d: set(b.pages) for d, b in blocks.items()}
    test_pages = {
        d: set(population.iloc[b.test]["document_id"].astype(str)) for d, b in blocks.items()
    }
    bought_outside = [
        r
        for r in purchases
        if not set(r["fit_pages"]) | set(r["calibration_pages"]) <= pool_pages[r["direction"]]
    ]
    record(
        "t02_no_test_page_is_ever_bought",
        "every purchased page is a pool page, and no pool page is a test page",
        not bought_outside and all(not (pool_pages[d] & test_pages[d]) for d in DIRECTIONS),
        {"purchases_outside_the_pool": len(bought_outside)},
    )
    record(
        "t03_blocks_are_rk3s_and_isolated",
        "the source and test blocks are RK3's held-out protocol, and nothing crosses into test",
        splits["isolated"]
        and all(row["equals_rk3_held_out_protocol"] for row in splits["directions"].values()),
        {d: row["isolation"] for d, row in splits["directions"].items()},
    )
    by_draw: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in purchases:
        by_draw.setdefault((r["direction"], r["draw"]), []).append(r)
    nested = True
    disjoint = True
    for rows in by_draw.values():
        ordered = sorted(rows, key=lambda r: r["pages"])
        for small, large in itertools.pairwise(ordered):
            nested &= set(small["fit_pages"]) <= set(large["fit_pages"])
            nested &= set(small["calibration_pages"]) <= set(large["calibration_pages"])
        disjoint &= all(not (set(r["fit_pages"]) & set(r["calibration_pages"])) for r in rows)
    record(
        "t04_budgets_nest_and_fit_and_calibration_pages_never_share",
        "within a draw every budget's fit and calibration pages contain the smaller budget's, and "
        "no page both fits and calibrates",
        nested and disjoint,
        {"draw_sequences": len(by_draw)},
    )
    mentioned = set().union(*(_mentions(f) for f in PURCHASE_FUNCTIONS))
    rebuilt = all(
        purchase(blocks[r["direction"]].pages, r["draw"], r["budget"])
        == (r["fit_pages"], r["calibration_pages"])
        for r in purchases
    )
    record(
        "t05_purchases_read_no_label",
        "the purchase, page-order and page-mean functions mention no label field, and every "
        "purchase is rebuilt from page ids alone",
        not (mentioned & label_fields) and rebuilt,
        {"label_fields_mentioned": sorted(mentioned & label_fields)},
    )
    for name, key in (
        ("t06_ds1_reproduces", "ds1_reliability_score"),
        ("t07_rk1_reproduces", "rk1_primary_ranker"),
        ("t08_rk2_reproduces", "rk2_arms"),
        ("t09_rk3_zero_shot_reproduces", "rk3_zero_shot"),
    ):
        record(
            name,
            f"{key} is rebuilt exactly before any adaptation",
            reproduction[key]["identical"],
            {k: v for k, v in reproduction[key].items() if k not in ("cells", "directions")},
        )
    evaluation = load_evaluation()
    sample = [cell_key(d, M1, POOL, 0) for d in DIRECTIONS]
    flipped_same = True
    for key in sample:
        cell = evaluation.cells[key]
        test = decisions(cell["test"])
        calibration = decisions(cell["calibration"])
        inverted = test.assign(is_harmful=~test["is_harmful"], exact=~test["exact"])
        for rule in RULES:
            honest = evaluate(
                rule, PRIMARY_EPSILON, test, calibration, True, evaluation.keys, evaluation.mapping
            )
            flipped = evaluate(
                rule,
                PRIMARY_EPSILON,
                inverted,
                calibration,
                True,
                evaluation.keys,
                evaluation.mapping,
            )
            flipped_same &= honest["cutoff"] == flipped["cutoff"]
    record(
        "t10_cutoffs_read_calibration_pages_only",
        "cutoffs are chosen on the calibration decisions only, and inverting every test label "
        "leaves every cutoff unchanged",
        _cutoff_first_arguments() == ["calibration"] and flipped_same,
        {"choose_cutoff_arguments": _cutoff_first_arguments()},
    )
    m1_rows = [r for r in registry["cells"] if r["arm"] == M1]
    source_rows = {d: int(b.source_fit.size) for d, b in blocks.items()}
    record(
        "t11_only_m1_uses_source_rows",
        "M1 trains on the source fit rows plus the target fit pages; every other adapted arm "
        "trains on the target fit pages alone",
        all(r["rows"] == source_rows[r["direction"]] + r["fit_rows"] for r in m1_rows)
        and all(
            r["rows"] == r["fit_rows"] for r in registry["cells"] if r["arm"] in (M2, M3, M4, B2)
        ),
        {"m1_cells": len(m1_rows)},
    )
    direction = DIRECTIONS[0]
    context = arm_context(population, matrix, blocks[direction])
    b = blocks[direction]
    zero = context["zero_shot_model"]
    again = risk_leaf_refit(
        zero,
        context["design5"][b.source_fit],
        context["utility"][b.source_fit],
        context["groups"][b.source_fit],
    )
    record(
        "t12_leaf_adaptation_reproduces_the_source_on_its_own_rows",
        "re-estimating the zero-shot ranker's leaves on its own training rows returns it exactly, "
        "split for split",
        bool(
            np.array_equal(
                again.score(context["design5"][b.test]), zero.score(context["design5"][b.test])
            )
            and all(t1 is t2 for t1, t2 in zip(again.trees, zero.trees, strict=True))
        ),
        {"trees": len(zero.trees)},
    )
    monotone = True
    checked = 0
    for d in DIRECTIONS:
        zero_test = evaluation.cells[cell_key(d, A0, "0", 0)]["test"].set_index("candidate_id")[
            "score"
        ]
        for draw in range(DRAWS):
            key = cell_key(d, M3, POOL, draw)
            if not evaluation.registry[key]["trainable"]:
                continue
            m3 = evaluation.cells[key]["test"].set_index("candidate_id")["score"]
            pair = pd.DataFrame({"z": zero_test, "m": m3}).dropna()
            order_z = np.argsort(pair["z"].to_numpy(), kind="mergesort")
            m_sorted = pair["m"].to_numpy()[order_z]
            steps = np.diff(m_sorted)
            monotone &= bool((steps >= -1e-12).all() or (steps <= 1e-12).all())
            checked += 1
    record(
        "t13_the_calibration_layer_is_monotone",
        "M3's test scores are a monotone function of the zero-shot scores in every cell",
        monotone and checked > 0,
        {"cells_checked": checked},
    )
    page_keys = matrix.merge(
        population[["candidate_id", "environment", "document_id"]], on="candidate_id"
    )
    varying = [
        f
        for f in PAGE_FEATURES
        if (
            page_keys.groupby(
                ["environment_y" if "environment_y" in page_keys else "environment", "document_id"]
            )[f].nunique()
            > 1
        ).any()
    ]
    record(
        "t14_page_features_are_constant_within_a_page_reading",
        "every page feature M4 reads takes one value per page and engine reading",
        not varying,
        {"varying": varying},
    )
    record(
        "t15_the_page_mean_reads_scores_only",
        "the page mean of the zero-shot score mentions no label field",
        not (_mentions("page_mean_scores") & label_fields),
        {"mentions": sorted(_mentions("page_mean_scores") & label_fields)},
    )
    unusable = policies[~policies["usable"].astype(bool)]
    record(
        "t16_an_untrainable_cell_never_accepts",
        "a cell without a trained scorer abstains under every rule",
        bool((unusable["accepted"] == 0).all()),
        {"untrainable_policy_rows": len(unusable)},
    )
    wide = policies.pivot_table(
        index=["direction", "arm", "budget", "draw", "epsilon"],
        columns="rule",
        values="cutoff",
        aggfunc="first",
    )
    both = wide.dropna(subset=[R_PAGE, R_CANDIDATE])
    record(
        "t17_the_page_bound_is_never_looser_than_the_candidate_bound",
        "wherever both bounds choose a cutoff, the page-corrected one is at least as strict",
        bool((both[R_PAGE] >= both[R_CANDIDATE] - 1e-12).all()),
        {"cells_with_both": len(both)},
    )
    a0_calibration = {
        d: set(
            scores[
                (scores["direction"] == d)
                & (scores["arm"] == A0)
                & (scores["evaluation_set"] == "calibration")
            ]["candidate_id"]
        )
        for d in DIRECTIONS
    }
    record(
        "t18_budget_zero_reads_the_source_calibration_block",
        "the zero-shot cutoff is chosen on the source calibration block and on no target row",
        all(
            a0_calibration[d]
            == set(population.iloc[blocks[d].source_calibration]["candidate_id"].astype(str))
            for d in DIRECTIONS
        ),
        {d: len(v) for d, v in a0_calibration.items()},
    )
    one_page = pd.DataFrame({"document_id": ["p"] * 4, "is_harmful": [True, False, False, False]})
    even = pd.DataFrame(
        {"document_id": ["p", "p", "q", "q"], "is_harmful": [True, False, True, False]}
    )
    record(
        "t19_design_effect_by_hand",
        "one calibration page gives no design effect, and pages with identical harm rates give "
        "the floor of one",
        th1.page_design_effect(one_page) is None and th1.page_design_effect(even) == 1.0,
        {"even": th1.page_design_effect(even)},
    )
    frontier = cc_read_json(FRONTIER)
    rederived = {
        d: {
            arm: {
                rule: minimal_budget(
                    {
                        budget: row["deployable"]
                        for budget, row in frontier["table"][d][arm][rule].items()
                    }
                )
                for rule in RULES
            }
            for arm in ADAPTED_ARMS
        }
        for d in DIRECTIONS
    }
    record(
        "t20_the_frontier_re_derives_from_the_curves",
        "every N* follows from the stored deployable flags by the frozen rule",
        rederived == frontier["n_star"],
        {"primary": frontier["primary"]["n_star"]},
    )
    decision_inputs = _mentions("criteria_from_artifacts") | _mentions("run_decide")
    record(
        "t21_the_oracle_never_reaches_a_decision",
        "no criterion or outcome reads an oracle cut",
        not [n for n in decision_inputs if "oracle" in n.lower()],
        {},
    )
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)
    }
    record(
        "t22_no_certification_and_no_reserve",
        "no certification or reserve machinery is called",
        not [n for n in called if "certif" in n.lower() or "reserve" in n.lower()],
        {},
    )
    controls = cc_read_json(CONTROL_RESULTS)
    record(
        "t23_target_grade_permutation_in_band",
        "RK2's target-only arm refitted on grades shuffled inside each purchased page ranks harm "
        "at chance",
        controls["permutation"]["b2_inside_the_band"],
        {"mean_harm_auroc": controls["permutation"]["b2_mean_harm_auroc"]},
    )
    budgets_scored = sorted(set(scores["budget"].astype(str)))
    record(
        "t24_no_budget_beyond_the_pool_is_evaluated",
        "every scored budget is a measured one, and every pool is smaller than the brief's "
        "smallest unmeasured budget",
        set(budgets_scored) <= set(BUDGETS)
        and all(len(b.pages) < min(BRIEF_BUDGETS_NOT_MEASURABLE) for b in blocks.values()),
        {"budgets": budgets_scored, "pools": {d: len(b.pages) for d, b in blocks.items()}},
    )
    generated = set(rl1._hy1_candidates()["candidate_id"].astype(str))
    record(
        "t25_every_candidate_is_one_hy1_generated_and_the_features_are_rk3s",
        "no candidate is generated here, and the feature matrix is RK3's frozen file",
        set(scores["candidate_id"].astype(str)) <= generated
        and file_sha256(rk3.FEATURE_MATRIX)
        == cc_read_json(RESEARCH_FREEZE)["upstream_sha256"][_relative(rk3.FEATURE_MATRIX)],
        {},
    )
    record(
        "t26_minimal_budget_by_hand",
        "N* is the smallest budget from which every larger budget holds",
        minimal_budget({"0": False, "5": True, "10": False, "25": True, POOL: True}) == "25"
        and minimal_budget({"0": True, "5": True, "10": True, "25": True, POOL: False}) is None,
        {},
    )
    passed = sum(t["passed"] for t in tests)
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
    if passed != len(tests):
        raise PhaseError(f"falsification failed: {[t['test'] for t in tests if not t['passed']]}")
    print(f"negative: {passed}/{len(tests)} pass ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the decision


def criteria_from_artifacts() -> dict[str, Any]:
    frontier = cc_read_json(FRONTIER)
    curves = cc_read_json(CURVES)["policies"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    n_star = frontier["primary"]["n_star"]
    ready = frontier["primary"]["ranking_ready"]
    shares = {
        d: {
            rule: curves[d][PRIMARY_ARM][rule][key][POOL]["violation_share"]
            for rule in (R_CANDIDATE, R_PAGE)
        }
        for d in DIRECTIONS
    }
    testable = [d for d in DIRECTIONS if shares[d][R_CANDIDATE] > VIOLATION_CEILING]
    criteria = {
        "R1": all(ready[d] is not None for d in DIRECTIONS),
        "R2": bool(testable) and all(shares[d][R_PAGE] <= VIOLATION_CEILING for d in testable),
        "R3": all(n_star[d] is not None for d in DIRECTIONS),
    }
    stats = cc_read_json(STATISTICAL_TESTS)["family"]
    return {
        "criteria": criteria,
        "n_star": n_star,
        "ranking_ready": ready,
        "violation_shares_at_pool": shares,
        "r2_testable_directions": testable,
        "outcome": assign_outcome(n_star, ready),
        "supported": {name: bool(row["survives_holm"]) for name, row in stats.items()},
    }


def outcome_label(state: dict[str, Any]) -> str:
    def shown(value: str | None) -> str:
        return (
            "not reached" if value is None else (f"{value} pages" if value != POOL else "the pool")
        )

    parts = [
        f"{DIRECTION_SHORT[d]}: ranking-ready at {shown(state['ranking_ready'][d])}, "
        f"N* {shown(state['n_star'][d])}"
        for d in DIRECTIONS
    ]
    return f"{state['outcome']}: {OUTCOME_TAXONOMY[state['outcome']]} -- " + "; ".join(parts)


def run_decide() -> int:
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    falsification = cc_read_json(FALSIFICATION)
    stats = cc_read_json(STATISTICAL_TESTS)
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "outcome": state["outcome"],
            "outcome_label": outcome_label(state),
            "criteria": state["criteria"],
            "n_star_primary": state["n_star"],
            "ranking_ready_primary": state["ranking_ready"],
            "violation_shares_at_pool": state["violation_shares_at_pool"],
            "r2_testable_directions": state["r2_testable_directions"],
            "statistically_supported": state["supported"],
            "statistical_family_surviving": stats["surviving"],
            "deployment_claim_permitted": bool(state["criteria"]["R3"]),
            "primary_arm": PRIMARY_ARM,
            "primary_rule": PRIMARY_RULE,
            "primary_epsilon": PRIMARY_EPSILON,
            "measured_budgets": list(BUDGETS),
            "not_measurable": list(BRIEF_BUDGETS_NOT_MEASURABLE),
            "falsification_tests_passed": falsification["passed"],
            "falsification_tests_total": falsification["total"],
            "recommended_next_stage": NEXT_STAGE[state["outcome"]],
            "issued_head": _git("rev-parse", "HEAD"),
            "ready_for_external_confirmation": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: {outcome_label(state)}")
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = rk3.FIGURE_NOTE
SURFACE = rk3.SURFACE
INK = rk3.INK
INK_SECONDARY = rk3.INK_SECONDARY
GRID = rk3.GRID
ARM_COLOURS = {
    A0: "#8a8a8a",
    M1: "#eb6834",
    M2: "#1baf7a",
    M3: "#b58b00",
    M4: "#9c6ade",
    B2: "#2a78d6",
}
RULE_COLOURS = {R_NAIVE: "#8a8a8a", R_CANDIDATE: "#2a78d6", R_PAGE: "#eb6834"}
FIGURES = (
    "fig1_label_efficiency.png",
    "fig2_safety_by_page_budget.png",
    "fig3_generator_transfer_matrix.png",
    "fig4_harm_recall_frontier.png",
    "fig5_page_calibration_effect.png",
    "fig6_feature_ablation.png",
)


def run_figures() -> int:
    """Six figures, every value read from a persisted artifact."""
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
            "svg.hashsalt": "rk4",
        }
    )
    manifest: dict[str, Any] = {}

    def save(fig: Any, name: str, sources: Sequence[Path], caption: str) -> None:
        fig.text(0.01, 0.005, FIGURE_NOTE, fontsize=5.5, color=INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(
            FIGURE_DIR / name,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata={"Software": None},
        )
        plt.close(fig)
        manifest[name] = {
            "path": f"{FIGURE_DIR.name}/{name}",
            "caption": caption,
            "sources": {s.name: rk3.xr1._signature(s) for s in sources},
        }

    curves = cc_read_json(CURVES)
    pools = curves["pool_pages"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)

    def pages(direction: str) -> list[int]:
        return [budget_pages(b, pools[direction]) for b in BUDGETS]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    reference = cc_read_json(FROZEN_CONFIGURATION)["rk3_in_distribution_reference"]
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        x = pages(direction)
        for arm in ADAPTED_ARMS:
            rows = [
                curves["ranking"][direction][arm if b != "0" else A0][b]["harm_auroc"]
                for b in BUDGETS
            ]
            middle = [np.nan if r["median"] is None else r["median"] for r in rows]
            low = [np.nan if r["q1"] is None else r["q1"] for r in rows]
            high = [np.nan if r["q3"] is None else r["q3"] for r in rows]
            ax.plot(
                x, middle, marker="o", ms=3, lw=1.2, color=ARM_COLOURS[arm], label=ARM_SHORT[arm]
            )
            ax.fill_between(x, low, high, color=ARM_COLOURS[arm], alpha=0.12, lw=0)
        ax.axhline(HARM_FLOOR, color=INK_SECONDARY, lw=0.8, ls="--")
        ax.axhline(reference[direction], color=INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xlabel("labelled target pages")
        ax.set_title(DIRECTION_SHORT[direction], loc="left", fontsize=8)
    axes[0].set_ylabel("test harm AUROC, median and IQR over draws")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    save(
        fig,
        FIGURES[0],
        (CURVES, FROZEN_CONFIGURATION),
        "Label efficiency: harm AUROC on the sealed target test pages by labelled page budget; "
        "dashed, the 0.75 floor; dotted, RK3 trained on both generators",
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex="col")
    for row, direction in enumerate(DIRECTIONS):
        x = pages(direction)
        for rule in RULES:
            rows = [
                curves["policies"][direction][PRIMARY_ARM if b != "0" else A0][rule][key][b]
                for b in BUDGETS
            ]
            axes[row, 0].plot(
                x,
                [r["violation_share"] for r in rows],
                marker="o",
                ms=3,
                color=RULE_COLOURS[rule],
                label=rule.replace("_", " "),
            )
            axes[row, 1].plot(
                x, [r["working_share"] for r in rows], marker="o", ms=3, color=RULE_COLOURS[rule]
            )
        axes[row, 0].axhline(VIOLATION_CEILING, color=INK_SECONDARY, lw=0.8, ls="--")
        axes[row, 1].axhline(WORKING_FLOOR, color=INK_SECONDARY, lw=0.8, ls="--")
        axes[row, 0].set_ylabel(f"{DIRECTION_SHORT[direction]}\nshare of draws")
        axes[row, 0].set_ylim(-0.05, 1.05)
        axes[row, 1].set_ylim(-0.05, 1.05)
    axes[0, 0].set_title("(a) draws whose test harm exceeds 0.1", loc="left", fontsize=8)
    axes[0, 1].set_title("(b) draws that hold and repair something", loc="left", fontsize=8)
    axes[1, 0].set_xlabel("labelled target pages")
    axes[1, 1].set_xlabel("labelled target pages")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper right")
    save(
        fig,
        FIGURES[1],
        (CURVES,),
        "M1 under each cutoff rule by page budget; a budget is deployable when (a) is at most "
        "0.1 and (b) at least 0.5",
    )

    transfer = cc_read_json(TRANSFER)
    columns = [
        ("adapted_to_Qwen|eval_Qwen", "to Qwen,\neval Qwen"),
        ("adapted_to_Qwen|eval_current", "to Qwen,\neval current"),
        ("adapted_to_current|eval_current", "to current,\neval current"),
        ("adapted_to_current|eval_Qwen", "to current,\neval Qwen"),
    ]
    grid = np.full((len(MATRIX_ARMS), len(columns)), np.nan)
    for i, arm in enumerate(MATRIX_ARMS):
        for j, (name, _label) in enumerate(columns):
            value = transfer["matrix"][arm][name]["median_harm_auroc"]
            grid[i, j] = np.nan if value is None else value
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.imshow(grid, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
    ax.grid(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j,
                i,
                "—" if np.isnan(grid[i, j]) else f"{grid[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=INK,
            )
    ax.set_xticks(range(len(columns)), [label for _n, label in columns])
    ax.set_yticks(range(len(MATRIX_ARMS)), [ARM_SHORT[a] for a in MATRIX_ARMS])
    save(
        fig,
        FIGURES[2],
        (TRANSFER,),
        "Generator transfer matrix: median test harm AUROC at the pool (zero shot at budget 0), "
        "on the generator adapted to and on the source generator",
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        for arm in SWEEP_ARMS:
            budget = "0" if arm == A0 else POOL
            rows = transfer["frontier"][direction][arm][budget]["median_frontier"]
            ax.plot(
                [r["harm"] for r in rows],
                [np.nan if r["recall"] is None else r["recall"] for r in rows],
                lw=1.2,
                color=ARM_COLOURS[arm],
                label=ARM_SHORT[arm],
            )
        ax.axvline(PRIMARY_EPSILON, color=INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xlabel("realized selective harm")
        ax.set_title(DIRECTION_SHORT[direction], loc="left", fontsize=8)
    axes[0].set_ylabel("highest exact-repair recall, median over draws")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    save(
        fig,
        FIGURES[3],
        (TRANSFER,),
        "Harm-recall frontier, ANALYSIS ONLY: the highest recall any cutoff reaches at each "
        "realized "
        "harm on the test labels, adapted arms at the pool",
    )

    effect = cc_read_json(CALIBRATION_EFFECT)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for direction, style in zip(DIRECTIONS, ("-", "--"), strict=True):
        x = pages(direction)
        for rule in RULES:
            rows = [effect["primary_arm_by_rule"][direction][rule][b] for b in BUDGETS]
            axes[0].plot(
                x,
                [
                    np.nan
                    if r["median_harm_estimation_error"] is None
                    else r["median_harm_estimation_error"]
                    for r in rows
                ],
                marker="o",
                ms=3,
                ls=style,
                color=RULE_COLOURS[rule],
                label=f"{rule.replace('_', ' ')}, {DIRECTION_SHORT[direction]}",
            )
        for arm in (M3, M4):
            rows = [effect["m4_vs_m3"][direction][b][arm][R_NAIVE] for b in ADAPTED_BUDGETS]
            axes[1].plot(
                x[1:],
                [r["violation_share"] for r in rows],
                marker="o",
                ms=3,
                ls=style,
                color=ARM_COLOURS[arm],
                label=f"{ARM_SHORT[arm]}, {DIRECTION_SHORT[direction]}",
            )
    axes[0].axhline(0.0, color=INK_SECONDARY, lw=0.8)
    axes[0].set_xlabel("labelled target pages")
    axes[0].set_ylabel("test harm minus calibration harm")
    axes[0].set_title("(a) M1: how optimistic each rule's pages are", loc="left", fontsize=8)
    axes[1].axhline(VIOLATION_CEILING, color=INK_SECONDARY, lw=0.8, ls="--")
    axes[1].set_xlabel("labelled target pages")
    axes[1].set_ylabel("violation share, naive rule")
    axes[1].set_title("(b) page-risk term against none", loc="left", fontsize=8)
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper right")
    axes[1].legend(frameon=False, fontsize=5.5, loc="upper right")
    save(
        fig,
        FIGURES[4],
        (CALIBRATION_EFFECT,),
        "Page calibration effect: (a) median realized-minus-calibration harm among draws that "
        "accept, by rule, where the page-level bound has no point because it never accepts; (b) "
        "violation shares with and without the page-risk term",
    )

    ablation = cc_read_json(ABLATION)["by_direction"]
    names = [n for n in ablation[DIRECTIONS[0]] if n != "full"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    y = np.arange(len(names))
    for offset, direction in zip((-0.2, 0.2), DIRECTIONS, strict=True):
        values = [ablation[direction][n]["delta_harm_auroc"] or 0.0 for n in names]
        ax.barh(
            y + offset,
            values,
            height=0.38,
            color=ARM_COLOURS[M1] if offset < 0 else ARM_COLOURS[B2],
            label=DIRECTION_SHORT[direction],
        )
    ax.axvline(0.0, color=INK_SECONDARY, lw=0.8)
    ax.set_yticks(y, [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("change in median test harm AUROC, M1 at the pool")
    ax.legend(frameon=False, fontsize=6.5)
    save(
        fig,
        FIGURES[5],
        (ABLATION,),
        "Feature ablations of M1 at the pool over the first draws, descriptive",
    )
    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_envelope("figure_manifest"),
            "figures": manifest,
            "note": FIGURE_NOTE,
            "synthetic": False,
        },
    )
    print(f"figures: {len(manifest)} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ determinism and record

DERIVED_OUTPUTS = (
    "SPLIT_REGISTRY",
    "PURCHASE_REGISTRY",
    "REPRODUCTION",
    "CELL_SCORES",
    "ADAPTATION_REGISTRY",
    "POLICY_OUTCOMES",
    "RANKING_OUTCOMES",
    "POLICY_REGISTRY",
    "CURVES",
    "FRONTIER",
    "TRANSFER",
    "CALIBRATION_EFFECT",
    "ABLATION",
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
        ("sample", run_sample),
        ("reproduce", run_reproduce),
        ("adapt", run_adapt),
        ("policies", run_policies),
        ("curves", run_curves),
        ("frontier", run_frontier),
        ("transfer", run_transfer),
        ("calibration", run_calibration),
        ("ablate", run_ablate),
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
            out[path.name] = rk3.xr1._signature(path)
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
                "pages": PAGE_SEED,
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
    "1. Motivation": (FROZEN_CONFIGURATION, RESEARCH_FREEZE),
    "2. Research Questions and Pre-Registration": (DESIGN_RECORD,),
    "3. What the Pages Can Measure": (SPLIT_REGISTRY, PURCHASE_REGISTRY, DESIGN_RECORD),
    "4. Adaptation Arms and Cutoff Rules": (DESIGN_RECORD, ADAPTATION_REGISTRY, POLICY_REGISTRY),
    "5. Reproduction and Isolation": (REPRODUCTION, SPLIT_REGISTRY),
    "6. Ranking Recovery by Page Budget": (CURVES, FROZEN_CONFIGURATION, ADAPTATION_REGISTRY),
    "7. Safety and Usefulness by Page Budget": (CURVES, POLICY_REGISTRY),
    "8. The Deployable Frontier": (FRONTIER, CURVES),
    "9. Page Calibration Effect": (CALIBRATION_EFFECT, CURVES),
    "10. Generator Transfer Matrix and Harm-Recall Frontier": (TRANSFER, FROZEN_CONFIGURATION),
    "11. Feature Ablation": (ABLATION,),
    "12. Statistical Tests": (STATISTICAL_TESTS,),
    "13. Controls and Falsification": (CONTROL_RESULTS, FALSIFICATION),
    "14. Limitations": (SPLIT_REGISTRY, PURCHASE_REGISTRY, CURVES, DESIGN_RECORD),
    "15. Research Decision": (DECISION, FRONTIER, STATISTICAL_TESTS, FALSIFICATION),
    "16. Next Research Direction": (DECISION, CURVES, TRANSFER, FRONTIER),
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
    audit = rk3.gen1.audit_report(REPORT, REPORT_SECTIONS)
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
            "regenerates_ocr": False,
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
        "reproduce": run_reproduce,
        "adapt": run_adapt,
        "policies": run_policies,
        "curves": run_curves,
        "frontier": run_frontier,
        "transfer": run_transfer,
        "calibration": run_calibration,
        "ablate": run_ablate,
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
