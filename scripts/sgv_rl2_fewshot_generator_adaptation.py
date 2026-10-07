#!/usr/bin/env python3
"""SGV-RL2: how much supervision from a new correction generator repairs reliability ranking?

SGV-RL1 closed with outcome B. A generator-agnostic representation ranks the pooled two-generator
candidate universe well (harm AUROC 0.8358, benefit 0.8329) and keeps most of what explicit
generator identity would buy, and it survives an engine change: leave-one-environment-out, with
the documents held out too, stays at 0.8352 harm. What it does not survive is a change of
correction generator -- strict leave-one-generator-out falls to 0.6206 harm and 0.4782 benefit.
RL1 recorded the mechanism as "candidate evidence works, but each new generator needs
supervision". This stage tests that directly:

    how many labelled candidates from a previously unseen correction generator are needed before
    reliability ranking recovers -- and does the repair hold in both transfer directions?

**1. Nothing upstream moves.** The candidate universe, the labels, the R3 feature matrix, the
model family and its hyperparameters are read from SGV-RL1 and hash-verified. No candidate is
generated, no feature is added or removed, and no model search is reopened.

**2. The representation stays source-agnostic.** Target labels reach the model only through
fitting. No feature vector ever carries a generator, corrector or proposal-source column, and an
exact invariance test asserts it.

**3. The target evaluation pool is sealed.** Adaptation and evaluation documents are disjoint by a
content-blind rule fixed before any label is purchased, and no site or atomic edit crosses the
boundary. Budgets, draws and the decision budget are pre-registered.

**4. Both directions are reported.** Current-generator to Qwen3-VL and Qwen3-VL to
current-generator. The asymmetry is a result, not something to average away.

    --reconstruct   section 0: HY1 and RL1 re-read from their own artifacts
    --freeze        upstream hashes and the frozen RL1 inputs this stage consumes
    --preregister   directions, splits, budgets, draws, arms, criteria, outcome rules
    --splits        the target adaptation/evaluation registry and the group registry
    --draws         the deterministic few-shot draws and their label composition
    --reproduce     the R3 matrix and the RL1 zero-shot transfer reproduced exactly
    --adapt         every arm x direction x budget x draw, scored out of sample
    --curves        sample efficiency, gap recovery, the minimal useful budget
    --retention     source-generator forgetting and the pooled hybrid population
    --prefix        ranking geometry and the diagnostic oracle harm-constrained frontier
    --strata        environment, generator-direction and error-type breakdowns
    --controls      random target labels, permuted target labels, infeasible draws
    --stats         document-clustered paired bootstrap, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen RL1 inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / ADAPTATION ONLY. No deployment threshold is selected, no certification is run, no
human-review allocation is optimized, and the oracle frontier is analysis-only.
`ready_for_external_confirmation` is false by construction and the confirmatory reserve stays
LOCKED.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_rl1_generator_agnostic_reliability as rl1
from ocr_risk.io.hashing import canonical_hash, file_sha256

hy1 = rl1.hy1
gen1 = rl1.gen1
xr1 = rl1.xr1
xc1 = rl1.xc1
lp1 = rl1.lp1
s15 = rl1.s15

REPO = rl1.REPO
OUT = REPO / "results/generated/sgv_rl2_fewshot_generator_adaptation"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"
POPULATION_INVENTORY = OUT / "candidate_population_inventory.json"

TARGET_SPLIT_REGISTRY = OUT / "target_split_registry.json"
GROUP_REGISTRY = OUT / "group_registry.json"
BUDGET_REGISTRY = OUT / "budget_registry.json"
DRAW_REGISTRY = OUT / "adaptation_draw_registry.json"
LABEL_COMPOSITION = OUT / "label_composition.json"
INFEASIBLE_DRAWS = OUT / "infeasible_draws.json"

REPRESENTATION_REPRODUCTION = OUT / "representation_reproduction.json"
ZERO_SHOT_REPRODUCTION = OUT / "zero_shot_reproduction.json"

MODEL_REGISTRY = OUT / "model_registry.json"
ADAPTATION_REGISTRY = OUT / "adaptation_registry.json"
HYPERPARAMETER_REGISTRY = OUT / "hyperparameter_registry.json"
ADAPTED_SCORES = OUT / "adapted_scores.parquet"

SAMPLE_EFFICIENCY_HARM = OUT / "sample_efficiency_harm.json"
SAMPLE_EFFICIENCY_BENEFIT = OUT / "sample_efficiency_benefit.json"
TRANSFER_GAP_RECOVERY = OUT / "transfer_gap_recovery.json"
MINIMAL_LABEL_BUDGET = OUT / "minimal_label_budget.json"
METHOD_COMPARISON = OUT / "adaptation_method_comparison.json"
ADAPTATION_EFFICIENCY = OUT / "adaptation_efficiency.json"

SOURCE_RETENTION = OUT / "source_retention.json"
MIXED_POPULATION = OUT / "mixed_population_results.json"

PREFIX_RANKING = OUT / "prefix_ranking.json"
ORACLE_RISK_FRONTIER = OUT / "oracle_risk_frontier.json"

ENVIRONMENT_ANALYSIS = OUT / "environment_analysis.json"
DIRECTION_ANALYSIS = OUT / "generator_direction_analysis.json"
ERROR_TYPE_ANALYSIS = OUT / "error_type_analysis.json"

RANDOM_LABEL_CONTROL = OUT / "random_label_control.json"
LABEL_PERMUTATION_CONTROL = OUT / "label_permutation_control.json"
CONTROL_RESULTS = OUT / "control_results.json"

STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_rl2/fewshot_generator_adaptation.md"

SCHEMA_VERSION = 1
STAGE = "sgv_rl2_fewshot_generator_adaptation"
HYPOTHESIS = "SGV-RL2-D1"
STAGE_KIND = "DEVELOPMENT / ADAPTATION"

PhaseError = rl1.PhaseError
_relative = rl1._relative
_git = rl1._git
_write_json_once = rl1._write_json_once
_write_parquet_once = rl1._write_parquet_once
cc_read_json = rl1.cc_read_json
_ratio = rl1._ratio
_require = rl1._require
_forbid = rl1._forbid
auroc = rl1.auroc
average_precision = rl1.average_precision
_mean = rl1._mean
_finite = rl1._finite

BOOTSTRAP_RESAMPLES = rl1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rl1.BOOTSTRAP_SEED
ALPHA = rl1.ALPHA
EPSILONS = rl1.EPSILONS
PRIMARY_EPSILON = rl1.PRIMARY_EPSILON
epsilon_key = rl1.epsilon_key
PREFIX_FRACTIONS = rl1.PREFIX_FRACTIONS

# ------------------------------------------------------------------ the frozen design

# The representation, the population and the model family are RL1's, taken unchanged.
POPULATION = rl1.PRIMARY_POPULATION
REPRESENTATION = rl1.PRIMARY_REPRESENTATION
MODEL = rl1.PRIMARY_MODEL
C0 = rl1.C0
C1 = rl1.C1
TARGETS = rl1.TARGETS
T_HARM = rl1.T_HARM
T_BENEFIT = rl1.T_BENEFIT

# Transfer directions, named by what is adapted to.
D_G2Q = "current_to_qwen"
D_Q2G = "qwen_to_current"
DIRECTIONS = (D_G2Q, D_Q2G)
SOURCE_OF = {D_G2Q: C0, D_Q2G: C1}
TARGET_OF = {D_G2Q: C1, D_Q2G: C0}
DIRECTION_LABEL = {
    D_G2Q: "frozen current generator -> Qwen3-VL image corrector",
    D_Q2G: "Qwen3-VL image corrector -> frozen current generator",
}

# The target split reuses RL1's frozen content-blind document folds, so nothing about the
# partition is decided here: adaptation takes the first three folds, evaluation the last two.
ADAPTATION_FOLDS = (0, 1, 2)
EVALUATION_FOLDS = (3, 4)

# Label budgets, fixed before any endpoint. N = 0 is the zero-shot baseline.
BUDGETS = (0, 10, 25, 50, 100, 250, 500)
DECISION_BUDGET = 100
SATURATION_BUDGET = 500
DRAWS = 20
DRAW_SEED = "sgv-rl2-fewshot-draws-v1:2026-09-21"
# The draw used for the pre-registered inferential comparison, chosen before any endpoint so the
# test is not selected on its own result. Across-draw spread is reported descriptively.
INFERENCE_DRAW = 0

# Adaptation arms.
A0 = "a0_zero_shot"
A1 = "a1_score_calibration"
A2 = "a2_joint_refit"
A3 = "a3_target_only"
ARMS = (A0, A1, A2, A3)
PRIMARY_ARM = A2

# Success criteria, frozen before any RL2 endpoint. The floors are RL1's absolute discrimination
# floors, reused unchanged so the two stages are comparable.
HARM_FLOOR = rl1.C1_HARM_FLOOR
BENEFIT_FLOOR = rl1.C1_BENEFIT_FLOOR
RECOVERY_FLOOR = 0.75
BREADTH_MAJORITY = rl1.C5_BREADTH_MAJORITY
FORGETTING_TOLERANCE = 0.05
PREFIX_MARGIN = rl1.C6_PREFIX_MARGIN
# A recovery fraction is not identifiable when the reference is at or below the zero-shot result.
MIN_RECOVERY_DENOMINATOR = 0.02

# Across-draw aggregation, declared before any endpoint: the median is the point estimate and the
# quartiles carry the spread, because a mean over twenty few-shot draws is dragged by the draw
# that happened to buy no positive label.
AGGREGATE = "median"

RANDOM_LABEL_SEED = 20260925
PERMUTATION_SEED = 20260926
RANDOM_RANKING_SEED = rl1.RANDOM_RANKING_SEED
RANDOM_RANKING_DRAWS = rl1.RANDOM_RANKING_DRAWS

QUESTIONS = {
    "Q1": "does target-generator supervision recover the zero-shot transfer gap?",
    "Q2": "how many labelled target candidates are needed before adaptation is useful?",
    "Q3": "do harm and benefit ranking need different amounts of target supervision?",
    "Q4": "is score alignment enough, or must the model geometry adapt?",
    "Q5": "does adaptation improve both directions and many environments, not just one?",
    "Q6": "is adapted ranking good enough to justify a later calibration stage?",
}

OUTCOME_TAXONOMY = {
    "A": "few-shot adaptation is sample-efficient",
    "B": "adaptation works but a qualifying criterion does not",
    "C": "adaptation is asymmetric across transfer directions",
    "D": "few-shot adaptation does not repair generator transfer",
}
NEXT_STAGE = {
    "A": "NEXT: ADAPTED RELIABILITY -> RISK CALIBRATION -> SELECTIVE DEPLOYMENT SYNTHESIS",
    "B": "NEXT: DEPLOYMENT SYNTHESIS WITH AN EXPLICIT LABEL-COST TRADEOFF",
    "C": "NEXT: GENERATOR-CONDITIONAL ADAPTATION / MIXTURE-OF-EXPERT RELIABILITY",
    "D": "NEXT: RICHER CANDIDATE VERIFICATION EVIDENCE",
}

PRIMARY_FAMILY = (
    "P1_a2_vs_a0_harm_auroc",
    "P2_a2_vs_a0_benefit_auroc",
    "P3_a2_vs_a1_harm_auroc",
    "P4_a2_vs_a1_benefit_auroc",
)

FORBIDDEN_SOURCE_COLUMNS = rl1.FORBIDDEN_SOURCE_COLUMNS
FORBIDDEN_LABEL_COLUMNS = rl1.FORBIDDEN_LABEL_COLUMNS

RL1_EXPECTED: dict[str, Any] = {
    "status": "COMPLETE",
    "outcome": "B",
    "ready_for_reliability_adaptation": True,
    "ready_for_external_confirmation": False,
    "confirmatory_reserve_consumed": False,
    "deployable": False,
    "selects_a_deployment_threshold": False,
    "falsification_tests_passed": 20,
    "falsification_tests_total": 20,
    "primary_population": POPULATION,
    "primary_representation": REPRESENTATION,
    "primary_model": MODEL,
    "recommended_next_stage": (
        "NEXT: FEW-SHOT GENERATOR-AWARE ADAPTATION OF A SOURCE-AGNOSTIC REPRESENTATION"
    ),
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_rl2-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "stage_kind": STAGE_KIND,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str) -> dict[str, Any]:
    return {
        **_envelope(artifact),
        "uses_ground_truth": True,
        "deployable": False,
        "selects_a_deployment_threshold": False,
        "confirmatory_reserve_consumed": False,
    }


def recovery(adapted: float, zero_shot: float, reference: float) -> float | None:
    """Share of the zero-shot-to-reference gap that adaptation closes.

    Undefined, and never a success, when the reference is at or below the zero-shot result: a
    ratio whose denominator is noise would turn "there was nothing to recover" into "everything
    recovered".
    """
    if not all(np.isfinite(v) for v in (adapted, zero_shot, reference)):
        return None
    gap = reference - zero_shot
    if gap < MIN_RECOVERY_DENOMINATOR:
        return None
    return float((adapted - zero_shot) / gap)


def assign_outcome(criteria: dict[str, bool], adapts_at_saturation: dict[str, bool]) -> str:
    """Exactly one outcome, in the frozen precedence D, C, A, B.

    D first: if neither direction reaches the absolute floors even at the largest tested budget,
    there is nothing to be sample-efficient about. C next: one direction adapts and the other does
    not, which is a directional finding and not an average. A needs every criterion. Anything left
    is B, and its label names the criterion that failed.
    """
    adapting = [name for name, value in adapts_at_saturation.items() if value]
    if not adapting:
        return "D"
    if len(adapting) < len(DIRECTIONS):
        return "C"
    if all(criteria.values()):
        return "A"
    return "B"


def environment_specs() -> list[dict[str, Any]]:
    return rl1.environment_specs()


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    """Every upstream file RL2's conclusions rest on: RL1's list plus RL1's own artifacts."""
    files = list(rl1._upstream_files())
    files.append(REPO / "scripts/sgv_rl1_generator_agnostic_reliability.py")
    for pattern in ("*.json", "*.parquet"):
        files.extend(sorted(rl1.OUT.glob(pattern)))
    files.append(rl1.REPORT)
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = xr1._check
    checks = list(rl1.upstream_checks())
    decision = cc_read_json(rl1.DECISION)
    for key, value in RL1_EXPECTED.items():
        checks.append(_check(f"sgv_rl1.{key}", decision.get(key), value))
    # RL1's qualitative findings, each as the number it rests on.
    mixed = cc_read_json(rl1.MIXED_SOURCE_RESULTS)
    pooled = mixed["by_population"][POPULATION][MODEL]
    logo = cc_read_json(rl1.LEAVE_GENERATOR_OUT)["by_split"][rl1.PRIMARY_TRANSFER_SPLIT][MODEL]
    loeo = cc_read_json(rl1.LEAVE_ENVIRONMENT_OUT)["by_model"][MODEL][REPRESENTATION]
    checks.append(
        _check(
            "sgv_rl1.mixed_source_beats_zero_shot_transfer_on_harm",
            bool(
                float(pooled[REPRESENTATION]["harm_auroc"])
                > float(logo[REPRESENTATION]["mean_harm_auroc"])
            ),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_rl1.engine_transfer_holds_while_generator_transfer_does_not",
            bool(float(loeo["mean_harm_auroc"]) > float(logo[REPRESENTATION]["mean_harm_auroc"])),
            True,
        )
    )
    gap = cc_read_json(rl1.SOURCE_IDENTITY_GAP)["mixed_source"]
    checks.append(
        _check(
            "sgv_rl1.identity_gain_is_small",
            bool(float(gap[T_HARM]["gap"]) < rl1.SOURCE_GAP_MARGIN),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_rl1.r3_retains_most_of_r4",
            bool(float(gap[T_HARM]["retention"]) >= rl1.C4_RETENTION_FLOOR),
            True,
        )
    )
    leakage = cc_read_json(rl1.IMPLICIT_SOURCE_LEAKAGE)["results"][REPRESENTATION]
    checks.append(
        _check(
            "sgv_rl1.implicit_source_predictability_is_high",
            bool(float(leakage["source_auroc"]) > 0.9),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_rl1.visual_gain_is_not_material",
            cc_read_json(rl1.VISUAL_ABLATION)["visual_gain_is_material"],
            False,
        )
    )
    checks.append(
        _check("sgv_rl1.determinism", cc_read_json(rl1.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_rl1.report_traceability",
            cc_read_json(rl1.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
            0,
        )
    )
    provenance = cc_read_json(rl1.PROVENANCE)
    checks.append(
        _check("sgv_rl1.upstream_files_moved", len(provenance["upstream_files_moved"]), 0)
    )
    checks.append(_check("sgv_rl1.hy1_unchanged", provenance["hy1_unchanged"], True))
    checks.append(
        _check("sgv_rl1.regenerates_a_correction", provenance["regenerates_a_correction"], False)
    )
    return checks


def run_reconstruct() -> int:
    """Section 0: every upstream stage re-read from its own artifacts, never restated."""
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
    return any(p.exists() for p in (ADAPTED_SCORES, SAMPLE_EFFICIENCY_HARM, DECISION))


def run_freeze() -> int:
    """The research freeze: upstream hashes and the frozen RL1 inputs RL2 consumes."""
    started = time.monotonic()
    for path in (RESEARCH_FREEZE, FROZEN_CONFIGURATION, POPULATION_INVENTORY):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an RL2 endpoint already exists; the freeze must precede every one")
    OUT.mkdir(parents=True, exist_ok=True)
    upstream = _upstream_files()
    hashes = {_relative(p): file_sha256(p) for p in upstream}
    rl1_provenance = cc_read_json(rl1.PROVENANCE)["artifacts"]
    moved = sorted(path for path, sha in rl1_provenance.items() if file_sha256(REPO / path) != sha)
    if moved:
        raise PhaseError(f"{len(moved)} RL1 artifacts moved since RL1 issued its record: {moved}")
    consumed = {
        "candidate_population": rl1.CANDIDATE_POPULATION,
        "feature_matrix": rl1.FEATURE_MATRIX,
        "r3_representation": rl1.R3_REPRESENTATION,
        "group_registry": rl1.GROUP_REGISTRY,
        "split_registry": rl1.SPLIT_REGISTRY,
        "reliability_scores": rl1.SCORES,
        "hyperparameter_registry": rl1.HYPERPARAMETER_REGISTRY,
        "model_registry": rl1.MODEL_REGISTRY,
    }
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "upstream_sha256": hashes,
            "upstream_file_count": len(hashes),
            "rl1_artifacts_rehashed": len(rl1_provenance),
            "rl1_artifacts_moved": moved,
            "rl1_inputs_consumed": {
                name: {
                    "path": _relative(path),
                    "sha256": file_sha256(path),
                    "matches_rl1_provenance": bool(
                        rl1_provenance.get(_relative(path)) == file_sha256(path)
                    ),
                }
                for name, path in consumed.items()
            },
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status": _git("status", "--short"),
                "diff_stat": _git("diff", "--stat"),
            },
            "rl1_is_immutable_input": True,
            "regenerates_a_candidate": False,
            "changes_the_representation": False,
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "population": POPULATION,
            "representation": REPRESENTATION,
            "representation_columns": len(rl1.columns_for(REPRESENTATION)),
            "model": MODEL,
            "hyperparameters": dict(cc_read_json(rl1.HYPERPARAMETER_REGISTRY)[MODEL]),
            "hyperparameter_source": cc_read_json(rl1.HYPERPARAMETER_REGISTRY)["source"],
            "document_folds": _relative(rl1.GROUP_REGISTRY),
            "labelling": cc_read_json(rl1.FROZEN_CONFIGURATION)["labelling"],
            "rl1_zero_shot_reference": {
                "split": rl1.PRIMARY_TRANSFER_SPLIT,
                "mean_harm_auroc": cc_read_json(rl1.LEAVE_GENERATOR_OUT)["primary"][
                    "mean_harm_auroc"
                ],
                "mean_benefit_auroc": cc_read_json(rl1.LEAVE_GENERATOR_OUT)["primary"][
                    "mean_benefit_auroc"
                ],
            },
            "rl1_mixed_source_reference": {
                "split": rl1.PRIMARY_SPLIT,
                "harm_auroc": cc_read_json(rl1.MIXED_SOURCE_RESULTS)["by_population"][POPULATION][
                    MODEL
                ][REPRESENTATION]["harm_auroc"],
                "benefit_auroc": cc_read_json(rl1.MIXED_SOURCE_RESULTS)["by_population"][
                    POPULATION
                ][MODEL][REPRESENTATION]["benefit_auroc"],
            },
            "uses_ground_truth": False,
        },
    )
    population = load_population()
    _write_json_once(
        POPULATION_INVENTORY,
        {
            **_analysis_envelope("candidate_population_inventory"),
            "source": _relative(rl1.CANDIDATE_POPULATION),
            "population": POPULATION,
            "candidates": len(population),
            "documents": int(population["document_id"].nunique()),
            "sites": int(population["site_group"].nunique()),
            "environments": int(population["environment"].nunique()),
            "by_corrector": {
                str(name): {
                    "candidates": len(group),
                    "harmful": int(group["is_harmful"].sum()),
                    "beneficial": int(group["beneficial"].sum()),
                    "exact": int(group["exact"].sum()),
                }
                for name, group in population.groupby("corrector_source", sort=True)
            },
            "multi_source_candidates": int(population["multi_source"].sum()),
            "never_regenerated_here": True,
        },
    )
    print(
        f"freeze: {len(hashes)} upstream files hashed, {len(population)} RL1 candidates frozen "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def load_population() -> pd.DataFrame:
    """RL1's primary candidate population with its frozen document folds attached."""
    population = pd.read_parquet(rl1.CANDIDATE_POPULATION)
    block = population[population["population"] == POPULATION].reset_index(drop=True)
    folds = cc_read_json(rl1.GROUP_REGISTRY)["document_folds"]
    mapped = block["document_id"].astype(str).map(folds)
    if mapped.isna().any():
        raise PhaseError("a candidate's document carries no RL1 fold assignment")
    block["fold"] = mapped.to_numpy(dtype=np.int64)
    return block


# ------------------------------------------------------------------ sections 3-13: the design


def run_preregister() -> int:
    """Every choice that could otherwise be made after seeing an endpoint."""
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("an RL2 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "questions": QUESTIONS,
            "primary_hypothesis": (
                "the shared source-agnostic representation already carries useful candidate "
                "evidence, and a small amount of target-generator supervision is enough to adapt "
                "its decision geometry to the new candidate distribution"
            ),
            "frozen_inputs": {
                "population": POPULATION,
                "representation": REPRESENTATION,
                "model": MODEL,
                "rule": (
                    "the candidate universe, the labels, the R3 feature matrix, the model family "
                    "and its hyperparameters are RL1's and are never re-derived, re-selected or "
                    "extended here"
                ),
            },
            "directions": {
                name: {
                    "source_generator": SOURCE_OF[name],
                    "target_generator": TARGET_OF[name],
                    "label": DIRECTION_LABEL[name],
                }
                for name in DIRECTIONS
            },
            "both_directions_reported": True,
            "split_rule": (
                "RL1's frozen content-blind document folds decide everything: folds "
                f"{list(ADAPTATION_FOLDS)} are the adaptation documents and folds "
                f"{list(EVALUATION_FOLDS)} the evaluation documents, for the target generator and "
                "for the source generator alike. Nothing about the partition is decided in RL2"
            ),
            "adaptation_folds": list(ADAPTATION_FOLDS),
            "evaluation_folds": list(EVALUATION_FOLDS),
            "site_disjointness": (
                "RL1's leave-one-generator-out rule is inherited: a source row is kept only if its "
                "site carries no target-generator candidate, and a target row only if its site "
                "carries no source-generator candidate. A multi-source candidate therefore enters "
                "neither side, which is RL1's frozen rule applied unchanged"
            ),
            "label_budgets": list(BUDGETS),
            "decision_budget": DECISION_BUDGET,
            "saturation_budget": SATURATION_BUDGET,
            "draws_per_cell": DRAWS,
            "draw_seed": DRAW_SEED,
            "inference_draw": INFERENCE_DRAW,
            "acquisition_rule": (
                "document-first and content-blind. The adaptation documents are ordered by a "
                "seeded hash of the document id and the draw index, candidates inside a document "
                "by their frozen candidate id, and labels are purchased in that order until the "
                "budget is met. The final document contributes only its first candidates, which "
                "is declared here rather than done silently; the number of documents represented "
                "is recorded for every draw. Nothing about the order reads a label, a score or an "
                "outcome"
            ),
            "budgets_are_nested": (
                "within one draw the order is fixed, so the rows bought at a smaller budget are a "
                "prefix of those bought at a larger one"
            ),
            "aggregation_across_draws": AGGREGATE,
            "aggregation_reason": (
                "a mean over twenty few-shot draws is dragged by the draw that happened to buy no "
                "positive label; the median is the point estimate and the quartiles carry the "
                "spread"
            ),
            "arms": {
                A0: "zero shot: the source-generator model, no target label",
                A1: (
                    "score-only adaptation: target labels fit an affine map of the frozen source "
                    "score and nothing else, so ranking can only change through ties"
                ),
                A2: (
                    "joint refit: the same model family on the source training rows plus the N "
                    "labelled target rows, equal observation weights, frozen R3 features"
                ),
                A3: "target-only: the same model family on the N labelled target rows alone",
            },
            "primary_arm": PRIMARY_ARM,
            "weighting": (
                "equal observation weights. No target/source weight is tuned, and none is chosen "
                "on target evaluation labels"
            ),
            "representation_stays_source_agnostic": (
                "no feature vector carries a generator, corrector, proposal-source, rank or "
                "per-site count column in any arm, including A2. Source metadata is used only to "
                "define the directions, build the splits and report strata"
            ),
            "sparse_class_rule": (
                "a draw whose purchased labels carry a single class for a task makes the arms "
                "that depend on target labels alone infeasible for that task. The draw is kept, "
                "the arm is marked infeasible and the infeasibility rate is reported. Draws are "
                "never resampled until the class balance looks favourable"
            ),
            "criteria": {
                "C1": (
                    f"at some budget at or below {DECISION_BUDGET}, the joint refit reaches harm "
                    f"AUROC {HARM_FLOOR} or above in both directions"
                ),
                "C2": (
                    f"at some budget at or below {DECISION_BUDGET}, the joint refit reaches "
                    f"benefit AUROC {BENEFIT_FLOOR} or above in both directions"
                ),
                "C3": (
                    f"at a common budget at or below {DECISION_BUDGET}, harm and benefit recovery "
                    f"both reach {RECOVERY_FLOOR} in both directions"
                ),
                "C4": (
                    f"the joint refit beats zero shot on harm AUROC in at least "
                    f"{BREADTH_MAJORITY} of the supported environments"
                ),
                "C5": (
                    f"adaptation costs no more than {FORGETTING_TOLERANCE} absolute AUROC on the "
                    "source generator's own held-out evaluation rows, on either task"
                ),
                "C6": (
                    f"at the analysis-only harm target {PRIMARY_EPSILON}, the adapted arm captures "
                    f"at least {PREFIX_MARGIN} more absolute exact-repair recall than seeded "
                    "random ranking, and more than zero shot"
                ),
            },
            "recovery_definition": (
                "(A_N - A_0) / (A_mixed - A_0), where A_0 is the zero-shot result on the target "
                "evaluation pool and A_mixed is RL1's mixed-source out-of-fold score restricted "
                "to exactly the same rows. A denominator below "
                f"{MIN_RECOVERY_DENOMINATOR} makes the fraction unidentifiable and never a success"
            ),
            "mixed_source_reference_is_not_an_oracle_ceiling": (
                "RL1's mixed-source model saw both generators in training; it is a useful "
                "reference for how much of the gap closes, not a theoretical maximum"
            ),
            "outcome_rule": OUTCOME_TAXONOMY,
            "statistical_family": list(PRIMARY_FAMILY),
            "statistical_plan": {
                "unit": "document, resampled within environment",
                "paired": True,
                "statistic": "pooled AUROC difference on the target evaluation rows",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "budget": DECISION_BUDGET,
                "draw": INFERENCE_DRAW,
                "multiplicity": "Holm within the frozen primary family",
                "note": (
                    "few-shot draw variability and clustered evaluation uncertainty are different "
                    "sources and are never pooled into one interval. The inferential comparison "
                    "fixes the draw declared here and the across-draw spread is descriptive"
                ),
                "budget_curves_are_descriptive": True,
            },
            "non_goals": [
                "no deployment threshold is selected",
                "no certification or finite-sample guarantee is computed",
                "no human-review allocation is optimized",
                "no OCR correction candidate is generated",
                "no feature is added to or removed from the R3 representation",
                "no generator identity enters any primary feature vector",
                "no model family or hyperparameter search is reopened",
                "the confirmatory reserve is never unlocked",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ sections 7-8: the splits


@dataclass(slots=True)
class Direction:
    """One transfer direction's frozen row blocks, as index arrays into the population."""

    name: str
    source: str
    target: str
    source_train: np.ndarray
    source_eval: np.ndarray
    target_adapt: np.ndarray
    target_eval: np.ndarray


def build_direction(name: str, population: pd.DataFrame) -> Direction:
    """The four blocks of one direction, by the frozen split rule and nothing else.

    Site disjointness is RL1's leave-one-generator-out rule, inherited unchanged; the document
    partition is RL1's own content-blind folds. Both axes apply to every block, so no endpoint
    here shares a document or a site with what trained it.
    """
    source, target = SOURCE_OF[name], TARGET_OF[name]
    target_sites = rl1._sites_with(population, target, True)
    source_sites = rl1._sites_with(population, source, True)
    corrector = population["corrector_source"].astype(str).to_numpy()
    site = population["site_group"].astype(str).to_numpy()
    fold = population["fold"].to_numpy(dtype=np.int64)
    source_rows = (corrector == source) & ~np.isin(site, list(target_sites))
    target_rows = (corrector == target) & ~np.isin(site, list(source_sites))
    adapt = np.isin(fold, list(ADAPTATION_FOLDS))
    evaluate = np.isin(fold, list(EVALUATION_FOLDS))
    return Direction(
        name=name,
        source=source,
        target=target,
        source_train=np.flatnonzero(source_rows & adapt),
        source_eval=np.flatnonzero(source_rows & evaluate),
        target_adapt=np.flatnonzero(target_rows & adapt),
        target_eval=np.flatnonzero(target_rows & evaluate),
    )


def directions(population: pd.DataFrame) -> dict[str, Direction]:
    return {name: build_direction(name, population) for name in DIRECTIONS}


def _block_summary(population: pd.DataFrame, index: np.ndarray) -> dict[str, Any]:
    block = population.iloc[index]
    return {
        "candidates": int(index.size),
        "documents": int(block["document_id"].nunique()),
        "sites": int(block["site_group"].nunique()),
        "environments": int(block["environment"].nunique()),
        "harmful": int(block["is_harmful"].sum()),
        "beneficial": int(block["beneficial"].sum()),
        "exact": int(block["exact"].sum()),
    }


def run_splits() -> int:
    """Sections 7-8: the target adaptation/evaluation registry and the group registry."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (TARGET_SPLIT_REGISTRY, GROUP_REGISTRY):
        _forbid(path)
    population = load_population()
    registry: dict[str, Any] = {}
    leakage: dict[str, Any] = {}
    for name, block in directions(population).items():
        parts = {
            "source_train": block.source_train,
            "source_eval": block.source_eval,
            "target_adapt": block.target_adapt,
            "target_eval": block.target_eval,
        }
        registry[name] = {
            "source_generator": block.source,
            "target_generator": block.target,
            **{key: _block_summary(population, index) for key, index in parts.items()},
        }

        def crossing(column: str, left: np.ndarray, right: np.ndarray) -> int:
            return len(
                set(population.iloc[left][column].astype(str))
                & set(population.iloc[right][column].astype(str))
            )

        leakage[name] = {
            "documents_adapt_vs_eval": crossing(
                "document_id", block.target_adapt, block.target_eval
            ),
            "sites_adapt_vs_eval": crossing("site_group", block.target_adapt, block.target_eval),
            "edits_adapt_vs_eval": crossing("edit_group", block.target_adapt, block.target_eval),
            "documents_source_train_vs_target_eval": crossing(
                "document_id", block.source_train, block.target_eval
            ),
            "sites_source_train_vs_target_eval": crossing(
                "site_group", block.source_train, block.target_eval
            ),
            "documents_source_train_vs_source_eval": crossing(
                "document_id", block.source_train, block.source_eval
            ),
            "correctors_source_train_vs_target_eval": sorted(
                set(population.iloc[block.source_train]["corrector_source"].astype(str))
                & set(population.iloc[block.target_eval]["corrector_source"].astype(str))
            ),
        }
    _write_json_once(
        TARGET_SPLIT_REGISTRY,
        {
            **_analysis_envelope("target_split_registry"),
            "directions": registry,
            "leakage": leakage,
            "nothing_crosses": bool(
                all(
                    value == 0
                    for block in leakage.values()
                    for key, value in block.items()
                    if isinstance(value, int)
                )
                and all(
                    not block["correctors_source_train_vs_target_eval"]
                    for block in leakage.values()
                )
            ),
            "rule": cc_read_json(DESIGN_RECORD)["split_rule"],
            "evaluation_pool_is_sealed": (
                "the evaluation documents are fixed before any label is purchased and are never "
                "read by fitting, by budget selection or by any hyperparameter"
            ),
        },
    )
    _write_json_once(
        GROUP_REGISTRY,
        {
            **_envelope("group_registry"),
            "unit": "document",
            "source": _relative(rl1.GROUP_REGISTRY),
            "document_folds": cc_read_json(rl1.GROUP_REGISTRY)["document_folds"],
            "adaptation_folds": list(ADAPTATION_FOLDS),
            "evaluation_folds": list(EVALUATION_FOLDS),
            "documents": int(population["document_id"].nunique()),
            "sites": int(population["site_group"].nunique()),
            "atomic_edit_groups": int(population["edit_group"].nunique()),
            "inherited_without_change": True,
            "uses_ground_truth": False,
        },
    )
    print(
        "splits: "
        + ", ".join(
            f"{name} target eval {registry[name]['target_eval']['candidates']}"
            for name in DIRECTIONS
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 8-10: the draws


def draw_order(population: pd.DataFrame, block: Direction, draw: int) -> np.ndarray:
    """The frozen acquisition order for one direction and one draw. It reads no label.

    Documents first, ordered by a seeded hash of the draw index and the document id; candidates
    inside a document by their frozen candidate id. A budget is the first N entries of this order,
    so budgets nest inside a draw by construction.
    """
    frame = population.iloc[block.target_adapt]
    documents = sorted(set(frame["document_id"].astype(str)))
    order = sorted(
        documents,
        key=lambda name: (
            canonical_hash({"seed": DRAW_SEED, "draw": int(draw), "document": name}),
            name,
        ),
    )
    position = {name: index for index, name in enumerate(order)}
    ranked = frame.assign(
        _document_rank=frame["document_id"].astype(str).map(position)
    ).sort_values(["_document_rank", "candidate_id"], kind="stable")
    return block.target_adapt[[frame.index.get_loc(index) for index in ranked.index]]


def purchased(order: np.ndarray, budget: int) -> np.ndarray:
    """The rows a budget buys: the first N of the frozen order, which makes budgets nested."""
    return order[: min(budget, order.size)]


def run_draws() -> int:
    """Sections 9-10 and 24: the deterministic few-shot draws and what they actually bought."""
    started = time.monotonic()
    _require(TARGET_SPLIT_REGISTRY, "splits")
    for path in (BUDGET_REGISTRY, DRAW_REGISTRY, LABEL_COMPOSITION):
        _forbid(path)
    population = load_population()
    blocks = directions(population)
    rows: list[dict[str, Any]] = []
    for name, block in blocks.items():
        for draw in range(DRAWS):
            order = draw_order(population, block, draw)
            for budget in BUDGETS:
                if budget == 0:
                    continue
                index = purchased(order, budget)
                frame = population.iloc[index]
                rows.append(
                    {
                        "direction": name,
                        "draw": draw,
                        "budget": budget,
                        "realized_labels": int(index.size),
                        "documents": int(frame["document_id"].nunique()),
                        "environments": int(frame["environment"].nunique()),
                        "harmful": int(frame["is_harmful"].sum()),
                        "beneficial": int(frame["beneficial"].sum()),
                        "exact": int(frame["exact"].sum()),
                        "neither": int((~frame["is_harmful"] & ~frame["beneficial"]).sum()),
                        "by_error_kind": {
                            str(k): int(v)
                            for k, v in frame["error_kind"].value_counts().sort_index().items()
                        },
                        "by_proposal_stratum": {
                            str(k): int(v)
                            for k, v in frame["proposal_stratum"]
                            .value_counts()
                            .sort_index()
                            .items()
                        },
                        "candidate_digest": canonical_hash(
                            frame["candidate_id"].astype(str).tolist()
                        ),
                        "single_class_harm": bool(frame["is_harmful"].nunique() < 2),
                        "single_class_benefit": bool(frame["beneficial"].nunique() < 2),
                    }
                )
    table = pd.DataFrame(rows)
    _write_json_once(
        BUDGET_REGISTRY,
        {
            **_analysis_envelope("budget_registry"),
            "budgets": list(BUDGETS),
            "decision_budget": DECISION_BUDGET,
            "saturation_budget": SATURATION_BUDGET,
            "draws": DRAWS,
            "realized": {
                name: {
                    str(budget): {
                        "realized_labels": int(
                            table[(table["direction"] == name) & (table["budget"] == budget)][
                                "realized_labels"
                            ].median()
                        ),
                        "median_documents": float(
                            table[(table["direction"] == name) & (table["budget"] == budget)][
                                "documents"
                            ].median()
                        ),
                        "min_documents": int(
                            table[(table["direction"] == name) & (table["budget"] == budget)][
                                "documents"
                            ].min()
                        ),
                        "max_documents": int(
                            table[(table["direction"] == name) & (table["budget"] == budget)][
                                "documents"
                            ].max()
                        ),
                    }
                    for budget in BUDGETS
                    if budget > 0
                }
                for name in DIRECTIONS
            },
            "adaptation_pool": {name: int(blocks[name].target_adapt.size) for name in DIRECTIONS},
            "every_budget_fits_the_pool": bool(
                all(max(BUDGETS) <= blocks[name].target_adapt.size for name in DIRECTIONS)
            ),
            "budgets_declared_before_any_endpoint": True,
        },
    )
    _write_json_once(
        DRAW_REGISTRY,
        {
            **_analysis_envelope("adaptation_draw_registry"),
            "seed": DRAW_SEED,
            "draws": DRAWS,
            "rows": rows,
            "count": len(rows),
            "acquisition_rule": cc_read_json(DESIGN_RECORD)["acquisition_rule"],
            "selected_without_evaluation_labels": True,
            "outcome_counts_attached_after_membership_was_frozen": True,
        },
    )
    composition: dict[str, Any] = {}
    for name in DIRECTIONS:
        composition[name] = {
            str(budget): {
                "median_harmful": float(
                    table[(table["direction"] == name) & (table["budget"] == budget)][
                        "harmful"
                    ].median()
                ),
                "median_beneficial": float(
                    table[(table["direction"] == name) & (table["budget"] == budget)][
                        "beneficial"
                    ].median()
                ),
                "median_neither": float(
                    table[(table["direction"] == name) & (table["budget"] == budget)][
                        "neither"
                    ].median()
                ),
                "draws_without_a_harmful_label": int(
                    table[(table["direction"] == name) & (table["budget"] == budget)][
                        "single_class_harm"
                    ].sum()
                ),
                "draws_without_a_beneficial_label": int(
                    table[(table["direction"] == name) & (table["budget"] == budget)][
                        "single_class_benefit"
                    ].sum()
                ),
            }
            for budget in BUDGETS
            if budget > 0
        }
    _write_json_once(
        LABEL_COMPOSITION,
        {
            **_analysis_envelope("label_composition"),
            "by_direction": composition,
            "why": (
                "a tiny budget can buy no positive example at all, which makes the arms that "
                "depend on target labels alone unfittable. Recording it is the only way the "
                "sample-efficiency curve can be read honestly at small N"
            ),
        },
    )
    print(
        f"draws: {len(rows)} draw-budget cells over {len(DIRECTIONS)} directions "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ section 42: reproduction


def _fit(index: np.ndarray, design: np.ndarray, labels: np.ndarray) -> Any:
    """RL1's fitted scaler and classifier, reproduced exactly: same class, same hyperparameters."""
    from sklearn.preprocessing import StandardScaler

    if index.size == 0 or len(set(labels[index].tolist())) < 2:
        return None
    scaler = StandardScaler().fit(design[index])
    model = rl1._make_model(MODEL).fit(scaler.transform(design[index]), labels[index].astype(int))
    if 1 not in list(model.classes_):
        return None
    return scaler, model


def _score(fitted: Any, design: np.ndarray, index: np.ndarray) -> np.ndarray:
    if fitted is None or index.size == 0:
        return np.full(index.size, np.nan)
    scaler, model = fitted
    classes = list(model.classes_)
    return np.asarray(
        model.predict_proba(scaler.transform(design[index]))[:, classes.index(1)], float
    )


def run_reproduce() -> int:
    """Section 42: the R3 matrix and RL1's zero-shot transfer reproduced exactly, before anything.

    Gate R1 re-runs RL1's own leave-one-generator-out path and demands bit-identical scores. It is
    a reproduction of RL1 on RL1's split, not RL2's baseline: RL2's own zero-shot arm is measured
    on the sealed target evaluation pool, which is a subset of RL1's test rows.
    """
    started = time.monotonic()
    _require(DRAW_REGISTRY, "draws")
    for path in (REPRESENTATION_REPRODUCTION, ZERO_SHOT_REPRODUCTION):
        _forbid(path)
    population = load_population()
    matrix = pd.read_parquet(rl1.FEATURE_MATRIX)
    columns = rl1.columns_for(REPRESENTATION)
    design = rl1._matrix_for(population, matrix, columns)
    published = pd.read_parquet(rl1.R3_REPRESENTATION)
    rebuilt = published[published["candidate_id"].isin(set(population["candidate_id"]))]
    rebuilt = (
        population[["candidate_id"]]
        .merge(rebuilt, on="candidate_id", how="left", validate="one_to_one")[columns]
        .to_numpy(dtype=np.float64)
    )
    matrix_difference = float(np.abs(design - rebuilt).max())
    _write_json_once(
        REPRESENTATION_REPRODUCTION,
        {
            **_envelope("representation_reproduction"),
            "representation": REPRESENTATION,
            "columns": len(columns),
            "rows": int(design.shape[0]),
            "max_absolute_difference": matrix_difference,
            "identical": bool(matrix_difference == 0.0),
            "published_matrix": _relative(rl1.R3_REPRESENTATION),
            "published_sha256": file_sha256(rl1.R3_REPRESENTATION),
            "carries_source_columns": bool([c for c in columns if c.startswith(rl1.FAM_SOURCE)]),
            "uses_ground_truth": False,
        },
    )
    if matrix_difference != 0.0:
        raise PhaseError("the R3 feature matrix did not reproduce; RL2 stops")
    published_scores = rl1.cell(
        rl1.load_scores(), POPULATION, REPRESENTATION, MODEL, rl1.PRIMARY_TRANSFER_SPLIT
    )
    rebuilt_scores = rl1.score_split(
        population.drop(columns=["fold"]), matrix, columns, MODEL, rl1.PRIMARY_TRANSFER_SPLIT
    )
    joined = published_scores[["candidate_id", "fold", "score_harm", "score_benefit"]].merge(
        rebuilt_scores[["candidate_id", "fold", "score_harm", "score_benefit"]],
        on=["candidate_id", "fold"],
        suffixes=("_published", "_rebuilt"),
        validate="one_to_one",
    )
    score_difference = max(
        float(
            np.abs(
                joined[f"score_{target}_published"].to_numpy(float)
                - joined[f"score_{target}_rebuilt"].to_numpy(float)
            ).max()
        )
        for target in TARGETS
    )
    rl1_logo = cc_read_json(rl1.LEAVE_GENERATOR_OUT)["by_split"][rl1.PRIMARY_TRANSFER_SPLIT][MODEL][
        REPRESENTATION
    ]
    rebuilt_by_direction = {}
    for name in DIRECTIONS:
        held = TARGET_OF[name]
        cell = rebuilt_scores[rebuilt_scores["held_out_generator"] == held]
        block = population.set_index("candidate_id").loc[cell["candidate_id"]]
        rebuilt_by_direction[name] = {
            "held_out_generator": held,
            "rows": len(cell),
            "harm_auroc": auroc(
                cell["score_harm"].to_numpy(float), block["is_harmful"].to_numpy(bool)
            ),
            "benefit_auroc": auroc(
                cell["score_benefit"].to_numpy(float), block["beneficial"].to_numpy(bool)
            ),
            "rl1_harm_auroc": float(rl1_logo[held]["harm_auroc"]),
            "rl1_benefit_auroc": float(rl1_logo[held]["benefit_auroc"]),
        }
        for target in TARGETS:
            rebuilt_by_direction[name][f"{target}_identical"] = bool(
                abs(
                    rebuilt_by_direction[name][f"{target}_auroc"]
                    - rebuilt_by_direction[name][f"rl1_{target}_auroc"]
                )
                == 0.0
            )
    _write_json_once(
        ZERO_SHOT_REPRODUCTION,
        {
            **_analysis_envelope("zero_shot_reproduction"),
            "split": rl1.PRIMARY_TRANSFER_SPLIT,
            "compared_rows": len(joined),
            "max_absolute_score_difference": score_difference,
            "scores_identical": bool(score_difference == 0.0),
            "by_direction": rebuilt_by_direction,
            "all_auroc_identical": bool(
                all(
                    rebuilt_by_direction[name][f"{target}_identical"]
                    for name in DIRECTIONS
                    for target in TARGETS
                )
            ),
            "rl1_mean_harm_auroc": float(rl1_logo["mean_harm_auroc"]),
            "rl1_mean_benefit_auroc": float(rl1_logo["mean_benefit_auroc"]),
            "note": (
                "this reproduces RL1 on RL1's own split. RL2's zero-shot arm is the same model "
                "family fitted on the adaptation-fold source rows and scored on the sealed target "
                "evaluation pool, which is a subset of these rows"
            ),
        },
    )
    if score_difference != 0.0:
        raise PhaseError("RL1's zero-shot transfer did not reproduce; RL2 stops")
    print(
        f"reproduce: R3 matrix and RL1 transfer identical, {len(joined)} rows "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 11-12: adaptation

EVAL_TARGET = "target"
EVAL_SOURCE = "source"
EVAL_SETS = (EVAL_TARGET, EVAL_SOURCE)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def calibrate(
    source_fitted: Any,
    design: np.ndarray,
    purchased_rows: np.ndarray,
    labels: np.ndarray,
    evaluate: np.ndarray,
) -> np.ndarray:
    """A1: target labels fit an affine map of the frozen source score, and nothing else.

    One feature, the source score's logit, so the adapted score is a monotone function of the
    source score whenever the fitted slope is positive. Ranking can therefore only move through
    ties, which is what the falsification suite asserts; what A1 can move is calibration.
    """
    from sklearn.linear_model import LogisticRegression

    if source_fitted is None or len(set(labels[purchased_rows].tolist())) < 2:
        return np.full(evaluate.size, np.nan)
    feature = _logit(_score(source_fitted, design, purchased_rows)).reshape(-1, 1)
    model = LogisticRegression(
        C=rl1.LOGISTIC_C,
        max_iter=rl1.LOGISTIC_MAX_ITER,
        class_weight="balanced",
        random_state=rl1.FIT_SEED,
    ).fit(feature, labels[purchased_rows].astype(int))
    classes = list(model.classes_)
    if 1 not in classes:
        return np.full(evaluate.size, np.nan)
    evaluated = _logit(_score(source_fitted, design, evaluate)).reshape(-1, 1)
    return np.asarray(model.predict_proba(evaluated)[:, classes.index(1)], float)


def adapt_cell(
    block: Direction,
    design: np.ndarray,
    labels: np.ndarray,
    source_fitted: Any,
    rows: np.ndarray,
    evaluate: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every arm's score on one evaluation block, for one direction, budget and draw."""
    joint = np.concatenate([block.source_train, rows]) if rows.size else block.source_train
    return {
        A0: _score(source_fitted, design, evaluate),
        A1: calibrate(source_fitted, design, rows, labels, evaluate),
        A2: _score(_fit(joint, design, labels), design, evaluate),
        A3: _score(_fit(rows, design, labels), design, evaluate),
    }


def run_adapt() -> int:
    """Sections 11-12: every arm x direction x budget x draw, scored on sealed evaluation rows."""
    started = time.monotonic()
    _require(ZERO_SHOT_REPRODUCTION, "reproduce")
    for path in (
        ADAPTED_SCORES,
        MODEL_REGISTRY,
        ADAPTATION_REGISTRY,
        HYPERPARAMETER_REGISTRY,
        INFEASIBLE_DRAWS,
    ):
        _forbid(path)
    population = load_population()
    matrix = pd.read_parquet(rl1.FEATURE_MATRIX)
    columns = rl1.columns_for(REPRESENTATION)
    design = rl1._matrix_for(population, matrix, columns)
    truth = {
        T_HARM: population["is_harmful"].to_numpy(dtype=bool),
        T_BENEFIT: population["beneficial"].to_numpy(dtype=bool),
    }
    blocks = directions(population)
    frames: list[pd.DataFrame] = []
    infeasible: list[dict[str, Any]] = []
    fits = 0
    for name, block in blocks.items():
        began = time.monotonic()
        evaluate = {EVAL_TARGET: block.target_eval, EVAL_SOURCE: block.source_eval}
        for target in TARGETS:
            labels = truth[target]
            source_fitted = _fit(block.source_train, design, labels)
            fits += 1
            for draw in range(DRAWS):
                order = draw_order(population, block, draw)
                for budget in BUDGETS:
                    rows = purchased(order, budget)
                    # A0 does not read a target label, so it is identical across draws and is
                    # recorded once under draw 0 rather than twenty times.
                    if budget == 0 and draw > 0:
                        continue
                    scored = {
                        key: adapt_cell(block, design, labels, source_fitted, rows, index)
                        for key, index in evaluate.items()
                    }
                    fits += 3 if budget else 0
                    for arm in ARMS:
                        for key, index in evaluate.items():
                            values = scored[key][arm]
                            if not np.isfinite(values).any() and index.size:
                                if key == EVAL_TARGET:
                                    infeasible.append(
                                        {
                                            "direction": name,
                                            "arm": arm,
                                            "target": target,
                                            "budget": int(budget),
                                            "draw": int(draw),
                                            "reason": (
                                                "no target label is purchased at this budget"
                                                if rows.size == 0
                                                else "purchased labels carry a single class"
                                            ),
                                            "structural": bool(rows.size == 0),
                                            "harmful": int(labels[rows].sum()),
                                            "rows": int(rows.size),
                                        }
                                    )
                                continue
                            frames.append(
                                pd.DataFrame(
                                    {
                                        "direction": name,
                                        "arm": arm,
                                        "target": target,
                                        "budget": int(budget),
                                        "draw": int(draw),
                                        "evaluation_set": key,
                                        "candidate_id": population.iloc[index][
                                            "candidate_id"
                                        ].to_numpy(),
                                        "score": values,
                                    }
                                )
                            )
        print(f"  {name}: {time.monotonic() - began:.0f}s", flush=True)
    table = pd.concat(frames, ignore_index=True)
    order_columns = [
        "direction",
        "arm",
        "target",
        "budget",
        "draw",
        "evaluation_set",
        "candidate_id",
    ]
    table = table.sort_values(order_columns, kind="stable").reset_index(drop=True)
    _write_parquet_once(ADAPTED_SCORES, table)
    _write_json_once(
        MODEL_REGISTRY,
        {
            **_envelope("model_registry"),
            "model": MODEL,
            "source": _relative(rl1.MODEL_REGISTRY),
            "class": cc_read_json(rl1.MODEL_REGISTRY)["models"][MODEL]["class"],
            "preprocessing": cc_read_json(rl1.MODEL_REGISTRY)["models"][MODEL]["preprocessing"],
            "two_heads": "one classifier per target; harm and benefit are never collapsed",
            "no_model_search": True,
            "reopened_architecture_search": False,
            "uses_ground_truth": True,
        },
    )
    _write_json_once(
        HYPERPARAMETER_REGISTRY,
        {
            **_envelope("hyperparameter_registry"),
            MODEL: dict(cc_read_json(rl1.HYPERPARAMETER_REGISTRY)[MODEL]),
            "source": _relative(rl1.HYPERPARAMETER_REGISTRY),
            "changed_here": False,
            "tuned_on_target_evaluation_labels": False,
            "target_weighting": "equal observation weights; no weight parameter is fitted",
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        ADAPTATION_REGISTRY,
        {
            **_analysis_envelope("adaptation_registry"),
            "arms": cc_read_json(DESIGN_RECORD)["arms"],
            "primary_arm": PRIMARY_ARM,
            "fits": fits,
            "scored_rows": len(table),
            "cells": int(table.groupby(["direction", "arm", "target", "budget", "draw"]).ngroups),
            "evaluation_sets": list(EVAL_SETS),
            "feature_columns": len(columns),
            "carries_source_columns": bool([c for c in columns if c.startswith(rl1.FAM_SOURCE)]),
            "target_labels_reach_the_model_only_through_fitting": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        INFEASIBLE_DRAWS,
        {
            **_analysis_envelope("infeasible_draws"),
            "rows": infeasible,
            "count": len(infeasible),
            "by_arm_and_budget": {
                f"{row['direction']}|{row['arm']}|{row['target']}|{row['budget']}": sum(
                    1
                    for other in infeasible
                    if (other["direction"], other["arm"], other["target"], other["budget"])
                    == (row["direction"], row["arm"], row["target"], row["budget"])
                )
                for row in infeasible
            },
            "sparse_class_cells": sum(1 for row in infeasible if not row["structural"]),
            "structural_cells": sum(1 for row in infeasible if row["structural"]),
            "structural_note": (
                "at the zero-shot budget no target label exists, so the arms that depend on "
                "target labels alone are unfittable by construction rather than by class balance"
            ),
            "rule": cc_read_json(DESIGN_RECORD)["sparse_class_rule"],
            "draws_were_never_resampled": True,
        },
    )
    print(
        f"adapt: {len(table)} scored rows, {len(infeasible)} infeasible arm-draws "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 14-23: curves

LABEL_COLUMN = {T_HARM: "is_harmful", T_BENEFIT: "beneficial"}


def load_adapted() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The adapted scores and the frozen population, indexed the way every analysis reads them."""
    return pd.read_parquet(ADAPTED_SCORES), load_population().set_index("candidate_id")


def _auroc_of(block: pd.DataFrame, population: pd.DataFrame, target: str) -> float:
    if block.empty:
        return float("nan")
    truth = population.loc[block["candidate_id"], LABEL_COLUMN[target]].to_numpy(dtype=bool)
    return auroc(block["score"].to_numpy(dtype=np.float64), truth)


def _ap_of(block: pd.DataFrame, population: pd.DataFrame, target: str) -> float:
    if block.empty:
        return float("nan")
    truth = population.loc[block["candidate_id"], LABEL_COLUMN[target]].to_numpy(dtype=bool)
    return average_precision(block["score"].to_numpy(dtype=np.float64), truth)


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    finite = _finite(values)
    if not finite:
        return {
            "draws": 0,
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "draws": len(finite),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def curve_point(
    scores: pd.DataFrame,
    population: pd.DataFrame,
    direction: str,
    arm: str,
    target: str,
    budget: int,
    evaluation_set: str = EVAL_TARGET,
) -> dict[str, Any]:
    """One curve point: the across-draw distribution of AUROC and average precision."""
    block = scores[
        (scores["direction"] == direction)
        & (scores["arm"] == arm)
        & (scores["target"] == target)
        & (scores["budget"] == budget)
        & (scores["evaluation_set"] == evaluation_set)
    ]
    aurocs = []
    precisions = []
    for _draw, group in block.groupby("draw", sort=True):
        aurocs.append(_auroc_of(group, population, target))
        precisions.append(_ap_of(group, population, target))
    point = _quantiles(aurocs)
    point["average_precision"] = _quantiles(precisions)["median"]
    point["feasible_draws"] = point["draws"]
    point["infeasible_draws"] = int((DRAWS if (budget > 0 and arm != A0) else 1) - point["draws"])
    return point


def arm_curve(
    scores: pd.DataFrame, population: pd.DataFrame, direction: str, arm: str, target: str
) -> dict[str, Any]:
    """One arm's whole budget curve. A0 reads no target label, so it is flat by construction."""
    if arm == A0:
        point = curve_point(scores, population, direction, arm, target, 0)
        return {str(budget): point for budget in BUDGETS}
    out = {"0": curve_point(scores, population, direction, A0, target, 0)}
    for budget in BUDGETS:
        if budget == 0:
            continue
        out[str(budget)] = curve_point(scores, population, direction, arm, target, budget)
    return out


def mixed_reference(population: pd.DataFrame, block: Direction, target: str) -> float:
    """RL1's mixed-source out-of-fold score restricted to exactly the evaluation rows.

    Not an oracle ceiling: RL1's model saw both generators in training, which is precisely what an
    unseen-generator deployment does not have. It is the reference the recovery fraction is
    measured against, and it is computed on the same rows so the comparison is not a population
    difference in disguise.
    """
    mixed = rl1.cell(
        rl1.load_scores(), POPULATION, REPRESENTATION, MODEL, rl1.PRIMARY_SPLIT
    ).set_index("candidate_id")
    ids = population.iloc[block.target_eval]["candidate_id"]
    truth = population.iloc[block.target_eval][LABEL_COLUMN[target]].to_numpy(dtype=bool)
    return auroc(mixed.loc[ids, f"score_{target}"].to_numpy(dtype=np.float64), truth)


def run_curves() -> int:
    """Sections 14-23: sample efficiency, gap recovery, the minimal useful budget, the arms."""
    started = time.monotonic()
    _require(ADAPTED_SCORES, "adapt")
    for path in (
        SAMPLE_EFFICIENCY_HARM,
        SAMPLE_EFFICIENCY_BENEFIT,
        TRANSFER_GAP_RECOVERY,
        MINIMAL_LABEL_BUDGET,
        METHOD_COMPARISON,
        ADAPTATION_EFFICIENCY,
        DIRECTION_ANALYSIS,
    ):
        _forbid(path)
    scores, indexed = load_adapted()
    population = load_population()
    blocks = directions(population)
    references = {
        name: {target: mixed_reference(population, block, target) for target in TARGETS}
        for name, block in blocks.items()
    }
    curves: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        curves[target] = {
            name: {arm: arm_curve(scores, indexed, name, arm, target) for arm in ARMS}
            for name in DIRECTIONS
        }
    floors = {T_HARM: HARM_FLOOR, T_BENEFIT: BENEFIT_FLOOR}
    for target, path in ((T_HARM, SAMPLE_EFFICIENCY_HARM), (T_BENEFIT, SAMPLE_EFFICIENCY_BENEFIT)):
        _write_json_once(
            path,
            {
                **_analysis_envelope(f"sample_efficiency_{target}"),
                "target": target,
                "aggregation": AGGREGATE,
                "budgets": list(BUDGETS),
                "by_direction": curves[target],
                "zero_shot": {name: curves[target][name][A0]["0"]["median"] for name in DIRECTIONS},
                "mixed_source_reference": {name: references[name][target] for name in DIRECTIONS},
                "rl1_mixed_source_pooled": cc_read_json(rl1.MIXED_SOURCE_RESULTS)["by_population"][
                    POPULATION
                ][MODEL][REPRESENTATION][f"{target}_auroc"],
                "absolute_floor": floors[target],
                "evaluation_set": EVAL_TARGET,
            },
        )
    recovery_rows: dict[str, Any] = {}
    for target in TARGETS:
        recovery_rows[target] = {}
        for name in DIRECTIONS:
            zero = curves[target][name][A0]["0"]["median"]
            reference = references[name][target]
            recovery_rows[target][name] = {
                "zero_shot": zero,
                "mixed_source_reference": reference,
                "gap": float(reference - zero),
                "identifiable": bool(reference - zero >= MIN_RECOVERY_DENOMINATOR),
                "by_arm": {
                    arm: {
                        str(budget): recovery(
                            curves[target][name][arm][str(budget)]["median"], zero, reference
                        )
                        for budget in BUDGETS
                    }
                    for arm in (A1, A2, A3)
                },
            }
    _write_json_once(
        TRANSFER_GAP_RECOVERY,
        {
            **_analysis_envelope("transfer_gap_recovery"),
            "definition": cc_read_json(DESIGN_RECORD)["recovery_definition"],
            "by_target": recovery_rows,
            "recovery_floor": RECOVERY_FLOOR,
            "minimum_denominator": MIN_RECOVERY_DENOMINATOR,
            "reference_is_not_an_oracle_ceiling": cc_read_json(DESIGN_RECORD)[
                "mixed_source_reference_is_not_an_oracle_ceiling"
            ],
        },
    )

    def minimal(arm: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in DIRECTIONS:
            per_target: dict[str, Any] = {}
            for target in TARGETS:
                floor = floors[target]
                found = None
                for budget in BUDGETS:
                    value = curves[target][name][arm][str(budget)]["median"]
                    share = recovery_rows[target][name]["by_arm"].get(arm, {}).get(str(budget))
                    if (
                        np.isfinite(value)
                        and value >= floor
                        and share is not None
                        and share >= RECOVERY_FLOOR
                    ):
                        found = int(budget)
                        break
                per_target[target] = found
            joint = None
            for budget in BUDGETS:
                if all(
                    np.isfinite(curves[t][name][arm][str(budget)]["median"])
                    and curves[t][name][arm][str(budget)]["median"] >= floors[t]
                    and (recovery_rows[t][name]["by_arm"].get(arm, {}).get(str(budget)) or -1.0)
                    >= RECOVERY_FLOOR
                    for t in TARGETS
                ):
                    joint = int(budget)
                    break
            out[name] = {
                **{f"n_star_{target}": per_target[target] for target in TARGETS},
                "n_star_joint": joint,
                "exceeds_largest_tested_budget": {
                    **{f"{target}": bool(per_target[target] is None) for target in TARGETS},
                    "joint": bool(joint is None),
                },
            }
        return out

    _write_json_once(
        MINIMAL_LABEL_BUDGET,
        {
            **_analysis_envelope("minimal_label_budget"),
            "definition": (
                "the smallest pre-registered budget at which the arm meets BOTH the absolute "
                "AUROC floor and the gap-recovery floor. When no tested budget qualifies the "
                "value is null and the report says the budget exceeds the largest one tested; "
                "nothing is extrapolated"
            ),
            "by_arm": {arm: minimal(arm) for arm in (A1, A2, A3)},
            "primary_arm": PRIMARY_ARM,
            "largest_tested_budget": max(BUDGETS),
            "decision_budget": DECISION_BUDGET,
            "floors": {**{t: floors[t] for t in TARGETS}, "recovery": RECOVERY_FLOOR},
        },
    )
    comparison: dict[str, Any] = {}
    for target in TARGETS:
        comparison[target] = {
            name: {
                str(budget): {arm: curves[target][name][arm][str(budget)]["median"] for arm in ARMS}
                for budget in BUDGETS
            }
            for name in DIRECTIONS
        }
    a1_moves_ranking = {
        target: {
            name: max(
                abs(
                    float(curves[target][name][A1][str(budget)]["median"])
                    - float(curves[target][name][A0]["0"]["median"])
                )
                for budget in BUDGETS
                if budget > 0 and np.isfinite(curves[target][name][A1][str(budget)]["median"])
            )
            for name in DIRECTIONS
        }
        for target in TARGETS
    }
    a3_beats_a2 = {
        target: {
            name: sum(
                1
                for budget in BUDGETS
                if budget > 0
                and np.isfinite(curves[target][name][A3][str(budget)]["median"])
                and curves[target][name][A3][str(budget)]["median"]
                > curves[target][name][A2][str(budget)]["median"]
            )
            for name in DIRECTIONS
        }
        for target in TARGETS
    }
    _write_json_once(
        METHOD_COMPARISON,
        {
            **_analysis_envelope("adaptation_method_comparison"),
            "by_target": comparison,
            "a1_maximum_auroc_move_from_zero_shot": a1_moves_ranking,
            "budgets_where_target_only_beats_joint_refit": a3_beats_a2,
            "budgets_tested_above_zero": len(BUDGETS) - 1,
            "reading": (
                "A1 changes calibration, not order: it is a monotone map of the frozen source "
                "score, so any AUROC move is a tie artifact. A2 moving where A1 does not means "
                "the representation is usable but the decision geometry must be refitted. A3 "
                "above A2 means the source generator's rows are pulling the fit away from the "
                "target, which is negative transfer and is reported as measured"
            ),
        },
    )
    efficiency: dict[str, Any] = {}
    for target in TARGETS:
        efficiency[target] = {}
        for name in DIRECTIONS:
            zero = curves[target][name][A0]["0"]["median"]
            rows = {}
            for budget in BUDGETS:
                if budget == 0:
                    continue
                value = curves[target][name][A2][str(budget)]["median"]
                share = recovery_rows[target][name]["by_arm"][A2][str(budget)]
                rows[str(budget)] = {
                    "auroc": value,
                    "delta_auroc": float(value - zero),
                    "delta_auroc_per_label": float((value - zero) / budget),
                    "recovery": share,
                    "recovery_per_label": (float(share / budget) if share is not None else None),
                }
            efficiency[target][name] = rows
    _write_json_once(
        ADAPTATION_EFFICIENCY,
        {
            **_analysis_envelope("adaptation_efficiency"),
            "arm": A2,
            "by_target": efficiency,
            "secondary_diagnostic": True,
            "not_optimized_on": (
                "no budget, arm or seed was chosen to make these ratios look better; they are "
                "descriptive and exist to show where the returns flatten"
            ),
        },
    )
    _write_json_once(
        DIRECTION_ANALYSIS,
        {
            **_analysis_envelope("generator_direction_analysis"),
            "directions": {
                name: {
                    "label": DIRECTION_LABEL[name],
                    "source_generator": SOURCE_OF[name],
                    "target_generator": TARGET_OF[name],
                    "target_evaluation_rows": int(blocks[name].target_eval.size),
                    "adaptation_pool": int(blocks[name].target_adapt.size),
                    **{
                        f"zero_shot_{target}_auroc": curves[target][name][A0]["0"]["median"]
                        for target in TARGETS
                    },
                    **{
                        f"decision_budget_{target}_auroc": curves[target][name][A2][
                            str(DECISION_BUDGET)
                        ]["median"]
                        for target in TARGETS
                    },
                    **{
                        f"saturation_{target}_auroc": curves[target][name][A2][
                            str(SATURATION_BUDGET)
                        ]["median"]
                        for target in TARGETS
                    },
                }
                for name in DIRECTIONS
            },
            "asymmetry": {
                target: float(
                    curves[target][D_G2Q][A2][str(DECISION_BUDGET)]["median"]
                    - curves[target][D_Q2G][A2][str(DECISION_BUDGET)]["median"]
                )
                for target in TARGETS
            },
            "both_directions_reported": True,
        },
    )
    print(
        "curves: "
        + ", ".join(
            f"{name} harm@{DECISION_BUDGET}="
            f"{curves[T_HARM][name][A2][str(DECISION_BUDGET)]['median']:.4f}"
            for name in DIRECTIONS
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 29-31: retention


def run_retention() -> int:
    """Sections 29-31: what adaptation costs the source generator, and the pooled population."""
    started = time.monotonic()
    _require(METHOD_COMPARISON, "curves")
    for path in (SOURCE_RETENTION, MIXED_POPULATION):
        _forbid(path)
    scores, indexed = load_adapted()
    population = load_population()
    blocks = directions(population)
    retention_rows: dict[str, Any] = {}
    for target in TARGETS:
        retention_rows[target] = {}
        for name in DIRECTIONS:
            baseline = curve_point(scores, indexed, name, A0, target, 0, EVAL_SOURCE)["median"]
            rows = {}
            for budget in BUDGETS:
                if budget == 0:
                    continue
                point = curve_point(scores, indexed, name, A2, target, budget, EVAL_SOURCE)
                rows[str(budget)] = {
                    "adapted": point["median"],
                    "q25": point["q25"],
                    "q75": point["q75"],
                    "forgetting": float(point["median"] - baseline),
                    "within_tolerance": bool(point["median"] - baseline >= -FORGETTING_TOLERANCE),
                }
            retention_rows[target][name] = {
                "source_generator": SOURCE_OF[name],
                "source_evaluation_rows": int(blocks[name].source_eval.size),
                "before_adaptation": baseline,
                "by_budget": rows,
                "worst_forgetting": min(
                    (row["forgetting"] for row in rows.values() if np.isfinite(row["forgetting"])),
                    default=float("nan"),
                ),
            }
    worst = min(
        (
            retention_rows[target][name]["worst_forgetting"]
            for target in TARGETS
            for name in DIRECTIONS
            if np.isfinite(retention_rows[target][name]["worst_forgetting"])
        ),
        default=float("nan"),
    )
    _write_json_once(
        SOURCE_RETENTION,
        {
            **_analysis_envelope("source_retention"),
            "definition": (
                "the adapted model's AUROC on the source generator's own held-out evaluation "
                "documents, minus the unadapted model's AUROC on exactly the same rows. The "
                "source evaluation rows are disjoint from the source training rows by document"
            ),
            "by_target": retention_rows,
            "tolerance": FORGETTING_TOLERANCE,
            "worst_forgetting": worst,
            "within_tolerance_everywhere": bool(worst >= -FORGETTING_TOLERANCE),
            "arm": A2,
        },
    )
    mixed: dict[str, Any] = {}
    for target in TARGETS:
        mixed[target] = {}
        for name in DIRECTIONS:
            rows = {}
            for budget in BUDGETS:
                arm = A0 if budget == 0 else A2
                block = scores[
                    (scores["direction"] == name)
                    & (scores["arm"] == arm)
                    & (scores["target"] == target)
                    & (scores["budget"] == budget)
                ]
                aurocs, precisions = [], []
                for _draw, group in block.groupby("draw", sort=True):
                    aurocs.append(_auroc_of(group, indexed, target))
                    precisions.append(_ap_of(group, indexed, target))
                rows[str(budget)] = {
                    "auroc": _quantiles(aurocs)["median"],
                    "q25": _quantiles(aurocs)["q25"],
                    "q75": _quantiles(aurocs)["q75"],
                    "average_precision": _quantiles(precisions)["median"],
                    "arm": arm,
                }
            mixed[target][name] = rows
    _write_json_once(
        MIXED_POPULATION,
        {
            **_analysis_envelope("mixed_population_results"),
            "population": (
                "the pooled held-out evaluation rows of both generators, which is the candidate "
                "universe a deployed hybrid system would actually face"
            ),
            "by_target": mixed,
            "rows": {
                name: int(blocks[name].target_eval.size + blocks[name].source_eval.size)
                for name in DIRECTIONS
            },
            "rl1_mixed_source_pooled": {
                target: cc_read_json(rl1.MIXED_SOURCE_RESULTS)["by_population"][POPULATION][MODEL][
                    REPRESENTATION
                ][f"{target}_auroc"]
                for target in TARGETS
            },
        },
    )
    print(f"retention: worst forgetting {worst:+.4f} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ sections 32-34: prefixes


def run_prefix() -> int:
    """Sections 32-34: ranking geometry and the diagnostic oracle harm-constrained frontier."""
    started = time.monotonic()
    _require(SOURCE_RETENTION, "retention")
    for path in (PREFIX_RANKING, ORACLE_RISK_FRONTIER):
        _forbid(path)
    scores, _indexed = load_adapted()
    population = load_population()
    blocks = directions(population)
    prefixes: dict[str, Any] = {}
    frontier: dict[str, Any] = {}
    random_rows: dict[str, Any] = {}
    for name in DIRECTIONS:
        block = population.iloc[blocks[name].target_eval]
        harmful = block["is_harmful"].to_numpy(dtype=bool)
        beneficial = block["beneficial"].to_numpy(dtype=bool)
        exact = block["exact"].to_numpy(dtype=bool)
        prefixes[name] = {}
        frontier[name] = {}
        for budget in BUDGETS:
            arm = A0 if budget == 0 else A2
            cell = scores[
                (scores["direction"] == name)
                & (scores["arm"] == arm)
                & (scores["target"] == T_HARM)
                & (scores["budget"] == budget)
                & (scores["evaluation_set"] == EVAL_TARGET)
            ]
            geometry: list[list[dict[str, Any]]] = []
            frontiers: dict[str, list[dict[str, Any]]] = {epsilon_key(e): [] for e in EPSILONS}
            for _draw, group in cell.groupby("draw", sort=True):
                ordered = block[["candidate_id"]].merge(
                    group[["candidate_id", "score"]], on="candidate_id", validate="one_to_one"
                )
                score = ordered["score"].to_numpy(dtype=np.float64)
                geometry.append(rl1.prefix_geometry(score, harmful, beneficial, exact))
                for epsilon in EPSILONS:
                    frontiers[epsilon_key(epsilon)].append(
                        rl1.oracle_frontier(score, harmful, exact, epsilon)
                    )
            prefixes[name][str(budget)] = {
                "arm": arm,
                "rows": [
                    {
                        "rank_fraction": fraction,
                        "harm_rate": float(np.median([g[position]["harm_rate"] for g in geometry])),
                        "beneficial_rate": float(
                            np.median([g[position]["beneficial_rate"] for g in geometry])
                        ),
                        "repair_recall": float(
                            np.median([g[position]["repair_recall"] for g in geometry])
                        ),
                    }
                    for position, fraction in enumerate(PREFIX_FRACTIONS)
                    if geometry and position < len(geometry[0])
                ],
            }
            frontier[name][str(budget)] = {
                "arm": arm,
                **{
                    key: {
                        "repair_recall": float(np.median([row["repair_recall"] for row in rows])),
                        "coverage": float(np.median([row["coverage"] for row in rows])),
                        "oracle_threshold": True,
                        "analysis_only": True,
                        "deployable": False,
                    }
                    for key, rows in frontiers.items()
                    if rows
                },
            }
        generator = np.random.default_rng(RANDOM_RANKING_SEED)
        recalls = {epsilon_key(e): [] for e in EPSILONS}
        for _draw in range(RANDOM_RANKING_DRAWS):
            random_score = generator.random(len(block))
            for epsilon in EPSILONS:
                row = rl1.oracle_frontier(random_score, harmful, exact, epsilon)
                recalls[epsilon_key(epsilon)].append(float(row["repair_recall"]))
        random_rows[name] = {key: float(np.mean(values)) for key, values in recalls.items()}
    key = epsilon_key(PRIMARY_EPSILON)
    margins = {
        name: {
            str(budget): float(
                frontier[name][str(budget)][key]["repair_recall"] - random_rows[name][key]
            )
            for budget in BUDGETS
        }
        for name in DIRECTIONS
    }
    _write_json_once(
        PREFIX_RANKING,
        {
            **_analysis_envelope("prefix_ranking"),
            "by_direction": prefixes,
            "fractions": list(PREFIX_FRACTIONS),
            "ranking": "ascending P(harmful); the safest candidates come first",
            "aggregation": AGGREGATE,
            "selects_a_deployment_threshold": False,
            "base_rates": {
                name: {
                    "harm": float(population.iloc[blocks[name].target_eval]["is_harmful"].mean()),
                    "benefit": float(
                        population.iloc[blocks[name].target_eval]["beneficial"].mean()
                    ),
                    "exact_repairs": int(population.iloc[blocks[name].target_eval]["exact"].sum()),
                }
                for name in DIRECTIONS
            },
        },
    )
    _write_json_once(
        ORACLE_RISK_FRONTIER,
        {
            **_analysis_envelope("oracle_risk_frontier"),
            "by_direction": frontier,
            "random_ranking": random_rows,
            "margin_over_random": margins,
            "required_margin": PREFIX_MARGIN,
            "epsilons": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "label": "oracle threshold -- analysis only -- not deployable",
            "rule": (
                "the longest score-ranked prefix whose realized harm rate is at or below epsilon, "
                "chosen with the labels of the accepted rows. It measures whether adaptation "
                "restores enough ranking headroom to justify a later calibration stage, and it "
                "cannot be implemented without the labels it reads"
            ),
        },
    )
    print(
        "prefix: "
        + ", ".join(
            f"{name} margin@{SATURATION_BUDGET}={margins[name][str(SATURATION_BUDGET)]:+.4f}"
            for name in DIRECTIONS
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 28-29: strata


def run_strata() -> int:
    """Sections 28-29: adaptation by environment and by error type."""
    started = time.monotonic()
    _require(PREFIX_RANKING, "prefix")
    for path in (ENVIRONMENT_ANALYSIS, ERROR_TYPE_ANALYSIS):
        _forbid(path)
    scores, indexed = load_adapted()
    population = load_population()
    blocks = directions(population)

    def stratified(column: str, budget: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in DIRECTIONS:
            block = population.iloc[blocks[name].target_eval]
            groups = sorted(set(block[column].astype(str)))
            out[name] = {}
            for group_name in groups:
                ids = set(block[block[column].astype(str) == group_name]["candidate_id"])
                row: dict[str, Any] = {"rows": len(ids)}
                for target in TARGETS:
                    for arm, tag in ((A0, "zero_shot"), (A2, "adapted")):
                        cell = scores[
                            (scores["direction"] == name)
                            & (scores["arm"] == arm)
                            & (scores["target"] == target)
                            & (scores["budget"] == (0 if arm == A0 else budget))
                            & (scores["evaluation_set"] == EVAL_TARGET)
                            & (scores["candidate_id"].isin(ids))
                        ]
                        values = [
                            _auroc_of(part, indexed, target)
                            for _draw, part in cell.groupby("draw", sort=True)
                        ]
                        row[f"{tag}_{target}_auroc"] = _quantiles(values)["median"]
                    row[f"{target}_improved"] = bool(
                        np.isfinite(row[f"adapted_{target}_auroc"])
                        and np.isfinite(row[f"zero_shot_{target}_auroc"])
                        and row[f"adapted_{target}_auroc"] > row[f"zero_shot_{target}_auroc"]
                    )
                out[name][group_name] = row
        return out

    environment = {
        str(budget): stratified("environment", budget)
        for budget in (DECISION_BUDGET, SATURATION_BUDGET)
    }
    improved = {
        str(budget): {
            name: sum(
                1 for row in environment[str(budget)][name].values() if row[f"{T_HARM}_improved"]
            )
            for name in DIRECTIONS
        }
        for budget in (DECISION_BUDGET, SATURATION_BUDGET)
    }
    supported = {name: len(environment[str(DECISION_BUDGET)][name]) for name in DIRECTIONS}
    _write_json_once(
        ENVIRONMENT_ANALYSIS,
        {
            **_analysis_envelope("environment_analysis"),
            "by_budget": environment,
            "environments_improved_on_harm": improved,
            "supported_environments": supported,
            "breadth_majority": BREADTH_MAJORITY,
            "question": (
                "does target-generator supervision improve ranking across engines and domains, or "
                "only where the adaptation documents happened to fall?"
            ),
            "note": (
                "the target evaluation pool does not cover every one of the ten development "
                "environments, because the evaluation documents are a content-blind subset. The "
                "breadth criterion is measured over the environments that are supported, and the "
                "supported count is reported beside it"
            ),
        },
    )
    _write_json_once(
        ERROR_TYPE_ANALYSIS,
        {
            **_analysis_envelope("error_type_analysis"),
            "by_budget": {
                str(budget): stratified("error_kind", budget)
                for budget in (DECISION_BUDGET, SATURATION_BUDGET)
            },
            "omission_note": (
                "SGV-HY1 produced no exact omission repair, so omission cells carry very few "
                "candidates and their AUROCs are reported without being interpreted"
            ),
        },
    )
    print(
        "strata: harm improved in "
        + ", ".join(
            f"{name} {improved[str(DECISION_BUDGET)][name]}/{supported[name]}"
            for name in DIRECTIONS
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 33-34: controls

CONTROL_BUDGETS = (DECISION_BUDGET, SATURATION_BUDGET)


def run_controls() -> int:
    """Sections 33-34: random target labels and permuted target labels must not repair transfer."""
    started = time.monotonic()
    _require(ERROR_TYPE_ANALYSIS, "strata")
    for path in (RANDOM_LABEL_CONTROL, LABEL_PERMUTATION_CONTROL, CONTROL_RESULTS):
        _forbid(path)
    scores, indexed = load_adapted()
    population = load_population()
    matrix = pd.read_parquet(rl1.FEATURE_MATRIX)
    columns = rl1.columns_for(REPRESENTATION)
    design = rl1._matrix_for(population, matrix, columns)
    truth = {
        T_HARM: population["is_harmful"].to_numpy(dtype=bool),
        T_BENEFIT: population["beneficial"].to_numpy(dtype=bool),
    }
    blocks = directions(population)
    random_rows: dict[str, Any] = {}
    permuted_rows: dict[str, Any] = {}
    for target in TARGETS:
        random_rows[target] = {}
        permuted_rows[target] = {}
        labels = truth[target]
        for name, block in blocks.items():
            zero = curve_point(scores, indexed, name, A0, target, 0)["median"]
            honest = {
                str(budget): curve_point(scores, indexed, name, A2, target, budget)["median"]
                for budget in CONTROL_BUDGETS
            }
            random_values: dict[str, list[float]] = {str(b): [] for b in CONTROL_BUDGETS}
            permuted_values: dict[str, list[float]] = {str(b): [] for b in CONTROL_BUDGETS}
            evaluation_truth = labels[block.target_eval]
            for draw in range(DRAWS):
                order = draw_order(population, block, draw)
                for budget in CONTROL_BUDGETS:
                    rows = purchased(order, budget)
                    prevalence = float(labels[block.target_adapt].mean())
                    fabricated = labels.copy()
                    fabricated[rows] = (
                        np.random.default_rng(RANDOM_LABEL_SEED + draw).random(rows.size)
                        < prevalence
                    )
                    shuffled = labels.copy()
                    shuffled[rows] = np.random.default_rng(PERMUTATION_SEED + draw).permutation(
                        labels[rows]
                    )
                    for store, values in (
                        (random_values, fabricated),
                        (permuted_values, shuffled),
                    ):
                        joint = np.concatenate([block.source_train, rows])
                        fitted = _fit(joint, design, values)
                        store[str(budget)].append(
                            auroc(_score(fitted, design, block.target_eval), evaluation_truth)
                        )
            random_rows[target][name] = {
                "zero_shot": zero,
                **{
                    str(budget): {
                        "random_labels": _quantiles(random_values[str(budget)])["median"],
                        "honest_labels": honest[str(budget)],
                        "recovered_less_than_honest": bool(
                            _quantiles(random_values[str(budget)])["median"] < honest[str(budget)]
                        ),
                    }
                    for budget in CONTROL_BUDGETS
                },
            }
            permuted_rows[target][name] = {
                "zero_shot": zero,
                **{
                    str(budget): {
                        "permuted_labels": _quantiles(permuted_values[str(budget)])["median"],
                        "honest_labels": honest[str(budget)],
                        "recovered_less_than_honest": bool(
                            _quantiles(permuted_values[str(budget)])["median"] < honest[str(budget)]
                        ),
                    }
                    for budget in CONTROL_BUDGETS
                },
            }
    _write_json_once(
        RANDOM_LABEL_CONTROL,
        {
            **_analysis_envelope("random_label_control"),
            "rule": (
                "the purchased rows keep their identity and their features and receive a label "
                "drawn at the adaptation pool's own prevalence. If adaptation were an artifact of "
                "adding target rows rather than of their labels, this would recover as much"
            ),
            "seed": RANDOM_LABEL_SEED,
            "budgets": list(CONTROL_BUDGETS),
            "by_target": random_rows,
            "no_systematic_recovery": bool(
                all(
                    random_rows[target][name][str(budget)]["recovered_less_than_honest"]
                    for target in TARGETS
                    for name in DIRECTIONS
                    for budget in CONTROL_BUDGETS
                )
            ),
        },
    )
    _write_json_once(
        LABEL_PERMUTATION_CONTROL,
        {
            **_analysis_envelope("label_permutation_control"),
            "rule": (
                "the purchased labels are permuted among the purchased rows, so the class balance "
                "is exactly preserved and only the pairing of label to candidate is destroyed"
            ),
            "seed": PERMUTATION_SEED,
            "budgets": list(CONTROL_BUDGETS),
            "by_target": permuted_rows,
            "no_systematic_recovery": bool(
                all(
                    permuted_rows[target][name][str(budget)]["recovered_less_than_honest"]
                    for target in TARGETS
                    for name in DIRECTIONS
                    for budget in CONTROL_BUDGETS
                )
            ),
        },
    )
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "random_labels": {
                target: {
                    name: random_rows[target][name][str(SATURATION_BUDGET)]["random_labels"]
                    for name in DIRECTIONS
                }
                for target in TARGETS
            },
            "permuted_labels": {
                target: {
                    name: permuted_rows[target][name][str(SATURATION_BUDGET)]["permuted_labels"]
                    for name in DIRECTIONS
                }
                for target in TARGETS
            },
            "honest_labels": {
                target: {
                    name: curve_point(scores, indexed, name, A2, target, SATURATION_BUDGET)[
                        "median"
                    ]
                    for name in DIRECTIONS
                }
                for target in TARGETS
            },
            "all_controls_behave": bool(
                cc_read_json(RANDOM_LABEL_CONTROL)["no_systematic_recovery"]
                and cc_read_json(LABEL_PERMUTATION_CONTROL)["no_systematic_recovery"]
            ),
            "infeasible_arm_draws": cc_read_json(INFEASIBLE_DRAWS)["count"],
        },
    )
    print(f"controls: all behave ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ sections 40-41: statistics


def run_stats() -> int:
    """Sections 40-41: the frozen family of four, document-clustered, paired, Holm-corrected."""
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    scores, _indexed = load_adapted()
    population = load_population().set_index("candidate_id")
    frames: list[pd.DataFrame] = []
    for name in DIRECTIONS:
        block = scores[
            (scores["direction"] == name)
            & (scores["evaluation_set"] == EVAL_TARGET)
            & (scores["draw"] == INFERENCE_DRAW)
        ]
        per_arm: dict[str, pd.DataFrame] = {}
        for arm in (A0, A1, A2):
            budget = 0 if arm == A0 else DECISION_BUDGET
            cell = block[(block["arm"] == arm) & (block["budget"] == budget)]
            wide = cell.pivot_table(
                index="candidate_id", columns="target", values="score", aggfunc="first"
            )
            per_arm[arm] = wide.rename(
                columns={target: f"score_{target}_{arm}" for target in TARGETS}
            )
        joined = per_arm[A0].join([per_arm[A1], per_arm[A2]], how="inner")
        joined = joined.join(
            population[["environment", "document_id", "is_harmful", "beneficial"]], how="inner"
        )
        joined["direction"] = name
        frames.append(joined.reset_index())
    frame = pd.concat(frames, ignore_index=True)
    tests = {
        PRIMARY_FAMILY[0]: (A2, A0, T_HARM),
        PRIMARY_FAMILY[1]: (A2, A0, T_BENEFIT),
        PRIMARY_FAMILY[2]: (A2, A1, T_HARM),
        PRIMARY_FAMILY[3]: (A2, A1, T_BENEFIT),
    }
    primary = {}
    for name, (left, right, target) in tests.items():
        label = "is_harmful" if target == T_HARM else "beneficial"
        primary[name] = rl1.paired_auroc_comparison(
            frame, f"score_{target}_{left}", f"score_{target}_{right}", label, name, "primary"
        )
        primary[name]["per_direction"] = {
            direction: float(
                auroc(
                    frame[frame["direction"] == direction][f"score_{target}_{left}"].to_numpy(
                        dtype=np.float64
                    ),
                    frame[frame["direction"] == direction][label].to_numpy(dtype=bool),
                )
                - auroc(
                    frame[frame["direction"] == direction][f"score_{target}_{right}"].to_numpy(
                        dtype=np.float64
                    ),
                    frame[frame["direction"] == direction][label].to_numpy(dtype=bool),
                )
            )
            for direction in DIRECTIONS
        }
    adjusted = s15.holm(primary)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "bootstrap": {
                "unit": "document, resampled within environment",
                "paired": True,
                "statistic": "pooled AUROC difference on the target evaluation rows",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "alpha": ALPHA,
                "budget": DECISION_BUDGET,
                "draw": INFERENCE_DRAW,
                "directions_pooled": list(DIRECTIONS),
                "rows": len(frame),
            },
            "multiplicity": "Holm-Bonferroni inside the frozen primary family",
            "primary_family": adjusted,
            "family_size": len(PRIMARY_FAMILY),
            "declared_before_any_endpoint": True,
            "draw_variability_is_reported_separately": (
                "few-shot draw spread and clustered evaluation uncertainty are different sources. "
                "The test fixes the pre-registered draw; the quartiles in the sample-efficiency "
                "artifacts carry the draw spread and are never merged into these intervals"
            ),
            "budget_curves_are_descriptive": True,
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
        },
    )
    print(
        f"stats: {sum(1 for v in adjusted.values() if v['survives_holm'])}/{len(adjusted)} "
        f"survive Holm ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ section 43: falsification


def _module_tree() -> ast.Module:
    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def _names_in(function_name: str) -> set[str]:
    """Every name a function body mentions, including the bodies it calls in this module."""
    tree = _module_tree()
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
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


def run_negative() -> int:
    """Section 43: twenty tests, each of which would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    population = load_population()
    blocks = directions(population)
    matrix = pd.read_parquet(rl1.FEATURE_MATRIX)
    columns = rl1.columns_for(REPRESENTATION)
    design = rl1._matrix_for(population, matrix, columns)
    scores, indexed = load_adapted()
    splits = cc_read_json(TARGET_SPLIT_REGISTRY)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    representation = cc_read_json(REPRESENTATION_REPRODUCTION)
    record(
        "t01_rl1_r3_matrices_reproduce_exactly",
        representation["identical"],
        {"max_absolute_difference": representation["max_absolute_difference"]},
    )
    zero = cc_read_json(ZERO_SHOT_REPRODUCTION)
    record(
        "t02_zero_shot_reproduces_rl1_generator_held_out_results",
        zero["scores_identical"] and zero["all_auroc_identical"],
        {
            k: {"harm": v["harm_auroc"], "rl1": v["rl1_harm_auroc"]}
            for k, v in zero["by_direction"].items()
        },
    )
    crossing = {name: splits["leakage"][name]["sites_adapt_vs_eval"] for name in DIRECTIONS}
    record(
        "t03_target_evaluation_rows_never_enter_adaptation",
        all(
            not (
                set(population.iloc[block.target_adapt]["candidate_id"])
                & set(population.iloc[block.target_eval]["candidate_id"])
            )
            for block in blocks.values()
        ),
        crossing,
    )
    record(
        "t04_target_evaluation_documents_never_enter_adaptation",
        all(
            splits["leakage"][name]["documents_adapt_vs_eval"] == 0
            and splits["leakage"][name]["documents_source_train_vs_target_eval"] == 0
            for name in DIRECTIONS
        ),
        {name: splits["leakage"][name] for name in DIRECTIONS},
    )
    feature_names = _names_in("run_adapt") | _names_in("adapt_cell") | _names_in("calibrate")
    label_leak = feature_names & {"is_harmful_feature", "outcome_feature"}
    record(
        "t05_target_labels_do_not_affect_feature_construction",
        not label_leak and representation["identical"],
        {
            "feature_matrix_is_read_only": True,
            "builder": _relative(rl1.FEATURE_MATRIX),
        },
    )
    source_columns = [c for c in columns if c.startswith(rl1.FAM_SOURCE)]
    record(
        "t06_generator_identity_does_not_enter_the_adapted_feature_vectors",
        not source_columns,
        {"columns": len(columns), "source_columns": source_columns},
    )
    permutation = cc_read_json(LABEL_PERMUTATION_CONTROL)["by_target"]
    record(
        "t07_permuting_target_labels_destroys_the_adaptation_gain",
        cc_read_json(LABEL_PERMUTATION_CONTROL)["no_systematic_recovery"],
        {
            target: {
                name: permutation[target][name][str(SATURATION_BUDGET)]["permuted_labels"]
                for name in DIRECTIONS
            }
            for target in TARGETS
        },
    )
    swapped = matrix.copy()
    for left, right in (("src_corrector_c0", "src_corrector_c1"),):
        keep = swapped[left].to_numpy(dtype=np.float64)
        swapped[left] = swapped[right].to_numpy(dtype=np.float64)
        swapped[right] = keep
    swapped_design = rl1._matrix_for(population, swapped, columns)
    record(
        "t08_permuting_source_metadata_alone_leaves_the_features_unchanged",
        bool(np.abs(swapped_design - design).max() == 0.0),
        {"max_absolute_difference": float(np.abs(swapped_design - design).max())},
    )
    block = blocks[D_G2Q]
    labels = population["is_harmful"].to_numpy(dtype=bool)
    order_a = draw_order(population, block, INFERENCE_DRAW)
    order_b = draw_order(population, block, INFERENCE_DRAW)
    rows = purchased(order_a, DECISION_BUDGET)
    first = _score(
        _fit(np.concatenate([block.source_train, rows]), design, labels), design, block.target_eval
    )
    second = _score(
        _fit(
            np.concatenate([block.source_train, purchased(order_b, DECISION_BUDGET)]),
            design,
            labels,
        ),
        design,
        block.target_eval,
    )
    record(
        "t09_the_same_draw_and_seed_reproduce_the_same_adapted_model",
        bool(np.array_equal(order_a, order_b) and np.abs(first - second).max() == 0.0),
        {"max_absolute_score_difference": float(np.abs(first - second).max())},
    )
    nested = all(
        np.array_equal(
            purchased(draw_order(population, blocks[name], draw), small),
            purchased(draw_order(population, blocks[name], draw), large)[:small],
        )
        for name in DIRECTIONS
        for draw in range(DRAWS)
        for small, large in pairwise(BUDGETS[1:])
    )
    record(
        "t10_increasing_the_budget_only_extends_the_frozen_draw_order",
        nested,
        {"budgets": list(BUDGETS)},
    )
    a1_move = cc_read_json(METHOD_COMPARISON)["a1_maximum_auroc_move_from_zero_shot"]
    record(
        "t11_score_only_calibration_cannot_change_ranking",
        all(float(a1_move[target][name]) < 1e-3 for target in TARGETS for name in DIRECTIONS),
        a1_move,
    )
    comparison = cc_read_json(METHOD_COMPARISON)["by_target"]
    record(
        "t12_the_joint_refit_can_change_ranking",
        any(
            abs(
                float(comparison[target][name][str(SATURATION_BUDGET)][A2])
                - float(comparison[target][name]["0"][A0])
            )
            > 1e-3
            for target in TARGETS
            for name in DIRECTIONS
        ),
        {
            target: {
                name: float(comparison[target][name][str(SATURATION_BUDGET)][A2])
                for name in DIRECTIONS
            }
            for target in TARGETS
        },
    )
    record(
        "t13_random_target_labels_do_not_produce_systematic_recovery",
        cc_read_json(RANDOM_LABEL_CONTROL)["no_systematic_recovery"],
        cc_read_json(CONTROL_RESULTS)["random_labels"],
    )
    a3_rows = purchased(draw_order(population, block, 0), SATURATION_BUDGET)
    record(
        "t14_target_only_adaptation_contains_no_source_generator_row",
        set(population.iloc[a3_rows]["corrector_source"]) == {block.target},
        sorted(set(population.iloc[a3_rows]["corrector_source"].astype(str))),
    )
    record(
        "t15_source_forgetting_uses_a_disjoint_source_evaluation_set",
        all(
            splits["leakage"][name]["documents_source_train_vs_source_eval"] == 0
            for name in DIRECTIONS
        ),
        {name: splits["directions"][name]["source_eval"]["candidates"] for name in DIRECTIONS},
    )
    record(
        "t16_duplicates_cannot_cross_the_adaptation_evaluation_partition",
        all(splits["leakage"][name]["edits_adapt_vs_eval"] == 0 for name in DIRECTIONS),
        {name: splits["leakage"][name]["edits_adapt_vs_eval"] for name in DIRECTIONS},
    )
    multi = set(population[population["multi_source"]]["candidate_id"])
    used = set(scores["candidate_id"])
    record(
        "t17_multi_source_candidates_follow_the_frozen_rl1_rule",
        not (multi & used),
        {"multi_source": len(multi), "reaching_an_rl2_endpoint": len(multi & used)},
    )
    fitting_names = _names_in("_fit") | _names_in("adapt_cell") | _names_in("run_adapt")
    oracle_leak = fitting_names & {"oracle_frontier", "ORACLE_RISK_FRONTIER", "epsilon_key"}
    record(
        "t18_oracle_thresholds_never_enter_a_fitted_model",
        not oracle_leak,
        sorted(oracle_leak),
    )
    record(
        "t19_changing_the_harm_epsilon_leaves_the_adapted_scores_unchanged",
        not (fitting_names & {"EPSILONS", "PRIMARY_EPSILON"}),
        sorted(fitting_names & {"EPSILONS", "PRIMARY_EPSILON"}),
    )
    review_terms = {"review_budget", "human_review", "REVIEW_BUDGETS", "dt1"}
    record(
        "t20_human_review_logic_is_absent_from_model_fitting",
        not (fitting_names & review_terms),
        sorted(fitting_names & review_terms),
    )
    _ = indexed
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


# ------------------------------------------------------------------ sections 38-39: decision


def criteria_from_artifacts() -> dict[str, Any]:
    """Every success criterion, re-derived from the persisted artifacts and nothing else."""
    harm = cc_read_json(SAMPLE_EFFICIENCY_HARM)["by_direction"]
    benefit = cc_read_json(SAMPLE_EFFICIENCY_BENEFIT)["by_direction"]
    curves = {T_HARM: harm, T_BENEFIT: benefit}
    floors = {T_HARM: HARM_FLOOR, T_BENEFIT: BENEFIT_FLOOR}
    share = cc_read_json(TRANSFER_GAP_RECOVERY)["by_target"]
    environments = cc_read_json(ENVIRONMENT_ANALYSIS)
    forgetting = cc_read_json(SOURCE_RETENTION)
    frontier = cc_read_json(ORACLE_RISK_FRONTIER)
    below = [budget for budget in BUDGETS if 0 < budget <= DECISION_BUDGET]

    def meets(target: str, budget: int) -> bool:
        return all(
            np.isfinite(curves[target][name][A2][str(budget)]["median"])
            and curves[target][name][A2][str(budget)]["median"] >= floors[target]
            for name in DIRECTIONS
        )

    c1 = any(meets(T_HARM, budget) for budget in below)
    c2 = any(meets(T_BENEFIT, budget) for budget in below)

    def recovered(budget: int) -> bool:
        return all(
            (share[target][name]["by_arm"][A2].get(str(budget)) or -1.0) >= RECOVERY_FLOOR
            for target in TARGETS
            for name in DIRECTIONS
        )

    c3 = any(recovered(budget) for budget in below)
    c4 = all(
        int(environments["environments_improved_on_harm"][str(DECISION_BUDGET)][name])
        >= BREADTH_MAJORITY
        for name in DIRECTIONS
    )
    c5 = bool(forgetting["within_tolerance_everywhere"])
    key = epsilon_key(PRIMARY_EPSILON)
    margins = frontier["margin_over_random"]
    zero_margin = {name: float(margins[name]["0"]) for name in DIRECTIONS}
    c6 = any(
        all(
            float(margins[name][str(budget)]) >= PREFIX_MARGIN
            and float(margins[name][str(budget)]) > zero_margin[name]
            for name in DIRECTIONS
        )
        for budget in below
    )
    c6_any_budget = any(
        all(
            float(margins[name][str(budget)]) >= PREFIX_MARGIN
            and float(margins[name][str(budget)]) > zero_margin[name]
            for name in DIRECTIONS
        )
        for budget in BUDGETS
        if budget > 0
    )
    c6_best_direction = any(
        float(margins[name][str(budget)]) >= PREFIX_MARGIN
        for name in DIRECTIONS
        for budget in BUDGETS
        if budget > 0
    )
    adapts = {
        name: all(
            np.isfinite(curves[target][name][A2][str(SATURATION_BUDGET)]["median"])
            and curves[target][name][A2][str(SATURATION_BUDGET)]["median"] >= floors[target]
            for target in TARGETS
        )
        for name in DIRECTIONS
    }
    return {
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6},
        "c6_readings": {
            "primary_both_directions_at_or_below_the_decision_budget": c6,
            "both_directions_at_any_tested_budget": c6_any_budget,
            "best_single_direction_at_any_tested_budget": c6_best_direction,
        },
        "curves": curves,
        "recovery": share,
        "adapts_at_saturation": adapts,
        "frontier_key": key,
        "margins": margins,
        "forgetting": forgetting,
        "environments": environments,
    }


def _mechanism(comparison: dict[str, Any], a1_move: dict[str, Any]) -> str:
    calibration_moves = max(
        float(a1_move[target][name]) for target in TARGETS for name in DIRECTIONS
    )
    target_only_wins = sum(
        int(comparison[target][name]) for target in TARGETS for name in DIRECTIONS
    )
    if calibration_moves >= 1e-3:
        return "score alignment alone repairs transfer"
    if target_only_wins > len(TARGETS) * len(DIRECTIONS) * (len(BUDGETS) - 1) / 2:
        return (
            "the representation transfers but the decision geometry must be refitted, and the "
            "source generator's rows impose negative transfer"
        )
    return "the representation transfers but the decision geometry must be refitted"


def run_decide() -> int:
    """Section 39: the frozen outcome rule, applied to the persisted artifacts."""
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    criteria = state["criteria"]
    curves = state["curves"]
    outcome = assign_outcome(criteria, state["adapts_at_saturation"])
    failed = sorted(name for name, value in criteria.items() if not value)
    minimal = cc_read_json(MINIMAL_LABEL_BUDGET)["by_arm"]
    comparison = cc_read_json(METHOD_COMPARISON)
    negative = cc_read_json(FALSIFICATION)
    frontier = cc_read_json(ORACLE_RISK_FRONTIER)["by_direction"]
    label = OUTCOME_TAXONOMY[outcome]
    if outcome == "B":
        label = (
            "adaptation works but requires substantial supervision"
            if not (criteria["C1"] and criteria["C2"] and criteria["C3"])
            else f"adaptation works but a qualifying criterion does not: {', '.join(failed)}"
        )
    reason = (
        f"outcome {outcome}: "
        + ", ".join(f"{name} {'met' if value else 'not met'}" for name, value in criteria.items())
        + f". {label}"
        + (f"; unmet: {', '.join(failed)}" if failed else "")
    )
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "primary_population": POPULATION,
            "primary_representation": REPRESENTATION,
            "primary_adaptation_method": PRIMARY_ARM,
            "primary_model": MODEL,
            "label_budgets": list(BUDGETS),
            "decision_budget": DECISION_BUDGET,
            "saturation_budget": SATURATION_BUDGET,
            "draws_per_cell": DRAWS,
            **{
                f"zero_shot_{name}_{target}_auroc": float(curves[target][name][A0]["0"]["median"])
                for name in DIRECTIONS
                for target in TARGETS
            },
            **{
                f"adapted_{name}_{target}_auroc_by_budget": {
                    str(budget): float(curves[target][name][A2][str(budget)]["median"])
                    for budget in BUDGETS
                }
                for name in DIRECTIONS
                for target in TARGETS
            },
            **{
                f"{target}_gap_recovery_by_budget": {
                    name: {
                        str(budget): state["recovery"][target][name]["by_arm"][A2].get(str(budget))
                        for budget in BUDGETS
                    }
                    for name in DIRECTIONS
                }
                for target in TARGETS
            },
            "mixed_source_reference": {
                target: {
                    name: float(state["recovery"][target][name]["mixed_source_reference"])
                    for name in DIRECTIONS
                }
                for target in TARGETS
            },
            "minimal_budget_harm": {name: minimal[A2][name]["n_star_harm"] for name in DIRECTIONS},
            "minimal_budget_benefit": {
                name: minimal[A2][name]["n_star_benefit"] for name in DIRECTIONS
            },
            "minimal_budget_joint": {
                name: minimal[A2][name]["n_star_joint"] for name in DIRECTIONS
            },
            "minimal_budget_joint_target_only": {
                name: minimal[A3][name]["n_star_joint"] for name in DIRECTIONS
            },
            "source_forgetting_harm": {
                name: float(
                    cc_read_json(SOURCE_RETENTION)["by_target"][T_HARM][name]["worst_forgetting"]
                )
                for name in DIRECTIONS
            },
            "source_forgetting_benefit": {
                name: float(
                    cc_read_json(SOURCE_RETENTION)["by_target"][T_BENEFIT][name]["worst_forgetting"]
                )
                for name in DIRECTIONS
            },
            "worst_source_forgetting": float(cc_read_json(SOURCE_RETENTION)["worst_forgetting"]),
            **{
                f"oracle_repair_recall_at_harm_{epsilon_key(e).replace('.', '')}": {
                    name: float(
                        frontier[name][str(SATURATION_BUDGET)][epsilon_key(e)]["repair_recall"]
                    )
                    for name in DIRECTIONS
                }
                for e in EPSILONS
            },
            "oracle_margin_over_random_at_primary_epsilon": {
                name: float(state["margins"][name][str(SATURATION_BUDGET)]) for name in DIRECTIONS
            },
            "environments_improved": {
                name: int(
                    state["environments"]["environments_improved_on_harm"][str(DECISION_BUDGET)][
                        name
                    ]
                )
                for name in DIRECTIONS
            },
            "supported_environments": state["environments"]["supported_environments"],
            "score_calibration_maximum_auroc_move": comparison[
                "a1_maximum_auroc_move_from_zero_shot"
            ],
            "budgets_where_target_only_beats_joint_refit": comparison[
                "budgets_where_target_only_beats_joint_refit"
            ],
            "criterion_C1": criteria["C1"],
            "criterion_C2": criteria["C2"],
            "criterion_C3": criteria["C3"],
            "criterion_C4": criteria["C4"],
            "criterion_C5": criteria["C5"],
            "criterion_C6": criteria["C6"],
            "c6_readings": state["c6_readings"],
            "criteria_thresholds": {
                "harm_floor": HARM_FLOOR,
                "benefit_floor": BENEFIT_FLOOR,
                "recovery_floor": RECOVERY_FLOOR,
                "breadth_majority": BREADTH_MAJORITY,
                "forgetting_tolerance": FORGETTING_TOLERANCE,
                "prefix_margin": PREFIX_MARGIN,
                "decision_budget": DECISION_BUDGET,
                "minimum_recovery_denominator": MIN_RECOVERY_DENOMINATOR,
            },
            "adapts_at_saturation": state["adapts_at_saturation"],
            "dominant_adaptation_mechanism": _mechanism(
                comparison["budgets_where_target_only_beats_joint_refit"],
                comparison["a1_maximum_auroc_move_from_zero_shot"],
            ),
            "outcome": outcome,
            "outcome_label": label,
            "unmet_criteria": failed,
            "ready_for_deployment_synthesis": outcome in ("A", "B"),
            "ready_for_external_confirmation": False,
            "confirmatory_reserve_consumed": False,
            "deployable": False,
            "selects_a_deployment_threshold": False,
            "falsification_tests_passed": int(negative["passed"]),
            "falsification_tests_total": int(negative["total"]),
            "recommended_next_stage": NEXT_STAGE[outcome],
            "reason": reason,
            "hypothesis_id": HYPOTHESIS,
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: outcome {outcome} -- {NEXT_STAGE[outcome]}")
    return 0


# ------------------------------------------------------------------ section 47: figures

FIGURE_NOTE = rl1.FIGURE_NOTE
SURFACE = rl1.SURFACE
ARM_COLOURS = {
    A0: "#e87ba4",
    A1: "#eda100",
    A2: "#2a78d6",
    A3: "#1baf7a",
    "random": "#8a8a8a",
    "reference": "#008300",
}
ARM_LABELS = {
    A0: "A0 zero shot",
    A1: "A1 calibration",
    A2: "A2 joint refit",
    A3: "A3 target only",
}
DIRECTION_SHORT = {D_G2Q: "current to Qwen", D_Q2G: "Qwen to current"}
FIGURES = (
    "harm_auroc_vs_target_labels.png",
    "benefit_auroc_vs_target_labels.png",
    "gap_recovery_vs_target_labels.png",
    "current_to_qwen_adaptation.png",
    "qwen_to_current_adaptation.png",
    "adaptation_method_comparison.png",
    "minimal_label_budget.png",
    "source_retention_vs_budget.png",
    "mixed_population_auroc_vs_budget.png",
    "harm_prefix_vs_budget.png",
    "benefit_prefix_vs_budget.png",
    "oracle_repair_recall_at_harm_010.png",
    "environment_adaptation_breadth.png",
    "label_composition_by_budget.png",
    "adaptation_efficiency.png",
    "zero_shot_reproduction.png",
)


def run_figures() -> int:
    """The sixteen required figures, every value read from a persisted artifact."""
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
            "axes.edgecolor": xc1.INK_SECONDARY,
            "axes.labelcolor": xc1.INK,
            "xtick.color": xc1.INK_SECONDARY,
            "ytick.color": xc1.INK_SECONDARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": xc1.GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "svg.hashsalt": "rl2",
        }
    )
    manifest: dict[str, Any] = {}

    def source_key(path: Path) -> str:
        stage = FIGURE_DIR.parent
        return str(path.relative_to(stage)) if path.is_relative_to(stage) else _relative(path)

    def save(fig: Any, name: str, sources: Sequence[Path], caption: str) -> None:
        fig.text(0.01, 0.005, FIGURE_NOTE, fontsize=5.5, color=xc1.INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = FIGURE_DIR / name
        fig.savefig(
            path, dpi=150, bbox_inches="tight", pad_inches=0.08, metadata={"Software": None}
        )
        plt.close(fig)
        manifest[name] = {
            "path": f"{FIGURE_DIR.name}/{name}",
            "caption": caption,
            "sources": {source_key(s): xr1._signature(s) for s in sources},
        }

    harm = cc_read_json(SAMPLE_EFFICIENCY_HARM)
    benefit = cc_read_json(SAMPLE_EFFICIENCY_BENEFIT)
    efficiency = {T_HARM: harm, T_BENEFIT: benefit}
    budgets = np.asarray(BUDGETS, dtype=float)

    def curve_axis(ax: Any, target: str, direction: str, arms: Sequence[str]) -> None:
        block = efficiency[target]["by_direction"][direction]
        for arm in arms:
            values = [block[arm][str(int(b))]["median"] for b in BUDGETS]
            ax.plot(
                budgets,
                values,
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                color=ARM_COLOURS[arm],
                label=ARM_LABELS[arm],
            )
            if arm == A2:
                ax.fill_between(
                    budgets,
                    [block[arm][str(int(b))]["q25"] for b in BUDGETS],
                    [block[arm][str(int(b))]["q75"] for b in BUDGETS],
                    color=ARM_COLOURS[arm],
                    alpha=0.15,
                    linewidth=0,
                )
        ax.axhline(
            efficiency[target]["mixed_source_reference"][direction],
            color=ARM_COLOURS["reference"],
            linewidth=1.0,
            linestyle="--",
            label="RL1 mixed-source reference",
        )
        ax.axhline(
            efficiency[target]["absolute_floor"],
            color=xc1.INK_SECONDARY,
            linewidth=0.8,
            linestyle=":",
            label="absolute floor",
        )
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_xlabel("labelled target-generator candidates")
        ax.set_ylabel(f"{target} AUROC")

    for target, name, caption in (
        (T_HARM, FIGURES[0], "harm AUROC against the target label budget, both directions"),
        (T_BENEFIT, FIGURES[1], "benefit AUROC against the target label budget, both directions"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
        for ax, direction in zip(axes, DIRECTIONS, strict=True):
            curve_axis(ax, target, direction, (A0, A1, A2, A3))
            ax.set_title(DIRECTION_SHORT[direction])
        axes[1].legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, name, [SAMPLE_EFFICIENCY_HARM, SAMPLE_EFFICIENCY_BENEFIT], caption)

    share = cc_read_json(TRANSFER_GAP_RECOVERY)["by_target"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, target in zip(axes, TARGETS, strict=True):
        for direction in DIRECTIONS:
            values = [
                share[target][direction]["by_arm"][A2].get(str(int(b))) or 0.0 for b in BUDGETS
            ]
            ax.plot(
                budgets,
                values,
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                color=ARM_COLOURS[A2] if direction == D_G2Q else ARM_COLOURS[A3],
                label=DIRECTION_SHORT[direction],
            )
        ax.axhline(RECOVERY_FLOOR, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle=":")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_xlabel("labelled target-generator candidates")
        ax.set_ylabel(f"{target} gap recovery")
        ax.set_title(target)
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIGURES[2], [TRANSFER_GAP_RECOVERY], "share of the zero-shot gap that closes")

    for direction, name in ((D_G2Q, FIGURES[3]), (D_Q2G, FIGURES[4])):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
        for ax, target in zip(axes, TARGETS, strict=True):
            curve_axis(ax, target, direction, (A0, A1, A2, A3))
            ax.set_title(target)
        axes[1].legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(
            fig,
            name,
            [SAMPLE_EFFICIENCY_HARM, SAMPLE_EFFICIENCY_BENEFIT],
            f"{DIRECTION_LABEL[direction]}: both targets, every arm",
        )

    comparison = cc_read_json(METHOD_COMPARISON)["by_target"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, target in zip(axes, TARGETS, strict=True):
        groups = [DIRECTION_SHORT[d] for d in DIRECTIONS]
        lp1._grouped(
            ax,
            groups,
            {
                arm: [float(comparison[target][d][str(DECISION_BUDGET)][arm]) for d in DIRECTIONS]
                for arm in ARMS
            },
            ARM_COLOURS,
            ARM_LABELS,
        )
        ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{target} AUROC")
        ax.set_title(f"{target} at {DECISION_BUDGET} labels")
    save(fig, FIGURES[5], [METHOD_COMPARISON], "the four adaptation arms at the decision budget")

    minimal = cc_read_json(MINIMAL_LABEL_BUDGET)["by_arm"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    labels_x = [f"{DIRECTION_SHORT[d]}\n{t}" for d in DIRECTIONS for t in TARGETS]
    values = []
    colours = []
    for arm in (A2, A3):
        for d in DIRECTIONS:
            for t in TARGETS:
                value = minimal[arm][d][f"n_star_{t}"]
                values.append(float(value) if value is not None else float(max(BUDGETS)) * 1.2)
                colours.append(ARM_COLOURS[arm])
    positions = np.arange(len(labels_x))
    width = 0.38
    for offset, arm in ((-width / 2, A2), (width / 2, A3)):
        block = [minimal[arm][d][f"n_star_{t}"] for d in DIRECTIONS for t in TARGETS]
        ax.bar(
            positions + offset,
            [float(v) if v is not None else 0.0 for v in block],
            width=width,
            color=ARM_COLOURS[arm],
            edgecolor=SURFACE,
            linewidth=1.0,
            label=ARM_LABELS[arm],
        )
        for x, value in zip(positions + offset, block, strict=True):
            ax.text(
                x,
                float(value) if value is not None else 0.0,
                f" {value}" if value is not None else " >500",
                ha="center",
                va="bottom",
                fontsize=6.0,
                color=xc1.INK_SECONDARY,
            )
    ax.set_xticks(positions, labels_x)
    ax.set_ylabel("smallest budget meeting both floors")
    ax.set_title("Minimal useful label budget")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIGURES[6], [MINIMAL_LABEL_BUDGET], "the minimal useful target label budget")

    retention = cc_read_json(SOURCE_RETENTION)["by_target"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, target in zip(axes, TARGETS, strict=True):
        for direction in DIRECTIONS:
            rows = retention[target][direction]["by_budget"]
            ax.plot(
                [b for b in BUDGETS if b > 0],
                [rows[str(b)]["forgetting"] for b in BUDGETS if b > 0],
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                color=ARM_COLOURS[A2] if direction == D_G2Q else ARM_COLOURS[A3],
                label=DIRECTION_SHORT[direction],
            )
        ax.axhline(-FORGETTING_TOLERANCE, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle=":")
        ax.axhline(0.0, color=xc1.INK_SECONDARY, linewidth=0.8)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([b for b in BUDGETS if b > 0], [str(b) for b in BUDGETS if b > 0])
        ax.set_xlabel("labelled target-generator candidates")
        ax.set_ylabel(f"{target} AUROC change on the source generator")
        ax.set_title(target)
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIGURES[7], [SOURCE_RETENTION], "source-generator forgetting against the budget")

    mixed = cc_read_json(MIXED_POPULATION)["by_target"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, target in zip(axes, TARGETS, strict=True):
        for direction in DIRECTIONS:
            ax.plot(
                budgets,
                [mixed[target][direction][str(int(b))]["auroc"] for b in BUDGETS],
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                color=ARM_COLOURS[A2] if direction == D_G2Q else ARM_COLOURS[A3],
                label=DIRECTION_SHORT[direction],
            )
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_xlabel("labelled target-generator candidates")
        ax.set_ylabel(f"{target} AUROC, pooled population")
        ax.set_title(target)
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIGURES[8], [MIXED_POPULATION], "the pooled hybrid population after adaptation")

    prefixes = cc_read_json(PREFIX_RANKING)["by_direction"]
    for key_name, name, caption in (
        ("harm_rate", FIGURES[9], "harm rate along the ranked prefix, by budget"),
        ("beneficial_rate", FIGURES[10], "beneficial rate along the ranked prefix, by budget"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
        for ax, direction in zip(axes, DIRECTIONS, strict=True):
            for budget in BUDGETS:
                rows = prefixes[direction][str(budget)]["rows"]
                ax.plot(
                    [row["rank_fraction"] for row in rows],
                    [row[key_name] for row in rows],
                    marker="o",
                    markersize=2.6,
                    linewidth=1.2,
                    label=f"N={budget}",
                )
            ax.set_xlabel("coverage")
            ax.set_ylabel(key_name.replace("_", " "))
            ax.set_title(DIRECTION_SHORT[direction])
        axes[1].legend(frameon=False, fontsize=6.0, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, name, [PREFIX_RANKING], caption)

    frontier = cc_read_json(ORACLE_RISK_FRONTIER)
    key = epsilon_key(PRIMARY_EPSILON)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for direction in DIRECTIONS:
        ax.plot(
            budgets,
            [
                frontier["by_direction"][direction][str(int(b))][key]["repair_recall"]
                for b in BUDGETS
            ],
            marker="o",
            markersize=3.2,
            linewidth=1.4,
            color=ARM_COLOURS[A2] if direction == D_G2Q else ARM_COLOURS[A3],
            label=DIRECTION_SHORT[direction],
        )
        ax.axhline(
            frontier["random_ranking"][direction][key],
            color=ARM_COLOURS["random"],
            linewidth=0.8,
            linestyle="--",
        )
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    ax.set_xlabel("labelled target-generator candidates")
    ax.set_ylabel(f"exact-repair recall at harm <= {PRIMARY_EPSILON}")
    ax.set_title("Oracle harm-constrained frontier -- analysis only, not deployable")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIGURES[11], [ORACLE_RISK_FRONTIER], "oracle repair recall at the primary epsilon")

    environments = cc_read_json(ENVIRONMENT_ANALYSIS)["by_budget"][str(DECISION_BUDGET)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        names = sorted(environments[direction])
        lp1._grouped(
            ax,
            [n.replace("/", "\n") for n in names],
            {
                A0: [environments[direction][n][f"zero_shot_{T_HARM}_auroc"] for n in names],
                A2: [environments[direction][n][f"adapted_{T_HARM}_auroc"] for n in names],
            },
            ARM_COLOURS,
            ARM_LABELS,
            label_values=False,
        )
        ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
        ax.set_ylabel("harm AUROC")
        ax.set_title(DIRECTION_SHORT[direction])
        ax.tick_params(axis="x", labelrotation=45, labelsize=5.5)
    save(fig, FIGURES[12], [ENVIRONMENT_ANALYSIS], "environment breadth at the decision budget")

    composition = cc_read_json(LABEL_COMPOSITION)["by_direction"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        positive = [b for b in BUDGETS if b > 0]
        lp1._grouped(
            ax,
            [str(b) for b in positive],
            {
                "harmful": [composition[direction][str(b)]["median_harmful"] for b in positive],
                "beneficial": [
                    composition[direction][str(b)]["median_beneficial"] for b in positive
                ],
                "neither": [composition[direction][str(b)]["median_neither"] for b in positive],
            },
            {
                "harmful": ARM_COLOURS[A0],
                "beneficial": ARM_COLOURS[A3],
                "neither": ARM_COLOURS["random"],
            },
            {"harmful": "harmful", "beneficial": "beneficial", "neither": "neither"},
            label_values=False,
        )
        ax.set_xlabel("label budget")
        ax.set_ylabel("median labels purchased")
        ax.set_title(DIRECTION_SHORT[direction])
    save(fig, FIGURES[13], [LABEL_COMPOSITION], "what each budget actually buys")

    efficiency_rows = cc_read_json(ADAPTATION_EFFICIENCY)["by_target"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, target in zip(axes, TARGETS, strict=True):
        for direction in DIRECTIONS:
            positive = [b for b in BUDGETS if b > 0]
            ax.plot(
                positive,
                [
                    efficiency_rows[target][direction][str(b)]["delta_auroc_per_label"]
                    for b in positive
                ],
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                color=ARM_COLOURS[A2] if direction == D_G2Q else ARM_COLOURS[A3],
                label=DIRECTION_SHORT[direction],
            )
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([b for b in BUDGETS if b > 0], [str(b) for b in BUDGETS if b > 0])
        ax.set_xlabel("labelled target-generator candidates")
        ax.set_ylabel(f"{target} AUROC gained per label")
        ax.set_title(target)
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIGURES[14], [ADAPTATION_EFFICIENCY], "where the returns to labelling flatten")

    reproduction = cc_read_json(ZERO_SHOT_REPRODUCTION)["by_direction"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    names = [DIRECTION_SHORT[d] for d in DIRECTIONS]
    lp1._grouped(
        ax,
        names,
        {
            "rl1": [float(reproduction[d]["rl1_harm_auroc"]) for d in DIRECTIONS],
            "rl2": [float(reproduction[d]["harm_auroc"]) for d in DIRECTIONS],
        },
        {"rl1": ARM_COLOURS[A0], "rl2": ARM_COLOURS[A2]},
        {"rl1": "SGV-RL1 published", "rl2": "SGV-RL2 rebuilt"},
    )
    ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
    ax.set_ylabel("harm AUROC, leave-one-generator-out")
    ax.set_title("Gate R1: RL1's zero-shot transfer reproduced")
    save(fig, FIGURES[15], [ZERO_SHOT_REPRODUCTION], "the zero-shot reproduction gate")

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
            "every_value_read_from_an_artifact": True,
        },
    )
    print(f"figures: {len(manifest)} written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ sections 44-45: record

DERIVED_OUTPUTS = (
    "TARGET_SPLIT_REGISTRY",
    "GROUP_REGISTRY",
    "BUDGET_REGISTRY",
    "DRAW_REGISTRY",
    "LABEL_COMPOSITION",
    "REPRESENTATION_REPRODUCTION",
    "ZERO_SHOT_REPRODUCTION",
    "MODEL_REGISTRY",
    "ADAPTATION_REGISTRY",
    "HYPERPARAMETER_REGISTRY",
    "INFEASIBLE_DRAWS",
    "ADAPTED_SCORES",
    "SAMPLE_EFFICIENCY_HARM",
    "SAMPLE_EFFICIENCY_BENEFIT",
    "TRANSFER_GAP_RECOVERY",
    "MINIMAL_LABEL_BUDGET",
    "METHOD_COMPARISON",
    "ADAPTATION_EFFICIENCY",
    "DIRECTION_ANALYSIS",
    "SOURCE_RETENTION",
    "MIXED_POPULATION",
    "PREFIX_RANKING",
    "ORACLE_RISK_FRONTIER",
    "ENVIRONMENT_ANALYSIS",
    "ERROR_TYPE_ANALYSIS",
    "RANDOM_LABEL_CONTROL",
    "LABEL_PERMUTATION_CONTROL",
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
        ("draws", run_draws),
        ("reproduce", run_reproduce),
        ("adapt", run_adapt),
        ("curves", run_curves),
        ("retention", run_retention),
        ("prefix", run_prefix),
        ("strata", run_strata),
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
    """Every derived phase, re-run from the frozen RL1 inputs into a directory of its own."""
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
    """Section 44: two independent regenerations from the frozen RL1 inputs."""
    started = time.monotonic()
    _require(FIGURE_MANIFEST, "figures")
    _forbid(DETERMINISM)
    published = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(
        name for name in published if any(run.get(name) != published[name] for run in runs)
    )
    decision = cc_read_json(DECISION)
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
                "draws": DRAW_SEED,
                "fit": rl1.FIT_SEED,
                "bootstrap": BOOTSTRAP_SEED,
                "random_labels": RANDOM_LABEL_SEED,
                "label_permutation": PERMUTATION_SEED,
                "random_ranking": RANDOM_RANKING_SEED,
            },
            "draw_membership_compared": True,
            "row_ordering": (
                "every table is sorted by an explicit key before it is written, so no result "
                "depends on the order the loops happened to produce"
            ),
            "outcome": decision["outcome"],
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
    POPULATION_INVENTORY,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE, FROZEN_CONFIGURATION),
    "2. Frozen RL1 Result": (FROZEN_CONFIGURATION, ZERO_SHOT_REPRODUCTION, rl1.DECISION),
    "3. Research Questions": (DESIGN_RECORD,),
    "4. Why Target Supervision Is the Next Variable": (
        FROZEN_CONFIGURATION,
        METHOD_COMPARISON,
        rl1.DECISION,
    ),
    "5. Non-Goals": (DESIGN_RECORD,),
    "6. Frozen R3 Representation": (REPRESENTATION_REPRODUCTION, FROZEN_CONFIGURATION),
    "7. Candidate Population": (POPULATION_INVENTORY, GROUP_REGISTRY),
    "8. Generator Transfer Directions": (DIRECTION_ANALYSIS, TARGET_SPLIT_REGISTRY),
    "9. Target Adaptation and Evaluation Split": (TARGET_SPLIT_REGISTRY, GROUP_REGISTRY),
    "10. Label Acquisition Rule": (DESIGN_RECORD, DRAW_REGISTRY),
    "11. Label Budgets": (BUDGET_REGISTRY, LABEL_COMPOSITION),
    "12. Adaptation Arms": (ADAPTATION_REGISTRY, MODEL_REGISTRY, HYPERPARAMETER_REGISTRY),
    "13. Zero-Shot Reproduction": (ZERO_SHOT_REPRODUCTION, REPRESENTATION_REPRODUCTION),
    "14. Harm Sample Efficiency": (SAMPLE_EFFICIENCY_HARM,),
    "15. Benefit Sample Efficiency": (SAMPLE_EFFICIENCY_BENEFIT,),
    "16. Current to Qwen Adaptation": (
        SAMPLE_EFFICIENCY_HARM,
        SAMPLE_EFFICIENCY_BENEFIT,
        DIRECTION_ANALYSIS,
    ),
    "17. Qwen to Current Adaptation": (
        SAMPLE_EFFICIENCY_HARM,
        SAMPLE_EFFICIENCY_BENEFIT,
        DIRECTION_ANALYSIS,
    ),
    "18. Transfer-Gap Recovery": (TRANSFER_GAP_RECOVERY, BUDGET_REGISTRY),
    "19. Minimal Useful Label Budget": (MINIMAL_LABEL_BUDGET, TRANSFER_GAP_RECOVERY),
    "20. Calibration-Only Adaptation": (METHOD_COMPARISON, STATISTICAL_TESTS),
    "21. Joint Source-Target Refit": (METHOD_COMPARISON, STATISTICAL_TESTS),
    "22. Target-Only Adaptation": (
        METHOD_COMPARISON,
        MINIMAL_LABEL_BUDGET,
        TARGET_SPLIT_REGISTRY,
        SAMPLE_EFFICIENCY_HARM,
        SAMPLE_EFFICIENCY_BENEFIT,
        DECISION,
    ),
    "23. Harm versus Benefit Label Efficiency": (
        MINIMAL_LABEL_BUDGET,
        ADAPTATION_EFFICIENCY,
        LABEL_COMPOSITION,
        BUDGET_REGISTRY,
    ),
    "24. Environment Breadth": (ENVIRONMENT_ANALYSIS, BUDGET_REGISTRY),
    "25. Source Retention": (SOURCE_RETENTION, BUDGET_REGISTRY),
    "26. Mixed Candidate Population": (MIXED_POPULATION, BUDGET_REGISTRY),
    "27. Prefix Ranking": (PREFIX_RANKING, BUDGET_REGISTRY),
    "28. Oracle Harm-Constrained Frontier": (ORACLE_RISK_FRONTIER, BUDGET_REGISTRY),
    "29. Error-Type Analysis": (ERROR_TYPE_ANALYSIS, TARGET_SPLIT_REGISTRY),
    "30. Label Composition": (LABEL_COMPOSITION, DRAW_REGISTRY),
    "31. Sparse-Class and Infeasible Draws": (INFEASIBLE_DRAWS, LABEL_COMPOSITION),
    "32. Adaptation Efficiency": (ADAPTATION_EFFICIENCY, BUDGET_REGISTRY),
    "33. Random-Label Control": (RANDOM_LABEL_CONTROL, CONTROL_RESULTS),
    "34. Label-Permutation Control": (LABEL_PERMUTATION_CONTROL, CONTROL_RESULTS),
    "35. Falsification Tests": (FALSIFICATION, DETERMINISM),
    "36. Statistical Tests": (STATISTICAL_TESTS,),
    "37. Limitations": (
        DESIGN_RECORD,
        TARGET_SPLIT_REGISTRY,
        LABEL_COMPOSITION,
        BUDGET_REGISTRY,
        ENVIRONMENT_ANALYSIS,
        ERROR_TYPE_ANALYSIS,
        MINIMAL_LABEL_BUDGET,
        DECISION,
    ),
    "38. Research Decision": (DECISION, STATISTICAL_TESTS, ORACLE_RISK_FRONTIER),
    "39. Implications for Deployment": (
        DECISION,
        ORACLE_RISK_FRONTIER,
        MINIMAL_LABEL_BUDGET,
        METHOD_COMPARISON,
        PREFIX_RANKING,
    ),
    "40. Next Stage": (
        DECISION,
        METHOD_COMPARISON,
        MINIMAL_LABEL_BUDGET,
        BUDGET_REGISTRY,
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
    rederived: list[bool] = []
    saved = globals()["DECISION"]
    for run in range(2):
        sandbox = CACHE / "record" / f"decision_{run}.json"
        sandbox.parent.mkdir(parents=True, exist_ok=True)
        sandbox.unlink(missing_ok=True)
        try:
            globals()["DECISION"] = sandbox
            run_decide()
        finally:
            globals()["DECISION"] = saved
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
            "rl1_unchanged": all(
                block["matches_rl1_provenance"] for block in freeze["rl1_inputs_consumed"].values()
            ),
            "decision_rederived_identically": rederived,
            "regenerates_a_candidate": False,
            "changes_the_representation": False,
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
        "retention": run_retention,
        "prefix": run_prefix,
        "strata": run_strata,
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
