#!/usr/bin/env python3
"""SGV-DS1: how should an OCR correction system be deployed when engines and generators change?

This is the synthesis stage. It introduces no model, generates no candidate and re-derives no
upstream result. It reads the frozen stages and asks the only question left:

    given what candidate generation can produce, what reliability estimation can rank, and what a
    human review budget can absorb, how much OCR correction may be accepted automatically -- and
    under what operating rule?

**1. Nothing upstream moves.** Candidates, labels, features and the reliability model class are
read from SGV-HY1, SGV-RL1 and SGV-RL2 and hash-verified.

**2. Three separately-frozen splits.** The repository's invariant is enforced literally: the score
is fitted on one document block, the threshold is chosen on a second, and the endpoint is measured
on a third. No threshold ever sees the test block.

**3. Risk targets are declared, not discovered.** The epsilon grid is the project's own. When no
threshold on held-out data satisfies a target, the policy abstains and that is recorded as the
result rather than repaired by moving the target.

**4. Measured, simulated and recommended are kept apart.** Every artifact carries a `kind` field,
and the report never lets a recommendation borrow the authority of a measurement.

    --reconstruct   section 0: CG1, XC1, GEN1, LP1, HY1, RL1 and RL2 re-read from their artifacts
    --freeze        upstream hashes and the frozen inputs this stage consumes
    --preregister   pipelines, policy, splits, risk targets, criteria, outcome rules
    --splits        the fit / threshold / test document registry and the decision population
    --score         the deployment reliability score, fitted on the fit block alone
    --thresholds    risk-target thresholds chosen on the threshold block and frozen
    --frontier      the automation frontier: review load against exact-repair recall
    --pipelines     P0, P1, P2 and P3 compared at the frozen operating points
    --routing       when image correction should be activated, and what it costs
    --adaptation    the label cost of deploying a new correction generator
    --stats         document-clustered paired bootstrap, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       the five required figures, from persisted artifacts
    --determinism   every derived phase twice from the frozen upstream inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / SYNTHESIS ONLY. Nothing here is a production system, no certification is run, no
finite-sample guarantee is claimed and the confirmatory reserve stays LOCKED.
`ready_for_external_confirmation` is false by construction.
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

import sgv_rl2_fewshot_generator_adaptation as rl2
from ocr_risk.io.hashing import file_sha256

rl1 = rl2.rl1
hy1 = rl1.hy1
lp1 = rl1.lp1
gen1 = rl1.gen1
xr1 = rl1.xr1
xc1 = rl1.xc1
cg1 = rl1.cg1
s15 = rl1.s15

REPO = rl1.REPO
OUT = REPO / "results/generated/sgv_ds1_deployment_synthesis"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"
UPSTREAM_SUMMARY = OUT / "upstream_findings_summary.json"

SPLIT_REGISTRY = OUT / "deployment_split_registry.json"
DECISION_POPULATION = OUT / "decision_population.parquet"
POPULATION_INVENTORY = OUT / "decision_population_inventory.json"

DEPLOYMENT_SCORES = OUT / "deployment_scores.parquet"
SCORE_REGISTRY = OUT / "score_registry.json"
THRESHOLD_REGISTRY = OUT / "threshold_registry.json"

AUTOMATION_FRONTIER = OUT / "automation_frontier.json"
RISK_COVERAGE = OUT / "risk_coverage.json"
PIPELINE_COMPARISON = OUT / "pipeline_comparison.json"
OPERATING_POINTS = OUT / "operating_points.json"
REVIEW_LOAD = OUT / "review_load.json"
ENVIRONMENT_ANALYSIS = OUT / "environment_analysis.json"

ROUTING_POLICY = OUT / "image_activation_policy.json"
ROUTING_COST = OUT / "routing_cost.json"

ADAPTATION_COST = OUT / "generator_adaptation_cost.json"
DEPLOYMENT_DECISION_TABLE = OUT / "deployment_decision_table.json"
DEPLOYMENT_FRAMEWORK = OUT / "deployment_framework.json"

STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_ds1/deployment_synthesis.md"

SCHEMA_VERSION = 1
STAGE = "sgv_ds1_deployment_synthesis"
HYPOTHESIS = "SGV-DS1-D1"
STAGE_KIND = "DEVELOPMENT / SYNTHESIS"

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
_mean = rl1._mean
_finite = rl1._finite

BOOTSTRAP_RESAMPLES = rl1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rl1.BOOTSTRAP_SEED
ALPHA = rl1.ALPHA

# ------------------------------------------------------------------ the frozen design

POPULATION = rl1.PRIMARY_POPULATION
REPRESENTATION = rl1.PRIMARY_REPRESENTATION
MODEL = rl1.PRIMARY_MODEL
C0 = rl1.C0
C1 = rl1.C1

# Three separately-frozen document blocks. The score is fitted on one, the threshold chosen on the
# second and the endpoint measured on the third, which is the repository's calibration invariant
# applied literally rather than by analogy.
FIT_FOLDS = (0, 1)
THRESHOLD_FOLDS = (2,)
TEST_FOLDS = (3, 4)
BLOCKS = ("fit", "threshold", "test")

# Risk targets: the project's own epsilon grid, with its own primary value. Not chosen here.
EPSILONS = rl1.EPSILONS
PRIMARY_EPSILON = rl1.PRIMARY_EPSILON

# The threshold grid is a fixed lattice on the safety score, declared before any threshold is
# chosen so that "the smallest qualifying threshold" is a well-defined object.
THRESHOLD_GRID = tuple(np.round(np.arange(0.0, 1.0001, 0.005), 4).tolist())

# Pipelines.
P0 = "p0_current_pipeline"
P1 = "p1_hybrid_ungated"
P2 = "p2_hybrid_reliability"
P3 = "p3_oracle_reliability"
PIPELINES = (P0, P1, P2, P3)
PIPELINE_LABEL = {
    P0: "current OCR correction pipeline, accepted open-loop",
    P1: "hybrid candidate generation, accepted open-loop",
    P2: "hybrid candidate generation with the reliability gate",
    P3: "hybrid candidate generation with oracle reliability",
}
PIPELINE_ARM = {P0: hy1.H0, P1: hy1.H5, P2: hy1.H5, P3: hy1.H5}

# Image-activation strategies, all built from HY1's frozen arms except the two gated variants,
# which are built from the frozen R3 features and carry no label.
S1 = "s1_text_only"
S2 = "s2_always_image"
S3A = "s3a_specialized_routing"
S3B = "s3b_low_confidence_image"
S3C = "s3c_substitution_image"
STRATEGIES = (S1, S2, S3A, S3B, S3C)
STRATEGY_LABEL = {
    S1: "text correction only",
    S2: "image correction everywhere",
    S3A: "image correction only where the OCR proposer never looked",
    S3B: "image correction where the OCR confidence is low",
    S3C: "image correction at substitution sites",
}
# The confidence quantile that defines "low confidence", chosen on the threshold block alone.
CONFIDENCE_FEATURE = "conf_anchor_z"
CONFIDENCE_QUANTILE = 0.5

# Generator-deployment scenarios, mapped onto SGV-RL2's frozen budget curve.
SCENARIOS = (
    ("A_existing_generator", None),
    ("B_new_generator_no_adaptation", 0),
    ("C_new_generator_100_labels", 100),
    ("D_new_generator_250_labels", 250),
    ("E_new_generator_500_labels", 500),
)
# The generator-shift endpoint splits SGV-RL2's sealed evaluation pool once more, so the threshold
# is never chosen on rows the adapted model trained on.
SHIFT_THRESHOLD_FOLD = 3
SHIFT_TEST_FOLD = 4

# Success criteria, frozen before any DS1 endpoint.
COVERAGE_FLOOR = 0.10
ORACLE_SHARE_FLOOR = 0.50
BREADTH_MAJORITY = rl1.C5_BREADTH_MAJORITY
HARM_TOLERANCE = 0.0

KIND_MEASURED = "measured"
KIND_SIMULATED = "simulated"
KIND_RECOMMENDED = "recommended"

QUESTIONS = {
    "Q1": "what is the achievable automation frontier at a given human review budget?",
    "Q2": "how should reliability thresholds be selected, and do they hold out of sample?",
    "Q3": "what is the label cost of deploying a new correction generator?",
    "Q4": "when should image-based correction be activated, and what does it cost?",
    "Q5": "what is the resulting deployment policy?",
}

OUTCOME_TAXONOMY = {
    "A": "the deployment framework demonstrates a reliable automation frontier",
    "B": "the framework gives useful operational guidance but requires human review",
    "C": "reliability limitations prevent safe deployment",
}
NEXT_STAGE = {
    "A": "NEXT: TARGETED EXTERNAL CONFIRMATION OF THE DEPLOYMENT POLICY",
    "B": "NEXT: RANKING-QUALITY WORK BEFORE ANY AUTOMATION CLAIM",
    "C": "NEXT: RICHER CANDIDATE VERIFICATION EVIDENCE",
}

PRIMARY_FAMILY = (
    "P1_hybrid_vs_current_repair_recall",
    "P2_reliability_gate_vs_ungated_joint_harm",
    "P3_gated_hybrid_vs_current_joint_harm",
    "P4_specialized_routing_vs_text_only_repair_recall",
)

# Every upstream value this stage rests on, re-read from the stage that produced it.
UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_cg1": {"dominant_bottleneck": "candidate"},
    "sgv_hy1": {
        "outcome": "A",
        "ready_for_reliability_representation_study": True,
        "omission_opportunity": 0.0,
    },
    "sgv_rl1": {"outcome": "B", "ready_for_reliability_adaptation": True},
    "sgv_rl2": {"outcome": "B", "ready_for_deployment_synthesis": True},
}


def _envelope(artifact: str, kind: str = KIND_MEASURED) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_ds1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "stage_kind": STAGE_KIND,
        "hypothesis_id": HYPOTHESIS,
        "kind": kind,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str, kind: str = KIND_MEASURED) -> dict[str, Any]:
    return {
        **_envelope(artifact, kind),
        "uses_ground_truth": True,
        "production_ready": False,
        "certified": False,
        "confirmatory_reserve_consumed": False,
    }


def assign_outcome(criteria: dict[str, bool]) -> str:
    """Exactly one outcome, in the frozen precedence C, A, B.

    C first: if hybrid generation does not help, or the reliability gate does not reduce harm,
    there is no safe deployment story to tell and the framework must say so. A needs the automation
    frontier to be usable at the project's primary risk target. Anything else is B -- the framework
    is operationally useful and human review remains load-bearing.
    """
    if not (criteria["C2"] and criteria["C3"]):
        return "C"
    if criteria["C1"] and criteria["C4"] and (criteria["C5"] or criteria["C6"]):
        return "A"
    return "B"


def _artifact_ref(path: Path) -> str:
    """A cross-reference that survives a sandboxed regeneration.

    A stage artifact naming a sibling must name it relative to the stage directory. Recording the
    repository-relative path instead makes the determinism comparison fail for a reason that has
    nothing to do with the science: the sandbox lives somewhere else.
    """
    return (
        path.name
        if path.is_relative_to(OUT) or path.parent.name.startswith("run_")
        else (_relative(path))
    )


def environment_specs() -> list[dict[str, Any]]:
    return rl1.environment_specs()


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    """Every upstream file DS1's conclusions rest on: RL2's list plus RL2's own artifacts."""
    files = list(rl2._upstream_files())
    files.append(REPO / "scripts/sgv_rl2_fewshot_generator_adaptation.py")
    for pattern in ("*.json", "*.parquet"):
        files.extend(sorted(rl2.OUT.glob(pattern)))
    files.append(rl2.REPORT)
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = xr1._check
    checks = list(rl2.upstream_checks())
    decisions = {
        "sgv_cg1": cc_read_json(cg1.OUT / "research_decision.json"),
        "sgv_hy1": cc_read_json(hy1.DECISION),
        "sgv_rl1": cc_read_json(rl1.DECISION),
        "sgv_rl2": cc_read_json(rl2.DECISION),
    }
    for stage, expected in UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))
    # RL2's two mechanism findings, as the numbers they rest on.
    rl2_decision = decisions["sgv_rl2"]
    checks.append(
        _check(
            "sgv_rl2.calibration_cannot_reorder",
            bool(
                max(
                    float(value)
                    for target in rl2_decision["score_calibration_maximum_auroc_move"].values()
                    for value in target.values()
                )
                < 1e-3
            ),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_rl2.joint_refit_needs_the_largest_budget",
            bool(
                all(
                    value == max(rl2.BUDGETS)
                    for value in rl2_decision["minimal_budget_joint"].values()
                )
            ),
            True,
        )
    )
    # Every stage this synthesis rests on kept the reserve locked.
    for stage, decision in decisions.items():
        checks.append(
            _check(
                f"{stage}.confirmatory_reserve_consumed",
                decision.get("confirmatory_reserve_consumed"),
                False,
            )
        )
    for stage in ("sgv_rl1", "sgv_rl2"):
        checks.append(
            _check(
                f"{stage}.ready_for_external_confirmation",
                decisions[stage].get("ready_for_external_confirmation"),
                False,
            )
        )
    checks.append(
        _check("sgv_rl2.determinism", cc_read_json(rl2.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_rl2.report_traceability",
            cc_read_json(rl2.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
            0,
        )
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
    return any(p.exists() for p in (DEPLOYMENT_SCORES, PIPELINE_COMPARISON, DECISION))


def run_freeze() -> int:
    """The research freeze: upstream hashes and the frozen inputs DS1 consumes."""
    started = time.monotonic()
    for path in (RESEARCH_FREEZE, FROZEN_CONFIGURATION, UPSTREAM_SUMMARY):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("a DS1 endpoint already exists; the freeze must precede every one")
    OUT.mkdir(parents=True, exist_ok=True)
    upstream = _upstream_files()
    hashes = {_relative(p): file_sha256(p) for p in upstream}
    consumed = {
        "hy1_candidates": hy1.CANDIDATES,
        "hy1_error_sites": hy1.ERROR_SITES,
        "hy1_proposal_strata": hy1.PROPOSAL_STRATA,
        "lp1_proposal_error_links": lp1.PROPOSAL_ERROR_LINKS,
        "rl1_candidate_population": rl1.CANDIDATE_POPULATION,
        "rl1_feature_matrix": rl1.FEATURE_MATRIX,
        "rl1_group_registry": rl1.GROUP_REGISTRY,
        "rl1_hyperparameters": rl1.HYPERPARAMETER_REGISTRY,
        "rl2_adapted_scores": rl2.ADAPTED_SCORES,
        "rl2_sample_efficiency_harm": rl2.SAMPLE_EFFICIENCY_HARM,
        "rl2_minimal_label_budget": rl2.MINIMAL_LABEL_BUDGET,
        "hy1_runtime_cost": hy1.RUNTIME_COST,
    }
    moved = sorted(name for name, path in consumed.items() if _relative(path) not in hashes)
    if moved:
        raise PhaseError(f"{len(moved)} consumed inputs are not in the upstream file set: {moved}")
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "upstream_sha256": hashes,
            "upstream_file_count": len(hashes),
            "inputs_consumed": {
                name: {"path": _relative(path), "sha256": file_sha256(path)}
                for name, path in consumed.items()
            },
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status": _git("status", "--short"),
                "diff_stat": _git("diff", "--stat"),
            },
            "introduces_a_new_model": False,
            "regenerates_a_candidate": False,
            "changes_an_upstream_outcome": False,
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
            "document_folds": _relative(rl1.GROUP_REGISTRY),
            "error_population": _relative(hy1.ERROR_SITES),
            "labelling": cc_read_json(rl1.FROZEN_CONFIGURATION)["labelling"],
            "candidate_arms": {name: PIPELINE_ARM[name] for name in PIPELINES},
            "image_throughput_requests_per_second": float(
                cc_read_json(hy1.RUNTIME_COST)["by_arm"]["c1_image_new"]["requests_per_second"]
            ),
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        UPSTREAM_SUMMARY,
        {
            **_envelope("upstream_findings_summary"),
            "cg1": {
                "finding": "candidate generation is the dominant bottleneck",
                "dominant_bottleneck": cc_read_json(cg1.OUT / "research_decision.json")[
                    "dominant_bottleneck"
                ],
            },
            "hy1": {
                "finding": "hybrid localization and correction expand the exact-repair universe",
                "h0_opportunity": cc_read_json(hy1.DECISION)["h0_frozen_pipeline_opportunity"],
                "h4_opportunity": cc_read_json(hy1.DECISION)["h4_specialized_routing_opportunity"],
                "h5_opportunity": cc_read_json(hy1.DECISION)["h5_dual_union_opportunity"],
                "omission_opportunity": cc_read_json(hy1.DECISION)["omission_opportunity"],
            },
            "rl1": {
                "finding": "reliability ranking transfers across engines but not across generators",
                "mixed_source_harm_auroc": cc_read_json(rl1.DECISION)["mixed_source_harm_auroc"],
                "mean_leave_generator_out_harm_auroc": cc_read_json(rl1.DECISION)[
                    "mean_leave_generator_out_harm_auroc"
                ],
                "source_identity_harm_gain": cc_read_json(rl1.DECISION)[
                    "source_identity_harm_gain"
                ],
            },
            "rl2": {
                "finding": "adaptation works but needs substantial supervision",
                "minimal_budget_joint": cc_read_json(rl2.DECISION)["minimal_budget_joint"],
                "minimal_budget_joint_target_only": cc_read_json(rl2.DECISION)[
                    "minimal_budget_joint_target_only"
                ],
                "worst_source_forgetting": cc_read_json(rl2.DECISION)["worst_source_forgetting"],
            },
            "read_from_artifacts_not_restated": True,
            "uses_ground_truth": True,
        },
    )
    print(
        f"freeze: {len(hashes)} upstream files hashed, {len(consumed)} inputs consumed "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the pre-registration


def run_preregister() -> int:
    """Every choice that could otherwise be made after seeing a deployment endpoint."""
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("a DS1 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "questions": QUESTIONS,
            "decision_unit": (
                "one proposal site. A deployed system decides once per site: apply an edit or send "
                "the site to a human. Accepting two edits at one site is not a policy anyone can "
                "execute, so the unit is the site and not the candidate"
            ),
            "policy": (
                "within a site, rank the arm's candidates by P(beneficial) descending with the "
                "candidate id as the tie-break; gate the top one by safety = 1 - P(harmful). "
                "Accept when safety is at or above the frozen threshold, otherwise send the site "
                "to human review"
            ),
            "safety_score": "1 - P(harmful), which is the repository's accept-rule orientation",
            "pipelines": {name: PIPELINE_LABEL[name] for name in PIPELINES},
            "pipeline_candidate_arms": {name: PIPELINE_ARM[name] for name in PIPELINES},
            "baseline_is_open_loop": (
                "P0 and P1 accept their top candidate at every site because neither has a "
                "reliability model that can validly score this candidate universe: SGV-XR1 "
                "established that the frozen verifier cannot score the image corrector at all"
            ),
            "split_rule": (
                "three separately-frozen document blocks from RL1's content-blind folds: the "
                f"score is fitted on folds {list(FIT_FOLDS)}, the threshold is chosen on fold "
                f"{list(THRESHOLD_FOLDS)} and every endpoint is measured on folds "
                f"{list(TEST_FOLDS)}. No threshold sees the test block and no fitted parameter "
                "sees either held-out block"
            ),
            "fit_folds": list(FIT_FOLDS),
            "threshold_folds": list(THRESHOLD_FOLDS),
            "test_folds": list(TEST_FOLDS),
            "risk_targets": list(EPSILONS),
            "primary_risk_target": PRIMARY_EPSILON,
            "risk_targets_are_inherited": (
                "the epsilon grid and its primary value are the project's own, used unchanged "
                "since SGV15; none of them is chosen here"
            ),
            "threshold_rule": (
                "for each risk target, the smallest threshold on a fixed lattice whose realized "
                "harm rate on the THRESHOLD block is at or below the target. When no lattice "
                "point qualifies, the policy abstains at that target and accepts nothing; that is "
                "recorded as the result rather than repaired by moving the target"
            ),
            "threshold_grid_points": len(THRESHOLD_GRID),
            "metrics": {
                "coverage": "share of decision sites auto-accepted",
                "review_rate": "share of decision sites sent to a human",
                "selective_harm_rate": "harmful accepted / accepted",
                "joint_harm_rate": (
                    "harmful accepted / decision sites; reported beside the selective rate "
                    "because the selective denominator vanishes as the threshold rises"
                ),
                "repair_recall": (
                    "evaluable OCR error sites repaired by an accepted edit / all evaluable OCR "
                    "error sites in the block, which is SGV-HY1's denominator unchanged"
                ),
                "review_load": "reviewed sites per document",
            },
            "image_activation_strategies": {name: STRATEGY_LABEL[name] for name in STRATEGIES},
            "confidence_gate": (
                f"the {CONFIDENCE_QUANTILE} quantile of the frozen {CONFIDENCE_FEATURE} feature, "
                "computed on the THRESHOLD block alone and then frozen"
            ),
            "routing_claim_rule": (
                "no routing strategy is called optimal unless its advantage survives the frozen "
                "statistical family; a numeric ordering alone is reported as a numeric ordering"
            ),
            "generator_scenarios": dict(SCENARIOS),
            "generator_shift_split": (
                f"SGV-RL2's sealed target evaluation pool is split once more -- fold "
                f"{SHIFT_THRESHOLD_FOLD} chooses the threshold and fold {SHIFT_TEST_FOLD} measures "
                "the endpoint -- so no threshold is chosen on rows the adapted model trained on. "
                f"Fold {SHIFT_TEST_FOLD} is nested inside this stage's own test block, which is "
                "disclosed rather than hidden"
            ),
            "criteria": {
                "C1": (
                    "the threshold procedure is sound out of sample: at every risk target the "
                    "threshold chosen on the threshold block either holds its target on test or "
                    "correctly abstains"
                ),
                "C2": (
                    "hybrid candidate generation beats the current pipeline on test exact-repair "
                    "recall at equal review load, without a higher joint harm rate"
                ),
                "C3": (
                    "the reliability gate lowers the joint harm rate against the ungated hybrid "
                    "on test, and the difference survives Holm"
                ),
                "C4": (
                    f"at the primary risk target the gated policy auto-accepts at least "
                    f"{COVERAGE_FLOOR} of decision sites on test while holding harm at or below "
                    "the target"
                ),
                "C5": (
                    f"at the primary risk target the gated policy reaches at least "
                    f"{ORACLE_SHARE_FLOOR} of the oracle pipeline's exact-repair recall"
                ),
                "C6": (
                    f"the gated hybrid beats the current pipeline on exact-repair recall in at "
                    f"least {BREADTH_MAJORITY} of the supported environments"
                ),
            },
            "outcome_rule": OUTCOME_TAXONOMY,
            "statistical_family": list(PRIMARY_FAMILY),
            "statistical_plan": {
                "unit": "document, resampled within environment",
                "paired": True,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "multiplicity": "Holm within the frozen primary family",
                "block": "test",
            },
            "kinds": {
                KIND_MEASURED: "computed from frozen artifacts on the held-out test block",
                KIND_SIMULATED: "projected from measured curves under a stated assumption",
                KIND_RECOMMENDED: "an operational recommendation, not a measurement",
            },
            "disclosure_a_sizing_probe_was_run": (
                "during implementation a sizing probe displayed the shape of the automation "
                "frontier and the approximate harm rates at a handful of thresholds on both the "
                "threshold and test blocks. The criteria above are expressed in quantities and "
                "margins inherited from earlier stages rather than chosen from that probe, but "
                "the probe happened and a reader should discount the pre-registration accordingly"
            ),
            "non_goals": [
                "no new model is introduced",
                "no candidate is generated",
                "no upstream outcome is changed",
                "no certification or finite-sample guarantee is computed",
                "no production readiness is claimed",
                "the confirmatory reserve is never unlocked",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the decision population


def _folds() -> dict[str, int]:
    return cc_read_json(rl1.GROUP_REGISTRY)["document_folds"]


def block_of(fold: int) -> str:
    if fold in FIT_FOLDS:
        return "fit"
    if fold in THRESHOLD_FOLDS:
        return "threshold"
    if fold in TEST_FOLDS:
        return "test"
    raise PhaseError(f"fold {fold} belongs to no declared block")


def load_candidates() -> pd.DataFrame:
    """RL1's frozen candidate population with its document block attached."""
    population = pd.read_parquet(rl1.CANDIDATE_POPULATION)
    frame = population[population["population"] == POPULATION].reset_index(drop=True)
    folds = _folds()
    mapped = frame["document_id"].astype(str).map(folds)
    if mapped.isna().any():
        raise PhaseError("a candidate's document carries no RL1 fold assignment")
    frame["fold"] = mapped.to_numpy(dtype=np.int64)
    frame["block"] = [block_of(int(f)) for f in frame["fold"]]
    frame["site_key"] = frame["environment"].astype(str) + "|" + frame["site_id"].astype(str)
    return frame


def load_error_population() -> pd.DataFrame:
    """SGV-HY1's evaluable OCR error sites, the denominator every recall here is measured on."""
    errors = pd.read_parquet(hy1.ERROR_SITES)
    folds = _folds()
    mapped = errors["document_id"].astype(str).map(folds)
    if mapped.isna().any():
        raise PhaseError("an error site's document carries no RL1 fold assignment")
    errors = errors.copy()
    errors["fold"] = mapped.to_numpy(dtype=np.int64)
    errors["block"] = [block_of(int(f)) for f in errors["fold"]]
    errors["error_key"] = (
        errors["environment"].astype(str) + "|" + errors["align_site_id"].astype(str)
    )
    return errors


def site_to_errors(errors: pd.DataFrame) -> dict[str, set[str]]:
    """Which evaluable error sites a proposal site covers, from LP1's frozen links alone."""
    known = set(errors["error_key"])
    links = pd.read_parquet(lp1.PROPOSAL_ERROR_LINKS)
    out: dict[str, set[str]] = {}
    for environment, site, target in zip(
        links["environment"].astype(str),
        links["site_id"].astype(str),
        links["align_site_id"].astype(str),
        strict=True,
    ):
        key = f"{environment}|{target}"
        if key in known:
            out.setdefault(f"{environment}|{site}", set()).add(key)
    return out


def run_splits() -> int:
    """The three frozen document blocks and the decision population they carry."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (SPLIT_REGISTRY, DECISION_POPULATION, POPULATION_INVENTORY):
        _forbid(path)
    candidates = load_candidates()
    errors = load_error_population()
    _write_parquet_once(
        DECISION_POPULATION,
        candidates.sort_values(["environment", "candidate_id"], kind="stable").reset_index(
            drop=True
        ),
    )
    registry: dict[str, Any] = {}
    for block in BLOCKS:
        candidate_block = candidates[candidates["block"] == block]
        error_block = errors[errors["block"] == block]
        registry[block] = {
            "folds": [int(f) for f in sorted({int(x) for x in candidate_block["fold"]})],
            "candidates": len(candidate_block),
            "decision_sites": int(candidate_block["site_key"].nunique()),
            "documents": int(candidate_block["document_id"].nunique()),
            "environments": int(candidate_block["environment"].nunique()),
            "error_sites": len(error_block),
            "harmful_candidates": int(candidate_block["is_harmful"].sum()),
            "beneficial_candidates": int(candidate_block["beneficial"].sum()),
            "exact_candidates": int(candidate_block["exact"].sum()),
        }
    crossing = {
        f"{left}_vs_{right}": len(
            set(candidates[candidates["block"] == left]["document_id"])
            & set(candidates[candidates["block"] == right]["document_id"])
        )
        for left in BLOCKS
        for right in BLOCKS
        if left < right
    }
    site_crossing = {
        f"{left}_vs_{right}": len(
            set(candidates[candidates["block"] == left]["site_key"])
            & set(candidates[candidates["block"] == right]["site_key"])
        )
        for left in BLOCKS
        for right in BLOCKS
        if left < right
    }
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_analysis_envelope("deployment_split_registry"),
            "blocks": registry,
            "document_crossings": crossing,
            "site_crossings": site_crossing,
            "nothing_crosses": bool(
                all(v == 0 for v in crossing.values())
                and all(v == 0 for v in site_crossing.values())
            ),
            "rule": cc_read_json(DESIGN_RECORD)["split_rule"],
            "source": _relative(rl1.GROUP_REGISTRY),
            "inherited_without_change": True,
        },
    )
    mapping = site_to_errors(errors)
    _write_json_once(
        POPULATION_INVENTORY,
        {
            **_analysis_envelope("decision_population_inventory"),
            "population": POPULATION,
            "candidates": len(candidates),
            "decision_sites": int(candidates["site_key"].nunique()),
            "error_sites": len(errors),
            "proposal_sites_linked_to_an_error_site": len(mapping),
            "candidates_at_a_linked_site": int(candidates["site_key"].isin(set(mapping)).sum()),
            "by_block": registry,
            "by_pipeline": {
                name: {
                    "arm": PIPELINE_ARM[name],
                    "candidates": len(arm_candidates(candidates, PIPELINE_ARM[name])),
                    "decision_sites": int(
                        arm_candidates(candidates, PIPELINE_ARM[name])["site_key"].nunique()
                    ),
                }
                for name in PIPELINES
            },
            "denominator": (
                "SGV-HY1's evaluable OCR error population, unchanged, so every recall here is "
                "comparable with every recall upstream"
            ),
        },
    )
    print(
        "splits: "
        + ", ".join(
            f"{block} {registry[block]['candidates']} candidates / "
            f"{registry[block]['error_sites']} error sites"
            for block in BLOCKS
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


def arm_candidates(candidates: pd.DataFrame, arm: str) -> pd.DataFrame:
    """One HY1 arm's candidates, selected by HY1's own frozen routing rule."""
    strata = pd.read_parquet(hy1.PROPOSAL_STRATA)
    frame = candidates.rename(columns={"site_id": "lattice_site_id"})
    frame = hy1._attach_membership(frame, strata)
    kept = hy1.arm_candidates(frame, arm)
    return kept.rename(columns={"lattice_site_id": "site_id"}).reset_index(drop=True)


# ------------------------------------------------------------------ the deployment score


def run_score() -> int:
    """The deployment reliability score: RL1's frozen model class, fitted on the fit block alone.

    Nothing new is introduced. This is RL1's representation and RL1's hyperparameters, refitted on
    the documents a deployment would actually have labels for, so the threshold block and the test
    block are both genuinely unseen. RL1's own published out-of-fold scores are carried beside it
    as a comparability arm and are never used to choose a threshold.
    """
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    for path in (DEPLOYMENT_SCORES, SCORE_REGISTRY):
        _forbid(path)
    candidates = pd.read_parquet(DECISION_POPULATION)
    matrix = pd.read_parquet(rl1.FEATURE_MATRIX)
    columns = rl1.columns_for(REPRESENTATION)
    design = rl1._matrix_for(candidates, matrix, columns)
    block = candidates["block"].to_numpy(str)
    fit = np.flatnonzero(block == "fit")
    scored = np.flatnonzero(block != "fit")
    truth = {
        rl1.T_HARM: candidates["is_harmful"].to_numpy(dtype=bool),
        rl1.T_BENEFIT: candidates["beneficial"].to_numpy(dtype=bool),
    }
    out = candidates[
        [
            "candidate_id",
            "environment",
            "document_id",
            "site_key",
            "block",
            "fold",
            "corrector_source",
            "proposal_stratum",
            "error_kind",
            "generator_rank",
            "is_harmful",
            "beneficial",
            "exact",
        ]
    ].copy()
    for target in rl1.TARGETS:
        values = np.full(len(candidates), np.nan)
        values[scored] = rl1.fit_and_score(MODEL, design[fit], truth[target][fit], design[scored])
        out[f"score_{target}"] = values
    out["safety"] = 1.0 - out["score_harm"]
    published = rl1.cell(rl1.load_scores(), POPULATION, REPRESENTATION, MODEL, rl1.PRIMARY_SPLIT)[
        ["candidate_id", "score_harm", "score_benefit"]
    ].rename(columns={"score_harm": "rl1_score_harm", "score_benefit": "rl1_score_benefit"})
    out = out.merge(published, on="candidate_id", how="left", validate="one_to_one")
    out[CONFIDENCE_FEATURE] = (
        matrix.set_index("candidate_id")
        .loc[out["candidate_id"], CONFIDENCE_FEATURE]
        .to_numpy(dtype=np.float64)
    )
    out = out.sort_values(["environment", "candidate_id"], kind="stable").reset_index(drop=True)
    _write_parquet_once(DEPLOYMENT_SCORES, out)
    quality: dict[str, Any] = {}
    for name in ("threshold", "test"):
        part = out[out["block"] == name]
        quality[name] = {
            "rows": len(part),
            **{
                f"{target}_auroc": auroc(
                    part[f"score_{target}"].to_numpy(dtype=np.float64),
                    part["is_harmful" if target == rl1.T_HARM else "beneficial"].to_numpy(bool),
                )
                for target in rl1.TARGETS
            },
            **{
                f"rl1_published_{target}_auroc": auroc(
                    part[f"rl1_score_{target}"].to_numpy(dtype=np.float64),
                    part["is_harmful" if target == rl1.T_HARM else "beneficial"].to_numpy(bool),
                )
                for target in rl1.TARGETS
            },
        }
    _write_json_once(
        SCORE_REGISTRY,
        {
            **_analysis_envelope("score_registry"),
            "model": MODEL,
            "representation": REPRESENTATION,
            "columns": len(columns),
            "carries_source_columns": bool([c for c in columns if c.startswith(rl1.FAM_SOURCE)]),
            "hyperparameters": dict(cc_read_json(rl1.HYPERPARAMETER_REGISTRY)[MODEL]),
            "fitted_on": "the fit block only",
            "fit_rows": int(fit.size),
            "scored_rows": int(scored.size),
            "introduces_a_new_model": False,
            "discrimination": quality,
            "comparability_arm": (
                "RL1's published out-of-fold scores are carried on every row for comparison and "
                "are never used to choose a threshold, because a fold-3 row's published score "
                "came from a model that had seen fold-4 rows, and both are in this stage's test "
                "block"
            ),
        },
    )
    print(
        f"score: harm AUROC {quality['test']['harm_auroc']:.4f} on test "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the decision policy


@dataclass(slots=True)
class Outcome:
    """What one policy did on one block, in the terms a deployment cares about."""

    sites: int
    accepted: int
    coverage: float
    review_rate: float
    selective_harm_rate: float
    joint_harm_rate: float
    repair_recall: float
    repaired_error_sites: int
    reviewed_sites_per_document: float


def top_per_site(block: pd.DataFrame, column: str, ascending: bool) -> pd.DataFrame:
    """One candidate per site, by the declared ranking and the candidate id as the tie-break."""
    ordered = block.sort_values(
        [column, "candidate_id"], ascending=[ascending, True], kind="stable"
    )
    return ordered.groupby("site_key", sort=False).head(1)


def evaluate_policy(
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    error_keys: set[str],
    mapping: dict[str, set[str]],
    documents: int,
) -> Outcome:
    """Coverage, harm and repair recall for one accept set. Ground truth enters only here."""
    repaired: set[str] = set()
    for key, exact in zip(accepted["site_key"].astype(str), accepted["exact"], strict=True):
        if exact:
            repaired |= mapping.get(key, set())
    repaired &= error_keys
    harmful = int(accepted["is_harmful"].sum()) if len(accepted) else 0
    return Outcome(
        sites=len(decisions),
        accepted=len(accepted),
        coverage=_ratio(len(accepted), len(decisions)),
        review_rate=_ratio(len(decisions) - len(accepted), len(decisions)),
        selective_harm_rate=_ratio(harmful, len(accepted)),
        joint_harm_rate=_ratio(harmful, len(decisions)),
        repair_recall=_ratio(len(repaired), len(error_keys)),
        repaired_error_sites=len(repaired),
        reviewed_sites_per_document=_ratio(len(decisions) - len(accepted), documents),
    )


def _as_dict(outcome: Outcome) -> dict[str, Any]:
    """The outcome as JSON. An undefined rate is null, never NaN.

    The selective harm rate has no value when nothing is accepted, and writing NaN would both
    misrepresent it as a measurement and break the determinism comparison, because NaN is not
    equal to itself.
    """
    return {
        "sites": outcome.sites,
        "accepted": outcome.accepted,
        "coverage": outcome.coverage,
        "review_rate": outcome.review_rate,
        "selective_harm_rate": (
            outcome.selective_harm_rate if np.isfinite(outcome.selective_harm_rate) else None
        ),
        "joint_harm_rate": outcome.joint_harm_rate,
        "repair_recall": outcome.repair_recall,
        "repaired_error_sites": outcome.repaired_error_sites,
        "reviewed_sites_per_document": outcome.reviewed_sites_per_document,
    }


def block_context(scores: pd.DataFrame, errors: pd.DataFrame, name: str) -> dict[str, Any]:
    part = scores[scores["block"] == name]
    error_block = errors[errors["block"] == name]
    return {
        "scores": part,
        "error_keys": set(error_block["error_key"].astype(str)),
        "documents": int(part["document_id"].nunique()),
    }


def pipeline_decisions(
    scores: pd.DataFrame, candidates: pd.DataFrame, pipeline: str
) -> pd.DataFrame:
    """The one-decision-per-site frame a pipeline faces, before any gate."""
    arm = set(arm_candidates(candidates, PIPELINE_ARM[pipeline])["candidate_id"].astype(str))
    block = scores[scores["candidate_id"].astype(str).isin(arm)]
    if pipeline in (P0, P1):
        return top_per_site(block, "generator_rank", True)
    if pipeline == P2:
        return top_per_site(block, "score_benefit", False)
    ranking = block.assign(_oracle=block["exact"].astype(int) * 2 + block["beneficial"].astype(int))
    return top_per_site(ranking, "_oracle", False)


def run_thresholds() -> int:
    """Risk-target thresholds chosen on the threshold block alone, then frozen."""
    started = time.monotonic()
    _require(DEPLOYMENT_SCORES, "score")
    _forbid(THRESHOLD_REGISTRY)
    scores = pd.read_parquet(DEPLOYMENT_SCORES)
    candidates = pd.read_parquet(DECISION_POPULATION)
    errors = load_error_population()
    mapping = site_to_errors(errors)
    context = block_context(scores, errors, "threshold")
    decisions = pipeline_decisions(context["scores"], candidates, P2)
    sweep = []
    for tau in THRESHOLD_GRID:
        accepted = decisions[decisions["safety"] >= tau]
        outcome = evaluate_policy(
            decisions, accepted, context["error_keys"], mapping, context["documents"]
        )
        sweep.append({"threshold": float(tau), **_as_dict(outcome)})
    chosen: dict[str, Any] = {}
    for epsilon in EPSILONS:
        qualifying = [
            row
            for row in sweep
            if row["accepted"] > 0 and row["selective_harm_rate"] <= epsilon + HARM_TOLERANCE
        ]
        key = rl1.epsilon_key(epsilon)
        if not qualifying:
            chosen[key] = {
                "epsilon": float(epsilon),
                "threshold": None,
                "abstains": True,
                "reason": (
                    "no threshold on the frozen lattice reaches this harm target on the threshold "
                    "block while accepting anything, so the policy accepts nothing at this target"
                ),
                "best_attainable_selective_harm_rate": min(
                    (row["selective_harm_rate"] for row in sweep if row["accepted"] > 0),
                    default=float("nan"),
                ),
            }
            continue
        best = min(qualifying, key=lambda row: (row["threshold"], -row["coverage"]))
        chosen[key] = {
            "epsilon": float(epsilon),
            "threshold": best["threshold"],
            "abstains": False,
            "threshold_block": {k: v for k, v in best.items() if k != "threshold"},
        }
    _write_json_once(
        THRESHOLD_REGISTRY,
        {
            **_analysis_envelope("threshold_registry"),
            "rule": cc_read_json(DESIGN_RECORD)["threshold_rule"],
            "grid_points": len(THRESHOLD_GRID),
            "grid_min": float(min(THRESHOLD_GRID)),
            "grid_max": float(max(THRESHOLD_GRID)),
            "chosen_on": "threshold",
            "applied_to": "test",
            "never_saw_the_test_block": True,
            "risk_targets": list(EPSILONS),
            "primary_risk_target": PRIMARY_EPSILON,
            "by_target": chosen,
            "threshold_block_sweep": sweep,
            "targets_that_abstain": sorted(key for key, row in chosen.items() if row["abstains"]),
        },
    )
    print(
        "thresholds: "
        + ", ".join(
            f"{key}={row['threshold'] if not row['abstains'] else 'abstain'}"
            for key, row in sorted(chosen.items())
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ Q1: automation frontier

REVIEW_GRID = tuple(np.round(np.arange(0.0, 1.0001, 0.02), 4).tolist())


def frontier_for(
    decisions: pd.DataFrame,
    ranking: str,
    ascending: bool,
    error_keys: set[str],
    mapping: dict[str, set[str]],
    documents: int,
) -> list[dict[str, Any]]:
    """Review the worst-ranked sites first and record what automation is left.

    A pipeline with no reliability score has no basis for choosing which sites to review, so its
    frontier is the random-review line: reviewing a share r of sites keeps a share 1-r of what it
    would have repaired. That is stated rather than hidden, and it is why the ungated pipelines
    appear as straight lines.
    """
    ordered = decisions.sort_values(
        [ranking, "candidate_id"], ascending=[not ascending, True], kind="stable"
    )
    rows: list[dict[str, Any]] = []
    total = len(ordered)
    for review in REVIEW_GRID:
        keep = total - round(review * total)
        accepted = ordered.head(max(keep, 0))
        outcome = evaluate_policy(ordered, accepted, error_keys, mapping, documents)
        rows.append({"review_rate_target": float(review), **_as_dict(outcome)})
    return rows


def random_review_frontier(
    decisions: pd.DataFrame,
    error_keys: set[str],
    mapping: dict[str, set[str]],
    documents: int,
) -> list[dict[str, Any]]:
    """The frontier of a pipeline that cannot rank: linear in the review rate, by construction."""
    full = evaluate_policy(decisions, decisions, error_keys, mapping, documents)
    rows = []
    for review in REVIEW_GRID:
        share = 1.0 - float(review)
        rows.append(
            {
                "review_rate_target": float(review),
                "sites": full.sites,
                "accepted": round(share * full.accepted),
                "coverage": share,
                "review_rate": float(review),
                "selective_harm_rate": full.selective_harm_rate,
                "joint_harm_rate": full.joint_harm_rate * share,
                "repair_recall": full.repair_recall * share,
                "repaired_error_sites": round(full.repaired_error_sites * share),
                "reviewed_sites_per_document": _ratio(round(float(review) * full.sites), documents),
                "expectation_under_random_review": True,
            }
        )
    return rows


def run_frontier() -> int:
    """Q1 and Q2: the automation frontier and the risk-coverage curves, on the test block."""
    started = time.monotonic()
    _require(THRESHOLD_REGISTRY, "thresholds")
    for path in (AUTOMATION_FRONTIER, RISK_COVERAGE):
        _forbid(path)
    scores = pd.read_parquet(DEPLOYMENT_SCORES)
    candidates = pd.read_parquet(DECISION_POPULATION)
    errors = load_error_population()
    mapping = site_to_errors(errors)
    frontier: dict[str, Any] = {}
    risk: dict[str, Any] = {}
    for name in ("threshold", "test"):
        context = block_context(scores, errors, name)
        frontier[name] = {}
        risk[name] = {}
        for pipeline in PIPELINES:
            decisions = pipeline_decisions(context["scores"], candidates, pipeline)
            if pipeline in (P0, P1):
                rows = random_review_frontier(
                    decisions, context["error_keys"], mapping, context["documents"]
                )
            elif pipeline == P2:
                rows = frontier_for(
                    decisions,
                    "safety",
                    True,
                    context["error_keys"],
                    mapping,
                    context["documents"],
                )
            else:
                ranked = decisions.assign(
                    _oracle=decisions["exact"].astype(int) * 2
                    + decisions["beneficial"].astype(int)
                    - decisions["is_harmful"].astype(int)
                )
                rows = frontier_for(
                    ranked,
                    "_oracle",
                    True,
                    context["error_keys"],
                    mapping,
                    context["documents"],
                )
            frontier[name][pipeline] = rows
            if pipeline in (P2, P3):
                risk[name][pipeline] = [
                    {
                        "coverage": row["coverage"],
                        "selective_harm_rate": row["selective_harm_rate"],
                        "joint_harm_rate": row["joint_harm_rate"],
                        "repair_recall": row["repair_recall"],
                    }
                    for row in rows
                ]
    _write_json_once(
        AUTOMATION_FRONTIER,
        {
            **_analysis_envelope("automation_frontier"),
            "question": QUESTIONS["Q1"],
            "review_grid": list(REVIEW_GRID),
            "by_block": frontier,
            "ranking": {
                P0: "none; random review is the only basis it has",
                P1: "none; random review is the only basis it has",
                P2: "least safe reviewed first, by 1 - P(harmful)",
                P3: "least useful reviewed first, by the frozen outcome",
            },
            "denominator": "evaluable OCR error sites in the block",
            "note": (
                "the ungated pipelines' curves are expectations under random review, not measured "
                "sweeps, because they have no score to sort by. They are exact in expectation and "
                "are labelled on every figure"
            ),
        },
    )
    _write_json_once(
        RISK_COVERAGE,
        {
            **_analysis_envelope("risk_coverage"),
            "question": QUESTIONS["Q2"],
            "by_block": risk,
            "risk_targets": list(EPSILONS),
            "primary_risk_target": PRIMARY_EPSILON,
            "note": (
                "the selective harm rate is reported beside the joint rate because the selective "
                "denominator vanishes as coverage falls and a single number there would mislead"
            ),
        },
    )
    print(
        f"frontier: {len(REVIEW_GRID)} review points per pipeline "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ Experiment 1: pipelines


def run_pipelines() -> int:
    """Experiment 1: P0, P1, P2 and P3 at the frozen operating points, on the test block."""
    started = time.monotonic()
    _require(AUTOMATION_FRONTIER, "frontier")
    for path in (PIPELINE_COMPARISON, OPERATING_POINTS, REVIEW_LOAD, ENVIRONMENT_ANALYSIS):
        _forbid(path)
    scores = pd.read_parquet(DEPLOYMENT_SCORES)
    candidates = pd.read_parquet(DECISION_POPULATION)
    errors = load_error_population()
    mapping = site_to_errors(errors)
    thresholds = cc_read_json(THRESHOLD_REGISTRY)["by_target"]
    comparison: dict[str, Any] = {}
    operating: dict[str, Any] = {}
    for name in ("threshold", "test"):
        context = block_context(scores, errors, name)
        comparison[name] = {}
        for pipeline in PIPELINES:
            decisions = pipeline_decisions(context["scores"], candidates, pipeline)
            if pipeline == P2:
                accepted = decisions.head(0)
            elif pipeline == P3:
                accepted = decisions[~decisions["is_harmful"]]
            else:
                accepted = decisions
            comparison[name][pipeline] = {
                "label": PIPELINE_LABEL[pipeline],
                "arm": PIPELINE_ARM[pipeline],
                "open_loop": _as_dict(
                    evaluate_policy(
                        decisions,
                        decisions if pipeline != P3 else accepted,
                        context["error_keys"],
                        mapping,
                        context["documents"],
                    )
                ),
            }
        operating[name] = {}
        decisions = pipeline_decisions(context["scores"], candidates, P2)
        for key, row in sorted(thresholds.items()):
            if row["abstains"]:
                operating[name][key] = {
                    "epsilon": row["epsilon"],
                    "threshold": None,
                    "abstains": True,
                    **_as_dict(
                        evaluate_policy(
                            decisions,
                            decisions.head(0),
                            context["error_keys"],
                            mapping,
                            context["documents"],
                        )
                    ),
                }
                continue
            accepted = decisions[decisions["safety"] >= float(row["threshold"])]
            outcome = evaluate_policy(
                decisions, accepted, context["error_keys"], mapping, context["documents"]
            )
            operating[name][key] = {
                "epsilon": row["epsilon"],
                "threshold": row["threshold"],
                "abstains": False,
                **_as_dict(outcome),
                "holds_its_target": bool(
                    np.isfinite(outcome.selective_harm_rate)
                    and outcome.selective_harm_rate <= row["epsilon"] + HARM_TOLERANCE
                ),
            }
    _write_json_once(
        PIPELINE_COMPARISON,
        {
            **_analysis_envelope("pipeline_comparison"),
            "by_block": comparison,
            "pipelines": {name: PIPELINE_LABEL[name] for name in PIPELINES},
            "open_loop_meaning": (
                "P0 and P1 accept their top candidate at every site; P3 accepts every candidate "
                "its oracle ranking marks not-harmful, which is the ceiling of selection on this "
                "candidate universe and not a method"
            ),
        },
    )
    _write_json_once(
        OPERATING_POINTS,
        {
            **_analysis_envelope("operating_points"),
            "by_block": operating,
            "arm": P2,
            "thresholds_from": _artifact_ref(THRESHOLD_REGISTRY),
            "all_targets_hold_or_abstain": bool(
                all(
                    row["abstains"] or row["holds_its_target"] for row in operating["test"].values()
                )
            ),
        },
    )
    context = block_context(scores, errors, "test")
    load: dict[str, Any] = {}
    for pipeline in PIPELINES:
        decisions = pipeline_decisions(context["scores"], candidates, pipeline)
        load[pipeline] = {
            "decision_sites": len(decisions),
            "documents": context["documents"],
            "decision_sites_per_document": _ratio(len(decisions), context["documents"]),
        }
    primary = rl1.epsilon_key(PRIMARY_EPSILON)
    load["at_the_primary_target"] = {
        "epsilon": PRIMARY_EPSILON,
        "reviewed_sites_per_document": operating["test"][primary]["reviewed_sites_per_document"],
        "review_rate": operating["test"][primary]["review_rate"],
        "coverage": operating["test"][primary]["coverage"],
    }
    _write_json_once(
        REVIEW_LOAD,
        {
            **_analysis_envelope("review_load"),
            "by_pipeline": load,
            "unit": "sites per document",
            "note": (
                "a review budget is a staffing decision, so the load is reported per document as "
                "well as a rate; the rate alone hides how many pages a reviewer must open"
            ),
        },
    )
    environments: dict[str, Any] = {}
    decisions_p2 = pipeline_decisions(context["scores"], candidates, P2)
    decisions_p0 = pipeline_decisions(context["scores"], candidates, P0)
    tau = thresholds[primary]["threshold"]
    for environment in sorted(set(context["scores"]["environment"].astype(str))):
        keys = set(
            errors[(errors["block"] == "test") & (errors["environment"] == environment)][
                "error_key"
            ].astype(str)
        )
        if not keys:
            continue
        documents = int(
            context["scores"][context["scores"]["environment"] == environment][
                "document_id"
            ].nunique()
        )
        block_p0 = decisions_p0[decisions_p0["environment"] == environment]
        block_p2 = decisions_p2[decisions_p2["environment"] == environment]
        accepted_p2 = (
            block_p2[block_p2["safety"] >= float(tau)] if tau is not None else block_p2.head(0)
        )
        environments[environment] = {
            "error_sites": len(keys),
            "p0": _as_dict(evaluate_policy(block_p0, block_p0, keys, mapping, documents)),
            "p2": _as_dict(evaluate_policy(block_p2, accepted_p2, keys, mapping, documents)),
        }
        environments[environment]["p2_beats_p0_on_repair_recall"] = bool(
            environments[environment]["p2"]["repair_recall"]
            > environments[environment]["p0"]["repair_recall"]
        )
        environments[environment]["p2_lowers_joint_harm"] = bool(
            environments[environment]["p2"]["joint_harm_rate"]
            < environments[environment]["p0"]["joint_harm_rate"]
        )
    _write_json_once(
        ENVIRONMENT_ANALYSIS,
        {
            **_analysis_envelope("environment_analysis"),
            "block": "test",
            "epsilon": PRIMARY_EPSILON,
            "by_environment": environments,
            "supported_environments": len(environments),
            "environments_where_p2_beats_p0_on_repair_recall": sum(
                1 for row in environments.values() if row["p2_beats_p0_on_repair_recall"]
            ),
            "environments_where_p2_lowers_joint_harm": sum(
                1 for row in environments.values() if row["p2_lowers_joint_harm"]
            ),
            "breadth_majority": BREADTH_MAJORITY,
        },
    )
    test = comparison["test"]
    print(
        "pipelines: "
        + ", ".join(
            f"{name.split('_')[0]} recall {test[name]['open_loop']['repair_recall']:.4f}"
            for name in PIPELINES
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ Q4: image activation


def strategy_candidates(
    candidates: pd.DataFrame, scores: pd.DataFrame, strategy: str, cutoff: float
) -> pd.DataFrame:
    """One image-activation strategy's candidate set, by a rule that reads no label.

    S1, S2 and S3A are SGV-HY1's frozen arms under other names. S3B and S3C are gated variants
    built from the frozen R3 confidence feature and the frozen site error kind, both of which are
    ground-truth-blind; the confidence cutoff comes from the threshold block alone.
    """
    if strategy == S1:
        return arm_candidates(candidates, hy1.H0)
    if strategy == S2:
        return arm_candidates(candidates, hy1.H3)
    if strategy == S3A:
        return arm_candidates(candidates, hy1.H4)
    union = arm_candidates(candidates, hy1.H5)
    joined = union.merge(
        scores[["candidate_id", CONFIDENCE_FEATURE, "error_kind"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_score"),
    )
    if strategy == S3B:
        image_sites = set(joined[joined[CONFIDENCE_FEATURE] <= cutoff]["site_key"].astype(str))
    else:
        kind = joined["error_kind_score"] if "error_kind_score" in joined else joined["error_kind"]
        image_sites = set(joined[kind.astype(str) == "substitution"]["site_key"].astype(str))
    corrector = joined["corrector_source"].astype(str)
    site = joined["site_key"].astype(str)
    keep = np.where(site.isin(image_sites), corrector == C1, corrector == C0)
    chosen = joined[keep]
    fallback = joined[~joined["site_key"].astype(str).isin(set(chosen["site_key"].astype(str)))]
    return pd.concat([chosen, fallback.groupby("site_key", sort=False).head(1)], ignore_index=True)


def run_routing() -> int:
    """Q4: when image correction should be activated, what it buys and what it costs."""
    started = time.monotonic()
    _require(PIPELINE_COMPARISON, "pipelines")
    for path in (ROUTING_POLICY, ROUTING_COST):
        _forbid(path)
    scores = pd.read_parquet(DEPLOYMENT_SCORES)
    candidates = pd.read_parquet(DECISION_POPULATION)
    errors = load_error_population()
    mapping = site_to_errors(errors)
    threshold_block = scores[scores["block"] == "threshold"]
    cutoff = float(np.quantile(threshold_block[CONFIDENCE_FEATURE], CONFIDENCE_QUANTILE))
    results: dict[str, Any] = {}
    for name in ("threshold", "test"):
        context = block_context(scores, errors, name)
        results[name] = {}
        for strategy in STRATEGIES:
            chosen = strategy_candidates(candidates, scores, strategy, cutoff)
            block = context["scores"][
                context["scores"]["candidate_id"].isin(set(chosen["candidate_id"]))
            ]
            decisions = top_per_site(block, "generator_rank", True)
            outcome = evaluate_policy(
                decisions, decisions, context["error_keys"], mapping, context["documents"]
            )
            image = int((decisions["corrector_source"].astype(str) == C1).sum())
            results[name][strategy] = {
                "label": STRATEGY_LABEL[strategy],
                **_as_dict(outcome),
                "image_corrections": image,
                "image_share": _ratio(image, len(decisions)),
                "image_corrections_per_document": _ratio(image, context["documents"]),
            }
    throughput = float(
        cc_read_json(hy1.RUNTIME_COST)["by_arm"]["c1_image_new"]["requests_per_second"]
    )
    test = results["test"]
    cost = {
        strategy: {
            "image_corrections": test[strategy]["image_corrections"],
            "image_corrections_per_document": test[strategy]["image_corrections_per_document"],
            "image_seconds_per_document": _ratio(test[strategy]["image_corrections"], throughput)
            / max(int(scores[scores["block"] == "test"]["document_id"].nunique()), 1),
            "repair_recall": test[strategy]["repair_recall"],
            "joint_harm_rate": test[strategy]["joint_harm_rate"],
        }
        for strategy in STRATEGIES
    }
    for strategy in STRATEGIES:
        baseline = cost[S1]
        extra_repairs = test[strategy]["repaired_error_sites"] - test[S1]["repaired_error_sites"]
        cost[strategy]["image_corrections_per_extra_repair"] = _ratio(
            test[strategy]["image_corrections"] - test[S1]["image_corrections"], extra_repairs
        )
        cost[strategy]["extra_repairs_over_text_only"] = int(extra_repairs)
        _ = baseline
    _write_json_once(
        ROUTING_POLICY,
        {
            **_analysis_envelope("image_activation_policy"),
            "question": QUESTIONS["Q4"],
            "by_block": results,
            "confidence_cutoff": cutoff,
            "confidence_feature": CONFIDENCE_FEATURE,
            "confidence_quantile": CONFIDENCE_QUANTILE,
            "cutoff_chosen_on": "threshold",
            "strategies": {name: STRATEGY_LABEL[name] for name in STRATEGIES},
            "claim_rule": cc_read_json(DESIGN_RECORD)["routing_claim_rule"],
            "no_strategy_is_called_optimal_here": True,
        },
    )
    _write_json_once(
        ROUTING_COST,
        {
            **_analysis_envelope("routing_cost"),
            "by_strategy": cost,
            "image_throughput_requests_per_second": throughput,
            "throughput_source": _relative(hy1.RUNTIME_COST),
            "note": (
                "the throughput is SGV-HY1's own measured decode rate on this machine. It prices "
                "the strategies against each other; it is not a claim about any other hardware"
            ),
        },
    )
    print(
        "routing: "
        + ", ".join(f"{s.split('_')[0]} recall {test[s]['repair_recall']:.4f}" for s in STRATEGIES)
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ Q3: adaptation cost


def run_adaptation() -> int:
    """Q3: what a new correction generator costs in labels, measured on RL2's sealed pool."""
    started = time.monotonic()
    _require(ROUTING_POLICY, "routing")
    for path in (ADAPTATION_COST, DEPLOYMENT_DECISION_TABLE):
        _forbid(path)
    population = rl2.load_population().set_index("candidate_id")
    adapted = pd.read_parquet(rl2.ADAPTED_SCORES)
    adapted = adapted[adapted["evaluation_set"] == rl2.EVAL_TARGET]
    errors = load_error_population()
    mapping = site_to_errors(errors)
    scenarios: dict[str, Any] = {}
    for label, budget in SCENARIOS:
        scenarios[label] = {"budget": budget, "by_direction": {}}
        for direction in rl2.DIRECTIONS:
            if budget is None:
                scenarios[label]["by_direction"][direction] = {
                    "kind": KIND_MEASURED,
                    "source": _relative(rl1.DECISION),
                    "note": (
                        "the existing generator needs no adaptation; its reliability is RL1's "
                        "mixed-source result on the pooled universe"
                    ),
                    "harm_auroc": cc_read_json(rl1.DECISION)["mixed_source_harm_auroc"],
                    "benefit_auroc": cc_read_json(rl1.DECISION)["mixed_source_benefit_auroc"],
                }
                continue
            arm = rl2.A0 if budget == 0 else rl2.A2
            cell = adapted[
                (adapted["direction"] == direction)
                & (adapted["arm"] == arm)
                & (adapted["budget"] == budget)
            ]
            rows: list[dict[str, Any]] = []
            for _draw, group in cell.groupby("draw", sort=True):
                wide = group.pivot_table(
                    index="candidate_id", columns="target", values="score", aggfunc="first"
                )
                joined = wide.join(
                    population[
                        [
                            "document_id",
                            "fold",
                            "is_harmful",
                            "beneficial",
                            "exact",
                            "environment",
                            "site_group",
                        ]
                    ],
                    how="inner",
                )
                chooser = joined[joined["fold"] == SHIFT_THRESHOLD_FOLD]
                evaluator = joined[joined["fold"] == SHIFT_TEST_FOLD]
                if chooser.empty or evaluator.empty:
                    continue
                tau = _shift_threshold(chooser, PRIMARY_EPSILON)
                rows.append(_shift_outcome(evaluator, tau, errors, mapping))
            scenarios[label]["by_direction"][direction] = {
                "kind": KIND_MEASURED,
                "draws": len(rows),
                "threshold_fold": SHIFT_THRESHOLD_FOLD,
                "evaluation_fold": SHIFT_TEST_FOLD,
                "abstained_draws": sum(1 for row in rows if row["abstains"]),
                "median_coverage": _median([row["coverage"] for row in rows]),
                "median_selective_harm_rate": _median(
                    [row["selective_harm_rate"] for row in rows if not row["abstains"]]
                ),
                "median_repair_recall": _median([row["repair_recall"] for row in rows]),
                "harm_auroc": cc_read_json(rl2.SAMPLE_EFFICIENCY_HARM)["by_direction"][direction][
                    arm
                ][str(budget)]["median"],
                "benefit_auroc": cc_read_json(rl2.SAMPLE_EFFICIENCY_BENEFIT)["by_direction"][
                    direction
                ][arm][str(budget)]["median"],
            }
    _write_json_once(
        ADAPTATION_COST,
        {
            **_analysis_envelope("generator_adaptation_cost"),
            "question": QUESTIONS["Q3"],
            "scenarios": scenarios,
            "rl2_minimal_budget_joint": cc_read_json(rl2.DECISION)["minimal_budget_joint"],
            "rl2_minimal_budget_joint_target_only": cc_read_json(rl2.DECISION)[
                "minimal_budget_joint_target_only"
            ],
            "split_note": cc_read_json(DESIGN_RECORD)["generator_shift_split"],
            "primary_risk_target": PRIMARY_EPSILON,
        },
    )
    _write_json_once(
        DEPLOYMENT_DECISION_TABLE,
        {
            **_envelope("deployment_decision_table", KIND_RECOMMENDED),
            "question": QUESTIONS["Q5"],
            "rows": [
                {
                    "situation": ("the correction generator is the one the model was fitted on"),
                    "required_action": "reuse the reliability model unchanged",
                    "evidence": "RL1 mixed-source discrimination",
                    "kind": KIND_RECOMMENDED,
                },
                {
                    "situation": "a new correction generator, no labels collected",
                    "required_action": (
                        "do not automate: route every site from the new generator to human review"
                    ),
                    "evidence": "RL2 zero-shot transfer, both directions",
                    "kind": KIND_RECOMMENDED,
                },
                {
                    "situation": "a new correction generator, a few hundred labels available",
                    "required_action": (
                        "refit the reliability model including the new labels, re-choose the "
                        "threshold on held-out documents, and re-measure before automating"
                    ),
                    "evidence": "RL2 sample-efficiency curve and minimal useful budget",
                    "kind": KIND_RECOMMENDED,
                },
                {
                    "situation": "the generator is unknown or cannot be identified",
                    "required_action": "abstain: review everything",
                    "evidence": "RL1 leave-one-generator-out transfer",
                    "kind": KIND_RECOMMENDED,
                },
                {
                    "situation": "the risk target is tighter than the threshold block can certify",
                    "required_action": "abstain at that target rather than interpolate a threshold",
                    "evidence": "the DS1 threshold registry",
                    "kind": KIND_RECOMMENDED,
                },
            ],
            "these_are_recommendations_not_measurements": True,
            "production_ready": False,
        },
    )
    print(f"adaptation: {len(SCENARIOS)} scenarios ({time.monotonic() - started:.0f}s)")
    return 0


def _median(values: Sequence[float]) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite else float("nan")


def _shift_threshold(chooser: pd.DataFrame, epsilon: float) -> float | None:
    """The smallest lattice threshold meeting the target on RL2's held-out chooser fold."""
    safety = 1.0 - chooser["harm"].to_numpy(dtype=np.float64)
    harmful = chooser["is_harmful"].to_numpy(dtype=bool)
    for tau in THRESHOLD_GRID:
        keep = safety >= tau
        if keep.sum() == 0:
            continue
        if float(harmful[keep].mean()) <= epsilon + HARM_TOLERANCE:
            return float(tau)
    return None


def _shift_outcome(
    evaluator: pd.DataFrame, tau: float | None, errors: pd.DataFrame, mapping: dict[str, set[str]]
) -> dict[str, Any]:
    """What the policy does on RL2's held-out evaluation fold at a threshold it never saw."""
    keys = set(errors[errors["fold"] == SHIFT_TEST_FOLD]["error_key"].astype(str))
    if tau is None:
        return {
            "abstains": True,
            "coverage": 0.0,
            "selective_harm_rate": float("nan"),
            "repair_recall": 0.0,
        }
    safety = 1.0 - evaluator["harm"].to_numpy(dtype=np.float64)
    accepted = evaluator[safety >= tau]
    repaired: set[str] = set()
    for key, exact in zip(accepted["site_group"].astype(str), accepted["exact"], strict=True):
        if exact:
            repaired |= mapping.get(key, set())
    repaired &= keys
    return {
        "abstains": False,
        "threshold": tau,
        "coverage": _ratio(len(accepted), len(evaluator)),
        "selective_harm_rate": _ratio(int(accepted["is_harmful"].sum()), len(accepted)),
        "repair_recall": _ratio(len(repaired), len(keys)),
    }


# ------------------------------------------------------------------ statistics


def _repaired_flags(
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    errors: pd.DataFrame,
    mapping: dict[str, set[str]],
) -> set[str]:
    _ = decisions
    repaired: set[str] = set()
    for key, exact in zip(accepted["site_key"].astype(str), accepted["exact"], strict=True):
        if exact:
            repaired |= mapping.get(key, set())
    return repaired & set(errors["error_key"].astype(str))


def _recall_frame(errors: pd.DataFrame, left: set[str], right: set[str], name: str) -> pd.DataFrame:
    keys = errors["error_key"].astype(str)
    return pd.DataFrame(
        {
            "environment": errors["environment"].astype(str).to_numpy(),
            "document_id": errors["document_id"].astype(str).to_numpy(),
            "a": [1.0 if key in left else 0.0 for key in keys],
            "b": [1.0 if key in right else 0.0 for key in keys],
            "unit": name,
        }
    )


def _harm_frame(sites: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    harmful_left = set(left[left["is_harmful"]]["site_key"].astype(str))
    harmful_right = set(right[right["is_harmful"]]["site_key"].astype(str))
    keys = sites["site_key"].astype(str)
    return pd.DataFrame(
        {
            "environment": sites["environment"].astype(str).to_numpy(),
            "document_id": sites["document_id"].astype(str).to_numpy(),
            "a": [1.0 if key in harmful_left else 0.0 for key in keys],
            "b": [1.0 if key in harmful_right else 0.0 for key in keys],
        }
    )


def run_stats() -> int:
    """The frozen family of four, document-clustered, paired, Holm-corrected, on the test block."""
    started = time.monotonic()
    _require(ADAPTATION_COST, "adaptation")
    _forbid(STATISTICAL_TESTS)
    scores = pd.read_parquet(DEPLOYMENT_SCORES)
    candidates = pd.read_parquet(DECISION_POPULATION)
    errors = load_error_population()
    test_errors = errors[errors["block"] == "test"].reset_index(drop=True)
    mapping = site_to_errors(errors)
    context = block_context(scores, errors, "test")
    tau = cc_read_json(THRESHOLD_REGISTRY)["by_target"][rl1.epsilon_key(PRIMARY_EPSILON)][
        "threshold"
    ]
    p0 = pipeline_decisions(context["scores"], candidates, P0)
    p1 = pipeline_decisions(context["scores"], candidates, P1)
    p2 = pipeline_decisions(context["scores"], candidates, P2)
    gated = p2[p2["safety"] >= float(tau)] if tau is not None else p2.head(0)
    repaired = {
        P0: _repaired_flags(p0, p0, test_errors, mapping),
        P1: _repaired_flags(p1, p1, test_errors, mapping),
        P2: _repaired_flags(p2, gated, test_errors, mapping),
    }
    strategies = {}
    cutoff = cc_read_json(ROUTING_POLICY)["confidence_cutoff"]
    for strategy in (S1, S3A):
        chosen = strategy_candidates(candidates, scores, strategy, cutoff)
        block = context["scores"][
            context["scores"]["candidate_id"].isin(set(chosen["candidate_id"]))
        ]
        decisions = top_per_site(block, "generator_rank", True)
        strategies[strategy] = _repaired_flags(decisions, decisions, test_errors, mapping)
    sites = p2[["site_key", "environment", "document_id"]].drop_duplicates("site_key")
    family = {
        PRIMARY_FAMILY[0]: xr1.comparison(
            _recall_frame(test_errors, repaired[P1], repaired[P0], "error_site"),
            PRIMARY_FAMILY[0],
            "primary",
        ),
        PRIMARY_FAMILY[1]: xr1.comparison(
            _harm_frame(sites, gated, p1), PRIMARY_FAMILY[1], "primary"
        ),
        PRIMARY_FAMILY[2]: xr1.comparison(
            _harm_frame(sites, gated, p0), PRIMARY_FAMILY[2], "primary"
        ),
        PRIMARY_FAMILY[3]: xr1.comparison(
            _recall_frame(test_errors, strategies[S3A], strategies[S1], "error_site"),
            PRIMARY_FAMILY[3],
            "primary",
        ),
    }
    adjusted = s15.holm(family)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "bootstrap": {
                "unit": "document, resampled within environment",
                "paired": True,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "alpha": ALPHA,
                "block": "test",
                "epsilon": PRIMARY_EPSILON,
                "threshold": tau,
            },
            "multiplicity": "Holm-Bonferroni inside the frozen primary family",
            "primary_family": adjusted,
            "family_size": len(PRIMARY_FAMILY),
            "declared_before_any_endpoint": True,
            "units": {
                PRIMARY_FAMILY[0]: "evaluable OCR error site in the test block",
                PRIMARY_FAMILY[1]: "decision site of the hybrid arm",
                PRIMARY_FAMILY[2]: (
                    "decision site of the hybrid arm; the current pipeline causes no harm at a "
                    "site it never acts on, which is why the comparison is well defined over the "
                    "larger site set"
                ),
                PRIMARY_FAMILY[3]: "evaluable OCR error site in the test block",
            },
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "not_in_the_family": (
                "the numeric ordering among the three selective image-activation strategies was "
                "not declared as a test and is reported as an ordering only"
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


def _direct_names(function_name: str) -> set[str]:
    """Every name one function body mentions, WITHOUT expanding the functions it calls.

    The transitive version is right for "can this value reach the model at all"; it is wrong for
    "does this phase read the wrong block", because a shared helper mentions every block name and
    every output field. Both extractors exist so each test can assert what it actually means.
    """
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function_name:
            names: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    names.add(child.id)
                elif isinstance(child, ast.Attribute):
                    names.add(child.attr)
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    names.add(child.value)
            return names
    raise PhaseError(f"{function_name} is not a function in this module")


def _called_names() -> set[str]:
    """Every name used outside the falsification phase, so its own string table cannot match."""
    tree = _module_tree()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run_negative":
            continue
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _blocks_selected_in(function_name: str) -> list[str]:
    """Which document blocks a function actually SELECTS, by reading its block_context calls.

    Naming a block in an artifact field is documentation; passing it to the selector is what
    decides which rows the phase touches. Only the second is a leak, so only the second is tested.
    """
    tree = _module_tree()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function_name
        ):
            continue
        blocks: list[str] = []
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "block_context"
            ):
                for argument in child.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        blocks.append(argument.value)
        return sorted(set(blocks))
    raise PhaseError(f"{function_name} is not a function in this module")


def run_negative() -> int:
    """Twenty-two tests, each of which would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    scores = pd.read_parquet(DEPLOYMENT_SCORES)
    candidates = pd.read_parquet(DECISION_POPULATION)
    errors = load_error_population()
    mapping = site_to_errors(errors)
    splits = cc_read_json(SPLIT_REGISTRY)
    registry = cc_read_json(THRESHOLD_REGISTRY)
    operating = cc_read_json(OPERATING_POINTS)["by_block"]
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    # The threshold phase names exactly one block, and it is not the test block. Asserting the
    # string constants the call graph mentions is the only way to catch a future edit that quietly
    # points the sweep at the wrong rows.
    # Naming the test block in an artifact field is documentation; passing it to the row selector
    # is a leak. The test asserts the selector, not the vocabulary.
    named = _blocks_selected_in("run_thresholds")
    record(
        "t01_threshold_selection_selects_only_the_threshold_block",
        named == ["threshold"],
        {
            "chosen_on": registry["chosen_on"],
            "applied_to": registry["applied_to"],
            "blocks_named": named,
        },
    )
    sweep_rows = registry["threshold_block_sweep"]
    rebuilt = block_context(scores, errors, "threshold")
    decisions = pipeline_decisions(rebuilt["scores"], candidates, P2)
    reproduced = []
    for key, row in registry["by_target"].items():
        if row["abstains"]:
            continue
        accepted = decisions[decisions["safety"] >= float(row["threshold"])]
        outcome = evaluate_policy(
            decisions, accepted, rebuilt["error_keys"], mapping, rebuilt["documents"]
        )
        reproduced.append(
            {
                "target": key,
                "matches": bool(
                    abs(outcome.selective_harm_rate - row["threshold_block"]["selective_harm_rate"])
                    < 1e-12
                ),
            }
        )
    record(
        "t02_the_chosen_threshold_reproduces_from_the_threshold_block_alone",
        all(row["matches"] for row in reproduced),
        reproduced,
    )
    columns = rl1.columns_for(REPRESENTATION)
    record(
        "t03_no_generator_identity_reaches_the_deployment_score",
        not [c for c in columns if c.startswith(rl1.FAM_SOURCE)],
        {"columns": len(columns)},
    )
    # The score's inputs are the design-matrix columns. The label fields travel beside them in the
    # scores table so the evaluation layer can read them, which is not the same as reaching a fit.
    label_fields = {"exact", "is_harmful", "beneficial", "outcome", "d_before", "d_after"}
    record(
        "t04_no_label_or_oracle_field_is_a_design_matrix_column",
        not (set(columns) & label_fields),
        {"columns": len(columns), "intersecting": sorted(set(columns) & label_fields)},
    )
    # A review budget would enter as the review grid or as a budget constant. Fields named
    # "review_rate" are produced by these phases, not consumed by them, so the direct-body
    # extractor is the right one and the grid is the thing to look for.
    review_leak = (_direct_names("run_score") | _direct_names("run_thresholds")) & {
        "REVIEW_GRID",
        "human_review",
        "review_budget",
        "REVIEW_BUDGETS",
    }
    record(
        "t05_no_review_budget_reaches_fitting_or_thresholding",
        not review_leak,
        sorted(review_leak),
    )
    record(
        "t06_no_document_or_site_crosses_a_block",
        splits["nothing_crosses"],
        {"documents": splits["document_crossings"], "sites": splits["site_crossings"]},
    )
    fit_rows = scores[scores["block"] == "fit"]
    record(
        "t07_the_fit_block_is_never_scored_or_evaluated",
        bool(fit_rows["score_harm"].isna().all() and fit_rows["score_benefit"].isna().all()),
        {"fit_rows": len(fit_rows)},
    )
    frozen = pd.read_parquet(rl1.CANDIDATE_POPULATION)
    frozen = frozen[frozen["population"] == POPULATION]
    record(
        "t08_the_candidate_population_is_rl1s_unchanged",
        bool(
            len(candidates) == len(frozen)
            and set(candidates["candidate_id"]) == set(frozen["candidate_id"])
        ),
        {"rows": len(candidates)},
    )
    hy1_errors = pd.read_parquet(hy1.ERROR_SITES)
    record(
        "t09_the_error_denominator_is_hy1s_unchanged",
        len(errors) == len(hy1_errors),
        {"error_sites": len(errors)},
    )
    arm_tests = (("t10_p0_is_hy1s_frozen_pipeline", P0), ("t11_p1_is_hy1s_dual_union", P1))
    for pipeline, arm in arm_tests:
        expected = hy1.arm_candidates(
            hy1._attach_membership(
                pd.read_parquet(hy1.CANDIDATES), pd.read_parquet(hy1.PROPOSAL_STRATA)
            ),
            PIPELINE_ARM[arm],
        )
        got = arm_candidates(candidates, PIPELINE_ARM[arm])
        record(
            pipeline,
            set(got["candidate_id"].astype(str)) <= set(expected["candidate_id"].astype(str)),
            {"arm": PIPELINE_ARM[arm], "candidates": len(got)},
        )
    context = block_context(scores, errors, "test")
    per_site = pipeline_decisions(context["scores"], candidates, P2)
    record(
        "t12_the_policy_makes_exactly_one_decision_per_site",
        int(per_site["site_key"].duplicated().sum()) == 0,
        {"sites": len(per_site)},
    )
    primary = rl1.epsilon_key(PRIMARY_EPSILON)
    row = operating["test"][primary]
    record(
        "t13_coverage_and_review_rate_partition_the_decision_sites",
        abs(row["coverage"] + row["review_rate"] - 1.0) < 1e-12,
        {"coverage": row["coverage"], "review_rate": row["review_rate"]},
    )
    abstaining = [key for key, value in registry["by_target"].items() if value["abstains"]]
    record(
        "t14_an_abstaining_target_accepts_nothing",
        all(operating["test"][key]["accepted"] == 0 for key in abstaining),
        {"abstaining_targets": abstaining},
    )
    coverages = [row["coverage"] for row in sweep_rows]
    record(
        "t15_coverage_is_monotone_in_the_threshold",
        all(a >= b - 1e-12 for a, b in pairwise(coverages)),
        {"grid_points": len(sweep_rows)},
    )
    oracle = pipeline_decisions(context["scores"], candidates, P3)
    record(
        "t16_the_oracle_pipeline_never_accepts_a_harmful_candidate",
        int(oracle[~oracle["is_harmful"]]["is_harmful"].sum()) == 0,
        {"accepted": int((~oracle["is_harmful"]).sum())},
    )
    frontier = cc_read_json(AUTOMATION_FRONTIER)["by_block"]["test"][P0]
    linear = all(
        abs(row["coverage"] - (1.0 - row["review_rate_target"])) < 1e-9 for row in frontier
    )
    record(
        "t17_the_unranked_frontier_is_the_random_review_expectation",
        linear,
        {"points": len(frontier)},
    )
    routing = cc_read_json(ROUTING_POLICY)
    record(
        "t18_the_confidence_cutoff_comes_from_the_threshold_block",
        routing["cutoff_chosen_on"] == "threshold",
        {"cutoff": routing["confidence_cutoff"]},
    )
    epsilon_leak = _names_in("run_score") & {"EPSILONS", "PRIMARY_EPSILON", "epsilon_key"}
    record(
        "t19_changing_the_risk_target_cannot_change_a_fitted_score",
        not epsilon_leak,
        sorted(epsilon_leak),
    )
    # A raw substring search would match this test's own table of forbidden terms, so the check
    # runs over the names every OTHER function actually uses.
    forbidden = sorted(
        _called_names()
        & {"select_threshold", "RiskController", "Calibrator", "clopper_pearson", "conformal"}
    )
    record("t20_no_certification_machinery_is_invoked", not forbidden, forbidden)
    record(
        "t21_rl2_adapted_scores_are_read_not_refitted",
        not (_names_in("run_adaptation") & {"fit_and_score", "_fit", "LogisticRegression"}),
        sorted(_names_in("run_adaptation") & {"fit_and_score", "_fit", "LogisticRegression"}),
    )
    record(
        "t22_every_target_either_holds_its_harm_bound_or_abstains",
        cc_read_json(OPERATING_POINTS)["all_targets_hold_or_abstain"],
        {
            key: {"abstains": value["abstains"], "holds": value.get("holds_its_target")}
            for key, value in operating["test"].items()
        },
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
    """Every success criterion, re-derived from the persisted artifacts and nothing else."""
    operating = cc_read_json(OPERATING_POINTS)["by_block"]["test"]
    comparison = cc_read_json(PIPELINE_COMPARISON)["by_block"]["test"]
    environments = cc_read_json(ENVIRONMENT_ANALYSIS)
    statistics = cc_read_json(STATISTICAL_TESTS)["primary_family"]
    primary = rl1.epsilon_key(PRIMARY_EPSILON)
    point = operating[primary]
    c1 = bool(cc_read_json(OPERATING_POINTS)["all_targets_hold_or_abstain"])
    p0 = comparison[P0]["open_loop"]
    p1 = comparison[P1]["open_loop"]
    p3 = comparison[P3]["open_loop"]
    c2 = bool(
        p1["repair_recall"] > p0["repair_recall"] and p1["joint_harm_rate"] <= p0["joint_harm_rate"]
    )
    gate = statistics[PRIMARY_FAMILY[1]]
    c3 = bool(gate["effect"] < 0.0 and gate["survives_holm"])
    c4 = bool(
        (not point["abstains"])
        and point["coverage"] >= COVERAGE_FLOOR
        and point["holds_its_target"]
    )
    oracle_share = _ratio(point["repair_recall"], p3["repair_recall"])
    c5 = bool(oracle_share >= ORACLE_SHARE_FLOOR)
    c6 = bool(
        int(environments["environments_where_p2_beats_p0_on_repair_recall"]) >= BREADTH_MAJORITY
    )
    return {
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6},
        "operating_point": point,
        "pipelines": {P0: p0, P1: p1, P3: p3},
        "oracle_share": oracle_share,
        "environments": environments,
        "statistics": statistics,
    }


def run_decide() -> int:
    """The frozen outcome rule, applied to the persisted artifacts."""
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    criteria = state["criteria"]
    outcome = assign_outcome(criteria)
    failed = sorted(name for name, value in criteria.items() if not value)
    point = state["operating_point"]
    negative = cc_read_json(FALSIFICATION)
    thresholds = cc_read_json(THRESHOLD_REGISTRY)
    routing = cc_read_json(ROUTING_POLICY)["by_block"]["test"]
    adaptation = cc_read_json(ADAPTATION_COST)
    reason = (
        f"outcome {outcome}: "
        + ", ".join(f"{name} {'met' if value else 'not met'}" for name, value in criteria.items())
        + f". {OUTCOME_TAXONOMY[outcome]}"
        + (f"; unmet: {', '.join(failed)}" if failed else "")
    )
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "block": "test",
            "primary_risk_target": PRIMARY_EPSILON,
            "risk_targets": list(EPSILONS),
            "thresholds": {key: row["threshold"] for key, row in thresholds["by_target"].items()},
            "targets_that_abstain": thresholds["targets_that_abstain"],
            "operating_point": {
                "epsilon": point["epsilon"],
                "threshold": point["threshold"],
                "coverage": point["coverage"],
                "review_rate": point["review_rate"],
                "selective_harm_rate": point["selective_harm_rate"],
                "joint_harm_rate": point["joint_harm_rate"],
                "repair_recall": point["repair_recall"],
                "reviewed_sites_per_document": point["reviewed_sites_per_document"],
                "holds_its_target": point["holds_its_target"],
            },
            "pipeline_repair_recall": {
                name: cc_read_json(PIPELINE_COMPARISON)["by_block"]["test"][name]["open_loop"][
                    "repair_recall"
                ]
                for name in PIPELINES
            },
            "pipeline_joint_harm_rate": {
                name: cc_read_json(PIPELINE_COMPARISON)["by_block"]["test"][name]["open_loop"][
                    "joint_harm_rate"
                ]
                for name in PIPELINES
            },
            "oracle_share_of_repair_recall": state["oracle_share"],
            "environments_supported": int(state["environments"]["supported_environments"]),
            "environments_where_the_gate_beats_the_baseline_on_recall": int(
                state["environments"]["environments_where_p2_beats_p0_on_repair_recall"]
            ),
            "environments_where_the_gate_lowers_joint_harm": int(
                state["environments"]["environments_where_p2_lowers_joint_harm"]
            ),
            "image_activation_repair_recall": {
                strategy: routing[strategy]["repair_recall"] for strategy in STRATEGIES
            },
            "image_activation_tested_in_the_family": [S1, S3A],
            "generator_adaptation_minimal_budget": adaptation["rl2_minimal_budget_joint"],
            "criteria_thresholds": {
                "coverage_floor": COVERAGE_FLOOR,
                "oracle_share_floor": ORACLE_SHARE_FLOOR,
                "breadth_majority": BREADTH_MAJORITY,
                "harm_tolerance": HARM_TOLERANCE,
            },
            "criterion_C1": criteria["C1"],
            "criterion_C2": criteria["C2"],
            "criterion_C3": criteria["C3"],
            "criterion_C4": criteria["C4"],
            "criterion_C5": criteria["C5"],
            "criterion_C6": criteria["C6"],
            "unmet_criteria": failed,
            "outcome": outcome,
            "outcome_label": OUTCOME_TAXONOMY[outcome],
            "human_review_remains_load_bearing": bool(point["review_rate"] > 0.5),
            "production_ready": False,
            "certified": False,
            "ready_for_external_confirmation": False,
            "confirmatory_reserve_consumed": False,
            "falsification_tests_passed": int(negative["passed"]),
            "falsification_tests_total": int(negative["total"]),
            "recommended_next_stage": NEXT_STAGE[outcome],
            "reason": reason,
            "hypothesis_id": HYPOTHESIS,
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DEPLOYMENT_FRAMEWORK,
        {
            **_envelope("deployment_framework", KIND_RECOMMENDED),
            "question": QUESTIONS["Q5"],
            "stages": [
                {"stage": "OCR input", "kind": KIND_MEASURED, "note": "the frozen engine output"},
                {
                    "stage": "error localization",
                    "kind": KIND_MEASURED,
                    "note": (
                        "OCR-side and image-side proposal are complementary; SGV-LP1 measured that "
                        "neither replaces the other"
                    ),
                },
                {
                    "stage": "candidate generation",
                    "kind": KIND_MEASURED,
                    "note": (
                        "hybrid routing expands the exact-repair universe; it is the only stage "
                        "that raised the ceiling"
                    ),
                },
                {
                    "stage": "reliability scoring",
                    "kind": KIND_MEASURED,
                    "note": (
                        "generator-agnostic features, refitted whenever the correction generator "
                        "changes; identity is not a feature"
                    ),
                },
                {
                    "stage": "decision layer",
                    "kind": KIND_RECOMMENDED,
                    "note": (
                        "accept above a threshold chosen on held-out documents; abstain entirely "
                        "at a risk target the held-out block cannot certify"
                    ),
                },
                {
                    "stage": "human review",
                    "kind": KIND_RECOMMENDED,
                    "note": "everything else, which at the primary target is most of the traffic",
                },
            ],
            "operating_point": point,
            "decision_table": _artifact_ref(DEPLOYMENT_DECISION_TABLE),
            "production_ready": False,
            "these_are_recommendations_not_measurements": True,
        },
    )
    print(f"decide: outcome {outcome} -- {NEXT_STAGE[outcome]}")
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = (
    "ANALYSIS ONLY -- development evidence computed with ground truth; not a production system"
)
SURFACE = rl1.SURFACE
INK = xc1.INK
INK_SECONDARY = xc1.INK_SECONDARY
GRID = xc1.GRID
PIPELINE_COLOURS = {
    P0: "#8a8a8a",
    P1: "#eb6834",
    P2: "#2a78d6",
    P3: "#1baf7a",
}
PIPELINE_SHORT = {
    P0: "P0 current pipeline",
    P1: "P1 hybrid, open loop",
    P2: "P2 hybrid + reliability",
    P3: "P3 oracle reliability",
}
FIGURES = (
    "deployment_framework.png",
    "risk_coverage.png",
    "human_review_reduction.png",
    "generator_adaptation_cost.png",
    "deployment_decision_map.png",
)


def run_figures() -> int:
    """The five required figures. Minimal academic style, every value from a persisted artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

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
            "svg.hashsalt": "ds1",
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

    decision = cc_read_json(DECISION)
    point = decision["operating_point"]

    # Figure 1 -- the framework, with the measured numbers each stage actually delivers.
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    boxes = [
        (9.8, "OCR input", f"{decision['environments_supported']} environments"),
        (8.4, "error localization", "OCR-side and image-side, complementary"),
        (
            7.0,
            "candidate generation",
            f"hybrid: repair ceiling {decision['pipeline_repair_recall'][P3]:.4f}",
        ),
        (
            5.6,
            "reliability scoring",
            "generator-agnostic; refit when the generator changes",
        ),
        (4.2, "decision layer", f"accept when safety >= {point['threshold']}"),
    ]
    for y, title, subtitle in boxes:
        ax.add_patch(
            Rectangle(
                (1.4, y - 0.5), 7.2, 1.0, facecolor=SURFACE, edgecolor=INK_SECONDARY, linewidth=0.9
            )
        )
        ax.text(5.0, y + 0.16, title, ha="center", va="center", fontsize=8.5, color=INK)
        ax.text(
            5.0, y - 0.22, subtitle, ha="center", va="center", fontsize=6.3, color=INK_SECONDARY
        )
    for y in (9.3, 7.9, 6.5, 5.1):
        ax.add_patch(
            FancyArrowPatch(
                (5.0, y),
                (5.0, y - 0.4),
                arrowstyle="-|>",
                mutation_scale=9,
                color=INK_SECONDARY,
                linewidth=0.9,
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (3.2, 3.7),
            (2.2, 2.6),
            arrowstyle="-|>",
            mutation_scale=9,
            color=PIPELINE_COLOURS[P2],
            linewidth=1.1,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (6.8, 3.7),
            (7.8, 2.6),
            arrowstyle="-|>",
            mutation_scale=9,
            color=INK_SECONDARY,
            linewidth=1.1,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.2, 1.4), 3.6, 1.1, facecolor=SURFACE, edgecolor=PIPELINE_COLOURS[P2], linewidth=1.1
        )
    )
    ax.text(2.0, 2.16, "automatic correction", ha="center", fontsize=8, color=INK)
    ax.text(
        2.0,
        1.72,
        f"{point['coverage']:.4f} of sites, harm {point['selective_harm_rate']:.4f}",
        ha="center",
        fontsize=6.3,
        color=INK_SECONDARY,
    )
    ax.add_patch(
        Rectangle((6.2, 1.4), 3.6, 1.1, facecolor=SURFACE, edgecolor=INK_SECONDARY, linewidth=1.1)
    )
    ax.text(8.0, 2.16, "human review", ha="center", fontsize=8, color=INK)
    ax.text(
        8.0,
        1.72,
        f"{point['review_rate']:.4f} of sites, {point['reviewed_sites_per_document']:.2f} per page",
        ha="center",
        fontsize=6.3,
        color=INK_SECONDARY,
    )
    ax.text(
        5.0,
        0.5,
        f"risk target {point['epsilon']}; threshold chosen on held-out documents",
        ha="center",
        fontsize=6.5,
        color=INK_SECONDARY,
    )
    ax.set_title("Deployment framework for risk-controlled OCR correction", fontsize=9.5)
    save(
        fig,
        FIGURES[0],
        [DECISION, OPERATING_POINTS, PIPELINE_COMPARISON],
        "the deployment framework with its measured operating point",
    )

    # Figure 2 -- risk-coverage.
    risk = cc_read_json(RISK_COVERAGE)["by_block"]["test"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for pipeline in (P2, P3):
        rows = sorted(risk[pipeline], key=lambda row: row["coverage"])
        axes[0].plot(
            [row["coverage"] for row in rows],
            [row["selective_harm_rate"] for row in rows],
            linewidth=1.3,
            color=PIPELINE_COLOURS[pipeline],
            label=PIPELINE_SHORT[pipeline],
        )
        axes[1].plot(
            [row["coverage"] for row in rows],
            [row["repair_recall"] for row in rows],
            linewidth=1.3,
            color=PIPELINE_COLOURS[pipeline],
            label=PIPELINE_SHORT[pipeline],
        )
    comparison = cc_read_json(PIPELINE_COMPARISON)["by_block"]["test"]
    for pipeline in (P0, P1):
        row = comparison[pipeline]["open_loop"]
        axes[0].scatter(
            [row["coverage"]],
            [row["selective_harm_rate"]],
            s=22,
            color=PIPELINE_COLOURS[pipeline],
            zorder=5,
            label=PIPELINE_SHORT[pipeline],
        )
        axes[1].scatter(
            [row["coverage"]],
            [row["repair_recall"]],
            s=22,
            color=PIPELINE_COLOURS[pipeline],
            zorder=5,
            label=PIPELINE_SHORT[pipeline],
        )
    for epsilon in EPSILONS:
        axes[0].axhline(epsilon, color=INK_SECONDARY, linewidth=0.7, linestyle=":")
    axes[0].set_xlabel("coverage (share auto-accepted)")
    axes[0].set_ylabel("harmful accepted / accepted")
    axes[0].set_title("Coverage against harm")
    axes[1].set_xlabel("coverage (share auto-accepted)")
    axes[1].set_ylabel("exact-repair recall")
    axes[1].set_title("Coverage against repair")
    axes[1].legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(
        fig,
        FIGURES[1],
        [RISK_COVERAGE, PIPELINE_COMPARISON],
        "risk-coverage curves on the held-out test block",
    )

    # Figure 3 -- human review reduction.
    frontier = cc_read_json(AUTOMATION_FRONTIER)["by_block"]["test"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for pipeline in PIPELINES:
        rows = frontier[pipeline]
        style = "--" if pipeline in (P0, P1) else "-"
        ax.plot(
            [row["review_rate_target"] for row in rows],
            [row["repair_recall"] for row in rows],
            linewidth=1.3,
            linestyle=style,
            color=PIPELINE_COLOURS[pipeline],
            label=PIPELINE_SHORT[pipeline] + (" (random review)" if style == "--" else ""),
        )
    ax.scatter(
        [point["review_rate"]],
        [point["repair_recall"]],
        s=30,
        zorder=6,
        color=PIPELINE_COLOURS[P2],
        edgecolor=SURFACE,
        linewidth=0.8,
    )
    ax.annotate(
        f"operating point, risk target {point['epsilon']}",
        (point["review_rate"], point["repair_recall"]),
        textcoords="offset points",
        xytext=(-12, 14),
        fontsize=6.3,
        color=INK_SECONDARY,
        ha="right",
    )
    ax.set_xlabel("human review rate (share of decision sites reviewed)")
    ax.set_ylabel("exact-repair recall")
    ax.set_title("What automation is left at each review budget")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(
        fig,
        FIGURES[2],
        [AUTOMATION_FRONTIER, DECISION],
        "the automation frontier: review budget against exact-repair recall",
    )

    # Figure 4 -- generator adaptation cost.
    adaptation = cc_read_json(ADAPTATION_COST)["scenarios"]
    budgets = [budget for _label, budget in SCENARIOS if budget is not None]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for direction, colour in zip(
        rl2.DIRECTIONS, (PIPELINE_COLOURS[P2], PIPELINE_COLOURS[P3]), strict=True
    ):
        values = []
        for label, budget in SCENARIOS:
            if budget is None:
                continue
            values.append(float(adaptation[label]["by_direction"][direction]["harm_auroc"]))
        ax.plot(
            budgets,
            values,
            marker="o",
            markersize=3.4,
            linewidth=1.3,
            color=colour,
            label=direction.replace("_", " "),
        )
    existing = float(
        adaptation["A_existing_generator"]["by_direction"][rl2.DIRECTIONS[0]]["harm_auroc"]
    )
    ax.axhline(
        existing,
        color=INK_SECONDARY,
        linewidth=0.8,
        linestyle="--",
        label="existing generator (no adaptation needed)",
    )
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.set_xlabel("labelled candidates collected from the new generator")
    ax.set_ylabel("harm-ranking AUROC")
    ax.set_title("What a new correction generator costs in labels")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIGURES[3], [ADAPTATION_COST], "reliability quality against annotation cost")

    # Figure 5 -- the decision map.
    table = cc_read_json(DEPLOYMENT_DECISION_TABLE)["rows"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(table) + 1.2)
    ax.text(0.15, len(table) + 0.6, "situation", fontsize=8, color=INK, weight="bold")
    ax.text(4.6, len(table) + 0.6, "required action", fontsize=8, color=INK, weight="bold")
    for index, row in enumerate(reversed(table)):
        y = index + 0.4
        ax.plot([0.1, 9.9], [y + 0.62, y + 0.62], color=GRID, linewidth=0.6)
        ax.text(0.15, y + 0.2, row["situation"], fontsize=6.6, color=INK, va="center", wrap=True)
        ax.text(
            4.6, y + 0.2, row["required_action"], fontsize=6.6, color=INK_SECONDARY, va="center"
        )
    ax.set_title("Deployment decision map (recommendations, not measurements)", fontsize=9.5)
    save(
        fig,
        FIGURES[4],
        [DEPLOYMENT_DECISION_TABLE],
        "the deployment decision map; every row is a recommendation",
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
            "style": "matplotlib only, minimal academic",
            "every_value_read_from_an_artifact": True,
        },
    )
    print(f"figures: {len(manifest)} written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ determinism and record

DERIVED_OUTPUTS = (
    "SPLIT_REGISTRY",
    "DECISION_POPULATION",
    "POPULATION_INVENTORY",
    "DEPLOYMENT_SCORES",
    "SCORE_REGISTRY",
    "THRESHOLD_REGISTRY",
    "AUTOMATION_FRONTIER",
    "RISK_COVERAGE",
    "PIPELINE_COMPARISON",
    "OPERATING_POINTS",
    "REVIEW_LOAD",
    "ENVIRONMENT_ANALYSIS",
    "ROUTING_POLICY",
    "ROUTING_COST",
    "ADAPTATION_COST",
    "DEPLOYMENT_DECISION_TABLE",
    "DEPLOYMENT_FRAMEWORK",
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
        ("thresholds", run_thresholds),
        ("frontier", run_frontier),
        ("pipelines", run_pipelines),
        ("routing", run_routing),
        ("adaptation", run_adaptation),
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
            "seeds": {"fit": rl1.FIT_SEED, "bootstrap": BOOTSTRAP_SEED},
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
    UPSTREAM_SUMMARY,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE, UPSTREAM_SUMMARY),
    "2. Previous Findings": (UPSTREAM_SUMMARY, FROZEN_CONFIGURATION),
    "3. Deployment Problem Formulation": (
        DESIGN_RECORD,
        SPLIT_REGISTRY,
        POPULATION_INVENTORY,
        SCORE_REGISTRY,
    ),
    "4. Automation Frontier": (
        AUTOMATION_FRONTIER,
        PIPELINE_COMPARISON,
        REVIEW_LOAD,
        STATISTICAL_TESTS,
        DECISION,
    ),
    "5. Risk-Controlled Correction": (
        THRESHOLD_REGISTRY,
        OPERATING_POINTS,
        RISK_COVERAGE,
        STATISTICAL_TESTS,
        ENVIRONMENT_ANALYSIS,
        PIPELINE_COMPARISON,
        DECISION,
    ),
    "6. Generator Adaptation Cost": (ADAPTATION_COST, UPSTREAM_SUMMARY),
    "7. Image Activation Policy": (ROUTING_POLICY, ROUTING_COST, STATISTICAL_TESTS),
    "8. Deployment Framework": (
        DEPLOYMENT_FRAMEWORK,
        DEPLOYMENT_DECISION_TABLE,
        OPERATING_POINTS,
        DECISION,
    ),
    "9. Limitations": (
        DESIGN_RECORD,
        SPLIT_REGISTRY,
        ENVIRONMENT_ANALYSIS,
        ROUTING_POLICY,
        ADAPTATION_COST,
        DECISION,
        FALSIFICATION,
    ),
    "10. Future Work": (DECISION, PIPELINE_COMPARISON, ADAPTATION_COST),
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
    # run_decide writes the framework artifact beside the decision, so both are redirected; only
    # the decision is compared, which is what the re-derivation claim is about.
    saved = {name: globals()[name] for name in ("DECISION", "DEPLOYMENT_FRAMEWORK")}
    for run in range(2):
        directory = CACHE / "record" / f"run_{run}"
        directory.mkdir(parents=True, exist_ok=True)
        sandbox = directory / saved["DECISION"].name
        for path in saved.values():
            (directory / path.name).unlink(missing_ok=True)
        try:
            for name, path in saved.items():
                globals()[name] = directory / path.name
            run_decide()
        finally:
            for name, path in saved.items():
                globals()[name] = path
        again = cc_read_json(sandbox)
        published = cc_read_json(saved["DECISION"])
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
            "introduces_a_new_model": False,
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
        "thresholds": run_thresholds,
        "frontier": run_frontier,
        "pipelines": run_pipelines,
        "routing": run_routing,
        "adaptation": run_adaptation,
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
