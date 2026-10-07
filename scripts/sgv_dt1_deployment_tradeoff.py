#!/usr/bin/env python3
"""SGV-DT1: risk x automation x human review x target annotation cost.

SGV15 asked whether a certified cut could preserve useful repair coverage and answered no.
SGV15b asked whether that failure was the inference unit, the action resolution, both or
neither, and answered something sharper than any of the four: the candidate-level certificate
really was anti-conservative under document clustering, the document-aware replacement really is
valid, the fine ladder really does recover the whole measured grid loss at the oracle -- and none
of it matters, because a distribution-free certificate for a candidate-weighted ratio needs on the
order of eight hundred independent target documents and the certification halves hold between
nineteen and fifty.

That is not a certification problem any more. Escalating the bound would answer a question whose
answer is already known.

    SGV-DT1: given the target-document and certification limits already measured, which
    combinations of harm tolerance, automated correction coverage, target annotation and human
    review produce a practically useful deployment regime -- and which do not?

**This stage measures a frontier. It is not built to make anything win.** The preferred output is
not "the method passes". It is the trade-off surface, with the parts that do not work stated as
plainly as the parts that do.

Nine decisions fix what the numbers below can mean.

**1. The model side is frozen shut.** SGV13's `a4_joint_refit`, SGV14's feature and candidate
universes, SGV15's fitting / certification document partition, SGV15's acquisition rule, SGV15's
250 adaptation labels, one fit per environment. No new ranking model, no new reliability model, no
new corrector, no new feature. Every arm below is downstream arithmetic on one frozen score table
per environment, which is both the honest design and the reason the sweep costs minutes.

**2. Automatic harm keeps its definition.** `Harm_auto` is harmful automatically applied edits over
all automatically applied edits. Reviewed corrections are NOT in that denominator. End-to-end
residual error -- what is still wrong after automatic edits, reviewed edits and preserved OCR --
is a different quantity with a different name, reported beside it and never substituted for it.

**3. Certified, empirically risk-controlled and unsafe are three different words.** A policy is
called certified only where it carries a valid finite-sample distribution-free certificate for the
operational estimand. A policy tuned to hold epsilon on revealed target labels is called
empirically risk-controlled, never certified, however well it does. SGV15's and SGV15b's frozen
certified procedures are carried through the whole stage as reference points precisely so the
price of leaving strict certification is measurable rather than asserted.

**4. Human review is not a correction oracle.** A perfect reviewer is simulated because it is the
upper bound, and it is labelled an upper bound in every artifact, table and figure. Two degraded
reviewers are simulated beside it under common random numbers, so the sensitivity is paired rather
than three separate experiments. No reviewer accuracy in this stage was measured on human beings.

**5. Review routing may not see an outcome label.** Which candidates are sent to review depends on
the frozen score, the deployed boundary and a declared budget. Ground truth is revealed only to
simulate what the reviewer then decides. The leakage suite asserts this against the source.

**6. Supervision is counted, never priced.** Cost is documents labelled, candidate labels revealed
and candidates reviewed. There is no empirical money or labour measurement in this project, so
there is no money or labour in this stage.

**7. The reviewed label is not free twice.** Adaptation labels and certification labels are counted
separately and summed. The primary scenario feeds no reviewed label back into the model; the reuse
scenario is secondary, sequential and may never let a reviewed label improve the decision that
routed that same candidate to review.

**8. Every category is declared before the endpoint.** The policy family, the routing strategies,
the review budgets, the reviewer accuracies, the document budgets, the deployment regimes and the
practicality criteria are written to `design_record.json` before any endpoint is read.

**9. This is DEVELOPMENT.** Every environment here was exposed by SGV14. Nothing in this stage may
be called confirmation, external validation, proven deployment or production readiness.

    --reconstruct   the section-0 freeze: SGV15 and SGV15b re-hashed and re-read
    --freeze        the frozen upstream configuration this stage may not vary
    --capacity      the environment inventory and what document budgets it can actually meet
    --preregister   policies, routings, budgets, reviewer scenarios, regimes and criteria
    --geometry      one adaptation fit per environment, cached as the deployment geometry
    --reproduce     the integrity gate against SGV15's and SGV15b's frozen cells
    --deploy        the main sweep: every policy x routing x review budget x document budget
    --review        routing comparison, review yield, harm interception
    --frontier      the risk, review, supervision and harm-tolerance frontiers
    --pareto        the non-dominated frontier and the full dominance relation
    --sensitivity   reviewer-accuracy sensitivity
    --sequential    the secondary batched-deployment and label-reuse scenario
    --controls      the ten required negative controls
    --negative      the falsification tests
    --stats         the pre-registered statistics
    --figures       the fourteen required figures
    --decide        the machine-readable development decision
    --determinism   two independent regenerations of the selected configuration
    --record        provenance, the dependency audit and the traceability index

DEVELOPMENT ONLY. The SGV1 CORD confirmatory reserve stays LOCKED and is absent from every
artifact. SGV13, SGV14, SGV15 and SGV15b artifacts are read and hash-verified; none is modified.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_verifier_pilot as pilot
import sgv13_fewshot_prefix_purity_adaptation as s13
import sgv14_confirmatory_validation as s14
import sgv15_target_risk_certification as s15
import sgv15b_document_level_certification as s15b
from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.risk import clopper_pearson_upper
from ocr_risk.risk.cluster_bounds import cluster_ratio_bound
from ocr_risk.risk.prefix_control import prefix_thresholds

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv_dt1_deployment_tradeoff"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"
CAPACITY_ANALYSIS = OUT / "capacity_analysis.json"
DESIGN_RECORD = OUT / "design_record.json"
DEPLOYMENT_POLICY_FAMILY = OUT / "deployment_policy_family.json"
REVIEW_BUDGET_FAMILY = OUT / "review_budget_family.json"
REVIEWER_ACCURACY_SCENARIOS = OUT / "reviewer_accuracy_scenarios.json"
TARGET_DOCUMENT_BUDGETS = OUT / "target_document_budgets.json"
SCORE_INVENTORY = OUT / "score_inventory.json"
GEOMETRY = OUT / "deployment_geometry.npz"
REPRODUCTION = OUT / "upstream_reproduction.json"
DEPLOYMENT_CELLS = OUT / "deployment_cells.parquet"
DOCUMENT_COUNTS = OUT / "document_counts.parquet"
AUTO_ONLY_RESULTS = OUT / "auto_only_results.json"
HUMAN_REVIEW_RESULTS = OUT / "human_review_results.json"
BOUNDARY_REVIEW_RESULTS = OUT / "boundary_review_results.json"
RANDOM_REVIEW_RESULTS = OUT / "random_review_results.json"
HIGH_RISK_REVIEW_RESULTS = OUT / "high_risk_review_results.json"
REVIEW_YIELD = OUT / "review_yield.json"
HARM_INTERCEPT = OUT / "harm_intercept.json"
HARM_TOLERANCE_FRONTIER = OUT / "harm_tolerance_frontier.json"
REVIEW_FRONTIER = OUT / "review_frontier.json"
SUPERVISION_FRONTIER = OUT / "supervision_frontier.json"
PARETO_FRONTIER = OUT / "pareto_frontier.json"
DOMINANCE_MATRIX = OUT / "dominance_matrix.json"
REVIEW_ACCURACY_SENSITIVITY = OUT / "review_accuracy_sensitivity.json"
SEQUENTIAL_DEPLOYMENT = OUT / "sequential_deployment.json"
CONTROL_RESULTS = OUT / "control_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
NESTED_SELECTION = OUT / "nested_selection.json"
DECISION = OUT / "development_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_dt1/deployment_tradeoff.md"

SCHEMA_VERSION = 1
STAGE = "sgv_dt1_deployment_tradeoff"
HYPOTHESIS = "SGV-DT1-Q1"

# ------------------------------------------------------------------ imported, never restated

PhaseError = s15.PhaseError
_relative = s15._relative
_git = s15._git
_write_json_once = s15._write_json_once
_write_parquet_once = s15._write_parquet_once
_ignored = s15._ignored
_tracked = s15._tracked
cc_read_json = s15.cc_read_json
_stable_seed = s15._stable_seed
_mean = s15._mean
holm = s15.holm
stratified_delta = s15.stratified_delta
fit_adaptation = s15.fit_adaptation
frozen_source_tau = s15.frozen_source_tau
per_document_counts = s13.per_document_counts

EPSILONS = s15.EPSILONS
PRIMARY_EPSILON = s15.PRIMARY_EPSILON
BOOTSTRAP_RESAMPLES = s13.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = s13.BOOTSTRAP_SEED
ALPHA = 0.05

# SGV15b's declared hundred-point ladder, imported rather than restated. SGV15b already froze it,
# already validated that it strictly refines SGV15's nine, and already measured that it recovers
# the whole oracle grid loss. Reusing it means this stage declares no new action family.
FINE_DEPTHS = s15b.FINE_DEPTHS
NINE_DEPTHS = s15.CUT_GRID

# ------------------------------------------------------------------ the pre-registered registry

SAMPLE_SEED = 20260909
REVIEWER_SEED = 4409
ROUTING_SEED = 771

# Section 8 axis C, as declared. Zero is the deployment that buys no target label at all; it is
# not an empty cell, it is the frozen source cut, which is the only operating point a system with
# no target supervision has. Budgets above an environment's capacity are clipped to it and
# recorded as clipped -- section 12 forbids manufacturing balanced capacities.
DOC_BUDGETS = (0, 5, 10, 20, 30, 50)
DOC_DRAWS = 20
# The largest declared budget that nine of the ten certification halves can meet exactly. The
# tenth holds nineteen documents and clips; that is reported, not hidden.
PRACTICAL_DOC_BUDGET = 20
# Section 34 control 9. One document cannot separate a document-level quantity from a single page.
CONTROL_DOC_BUDGET = 1

# Section 8 axis B, as declared: the maximum fraction of eligible candidates routed to review.
REVIEW_BUDGETS = (0.0, 0.05, 0.10, 0.20, 0.40)
PRACTICAL_REVIEW_BUDGET = 0.20

# Section 6 / section 22. These are SIMULATION SCENARIOS. No reviewer accuracy in this project was
# measured on human beings, and every artifact says so. H1 is an upper bound and is labelled one.
H0_NO_REVIEW = "h0_no_review"
H1_IDEAL = "h1_ideal_reviewer"
H2_HIGH = "h2_high_quality_reviewer"
H3_MODERATE = "h3_moderate_reviewer"
H4_COIN_FLIP = "h4_coin_flip_control"
REVIEWER_ACCURACY = {H1_IDEAL: 1.00, H2_HIGH: 0.98, H3_MODERATE: 0.95, H4_COIN_FLIP: 0.50}
# H4 is section 34 control 4 -- reviewer accuracy degradation -- and is a NEGATIVE CONTROL. It is
# measured everywhere and may never satisfy a criterion or be selected.
REVIEWER_SCENARIOS = (H1_IDEAL, H2_HIGH, H3_MODERATE)
CONTROL_SCENARIOS = (H4_COIN_FLIP,)
PRIMARY_SCENARIO = H2_HIGH
# Criterion 5 of section 49: a regime that only works with a perfect reviewer is not a regime.
NOT_PERFECT_REVIEWER_SCENARIO = H3_MODERATE

# How the automatic boundary is placed. Only the first two consume target labels from the revealed
# certification documents; `source` consumes none and `preserve` places no boundary at all.
CUT_EMPIRICAL = "empirical_risk_controlled"
CUT_CERTIFIED_CANDIDATE = "certified_candidate_clopper_pearson"
CUT_CERTIFIED_DOCUMENT = "certified_document_union_finite"
CUT_SOURCE = "frozen_source_cut"
CUT_PRESERVE = "preserve_all"

# Section 15's three words, kept apart. The label describes the CONSTRUCTION of the operating
# point, never its observed outcome; whether it held epsilon on the evaluation block is a separate
# measured field and the two are never merged.
SAFETY_CERTIFIED = "certified_safe"
SAFETY_EMPIRICAL = "empirically_risk_controlled"
SAFETY_NONE = "no_risk_control"

CUT_SAFETY = {
    CUT_EMPIRICAL: SAFETY_EMPIRICAL,
    CUT_CERTIFIED_CANDIDATE: SAFETY_CERTIFIED,
    CUT_CERTIFIED_DOCUMENT: SAFETY_CERTIFIED,
    CUT_SOURCE: SAFETY_NONE,
    CUT_PRESERVE: SAFETY_NONE,
}

# Section 18's routing strategies, plus the band the conservative policy reviews and the secondary
# expected-value ordering. A routing is a POOL and an ORDER over that pool, and both are computed
# from the frozen score and the deployed boundary alone.
ROUTE_NONE = "none"
ROUTE_BELOW = "below_boundary"
ROUTE_BOUNDARY = "boundary"
ROUTE_RANDOM = "random"
ROUTE_HIGH_RISK = "high_risk"
ROUTE_EXPECTED_VALUE = "expected_value"
ROUTE_SHUFFLED = "shuffled_score"
ROUTINGS = (ROUTE_BELOW, ROUTE_BOUNDARY, ROUTE_RANDOM, ROUTE_HIGH_RISK)

# Section 9's policy family and section 33's baselines are one table, because they are the same
# kind of object: a boundary rule, a routing rule and whether review is spent at all. B0 is P0 and
# B5 is P3; naming them twice would have let the same arm appear as two independent results.
P0_PRESERVE = "p0_preserve_all"
P1_AUTO_ONLY = "p1_auto_only"
P2_CONSERVATIVE = "p2_conservative_auto_review"
P3_BOUNDARY = "p3_boundary_review"
P4_EXPECTED_VALUE = "p4_expected_value_review"
B1_SOURCE = "b1_frozen_source_cut"
B2_SGV15 = "b2_sgv15_certified_candidate"
B3_SGV15B = "b3_sgv15b_certified_document"
B4_RANDOM_REVIEW = "b4_auto_random_review"
B6_HIGH_RISK_REVIEW = "b6_auto_high_risk_review"

# (arm, boundary rule, routing, reviews, primary)
POLICIES: tuple[tuple[str, str, str, bool, bool], ...] = (
    (P0_PRESERVE, CUT_PRESERVE, ROUTE_NONE, False, True),
    (P1_AUTO_ONLY, CUT_EMPIRICAL, ROUTE_NONE, False, True),
    (P2_CONSERVATIVE, CUT_CERTIFIED_CANDIDATE, ROUTE_BELOW, True, True),
    (P3_BOUNDARY, CUT_EMPIRICAL, ROUTE_BOUNDARY, True, True),
    (P4_EXPECTED_VALUE, CUT_EMPIRICAL, ROUTE_EXPECTED_VALUE, True, False),
    (B1_SOURCE, CUT_SOURCE, ROUTE_NONE, False, True),
    (B2_SGV15, CUT_CERTIFIED_CANDIDATE, ROUTE_NONE, False, True),
    (B3_SGV15B, CUT_CERTIFIED_DOCUMENT, ROUTE_NONE, False, True),
    (B4_RANDOM_REVIEW, CUT_EMPIRICAL, ROUTE_RANDOM, True, True),
    (B6_HIGH_RISK_REVIEW, CUT_EMPIRICAL, ROUTE_HIGH_RISK, True, True),
)
POLICY_NAMES = tuple(name for name, _, _, _, _ in POLICIES)
POLICY_RULE = {name: (cut, route, reviews) for name, cut, route, reviews, _ in POLICIES}
PRIMARY_POLICIES = tuple(name for name, _, _, _, primary in POLICIES if primary)
# P4 needs an outcome-rate map estimated on the revealed target labels. That is a fit, however
# small, so section 9 keeps it out of the primary family; it is measured and reported and may
# never be selected as the deployment regime.
SECONDARY_POLICIES = (P4_EXPECTED_VALUE,)
REVIEW_POLICIES = tuple(name for name, _, _, reviews, _ in POLICIES if reviews)
SELECTABLE_POLICIES = tuple(
    name for name in PRIMARY_POLICIES if name not in (P0_PRESERVE, B1_SOURCE)
)

# Section 18's pre-registered comparison, at matched budget on the same boundary.
R0_RANDOM = "r0_random_review"
R1_BOUNDARY = "r1_boundary_review"
R2_HIGH_RISK = "r2_high_risk_review"
ROUTING_ARMS = {
    R0_RANDOM: B4_RANDOM_REVIEW,
    R1_BOUNDARY: P3_BOUNDARY,
    R2_HIGH_RISK: B6_HIGH_RISK_REVIEW,
}

# Section 27's regimes, declared before any endpoint is read.
REGIME_A = "regime_a_fully_automatic"
REGIME_B = "regime_b_low_review"
REGIME_C = "regime_c_moderate_review"
REGIME_D = "regime_d_review_heavy"
REGIMES: tuple[tuple[str, float], ...] = (
    (REGIME_A, 0.0),
    (REGIME_B, 0.05),
    (REGIME_C, 0.20),
    (REGIME_D, 0.40),
)

# Section 20's two label-reuse scenarios.
L0_NO_REUSE = "l0_no_reuse"
L1_REUSE_AFTER_REVIEW = "l1_reuse_after_review"

ENVIRONMENT_COUNT = 10
BREADTH_REQUIRED = 7
MAJORITY_REQUIRED = 6
SEQUENTIAL_BATCHES = 4
DETERMINISM_RUNS = 2

# Section 29's minimum non-degenerate automation criterion is the project's existing convention,
# recovered from the frozen configuration rather than restated here. `load_floor` reads it.

# What section 0 expects to find frozen. If the repository disagrees the stage refuses to start.
SGV15_STATE = ("NOT SUPPORTED", 2, 6, "D")
SGV15B_STATE = ("NOT SUPPORTED", 3, 7, "D")

UPSTREAM_SCRIPTS = (
    "scripts/sgv_dt1_deployment_tradeoff.py",
    "scripts/sgv15b_document_level_certification.py",
    *s15b.UPSTREAM_SCRIPTS[1:],
)


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_dt1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": _iso(),
    }


def load_floor() -> float:
    """The project's existing non-degeneracy convention, read from SGV14's frozen record."""
    return float(s15.load_frozen().coverage_floor)


# ------------------------------------------------------------- section 0: the research freeze
#
# SGV-DT1's whole premise is two other stages' measured failure. It therefore reads those stages'
# artifacts rather than a prompt's description of them, re-hashes everything their own provenance
# recorded, and refuses to start on any discrepancy.


def _frozen_state(decision: Path) -> tuple[str, int, int, str]:
    record = cc_read_json(decision)
    return (
        str(record["verdict"]),
        int(record["criteria_met"]),
        int(record["criteria_total"]),
        str(record["outcome"]["label"]),
    )


def _verify_stage(
    tag: str, provenance: Path, decision: Path, expected: tuple[str, int, int, str]
) -> dict[str, Any]:
    if not provenance.is_file():
        raise PhaseError(f"{tag}'s provenance.json is missing; {tag} is not closed")
    observed = _frozen_state(decision)
    if observed != expected:
        raise PhaseError(
            f"{tag}'s frozen verdict is {observed[0]} {observed[1]}/{observed[2]} outcome "
            f"{observed[3]}, not the state SGV-DT1 was written against. Repository evidence is "
            "authoritative: resolve the discrepancy before SGV-DT1."
        )
    record = cc_read_json(decision)
    if bool(record["ready_for_sgv16"]):
        raise PhaseError(f"{tag} declared itself ready for SGV16; SGV-DT1's premise does not hold")
    if bool(record["confirmatory_reserve_consumed"]):
        raise PhaseError("the confirmatory reserve has been spent; SGV-DT1 may not proceed")
    artifacts = dict(cc_read_json(provenance)["artifacts"])
    drifted = [
        path
        for path, digest in artifacts.items()
        if (REPO / path).is_file() and file_sha256(REPO / path) != digest
    ]
    missing = [path for path in artifacts if not (REPO / path).is_file()]
    if drifted or missing:
        raise PhaseError(
            f"{tag} artifacts moved since it was frozen: {len(drifted)} changed, {len(missing)} "
            f"missing. {sorted(drifted + missing)[:4]}"
        )
    return {
        "verdict": observed[0],
        "criteria_met": observed[1],
        "criteria_total": observed[2],
        "outcome": observed[3],
        "ready_for_sgv16": False,
        "confirmatory_reserve_consumed": False,
        "artifacts_verified": len(artifacts),
    }


def verify_upstream() -> dict[str, Any]:
    """Section 0's eight expected findings, each read from the artifact that measured it."""
    sgv15 = _verify_stage("SGV15", s15.PROVENANCE, s15.DECISION, SGV15_STATE)
    sgv15b = _verify_stage("SGV15b", s15b.PROVENANCE, s15b.DECISION, SGV15B_STATE)

    decision = cc_read_json(s15b.DECISION)
    projection = cc_read_json(s15b.PROJECTION)
    inventory = cc_read_json(s15b.ENVIRONMENT_INVENTORY)
    simulation = cc_read_json(s15b.SIMULATION_COVERAGE)
    duplication = cc_read_json(s15b.DOCUMENT_DUPLICATION_TEST)
    resolution = cc_read_json(s15b.ORACLE_RESULTS)
    pools = [record["certification_documents"] for record in inventory["environments"]]

    cluster = decision["criteria"]["4_cluster_awareness_is_load_bearing"]
    findings = {
        "candidate_iid_inference_is_anti_conservative": {
            "candidate_false_certification_rate": float(
                cluster["simulation_candidate_false_certification_rate"]
            ),
            "document_false_certification_rate": float(
                cluster["simulation_document_false_certification_rate"]
            ),
            "nominal_alpha": float(decision["alpha"]),
            "holds": bool(
                float(cluster["simulation_candidate_false_certification_rate"])
                > float(decision["alpha"])
            ),
        },
        "document_aware_inference_is_safer": {
            "environments_where_the_candidate_bound_is_below": int(
                cluster["environments_where_the_candidate_bound_is_below"]
            ),
            "duplication_falsification_passes": bool(
                duplication["summary"]["falsification_passes"]
            ),
            "holds": bool(cluster["met"]),
        },
        "the_fine_grid_recovers_the_oracle_grid_loss": {
            "recovered_fraction": float(
                decision["criteria"]["5_fine_prefix_control_is_load_bearing"][
                    "oracle_recovered_fraction"
                ]
            ),
            "holds": bool(
                float(
                    decision["criteria"]["5_fine_prefix_control_is_load_bearing"][
                        "oracle_recovered_fraction"
                    ]
                )
                >= 1.0
            ),
        },
        "valid_procedures_still_refuse_almost_everywhere": {
            "environments_reaching_non_degenerate_deployment": int(
                decision["criteria"]["2_non_degenerate_safe_deployment"]["observed"]
            ),
            "required": int(decision["criteria"]["2_non_degenerate_safe_deployment"]["required"]),
            "holds": not bool(decision["criteria"]["2_non_degenerate_safe_deployment"]["met"]),
        },
        "the_document_requirement_is_out_of_reach": {
            "median_documents_required_finite_sample": float(
                projection["summary"]["median_documents_required_finite"]
            ),
            "median_documents_required_asymptotic": float(
                projection["summary"]["median_documents_required_asymptotic"]
            ),
            "smallest_certification_pool": int(min(pools)),
            "largest_certification_pool": int(max(pools)),
            "holds": bool(
                float(projection["summary"]["median_documents_required_finite"]) > max(pools)
            ),
        },
        "no_certification_method_supports_useful_breadth": {
            "best_arm_environments_at_a_practical_budget": max(
                int(record["environments"])
                for record in decision["outcome"]["evidence"][
                    "useful_safe_deployment_at_a_practical_budget"
                ].values()
            ),
            "majority_required": int(
                decision["outcome"]["evidence"]["useful_safe_deployment_at_a_practical_budget"][
                    "m00_candidate_nine_union"
                ]["majority_required"]
            ),
            "holds": not any(
                bool(record["useful"])
                for record in decision["outcome"]["evidence"][
                    "useful_safe_deployment_at_a_practical_budget"
                ].values()
            ),
        },
        "sgv16_remains_gated": {
            "sgv15_ready": False,
            "sgv15b_ready": False,
            "holds": True,
        },
    }
    failed = sorted(name for name, record in findings.items() if not record["holds"])
    if failed:
        raise PhaseError(
            "section 0's expected frozen findings disagree with the repository artifacts: "
            f"{failed}. Resolve the discrepancy before SGV-DT1; do not change SGV15b."
        )
    unused = float(simulation["nominal_alpha"]) if "nominal_alpha" in simulation else float("nan")
    return {
        "sgv15": sgv15,
        "sgv15b": sgv15b,
        "findings": findings,
        "simulation_nominal_alpha": unused,
        "resolution_artifact": _relative(s15b.ORACLE_RESULTS),
        "oracle_environments": len(resolution.get("per_environment", {})),
    }


def run_reconstruct() -> int:
    """Verify the frozen state SGV-DT1 builds on, and refuse to start if it has moved."""
    started = time.monotonic()
    upstream = verify_upstream()
    _reach = upstream["findings"]["the_document_requirement_is_out_of_reach"]
    payload = {
        **_envelope("research_freeze"),
        "issued_head": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "upstream": upstream,
        "why_certification_development_stops_here": (
            "SGV15b measured the median finite-sample requirement at "
            f"{_reach['median_documents_required_finite_sample']:.0f} independent target "
            "documents against certification halves holding between "
            f"{_reach['smallest_certification_pool']} and "
            f"{_reach['largest_certification_pool']}. "
            "A further threshold heuristic cannot close a gap of that shape, so SGV-DT1 asks what "
            "deployment is achievable under that limit rather than how to certify around it."
        ),
        "verification": s15b._verification_state(),
        "upstream_script_sha256": {
            path: file_sha256(REPO / path) for path in sorted(UPSTREAM_SCRIPTS)
        },
        "confirmatory_reserve": "LOCKED. SGV-DT1 is a development stage and may not spend it.",
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(RESEARCH_FREEZE, payload)
    print(
        f"reconstruct: SGV15 {upstream['sgv15']['verdict']} "
        f"{upstream['sgv15']['criteria_met']}/{upstream['sgv15']['criteria_total']} outcome "
        f"{upstream['sgv15']['outcome']}, SGV15b {upstream['sgv15b']['verdict']} "
        f"{upstream['sgv15b']['criteria_met']}/{upstream['sgv15b']['criteria_total']} outcome "
        f"{upstream['sgv15b']['outcome']}, "
        f"{upstream['sgv15']['artifacts_verified'] + upstream['sgv15b']['artifacts_verified']} "
        f"artifacts re-hashed -> {_relative(RESEARCH_FREEZE)}"
    )
    return 0


# ------------------------------------------------------- section 2: the frozen model side
#
# SGV-DT1 varies deployment policy. It varies nothing that produces a score. Everything the
# ranking depends on is recovered from the frozen artifacts and asserted, not restated.


def run_freeze() -> int:
    """Everything SGV-DT1 inherits and may not vary, recovered from the frozen artifacts."""
    started = time.monotonic()
    if not RESEARCH_FREEZE.is_file():
        raise PhaseError("run --reconstruct first")
    frozen = s15.load_frozen()
    s15.assert_frozen(frozen)
    inherited = cc_read_json(s15b.FROZEN_CONFIGURATION)
    payload = {
        **_envelope("frozen_upstream_configuration"),
        "configuration_sha256": frozen.configuration_sha256,
        "frozen_ranking": inherited["frozen_ranking"],
        "frozen_partition": inherited["frozen_partition"],
        "inherited_primary_epsilon": float(frozen.epsilon),
        "inherited_coverage_floor": float(frozen.coverage_floor),
        "inherited_epsilon_grid": [float(value) for value in EPSILONS],
        "inherited_action_family": {
            "source": _relative(s15b.PREFIX_FAMILY),
            "family": "fine",
            "size": len(FINE_DEPTHS),
            "why": (
                "SGV15b declared this hundred-point ladder before any of its endpoints, validated "
                "that it strictly refines SGV15's nine, and measured that it recovers the whole "
                "oracle grid loss. Reusing it means SGV-DT1 declares no new action family and no "
                "new resolution question."
            ),
            "hash": cc_read_json(s15b.PREFIX_FAMILY)["family_hash"],
        },
        "what_sgv_dt1_may_not_vary": [
            "the adaptation arm, its weighting, its regularisation or its feature universe",
            "the candidate universe, the harm definition or the label definitions",
            "the ranking score, the reliability model or the correction generator",
            "the primary epsilon",
            "the fitting / certification / evaluation document partition",
            "any SGV13, SGV14, SGV15 or SGV15b artifact",
        ],
        "what_sgv_dt1_varies": [
            "the deployment action set: automatic application, human review, preservation",
            "where the automatic boundary is placed and how it is justified",
            "how much human review is bought and which candidates it is spent on",
            "the simulated reviewer's accuracy",
            "the harm tolerance and the target-document supervision budget",
        ],
        "not_introduced": [
            "ByT5",
            "GPT or other LLM correction",
            "VLM correction",
            "multimodal post-correction",
            "external correctors",
            "a new certification method",
            "a new ranking or reliability model",
        ],
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(FROZEN_CONFIGURATION, payload)
    print(
        f"freeze: {frozen.arm} / {frozen.acquisition} / {frozen.adapt_budget} adaptation labels, "
        f"epsilon {frozen.epsilon}, coverage floor {frozen.coverage_floor:.6f} -> "
        f"{_relative(FROZEN_CONFIGURATION)}"
    )
    return 0


# ------------------------------------------------- section 32: capacity, reported not engineered


def run_capacity() -> int:
    """The document inventory and which of this stage's budgets each environment can meet."""
    started = time.monotonic()
    if not FROZEN_CONFIGURATION.is_file():
        raise PhaseError("run --freeze first")
    certification = s15b._certification_table()
    upstream = {
        record["environment"]: record
        for record in cc_read_json(s15b.ENVIRONMENT_INVENTORY)["environments"]
    }

    records: list[dict[str, Any]] = []
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        block = certification[certification["environment"] == name]
        per_document = block.groupby("document_id", observed=True).size()
        sizes = np.asarray(sorted(per_document.to_numpy().tolist()), dtype=float)
        available = int(per_document.size)
        clipped = [budget for budget in DOC_BUDGETS if budget > available]
        records.append(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "base_engine": spec["base_engine"],
                "certification_documents": available,
                "certification_candidates": len(block),
                "certification_harmful": int(block["is_harmful"].sum()),
                "candidates_per_document_mean": float(sizes.mean()) if sizes.size else 0.0,
                "candidates_per_document_median": float(np.median(sizes)) if sizes.size else 0.0,
                "candidates_per_document_min": int(sizes.min()) if sizes.size else 0,
                "candidates_per_document_max": int(sizes.max()) if sizes.size else 0,
                "evaluation_rows": int(upstream[name]["evaluation_rows"]),
                "evaluation_documents": int(upstream[name]["evaluation_documents"]),
                "largest_feasible_document_budget": max(
                    [budget for budget in DOC_BUDGETS if budget <= available] or [0]
                ),
                "clipped_budgets": clipped,
                "capacity_limited": bool(clipped),
                "meets_the_practical_budget_exactly": bool(available >= PRACTICAL_DOC_BUDGET),
                "sgv15b_certification_documents": int(upstream[name]["certification_documents"]),
            }
        )

    drifted = [
        record["environment"]
        for record in records
        if record["certification_documents"] != record["sgv15b_certification_documents"]
    ]
    if drifted:
        raise PhaseError(f"the certification halves moved since SGV15b: {drifted}")

    limited = [record["environment"] for record in records if record["capacity_limited"]]
    inventory = {
        **_envelope("environment_inventory"),
        "environments": records,
        "capacity_limited": limited,
        "totals": {
            "environments": len(records),
            "certification_documents": int(sum(r["certification_documents"] for r in records)),
            "certification_candidates": int(sum(r["certification_candidates"] for r in records)),
            "evaluation_rows": int(sum(r["evaluation_rows"] for r in records)),
            "evaluation_documents": int(sum(r["evaluation_documents"] for r in records)),
        },
        "scientific_status": "DEVELOPMENT -- every environment here was exposed by SGV14",
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(ENVIRONMENT_INVENTORY, inventory)

    by_budget = {
        str(budget): {
            "environments_that_can_meet_it": sorted(
                record["environment"]
                for record in records
                if record["certification_documents"] >= budget
            ),
            "environments_that_clip": sorted(
                record["environment"]
                for record in records
                if record["certification_documents"] < budget
            ),
        }
        for budget in DOC_BUDGETS
    }
    _write_json_once(
        CAPACITY_ANALYSIS,
        {
            **_envelope("capacity_analysis"),
            "declared_budgets": list(DOC_BUDGETS),
            "practical_budget": PRACTICAL_DOC_BUDGET,
            "by_budget": by_budget,
            "capacity_limited": limited,
            "smallest_certification_pool": int(min(r["certification_documents"] for r in records)),
            "largest_certification_pool": int(max(r["certification_documents"] for r in records)),
            "rule": (
                "a budget above an environment's capacity is CLIPPED to that capacity and the "
                "achieved count is recorded. No environment is dropped, no document is "
                "duplicated, and no nominal budget is reported as achieved."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"capacity: {len(records)} environments, "
        f"{inventory['totals']['certification_documents']} certification documents, "
        f"{len(limited)} capacity-limited -> {_relative(ENVIRONMENT_INVENTORY)}"
    )
    return 0


# --------------------------------------------------- sections 8-9, 18, 20, 22, 27-30: declared
#
# Everything a result could otherwise be shaped to fit is written here, before any endpoint is
# read: the action set, the policy family, the routing strategies, the budgets, the reviewer
# scenarios, the regimes, the practical questions and the criteria that answer them.

ACTIONS = ("auto_apply", "human_review", "preserve")

QUESTIONS = {
    "Q1": (
        "does low or moderate human review materially improve the achievable safety-coverage "
        "frontier?"
    ),
    "Q2": "does boundary-targeted review beat matched random review?",
    "Q3": (
        "can automatic harm at or below the primary epsilon be achieved with non-degenerate "
        "automatic repair recall at a review rate at or below the practical budget, on a majority "
        "of development environments?"
    ),
    "Q4": "does additional target-document supervision reduce review burden?",
    "Q5": (
        "is any practical operating regime close to the strict certified baseline's safety "
        "without its near-total refusal?"
    ),
}


def _policy_family() -> dict[str, Any]:
    return {
        **_envelope("deployment_policy_family"),
        "actions": list(ACTIONS),
        "action_rule": (
            "candidates are ranked by the frozen adapted score, higher is safer. A candidate whose "
            "score clears the automatic boundary and was not routed to review is applied "
            "automatically; a routed candidate is decided by the simulated reviewer; everything "
            "else is preserved. The three actions partition the evaluation block exactly."
        ),
        "boundary_constraint": (
            "k_auto <= k_review, enforced because the review band is defined relative to k_auto"
        ),
        "policies": {
            name: {
                "boundary_rule": cut,
                "routing": route,
                "spends_review": reviews,
                "status": "PRIMARY" if primary else "SECONDARY",
                "safety_label": CUT_SAFETY[cut],
            }
            for name, cut, route, reviews, primary in POLICIES
        },
        "boundary_rules": {
            CUT_PRESERVE: "no automatic boundary is placed; nothing is applied automatically",
            CUT_SOURCE: (
                "SGV14's frozen source-side cut, computed on the SOURCE engines' calibration rows "
                "through SGV15's own `frozen_source_tau`. It consumes NO target label and is the "
                "only operating point a deployment with zero target supervision has."
            ),
            CUT_EMPIRICAL: (
                "the deepest declared prefix whose harm rate ON THE REVEALED TARGET DOCUMENTS is "
                "at or below epsilon. A plug-in estimate with no distribution-free guarantee. It "
                "is called empirically risk-controlled and never certified."
            ),
            CUT_CERTIFIED_CANDIDATE: (
                "SGV15's frozen procedure: an exact Clopper-Pearson upper limit on the accepted "
                "set's harm rate, over SGV15's nine declared depths at a union-bound level of "
                "alpha / 9. SGV15b measured this to be anti-conservative under document "
                "clustering; it is carried as a REFERENCE POINT, not as a recommendation."
            ),
            CUT_CERTIFIED_DOCUMENT: (
                "SGV15b's frozen document-aware procedure: the union of three finite-sample "
                "bounds on the ratio-null reduction, documents as the inference unit, over the "
                "same nine depths at alpha / 9. This is the valid distribution-free certificate."
            ),
        },
        "routings": {
            ROUTE_NONE: "no review is spent",
            ROUTE_BELOW: (
                "pool: candidates below the automatic boundary. Order: descending score, so the "
                "band immediately below the boundary is reviewed first. Review can recover "
                "repairs the boundary refused; it can never intercept an automatic edit."
            ),
            ROUTE_BOUNDARY: (
                "pool: every eligible candidate. Order: ascending distance from the automatic "
                "boundary, so the review band straddles it. Review can both intercept an edit "
                "about to be applied and recover a repair about to be preserved."
            ),
            ROUTE_RANDOM: (
                "pool: every eligible candidate. Order: a frozen permutation under this stage's "
                "routing seed. Section 34's control 1 and section 18's R0."
            ),
            ROUTE_HIGH_RISK: (
                "pool: candidates at or above the automatic boundary -- the ones about to be "
                "applied. Order: ascending score, so the frozen model's highest predicted harm "
                "inside the deployment band is reviewed first. Section 18's R2."
            ),
            ROUTE_EXPECTED_VALUE: (
                "pool: every eligible candidate. Order: descending estimated gain from review, "
                "where the gain is a step function of the frozen score estimated on the REVEALED "
                "target labels alone. It is an estimate, so the policy that uses it is SECONDARY."
            ),
            ROUTE_SHUFFLED: (
                "section 34's control 2: the boundary routing run against a permuted score, so "
                "the band is the same size and sits nowhere in particular."
            ),
        },
        "routing_may_use": [
            "the frozen adapted score",
            "the deployed automatic boundary",
            "the declared review budget",
            "labels revealed by the target-document purchase, for the expected-value ordering only",
        ],
        "routing_may_not_use": [
            "any evaluation outcome label",
            "realised harm on the evaluation block",
            "the reviewer's own decision",
            "any oracle quantity",
        ],
        "ground_truth_enters": (
            "only to SIMULATE what a reviewer decides about a candidate already routed to review, "
            "and to measure the endpoint. It never selects the boundary and never routes."
        ),
    }


def _review_budgets() -> dict[str, Any]:
    return {
        **_envelope("review_budget_family"),
        "budgets": list(REVIEW_BUDGETS),
        "practical_budget": PRACTICAL_REVIEW_BUDGET,
        "definition": (
            "the maximum fraction of ELIGIBLE candidates routed to human review, where eligible "
            "means every candidate in the evaluation block -- the whole stream the deployed "
            "system sees. The realised review rate can fall below the budget when the routing's "
            "pool is smaller than the budget allows; it is never above it."
        ),
        "cost_units": [
            "reviewed candidates",
            "reviewed documents",
            "candidate labels revealed by target-document purchase",
            "adaptation labels",
        ],
        "no_monetary_units": (
            "this project has no empirical annotation-cost measurement, so this stage reports "
            "supervision cost in labels and reviews and never in money, labour or time."
        ),
        "declared_before": "any endpoint evaluation",
    }


def _reviewer_scenarios() -> dict[str, Any]:
    return {
        **_envelope("reviewer_accuracy_scenarios"),
        "status": (
            "SIMULATION SCENARIOS. No reviewer accuracy in this project was measured on human "
            "beings. H1 is an IDEALIZED UPPER BOUND and is labelled one everywhere it appears."
        ),
        "scenarios": {
            H1_IDEAL: {
                "accuracy": REVIEWER_ACCURACY[H1_IDEAL],
                "role": "IDEALIZED UPPER BOUND",
                "selectable": True,
            },
            H2_HIGH: {
                "accuracy": REVIEWER_ACCURACY[H2_HIGH],
                "role": "high-quality reviewer sensitivity scenario",
                "selectable": True,
            },
            H3_MODERATE: {
                "accuracy": REVIEWER_ACCURACY[H3_MODERATE],
                "role": "moderate reviewer sensitivity scenario",
                "selectable": True,
            },
            H4_COIN_FLIP: {
                "accuracy": REVIEWER_ACCURACY[H4_COIN_FLIP],
                "role": "NEGATIVE CONTROL -- section 34 control 4. Never selectable.",
                "selectable": False,
            },
            H0_NO_REVIEW: {
                "accuracy": None,
                "role": "no review is spent, so reviewer accuracy is not applicable",
                "selectable": True,
            },
        },
        "primary_scenario": PRIMARY_SCENARIO,
        "must_survive": NOT_PERFECT_REVIEWER_SCENARIO,
        "reviewer_model": (
            "a reviewer shown one candidate decides to apply it or to preserve it. A CORRECT "
            "decision applies the candidate if and only if it is beneficial. An INCORRECT "
            "decision is the opposite of the correct one, so an incorrect reviewer can both drop "
            "a repair and admit a harmful edit. Correctness is drawn once per evaluation row "
            "under a frozen seed and thresholded by the accuracy, so the scenarios are NESTED "
            "under common random numbers: every candidate the 0.95 reviewer gets right, the 0.98 "
            "and the 1.00 reviewer also get right. The sensitivity is therefore paired."
        ),
        "seed": REVIEWER_SEED,
    }


def _document_budgets() -> dict[str, Any]:
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    capacity = {
        record["environment"]: record["certification_documents"]
        for record in inventory["environments"]
    }
    return {
        **_envelope("target_document_budgets"),
        "budgets": list(DOC_BUDGETS),
        "control_budget": CONTROL_DOC_BUDGET,
        "practical_budget": PRACTICAL_DOC_BUDGET,
        "draws_per_budget": DOC_DRAWS,
        "draws_at_zero": 1,
        "seed": SAMPLE_SEED,
        "unit": (
            "documents. Selecting a target document reveals every eligible candidate inside it, "
            "which is what keeps the purchased units independent and what makes the candidate "
            "label count -- not the document count -- the number an annotation campaign pays."
        ),
        "zero_budget_meaning": (
            "no target label is bought at all. The only operating point available is SGV14's "
            "frozen source cut, so the zero-budget column is the frozen source baseline and not "
            "an empty cell."
        ),
        "supervision_accounting": {
            "adaptation_labels": int(s15.ADAPT_BUDGET),
            "adaptation_source": "the fitting half, bought by SGV15's frozen acquisition rule",
            "certification_labels": "every eligible candidate inside each purchased document",
            "double_counting": (
                "adaptation labels and certification labels are disjoint by construction -- they "
                "come from disjoint document halves -- and are reported separately and summed. "
                "No revealed label is counted as free."
            ),
        },
        "capacity": capacity,
        "clipping": {
            str(budget): sorted(name for name, size in capacity.items() if size < budget)
            for budget in DOC_BUDGETS
        },
    }


def run_preregister() -> int:
    """The design record. Written before any endpoint; `_write_json_once` refuses a rewrite."""
    started = time.monotonic()
    if not ENVIRONMENT_INVENTORY.is_file():
        raise PhaseError("run --capacity first")
    floor = load_floor()
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    environments = [record["environment"] for record in inventory["environments"]]

    _write_json_once(DEPLOYMENT_POLICY_FAMILY, _policy_family())
    _write_json_once(REVIEW_BUDGET_FAMILY, _review_budgets())
    _write_json_once(REVIEWER_ACCURACY_SCENARIOS, _reviewer_scenarios())
    _write_json_once(TARGET_DOCUMENT_BUDGETS, _document_budgets())

    payload = {
        **_envelope("design_record"),
        "scientific_status": {
            "kind": "DEVELOPMENT",
            "why": (
                "SGV14 already exposed all ten environments and this stage reads their outcomes. "
                "Nothing here is confirmatory however well it performs."
            ),
            "prohibited_language": [
                "externally validated",
                "deployment proven",
                "safely deployable in general",
                "production ready",
                "confirmed",
            ],
            "allowed_framing": "development evidence suggests",
            "confirmatory_stage": (
                "a later external confirmation on environments nothing has touched is still "
                "required if a practical regime emerges here"
            ),
        },
        "environments": environments,
        "axes": {
            "A_harm_tolerance": {
                "values": [float(value) for value in EPSILONS],
                "primary": float(PRIMARY_EPSILON),
                "rule": (
                    "the primary target does not move. The wider tolerance exists to price "
                    "safety, not to be promoted because it performs better."
                ),
            },
            "B_review_budget": {"values": list(REVIEW_BUDGETS), "primary": PRACTICAL_REVIEW_BUDGET},
            "C_target_document_budget": {
                "values": list(DOC_BUDGETS),
                "primary": PRACTICAL_DOC_BUDGET,
                "draws": DOC_DRAWS,
            },
            "D_deployment_policy": {
                "primary": list(PRIMARY_POLICIES),
                "secondary": list(SECONDARY_POLICIES),
            },
        },
        "questions": QUESTIONS,
        "two_questions_kept_apart": {
            "A_operational_frontier": (
                "with the ranking already available, which safety / coverage / review "
                "combinations are reachable at all? Answered by the review and harm-tolerance "
                "frontiers at a fixed document budget."
            ),
            "B_supervision_cost": (
                "how does the limited target-document budget move that frontier? Answered by the "
                "supervision frontier, which holds the policy fixed and varies only the budget."
            ),
        },
        "metrics": {
            "automatic_harm": (
                "harmful auto-applied edits / all auto-applied edits. Reviewed edits are NOT in "
                "this denominator."
            ),
            "end_to_end_residual_error": (
                "(harmful edits applied by any route + beneficial edits not applied by any route) "
                "/ evaluation rows. A DIFFERENT quantity from automatic harm, never substituted "
                "for it."
            ),
            "automatic_repair_recall": (
                "beneficial edits applied automatically / all beneficial edits"
            ),
            "total_repair_recall": (
                "beneficial edits applied automatically or through review / all beneficial edits"
            ),
            "automatic_coverage": "auto-applied edits / evaluation rows",
            "review_rate": "reviewed candidates / eligible candidates",
            "preserve_rate": "preserved candidates / evaluation rows",
            "false_automatic_corrections": "count of harmful auto-applied edits",
            "review_yield": (
                "reviewed candidates whose review outcome is strictly better than the no-review "
                "counterfactual, over reviewed candidates. A harmful candidate blocked before "
                "automatic application counts; a beneficial candidate applied that would have "
                "been preserved counts; agreeing with the counterfactual does not."
            ),
            "net_correct_repairs": (
                "beneficial edits applied by any route minus harmful edits applied by any route"
            ),
            "harm_intercept_rate": (
                "harmful candidates ROUTED AWAY from automatic application / harmful candidates "
                "that would otherwise have been auto-applied. This measures routing. The realised "
                "interception, after the reviewer's own errors, is reported separately."
            ),
            "no_composite_primary": (
                "no weighted combination of these is an endpoint. The frontier is the output."
            ),
        },
        "safety_labels": {
            SAFETY_CERTIFIED: (
                "carries a valid finite-sample distribution-free certificate for the operational "
                "estimand"
            ),
            SAFETY_EMPIRICAL: (
                "satisfies the harm target on revealed development labels; NO distribution-free "
                "guarantee"
            ),
            SAFETY_NONE: "no target-label risk control at all",
            "rule": (
                "an empirically risk-controlled policy is never called certified, whatever it "
                "achieves"
            ),
        },
        "label_reuse": {
            L0_NO_REUSE: "PRIMARY. Reviewed labels are never fed back into the model.",
            L1_REUSE_AFTER_REVIEW: (
                "SECONDARY, sequential only. A reviewed label may inform a LATER batch and may "
                "never improve the decision that routed that same candidate to review."
            ),
        },
        "regimes": {name: {"max_review_rate": bound} for name, bound in REGIMES},
        "practical_regime_criteria": {
            "1_safety": {
                "requirement": f"automatic harm at or below {PRIMARY_EPSILON} on every draw",
                "breadth_required": BREADTH_REQUIRED,
                "of": ENVIRONMENT_COUNT,
            },
            "2_non_degenerate_automation": {
                "requirement": f"automatic coverage at or above the inherited floor {floor}",
                "breadth_required": BREADTH_REQUIRED,
                "of": ENVIRONMENT_COUNT,
                "convention": (
                    "the project's existing non-degeneracy floor, read from SGV14's frozen record"
                ),
            },
            "3_review_budget": {
                "requirement": f"review rate at or below {PRACTICAL_REVIEW_BUDGET}"
            },
            "4_realistic_supervision": {
                "requirement": f"target-document budget at or below {PRACTICAL_DOC_BUDGET}"
            },
            "5_not_dependent_on_perfect_review": {
                "requirement": (
                    f"criteria 1-3 also hold under the {NOT_PERFECT_REVIEWER_SCENARIO} scenario, "
                    f"accuracy {REVIEWER_ACCURACY[NOT_PERFECT_REVIEWER_SCENARIO]}"
                )
            },
            "6_survives_nested_selection": {
                "requirement": (
                    "the leave-one-environment-out selection picks it on a majority of folds"
                ),
                "majority_required": MAJORITY_REQUIRED,
            },
            "7_fixed_complete_policy": {
                "requirement": (
                    "one frozen tuple of (policy, routing, review budget, document budget, "
                    "reviewer scenario)"
                )
            },
        },
        "answering_rules": {
            "Q1": (
                "paired across environments at the primary epsilon and the practical document "
                "budget: total repair recall under a review budget at or below the practical one, "
                "against the same policy family's best review-free arm, with automatic harm held "
                "at or below epsilon. Material means the paired effect is positive and survives "
                "Holm adjustment across the declared family."
            ),
            "Q2": (
                "R1 boundary against R0 random at MATCHED review budget and the same automatic "
                "boundary, on harm interception and on risk-controlled automatic recall."
            ),
            "Q3": (
                f"count environments where automatic harm holds {PRIMARY_EPSILON} on every draw, "
                f"automatic coverage clears the floor, and the review rate is at or below "
                f"{PRACTICAL_REVIEW_BUDGET}, at the practical document budget. A majority is "
                f"{MAJORITY_REQUIRED} of {ENVIRONMENT_COUNT}."
            ),
            "Q4": (
                "the smallest review budget that reaches a fixed operating point, as a function "
                "of the target-document budget, at the primary epsilon."
            ),
            "Q5": (
                "the frozen certified baselines' safety and automation against the best practical "
                "regime's, on the same environments and the same ranking."
            ),
        },
        "outcome_taxonomy": {
            "A": (
                "a policy with harm <= epsilon, meaningful automatic recall, review <= 20% and "
                "majority breadth exists"
            ),
            "B": "review works but only at high review rates; the break-even burden is quantified",
            "C": "deployment becomes useful only at the wider tolerance, not at the primary one",
            "D": (
                "human review does not rescue deployment; the bottleneck lies upstream of this "
                "stage"
            ),
        },
        "stopping_rule": (
            "if the outcome is D the next investigation moves UPSTREAM -- candidate quality, the "
            "correction generator, correction-method diversity. It does not invent another "
            "threshold or certification stage."
        ),
        "seeds": {
            "document_sample": SAMPLE_SEED,
            "reviewer_correctness": REVIEWER_SEED,
            "random_routing": ROUTING_SEED,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "coverage_floor": floor,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "alpha": ALPHA,
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DESIGN_RECORD, payload)
    print(
        f"preregister: {len(POLICIES)} policies, {len(REVIEW_BUDGETS)} review budgets, "
        f"{len(DOC_BUDGETS)} document budgets, {len(REVIEWER_SCENARIOS)} reviewer scenarios, "
        f"{len(QUESTIONS)} questions -> {_relative(DESIGN_RECORD)}"
    )
    return 0


# ------------------------------------------------- section 43: fit once, then do arithmetic
#
# The whole stage is downstream arithmetic on one frozen score table per environment. The fit
# happens here, once, and every later phase reads the cache. Section 42's discipline is that a
# cache is only legitimate when identity is PROVEN, so this phase also compares what it just
# computed against SGV15b's stored geometry field by field and refuses on any difference.


@dataclass(frozen=True, slots=True)
class Deployment:
    """One environment's frozen ranking, its evaluation block, and both purchasable sides."""

    name: str
    corpus: str
    base_engine: str
    cert_documents: np.ndarray
    cert_doc_of: np.ndarray
    cert_scores: np.ndarray
    cert_harm: np.ndarray
    eval_scores: np.ndarray
    eval_harm: np.ndarray
    eval_beneficial: np.ndarray
    eval_doc_of: np.ndarray
    eval_documents: np.ndarray
    fine_thresholds: np.ndarray
    nine_thresholds: np.ndarray
    source_tau: dict[float, float]
    accepted: np.ndarray
    harmful: np.ndarray
    nine_accepted: np.ndarray
    nine_harmful: np.ndarray

    @property
    def n_documents(self) -> int:
        return int(self.cert_documents.size)

    @property
    def n_rows(self) -> int:
        return int(self.eval_scores.size)

    def doc_sizes(self) -> np.ndarray:
        return np.bincount(self.cert_doc_of, minlength=self.n_documents).astype(np.int64)

    def doc_harm(self) -> np.ndarray:
        return np.bincount(
            self.cert_doc_of, weights=self.cert_harm.astype(float), minlength=self.n_documents
        ).astype(np.int64)


def _per_document_prefix_counts(
    scores: np.ndarray, harm: np.ndarray, doc_of: np.ndarray, taus: np.ndarray, n_documents: int
) -> tuple[np.ndarray, np.ndarray]:
    """Accepted and harmful-accepted counts per document at every threshold, in one pass each."""
    accepted = np.zeros((n_documents, taus.size), dtype=np.int64)
    harmful = np.zeros_like(accepted)
    for index in range(n_documents):
        mask = doc_of == index
        ascending = np.sort(scores[mask])
        accepted[index] = ascending.size - np.searchsorted(ascending, taus, side="left")
        ascending_harm = np.sort(scores[mask & harm])
        harmful[index] = ascending_harm.size - np.searchsorted(ascending_harm, taus, side="left")
    return accepted, harmful


def _geometry_key(name: str, field: str) -> str:
    return f"{name.replace('/', '__')}|{field}"


def run_geometry() -> int:
    """Fit the frozen adaptation once per environment and cache the deployment geometry."""
    started = time.monotonic()
    if not DESIGN_RECORD.is_file():
        raise PhaseError("run --preregister first")
    if GEOMETRY.exists():
        raise PhaseError(f"{_relative(GEOMETRY)} already exists; delete it deliberately")
    built, _ = s15._environments()
    stored = np.load(s15b.GEOMETRY, allow_pickle=False)

    payload: dict[str, np.ndarray] = {}
    inventory: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    for environment in built:
        began = time.monotonic()
        seed = _stable_seed(
            "sgv15-adapt", str(s15.SAMPLE_SEED), environment.name, s15.A_UNCERTAINTY
        )
        adapted = fit_adaptation(environment, s15.A_UNCERTAINTY, s15.ADAPT_BUDGET, seed)
        block = environment.setup.evaluation
        name = environment.name

        documents = np.asarray([str(d) for d in environment.cert_documents.tolist()], dtype=object)
        names = np.asarray(sorted(set(documents.tolist())), dtype=object)
        lookup = {value: index for index, value in enumerate(names.tolist())}
        doc_of = np.asarray([lookup[str(d)] for d in documents.tolist()], dtype=np.int64)
        cert_scores = np.asarray(adapted.cert_scores, dtype=float)
        cert_harm = np.asarray(environment.cert_harmful, dtype=bool)
        fine = prefix_thresholds(cert_scores, FINE_DEPTHS)
        nine = prefix_thresholds(cert_scores, NINE_DEPTHS)

        # Every field SGV15b stored is recomputed here from the same frozen inputs and compared
        # exactly. A cached score table is only usable if it is provably the same table.
        for field, computed in (
            ("documents", names.astype(str)),
            ("doc_of", doc_of),
            ("scores", cert_scores),
            ("harm", cert_harm),
            ("thresholds_fine", fine),
            ("thresholds_nine", nine),
            ("eval_scores", np.asarray(adapted.eval_scores, dtype=float)),
            ("eval_harm", np.asarray(block.harmful, dtype=bool)),
            ("eval_beneficial", np.asarray(block.beneficial, dtype=bool)),
            ("eval_documents", np.asarray(block.documents, dtype=str)),
        ):
            key = _geometry_key(name, field)
            if key not in stored:
                differences.append(
                    {"environment": name, "field": field, "problem": "absent upstream"}
                )
                continue
            reference = stored[key]
            if reference.shape != computed.shape:
                differences.append({"environment": name, "field": field, "problem": "shape"})
            elif computed.dtype.kind in "fiub":
                gap = (
                    float(np.abs(computed.astype(float) - reference.astype(float)).max())
                    if computed.size
                    else 0.0
                )
                if gap != 0.0:
                    differences.append(
                        {"environment": name, "field": field, "max_absolute_difference": gap}
                    )
            elif not bool((computed.astype(str) == reference.astype(str)).all()):
                differences.append({"environment": name, "field": field, "problem": "values"})

        eval_names = sorted({str(d) for d in block.documents.tolist()})
        eval_lookup = {value: index for index, value in enumerate(eval_names)}
        payload[_geometry_key(name, "cert_documents")] = names.astype(str)
        payload[_geometry_key(name, "cert_doc_of")] = doc_of
        payload[_geometry_key(name, "cert_scores")] = cert_scores
        payload[_geometry_key(name, "cert_harm")] = cert_harm
        payload[_geometry_key(name, "eval_scores")] = np.asarray(adapted.eval_scores, dtype=float)
        payload[_geometry_key(name, "eval_harm")] = np.asarray(block.harmful, dtype=bool)
        payload[_geometry_key(name, "eval_beneficial")] = np.asarray(block.beneficial, dtype=bool)
        payload[_geometry_key(name, "eval_documents")] = np.asarray(eval_names, dtype=str)
        payload[_geometry_key(name, "eval_doc_of")] = np.asarray(
            [eval_lookup[str(d)] for d in block.documents.tolist()], dtype=np.int64
        )
        payload[_geometry_key(name, "fine_thresholds")] = fine
        payload[_geometry_key(name, "nine_thresholds")] = nine
        payload[_geometry_key(name, "source_tau")] = np.asarray(
            [frozen_source_tau(environment, adapted.model, epsilon) for epsilon in EPSILONS],
            dtype=float,
        )
        accepted, harmful = _per_document_prefix_counts(
            cert_scores, cert_harm, doc_of, fine, names.size
        )
        nine_accepted, nine_harmful = _per_document_prefix_counts(
            cert_scores, cert_harm, doc_of, nine, names.size
        )
        payload[_geometry_key(name, "accepted")] = accepted
        payload[_geometry_key(name, "harmful")] = harmful
        payload[_geometry_key(name, "nine_accepted")] = nine_accepted
        payload[_geometry_key(name, "nine_harmful")] = nine_harmful
        payload[_geometry_key(name, "meta")] = np.asarray(
            [environment.corpus, environment.base_engine], dtype=str
        )
        inventory.append(
            {
                "environment": name,
                "corpus": environment.corpus,
                "base_engine": environment.base_engine,
                "certification_documents": int(names.size),
                "certification_candidates": int(cert_scores.size),
                "evaluation_rows": int(block.size),
                "evaluation_documents": len(eval_names),
                "evaluation_harmful": int(block.harmful.sum()),
                "evaluation_beneficial": int(block.beneficial.sum()),
                "adaptation_labels": int(adapted.n_adapt),
                "adaptation_documents": int(adapted.documents),
                "score_sha256": canonical_hash(
                    {"eval": adapted.eval_scores.tolist(), "cert": cert_scores.tolist()}
                ),
                "fit_seconds": time.monotonic() - began,
            }
        )
        print(
            f"  {name}: {int(names.size)} certification documents, {int(cert_scores.size)} "
            f"certification candidates, {int(block.size)} evaluation rows "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )

    if differences:
        raise PhaseError(
            "the refitted score table does not reproduce SGV15b's stored geometry: "
            f"{differences[:4]}. Identity was not proven, so nothing may be cached or reused."
        )

    np.savez_compressed(GEOMETRY, **payload)
    _write_json_once(
        SCORE_INVENTORY,
        {
            **_envelope("score_inventory"),
            "reuse_check": (
                "SGV15b's `certification_geometry.npz` already holds the frozen adapted score "
                "table for these ten environments. Section 43 asks whether the required scores "
                "already exist; section 42 forbids caching without proven identity. This phase "
                "therefore refits the frozen adaptation once per environment, compares every "
                "stored field exactly, and refuses on any difference. It is the only refit in "
                "the stage."
            ),
            "upstream_geometry": _relative(s15b.GEOMETRY),
            "upstream_geometry_sha256": file_sha256(s15b.GEOMETRY),
            "fields_compared_per_environment": 10,
            "environments_compared": len(inventory),
            "differences": differences,
            "identical_to_upstream": not differences,
            "environments": inventory,
            "fits_performed": len(inventory),
            "fits_avoided": (
                "one fit per environment serves every epsilon, every review budget, every "
                "reviewer scenario, every document budget, every draw and every policy: "
                f"{len(EPSILONS)} x {len(REVIEW_BUDGETS)} x {len(REVIEWER_SCENARIOS)} x "
                f"{len(DOC_BUDGETS)} x {DOC_DRAWS} x {len(POLICIES)} cells from ten fits."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"geometry: {len(inventory)} environments fitted once, all fields identical to "
        f"{_relative(s15b.GEOMETRY)} -> {_relative(GEOMETRY)}"
    )
    return 0


def load_deployments() -> dict[str, Deployment]:
    if not GEOMETRY.is_file():
        raise PhaseError("run --geometry first")
    stored = np.load(GEOMETRY, allow_pickle=False)
    out: dict[str, Deployment] = {}
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        if _geometry_key(name, "eval_scores") not in stored:
            continue
        meta = stored[_geometry_key(name, "meta")]
        out[name] = Deployment(
            name=name,
            corpus=str(meta[0]),
            base_engine=str(meta[1]),
            cert_documents=stored[_geometry_key(name, "cert_documents")],
            cert_doc_of=stored[_geometry_key(name, "cert_doc_of")],
            cert_scores=stored[_geometry_key(name, "cert_scores")],
            cert_harm=stored[_geometry_key(name, "cert_harm")],
            eval_scores=stored[_geometry_key(name, "eval_scores")],
            eval_harm=stored[_geometry_key(name, "eval_harm")],
            eval_beneficial=stored[_geometry_key(name, "eval_beneficial")],
            eval_doc_of=stored[_geometry_key(name, "eval_doc_of")],
            eval_documents=stored[_geometry_key(name, "eval_documents")],
            fine_thresholds=stored[_geometry_key(name, "fine_thresholds")],
            nine_thresholds=stored[_geometry_key(name, "nine_thresholds")],
            source_tau=dict(
                zip(
                    [float(e) for e in EPSILONS],
                    [float(v) for v in stored[_geometry_key(name, "source_tau")].tolist()],
                    strict=True,
                )
            ),
            accepted=stored[_geometry_key(name, "accepted")],
            harmful=stored[_geometry_key(name, "harmful")],
            nine_accepted=stored[_geometry_key(name, "nine_accepted")],
            nine_harmful=stored[_geometry_key(name, "nine_harmful")],
        )
    return out


# ------------------------------------------------ sections 4-7, 10, 12: the deployment policy
#
# Three actions, two boundaries, one ranking. Everything below is computed from the frozen score,
# the declared boundary and the declared budget; an outcome label enters only to simulate what a
# reviewer decides about a candidate that has already been routed.

REFUSE = float("inf")


def document_sample(deployment: Deployment, size: int, seed: int) -> np.ndarray:
    """Which certification documents this draw buys, clipped to what the environment has."""
    available = deployment.n_documents
    take = min(int(size), available)
    if take <= 0:
        return np.zeros(0, dtype=int)
    return np.sort(np.random.default_rng(seed).permutation(available)[:take])


def _sample_seed(environment: str, budget: int, draw: int) -> int:
    return _stable_seed("sgv-dt1-docs", str(SAMPLE_SEED), environment, str(budget), str(draw))


def select_boundary(
    deployment: Deployment, rule: str, rows: np.ndarray, epsilon: float
) -> dict[str, Any]:
    """Where the automatic boundary goes, and what justifies it.

    Every rule that consumes a target label consumes only the labels inside `rows`, which are the
    documents this draw actually bought. No rule sees an evaluation outcome.
    """
    if rule == CUT_PRESERVE:
        return {
            "tau": REFUSE,
            "feasible": False,
            "depth": 0.0,
            "index": -1,
            "accepted_in_sample": 0,
            "harmful_in_sample": 0,
            "risk_upper_bound": float("nan"),
        }
    if rule == CUT_SOURCE:
        tau = float(deployment.source_tau[float(epsilon)])
        return {
            "tau": tau,
            "feasible": bool(np.isfinite(tau)),
            "depth": float("nan"),
            "index": -1,
            "accepted_in_sample": 0,
            "harmful_in_sample": 0,
            "risk_upper_bound": float("nan"),
        }
    if rows.size == 0:
        return {
            "tau": REFUSE,
            "feasible": False,
            "depth": 0.0,
            "index": -1,
            "accepted_in_sample": 0,
            "harmful_in_sample": 0,
            "risk_upper_bound": 1.0,
        }

    if rule == CUT_EMPIRICAL:
        accepted = deployment.accepted[rows].sum(axis=0)
        harmful = deployment.harmful[rows].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(accepted > 0, harmful / np.maximum(accepted, 1), np.inf)
        feasible = np.flatnonzero((accepted > 0) & (rate <= epsilon))
        if feasible.size == 0:
            return {
                "tau": REFUSE,
                "feasible": False,
                "depth": 0.0,
                "index": -1,
                "accepted_in_sample": 0,
                "harmful_in_sample": 0,
                "risk_upper_bound": 1.0,
            }
        index = int(feasible.max())
        return {
            "tau": float(deployment.fine_thresholds[index]),
            "feasible": True,
            "depth": float(FINE_DEPTHS[index]),
            "index": index,
            "accepted_in_sample": int(accepted[index]),
            "harmful_in_sample": int(harmful[index]),
            "risk_upper_bound": float(rate[index]),
        }

    # Both certified rules run over SGV15's nine declared depths at a union-bound level of
    # alpha / 9, which is exactly what SGV15 and SGV15b froze. Neither is re-derived here.
    accepted = deployment.nine_accepted[rows]
    harmful = deployment.nine_harmful[rows]
    span = deployment.nine_accepted.max(axis=0).astype(float)
    level = ALPHA / len(NINE_DEPTHS)
    best = -1
    limit = 1.0
    for index in range(len(NINE_DEPTHS)):
        total = int(accepted[:, index].sum())
        count = int(harmful[:, index].sum())
        if rule == CUT_CERTIFIED_CANDIDATE:
            certifies = bool(total > 0 and clopper_pearson_upper(count, total, level) <= epsilon)
            upper = float(clopper_pearson_upper(count, total, level)) if total > 0 else 1.0
        else:
            bound = cluster_ratio_bound(
                harmful[:, index].astype(float),
                accepted[:, index].astype(float),
                epsilon,
                delta=level,
                span=float(span[index]),
                method="union_finite",
            )
            certifies, upper = bool(bound.certifies), float(bound.ratio_upper)
        if certifies:
            best, limit = index, upper
    if best < 0:
        return {
            "tau": REFUSE,
            "feasible": False,
            "depth": 0.0,
            "index": -1,
            "accepted_in_sample": 0,
            "harmful_in_sample": 0,
            "risk_upper_bound": 1.0,
        }
    return {
        "tau": float(deployment.nine_thresholds[best]),
        "feasible": True,
        "depth": float(NINE_DEPTHS[best]),
        "index": best,
        "accepted_in_sample": int(accepted[:, best].sum()),
        "harmful_in_sample": int(harmful[:, best].sum()),
        "risk_upper_bound": limit,
    }


def reviewer_correctness(deployment: Deployment) -> np.ndarray:
    """One uniform per evaluation row, frozen. A reviewer of accuracy `a` is right iff u <= a.

    Common random numbers, so the accuracy scenarios are NESTED: every candidate the moderate
    reviewer gets right, the high-quality and the ideal reviewer also get right. The sensitivity
    is then a paired comparison rather than three unrelated experiments.
    """
    seed = _stable_seed("sgv-dt1-reviewer", str(REVIEWER_SEED), deployment.name)
    return np.random.default_rng(seed).random(deployment.n_rows)


def review_order(
    deployment: Deployment,
    tau: float,
    routing: str,
    scores: np.ndarray | None = None,
    gain: np.ndarray | None = None,
) -> np.ndarray:
    """The review priority order: which eligible candidates a budget buys, and in what order.

    Deterministic in every branch. `scores` overrides the ranking, which is the only way section
    34's shuffled-routing control can reach a routing decision.
    """
    values = deployment.eval_scores if scores is None else scores
    size = values.size
    if routing == ROUTE_NONE:
        return np.zeros(0, dtype=int)
    if routing == ROUTE_RANDOM:
        seed = _stable_seed("sgv-dt1-routing", str(ROUTING_SEED), deployment.name)
        return np.random.default_rng(seed).permutation(size)
    if routing == ROUTE_BELOW:
        pool = np.flatnonzero(values < tau)
        return pool[np.argsort(-values[pool], kind="stable")]
    if routing == ROUTE_HIGH_RISK:
        pool = np.flatnonzero(values >= tau) if np.isfinite(tau) else np.zeros(0, dtype=int)
        return pool[np.argsort(values[pool], kind="stable")]
    if routing == ROUTE_EXPECTED_VALUE:
        if gain is None:
            raise PhaseError("the expected-value routing needs an estimated gain vector")
        return np.lexsort((np.abs(values - tau) if np.isfinite(tau) else -values, -gain))
    if routing in (ROUTE_BOUNDARY, ROUTE_SHUFFLED):
        if not np.isfinite(tau):
            # No boundary was placed, so "nearest the boundary" is undefined. The safest
            # candidates are the ones an automatic policy would have reached first.
            return np.argsort(-values, kind="stable")
        return np.argsort(np.abs(values - tau), kind="stable")
    raise PhaseError(f"unknown routing {routing}")


def expected_gain(
    deployment: Deployment, rows: np.ndarray, tau: float, bins: int = 10
) -> np.ndarray:
    """Estimated gain from reviewing each evaluation candidate, from REVEALED labels only.

    The bin edges are quantiles of the UNLABELLED certification scores, so the binning itself
    carries no outcome information. Inside a bin the revealed candidates give an outcome rate,
    and the gain from review is the chance of blocking a harmful automatic edit above the
    boundary or of recovering a repair below it. An empty bin contributes no gain.
    """
    revealed = np.isin(deployment.cert_doc_of, rows)
    edges = np.quantile(deployment.cert_scores, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    cert_bin = np.searchsorted(edges, deployment.cert_scores, side="right")
    eval_bin = np.searchsorted(edges, deployment.eval_scores, side="right")
    harm_rate = np.zeros(bins, dtype=float)
    benefit_rate = np.zeros(bins, dtype=float)
    for index in range(bins):
        mask = revealed & (cert_bin == index)
        total = int(mask.sum())
        if total == 0:
            continue
        harm_rate[index] = float(deployment.cert_harm[mask].mean())
        benefit_rate[index] = float(total - int(deployment.cert_harm[mask].sum())) / total
    above = deployment.eval_scores >= tau if np.isfinite(tau) else np.zeros(deployment.n_rows, bool)
    return np.where(above, harm_rate[eval_bin], benefit_rate[eval_bin])


# The columns a review budget slices. Everything a cell reports is a cumulative count along the
# review order, so one pass over an ordering answers every declared budget at once instead of
# re-deriving the action assignment thirty thousand times.
def _measure(
    deployment: Deployment,
    tau: float,
    order: np.ndarray,
    budget_counts: Sequence[int],
    accuracies: Sequence[float],
    correctness: np.ndarray,
    floor: float,
) -> dict[tuple[int, float], dict[str, Any]]:
    """Every (review budget, reviewer accuracy) outcome for one boundary and one ordering.

    Deliberately epsilon-free. Only three reported fields depend on the tolerance, and keeping
    them out means one pass over an ordering serves all three declared tolerances instead of
    three, which is what makes the cache key (routing, boundary) rather than (routing, boundary,
    epsilon).
    """
    harm = deployment.eval_harm
    benefit = deployment.eval_beneficial
    rows = deployment.n_rows
    auto0 = (deployment.eval_scores >= tau) if np.isfinite(tau) else np.zeros(rows, dtype=bool)
    total_beneficial = int(benefit.sum())
    auto0_n = int(auto0.sum())
    auto0_h = int((auto0 & harm).sum())
    auto0_b = int((auto0 & benefit).sum())

    ordered_auto = auto0[order]
    ordered_harm = harm[order]
    ordered_benefit = benefit[order]
    first_document = np.zeros(order.size, dtype=bool)
    if order.size:
        seen = deployment.eval_doc_of[order]
        _, first = np.unique(seen, return_index=True)
        first_document[first] = True

    def prefix(values: np.ndarray) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(values.astype(np.int64))))

    c_auto = prefix(ordered_auto)
    c_auto_harm = prefix(ordered_auto & ordered_harm)
    c_auto_benefit = prefix(ordered_auto & ordered_benefit)
    c_documents = prefix(first_document)

    out: dict[tuple[int, float], dict[str, Any]] = {}
    for accuracy in accuracies:
        correct = correctness <= accuracy
        applies = (correct & benefit) | (~correct & ~benefit)
        ordered_applies = applies[order]
        c_applied = prefix(ordered_applies)
        c_applied_harm = prefix(ordered_applies & ordered_harm)
        c_applied_benefit = prefix(ordered_applies & ordered_benefit)
        c_applied_auto_harm = prefix(ordered_applies & ordered_auto & ordered_harm)
        c_blocked_auto_harm = prefix(~ordered_applies & ordered_auto & ordered_harm)
        c_recovered = prefix(ordered_applies & ~ordered_auto & ordered_benefit)
        c_lost = prefix(~ordered_applies & ordered_auto & ordered_benefit)
        c_admitted = prefix(ordered_applies & ~ordered_auto & ordered_harm)
        for count in budget_counts:
            # keyed by the REQUESTED budget, clamped to the pool the routing actually offers.
            k = min(int(count), int(order.size))
            reviewed_auto = int(c_auto[k])
            reviewed_auto_harm = int(c_auto_harm[k])
            reviewed_auto_benefit = int(c_auto_benefit[k])
            auto_n = auto0_n - reviewed_auto
            auto_h = auto0_h - reviewed_auto_harm
            auto_b = auto0_b - reviewed_auto_benefit
            applied_b = auto_b + int(c_applied_benefit[k])
            applied_h = auto_h + int(c_applied_harm[k])
            harm_auto = float(auto_h / auto_n) if auto_n else float("nan")
            coverage = float(auto_n / rows)
            automatic_recall = float(auto_b / total_beneficial) if total_beneficial else 0.0
            intercepted = reviewed_auto_harm
            out[(int(count), accuracy)] = {
                "reviewed": k,
                "reviewed_documents": int(c_documents[k]),
                "review_rate": float(k / rows),
                "auto_applied": auto_n,
                "auto_harmful": auto_h,
                "auto_beneficial": auto_b,
                "automatic_harm": harm_auto,
                "automatic_coverage": coverage,
                "non_degenerate": bool(coverage >= floor),
                "automatic_repair_recall": automatic_recall,
                "review_applied": int(c_applied[k]),
                "review_applied_beneficial": int(c_applied_benefit[k]),
                "review_applied_harmful": int(c_applied_harm[k]),
                "total_repair_recall": float(applied_b / total_beneficial)
                if total_beneficial
                else 0.0,
                "preserve_rate": float((rows - auto_n - int(c_applied[k])) / rows),
                "false_automatic_corrections": auto_h,
                "net_correct_repairs": int(applied_b - applied_h),
                "end_to_end_residual_error": float(
                    (applied_h + total_beneficial - applied_b) / rows
                ),
                "harm_routed_away_from_auto": intercepted,
                "harm_intercept_rate": float(intercepted / auto0_h) if auto0_h else float("nan"),
                "harm_intercepted_realised": int(c_blocked_auto_harm[k]),
                "harm_intercept_rate_realised": (
                    float(int(c_blocked_auto_harm[k]) / auto0_h) if auto0_h else float("nan")
                ),
                "harm_readmitted_by_review": int(c_applied_auto_harm[k]),
                "repairs_recovered_by_review": int(c_recovered[k]),
                "repairs_lost_to_review": int(c_lost[k]),
                "harm_introduced_by_review": int(c_admitted[k]),
                "review_value_added": int(c_blocked_auto_harm[k]) + int(c_recovered[k]),
                "review_yield": (
                    float((int(c_blocked_auto_harm[k]) + int(c_recovered[k])) / k)
                    if k
                    else float("nan")
                ),
                "harm_found_per_100_reviews": (
                    float(100.0 * int(c_blocked_auto_harm[k]) / k) if k else float("nan")
                ),
                "benefit_recovered_per_100_reviews": (
                    float(100.0 * int(c_recovered[k]) / k) if k else float("nan")
                ),
                "would_be_auto_applied": auto0_n,
                "would_be_auto_harmful": auto0_h,
            }
    return out


def _finalize(record: dict[str, Any], epsilon: float) -> dict[str, Any]:
    """The three fields that depend on the harm tolerance, added to an epsilon-free measurement.

    An automatic set that is empty holds the bound vacuously -- refusing is safe -- and scores
    zero risk-controlled recall, which is exactly why the non-degeneracy criterion exists as a
    separate gate.
    """
    auto = int(record["auto_applied"])
    harm = float(record["automatic_harm"])
    holds = True if auto == 0 else bool(harm <= epsilon)
    return {
        **record,
        "epsilon": float(epsilon),
        "holds_bound": holds,
        "risk_controlled_automatic_recall": float(
            record["automatic_repair_recall"] if holds and auto else 0.0
        ),
        "risk_controlled_total_recall": float(record["total_repair_recall"] if holds else 0.0),
    }


def _action_masks(
    deployment: Deployment,
    tau: float,
    order: np.ndarray,
    k: int,
    accuracy: float,
    correctness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The automatically applied set and the set applied by EITHER route.

    Both are needed by the clustered bootstrap, because automatic harm is read on the first and
    total repair recall on the second, and section 3 forbids collapsing them into one denominator.
    """
    rows = deployment.n_rows
    auto0 = (deployment.eval_scores >= tau) if np.isfinite(tau) else np.zeros(rows, dtype=bool)
    reviewed = np.zeros(rows, dtype=bool)
    if k:
        reviewed[order[:k]] = True
    correct = correctness <= accuracy
    applies = (correct & deployment.eval_beneficial) | (~correct & ~deployment.eval_beneficial)
    auto = auto0 & ~reviewed
    return auto, auto | (reviewed & applies)


# ------------------------------------------------------- sections 11-12: the main sweep
#
# One environment at a time: fit nothing, place every boundary the policy family asks for, order
# the review pool once per (routing, boundary), and read every declared budget off that ordering.


class Lattice:
    """Per-environment cache of review orderings and their budget slices.

    A policy family of ten arms, five review budgets, four reviewer scenarios, three tolerances,
    six document budgets and twenty draws resolves to a few hundred distinct (routing, boundary)
    pairs per environment. Computing the action assignment per cell instead would repeat the same
    arithmetic tens of thousands of times and return the same answer.
    """

    def __init__(self, deployment: Deployment, floor: float, accuracies: Sequence[float]) -> None:
        self.deployment = deployment
        self.floor = floor
        self.accuracies = tuple(float(value) for value in accuracies)
        self.correctness = reviewer_correctness(deployment)
        self.counts = tuple(int(rate * deployment.n_rows) for rate in REVIEW_BUDGETS)
        self._orders: dict[tuple[str, float], np.ndarray] = {}
        self._measures: dict[tuple[str, float], dict[tuple[int, float], dict[str, Any]]] = {}

    def order(self, routing: str, tau: float) -> np.ndarray:
        key = (routing, float(tau))
        if key not in self._orders:
            self._orders[key] = review_order(self.deployment, tau, routing)
        return self._orders[key]

    def measures(self, routing: str, tau: float) -> dict[tuple[int, float], dict[str, Any]]:
        key = (routing, float(tau))
        if key not in self._measures:
            self._measures[key] = _measure(
                self.deployment,
                tau,
                self.order(routing, tau),
                self.counts,
                self.accuracies,
                self.correctness,
                self.floor,
            )
        return self._measures[key]

    def budget_count(self, rate: float) -> int:
        return int(rate * self.deployment.n_rows)

    def transient(
        self, routing: str, tau: float, order: np.ndarray
    ) -> dict[tuple[int, float], dict[str, Any]]:
        """An ordering that depends on the purchased sample and so cannot be cached by boundary."""
        return _measure(
            self.deployment, tau, order, self.counts, self.accuracies, self.correctness, self.floor
        )


def _scenario_accuracy(scenario: str) -> float:
    return float(REVIEWER_ACCURACY[scenario]) if scenario in REVIEWER_ACCURACY else float("nan")


def _cell_rows(
    lattice: Lattice,
    epsilon: float,
    budget: int,
    draw: int,
    sample: np.ndarray,
    policies: Sequence[str],
    scenarios: Sequence[str],
) -> tuple[list[dict[str, Any]], list[tuple[str, np.ndarray, np.ndarray]]]:
    """Every declared (policy, review budget, reviewer scenario) row for one purchased sample."""
    deployment = lattice.deployment
    boundaries: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    keep: list[tuple[str, np.ndarray, np.ndarray]] = []
    labels = int(deployment.doc_sizes()[sample].sum()) if sample.size else 0
    revealed_harmful = int(deployment.doc_harm()[sample].sum()) if sample.size else 0
    context = {
        "environment": deployment.name,
        "corpus": deployment.corpus,
        "base_engine": deployment.base_engine,
        "epsilon": float(epsilon),
        "requested_documents": int(budget),
        "achieved_documents": int(sample.size),
        "clipped": bool(sample.size < budget),
        "draw": int(draw),
        "certification_labels": labels,
        "certification_harmful_labels": revealed_harmful,
        "adaptation_labels": int(s15.ADAPT_BUDGET),
        "supervision_labels": int(labels + s15.ADAPT_BUDGET),
        "evaluation_rows": deployment.n_rows,
        "evaluation_beneficial": int(deployment.eval_beneficial.sum()),
        "evaluation_harmful": int(deployment.eval_harm.sum()),
        "label_reuse": L0_NO_REUSE,
    }
    for policy in policies:
        rule, routing, reviews = POLICY_RULE[policy]
        if rule not in boundaries:
            boundaries[rule] = select_boundary(deployment, rule, sample, epsilon)
        boundary = boundaries[rule]
        tau = float(boundary["tau"])
        if routing == ROUTE_EXPECTED_VALUE:
            order = review_order(
                deployment, tau, routing, gain=expected_gain(deployment, sample, tau)
            )
            table = lattice.transient(routing, tau, order)
        else:
            table = lattice.measures(routing, tau)
        rates = REVIEW_BUDGETS if reviews else (0.0,)
        for rate in rates:
            count = lattice.budget_count(rate)
            names = (H0_NO_REVIEW,) if (not reviews or rate == 0.0) else tuple(scenarios)
            for scenario in names:
                accuracy = (
                    lattice.accuracies[0]
                    if scenario == H0_NO_REVIEW
                    else _scenario_accuracy(scenario)
                )
                record = table[(count, accuracy)]
                row = {
                    **context,
                    "policy": policy,
                    "boundary_rule": rule,
                    "routing": routing if rate > 0.0 else ROUTE_NONE,
                    "safety_label": CUT_SAFETY[rule],
                    "review_budget": float(rate),
                    "reviewer_scenario": scenario,
                    "reviewer_accuracy": (
                        float("nan") if scenario == H0_NO_REVIEW else _scenario_accuracy(scenario)
                    ),
                    "cut_feasible": bool(boundary["feasible"]),
                    "declared_depth": float(boundary["depth"]),
                    "boundary_index": int(boundary["index"]),
                    "tau": tau,
                    "certified_risk_upper_bound": float(boundary["risk_upper_bound"]),
                    "cut_accepted_in_sample": int(boundary["accepted_in_sample"]),
                    "cut_harmful_in_sample": int(boundary["harmful_in_sample"]),
                    **_finalize(record, epsilon),
                }
                row["cell"] = "|".join(
                    (deployment.name, policy, f"{rate:.2f}", scenario, str(budget), str(draw))
                )
                out.append(row)
                if (
                    epsilon == PRIMARY_EPSILON
                    and budget == PRACTICAL_DOC_BUDGET
                    and scenario in (H0_NO_REVIEW, PRIMARY_SCENARIO)
                    and rate in (0.0, PRACTICAL_REVIEW_BUDGET)
                ):
                    order = (
                        review_order(
                            deployment, tau, routing, gain=expected_gain(deployment, sample, tau)
                        )
                        if routing == ROUTE_EXPECTED_VALUE
                        else lattice.order(routing, tau)
                    )
                    keep.append(
                        (
                            row["cell"],
                            *_action_masks(
                                deployment,
                                tau,
                                order,
                                min(count, int(order.size)),
                                accuracy,
                                lattice.correctness,
                            ),
                        )
                    )
    return out, keep


def run_deploy(only: str | None = None) -> int:
    """The main sweep. Every declared policy x routing x review budget x document budget."""
    started = time.monotonic()
    if not DESIGN_RECORD.is_file():
        raise PhaseError("run --preregister first")
    deployments = load_deployments()
    if only is not None:
        deployments = {only: deployments[only]}
    floor = load_floor()
    accuracies = tuple(
        REVIEWER_ACCURACY[name] for name in (*REVIEWER_SCENARIOS, *CONTROL_SCENARIOS)
    )
    rows: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []

    for name, deployment in deployments.items():
        began = time.monotonic()
        lattice = Lattice(deployment, floor, accuracies)
        for epsilon in EPSILONS:
            # The coin-flip reviewer is a negative control, so it is measured at the primary
            # tolerance only: it exists to show that routing without competent review buys
            # nothing, not to be swept.
            scenarios = (
                (*REVIEWER_SCENARIOS, *CONTROL_SCENARIOS)
                if epsilon == PRIMARY_EPSILON
                else REVIEWER_SCENARIOS
            )
            for budget in DOC_BUDGETS:
                draws = 1 if budget == 0 else DOC_DRAWS
                for draw in range(draws):
                    sample = document_sample(deployment, budget, _sample_seed(name, budget, draw))
                    cells, keep = _cell_rows(
                        lattice, epsilon, budget, draw, sample, POLICY_NAMES, scenarios
                    )
                    rows.extend(cells)
                    for cell, auto, applied in keep:
                        documents = sorted(deployment.eval_documents.tolist())
                        block = s13.Block(
                            name="evaluation",
                            index=np.arange(deployment.n_rows),
                            features=np.zeros((deployment.n_rows, 0)),
                            frozen_score=deployment.eval_scores,
                            documents=deployment.eval_documents[deployment.eval_doc_of],
                            sites=np.zeros(deployment.n_rows, dtype=int),
                            candidates=np.zeros(deployment.n_rows, dtype=int),
                            harmful=deployment.eval_harm,
                            beneficial=deployment.eval_beneficial,
                        )
                        auto_counts = per_document_counts(block, auto, documents)
                        applied_counts = per_document_counts(block, applied, documents)
                        for position, document in enumerate(documents):
                            counts.append(
                                {
                                    "cell": cell,
                                    "document_id": document,
                                    "auto_accepted": float(auto_counts["accepted"][position]),
                                    "auto_harmful": float(
                                        auto_counts["harmful_accepted"][position]
                                    ),
                                    "auto_beneficial": float(
                                        auto_counts["beneficial_accepted"][position]
                                    ),
                                    "applied_beneficial": float(
                                        applied_counts["beneficial_accepted"][position]
                                    ),
                                    "applied_harmful": float(
                                        applied_counts["harmful_accepted"][position]
                                    ),
                                    "beneficial_total": float(
                                        auto_counts["beneficial_total"][position]
                                    ),
                                }
                            )
        print(
            f"  {name}: {len([r for r in rows if r['environment'] == name])} cells, "
            f"{len(lattice._measures)} distinct (routing, boundary) pairs "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    _write_parquet_once(DEPLOYMENT_CELLS, frame)
    _write_parquet_once(DOCUMENT_COUNTS, pd.DataFrame(counts))
    print(
        f"deploy: {len(frame)} cells over {frame['environment'].nunique()} environments, "
        f"{frame['policy'].nunique()} policies, {len(counts)} per-document count rows "
        f"({time.monotonic() - started:.0f}s) -> {_relative(DEPLOYMENT_CELLS)}"
    )
    return 0


def load_cells() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not DEPLOYMENT_CELLS.is_file():
        raise PhaseError("run --deploy first")
    return pd.read_parquet(DEPLOYMENT_CELLS), cc_read_json(DESIGN_RECORD)


# ---------------------------------------------- section 44: the upstream reproduction gate
#
# Two of this stage's baselines ARE SGV15's and SGV15b's frozen certified procedures, and this
# file implements them again -- `select_boundary` places the nine-depth union-bound cut directly
# rather than driving SGV15b's `select_prefix`. That is a second implementation, so it has to be
# checked against the first on the first's own seeds and budgets before anything downstream is
# believed. A difference stops the stage.

REPRODUCED_ARMS = {
    B2_SGV15: (CUT_CERTIFIED_CANDIDATE, "m00_candidate_nine_union"),
    B3_SGV15B: (CUT_CERTIFIED_DOCUMENT, "m10_document_nine_union"),
}
REPRODUCED_FIELDS = (
    "tau",
    "cut_feasible",
    "declared_depth",
    "coverage",
    "realized_harm_rate",
    "repair_recall",
    "cut_accepted_in_sample",
    "cut_harmful_in_sample",
)


def run_reproduce(only: str | None = None) -> int:
    """Reproduce SGV15's and SGV15b's frozen certified cells with this stage's own code."""
    started = time.monotonic()
    deployments = load_deployments()
    if only is not None:
        deployments = {only: deployments[only]}
    stored = pd.read_parquet(s15b.FACTORIAL_CELLS)
    stored = stored[stored["cell_kind"] == "arm"] if "cell_kind" in stored else stored

    comparisons: list[dict[str, Any]] = []
    worst = 0.0
    mismatches: list[dict[str, Any]] = []
    for name, deployment in deployments.items():
        for baseline, (rule, arm) in REPRODUCED_ARMS.items():
            block = stored[(stored["environment"] == name) & (stored["arm"] == arm)]
            for _, reference in block.iterrows():
                budget = int(reference["requested_docs"])
                draw = int(reference["draw"])
                epsilon = float(reference["epsilon"])
                sample = document_sample(deployment, budget, s15b._sample_seed(name, budget, draw))
                boundary = select_boundary(deployment, rule, sample, epsilon)
                tau = float(boundary["tau"])
                point = s13.deployed(
                    deployment.eval_scores,
                    deployment.eval_harm,
                    deployment.eval_beneficial,
                    tau,
                    epsilon,
                )
                mine = {
                    "tau": tau,
                    "cut_feasible": bool(boundary["feasible"]),
                    "declared_depth": float(boundary["depth"]),
                    "coverage": float(point["coverage"]),
                    "realized_harm_rate": float(point["realized_harm_rate"]),
                    "repair_recall": float(point["repair_recall"]),
                    "cut_accepted_in_sample": int(boundary["accepted_in_sample"]),
                    "cut_harmful_in_sample": int(boundary["harmful_in_sample"]),
                }
                for field in REPRODUCED_FIELDS:
                    theirs = reference[field]
                    ours = mine[field]
                    if isinstance(ours, bool):
                        if bool(theirs) != ours:
                            mismatches.append(
                                {
                                    "environment": name,
                                    "arm": arm,
                                    "field": field,
                                    "budget": budget,
                                    "draw": draw,
                                    "epsilon": epsilon,
                                }
                            )
                        continue
                    left, right = float(theirs), float(ours)
                    if math.isnan(left) and math.isnan(right):
                        continue
                    if left == right:
                        continue
                    if math.isinf(left) and math.isinf(right) and left == right:
                        continue
                    gap = abs(left - right)
                    worst = max(worst, gap)
                    mismatches.append(
                        {
                            "environment": name,
                            "arm": arm,
                            "field": field,
                            "budget": budget,
                            "draw": draw,
                            "epsilon": epsilon,
                            "difference": gap,
                        }
                    )
            comparisons.append(
                {
                    "environment": name,
                    "baseline": baseline,
                    "upstream_arm": arm,
                    "cells": len(block),
                }
            )

    if mismatches:
        raise PhaseError(
            f"this stage's certified baselines do not reproduce the frozen upstream cells: "
            f"{len(mismatches)} differences, worst {worst}. {mismatches[:3]}"
        )
    payload = {
        **_envelope("upstream_reproduction"),
        "what_is_reproduced": (
            "SGV15's frozen candidate-level Clopper-Pearson procedure and SGV15b's frozen "
            "document-aware finite-sample procedure, both over SGV15's nine declared depths at a "
            "union-bound level of alpha / 9, replayed on SGV15b's own document-sample seeds and "
            "budgets and compared field by field against SGV15b's frozen cell table."
        ),
        "source": _relative(s15b.FACTORIAL_CELLS),
        "source_sha256": file_sha256(s15b.FACTORIAL_CELLS),
        "fields_compared": list(REPRODUCED_FIELDS),
        "comparisons": comparisons,
        "cells_compared": int(sum(record["cells"] for record in comparisons)),
        "mismatches": len(mismatches),
        "max_absolute_difference": worst,
        "identical": True,
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(REPRODUCTION, payload)
    print(
        f"reproduce: {payload['cells_compared']} upstream cells, "
        f"{len(REPRODUCED_FIELDS)} fields each, max difference {worst} -> {_relative(REPRODUCTION)}"
    )
    return 0


# ------------------------------------------------------------------ reading the cell table
#
# Two aggregation conventions, both inherited from SGV15 and SGV15b rather than invented here.
# Safety is judged on EVERY draw, because a deployment gets one sample and a rule that exceeds the
# tolerance on one draw in twenty has not controlled risk. Automation is judged on the MEAN over
# draws against the project's existing non-degeneracy floor.

AGGREGATED = (
    "automatic_harm",
    "automatic_coverage",
    "automatic_repair_recall",
    "risk_controlled_automatic_recall",
    "total_repair_recall",
    "risk_controlled_total_recall",
    "review_rate",
    "preserve_rate",
    "end_to_end_residual_error",
    "review_yield",
    "harm_intercept_rate",
    "harm_intercept_rate_realised",
    "harm_found_per_100_reviews",
    "benefit_recovered_per_100_reviews",
    "net_correct_repairs",
    "false_automatic_corrections",
    "reviewed",
    "reviewed_documents",
    "auto_applied",
    "repairs_recovered_by_review",
    "repairs_lost_to_review",
    "harm_introduced_by_review",
    "harm_readmitted_by_review",
    "certification_labels",
    "supervision_labels",
    "achieved_documents",
)


def slice_cells(
    cells: pd.DataFrame,
    policy: str,
    *,
    epsilon: float = PRIMARY_EPSILON,
    review_budget: float = 0.0,
    scenario: str | None = None,
    documents: int = PRACTICAL_DOC_BUDGET,
) -> pd.DataFrame:
    name = (
        scenario
        if scenario is not None
        else (H0_NO_REVIEW if review_budget == 0.0 else PRIMARY_SCENARIO)
    )
    return cells[
        (cells["policy"] == policy)
        & (cells["epsilon"] == epsilon)
        & (cells["review_budget"] == review_budget)
        & (cells["reviewer_scenario"] == name)
        & (cells["requested_documents"] == documents)
    ]


def per_environment(block: pd.DataFrame, floor: float, epsilon: float) -> dict[str, dict[str, Any]]:
    """One record per environment, aggregated over that environment's draws."""
    out: dict[str, dict[str, Any]] = {}
    for name, rows in block.groupby("environment", observed=True):
        record: dict[str, Any] = {
            field: _mean([float(value) for value in rows[field].tolist()]) for field in AGGREGATED
        }
        record["draws"] = len(rows)
        record["holds_bound_fraction"] = float(rows["holds_bound"].mean())
        record["holds_on_every_draw"] = bool(float(rows["holds_bound"].mean()) >= 1.0)
        record["feasible_fraction"] = float(rows["cut_feasible"].mean())
        record["non_degenerate"] = bool(record["automatic_coverage"] >= floor)
        record["safe_and_non_degenerate"] = bool(
            record["holds_on_every_draw"] and record["non_degenerate"]
        )
        record["epsilon"] = float(epsilon)
        out[str(name)] = record
    return out


def _summary(per_env: dict[str, dict[str, Any]], floor: float) -> dict[str, Any]:
    names = sorted(per_env)
    return {
        "environments": len(names),
        "environments_holding_the_bound": sorted(
            name for name in names if per_env[name]["holds_on_every_draw"]
        ),
        "environments_non_degenerate": sorted(
            name for name in names if per_env[name]["non_degenerate"]
        ),
        "environments_safe_and_non_degenerate": sorted(
            name for name in names if per_env[name]["safe_and_non_degenerate"]
        ),
        "mean_automatic_harm": _mean([per_env[n]["automatic_harm"] for n in names]),
        "mean_automatic_coverage": _mean([per_env[n]["automatic_coverage"] for n in names]),
        "mean_automatic_repair_recall": _mean(
            [per_env[n]["automatic_repair_recall"] for n in names]
        ),
        "mean_risk_controlled_automatic_recall": _mean(
            [per_env[n]["risk_controlled_automatic_recall"] for n in names]
        ),
        "mean_total_repair_recall": _mean([per_env[n]["total_repair_recall"] for n in names]),
        "mean_review_rate": _mean([per_env[n]["review_rate"] for n in names]),
        "mean_end_to_end_residual_error": _mean(
            [per_env[n]["end_to_end_residual_error"] for n in names]
        ),
        "coverage_floor": floor,
    }


def run_auto_only() -> int:
    """Sections 13 and 14: what the review-free policies do, including the certified baselines."""
    started = time.monotonic()
    cells, _design = load_cells()
    floor = load_floor()
    review_free = (P0_PRESERVE, P1_AUTO_ONLY, B1_SOURCE, B2_SGV15, B3_SGV15B)

    by_policy: dict[str, Any] = {}
    for policy in review_free:
        per_budget: dict[str, Any] = {}
        for epsilon in EPSILONS:
            per_documents: dict[str, Any] = {}
            for documents in DOC_BUDGETS:
                block = slice_cells(
                    cells, policy, epsilon=epsilon, review_budget=0.0, documents=documents
                )
                if block.empty:
                    continue
                per_env = per_environment(block, floor, epsilon)
                per_documents[str(documents)] = {
                    "per_environment": per_env,
                    "summary": _summary(per_env, floor),
                }
            per_budget[f"epsilon_{epsilon}"] = per_documents
        by_policy[policy] = {
            "safety_label": CUT_SAFETY[POLICY_RULE[policy][0]],
            "by_epsilon": per_budget,
        }

    _write_json_once(
        AUTO_ONLY_RESULTS,
        {
            **_envelope("auto_only_results"),
            "question": "what is reachable with no human review at all?",
            "policies": list(review_free),
            "aggregation": {
                "safety": "the bound must hold on EVERY draw",
                "automation": "the mean over draws against the inherited non-degeneracy floor",
                "source": "SGV15's and SGV15b's own conventions, not new ones",
            },
            "certified_reference_points": {
                B2_SGV15: "SGV15's frozen candidate-level Clopper-Pearson procedure",
                B3_SGV15B: "SGV15b's frozen document-aware finite-sample procedure",
            },
            "by_policy": by_policy,
            "coverage_floor": floor,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"auto-only: {len(review_free)} review-free policies -> {_relative(AUTO_ONLY_RESULTS)}")
    return 0


# ---------------------------------------------- sections 15, 17, 18, 21-22: what review buys


def run_review() -> int:
    """The review-bearing policies, the routing comparison, review yield and interception."""
    started = time.monotonic()
    cells, _design = load_cells()
    floor = load_floor()

    by_policy: dict[str, Any] = {}
    for policy in REVIEW_POLICIES:
        per_epsilon: dict[str, Any] = {}
        for epsilon in EPSILONS:
            per_documents: dict[str, Any] = {}
            for documents in DOC_BUDGETS:
                per_rate: dict[str, Any] = {}
                for rate in REVIEW_BUDGETS:
                    per_scenario: dict[str, Any] = {}
                    names = (
                        (H0_NO_REVIEW,)
                        if rate == 0.0
                        else (
                            (*REVIEWER_SCENARIOS, *CONTROL_SCENARIOS)
                            if epsilon == PRIMARY_EPSILON
                            else REVIEWER_SCENARIOS
                        )
                    )
                    for scenario in names:
                        block = slice_cells(
                            cells,
                            policy,
                            epsilon=epsilon,
                            review_budget=rate,
                            scenario=scenario,
                            documents=documents,
                        )
                        if block.empty:
                            continue
                        per_env = per_environment(block, floor, epsilon)
                        per_scenario[scenario] = {
                            "per_environment": per_env,
                            "summary": _summary(per_env, floor),
                        }
                    if per_scenario:
                        per_rate[f"review_{rate}"] = per_scenario
                if per_rate:
                    per_documents[str(documents)] = per_rate
            per_epsilon[f"epsilon_{epsilon}"] = per_documents
        by_policy[policy] = {
            "boundary_rule": POLICY_RULE[policy][0],
            "routing": POLICY_RULE[policy][1],
            "safety_label": CUT_SAFETY[POLICY_RULE[policy][0]],
            "status": "SECONDARY" if policy in SECONDARY_POLICIES else "PRIMARY",
            "by_epsilon": per_epsilon,
        }

    _write_json_once(
        HUMAN_REVIEW_RESULTS,
        {
            **_envelope("human_review_results"),
            "question": "what does buying human review change?",
            "policies": list(REVIEW_POLICIES),
            "reviewer_scenarios_are_simulations": (
                "no reviewer accuracy here was measured on human beings. The ideal reviewer is an "
                "IDEALIZED UPPER BOUND and the coin-flip reviewer is a negative control."
            ),
            "by_policy": by_policy,
            "coverage_floor": floor,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    # Section 18's pre-registered routing comparison, at matched budget and the same boundary.
    routing_tables: dict[str, tuple[Path, str]] = {
        R1_BOUNDARY: (BOUNDARY_REVIEW_RESULTS, "boundary_review_results"),
        R0_RANDOM: (RANDOM_REVIEW_RESULTS, "random_review_results"),
        R2_HIGH_RISK: (HIGH_RISK_REVIEW_RESULTS, "high_risk_review_results"),
    }
    per_routing: dict[str, dict[str, Any]] = {}
    for arm, (path, artifact) in routing_tables.items():
        policy = ROUTING_ARMS[arm]
        table: dict[str, Any] = {}
        for rate in REVIEW_BUDGETS:
            per_scenario: dict[str, Any] = {}
            for scenario in (H0_NO_REVIEW,) if rate == 0.0 else REVIEWER_SCENARIOS:
                block = slice_cells(
                    cells,
                    policy,
                    review_budget=rate,
                    scenario=scenario,
                    documents=PRACTICAL_DOC_BUDGET,
                )
                if block.empty:
                    continue
                per_env = per_environment(block, floor, PRIMARY_EPSILON)
                per_scenario[scenario] = {
                    "per_environment": per_env,
                    "summary": _summary(per_env, floor),
                }
            table[f"review_{rate}"] = per_scenario
        per_routing[arm] = table
        _write_json_once(
            path,
            {
                **_envelope(artifact),
                "routing_arm": arm,
                "policy": policy,
                "routing": POLICY_RULE[policy][1],
                "epsilon": float(PRIMARY_EPSILON),
                "document_budget": PRACTICAL_DOC_BUDGET,
                "matched": (
                    "the same empirical boundary and the same review budget as the other two "
                    "routings"
                ),
                "by_review_budget": table,
                "runtime_seconds": time.monotonic() - started,
            },
        )

    # Sections 17 and 18's efficiency questions, read off the same slices.
    yields: dict[str, Any] = {}
    intercepts: dict[str, Any] = {}
    for arm, table in per_routing.items():
        for key, per_scenario in table.items():
            for scenario, record in per_scenario.items():
                names = sorted(record["per_environment"])
                cell = f"{arm}|{key}|{scenario}"
                yields[cell] = {
                    "mean_review_yield": _mean(
                        [record["per_environment"][n]["review_yield"] for n in names]
                    ),
                    "mean_harm_found_per_100_reviews": _mean(
                        [record["per_environment"][n]["harm_found_per_100_reviews"] for n in names]
                    ),
                    "mean_benefit_recovered_per_100_reviews": _mean(
                        [
                            record["per_environment"][n]["benefit_recovered_per_100_reviews"]
                            for n in names
                        ]
                    ),
                    "mean_reviewed_candidates": _mean(
                        [record["per_environment"][n]["reviewed"] for n in names]
                    ),
                    "mean_reviewed_documents": _mean(
                        [record["per_environment"][n]["reviewed_documents"] for n in names]
                    ),
                    "mean_repairs_lost_to_review": _mean(
                        [record["per_environment"][n]["repairs_lost_to_review"] for n in names]
                    ),
                    "mean_harm_introduced_by_review": _mean(
                        [record["per_environment"][n]["harm_introduced_by_review"] for n in names]
                    ),
                }
                intercepts[cell] = {
                    "mean_routing_intercept_rate": _mean(
                        [record["per_environment"][n]["harm_intercept_rate"] for n in names]
                    ),
                    "mean_realised_intercept_rate": _mean(
                        [
                            record["per_environment"][n]["harm_intercept_rate_realised"]
                            for n in names
                        ]
                    ),
                    "mean_harm_readmitted_by_review": _mean(
                        [record["per_environment"][n]["harm_readmitted_by_review"] for n in names]
                    ),
                    "mean_false_automatic_corrections": _mean(
                        [record["per_environment"][n]["false_automatic_corrections"] for n in names]
                    ),
                }

    _write_json_once(
        REVIEW_YIELD,
        {
            **_envelope("review_yield"),
            "definition": (
                "reviewed candidates whose review outcome is strictly better than the no-review "
                "counterfactual, over reviewed candidates: a harmful candidate blocked before "
                "automatic application, or a beneficial candidate applied that would have been "
                "preserved. Agreeing with the counterfactual is not yield."
            ),
            "decomposition": "review_yield = (harm found + benefit recovered) per review",
            "costs_are_reported_too": (
                "repairs lost to review and harm introduced by review are reported beside the "
                "yield, because a fallible reviewer moves candidates in both directions."
            ),
            "by_cell": yields,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        HARM_INTERCEPT,
        {
            **_envelope("harm_intercept"),
            "definition": (
                "harmful candidates routed away from automatic application, over harmful "
                "candidates that would otherwise have been auto-applied. That is a property of "
                "the ROUTING. The realised rate, after the reviewer's own errors, is reported "
                "beside it and is the one a deployment experiences."
            ),
            "by_cell": intercepts,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"review: {len(REVIEW_POLICIES)} review policies, {len(per_routing)} routings, "
        f"{len(yields)} efficiency cells -> {_relative(HUMAN_REVIEW_RESULTS)}"
    )
    return 0


# ---------------------------------------------- sections 13, 23, 24, 25: the frontiers
#
# Three frontiers, kept apart because they answer different questions. The review frontier holds
# supervision fixed and prices review. The supervision frontier holds review fixed and prices
# target documents. The harm-tolerance frontier holds both fixed and prices safety.


def _frontier_point(
    cells: pd.DataFrame,
    policy: str,
    epsilon: float,
    rate: float,
    documents: int,
    scenario: str,
    floor: float,
) -> dict[str, Any] | None:
    block = slice_cells(
        cells, policy, epsilon=epsilon, review_budget=rate, scenario=scenario, documents=documents
    )
    if block.empty:
        return None
    per_env = per_environment(block, floor, epsilon)
    summary = _summary(per_env, floor)
    return {
        "policy": policy,
        "epsilon": float(epsilon),
        "review_budget": float(rate),
        "document_budget": int(documents),
        "reviewer_scenario": scenario,
        "safety_label": CUT_SAFETY[POLICY_RULE[policy][0]],
        "mean_automatic_harm": summary["mean_automatic_harm"],
        "mean_automatic_repair_recall": summary["mean_automatic_repair_recall"],
        "mean_risk_controlled_automatic_recall": summary["mean_risk_controlled_automatic_recall"],
        "mean_total_repair_recall": summary["mean_total_repair_recall"],
        "mean_review_rate": summary["mean_review_rate"],
        "mean_automatic_coverage": summary["mean_automatic_coverage"],
        "mean_end_to_end_residual_error": summary["mean_end_to_end_residual_error"],
        "environments_holding_the_bound": len(summary["environments_holding_the_bound"]),
        "environments_safe_and_non_degenerate": len(
            summary["environments_safe_and_non_degenerate"]
        ),
        "mean_certification_labels": _mean(
            [per_env[n]["certification_labels"] for n in sorted(per_env)]
        ),
        "mean_supervision_labels": _mean(
            [per_env[n]["supervision_labels"] for n in sorted(per_env)]
        ),
        "mean_reviewed_candidates": _mean([per_env[n]["reviewed"] for n in sorted(per_env)]),
        "mean_reviewed_documents": _mean(
            [per_env[n]["reviewed_documents"] for n in sorted(per_env)]
        ),
        "per_environment": per_env,
    }


def _scenarios_for(rate: float, epsilon: float) -> tuple[str, ...]:
    if rate == 0.0:
        return (H0_NO_REVIEW,)
    return (
        (*REVIEWER_SCENARIOS, *CONTROL_SCENARIOS)
        if epsilon == PRIMARY_EPSILON
        else REVIEWER_SCENARIOS
    )


def _all_points(cells: pd.DataFrame, floor: float) -> list[dict[str, Any]]:
    """Every declared operating point in the stage, as one flat list."""
    points: list[dict[str, Any]] = []
    for policy in POLICY_NAMES:
        rates = REVIEW_BUDGETS if POLICY_RULE[policy][2] else (0.0,)
        for epsilon in EPSILONS:
            for documents in DOC_BUDGETS:
                for rate in rates:
                    for scenario in _scenarios_for(rate, epsilon):
                        point = _frontier_point(
                            cells, policy, epsilon, rate, documents, scenario, floor
                        )
                        if point is not None:
                            points.append(point)
    return points


def run_frontier() -> int:
    """The review, supervision and harm-tolerance frontiers, plus section 14's primary table."""
    started = time.monotonic()
    cells, _design = load_cells()
    floor = load_floor()

    # Section 24: review rate -> automation, at the primary tolerance.
    review: dict[str, Any] = {}
    for policy in REVIEW_POLICIES:
        per_scenario: dict[str, Any] = {}
        for scenario in REVIEWER_SCENARIOS:
            series: list[dict[str, Any]] = []
            for rate in REVIEW_BUDGETS:
                point = _frontier_point(
                    cells,
                    policy,
                    PRIMARY_EPSILON,
                    rate,
                    PRACTICAL_DOC_BUDGET,
                    H0_NO_REVIEW if rate == 0.0 else scenario,
                    floor,
                )
                if point is not None:
                    series.append({k: v for k, v in point.items() if k != "per_environment"})
            per_scenario[scenario] = series
        review[policy] = per_scenario

    # Section 25: target documents -> automation and review burden, at fixed review budgets.
    supervision: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        rates = REVIEW_BUDGETS if POLICY_RULE[policy][2] else (0.0,)
        per_rate: dict[str, Any] = {}
        for rate in rates:
            series = []
            for documents in DOC_BUDGETS:
                point = _frontier_point(
                    cells,
                    policy,
                    PRIMARY_EPSILON,
                    rate,
                    documents,
                    H0_NO_REVIEW if rate == 0.0 else PRIMARY_SCENARIO,
                    floor,
                )
                if point is not None:
                    series.append({k: v for k, v in point.items() if k != "per_environment"})
            per_rate[f"review_{rate}"] = series
        supervision[policy] = per_rate

    # Section 23: the price of safety, at a fixed review budget and document budget.
    tolerance: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        rates = (0.0, PRACTICAL_REVIEW_BUDGET) if POLICY_RULE[policy][2] else (0.0,)
        per_rate = {}
        for rate in rates:
            series = []
            for epsilon in EPSILONS:
                point = _frontier_point(
                    cells,
                    policy,
                    epsilon,
                    rate,
                    PRACTICAL_DOC_BUDGET,
                    H0_NO_REVIEW if rate == 0.0 else PRIMARY_SCENARIO,
                    floor,
                )
                if point is not None:
                    series.append({k: v for k, v in point.items() if k != "per_environment"})
            per_rate[f"review_{rate}"] = series
        tolerance[policy] = per_rate

    # Section 14's primary question, as a table: the maximum automatic repair recall reachable at
    # each (target documents, review budget) under the primary tolerance, over the primary
    # policies, with the harm constraint enforced rather than assumed.
    primary_table: dict[str, Any] = {}
    for documents in DOC_BUDGETS:
        row: dict[str, Any] = {}
        for rate in REVIEW_BUDGETS:
            best: dict[str, Any] | None = None
            for policy in SELECTABLE_POLICIES:
                if rate > 0.0 and not POLICY_RULE[policy][2]:
                    continue
                if rate == 0.0 and POLICY_RULE[policy][2] and policy != P3_BOUNDARY:
                    continue
                point = _frontier_point(
                    cells,
                    policy,
                    PRIMARY_EPSILON,
                    rate,
                    documents,
                    H0_NO_REVIEW if rate == 0.0 else PRIMARY_SCENARIO,
                    floor,
                )
                if point is None:
                    continue
                value = float(point["mean_risk_controlled_automatic_recall"])
                if best is None or value > float(best["mean_risk_controlled_automatic_recall"]):
                    best = point
            row[f"review_{rate}"] = (
                None
                if best is None
                else {
                    "policy": best["policy"],
                    "mean_risk_controlled_automatic_recall": best[
                        "mean_risk_controlled_automatic_recall"
                    ],
                    "mean_automatic_repair_recall": best["mean_automatic_repair_recall"],
                    "mean_total_repair_recall": best["mean_total_repair_recall"],
                    "mean_automatic_harm": best["mean_automatic_harm"],
                    "mean_review_rate": best["mean_review_rate"],
                    "environments_safe_and_non_degenerate": best[
                        "environments_safe_and_non_degenerate"
                    ],
                }
            )
        primary_table[str(documents)] = row

    _write_json_once(
        REVIEW_FRONTIER,
        {
            **_envelope("review_frontier"),
            "question": (
                "at the primary tolerance and the practical document budget, what does review "
                "rate buy?"
            ),
            "epsilon": float(PRIMARY_EPSILON),
            "document_budget": PRACTICAL_DOC_BUDGET,
            "by_policy": review,
            "primary_table": primary_table,
            "primary_table_rule": (
                "the best mean risk-controlled automatic repair recall over the selectable "
                "primary policies at each (target documents, review budget). Risk-controlled "
                "means the harm constraint is enforced inside the endpoint, so a policy cannot "
                "enter the table by breaking the bound it was given."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        SUPERVISION_FRONTIER,
        {
            **_envelope("supervision_frontier"),
            "question": "does more target supervision reduce the review burden?",
            "epsilon": float(PRIMARY_EPSILON),
            "reviewer_scenario": PRIMARY_SCENARIO,
            "certification_impossibility_reference": {
                "median_documents_required_finite_sample": float(
                    cc_read_json(s15b.PROJECTION)["summary"]["median_documents_required_finite"]
                ),
                "median_documents_required_asymptotic": float(
                    cc_read_json(s15b.PROJECTION)["summary"]["median_documents_required_asymptotic"]
                ),
                "largest_size_tested": int(
                    cc_read_json(s15b.PROJECTION)["summary"]["largest_size_tested"]
                ),
                "status": cc_read_json(s15b.PROJECTION)["status"],
                "available_pool_documents": {
                    record["environment"]: record["certification_documents"]
                    for record in cc_read_json(ENVIRONMENT_INVENTORY)["environments"]
                },
                "why_it_is_here": (
                    "the requirement and the availability belong on the same axis. It is an "
                    "EXTRAPOLATION from SGV15b and is labelled one; no criterion reads it."
                ),
            },
            "by_policy": supervision,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        HARM_TOLERANCE_FRONTIER,
        {
            **_envelope("harm_tolerance_frontier"),
            "question": "how much automation is lost by tightening the tolerance?",
            "epsilons": [float(value) for value in EPSILONS],
            "primary_epsilon": float(PRIMARY_EPSILON),
            "rule": (
                "the primary target does not move. This axis prices safety; it does not choose "
                "the tolerance that performs best."
            ),
            "document_budget": PRACTICAL_DOC_BUDGET,
            "by_policy": tolerance,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"frontier: review {len(review)} policies, supervision {len(supervision)}, "
        f"tolerance {len(tolerance)}, primary table {len(primary_table)} rows -> "
        f"{_relative(REVIEW_FRONTIER)}"
    )
    return 0


# ------------------------------------------------------------------ section 36: Pareto
#
# Four axes, three minimised and one maximised, with no weighting between them. A weighted score
# would decide the trade-off this stage exists to expose.

PARETO_MINIMISE = ("mean_automatic_harm", "mean_review_rate", "document_budget")
PARETO_MAXIMISE = ("mean_risk_controlled_automatic_recall",)
PARETO_SECONDARY_MAXIMISE = ("mean_total_repair_recall",)


def _point_key(point: dict[str, Any]) -> str:
    return "|".join(
        (
            str(point["policy"]),
            f"{float(point['epsilon']):.2f}",
            f"{float(point['review_budget']):.2f}",
            str(int(point["document_budget"])),
            str(point["reviewer_scenario"]),
        )
    )


def _vector(point: dict[str, Any], maximise: Sequence[str]) -> tuple[float, ...]:
    """A point in dominance space. An undefined harm rate means nothing was applied.

    A policy that applies nothing has no accepted set to be harmful, so its harm is not zero and
    must not be allowed to dominate on the safety axis. It is placed at the tolerance, which is
    the worst value a point that DOES apply something is allowed to reach.
    """
    values: list[float] = []
    for field in PARETO_MINIMISE:
        raw = float(point[field])
        if field == "mean_automatic_harm" and math.isnan(raw):
            raw = float(point["epsilon"])
        values.append(raw)
    for field in maximise:
        values.append(-float(point[field]))
    return tuple(values)


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True)) and any(
        a < b for a, b in zip(left, right, strict=True)
    )


def run_pareto() -> int:
    """The non-dominated frontier and the full dominance relation, deterministically."""
    started = time.monotonic()
    cells, _design = load_cells()
    floor = load_floor()
    points = _all_points(cells, floor)
    flat = [{k: v for k, v in point.items() if k != "per_environment"} for point in points]
    keys = [_point_key(point) for point in flat]
    order = sorted(range(len(flat)), key=lambda index: keys[index])
    flat = [flat[index] for index in order]
    keys = [keys[index] for index in order]

    frontiers: dict[str, Any] = {}
    dominance: dict[str, Any] = {}
    for label, maximise in (
        ("automatic_repair_recall", PARETO_MAXIMISE),
        ("total_repair_recall", PARETO_SECONDARY_MAXIMISE),
    ):
        vectors = [_vector(point, maximise) for point in flat]
        dominated_by: list[list[str]] = [[] for _ in flat]
        for i, left in enumerate(vectors):
            for j, right in enumerate(vectors):
                if i != j and _dominates(left, right):
                    dominated_by[j].append(keys[i])
        non_dominated = [
            {**flat[index], "key": keys[index]}
            for index in range(len(flat))
            if not dominated_by[index]
        ]
        frontiers[label] = {
            "objective": {
                "minimise": list(PARETO_MINIMISE),
                "maximise": list(maximise),
                "no_weighting": "the axes are never combined into a scalar",
            },
            "points_considered": len(flat),
            "non_dominated_count": len(non_dominated),
            "non_dominated": sorted(non_dominated, key=lambda record: record["key"]),
        }
        dominance[label] = {
            keys[index]: sorted(dominated_by[index])
            for index in range(len(flat))
            if dominated_by[index]
        }

    _write_json_once(
        PARETO_FRONTIER,
        {
            **_envelope("pareto_frontier"),
            "tie_handling": (
                "points are sorted by their canonical key before the sweep and a point never "
                "dominates itself, so equal points are mutually non-dominated and both survive. "
                "The result does not depend on iteration order."
            ),
            "undefined_harm": (
                "a policy that applies nothing has an undefined harm rate, not a zero one. On the "
                "dominance axis it is placed AT the tolerance so that refusing everything cannot "
                "dominate a policy that actually applies edits."
            ),
            "frontiers": frontiers,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DOMINANCE_MATRIX,
        {
            **_envelope("dominance_matrix"),
            "encoding": (
                "for each dominated point, the sorted keys of every point that dominates it"
            ),
            "points": len(flat),
            "by_objective": dominance,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"pareto: {len(flat)} operating points, "
        f"{frontiers['automatic_repair_recall']['non_dominated_count']} non-dominated on "
        f"automatic recall, {frontiers['total_repair_recall']['non_dominated_count']} on total "
        f"recall -> {_relative(PARETO_FRONTIER)}"
    )
    return 0


# ---------------------------------------------------- section 22: reviewer-accuracy sensitivity


def run_sensitivity() -> int:
    """How much of each review policy's benefit survives a fallible reviewer."""
    started = time.monotonic()
    cells, _design = load_cells()
    floor = load_floor()
    by_policy: dict[str, Any] = {}
    for policy in REVIEW_POLICIES:
        per_rate: dict[str, Any] = {}
        for rate in REVIEW_BUDGETS:
            if rate == 0.0:
                continue
            per_scenario: dict[str, Any] = {}
            for scenario in (*REVIEWER_SCENARIOS, *CONTROL_SCENARIOS):
                point = _frontier_point(
                    cells, policy, PRIMARY_EPSILON, rate, PRACTICAL_DOC_BUDGET, scenario, floor
                )
                if point is None:
                    continue
                per_scenario[scenario] = {k: v for k, v in point.items() if k != "per_environment"}
                per_scenario[scenario]["reviewer_accuracy"] = REVIEWER_ACCURACY[scenario]
                per_scenario[scenario]["role"] = (
                    "IDEALIZED UPPER BOUND"
                    if scenario == H1_IDEAL
                    else (
                        "NEGATIVE CONTROL"
                        if scenario in CONTROL_SCENARIOS
                        else "sensitivity scenario"
                    )
                )
            if not per_scenario:
                continue
            ideal = per_scenario.get(H1_IDEAL)
            moderate = per_scenario.get(H3_MODERATE)
            per_rate[f"review_{rate}"] = {
                "by_scenario": per_scenario,
                "total_recall_lost_from_ideal_to_moderate": (
                    None
                    if ideal is None or moderate is None
                    else float(
                        ideal["mean_total_repair_recall"] - moderate["mean_total_repair_recall"]
                    )
                ),
                "automatic_harm_change_from_ideal_to_moderate": (
                    None
                    if ideal is None or moderate is None
                    else float(moderate["mean_automatic_harm"] - ideal["mean_automatic_harm"])
                ),
                "depends_on_a_near_perfect_reviewer": (
                    None
                    if ideal is None or moderate is None
                    else bool(
                        int(ideal["environments_safe_and_non_degenerate"])
                        > int(moderate["environments_safe_and_non_degenerate"])
                    )
                ),
            }
        by_policy[policy] = per_rate

    _write_json_once(
        REVIEW_ACCURACY_SENSITIVITY,
        {
            **_envelope("review_accuracy_sensitivity"),
            "status": (
                "SIMULATION SCENARIOS. No reviewer accuracy here was measured on human beings. "
                "The ideal reviewer is an IDEALIZED UPPER BOUND, never a deployment claim."
            ),
            "common_random_numbers": (
                "the scenarios are nested under one frozen uniform per evaluation row, so the "
                "comparison across accuracies is paired on the same candidates."
            ),
            "epsilon": float(PRIMARY_EPSILON),
            "document_budget": PRACTICAL_DOC_BUDGET,
            "by_policy": by_policy,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"sensitivity: {len(by_policy)} policies across "
        f"{len(REVIEWER_SCENARIOS) + len(CONTROL_SCENARIOS)} reviewer scenarios -> "
        f"{_relative(REVIEW_ACCURACY_SENSITIVITY)}"
    )
    return 0


# ------------------------------------------------- sections 20-21: sequential deployment
#
# SECONDARY. The primary analysis is static and feeds no reviewed label back anywhere. This
# scenario asks a different question: if a deployment runs in batches and keeps what its reviewers
# decided, does that supervision reduce what the next batch has to review?
#
# Note what is and is not updated. Section 2 freezes the model, so a reviewed label may move the
# BOUNDARY -- a deployment decision -- and may never refit the ranking. And a label revealed in
# batch t is available from batch t+1 onwards and never earlier, so no reviewed label can improve
# the decision that routed that same candidate to review.


def _batches(deployment: Deployment, count: int) -> list[np.ndarray]:
    """Whole documents, split deterministically by their sorted position. Never by row."""
    documents = np.arange(deployment.eval_documents.size)
    chunks = np.array_split(documents, count)
    return [np.flatnonzero(np.isin(deployment.eval_doc_of, chunk)) for chunk in chunks]


def _empirical_boundary_from(
    deployment: Deployment, accepted: np.ndarray, harmful: np.ndarray, epsilon: float
) -> dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(accepted > 0, harmful / np.maximum(accepted, 1), np.inf)
    feasible = np.flatnonzero((accepted > 0) & (rate <= epsilon))
    if feasible.size == 0:
        return {"tau": REFUSE, "feasible": False, "index": -1}
    index = int(feasible.max())
    return {"tau": float(deployment.fine_thresholds[index]), "feasible": True, "index": index}


def run_sequential() -> int:
    """The batched-deployment scenario, run under both label-reuse regimes."""
    started = time.monotonic()
    deployments = load_deployments()
    floor = load_floor()
    epsilon = float(PRIMARY_EPSILON)
    rate = float(PRACTICAL_REVIEW_BUDGET)
    accuracy = float(REVIEWER_ACCURACY[PRIMARY_SCENARIO])

    per_environment_out: dict[str, Any] = {}
    for name, deployment in deployments.items():
        correctness = reviewer_correctness(deployment)
        sample = document_sample(
            deployment, PRACTICAL_DOC_BUDGET, _sample_seed(name, PRACTICAL_DOC_BUDGET, 0)
        )
        base_accepted = deployment.accepted[sample].sum(axis=0).astype(float)
        base_harmful = deployment.harmful[sample].sum(axis=0).astype(float)
        batches = _batches(deployment, SEQUENTIAL_BATCHES)
        per_regime: dict[str, Any] = {}
        for regime in (L0_NO_REUSE, L1_REUSE_AFTER_REVIEW):
            accepted = base_accepted.copy()
            harmful = base_harmful.copy()
            history: list[dict[str, Any]] = []
            for index, rows in enumerate(batches):
                boundary = _empirical_boundary_from(deployment, accepted, harmful, epsilon)
                tau = float(boundary["tau"])
                scores = deployment.eval_scores[rows]
                auto0 = (scores >= tau) if np.isfinite(tau) else np.zeros(rows.size, dtype=bool)
                order = rows[
                    np.argsort(np.abs(scores - tau) if np.isfinite(tau) else -scores, kind="stable")
                ]
                take = min(int(rate * rows.size), order.size)
                reviewed = order[:take]
                mask = np.zeros(deployment.n_rows, dtype=bool)
                mask[reviewed] = True
                correct = correctness <= accuracy
                benefit = deployment.eval_beneficial
                applies = (correct & benefit) | (~correct & ~benefit)
                auto = np.zeros(deployment.n_rows, dtype=bool)
                auto[rows[auto0]] = True
                auto &= ~mask
                applied = auto | (mask & applies)
                n_auto = int(auto.sum())
                n_harm = int((auto & deployment.eval_harm).sum())
                total_beneficial = int(benefit[rows].sum())
                history.append(
                    {
                        "batch": index,
                        "documents": int(np.unique(deployment.eval_doc_of[rows]).size),
                        "rows": int(rows.size),
                        "boundary_feasible": bool(boundary["feasible"]),
                        "boundary_index": int(boundary["index"]),
                        "labels_available_to_this_batch": int(accepted[-1]) if accepted.size else 0,
                        "auto_applied": n_auto,
                        "automatic_harm": float(n_harm / n_auto) if n_auto else float("nan"),
                        "holds_bound": True if n_auto == 0 else bool(n_harm / n_auto <= epsilon),
                        "automatic_coverage": float(n_auto / rows.size),
                        "automatic_repair_recall": (
                            float(int((auto & benefit).sum()) / total_beneficial)
                            if total_beneficial
                            else 0.0
                        ),
                        "total_repair_recall": (
                            float(int((applied & benefit)[rows].sum()) / total_beneficial)
                            if total_beneficial
                            else 0.0
                        ),
                        "reviewed": int(take),
                        "review_rate": float(take / rows.size),
                    }
                )
                if regime == L1_REUSE_AFTER_REVIEW and take:
                    # Available from the NEXT batch onwards, never to the batch it came from.
                    revealed = deployment.eval_scores[reviewed]
                    for position, threshold in enumerate(deployment.fine_thresholds):
                        clears = revealed >= threshold
                        accepted[position] += float(clears.sum())
                        harmful[position] += float((clears & deployment.eval_harm[reviewed]).sum())
            per_regime[regime] = {
                "batches": history,
                "mean_automatic_harm": _mean([b["automatic_harm"] for b in history]),
                "mean_automatic_coverage": _mean([b["automatic_coverage"] for b in history]),
                "mean_total_repair_recall": _mean([b["total_repair_recall"] for b in history]),
                "batches_holding_the_bound": int(sum(1 for b in history if b["holds_bound"])),
                "boundary_moved": bool(len({b["boundary_index"] for b in history}) > 1),
            }
        per_environment_out[name] = per_regime

    names = sorted(per_environment_out)
    _write_json_once(
        SEQUENTIAL_DEPLOYMENT,
        {
            **_envelope("sequential_deployment"),
            "status": "SECONDARY. The primary frontier is static and reuses no reviewed label.",
            "what_updates": (
                "the BOUNDARY only. Section 2 freezes the ranking model, so a reviewed label may "
                "move a deployment decision and may never refit a score."
            ),
            "no_temporal_leakage": (
                "a label revealed in batch t enters the pool used from batch t+1 onwards. It "
                "cannot improve the decision that routed that same candidate to review."
            ),
            "batches": SEQUENTIAL_BATCHES,
            "batch_unit": "whole evaluation documents, split deterministically by sorted position",
            "epsilon": epsilon,
            "review_budget": rate,
            "reviewer_scenario": PRIMARY_SCENARIO,
            "initial_document_budget": PRACTICAL_DOC_BUDGET,
            "per_environment": per_environment_out,
            "summary": {
                regime: {
                    "mean_automatic_harm": _mean(
                        [per_environment_out[n][regime]["mean_automatic_harm"] for n in names]
                    ),
                    "mean_automatic_coverage": _mean(
                        [per_environment_out[n][regime]["mean_automatic_coverage"] for n in names]
                    ),
                    "mean_total_repair_recall": _mean(
                        [per_environment_out[n][regime]["mean_total_repair_recall"] for n in names]
                    ),
                    "environments_where_the_boundary_moved": int(
                        sum(1 for n in names if per_environment_out[n][regime]["boundary_moved"])
                    ),
                }
                for regime in (L0_NO_REUSE, L1_REUSE_AFTER_REVIEW)
            },
            "coverage_floor": floor,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"sequential: {len(names)} environments x {SEQUENTIAL_BATCHES} batches x 2 reuse regimes "
        f"-> {_relative(SEQUENTIAL_DEPLOYMENT)}"
    )
    return 0


# ------------------------------------------------------------------ section 34: the controls
#
# Each control breaks exactly one thing and keeps everything else. If breaking it changes nothing,
# the thing it broke was not carrying the result.


def _run_configuration(
    deployments: dict[str, Deployment],
    floor: float,
    *,
    rule: str,
    routing: str,
    rate: float,
    accuracy: float,
    documents: int,
    epsilon: float = PRIMARY_EPSILON,
    permute_labels: int | None = None,
    shuffle_routing: int | None = None,
) -> dict[str, dict[str, Any]]:
    """One ad-hoc (boundary, routing, budget, reviewer) configuration, over every environment.

    `permute_labels` destroys the label-to-candidate assignment inside the purchased documents,
    holding the documents, the rows and the class balance. `shuffle_routing` permutes the score
    the ROUTING sees while leaving the boundary and the deployment untouched.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, deployment in deployments.items():
        correctness = reviewer_correctness(deployment)
        rows_out: list[dict[str, Any]] = []
        draws = 1 if documents == 0 else DOC_DRAWS
        for draw in range(draws):
            sample = document_sample(deployment, documents, _sample_seed(name, documents, draw))
            local = deployment
            if permute_labels is not None and sample.size:
                generator = np.random.default_rng(
                    _stable_seed("sgv-dt1-permute", str(permute_labels), name, str(draw))
                )
                inside = np.isin(deployment.cert_doc_of, sample)
                harm = deployment.cert_harm.copy()
                harm[inside] = generator.permutation(harm[inside])
                accepted, harmful = _per_document_prefix_counts(
                    deployment.cert_scores,
                    harm,
                    deployment.cert_doc_of,
                    deployment.fine_thresholds,
                    deployment.n_documents,
                )
                nine_accepted, nine_harmful = _per_document_prefix_counts(
                    deployment.cert_scores,
                    harm,
                    deployment.cert_doc_of,
                    deployment.nine_thresholds,
                    deployment.n_documents,
                )
                local = Deployment(
                    **{
                        **{
                            field: getattr(deployment, field)
                            for field in deployment.__slots__
                            if field
                            not in (
                                "cert_harm",
                                "accepted",
                                "harmful",
                                "nine_accepted",
                                "nine_harmful",
                            )
                        },
                        "cert_harm": harm,
                        "accepted": accepted,
                        "harmful": harmful,
                        "nine_accepted": nine_accepted,
                        "nine_harmful": nine_harmful,
                    }
                )
            boundary = select_boundary(local, rule, sample, epsilon)
            tau = float(boundary["tau"])
            if shuffle_routing is not None:
                generator = np.random.default_rng(
                    _stable_seed("sgv-dt1-shuffle", str(shuffle_routing), name, str(draw))
                )
                order = review_order(
                    deployment, tau, routing, scores=generator.permutation(deployment.eval_scores)
                )
            elif routing == ROUTE_EXPECTED_VALUE:
                order = review_order(
                    deployment, tau, routing, gain=expected_gain(deployment, sample, tau)
                )
            else:
                order = review_order(deployment, tau, routing)
            table = _measure(
                deployment,
                tau,
                order,
                (int(rate * deployment.n_rows),),
                (accuracy,),
                correctness,
                floor,
            )
            record = _finalize(table[(int(rate * deployment.n_rows), accuracy)], epsilon)
            record["cut_feasible"] = bool(boundary["feasible"])
            record["achieved_documents"] = int(sample.size)
            record["certification_labels"] = (
                int(deployment.doc_sizes()[sample].sum()) if sample.size else 0
            )
            record["supervision_labels"] = int(record["certification_labels"] + s15.ADAPT_BUDGET)
            rows_out.append(record)
        frame = pd.DataFrame(rows_out)
        out[name] = per_environment(frame.assign(environment=name), floor, epsilon)[name]
    return out


def _control(
    label: str,
    question: str,
    real: dict[str, dict[str, Any]],
    control: dict[str, dict[str, Any]],
    field: str,
    direction: str = "higher_is_better",
) -> dict[str, Any]:
    names = sorted(set(real) & set(control))
    left = _mean([real[name][field] for name in names])
    right = _mean([control[name][field] for name in names])
    delta = float(left - right)
    discriminates = bool(delta > 0.0) if direction == "higher_is_better" else bool(delta < 0.0)
    return {
        "question": question,
        "field": field,
        "direction": direction,
        "real": left,
        "control": right,
        "delta": delta,
        "environments_improved": int(
            sum(
                1
                for name in names
                if (real[name][field] > control[name][field]) == (direction == "higher_is_better")
                and real[name][field] != control[name][field]
            )
        ),
        "environments": len(names),
        "real_environments_safe_and_non_degenerate": int(
            sum(1 for name in names if real[name]["safe_and_non_degenerate"])
        ),
        "control_environments_safe_and_non_degenerate": int(
            sum(1 for name in names if control[name]["safe_and_non_degenerate"])
        ),
        "discriminates": discriminates,
        "label": label,
    }


def run_controls() -> int:
    """The ten required negative controls, each breaking exactly one thing."""
    started = time.monotonic()
    deployments = load_deployments()
    floor = load_floor()
    cells, _design = load_cells()
    rate = float(PRACTICAL_REVIEW_BUDGET)
    accuracy = float(REVIEWER_ACCURACY[PRIMARY_SCENARIO])

    def slice_env(
        policy: str, budget: float, scenario: str, documents: int
    ) -> dict[str, dict[str, Any]]:
        block = slice_cells(
            cells, policy, review_budget=budget, scenario=scenario, documents=documents
        )
        return per_environment(block, floor, PRIMARY_EPSILON)

    boundary_real = slice_env(P3_BOUNDARY, rate, PRIMARY_SCENARIO, PRACTICAL_DOC_BUDGET)
    auto_only = slice_env(P1_AUTO_ONLY, 0.0, H0_NO_REVIEW, PRACTICAL_DOC_BUDGET)

    shuffled = _run_configuration(
        deployments,
        floor,
        rule=CUT_EMPIRICAL,
        routing=ROUTE_SHUFFLED,
        rate=rate,
        accuracy=accuracy,
        documents=PRACTICAL_DOC_BUDGET,
        shuffle_routing=13,
    )
    permuted = _run_configuration(
        deployments,
        floor,
        rule=CUT_EMPIRICAL,
        routing=ROUTE_BOUNDARY,
        rate=rate,
        accuracy=accuracy,
        documents=PRACTICAL_DOC_BUDGET,
        permute_labels=29,
    )
    one_document = _run_configuration(
        deployments,
        floor,
        rule=CUT_EMPIRICAL,
        routing=ROUTE_BOUNDARY,
        rate=rate,
        accuracy=accuracy,
        documents=CONTROL_DOC_BUDGET,
    )

    controls = {
        "control_1_random_review": _control(
            "random review",
            "does WHERE review is spent matter, or only how much?",
            boundary_real,
            slice_env(B4_RANDOM_REVIEW, rate, PRIMARY_SCENARIO, PRACTICAL_DOC_BUDGET),
            "risk_controlled_automatic_recall",
        ),
        "control_2_shuffled_review_ranking": _control(
            "shuffled routing score",
            "does the routing need the ranking, or would any band of the same size do?",
            boundary_real,
            shuffled,
            "risk_controlled_automatic_recall",
        ),
        "control_3_label_permutation": _control(
            "permuted certification labels",
            "do the purchased labels carry information the boundary uses?",
            boundary_real,
            permuted,
            "risk_controlled_automatic_recall",
        ),
        "control_4_reviewer_accuracy_degradation": _control(
            "coin-flip reviewer",
            "does the benefit need a competent reviewer?",
            boundary_real,
            slice_env(P3_BOUNDARY, rate, H4_COIN_FLIP, PRACTICAL_DOC_BUDGET),
            "total_repair_recall",
        ),
        "control_5_review_labels_unavailable_to_the_current_decision": {
            "question": (
                "can a reviewed label improve the decision that routed that same candidate?"
            ),
            "design": (
                "the static primary analysis never reuses a reviewed label at all, and the "
                "sequential scenario admits one only from the following batch. Both are asserted "
                "in the leakage suite against the source rather than claimed here."
            ),
            "static_regime": L0_NO_REUSE,
            "sequential_artifact": _relative(SEQUENTIAL_DEPLOYMENT),
            "boundary_moved_under_reuse": int(
                cc_read_json(SEQUENTIAL_DEPLOYMENT)["summary"][L1_REUSE_AFTER_REVIEW][
                    "environments_where_the_boundary_moved"
                ]
            ),
            "boundary_moved_without_reuse": int(
                cc_read_json(SEQUENTIAL_DEPLOYMENT)["summary"][L0_NO_REUSE][
                    "environments_where_the_boundary_moved"
                ]
            ),
            "discriminates": bool(
                int(
                    cc_read_json(SEQUENTIAL_DEPLOYMENT)["summary"][L0_NO_REUSE][
                        "environments_where_the_boundary_moved"
                    ]
                )
                == 0
            ),
        },
        "control_6_no_review_baseline": _control(
            "no review at all",
            "does buying review change the automatic side at all?",
            boundary_real,
            auto_only,
            "risk_controlled_automatic_recall",
        ),
        "control_7_preserve_all": _control(
            "preserve everything",
            "is the deployed policy doing more than refusing?",
            boundary_real,
            slice_env(P0_PRESERVE, 0.0, H0_NO_REVIEW, PRACTICAL_DOC_BUDGET),
            "total_repair_recall",
        ),
        "control_8_ideal_review_upper_bound": _control(
            "ideal reviewer",
            "how much of the ceiling does a realistic reviewer keep?",
            slice_env(P3_BOUNDARY, rate, H1_IDEAL, PRACTICAL_DOC_BUDGET),
            boundary_real,
            "total_repair_recall",
        ),
        "control_9_one_document_target_supervision": _control(
            "one target document",
            "is the document budget load-bearing, or would a single page do?",
            boundary_real,
            one_document,
            "risk_controlled_automatic_recall",
        ),
        "control_10_zero_target_label_deployment": _control(
            "no target label at all",
            "does target supervision buy anything over the frozen source cut?",
            boundary_real,
            slice_env(B1_SOURCE, 0.0, H0_NO_REVIEW, 0),
            "risk_controlled_automatic_recall",
        ),
    }

    positive = sorted(name for name, record in controls.items() if record.get("discriminates"))
    negative = sorted(name for name, record in controls.items() if not record.get("discriminates"))
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_envelope("control_results"),
            "reference": {
                "policy": P3_BOUNDARY,
                "review_budget": rate,
                "reviewer_scenario": PRIMARY_SCENARIO,
                "document_budget": PRACTICAL_DOC_BUDGET,
                "epsilon": float(PRIMARY_EPSILON),
            },
            "controls": controls,
            "controls_that_discriminate": positive,
            "controls_that_do_not": negative,
            "reading": (
                "a control that does NOT discriminate is a finding about the method, not a bug in "
                "the control. Control 8 is deliberately inverted: it asks how much of the ideal "
                "reviewer's ceiling a realistic reviewer keeps, so a small gap is the good result."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"controls: {len(positive)} of {len(controls)} discriminate -> {_relative(CONTROL_RESULTS)}"
    )
    return 0


# ---------------------------------------- section 49 criterion 6: nested selection
#
# The configuration is chosen on nine environments and read on the tenth, ten times. A
# configuration that only wins because it was picked after looking at the environment it is
# scored on has not been shown to generalise, and the selection frequency says how close the
# choice was rather than presenting the modal winner as if it were the only one.

SELECTION_DOC_BUDGETS = tuple(b for b in DOC_BUDGETS if b <= PRACTICAL_DOC_BUDGET)
SELECTION_REVIEW_BUDGETS = tuple(r for r in REVIEW_BUDGETS if r <= PRACTICAL_REVIEW_BUDGET)


def _configurations() -> list[tuple[str, float, int]]:
    out: list[tuple[str, float, int]] = []
    for policy in SELECTABLE_POLICIES:
        rates = SELECTION_REVIEW_BUDGETS if POLICY_RULE[policy][2] else (0.0,)
        for rate in rates:
            for documents in SELECTION_DOC_BUDGETS:
                out.append((policy, float(rate), int(documents)))
    return sorted(out)


def _configuration_table(
    cells: pd.DataFrame, floor: float
) -> dict[tuple[str, float, int], dict[str, dict[str, Any]]]:
    """Every selectable configuration's per-environment endpoint at the primary tolerance.

    The reviewer scenario is NOT a selection axis. A deployment does not choose how accurate its
    reviewers are; it inherits that. Selection runs at the declared primary scenario and the
    dependence on reviewer quality is a separate criterion.
    """
    table: dict[tuple[str, float, int], dict[str, dict[str, Any]]] = {}
    for policy, rate, documents in _configurations():
        block = slice_cells(
            cells,
            policy,
            review_budget=rate,
            scenario=H0_NO_REVIEW if rate == 0.0 else PRIMARY_SCENARIO,
            documents=documents,
        )
        if block.empty:
            continue
        table[(policy, rate, documents)] = per_environment(block, floor, PRIMARY_EPSILON)
    return table


def _configuration_key(configuration: tuple[str, float, int]) -> str:
    policy, rate, documents = configuration
    return f"{policy}|{rate:.2f}|{documents}"


def run_select() -> int:
    """Leave-one-environment-out selection over the declared configuration space."""
    started = time.monotonic()
    cells, _design = load_cells()
    floor = load_floor()
    table = _configuration_table(cells, floor)
    environments = sorted({name for record in table.values() for name in record})

    folds: list[dict[str, Any]] = []
    for held_out in environments:
        best: tuple[str, float, int] | None = None
        best_score = -1.0
        for configuration, per_env in sorted(
            table.items(), key=lambda item: _configuration_key(item[0])
        ):
            training = [name for name in environments if name != held_out and name in per_env]
            if not training:
                continue
            score = _mean([per_env[name]["risk_controlled_automatic_recall"] for name in training])
            if math.isnan(score):
                continue
            if score > best_score:
                best, best_score = configuration, float(score)
        if best is None:
            continue
        record = table[best].get(held_out)
        folds.append(
            {
                "held_out": held_out,
                "selected": _configuration_key(best),
                "policy": best[0],
                "review_budget": best[1],
                "document_budget": best[2],
                "training_score": best_score,
                "held_out_risk_controlled_automatic_recall": (
                    None if record is None else record["risk_controlled_automatic_recall"]
                ),
                "held_out_total_repair_recall": (
                    None if record is None else record["total_repair_recall"]
                ),
                "held_out_automatic_harm": None if record is None else record["automatic_harm"],
                "held_out_holds_bound": None if record is None else record["holds_on_every_draw"],
                "held_out_safe_and_non_degenerate": (
                    None if record is None else record["safe_and_non_degenerate"]
                ),
            }
        )

    frequency: dict[str, int] = {}
    for fold in folds:
        frequency[fold["selected"]] = frequency.get(fold["selected"], 0) + 1
    modal = max(sorted(frequency), key=lambda key: (frequency[key], key)) if frequency else None
    pooled = {
        _configuration_key(configuration): _mean(
            [per_env[name]["risk_controlled_automatic_recall"] for name in sorted(per_env)]
        )
        for configuration, per_env in table.items()
    }
    pooled_best = max(sorted(pooled), key=lambda key: (pooled[key], key)) if pooled else None

    _write_json_once(
        NESTED_SELECTION,
        {
            **_envelope("nested_selection"),
            "objective": (
                "the mean risk-controlled automatic repair recall on the NINE training "
                "environments, at the primary tolerance. Risk-controlled means the harm "
                "constraint is inside the endpoint, so a configuration cannot be selected for "
                "breaking the bound it was given."
            ),
            "search_space": {
                "policies": list(SELECTABLE_POLICIES),
                "review_budgets": list(SELECTION_REVIEW_BUDGETS),
                "document_budgets": list(SELECTION_DOC_BUDGETS),
                "reviewer_scenario": PRIMARY_SCENARIO,
                "why_the_reviewer_is_not_a_selection_axis": (
                    "a deployment inherits its reviewers' accuracy; it does not choose it. "
                    "Selecting over it would be selecting an assumption."
                ),
                "size": len(table),
            },
            "folds": folds,
            "selection_frequency": dict(sorted(frequency.items())),
            "distinct_selections": len(frequency),
            "modal_selection": modal,
            "modal_frequency": int(frequency.get(modal, 0)) if modal else 0,
            "selection_is_stable": bool(
                modal is not None and frequency.get(modal, 0) >= MAJORITY_REQUIRED
            ),
            "mean_held_out_risk_controlled_automatic_recall": _mean(
                [
                    float(fold["held_out_risk_controlled_automatic_recall"])
                    for fold in folds
                    if fold["held_out_risk_controlled_automatic_recall"] is not None
                ]
            ),
            "pooled_best": pooled_best,
            "pooled_best_value": None if pooled_best is None else pooled[pooled_best],
            "pooled_is_not_unbiased": (
                "the pooled winner is reported because the difference between it and the "
                "held-out mean is the selection cost. It is not an unbiased estimate of "
                "anything and no criterion reads it."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"select: {len(table)} configurations, {len(folds)} folds, modal {modal} on "
        f"{frequency.get(modal, 0) if modal else 0}, stable="
        f"{bool(modal is not None and frequency.get(modal, 0) >= MAJORITY_REQUIRED)} -> "
        f"{_relative(NESTED_SELECTION)}"
    )
    return 0


def selected_configuration() -> tuple[str, float, int]:
    record = cc_read_json(NESTED_SELECTION)
    policy, rate, documents = str(record["modal_selection"]).split("|")
    return policy, float(rate), int(documents)


# ------------------------------------------------------------------ section 35: statistics
#
# The resampling unit is the document. Comparisons on the same documents are paired. The family
# is stated and Holm-corrected. All three are the project's standing conventions, imported here
# rather than re-decided.

CELL_FIELDS = ("environment", "policy", "review_budget", "reviewer_scenario", "documents", "draw")
RECALL = s13.RECALL
HARM = s13.HARM


def indexed_counts() -> pd.DataFrame:
    counts = pd.read_parquet(DOCUMENT_COUNTS)
    parts = counts["cell"].astype(str).str.split("|", expand=True)
    parts.columns = list(CELL_FIELDS)
    return pd.concat([counts.reset_index(drop=True), parts], axis=1)


def count_blocks(
    indexed: pd.DataFrame, environment: str, policy: str, rate: float, scenario: str, documents: int
) -> tuple[Any, Any]:
    """One cell's per-document counts as two blocks: the automatic side and both routes.

    Both carry the SAME accepted and harmful vectors -- the automatically applied set -- because
    section 3 keeps reviewed edits out of the automatic-harm denominator. They differ only in
    which beneficial count they carry, so `statistic_draws` reads risk-controlled automatic recall
    off the first and risk-controlled total recall off the second, both gated on automatic harm.
    """
    block = indexed[
        (indexed["environment"] == environment)
        & (indexed["policy"] == policy)
        & (indexed["review_budget"] == f"{rate:.2f}")
        & (indexed["reviewer_scenario"] == scenario)
        & (indexed["documents"] == str(documents))
    ]
    if block.empty:
        raise PhaseError(
            f"no deployment counts for {environment}/{policy}/{rate}/{scenario}/{documents}"
        )
    names = tuple(sorted({str(value) for value in block["document_id"].tolist()}))
    draws = sorted({str(value) for value in block["draw"].tolist()})
    position = {name: index for index, name in enumerate(names)}
    row_of = {draw: index for index, draw in enumerate(draws)}
    shape = (len(draws), len(names))
    rows = np.array([row_of[str(v)] for v in block["draw"].tolist()], dtype=int)
    columns = np.array([position[str(v)] for v in block["document_id"].tolist()], dtype=int)
    matrices: dict[str, np.ndarray] = {}
    for field in (
        "auto_accepted",
        "auto_harmful",
        "auto_beneficial",
        "applied_beneficial",
        "beneficial_total",
    ):
        matrix = np.zeros(shape, dtype=float)
        matrix[rows, columns] = block[field].to_numpy(dtype=float)
        matrices[field] = matrix
    automatic = s13.CountBlock(
        documents=names,
        accepted=matrices["auto_accepted"],
        harmful=matrices["auto_harmful"],
        beneficial=matrices["auto_beneficial"],
        total=matrices["beneficial_total"],
    )
    combined = s13.CountBlock(
        documents=names,
        accepted=matrices["auto_accepted"],
        harmful=matrices["auto_harmful"],
        beneficial=matrices["applied_beneficial"],
        total=matrices["beneficial_total"],
    )
    return automatic, combined


def _scenario_of(rate: float) -> str:
    return H0_NO_REVIEW if rate == 0.0 else PRIMARY_SCENARIO


def run_stats() -> int:
    """The pre-registered tests: the selected regime against its declared comparators."""
    started = time.monotonic()
    _cells, design = load_cells()
    indexed = indexed_counts()
    policy, rate, documents = selected_configuration()
    environments = list(design["environments"])

    def pair(
        left: tuple[str, float, int], right: tuple[str, float, int], combined: bool
    ) -> list[Any]:
        out = []
        for name in environments:
            a = count_blocks(indexed, name, left[0], left[1], _scenario_of(left[1]), left[2])
            b = count_blocks(indexed, name, right[0], right[1], _scenario_of(right[1]), right[2])
            out.append((a[1] if combined else a[0], b[1] if combined else b[0]))
        return out

    selected = (policy, rate, documents)
    comparator = (P1_AUTO_ONLY, 0.0, documents)
    automatic = pair(selected, comparator, combined=False)
    total = pair(selected, comparator, combined=True)

    primary_recall = stratified_delta(automatic, RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED)
    primary_total = stratified_delta(total, RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED)
    primary_harm = stratified_delta(automatic, HARM, PRIMARY_EPSILON, BOOTSTRAP_SEED)

    family: dict[str, dict[str, float]] = {}
    for name, (left, right) in zip(environments, automatic, strict=True):
        family[f"{name}|risk_controlled_automatic_recall"] = s13.paired_delta(
            left, right, RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED
        )
        family[f"{name}|automatic_harm"] = s13.paired_delta(
            left, right, HARM, PRIMARY_EPSILON, BOOTSTRAP_SEED
        )
    corrected = holm(family)
    surviving = sorted(key for key, value in corrected.items() if value["survives_holm"])

    secondary = {
        "Q2_boundary_versus_random_review": {
            "left": _configuration_key((P3_BOUNDARY, PRACTICAL_REVIEW_BUDGET, documents)),
            "right": _configuration_key((B4_RANDOM_REVIEW, PRACTICAL_REVIEW_BUDGET, documents)),
            "risk_controlled_automatic_recall": stratified_delta(
                pair(
                    (P3_BOUNDARY, PRACTICAL_REVIEW_BUDGET, documents),
                    (B4_RANDOM_REVIEW, PRACTICAL_REVIEW_BUDGET, documents),
                    combined=False,
                ),
                RECALL,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            ),
            "automatic_harm": stratified_delta(
                pair(
                    (P3_BOUNDARY, PRACTICAL_REVIEW_BUDGET, documents),
                    (B4_RANDOM_REVIEW, PRACTICAL_REVIEW_BUDGET, documents),
                    combined=False,
                ),
                HARM,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            ),
        },
        "Q2_high_risk_versus_random_review": {
            "left": _configuration_key((B6_HIGH_RISK_REVIEW, PRACTICAL_REVIEW_BUDGET, documents)),
            "right": _configuration_key((B4_RANDOM_REVIEW, PRACTICAL_REVIEW_BUDGET, documents)),
            "risk_controlled_automatic_recall": stratified_delta(
                pair(
                    (B6_HIGH_RISK_REVIEW, PRACTICAL_REVIEW_BUDGET, documents),
                    (B4_RANDOM_REVIEW, PRACTICAL_REVIEW_BUDGET, documents),
                    combined=False,
                ),
                RECALL,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            ),
        },
        "Q5_selected_versus_sgv15_certified": {
            "left": _configuration_key(selected),
            "right": _configuration_key((B2_SGV15, 0.0, documents)),
            "risk_controlled_automatic_recall": stratified_delta(
                pair(selected, (B2_SGV15, 0.0, documents), combined=False),
                RECALL,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            ),
            "automatic_harm": stratified_delta(
                pair(selected, (B2_SGV15, 0.0, documents), combined=False),
                HARM,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            ),
            "total_repair_recall": stratified_delta(
                pair(selected, (B2_SGV15, 0.0, documents), combined=True),
                RECALL,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            ),
        },
        "Q5_selected_versus_sgv15b_certified": {
            "left": _configuration_key(selected),
            "right": _configuration_key((B3_SGV15B, 0.0, documents)),
            "risk_controlled_automatic_recall": stratified_delta(
                pair(selected, (B3_SGV15B, 0.0, documents), combined=False),
                RECALL,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            ),
            "total_repair_recall": stratified_delta(
                pair(selected, (B3_SGV15B, 0.0, documents), combined=True),
                RECALL,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            ),
        },
    }

    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_envelope("statistical_tests"),
            "scientific_status": (
                "DEVELOPMENT. These are the environments SGV14 already exposed; a surviving "
                "p-value here is not external confirmation."
            ),
            "selected_configuration": {
                "policy": policy,
                "review_budget": rate,
                "document_budget": documents,
                "reviewer_scenario": _scenario_of(rate),
            },
            "primary_comparator": {
                "configuration": _configuration_key(comparator),
                "why": (
                    "the same boundary rule and the same purchased documents with the review "
                    "budget set to zero. The paired difference therefore isolates what buying "
                    "review changed, which is question Q1."
                ),
            },
            "resampling_unit": "document",
            "pairing": "the same resampled documents are given to both arms in every resample",
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_risk_controlled_automatic_recall": primary_recall,
            "primary_risk_controlled_total_recall": primary_total,
            "primary_automatic_harm": primary_harm,
            "family": corrected,
            "family_size": len(family),
            "family_definition": (
                "one test per environment for each of the two primary statistics, which is the "
                "family shape SGV13, SGV14, SGV15 and SGV15b all used."
            ),
            "multiplicity": "Holm-Bonferroni at 0.05",
            "surviving_after_holm": surviving,
            "count_surviving": len(surviving),
            "secondary": secondary,
            "secondary_are_not_holm_corrected_into_the_primary_family": (
                "they answer different declared questions and are reported with their own "
                "intervals; the primary family's size was fixed before any of them was computed."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"stats: automatic recall {primary_recall['delta']:+.4f} CI "
        f"[{primary_recall['ci_lower']:+.4f}, {primary_recall['ci_upper']:+.4f}] "
        f"p={primary_recall['p_value']:.4f}; harm {primary_harm['delta']:+.4f}; "
        f"{len(surviving)}/{len(family)} survive Holm -> {_relative(STATISTICAL_TESTS)}"
    )
    return 0


# ------------------------------------------------------------------ the falsification tests
#
# Each of these would FAIL if the corresponding claim were false. They are run here, on the real
# artifacts, and again in `tests/leakage` against the source.


def run_negative() -> int:
    """Executable falsifications of the claims this stage's numbers depend on."""
    started = time.monotonic()
    cells, _design = load_cells()
    deployments = load_deployments()
    floor = load_floor()
    tests: dict[str, dict[str, Any]] = {}

    # 1. The three actions partition the evaluation block exactly, on every sampled cell.
    partition_failures = 0
    checked = 0
    for name, deployment in deployments.items():
        correctness = reviewer_correctness(deployment)
        sample = document_sample(
            deployment, PRACTICAL_DOC_BUDGET, _sample_seed(name, PRACTICAL_DOC_BUDGET, 0)
        )
        for routing in ROUTINGS:
            boundary = select_boundary(deployment, CUT_EMPIRICAL, sample, PRIMARY_EPSILON)
            tau = float(boundary["tau"])
            order = review_order(deployment, tau, routing)
            take = min(int(PRACTICAL_REVIEW_BUDGET * deployment.n_rows), int(order.size))
            reviewed = np.zeros(deployment.n_rows, dtype=bool)
            reviewed[order[:take]] = True
            auto0 = (
                (deployment.eval_scores >= tau)
                if np.isfinite(tau)
                else np.zeros(deployment.n_rows, dtype=bool)
            )
            auto = auto0 & ~reviewed
            preserved = ~auto & ~reviewed
            checked += 1
            if (
                int((auto & reviewed).sum())
                or int((auto & preserved).sum())
                or int((reviewed & preserved).sum())
            ):
                partition_failures += 1
            if int(auto.sum() + reviewed.sum() + preserved.sum()) != deployment.n_rows:
                partition_failures += 1
            # k_auto <= k_review by construction: the review band is defined relative to the
            # boundary, so no candidate can be reviewed and automatically applied at once.
            _, applied = _action_masks(deployment, tau, order, take, 1.0, correctness)
            if int((applied & ~auto & ~reviewed).sum()):
                partition_failures += 1
    tests["the_three_actions_partition_the_block"] = {
        "claim": "auto-apply, human review and preserve are disjoint and exhaustive",
        "assignments_checked": checked,
        "failures": partition_failures,
        "passes": partition_failures == 0,
    }

    # 2. A zero review budget is exactly the auto-only policy.
    left = slice_cells(cells, P3_BOUNDARY, review_budget=0.0, documents=PRACTICAL_DOC_BUDGET)
    right = slice_cells(cells, P1_AUTO_ONLY, review_budget=0.0, documents=PRACTICAL_DOC_BUDGET)
    fields = (
        "automatic_harm",
        "automatic_coverage",
        "automatic_repair_recall",
        "total_repair_recall",
    )
    worst = 0.0
    for field in fields:
        a = left.sort_values(["environment", "draw"])[field].to_numpy(dtype=float)
        b = right.sort_values(["environment", "draw"])[field].to_numpy(dtype=float)
        equal = np.isnan(a) & np.isnan(b)
        worst = max(worst, float(np.abs(np.where(equal, 0.0, a - b)).max()) if a.size else 0.0)
    tests["a_zero_review_budget_reproduces_the_auto_only_policy"] = {
        "claim": "spending no review is exactly not spending review",
        "cells": len(left),
        "max_absolute_difference": worst,
        "passes": worst == 0.0,
    }

    # 3. The ideal reviewer is an upper bound, and the ordering across accuracies is monotone.
    monotone = True
    ladder: dict[str, float] = {}
    for scenario in (H1_IDEAL, H2_HIGH, H3_MODERATE, H4_COIN_FLIP):
        block = slice_cells(
            cells,
            P3_BOUNDARY,
            review_budget=PRACTICAL_REVIEW_BUDGET,
            scenario=scenario,
            documents=PRACTICAL_DOC_BUDGET,
        )
        per_env = per_environment(block, floor, PRIMARY_EPSILON)
        ladder[scenario] = _mean([per_env[n]["total_repair_recall"] for n in sorted(per_env)])
    order = (H1_IDEAL, H2_HIGH, H3_MODERATE, H4_COIN_FLIP)
    for first, second in itertools.pairwise(order):
        if not ladder[first] >= ladder[second]:
            monotone = False
    tests["the_ideal_reviewer_is_an_upper_bound"] = {
        "claim": (
            "under common random numbers, total repair recall is monotone in reviewer accuracy"
        ),
        "ladder": ladder,
        "passes": bool(monotone),
    }

    # 4. The declared review budget is never exceeded.
    over = cells[cells["review_rate"] > cells["review_budget"] + 1e-12]
    tests["the_review_budget_is_never_exceeded"] = {
        "claim": "the realised review rate never rises above the declared budget",
        "cells": len(cells),
        "violations": len(over),
        "passes": len(over) == 0,
    }

    # 5. Reviewed edits are absent from the automatic-harm denominator.
    recomputed = 0
    denominator_failures = 0
    for name, deployment in deployments.items():
        correctness = reviewer_correctness(deployment)
        sample = document_sample(
            deployment, PRACTICAL_DOC_BUDGET, _sample_seed(name, PRACTICAL_DOC_BUDGET, 3)
        )
        boundary = select_boundary(deployment, CUT_EMPIRICAL, sample, PRIMARY_EPSILON)
        tau = float(boundary["tau"])
        order = review_order(deployment, tau, ROUTE_BOUNDARY)
        take = min(int(PRACTICAL_REVIEW_BUDGET * deployment.n_rows), int(order.size))
        auto, _applied = _action_masks(deployment, tau, order, take, 0.98, correctness)
        reviewed = np.zeros(deployment.n_rows, dtype=bool)
        reviewed[order[:take]] = True
        table = _measure(
            deployment,
            tau,
            order,
            (int(PRACTICAL_REVIEW_BUDGET * deployment.n_rows),),
            (0.98,),
            correctness,
            floor,
        )
        record = table[(int(PRACTICAL_REVIEW_BUDGET * deployment.n_rows), 0.98)]
        direct = (
            float(int((auto & deployment.eval_harm).sum()) / int(auto.sum()))
            if int(auto.sum())
            else float("nan")
        )
        recomputed += 1
        reported = float(record["automatic_harm"])
        if not (math.isnan(direct) and math.isnan(reported)) and direct != reported:
            denominator_failures += 1
        if int((auto & reviewed).sum()):
            denominator_failures += 1
    tests["reviewed_edits_are_absent_from_the_automatic_harm_denominator"] = {
        "claim": "automatic harm counts only edits applied without review",
        "environments_recomputed": recomputed,
        "failures": denominator_failures,
        "passes": denominator_failures == 0,
    }

    # 6-7. Evaluation labels can move neither the routing nor the boundary.
    routing_moved = 0
    boundary_moved = 0
    for name, deployment in deployments.items():
        generator = np.random.default_rng(_stable_seed("sgv-dt1-labelperm", name))
        permuted = Deployment(
            **{
                **{
                    field: getattr(deployment, field)
                    for field in deployment.__slots__
                    if field not in ("eval_harm", "eval_beneficial")
                },
                "eval_harm": generator.permutation(deployment.eval_harm),
                "eval_beneficial": generator.permutation(deployment.eval_beneficial),
            }
        )
        sample = document_sample(
            deployment, PRACTICAL_DOC_BUDGET, _sample_seed(name, PRACTICAL_DOC_BUDGET, 0)
        )
        before = select_boundary(deployment, CUT_EMPIRICAL, sample, PRIMARY_EPSILON)
        after = select_boundary(permuted, CUT_EMPIRICAL, sample, PRIMARY_EPSILON)
        if float(before["tau"]) != float(after["tau"]):
            boundary_moved += 1
        for routing in ROUTINGS:
            if not np.array_equal(
                review_order(deployment, float(before["tau"]), routing),
                review_order(permuted, float(after["tau"]), routing),
            ):
                routing_moved += 1
    tests["permuting_evaluation_labels_moves_neither_routing_nor_boundary"] = {
        "claim": "no evaluation outcome label reaches the routing or the boundary",
        "environments": len(deployments),
        "boundaries_moved": boundary_moved,
        "routings_moved": routing_moved,
        "passes": boundary_moved == 0 and routing_moved == 0,
    }

    # 8. The reviewer is deterministic under its frozen seed.
    unstable = sum(
        1
        for deployment in deployments.values()
        if not np.array_equal(reviewer_correctness(deployment), reviewer_correctness(deployment))
    )
    tests["the_reviewer_simulation_is_deterministic"] = {
        "claim": "reviewer correctness is a frozen function of the environment and the seed",
        "environments": len(deployments),
        "unstable": unstable,
        "passes": unstable == 0,
    }

    # 9-10. The dominance relation is irreflexive, and equal points survive together.
    frontier = cc_read_json(PARETO_FRONTIER)["frontiers"]["automatic_repair_recall"]
    dominance = cc_read_json(DOMINANCE_MATRIX)["by_objective"]["automatic_repair_recall"]
    reflexive = sum(1 for key, sources in dominance.items() if key in sources)
    sample_point = {
        "mean_automatic_harm": 0.05,
        "mean_review_rate": 0.1,
        "document_budget": 10,
        "mean_risk_controlled_automatic_recall": 0.3,
        "epsilon": PRIMARY_EPSILON,
    }
    twin = _vector(sample_point, PARETO_MAXIMISE)
    tests["the_dominance_relation_is_well_formed"] = {
        "claim": "no point dominates itself and two identical points are mutually non-dominated",
        "points": int(cc_read_json(DOMINANCE_MATRIX)["points"]),
        "non_dominated": int(frontier["non_dominated_count"]),
        "self_dominating": reflexive,
        "identical_points_dominate_each_other": bool(_dominates(twin, twin)),
        "passes": bool(reflexive == 0 and not _dominates(twin, twin)),
    }

    # 11. A policy that applies nothing cannot dominate one that applies something on safety.
    refusing = {
        "mean_automatic_harm": float("nan"),
        "mean_review_rate": 0.0,
        "document_budget": 0,
        "mean_risk_controlled_automatic_recall": 0.0,
        "epsilon": PRIMARY_EPSILON,
    }
    applying = {
        "mean_automatic_harm": 0.0,
        "mean_review_rate": 0.0,
        "document_budget": 0,
        "mean_risk_controlled_automatic_recall": 0.2,
        "epsilon": PRIMARY_EPSILON,
    }
    tests["refusing_everything_cannot_dominate_on_safety"] = {
        "claim": "an undefined harm rate is placed at the tolerance, not at zero",
        "refusing_dominates_applying": bool(
            _dominates(_vector(refusing, PARETO_MAXIMISE), _vector(applying, PARETO_MAXIMISE))
        ),
        "passes": not bool(
            _dominates(_vector(refusing, PARETO_MAXIMISE), _vector(applying, PARETO_MAXIMISE))
        ),
    }

    # 12. The certified baselines' in-sample limit never exceeds the tolerance they were given.
    certified = cells[
        (cells["safety_label"] == SAFETY_CERTIFIED)
        & (cells["cut_feasible"])
        & (cells["certified_risk_upper_bound"].notna())
    ]
    breached = certified[certified["certified_risk_upper_bound"] > certified["epsilon"] + 1e-12]
    tests["a_certified_boundary_never_exceeds_its_own_limit"] = {
        "claim": "a boundary called certified carried an upper limit at or below its tolerance",
        "cells": len(certified),
        "breaches": len(breached),
        "passes": len(breached) == 0,
    }

    failed = sorted(name for name, record in tests.items() if not record["passes"])
    if failed:
        raise PhaseError(f"falsification tests failed: {failed}")
    _write_json_once(
        NEGATIVE_TESTS,
        {
            **_envelope("negative_tests"),
            "rule": "each of these would FAIL if the claim it names were false",
            "tests": tests,
            "count": len(tests),
            "all_passed": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"negative: {len(tests)} falsification tests, all passed -> {_relative(NEGATIVE_TESTS)}")
    return 0


# ------------------------------------------ sections 28-30, 48, 49: the development decision


def _regime_of(rate: float) -> str:
    for name, bound in REGIMES:
        if rate <= bound:
            return name
    return REGIME_D


def _breadth(per_env: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = sorted(per_env)
    safe = sorted(name for name in names if per_env[name]["holds_on_every_draw"])
    non_degenerate = sorted(name for name in names if per_env[name]["non_degenerate"])
    both = sorted(name for name in names if per_env[name]["safe_and_non_degenerate"])
    return {
        "environments_safe": len(safe),
        "environments_safe_names": safe,
        "environments_non_degenerate": len(non_degenerate),
        "environments_non_degenerate_names": non_degenerate,
        "environments_safe_and_non_degenerate": len(both),
        "environments_safe_and_non_degenerate_names": both,
        "mean_automatic_harm": _mean([per_env[n]["automatic_harm"] for n in names]),
        "mean_automatic_coverage": _mean([per_env[n]["automatic_coverage"] for n in names]),
        "mean_automatic_repair_recall": _mean(
            [per_env[n]["automatic_repair_recall"] for n in names]
        ),
        "mean_risk_controlled_automatic_recall": _mean(
            [per_env[n]["risk_controlled_automatic_recall"] for n in names]
        ),
        "mean_total_repair_recall": _mean([per_env[n]["total_repair_recall"] for n in names]),
        "mean_review_rate": _mean([per_env[n]["review_rate"] for n in names]),
        "mean_reviewed_candidates": _mean([per_env[n]["reviewed"] for n in names]),
        "mean_certification_labels": _mean([per_env[n]["certification_labels"] for n in names]),
        "mean_supervision_labels": _mean([per_env[n]["supervision_labels"] for n in names]),
    }


def _scan(
    cells: pd.DataFrame,
    floor: float,
    *,
    epsilon: float,
    max_review: float,
    max_documents: int,
    scenario: str,
    pool: Sequence[str] = SELECTABLE_POLICIES,
) -> dict[str, Any]:
    """The widest breadth any declared policy reaches inside one corner of the design."""
    best: dict[str, Any] | None = None
    for policy in pool:
        rates = REVIEW_BUDGETS if POLICY_RULE[policy][2] else (0.0,)
        for rate in rates:
            if rate > max_review:
                continue
            for documents in DOC_BUDGETS:
                if documents > max_documents:
                    continue
                block = slice_cells(
                    cells,
                    policy,
                    epsilon=epsilon,
                    review_budget=rate,
                    scenario=H0_NO_REVIEW if rate == 0.0 else scenario,
                    documents=documents,
                )
                if block.empty:
                    continue
                record = _breadth(per_environment(block, floor, epsilon))
                record.update(
                    {
                        "policy": policy,
                        "review_budget": float(rate),
                        "document_budget": int(documents),
                        "epsilon": float(epsilon),
                        "reviewer_scenario": H0_NO_REVIEW if rate == 0.0 else scenario,
                        "regime": _regime_of(float(rate)),
                    }
                )
                key = (
                    record["environments_safe_and_non_degenerate"],
                    record["mean_risk_controlled_automatic_recall"],
                )
                if best is None or key > (
                    best["environments_safe_and_non_degenerate"],
                    best["mean_risk_controlled_automatic_recall"],
                ):
                    best = record
    return best or {}


def run_decide() -> int:
    """The seven pre-registered criteria, the five questions, the outcome and the gate."""
    started = time.monotonic()
    cells, design = load_cells()
    floor = load_floor()
    policy, rate, documents = selected_configuration()
    selection = cc_read_json(NESTED_SELECTION)
    controls = cc_read_json(CONTROL_RESULTS)
    stats = cc_read_json(STATISTICAL_TESTS)
    sensitivity = cc_read_json(REVIEW_ACCURACY_SENSITIVITY)

    def measured(scenario: str) -> dict[str, Any]:
        block = slice_cells(
            cells,
            policy,
            review_budget=rate,
            scenario=H0_NO_REVIEW if rate == 0.0 else scenario,
            documents=documents,
        )
        return _breadth(per_environment(block, floor, PRIMARY_EPSILON))

    primary = measured(PRIMARY_SCENARIO)
    degraded = measured(NOT_PERFECT_REVIEWER_SCENARIO)

    criteria = {
        "1_safety": {
            "requirement": (
                f"automatic harm at or below {PRIMARY_EPSILON} on every draw, on at least "
                f"{BREADTH_REQUIRED} of {ENVIRONMENT_COUNT}"
            ),
            "observed": primary["environments_safe"],
            "environments": primary["environments_safe_names"],
            "required": BREADTH_REQUIRED,
            "met": bool(primary["environments_safe"] >= BREADTH_REQUIRED),
        },
        "2_non_degenerate_automation": {
            "requirement": (
                f"automatic coverage at or above the inherited floor {floor}, on at least "
                f"{BREADTH_REQUIRED} of {ENVIRONMENT_COUNT}"
            ),
            "observed": primary["environments_non_degenerate"],
            "environments": primary["environments_non_degenerate_names"],
            "required": BREADTH_REQUIRED,
            "coverage_floor": floor,
            "met": bool(primary["environments_non_degenerate"] >= BREADTH_REQUIRED),
        },
        "3_review_budget": {
            "requirement": f"review rate at or below {PRACTICAL_REVIEW_BUDGET}",
            "observed": primary["mean_review_rate"],
            "declared_budget": rate,
            "met": bool(rate <= PRACTICAL_REVIEW_BUDGET),
        },
        "4_realistic_supervision": {
            "requirement": f"target-document budget at or below {PRACTICAL_DOC_BUDGET}",
            "observed": documents,
            "mean_certification_labels": primary["mean_certification_labels"],
            "mean_supervision_labels": primary["mean_supervision_labels"],
            "met": bool(documents <= PRACTICAL_DOC_BUDGET),
        },
        "5_not_dependent_on_perfect_review": {
            "requirement": (
                f"criteria 1-3 also hold under {NOT_PERFECT_REVIEWER_SCENARIO}, accuracy "
                f"{REVIEWER_ACCURACY[NOT_PERFECT_REVIEWER_SCENARIO]}"
            ),
            "environments_safe_under_a_moderate_reviewer": degraded["environments_safe"],
            "environments_safe_and_non_degenerate_under_a_moderate_reviewer": degraded[
                "environments_safe_and_non_degenerate"
            ],
            "total_repair_recall_lost": float(
                primary["mean_total_repair_recall"] - degraded["mean_total_repair_recall"]
            ),
            "met": bool(
                degraded["environments_safe"] >= BREADTH_REQUIRED
                and degraded["environments_non_degenerate"] >= BREADTH_REQUIRED
            ),
        },
        "6_survives_nested_selection": {
            "requirement": (
                "the leave-one-environment-out selection picks it on at least "
                f"{MAJORITY_REQUIRED} of {ENVIRONMENT_COUNT} folds"
            ),
            "observed": int(selection["modal_frequency"]),
            "required": MAJORITY_REQUIRED,
            "distinct_selections": int(selection["distinct_selections"]),
            "mean_held_out_risk_controlled_automatic_recall": selection[
                "mean_held_out_risk_controlled_automatic_recall"
            ],
            "met": bool(selection["selection_is_stable"]),
        },
        "7_fixed_complete_policy": {
            "requirement": (
                "one frozen tuple of policy, routing, review budget, document budget and "
                "reviewer scenario"
            ),
            "policy": policy,
            "routing": POLICY_RULE[policy][1],
            "boundary_rule": POLICY_RULE[policy][0],
            "review_budget": rate,
            "document_budget": documents,
            "reviewer_scenario": PRIMARY_SCENARIO,
            "met": True,
        },
    }
    met = sorted(name for name, record in criteria.items() if record["met"])

    corners = {
        "primary_epsilon_moderate_review": _scan(
            cells,
            floor,
            epsilon=PRIMARY_EPSILON,
            max_review=PRACTICAL_REVIEW_BUDGET,
            max_documents=PRACTICAL_DOC_BUDGET,
            scenario=PRIMARY_SCENARIO,
        ),
        "primary_epsilon_review_heavy": _scan(
            cells,
            floor,
            epsilon=PRIMARY_EPSILON,
            max_review=max(REVIEW_BUDGETS),
            max_documents=PRACTICAL_DOC_BUDGET,
            scenario=PRIMARY_SCENARIO,
        ),
        "primary_epsilon_no_review": _scan(
            cells,
            floor,
            epsilon=PRIMARY_EPSILON,
            max_review=0.0,
            max_documents=PRACTICAL_DOC_BUDGET,
            scenario=H0_NO_REVIEW,
        ),
        "wider_tolerance_moderate_review": _scan(
            cells,
            floor,
            epsilon=max(EPSILONS),
            max_review=PRACTICAL_REVIEW_BUDGET,
            max_documents=PRACTICAL_DOC_BUDGET,
            scenario=PRIMARY_SCENARIO,
        ),
        "primary_epsilon_moderate_review_beyond_the_practical_budget": _scan(
            cells,
            floor,
            epsilon=PRIMARY_EPSILON,
            max_review=PRACTICAL_REVIEW_BUDGET,
            max_documents=max(DOC_BUDGETS),
            scenario=PRIMARY_SCENARIO,
        ),
    }
    # Reported, never read by a criterion. The expected-value policy estimates an outcome-rate map
    # from the revealed labels, which is a fit however small, so section 9 keeps it out of the
    # primary family and out of the selection pool. Leaving its numbers out of the record entirely
    # would hide the corner of the design where the majority IS reached at the primary tolerance.
    secondary_diagnostic = {
        "widest_breadth_including_the_secondary_policy": _scan(
            cells,
            floor,
            epsilon=PRIMARY_EPSILON,
            max_review=max(REVIEW_BUDGETS),
            max_documents=PRACTICAL_DOC_BUDGET,
            scenario=PRIMARY_SCENARIO,
            pool=(*SELECTABLE_POLICIES, *SECONDARY_POLICIES),
        ),
        "status": (
            "DIAGNOSTIC. The secondary policy is excluded from every criterion and from the "
            "selection pool because its routing estimates an outcome-rate map from the revealed "
            "labels. No outcome, criterion or gate reads this field."
        ),
    }

    q1 = {
        "question": QUESTIONS["Q1"],
        "total_repair_recall_delta": stats["primary_risk_controlled_total_recall"],
        "automatic_harm_delta": stats["primary_automatic_harm"],
        "risk_controlled_automatic_recall_delta": stats["primary_risk_controlled_automatic_recall"],
        "breadth_with_review": corners["primary_epsilon_moderate_review"][
            "environments_safe_and_non_degenerate"
        ],
        "breadth_without_review": corners["primary_epsilon_no_review"][
            "environments_safe_and_non_degenerate"
        ],
        "answer": bool(
            float(stats["primary_risk_controlled_total_recall"]["delta"]) > 0.0
            and float(stats["primary_risk_controlled_total_recall"]["ci_lower"]) > 0.0
        ),
    }
    q2 = {
        "question": QUESTIONS["Q2"],
        "boundary_versus_random": stats["secondary"]["Q2_boundary_versus_random_review"],
        "high_risk_versus_random": stats["secondary"]["Q2_high_risk_versus_random_review"],
        "control_1": controls["controls"]["control_1_random_review"],
        "control_2": controls["controls"]["control_2_shuffled_review_ranking"],
        "answer": bool(
            float(
                stats["secondary"]["Q2_boundary_versus_random_review"][
                    "risk_controlled_automatic_recall"
                ]["ci_lower"]
            )
            > 0.0
        ),
    }
    q3 = {
        "question": QUESTIONS["Q3"],
        "observed": corners["primary_epsilon_moderate_review"][
            "environments_safe_and_non_degenerate"
        ],
        "majority_required": MAJORITY_REQUIRED,
        "at": {
            key: corners["primary_epsilon_moderate_review"].get(key)
            for key in ("policy", "review_budget", "document_budget")
        },
        "answer": bool(
            corners["primary_epsilon_moderate_review"]["environments_safe_and_non_degenerate"]
            >= MAJORITY_REQUIRED
        ),
    }
    supervision = cc_read_json(SUPERVISION_FRONTIER)["by_policy"][P3_BOUNDARY]
    q4 = {
        "question": QUESTIONS["Q4"],
        "control_9_one_document": controls["controls"]["control_9_one_document_target_supervision"],
        "control_10_zero_labels": controls["controls"]["control_10_zero_target_label_deployment"],
        "breadth_by_document_budget_at_the_practical_review_budget": {
            str(point["document_budget"]): point["environments_safe_and_non_degenerate"]
            for point in supervision[f"review_{PRACTICAL_REVIEW_BUDGET}"]
        },
        "smallest_review_budget_reaching_the_best_breadth": min(
            [
                float(rate_)
                for rate_ in REVIEW_BUDGETS
                if _scan(
                    cells,
                    floor,
                    epsilon=PRIMARY_EPSILON,
                    max_review=rate_,
                    max_documents=PRACTICAL_DOC_BUDGET,
                    scenario=PRIMARY_SCENARIO,
                )["environments_safe_and_non_degenerate"]
                >= corners["primary_epsilon_moderate_review"][
                    "environments_safe_and_non_degenerate"
                ]
            ]
            or [float("nan")]
        ),
        "answer": bool(
            float(controls["controls"]["control_9_one_document_target_supervision"]["delta"]) > 0.0
        ),
    }
    q5 = {
        "question": QUESTIONS["Q5"],
        "versus_sgv15_certified": stats["secondary"]["Q5_selected_versus_sgv15_certified"],
        "versus_sgv15b_certified": stats["secondary"]["Q5_selected_versus_sgv15b_certified"],
        "certified_baseline_breadth": {
            B2_SGV15: _breadth(
                per_environment(
                    slice_cells(cells, B2_SGV15, review_budget=0.0, documents=documents),
                    floor,
                    PRIMARY_EPSILON,
                )
            ),
            B3_SGV15B: _breadth(
                per_environment(
                    slice_cells(cells, B3_SGV15B, review_budget=0.0, documents=documents),
                    floor,
                    PRIMARY_EPSILON,
                )
            ),
        },
        "answer": bool(
            primary["environments_safe"]
            >= _breadth(
                per_environment(
                    slice_cells(cells, B2_SGV15, review_budget=0.0, documents=documents),
                    floor,
                    PRIMARY_EPSILON,
                )
            )["environments_safe"]
            - 1
            and primary["mean_total_repair_recall"] > 0.0
        ),
    }

    practical = bool(
        corners["primary_epsilon_moderate_review"]["environments_safe_and_non_degenerate"]
        >= MAJORITY_REQUIRED
    )
    heavy = bool(
        corners["primary_epsilon_review_heavy"]["environments_safe_and_non_degenerate"]
        >= MAJORITY_REQUIRED
    )
    wider = bool(
        corners["wider_tolerance_moderate_review"]["environments_safe_and_non_degenerate"]
        >= MAJORITY_REQUIRED
    )
    review_helps = bool(
        corners["primary_epsilon_moderate_review"]["environments_safe_and_non_degenerate"]
        > corners["primary_epsilon_no_review"]["environments_safe_and_non_degenerate"]
    )
    if practical:
        outcome, summary = (
            "A",
            (
                "a policy holding the primary tolerance with non-degenerate automation exists "
                "inside the moderate-review, realistic-supervision corner"
            ),
        )
    elif heavy:
        outcome, summary = (
            "B",
            (
                "review moves the frontier and reaches majority breadth, but only above the "
                "declared practical review budget"
            ),
        )
    elif wider:
        outcome, summary = (
            "C",
            (
                "deployment reaches majority breadth at the wider tolerance and not at the primary "
                "one; the primary target does not move"
            ),
        )
    else:
        outcome, summary = (
            "D",
            (
                "human review measurably improves the frontier but does not produce safe, "
                "non-degenerate automatic correction on a majority of environments at any declared "
                "review budget; the remaining bottleneck is upstream of this stage"
            ),
        )

    ready = bool(len(met) == len(criteria) and practical)
    payload = {
        **_envelope("development_decision"),
        "stage": STAGE,
        "status": "COMPLETE",
        "verdict": "PRACTICAL REGIME IDENTIFIED"
        if practical
        else "NO PRACTICAL REGIME AT THE DECLARED BUDGETS",
        "scientific_status": design["scientific_status"],
        "primary_epsilon": float(PRIMARY_EPSILON),
        "alpha": ALPHA,
        "coverage_floor": floor,
        "best_practical_policy": {
            "policy": policy,
            "boundary_rule": POLICY_RULE[policy][0],
            "routing": POLICY_RULE[policy][1],
            "safety_label": CUT_SAFETY[POLICY_RULE[policy][0]],
            "review_budget": rate,
            "target_document_budget": documents,
            "reviewer_scenario": PRIMARY_SCENARIO,
            "regime": _regime_of(rate),
        },
        "automatic_harm": primary["mean_automatic_harm"],
        "automatic_repair_recall": primary["mean_automatic_repair_recall"],
        "risk_controlled_automatic_repair_recall": primary["mean_risk_controlled_automatic_recall"],
        "total_repair_recall": primary["mean_total_repair_recall"],
        "review_rate": primary["mean_review_rate"],
        "target_document_budget": documents,
        "mean_certification_labels": primary["mean_certification_labels"],
        "mean_supervision_labels": primary["mean_supervision_labels"],
        "mean_reviewed_candidates": primary["mean_reviewed_candidates"],
        "environments_safe": primary["environments_safe"],
        "environments_nondegenerate": primary["environments_non_degenerate"],
        "environments_safe_and_nondegenerate": primary["environments_safe_and_non_degenerate"],
        "human_review_load_bearing": bool(
            review_helps and float(stats["primary_risk_controlled_total_recall"]["ci_lower"]) > 0.0
        ),
        "target_supervision_load_bearing": bool(
            float(controls["controls"]["control_3_label_permutation"]["delta"]) > 0.0
            and float(controls["controls"]["control_10_zero_target_label_deployment"]["delta"])
            > 0.0
        ),
        "review_routing_load_bearing": bool(
            controls["controls"]["control_1_random_review"]["discriminates"]
            and controls["controls"]["control_2_shuffled_review_ranking"]["discriminates"]
        ),
        "review_accuracy_sensitivity": {
            "scenarios": {
                name: REVIEWER_ACCURACY[name] for name in (*REVIEWER_SCENARIOS, *CONTROL_SCENARIOS)
            },
            "status": "SIMULATION SCENARIOS -- no reviewer accuracy here was measured on people",
            "total_repair_recall_lost_from_ideal_to_moderate": sensitivity["by_policy"][policy][
                f"review_{rate}"
            ]["total_recall_lost_from_ideal_to_moderate"],
            "depends_on_a_near_perfect_reviewer": sensitivity["by_policy"][policy][
                f"review_{rate}"
            ]["depends_on_a_near_perfect_reviewer"],
        },
        "selection_and_breadth_are_different_readings": {
            "selected_by_nested_selection": _configuration_key((policy, rate, documents)),
            "selection_objective": (
                "mean risk-controlled automatic repair recall on the nine training environments"
            ),
            "widest_breadth_in_the_practical_corner": {
                key: corners["primary_epsilon_moderate_review"].get(key)
                for key in (
                    "policy",
                    "review_budget",
                    "document_budget",
                    "environments_safe_and_non_degenerate",
                    "mean_risk_controlled_automatic_recall",
                )
            },
            "why_they_differ": (
                "the nested selection maximises an endpoint on held-out environments; the corner "
                "scan maximises BREADTH over every environment at once and is therefore an "
                "in-sample maximum. Neither is a substitute for the other and the criteria are "
                "read on the selected regime, which is the one a deployment would actually get."
            ),
        },
        "criteria": criteria,
        "criteria_met": len(met),
        "criteria_total": len(criteria),
        "criteria_met_names": met,
        "questions": {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5},
        "design_corners": corners,
        "secondary_policy_diagnostic": secondary_diagnostic,
        "outcome": {"label": outcome, "summary": summary},
        "outcome_taxonomy": design["outcome_taxonomy"],
        "ready_for_external_confirmation": ready,
        "reason": (
            f"{len(met)} of {len(criteria)} pre-registered criteria met; the widest breadth inside "
            f"the declared practical corner is "
            f"{corners['primary_epsilon_moderate_review']['environments_safe_and_non_degenerate']} "
            f"of {ENVIRONMENT_COUNT} against a majority of {MAJORITY_REQUIRED}. "
            + (
                "A single frozen regime satisfies every criterion, so an external confirmation "
                "stage is warranted."
                if ready
                else "External confirmation is not warranted and the reserve stays locked."
            )
        ),
        "next_investigation": (
            design["stopping_rule"]
            if outcome == "D"
            else "freeze the exact deployment regime before any confirmatory stage is considered"
        ),
        "confirmatory_reserve_consumed": False,
        "synthetic": False,
        "issued_head": _git("rev-parse", "HEAD"),
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DECISION, payload)
    print(
        f"decide: {len(met)}/{len(criteria)} criteria, outcome {outcome}, "
        f"{primary['environments_safe']} safe / {primary['environments_non_degenerate']} "
        f"non-degenerate, ready_for_external_confirmation={ready} -> {_relative(DECISION)}"
    )
    return 0


# ------------------------------------------------------------------ section 37: the figures

POLICY_COLOUR = {
    P0_PRESERVE: "#000000",
    P1_AUTO_ONLY: "#8c8c8c",
    P2_CONSERVATIVE: "#1f77b4",
    P3_BOUNDARY: "#d62728",
    P4_EXPECTED_VALUE: "#9467bd",
    B1_SOURCE: "#bcbd22",
    B2_SGV15: "#2ca02c",
    B3_SGV15B: "#17becf",
    B4_RANDOM_REVIEW: "#ff7f0e",
    B6_HIGH_RISK_REVIEW: "#7f0000",
}
POLICY_LABEL = {
    P0_PRESERVE: "P0 preserve all",
    P1_AUTO_ONLY: "P1 auto only",
    P2_CONSERVATIVE: "P2 certified auto + review",
    P3_BOUNDARY: "P3 boundary review",
    P4_EXPECTED_VALUE: "P4 expected-value review (secondary)",
    B1_SOURCE: "B1 frozen source cut",
    B2_SGV15: "B2 SGV15 certified",
    B3_SGV15B: "B3 SGV15b certified",
    B4_RANDOM_REVIEW: "B4 random review",
    B6_HIGH_RISK_REVIEW: "B6 high-risk review",
}
SCENARIO_LABEL = {
    H1_IDEAL: "ideal reviewer (UPPER BOUND)",
    H2_HIGH: "reviewer 0.98",
    H3_MODERATE: "reviewer 0.95",
    H4_COIN_FLIP: "coin flip (control)",
}


def _short(name: str) -> str:
    return f"{name.split('/')[0][:2]}:{name.split('/')[-1]}"


def run_figures() -> int:
    """The fourteen required figures. Every plotted number comes out of a saved artifact."""
    started = time.monotonic()
    plt = s15b._figure_style()
    review = cc_read_json(REVIEW_FRONTIER)
    supervision = cc_read_json(SUPERVISION_FRONTIER)
    tolerance = cc_read_json(HARM_TOLERANCE_FRONTIER)
    auto_only = cc_read_json(AUTO_ONLY_RESULTS)
    yields = cc_read_json(REVIEW_YIELD)["by_cell"]
    intercepts = cc_read_json(HARM_INTERCEPT)["by_cell"]
    sensitivity = cc_read_json(REVIEW_ACCURACY_SENSITIVITY)["by_policy"]
    pareto = cc_read_json(PARETO_FRONTIER)["frontiers"]
    decision = cc_read_json(DECISION)
    written: list[Path] = []

    def series(policy: str, scenario: str, field: str) -> tuple[list[float], list[float]]:
        rows = review["by_policy"][policy][scenario]
        return (
            [float(row["mean_review_rate"]) for row in rows],
            [float(row[field]) for row in rows],
        )

    # 1. Risk against automation, per environment, at the selected regime and its comparators.
    figure, axis = plt.subplots(figsize=(6.6, 4.0))
    for policy in (P1_AUTO_ONLY, P2_CONSERVATIVE, P3_BOUNDARY, B2_SGV15, B3_SGV15B):
        rate = PRACTICAL_REVIEW_BUDGET if POLICY_RULE[policy][2] else 0.0
        scenario = PRIMARY_SCENARIO if rate > 0.0 else H0_NO_REVIEW
        if POLICY_RULE[policy][2]:
            record = cc_read_json(HUMAN_REVIEW_RESULTS)["by_policy"][policy]["by_epsilon"][
                f"epsilon_{PRIMARY_EPSILON}"
            ][str(PRACTICAL_DOC_BUDGET)][f"review_{rate}"][scenario]["per_environment"]
        else:
            record = auto_only["by_policy"][policy]["by_epsilon"][f"epsilon_{PRIMARY_EPSILON}"][
                str(PRACTICAL_DOC_BUDGET)
            ]["per_environment"]
        xs = [float(record[n]["automatic_coverage"]) for n in sorted(record)]
        ys = [float(record[n]["automatic_harm"]) for n in sorted(record)]
        axis.scatter(xs, ys, s=18, color=POLICY_COLOUR[policy], label=POLICY_LABEL[policy])
    axis.axhline(PRIMARY_EPSILON, color="#d62728", linestyle="--", lw=1, label="epsilon")
    axis.axvline(
        decision["coverage_floor"], color="#555555", linestyle=":", lw=1, label="coverage floor"
    )
    axis.set_xlabel("automatic coverage")
    axis.set_ylabel("automatic harm")
    axis.set_title(
        f"Risk against automation, per environment ({PRACTICAL_DOC_BUDGET} target documents, "
        f"review {PRACTICAL_REVIEW_BUDGET:.0%})"
    )
    axis.legend(fontsize=6)
    written.append(s15b._finish(figure, FIGURE_DIR / "risk_coverage_frontier.png"))

    # 2. Review rate against automatic harm.
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for policy in REVIEW_POLICIES:
        xs, ys = series(policy, PRIMARY_SCENARIO, "mean_automatic_harm")
        axis.plot(
            xs, ys, "o-", color=POLICY_COLOUR[policy], label=POLICY_LABEL[policy], lw=1.2, ms=3
        )
    axis.axhline(PRIMARY_EPSILON, color="#d62728", linestyle="--", lw=1, label="epsilon")
    axis.set_xlabel("review rate")
    axis.set_ylabel("mean automatic harm")
    axis.set_title("What review buys on the automatic side")
    axis.legend(fontsize=6)
    written.append(s15b._finish(figure, FIGURE_DIR / "risk_review_frontier.png"))

    # 3. Review rate against risk-controlled automatic recall.
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for policy in REVIEW_POLICIES:
        xs, ys = series(policy, PRIMARY_SCENARIO, "mean_risk_controlled_automatic_recall")
        axis.plot(
            xs, ys, "o-", color=POLICY_COLOUR[policy], label=POLICY_LABEL[policy], lw=1.2, ms=3
        )
    axis.set_xlabel("review rate")
    axis.set_ylabel("mean risk-controlled automatic repair recall")
    axis.set_title("Automation against review burden")
    axis.legend(fontsize=6)
    written.append(s15b._finish(figure, FIGURE_DIR / "automation_review_frontier.png"))

    # 4. Review rate against total repair recall.
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for policy in REVIEW_POLICIES:
        xs, ys = series(policy, PRIMARY_SCENARIO, "mean_total_repair_recall")
        axis.plot(
            xs, ys, "o-", color=POLICY_COLOUR[policy], label=POLICY_LABEL[policy], lw=1.2, ms=3
        )
    axis.set_xlabel("review rate")
    axis.set_ylabel("mean total repair recall (automatic plus reviewed)")
    axis.set_title("Total repair recall against review burden")
    axis.legend(fontsize=6)
    written.append(s15b._finish(figure, FIGURE_DIR / "total_repair_review_frontier.png"))

    # 5. Target documents against automation, at several fixed review budgets.
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for rate in REVIEW_BUDGETS:
        rows = supervision["by_policy"][P3_BOUNDARY].get(f"review_{rate}", [])
        if not rows:
            continue
        axis.plot(
            [int(row["document_budget"]) for row in rows],
            [float(row["mean_risk_controlled_automatic_recall"]) for row in rows],
            "o-",
            lw=1.2,
            ms=3,
            label=f"review {rate:.0%}",
        )
    axis.set_xlabel("target documents purchased")
    axis.set_ylabel("mean risk-controlled automatic repair recall")
    axis.set_title(f"Supervision against automation, {POLICY_LABEL[P3_BOUNDARY]}")
    axis.legend(fontsize=6)
    written.append(s15b._finish(figure, FIGURE_DIR / "target_docs_automation_frontier.png"))

    # 6. Target documents against the review burden actually spent, with the certification
    #    requirement SGV15b measured on the same axis.
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    for rate in REVIEW_BUDGETS:
        rows = supervision["by_policy"][P3_BOUNDARY].get(f"review_{rate}", [])
        if not rows:
            continue
        axis.plot(
            [int(row["document_budget"]) for row in rows],
            [float(row["mean_reviewed_candidates"]) for row in rows],
            "o-",
            lw=1.2,
            ms=3,
            label=f"review budget {rate:.0%}",
        )
    axis.set_xlabel("target documents purchased")
    axis.set_ylabel("mean candidates sent to review")
    axis.set_title("Supervision against review burden")
    axis.legend(fontsize=6)
    written.append(s15b._finish(figure, FIGURE_DIR / "target_docs_review_frontier.png"))

    # 7. The price of safety.
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for policy in (P1_AUTO_ONLY, P2_CONSERVATIVE, P3_BOUNDARY, B2_SGV15, B3_SGV15B):
        for rate, style in ((0.0, "--"), (PRACTICAL_REVIEW_BUDGET, "-")):
            rows = tolerance["by_policy"][policy].get(f"review_{rate}", [])
            if not rows:
                continue
            axis.plot(
                [float(row["epsilon"]) for row in rows],
                [float(row["mean_risk_controlled_automatic_recall"]) for row in rows],
                style,
                marker="o",
                ms=3,
                lw=1.2,
                color=POLICY_COLOUR[policy],
                label=f"{POLICY_LABEL[policy]}, review {rate:.0%}",
            )
    axis.axvline(PRIMARY_EPSILON, color="#d62728", linestyle=":", lw=1, label="primary epsilon")
    axis.set_xlabel("harm tolerance")
    axis.set_ylabel("mean risk-controlled automatic repair recall")
    axis.set_title("The price of safety")
    axis.legend(fontsize=5)
    written.append(s15b._finish(figure, FIGURE_DIR / "harm_tolerance_frontier.png"))

    # 8. Section 18's pre-registered routing comparison.
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for arm, policy in ROUTING_ARMS.items():
        xs, ys = series(policy, PRIMARY_SCENARIO, "mean_risk_controlled_automatic_recall")
        axes[0].plot(xs, ys, "o-", color=POLICY_COLOUR[policy], label=arm, lw=1.2, ms=3)
        _, hs = series(policy, PRIMARY_SCENARIO, "mean_automatic_harm")
        axes[1].plot(xs, hs, "o-", color=POLICY_COLOUR[policy], label=arm, lw=1.2, ms=3)
    axes[0].set_ylabel("mean risk-controlled automatic recall")
    axes[1].set_ylabel("mean automatic harm")
    axes[1].axhline(PRIMARY_EPSILON, color="#d62728", linestyle="--", lw=1)
    for axis in axes:
        axis.set_xlabel("review rate")
        axis.legend(fontsize=6)
    figure.suptitle("Where review is spent, at matched budget", fontsize=9)
    written.append(s15b._finish(figure, FIGURE_DIR / "boundary_vs_random_review.png"))

    # 9. Review yield, decomposed.
    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    labels = []
    found = []
    recovered = []
    for arm in ROUTING_ARMS:
        for rate in REVIEW_BUDGETS:
            key = f"{arm}|review_{rate}|{PRIMARY_SCENARIO}"
            if key not in yields:
                continue
            labels.append(f"{arm.split('_')[0]}\n{rate:.0%}")
            found.append(float(yields[key]["mean_harm_found_per_100_reviews"]))
            recovered.append(float(yields[key]["mean_benefit_recovered_per_100_reviews"]))
    positions = np.arange(len(labels))
    axis.bar(positions, found, color="#d62728", label="harmful edits blocked per 100 reviews")
    axis.bar(
        positions,
        recovered,
        bottom=found,
        color="#2ca02c",
        label="repairs recovered per 100 reviews",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=5)
    axis.set_ylabel("per 100 reviews")
    axis.set_title(f"What a review is spent on ({SCENARIO_LABEL[PRIMARY_SCENARIO]})")
    axis.legend(fontsize=6)
    written.append(s15b._finish(figure, FIGURE_DIR / "review_yield.png"))

    # 10. Routing interception against realised interception.
    figure, axis = plt.subplots(figsize=(6.6, 3.4))
    for arm in ROUTING_ARMS:
        xs, routed, realised = [], [], []
        for rate in REVIEW_BUDGETS:
            key = f"{arm}|review_{rate}|{PRIMARY_SCENARIO}"
            if key not in intercepts:
                continue
            xs.append(rate)
            routed.append(float(intercepts[key]["mean_routing_intercept_rate"]))
            realised.append(float(intercepts[key]["mean_realised_intercept_rate"]))
        colour = POLICY_COLOUR[ROUTING_ARMS[arm]]
        axis.plot(xs, routed, "o-", color=colour, lw=1.2, ms=3, label=f"{arm} routed away")
        axis.plot(xs, realised, "s--", color=colour, lw=1.0, ms=3, label=f"{arm} actually blocked")
    axis.set_xlabel("review rate")
    axis.set_ylabel("share of would-be automatic harmful edits")
    axis.set_title("Interception: what routing removes, and what the reviewer keeps out")
    axis.legend(fontsize=5)
    written.append(s15b._finish(figure, FIGURE_DIR / "harm_intercept_rate.png"))

    # 11. Reviewer-accuracy sensitivity.
    figure, axis = plt.subplots(figsize=(6.6, 3.4))
    for policy in REVIEW_POLICIES:
        record = sensitivity.get(policy, {}).get(f"review_{PRACTICAL_REVIEW_BUDGET}", {})
        by_scenario = record.get("by_scenario", {})
        names = [n for n in (H1_IDEAL, H2_HIGH, H3_MODERATE, H4_COIN_FLIP) if n in by_scenario]
        axis.plot(
            [REVIEWER_ACCURACY[n] for n in names],
            [float(by_scenario[n]["mean_total_repair_recall"]) for n in names],
            "o-",
            color=POLICY_COLOUR[policy],
            lw=1.2,
            ms=3,
            label=POLICY_LABEL[policy],
        )
    axis.set_xlabel("simulated reviewer accuracy (SCENARIO, not measured on people)")
    axis.set_ylabel("mean total repair recall")
    axis.set_title(f"Reviewer-accuracy sensitivity at review {PRACTICAL_REVIEW_BUDGET:.0%}")
    axis.legend(fontsize=6)
    written.append(s15b._finish(figure, FIGURE_DIR / "review_accuracy_sensitivity.png"))

    # 12. Certified against empirically risk-controlled, on the same ranking.
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    for policy in (B2_SGV15, B3_SGV15B, P2_CONSERVATIVE, P1_AUTO_ONLY, P3_BOUNDARY):
        rates = (0.0, PRACTICAL_REVIEW_BUDGET) if POLICY_RULE[policy][2] else (0.0,)
        for rate in rates:
            rows = supervision["by_policy"][policy].get(f"review_{rate}", [])
            if not rows:
                continue
            marker = "o" if CUT_SAFETY[POLICY_RULE[policy][0]] == SAFETY_CERTIFIED else "^"
            axis.plot(
                [float(row["mean_automatic_harm"]) for row in rows],
                [float(row["mean_risk_controlled_automatic_recall"]) for row in rows],
                marker=marker,
                linestyle="none",
                color=POLICY_COLOUR[policy],
                ms=5,
                label=(
                    f"{POLICY_LABEL[policy]} "
                    f"({CUT_SAFETY[POLICY_RULE[policy][0]]}), review {rate:.0%}"
                ),
            )
    axis.axvline(PRIMARY_EPSILON, color="#d62728", linestyle="--", lw=1, label="epsilon")
    axis.set_xlabel("mean automatic harm")
    axis.set_ylabel("mean risk-controlled automatic repair recall")
    axis.set_title("Certified circles, empirically risk-controlled triangles")
    axis.legend(fontsize=5)
    written.append(s15b._finish(figure, FIGURE_DIR / "certified_vs_empirical_frontier.png"))

    # 13. The non-dominated frontier, faceted by document budget.
    budgets = [b for b in DOC_BUDGETS if b <= PRACTICAL_DOC_BUDGET]
    figure, axes = plt.subplots(1, len(budgets), figsize=(3.0 * len(budgets), 3.2), sharey=True)
    points = pareto["automatic_repair_recall"]["non_dominated"]
    for axis, budget in zip(np.atleast_1d(axes), budgets, strict=True):
        subset = [p for p in points if int(p["document_budget"]) == budget]
        for policy in POLICY_NAMES:
            block = [p for p in subset if p["policy"] == policy]
            if not block:
                continue
            axis.scatter(
                [float(p["mean_review_rate"]) for p in block],
                [float(p["mean_risk_controlled_automatic_recall"]) for p in block],
                s=20,
                color=POLICY_COLOUR[policy],
                label=POLICY_LABEL[policy],
            )
        axis.set_title(f"{budget} target documents", fontsize=8)
        axis.set_xlabel("review rate")
    np.atleast_1d(axes)[0].set_ylabel("risk-controlled automatic recall")
    np.atleast_1d(axes)[-1].legend(fontsize=4)
    figure.suptitle("Non-dominated operating points", fontsize=9)
    written.append(s15b._finish(figure, FIGURE_DIR / "pareto_frontier_3d_or_faceted.png"))

    # 14. The regime matrix: breadth at every (document budget, review budget).
    figure, axis = plt.subplots(figsize=(6.2, 3.4))
    matrix = np.full((len(DOC_BUDGETS), len(REVIEW_BUDGETS)), np.nan)
    table = review["primary_table"]
    for row, budget in enumerate(DOC_BUDGETS):
        for column, rate in enumerate(REVIEW_BUDGETS):
            record = table.get(str(budget), {}).get(f"review_{rate}")
            if record is not None:
                matrix[row, column] = float(record["environments_safe_and_non_degenerate"])
    image = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=ENVIRONMENT_COUNT)
    axis.set_xticks(range(len(REVIEW_BUDGETS)))
    axis.set_xticklabels([f"{rate:.0%}" for rate in REVIEW_BUDGETS])
    axis.set_yticks(range(len(DOC_BUDGETS)))
    axis.set_yticklabels([str(budget) for budget in DOC_BUDGETS])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if not math.isnan(matrix[row, column]):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )
    axis.set_xlabel("review budget")
    axis.set_ylabel("target documents")
    axis.set_title(
        f"Environments safe AND non-degenerate at epsilon {PRIMARY_EPSILON}, of {ENVIRONMENT_COUNT}"
    )
    figure.colorbar(image, ax=axis, shrink=0.8)
    written.append(s15b._finish(figure, FIGURE_DIR / "deployment_regime_matrix.png"))

    sources = (
        REVIEW_FRONTIER,
        SUPERVISION_FRONTIER,
        HARM_TOLERANCE_FRONTIER,
        AUTO_ONLY_RESULTS,
        HUMAN_REVIEW_RESULTS,
        REVIEW_YIELD,
        HARM_INTERCEPT,
        REVIEW_ACCURACY_SENSITIVITY,
        PARETO_FRONTIER,
        DECISION,
    )
    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_envelope("figure_manifest"),
            "synthetic": False,
            "scientific_status": "DEVELOPMENT -- SGV14-exposed environments",
            "reviewer_accuracy_is_simulated": (
                "every figure that varies reviewer accuracy plots SIMULATION SCENARIOS. The "
                "ideal reviewer is an idealized upper bound."
            ),
            "figures": {
                _relative(path): {
                    "sha256": file_sha256(path),
                    "derived_from": [_relative(source) for source in sources],
                    "source_sha256": {_relative(source): file_sha256(source) for source in sources},
                }
                for path in written
            },
            "figure_count": len(written),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {_relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ section 46: determinism
#
# Two independent regenerations of the selected configuration, from the raw environment build
# forward: the frozen adaptation is refitted, the prefix ladder is replaced on the unlabelled
# certification pool, the document samples are redrawn under their recorded seeds, the boundary is
# re-placed, the review order is rebuilt, the reviewer's own draws are regenerated and the
# deployment is re-measured. Nothing is read from the stored table except for the comparison.

compare_frames = s15b.compare_frames


def _regenerate(policy: str, rate: float, documents: int) -> tuple[pd.DataFrame, dict[str, str]]:
    built, _ = s15._environments()
    floor = load_floor()
    accuracies = tuple(
        REVIEWER_ACCURACY[name] for name in (*REVIEWER_SCENARIOS, *CONTROL_SCENARIOS)
    )
    rows: list[dict[str, Any]] = []
    signature: dict[str, str] = {}
    for environment in built:
        name = environment.name
        seed = _stable_seed("sgv15-adapt", str(s15.SAMPLE_SEED), name, s15.A_UNCERTAINTY)
        adapted = fit_adaptation(environment, s15.A_UNCERTAINTY, s15.ADAPT_BUDGET, seed)
        block = environment.setup.evaluation
        cert_documents = np.asarray(
            [str(d) for d in environment.cert_documents.tolist()], dtype=object
        )
        names = np.asarray(sorted(set(cert_documents.tolist())), dtype=object)
        lookup = {value: index for index, value in enumerate(names.tolist())}
        doc_of = np.asarray([lookup[str(d)] for d in cert_documents.tolist()], dtype=np.int64)
        cert_scores = np.asarray(adapted.cert_scores, dtype=float)
        cert_harm = np.asarray(environment.cert_harmful, dtype=bool)
        fine = prefix_thresholds(cert_scores, FINE_DEPTHS)
        nine = prefix_thresholds(cert_scores, NINE_DEPTHS)
        accepted, harmful = _per_document_prefix_counts(
            cert_scores, cert_harm, doc_of, fine, names.size
        )
        nine_accepted, nine_harmful = _per_document_prefix_counts(
            cert_scores, cert_harm, doc_of, nine, names.size
        )
        eval_names = sorted({str(d) for d in block.documents.tolist()})
        eval_lookup = {value: index for index, value in enumerate(eval_names)}
        deployment = Deployment(
            name=name,
            corpus=environment.corpus,
            base_engine=environment.base_engine,
            cert_documents=names.astype(str),
            cert_doc_of=doc_of,
            cert_scores=cert_scores,
            cert_harm=cert_harm,
            eval_scores=np.asarray(adapted.eval_scores, dtype=float),
            eval_harm=np.asarray(block.harmful, dtype=bool),
            eval_beneficial=np.asarray(block.beneficial, dtype=bool),
            eval_doc_of=np.asarray(
                [eval_lookup[str(d)] for d in block.documents.tolist()], dtype=np.int64
            ),
            eval_documents=np.asarray(eval_names, dtype=str),
            fine_thresholds=fine,
            nine_thresholds=nine,
            source_tau={
                float(epsilon): float(frozen_source_tau(environment, adapted.model, epsilon))
                for epsilon in EPSILONS
            },
            accepted=accepted,
            harmful=harmful,
            nine_accepted=nine_accepted,
            nine_harmful=nine_harmful,
        )
        lattice = Lattice(deployment, floor, accuracies)
        samples: list[list[int]] = []
        orders: list[list[int]] = []
        for epsilon in EPSILONS:
            scenarios = (
                (*REVIEWER_SCENARIOS, *CONTROL_SCENARIOS)
                if epsilon == PRIMARY_EPSILON
                else REVIEWER_SCENARIOS
            )
            for draw in range(1 if documents == 0 else DOC_DRAWS):
                sample = document_sample(deployment, documents, _sample_seed(name, documents, draw))
                samples.append([int(value) for value in sample.tolist()])
                cells, _keep = _cell_rows(
                    lattice, epsilon, documents, draw, sample, (policy,), scenarios
                )
                rows.extend(cells)
                rule, routing, _reviews = POLICY_RULE[policy]
                boundary = select_boundary(deployment, rule, sample, epsilon)
                order = (
                    review_order(
                        deployment,
                        float(boundary["tau"]),
                        routing,
                        gain=expected_gain(deployment, sample, float(boundary["tau"])),
                    )
                    if routing == ROUTE_EXPECTED_VALUE
                    else lattice.order(routing, float(boundary["tau"]))
                )
                take = min(lattice.budget_count(rate), int(order.size))
                orders.append([int(value) for value in order[:take].tolist()])
        signature[f"{name}|documents"] = canonical_hash({"documents": names.astype(str).tolist()})
        signature[f"{name}|thresholds"] = canonical_hash(
            {"fine": fine.tolist(), "nine": nine.tolist()}
        )
        signature[f"{name}|samples"] = canonical_hash({"samples": samples})
        signature[f"{name}|review_selection"] = canonical_hash({"orders": orders})
        signature[f"{name}|reviewer_draws"] = canonical_hash(
            {"u": reviewer_correctness(deployment).tolist()}
        )
    return pd.DataFrame(rows), signature


def run_determinism() -> int:
    """Regenerate the selected configuration twice and compare everything it touched."""
    started = time.monotonic()
    policy, rate, documents = selected_configuration()
    stored, _design = load_cells()
    stored = stored[
        (stored["policy"] == policy) & (stored["requested_documents"] == documents)
    ].reset_index(drop=True)
    runs: list[dict[str, Any]] = []
    signatures: list[dict[str, str]] = []
    derived: list[dict[str, Any]] = []
    floor = load_floor()
    for index in range(DETERMINISM_RUNS):
        began = time.monotonic()
        regenerated, hashes = _regenerate(policy, rate, documents)
        signatures.append(hashes)
        report = compare_frames(stored, regenerated, ["cell", "epsilon"])
        block = regenerated[
            (regenerated["review_budget"] == rate)
            & (regenerated["reviewer_scenario"] == _scenario_of(rate))
            & (regenerated["epsilon"] == PRIMARY_EPSILON)
        ]
        summary = _breadth(per_environment(block, floor, PRIMARY_EPSILON))
        point = {
            "mean_automatic_harm": summary["mean_automatic_harm"],
            "mean_review_rate": summary["mean_review_rate"],
            "document_budget": documents,
            "mean_risk_controlled_automatic_recall": summary[
                "mean_risk_controlled_automatic_recall"
            ],
            "epsilon": PRIMARY_EPSILON,
        }
        derived.append(
            {
                "breadth": {
                    key: summary[key]
                    for key in (
                        "environments_safe",
                        "environments_non_degenerate",
                        "environments_safe_and_non_degenerate",
                    )
                },
                "pareto_vector": [float(value) for value in _vector(point, PARETO_MAXIMISE)],
                "endpoint_hash": canonical_hash(
                    {key: summary[key] for key in sorted(summary) if not key.endswith("_names")}
                ),
            }
        )
        runs.append(
            {
                "run": index + 1,
                "issued_utc": _iso(),
                "elapsed_seconds": time.monotonic() - began,
                **report,
            }
        )
        print(
            f"  run {index + 1}: {report['rows_regenerated']} rows, "
            f"max difference {report['max_absolute_numeric_difference']}, "
            f"identical={report['identical']}",
            flush=True,
        )
    fields = sorted(signatures[0])
    signature_agrees = all(
        signature[field] == signatures[0][field] for signature in signatures for field in fields
    )
    derived_agrees = all(record == derived[0] for record in derived)
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "what_is_regenerated": (
                "the SELECTED configuration: every environment rebuilt from scratch, the frozen "
                "adaptation refitted, the prefix ladder replaced on the unlabelled certification "
                "pool, the document samples redrawn under their recorded seeds, the boundary "
                "re-placed, the review order rebuilt, the reviewer's own uniforms regenerated and "
                "the deployment re-measured. Nothing is read from the stored table except for the "
                "comparison itself."
            ),
            "selected_configuration": {
                "policy": policy,
                "review_budget": rate,
                "document_budget": documents,
            },
            "runs_completed": len(runs),
            "runs": runs,
            "all_runs_identical": all(record["identical"] for record in runs),
            "max_absolute_numeric_difference_over_all_runs": max(
                record["max_absolute_numeric_difference"] for record in runs
            ),
            "structural_signatures_compared": len(fields),
            "structural_signatures_agree": bool(signature_agrees),
            "structural_signature_fields": [
                "the certification-pool document list per environment",
                "the placed prefix ladder per environment",
                "the sampled document set for every draw",
                "the review selection for every draw",
                "the reviewer's uniform draws per environment",
            ],
            "derived_results_compared": [
                "the breadth counts",
                "the selected point's Pareto vector",
                "a canonical hash of the whole per-environment endpoint summary",
            ],
            "derived_results": derived,
            "derived_results_agree": bool(derived_agrees),
            "inf_handling": (
                "a refusal threshold is +inf and inf - inf is nan, so equality is established "
                "before any subtraction; identical refusals compare as identical"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"determinism: {len(runs)} runs, max difference "
        f"{max(record['max_absolute_numeric_difference'] for record in runs)}, signatures agree="
        f"{signature_agrees}, derived agree={derived_agrees} -> {_relative(DETERMINISM)}"
    )
    return 0


# --------------------------------------------- sections 39, 45: provenance and traceability

PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    ENVIRONMENT_INVENTORY,
    CAPACITY_ANALYSIS,
    DESIGN_RECORD,
    DEPLOYMENT_POLICY_FAMILY,
    REVIEW_BUDGET_FAMILY,
    REVIEWER_ACCURACY_SCENARIOS,
    TARGET_DOCUMENT_BUDGETS,
    SCORE_INVENTORY,
    GEOMETRY,
    REPRODUCTION,
    DEPLOYMENT_CELLS,
    DOCUMENT_COUNTS,
    AUTO_ONLY_RESULTS,
    HUMAN_REVIEW_RESULTS,
    BOUNDARY_REVIEW_RESULTS,
    RANDOM_REVIEW_RESULTS,
    HIGH_RISK_REVIEW_RESULTS,
    REVIEW_YIELD,
    HARM_INTERCEPT,
    HARM_TOLERANCE_FRONTIER,
    REVIEW_FRONTIER,
    SUPERVISION_FRONTIER,
    PARETO_FRONTIER,
    DOMINANCE_MATRIX,
    REVIEW_ACCURACY_SENSITIVITY,
    SEQUENTIAL_DEPLOYMENT,
    CONTROL_RESULTS,
    NEGATIVE_TESTS,
    STATISTICAL_TESTS,
    NESTED_SELECTION,
    DECISION,
    DETERMINISM,
    FIGURE_MANIFEST,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE, FROZEN_CONFIGURATION),
    "2. Why certification development stops here": (RESEARCH_FREEZE, SUPERVISION_FRONTIER),
    "3. Frozen SGV15 and SGV15b findings": (RESEARCH_FREEZE, REPRODUCTION),
    "4. Research questions": (DESIGN_RECORD,),
    "5. Deployment actions": (DEPLOYMENT_POLICY_FAMILY, DESIGN_RECORD),
    "6. The automatic harm definition": (DESIGN_RECORD,),
    "7. The human-review model": (DEPLOYMENT_POLICY_FAMILY, REVIEW_BUDGET_FAMILY),
    "8. Reviewer-accuracy scenarios": (REVIEWER_ACCURACY_SCENARIOS, REVIEW_ACCURACY_SENSITIVITY),
    "9. Target-supervision budgets": (
        TARGET_DOCUMENT_BUDGETS,
        ENVIRONMENT_INVENTORY,
        CAPACITY_ANALYSIS,
    ),
    "10. The deployment policy family": (DEPLOYMENT_POLICY_FAMILY,),
    "11. Human-review routing strategies": (DEPLOYMENT_POLICY_FAMILY,),
    "12. Experimental design": (
        DESIGN_RECORD,
        ENVIRONMENT_INVENTORY,
        SCORE_INVENTORY,
        REPRODUCTION,
        PROVENANCE,
    ),
    "13. Automatic-only baselines": (AUTO_ONLY_RESULTS,),
    "14. Certified baselines": (AUTO_ONLY_RESULTS, REPRODUCTION),
    "15. Human-review results": (HUMAN_REVIEW_RESULTS, REVIEW_FRONTIER),
    "16. The risk-coverage frontier": (AUTO_ONLY_RESULTS, HUMAN_REVIEW_RESULTS),
    "17. The risk-review frontier": (REVIEW_FRONTIER,),
    "18. The automation-review frontier": (REVIEW_FRONTIER,),
    "19. The target-supervision frontier": (SUPERVISION_FRONTIER, TARGET_DOCUMENT_BUDGETS),
    "20. The harm-tolerance frontier": (HARM_TOLERANCE_FRONTIER,),
    "21. Review efficiency": (REVIEW_YIELD, HARM_INTERCEPT),
    "22. Boundary against random review": (
        BOUNDARY_REVIEW_RESULTS,
        RANDOM_REVIEW_RESULTS,
        HIGH_RISK_REVIEW_RESULTS,
        STATISTICAL_TESTS,
        CONTROL_RESULTS,
    ),
    "23. Reviewer-accuracy sensitivity": (REVIEW_ACCURACY_SENSITIVITY, REVIEWER_ACCURACY_SCENARIOS),
    "24. Practical deployment regimes": (DECISION, REVIEW_FRONTIER),
    "25. Pareto analysis": (PARETO_FRONTIER, DOMINANCE_MATRIX),
    "26. Controls": (CONTROL_RESULTS,),
    "27. Negative tests": (NEGATIVE_TESTS,),
    "28. Statistics": (STATISTICAL_TESTS, NESTED_SELECTION, DESIGN_RECORD),
    "29. Sequential deployment and label reuse": (SEQUENTIAL_DEPLOYMENT,),
    "30. Limitations": (
        CAPACITY_ANALYSIS,
        REVIEWER_ACCURACY_SCENARIOS,
        SUPERVISION_FRONTIER,
        NESTED_SELECTION,
        DESIGN_RECORD,
    ),
    "31. Determinism and reproduction": (DETERMINISM, REPRODUCTION, SCORE_INVENTORY),
    "32. The development decision": (
        DECISION,
        TARGET_DOCUMENT_BUDGETS,
        CONTROL_RESULTS,
        STATISTICAL_TESTS,
    ),
    "33. Implications for the final framework": (
        DECISION,
        CONTROL_RESULTS,
        STATISTICAL_TESTS,
        REVIEW_YIELD,
        HARM_INTERCEPT,
        HARM_TOLERANCE_FRONTIER,
        AUTO_ONLY_RESULTS,
    ),
    "34. Future external confirmation": (DECISION, DESIGN_RECORD),
}


def run_record() -> int:
    """Provenance, the dependency audit and the traceability index."""
    started = time.monotonic()
    missing = [path for path in PRODUCED if not path.exists()]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {[_relative(p) for p in missing][:5]}")
    freeze = cc_read_json(RESEARCH_FREEZE)
    inputs = dict(cc_read_json(s15b.PROVENANCE)["artifacts"])
    inputs.update(cc_read_json(s15.PROVENANCE)["artifacts"])
    inputs.update({path: file_sha256(REPO / path) for path in sorted(UPSTREAM_SCRIPTS)})
    cells = pd.read_parquet(DEPLOYMENT_CELLS)
    tracked = [path for path in PRODUCED if _tracked(path)]
    unignored = [path for path in PRODUCED if not _ignored(path)]
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "issued_head": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(_git("status", "--porcelain")),
            "artifacts": {_relative(path): file_sha256(path) for path in PRODUCED},
            "artifact_count": len(PRODUCED),
            "upstream_inputs": inputs,
            "upstream_input_count": len(inputs),
            "documents": {
                _relative(REPORT): file_sha256(REPORT) if REPORT.is_file() else None,
            },
            "upstream_script_sha256_at_section_zero": freeze["upstream_script_sha256"],
            "table_rows": {
                _relative(DEPLOYMENT_CELLS): len(cells),
                _relative(DOCUMENT_COUNTS): len(pd.read_parquet(DOCUMENT_COUNTS)),
            },
            "aggregate": {
                "environments": int(cells["environment"].nunique()),
                "policies": int(cells["policy"].nunique()),
                "review_budgets": len(REVIEW_BUDGETS),
                "document_budgets": len(DOC_BUDGETS),
                "reviewer_scenarios": len(REVIEWER_SCENARIOS) + len(CONTROL_SCENARIOS),
                "draws": DOC_DRAWS,
                "epsilons": len(EPSILONS),
                "declared_prefixes": len(FINE_DEPTHS),
                "certification_documents": int(
                    cc_read_json(ENVIRONMENT_INVENTORY)["totals"]["certification_documents"]
                ),
                "certification_candidates": int(
                    cc_read_json(ENVIRONMENT_INVENTORY)["totals"]["certification_candidates"]
                ),
                "evaluation_rows": int(
                    cc_read_json(ENVIRONMENT_INVENTORY)["totals"]["evaluation_rows"]
                ),
            },
            "dependency_audit": {
                "artifacts_tracked_by_git": [_relative(path) for path in tracked],
                "artifacts_not_git_ignored": [_relative(path) for path in unignored],
                "raw_data_written": False,
                "sgv13_sgv14_sgv15_sgv15b_artifacts_written": False,
                "confirmatory_reserve_consumed": False,
                "model_refits_performed": int(cc_read_json(SCORE_INVENTORY)["fits_performed"]),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        TRACEABILITY,
        {
            **_envelope("traceability"),
            "report": _relative(REPORT),
            "rule": (
                "every number in the report is read from one of the artifacts its section names. "
                "A number that cannot be traced to one does not belong in the report."
            ),
            "audited_classes": ["decimals", "grouped integers", "percentages", "bare integers"],
            "sections": {
                name: [_relative(path) for path in paths] for name, paths in REPORT_SECTIONS.items()
            },
        },
    )
    print(
        f"record: {len(PRODUCED)} artifacts, {len(inputs)} upstream inputs, "
        f"{len(REPORT_SECTIONS)} report sections -> {_relative(PROVENANCE)}"
    )
    return 0


PHASES: tuple[tuple[str, Callable[[], int]], ...] = (
    ("reconstruct", run_reconstruct),
    ("freeze", run_freeze),
    ("capacity", run_capacity),
    ("preregister", run_preregister),
    ("geometry", run_geometry),
    ("reproduce", run_reproduce),
    ("deploy", run_deploy),
    ("auto", run_auto_only),
    ("review", run_review),
    ("frontier", run_frontier),
    ("pareto", run_pareto),
    ("sensitivity", run_sensitivity),
    ("sequential", run_sequential),
    ("select", run_select),
    ("controls", run_controls),
    ("negative", run_negative),
    ("stats", run_stats),
    ("figures", run_figures),
    ("decide", run_decide),
    ("determinism", run_determinism),
    ("record", run_record),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name, _ in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--all", action="store_true", help="run every phase in order")
    parser.add_argument(
        "--only", type=str, default=None, help="restrict an experiment to one environment"
    )
    arguments = parser.parse_args()
    requested = [name for name, _ in PHASES if getattr(arguments, name)]
    if arguments.all:
        requested = [name for name, _ in PHASES]
    if not requested:
        parser.print_help()
        return 2
    for name, phase in PHASES:
        if name not in requested:
            continue
        if name in ("reproduce", "deploy") and arguments.only:
            phase(arguments.only)  # type: ignore[call-arg]
        else:
            phase()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
