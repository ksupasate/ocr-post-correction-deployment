#!/usr/bin/env python3
"""SGV15b: was it the clustering, the coarse grid, both, or neither?

SGV15 returned NOT SUPPORTED on 2 of 6 criteria and left two specific, measurable suspects
behind rather than a general disappointment.

The first is the inference unit. SGV15's certificate was an exact Clopper-Pearson interval read
on candidates, and candidates are not independent: it measured document-cluster design effects of
7.6, 20.6 and 37.7 on its deployed environments and found the candidate-level limit sitting BELOW
the document-clustered bootstrap limit on two of the three. An exact interval for the wrong model
is not a guarantee.

The second is the action resolution. SGV15's nine declared acceptance depths cost 0.1025 repair
recall against a dense frontier -- more than the certified method delivered at its operating
point. The nine points bought a union bound over nine tests; the question is whether that trade
was worth making.

    SGV15b-P1: SGV15's certification failure is explained by its inference unit, by its action
    resolution, by both, or by neither -- and a 2x2 that varies exactly those two things, holding
    the ranking, the labels, the estimand and the evaluation blocks fixed, distinguishes the four.

**This is a mechanism stage, and a DEVELOPMENT one.** It does not try to make the model better.
It does not introduce a corrector, a feature, a loss or a model class. It changes how a risk
estimate is computed from labels that a deployment could actually buy, and it measures what that
costs and what it buys. Every environment here was already exposed by SGV14; nothing in this
stage may be called confirmation.

Nine decisions fix what the numbers below can mean.

**1. The estimand does not move.** Harm stays the candidate-weighted pooled accepted-edit rate,
`sum_d H_d / sum_d A_d`. Documents become the INFERENCE unit; they do not become the estimand.
Substituting the mean per-document rate would reweight a two-edit page against a two-hundred-edit
page and no deployment consumes risk that way. `docs/sgv15b/risk_bound_design.md` states the
reduction that makes document-level inference possible without changing what is estimated.

**2. Naming follows mathematics, not results.** A method is called a certificate only if it
carries a finite-sample distribution-free guarantee for that estimand. The cluster-robust
delta-method limit is tighter and is carried throughout, and it is labelled ASYMPTOTIC in every
artifact, may never meet a criterion on its own, and is never called certified.

**3. The ranking side is frozen shut.** SGV13's `a4_joint_refit`, SGV14's feature universe,
SGV15's fitting/certification document partition, SGV15's acquisition rule, SGV15's 250 adaptation
labels, one fit per environment. The adapted score table is computed once and reused by every arm,
which is both the honest design and the reason this stage costs minutes rather than hours.

**4. M00 must reproduce SGV15 exactly.** The candidate-level nine-depth arm is not a
reimplementation: it calls SGV15's own `evaluate_cell` through SGV15's own module, and the
reproduction gate compares the resulting cells against SGV15's frozen table. A non-zero difference
stops the stage.

**5. The fine family strictly refines the frozen one.** All nine SGV15 depths are members of the
hundred-point family, so a resolution comparison is a comparison of resolution and not of range.
The family is declared, written to `prefix_family.json` before any endpoint is read, and placed at
quantiles of the UNLABELLED certification-pool scores.

**6. The budget is documents.** Independence arises at the document level, so sample complexity is
reported in documents first and in candidate labels second. Selecting a document reveals every
eligible candidate in it, which is what keeps the documents independent and what makes the
annotation cost large enough to need its own accounting.

**7. Capacity is reported, not engineered.** The certification halves hold between 19 and 50
documents. Budgets above an environment's capacity are clipped to it and recorded as clipped; no
environment is dropped, no row is duplicated, and no nominal budget is claimed as achieved.

**8. The bounds are validated by simulation before they are believed.** Section 23's clustered
simulation runs first and its verdict is pre-registered: a bound that over-certifies under its own
declared assumptions is demoted to a diagnostic, and the demotion is reported.

**9. Duplication is the falsification.** Duplicating every candidate inside every certification
document adds no independent document. A document-aware certificate must be exactly invariant; a
candidate-level one becomes measurably more confident. The stage runs this on real certification
samples and saves the result.

    --reconstruct  the section-0 freeze, including SGV15's verification state
    --freeze       the frozen SGV15 configuration this stage may not vary
    --preregister  the prefix families, document budgets, arms, criteria and selection plan
    --simulate     the clustered-data validity gate for the bounds and for the prefix controls
    --reproduce    the M00 integrity gate against SGV15's frozen cells
    --factorial    the 2x2, its declared baselines, its diagnostics and its oracles
    --duplicate    the document-duplication falsification on real certification samples
    --cluster      intra-document dependence, design effects and effective sample size
    --oracle       O0/O1/O2 and how much of SGV15's grid loss the finer family recovers
    --results      the 2x2 tables, the per-arm tables and the failure taxonomy
    --complexity   sample complexity in documents and in candidate labels, and annotation cost
    --controls     the twelve required controls
    --negative     the falsification tests
    --select       nested leave-environment-out selection
    --stats        the pre-registered tests
    --figures      the thirteen required figures
    --decide       the machine-readable finding
    --determinism  two independent regenerations of the selected configuration
    --record       provenance, the dependency audit and the traceability index

DEVELOPMENT ONLY. The SGV1 CORD confirmatory reserve stays LOCKED and is absent from every
artifact. SGV13, SGV14 and SGV15 artifacts are read and hash-verified; none is modified.
"""

from __future__ import annotations

import argparse
import math
import subprocess
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
from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.risk import clopper_pearson_upper
from ocr_risk.risk.cluster_bounds import (
    cluster_ratio_bound,
    cluster_ratio_upper,
    design_effect,
)
from ocr_risk.risk.prefix_control import prefix_thresholds, select_prefix

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv15b_document_level_certification"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_sgv15_configuration.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"
DOCUMENT_PARTITION = OUT / "document_partition.json"
DOCUMENT_CAPACITY = OUT / "document_capacity.json"
DESIGN_RECORD = OUT / "design_record.json"
PREFIX_FAMILY = OUT / "prefix_family.json"
SIMULATION_COVERAGE = OUT / "simulation_coverage.json"
PREFIX_CONTROL_VALIDATION = OUT / "prefix_control_validation.json"
SGV15_REPRODUCTION = OUT / "sgv15_reproduction.json"
FACTORIAL_CELLS = OUT / "factorial_cells.parquet"
DOCUMENT_COUNTS = OUT / "document_counts.parquet"
DOCUMENT_DUPLICATION_TEST = OUT / "document_duplication_test.json"
CLUSTER_STRUCTURE = OUT / "cluster_structure.json"
DESIGN_EFFECTS = OUT / "design_effects.json"
EFFECTIVE_SAMPLE_SIZE = OUT / "effective_sample_size.json"
ORACLE_RESULTS = OUT / "oracle_results.json"
GRID_RESOLUTION_ANALYSIS = OUT / "grid_resolution_analysis.json"
FACTORIAL_RESULTS = OUT / "factorial_2x2_results.json"
CANDIDATE_NINE_RESULTS = OUT / "candidate_9grid_results.json"
DOCUMENT_NINE_RESULTS = OUT / "document_9grid_results.json"
CANDIDATE_FINE_RESULTS = OUT / "candidate_fine_results.json"
DOCUMENT_FINE_RESULTS = OUT / "document_fine_results.json"
SAMPLE_COMPLEXITY_DOCUMENTS = OUT / "sample_complexity_documents.json"
SAMPLE_COMPLEXITY_CANDIDATES = OUT / "sample_complexity_candidates.json"
ANNOTATION_COST = OUT / "annotation_cost.json"
CONTROL_RESULTS = OUT / "control_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
NESTED_SELECTION = OUT / "nested_selection.json"
DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

SCHEMA_VERSION = 1
STAGE = "sgv15b_document_level_certification"
HYPOTHESIS = "SGV15b-P1"

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
deploy = s15.deploy
fit_adaptation = s15.fit_adaptation
classify = s15.classify
holds_the_bound = s15.holds_the_bound
_mean = s15._mean
holm = s15.holm
per_document_counts = s13.per_document_counts
epsilon_key = s13.epsilon_key
ranking_quality = s13.ranking_quality
frontier = s13.frontier

EPSILONS = s15.EPSILONS
PRIMARY_EPSILON = s15.PRIMARY_EPSILON
BOOTSTRAP_RESAMPLES = s13.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = s13.BOOTSTRAP_SEED

F1_NO_FEASIBLE_CUT = s15.F1_NO_FEASIBLE_CUT
F2_DEGENERATE = s15.F2_DEGENERATE
F3_NON_DEGENERATE = s15.F3_NON_DEGENERATE
F4_FALSE_CERTIFICATION = s15.F4_FALSE_CERTIFICATION
FAILURE_MODES = s15.FAILURE_MODES

# ------------------------------------------------------------------ the pre-registered registry

SAMPLE_SEED = 20260908
SIMULATION_SEED = 815

# Section 12's budget: documents, because that is where independence lives. The certification
# halves hold between 19 and 50 documents, so 25 is the deepest budget every environment can
# supply something for and 50 is the deepest FUNSD can. Budgets above an environment's capacity
# are clipped to it and recorded as clipped.
DOC_BUDGETS = (5, 10, 15, 20, 25, 50)
DOC_DRAWS = 20
PRACTICAL_DOC_BUDGET = 25

# Section 7's two factors.
UNIT_CANDIDATE = "candidate"
UNIT_DOCUMENT = "document"
UNIT_DOCUMENT_ASYMPTOTIC = "document_asymptotic"
UNIT_NAIVE = "naive_empirical"

FAMILY_NINE = "nine"
FAMILY_FINE = "fine"

CONTROL_UNION = "union"
CONTROL_FIXED_SEQUENCE = "fixed_sequence"

# SGV15's frozen nine acceptance depths, imported rather than restated.
NINE_DEPTHS = s15.CUT_GRID
# Section 9's finer family: a hundred nested prefixes on a deterministic 0.005 ladder. It is a
# strict REFINEMENT -- every one of the nine frozen depths is a member -- so a comparison between
# the two families is a comparison of resolution and not of range.
FINE_DEPTHS = tuple(round(0.005 * step, 3) for step in range(1, 101))

ALPHA = 0.05

M00 = "m00_candidate_nine_union"
M10 = "m10_document_nine_union"
M01 = "m01_candidate_fine_sequence"
M11 = "m11_document_fine_sequence"
P1A = "p1a_candidate_fine_union"
P1B = "p1b_document_fine_union"
D0 = "d0_document_asymptotic_nine"
D1 = "d1_document_asymptotic_fine"
N0 = "n0_naive_empirical_fine"

O0_UNRESTRICTED = "o0_unrestricted_oracle"
O1_NINE = "o1_nine_depth_oracle"
O2_FINE = "o2_fine_family_oracle"

ARMS: tuple[tuple[str, str, str, str], ...] = (
    (M00, UNIT_CANDIDATE, FAMILY_NINE, CONTROL_UNION),
    (M10, UNIT_DOCUMENT, FAMILY_NINE, CONTROL_UNION),
    (M01, UNIT_CANDIDATE, FAMILY_FINE, CONTROL_FIXED_SEQUENCE),
    (M11, UNIT_DOCUMENT, FAMILY_FINE, CONTROL_FIXED_SEQUENCE),
    (P1A, UNIT_CANDIDATE, FAMILY_FINE, CONTROL_UNION),
    (P1B, UNIT_DOCUMENT, FAMILY_FINE, CONTROL_UNION),
    (D0, UNIT_DOCUMENT_ASYMPTOTIC, FAMILY_NINE, CONTROL_UNION),
    (D1, UNIT_DOCUMENT_ASYMPTOTIC, FAMILY_FINE, CONTROL_FIXED_SEQUENCE),
    (N0, UNIT_NAIVE, FAMILY_FINE, CONTROL_UNION),
)
ARM_NAMES = tuple(name for name, _, _, _ in ARMS)
FACTORIAL = (M00, M10, M01, M11)
# Measured everywhere, selectable nowhere. The two asymptotic arms carry a limit theorem and not
# a bound, and the naive arm is a registered negative; selecting either would let a diagnostic
# become the deployed rule, which is the defect SGV15 caught in its own selection pool.
SELECTABLE_ARMS = (M00, M10, M01, M11, P1A, P1B)
NOT_SELECTABLE_ARMS = (D0, D1, N0, O0_UNRESTRICTED, O1_NINE, O2_FINE)
ORACLES = (O0_UNRESTRICTED, O1_NINE, O2_FINE)

BREADTH_REQUIRED = 7
MAJORITY_REQUIRED = 6
ENVIRONMENT_COUNT = 10
# Section 30's criterion 5 threshold, fixed before any oracle is computed: the finer family must
# recover at least half of the frozen family's measured grid loss to count as load-bearing.
RESOLUTION_RECOVERY_REQUIRED = 0.5

SIMULATION_TRIALS = 2000
SIMULATION_DOCUMENTS = (10, 20, 40)
SIMULATION_CORRELATIONS = ("iid", "weak", "moderate", "strong")
FWER_TRIALS = 2000
# Extended after the first run of this simulation: at 20 and 40 documents every procedure refused
# on essentially every trial, which makes a familywise error rate unmeasurable rather than small.
# The larger counts are a resolution choice about the VALIDATION and change no method, no
# criterion and no real-data design; the amendment is recorded in `prefix_control_validation.json`.
FWER_DOCUMENTS = (20, 40, 100, 200, 400)


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv15b-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": _iso(),
    }


# ------------------------------------------------------------- section 0: the research freeze
#
# SGV15b may not start until SGV15 is closed, and "closed" is checked against SGV15's artifacts
# rather than assumed from a prompt. Every path SGV15's provenance names is re-hashed here.

SGV15_GRID_RESOLUTION = s15.OUT / "grid_resolution_analysis.json"

SGV15_VERDICT = "NOT SUPPORTED"
SGV15_CRITERIA_MET = 2
SGV15_CRITERIA_TOTAL = 6
SGV15_OUTCOME = "D"

UPSTREAM_SCRIPTS = (
    "scripts/sgv15b_document_level_certification.py",
    "scripts/sgv15_target_risk_certification.py",
    *s14.DEPENDENCY_SCRIPTS,
)


def verify_sgv15() -> dict[str, Any]:
    """SGV15's artifacts, re-hashed against the record SGV15's own provenance wrote.

    A stage whose entire motivation is another stage's failure has to be sure it is reading that
    stage's actual numbers, and that nobody quietly improved them in between.
    """
    if not s15.PROVENANCE.is_file():
        raise PhaseError("SGV15's provenance.json is missing; SGV15 is not closed")
    record = cc_read_json(s15.PROVENANCE)
    decision = cc_read_json(s15.DECISION)

    verdict = str(decision["verdict"])
    met = int(decision["criteria_met"])
    total = int(decision["criteria_total"])
    outcome = str(decision["outcome"]["label"])
    if (verdict, met, total, outcome) != (
        SGV15_VERDICT,
        SGV15_CRITERIA_MET,
        SGV15_CRITERIA_TOTAL,
        SGV15_OUTCOME,
    ):
        raise PhaseError(
            f"SGV15's frozen verdict is {verdict} {met}/{total} outcome {outcome}, not the state "
            "SGV15b was written against. Repository evidence is authoritative: resolve the "
            "discrepancy before running SGV15b."
        )
    if bool(decision["ready_for_sgv16"]):
        raise PhaseError("SGV15 declared itself ready for SGV16; SGV15b's premise does not hold")
    if bool(decision["confirmatory_reserve_consumed"]):
        raise PhaseError("the confirmatory reserve has been spent; SGV15b may not proceed")

    drifted = [
        path
        for path, digest in record["artifacts"].items()
        if (REPO / path).is_file() and file_sha256(REPO / path) != digest
    ]
    missing = [path for path in record["artifacts"] if not (REPO / path).is_file()]
    if drifted or missing:
        raise PhaseError(
            f"SGV15 artifacts moved since it was frozen: {len(drifted)} changed, "
            f"{len(missing)} missing. {sorted(drifted + missing)[:4]}"
        )

    sgv14 = cc_read_json(s14.DECISION)
    if str(sgv14["verdict"]) != s15.SGV14_VERDICT or int(sgv14["criteria_met"]) != (
        s15.SGV14_CRITERIA_MET
    ):
        raise PhaseError("SGV14's frozen verdict moved; SGV15b may not modify or depend on a drift")

    return {
        "verdict": verdict,
        "criteria_met": met,
        "criteria_total": total,
        "outcome": outcome,
        "ready_for_sgv16": bool(decision["ready_for_sgv16"]),
        "confirmatory_reserve_consumed": bool(decision["confirmatory_reserve_consumed"]),
        "artifacts_verified": len(record["artifacts"]),
        "determinism": {
            "runs": int(cc_read_json(s15.DETERMINISM)["runs_completed"]),
            "all_runs_identical": bool(cc_read_json(s15.DETERMINISM)["all_runs_identical"]),
            "max_absolute_numeric_difference": float(
                cc_read_json(s15.DETERMINISM)["max_absolute_numeric_difference_over_all_runs"]
            ),
        },
        "sgv14_verdict": str(sgv14["verdict"]),
        "sgv14_criteria_met": int(sgv14["criteria_met"]),
        "sgv14_criteria_total": int(sgv14["criteria_total"]),
    }


def _verification_state() -> dict[str, Any]:
    """The gates, run now rather than quoted from a prompt. Slow ones are named, not run."""
    lint = subprocess.run(
        ["uv", "run", "ruff", "check", "src", "tests", "scripts"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    typing = subprocess.run(
        ["uv", "run", "mypy"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return {
        "lint_passed": lint.returncode == 0,
        "mypy_passed": typing.returncode == 0,
        "mypy_summary": typing.stdout.strip().splitlines()[-1] if typing.stdout.strip() else "",
        "suite": (
            "`make test` and `make audit` are run outside this phase because they take minutes; "
            "their observed state is recorded in the stage report and in the research decision."
        ),
    }


def run_reconstruct() -> int:
    """Verify the frozen state SGV15b builds on, and refuse to start if it has moved."""
    started = time.monotonic()
    sgv15 = verify_sgv15()
    payload = {
        **_envelope("research_freeze"),
        "issued_head": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "sgv15": sgv15,
        "sgv15_findings_this_stage_is_built_on": {
            "design_effects": [
                float(record["design_effect_on_the_harm_rate"])
                for record in cc_read_json(s15.RISK_BOUND_RESULTS)["per_environment"].values()
                if record.get("deployed")
            ],
            "environments_where_the_candidate_bound_is_anti_conservative": cc_read_json(
                s15.RISK_BOUND_RESULTS
            )["environments_where_the_candidate_bound_is_anti_conservative"],
            "grid_resolution_gap": float(
                cc_read_json(SGV15_GRID_RESOLUTION)["mean_grid_resolution_gap"]
            ),
            "environments_where_the_grid_costs_recall": int(
                cc_read_json(SGV15_GRID_RESOLUTION)["environments_where_the_grid_costs_recall"]
            ),
        },
        "verification": _verification_state(),
        "upstream_script_sha256": {
            path: file_sha256(REPO / path) for path in sorted(UPSTREAM_SCRIPTS)
        },
        "confirmatory_reserve": "LOCKED. SGV15b is a development stage and may not spend it.",
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(RESEARCH_FREEZE, payload)
    print(
        f"reconstruct: SGV15 {sgv15['verdict']} {sgv15['criteria_met']}/"
        f"{sgv15['criteria_total']} outcome {sgv15['outcome']}, "
        f"{sgv15['artifacts_verified']} artifacts re-hashed -> {_relative(RESEARCH_FREEZE)}"
    )
    return 0


# ------------------------------------------------------------------ the frozen configuration


def run_freeze() -> int:
    """Everything SGV15b inherits and may not vary, recovered from the frozen artifacts."""
    started = time.monotonic()
    if not RESEARCH_FREEZE.is_file():
        raise PhaseError("run --reconstruct first")
    frozen = s15.load_frozen()
    s15.assert_frozen(frozen)
    sgv15_design = cc_read_json(s15.DESIGN_RECORD)
    payload = {
        **_envelope("frozen_sgv15_configuration"),
        "frozen_ranking": {
            "arm": frozen.arm,
            "acquisition": frozen.acquisition,
            "adaptation_labels": frozen.adapt_budget,
            "source_to_target_weighting": {"source_rows": 1.0, "target_rows": frozen.target_weight},
            "regularization": (
                "SGV5's selected model class and lambda per fold; the refit adds none of its own"
            ),
            "feature_universe": "SGV1's 96 design columns plus SGV5's 24 structural columns",
            "candidate_universe": "SGV1's frozen g8_union generator ladder",
            "label_definitions": "edits/outcome.py, harm policy strict_worsening",
            "score_orientation": "higher is safer; a prefix accepts scores at or above its cut",
            "fits_per_environment": 1,
        },
        "frozen_partition": {
            "source": _relative(s15.CERTIFICATION_ROWS),
            "sha256": file_sha256(s15.CERTIFICATION_ROWS),
            "rule": (
                "SGV15's own fitting / certification split of SGV14's adaptation half, reused "
                "unchanged. Whole documents, whole volumes on OCR-D-SBB, content-blind hash "
                "under SGV15's seed. SGV15b introduces no new partition and weakens no separation."
            ),
            "partition_seed": s15.PARTITION_SEED,
        },
        "inherited_coverage_floor": frozen.coverage_floor,
        "inherited_primary_epsilon": frozen.epsilon,
        "inherited_cut_grid": list(NINE_DEPTHS),
        "sgv15_multiplicity": sgv15_design["cut_grid"]["multiplicity"],
        "what_sgv15b_varies": [
            "the statistical unit the risk estimate is made on",
            "the declared prefix family and how the confidence level is spread across it",
            "the certification budget, counted in documents",
        ],
        "what_sgv15b_may_not_vary": [
            "the adaptation arm, its weighting, its regularisation or its feature universe",
            "the candidate universe, the harm definition or the label definitions",
            "the primary epsilon",
            "the fitting / certification / evaluation document partition",
            "any SGV13, SGV14 or SGV15 artifact",
        ],
        "configuration_sha256": frozen.configuration_sha256,
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(FROZEN_CONFIGURATION, payload)
    print(
        f"freeze: {frozen.arm} + {frozen.acquisition} + N_adapt={frozen.adapt_budget}, "
        f"coverage floor {frozen.coverage_floor:.6f} -> {_relative(FROZEN_CONFIGURATION)}"
    )
    return 0


# --------------------------------------------------------- section 12: what capacity there is
#
# Independence lives at the document level, so the budget is documents and the constraint is how
# many independent certification documents an environment actually has. This phase reads SGV15's
# frozen partition and reports what it holds. Nothing here is engineered to be balanced.


def _certification_table() -> pd.DataFrame:
    """SGV15's frozen fitting / certification assignment, joined to SGV14's rows."""
    roles = pd.read_parquet(s15.CERTIFICATION_ROWS)
    roles = roles[roles["sgv15_role"] == s15.ROLE_CERT]
    rows = pd.read_parquet(s14.ENVIRONMENT_ROWS)
    rows = rows[rows["role"] == s14.ROLE_ADAPTATION]
    merged = rows.merge(
        roles[["environment", "candidate_id"]].astype(str),
        on=["environment", "candidate_id"],
        how="inner",
    )
    return merged


def run_capacity() -> int:
    """The document inventory, the capacity limits, and which budgets each environment can meet."""
    started = time.monotonic()
    if not FROZEN_CONFIGURATION.is_file():
        raise PhaseError("run --freeze first")
    certification = _certification_table()
    sgv15_inventory = {
        record["environment"]: record
        for record in cc_read_json(s15.ENVIRONMENT_INVENTORY)["environments"]
    }

    records: list[dict[str, Any]] = []
    partition: dict[str, Any] = {}
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        block = certification[certification["environment"] == name]
        per_document = block.groupby("document_id", observed=True).size()
        documents = sorted(str(d) for d in per_document.index.tolist())
        sizes = np.asarray(sorted(per_document.to_numpy().tolist()), dtype=float)
        available = len(documents)
        clipped = [budget for budget in DOC_BUDGETS if budget > available]
        records.append(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "base_engine": spec["base_engine"],
                "certification_documents": available,
                "certification_candidates": len(block),
                "candidates_per_document_mean": float(sizes.mean()) if sizes.size else 0.0,
                "candidates_per_document_median": float(np.median(sizes)) if sizes.size else 0.0,
                "candidates_per_document_min": int(sizes.min()) if sizes.size else 0,
                "candidates_per_document_max": int(sizes.max()) if sizes.size else 0,
                "certification_harmful": int(block["is_harmful"].sum()),
                "evaluation_rows": int(sgv15_inventory[name]["evaluation_rows"]),
                "evaluation_documents": int(sgv15_inventory[name]["evaluation_documents"]),
                "sgv15_capacity_limited": bool(sgv15_inventory[name]["capacity_limited"]),
                "largest_feasible_document_budget": max(
                    [budget for budget in DOC_BUDGETS if budget <= available] or [0]
                ),
                "clipped_budgets": clipped,
                "capacity_limited": bool(clipped),
            }
        )
        partition[name] = {
            "certification_documents": available,
            "document_hash": canonical_hash({"documents": documents}),
        }

    limited = [record["environment"] for record in records if record["capacity_limited"]]
    inventory = {
        **_envelope("environment_inventory"),
        "environments": records,
        "capacity_limited": limited,
        "totals": {
            "environments": len(records),
            "certification_documents": int(sum(r["certification_documents"] for r in records)),
            "certification_candidates": int(sum(r["certification_candidates"] for r in records)),
        },
    }
    _write_json_once(ENVIRONMENT_INVENTORY, inventory)

    _write_json_once(
        DOCUMENT_PARTITION,
        {
            **_envelope("document_partition"),
            "rule": (
                "SGV15's frozen fitting / certification split, reused byte-for-byte. SGV15b "
                "creates no partition of its own, so adaptation documents, certification "
                "documents and evaluation documents remain the three disjoint sets SGV14 and "
                "SGV15 established."
            ),
            "source": _relative(s15.CERTIFICATION_ROWS),
            "sha256": file_sha256(s15.CERTIFICATION_ROWS),
            "seed": s15.PARTITION_SEED,
            "unit": "document, or volume on OCR-D-SBB",
            "per_environment": partition,
        },
    )

    _write_json_once(
        DOCUMENT_CAPACITY,
        {
            **_envelope("document_capacity"),
            "declared_budgets": list(DOC_BUDGETS),
            "clipping_rule": (
                "a budget larger than an environment's certification pool is clipped to the pool "
                "and the cell records both the requested and the achieved count. No row is "
                "duplicated, no environment is dropped, and no nominal budget is reported as "
                "achieved when it was not."
            ),
            "feasible_environments_by_budget": {
                str(budget): [
                    record["environment"]
                    for record in records
                    if record["certification_documents"] >= budget
                ]
                for budget in DOC_BUDGETS
            },
            "capacity_limited": limited,
            "practical_budget_ceiling": PRACTICAL_DOC_BUDGET,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"capacity: {len(records)} environments, "
        f"{inventory['totals']['certification_documents']} certification documents, "
        f"{len(limited)} cannot supply every declared budget -> {_relative(DOCUMENT_CAPACITY)}"
    )
    return 0


# ------------------------------------------------------------------- the pre-registration


def run_preregister() -> int:
    """Freeze the families, budgets, arms, criteria and the selection plan before any endpoint."""
    started = time.monotonic()
    if not ENVIRONMENT_INVENTORY.is_file():
        raise PhaseError("run --capacity before pre-registering")
    frozen = s15.load_frozen()

    _write_json_once(
        PREFIX_FAMILY,
        {
            **_envelope("prefix_family"),
            "families": {
                FAMILY_NINE: {
                    "depths": list(NINE_DEPTHS),
                    "size": len(NINE_DEPTHS),
                    "source": "SGV15's frozen declared cut grid, imported and not restated",
                    "control": CONTROL_UNION,
                },
                FAMILY_FINE: {
                    "depths": list(FINE_DEPTHS),
                    "size": len(FINE_DEPTHS),
                    "source": (
                        "a deterministic 0.005 ladder from 0.005 to 0.500. It is a strict "
                        "REFINEMENT of the frozen nine: every one of those depths is a member, "
                        "so a comparison between the families is a comparison of resolution and "
                        "not of range."
                    ),
                    "control": CONTROL_FIXED_SEQUENCE,
                    "refines_the_frozen_family": all(d in FINE_DEPTHS for d in NINE_DEPTHS),
                },
            },
            "placement": (
                "quantiles of the UNLABELLED certification-pool scores under the frozen adapted "
                "model, so a prefix location depends on the ranking and the declared depth alone"
            ),
            "may_depend_on": [
                "the number of candidates",
                "the ranking order",
                "declared fractions",
            ],
            "may_not_depend_on": [
                "target outcome labels",
                "evaluation harm",
                "the oracle cut",
                "any SGV15b endpoint",
            ],
            "declared_before": "any SGV15b endpoint evaluation",
            "family_hash": canonical_hash({"nine": list(NINE_DEPTHS), "fine": list(FINE_DEPTHS)}),
        },
    )

    payload = {
        **_envelope("design_record"),
        "issued_head": _git("rev-parse", "HEAD"),
        "hypothesis": (
            "SGV15's certification failure is explained by its inference unit, by its action "
            "resolution, by both, or by neither"
        ),
        "kind": "DEVELOPMENT",
        "alpha": ALPHA,
        "primary_epsilon": PRIMARY_EPSILON,
        "secondary_epsilons": [e for e in EPSILONS if e != PRIMARY_EPSILON],
        "epsilon_rule": (
            "the secondary tolerances are reported and may not rescue a failure at the primary "
            "one; no criterion reads them"
        ),
        "estimand": {
            "operational": "candidate-weighted accepted-edit harm, sum_d H_d / sum_d A_d",
            "inference_unit": "the document, or the volume on OCR-D-SBB",
            "not_used": (
                "the mean per-document harm rate. It is a different population quantity and no "
                "deployment consumes risk that way."
            ),
            "design_note": "docs/sgv15b/risk_bound_design.md",
        },
        "factors": {
            "A_certification_unit": {
                UNIT_CANDIDATE: "SGV15's exact Clopper-Pearson interval on accepted candidates",
                UNIT_DOCUMENT: (
                    "the union at alpha/3 of the Hoeffding, Maurer-Pontil empirical-Bernstein and "
                    "Waudby-Smith-Ramdas betting bounds on the per-document ratio null. "
                    "Finite-sample, distribution-free."
                ),
            },
            "B_prefix_control": {
                FAMILY_NINE: "SGV15's nine declared depths under a union bound at alpha/9",
                FAMILY_FINE: (
                    "one hundred declared depths under fixed-sequence gatekeeping at the full "
                    "alpha, shallow to deep, stopping at the first prefix that cannot be certified"
                ),
            },
        },
        "arms": {
            name: {
                "unit": unit,
                "family": family,
                "control": control,
                "selectable": name in SELECTABLE_ARMS,
                "guarantee": (
                    "asymptotic"
                    if unit == UNIT_DOCUMENT_ASYMPTOTIC
                    else ("none" if unit == UNIT_NAIVE else "finite-sample")
                ),
            }
            for name, unit, family, control in ARMS
        },
        "arms_measured_but_never_selectable": {
            D0: "an asymptotic limit theorem, not a certificate",
            D1: "an asymptotic limit theorem, not a certificate",
            N0: "a registered negative: the observed rate taken at face value",
            O0_UNRESTRICTED: "ANALYSIS ONLY: reads evaluation labels",
            O1_NINE: "ANALYSIS ONLY: reads evaluation labels",
            O2_FINE: "ANALYSIS ONLY: reads evaluation labels",
        },
        "budgets": {
            "unit": "documents",
            "declared": list(DOC_BUDGETS),
            "draws": DOC_DRAWS,
            "annotation_rule": (
                "selecting a certification document reveals every eligible candidate in it. That "
                "is what keeps the documents independent, and it is why the candidate-label cost "
                "is reported separately."
            ),
            "within_document_subsampling": (
                "REGISTERED AND NOT RUN. Section 14 permits a fixed-cost within-document arm only "
                "if its sampling design is reflected in the bound before evaluation. No such "
                "inverse-probability-weighted bound is established here, so the arm is not run "
                "rather than run without one."
            ),
            "practical_ceiling": PRACTICAL_DOC_BUDGET,
        },
        "non_degeneracy": {
            "coverage_floor": frozen.coverage_floor,
            "source": "SGV13's own floor, inherited unchanged through SGV14 and SGV15",
            "rule": (
                "a deployment is non-degenerate when it is feasible, accepts at least one edit, "
                "and reaches the frozen coverage floor on the evaluation block"
            ),
        },
        "failure_taxonomy": {
            F1_NO_FEASIBLE_CUT: "the method refused: no prefix could be certified",
            F2_DEGENERATE: "certified, but coverage falls below the frozen floor",
            F3_NON_DEGENERATE: "certified, safe, and above the floor",
            F4_FALSE_CERTIFICATION: (
                "certified, and realised harm on disjoint evaluation exceeds epsilon"
            ),
        },
        "criteria": {
            "1_safety": (
                f"the selected method holds realised harm at or below {PRIMARY_EPSILON} on at "
                f"least {BREADTH_REQUIRED} of {ENVIRONMENT_COUNT} held-out development environments"
            ),
            "2_non_degenerate_safe_deployment": (
                f"at least {BREADTH_REQUIRED} of {ENVIRONMENT_COUNT} environments reach F3"
            ),
            "3_operational_improvement": (
                "risk-controlled repair recall improves on SGV15's certified baseline "
                f"(m2_certified_clopper_pearson) on at least {BREADTH_REQUIRED} of "
                f"{ENVIRONMENT_COUNT}"
            ),
            "4_cluster_awareness_is_load_bearing": (
                "with the prefix family held fixed, all three must hold: the duplication "
                "falsification passes for the document unit and fails for the candidate unit; the "
                "document-aware limit is at least as wide as the candidate-level limit on at "
                f"least {BREADTH_REQUIRED} of {ENVIRONMENT_COUNT}; and the candidate unit's "
                "false-certification count exceeds the document unit's. Better recall is NOT "
                "required."
            ),
            "5_fine_prefix_control_is_load_bearing": (
                "with the inference unit held fixed, the finer family recovers at least "
                f"{RESOLUTION_RECOVERY_REQUIRED:.0%} of the frozen family's measured oracle grid "
                "loss, and the fine-prefix arm's mean risk-controlled repair recall exceeds the "
                "nine-point arm's"
            ),
            "6_false_certification_control": (
                f"the selected method's false-certification rate over its cells is at most alpha "
                f"= {ALPHA} and does not exceed M00's"
            ),
            "7_practical_sample_complexity": (
                f"some declared budget of at most {PRACTICAL_DOC_BUDGET} documents gives safe "
                f"non-degenerate deployment on at least {MAJORITY_REQUIRED} of "
                f"{ENVIRONMENT_COUNT} environments"
            ),
        },
        "criteria_total": 7,
        "breadth_required": BREADTH_REQUIRED,
        "majority_required": MAJORITY_REQUIRED,
        "resolution_recovery_required": RESOLUTION_RECOVERY_REQUIRED,
        "outcome_taxonomy": {
            "A": "clustering was the dominant bottleneck",
            "B": "prefix resolution was the dominant bottleneck",
            "C": "both mechanisms are load-bearing",
            "D": "neither fixes deployment at practical document budgets",
        },
        "selection_plan": (
            "leave-one-environment-out inside the ten development environments. For each held-out "
            "environment the arm and the document budget are chosen on the remaining nine and "
            "frozen before the held-out endpoint is read. The pooled maximum is reported "
            "separately and is never described as unbiased."
        ),
        "selection_pool": list(SELECTABLE_ARMS),
        "environments": [spec["environment"] for spec in s14.ENVIRONMENTS],
        "seeds": {
            "sample_seed": SAMPLE_SEED,
            "simulation_seed": SIMULATION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "document_order_for_the_betting_bound": (
                "ascending document identifier: the betting bound's plug-in is predictable, so "
                "the order must be fixed before the data are read"
            ),
        },
        "prohibited": [
            "externally confirmed",
            "generally validated",
            "safe deployment solved",
            "universal reliability achieved",
        ],
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DESIGN_RECORD, payload)
    print(
        f"preregister: {len(ARMS)} arms, {len(NINE_DEPTHS)} + {len(FINE_DEPTHS)} declared "
        f"prefixes, budgets {list(DOC_BUDGETS)}, 7 criteria -> {_relative(DESIGN_RECORD)}"
    )
    return 0


# ------------------------------------------------------------------ the certification geometry
#
# Everything a cell needs that does not depend on the draw is computed once per environment: one
# adaptation fit, both declared families placed on the unlabelled certification-pool scores, and
# the per-document acceptance counts at every prefix. Section 42's rule, applied: the fit depends
# on the environment and the seed and on nothing else, so it happens once and never inside a loop
# over epsilon, prefixes or draws.


@dataclass(frozen=True, slots=True)
class Placed:
    """One environment's frozen ranking, with both prefix families placed and counted.

    `accepted` and `span` are LABEL-FREE: they count candidates whose adapted score clears a
    prefix, and the adapted score is fitted on the disjoint fitting half. `span` in particular is
    the design constant the document-level bound needs, and it is read off the whole certification
    pool rather than off a draw, so it does not move with the sample.
    """

    environment: Any
    adapted: Any
    documents: np.ndarray
    doc_sizes: np.ndarray
    doc_harm: np.ndarray
    thresholds: dict[str, np.ndarray]
    accepted: dict[str, np.ndarray]
    harmful: dict[str, np.ndarray]
    span: dict[str, np.ndarray]
    ranking: dict[str, float]


FAMILIES: dict[str, tuple[float, ...]] = {FAMILY_NINE: NINE_DEPTHS, FAMILY_FINE: FINE_DEPTHS}


def place(environment: Any) -> Placed:
    """Fit SGV15's frozen adaptation arm once, then place and count both declared families."""
    seed = _stable_seed("sgv15-adapt", str(s15.SAMPLE_SEED), environment.name, s15.A_UNCERTAINTY)
    adapted = fit_adaptation(environment, s15.A_UNCERTAINTY, s15.ADAPT_BUDGET, seed)
    scores = np.asarray(adapted.cert_scores, dtype=float)
    documents = np.asarray(
        [str(name) for name in environment.cert_documents.tolist()], dtype=object
    )
    harm = np.asarray(environment.cert_harmful, dtype=bool)
    names = np.asarray(sorted(set(documents.tolist())), dtype=object)
    masks = [documents == name for name in names.tolist()]

    thresholds: dict[str, np.ndarray] = {}
    accepted: dict[str, np.ndarray] = {}
    harmful: dict[str, np.ndarray] = {}
    span: dict[str, np.ndarray] = {}
    for family, depths in FAMILIES.items():
        taus = prefix_thresholds(scores, depths)
        counts = np.zeros((names.size, len(depths)), dtype=np.int64)
        harms = np.zeros_like(counts)
        for index, mask in enumerate(masks):
            ascending = np.sort(scores[mask])
            counts[index] = ascending.size - np.searchsorted(ascending, taus, side="left")
            ascending_harm = np.sort(scores[mask & harm])
            harms[index] = ascending_harm.size - np.searchsorted(ascending_harm, taus, side="left")
        thresholds[family] = taus
        accepted[family] = counts
        harmful[family] = harms
        span[family] = counts.max(axis=0).astype(float)

    return Placed(
        environment=environment,
        adapted=adapted,
        documents=names,
        doc_sizes=np.asarray([int(mask.sum()) for mask in masks], dtype=np.int64),
        doc_harm=np.asarray([int((mask & harm).sum()) for mask in masks], dtype=np.int64),
        thresholds=thresholds,
        accepted=accepted,
        harmful=harmful,
        span=span,
        ranking=s15._ranking_row(environment, adapted),
    )


def document_sample(placed: Placed, size: int, seed: int) -> np.ndarray:
    """Which certification documents this draw buys, clipped to what the environment has.

    A uniform draw over DOCUMENTS, not over rows. Every eligible candidate inside a chosen
    document is then revealed, which is what keeps the drawn units independent.
    """
    available = int(placed.documents.size)
    take = min(int(size), available)
    if take <= 0:
        return np.zeros(0, dtype=int)
    generator = np.random.default_rng(seed)
    return np.sort(generator.permutation(available)[:take])


def certifier(
    placed: Placed, family: str, unit: str, rows: np.ndarray, epsilon: float
) -> Callable[[int, float], bool]:
    """The per-prefix test for one inference unit, as a callback the control procedure drives.

    The callback closes over the sampled documents and nothing else, so no arm can see a row it
    did not buy, and the prefix-control procedure never learns which unit it is serving.
    """
    accepted = placed.accepted[family][rows]
    harmful = placed.harmful[family][rows]
    span = placed.span[family]

    def certifies(index: int, level: float) -> bool:
        total = int(accepted[:, index].sum())
        harm = int(harmful[:, index].sum())
        if unit == UNIT_CANDIDATE:
            return bool(total > 0 and clopper_pearson_upper(harm, total, level) <= epsilon)
        if unit == UNIT_NAIVE:
            return bool(total > 0 and harm / total <= epsilon)
        method = "union_finite" if unit == UNIT_DOCUMENT else "cluster_robust_asymptotic"
        return cluster_ratio_bound(
            harmful[:, index].astype(float),
            accepted[:, index].astype(float),
            epsilon,
            delta=level,
            span=float(span[index]),
            method=method,
        ).certifies

    return certifies


def deployed_bound(
    placed: Placed,
    family: str,
    unit: str,
    rows: np.ndarray,
    index: int,
    level: float,
    epsilon: float,
) -> tuple[float, float]:
    """The certificate the deployed prefix carries: (ratio limit, limit on E[Z])."""
    total = int(placed.accepted[family][rows, index].sum())
    harm = int(placed.harmful[family][rows, index].sum())
    if unit in (UNIT_CANDIDATE, UNIT_NAIVE):
        return float(clopper_pearson_upper(harm, total, level)), float("nan")
    result = cluster_ratio_bound(
        placed.harmful[family][rows, index].astype(float),
        placed.accepted[family][rows, index].astype(float),
        epsilon,
        delta=level,
        span=float(placed.span[family][index]),
        method="union_finite" if unit == UNIT_DOCUMENT else "cluster_robust_asymptotic",
    )
    return float(result.ratio_upper), float(result.z_upper)


def deployer(placed: Placed) -> Callable[[float, float], tuple[dict[str, Any], np.ndarray]]:
    """Memoised deployment.

    A prefix family has at most a hundred members plus a refusal, so a whole stage of cells
    resolves to a hundred distinct thresholds per environment. Recomputing the evaluation-block
    outcome for each of thirty thousand cells would repeat the identical arithmetic; the cache key
    is the threshold and the tolerance, which is exactly what the result depends on.
    """
    cache: dict[tuple[float, float], tuple[dict[str, Any], np.ndarray]] = {}

    def run(tau: float, epsilon: float) -> tuple[dict[str, Any], np.ndarray]:
        key = (float(tau), float(epsilon))
        if key not in cache:
            point = deploy(placed.environment, placed.adapted.eval_scores, tau, epsilon)
            accepted = point.pop("accepted")
            cache[key] = (point, accepted)
        point, accepted = cache[key]
        return dict(point), accepted

    return run


def _context(placed: Placed, **extra: Any) -> dict[str, Any]:
    environment = placed.environment
    return {
        "environment": environment.name,
        "corpus": environment.corpus,
        "base_engine": environment.base_engine,
        "capacity_limited": bool(environment.capacity_limited),
        "certification_pool_documents": int(placed.documents.size),
        "certification_pool_candidates": int(placed.doc_sizes.sum()),
        "evaluation_rows": int(environment.setup.evaluation.size),
        **extra,
    }


def evaluate_arm(
    placed: Placed,
    arm: str,
    unit: str,
    family: str,
    control: str,
    rows: np.ndarray,
    epsilon: float,
    run: Callable[[float, float], tuple[dict[str, Any], np.ndarray]],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """One (environment, budget, draw, arm, epsilon) cell."""
    depths = FAMILIES[family]
    decision = select_prefix(
        depths, certifier(placed, family, unit, rows, epsilon), delta=ALPHA, control=control
    )
    if decision.feasible:
        tau = float(placed.thresholds[family][decision.selected])
        depth = float(depths[decision.selected])
        ratio_upper, z_upper = deployed_bound(
            placed, family, unit, rows, decision.selected, decision.per_test_delta, epsilon
        )
        deployed_span = float(placed.span[family][decision.selected])
        certified_in_sample = int(placed.accepted[family][rows, decision.selected].sum())
        harmful_in_sample = int(placed.harmful[family][rows, decision.selected].sum())
    else:
        tau, depth = float("inf"), 0.0
        ratio_upper, z_upper = 1.0, float("inf")
        deployed_span = float("nan")
        certified_in_sample = 0
        harmful_in_sample = 0

    point, accepted = run(tau, epsilon)
    row: dict[str, Any] = dict(context)
    row.update(
        {
            "arm": arm,
            "unit": unit,
            "family": family,
            "control": control,
            "epsilon": float(epsilon),
            "n_docs": int(rows.size),
            "n_candidate_labels": int(placed.doc_sizes[rows].sum()) if rows.size else 0,
            "cert_harmful_labels": int(placed.doc_harm[rows].sum()) if rows.size else 0,
            "family_size": int(decision.family_size),
            "per_test_delta": float(decision.per_test_delta),
            "prefixes_tested": int(decision.tested),
            "prefixes_certified": len(decision.certified),
            "stopped_at": int(decision.stopped_at),
            "cut_feasible": bool(decision.feasible),
            "selected_index": int(decision.selected),
            "declared_depth": depth,
            "certified_ratio_upper": ratio_upper,
            "certified_z_upper": z_upper,
            "deployed_span": deployed_span,
            "cut_accepted_in_sample": certified_in_sample,
            "cut_harmful_in_sample": harmful_in_sample,
        }
    )
    row.update(point)
    row.update(placed.ranking)
    counts = None
    if epsilon == PRIMARY_EPSILON and arm in SELECTABLE_ARMS:
        # Every arm the nested selection may choose needs per-document counts, because the
        # paired document-clustered bootstrap resamples them. Writing them only for the four
        # factorial cells would leave the statistics unable to test a declared baseline that
        # happened to win.
        block = placed.environment.setup.evaluation
        counts = per_document_counts(block, accepted, sorted({*block.documents.tolist()}))
    return row, counts


def oracle_row(
    placed: Placed,
    oracle: str,
    epsilon: float,
    run: Callable[[float, float], tuple[dict[str, Any], np.ndarray]],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """An analysis-only cut chosen with evaluation labels. It may never reach a selection path.

    O0 searches the dense two-hundred-point depth ladder SGV15 used to price its own grid, O1 the
    frozen nine and O2 the declared hundred. The three are strictly nested, so the two grid losses
    are non-negative by construction and directly comparable to SGV15's frozen number.
    """
    block = placed.environment.setup.evaluation
    scores = placed.adapted.eval_scores
    if oracle == O0_UNRESTRICTED:
        grid = s15.fine_grid(placed.adapted.cert_scores)
    else:
        grid = placed.thresholds[FAMILY_NINE if oracle == O1_NINE else FAMILY_FINE]
    decision = s15.select_threshold(
        scores, block.harmful, epsilon, controller="empirical", thresholds=grid
    )
    tau = s15.refusal_tau(decision)
    depth = s15._depth_of(placed.adapted.cert_scores, tau)
    point, accepted = run(tau, epsilon)
    row: dict[str, Any] = dict(context)
    row.update(
        {
            "arm": oracle,
            "unit": "oracle",
            "family": "oracle",
            "control": "none",
            "epsilon": float(epsilon),
            "n_docs": 0,
            "n_candidate_labels": 0,
            "cert_harmful_labels": 0,
            "family_size": len(grid),
            "per_test_delta": float("nan"),
            "prefixes_tested": len(grid),
            "prefixes_certified": 0,
            "stopped_at": -1,
            "cut_feasible": bool(decision.feasible),
            "selected_index": -1,
            "declared_depth": depth,
            "certified_ratio_upper": float("nan"),
            "certified_z_upper": float("nan"),
            "deployed_span": float("nan"),
            "cut_accepted_in_sample": 0,
            "cut_harmful_in_sample": 0,
        }
    )
    row.update(point)
    row.update(placed.ranking)
    counts = None
    if epsilon == PRIMARY_EPSILON:
        counts = per_document_counts(block, accepted, sorted({*block.documents.tolist()}))
    return row, counts


# ------------------------------------------------- sections 23 and 24: the validity simulation
#
# The bounds are believed only after they are checked on data whose dependence structure is known,
# because on real data the truth is exactly what is missing. Two simulations run here:
#
#   * a RATIO simulation, where the true candidate-weighted risk sits exactly on the tolerance and
#     the question is how often each method certifies it anyway;
#   * a PREFIX simulation, where a whole ranked family is generated with a known safe frontier and
#     the question is how often each control procedure deploys past it.
#
# The generative model is chosen so the truth is available in closed form rather than estimated:
# scores are uniform, the harm probability is linear in the score, and the per-document random
# effect has mean one, so the population risk at acceptance depth q is exactly RISK_SLOPE * q.

REGIMES: tuple[tuple[str, float, float, str], ...] = (
    ("r1_iid_candidates", 0.0, 12.0, "constant"),
    ("r2_weak_correlation", 50.0, 12.0, "constant"),
    ("r3_moderate_correlation", 5.0, 12.0, "constant"),
    ("r4_strong_correlation", 0.5, 12.0, "constant"),
    ("r5_heterogeneous_sizes", 5.0, 12.0, "heavy_tailed"),
    ("r6_heterogeneous_risks", 0.0, 12.0, "two_point"),
)
"""``(name, concentration, mean document size, size law)``. Concentration ``0`` means the
per-document harm probability is degenerate at the target rate -- genuinely IID candidates for
``r1``, and a two-point mixture for ``r6``, which is the worst clustering a bounded rate allows."""

FINITE_SAMPLE_NAMES = ("union_finite", "hoeffding", "empirical_bernstein", "betting")

RISK_SLOPE = 0.4
"""``R(q) = RISK_SLOPE * q`` under the prefix simulation's generative model, so the safe frontier
is at ``q = epsilon / RISK_SLOPE`` and no estimate of the truth is needed."""

SIM_POOL_DOCUMENTS = 500


def _regime_draw(
    generator: np.random.Generator,
    name: str,
    concentration: float,
    mean_size: float,
    law: str,
    documents: int,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-document (harmful, accepted) counts whose population ratio is exactly ``rate``."""
    if law == "constant":
        sizes = np.full(documents, int(mean_size), dtype=np.int64)
    elif law == "heavy_tailed":
        sizes = np.clip(generator.geometric(1.0 / mean_size, documents), 1, 120).astype(np.int64)
    else:
        sizes = np.full(documents, int(mean_size), dtype=np.int64)
    if name == "r6_heterogeneous_risks":
        probabilities = np.where(generator.random(documents) < 0.2, 0.5, 0.0)
    elif concentration <= 0.0:
        probabilities = np.full(documents, rate)
    else:
        probabilities = generator.beta(
            rate * concentration, (1.0 - rate) * concentration, documents
        )
    harmful = generator.binomial(sizes, probabilities).astype(float)
    return harmful, sizes.astype(float)


def _ratio_simulation() -> dict[str, Any]:
    """How often does each method certify a risk that is exactly at the tolerance?"""
    generator = np.random.default_rng(SIMULATION_SEED)
    methods = (
        "candidate_clopper_pearson",
        "union_finite",
        "hoeffding",
        "empirical_bernstein",
        "betting",
        "cluster_robust_asymptotic",
    )
    results: dict[str, Any] = {}
    for name, concentration, mean_size, law in REGIMES:
        pool_harm, pool_size = _regime_draw(
            generator, name, concentration, mean_size, law, SIM_POOL_DOCUMENTS, PRIMARY_EPSILON
        )
        span = float(pool_size.max())
        per_budget: dict[str, Any] = {}
        for documents in SIMULATION_DOCUMENTS:
            counts = dict.fromkeys(methods, 0)
            for _ in range(SIMULATION_TRIALS):
                harmful, accepted = _regime_draw(
                    generator, name, concentration, mean_size, law, documents, PRIMARY_EPSILON
                )
                total, harm = int(accepted.sum()), int(harmful.sum())
                if total > 0 and clopper_pearson_upper(harm, total, ALPHA) <= PRIMARY_EPSILON:
                    counts["candidate_clopper_pearson"] += 1
                for method in methods[1:]:
                    if cluster_ratio_bound(
                        harmful, accepted, PRIMARY_EPSILON, delta=ALPHA, span=span, method=method
                    ).certifies:
                        counts[method] += 1
            per_budget[str(documents)] = {
                method: counts[method] / SIMULATION_TRIALS for method in methods
            }
        results[name] = {
            "concentration": concentration,
            "size_law": law,
            "span": span,
            "population_design_effect": design_effect(pool_harm, pool_size),
            "false_certification_rate": per_budget,
        }
    return results


def _prefix_pool(
    generator: np.random.Generator, documents: int, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, harm flags and document identifiers under the closed-form generative model."""
    effects = 1.25 * generator.beta(1.6, 0.4, documents)
    scores = generator.random((documents, size))
    probabilities = effects[:, None] * 0.8 * (1.0 - scores)
    harmful = generator.random((documents, size)) < probabilities
    return scores, harmful, np.repeat(np.arange(documents), size)


def _prefix_counts(
    scores: np.ndarray, harmful: np.ndarray, depths: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-document accepted and harmful-accepted counts at each declared depth."""
    taus = np.asarray([1.0 - d for d in depths], dtype=float)
    accepted = (scores[:, :, None] >= taus[None, None, :]).astype(np.int64)
    return (
        (accepted & harmful[:, :, None]).sum(axis=1).astype(float),
        accepted.sum(axis=1).astype(float),
    )


def _prefix_simulation() -> dict[str, Any]:
    """How often does each control procedure deploy past the known safe frontier?"""
    generator = np.random.default_rng(SIMULATION_SEED + 1)
    frontier_depth = PRIMARY_EPSILON / RISK_SLOPE
    procedures = (
        (UNIT_CANDIDATE, FAMILY_NINE, CONTROL_UNION),
        (UNIT_CANDIDATE, FAMILY_FINE, CONTROL_UNION),
        (UNIT_CANDIDATE, FAMILY_FINE, CONTROL_FIXED_SEQUENCE),
        (UNIT_DOCUMENT, FAMILY_NINE, CONTROL_UNION),
        (UNIT_DOCUMENT, FAMILY_FINE, CONTROL_UNION),
        (UNIT_DOCUMENT, FAMILY_FINE, CONTROL_FIXED_SEQUENCE),
    )
    size = 12
    pool_scores, _pool_harm, _ = _prefix_pool(generator, SIM_POOL_DOCUMENTS, size)
    spans = {
        family: np.asarray(
            [float((pool_scores >= (1.0 - depth)).sum(axis=1).max()) for depth in FAMILIES[family]],
            dtype=float,
        )
        for family in FAMILIES
    }
    results: dict[str, Any] = {}
    for documents in FWER_DOCUMENTS:
        tallies = {name: {"unsafe": 0, "refused": 0, "depth": 0.0} for name in procedures}
        for _ in range(FWER_TRIALS):
            scores, harmful, _ = _prefix_pool(generator, documents, size)
            counts = {
                family: _prefix_counts(scores, harmful, FAMILIES[family]) for family in FAMILIES
            }
            for key in procedures:
                unit, family, control = key
                harm_counts, accept_counts = counts[family]

                def certifies(
                    index: int,
                    level: float,
                    unit: str = unit,
                    family: str = family,
                    harm_counts: np.ndarray = harm_counts,
                    accept_counts: np.ndarray = accept_counts,
                ) -> bool:
                    total = int(accept_counts[:, index].sum())
                    if unit == UNIT_CANDIDATE:
                        harm = int(harm_counts[:, index].sum())
                        return bool(
                            total > 0
                            and clopper_pearson_upper(harm, total, level) <= PRIMARY_EPSILON
                        )
                    return cluster_ratio_bound(
                        harm_counts[:, index],
                        accept_counts[:, index],
                        PRIMARY_EPSILON,
                        delta=level,
                        span=float(spans[family][index]),
                        method="union_finite",
                    ).certifies

                decision = select_prefix(FAMILIES[family], certifies, delta=ALPHA, control=control)
                if not decision.feasible:
                    tallies[key]["refused"] += 1
                    continue
                depth = FAMILIES[family][decision.selected]
                tallies[key]["depth"] += depth
                if depth > frontier_depth + 1e-12:
                    tallies[key]["unsafe"] += 1
        results[str(documents)] = {
            f"{unit}|{family}|{control}": {
                "familywise_false_certification_rate": tallies[key]["unsafe"] / FWER_TRIALS,
                "refusal_rate": tallies[key]["refused"] / FWER_TRIALS,
                "mean_deployed_depth_when_feasible": (
                    tallies[key]["depth"] / (FWER_TRIALS - tallies[key]["refused"])
                    if tallies[key]["refused"] < FWER_TRIALS
                    else float("nan")
                ),
                "selected_depth_bias": (
                    tallies[key]["depth"] / (FWER_TRIALS - tallies[key]["refused"]) - frontier_depth
                    if tallies[key]["refused"] < FWER_TRIALS
                    else float("nan")
                ),
            }
            for key in procedures
            for unit, family, control in [key]
        }
    return results


def run_simulate() -> int:
    """The pre-registered validity gate. A bound that over-certifies here is demoted."""
    started = time.monotonic()
    if not DESIGN_RECORD.is_file():
        raise PhaseError("run --preregister before the validity gate")
    ratio = _ratio_simulation()
    prefix = _prefix_simulation()

    worst = {
        method: max(
            record["false_certification_rate"][str(documents)][method]
            for record in ratio.values()
            for documents in SIMULATION_DOCUMENTS
        )
        for method in (
            "candidate_clopper_pearson",
            "union_finite",
            "hoeffding",
            "empirical_bernstein",
            "betting",
            "cluster_robust_asymptotic",
        )
    }
    demoted = [
        method
        for method in ("hoeffding", "empirical_bernstein", "betting", "union_finite")
        if worst[method] > ALPHA
    ]
    _write_json_once(
        SIMULATION_COVERAGE,
        {
            **_envelope("simulation_coverage"),
            "question": (
                "with the true candidate-weighted risk sitting exactly on the tolerance, how "
                "often does each method certify it anyway?"
            ),
            "nominal_alpha": ALPHA,
            "trials_per_cell": SIMULATION_TRIALS,
            "documents": list(SIMULATION_DOCUMENTS),
            "seed": SIMULATION_SEED,
            "regimes": ratio,
            "worst_false_certification_rate": worst,
            "demoted_to_diagnostic": demoted,
            "pre_registered_contingency": (
                "declared in docs/sgv15b/risk_bound_design.md before this simulation was run: a "
                "component of the finite-sample union that exceeds its nominal level here is "
                "demoted to a diagnostic and the union is recomputed from the survivors."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        PREFIX_CONTROL_VALIDATION,
        {
            **_envelope("prefix_control_validation"),
            "question": (
                "on a ranked family whose safe frontier is known in closed form, how often does "
                "each control procedure deploy a prefix past it?"
            ),
            "generative_model": (
                "scores uniform on (0, 1), harm probability 0.8 * (1 - score) times a "
                "mean-one per-document effect, so the population risk at acceptance depth q is "
                f"exactly {RISK_SLOPE} * q"
            ),
            "safe_frontier_depth": PRIMARY_EPSILON / RISK_SLOPE,
            "nominal_alpha": ALPHA,
            "trials": FWER_TRIALS,
            "documents": list(FWER_DOCUMENTS),
            "seed": SIMULATION_SEED + 1,
            "amendment": (
                "the document counts were extended from (20, 40) after the first run showed every "
                "procedure refusing on essentially every trial at those sizes, which makes a "
                "familywise error rate unmeasurable rather than small. This changes the "
                "resolution of the validation and no method, criterion or real-data design. The "
                "refusal rates at 20 and 40 are retained below and are themselves a result: a "
                "declared family that starts at depth 0.005 spends its shallowest tests where "
                "there is least evidence, which is what stops fixed-sequence gatekeeping."
            ),
            "by_documents": prefix,
            "fixed_sequence_holds_its_level": all(
                record[key]["familywise_false_certification_rate"] <= ALPHA
                for record in prefix.values()
                for key in record
                if key.endswith(CONTROL_FIXED_SEQUENCE)
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"simulate: {len(REGIMES)} regimes x {len(SIMULATION_DOCUMENTS)} budgets, worst "
        f"finite-sample rate {max(worst[m] for m in FINITE_SAMPLE_NAMES):.4f} "
        f"vs candidate {worst['candidate_clopper_pearson']:.4f}, "
        f"{len(demoted)} demoted -> {_relative(SIMULATION_COVERAGE)}"
    )
    return 0


# ---------------------------------------------------------- section 8: the reproduction gate
#
# The candidate-level nine-depth arm is SGV15's method. If SGV15b cannot reproduce SGV15's own
# cells from SGV15's own code, then nothing downstream is a comparison -- it is two different
# experiments. So the gate runs first, on the whole document-stratified slice of experiment A,
# and any non-zero difference stops the stage.


def compare_frames(
    stored: pd.DataFrame, regenerated: pd.DataFrame, keys: list[str]
) -> dict[str, Any]:
    """Column-by-column comparison of two cell tables on a shared key.

    ``inf - inf`` is ``nan``, and a refusal threshold is ``+inf``, so equality is established
    before any subtraction; two identical refusals must compare as identical rather than as a
    missing value.
    """
    left = stored.set_index(keys).sort_index()
    right = regenerated.set_index(keys).sort_index()
    report: dict[str, Any] = {
        "rows_stored": len(left),
        "rows_regenerated": len(right),
        "keys_only_in_stored": len(left.index.difference(right.index)),
        "keys_only_in_regenerated": len(right.index.difference(left.index)),
    }
    shared_index = left.index.intersection(right.index)
    left = left.loc[shared_index]
    right = right.loc[shared_index]
    columns = [column for column in left.columns if column in right.columns]
    worst, worst_column = 0.0, None
    mismatches: list[str] = []
    for column in columns:
        lv, rv = left[column], right[column]
        if pd.api.types.is_numeric_dtype(lv) and pd.api.types.is_numeric_dtype(rv):
            lva = lv.to_numpy(dtype=float)
            rva = rv.to_numpy(dtype=float)
            equal = (lva == rva) | (np.isnan(lva) & np.isnan(rva))
            difference = float(np.abs(np.where(equal, 0.0, lva - rva)).max()) if lva.size else 0.0
            if not np.isfinite(difference):
                mismatches.append(f"{column} (non-finite difference)")
                continue
            if difference > worst:
                worst, worst_column = difference, column
        elif not lv.equals(rv):
            mismatches.append(column)
    report.update(
        {
            "columns_compared": len(columns),
            "columns_not_compared": sorted(set(left.columns).symmetric_difference(right.columns)),
            "max_absolute_numeric_difference": worst,
            "worst_column": worst_column,
            "non_numeric_mismatches": mismatches,
            "identical": bool(
                worst == 0.0
                and not mismatches
                and report["keys_only_in_stored"] == 0
                and report["keys_only_in_regenerated"] == 0
            ),
        }
    )
    return report


def _replay_sgv15(built: list[Any]) -> pd.DataFrame:
    """SGV15's experiment-A cells at its primary sampler, rebuilt through SGV15's own functions."""
    rows: list[dict[str, Any]] = []
    for environment in built:
        seed = _stable_seed(
            "sgv15-adapt", str(s15.SAMPLE_SEED), environment.name, s15.A_UNCERTAINTY
        )
        adapted = fit_adaptation(environment, s15.A_UNCERTAINTY, s15.ADAPT_BUDGET, seed)
        ranking = s15._ranking_row(environment, adapted)
        grid = s15.cut_grid(adapted.cert_scores)
        for size in s15.CERT_BUDGETS:
            for draw in range(1 if size == 0 else s15.CERT_DRAWS):
                sample_seed = _stable_seed(
                    "sgv15-cert",
                    str(s15.SAMPLE_SEED),
                    environment.name,
                    s15.PRIMARY_SAMPLER,
                    str(size),
                    str(draw),
                )
                sample = s15.certification_sample(
                    environment, s15.PRIMARY_SAMPLER, size, sample_seed
                )
                context = s15._context(
                    environment,
                    "A",
                    sampler=s15.PRIMARY_SAMPLER,
                    requested_cert=size,
                    draw=draw,
                    acquisition=s15.A_UNCERTAINTY,
                    allocation="250+n",
                )
                methods = (
                    (s15.M0_FROZEN_SOURCE, s15.C1_ORACLE)
                    if size == 0
                    else (*s15.METHODS[1:], s15.C1_ORACLE)
                )
                for method in methods:
                    for epsilon in EPSILONS:
                        row, _ = s15.evaluate_cell(
                            environment, adapted, grid, sample, method, epsilon, context
                        )
                        row.update(ranking)
                        row["cell"] = "|".join(
                            (
                                environment.name,
                                "A",
                                s15.PRIMARY_SAMPLER,
                                str(size),
                                str(draw),
                                method,
                            )
                        )
                        rows.append(row)
    return pd.DataFrame(rows)


def run_reproduce(only: str | None = None) -> int:
    """Rebuild SGV15's document-stratified experiment-A cells and demand an exact match."""
    started = time.monotonic()
    if not SIMULATION_COVERAGE.is_file():
        raise PhaseError("run --simulate before reading any real data")
    built, _ = s15._environments(only)
    regenerated = _replay_sgv15(built)
    stored = pd.read_parquet(s15.CELL_TABLE)
    stored = stored[
        (stored["experiment"] == "A") & (stored["sampler"] == s15.PRIMARY_SAMPLER)
    ].reset_index(drop=True)
    if only is not None:
        stored = stored[stored["environment"] == only].reset_index(drop=True)
    report = compare_frames(stored, regenerated, ["cell", "epsilon"])
    payload = {
        **_envelope("sgv15_reproduction"),
        "gate": (
            "the candidate-level nine-depth arm is SGV15's own method, replayed through SGV15's "
            "own `evaluate_cell`. Anything other than an exact match means SGV15b is not "
            "comparing itself to SGV15."
        ),
        "slice": {
            "experiment": "A",
            "sampler": s15.PRIMARY_SAMPLER,
            "certification_budgets": list(s15.CERT_BUDGETS),
            "draws": s15.CERT_DRAWS,
            "methods": [*s15.METHODS[1:], s15.M0_FROZEN_SOURCE, s15.C1_ORACLE],
            "epsilons": list(EPSILONS),
        },
        "comparison": report,
        "verified_fields": [
            "candidate identifiers, through the frozen certification-row assignment",
            "adaptation rows and their label accounting",
            "certification rows, through the recorded sampler seed",
            "the selected threshold",
            "the certified risk upper bound",
            "realised harm, repair recall and coverage on the evaluation block",
        ],
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(SGV15_REPRODUCTION, payload)
    print(
        f"reproduce: {report['rows_regenerated']} cells, "
        f"max absolute difference {report['max_absolute_numeric_difference']}, "
        f"identical={report['identical']} -> {_relative(SGV15_REPRODUCTION)}"
    )
    if not report["identical"]:
        raise PhaseError(
            "M00 did not reproduce SGV15's frozen cells; SGV15b stops here rather than comparing "
            "against a number it cannot rebuild"
        )
    return 0


# ------------------------------------------------------------ section 7: the 2x2 factorial


def _sample_seed(environment: str, budget: int, draw: int) -> int:
    return _stable_seed("sgv15b-docs", str(SAMPLE_SEED), environment, str(budget), str(draw))


def run_factorial(only: str | None = None) -> int:
    """Every arm, every declared document budget, every draw, every tolerance."""
    started = time.monotonic()
    if not SGV15_REPRODUCTION.is_file():
        raise PhaseError("run --reproduce before the factorial; the gate is not optional")
    built, _ = s15._environments(only)
    rows: list[dict[str, Any]] = []
    counts: list[tuple[tuple[str, str], np.ndarray]] = []
    geometry: list[Placed] = []
    for environment in built:
        began = time.monotonic()
        placed = place(environment)
        geometry.append(placed)
        run = deployer(placed)
        for epsilon in EPSILONS:
            for oracle in ORACLES:
                context = _context(
                    placed, requested_docs=0, draw=0, clipped=False, cell_kind="oracle"
                )
                row, block = oracle_row(placed, oracle, epsilon, run, context)
                _push(rows, counts, environment, (environment.name, "0", "0", oracle), row, block)
        for budget in DOC_BUDGETS:
            for draw in range(DOC_DRAWS):
                sample = document_sample(
                    placed, budget, _sample_seed(environment.name, budget, draw)
                )
                context = _context(
                    placed,
                    requested_docs=budget,
                    draw=draw,
                    clipped=bool(sample.size < budget),
                    cell_kind="arm",
                )
                for arm, unit, family, control in ARMS:
                    for epsilon in EPSILONS:
                        row, block = evaluate_arm(
                            placed, arm, unit, family, control, sample, epsilon, run, context
                        )
                        _push(
                            rows,
                            counts,
                            environment,
                            (environment.name, str(budget), str(draw), arm),
                            row,
                            block,
                        )
        print(
            f"  {environment.name}: {len(rows)} cells so far ({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    _append_cells(rows, counts)
    write_geometry(geometry)
    print(
        f"factorial: {len(rows)} cells over {len(built)} environments "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _push(
    rows: list[dict[str, Any]],
    counts: list[tuple[tuple[str, str], np.ndarray]],
    environment: Any,
    key: tuple[str, ...],
    row: dict[str, Any],
    block: dict[str, np.ndarray] | None,
) -> None:
    row["cell"] = "|".join(key)
    rows.append(row)
    if block is not None:
        counts.append(
            (
                (environment.name, row["cell"]),
                np.vstack(
                    [
                        block["accepted"],
                        block["harmful_accepted"],
                        block["beneficial_accepted"],
                        block["beneficial_total"],
                    ]
                ),
            )
        )


def _append_cells(
    rows: list[dict[str, Any]], counts: list[tuple[tuple[str, str], np.ndarray]]
) -> None:
    """Write-once, so a partial rerun cannot silently mix two generations of the same cell."""
    _write_parquet_once(FACTORIAL_CELLS, pd.DataFrame(rows))
    if not counts:
        return
    documents = s15._evaluation_documents()
    widths = [block.shape[1] for _, block in counts]
    _write_parquet_once(
        DOCUMENT_COUNTS,
        pd.DataFrame(
            {
                "cell": pd.Categorical(np.repeat([key for (_, key), _ in counts], widths)),
                "document_id": pd.Categorical(
                    np.concatenate([documents[name] for (name, _), _ in counts])
                ),
                "accepted": np.concatenate([b[0] for _, b in counts]).astype(float),
                "harmful_accepted": np.concatenate([b[1] for _, b in counts]).astype(float),
                "beneficial_accepted": np.concatenate([b[2] for _, b in counts]).astype(float),
                "beneficial_total": np.concatenate([b[3] for _, b in counts]).astype(float),
            }
        ),
    )


# ------------------------------------------------------------------ the reusable geometry
#
# Section 42's rule again. The adaptation fit, the placed prefix families and the certification
# pool's scores and labels are the only expensive things in this stage, and they depend on the
# environment and the frozen seed alone. They are written once by the factorial phase, and every
# downstream analysis -- the controls, the permutations, the duplication falsification, the
# clustering diagnostics -- reads them instead of refitting.

GEOMETRY = OUT / "certification_geometry.npz"


@dataclass(frozen=True, slots=True)
class Geometry:
    """One environment's certification pool and evaluation block, without the model."""

    name: str
    corpus: str
    base_engine: str
    documents: np.ndarray
    doc_of: np.ndarray
    scores: np.ndarray
    harm: np.ndarray
    thresholds: dict[str, np.ndarray]
    eval_scores: np.ndarray
    eval_harm: np.ndarray
    eval_beneficial: np.ndarray
    eval_documents: np.ndarray

    def counts(self, family: str, harm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Per-document (harmful accepted, accepted) counts at every prefix of ``family``.

        ``harm`` overrides the recorded labels, which is what the permutation controls need and
        the only way any of them can reach a certification decision.
        """
        taus = self.thresholds[family]
        labels = self.harm if harm is None else harm
        accepted = np.zeros((self.documents.size, taus.size), dtype=np.int64)
        harmful = np.zeros_like(accepted)
        for index in range(self.documents.size):
            mask = self.doc_of == index
            ascending = np.sort(self.scores[mask])
            accepted[index] = ascending.size - np.searchsorted(ascending, taus, side="left")
            ascending_harm = np.sort(self.scores[mask & labels])
            harmful[index] = ascending_harm.size - np.searchsorted(
                ascending_harm, taus, side="left"
            )
        return harmful.astype(float), accepted.astype(float)

    def sizes(self) -> np.ndarray:
        return np.asarray(
            [int((self.doc_of == index).sum()) for index in range(self.documents.size)], dtype=float
        )


def _key(name: str, field: str) -> str:
    return f"{name.replace('/', '__')}|{field}"


def write_geometry(placed: list[Placed]) -> None:
    payload: dict[str, np.ndarray] = {}
    for item in placed:
        environment = item.environment
        block = environment.setup.evaluation
        documents = np.asarray([str(d) for d in environment.cert_documents.tolist()], dtype=object)
        lookup = {name: index for index, name in enumerate(item.documents.tolist())}
        name = environment.name
        payload[_key(name, "documents")] = item.documents.astype(str)
        payload[_key(name, "doc_of")] = np.asarray(
            [lookup[str(d)] for d in documents.tolist()], dtype=np.int64
        )
        payload[_key(name, "scores")] = np.asarray(item.adapted.cert_scores, dtype=float)
        payload[_key(name, "harm")] = np.asarray(environment.cert_harmful, dtype=bool)
        for family in FAMILIES:
            payload[_key(name, f"thresholds_{family}")] = item.thresholds[family]
        payload[_key(name, "eval_scores")] = np.asarray(item.adapted.eval_scores, dtype=float)
        payload[_key(name, "eval_harm")] = np.asarray(block.harmful, dtype=bool)
        payload[_key(name, "eval_beneficial")] = np.asarray(block.beneficial, dtype=bool)
        payload[_key(name, "eval_documents")] = np.asarray(block.documents, dtype=str)
        payload[_key(name, "meta")] = np.asarray(
            [environment.corpus, environment.base_engine], dtype=str
        )
    if GEOMETRY.exists():
        raise PhaseError(f"{_relative(GEOMETRY)} already exists; delete it deliberately")
    np.savez_compressed(GEOMETRY, **payload)


def load_geometry() -> dict[str, Geometry]:
    if not GEOMETRY.is_file():
        raise PhaseError("run --factorial first; the certification geometry is written there")
    stored = np.load(GEOMETRY, allow_pickle=False)
    out: dict[str, Geometry] = {}
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        if _key(name, "scores") not in stored:
            continue
        meta = stored[_key(name, "meta")]
        out[name] = Geometry(
            name=name,
            corpus=str(meta[0]),
            base_engine=str(meta[1]),
            documents=stored[_key(name, "documents")],
            doc_of=stored[_key(name, "doc_of")],
            scores=stored[_key(name, "scores")],
            harm=stored[_key(name, "harm")],
            thresholds={family: stored[_key(name, f"thresholds_{family}")] for family in FAMILIES},
            eval_scores=stored[_key(name, "eval_scores")],
            eval_harm=stored[_key(name, "eval_harm")],
            eval_beneficial=stored[_key(name, "eval_beneficial")],
            eval_documents=stored[_key(name, "eval_documents")],
        )
    return out


def geometry_deploy(geometry: Geometry, tau: float, epsilon: float) -> dict[str, float]:
    """SGV12's own `deployed`, on the stored evaluation arrays rather than on a rebuilt block."""
    point = s13.deployed(
        geometry.eval_scores, geometry.eval_harm, geometry.eval_beneficial, tau, epsilon
    )
    return {
        "coverage": float(point["coverage"]),
        "realized_harm_rate": float(point["realized_harm_rate"]),
        "holds_bound": bool(point["holds_bound"]),
        "repair_recall": float(point["repair_recall"]),
        "risk_controlled": float(point["risk_controlled_repair_recall"]),
        "n_accepted": int(point["n_accepted"]),
    }


def geometry_certify(
    geometry: Geometry,
    family: str,
    unit: str,
    control: str,
    rows: np.ndarray,
    epsilon: float,
    harm: np.ndarray | None = None,
    multiplier: int = 1,
) -> tuple[Any, float]:
    """Run one arm's whole certification procedure on stored counts. Returns (decision, tau).

    ``multiplier`` duplicates every candidate inside every document, which is section 26's
    falsification: it multiplies both per-document counts and the population support, so a
    document-level certificate is unchanged by construction and a candidate-level one is not.
    """
    harmful, accepted = geometry.counts(family, harm)
    span = accepted.max(axis=0) * multiplier
    harmful = harmful[rows] * multiplier
    accepted = accepted[rows] * multiplier

    def certifies(index: int, level: float) -> bool:
        total = int(accepted[:, index].sum())
        count = int(harmful[:, index].sum())
        if unit == UNIT_CANDIDATE:
            return bool(total > 0 and clopper_pearson_upper(count, total, level) <= epsilon)
        if unit == UNIT_NAIVE:
            return bool(total > 0 and count / total <= epsilon)
        return cluster_ratio_bound(
            harmful[:, index],
            accepted[:, index],
            epsilon,
            delta=level,
            span=float(span[index]),
            method="union_finite" if unit == UNIT_DOCUMENT else "cluster_robust_asymptotic",
        ).certifies

    decision = select_prefix(FAMILIES[family], certifies, delta=ALPHA, control=control)
    tau = (
        float(geometry.thresholds[family][decision.selected]) if decision.feasible else float("inf")
    )
    return decision, tau


# --------------------------------------------------------------------- reading the cell table


def load_cells() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not FACTORIAL_CELLS.is_file():
        raise PhaseError("run --factorial first")
    return pd.read_parquet(FACTORIAL_CELLS), cc_read_json(DESIGN_RECORD)


def arm_slice(
    cells: pd.DataFrame, arm: str, budget: int | None = None, epsilon: float = PRIMARY_EPSILON
) -> pd.DataFrame:
    block = cells[(cells["arm"] == arm) & (cells["epsilon"] == epsilon)]
    if budget is not None:
        block = block[block["requested_docs"] == budget]
    return block


def per_environment(block: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Aggregate a slice over its draws, one record per environment.

    Realised harm is averaged only over the draws where it is DEFINED. A draw that accepted
    nothing has no harm rate; recording it as zero would let refusal look like safety.
    """
    out: dict[str, dict[str, float]] = {}
    for name, group in block.groupby("environment", observed=True):
        out[str(name)] = {
            "draws": len(group),
            "cut_feasible_fraction": float(group["cut_feasible"].mean()),
            "coverage": float(group["coverage"].mean()),
            "realized_harm_rate": _mean(group["realized_harm_rate"].tolist()),
            "holds_bound_fraction": float(group["holds_bound"].mean()),
            "repair_recall": float(group["repair_recall"].mean()),
            "risk_controlled": float(group["risk_controlled"].mean()),
            "n_accepted": float(group["n_accepted"].mean()),
            "declared_depth": float(group["declared_depth"].mean()),
            "n_docs": float(group["n_docs"].mean()),
            "n_candidate_labels": float(group["n_candidate_labels"].mean()),
            "prefixes_certified": float(group["prefixes_certified"].mean()),
            "prefixes_tested": float(group["prefixes_tested"].mean()),
            "certified_ratio_upper": _mean(group["certified_ratio_upper"].tolist()),
        }
    return out


def taxonomy_counts(block: pd.DataFrame, floor: float) -> dict[str, int]:
    modes = block.apply(lambda row: classify(row, floor), axis=1)
    return {mode: int((modes == mode).sum()) for mode in FAILURE_MODES}


def environments_holding(records: dict[str, dict[str, float]]) -> list[str]:
    """Environments where EVERY draw held the bound. A single violation is a violation."""
    return sorted(name for name, record in records.items() if holds_the_bound(record))


def environments_non_degenerate(records: dict[str, dict[str, float]], floor: float) -> list[str]:
    return sorted(
        name
        for name, record in records.items()
        if holds_the_bound(record)
        and record["cut_feasible_fraction"] >= 1.0
        and record["coverage"] >= floor
    )


# ------------------------------------------------- section 26: the duplication falsification


DUPLICATION_MULTIPLIERS = (1, 2, 5, 10)


def run_duplicate() -> int:
    """Duplicate every candidate inside every certification document and re-certify.

    The comparison is made at EVERY one of the nine declared depths rather than at a chosen one,
    so no prefix is picked after seeing which of them behaves well.
    """
    started = time.monotonic()
    geometry = load_geometry()
    per_env: dict[str, Any] = {}
    for name, item in geometry.items():
        rows = document_sample_rows(item, PRACTICAL_DOC_BUDGET, name)
        harmful, accepted = item.counts(FAMILY_NINE)
        base_span = accepted.max(axis=0)
        level = ALPHA / len(NINE_DEPTHS)
        record: dict[str, Any] = {}
        for unit, arm in ((UNIT_CANDIDATE, M00), (UNIT_DOCUMENT, M10)):
            by_multiplier: dict[str, Any] = {}
            for multiplier in DUPLICATION_MULTIPLIERS:
                decision, _tau = geometry_certify(
                    item,
                    FAMILY_NINE,
                    unit,
                    CONTROL_UNION,
                    rows,
                    PRIMARY_EPSILON,
                    multiplier=multiplier,
                )
                limits: list[float] = []
                for index in range(len(NINE_DEPTHS)):
                    total = int(accepted[rows, index].sum()) * multiplier
                    harm = int(harmful[rows, index].sum()) * multiplier
                    if unit == UNIT_CANDIDATE:
                        limits.append(float(clopper_pearson_upper(harm, total, level)))
                    else:
                        limits.append(
                            float(
                                cluster_ratio_bound(
                                    harmful[rows, index] * multiplier,
                                    accepted[rows, index] * multiplier,
                                    PRIMARY_EPSILON,
                                    delta=level,
                                    span=float(base_span[index] * multiplier),
                                ).z_upper
                            )
                        )
                by_multiplier[str(multiplier)] = {
                    "deployed_index": int(decision.selected),
                    "deployed_depth": (
                        float(NINE_DEPTHS[decision.selected]) if decision.feasible else float("nan")
                    ),
                    "feasible": bool(decision.feasible),
                    "prefixes_certified": len(decision.certified),
                    "bounds_at_every_declared_depth": limits,
                    "accepted_in_sample": int(accepted[rows].sum(axis=0).max()) * multiplier,
                }
            baseline = by_multiplier["1"]["bounds_at_every_declared_depth"]
            record[arm] = {
                "unit": unit,
                "by_multiplier": by_multiplier,
                "decision_invariant": all(
                    entry["deployed_index"] == by_multiplier["1"]["deployed_index"]
                    for entry in by_multiplier.values()
                ),
                "bound_scales_with_the_multiplier": all(
                    abs(value - baseline[index] * int(key))
                    <= 1e-8 * max(1.0, abs(baseline[index] * int(key)))
                    for key, entry in by_multiplier.items()
                    for index, value in enumerate(entry["bounds_at_every_declared_depth"])
                ),
                "bound_shrinks_with_duplication": any(
                    value < baseline[index] - 1e-12
                    for value in [
                        by_multiplier[str(DUPLICATION_MULTIPLIERS[-1])][
                            "bounds_at_every_declared_depth"
                        ][index]
                    ]
                    for index in range(len(NINE_DEPTHS))
                ),
            }
        per_env[name] = record

    document_invariant = [
        name
        for name, record in per_env.items()
        if record[M10]["decision_invariant"] and record[M10]["bound_scales_with_the_multiplier"]
    ]
    candidate_invariant = [
        name for name, record in per_env.items() if record[M00]["decision_invariant"]
    ]
    candidate_overconfident = [
        name for name, record in per_env.items() if record[M00]["bound_shrinks_with_duplication"]
    ]
    _write_json_once(
        DOCUMENT_DUPLICATION_TEST,
        {
            **_envelope("document_duplication_test"),
            "question": (
                "duplicating every candidate inside every certification document adds no "
                "independent document. Does the certificate know that?"
            ),
            "multipliers": list(DUPLICATION_MULTIPLIERS),
            "budget_documents": PRACTICAL_DOC_BUDGET,
            "family": FAMILY_NINE,
            "epsilon": PRIMARY_EPSILON,
            "per_environment": per_env,
            "summary": {
                "environments": len(per_env),
                "document_aware_invariant": len(document_invariant),
                "candidate_level_decision_invariant": len(candidate_invariant),
                "candidate_level_bound_shrinks": len(candidate_overconfident),
                "falsification_passes": bool(
                    len(document_invariant) == len(per_env)
                    and len(candidate_overconfident) == len(per_env)
                ),
            },
            "interpretation": (
                "the document-aware limit on E[Z] is positively homogeneous in the per-document "
                "counts once the support scales with them, so it scales by exactly the multiplier "
                "and the certification decision is invariant by construction rather than by luck. "
                "The candidate-level interval treats the copies as new observations and narrows, "
                "which is the same error the clustered simulation measures as an inflated "
                "false-certification rate."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"duplicate: document-aware invariant on {len(document_invariant)}/{len(per_env)}, "
        f"candidate-level bound shrinks on {len(candidate_overconfident)}/{len(per_env)} "
        f"-> {_relative(DOCUMENT_DUPLICATION_TEST)}"
    )
    return 0


def document_sample_rows(geometry: Geometry, budget: int, name: str, draw: int = 0) -> np.ndarray:
    """The same draw the factorial used, rebuilt from the recorded seed."""
    available = int(geometry.documents.size)
    take = min(int(budget), available)
    generator = np.random.default_rng(_sample_seed(name, budget, draw))
    return np.sort(generator.permutation(available)[:take])


# ------------------------------------------- section 22: how much dependence is actually there


def run_cluster() -> int:
    """Intra-document dependence, effective sample size, and the two bounds side by side."""
    started = time.monotonic()
    geometry = load_geometry()
    structure: dict[str, Any] = {}
    effects: dict[str, Any] = {}
    effective: dict[str, Any] = {}
    anti_conservative: list[str] = []
    for name, item in geometry.items():
        harmful, accepted = item.counts(FAMILY_NINE)
        rows = document_sample_rows(item, PRACTICAL_DOC_BUDGET, name)
        by_depth: dict[str, Any] = {}
        effect_by_depth: dict[str, Any] = {}
        effective_by_depth: dict[str, Any] = {}
        below = 0
        for index, depth in enumerate(NINE_DEPTHS):
            column_a = accepted[:, index]
            column_h = harmful[:, index]
            total = float(column_a.sum())
            contributing = int((column_a > 0).sum())
            rates = np.divide(column_h, column_a, out=np.zeros_like(column_h), where=column_a > 0)
            deff = design_effect(column_h, column_a)
            mean_cluster = total / contributing if contributing else float("nan")
            # The standard identity for a one-way clustered design: deff = 1 + (nbar - 1) * ICC.
            # Reported as a derived quantity, not as a second estimator of the same thing.
            icc = (
                (deff - 1.0) / (mean_cluster - 1.0)
                if contributing and mean_cluster > 1.0 and not math.isnan(deff)
                else float("nan")
            )
            by_depth[f"{depth:g}"] = {
                "documents_contributing": contributing,
                "accepted": total,
                "harmful_accepted": float(column_h.sum()),
                "observed_ratio": float(column_h.sum() / total) if total > 0 else float("nan"),
                "accepted_per_document_mean": mean_cluster,
                "accepted_per_document_max": float(column_a.max()),
                "per_document_harm_rate_mean": float(rates[column_a > 0].mean())
                if contributing
                else float("nan"),
                "per_document_harm_rate_sd": float(rates[column_a > 0].std(ddof=1))
                if contributing > 1
                else float("nan"),
            }
            effect_by_depth[f"{depth:g}"] = {
                "design_effect": deff,
                "intraclass_correlation_from_design_effect": icc,
            }
            sample_a = accepted[rows, index]
            sample_h = harmful[rows, index]
            sample_total = int(sample_a.sum())
            candidate_limit = (
                float(clopper_pearson_upper(int(sample_h.sum()), sample_total, ALPHA))
                if sample_total > 0
                else 1.0
            )
            document_limit = cluster_ratio_upper(
                sample_h, sample_a, delta=ALPHA, span=float(column_a.max()), resolution=1000
            )
            asymptotic = cluster_ratio_bound(
                sample_h,
                sample_a,
                PRIMARY_EPSILON,
                delta=ALPHA,
                span=float(column_a.max()),
                method="cluster_robust_asymptotic",
            ).ratio_upper
            if candidate_limit < document_limit:
                below += 1
            effective_by_depth[f"{depth:g}"] = {
                "sample_documents": int(rows.size),
                "sample_accepted": sample_total,
                "sample_harmful": int(sample_h.sum()),
                "effective_sample_size": (
                    float(sample_total / deff)
                    if not math.isnan(deff) and deff > 0
                    else float("nan")
                ),
                "candidate_level_clopper_pearson_upper": candidate_limit,
                "document_aware_finite_upper": document_limit,
                "cluster_robust_asymptotic_upper": asymptotic,
                "candidate_bound_is_below_the_document_bound": bool(
                    candidate_limit < document_limit
                ),
            }
        if below > 0:
            anti_conservative.append(name)
        structure[name] = {"corpus": item.corpus, "by_depth": by_depth}
        effects[name] = {
            "by_depth": effect_by_depth,
            "mean_design_effect": _mean(
                [record["design_effect"] for record in effect_by_depth.values()]
            ),
            "max_design_effect": max(
                [
                    record["design_effect"]
                    for record in effect_by_depth.values()
                    if not math.isnan(record["design_effect"])
                ]
                or [float("nan")]
            ),
        }
        effective[name] = {
            "by_depth": effective_by_depth,
            "depths_where_the_candidate_bound_is_below": below,
        }

    _write_json_once(
        CLUSTER_STRUCTURE,
        {
            **_envelope("cluster_structure"),
            "unit": "document, or volume on OCR-D-SBB",
            "measured_on": "the whole certification pool, at each of the nine declared depths",
            "per_environment": structure,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    defined = [
        record["intraclass_correlation_from_design_effect"]
        for environment in effects.values()
        for record in environment["by_depth"].values()
        if not math.isnan(record["intraclass_correlation_from_design_effect"])
    ]
    _write_json_once(
        DESIGN_EFFECTS,
        {
            **_envelope("design_effects"),
            "intraclass_correlation": {
                "cells_where_it_is_defined": len(defined),
                "cells_total": len(effects) * len(NINE_DEPTHS),
                "median": float(np.median(defined)) if defined else float("nan"),
                "minimum": min(defined) if defined else float("nan"),
                "maximum": max(defined) if defined else float("nan"),
                "note": (
                    "derived from the design effect through the one-way identity rather than "
                    "estimated a second way, so it is not constrained to the unit interval: a "
                    "cell whose clustered variance falls below the binomial one returns a "
                    "negative value."
                ),
            },
            "definition": (
                "the ratio of the document-clustered variance of the accepted-set harm rate to "
                "the binomial variance the candidate-level bound assumes. One is no clustering."
            ),
            "identity": "deff = 1 + (mean accepted per document - 1) * ICC",
            "per_environment": effects,
            "sgv15_reference": {
                "values": [
                    float(record["design_effect_on_the_harm_rate"])
                    for record in cc_read_json(s15.RISK_BOUND_RESULTS)["per_environment"].values()
                    if record.get("deployed")
                ],
                "note": "SGV15 measured these on its deployed accepted sets, not on the pool",
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        EFFECTIVE_SAMPLE_SIZE,
        {
            **_envelope("effective_sample_size"),
            "budget_documents": PRACTICAL_DOC_BUDGET,
            "definition": (
                "N_eff = accepted candidates / design effect. A diagnostic, not a method."
            ),
            "per_environment": effective,
            "environments_where_the_candidate_bound_falls_below": anti_conservative,
            "count_anti_conservative": len(anti_conservative),
            "interpretation": (
                "where the exact Bernoulli limit sits BELOW the distribution-free document-level "
                "limit on the same accepted set, the candidate-level certificate is reading its "
                "guarantee off a sample that does not satisfy the independence assumption it "
                "needs. SGV15 reported this on two of three deployed environments and did not "
                "correct it; the correction is the document-aware arm."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"cluster: {len(structure)} environments, candidate bound below the document bound on "
        f"{len(anti_conservative)} -> {_relative(EFFECTIVE_SAMPLE_SIZE)}"
    )
    return 0


# --------------------------------------------------------- section 21: the oracle decomposition


def run_oracle() -> int:
    """What each declared family could have achieved with evaluation labels, and what it costs."""
    started = time.monotonic()
    cells, _design = load_cells()
    per_env: dict[str, Any] = {}
    for name in sorted(cells["environment"].unique()):
        record: dict[str, Any] = {}
        for oracle in ORACLES:
            block = cells[
                (cells["environment"] == name)
                & (cells["arm"] == oracle)
                & (cells["epsilon"] == PRIMARY_EPSILON)
            ]
            row = block.iloc[0]
            record[oracle] = {
                "repair_recall": float(row["repair_recall"]),
                "risk_controlled": float(row["risk_controlled"]),
                "coverage": float(row["coverage"]),
                "realized_harm_rate": float(row["realized_harm_rate"]),
                "declared_depth": float(row["declared_depth"]),
                "feasible": bool(row["cut_feasible"]),
                "family_size": int(row["family_size"]),
            }
        loss_nine = record[O0_UNRESTRICTED]["repair_recall"] - record[O1_NINE]["repair_recall"]
        loss_fine = record[O0_UNRESTRICTED]["repair_recall"] - record[O2_FINE]["repair_recall"]
        record["grid_loss_nine"] = loss_nine
        record["grid_loss_fine"] = loss_fine
        record["recovered"] = loss_nine - loss_fine
        record["recovered_fraction"] = (
            (loss_nine - loss_fine) / loss_nine if loss_nine > 0 else float("nan")
        )
        per_env[str(name)] = record

    losses_nine = [record["grid_loss_nine"] for record in per_env.values()]
    losses_fine = [record["grid_loss_fine"] for record in per_env.values()]
    recovered = float(np.mean(losses_nine) - np.mean(losses_fine))
    fraction = recovered / float(np.mean(losses_nine)) if float(np.mean(losses_nine)) > 0 else 0.0
    sgv15_gap = float(cc_read_json(SGV15_GRID_RESOLUTION)["mean_grid_resolution_gap"])
    payload = {
        **_envelope("oracle_results"),
        "status": "ANALYSIS ONLY. Every oracle reads evaluation labels and none may select.",
        "oracles": {
            O0_UNRESTRICTED: (
                "the deepest cut on SGV15's own dense two-hundred-point acceptance-depth ladder "
                "whose realised harm holds on the evaluation block"
            ),
            O1_NINE: "the same, restricted to SGV15's frozen nine declared depths",
            O2_FINE: "the same, restricted to SGV15b's hundred declared depths",
        },
        "nesting": "the nine are a subset of the hundred, which are a subset of the two hundred",
        "per_environment": per_env,
        "mean_grid_loss_nine": float(np.mean(losses_nine)),
        "mean_grid_loss_fine": float(np.mean(losses_fine)),
        "mean_recovered": recovered,
        "recovered_fraction": fraction,
        "environments_where_the_nine_cost_recall": int(sum(1 for v in losses_nine if v > 0)),
        "environments_where_the_hundred_still_cost_recall": int(
            sum(1 for v in losses_fine if v > 0)
        ),
        "sgv15_frozen_grid_resolution_gap": sgv15_gap,
        "agrees_with_sgv15": bool(abs(float(np.mean(losses_nine)) - sgv15_gap) < 1e-9),
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(ORACLE_RESULTS, payload)
    _write_json_once(
        GRID_RESOLUTION_ANALYSIS,
        {
            **_envelope("grid_resolution_analysis"),
            "question": (
                "SGV15's nine declared depths cost repair recall against a dense frontier. How "
                "much of that does a hundred declared depths recover, before any multiplicity "
                "correction is paid for it?"
            ),
            "grid_loss_nine": float(np.mean(losses_nine)),
            "grid_loss_fine": float(np.mean(losses_fine)),
            "recovered": recovered,
            "recovered_fraction": fraction,
            "required_fraction": RESOLUTION_RECOVERY_REQUIRED,
            "criterion_5_resolution_component_met": bool(fraction >= RESOLUTION_RECOVERY_REQUIRED),
            "per_environment": {
                name: {
                    "grid_loss_nine": record["grid_loss_nine"],
                    "grid_loss_fine": record["grid_loss_fine"],
                    "recovered_fraction": record["recovered_fraction"],
                }
                for name, record in per_env.items()
            },
            "caveat": (
                "an oracle recovery is an upper bound on what any procedure can gain from the "
                "finer family. It says nothing about whether a certificate can reach it, which "
                "is what the factorial measures."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"oracle: grid loss {float(np.mean(losses_nine)):.4f} -> "
        f"{float(np.mean(losses_fine)):.4f}, "
        f"{fraction:.1%} recovered -> {_relative(ORACLE_RESULTS)}"
    )
    return 0


# ------------------------------------------------------ section 20: the mechanism decomposition

ENDPOINTS = ("risk_controlled", "repair_recall", "coverage", "realized_harm_rate")


def _effect(
    left: dict[str, dict[str, float]], right: dict[str, dict[str, float]], field: str
) -> dict[str, Any]:
    """One factor's effect: the paired per-environment difference, and its mean."""
    shared = sorted(set(left) & set(right))
    differences = {
        name: float(left[name][field] - right[name][field])
        for name in shared
        if not (math.isnan(left[name][field]) or math.isnan(right[name][field]))
    }
    return {
        "per_environment": differences,
        "mean": _mean(list(differences.values())),
        "environments_improved": int(sum(1 for value in differences.values() if value > 0)),
        "environments_defined": len(differences),
    }


def run_results() -> int:
    """The 2x2, its per-arm tables, and the failure taxonomy at every declared budget."""
    started = time.monotonic()
    cells, _design = load_cells()
    floor = float(s15.load_frozen().coverage_floor)

    by_arm: dict[str, dict[str, Any]] = {}
    for arm, _unit, _family, _control in ARMS:
        per_budget: dict[str, Any] = {}
        for budget in DOC_BUDGETS:
            block = arm_slice(cells, arm, budget)
            records = per_environment(block)
            per_budget[str(budget)] = {
                "per_environment": records,
                "environments_holding_the_bound": len(environments_holding(records)),
                "environments_non_degenerate": len(environments_non_degenerate(records, floor)),
                "mean_risk_controlled": _mean(
                    [record["risk_controlled"] for record in records.values()]
                ),
                "mean_repair_recall": _mean(
                    [record["repair_recall"] for record in records.values()]
                ),
                "mean_coverage": _mean([record["coverage"] for record in records.values()]),
                "mean_realized_harm_rate": _mean(
                    [record["realized_harm_rate"] for record in records.values()]
                ),
                "mean_declared_depth": _mean(
                    [record["declared_depth"] for record in records.values()]
                ),
                "refusal_rate": float(1.0 - block["cut_feasible"].mean()),
                "failure_modes": taxonomy_counts(block, floor),
                "mean_candidate_labels": float(block["n_candidate_labels"].mean()),
                "mean_documents": float(block["n_docs"].mean()),
            }
        by_arm[arm] = per_budget

    factorial: dict[str, Any] = {}
    for budget in DOC_BUDGETS:
        records = {arm: per_environment(arm_slice(cells, arm, budget)) for arm in FACTORIAL}
        entry: dict[str, Any] = {}
        for field in ENDPOINTS:
            clustering = _effect(records[M10], records[M00], field)
            resolution = _effect(records[M01], records[M00], field)
            joint = _effect(records[M11], records[M00], field)
            deep = _effect(records[M11], records[M01], field)
            entry[field] = {
                "clustering_effect": clustering,
                "resolution_effect": resolution,
                "joint_effect": joint,
                "interaction": {
                    "mean": (
                        deep["mean"] - clustering["mean"]
                        if not (math.isnan(deep["mean"]) or math.isnan(clustering["mean"]))
                        else float("nan")
                    ),
                    "definition": "(M11 - M01) - (M10 - M00)",
                },
            }
        entry["safety"] = {arm: len(environments_holding(records[arm])) for arm in FACTORIAL}
        entry["non_degenerate"] = {
            arm: len(environments_non_degenerate(records[arm], floor)) for arm in FACTORIAL
        }
        entry["false_certifications"] = {
            arm: taxonomy_counts(arm_slice(cells, arm, budget), floor)[F4_FALSE_CERTIFICATION]
            for arm in FACTORIAL
        }
        entry["refusals"] = {
            arm: taxonomy_counts(arm_slice(cells, arm, budget), floor)[F1_NO_FEASIBLE_CUT]
            for arm in FACTORIAL
        }
        factorial[str(budget)] = entry

    _write_json_once(
        FACTORIAL_RESULTS,
        {
            **_envelope("factorial_2x2_results"),
            "design": {
                "A_certification_unit": [UNIT_CANDIDATE, UNIT_DOCUMENT],
                "B_prefix_control": [
                    f"{FAMILY_NINE}|{CONTROL_UNION}",
                    f"{FAMILY_FINE}|{CONTROL_FIXED_SEQUENCE}",
                ],
                "cells": {M00: "A0B0", M10: "A1B0", M01: "A0B1", M11: "A1B1"},
            },
            "coverage_floor": floor,
            "by_budget": factorial,
            "declared_baselines": {
                P1A: "candidate unit, fine family, union bound: resolution without gatekeeping",
                P1B: "document unit, fine family, union bound",
            },
            "diagnostics_never_selectable": {D0: "asymptotic", D1: "asymptotic", N0: "negative"},
            "runtime_seconds": time.monotonic() - started,
        },
    )
    for arm, path in (
        (M00, CANDIDATE_NINE_RESULTS),
        (M10, DOCUMENT_NINE_RESULTS),
        (M01, CANDIDATE_FINE_RESULTS),
        (M11, DOCUMENT_FINE_RESULTS),
    ):
        unit, family, control = next((u, f, c) for name, u, f, c in ARMS if name == arm)
        _write_json_once(
            path,
            {
                **_envelope(path.stem),
                "arm": arm,
                "unit": unit,
                "family": family,
                "control": control,
                "family_size": len(FAMILIES[family]),
                "coverage_floor": floor,
                "by_budget": by_arm[arm],
                "declared_baselines_measured_alongside": (
                    {P1A: by_arm[P1A], D0: by_arm[D0]}
                    if unit == UNIT_CANDIDATE
                    else {P1B: by_arm[P1B], D1: by_arm[D1]}
                ),
                "runtime_seconds": time.monotonic() - started,
            },
        )

    reference = factorial[str(PRACTICAL_DOC_BUDGET)]
    print(
        f"results: at {PRACTICAL_DOC_BUDGET} documents -- clustering effect "
        f"{reference['risk_controlled']['clustering_effect']['mean']:+.4f}, resolution "
        f"{reference['risk_controlled']['resolution_effect']['mean']:+.4f}, joint "
        f"{reference['risk_controlled']['joint_effect']['mean']:+.4f} "
        f"-> {_relative(FACTORIAL_RESULTS)}"
    )
    return 0


# ------------------------------------------ sections 27 and 28: complexity and annotation cost


def run_complexity() -> int:
    """Sample complexity in documents and in candidate labels, and what the labels cost."""
    started = time.monotonic()
    cells, _design = load_cells()
    floor = float(s15.load_frozen().coverage_floor)

    documents_curve: dict[str, Any] = {}
    candidates_curve: dict[str, Any] = {}
    cost: dict[str, Any] = {}
    for arm, _unit, _family, _control in ARMS:
        by_budget: dict[str, Any] = {}
        by_labels: dict[str, Any] = {}
        by_cost: dict[str, Any] = {}
        for budget in DOC_BUDGETS:
            block = arm_slice(cells, arm, budget)
            modes = block.apply(lambda row: classify(row, floor), axis=1)
            records = per_environment(block)
            safe = environments_non_degenerate(records, floor)
            labels = float(block["n_candidate_labels"].mean())
            by_budget[str(budget)] = {
                "documents_requested": budget,
                "documents_achieved_mean": float(block["n_docs"].mean()),
                "probability_safe_non_degenerate_deployment": float(
                    (modes == F3_NON_DEGENERATE).mean()
                ),
                "environments_safe_and_non_degenerate": len(safe),
                "environments": sorted(safe),
                "mean_candidate_labels": labels,
            }
            by_labels[f"{labels:.0f}"] = by_budget[str(budget)][
                "probability_safe_non_degenerate_deployment"
            ]
            per_document = block["n_candidate_labels"] / block["n_docs"].replace(0, np.nan)
            by_cost[str(budget)] = {
                "documents_labelled": float(block["n_docs"].mean()),
                "candidates_labelled": labels,
                "candidates_per_document_mean": float(per_document.mean()),
                "candidates_per_document_median": float(per_document.median()),
                "total_annotation_units": float(block["n_candidate_labels"].sum()),
                "safe_repair_recall": _mean(
                    [
                        records[name]["risk_controlled"]
                        for name in environments_non_degenerate(records, floor)
                    ]
                ),
                "mean_risk_controlled": _mean(
                    [record["risk_controlled"] for record in records.values()]
                ),
            }
        documents_curve[arm] = by_budget
        candidates_curve[arm] = by_labels
        cost[arm] = by_cost

    smallest = {
        arm: next(
            (
                budget
                for budget in DOC_BUDGETS
                if documents_curve[arm][str(budget)]["environments_safe_and_non_degenerate"]
                >= MAJORITY_REQUIRED
            ),
            None,
        )
        for arm in ARM_NAMES
    }
    _write_json_once(
        SAMPLE_COMPLEXITY_DOCUMENTS,
        {
            **_envelope("sample_complexity_documents"),
            "question": "how many genuinely independent target documents are needed?",
            "budgets": list(DOC_BUDGETS),
            "majority_required": MAJORITY_REQUIRED,
            "practical_ceiling": PRACTICAL_DOC_BUDGET,
            "by_arm": documents_curve,
            "smallest_budget_reaching_a_majority": smallest,
            "capacity_note": (
                "the certification halves hold between 19 and 50 documents, so a budget larger "
                "than an environment's pool is clipped to it and the achieved count is recorded "
                "beside the requested one"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        SAMPLE_COMPLEXITY_CANDIDATES,
        {
            **_envelope("sample_complexity_candidates"),
            "question": (
                "the same curve against the candidate-label axis SGV15 used, so the two stages "
                "can be compared on the annotation cost a deployment actually pays"
            ),
            "sgv15_candidate_budgets": list(s15.CERT_BUDGETS),
            "by_arm": candidates_curve,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        ANNOTATION_COST,
        {
            **_envelope("annotation_cost"),
            "rule": "selecting a certification document reveals every eligible candidate in it",
            "by_arm": cost,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "complexity: smallest budget reaching a majority -- "
        + ", ".join(f"{arm}={smallest[arm]}" for arm in FACTORIAL)
        + f" -> {_relative(SAMPLE_COMPLEXITY_DOCUMENTS)}"
    )
    return 0


# ------------------------------------------------------------- section 25: the twelve controls


def _permute_labels(geometry: Geometry, seed: int) -> np.ndarray:
    """Destroy the association between the score and the label, keeping the prevalence."""
    generator = np.random.default_rng(seed)
    return np.asarray(generator.permutation(geometry.harm), dtype=bool)


def _permute_document_labels(geometry: Geometry, seed: int) -> np.ndarray:
    """Move whole documents' label vectors between documents of the same size where possible.

    Within-document structure survives; between-document heterogeneity is reassigned. The
    document-aware bound should notice, because that heterogeneity is what it is priced against.
    """
    generator = np.random.default_rng(seed)
    labels = np.array(geometry.harm, dtype=bool)
    order = generator.permutation(geometry.documents.size)
    out = np.zeros_like(labels)
    for source, target in enumerate(order.tolist()):
        source_rows = np.flatnonzero(geometry.doc_of == source)
        target_rows = np.flatnonzero(geometry.doc_of == target)
        take = min(source_rows.size, target_rows.size)
        out[target_rows[:take]] = labels[source_rows[:take]]
    return out


def _shuffle_documents(geometry: Geometry, seed: int) -> np.ndarray:
    """Reassign candidates to documents at random, which removes the clustering entirely."""
    generator = np.random.default_rng(seed)
    return np.asarray(generator.permutation(geometry.doc_of), dtype=np.int64)


def _control_outcome(
    geometry: Geometry,
    unit: str,
    family: str,
    control: str,
    rows: np.ndarray,
    harm: np.ndarray | None = None,
) -> dict[str, Any]:
    decision, tau = geometry_certify(
        geometry, family, unit, control, rows, PRIMARY_EPSILON, harm=harm
    )
    point = geometry_deploy(geometry, tau, PRIMARY_EPSILON)
    return {
        "feasible": bool(decision.feasible),
        "deployed_depth": (
            float(FAMILIES[family][decision.selected]) if decision.feasible else float("nan")
        ),
        "prefixes_certified": len(decision.certified),
        **point,
    }


def _aggregate_control(records: dict[str, dict[str, Any]], floor: float) -> dict[str, Any]:
    values = list(records.values())
    return {
        "environments": len(values),
        "feasible": int(sum(1 for record in values if record["feasible"])),
        "environments_holding_epsilon": int(
            sum(
                1
                for record in values
                if record["feasible"]
                and not math.isnan(record["realized_harm_rate"])
                and record["realized_harm_rate"] <= PRIMARY_EPSILON
            )
        ),
        "environments_violating_epsilon": int(
            sum(
                1
                for record in values
                if record["feasible"]
                and not math.isnan(record["realized_harm_rate"])
                and record["realized_harm_rate"] > PRIMARY_EPSILON
            )
        ),
        "environments_non_degenerate": int(
            sum(1 for record in values if record["feasible"] and record["coverage"] >= floor)
        ),
        "mean_risk_controlled": _mean([record["risk_controlled"] for record in values]),
        "mean_coverage": _mean([record["coverage"] for record in values]),
        "mean_realized_harm_rate": _mean([record["realized_harm_rate"] for record in values]),
        "per_environment": records,
    }


def run_controls() -> int:
    """The twelve controls the brief requires, and the discrimination checks they support."""
    started = time.monotonic()
    cells, _design = load_cells()
    geometry = load_geometry()
    floor = float(s15.load_frozen().coverage_floor)
    duplication = cc_read_json(DOCUMENT_DUPLICATION_TEST)

    def from_cells(arm: str, budget: int = PRACTICAL_DOC_BUDGET) -> dict[str, Any]:
        # The oracles carry no document budget -- they buy no labels -- so they are sliced at 0.
        records = per_environment(arm_slice(cells, arm, budget))
        return {
            "environments": len(records),
            "feasible": int(
                sum(1 for record in records.values() if record["cut_feasible_fraction"] > 0.0)
            ),
            "environments_holding_the_bound": len(environments_holding(records)),
            "environments_non_degenerate": len(environments_non_degenerate(records, floor)),
            "mean_risk_controlled": _mean(
                [record["risk_controlled"] for record in records.values()]
            ),
            "mean_coverage": _mean([record["coverage"] for record in records.values()]),
            "mean_realized_harm_rate": _mean(
                [record["realized_harm_rate"] for record in records.values()]
            ),
            "refusal_rate": float(1.0 - arm_slice(cells, arm, budget)["cut_feasible"].mean()),
            "per_environment": records,
        }

    controls: dict[str, Any] = {
        "c1_sgv15_candidate_nine_depth": {
            "purpose": (
                "SGV15's own method at SGV15b's budget: the baseline everything else moves from"
            ),
            **from_cells(M00),
        },
        "c2_candidate_fine_grid": {
            "purpose": "the same inference unit on the finer family under a union bound",
            **from_cells(P1A),
        },
        "c3_document_nine_depth": {
            "purpose": "the frozen family, certified at the document level",
            **from_cells(M10),
        },
        "c4_document_fine_grid": {
            "purpose": "both changes at once, under a union bound",
            **from_cells(P1B),
        },
        "c5_naive_empirical_fine_search": {
            "purpose": (
                "the observed rate at face value on a hundred prefixes: a registered negative"
            ),
            **from_cells(N0),
        },
        "c12_unrestricted_evaluation_oracle": {
            "purpose": "the ceiling any procedure on this ranking could reach. ANALYSIS ONLY.",
            **from_cells(O0_UNRESTRICTED, budget=0),
        },
    }

    units = (UNIT_CANDIDATE, UNIT_DOCUMENT)
    # Every degenerate and permutation control is run at BOTH inference units. Running them only
    # at the document unit would make the discrimination checks vacuous whenever that unit
    # refuses everywhere, which is a property of the result and must not decide what is measured.
    gathered: dict[str, dict[str, dict[str, Any]]] = {
        key: {unit: {} for unit in units}
        for key in ("permuted", "document_permuted", "one", "few", "shuffled")
    }
    for name, item in geometry.items():
        rows = document_sample_rows(item, PRACTICAL_DOC_BUDGET, name)
        seed = _stable_seed("sgv15b-control", str(SAMPLE_SEED), name)
        labels = _permute_labels(item, seed)
        document_labels = _permute_document_labels(item, seed + 1)
        reassigned = Geometry(
            name=item.name,
            corpus=item.corpus,
            base_engine=item.base_engine,
            documents=item.documents,
            doc_of=_shuffle_documents(item, seed + 2),
            scores=item.scores,
            harm=item.harm,
            thresholds=item.thresholds,
            eval_scores=item.eval_scores,
            eval_harm=item.eval_harm,
            eval_beneficial=item.eval_beneficial,
            eval_documents=item.eval_documents,
        )
        for unit in units:
            gathered["permuted"][unit][name] = _control_outcome(
                item, unit, FAMILY_NINE, CONTROL_UNION, rows, labels
            )
            gathered["document_permuted"][unit][name] = _control_outcome(
                item, unit, FAMILY_NINE, CONTROL_UNION, rows, document_labels
            )
            gathered["one"][unit][name] = _control_outcome(
                item, unit, FAMILY_NINE, CONTROL_UNION, document_sample_rows(item, 1, name)
            )
            gathered["few"][unit][name] = _control_outcome(
                item,
                unit,
                FAMILY_NINE,
                CONTROL_UNION,
                document_sample_rows(item, s15.FEW_DOCUMENTS, name),
            )
            gathered["shuffled"][unit][name] = _control_outcome(
                reassigned, unit, FAMILY_NINE, CONTROL_UNION, rows
            )

    def both(key: str, purpose: str) -> dict[str, Any]:
        return {
            "purpose": purpose,
            "by_unit": {unit: _aggregate_control(gathered[key][unit], floor) for unit in units},
        }

    controls["c6_permuted_candidate_labels"] = both(
        "permuted", "destroy the score-to-label association at the candidate level"
    )
    controls["c7_permuted_document_labels"] = both(
        "document_permuted", "move whole documents' label vectors between documents"
    )
    controls["c8_one_document_certification"] = both(
        "one", "one document is one independent observation and must not certify"
    )
    controls["c9_few_document_certification"] = both(
        "few", f"{s15.FEW_DOCUMENTS} documents, the same question with slightly more evidence"
    )
    controls["c11_shuffled_document_assignment"] = both(
        "shuffled", "reassign candidates to documents at random, which removes the clustering"
    )
    controls["c10_duplicated_candidate_rows"] = {
        "purpose": "duplicate every candidate inside every document; no independent unit is added",
        "document_aware_invariant": int(duplication["summary"]["document_aware_invariant"]),
        "candidate_level_bound_shrinks": int(
            duplication["summary"]["candidate_level_bound_shrinks"]
        ),
        "environments": int(duplication["summary"]["environments"]),
        "falsification_passes": bool(duplication["summary"]["falsification_passes"]),
    }

    def unit_of(key: str, unit: str) -> dict[str, Any]:
        return dict(controls[key]["by_unit"][unit])

    baseline = controls["c1_sgv15_candidate_nine_depth"]
    document_arm = controls["c3_document_nine_depth"]
    checks = {
        "the_ranking_is_load_bearing": bool(
            baseline["mean_risk_controlled"]
            > unit_of("c6_permuted_candidate_labels", UNIT_CANDIDATE)["mean_risk_controlled"]
        ),
        "certification_labels_carry_information": bool(
            unit_of("c6_permuted_candidate_labels", UNIT_CANDIDATE)[
                "environments_violating_epsilon"
            ]
            > 0
            or unit_of("c6_permuted_candidate_labels", UNIT_CANDIDATE)["mean_coverage"]
            != baseline["mean_coverage"]
        ),
        "document_diversity_matters": bool(
            unit_of("c8_one_document_certification", UNIT_CANDIDATE)["environments_non_degenerate"]
            <= unit_of("c9_few_document_certification", UNIT_CANDIDATE)[
                "environments_non_degenerate"
            ]
            <= baseline["environments_non_degenerate"]
        ),
        "one_document_cannot_certify_at_the_document_level": bool(
            unit_of("c8_one_document_certification", UNIT_DOCUMENT)["feasible"] == 0
        ),
        "removing_the_clustering_makes_the_document_bound_feasible": bool(
            unit_of("c11_shuffled_document_assignment", UNIT_DOCUMENT)["feasible"]
            > document_arm["feasible"]
        ),
        "duplication_falsification_passes": bool(
            controls["c10_duplicated_candidate_rows"]["falsification_passes"]
        ),
        "the_naive_rule_is_the_unsafe_one": bool(
            controls["c5_naive_empirical_fine_search"]["environments_holding_the_bound"]
            <= document_arm["environments_holding_the_bound"]
        ),
    }
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_envelope("control_results"),
            "budget_documents": PRACTICAL_DOC_BUDGET,
            "epsilon": PRIMARY_EPSILON,
            "coverage_floor": floor,
            "controls": controls,
            "discrimination_checks": checks,
            "interpretation": {
                "removing_the_clustering_makes_the_document_bound_feasible": (
                    "reassigning candidates to documents at random destroys the within-document "
                    "correlation and leaves the number of documents and the support width "
                    "untouched. If the document-level certificate still refuses, its refusal is "
                    "priced by the ratio of the largest document's accepted count to the number "
                    "of documents rather than by the measured correlation, and the two are "
                    "different explanations of the same refusal."
                ),
                "candidate_unit_under_shuffling": (
                    "the candidate-level arm is measured under the same shuffle, where its "
                    "independence assumption is actually true, so the pair separates 'the "
                    "assumption was wrong' from 'there were not enough documents'."
                ),
            },
            "checks_positive": int(sum(1 for value in checks.values() if value)),
            "checks_total": len(checks),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"controls: {len(controls)} controls, "
        f"{sum(1 for value in checks.values() if value)}/{len(checks)} discrimination checks "
        f"positive -> {_relative(CONTROL_RESULTS)}"
    )
    return 0


# ------------------------------------------------------------- the falsification tests


def _test(
    results: list[dict[str, Any]], name: str, question: str, passed: bool, evidence: Any
) -> None:
    results.append(
        {"test": name, "question": question, "passed": bool(passed), "evidence": evidence}
    )


def run_negative() -> int:
    """Tests that would fail if the stage's claims were false. Each names its own evidence."""
    started = time.monotonic()
    cells, design = load_cells()
    geometry = load_geometry()
    simulation = cc_read_json(SIMULATION_COVERAGE)
    prefixes = cc_read_json(PREFIX_CONTROL_VALIDATION)
    duplication = cc_read_json(DOCUMENT_DUPLICATION_TEST)
    reproduction = cc_read_json(SGV15_REPRODUCTION)
    results: list[dict[str, Any]] = []

    _test(
        results,
        "n1_the_oracle_never_reaches_a_selection",
        "can an arm that reads evaluation labels be selected?",
        all(oracle not in SELECTABLE_ARMS for oracle in ORACLES)
        and all(oracle in NOT_SELECTABLE_ARMS for oracle in ORACLES),
        {"selection_pool": list(SELECTABLE_ARMS), "excluded": list(NOT_SELECTABLE_ARMS)},
    )
    _test(
        results,
        "n2_the_asymptotic_arms_never_reach_a_selection",
        "can a limit theorem be deployed as a certificate?",
        D0 not in SELECTABLE_ARMS and D1 not in SELECTABLE_ARMS and N0 not in SELECTABLE_ARMS,
        {"asymptotic": [D0, D1], "negative": [N0]},
    )
    frozen_partition = pd.read_parquet(s15.CERTIFICATION_ROWS)
    evaluation = pd.read_parquet(
        s14.ENVIRONMENT_ROWS, columns=["environment", "document_id", "role"]
    )
    overlaps = {}
    for name, item in geometry.items():
        evaluation_documents = set(
            evaluation[
                (evaluation["environment"] == name) & (evaluation["role"] == s14.ROLE_EVALUATION)
            ]["document_id"].astype(str)
        )
        overlaps[name] = len(set(item.documents.tolist()) & evaluation_documents)
    _test(
        results,
        "n3_certification_and_evaluation_documents_are_disjoint",
        "can a document be certified on and evaluated on in the same environment?",
        all(value == 0 for value in overlaps.values()),
        overlaps,
    )
    fit_documents = frozen_partition[frozen_partition["sgv15_role"] == s15.ROLE_FIT]
    adaptation_overlap = {}
    for name, item in geometry.items():
        block = set(fit_documents[fit_documents["environment"] == name]["document_id"].astype(str))
        adaptation_overlap[name] = len(set(item.documents.tolist()) & block)
    _test(
        results,
        "n4_adaptation_and_certification_documents_are_disjoint",
        "can a document both fit the ranking and certify the cut?",
        all(value == 0 for value in adaptation_overlap.values()),
        adaptation_overlap,
    )
    _test(
        results,
        "n5_m00_reproduces_sgv15_exactly",
        "is the baseline arm SGV15's own method, or a lookalike?",
        bool(reproduction["comparison"]["identical"])
        and reproduction["comparison"]["max_absolute_numeric_difference"] == 0.0,
        reproduction["comparison"],
    )
    _test(
        results,
        "n6_the_fine_family_strictly_refines_the_frozen_one",
        "is the resolution comparison a comparison of resolution, or of range?",
        all(depth in FINE_DEPTHS for depth in NINE_DEPTHS)
        and min(FINE_DEPTHS) <= min(NINE_DEPTHS)
        and max(FINE_DEPTHS) == max(NINE_DEPTHS),
        {
            "nine": list(NINE_DEPTHS),
            "fine_size": len(FINE_DEPTHS),
            "fine_range": [min(FINE_DEPTHS), max(FINE_DEPTHS)],
        },
    )
    label_free = {}
    for name, item in geometry.items():
        permuted = _permute_labels(item, 17)
        base_harm, base_accept = item.counts(FAMILY_NINE)
        other_harm, other_accept = item.counts(FAMILY_NINE, permuted)
        label_free[name] = {
            "accepted_counts_identical": bool(np.array_equal(base_accept, other_accept)),
            "span_identical": bool(
                np.array_equal(base_accept.max(axis=0), other_accept.max(axis=0))
            ),
            "harm_counts_changed": bool(not np.array_equal(base_harm, other_harm)),
        }
    _test(
        results,
        "n7_the_prefix_family_and_its_support_are_label_free",
        "does the declared family or the bound's support move when the labels are permuted?",
        all(
            record["accepted_counts_identical"] and record["span_identical"]
            for record in label_free.values()
        ),
        label_free,
    )
    nested = {
        name: bool(np.all(np.diff(item.counts(FAMILY_FINE)[1], axis=1) >= 0))
        for name, item in geometry.items()
    }
    _test(
        results,
        "n8_the_declared_prefixes_are_nested",
        "does a deeper prefix ever accept fewer candidates than a shallower one?",
        all(nested.values()),
        nested,
    )
    leaked = cells[(~cells["cut_feasible"]) & (cells["n_accepted"] > 0)]
    _test(
        results,
        "n9_no_refusal_deployed_an_edit",
        "does a method that refused still accept edits on the evaluation block?",
        len(leaked) == 0,
        {"cells": len(leaked), "total_cells": len(cells)},
    )
    _test(
        results,
        "n10_the_document_aware_certificate_is_duplication_invariant",
        "does duplicating candidates inside a document change a document-level decision?",
        int(duplication["summary"]["document_aware_invariant"])
        == int(duplication["summary"]["environments"]),
        duplication["summary"],
    )
    _test(
        results,
        "n11_the_candidate_level_certificate_is_not",
        "would the falsification be vacuous because the duplication did nothing?",
        int(duplication["summary"]["candidate_level_bound_shrinks"])
        == int(duplication["summary"]["environments"]),
        duplication["summary"],
    )
    worst = simulation["worst_false_certification_rate"]
    _test(
        results,
        "n12_the_finite_sample_bounds_hold_their_level_under_clustering",
        "does the declared certificate over-certify on data whose dependence is known?",
        all(
            worst[method] <= ALPHA
            for method in ("union_finite", "hoeffding", "empirical_bernstein", "betting")
        ),
        worst,
    )
    _test(
        results,
        "n13_the_candidate_level_bound_does_not",
        "would that check be vacuous because the simulation is too easy?",
        worst["candidate_clopper_pearson"] > ALPHA,
        {"candidate_clopper_pearson": worst["candidate_clopper_pearson"], "alpha": ALPHA},
    )
    _test(
        results,
        "n14_every_prefix_control_holds_its_familywise_level",
        "does any control procedure deploy past the known safe frontier more often than alpha?",
        all(
            record[key]["familywise_false_certification_rate"] <= ALPHA
            for record in prefixes["by_documents"].values()
            for key in record
        ),
        {
            documents: {key: record[key]["familywise_false_certification_rate"] for key in record}
            for documents, record in prefixes["by_documents"].items()
        },
    )
    clipped = cells[cells["clipped"]]
    _test(
        results,
        "n15_a_clipped_budget_is_recorded_as_clipped",
        "is a nominal document budget ever reported as achieved when the pool was smaller?",
        bool((clipped["n_docs"] < clipped["requested_docs"]).all())
        and bool(
            (cells[~cells["clipped"]]["n_docs"] >= cells[~cells["clipped"]]["requested_docs"]).all()
        ),
        {
            "clipped_cells": len(clipped),
            "environments_ever_clipped": sorted(clipped["environment"].unique().tolist()),
        },
    )
    _test(
        results,
        "n16_the_criteria_predate_the_endpoints",
        "were the criteria written before the numbers they judge?",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(FACTORIAL_CELLS.stat().st_mtime))
        >= design["issued_utc"]
        and set(design["criteria"])
        == {
            "1_safety",
            "2_non_degenerate_safe_deployment",
            "3_operational_improvement",
            "4_cluster_awareness_is_load_bearing",
            "5_fine_prefix_control_is_load_bearing",
            "6_false_certification_control",
            "7_practical_sample_complexity",
        },
        {
            "design_record_issued": design["issued_utc"],
            "cells_written": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(FACTORIAL_CELLS.stat().st_mtime)
            ),
            "criteria": sorted(design["criteria"]),
        },
    )

    passed = sum(1 for record in results if record["passed"])
    _write_json_once(
        NEGATIVE_TESTS,
        {
            **_envelope("negative_tests"),
            "tests": results,
            "passed": passed,
            "total": len(results),
            "failed": [record["test"] for record in results if not record["passed"]],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"negative: {passed}/{len(results)} falsification tests pass -> {_relative(NEGATIVE_TESTS)}"
    )
    return 0


# ------------------------------------------------------- section 29: nested selection


CELL_FIELDS = ("environment", "budget", "draw", "arm")


def indexed_counts() -> pd.DataFrame:
    """The per-document deployment counts with the cell key split back into its fields."""
    counts = pd.read_parquet(DOCUMENT_COUNTS)
    parts = counts["cell"].astype(str).str.split("|", expand=True)
    parts.columns = list(CELL_FIELDS)
    return pd.concat([counts.reset_index(drop=True), parts], axis=1)


def cell_block(indexed: pd.DataFrame, **filters: str) -> Any:
    """One cell's per-document counts, one matrix row per draw, in sorted document order."""
    block = indexed
    for column, value in filters.items():
        block = block[block[column] == value]
    if block.empty:
        raise PhaseError(f"no deployment counts for {filters}")
    documents = tuple(sorted({str(name) for name in block["document_id"].tolist()}))
    draws = sorted({str(value) for value in block["draw"].tolist()})
    position = {name: index for index, name in enumerate(documents)}
    row_of = {draw: index for index, draw in enumerate(draws)}
    names = ("accepted", "harmful_accepted", "beneficial_accepted", "beneficial_total")
    matrices = {name: np.zeros((len(draws), len(documents)), dtype=float) for name in names}
    rows = np.array([row_of[str(v)] for v in block["draw"].tolist()], dtype=int)
    columns = np.array([position[str(v)] for v in block["document_id"].tolist()], dtype=int)
    for name in names:
        matrices[name][rows, columns] = block[name].to_numpy(dtype=float)
    return s13.CountBlock(
        documents=documents,
        accepted=matrices["accepted"],
        harmful=matrices["harmful_accepted"],
        beneficial=matrices["beneficial_accepted"],
        total=matrices["beneficial_total"],
    )


def _config_table(cells: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
    """Every selectable (arm, document budget) configuration's per-environment endpoint.

    The pool is `SELECTABLE_ARMS`. The two asymptotic arms and the naive arm are measured in
    every table above and cannot appear here, for the same reason the oracle cannot: a diagnostic
    that can be selected is no longer a diagnostic.
    """
    table: dict[tuple[str, int], dict[str, float]] = {}
    for arm in SELECTABLE_ARMS:
        for budget in DOC_BUDGETS:
            block = arm_slice(cells, arm, budget)
            if block.empty:
                continue
            table[(arm, budget)] = {
                str(name): float(group["risk_controlled"].mean())
                for name, group in block.groupby("environment", observed=True)
            }
    return table


def _nested(table: dict[Any, dict[str, float]], environments: list[str]) -> dict[str, Any]:
    """Leave-one-environment-out selection, and the held-out value of whatever it chose."""
    chosen: dict[str, Any] = {}
    for held_out in environments:
        development = [name for name in environments if name != held_out]
        best: tuple[float, tuple[str, ...]] | None = None
        for config, values in table.items():
            inner = [values[name] for name in development if name in values]
            if not inner:
                continue
            score = float(np.mean(inner))
            label = tuple(str(part) for part in config)
            if best is None or score > best[0] or (score == best[0] and label < best[1]):
                best = (score, label)
        if best is None:
            continue
        config = next(key for key in table if tuple(str(p) for p in key) == best[1])
        chosen[held_out] = {
            "selected": list(best[1]),
            "development_mean": best[0],
            "held_out_value": table[config].get(held_out, float("nan")),
        }
    frequency: dict[str, int] = {}
    for entry in chosen.values():
        key = "|".join(entry["selected"])
        frequency[key] = frequency.get(key, 0) + 1
    return {
        "by_held_out_environment": chosen,
        "selection_frequency": frequency,
        "distinct_configurations_selected": len(frequency),
        "modal_configuration": max(frequency, key=lambda k: frequency[k]) if frequency else None,
        "modal_share": max(frequency.values()) / len(chosen) if chosen else float("nan"),
        "mean_held_out_endpoint": _mean([entry["held_out_value"] for entry in chosen.values()]),
        "stable": bool(frequency and max(frequency.values()) == len(chosen)),
    }


def run_select() -> int:
    """Leave-one-environment-out selection, and the pooled maximum it exists to distrust."""
    started = time.monotonic()
    cells, design = load_cells()
    environments = list(design["environments"])
    table = _config_table(cells)
    nested = _nested(table, environments)
    pooled = {
        "|".join(str(part) for part in config): _mean(list(values.values()))
        for config, values in table.items()
    }
    pooled_best = max(pooled, key=lambda k: pooled[k]) if pooled else None
    modal = nested["modal_configuration"]
    selected = modal.split("|") if modal else None
    payload = {
        **_envelope("nested_selection"),
        "scientific_status": "DEVELOPMENT -- these ten environments are exposed, not held out",
        "procedure": design["selection_plan"],
        "endpoint": f"risk-controlled repair recall at epsilon = {PRIMARY_EPSILON}",
        "selection_pool": {"arms": list(SELECTABLE_ARMS), "budgets": list(DOC_BUDGETS)},
        "excluded_from_selection": {
            D0: "asymptotic: a limit theorem, not a certificate",
            D1: "asymptotic: a limit theorem, not a certificate",
            N0: "registered negative: the observed rate at face value",
            **dict.fromkeys(ORACLES, "ANALYSIS ONLY: reads evaluation labels"),
        },
        "candidate_configurations": len(table),
        "nested_leave_one_environment_out": nested,
        "pooled_maximum": {
            "configuration": pooled_best,
            "value": pooled.get(pooled_best) if pooled_best else None,
            "why_it_is_not_the_answer": (
                "the pooled maximum is chosen on the same ten environments it is read on. It is "
                "recorded so the optimism of that shortcut is visible, and it does not select."
            ),
        },
        "selected_arm": selected[0] if selected else None,
        "selected_budget": int(selected[1]) if selected else None,
        "selection_is_stable": nested["stable"],
        "instability_note": (
            None
            if nested["stable"]
            else (
                "the leave-one-environment-out selection did not converge on one configuration; "
                f"{nested['distinct_configurations_selected']} distinct configurations were "
                "chosen across the ten folds. The modal choice is reported as the candidate and "
                "the instability is part of the finding."
            )
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(NESTED_SELECTION, payload)
    print(
        f"select: {len(table)} configurations, modal choice {modal} "
        f"(stable={nested['stable']}) -> {_relative(NESTED_SELECTION)}"
    )
    return 0


def selected_configuration() -> tuple[str, int]:
    """Whatever the nested selection chose, read from its artifact and never re-derived."""
    if not NESTED_SELECTION.is_file():
        raise PhaseError("run --select first; the statistics test the SELECTED arm")
    record = cc_read_json(NESTED_SELECTION)
    if record["selected_arm"] is None:
        raise PhaseError("the nested selection chose nothing; there is no arm to test")
    return str(record["selected_arm"]), int(record["selected_budget"])


# ------------------------------------------------------------------ the statistical tests


def run_stats() -> int:
    """The selected arm against SGV15's own method, on the same labels and the same documents."""
    started = time.monotonic()
    _cells, design = load_cells()
    arm, budget = selected_configuration()
    indexed = indexed_counts()
    environments = list(design["environments"])
    comparator = M00 if arm != M00 else M10

    def blocks(name: str) -> tuple[Any, Any]:
        return (
            cell_block(indexed, environment=name, budget=str(budget), arm=arm),
            cell_block(indexed, environment=name, budget=str(budget), arm=comparator),
        )

    paired = [blocks(name) for name in environments]
    primary = s15.stratified_delta(paired, s15.RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED)
    harm_overall = s15.stratified_delta(paired, s15.HARM, PRIMARY_EPSILON, BOOTSTRAP_SEED)
    family: dict[str, dict[str, float]] = {}
    for name, (left, right) in zip(environments, paired, strict=True):
        family[f"{name}|repair_recall"] = s13.paired_delta(
            left, right, s15.RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED
        )
        family[f"{name}|realized_harm"] = s13.paired_delta(
            left, right, s15.HARM, PRIMARY_EPSILON, BOOTSTRAP_SEED
        )
    corrected = holm(family)
    surviving = [key for key, value in corrected.items() if value["survives_holm"]]
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_envelope("statistical_tests"),
            "scientific_status": (
                "DEVELOPMENT. These are the environments SGV14 already exposed; a surviving "
                "p-value here is not external confirmation."
            ),
            "selected_configuration": {"arm": arm, "document_budget": budget},
            "comparator": f"{comparator} on the same documents and the same labels",
            "why_that_comparator": (
                "the factorial's whole point is that the two arms differ only in how the risk "
                "estimate is computed. Both buy the same documents and reveal the same "
                "candidates, so the paired difference isolates the inference unit."
            ),
            "resampling_unit": "document",
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_endpoint_repair_recall": primary,
            "primary_endpoint_realized_harm": harm_overall,
            "family": corrected,
            "family_size": len(family),
            "multiplicity": "Holm-Bonferroni at 0.05",
            "surviving_after_holm": surviving,
            "count_surviving": len(surviving),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"stats: delta {primary['delta']:+.4f} CI [{primary['ci_lower']:+.4f}, "
        f"{primary['ci_upper']:+.4f}] p={primary['p_value']:.4f}, {len(surviving)}/{len(family)} "
        f"survive Holm -> {_relative(STATISTICAL_TESTS)}"
    )
    return 0


# ------------------------------------------------- sections 30, 31 and 32: the research decision


def _useful(cells: pd.DataFrame, arm: str, floor: float) -> dict[str, Any]:
    """Does this arm reach safe, non-degenerate deployment on a majority, at a practical budget?"""
    best_budget: int | None = None
    best_count = 0
    for budget in DOC_BUDGETS:
        if budget > PRACTICAL_DOC_BUDGET:
            continue
        records = per_environment(arm_slice(cells, arm, budget))
        count = len(environments_non_degenerate(records, floor))
        if count > best_count:
            best_budget, best_count = budget, count
    return {
        "budget": best_budget,
        "environments": best_count,
        "useful": bool(best_count >= MAJORITY_REQUIRED),
        "majority_required": MAJORITY_REQUIRED,
    }


def run_decide() -> int:
    """The seven pre-registered criteria, the outcome, and the SGV16 gate."""
    started = time.monotonic()
    cells, design = load_cells()
    floor = float(s15.load_frozen().coverage_floor)
    arm, budget = selected_configuration()
    selection = cc_read_json(NESTED_SELECTION)
    factorial = cc_read_json(FACTORIAL_RESULTS)
    resolution = cc_read_json(GRID_RESOLUTION_ANALYSIS)
    clustering = cc_read_json(EFFECTIVE_SAMPLE_SIZE)
    duplication = cc_read_json(DOCUMENT_DUPLICATION_TEST)
    complexity = cc_read_json(SAMPLE_COMPLEXITY_DOCUMENTS)
    controls = cc_read_json(CONTROL_RESULTS)
    negatives = cc_read_json(NEGATIVE_TESTS)
    statistics = cc_read_json(STATISTICAL_TESTS)
    simulation = cc_read_json(SIMULATION_COVERAGE)

    selected = per_environment(arm_slice(cells, arm, budget))
    baseline = per_environment(arm_slice(cells, M00, budget))
    holding = environments_holding(selected)
    non_degenerate = environments_non_degenerate(selected, floor)
    improved = sorted(
        name
        for name in selected
        if name in baseline
        and selected[name]["risk_controlled"] > baseline[name]["risk_controlled"]
    )
    selected_modes = taxonomy_counts(arm_slice(cells, arm, budget), floor)
    baseline_modes = taxonomy_counts(arm_slice(cells, M00, budget), floor)
    selected_cells = len(arm_slice(cells, arm, budget))
    baseline_cells = len(arm_slice(cells, M00, budget))
    false_rate = selected_modes[F4_FALSE_CERTIFICATION] / selected_cells if selected_cells else 0.0
    baseline_rate = (
        baseline_modes[F4_FALSE_CERTIFICATION] / baseline_cells if baseline_cells else 0.0
    )

    wider = sum(
        1
        for record in clustering["per_environment"].values()
        if record["depths_where_the_candidate_bound_is_below"] > 0
    )
    document_f4 = sum(
        taxonomy_counts(arm_slice(cells, M10, b), floor)[F4_FALSE_CERTIFICATION]
        for b in DOC_BUDGETS
    )
    candidate_f4 = sum(
        taxonomy_counts(arm_slice(cells, M00, b), floor)[F4_FALSE_CERTIFICATION]
        for b in DOC_BUDGETS
    )
    fine_mean = _mean(
        [
            factorial["by_budget"][str(b)]["risk_controlled"]["resolution_effect"]["mean"]
            for b in DOC_BUDGETS
        ]
    )

    criteria = {
        "1_safety": {
            "requirement": design["criteria"]["1_safety"],
            "required": BREADTH_REQUIRED,
            "observed": len(holding),
            "environments": holding,
            "met": len(holding) >= BREADTH_REQUIRED,
        },
        "2_non_degenerate_safe_deployment": {
            "requirement": design["criteria"]["2_non_degenerate_safe_deployment"],
            "required": BREADTH_REQUIRED,
            "observed": len(non_degenerate),
            "environments": non_degenerate,
            "coverage_floor": floor,
            "met": len(non_degenerate) >= BREADTH_REQUIRED,
        },
        "3_operational_improvement": {
            "requirement": design["criteria"]["3_operational_improvement"],
            "comparator": f"{M00}, which is SGV15's m2_certified_clopper_pearson procedure",
            "required": BREADTH_REQUIRED,
            "observed": len(improved),
            "environments": improved,
            "sgv15_frozen_selected_endpoint": float(
                cc_read_json(s15.DECISION)["criteria"]["4_representative_sampling_matters"][
                    "observed"
                ]["by_sampler"]["cs2_document_stratified"]["mean_risk_controlled"]
            ),
            "met": len(improved) >= BREADTH_REQUIRED,
        },
        "4_cluster_awareness_is_load_bearing": {
            "requirement": design["criteria"]["4_cluster_awareness_is_load_bearing"],
            "duplication_falsification_passes": bool(
                duplication["summary"]["falsification_passes"]
            ),
            "environments_where_the_candidate_bound_is_below": wider,
            "required_environments": BREADTH_REQUIRED,
            "candidate_false_certifications": candidate_f4,
            "document_false_certifications": document_f4,
            "simulation_candidate_false_certification_rate": float(
                simulation["worst_false_certification_rate"]["candidate_clopper_pearson"]
            ),
            "simulation_document_false_certification_rate": float(
                simulation["worst_false_certification_rate"]["union_finite"]
            ),
            "met": bool(
                duplication["summary"]["falsification_passes"]
                and wider >= BREADTH_REQUIRED
                and candidate_f4 > document_f4
            ),
        },
        "5_fine_prefix_control_is_load_bearing": {
            "requirement": design["criteria"]["5_fine_prefix_control_is_load_bearing"],
            "oracle_recovered_fraction": float(resolution["recovered_fraction"]),
            "required_fraction": RESOLUTION_RECOVERY_REQUIRED,
            "mean_resolution_effect_on_risk_controlled": fine_mean,
            "met": bool(
                resolution["recovered_fraction"] >= RESOLUTION_RECOVERY_REQUIRED
                and not math.isnan(fine_mean)
                and fine_mean > 0.0
            ),
        },
        "6_false_certification_control": {
            "requirement": design["criteria"]["6_false_certification_control"],
            "selected_false_certification_rate": false_rate,
            "baseline_false_certification_rate": baseline_rate,
            "alpha": ALPHA,
            "met": bool(false_rate <= ALPHA and false_rate <= baseline_rate),
        },
        "7_practical_sample_complexity": {
            "requirement": design["criteria"]["7_practical_sample_complexity"],
            "practical_ceiling": PRACTICAL_DOC_BUDGET,
            "majority_required": MAJORITY_REQUIRED,
            "best_practical_budget": _useful(cells, arm, floor),
            "met": bool(_useful(cells, arm, floor)["useful"]),
        },
    }
    met = sum(1 for record in criteria.values() if record["met"])

    useful = {name: _useful(cells, name, floor) for name in FACTORIAL}
    if not useful[M11]["useful"]:
        label = "D"
        summary = (
            "neither a document-aware inference unit nor a finer declared prefix family, alone or "
            "together, produces safe non-degenerate deployment at a practical document budget"
        )
    elif useful[M10]["useful"] and not useful[M01]["useful"]:
        label = "A"
        summary = "the inference unit was the dominant bottleneck"
    elif useful[M01]["useful"] and not useful[M10]["useful"]:
        label = "B"
        summary = "the action resolution was the dominant bottleneck"
    else:
        label = "C"
        summary = "both mechanisms are load-bearing"

    ready = met == len(criteria)
    payload = {
        **_envelope("research_decision"),
        "issued_head": _git("rev-parse", "HEAD"),
        "scientific_status": {
            "kind": "DEVELOPMENT",
            "why": (
                "SGV14 already exposed all ten environments and this stage reads their outcomes. "
                "Nothing here is confirmatory however well it performs."
            ),
            "confirmatory_stage": (
                "SGV16 would evaluate a frozen rule on environments nothing has touched"
            ),
            "prohibited_language": design["prohibited"],
        },
        "verdict": "SUPPORTED" if ready else "NOT SUPPORTED",
        "criteria": criteria,
        "criteria_met": met,
        "criteria_total": len(criteria),
        "outcome": {
            "label": label,
            "summary": summary,
            "evidence": {
                "useful_safe_deployment_at_a_practical_budget": useful,
                "clustering_effect_on_risk_controlled": {
                    str(b): factorial["by_budget"][str(b)]["risk_controlled"]["clustering_effect"][
                        "mean"
                    ]
                    for b in DOC_BUDGETS
                },
                "resolution_effect_on_risk_controlled": {
                    str(b): factorial["by_budget"][str(b)]["risk_controlled"]["resolution_effect"][
                        "mean"
                    ]
                    for b in DOC_BUDGETS
                },
                "joint_effect_on_risk_controlled": {
                    str(b): factorial["by_budget"][str(b)]["risk_controlled"]["joint_effect"][
                        "mean"
                    ]
                    for b in DOC_BUDGETS
                },
                "interaction_on_risk_controlled": {
                    str(b): factorial["by_budget"][str(b)]["risk_controlled"]["interaction"]["mean"]
                    for b in DOC_BUDGETS
                },
            },
        },
        "selected_arm": arm,
        "selected_budget": budget,
        "selection_is_stable": bool(selection["selection_is_stable"]),
        "sgv15_reproduction": {
            "identical": bool(cc_read_json(SGV15_REPRODUCTION)["comparison"]["identical"]),
            "max_absolute_numeric_difference": float(
                cc_read_json(SGV15_REPRODUCTION)["comparison"]["max_absolute_numeric_difference"]
            ),
            "cells": int(cc_read_json(SGV15_REPRODUCTION)["comparison"]["rows_regenerated"]),
        },
        "failure_taxonomy": {
            "selected": selected_modes,
            "baseline_m00": baseline_modes,
            "by_arm_at_the_practical_budget": {
                name: taxonomy_counts(arm_slice(cells, name, PRACTICAL_DOC_BUDGET), floor)
                for name in ARM_NAMES
            },
        },
        "sample_complexity": {
            "unit": "documents",
            "smallest_budget_reaching_a_majority": complexity[
                "smallest_budget_reaching_a_majority"
            ],
            "declared_budgets": list(DOC_BUDGETS),
            "capacity": {
                "certification_documents_per_environment": {
                    record["environment"]: record["certification_documents"]
                    for record in cc_read_json(ENVIRONMENT_INVENTORY)["environments"]
                },
                "capacity_limited": cc_read_json(ENVIRONMENT_INVENTORY)["capacity_limited"],
            },
        },
        "statistical_support": {
            "delta": float(statistics["primary_endpoint_repair_recall"]["delta"]),
            "ci_lower": float(statistics["primary_endpoint_repair_recall"]["ci_lower"]),
            "ci_upper": float(statistics["primary_endpoint_repair_recall"]["ci_upper"]),
            "p_value": float(statistics["primary_endpoint_repair_recall"]["p_value"]),
            "harm_delta": float(statistics["primary_endpoint_realized_harm"]["delta"]),
            "harm_ci_lower": float(statistics["primary_endpoint_realized_harm"]["ci_lower"]),
            "harm_ci_upper": float(statistics["primary_endpoint_realized_harm"]["ci_upper"]),
            "family_size": int(statistics["family_size"]),
            "surviving_after_holm": int(statistics["count_surviving"]),
        },
        "controls_summary": controls["discrimination_checks"],
        "negative_tests": {
            "passed": int(negatives["passed"]),
            "total": int(negatives["total"]),
            "failed": negatives["failed"],
        },
        "ready_for_sgv16": bool(ready),
        "reason": (
            f"{'SUPPORTED' if ready else 'NOT SUPPORTED'} on {met} of {len(criteria)} frozen "
            f"development criteria; outcome {label} -- {summary}."
        ),
        "confirmatory_reserve_consumed": False,
        "alpha": ALPHA,
        "primary_epsilon": PRIMARY_EPSILON,
        "synthetic": False,
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DECISION, payload)
    print(
        f"decide: {payload['verdict']} -- {met}/{len(criteria)} criteria, outcome {label}, "
        f"ready_for_sgv16={ready} -> {_relative(DECISION)}"
    )
    return 0


# ------------------------------------------- section 27: how many documents would be enough
#
# The declared budgets stop at what the certification halves can supply. When a bound refuses at
# every one of them, "not enough documents" is a claim that has to be quantified rather than
# asserted, so this phase resamples DOCUMENTS with replacement from each environment's own
# certification pool out to sizes the pool does not contain, and reports the smallest size at
# which the certificate holds on a majority of resamples.
#
# It is an EXTRAPOLATION and is labelled as one everywhere it appears. It assumes further
# documents drawn from the same environment's own document distribution, it inherits that pool's
# harm rate and its size heterogeneity, and it cannot see whether a larger annotation campaign
# would meet different pages. No criterion reads it.

PROJECTION = OUT / "document_requirement_projection.json"
PROJECTION_SIZES = (25, 50, 100, 200, 400, 800, 1600, 3200)
PROJECTION_RESAMPLES = 200
PROJECTION_MAJORITY = 0.5


def run_projection() -> int:
    """Resample documents beyond the pool and find where each certificate would start to hold."""
    started = time.monotonic()
    geometry = load_geometry()
    per_env: dict[str, Any] = {}
    for name, item in geometry.items():
        harmful, accepted = item.counts(FAMILY_NINE)
        span = accepted.max(axis=0)
        level = ALPHA / len(NINE_DEPTHS)
        generator = np.random.default_rng(_stable_seed("sgv15b-projection", str(SAMPLE_SEED), name))
        record: dict[str, Any] = {}
        for method, tag in (
            ("union_finite", "document_finite"),
            ("cluster_robust_asymptotic", "document_asymptotic"),
        ):
            by_size: dict[str, float] = {}
            smallest: int | None = None
            for size in PROJECTION_SIZES:
                held = 0
                for _ in range(PROJECTION_RESAMPLES):
                    rows = generator.integers(0, item.documents.size, size)
                    feasible = any(
                        cluster_ratio_bound(
                            harmful[rows, index],
                            accepted[rows, index],
                            PRIMARY_EPSILON,
                            delta=level,
                            span=float(span[index]),
                            method=method,
                        ).certifies
                        and accepted[rows, index].sum() > 0
                        for index in range(len(NINE_DEPTHS))
                    )
                    held += int(feasible)
                by_size[str(size)] = held / PROJECTION_RESAMPLES
                if smallest is None and by_size[str(size)] >= PROJECTION_MAJORITY:
                    smallest = size
            record[tag] = {
                "certification_rate_by_document_count": by_size,
                "smallest_document_count_certifying_on_a_majority": smallest,
            }
        record["pool_documents"] = int(item.documents.size)
        record["pool_candidates_per_document_mean"] = float(item.sizes().mean())
        per_env[name] = record

    finite = [
        record["document_finite"]["smallest_document_count_certifying_on_a_majority"]
        for record in per_env.values()
    ]
    asymptotic = [
        record["document_asymptotic"]["smallest_document_count_certifying_on_a_majority"]
        for record in per_env.values()
    ]
    _write_json_once(
        PROJECTION,
        {
            **_envelope("document_requirement_projection"),
            "status": (
                "EXTRAPOLATION. Documents are resampled WITH REPLACEMENT from each environment's "
                "own certification pool out to sizes the pool does not contain. It assumes further "
                "documents from the same distribution, inherits that pool's harm rate and size "
                "heterogeneity, and cannot see whether a larger annotation campaign would meet "
                "different pages. No criterion reads it."
            ),
            "family": FAMILY_NINE,
            "control": CONTROL_UNION,
            "epsilon": PRIMARY_EPSILON,
            "per_test_delta": ALPHA / len(NINE_DEPTHS),
            "sizes": list(PROJECTION_SIZES),
            "resamples": PROJECTION_RESAMPLES,
            "majority": PROJECTION_MAJORITY,
            "per_environment": per_env,
            "summary": {
                "environments": len(per_env),
                "finite_sample_certificate_reached_within_the_largest_size": int(
                    sum(1 for value in finite if value is not None)
                ),
                "asymptotic_comparator_reached_within_the_largest_size": int(
                    sum(1 for value in asymptotic if value is not None)
                ),
                "largest_size_tested": max(PROJECTION_SIZES),
                "median_documents_required_finite": (
                    float(np.median([v for v in finite if v is not None]))
                    if any(value is not None for value in finite)
                    else None
                ),
                "median_documents_required_asymptotic": (
                    float(np.median([v for v in asymptotic if v is not None]))
                    if any(value is not None for value in asymptotic)
                    else None
                ),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"projection: finite-sample certificate reachable on "
        f"{sum(1 for v in finite if v is not None)}/{len(per_env)} environments within "
        f"{max(PROJECTION_SIZES)} documents -> {_relative(PROJECTION)}"
    )
    return 0


# --------------------------------------------------- section 39: determinism, by regeneration
#
# The comparison is produced by the regeneration itself. Nothing here is copied from a log.


def _regenerate(arm: str, budget: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rebuild the selected configuration from scratch: environments, fit, draws, cut, deploy."""
    unit, family, control = next((u, f, c) for name, u, f, c in ARMS if name == arm)
    built, _ = s15._environments()
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for environment in built:
        placed = place(environment)
        run = deployer(placed)
        hashes[f"{environment.name}|documents"] = canonical_hash(
            {"documents": sorted(placed.documents.tolist())}
        )
        for name, taus in placed.thresholds.items():
            hashes[f"{environment.name}|prefix_family|{name}"] = canonical_hash(
                {"depths": list(FAMILIES[name]), "thresholds": [float(t) for t in taus.tolist()]}
            )
        for draw in range(DOC_DRAWS):
            sample = document_sample(placed, budget, _sample_seed(environment.name, budget, draw))
            hashes[f"{environment.name}|sample|{draw}"] = canonical_hash(
                {"documents": sorted(placed.documents[sample].tolist())}
            )
            context = _context(
                placed,
                requested_docs=budget,
                draw=draw,
                clipped=bool(sample.size < budget),
                cell_kind="arm",
            )
            for epsilon in EPSILONS:
                row, _block = evaluate_arm(
                    placed, arm, unit, family, control, sample, epsilon, run, context
                )
                row["cell"] = "|".join((environment.name, str(budget), str(draw), arm))
                rows.append(row)
    return pd.DataFrame(rows), hashes


DETERMINISM_RUNS = 2


def run_determinism() -> int:
    """Regenerate the selected configuration twice and compare everything it touched."""
    started = time.monotonic()
    arm, budget = selected_configuration()
    stored, _design = load_cells()
    stored = stored[(stored["arm"] == arm) & (stored["requested_docs"] == budget)].reset_index(
        drop=True
    )
    runs: list[dict[str, Any]] = []
    signatures: list[dict[str, str]] = []
    for index in range(DETERMINISM_RUNS):
        began = time.monotonic()
        regenerated, hashes = _regenerate(arm, budget)
        signatures.append(hashes)
        report = compare_frames(stored, regenerated, ["cell", "epsilon"])
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
    signature_fields = sorted(signatures[0])
    signature_agrees = all(
        signature[field] == signatures[0][field]
        for signature in signatures
        for field in signature_fields
    )
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "what_is_regenerated": (
                "the SELECTED arm at its selected document budget: every environment rebuilt from "
                "scratch, the frozen adaptation arm refitted, both prefix families replaced on the "
                "unlabelled certification pool, the document samples redrawn under their recorded "
                "seeds, the certificate recomputed and the deployment re-evaluated. Nothing is "
                "read from the stored cell table except for the comparison itself."
            ),
            "selected_arm": arm,
            "selected_budget": budget,
            "runs_completed": len(runs),
            "runs": runs,
            "all_runs_identical": all(record["identical"] for record in runs),
            "max_absolute_numeric_difference_over_all_runs": max(
                record["max_absolute_numeric_difference"] for record in runs
            ),
            "structural_signatures_compared": len(signature_fields),
            "structural_signatures_agree": bool(signature_agrees),
            "structural_signature_fields": [
                "the certification-pool document list per environment",
                "both declared prefix families and their placed thresholds per environment",
                "the sampled document set for every draw",
            ],
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
        f"{signature_agrees} -> {_relative(DETERMINISM)}"
    )
    return 0


# ------------------------------------------------------------------ provenance and the cli

REPORT = REPO / "docs/sgv15b/document_level_certification.md"
DESIGN_NOTE = REPO / "docs/sgv15b/risk_bound_design.md"

PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    ENVIRONMENT_INVENTORY,
    DOCUMENT_PARTITION,
    DOCUMENT_CAPACITY,
    DESIGN_RECORD,
    PREFIX_FAMILY,
    SIMULATION_COVERAGE,
    PREFIX_CONTROL_VALIDATION,
    SGV15_REPRODUCTION,
    FACTORIAL_CELLS,
    DOCUMENT_COUNTS,
    GEOMETRY,
    DOCUMENT_DUPLICATION_TEST,
    CLUSTER_STRUCTURE,
    DESIGN_EFFECTS,
    EFFECTIVE_SAMPLE_SIZE,
    ORACLE_RESULTS,
    GRID_RESOLUTION_ANALYSIS,
    FACTORIAL_RESULTS,
    CANDIDATE_NINE_RESULTS,
    DOCUMENT_NINE_RESULTS,
    CANDIDATE_FINE_RESULTS,
    DOCUMENT_FINE_RESULTS,
    SAMPLE_COMPLEXITY_DOCUMENTS,
    SAMPLE_COMPLEXITY_CANDIDATES,
    ANNOTATION_COST,
    PROJECTION,
    CONTROL_RESULTS,
    NEGATIVE_TESTS,
    NESTED_SELECTION,
    STATISTICAL_TESTS,
    DECISION,
    DETERMINISM,
    FIGURE_MANIFEST,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE, s15.DECISION, s15.RISK_BOUND_RESULTS, SGV15_GRID_RESOLUTION),
    "2. Frozen SGV15 finding": (RESEARCH_FREEZE, s15.DECISION),
    "3. Research question": (DESIGN_RECORD,),
    "4. Non-goals": (DESIGN_RECORD, FROZEN_CONFIGURATION),
    "5. Operational harm estimand": (DESIGN_RECORD,),
    "6. Why candidate independence fails": (
        DESIGN_EFFECTS,
        EFFECTIVE_SAMPLE_SIZE,
        SIMULATION_COVERAGE,
        s15.RISK_BOUND_RESULTS,
    ),
    "7. Document-level statistical design": (
        DOCUMENT_PARTITION,
        DOCUMENT_CAPACITY,
        ENVIRONMENT_INVENTORY,
    ),
    "8. Risk-bound derivation": (DESIGN_RECORD, PREFIX_FAMILY),
    "9. Prefix-control design": (PREFIX_FAMILY, DESIGN_RECORD),
    "10. Simulation validation": (SIMULATION_COVERAGE, PREFIX_CONTROL_VALIDATION),
    "11. SGV15 reproduction": (SGV15_REPRODUCTION,),
    "12. The 2\u00d72 factorial": (FACTORIAL_RESULTS, SAMPLE_COMPLEXITY_DOCUMENTS),
    "13. Candidate-level and the frozen family": (
        CANDIDATE_NINE_RESULTS,
        FACTORIAL_RESULTS,
        ANNOTATION_COST,
    ),
    "14. Document-level and the frozen family": (DOCUMENT_NINE_RESULTS, FACTORIAL_RESULTS),
    "15. Candidate-level and the fine family": (
        CANDIDATE_FINE_RESULTS,
        SAMPLE_COMPLEXITY_DOCUMENTS,
    ),
    "16. Document-level and the fine family": (DOCUMENT_FINE_RESULTS,),
    "17. Safety": (DECISION, FACTORIAL_RESULTS),
    "18. False certifications": (DECISION, SIMULATION_COVERAGE, FACTORIAL_RESULTS),
    "19. Coverage and non-degeneracy": (DECISION, FACTORIAL_RESULTS, ORACLE_RESULTS),
    "20. Grid resolution": (GRID_RESOLUTION_ANALYSIS, ORACLE_RESULTS, DECISION),
    "21. Clustering effects": (
        DESIGN_EFFECTS,
        CLUSTER_STRUCTURE,
        EFFECTIVE_SAMPLE_SIZE,
        CONTROL_RESULTS,
    ),
    "22. Document-duplication falsification": (DOCUMENT_DUPLICATION_TEST,),
    "23. Document sample complexity": (
        SAMPLE_COMPLEXITY_DOCUMENTS,
        PROJECTION,
        ENVIRONMENT_INVENTORY,
    ),
    "24. Candidate annotation cost": (
        ANNOTATION_COST,
        SAMPLE_COMPLEXITY_CANDIDATES,
        ENVIRONMENT_INVENTORY,
        PROJECTION,
    ),
    "25. Oracle decomposition": (ORACLE_RESULTS, CONTROL_RESULTS, CANDIDATE_NINE_RESULTS),
    "26. Controls": (CONTROL_RESULTS,),
    "27. Negative tests": (NEGATIVE_TESTS,),
    "28. Nested selection": (NESTED_SELECTION,),
    "29. Statistical tests": (STATISTICAL_TESTS,),
    "30. Limitations": (
        DECISION,
        ORACLE_RESULTS,
        EFFECTIVE_SAMPLE_SIZE,
        PREFIX_CONTROL_VALIDATION,
        PROJECTION,
        ENVIRONMENT_INVENTORY,
        SIMULATION_COVERAGE,
        SAMPLE_COMPLEXITY_DOCUMENTS,
    ),
    "31. Research decision": (
        DECISION,
        DETERMINISM,
        GRID_RESOLUTION_ANALYSIS,
        PROJECTION,
        SAMPLE_COMPLEXITY_DOCUMENTS,
        SIMULATION_COVERAGE,
        DOCUMENT_DUPLICATION_TEST,
    ),
    "32. SGV16 readiness": (DECISION, PROJECTION, NESTED_SELECTION),
    "Amendments and defects": (
        SGV15_REPRODUCTION,
        DETERMINISM,
        PREFIX_CONTROL_VALIDATION,
        CONTROL_RESULTS,
        DESIGN_RECORD,
    ),
    "Artifacts": (PROVENANCE, TRACEABILITY, FIGURE_MANIFEST),
}


def run_record() -> int:
    """Provenance, the dependency audit and the traceability index."""
    started = time.monotonic()
    missing = [path for path in PRODUCED if not path.exists()]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {[_relative(p) for p in missing][:5]}")
    freeze = cc_read_json(RESEARCH_FREEZE)
    inputs = dict(cc_read_json(s15.PROVENANCE)["artifacts"])
    # Re-hashed here rather than copied from the section-0 freeze: the freeze records the state
    # the stage STARTED from, and provenance must record the code that produced these artifacts.
    inputs.update({path: file_sha256(REPO / path) for path in sorted(UPSTREAM_SCRIPTS)})
    cells = pd.read_parquet(FACTORIAL_CELLS)
    tracked = [path for path in PRODUCED if _tracked(path)]
    unignored = [path for path in PRODUCED if not _ignored(path)]
    payload = {
        **_envelope("provenance"),
        "issued_head": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "artifacts": {_relative(path): file_sha256(path) for path in PRODUCED},
        "artifact_count": len(PRODUCED),
        "upstream_inputs": inputs,
        "upstream_input_count": len(inputs),
        "documents": {
            _relative(REPORT): file_sha256(REPORT) if REPORT.is_file() else None,
            _relative(DESIGN_NOTE): file_sha256(DESIGN_NOTE),
        },
        "upstream_script_sha256_at_section_zero": freeze["upstream_script_sha256"],
        "table_rows": {
            _relative(FACTORIAL_CELLS): len(cells),
            _relative(DOCUMENT_COUNTS): len(pd.read_parquet(DOCUMENT_COUNTS)),
        },
        "aggregate": {
            "environments": int(cells["environment"].nunique()),
            "arms": int(cells["arm"].nunique()),
            "document_budgets": len(DOC_BUDGETS),
            "draws": DOC_DRAWS,
            "epsilons": len(EPSILONS),
            "declared_prefixes": len(NINE_DEPTHS) + len(FINE_DEPTHS),
            "certification_documents": int(
                cc_read_json(ENVIRONMENT_INVENTORY)["totals"]["certification_documents"]
            ),
            "certification_candidates": int(
                cc_read_json(ENVIRONMENT_INVENTORY)["totals"]["certification_candidates"]
            ),
        },
        "dependency_audit": {
            "artifacts_tracked_by_git": [_relative(path) for path in tracked],
            "artifacts_not_git_ignored": [_relative(path) for path in unignored],
            "raw_data_written": False,
            "sgv13_sgv14_sgv15_artifacts_written": False,
            "confirmatory_reserve_consumed": False,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(PROVENANCE, payload)
    _write_json_once(
        TRACEABILITY,
        {
            **_envelope("traceability"),
            "report": _relative(REPORT),
            "design_note": _relative(DESIGN_NOTE),
            "rule": (
                "every number in the report is read from one of the artifacts its section names. "
                "A number that cannot be traced to one does not belong in the report."
            ),
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


# ------------------------------------------------------------------ section 36: the figures

ARM_COLOUR = {
    M00: "#8c8c8c",
    M10: "#1f77b4",
    M01: "#ff7f0e",
    M11: "#2ca02c",
    P1A: "#d62728",
    P1B: "#9467bd",
    D0: "#17becf",
    D1: "#bcbd22",
    N0: "#7f0000",
    O0_UNRESTRICTED: "#000000",
    O1_NINE: "#555555",
    O2_FINE: "#aaaaaa",
}
ARM_LABEL = {
    M00: "M00 candidate x nine (SGV15)",
    M10: "M10 document x nine",
    M01: "M01 candidate x fine, gatekeeping",
    M11: "M11 document x fine, gatekeeping",
    P1A: "P1a candidate x fine, union",
    P1B: "P1b document x fine, union",
    D0: "D0 asymptotic x nine (diagnostic)",
    D1: "D1 asymptotic x fine (diagnostic)",
    N0: "N0 naive empirical (negative)",
    O0_UNRESTRICTED: "O0 unrestricted oracle",
    O1_NINE: "O1 nine-depth oracle",
    O2_FINE: "O2 fine-family oracle",
}


def _figure_style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _finish(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    figure.clf()
    return path


def run_figures() -> int:
    """The thirteen required figures. Every plotted number comes out of a saved artifact."""
    started = time.monotonic()
    plt = _figure_style()
    effects = cc_read_json(DESIGN_EFFECTS)
    effective = cc_read_json(EFFECTIVE_SAMPLE_SIZE)
    simulation = cc_read_json(SIMULATION_COVERAGE)
    prefixes = cc_read_json(PREFIX_CONTROL_VALIDATION)
    oracle = cc_read_json(ORACLE_RESULTS)
    complexity = cc_read_json(SAMPLE_COMPLEXITY_DOCUMENTS)
    cost = cc_read_json(ANNOTATION_COST)["by_arm"]
    duplication = cc_read_json(DOCUMENT_DUPLICATION_TEST)
    per_arm = {
        M00: cc_read_json(CANDIDATE_NINE_RESULTS),
        M10: cc_read_json(DOCUMENT_NINE_RESULTS),
        M01: cc_read_json(CANDIDATE_FINE_RESULTS),
        M11: cc_read_json(DOCUMENT_FINE_RESULTS),
    }
    environments = sorted(effects["per_environment"])
    # Both corpora contribute a `doctr` environment, so the corpus prefix has to survive the
    # abbreviation or two different environments share a tick label and a legend entry.
    short = [f"{name.split('/')[0][:2]}:{name.split('/')[-1]}" for name in environments]
    written: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    values = [effects["per_environment"][name]["mean_design_effect"] for name in environments]
    peaks = [effects["per_environment"][name]["max_design_effect"] for name in environments]
    axis.bar(short, values, color="#1f77b4", label="mean over the nine declared depths")
    axis.plot(short, peaks, "k.", label="worst declared depth")
    axis.axhline(1.0, color="#d62728", linestyle="--", lw=1, label="no clustering")
    axis.set_ylabel("design effect")
    axis.set_title("Document clustering of the accepted-set harm rate, certification pool")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(fontsize=6)
    written.append(_finish(figure, FIGURE_DIR / "design_effect_by_environment.png"))

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for name in environments:
        record = effective["per_environment"][name]["by_depth"]
        axis.plot(
            [record[key]["candidate_level_clopper_pearson_upper"] for key in record],
            [record[key]["document_aware_finite_upper"] for key in record],
            "o",
            ms=3,
            alpha=0.7,
            label=f"{name.split('/')[0][:2]}:{name.split('/')[-1]}",
        )
    limits = [0.0, 1.02]
    axis.plot(limits, limits, "k--", lw=1, label="equal")
    axis.set_xlabel("candidate-level Clopper-Pearson upper limit")
    axis.set_ylabel("document-aware finite-sample upper limit")
    axis.set_title(f"The same accepted set, two inference units ({PRACTICAL_DOC_BUDGET} documents)")
    axis.legend(fontsize=5, ncol=2)
    written.append(_finish(figure, FIGURE_DIR / "candidate_vs_document_bound.png"))

    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for method, colour in (
        ("candidate_clopper_pearson", "#d62728"),
        ("union_finite", "#1f77b4"),
        ("cluster_robust_asymptotic", "#17becf"),
    ):
        points = [
            (
                record["population_design_effect"],
                record["false_certification_rate"][str(max(SIMULATION_DOCUMENTS))][method],
            )
            for record in simulation["regimes"].values()
        ]
        points.sort()
        axis.plot([p[0] for p in points], [p[1] for p in points], "o-", color=colour, label=method)
    axis.axhline(ALPHA, color="k", linestyle="--", lw=1, label=f"nominal alpha = {ALPHA}")
    axis.set_xlabel("population design effect")
    axis.set_ylabel("false-certification rate")
    axis.set_title(
        f"Simulated clustering vs over-certification ({max(SIMULATION_DOCUMENTS)} documents)"
    )
    axis.legend(fontsize=6)
    written.append(_finish(figure, FIGURE_DIR / "false_certification_vs_clustering.png"))

    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    width = 0.38
    positions = np.arange(len(environments))
    axis.bar(
        positions - width / 2,
        [oracle["per_environment"][name]["grid_loss_nine"] for name in environments],
        width,
        color="#8c8c8c",
        label=f"nine declared depths (loss {oracle['mean_grid_loss_nine']:.4f})",
    )
    axis.bar(
        positions + width / 2,
        [oracle["per_environment"][name]["grid_loss_fine"] for name in environments],
        width,
        color="#2ca02c",
        label=f"{len(FINE_DEPTHS)} declared depths (loss {oracle['mean_grid_loss_fine']:.4f})",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(short, rotation=45)
    axis.set_ylabel("oracle repair recall lost to the declared family")
    axis.set_title("What each declared family costs against a dense frontier")
    axis.legend(fontsize=6)
    written.append(_finish(figure, FIGURE_DIR / "grid_resolution_comparison.png"))

    figure, axis = plt.subplots(figsize=(6.6, 3.4))
    for key in sorted(next(iter(prefixes["by_documents"].values()))):
        counts = sorted(int(d) for d in prefixes["by_documents"])
        axis.plot(
            counts,
            [1.0 - prefixes["by_documents"][str(d)][key]["refusal_rate"] for d in counts],
            "o-",
            label=key,
        )
    axis.set_xscale("log")
    axis.set_xlabel("certification documents (simulated)")
    axis.set_ylabel("fraction of trials that deployed")
    axis.set_title("Prefix control: what each rule can actually certify")
    axis.legend(fontsize=6)
    written.append(_finish(figure, FIGURE_DIR / "prefix_control_comparison.png"))

    for field, filename, label in (
        (
            "mean_risk_controlled",
            "factorial_2x2_repair_recall.png",
            "risk-controlled repair recall",
        ),
        ("mean_realized_harm_rate", "factorial_2x2_harm.png", "realised harm rate"),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 3.4))
        for arm in FACTORIAL:
            axis.plot(
                DOC_BUDGETS,
                [per_arm[arm]["by_budget"][str(b)][field] for b in DOC_BUDGETS],
                "o-",
                color=ARM_COLOUR[arm],
                label=ARM_LABEL[arm],
            )
        if field == "mean_realized_harm_rate":
            axis.axhline(
                PRIMARY_EPSILON,
                color="k",
                linestyle="--",
                lw=1,
                label=f"epsilon = {PRIMARY_EPSILON}",
            )
        axis.set_xlabel("certification documents")
        axis.set_ylabel(label)
        axis.set_title(f"The 2x2, {label}")
        axis.legend(fontsize=6)
        written.append(_finish(figure, FIGURE_DIR / filename))

    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    reference = str(PRACTICAL_DOC_BUDGET)
    for arm in FACTORIAL:
        records = per_arm[arm]["by_budget"][reference]["per_environment"]
        axis.plot(
            [records[name]["coverage"] if name in records else np.nan for name in environments],
            [
                records[name]["realized_harm_rate"] if name in records else np.nan
                for name in environments
            ],
            "o",
            color=ARM_COLOUR[arm],
            label=ARM_LABEL[arm],
        )
    axis.axhline(
        PRIMARY_EPSILON, color="k", linestyle="--", lw=1, label=f"epsilon = {PRIMARY_EPSILON}"
    )
    axis.set_xlabel("accepted coverage")
    axis.set_ylabel("realised harm rate")
    axis.set_title(f"Risk against coverage, {PRACTICAL_DOC_BUDGET} certification documents")
    axis.legend(fontsize=6)
    written.append(_finish(figure, FIGURE_DIR / "risk_coverage_by_environment.png"))

    for axis_field, filename, xlabel in (
        ("documents", "safe_repair_recall_vs_documents.png", "certification documents"),
        ("candidates", "safe_repair_recall_vs_candidate_labels.png", "candidate labels revealed"),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 3.4))
        for arm in FACTORIAL:
            xs = [
                cost[arm][str(b)]["documents_labelled"]
                if axis_field == "documents"
                else cost[arm][str(b)]["candidates_labelled"]
                for b in DOC_BUDGETS
            ]
            axis.plot(
                xs,
                [cost[arm][str(b)]["mean_risk_controlled"] for b in DOC_BUDGETS],
                "o-",
                color=ARM_COLOUR[arm],
                label=ARM_LABEL[arm],
            )
        if axis_field == "candidates":
            axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("mean risk-controlled repair recall")
        axis.set_title(f"Annotation cost: repair recall against {xlabel}")
        axis.legend(fontsize=6)
        written.append(_finish(figure, FIGURE_DIR / filename))

    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for arm in FACTORIAL:
        axis.plot(
            DOC_BUDGETS,
            [
                complexity["by_arm"][arm][str(b)]["probability_safe_non_degenerate_deployment"]
                for b in DOC_BUDGETS
            ],
            "o-",
            color=ARM_COLOUR[arm],
            label=ARM_LABEL[arm],
        )
    axis.set_xlabel("certification documents")
    axis.set_ylabel("P(safe, non-degenerate deployment)")
    axis.set_title("Certification feasibility against the document budget")
    axis.legend(fontsize=6)
    written.append(_finish(figure, FIGURE_DIR / "certification_feasibility_vs_documents.png"))

    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    # The two limits live on different scales -- the candidate one bounds a rate, the document one
    # bounds E[Z] in accepted-edit units -- so each is drawn relative to its own single-copy value.
    # The claim is about relative change, and plotting the raw values would flatten the shrinkage
    # that is the entire point of the comparison.
    for arm in (M00, M10):
        for index, name in enumerate(environments):
            record = duplication["per_environment"][name][arm]["by_multiplier"]
            base = record["1"]["bounds_at_every_declared_depth"]
            scaled = [
                float(
                    np.mean(
                        [
                            record[str(m)]["bounds_at_every_declared_depth"][depth]
                            / (base[depth] * (m if arm == M10 else 1))
                            for depth in range(len(NINE_DEPTHS))
                            if base[depth] > 0.0
                        ]
                    )
                )
                for m in DUPLICATION_MULTIPLIERS
            ]
            axis.plot(
                DUPLICATION_MULTIPLIERS,
                scaled,
                "o-",
                color=ARM_COLOUR[arm],
                alpha=0.6,
                label=ARM_LABEL[arm] if index == 0 else None,
            )
    axis.axhline(1.0, color="k", linestyle="--", lw=1, label="unchanged by duplication")
    axis.set_xlabel("copies of every candidate inside every document")
    axis.set_ylabel("limit relative to its single-copy value")
    axis.set_title("Duplication adds no independent document. Does the bound know?")
    axis.legend(fontsize=6)
    written.append(_finish(figure, FIGURE_DIR / "document_duplication_invariance.png"))

    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    for oracle_name in ORACLES:
        axis.plot(
            short,
            [
                oracle["per_environment"][name][oracle_name]["repair_recall"]
                for name in environments
            ],
            "o-",
            color=ARM_COLOUR[oracle_name],
            label=ARM_LABEL[oracle_name],
        )
    axis.set_ylabel("oracle repair recall")
    axis.set_title("ANALYSIS ONLY: what each declared family could have reached")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(fontsize=6)
    written.append(_finish(figure, FIGURE_DIR / "oracle_grid_loss.png"))

    sources = (
        DESIGN_EFFECTS,
        EFFECTIVE_SAMPLE_SIZE,
        SIMULATION_COVERAGE,
        PREFIX_CONTROL_VALIDATION,
        ORACLE_RESULTS,
        GRID_RESOLUTION_ANALYSIS,
        FACTORIAL_RESULTS,
        CANDIDATE_NINE_RESULTS,
        DOCUMENT_NINE_RESULTS,
        CANDIDATE_FINE_RESULTS,
        DOCUMENT_FINE_RESULTS,
        SAMPLE_COMPLEXITY_DOCUMENTS,
        ANNOTATION_COST,
        DOCUMENT_DUPLICATION_TEST,
    )
    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_envelope("figure_manifest"),
            "synthetic": False,
            "scientific_status": "DEVELOPMENT -- SGV14-exposed environments",
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


PHASES: tuple[tuple[str, Callable[[], int]], ...] = (
    ("reconstruct", run_reconstruct),
    ("freeze", run_freeze),
    ("capacity", run_capacity),
    ("preregister", run_preregister),
    ("simulate", run_simulate),
    ("reproduce", run_reproduce),
    ("factorial", run_factorial),
    ("duplicate", run_duplicate),
    ("cluster", run_cluster),
    ("oracle", run_oracle),
    ("results", run_results),
    ("complexity", run_complexity),
    ("projection", run_projection),
    ("controls", run_controls),
    ("negative", run_negative),
    ("select", run_select),
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
        if name in ("reproduce", "factorial") and arguments.only:
            phase(arguments.only)  # type: ignore[call-arg]
        else:
            phase()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
