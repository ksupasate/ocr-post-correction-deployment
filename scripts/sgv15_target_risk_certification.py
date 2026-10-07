#!/usr/bin/env python3
"""SGV15: the ranking transferred, the threshold did not. How much target evidence certifies a cut?

SGV14 ran SGV13's frozen procedure on ten OCR environments it had never seen and returned NOT
SUPPORTED on 1 of 4 criteria. It also located the failure to a single component. The target
labels worked: AUROC rose on nine of ten environments, the achievable risk-controlled frontier
rose on eight, accepted-set harm fell on all ten, a source-only refit reproduced the frozen
baseline exactly and permuting the purchased labels destroyed the gain. What failed was the
deployment cut. `frozen_source` is `select_threshold` on CORD receipt calibration rows, and
transported unchanged to scanned forms and to Fraktur it admitted prefixes whose realised harm
ran from 0.1018 to 0.5731 against a bound of 0.10.

    SGV15-P1: given a ranking already improved by limited target supervision, a small,
    representative, independently held target sample can certify how far into that ranking a
    deployment may safely cut -- and doing so converts SGV14's measured ranking gains into
    realised risk-controlled repair.

**The hypothesis is recorded as SGV15-P1.** `docs/sgv1/protocol.md` binds SGV1-H1..H4 and says a
frozen identifier is never reused for a different claim; every stage since SGV2 has opened its
own family. The brief's RQ1..RQ5 are carried as sub-claims P1a (a certified target cut beats the
frozen source cut), P1b (the label cost of certification is measurable), P1c (adaptation and
certification labels should be separated), P1d (representative sampling beats targeted sampling
for risk estimation) and P1e (a conservative finite-sample bound prevents overshoot without
collapsing coverage), all under P1.

**This is a DEVELOPMENT stage, not a confirmatory one.** SGV14 already exposed all ten
environments and this stage reads their outcomes. Nothing here may be called confirmation, however
well it performs. The intended sequence is: SGV15 develops and freezes a certification rule;
SGV16 evaluates that frozen rule on environments nothing has touched. The CORD confirmatory
reserve stays LOCKED throughout -- a one-way door is not spent on method development.

Ten decisions fix what the numbers below can mean.

**1. The adaptation model is not the variable.** Every arm reuses SGV13's `a4_joint_refit`
through SGV13's own function, on SGV14's feature universe, model class, source rows, weighting
and regularisation. No stronger model is built to rescue this stage. The variable is how target
labels are SAMPLED and USED to place a cut.

**2. Adaptation labels and certification labels are different objects.** SGV14's adaptation
partition is split by DOCUMENT into a fitting half and a certification half. Adaptation labels
are bought from the fitting half; certification labels are drawn from the certification half. A
certification label never reaches a fit and an adaptation label never reaches the primary risk
estimate. The split is content-blind and whole volumes move together.

**3. Nothing here reimplements risk control.** The cut is placed by `risk.select_threshold` --
the repository's single implementation, which already splits the confidence level across the
threshold grid by union bound. SGV15 adds two things to the library rather than to this script: an
exact Clopper-Pearson upper bound, because the harm indicator is a Bernoulli draw and Bentkus'
factor of e is a price paid for generality this problem does not need; and a `thresholds`
parameter, so a caller can declare its cut family instead of searching every row.

**4. The cut grid is declared, and the multiplicity is over the declared grid.** Nine acceptance
depths, fixed before any endpoint is read, evaluated at the corresponding quantiles of the
UNLABELLED certification-partition scores. The labelled sample is then spent purely on estimating
risk, never on choosing where to look.

**5. Representative beats targeted, and the stage assumes neither.** The primary certification
sampler is document-stratified random. Uncertainty-acquired certification is carried as an
explicitly diagnostic NEGATIVE baseline, because SGV13 and SGV14 both showed targeted acquisition
distorts label prevalence -- a budget drawn at the deployed threshold was 47% to 85% harmful.

**6. Two experiments, and they answer different questions.** Experiment A holds the adaptation
budget at SGV13's 250 and adds independent certification budgets, which isolates the mechanism at
the cost of spending more labels. Experiment B holds the TOTAL at 250 and splits it, which is the
label-efficiency question. Criticising A for label cost is a category error and the report says so.

**7. The oracle cut exists and may never select anything.** C1 is the deepest cut whose realised
harm holds on the evaluation block. It bounds what any certification procedure could achieve and
it is analysis only; the leakage suite asserts it reaches no selection path.

**8. Selection is nested, and the outer number is not blinded.** Method, grid and allocation are
chosen by leave-one-environment-out inside the development set. That is the honest procedure
available once outcomes have been seen, and the report states plainly that the resulting outer
number is developmental.

**9. Candidates are not IID and the bound knows it is not.** Several candidates come from one
page. Every deployed endpoint is compared under a document-clustered bootstrap as well as the
candidate-level binomial bound, and where the IID bound is anti-conservative that is reported.

**10. A failure is reported, not patched.** If representative target evidence cannot certify a
safe cut at these budgets, this stage says so and quantifies why, rather than adding another
downstream threshold heuristic.

    --reconstruct  the section-0 freeze, including SGV14's verification state
    --freeze       the frozen SGV14 configuration this stage may not vary
    --partition    the fitting / certification split, and its capacity
    --preregister  the methods, samplers, grid, budgets, criteria and selection plan
    --experiment-a certification budgets at a fixed adaptation budget
    --experiment-b a fixed total budget, split between adaptation and certification
    --sampling     the certification sampler comparison
    --decompose    the cut decomposition and the adaptation-by-cut table
    --samplesize   what a certification sample of each size can actually certify
    --controls     the twelve required controls
    --negative     the fifteen falsification tests
    --stats        the pre-registered tests
    --select       nested leave-environment-out method selection
    --figures      the seven required figures
    --decide       the machine-readable finding
    --record       provenance, the dependency audit and the traceability index

DEVELOPMENT ONLY. The SGV1 CORD confirmatory reserve stays LOCKED and is absent from every
artifact. SGV14's artifacts are read and hash-verified; none is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_verifier_pilot as pilot
import sgv6_target_engine_adaptation as s6
import sgv9_target_calibration as s9
import sgv11_minimax_policy_transfer as s11
import sgv13_fewshot_prefix_purity_adaptation as s13
import sgv14_confirmatory_validation as s14
from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.risk import clopper_pearson_upper, risk_upper_bound, select_threshold

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv15_target_risk_certification"

CERTIFICATION_ROWS = OUT / "certification_rows.parquet"
CELL_TABLE = OUT / "certification_cells.parquet"
DEPLOYMENT_COUNTS = OUT / "deployment_counts.parquet"
RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_sgv14_configuration.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"
DESIGN_RECORD = OUT / "design_record.json"
ADAPTATION_BUDGET_RESULTS = OUT / "adaptation_budget_results.json"
CERTIFICATION_BUDGET_RESULTS = OUT / "certification_budget_results.json"
BUDGET_ALLOCATION_RESULTS = OUT / "budget_allocation_results.json"
CERTIFICATION_SAMPLING = OUT / "certification_sampling.json"
RISK_BOUND_RESULTS = OUT / "risk_bound_results.json"
CUT_RESULTS = OUT / "cut_results.json"
CUT_DECOMPOSITION = OUT / "cut_decomposition.json"
ADAPTATION_CERTIFICATION_DECOMPOSITION = OUT / "adaptation_certification_decomposition.json"
SAMPLE_SIZE_ANALYSIS = OUT / "sample_size_analysis.json"
CONTROL_RESULTS = OUT / "control_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
METHOD_SELECTION = OUT / "method_selection.json"
DECISION = OUT / "research_decision.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

SCHEMA_VERSION = 1
STAGE = "sgv15_target_risk_certification"
HYPOTHESIS = "SGV15-P1"

# ------------------------------------------------------------------ imported, never restated

deployed = s13.deployed
frontier = s13.frontier
ranking_quality = s13.ranking_quality
epsilon_key = s13.epsilon_key
per_document_counts = s13.per_document_counts
document_multiplicities = s11.document_multiplicities
_interval = s9._interval
_stable_seed = s6._stable_seed
build_setup = s13.build_setup
acquisition_order = s13.acquisition_order
purchase = s13.purchase
fit_joint_arm = s13.fit_joint_arm
frozen_arm = s13.frozen_arm
cc_read_json = s13.cc_read_json
combine = s14.combine
reference_state = s14.reference_state
environment_setup = s14.environment_setup
group_of = s14.group_of
count_block = s14.count_block
stratified_delta = s14.stratified_delta
holm = s14.holm

EPSILONS = s13.EPSILONS
PRIMARY_EPSILON = s13.PRIMARY_EPSILON
PRIMARY_KEY = s13.PRIMARY_KEY
BOOTSTRAP_RESAMPLES = s13.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = s13.BOOTSTRAP_SEED
RECALL = s13.RECALL
HARM = s13.HARM

FIT = s13.FIT
CALIBRATION = s13.CALIBRATION
EVALUATION = s13.EVALUATION
POOL = s13.POOL


class PhaseError(RuntimeError):
    """A guard this stage refuses to run past."""


# ------------------------------------------------------------------ the pre-registered registry

# The fitting / certification split of SGV14's adaptation half. Content-blind: the key is a hash
# of the document -- or, on OCR-D-SBB, of the volume -- under a fixed seed.
PARTITION_SEED = "sgv15-certification-partition-v1:2026-09-07"
FIT_FRACTION = 0.5
ROLE_FIT = "SGV15_FIT"
ROLE_CERT = "SGV15_CERT"

SAMPLE_SEED = 20260907

# Section 8's two experiments.
ADAPT_BUDGET = 250
CERT_BUDGETS = (0, 25, 50, 100, 150, 250, 500)
TOTAL_BUDGET = 250
ALLOCATIONS = ((250, 0), (225, 25), (200, 50), (175, 75), (150, 100), (125, 125))

# Section 15's declared cut family: acceptance depths, evaluated at the corresponding quantiles
# of the UNLABELLED certification-partition scores. Nine points, so the union bound costs a
# factor of nine rather than a factor of two hundred.
CUT_GRID = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
FINE_GRID_POINTS = 200
CERT_DELTA = 0.05

# Section 13's methods.
M0_FROZEN_SOURCE = "m0_frozen_source_cut"
M1_EMPIRICAL = "m1_naive_empirical_target_cut"
M2_CLOPPER = "m2_certified_clopper_pearson"
M2B_BENTKUS = "m2b_certified_bentkus"
M3_SHRINKAGE = "m3_beta_binomial_shrinkage"
M4_CROSS_FITTED = "m4_cross_fitted_certification"
C1_ORACLE = "c1_oracle_evaluation_cut"
METHODS = (M0_FROZEN_SOURCE, M1_EMPIRICAL, M2_CLOPPER, M2B_BENTKUS, M3_SHRINKAGE)
ANALYSIS_ONLY_METHODS = (C1_ORACLE,)
CONTROLLER_OF = {
    M1_EMPIRICAL: "empirical",
    M2_CLOPPER: "ltt_clopper_pearson",
    M2B_BENTKUS: "ltt_bentkus",
    M3_SHRINKAGE: "ltt_beta_binomial",
}

# Section 10's samplers.
CS0_UNCERTAINTY = "cs0_uncertainty_biased"
CS1_UNIFORM = "cs1_uniform_random"
CS2_DOCUMENT = "cs2_document_stratified"
CS_ONE_DOCUMENT = "cx_one_document"
CS_FEW_DOCUMENTS = "cx_few_documents"
SAMPLERS = (CS0_UNCERTAINTY, CS1_UNIFORM, CS2_DOCUMENT)
CONTROL_SAMPLERS = (CS_ONE_DOCUMENT, CS_FEW_DOCUMENTS)
# Measured everywhere, selectable nowhere. `cs0_uncertainty_biased` is registered in the design
# record as a DIAGNOSTIC NEGATIVE and section 10 of the stage brief forbids it as the primary
# certification sample, so it is excluded from the selection pool for exactly the reason the
# oracle is. It still appears in every comparison table.
SELECTABLE_SAMPLERS = (CS1_UNIFORM, CS2_DOCUMENT)
NOT_SELECTABLE_SAMPLERS = (CS0_UNCERTAINTY, *CONTROL_SAMPLERS)
FEW_DOCUMENTS = 3
PRIMARY_SAMPLER = CS2_DOCUMENT

# Adaptation acquisition: SGV14's frozen rule, plus the document-random comparison section 11
# asks for as an explicitly secondary development question.
A_UNCERTAINTY = s13.Q_UNCERTAINTY
A_DOCUMENT_RANDOM = s13.Q_RANDOM_DOCUMENT
ACQUISITIONS = (A_UNCERTAINTY, A_DOCUMENT_RANDOM)

CERT_DRAWS = 20
ADAPT_DRAWS = 5
CROSS_FIT_FOLDS = 5

# Section 29's criteria. Fixed here, before any SGV15 endpoint exists.
BREADTH_REQUIRED = 7
ENVIRONMENT_COUNT = 10
SHRINKAGE_PRIOR_WEIGHT = 20.0


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PhaseError(f"{_relative(path)} already exists; delete it deliberately to rewrite")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PhaseError(f"{_relative(path)} already exists; delete it deliberately to rewrite")
    frame.to_parquet(path, index=False)


def _ignored(path: Path) -> bool:
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=REPO, capture_output=True, check=False
        ).returncode
        == 0
    )


def _tracked(path: Path) -> bool:
    return bool(_git("ls-files", "--error-unmatch", str(path)))


# ------------------------------------------------------------- section 0: the research freeze
#
# SGV15 may not start until SGV14's verification is closed, and "closed" is checked rather than
# assumed: the decision artifact must exist, its verdict must be the one SGV14 published, and the
# artifacts it names must still hash to what its own provenance recorded.

SGV14_VERDICT = "NOT SUPPORTED"
SGV14_CRITERIA_MET = 1
SGV14_CRITERIA_TOTAL = 4

UPSTREAM_SCRIPTS = ("scripts/sgv15_target_risk_certification.py", *s14.DEPENDENCY_SCRIPTS)


def verify_sgv14() -> dict[str, Any]:
    """SGV14's artifacts, checked against the hashes SGV14's own provenance recorded.

    A stage that motivates itself by another stage's failure has to be sure it is reading that
    stage's actual numbers. Every path SGV14's provenance names is re-hashed here, and a
    difference is a hard stop rather than a warning.
    """
    if not s14.PROVENANCE.is_file():
        raise PhaseError("SGV14's provenance.json is missing; SGV14 is not closed")
    record = cc_read_json(s14.PROVENANCE)
    decision = cc_read_json(s14.DECISION)
    if decision["verdict"] != SGV14_VERDICT:
        raise PhaseError(
            f"SGV14's verdict is {decision['verdict']!r}, not {SGV14_VERDICT!r}; SGV15's "
            "motivation is a specific published failure and may not be built on a different one"
        )
    if (decision["criteria_met"], decision["criteria_total"]) != (
        SGV14_CRITERIA_MET,
        SGV14_CRITERIA_TOTAL,
    ):
        raise PhaseError("SGV14's criteria tally moved since it was frozen")
    drifted: list[str] = []
    missing: list[str] = []
    for path, digest in record["artifacts"].items():
        target = REPO / path
        if not target.is_file():
            if not _ignored(target):
                missing.append(path)
            continue
        if file_sha256(target) != digest:
            drifted.append(path)
    if drifted:
        raise PhaseError(
            f"{len(drifted)} SGV14 artifact(s) no longer hash to what SGV14 recorded: "
            f"{drifted[:3]}. SGV14 must be reconciled before SGV15 may read it"
        )
    return {
        "verdict": decision["verdict"],
        "criteria_met": decision["criteria_met"],
        "criteria_total": decision["criteria_total"],
        "outcome": decision["outcome"]["label"],
        "artifacts_verified": len(record["artifacts"]) - len(missing),
        "artifacts_regenerable_and_absent": sorted(missing),
        "provenance_sha256": file_sha256(s14.PROVENANCE),
        "decision_sha256": file_sha256(s14.DECISION),
    }


def run_reconstruct() -> int:
    """The section-0 freeze. Refuses to proceed unless SGV14 is closed and unchanged."""
    started = time.monotonic()
    sgv14 = verify_sgv14()
    lock = cc_read_json(REPO / "manifests/sgv1/confirmatory_reserve_lock.json")
    if lock.get("status") != "LOCKED" or lock.get("unlock_record") is not None:
        raise PhaseError("the SGV1 confirmatory reserve is not in the state SGV14 left it")
    status = [line for line in _git("status", "--porcelain").splitlines() if line.strip()]
    payload = {
        "artifact": "research_freeze",
        "schema_version": f"sgv15-research_freeze-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_status": (
            "DEVELOPMENT. SGV14 already exposed all ten environments and this stage reads their "
            "outcomes, so no SGV15 result is confirmatory however well it performs. The rule "
            "developed here is frozen for SGV16, which evaluates it on environments nothing has "
            "touched."
        ),
        "repository": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "HEAD"),
            "dirty": bool(status),
            "dirty_entries": status,
        },
        "sgv14_verification": sgv14,
        "sgv14_finding_preserved": {
            "verdict": f"{SGV14_VERDICT} -- {SGV14_CRITERIA_MET} of {SGV14_CRITERIA_TOTAL}",
            "diagnosis": (
                "target supervision improved the ranking on nine of ten environments and reduced "
                "accepted-set harm on all ten; the frozen source cut, calibrated on CORD receipt "
                "rows, admitted prefixes whose realised harm ran from 0.1018 to 0.5731"
            ),
            "not_rewritten": (
                "SGV15 does not repair SGV14 and does not present target certification as though "
                "it had always been part of the frozen method. SGV14 diagnosed the failure; "
                "SGV15 tests the targeted remedy."
            ),
        },
        "library_changes": {
            "src/ocr_risk/risk/bounds.py": (
                "added `clopper_pearson_upper`, an exact one-sided Bernoulli bound. Additive: no "
                "existing caller's behaviour changes."
            ),
            "src/ocr_risk/risk/controller.py": (
                "added a `thresholds` parameter so a caller can declare its cut family, and the "
                "`ltt_clopper_pearson` controller. Both additive; the default path is unchanged."
            ),
            "why_in_the_library": (
                "threshold selection and risk bounds are defined once, under `risk/`. A stage "
                "that needed its own copy would be reimplementing the thing the rules forbid "
                "reimplementing."
            ),
        },
        "dependency_graph": {
            "depth": len(UPSTREAM_SCRIPTS),
            "scripts": list(UPSTREAM_SCRIPTS),
        },
        "upstream_script_sha256": {
            name: file_sha256(REPO / name) for name in UPSTREAM_SCRIPTS if (REPO / name).is_file()
        },
        "sgv1_confirmatory_reserve": {
            "status": lock["status"],
            "unlock_record": lock.get("unlock_record"),
            "reserve_count": int(lock["reserve_count"]),
            "consumed_by_sgv15": False,
            "note": (
                "section 5: a one-way door is not spent on method development. The reserve stays "
                "available for SGV16."
            ),
        },
        "known_defects": [s14.KNOWN_PROVENANCE_DEFECT],
        "frozen_stage_modifications": [],
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(RESEARCH_FREEZE, payload)
    print(
        f"reconstruct: SGV14 {sgv14['verdict']} ({sgv14['criteria_met']}/"
        f"{sgv14['criteria_total']}, outcome {sgv14['outcome']}), "
        f"{sgv14['artifacts_verified']} artifacts verified -> {_relative(RESEARCH_FREEZE)}"
    )
    return 0


# ------------------------------------------------------------------ the frozen configuration


@dataclass(frozen=True, slots=True)
class Frozen:
    """What SGV14 froze and SGV15 may not vary."""

    arm: str
    acquisition: str
    adapt_budget: int
    source_cut: str
    epsilon: float
    target_weight: float
    band_multiple: float
    coverage_floor: float
    configuration_sha256: str


def load_frozen() -> Frozen:
    record = cc_read_json(s14.FROZEN_CONFIGURATION)["method"]
    return Frozen(
        arm=str(record["adaptation_arm"]),
        acquisition=str(record["acquisition"]),
        adapt_budget=int(record["target_labels"]),
        source_cut=str(record["deployment_cut"]),
        epsilon=float(record["primary_epsilon"]),
        target_weight=float(record["source_to_target_weighting"]["target_rows"]),
        band_multiple=float(record["band_multiple"]),
        coverage_floor=float(record["coverage_floor"]),
        configuration_sha256=file_sha256(s14.FROZEN_CONFIGURATION),
    )


def assert_frozen(frozen: Frozen) -> None:
    """The adaptation mechanism is not the variable of interest; refuse if it has moved."""
    expected = {
        "arm": s13.A4_JOINT,
        "acquisition": s13.Q_UNCERTAINTY,
        "source_cut": s13.CUT_FROZEN,
        "adapt_budget": ADAPT_BUDGET,
        "epsilon": PRIMARY_EPSILON,
    }
    actual = {
        "arm": frozen.arm,
        "acquisition": frozen.acquisition,
        "source_cut": frozen.source_cut,
        "adapt_budget": frozen.adapt_budget,
        "epsilon": frozen.epsilon,
    }
    if actual != expected:
        raise PhaseError(f"the frozen adaptation configuration moved: {actual} != {expected}")


def run_freeze() -> int:
    started = time.monotonic()
    frozen = load_frozen()
    assert_frozen(frozen)
    sgv14 = verify_sgv14()
    payload = {
        "artifact": "frozen_sgv14_configuration",
        "schema_version": f"sgv15-frozen_configuration-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_adaptation": {
            "arm": frozen.arm,
            "acquisition": frozen.acquisition,
            "adaptation_labels": frozen.adapt_budget,
            "source_to_target_weighting": {"source_rows": 1.0, "target_rows": frozen.target_weight},
            "band_multiple": frozen.band_multiple,
            "regularization": (
                "SGV5's selected model class and lambda per fold; the joint refit adds none of "
                "its own and raises no capacity"
            ),
            "feature_universe": "SGV1's 96 design columns plus SGV5's 24 structural columns",
            "candidate_universe": "SGV1's frozen g8_union generator ladder",
            "label_definitions": "edits/outcome.py, harm policy strict_worsening",
        },
        "what_sgv15_varies": [
            "where certification labels are sampled from",
            "how many certification labels are bought",
            "how the total budget is split between adaptation and certification",
            "which finite-sample bound places the cut",
            "the declared cut grid",
        ],
        "what_sgv15_may_not_vary": [
            "the adaptation arm, its weighting, its regularisation or its feature universe",
            "the candidate universe or the label definitions",
            "the primary epsilon",
            "the evaluation partition",
        ],
        "sgv14_baseline_cut": frozen.source_cut,
        "sgv14_coverage_floor": frozen.coverage_floor,
        "sgv14_verification": sgv14,
        "configuration_sha256": frozen.configuration_sha256,
        "seeds": {
            "partition_seed": PARTITION_SEED,
            "sample_seed": SAMPLE_SEED,
            "certification_draws": CERT_DRAWS,
            "adaptation_draws": ADAPT_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(FROZEN_CONFIGURATION, payload)
    print(
        f"freeze: {frozen.arm} + {frozen.acquisition} + N_adapt={frozen.adapt_budget}, "
        f"baseline cut {frozen.source_cut} -> {_relative(FROZEN_CONFIGURATION)}"
    )
    return 0


# ------------------------------------------------- the fitting / certification split
#
# Section 7's independence requirement, made structural. The split is by DOCUMENT (by VOLUME on
# OCR-D-SBB), content-blind, and it partitions SGV14's ADAPTATION half only. SGV14's evaluation
# half is never touched by anything in this stage except the final endpoint read.


def certification_partition(corpus: str, documents: list[str]) -> dict[str, str]:
    """Split one environment's adaptation documents into a fitting and a certification half.

    The key is a hash of the group name under a fixed seed, so which half a page lands in is
    decided by its name and nothing else. Whole groups move together, for the same reason SGV14
    moved whole volumes: pages of one book share a typeface, a compositor and a scanning session,
    and certifying a cut on the very book the model was fitted on is not an independent estimate.
    """
    groups: dict[str, list[str]] = {}
    for name in sorted(documents):
        groups.setdefault(group_of(corpus, name), []).append(name)
    order = sorted(
        groups, key=lambda key: (canonical_hash({"seed": PARTITION_SEED, "group": key}), key)
    )
    target = FIT_FRACTION * len(documents)
    roles: dict[str, str] = {}
    taken = 0
    for key in order:
        members = groups[key]
        role = ROLE_FIT if taken + len(members) / 2.0 <= target else ROLE_CERT
        if role == ROLE_FIT:
            taken += len(members)
        roles.update(dict.fromkeys(members, role))
    return roles


def run_partition() -> int:
    """Write the split and its capacity. An environment that cannot supply both is marked."""
    started = time.monotonic()
    if not RESEARCH_FREEZE.is_file():
        raise PhaseError("run --reconstruct first; SGV14 must be verified before SGV15 reads it")
    rows = pd.read_parquet(s14.ENVIRONMENT_ROWS)
    sgv14_inventory = {
        record["environment"]: record
        for record in cc_read_json(s14.ENVIRONMENT_INVENTORY)["environments"]
    }
    sufficient = set(cc_read_json(s14.DESIGN_RECORD)["environments_sufficient"])
    records: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        block = rows[(rows["environment"] == name) & (rows["role"] == s14.ROLE_ADAPTATION)]
        documents = sorted(block["document_id"].astype(str).unique())
        roles = certification_partition(spec["corpus"], documents)
        assigned = block["document_id"].astype(str).map(roles)
        fit = block[assigned == ROLE_FIT]
        cert = block[assigned == ROLE_CERT]
        capacity = {
            "fit_rows": len(fit),
            "fit_documents": int(fit["document_id"].nunique()),
            "certification_rows": len(cert),
            "certification_documents": int(cert["document_id"].nunique()),
        }
        limits = []
        if capacity["fit_rows"] < ADAPT_BUDGET:
            limits.append(
                f"{capacity['fit_rows']} fitting rows cannot supply {ADAPT_BUDGET} "
                "adaptation labels"
            )
        largest = max(CERT_BUDGETS)
        if capacity["certification_rows"] < largest:
            limits.append(
                f"{capacity['certification_rows']} certification rows cannot supply the largest "
                f"certification budget of {largest}"
            )
        records.append(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "base_engine": spec["base_engine"],
                "sgv14_sufficient": name in sufficient,
                "adaptation_rows": len(block),
                "adaptation_documents": len(documents),
                "evaluation_rows": int(sgv14_inventory[name]["rows_evaluation"]),
                "evaluation_documents": int(sgv14_inventory[name]["documents_evaluation"]),
                "evaluation_beneficial": int(sgv14_inventory[name]["beneficial_evaluation"]),
                "evaluation_harmful": int(sgv14_inventory[name]["harmful_evaluation"]),
                **capacity,
                "certification_harmful_rate": float(cert["is_harmful"].mean())
                if len(cert)
                else float("nan"),
                "fit_harmful_rate": float(fit["is_harmful"].mean()) if len(fit) else float("nan"),
                "largest_feasible_certification_budget": max(
                    [b for b in CERT_BUDGETS if b <= capacity["certification_rows"]] or [0]
                ),
                "capacity_limited": bool(limits),
                "capacity_limits": limits,
            }
        )
        assignments.append(
            pd.DataFrame(
                {
                    "environment": name,
                    "candidate_id": block["candidate_id"].astype(str).to_numpy(),
                    "document_id": block["document_id"].astype(str).to_numpy(),
                    "sgv15_role": assigned.to_numpy(str),
                }
            )
        )

    table = pd.concat(assignments, ignore_index=True)
    for name, group in table.groupby("environment", observed=True):
        fit_documents = set(group.loc[group["sgv15_role"] == ROLE_FIT, "document_id"])
        cert_documents = set(group.loc[group["sgv15_role"] == ROLE_CERT, "document_id"])
        if fit_documents & cert_documents:
            raise PhaseError(f"{name}: the fitting and certification halves share a document")
    _write_parquet_once(CERTIFICATION_ROWS, table)
    limited = [record["environment"] for record in records if record["capacity_limited"]]
    payload = {
        "artifact": "environment_inventory",
        "schema_version": f"sgv15-environment_inventory-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partition": {
            "seed": PARTITION_SEED,
            "fit_fraction": FIT_FRACTION,
            "unit": "document, or volume on OCR-D-SBB",
            "splits": "SGV14's ADAPTATION half only; its EVALUATION half is untouched",
            "inputs_allowed": ["document_id"],
            "inputs_forbidden": [
                "image content",
                "OCR output",
                "candidate scores",
                "outcome labels",
            ],
        },
        "environments": records,
        "capacity_limited": limited,
        "totals": {
            "environments": len(records),
            "fit_rows": int(sum(r["fit_rows"] for r in records)),
            "certification_rows": int(sum(r["certification_rows"] for r in records)),
        },
        "artifacts": {_relative(CERTIFICATION_ROWS): file_sha256(CERTIFICATION_ROWS)},
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(ENVIRONMENT_INVENTORY, payload)
    print(
        f"partition: {len(records)} environments, "
        f"{payload['totals']['fit_rows']} fitting / "
        f"{payload['totals']['certification_rows']} certification rows, "
        f"{len(limited)} capacity-limited -> {_relative(ENVIRONMENT_INVENTORY)}"
    )
    return 0


# ------------------------------------------------------------------ per-environment setup


@dataclass(frozen=True, slots=True)
class Environment:
    """One environment, with SGV14's fold rebuilt and this stage's two halves resolved."""

    name: str
    corpus: str
    base_engine: str
    setup: Any
    fit_positions: np.ndarray
    cert_positions: np.ndarray
    cert_documents: np.ndarray
    cert_harmful: np.ndarray
    source_prior: tuple[float, float]
    sufficient: bool
    capacity_limited: bool


def _roles(name: str) -> dict[str, str]:
    table = pd.read_parquet(CERTIFICATION_ROWS)
    block = table[table["environment"] == name]
    return dict(
        zip(block["candidate_id"].astype(str), block["sgv15_role"].astype(str), strict=True)
    )


def build_environment(
    name: str,
    rows: pd.DataFrame,
    sgv1: tuple[Any, np.ndarray, list[int]],
    selections: dict[str, Any],
    frozen_scores: pd.DataFrame,
    references: dict[str, Any],
    frozen: Frozen,
    roles: dict[str, str],
    inventory: dict[str, Any],
) -> Environment:
    """SGV14's fold, plus the fitting / certification split expressed as pool positions."""
    spec = s14.environment_of(name)
    engine = spec["base_engine"]
    if engine not in references:
        references[engine] = reference_state(engine, sgv1, selections[engine], frozen_scores)
    combined = combine(name, rows, sgv1)
    setup = environment_setup(combined, selections[engine], s14.load_frozen(), references[engine])
    view = setup.view
    assignment = np.array([roles.get(str(c), "") for c in view.candidates.tolist()], dtype=object)
    unknown = int((assignment == "").sum())
    if unknown:
        raise PhaseError(f"{name}: {unknown} pool rows carry no SGV15 role")
    fit_positions = np.flatnonzero(assignment == ROLE_FIT)
    cert_positions = np.flatnonzero(assignment == ROLE_CERT)
    outcomes = setup.outcomes["harmful"]
    record = next(r for r in inventory["environments"] if r["environment"] == name)
    # The source-informed prior for M3, formed BEFORE any target row is read: the harm rate the
    # published source cut takes on the SOURCE engines' calibration rows, at the primary epsilon.
    calibration = setup.calibration
    tau = float(setup.state.source_threshold[PRIMARY_KEY])
    accepted = calibration.frozen_score >= tau if np.isfinite(tau) else np.zeros(0, dtype=bool)
    rate = float(calibration.harmful[accepted].mean()) if int(accepted.sum()) else 0.0
    return Environment(
        name=name,
        corpus=spec["corpus"],
        base_engine=engine,
        setup=setup,
        fit_positions=fit_positions,
        cert_positions=cert_positions,
        cert_documents=view.documents[cert_positions],
        cert_harmful=outcomes[cert_positions],
        source_prior=(rate * SHRINKAGE_PRIOR_WEIGHT, SHRINKAGE_PRIOR_WEIGHT),
        sufficient=bool(record["sgv14_sufficient"]),
        capacity_limited=bool(record["capacity_limited"]),
    )


def adaptation_order(environment: Environment, rule: str, seed: int, restrict: bool) -> np.ndarray:
    """SGV14's acquisition ordering, optionally restricted to the fitting half.

    The ordering itself is SGV13's function on SGV13's `PoolView`, which carries no outcome. The
    restriction keeps the relative order and drops the certification half, so "acquire from the
    fitting documents" is a filter on a frozen ordering rather than a different rule.
    """
    order = acquisition_order(environment.setup.view, rule, environment.setup.base, seed)
    if not restrict:
        return order
    allowed = np.zeros(environment.setup.view.size, dtype=bool)
    allowed[environment.fit_positions] = True
    return order[allowed[order]]


# ------------------------------------------------------------------ certification samplers
#
# Every sampler is handed the certification half's POSITIONS, its DOCUMENTS and the frozen score,
# and returns a subset of positions. None of them may read an outcome: the leakage suite walks
# each of their syntax trees for `harmful`, `beneficial` and `outcomes`.


def _document_stratified(documents: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Round-robin over a shuffled document order, so a page contributes its first row first.

    This is what "representative" has to mean when candidates are clustered: a uniform draw over
    ROWS is dominated by whichever pages happen to carry the most candidates, and a risk estimate
    from it is really an estimate of those pages' risk.
    """
    generator = np.random.default_rng(seed)
    by_document: dict[str, list[int]] = {}
    for position, name in enumerate(documents.tolist()):
        by_document.setdefault(str(name), []).append(position)
    names = sorted(by_document)
    for name in names:
        rows = by_document[name]
        generator.shuffle(rows)
    ordered = [names[int(i)] for i in generator.permutation(len(names))]
    out: list[int] = []
    depth = 0
    while len(out) < size:
        progressed = False
        for name in ordered:
            rows = by_document[name]
            if depth < len(rows):
                out.append(rows[depth])
                progressed = True
                if len(out) >= size:
                    break
        if not progressed:
            break
        depth += 1
    return np.asarray(out, dtype=int)


def certification_sample(
    environment: Environment, sampler: str, size: int, seed: int
) -> np.ndarray:
    """Positions in the pool, drawn from the certification half only."""
    positions = environment.cert_positions
    if size <= 0 or positions.size == 0:
        return np.zeros(0, dtype=int)
    take = int(min(size, positions.size))
    generator = np.random.default_rng(seed)
    if sampler == CS1_UNIFORM:
        return positions[generator.permutation(positions.size)[:take]]
    if sampler == CS2_DOCUMENT:
        local = _document_stratified(environment.cert_documents, take, seed)
        return positions[local]
    if sampler == CS0_UNCERTAINTY:
        # SGV13's own uncertainty ordering, restricted to the certification half. Carried as a
        # NEGATIVE baseline: it concentrates the sample where the deployment is about to cut,
        # which is exactly where the harmful rate is unrepresentative of the accepted set.
        view = environment.setup.view
        distance = np.abs(view.frozen_score[positions] - view.frozen_threshold)
        return positions[np.lexsort((np.arange(positions.size), distance))[:take]]
    if sampler in (CS_ONE_DOCUMENT, CS_FEW_DOCUMENTS):
        wanted = 1 if sampler == CS_ONE_DOCUMENT else FEW_DOCUMENTS
        names = sorted({str(n) for n in environment.cert_documents.tolist()})
        chosen = {names[int(i)] for i in generator.permutation(len(names))[:wanted]}
        mask = np.array([str(n) in chosen for n in environment.cert_documents.tolist()])
        pool = positions[mask]
        return pool[generator.permutation(pool.size)[: min(take, pool.size)]]
    raise PhaseError(f"unknown certification sampler {sampler!r}")


# ------------------------------------------------------------------ the cut methods
#
# Every cut goes through `risk.select_threshold`, the repository's single implementation, which
# already splits the confidence level across the declared grid by union bound. What varies
# between methods is which rows the estimate is made on and which finite-sample bound is used.


def cut_grid(scores: np.ndarray) -> np.ndarray:
    """The declared cut family, evaluated on the UNLABELLED certification-partition scores.

    Nine acceptance depths, fixed before any endpoint was read. Reading the grid off the
    unlabelled half means the labelled sample is spent entirely on estimating risk and none of
    it on choosing where to look, which is what makes the union bound over nine points the whole
    multiplicity story.
    """
    if scores.size == 0:
        return np.zeros(0, dtype=float)
    return np.unique(np.quantile(scores, [1.0 - q for q in CUT_GRID]))


def place_cut(
    method: str,
    grid: np.ndarray,
    scores: np.ndarray,
    harmful: np.ndarray,
    epsilon: float,
    prior: tuple[float, float] | None,
) -> Any:
    """One method's threshold decision on one sample. No method may see an evaluation row."""
    controller = CONTROLLER_OF.get(method)
    if controller is None:
        raise PhaseError(f"{method} has no controller and cannot place a cut")
    return select_threshold(
        scores,
        harmful,
        epsilon,
        delta=CERT_DELTA,
        controller=controller,
        thresholds=grid,
        prior=prior if controller == "ltt_beta_binomial" else None,
    )


def frozen_source_tau(environment: Environment, adapted: Any, epsilon: float) -> float:
    """SGV14's cut: `select_threshold` on the SOURCE engines' calibration rows, unchanged."""
    thresholds = s13.thresholds_for(
        environment.setup, adapted, _empty_budget(environment), s13.CUT_FROZEN
    )
    return float(thresholds[epsilon_key(epsilon)]["tau"])


def _empty_budget(environment: Environment) -> Any:
    """A zero-size budget, so `thresholds_for` takes its frozen-source branch."""
    return purchase(environment.setup, np.zeros(0, dtype=int), 0)


def oracle_decision(
    environment: Environment, eval_scores: np.ndarray, grid: np.ndarray, epsilon: float
) -> Any:
    """The deepest declared cut whose realised harm holds ON THE EVALUATION BLOCK.

    Analysis only. It bounds what any certification procedure could have achieved on this
    ranking and it may never reach a selection path; the leakage suite asserts its call sites.
    """
    block = environment.setup.evaluation
    return select_threshold(
        eval_scores, block.harmful, epsilon, controller="empirical", thresholds=grid
    )


def refusal_tau(decision: Any) -> float:
    """The threshold a decision actually deploys at, with abstention meaning abstention.

    `select_threshold` signals infeasibility by returning a threshold just above the largest
    score IT WAS SHOWN, which accepts nothing on that sample. Carried across to a different
    block that is no longer true: evaluation rows scoring above everything in a small
    certification sample would still be accepted, so a method that had just declared it could
    not certify any cut would deploy one anyway. Infeasible means accept nothing, everywhere.
    """
    return float(decision.tau) if decision.feasible else float("inf")


def deploy(
    environment: Environment, eval_scores: np.ndarray, tau: float, epsilon: float
) -> dict[str, Any]:
    """Apply one threshold to the evaluation block, through SGV12's own `deployed`.

    The evaluation scores are computed ONCE per fitted model and passed in. Rescoring several
    thousand rows for every (method, epsilon, draw) cell would multiply the stage's runtime by
    two orders of magnitude and would return the identical vector every time.
    """
    block = environment.setup.evaluation
    scores = eval_scores
    point = deployed(scores, block.harmful, block.beneficial, tau, epsilon)
    accepted = np.asarray(scores >= tau) if np.isfinite(tau) else np.zeros(block.size, dtype=bool)
    return {
        "tau": float(tau),
        "coverage": float(point["coverage"]),
        "realized_harm_rate": float(point["realized_harm_rate"]),
        "holds_bound": bool(point["holds_bound"]),
        "repair_recall": float(point["repair_recall"]),
        "risk_controlled": float(point["risk_controlled_repair_recall"]),
        "n_accepted": int(point["n_accepted"]),
        "harmful_accepted": int((accepted & block.harmful).sum()),
        "beneficial_accepted": int((accepted & block.beneficial).sum()),
        "repairs_captured": int(point["repairs_captured"]),
        "joint_harm_rate": float(point["joint_harm_rate"]),
        "accepted": accepted,
    }


@dataclass(frozen=True, slots=True)
class Adapted:
    """One fitted arm, with the label accounting that produced it and its scored blocks.

    `eval_scores` and `cert_scores` are the model's output on the evaluation block and on the
    whole certification half. Both are fixed the moment the model is, so they are computed once
    here rather than once per cell.
    """

    model: Any
    n_adapt: int
    documents: int
    harmful: int
    beneficial: int
    neutral: int
    eval_scores: np.ndarray
    cert_scores: np.ndarray


def fit_adaptation(
    environment: Environment,
    rule: str,
    size: int,
    seed: int,
    restrict: bool = True,
    permute_seed: int | None = None,
) -> Adapted:
    """SGV13's frozen joint refit on `size` labels bought from the fitting half.

    `permute_seed` is section 22's control 3, and it is SGV13's own `purchase` argument rather
    than a second implementation: the rows, the documents, the acquisition ordering and the class
    balance are all held exactly, and only the label-to-row assignment is destroyed.
    """
    if size <= 0:
        return _scored(environment, frozen_arm(), 0, 0, 0, 0, 0)
    order = adaptation_order(environment, rule, seed, restrict)
    budget = purchase(environment.setup, order, size, permute_seed=permute_seed)
    frozen = load_frozen()
    model = fit_joint_arm(environment.setup, budget, frozen.target_weight, False, s13.A4_JOINT)
    return _scored(
        environment,
        model,
        int(budget.size),
        int(budget.n_documents),
        int(budget.harmful.sum()),
        int(budget.beneficial.sum()),
        int((~budget.harmful & ~budget.beneficial).sum()),
    )


def _scored(
    environment: Environment,
    model: Any,
    n_adapt: int,
    documents: int,
    harmful: int,
    beneficial: int,
    neutral: int,
) -> Adapted:
    """Score the evaluation block and the certification half once, and freeze both."""
    setup = environment.setup
    return Adapted(
        model=model,
        n_adapt=n_adapt,
        documents=documents,
        harmful=harmful,
        beneficial=beneficial,
        neutral=neutral,
        eval_scores=np.asarray(model.score(setup, setup.evaluation), dtype=float),
        cert_scores=np.asarray(model.score(setup, _cert_block(environment)), dtype=float),
    )


# ------------------------------------------------------------------ the pre-registration


def run_preregister() -> int:
    """Freeze the methods, samplers, grid, budgets, criteria and the selection plan."""
    started = time.monotonic()
    if CELL_TABLE.exists():
        raise PhaseError("cells already exist; the design must be frozen before any is scored")
    if not ENVIRONMENT_INVENTORY.is_file():
        raise PhaseError("run --partition first; the capacity limits belong in the design")
    frozen = load_frozen()
    assert_frozen(frozen)
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    environments = [record["environment"] for record in inventory["environments"]]
    payload = {
        "artifact": "design_record",
        "schema_version": f"sgv15-design_record-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git("rev-parse", "HEAD"),
        "scientific_status": {
            "kind": "DEVELOPMENT",
            "why": (
                "SGV14 already exposed all ten environments and this stage reads their outcomes. "
                "Nothing here is confirmatory however well it performs."
            ),
            "confirmatory_stage": (
                "SGV16 evaluates the rule frozen here on environments nothing has touched"
            ),
            "outer_number_is_blinded": False,
        },
        "hypothesis": (
            "given a ranking already improved by limited target supervision, a small, "
            "representative, independently held target sample can certify how far into that "
            "ranking a deployment may safely cut"
        ),
        "sub_claims": {
            "P1a": "a certified target cut beats the frozen source cut",
            "P1b": "the label cost of certification is measurable",
            "P1c": "adaptation labels and certification labels should be separated",
            "P1d": "representative sampling beats targeted sampling for risk estimation",
            "P1e": "a conservative bound prevents overshoot without collapsing coverage",
        },
        "frozen_adaptation": {
            "arm": frozen.arm,
            "acquisition": frozen.acquisition,
            "weighting": {"source_rows": 1.0, "target_rows": frozen.target_weight},
            "source": _relative(s14.FROZEN_CONFIGURATION),
        },
        "label_roles": {
            "adaptation": "bought from the SGV15 fitting half; may reach a fit",
            "certification": (
                "sampled from the SGV15 certification half; may reach a risk estimate and a cut, "
                "and may never reach a fit"
            ),
            "separation": "by DOCUMENT, so the two halves share no page and no volume",
            "evaluation": "never labelled for any purpose in this stage except the endpoint read",
        },
        "experiments": {
            "A_mechanism_isolation": {
                "adaptation_labels": ADAPT_BUDGET,
                "certification_labels": list(CERT_BUDGETS),
                "question": (
                    "does independent representative target risk evidence solve the SGV14 cut "
                    "failure at all?"
                ),
                "note": (
                    "this experiment deliberately increases the TOTAL label count. Criticising it "
                    "for label efficiency is a category error; that is experiment B's question."
                ),
            },
            "B_fixed_total_budget": {
                "total_labels": TOTAL_BUDGET,
                "allocations": [list(a) for a in ALLOCATIONS],
                "question": (
                    "what is the adaptation-versus-certification tradeoff under SGV13's own "
                    "annotation budget?"
                ),
                "grid_frozen": "no allocation may be added after an endpoint is inspected",
            },
        },
        "methods": {
            M0_FROZEN_SOURCE: (
                "SGV14's baseline: `select_threshold` on the SOURCE calibration rows, "
                "transported unchanged. Reads no target row."
            ),
            M1_EMPIRICAL: (
                "the deepest declared cut whose OBSERVED risk on the certification sample is at "
                "most epsilon. Deliberately non-conservative, so that finite-sample uncertainty "
                "has something to be measured against."
            ),
            M2_CLOPPER: (
                "the deepest declared cut whose exact one-sided Clopper-Pearson upper bound is "
                "at most epsilon, at alpha = 0.05 split across the nine-point grid. The primary "
                "candidate."
            ),
            M2B_BENTKUS: (
                "the same, under the Hoeffding-Bentkus bound the repository already used. "
                "Carried because it is distribution-free over any bounded loss and therefore "
                "strictly more conservative than the exact Bernoulli bound."
            ),
            M3_SHRINKAGE: (
                "a Beta posterior whose prior is the harm rate the published source cut takes on "
                "the SOURCE calibration rows, worth "
                f"{SHRINKAGE_PRIOR_WEIGHT:g} pseudo-observations. Formed before any target row "
                "is read. It is a credible bound, not a guarantee, and SGV14 gives every reason "
                "to expect a source prior to be wrong in the unsafe direction."
            ),
            M4_CROSS_FITTED: (
                "K-fold cross-fitting over the labelled target documents: fit without a fold, "
                "certify on it. Registered as SECONDARY and run only at the selected allocation, "
                "because sample-split certification is the methodologically cleaner primary."
            ),
            C1_ORACLE: (
                "ANALYSIS ONLY. The deepest declared cut whose realised harm holds on the "
                "EVALUATION block. It bounds what any certification procedure could achieve and "
                "may never influence a selection."
            ),
        },
        "primary_method_candidate": M2_CLOPPER,
        "alpha": CERT_DELTA,
        "cut_grid": {
            "acceptance_depths": list(CUT_GRID),
            "evaluated_on": "the unlabelled certification-partition scores",
            "size": len(CUT_GRID),
            "multiplicity": (
                "the confidence level is split across the declared grid by union bound inside "
                "`risk.select_threshold`, so the per-test level is alpha / 9"
            ),
            "secondary_fine_grid": FINE_GRID_POINTS,
        },
        "samplers": {
            CS0_UNCERTAINTY: "DIAGNOSTIC NEGATIVE. SGV13's uncertainty ordering, restricted.",
            CS1_UNIFORM: "uniform over certification-half rows",
            CS2_DOCUMENT: "document round-robin. The PRIMARY sampler.",
            CS_ONE_DOCUMENT: "control: every certification label from one document",
            CS_FEW_DOCUMENTS: f"control: every certification label from {FEW_DOCUMENTS} documents",
        },
        "primary_sampler": PRIMARY_SAMPLER,
        "excluded_samplers": {
            "cs3_score_stratified_probability_sample": (
                "registered and NOT implemented. A weighted risk estimate has no conservative "
                "finite-sample bound in the repository's single implementation, and adding one "
                "would be a method change this stage does not need. Section 10 marks it optional."
            )
        },
        "adaptation_acquisition_comparison": {
            "primary": A_UNCERTAINTY,
            "secondary": A_DOCUMENT_RANDOM,
            "status": (
                "SECONDARY development question. SGV14 found random matched or beat "
                "`q2_uncertainty` on 7 of 8 relevant comparisons; it is measured again here and "
                "it may not contaminate the cut-certification question."
            ),
        },
        "draws": {
            "certification": CERT_DRAWS,
            "adaptation": ADAPT_DRAWS,
            "note": (
                "`q2_uncertainty` is a deterministic function of the frozen score and the "
                "transported threshold, so its adaptation draws are identical by construction; "
                "SGV14 measured a standard deviation of 0. The document-random acquisition "
                "comparison is stochastic and carries the full count."
            ),
        },
        "primary_endpoint": (
            "risk-controlled repair recall on SGV14's evaluation block at epsilon = 0.10: repair "
            "recall where the realised accepted-set harm rate holds the bound, and zero where it "
            "does not"
        ),
        "secondary_epsilons": [e for e in EPSILONS if e != PRIMARY_EPSILON],
        "secondary_epsilon_rule": "reported, never used to rescue a failure at 0.10",
        "criteria": {
            "1_safety_recovery": (
                f"the certified cut holds the realised harm bound on at least {BREADTH_REQUIRED} "
                f"of {ENVIRONMENT_COUNT} development environments, against SGV14's 3"
            ),
            "2_operational_gain": (
                f"risk-controlled repair recall at epsilon = 0.10 exceeds SGV14's frozen source "
                f"cut on at least {BREADTH_REQUIRED} of {ENVIRONMENT_COUNT}"
            ),
            "3_non_degeneracy": (
                "safety is not obtained through near-zero deployment: accepted coverage is at "
                f"least SGV13's own floor of {frozen.coverage_floor:.6f} on at least "
                f"{BREADTH_REQUIRED} of {ENVIRONMENT_COUNT}"
            ),
            "4_representative_sampling_matters": (
                "the document-stratified sampler beats the uncertainty-biased sampler on the "
                "primary endpoint, averaged over environments at the selected budget"
            ),
            "5_cut_mechanism_is_load_bearing": (
                "with the adapted ranking held fixed, the certified cut beats the frozen source "
                f"cut on at least {BREADTH_REQUIRED} of {ENVIRONMENT_COUNT}"
            ),
            "6_fixed_budget_feasibility": (
                f"at least one allocation with N_total = {TOTAL_BUDGET} achieves criteria 1 and 2"
            ),
        },
        "criteria_total": 6,
        "breadth_required": BREADTH_REQUIRED,
        "coverage_floor": frozen.coverage_floor,
        "verdict_rule": (
            "SUPPORTED requires all six; otherwise NOT SUPPORTED. SUPPORTED here means the "
            "mechanism works in development and is worth freezing for SGV16 -- it is not "
            "external confirmation and the report may not describe it as such."
        ),
        "selection_plan": {
            "phase_a": "mechanism isolation over all development environments",
            "phase_b": (
                "nested leave-one-environment-out selection of the certification method, the cut "
                "grid and the allocation. The winner is the configuration with the best mean "
                "held-out-environment endpoint, never the best pooled number."
            ),
            "phase_c": "freeze one configuration as the SGV16 confirmatory method",
            "oracle_excluded": "C1 may not enter any selection",
        },
        "statistics": {
            "resampling_unit": "document",
            "primary_family": (
                "the selected certified method against SGV14's frozen source cut, on "
                "risk-controlled repair recall and on realised harm, per environment"
            ),
            "multiplicity": "Holm at 0.05 over the pre-registered family",
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "separation": (
                "the confidence procedure that CHOOSES the cut and the bootstrap that EVALUATES "
                "the method are different objects on different rows, and never share data"
            ),
        },
        "environments": environments,
        "environments_capacity_limited": inventory["capacity_limited"],
        "prohibited_after_this_file_exists": [
            "adding a certification budget, an allocation, a sampler or a method",
            "changing the cut grid, alpha, epsilon or the coverage floor",
            "changing the adaptation arm, its weighting or its feature universe",
            "letting the oracle cut influence any selection",
            "spending the CORD confirmatory reserve",
        ],
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DESIGN_RECORD, payload)
    print(
        f"preregister: {len(METHODS)} methods, {len(SAMPLERS)} samplers, "
        f"{len(CERT_BUDGETS)} certification budgets, {len(ALLOCATIONS)} allocations, "
        f"6 criteria -> {_relative(DESIGN_RECORD)}"
    )
    return 0


# ------------------------------------------------------------------ one cell


def evaluate_cell(
    environment: Environment,
    adapted: Adapted,
    grid: np.ndarray,
    sample: np.ndarray,
    method: str,
    epsilon: float,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """One (environment, adaptation, sampler, N_cert, method, draw) cell at one epsilon."""
    setup = environment.setup
    view = setup.view
    row: dict[str, Any] = dict(context)
    row["method"] = method
    row["epsilon"] = float(epsilon)
    row["n_cert"] = int(sample.size)
    row["cert_documents"] = len({str(n) for n in view.documents[sample].tolist()})
    harmful = setup.outcomes["harmful"][sample] if sample.size else np.zeros(0, dtype=bool)
    row["cert_harmful"] = int(harmful.sum())
    row["cert_harmful_rate"] = float(harmful.mean()) if sample.size else float("nan")

    if method == M0_FROZEN_SOURCE:
        tau = frozen_source_tau(environment, adapted.model, epsilon)  # source rows, not target
        row["certified_risk_upper_bound"] = float("nan")
        row["certified_observed_risk"] = float("nan")
        row["cut_feasible"] = bool(np.isfinite(tau))
        row["cut_accepted_in_sample"] = 0
    elif method == C1_ORACLE:
        decision = oracle_decision(environment, adapted.eval_scores, grid, epsilon)
        tau = refusal_tau(decision)
        row["certified_risk_upper_bound"] = float("nan")
        row["certified_observed_risk"] = float("nan")
        row["cut_feasible"] = bool(decision.feasible)
        row["cut_accepted_in_sample"] = 0
    else:
        if sample.size == 0:
            tau = float("inf")
            row["certified_risk_upper_bound"] = 1.0
            row["certified_observed_risk"] = float("nan")
            row["cut_feasible"] = False
            row["cut_accepted_in_sample"] = 0
        else:
            scores = adapted.cert_scores[_cert_index(environment, sample)]
            decision = place_cut(method, grid, scores, harmful, epsilon, environment.source_prior)
            tau = refusal_tau(decision)
            row["certified_risk_upper_bound"] = float(decision.risk_upper_bound)
            row["certified_observed_risk"] = float(decision.observed_risk)
            row["cut_feasible"] = bool(decision.feasible)
            row["cut_accepted_in_sample"] = int(decision.n_accepted)

    point = deploy(environment, adapted.eval_scores, tau, epsilon)
    accepted = point.pop("accepted")
    row.update(point)
    row["n_adapt"] = int(adapted.n_adapt)
    row["adapt_documents"] = int(adapted.documents)
    row["adapt_harmful"] = int(adapted.harmful)
    row["adapt_beneficial"] = int(adapted.beneficial)
    row["adapt_neutral"] = int(adapted.neutral)
    row["n_total"] = int(adapted.n_adapt + sample.size)
    counts = None
    if epsilon == PRIMARY_EPSILON:
        block = setup.evaluation
        counts = per_document_counts(block, accepted, sorted({*block.documents.tolist()}))
    return row, counts


def _cert_index(environment: Environment, sample: np.ndarray) -> np.ndarray:
    """Where each sampled position sits inside the certification half's score vector."""
    lookup = {int(p): i for i, p in enumerate(environment.cert_positions.tolist())}
    return np.asarray([lookup[int(p)] for p in sample.tolist()], dtype=int)


def _sample_block(environment: Environment, sample: np.ndarray) -> Any:
    """The certification sample dressed as a block, so an arm can be scored on it."""
    setup = environment.setup
    view = setup.view
    return s13.Block(
        name="certification",
        index=view.index[sample],
        features=view.features[sample],
        frozen_score=view.frozen_score[sample],
        documents=view.documents[sample],
        sites=view.sites[sample],
        candidates=view.candidates[sample],
        harmful=setup.outcomes["harmful"][sample],
        beneficial=setup.outcomes["beneficial"][sample],
    )


def _ranking_row(environment: Environment, adapted: Adapted) -> dict[str, float]:
    """The ranking diagnostics, so a cut result can be read against the ordering it cut."""
    block = environment.setup.evaluation
    scores = adapted.eval_scores
    quality = ranking_quality(scores, block.harmful, block.beneficial)
    achievable = frontier(scores, block.harmful, block.beneficial)
    return {
        "rank__auroc_safe": float(quality["auroc_safe"]),
        "rank__average_precision_beneficial": float(quality["average_precision_beneficial"]),
        **{
            f"frontier__{epsilon_key(e)}": float(achievable[epsilon_key(e)]["repair_recall"])
            for e in EPSILONS
        },
    }


# ------------------------------------------------------------------ the experiments


def _context(environment: Environment, experiment: str, **extra: Any) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "environment": environment.name,
        "corpus": environment.corpus,
        "base_engine": environment.base_engine,
        "sgv14_sufficient": environment.sufficient,
        "capacity_limited": environment.capacity_limited,
        "evaluation_rows": int(environment.setup.evaluation.size),
        "evaluation_documents": len({*environment.setup.evaluation.documents.tolist()}),
        "certification_pool_rows": int(environment.cert_positions.size),
        "certification_pool_documents": len({*environment.cert_documents.tolist()}),
        **extra,
    }


def _environments(
    only: str | None = None,
) -> tuple[list[Environment], dict[str, Any]]:
    """Build every environment once. This is the expensive part and it is done a single time."""
    rows = pd.read_parquet(s14.ENVIRONMENT_ROWS)
    sgv1 = s6.load_base()
    selections = s6.frozen_selection()
    frozen_scores = s6.load_frozen_scores()
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    references: dict[str, Any] = {}
    frozen = load_frozen()
    built: list[Environment] = []
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        if only is not None and name != only:
            continue
        began = time.monotonic()
        environment = build_environment(
            name,
            rows,
            sgv1,
            selections,
            frozen_scores,
            references,
            frozen,
            _roles(name),
            inventory,
        )
        built.append(environment)
        print(
            f"  {name}: fit {environment.fit_positions.size} / cert "
            f"{environment.cert_positions.size} pool rows, "
            f"{environment.setup.evaluation.size} evaluation rows "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    return built, inventory


def _collect(
    rows: list[dict[str, Any]],
    counts: list[tuple[tuple[str, str], np.ndarray]],
    environment: Environment,
    adapted: Adapted,
    grid: np.ndarray,
    sample: np.ndarray,
    methods: tuple[str, ...],
    context: dict[str, Any],
    ranking: dict[str, float],
    key: tuple[str, ...],
) -> None:
    for method in methods:
        for epsilon in EPSILONS:
            row, block = evaluate_cell(environment, adapted, grid, sample, method, epsilon, context)
            row.update(ranking)
            row["cell"] = "|".join((*key, method))
            rows.append(row)
            if block is not None:
                counts.append(
                    (
                        (environment.name, "|".join((*key, method))),
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


def run_experiment_a(only: str | None = None) -> int:
    """Experiment A: the adaptation budget fixed at 250, independent certification budgets added."""
    started = time.monotonic()
    if not DESIGN_RECORD.is_file():
        raise PhaseError("run --preregister before any experiment")
    frozen = load_frozen()
    assert_frozen(frozen)
    built, _inventory = _environments(only)
    rows: list[dict[str, Any]] = []
    counts: list[tuple[tuple[str, str], np.ndarray]] = []
    for environment in built:
        began = time.monotonic()
        seed = _stable_seed("sgv15-adapt", str(SAMPLE_SEED), environment.name, A_UNCERTAINTY)
        adapted = fit_adaptation(environment, A_UNCERTAINTY, ADAPT_BUDGET, seed)
        ranking = _ranking_row(environment, adapted)
        grid = cut_grid(adapted.cert_scores)
        for sampler in (*SAMPLERS, *CONTROL_SAMPLERS):
            for size in CERT_BUDGETS:
                draws = 1 if size == 0 else CERT_DRAWS
                for draw in range(draws):
                    sample_seed = _stable_seed(
                        "sgv15-cert",
                        str(SAMPLE_SEED),
                        environment.name,
                        sampler,
                        str(size),
                        str(draw),
                    )
                    sample = certification_sample(environment, sampler, size, sample_seed)
                    context = _context(
                        environment,
                        "A",
                        sampler=sampler,
                        requested_cert=size,
                        draw=draw,
                        acquisition=A_UNCERTAINTY,
                        allocation="250+n",
                    )
                    methods = (
                        (M0_FROZEN_SOURCE, C1_ORACLE) if size == 0 else (*METHODS[1:], C1_ORACLE)
                    )
                    _collect(
                        rows,
                        counts,
                        environment,
                        adapted,
                        grid,
                        sample,
                        methods,
                        context,
                        ranking,
                        (environment.name, "A", sampler, str(size), str(draw)),
                    )
        print(
            f"  {environment.name}: {len(rows)} cells so far ({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    _append_cells(rows, counts)
    print(f"experiment-a: {len(rows)} cells ({time.monotonic() - started:.0f}s)")
    return 0


def _evaluation_documents() -> dict[str, np.ndarray]:
    """Each environment's evaluation documents in the sorted order the count matrices use."""
    rows = pd.read_parquet(s14.ENVIRONMENT_ROWS, columns=["environment", "document_id", "role"])
    rows = rows[rows["role"] == s14.ROLE_EVALUATION]
    return {
        str(name): np.asarray(sorted(group["document_id"].astype(str).unique()), dtype=object)
        for name, group in rows.groupby("environment", observed=True)
    }


def _cert_block(environment: Environment) -> Any:
    """The whole certification half, unlabelled -- used only to place the declared grid."""
    return _sample_block(environment, environment.cert_positions)


def run_experiment_b(only: str | None = None) -> int:
    """Experiment B: a fixed total of 250, split between adaptation and certification."""
    started = time.monotonic()
    if not DESIGN_RECORD.is_file():
        raise PhaseError("run --preregister before any experiment")
    frozen = load_frozen()
    assert_frozen(frozen)
    built, _inventory = _environments(only)
    rows: list[dict[str, Any]] = []
    counts: list[tuple[tuple[str, str], np.ndarray]] = []
    for environment in built:
        began = time.monotonic()
        for n_adapt, n_cert in ALLOCATIONS:
            seed = _stable_seed(
                "sgv15-adapt", str(SAMPLE_SEED), environment.name, A_UNCERTAINTY, str(n_adapt)
            )
            adapted = fit_adaptation(environment, A_UNCERTAINTY, n_adapt, seed)
            ranking = _ranking_row(environment, adapted)
            grid = cut_grid(adapted.cert_scores)
            draws = 1 if n_cert == 0 else CERT_DRAWS
            for draw in range(draws):
                sample_seed = _stable_seed(
                    "sgv15-cert",
                    str(SAMPLE_SEED),
                    environment.name,
                    PRIMARY_SAMPLER,
                    str(n_cert),
                    str(draw),
                    "B",
                )
                sample = certification_sample(environment, PRIMARY_SAMPLER, n_cert, sample_seed)
                context = _context(
                    environment,
                    "B",
                    sampler=PRIMARY_SAMPLER,
                    requested_cert=n_cert,
                    draw=draw,
                    acquisition=A_UNCERTAINTY,
                    allocation=f"{n_adapt}+{n_cert}",
                )
                methods = (
                    (M0_FROZEN_SOURCE, C1_ORACLE)
                    if n_cert == 0
                    else (M0_FROZEN_SOURCE, *METHODS[1:], C1_ORACLE)
                )
                _collect(
                    rows,
                    counts,
                    environment,
                    adapted,
                    grid,
                    sample,
                    methods,
                    context,
                    ranking,
                    (environment.name, "B", f"{n_adapt}+{n_cert}", str(draw), "x"),
                )
        print(
            f"  {environment.name}: {len(rows)} cells so far ({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    _append_cells(rows, counts)
    print(f"experiment-b: {len(rows)} cells ({time.monotonic() - started:.0f}s)")
    return 0


def _append_cells(
    rows: list[dict[str, Any]], counts: list[tuple[tuple[str, str], np.ndarray]]
) -> None:
    """Write-once per experiment, appended into one table so the analysis reads a single file."""
    frame = pd.DataFrame(rows)
    if CELL_TABLE.exists():
        frame = pd.concat([pd.read_parquet(CELL_TABLE), frame], ignore_index=True)
        CELL_TABLE.unlink()
    _write_parquet_once(CELL_TABLE, frame)
    if not counts:
        return
    documents = _evaluation_documents()
    widths = [block.shape[1] for _, block in counts]
    block = pd.DataFrame(
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
    )
    if DEPLOYMENT_COUNTS.exists():
        block = pd.concat([pd.read_parquet(DEPLOYMENT_COUNTS), block], ignore_index=True)
        DEPLOYMENT_COUNTS.unlink()
    _write_parquet_once(DEPLOYMENT_COUNTS, block)


# ------------------------------------------------------------------ sample-complexity analysis
#
# Section 25. This phase reads no experiment at all: it is arithmetic on the bounds, and it says
# what a certification sample of each size CAN certify before anything is measured.


def run_samplesize() -> int:
    """What each certification budget can certify, and the smallest rate it can rule out."""
    started = time.monotonic()
    if not DESIGN_RECORD.is_file():
        raise PhaseError("run --preregister first")
    per_test = CERT_DELTA / len(CUT_GRID)
    resolution: dict[str, Any] = {}
    for size in CERT_BUDGETS:
        if size == 0:
            continue
        entry: dict[str, Any] = {"n_cert": size, "per_test_delta": per_test}
        for label, delta in (("alpha", CERT_DELTA), ("alpha_over_grid", per_test)):
            entry[label] = {
                "clopper_pearson_zero_harm": clopper_pearson_upper(0, size, delta),
                "bentkus_zero_harm": risk_upper_bound(0, size, delta),
                "clopper_pearson_one_harm": clopper_pearson_upper(1, size, delta),
                "certifies_at_primary_epsilon_with_zero_harm": bool(
                    clopper_pearson_upper(0, size, delta) <= PRIMARY_EPSILON
                ),
                "max_harmful_still_certifiable": max(
                    [
                        k
                        for k in range(size + 1)
                        if clopper_pearson_upper(k, size, delta) <= PRIMARY_EPSILON
                    ]
                    or [-1]
                ),
            }
        entry["note"] = (
            "the accepted-set size at a cut is what the bound is computed on, and it is at most "
            "n_cert. A deep cut spends the whole sample; a shallow one spends a fraction of it."
        )
        resolution[str(size)] = entry

    # The smallest sample that can certify a genuinely-zero-harm prefix below epsilon.
    smallest = {}
    for label, delta in (("alpha", CERT_DELTA), ("alpha_over_grid", per_test)):
        found = next(
            (n for n in range(1, 4001) if clopper_pearson_upper(0, n, delta) <= PRIMARY_EPSILON),
            None,
        )
        smallest[label] = found

    # Section 24's near miss, as a sample-size question and nothing else.
    near_miss = {
        "environment": "sbb/tesseract_frk",
        "sgv14_realised_harm": 0.1018,
        "epsilon": PRIMARY_EPSILON,
        "question": (
            "how many representative certification labels would be needed to tell 0.1018 apart "
            "from a genuinely safe operating point?"
        ),
        "answer": {
            str(size): {
                "expected_harmful_at_0p1018": round(0.1018 * size, 2),
                "upper_bound_at_that_count": clopper_pearson_upper(
                    round(0.1018 * size), size, per_test
                ),
                "would_certify": bool(
                    clopper_pearson_upper(round(0.1018 * size), size, per_test) <= PRIMARY_EPSILON
                ),
            }
            for size in (*CERT_BUDGETS[1:], 1000, 2000)
        },
        "interpretation": (
            "a procedure that certifies this prefix is making an error, because its true rate is "
            "above the bound. The useful number is the opposite one: how large a sample is "
            "needed before a prefix whose true rate is 0.1018 is reliably REFUSED."
        ),
        "not_a_special_case": (
            "this analysis changes no operational method and no environment is treated "
            "differently because of it"
        ),
    }
    payload = {
        "artifact": "sample_size_analysis",
        "schema_version": f"sgv15-sample_size_analysis-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epsilon": PRIMARY_EPSILON,
        "alpha": CERT_DELTA,
        "grid_points": len(CUT_GRID),
        "per_test_delta": per_test,
        "resolution": resolution,
        "smallest_sample_certifying_zero_harm_below_epsilon": smallest,
        "near_miss_analysis": near_miss,
        "headline": (
            "observing no harmful edit among 25 certification candidates certifies a risk of at "
            f"most {clopper_pearson_upper(0, 25, CERT_DELTA):.4f} at alpha = {CERT_DELTA}, which "
            f"does not clear epsilon = {PRIMARY_EPSILON}. Under the grid-corrected level it is "
            f"{clopper_pearson_upper(0, 25, per_test):.4f}. Finite-sample resolution, not "
            "estimator quality, is the binding constraint at the smallest budgets."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(SAMPLE_SIZE_ANALYSIS, payload)
    print(
        f"samplesize: zero harm certifies below {PRIMARY_EPSILON} from n = "
        f"{smallest['alpha']} (alpha) / {smallest['alpha_over_grid']} (grid-corrected) "
        f"-> {_relative(SAMPLE_SIZE_ANALYSIS)}"
    )
    return 0


# ------------------------------------------------------------------ reading the cells


def load_cells() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not CELL_TABLE.is_file():
        raise PhaseError("run --experiment-a and --experiment-b first")
    if not DESIGN_RECORD.is_file():
        raise PhaseError("the pre-registration is missing")
    return pd.read_parquet(CELL_TABLE), cc_read_json(DESIGN_RECORD)


def primary_slice(
    cells: pd.DataFrame,
    experiment: str,
    method: str,
    sampler: str | None = None,
    n_cert: int | None = None,
    allocation: str | None = None,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    block = cells[
        (cells["experiment"] == experiment)
        & (cells["method"] == method)
        & (np.isclose(cells["epsilon"], epsilon))
    ]
    if sampler is not None:
        block = block[block["sampler"] == sampler]
    if n_cert is not None:
        block = block[block["requested_cert"] == n_cert]
    if allocation is not None:
        block = block[block["allocation"] == allocation]
    return block


def _per_environment(block: pd.DataFrame) -> dict[str, dict[str, float]]:
    """One row per environment, averaged over draws."""
    out: dict[str, dict[str, float]] = {}
    for name, group in block.groupby("environment", observed=True):
        out[str(name)] = {
            "draws": len(group),
            "risk_controlled": float(group["risk_controlled"].mean()),
            "risk_controlled_sd": float(group["risk_controlled"].std(ddof=0)),
            "repair_recall": float(group["repair_recall"].mean()),
            "realized_harm_rate": float(group["realized_harm_rate"].mean()),
            "holds_bound_fraction": float(group["holds_bound"].mean()),
            "coverage": float(group["coverage"].mean()),
            "n_accepted": float(group["n_accepted"].mean()),
            "harmful_accepted": float(group["harmful_accepted"].mean()),
            "beneficial_accepted": float(group["beneficial_accepted"].mean()),
            "cut_feasible_fraction": float(group["cut_feasible"].mean()),
            "certified_upper_bound": float(group["certified_risk_upper_bound"].mean()),
            "cert_harmful_rate": float(group["cert_harmful_rate"].mean()),
            "auroc_safe": float(group["rank__auroc_safe"].mean()),
            "achievable_frontier": float(group[f"frontier__{PRIMARY_KEY}"].mean()),
        }
    return out


def holds_the_bound(row: dict[str, float]) -> bool:
    """Whether an environment's deployment held epsilon, aggregated over draws.

    The design record freezes criterion 1 as "holds the realised harm bound" without saying how
    to aggregate the certification draws, so the convention is fixed here, before any criterion
    is computed, and it is the conservative one: EVERY draw must hold. A deployment gets one
    sample, and a rule that exceeds the tolerance on one draw in twenty has not controlled risk.
    The per-draw fraction is reported alongside it so the strictness is visible rather than
    buried.

    A draw that accepted nothing holds vacuously -- refusing is safe. That is precisely why
    criterion 3 exists: safety bought by refusing everything has to fail a separate gate.
    """
    return float(row["holds_bound_fraction"]) >= 1.0


def _mean(values: list[float]) -> float:
    """Mean over the environments where the quantity is defined.

    Realised harm is undefined, not zero, where a method accepted nothing: there is no accepted
    set to be harmful. Counting those as 0.0 would let a method that refuses everywhere post the
    best safety number in the stage.
    """
    defined = [v for v in values if not math.isnan(v)]
    return float(np.mean(defined)) if defined else float("nan")


def _baseline(cells: pd.DataFrame, epsilon: float = PRIMARY_EPSILON) -> dict[str, dict[str, float]]:
    """SGV14's frozen source cut on the SGV15 adapted ranking, at N_cert = 0."""
    return _per_environment(primary_slice(cells, "A", M0_FROZEN_SOURCE, n_cert=0, epsilon=epsilon))


def run_certification_results() -> int:
    """Experiment A and B, read as endpoints. Written by `--results`."""
    started = time.monotonic()
    cells, design = load_cells()
    frozen = load_frozen()
    baseline = _baseline(cells)
    environments = list(design["environments"])

    budget_curve: dict[str, Any] = {}
    for method in METHODS[1:]:
        per_budget: dict[str, Any] = {}
        for size in CERT_BUDGETS:
            if size == 0:
                continue
            block = primary_slice(cells, "A", method, sampler=PRIMARY_SAMPLER, n_cert=size)
            if block.empty:
                continue
            rows = _per_environment(block)
            per_budget[str(size)] = {
                "per_environment": rows,
                "environments_holding_the_bound": sum(
                    1 for r in rows.values() if holds_the_bound(r)
                ),
                "environments_beating_the_baseline": sum(
                    1
                    for name, r in rows.items()
                    if r["risk_controlled"] > baseline[name]["risk_controlled"]
                ),
                "environments_above_the_coverage_floor": sum(
                    1 for r in rows.values() if r["coverage"] >= frozen.coverage_floor
                ),
                "mean_risk_controlled": _mean([r["risk_controlled"] for r in rows.values()]),
                "mean_realized_harm": _mean([r["realized_harm_rate"] for r in rows.values()]),
                "mean_coverage": _mean([r["coverage"] for r in rows.values()]),
                "environments_with_a_defined_harm_rate": sum(
                    1 for r in rows.values() if not math.isnan(r["realized_harm_rate"])
                ),
            }
        budget_curve[method] = per_budget

    _write_json_once(
        CERTIFICATION_BUDGET_RESULTS,
        {
            "artifact": "certification_budget_results",
            "schema_version": f"sgv15-certification_budget_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "experiment": "A -- adaptation fixed at 250, certification budgets added",
            "sampler": PRIMARY_SAMPLER,
            "epsilon": PRIMARY_EPSILON,
            "baseline": baseline,
            "sgv14_baseline_note": (
                "M0 here is SGV14's frozen source cut applied to the SGV15 adapted ranking, "
                "which is fitted on the fitting half rather than on the whole adaptation "
                "partition. It is the like-for-like baseline for every SGV15 comparison; "
                "SGV14's own published numbers are the separate reference in the report."
            ),
            "by_method": budget_curve,
            "environments": environments,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    allocation_results: dict[str, Any] = {}
    for n_adapt, n_cert in ALLOCATIONS:
        label = f"{n_adapt}+{n_cert}"
        method = M0_FROZEN_SOURCE if n_cert == 0 else M2_CLOPPER
        block = primary_slice(cells, "B", method, allocation=label)
        if block.empty:
            continue
        rows = _per_environment(block)
        allocation_results[label] = {
            "n_adapt": n_adapt,
            "n_cert": n_cert,
            "method": method,
            "per_environment": rows,
            "environments_holding_the_bound": sum(1 for r in rows.values() if holds_the_bound(r)),
            "environments_beating_the_baseline": sum(
                1
                for name, r in rows.items()
                if r["risk_controlled"] > baseline[name]["risk_controlled"]
            ),
            "environments_above_the_coverage_floor": sum(
                1 for r in rows.values() if r["coverage"] >= frozen.coverage_floor
            ),
            "mean_risk_controlled": _mean([r["risk_controlled"] for r in rows.values()]),
            "mean_realized_harm": _mean([r["realized_harm_rate"] for r in rows.values()]),
            "mean_coverage": _mean([r["coverage"] for r in rows.values()]),
            "mean_auroc": _mean([r["auroc_safe"] for r in rows.values()]),
        }
    _write_json_once(
        BUDGET_ALLOCATION_RESULTS,
        {
            "artifact": "budget_allocation_results",
            "schema_version": f"sgv15-budget_allocation_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "experiment": "B -- a fixed total of 250 split between adaptation and certification",
            "total_budget": TOTAL_BUDGET,
            "sampler": PRIMARY_SAMPLER,
            "method": M2_CLOPPER,
            "allocations": allocation_results,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    adaptation_rows = {
        label: {
            "n_adapt": entry["n_adapt"],
            "mean_auroc": entry["mean_auroc"],
            "mean_achievable_frontier": _mean(
                [r["achievable_frontier"] for r in entry["per_environment"].values()]
            ),
        }
        for label, entry in allocation_results.items()
    }
    _write_json_once(
        ADAPTATION_BUDGET_RESULTS,
        {
            "artifact": "adaptation_budget_results",
            "schema_version": f"sgv15-adaptation_budget_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": "what does shrinking the adaptation budget cost the RANKING?",
            "by_allocation": adaptation_rows,
            "note": (
                "the ranking is what adaptation labels buy; the cut is what certification labels "
                "buy. Reading these two tables together is how experiment B's tradeoff is read."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"results: {len(budget_curve)} methods x {len(CERT_BUDGETS) - 1} certification budgets, "
        f"{len(allocation_results)} allocations -> {_relative(CERTIFICATION_BUDGET_RESULTS)}"
    )
    return 0


# ------------------------------------------------------------------ the mechanism pass
#
# Sections 19, 20 and 22 ask four questions the two budget experiments cannot answer from their
# own cells, because each needs a model or a label set that neither experiment fits:
#
#   * what the declared nine-point grid costs, which needs an oracle searched off the grid;
#   * which of ranking and cut is load-bearing, which needs the SOURCE ranking carried forward
#     to a target-certified cut;
#   * whether the gains survive destroying the labels, which needs two permutation refits;
#   * what cross-fitting buys, which needs K models per draw.
#
# All four are computed here, in one pass over the environments, and appended to the same cell
# table so that every downstream analysis reads one file and one schema.

X_SOURCE_RANK_SOURCE_CUT = "x_source_rank__source_cut"
X_ADAPTED_RANK_SOURCE_CUT = "x_adapted_rank__source_cut"
X_SOURCE_RANK_CERTIFIED_CUT = "x_source_rank__certified_cut"
X_ADAPTED_RANK_CERTIFIED_CUT = "x_adapted_rank__certified_cut"
TWO_BY_TWO = (
    X_SOURCE_RANK_SOURCE_CUT,
    X_ADAPTED_RANK_SOURCE_CUT,
    X_SOURCE_RANK_CERTIFIED_CUT,
    X_ADAPTED_RANK_CERTIFIED_CUT,
)
C3_PERMUTED_ADAPTATION = "c3_permuted_adaptation_labels"
C4_PERMUTED_CERTIFICATION = "c4_permuted_certification_labels"
C1_ORACLE_FINE = "c1_oracle_fine_grid"
MECHANISM_BUDGET = 250
M4_DRAWS = 3
MECHANISM_METHODS = (
    *TWO_BY_TWO,
    C3_PERMUTED_ADAPTATION,
    C4_PERMUTED_CERTIFICATION,
    C1_ORACLE_FINE,
    M4_CROSS_FITTED,
)


def fine_grid(scores: np.ndarray) -> np.ndarray:
    """A dense acceptance-depth grid, used ONLY to price the declared nine-point one.

    Never a deployment family: nothing selected on this grid may be certified, because the union
    bound over two hundred points is what the nine-point grid exists to avoid paying.
    """
    if scores.size == 0:
        return np.zeros(0, dtype=float)
    depths = np.linspace(1.0 / FINE_GRID_POINTS, 1.0, FINE_GRID_POINTS)
    return np.unique(np.quantile(scores, [1.0 - d for d in depths]))


def _depth_of(scores: np.ndarray, tau: float) -> float:
    """The acceptance depth a threshold represents on a score vector."""
    if scores.size == 0 or not np.isfinite(tau):
        return 0.0
    return float(np.mean(scores >= tau))


def crossfit_selection(environment: Environment, rule: str, n_total: int, seed: int) -> np.ndarray:
    """Which target rows a cross-fitted draw buys, from the whole pool rather than one half."""
    setup = environment.setup
    view = setup.view
    pool = np.concatenate([environment.fit_positions, environment.cert_positions])
    allowed = np.zeros(view.size, dtype=bool)
    allowed[pool] = True
    order = acquisition_order(view, rule, setup.base, seed)
    return np.asarray(order[allowed[order]][:n_total], dtype=int)


def cross_fitted_counts(
    environment: Environment, rule: str, n_total: int, seed: int
) -> tuple[Adapted, dict[float, tuple[int, int]], dict[str, Any]]:
    """K-fold cross-fitted certification counts: every target label both fits and certifies,
    never both for the same row.

    The sample-split design spends half its labels on the ranking and half on the risk estimate.
    Cross-fitting spends all of them twice by rotating: fold `k` is certified by a model that
    never saw it. The complication is that `K` models put scores on `K` incomparable scales, so
    the pooling is done in ACCEPTANCE DEPTH -- the declared grid is already a depth grid, which is
    what makes this possible at all -- and each fold contributes the harm indicators of the rows
    its own model would have accepted at that depth.

    The `K + 1` refits depend on the environment, the acquisition and the seed, and on NOTHING
    else. In particular they do not depend on epsilon, so they are done once here and every
    tolerance is priced from the returned counts by `cross_fitted_cut_at`.

    Registered SECONDARY. The deployed model is the one fitted on all `n_total` labels while the
    risk estimate is pooled from `K` slightly different models, and that mismatch is exactly why
    sample splitting is the primary.
    """
    setup = environment.setup
    view = setup.view
    chosen = crossfit_selection(environment, rule, n_total, seed)
    if chosen.size == 0:
        return fit_adaptation(environment, rule, 0, seed), {}, {"folds": 0, "n_total": 0}

    documents = np.asarray([str(d) for d in view.documents[chosen].tolist()], dtype=object)
    names = sorted(set(documents.tolist()))
    fold_of = {name: index % min(CROSS_FIT_FOLDS, len(names)) for index, name in enumerate(names)}
    assignment = np.asarray([fold_of[name] for name in documents.tolist()], dtype=int)
    harmful = setup.outcomes["harmful"][chosen]

    folds = sorted(set(assignment.tolist()))
    per_depth_harm: dict[float, list[int]] = {q: [] for q in CUT_GRID}
    per_depth_total: dict[float, list[int]] = {q: [] for q in CUT_GRID}
    for fold in folds:
        held = assignment == fold
        rest = np.flatnonzero(~held)
        if rest.size == 0 or int(held.sum()) == 0:
            continue
        budget = purchase(setup, chosen[rest], int(rest.size))
        model = fit_joint_arm(setup, budget, load_frozen().target_weight, False, s13.A4_JOINT)
        reference = np.asarray(model.score(setup, _cert_block(environment)), dtype=float)
        held_scores = np.asarray(
            model.score(setup, _sample_block(environment, chosen[held])), dtype=float
        )
        for q in CUT_GRID:
            tau = float(np.quantile(reference, 1.0 - q)) if reference.size else float("inf")
            accepted = held_scores >= tau
            per_depth_total[q].append(int(accepted.sum()))
            per_depth_harm[q].append(int((accepted & harmful[held]).sum()))

    pooled = {q: (int(sum(per_depth_total[q])), int(sum(per_depth_harm[q]))) for q in CUT_GRID}
    final = fit_adaptation(environment, rule, int(chosen.size), seed, restrict=False)
    return final, pooled, {"folds": len(folds), "n_total": int(chosen.size)}


def cross_fitted_cut_at(
    final: Adapted, pooled: dict[float, tuple[int, int]], epsilon: float
) -> tuple[float, dict[str, Any]]:
    """Price one tolerance from the pooled cross-fitted counts. Arithmetic only, no refit."""
    per_test = CERT_DELTA / len(CUT_GRID)
    deepest: float | None = None
    by_depth: dict[str, Any] = {}
    for q in CUT_GRID:
        total, harm = pooled.get(q, (0, 0))
        bound = clopper_pearson_upper(harm, total, per_test) if total else 1.0
        by_depth[f"{q:g}"] = {
            "n_accepted": total,
            "n_harmful": harm,
            "upper_bound": float(bound),
            "feasible": bool(total > 0 and bound <= epsilon),
        }
        if total > 0 and bound <= epsilon and (deepest is None or q > deepest):
            deepest = q
    detail: dict[str, Any] = {"by_depth": by_depth, "selected_depth": deepest}
    if deepest is None or final.cert_scores.size == 0:
        return float("inf"), detail
    return float(np.quantile(final.cert_scores, 1.0 - deepest)), detail


def _mechanism_row(
    environment: Environment,
    adapted: Adapted,
    tau: float,
    method: str,
    epsilon: float,
    context: dict[str, Any],
    sample: np.ndarray,
    feasible: bool,
    bound: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """One mechanism cell, written into exactly the schema the two experiments produced."""
    setup = environment.setup
    view = setup.view
    row: dict[str, Any] = dict(context)
    row["method"] = method
    row["epsilon"] = float(epsilon)
    row["n_cert"] = int(sample.size)
    row["cert_documents"] = len({str(n) for n in view.documents[sample].tolist()})
    harmful = setup.outcomes["harmful"][sample] if sample.size else np.zeros(0, dtype=bool)
    row["cert_harmful"] = int(harmful.sum())
    row["cert_harmful_rate"] = float(harmful.mean()) if sample.size else float("nan")
    row["certified_risk_upper_bound"] = float(bound)
    row["certified_observed_risk"] = float("nan")
    row["cut_feasible"] = bool(feasible)
    row["cut_accepted_in_sample"] = 0
    point = deploy(environment, adapted.eval_scores, tau, epsilon)
    accepted = point.pop("accepted")
    row.update(point)
    row["n_adapt"] = int(adapted.n_adapt)
    row["adapt_documents"] = int(adapted.documents)
    row["adapt_harmful"] = int(adapted.harmful)
    row["adapt_beneficial"] = int(adapted.beneficial)
    row["adapt_neutral"] = int(adapted.neutral)
    row["n_total"] = int(adapted.n_adapt + sample.size)
    counts = None
    if epsilon == PRIMARY_EPSILON:
        block = setup.evaluation
        counts = per_document_counts(block, accepted, sorted({*block.documents.tolist()}))
    return row, counts


def _push(
    rows: list[dict[str, Any]],
    counts: list[tuple[tuple[str, str], np.ndarray]],
    environment: Environment,
    key: tuple[str, ...],
    method: str,
    ranking: dict[str, float],
    row: dict[str, Any],
    block: dict[str, np.ndarray] | None,
) -> None:
    row.update(ranking)
    row["cell"] = "|".join((*key, method))
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


def run_mechanism(only: str | None = None) -> int:
    """The 2x2, the fine-grid oracle, the two permutation controls and the cross-fitted arm."""
    started = time.monotonic()
    if not CELL_TABLE.is_file():
        raise PhaseError("run --experiment-a and --experiment-b first")
    frozen = load_frozen()
    assert_frozen(frozen)
    built, _inventory = _environments(only)
    rows: list[dict[str, Any]] = []
    counts: list[tuple[tuple[str, str], np.ndarray]] = []
    diagnostics: dict[str, Any] = {}

    for environment in built:
        began = time.monotonic()
        seed = _stable_seed("sgv15-adapt", str(SAMPLE_SEED), environment.name, A_UNCERTAINTY)
        adapted = fit_adaptation(environment, A_UNCERTAINTY, ADAPT_BUDGET, seed)
        source = fit_adaptation(environment, A_UNCERTAINTY, 0, seed)
        permuted = fit_adaptation(
            environment,
            A_UNCERTAINTY,
            ADAPT_BUDGET,
            seed,
            permute_seed=_stable_seed("sgv15-permute-adapt", str(SAMPLE_SEED), environment.name),
        )
        adapted_grid = cut_grid(adapted.cert_scores)
        source_grid = cut_grid(source.cert_scores)
        permuted_grid = cut_grid(permuted.cert_scores)
        adapted_rank = _ranking_row(environment, adapted)
        source_rank = _ranking_row(environment, source)
        permuted_rank = _ranking_row(environment, permuted)

        # The fine-grid oracle: what the SAME ranking could have reached had the declared family
        # not been nine points wide. Analysis only, and it is priced against the on-grid oracle.
        for epsilon in EPSILONS:
            decision = oracle_decision(
                environment, adapted.eval_scores, fine_grid(adapted.cert_scores), epsilon
            )
            context = _context(
                environment,
                "C",
                sampler=PRIMARY_SAMPLER,
                requested_cert=0,
                draw=0,
                acquisition=A_UNCERTAINTY,
                allocation=f"{ADAPT_BUDGET}+0",
            )
            row, block = _mechanism_row(
                environment,
                adapted,
                refusal_tau(decision),
                C1_ORACLE_FINE,
                epsilon,
                context,
                np.zeros(0, dtype=int),
                bool(decision.feasible),
                float("nan"),
            )
            _push(
                rows,
                counts,
                environment,
                (environment.name, "C", "fine", "0", "0"),
                C1_ORACLE_FINE,
                adapted_rank,
                row,
                block,
            )

        for size in CERT_BUDGETS:
            if size == 0:
                continue
            for draw in range(CERT_DRAWS):
                sample_seed = _stable_seed(
                    "sgv15-cert",
                    str(SAMPLE_SEED),
                    environment.name,
                    PRIMARY_SAMPLER,
                    str(size),
                    str(draw),
                )
                sample = certification_sample(environment, PRIMARY_SAMPLER, size, sample_seed)
                key = (environment.name, "C", PRIMARY_SAMPLER, str(size), str(draw))
                context = _context(
                    environment,
                    "C",
                    sampler=PRIMARY_SAMPLER,
                    requested_cert=size,
                    draw=draw,
                    acquisition=A_UNCERTAINTY,
                    allocation=f"{ADAPT_BUDGET}+{size}",
                )
                harmful = environment.setup.outcomes["harmful"][sample]
                index = _cert_index(environment, sample)
                shuffled = harmful[
                    np.random.default_rng(
                        _stable_seed(
                            "sgv15-permute-cert",
                            str(SAMPLE_SEED),
                            environment.name,
                            str(size),
                            str(draw),
                        )
                    ).permutation(harmful.size)
                ]
                # The 2x2 of section 20, plus the two permutation controls, all at one epsilon
                # sweep. Each pairing states which ranking and which cut produced it.
                for method, model, grid, rank, labels in (
                    (X_SOURCE_RANK_SOURCE_CUT, source, source_grid, source_rank, None),
                    (X_ADAPTED_RANK_SOURCE_CUT, adapted, adapted_grid, adapted_rank, None),
                    (X_SOURCE_RANK_CERTIFIED_CUT, source, source_grid, source_rank, harmful),
                    (X_ADAPTED_RANK_CERTIFIED_CUT, adapted, adapted_grid, adapted_rank, harmful),
                    (C3_PERMUTED_ADAPTATION, permuted, permuted_grid, permuted_rank, harmful),
                    (C4_PERMUTED_CERTIFICATION, adapted, adapted_grid, adapted_rank, shuffled),
                ):
                    for epsilon in EPSILONS:
                        if labels is None:
                            tau = frozen_source_tau(environment, model.model, epsilon)
                            feasible, bound = bool(np.isfinite(tau)), float("nan")
                        else:
                            decision = place_cut(
                                M2_CLOPPER,
                                grid,
                                model.cert_scores[index],
                                labels,
                                epsilon,
                                environment.source_prior,
                            )
                            tau = refusal_tau(decision)
                            feasible = bool(decision.feasible)
                            bound = float(decision.risk_upper_bound)
                        row, block = _mechanism_row(
                            environment,
                            model,
                            tau,
                            method,
                            epsilon,
                            context,
                            sample,
                            feasible,
                            bound,
                        )
                        _push(rows, counts, environment, key, method, rank, row, block)

        print(
            f"  {environment.name}: {len(rows)} mechanism cells ({time.monotonic() - began:.0f}s)",
            flush=True,
        )
        diagnostics[environment.name] = {
            "adapted_auroc": adapted_rank["rank__auroc_safe"],
            "source_auroc": source_rank["rank__auroc_safe"],
            "permuted_auroc": permuted_rank["rank__auroc_safe"],
        }

    _append_cells(rows, counts)
    print(f"mechanism: {len(rows)} cells ({time.monotonic() - started:.0f}s)")
    return 0


def run_crossfit(only: str | None = None) -> int:
    """The registered secondary cross-fitted arm, at the mechanism budget only."""
    started = time.monotonic()
    if not CELL_TABLE.is_file():
        raise PhaseError("run the experiments first")
    built, _inventory = _environments(only)
    rows: list[dict[str, Any]] = []
    counts: list[tuple[tuple[str, str], np.ndarray]] = []
    diagnostics: dict[str, Any] = {}
    for environment in built:
        began = time.monotonic()
        per_draw = []
        # `q2_uncertainty` is a deterministic function of the frozen score and the transported
        # threshold, so two draws buy the same rows and would produce identical refits. That is
        # VERIFIED here rather than assumed -- the selections are compared and the K + 1 fits are
        # reused only when they are literally equal -- and the count of distinct selections is
        # reported, so a stochastic acquisition would still get its full draws.
        fitted: dict[bytes, tuple[Adapted, dict[float, tuple[int, int]], dict[str, Any]]] = {}
        selections: list[str] = []
        for draw in range(M4_DRAWS):
            seed = _stable_seed(
                "sgv15-crossfit",
                str(SAMPLE_SEED),
                environment.name,
                str(MECHANISM_BUDGET),
                str(draw),
            )
            chosen = crossfit_selection(environment, A_UNCERTAINTY, MECHANISM_BUDGET, seed)
            key = np.sort(chosen).tobytes()
            selections.append(canonical_hash({"rows": sorted(int(c) for c in chosen.tolist())}))
            if key not in fitted:
                fitted[key] = cross_fitted_counts(
                    environment, A_UNCERTAINTY, MECHANISM_BUDGET, seed
                )
            model, pooled, shared = fitted[key]
            ranking = _ranking_row(environment, model)
            for epsilon in EPSILONS:
                tau, priced = cross_fitted_cut_at(model, pooled, epsilon)
                detail = {**shared, **priced, "epsilon": epsilon}
                context = _context(
                    environment,
                    "C",
                    sampler=PRIMARY_SAMPLER,
                    requested_cert=MECHANISM_BUDGET,
                    draw=draw,
                    acquisition=A_UNCERTAINTY,
                    allocation=f"crossfit-{MECHANISM_BUDGET}",
                )
                row, block = _mechanism_row(
                    environment,
                    model,
                    tau,
                    M4_CROSS_FITTED,
                    epsilon,
                    context,
                    np.zeros(0, dtype=int),
                    detail.get("selected_depth") is not None,
                    float("nan"),
                )
                _push(
                    rows,
                    counts,
                    environment,
                    (environment.name, "C", "crossfit", str(MECHANISM_BUDGET), str(draw)),
                    M4_CROSS_FITTED,
                    ranking,
                    row,
                    block,
                )
                if epsilon == PRIMARY_EPSILON:
                    per_draw.append(detail)
        diagnostics[environment.name] = {
            "per_draw": per_draw,
            "draws": M4_DRAWS,
            "distinct_label_selections": len(fitted),
            "selection_hashes": selections,
            "acquisition_is_deterministic": len(fitted) == 1,
        }
        print(
            f"  {environment.name}: cross-fit done, {len(fitted)} distinct selection(s) over "
            f"{M4_DRAWS} draws ({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    # The endpoint summary lives here rather than in the primary results, because M4 is secondary
    # and must not enter a selection. Reporting it from its own artifact keeps it traceable
    # without letting it compete.
    frame = pd.DataFrame(rows)
    primary = frame[np.isclose(frame["epsilon"], PRIMARY_EPSILON)]
    endpoint = {
        str(name): {
            "draws": len(group),
            "risk_controlled": float(group["risk_controlled"].mean()),
            "realized_harm_rate": _mean(group["realized_harm_rate"].tolist()),
            "coverage": float(group["coverage"].mean()),
            "cut_feasible_fraction": float(group["cut_feasible"].mean()),
            "holds_bound_fraction": float(group["holds_bound"].mean()),
            "n_adapt": float(group["n_adapt"].mean()),
        }
        for name, group in primary.groupby("environment", observed=True)
    }
    _append_cells(rows, counts)
    _write_json_once(
        OUT / "cross_fitted_diagnostics.json",
        {
            "artifact": "cross_fitted_diagnostics",
            "schema_version": f"sgv15-cross_fitted_diagnostics-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "SECONDARY -- registered in the design record, reported, never selected",
            "n_total": MECHANISM_BUDGET,
            "folds": CROSS_FIT_FOLDS,
            "draws": M4_DRAWS,
            "draws_note": (
                f"{M4_DRAWS} draws rather than {CERT_DRAWS}: each draw costs "
                f"{CROSS_FIT_FOLDS + 1} refits, and the arm is secondary. The reduced count is "
                "stated wherever the arm is reported and it may not carry a headline."
            ),
            "by_environment": diagnostics,
            "endpoint_by_environment": endpoint,
            "mean_risk_controlled": _mean([r["risk_controlled"] for r in endpoint.values()]),
            "mean_realized_harm": _mean([r["realized_harm_rate"] for r in endpoint.values()]),
            "mean_coverage": _mean([r["coverage"] for r in endpoint.values()]),
            "environments_holding_the_bound": sum(
                1 for r in endpoint.values() if r["holds_bound_fraction"] >= 1.0
            ),
            "epsilon": PRIMARY_EPSILON,
            "may_not_be_selected": (
                "M4 is absent from METHODS and therefore from the selection table by "
                "construction; the leakage suite asserts the selected method is one of METHODS."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"crossfit: {len(rows)} cells ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ reading the per-document counts

CELL_FIELDS = ("environment", "experiment", "f2", "f3", "f4", "method")


def indexed_counts() -> pd.DataFrame:
    """The per-document deployment counts with the cell key split back into its fields.

    The key is positional and its meaning depends on the experiment: A and the mechanism pass
    write `sampler|budget|draw`, B writes `allocation|draw|x`. Naming the columns f2..f4 keeps
    that honest rather than pretending one schema fits both.
    """
    counts = pd.read_parquet(DEPLOYMENT_COUNTS)
    parts = counts["cell"].astype(str).str.split("|", expand=True)
    parts.columns = list(CELL_FIELDS)
    return pd.concat([counts.reset_index(drop=True), parts], axis=1)


def cell_block(indexed: pd.DataFrame, draw_field: str = "f4", **filters: str) -> Any:
    """One cell's per-document counts, one matrix row per draw, in sorted document order."""
    block = indexed
    for column, value in filters.items():
        block = block[block[column] == value]
    if block.empty:
        raise PhaseError(f"no deployment counts for {filters}")
    documents = tuple(sorted({str(name) for name in block["document_id"].tolist()}))
    draws = sorted({str(value) for value in block[draw_field].tolist()})
    position = {name: index for index, name in enumerate(documents)}
    row_of = {draw: index for index, draw in enumerate(draws)}
    shape = (len(draws), len(documents))
    names = ("accepted", "harmful_accepted", "beneficial_accepted", "beneficial_total")
    matrices = {name: np.zeros(shape, dtype=float) for name in names}
    rows = np.array([row_of[str(v)] for v in block[draw_field].tolist()], dtype=int)
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


def _evaluation_prevalence() -> dict[str, dict[str, float]]:
    """Each environment's evaluation-block harm and benefit prevalence, from the inventory."""
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    out = {}
    for record in inventory["environments"]:
        size = float(record["evaluation_rows"])
        out[record["environment"]] = {
            "evaluation_rows": size,
            "harmful_prevalence": float(record["evaluation_harmful"]) / size if size else 0.0,
            "beneficial_prevalence": float(record["evaluation_beneficial"]) / size if size else 0.0,
        }
    return out


# ------------------------------------------------------------------ section 21: the samplers


def run_sampling() -> int:
    """Does a representative certification sample estimate deployment risk better than a
    targeted one?"""
    started = time.monotonic()
    cells, design = load_cells()
    prevalence = _evaluation_prevalence()
    baseline = _baseline(cells)
    frozen = load_frozen()
    by_sampler: dict[str, Any] = {}

    for sampler in (*SAMPLERS, *CONTROL_SAMPLERS):
        per_budget: dict[str, Any] = {}
        for size in CERT_BUDGETS:
            if size == 0:
                continue
            block = primary_slice(cells, "A", M2_CLOPPER, sampler=sampler, n_cert=size)
            if block.empty:
                continue
            rows: dict[str, Any] = {}
            for name, group in block.groupby("environment", observed=True):
                realised = group["realized_harm_rate"]
                observed = group["certified_observed_risk"]
                paired = group.dropna(subset=["realized_harm_rate", "certified_observed_risk"])
                mean_rate = float(group["cert_harmful_rate"].mean())
                # A design effect measured from the draws themselves: how much wider the
                # sample-to-sample spread of the estimated prevalence is than an independent
                # draw of the same size would give. Above one means the document clustering is
                # costing real information the row count does not show.
                spread = float(group["cert_harmful_rate"].var(ddof=1))
                mean_n = float(group["n_cert"].mean())
                binomial = mean_rate * (1.0 - mean_rate) / mean_n if mean_n > 0 else float("nan")
                deff = spread / binomial if binomial and binomial > 0 else float("nan")
                rows[str(name)] = {
                    "draws": len(group),
                    "certification_prevalence": mean_rate,
                    "evaluation_prevalence": prevalence[str(name)]["harmful_prevalence"],
                    "prevalence_absolute_error": abs(
                        mean_rate - prevalence[str(name)]["harmful_prevalence"]
                    ),
                    "mean_certification_documents": float(group["cert_documents"].mean()),
                    "mean_certification_rows": mean_n,
                    "design_effect": deff,
                    "effective_sample_size": (
                        mean_n / deff
                        if deff and deff > 0 and not math.isnan(deff)
                        else float("nan")
                    ),
                    "tau": float(group["tau"].replace([np.inf, -np.inf], np.nan).mean()),
                    "realized_harm_rate": float(realised.mean()),
                    "certified_observed_risk": float(observed.mean()),
                    "risk_estimation_absolute_error": (
                        float(
                            (paired["certified_observed_risk"] - paired["realized_harm_rate"])
                            .abs()
                            .mean()
                        )
                        if not paired.empty
                        else float("nan")
                    ),
                    "coverage": float(group["coverage"].mean()),
                    "repair_recall": float(group["repair_recall"].mean()),
                    "risk_controlled": float(group["risk_controlled"].mean()),
                    "holds_bound_fraction": float(group["holds_bound"].mean()),
                    "cut_feasible_fraction": float(group["cut_feasible"].mean()),
                }
            per_budget[str(size)] = {
                "per_environment": rows,
                "mean_risk_controlled": _mean([r["risk_controlled"] for r in rows.values()]),
                "mean_prevalence_absolute_error": _mean(
                    [r["prevalence_absolute_error"] for r in rows.values()]
                ),
                "mean_risk_estimation_absolute_error": _mean(
                    [r["risk_estimation_absolute_error"] for r in rows.values()]
                ),
                "mean_design_effect": _mean([r["design_effect"] for r in rows.values()]),
                "environments_holding_the_bound": sum(
                    1 for r in rows.values() if r["holds_bound_fraction"] >= 1.0
                ),
                "environments_above_the_coverage_floor": sum(
                    1 for r in rows.values() if r["coverage"] >= frozen.coverage_floor
                ),
            }
        by_sampler[sampler] = per_budget

    comparison = {}
    for size in CERT_BUDGETS:
        if size == 0 or str(size) not in by_sampler[PRIMARY_SAMPLER]:
            continue
        entry = {}
        for sampler in (*SAMPLERS, *CONTROL_SAMPLERS):
            record = by_sampler[sampler].get(str(size))
            if record is None:
                continue
            entry[sampler] = {
                "mean_risk_controlled": record["mean_risk_controlled"],
                "mean_prevalence_absolute_error": record["mean_prevalence_absolute_error"],
                "mean_risk_estimation_absolute_error": record[
                    "mean_risk_estimation_absolute_error"
                ],
                "environments_holding_the_bound": record["environments_holding_the_bound"],
            }
        primary = entry.get(PRIMARY_SAMPLER, {})
        diagnostic = entry.get(CS0_UNCERTAINTY, {})
        comparison[str(size)] = {
            "by_sampler": entry,
            "representative_beats_uncertainty_on_endpoint": bool(
                primary.get("mean_risk_controlled", float("nan"))
                > diagnostic.get("mean_risk_controlled", float("nan"))
            ),
            "representative_estimates_prevalence_better": bool(
                primary.get("mean_prevalence_absolute_error", float("inf"))
                < diagnostic.get("mean_prevalence_absolute_error", float("inf"))
            ),
        }

    _write_json_once(
        CERTIFICATION_SAMPLING,
        {
            "artifact": "certification_sampling",
            "schema_version": f"sgv15-certification_sampling-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": design["sub_claims"]["P1d"],
            "method": M2_CLOPPER,
            "epsilon": PRIMARY_EPSILON,
            "primary_sampler": PRIMARY_SAMPLER,
            "diagnostic_negative_sampler": CS0_UNCERTAINTY,
            "baseline": baseline,
            "by_sampler": by_sampler,
            "comparison": comparison,
            "design_effect_note": (
                "measured from the spread of the estimated prevalence across draws against the "
                "independent-draw variance at the same size. It is a property of the sample, not "
                "of the bound, and the bound does not use it -- which is the point of section 10."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"sampling: {len(by_sampler)} samplers x {len(CERT_BUDGETS) - 1} budgets "
        f"-> {_relative(CERTIFICATION_SAMPLING)}"
    )
    return 0


# ------------------------------------------------------------------ sections 19 and 20


def _cut_row(cells: pd.DataFrame, environment: str, method: str, **kwargs: Any) -> dict[str, float]:
    """One environment's mean deployment under one method, or an explicit absence."""
    block = primary_slice(cells, kwargs.pop("experiment", "A"), method, **kwargs)
    block = block[block["environment"] == environment]
    if block.empty:
        return {k: float("nan") for k in ("risk_controlled", "realized_harm_rate", "coverage")}
    return {
        "risk_controlled": float(block["risk_controlled"].mean()),
        "realized_harm_rate": float(block["realized_harm_rate"].mean()),
        "coverage": float(block["coverage"].mean()),
        "repair_recall": float(block["repair_recall"].mean()),
        "tau": float(block["tau"].replace([np.inf, -np.inf], np.nan).mean()),
        "acceptance_depth": float(block["coverage"].mean()),
        "cut_feasible_fraction": float(block["cut_feasible"].mean()),
        "holds_bound_fraction": float(block["holds_bound"].mean()),
        "draws": len(block),
    }


def run_decompose() -> int:
    """C0-C3, the price of the declared grid, and the adaptation-versus-certification 2x2."""
    started = time.monotonic()
    cells, design = load_cells()
    environments = list(design["environments"])
    reference = MECHANISM_BUDGET

    per_environment: dict[str, Any] = {}
    for name in environments:
        c0 = _cut_row(cells, name, M0_FROZEN_SOURCE, n_cert=0)
        grid_oracle = _cut_row(cells, name, C1_ORACLE, sampler=PRIMARY_SAMPLER, n_cert=reference)
        fine_oracle = _cut_row(cells, name, C1_ORACLE_FINE, experiment="C", n_cert=0)
        c2 = _cut_row(cells, name, M1_EMPIRICAL, sampler=PRIMARY_SAMPLER, n_cert=reference)
        c3 = _cut_row(cells, name, M2_CLOPPER, sampler=PRIMARY_SAMPLER, n_cert=reference)
        per_environment[name] = {
            "C0_frozen_source_cut": c0,
            "C1_oracle_evaluation_cut_on_grid": grid_oracle,
            "C1f_oracle_evaluation_cut_fine_grid": fine_oracle,
            "C2_empirical_target_certification_cut": c2,
            "C3_conservative_certified_cut": c3,
            "gaps": {
                "source_cut_transfer_gap": grid_oracle["risk_controlled"] - c0["risk_controlled"],
                "certification_estimation_gap": (
                    grid_oracle["risk_controlled"] - c2["risk_controlled"]
                ),
                "conservatism_cost": c2["risk_controlled"] - c3["risk_controlled"],
                "grid_resolution_gap": (
                    fine_oracle["risk_controlled"] - grid_oracle["risk_controlled"]
                ),
                "total_gap_from_oracle": fine_oracle["risk_controlled"] - c3["risk_controlled"],
            },
            "harm_gaps": {
                "C0_realized_harm": c0["realized_harm_rate"],
                "C3_realized_harm": c3["realized_harm_rate"],
                "harm_reduction": c0["realized_harm_rate"] - c3["realized_harm_rate"],
            },
        }

    def gap(name: str) -> list[float]:
        return [per_environment[e]["gaps"][name] for e in environments]

    _write_json_once(
        CUT_DECOMPOSITION,
        {
            "artifact": "cut_decomposition",
            "schema_version": f"sgv15-cut_decomposition-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epsilon": PRIMARY_EPSILON,
            "endpoint": "risk-controlled repair recall",
            "reference_certification_budget": reference,
            "sampler": PRIMARY_SAMPLER,
            "definitions": {
                "C0": "SGV14's frozen source cut, transported unchanged",
                "C1": "the deepest DECLARED cut whose realised harm holds on the evaluation block",
                "C1f": f"the same search over a {FINE_GRID_POINTS}-point depth grid",
                "C2": "the deepest declared cut whose OBSERVED certification-sample risk holds",
                "C3": "the deepest declared cut whose exact upper BOUND holds",
                "oracle_status": (
                    "C1 and C1f are ANALYSIS ONLY. Neither reaches a selection path; the leakage "
                    "suite asserts it."
                ),
            },
            "per_environment": per_environment,
            "means": {
                "source_cut_transfer_gap": _mean(gap("source_cut_transfer_gap")),
                "certification_estimation_gap": _mean(gap("certification_estimation_gap")),
                "conservatism_cost": _mean(gap("conservatism_cost")),
                "grid_resolution_gap": _mean(gap("grid_resolution_gap")),
                "total_gap_from_oracle": _mean(gap("total_gap_from_oracle")),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )

    _write_json_once(
        OUT / "grid_resolution_analysis.json",
        {
            "artifact": "grid_resolution_analysis",
            "schema_version": f"sgv15-grid_resolution_analysis-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": "what does restricting the cut family to nine declared depths cost?",
            "declared_grid": list(CUT_GRID),
            "fine_grid_points": FINE_GRID_POINTS,
            "per_environment": {
                name: {
                    "on_grid_oracle": per_environment[name]["C1_oracle_evaluation_cut_on_grid"],
                    "fine_grid_oracle": per_environment[name][
                        "C1f_oracle_evaluation_cut_fine_grid"
                    ],
                    "grid_resolution_gap": per_environment[name]["gaps"]["grid_resolution_gap"],
                }
                for name in environments
            },
            "mean_grid_resolution_gap": _mean(gap("grid_resolution_gap")),
            "environments_where_the_grid_costs_recall": sum(
                1 for e in environments if per_environment[e]["gaps"]["grid_resolution_gap"] > 0.0
            ),
            "interpretation": (
                "the declared grid is not free. It is the price of a union bound over nine points "
                "instead of two hundred, and the certified methods are judged against a ceiling "
                "that the grid itself lowers. Widening the grid to recover this would widen the "
                "multiplicity correction that makes certification possible at these sample sizes, "
                "so the grid is NOT changed -- the cost is reported."
            ),
            "grid_not_changed": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    matrix: dict[str, Any] = {}
    for name in environments:
        entry = {}
        for method in TWO_BY_TWO:
            entry[method] = _cut_row(
                cells, name, method, experiment="C", sampler=PRIMARY_SAMPLER, n_cert=reference
            )
        entry["effects"] = {
            "adaptation_only": (
                entry[X_ADAPTED_RANK_SOURCE_CUT]["risk_controlled"]
                - entry[X_SOURCE_RANK_SOURCE_CUT]["risk_controlled"]
            ),
            "certification_only": (
                entry[X_SOURCE_RANK_CERTIFIED_CUT]["risk_controlled"]
                - entry[X_SOURCE_RANK_SOURCE_CUT]["risk_controlled"]
            ),
            "both": (
                entry[X_ADAPTED_RANK_CERTIFIED_CUT]["risk_controlled"]
                - entry[X_SOURCE_RANK_SOURCE_CUT]["risk_controlled"]
            ),
            "interaction": (
                entry[X_ADAPTED_RANK_CERTIFIED_CUT]["risk_controlled"]
                - entry[X_ADAPTED_RANK_SOURCE_CUT]["risk_controlled"]
                - entry[X_SOURCE_RANK_CERTIFIED_CUT]["risk_controlled"]
                + entry[X_SOURCE_RANK_SOURCE_CUT]["risk_controlled"]
            ),
        }
        matrix[name] = entry

    def effect(name: str) -> list[float]:
        return [matrix[e]["effects"][name] for e in environments]

    adaptation_mean = _mean(effect("adaptation_only"))
    certification_mean = _mean(effect("certification_only"))
    _write_json_once(
        ADAPTATION_CERTIFICATION_DECOMPOSITION,
        {
            "artifact": "adaptation_certification_decomposition",
            "schema_version": (f"sgv15-adaptation_certification_decomposition-v{SCHEMA_VERSION}"),
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": "which component is load-bearing: the ranking, the cut, or both?",
            "epsilon": PRIMARY_EPSILON,
            "reference_certification_budget": reference,
            "design": {
                X_SOURCE_RANK_SOURCE_CUT: "source ranking, source cut -- SGV14's starting point",
                X_ADAPTED_RANK_SOURCE_CUT: "target-adapted ranking, source cut -- SGV14's finding",
                X_SOURCE_RANK_CERTIFIED_CUT: "source ranking, target-certified cut",
                X_ADAPTED_RANK_CERTIFIED_CUT: "target-adapted ranking, target-certified cut",
                "held_fixed": (
                    "the certification sample, its seed, the declared grid, alpha and epsilon are "
                    "identical across the four corners; only the ranking and the cut rule move"
                ),
            },
            "per_environment": matrix,
            "means": {
                "adaptation_only": adaptation_mean,
                "certification_only": certification_mean,
                "both": _mean(effect("both")),
                "interaction": _mean(effect("interaction")),
            },
            "environments_where_certification_helps": sum(
                1 for e in environments if matrix[e]["effects"]["certification_only"] > 0.0
            ),
            "environments_where_adaptation_helps": sum(
                1 for e in environments if matrix[e]["effects"]["adaptation_only"] > 0.0
            ),
            "load_bearing_component": (
                "certification"
                if certification_mean > adaptation_mean
                else "adaptation"
                if adaptation_mean > certification_mean
                else "neither separates"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )

    _write_json_once(
        CUT_RESULTS,
        {
            "artifact": "cut_results",
            "schema_version": f"sgv15-cut_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epsilon": PRIMARY_EPSILON,
            "per_environment_by_method": {
                name: {
                    method: _cut_row(
                        cells,
                        name,
                        method,
                        sampler=PRIMARY_SAMPLER,
                        n_cert=(0 if method == M0_FROZEN_SOURCE else reference),
                    )
                    for method in (*METHODS, C1_ORACLE)
                }
                for name in environments
            },
            "reference_certification_budget": reference,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"decompose: C0-C3 on {len(environments)} environments, grid cost "
        f"{_mean(gap('grid_resolution_gap')):.4f} -> {_relative(CUT_DECOMPOSITION)}"
    )
    return 0


# ------------------------------------------------------------------ sections 8 and 10
#
# Four outcomes, and collapsing them would hide the two that matter. A method that refuses is not
# a method that deployed unsafely, and a method that deployed safely into nothing is not a method
# that worked.

F1_NO_FEASIBLE_CUT = "F1_no_feasible_certified_cut"
F2_DEGENERATE = "F2_certified_but_degenerate"
F3_NON_DEGENERATE = "F3_certified_and_non_degenerate"
F4_FALSE_CERTIFICATION = "F4_false_certification"
FAILURE_MODES = (F1_NO_FEASIBLE_CUT, F2_DEGENERATE, F3_NON_DEGENERATE, F4_FALSE_CERTIFICATION)


def classify(row: Any, coverage_floor: float, epsilon: float = PRIMARY_EPSILON) -> str:
    """One deployment's failure mode. Order matters: a false certification outranks everything."""
    feasible = bool(row["cut_feasible"])
    harm = float(row["realized_harm_rate"])
    if feasible and not math.isnan(harm) and harm > epsilon:
        return F4_FALSE_CERTIFICATION
    if not feasible:
        return F1_NO_FEASIBLE_CUT
    if float(row["coverage"]) < coverage_floor:
        return F2_DEGENERATE
    return F3_NON_DEGENERATE


def run_feasibility() -> int:
    """The failure-mode matrix, the feasibility curve, and the document-clustering audit."""
    started = time.monotonic()
    cells, design = load_cells()
    frozen = load_frozen()
    environments = list(design["environments"])
    floor = frozen.coverage_floor

    matrix: dict[str, Any] = {}
    curve: dict[str, Any] = {}
    for method in METHODS[1:]:
        per_budget: dict[str, Any] = {}
        for size in CERT_BUDGETS:
            if size == 0:
                continue
            block = primary_slice(cells, "A", method, sampler=PRIMARY_SAMPLER, n_cert=size)
            if block.empty:
                continue
            modes = block.apply(lambda r: classify(r, floor), axis=1)
            by_environment = {}
            for name, group in block.groupby("environment", observed=True):
                labels = group.apply(lambda r: classify(r, floor), axis=1)
                counts = {mode: int((labels == mode).sum()) for mode in FAILURE_MODES}
                by_environment[str(name)] = {
                    **counts,
                    "draws": len(group),
                    "dominant": max(counts, key=lambda k: counts[k]),
                    "any_false_certification": counts[F4_FALSE_CERTIFICATION] > 0,
                }
            per_budget[str(size)] = {
                "cells": len(block),
                **{mode: int((modes == mode).sum()) for mode in FAILURE_MODES},
                "feasibility_rate": float(block["cut_feasible"].mean()),
                "refusal_rate": float((block["n_accepted"] == 0).mean()),
                "mean_certified_upper_bound": float(
                    block["certified_risk_upper_bound"].replace([np.inf], np.nan).mean()
                ),
                "mean_realized_risk": _mean(block["realized_harm_rate"].tolist()),
                "mean_coverage": float(block["coverage"].mean()),
                "mean_repair_recall": float(block["repair_recall"].mean()),
                "mean_risk_controlled": float(block["risk_controlled"].mean()),
                "mean_acceptance_depth": float(block["coverage"].mean()),
                "environments_non_degenerate": sum(
                    1 for r in by_environment.values() if r["dominant"] == F3_NON_DEGENERATE
                ),
                "environments_with_a_false_certification": sum(
                    1 for r in by_environment.values() if r["any_false_certification"]
                ),
                "by_environment": by_environment,
            }
        curve[method] = per_budget
        matrix[method] = {
            size: {
                name: entry["by_environment"][name]["dominant"]
                for name in environments
                if name in entry["by_environment"]
            }
            for size, entry in per_budget.items()
        }

    _write_json_once(
        OUT / "certification_feasibility.json",
        {
            "artifact": "certification_feasibility",
            "schema_version": f"sgv15-certification_feasibility-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epsilon": PRIMARY_EPSILON,
            "sampler": PRIMARY_SAMPLER,
            "coverage_floor": floor,
            "failure_modes": {
                F1_NO_FEASIBLE_CUT: "the bound never clears epsilon; the method deploys nothing",
                F2_DEGENERATE: "a safe cut exists but coverage is below SGV13's floor",
                F3_NON_DEGENERATE: "safe and operationally meaningful",
                F4_FALSE_CERTIFICATION: (
                    "the method certified a cut and the realised evaluation harm exceeded "
                    "epsilon anyway. The most serious failure in the stage."
                ),
            },
            "by_method": curve,
            "matrix": matrix,
            # The headline safety comparison of the stage, aggregated once here so it is read
            # from an artifact rather than recomputed in prose.
            "aggregate": {
                "conservative_methods": list(METHODS[2:]),
                "conservative_cells": sum(
                    entry["cells"] for m in METHODS[2:] for entry in curve[m].values()
                ),
                "conservative_false_certifications": sum(
                    entry[F4_FALSE_CERTIFICATION]
                    for m in METHODS[2:]
                    for entry in curve[m].values()
                ),
                "naive_method": M1_EMPIRICAL,
                "naive_cells": sum(entry["cells"] for entry in curve[M1_EMPIRICAL].values()),
                "naive_false_certifications": sum(
                    entry[F4_FALSE_CERTIFICATION] for entry in curve[M1_EMPIRICAL].values()
                ),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )

    # Section 10. The certification bound treats accepted candidates as independent Bernoulli
    # draws. They are not: they are clustered inside documents. This measures the gap.
    indexed = indexed_counts()
    audit: dict[str, Any] = {}
    for name in environments:
        try:
            block = cell_block(
                indexed,
                environment=name,
                experiment="A",
                f2=PRIMARY_SAMPLER,
                f3=str(MECHANISM_BUDGET),
                method=M2_CLOPPER,
            )
        except PhaseError:
            continue
        accepted = block.accepted.sum(axis=0)
        harmful = block.harmful.sum(axis=0)
        total_accepted = float(accepted.sum())
        total_harmful = float(harmful.sum())
        if total_accepted <= 0:
            audit[name] = {"deployed": False, "note": "the method accepted nothing on any draw"}
            continue
        rate = total_harmful / total_accepted
        # Document-clustered bootstrap of the realised accepted-set harm rate.
        multiplicity = document_multiplicities(
            np.asarray(block.documents), BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES
        ).astype(float)
        draws = s13.statistic_draws(block, multiplicity, HARM, PRIMARY_EPSILON).mean(axis=1)
        interval = _interval(draws)
        per_document = np.where(accepted > 0, harmful / np.maximum(accepted, 1e-12), np.nan)
        observed = per_document[~np.isnan(per_document)]
        binomial_variance = rate * (1.0 - rate) / total_accepted
        clustered_variance = float(np.var(draws, ddof=1))
        audit[name] = {
            "deployed": True,
            "accepted_candidates": total_accepted,
            "harmful_accepted": total_harmful,
            "documents": len(block.documents),
            "documents_contributing_an_accepted_edit": int((accepted > 0).sum()),
            "realized_harm_rate": rate,
            "candidate_level_clopper_pearson_upper": clopper_pearson_upper(
                round(total_harmful), round(total_accepted), CERT_DELTA / len(CUT_GRID)
            ),
            "document_clustered_bootstrap_upper": float(interval["ci_upper"]),
            "document_clustered_bootstrap_lower": float(interval["ci_lower"]),
            "per_document_harm_rate_mean": float(observed.mean())
            if observed.size
            else float("nan"),
            "per_document_harm_rate_sd": (
                float(observed.std(ddof=1)) if observed.size > 1 else float("nan")
            ),
            "design_effect_on_the_harm_rate": (
                clustered_variance / binomial_variance if binomial_variance > 0 else float("nan")
            ),
            "candidate_bound_is_anti_conservative": bool(
                clopper_pearson_upper(
                    round(total_harmful),
                    round(total_accepted),
                    CERT_DELTA / len(CUT_GRID),
                )
                < float(interval["ci_upper"])
            ),
        }

    anti = [k for k, v in audit.items() if v.get("candidate_bound_is_anti_conservative")]
    _write_json_once(
        RISK_BOUND_RESULTS,
        {
            "artifact": "risk_bound_results",
            "schema_version": f"sgv15-risk_bound_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": (
                "the certification bound assumes independent Bernoulli draws. Candidate rows are "
                "clustered inside documents. How much does the assumption cost?"
            ),
            "method": M2_CLOPPER,
            "certification_budget": MECHANISM_BUDGET,
            "alpha": CERT_DELTA,
            "per_test_alpha": CERT_DELTA / len(CUT_GRID),
            "resamples": BOOTSTRAP_RESAMPLES,
            "resampling_unit": "document",
            "per_environment": audit,
            "environments_where_the_candidate_bound_is_anti_conservative": anti,
            "count_anti_conservative": len(anti),
            "interpretation": (
                "where the candidate-level bound sits BELOW the document-clustered upper "
                "confidence limit on the same accepted set, the exact Bernoulli guarantee is "
                "being read on a sample that does not satisfy its independence assumption, and "
                "the certificate is weaker than its nominal level. This is reported, not "
                "corrected: correcting it would be a new method, and SGV15 is not permitted to "
                "invent one."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"feasibility: {len(curve)} methods classified, {len(anti)} environments where the "
        f"candidate-level bound is anti-conservative -> {_relative(RISK_BOUND_RESULTS)}"
    )
    return 0


# ------------------------------------------------------------------ section 22: the twelve controls


def _control_row(
    cells: pd.DataFrame, label: str, description: str, **kwargs: Any
) -> dict[str, Any]:
    """One control, summarised over every environment it is defined on."""
    experiment = kwargs.pop("experiment", "A")
    method = kwargs.pop("method")
    block = primary_slice(cells, experiment, method, **kwargs)
    if block.empty:
        return {"control": label, "description": description, "available": False}
    rows = _per_environment(block)
    return {
        "control": label,
        "description": description,
        "available": True,
        "method": method,
        "experiment": experiment,
        "environments": len(rows),
        "mean_risk_controlled": _mean([r["risk_controlled"] for r in rows.values()]),
        "mean_realized_harm": _mean([r["realized_harm_rate"] for r in rows.values()]),
        "mean_coverage": _mean([r["coverage"] for r in rows.values()]),
        "mean_auroc": _mean([r["auroc_safe"] for r in rows.values()]),
        "environments_holding_the_bound": sum(1 for r in rows.values() if holds_the_bound(r)),
        "mean_cut_feasible_fraction": _mean([r["cut_feasible_fraction"] for r in rows.values()]),
        "per_environment": rows,
    }


def run_controls() -> int:
    """Every control the frozen design can support, each against the arm it is a control for."""
    started = time.monotonic()
    cells, design = load_cells()
    reference = MECHANISM_BUDGET
    primary = _control_row(
        cells,
        "primary",
        "the adapted ranking with the conservative certified cut -- the arm under test",
        method=M2_CLOPPER,
        sampler=PRIMARY_SAMPLER,
        n_cert=reference,
    )
    controls = {
        "control_1_frozen_source_cut": _control_row(
            cells,
            "control_1_frozen_source_cut",
            "SGV14's cut, transported unchanged. Reads no target label.",
            method=M0_FROZEN_SOURCE,
            n_cert=0,
        ),
        "control_2_source_only_refit": _control_row(
            cells,
            "control_2_source_only_refit",
            "no target adaptation labels at all: the frozen source ranking with the source cut.",
            experiment="C",
            method=X_SOURCE_RANK_SOURCE_CUT,
            sampler=PRIMARY_SAMPLER,
            n_cert=reference,
        ),
        "control_3_permuted_adaptation_labels": _control_row(
            cells,
            "control_3_permuted_adaptation_labels",
            "the same rows, documents and class balance; the label-to-row assignment destroyed.",
            experiment="C",
            method=C3_PERMUTED_ADAPTATION,
            sampler=PRIMARY_SAMPLER,
            n_cert=reference,
        ),
        "control_4_permuted_certification_labels": _control_row(
            cells,
            "control_4_permuted_certification_labels",
            "the adapted ranking, certified against shuffled certification outcomes.",
            experiment="C",
            method=C4_PERMUTED_CERTIFICATION,
            sampler=PRIMARY_SAMPLER,
            n_cert=reference,
        ),
        "control_5_uncertainty_biased_certification": _control_row(
            cells,
            "control_5_uncertainty_biased_certification",
            "the diagnostic negative sampler: labels bought where the deployment is about to cut.",
            method=M2_CLOPPER,
            sampler=CS0_UNCERTAINTY,
            n_cert=reference,
        ),
        "control_6_representative_certification": _control_row(
            cells,
            "control_6_representative_certification",
            "uniform-random certification rows, the other representative sampler.",
            method=M2_CLOPPER,
            sampler=CS1_UNIFORM,
            n_cert=reference,
        ),
        "control_7_certified_cut_source_ranking": _control_row(
            cells,
            "control_7_certified_cut_source_ranking",
            "the target-certified cut applied to the SOURCE ranking. Isolates the cut.",
            experiment="C",
            method=X_SOURCE_RANK_CERTIFIED_CUT,
            sampler=PRIMARY_SAMPLER,
            n_cert=reference,
        ),
        "control_8_adapted_ranking_source_cut": _control_row(
            cells,
            "control_8_adapted_ranking_source_cut",
            "the adapted ranking with the frozen source cut. Isolates the ranking. SGV14's arm.",
            experiment="C",
            method=X_ADAPTED_RANK_SOURCE_CUT,
            sampler=PRIMARY_SAMPLER,
            n_cert=reference,
        ),
        "control_9_oracle_evaluation_cut": _control_row(
            cells,
            "control_9_oracle_evaluation_cut",
            "ANALYSIS ONLY. The best declared cut on the evaluation block. Never selected.",
            method=C1_ORACLE,
            sampler=PRIMARY_SAMPLER,
            n_cert=reference,
        ),
        "control_10_certification_from_few_documents": _control_row(
            cells,
            "control_10_certification_from_few_documents",
            f"every certification label drawn from {FEW_DOCUMENTS} documents.",
            method=M2_CLOPPER,
            sampler=CS_FEW_DOCUMENTS,
            n_cert=reference,
        ),
        "control_11_certification_from_one_document": _control_row(
            cells,
            "control_11_certification_from_one_document",
            "every certification label drawn from a single document.",
            method=M2_CLOPPER,
            sampler=CS_ONE_DOCUMENT,
            n_cert=reference,
        ),
        "control_12_full_document_stratified": _control_row(
            cells,
            "control_12_full_document_stratified",
            "the primary sampler at the largest declared budget.",
            method=M2_CLOPPER,
            sampler=CS2_DOCUMENT,
            n_cert=max(CERT_BUDGETS),
        ),
    }

    def delta(name: str) -> float:
        entry = controls[name]
        if not entry["available"]:
            return float("nan")
        return primary["mean_risk_controlled"] - entry["mean_risk_controlled"]

    verdicts = {
        "adaptation_information_is_real": bool(delta("control_3_permuted_adaptation_labels") > 0.0),
        "certification_information_is_real": bool(
            delta("control_4_permuted_certification_labels") > 0.0
        ),
        "representativeness_matters": bool(
            delta("control_5_uncertainty_biased_certification") > 0.0
        ),
        "document_diversity_matters": bool(
            delta("control_11_certification_from_one_document") > 0.0
        ),
        "the_cut_is_load_bearing": bool(delta("control_8_adapted_ranking_source_cut") > 0.0),
        "the_ranking_is_load_bearing": bool(delta("control_7_certified_cut_source_ranking") > 0.0),
    }
    _write_json_once(
        CONTROL_RESULTS,
        {
            "artifact": "control_results",
            "schema_version": f"sgv15-control_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epsilon": PRIMARY_EPSILON,
            "reference_certification_budget": reference,
            "primary_arm": primary,
            "controls": controls,
            "deltas_against_the_primary_arm": {name: delta(name) for name in controls},
            "verdicts": verdicts,
            "reading": (
                "a control that MATCHES the primary arm has falsified the claim that the "
                "information the control destroys was what produced the result. Controls are not "
                "reinterpreted after the fact; each one's direction was fixed with the design."
            ),
            "environments": list(design["environments"]),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    passed = sum(1 for v in verdicts.values() if v)
    print(
        f"controls: {len(controls)} controls, {passed}/{len(verdicts)} discrimination checks "
        f"positive -> {_relative(CONTROL_RESULTS)}"
    )
    return 0


# ------------------------------------------------------------------ section 23: falsification
#
# Fifteen statements that would each, if false, mean something specific about SGV15 is broken.
# They are recorded with their outcome whichever way they land: a failed test here is a finding
# and is reported as one, never quietly dropped or reinterpreted into a pass.


def _test(
    name: str, question: str, expectation: str, passed: bool, **evidence: Any
) -> dict[str, Any]:
    return {
        "test": name,
        "question": question,
        "expectation": expectation,
        "passed": bool(passed),
        "evidence": evidence,
    }


def run_negative() -> int:
    """The fifteen falsification tests, computed from the frozen cells and bounds."""
    started = time.monotonic()
    cells, design = load_cells()
    frozen = load_frozen()
    reference = MECHANISM_BUDGET
    per_test = CERT_DELTA / len(CUT_GRID)
    tests: list[dict[str, Any]] = []

    def slice_of(method: str, **kwargs: Any) -> pd.DataFrame:
        return primary_slice(cells, kwargs.pop("experiment", "A"), method, **kwargs)

    zero = cells[
        (cells["experiment"] == "A")
        & (cells["requested_cert"] == 0)
        & (cells["method"].isin(METHODS[1:]))
    ]
    tests.append(
        _test(
            "N1_zero_labels_cannot_certify",
            "can a certified method place a cut with no certification labels at all?",
            "no cell at N_cert = 0 is feasible, and none deploys an edit",
            bool(zero.empty or ((~zero["cut_feasible"]).all() and (zero["n_accepted"] == 0).all())),
            cells=len(zero),
            feasible_cells=int(zero["cut_feasible"].sum()) if not zero.empty else 0,
            accepting_cells=int((zero["n_accepted"] > 0).sum()) if not zero.empty else 0,
        )
    )

    bound_at_25 = clopper_pearson_upper(0, 25, CERT_DELTA)
    tests.append(
        _test(
            "N2_twenty_five_labels_are_provably_insufficient",
            "can 25 certification labels certify below epsilon in the best possible case?",
            "the zero-harm exact upper bound at n = 25 exceeds epsilon",
            bool(bound_at_25 > PRIMARY_EPSILON),
            n_cert=25,
            observed_harmful=0,
            alpha=CERT_DELTA,
            clopper_pearson_upper=bound_at_25,
            epsilon=PRIMARY_EPSILON,
            grid_corrected_upper=clopper_pearson_upper(0, 25, per_test),
            status="A PRIORI -- arithmetic on the bound, computed before any endpoint was read",
        )
    )

    real = slice_of(M2_CLOPPER, sampler=PRIMARY_SAMPLER, n_cert=reference)
    shuffled = slice_of(
        C4_PERMUTED_CERTIFICATION, experiment="C", sampler=PRIMARY_SAMPLER, n_cert=reference
    )
    real_mean = float(real["risk_controlled"].mean()) if not real.empty else float("nan")
    shuffled_mean = (
        float(shuffled["risk_controlled"].mean()) if not shuffled.empty else float("nan")
    )
    tests.append(
        _test(
            "N3_permuted_certification_labels_do_not_help",
            "do shuffled certification outcomes certify as well as real ones?",
            "the permuted arm does not beat the real one on the primary endpoint",
            bool(shuffled_mean <= real_mean or math.isnan(shuffled_mean)),
            real_risk_controlled=real_mean,
            permuted_risk_controlled=shuffled_mean,
            delta=real_mean - shuffled_mean,
        )
    )

    permuted_adapt = slice_of(
        C3_PERMUTED_ADAPTATION, experiment="C", sampler=PRIMARY_SAMPLER, n_cert=reference
    )
    adapted_auroc = float(real["rank__auroc_safe"].mean()) if not real.empty else float("nan")
    permuted_auroc = (
        float(permuted_adapt["rank__auroc_safe"].mean())
        if not permuted_adapt.empty
        else float("nan")
    )
    tests.append(
        _test(
            "N4_permuted_adaptation_labels_destroy_the_ranking_gain",
            "does the adapted ranking survive destroying the label-to-row assignment?",
            "the permuted refit ranks no better than the real one",
            bool(permuted_auroc <= adapted_auroc or math.isnan(permuted_auroc)),
            adapted_auroc=adapted_auroc,
            permuted_auroc=permuted_auroc,
            delta=adapted_auroc - permuted_auroc,
        )
    )

    source_source = slice_of(
        X_SOURCE_RANK_SOURCE_CUT, experiment="C", sampler=PRIMARY_SAMPLER, n_cert=reference
    )
    baseline_rows = _baseline(cells)
    source_rows = _per_environment(source_source) if not source_source.empty else {}
    labels_read = int(source_source["n_adapt"].abs().max()) if not source_source.empty else -1
    spread = _mean(
        [
            float(group["rank__auroc_safe"].std(ddof=0))
            for _, group in source_source.groupby("environment", observed=True)
        ]
    )
    adaptation_gain = _mean(
        [
            baseline_rows[name]["auroc_safe"] - source_rows[name]["auroc_safe"]
            for name in source_rows
            if name in baseline_rows
        ]
    )
    tests.append(
        _test(
            "N5_source_only_ranking_is_genuinely_source_only",
            "does the source-only corner of the 2x2 read any target label?",
            "it buys zero adaptation labels and its ranking is identical across every draw",
            bool(labels_read == 0 and not math.isnan(spread) and spread < 1e-12),
            adaptation_labels_bought=labels_read,
            mean_within_environment_auroc_spread=spread,
            environments=len(source_rows),
            adaptation_auroc_gain_over_this_corner=adaptation_gain,
            note=(
                "the SGV15 baseline arm carries the ADAPTED ranking at N_adapt = 250 with the "
                "frozen source cut, so it is not the comparator for this identity. The gain over "
                "this corner is reported here because it is the size of what adaptation bought."
            ),
        )
    )

    oracle = slice_of(C1_ORACLE, sampler=PRIMARY_SAMPLER, n_cert=reference)
    oracle_rows = _per_environment(oracle) if not oracle.empty else {}
    certified_rows = _per_environment(real) if not real.empty else {}
    exceed = [
        name
        for name in certified_rows
        if name in oracle_rows
        and certified_rows[name]["risk_controlled"] > oracle_rows[name]["risk_controlled"] + 1e-9
    ]
    tests.append(
        _test(
            "N6_no_method_beats_the_on_grid_oracle",
            "can a certification procedure exceed the best achievable declared cut?",
            "no environment where the certified method beats the on-grid oracle",
            len(exceed) == 0,
            environments_exceeding=exceed,
            oracle_mean=_mean([r["risk_controlled"] for r in oracle_rows.values()]),
            certified_mean=_mean([r["risk_controlled"] for r in certified_rows.values()]),
        )
    )

    empirical = slice_of(M1_EMPIRICAL, sampler=PRIMARY_SAMPLER, n_cert=reference)
    tests.append(
        _test(
            "N7_the_bound_is_more_conservative_than_the_point_estimate",
            "does the exact bound ever cut deeper than the naive empirical rule?",
            "certified coverage never exceeds empirical coverage on the same sample",
            bool(float(real["coverage"].mean()) <= float(empirical["coverage"].mean()) + 1e-9),
            empirical_coverage=float(empirical["coverage"].mean()),
            certified_coverage=float(real["coverage"].mean()),
        )
    )

    bentkus = slice_of(M2B_BENTKUS, sampler=PRIMARY_SAMPLER, n_cert=reference)
    tests.append(
        _test(
            "N8_bentkus_is_never_tighter_than_clopper_pearson",
            "is the distribution-free bound ever tighter than the exact Bernoulli one?",
            "Bentkus coverage never exceeds Clopper-Pearson coverage",
            bool(float(bentkus["coverage"].mean()) <= float(real["coverage"].mean()) + 1e-9),
            bentkus_coverage=float(bentkus["coverage"].mean()),
            clopper_pearson_coverage=float(real["coverage"].mean()),
            why=(
                "Bentkus carries a factor of e because it must hold for any loss bounded in "
                "[0, 1]; the harm indicator is Bernoulli, so the exact bound is both valid and "
                "tighter. A violation would mean one of the two is misimplemented."
            ),
        )
    )

    one_doc = slice_of(M2_CLOPPER, sampler=CS_ONE_DOCUMENT, n_cert=reference)
    floor = frozen.coverage_floor
    false_one = float(
        one_doc.apply(lambda r: classify(r, floor) == F4_FALSE_CERTIFICATION, axis=1).mean()
    )
    false_rep = float(
        real.apply(lambda r: classify(r, floor) == F4_FALSE_CERTIFICATION, axis=1).mean()
    )
    tests.append(
        _test(
            "N9_single_document_certification_is_less_reliable",
            "does certifying from one document produce more false certifications?",
            "the single-document sampler's false-certification rate is at least the primary's",
            bool(false_one >= false_rep),
            single_document_false_certification_rate=false_one,
            representative_false_certification_rate=false_rep,
        )
    )

    uncertainty = slice_of(M2_CLOPPER, sampler=CS0_UNCERTAINTY, n_cert=reference)
    prevalence = _evaluation_prevalence()

    def prevalence_error(block: pd.DataFrame) -> float:
        if block.empty:
            return float("nan")
        return _mean(
            [
                abs(
                    float(group["cert_harmful_rate"].mean())
                    - prevalence[str(name)]["harmful_prevalence"]
                )
                for name, group in block.groupby("environment", observed=True)
            ]
        )

    tests.append(
        _test(
            "N10_uncertainty_sampling_misestimates_prevalence",
            "does the targeted sampler estimate the target harm prevalence as well as a "
            "representative one?",
            "the uncertainty-biased sampler's prevalence error is larger",
            bool(prevalence_error(uncertainty) > prevalence_error(real)),
            uncertainty_prevalence_error=prevalence_error(uncertainty),
            representative_prevalence_error=prevalence_error(real),
        )
    )

    certifying = cells[cells["method"].isin(METHODS[1:])]
    leaked = certifying[(~certifying["cut_feasible"]) & (certifying["n_accepted"] > 0)]
    tests.append(
        _test(
            "N11_refusal_means_refusal",
            "does a method that declared no feasible cut still deploy edits?",
            "no infeasible cell accepts an evaluation row",
            len(leaked) == 0,
            infeasible_cells=int((~certifying["cut_feasible"]).sum()),
            infeasible_cells_that_deployed=len(leaked),
            note=(
                "`select_threshold` signals infeasibility with a threshold above the largest "
                "score IT SAW. Carried to a different block that is not a refusal, and this stage "
                "maps infeasibility to an infinite threshold before deployment."
            ),
        )
    )

    feasibility = {
        size: float(
            slice_of(M2_CLOPPER, sampler=PRIMARY_SAMPLER, n_cert=size)["cut_feasible"].mean()
        )
        for size in CERT_BUDGETS
        if size > 0
    }
    ordered = [feasibility[s] for s in sorted(feasibility)]
    tests.append(
        _test(
            "N12_feasibility_increases_with_certification_labels",
            "does buying more certification labels make certification more often possible?",
            "the feasibility rate at the largest budget exceeds that at the smallest",
            bool(ordered[-1] >= ordered[0]),
            feasibility_by_budget={str(k): v for k, v in sorted(feasibility.items())},
        )
    )

    certified_cells = cells[
        cells["method"].isin(METHODS[1:]) & np.isclose(cells["epsilon"], PRIMARY_EPSILON)
    ]
    cardinality = {
        f"{name}|{method}|{experiment}|{allocation}": int(group["tau"].nunique(dropna=False))
        for (name, method, experiment, allocation), group in certified_cells.groupby(
            ["environment", "method", "experiment", "allocation"], observed=True
        )
    }
    worst = max(cardinality.values()) if cardinality else 0
    tests.append(
        _test(
            "N13_only_the_declared_family_is_searched",
            "was any cut placed outside the declared nine-point family?",
            "no environment-method pair produces more distinct thresholds than the grid has "
            "points, plus the abstain threshold",
            bool(worst <= len(CUT_GRID) + 1),
            declared_grid_points=len(CUT_GRID),
            permitted_distinct_thresholds=len(CUT_GRID) + 1,
            largest_observed=worst,
            groups_checked=len(cardinality),
            grouping=(
                "environment x method x experiment x allocation -- one FITTED MODEL per group. "
                "The grid is a set of quantiles of the certification half's scores under a given "
                "model, so experiment B's six allocations and the mechanism pass's three refits "
                "each carry their own nine thresholds. Pooling across models would compare "
                "different grids and count their union."
            ),
            note=(
                "realised evaluation coverage is deliberately NOT the quantity tested. The grid "
                "is a set of quantiles of the certification half's scores; the evaluation block "
                "is a different score distribution, so a declared depth of 0.50 can accept more "
                "than half the evaluation rows. That is the distribution shift this project "
                "measures, not a grid violation. What would be a violation is a method that "
                "produced more distinct thresholds than the declared family contains."
            ),
        )
    )

    rows = pd.read_parquet(CERTIFICATION_ROWS)
    within = {
        str(name): int(
            block.groupby("document_id", observed=True)["sgv15_role"].nunique().gt(1).sum()
        )
        for name, block in rows.groupby("environment", observed=True)
    }
    across = int(rows.groupby("document_id", observed=True)["sgv15_role"].nunique().gt(1).sum())
    tests.append(
        _test(
            "N14_the_two_halves_share_no_document_within_an_environment",
            "can a document contribute both a fitting label and a certification label to the "
            "same model?",
            "no document carries both roles inside one environment",
            sum(within.values()) == 0,
            documents_with_both_roles_by_environment=within,
            documents_whose_role_differs_across_environments=across,
            documents_total=int(rows["document_id"].nunique()),
            note=(
                "the cross-environment count is reported and is NOT a failure. The split balances "
                "to a target count over each environment's own adaptation documents, and those "
                "document sets differ -- most visibly on the capacity-limited environment. No "
                "environment's fit reads another environment's rows, so a page landing in "
                "different halves in two environments is not a shared label."
            ),
        )
    )

    tight = primary_slice(
        cells, "A", M2_CLOPPER, sampler=PRIMARY_SAMPLER, n_cert=reference, epsilon=0.05
    )
    tests.append(
        _test(
            "N15_a_tighter_tolerance_is_never_more_permissive",
            "does demanding a tighter harm tolerance ever admit MORE of the ranking?",
            "coverage at epsilon = 0.05 does not exceed coverage at epsilon = 0.10",
            bool(float(tight["coverage"].mean()) <= float(real["coverage"].mean()) + 1e-9),
            coverage_at_0p05=float(tight["coverage"].mean()),
            coverage_at_0p10=float(real["coverage"].mean()),
        )
    )

    passed = sum(1 for t in tests if t["passed"])
    _write_json_once(
        NEGATIVE_TESTS,
        {
            "artifact": "negative_tests",
            "schema_version": f"sgv15-negative_tests-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epsilon": PRIMARY_EPSILON,
            "reference_certification_budget": reference,
            "tests": tests,
            "passed": passed,
            "total": len(tests),
            "failed": [t["test"] for t in tests if not t["passed"]],
            "policy": (
                "a failed falsification test is a finding about SGV15 and is reported as one. "
                "None is reinterpreted, and none is removed after its outcome was seen."
            ),
            "environments": list(design["environments"]),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"negative: {passed}/{len(tests)} falsification tests pass -> {_relative(NEGATIVE_TESTS)}"
    )
    return 0


# ------------------------------------------------------------------ section 27: nested selection
#
# The development environments are all ten, and pooling their best number would be selection on
# the same data the number is read from. Each environment is held out in turn, the configuration
# is chosen on the other nine, and only then is the held-out environment's endpoint read.


def _config_table(cells: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, float]]:
    """Every selectable (method, sampler, budget) configuration's per-environment endpoint."""
    table: dict[tuple[str, str, int], dict[str, float]] = {}
    for method in METHODS[1:]:
        for sampler in SELECTABLE_SAMPLERS:
            for size in CERT_BUDGETS:
                if size == 0:
                    continue
                block = primary_slice(cells, "A", method, sampler=sampler, n_cert=size)
                if block.empty:
                    continue
                table[(method, sampler, size)] = {
                    str(name): float(group["risk_controlled"].mean())
                    for name, group in block.groupby("environment", observed=True)
                }
    return table


def _nested(table: dict[Any, dict[str, float]], environments: list[str]) -> dict[str, Any]:
    """Leave-one-environment-out selection, and the held-out value of whatever it chose."""
    chosen: dict[str, Any] = {}
    for held_out in environments:
        development = [e for e in environments if e != held_out]
        best: tuple[float, Any] | None = None
        for config, values in table.items():
            inner = [values[e] for e in development if e in values]
            if not inner:
                continue
            score = float(np.mean(inner))
            key = (score, tuple(str(part) for part in config))
            if best is None or (score > best[0]) or (score == best[0] and key[1] < best[1]):
                best = (score, config)
        if best is None:
            continue
        config = best[1]
        chosen[held_out] = {
            "selected": [str(part) for part in config],
            "development_mean": best[0],
            "held_out_value": table[config].get(held_out, float("nan")),
        }
    frequency: dict[str, int] = {}
    for entry in chosen.values():
        frequency["|".join(entry["selected"])] = frequency.get("|".join(entry["selected"]), 0) + 1
    return {
        "by_held_out_environment": chosen,
        "selection_frequency": frequency,
        "distinct_configurations_selected": len(frequency),
        "modal_configuration": max(frequency, key=lambda k: frequency[k]) if frequency else None,
        "modal_share": (max(frequency.values()) / len(chosen) if chosen else float("nan")),
        "mean_held_out_endpoint": _mean([e["held_out_value"] for e in chosen.values()]),
        "stable": bool(frequency and max(frequency.values()) == len(chosen)),
    }


def run_select() -> int:
    """Phase B of the selection plan, plus the pooled maximum it is there to distrust."""
    started = time.monotonic()
    cells, design = load_cells()
    environments = list(design["environments"])
    table = _config_table(cells)
    nested = _nested(table, environments)

    pooled = {
        "|".join(str(p) for p in config): _mean(list(values.values()))
        for config, values in table.items()
    }
    pooled_best = max(pooled, key=lambda k: pooled[k]) if pooled else None

    allocation_table: dict[Any, dict[str, float]] = {}
    for n_adapt, n_cert in ALLOCATIONS:
        label = f"{n_adapt}+{n_cert}"
        method = M0_FROZEN_SOURCE if n_cert == 0 else M2_CLOPPER
        block = primary_slice(cells, "B", method, allocation=label)
        if block.empty:
            continue
        allocation_table[(label,)] = {
            str(name): float(group["risk_controlled"].mean())
            for name, group in block.groupby("environment", observed=True)
        }
    allocation_nested = _nested(allocation_table, environments)

    modal = nested["modal_configuration"]
    selected = modal.split("|") if modal else None
    payload = {
        "artifact": "method_selection",
        "schema_version": f"sgv15-method_selection-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_status": "DEVELOPMENT -- these ten environments are exposed, not held out",
        "procedure": design["selection_plan"]["phase_b"],
        "endpoint": "risk-controlled repair recall at epsilon = 0.10",
        "oracle_excluded": True,
        "samplers_excluded_from_selection": {
            sampler: (
                "registered DIAGNOSTIC NEGATIVE; measured in every comparison, never selectable"
                if sampler == CS0_UNCERTAINTY
                else "registered CONTROL; measured, never selectable"
            )
            for sampler in NOT_SELECTABLE_SAMPLERS
        },
        "selection_pool": {
            "methods": list(METHODS[1:]),
            "samplers": list(SELECTABLE_SAMPLERS),
            "certification_budgets": [s for s in CERT_BUDGETS if s > 0],
        },
        "candidate_configurations": len(table),
        "nested_leave_one_environment_out": nested,
        "allocation_selection": allocation_nested,
        "pooled_maximum": {
            "configuration": pooled_best,
            "value": pooled.get(pooled_best) if pooled_best else None,
            "why_it_is_not_the_answer": (
                "the pooled maximum is chosen on the same ten environments it is read on. It is "
                "recorded so the optimism of that shortcut is visible, and it does not select."
            ),
        },
        "selected_method": selected[0] if selected else None,
        "selected_sampler": selected[1] if selected else None,
        "selected_certification_budget": int(selected[2]) if selected else None,
        "selected_allocation": (
            allocation_nested["modal_configuration"]
            if allocation_nested["modal_configuration"]
            else None
        ),
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
    _write_json_once(METHOD_SELECTION, payload)
    print(
        f"select: {len(table)} configurations, modal choice {modal} "
        f"(stable={nested['stable']}) -> {_relative(METHOD_SELECTION)}"
    )
    return 0


# ------------------------------------------------------------------ section 12: statistics


def selected_configuration() -> tuple[str, str, int]:
    """Whatever the nested selection chose, read from its artifact and never re-derived."""
    if not METHOD_SELECTION.is_file():
        raise PhaseError("run --select first; the statistics test the SELECTED method")
    record = cc_read_json(METHOD_SELECTION)
    method = record["selected_method"]
    if method is None:
        raise PhaseError("the nested selection chose nothing; there is no method to test")
    return (
        str(method),
        str(record["selected_sampler"]),
        int(record["selected_certification_budget"]),
    )


def run_stats() -> int:
    """The pre-registered primary family: the selected method against the frozen source cut."""
    started = time.monotonic()
    _cells, design = load_cells()
    method, sampler, size = selected_configuration()
    indexed = indexed_counts()
    environments = list(design["environments"])

    def blocks(name: str) -> tuple[Any, Any]:
        left = cell_block(
            indexed,
            environment=name,
            experiment="A",
            f2=sampler,
            f3=str(size),
            method=method,
        )
        right = cell_block(
            indexed,
            environment=name,
            experiment="A",
            f2=sampler,
            f3="0",
            method=M0_FROZEN_SOURCE,
        )
        return left, right

    paired = [blocks(name) for name in environments]
    primary = stratified_delta(paired, RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED)
    harm_overall = stratified_delta(paired, HARM, PRIMARY_EPSILON, BOOTSTRAP_SEED)

    family: dict[str, dict[str, float]] = {}
    for name, (left, right) in zip(environments, paired, strict=True):
        family[f"{name}|repair_recall"] = s13.paired_delta(
            left, right, RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED
        )
        family[f"{name}|realized_harm"] = s13.paired_delta(
            left, right, HARM, PRIMARY_EPSILON, BOOTSTRAP_SEED
        )
    corrected = holm(family)
    surviving = [k for k, v in corrected.items() if v["survives_holm"]]

    _write_json_once(
        STATISTICAL_TESTS,
        {
            "artifact": "statistical_tests",
            "schema_version": f"sgv15-statistical_tests-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scientific_status": (
                "DEVELOPMENT. These are the environments SGV14 already exposed; a surviving "
                "p-value here is not external confirmation."
            ),
            "selected_configuration": {
                "method": method,
                "sampler": sampler,
                "certification_budget": size,
            },
            "comparator": f"{M0_FROZEN_SOURCE} at N_cert = 0 on the same adapted ranking",
            "resampling_unit": "document",
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_endpoint_repair_recall": primary,
            "primary_endpoint_realized_harm": harm_overall,
            "family": corrected,
            "family_size": len(family),
            "family_definition": design["statistics"]["primary_family"],
            "multiplicity": "Holm-Bonferroni at 0.05",
            "surviving_after_holm": surviving,
            "count_surviving": len(surviving),
            "separation": design["statistics"]["separation"],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"stats: delta {primary['delta']:+.4f} CI [{primary['ci_lower']:+.4f}, "
        f"{primary['ci_upper']:+.4f}] p={primary['p_value']:.4f}, {len(surviving)}/{len(family)} "
        f"survive Holm -> {_relative(STATISTICAL_TESTS)}"
    )
    return 0


# ------------------------------------------------------------------ section 16: the figures


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


METHOD_COLOUR = {
    M0_FROZEN_SOURCE: "#8c8c8c",
    M1_EMPIRICAL: "#d62728",
    M2_CLOPPER: "#1f77b4",
    M2B_BENTKUS: "#9467bd",
    M3_SHRINKAGE: "#2ca02c",
    C1_ORACLE: "#000000",
}
METHOD_LABEL = {
    M0_FROZEN_SOURCE: "M0 frozen source cut",
    M1_EMPIRICAL: "M1 naive empirical",
    M2_CLOPPER: "M2 Clopper-Pearson",
    M2B_BENTKUS: "M2b Bentkus",
    M3_SHRINKAGE: "M3 beta-binomial",
    C1_ORACLE: "C1 oracle (analysis only)",
}


def run_figures() -> int:
    """The eleven required figures. Every plotted number comes out of a saved artifact."""
    started = time.monotonic()
    plt = _figure_style()
    budgets = cc_read_json(CERTIFICATION_BUDGET_RESULTS)
    allocations = cc_read_json(BUDGET_ALLOCATION_RESULTS)["allocations"]
    adaptation = cc_read_json(ADAPTATION_BUDGET_RESULTS)["by_allocation"]
    sampling = cc_read_json(CERTIFICATION_SAMPLING)
    decomposition = cc_read_json(CUT_DECOMPOSITION)
    two_by_two = cc_read_json(ADAPTATION_CERTIFICATION_DECOMPOSITION)
    feasibility = cc_read_json(OUT / "certification_feasibility.json")
    resolution = cc_read_json(OUT / "grid_resolution_analysis.json")
    design = cc_read_json(DESIGN_RECORD)
    frozen = load_frozen()
    environments = list(design["environments"])
    sizes = [s for s in CERT_BUDGETS if s > 0]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    labels = [e.replace("/", "\n") for e in environments]

    note = (
        f"SGV15 target risk certification | DEVELOPMENT stage on SGV14-exposed environments | "
        f"frozen adaptation {frozen.arm} + {frozen.acquisition} | epsilon={PRIMARY_EPSILON}, "
        f"alpha={CERT_DELTA} over a {len(CUT_GRID)}-point declared cut grid"
    )

    def _finish(figure: Any, path: Path) -> None:
        figure.text(0.005, 0.005, note, fontsize=5.5, color="#555555")
        figure.tight_layout(rect=(0, 0.035, 1, 1))
        figure.savefig(path)
        plt.close(figure)
        written.append(path)

    def curve(method: str, field: str) -> list[float]:
        return [
            budgets["by_method"][method].get(str(s), {}).get(field, float("nan")) for s in sizes
        ]

    baseline = budgets["baseline"]
    base_harm = _mean([r["realized_harm_rate"] for r in baseline.values()])
    base_recall = _mean([r["risk_controlled"] for r in baseline.values()])

    # 1. harm against certification labels.
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    for method in METHODS[1:]:
        axis.plot(
            sizes,
            curve(method, "mean_realized_harm"),
            marker="o",
            color=METHOD_COLOUR[method],
            label=METHOD_LABEL[method],
        )
    axis.axhline(PRIMARY_EPSILON, color="#d62728", linestyle="--", lw=1, label="epsilon = 0.10")
    axis.axhline(base_harm, color="#8c8c8c", linestyle=":", lw=1, label="M0 frozen source cut")
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(s) for s in sizes])
    axis.set_xlabel("certification labels (N_cert)")
    axis.set_ylabel("mean realised accepted-set harm rate")
    axis.set_title("SGV15: realised harm against certification budget")
    axis.legend(fontsize=6.5)
    _finish(figure, FIGURE_DIR / "harm_vs_certification_labels.png")

    # 2. repair recall against certification labels.
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    for method in METHODS[1:]:
        axis.plot(
            sizes,
            curve(method, "mean_risk_controlled"),
            marker="o",
            color=METHOD_COLOUR[method],
            label=METHOD_LABEL[method],
        )
    axis.axhline(base_recall, color="#8c8c8c", linestyle=":", lw=1, label="M0 frozen source cut")
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(s) for s in sizes])
    axis.set_xlabel("certification labels (N_cert)")
    axis.set_ylabel(f"mean risk-controlled repair recall at eps={PRIMARY_EPSILON}")
    axis.set_title("SGV15: operational gain against certification budget")
    axis.legend(fontsize=6.5)
    _finish(figure, FIGURE_DIR / "repair_recall_vs_certification_labels.png")

    # 3. feasibility against certification labels.
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    for method in METHODS[1:]:
        axis.plot(
            sizes,
            [
                feasibility["by_method"][method]
                .get(str(s), {})
                .get("feasibility_rate", float("nan"))
                for s in sizes
            ],
            marker="o",
            color=METHOD_COLOUR[method],
            label=METHOD_LABEL[method],
        )
    smallest = cc_read_json(SAMPLE_SIZE_ANALYSIS)[
        "smallest_sample_certifying_zero_harm_below_epsilon"
    ]
    axis.axvline(
        smallest["alpha_over_grid"],
        color="#333333",
        linestyle="--",
        lw=1,
        label=f"a priori floor: n = {smallest['alpha_over_grid']} (grid-corrected)",
    )
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(s) for s in sizes])
    axis.set_xlabel("certification labels (N_cert)")
    axis.set_ylabel("fraction of draws with a feasible certified cut")
    axis.set_title("SGV15: when certification is possible at all")
    axis.legend(fontsize=6.5)
    _finish(figure, FIGURE_DIR / "certification_feasibility_vs_labels.png")

    # 4. the fixed-budget tradeoff.
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    order = [f"{a}+{c}" for a, c in ALLOCATIONS if f"{a}+{c}" in allocations]
    axis.plot(
        range(len(order)),
        [allocations[k]["mean_risk_controlled"] for k in order],
        marker="o",
        color="#1f77b4",
        label="risk-controlled repair recall",
    )
    axis.set_ylabel(f"risk-controlled repair recall at eps={PRIMARY_EPSILON}", color="#1f77b4")
    twin = axis.twinx()
    twin.plot(
        range(len(order)),
        [adaptation[k]["mean_auroc"] for k in order],
        marker="s",
        color="#d68910",
        label="ranking AUROC",
    )
    twin.set_ylabel("mean ranking AUROC (what adaptation buys)", color="#d68910")
    twin.grid(visible=False)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order, rotation=20)
    axis.set_xlabel("allocation  N_adapt + N_cert  (total = 250)")
    axis.set_title("SGV15 experiment B: the same 250 labels, split two ways")
    _finish(figure, FIGURE_DIR / "adaptation_vs_certification_budget.png")

    # 5. the frozen source cut against the certified cut, per environment.
    figure, axis = plt.subplots(figsize=(9.5, 4.0))
    position = np.arange(len(environments))
    width = 0.38
    per_env = decomposition["per_environment"]
    axis.bar(
        position - width / 2,
        [per_env[e]["C0_frozen_source_cut"]["realized_harm_rate"] for e in environments],
        width,
        color="#8c8c8c",
        label="C0 frozen source cut",
    )
    axis.bar(
        position + width / 2,
        [per_env[e]["C3_conservative_certified_cut"]["realized_harm_rate"] for e in environments],
        width,
        color="#1f77b4",
        label="C3 conservative certified cut",
    )
    axis.axhline(PRIMARY_EPSILON, color="#d62728", linestyle="--", lw=1, label="epsilon = 0.10")
    axis.set_xticks(position)
    axis.set_xticklabels(labels, fontsize=6.5)
    axis.set_ylabel("realised accepted-set harm rate")
    axis.set_title("SGV15: the cut that failed to transport, against the certified one")
    axis.legend(fontsize=7)
    _finish(figure, FIGURE_DIR / "frozen_vs_target_cut.png")

    # 6. what the certificate resolves, against what it costs.
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.plot(
        sizes,
        [clopper_pearson_upper(0, s, CERT_DELTA / len(CUT_GRID)) for s in sizes],
        marker="o",
        color="#1f77b4",
        label="best certifiable bound (zero observed harm, grid-corrected)",
    )
    axis.plot(
        sizes,
        [clopper_pearson_upper(0, s, CERT_DELTA) for s in sizes],
        marker="s",
        color="#9467bd",
        label="best certifiable bound (uncorrected alpha)",
    )
    axis.plot(
        sizes,
        curve(M2_CLOPPER, "mean_coverage"),
        marker="^",
        color="#2ca02c",
        label="realised mean coverage of M2",
    )
    axis.axhline(PRIMARY_EPSILON, color="#d62728", linestyle="--", lw=1, label="epsilon = 0.10")
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(s) for s in sizes])
    axis.set_xlabel("certification labels (N_cert)")
    axis.set_ylabel("risk / coverage")
    axis.set_title("SGV15: certification resolution against the budget that buys it")
    axis.legend(fontsize=6.5)
    _finish(figure, FIGURE_DIR / "certification_resolution_vs_labels.png")

    # 7. the samplers.
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    palette = {
        CS0_UNCERTAINTY: "#d62728",
        CS1_UNIFORM: "#9467bd",
        CS2_DOCUMENT: "#1f77b4",
        CS_FEW_DOCUMENTS: "#d68910",
        CS_ONE_DOCUMENT: "#8c8c8c",
    }
    for sampler, colour in palette.items():
        values = [
            sampling["by_sampler"][sampler]
            .get(str(s), {})
            .get("mean_prevalence_absolute_error", float("nan"))
            for s in sizes
        ]
        axis.plot(sizes, values, marker="o", color=colour, label=sampler)
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(s) for s in sizes])
    axis.set_xlabel("certification labels (N_cert)")
    axis.set_ylabel("|sample harm prevalence - evaluation harm prevalence|")
    axis.set_title("SGV15: which certification sample represents the target")
    axis.legend(fontsize=6.5)
    _finish(figure, FIGURE_DIR / "sampling_strategy_comparison.png")

    # 8. risk against coverage, per environment, at the reference budget.
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for name in environments:
        base = baseline[name]
        certified = per_env[name]["C3_conservative_certified_cut"]
        axis.scatter(base["coverage"], base["realized_harm_rate"], color="#8c8c8c", s=22)
        axis.scatter(certified["coverage"], certified["realized_harm_rate"], color="#1f77b4", s=22)
        axis.annotate(
            name,
            (certified["coverage"], certified["realized_harm_rate"]),
            fontsize=5.5,
            xytext=(3, 3),
            textcoords="offset points",
        )
        if not math.isnan(certified["realized_harm_rate"]):
            axis.annotate(
                "",
                xy=(certified["coverage"], certified["realized_harm_rate"]),
                xytext=(base["coverage"], base["realized_harm_rate"]),
                arrowprops={"arrowstyle": "->", "color": "#cccccc", "lw": 0.7},
            )
    axis.axhline(PRIMARY_EPSILON, color="#d62728", linestyle="--", lw=1, label="epsilon = 0.10")
    axis.axvline(
        frozen.coverage_floor, color="#2ca02c", linestyle=":", lw=1, label="SGV13 coverage floor"
    )
    axis.scatter([], [], color="#8c8c8c", label="C0 frozen source cut")
    axis.scatter([], [], color="#1f77b4", label="C3 certified cut")
    axis.set_xlabel("accepted coverage")
    axis.set_ylabel("realised accepted-set harm rate")
    axis.set_title("SGV15: where certification moves each environment")
    axis.legend(fontsize=6.5)
    _finish(figure, FIGURE_DIR / "risk_coverage_by_environment.png")

    # 9. the cut decomposition.
    figure, axis = plt.subplots(figsize=(9.5, 4.0))
    series = (
        ("C0_frozen_source_cut", "#8c8c8c", "C0 frozen source"),
        ("C3_conservative_certified_cut", "#1f77b4", "C3 certified"),
        ("C2_empirical_target_certification_cut", "#d62728", "C2 empirical"),
        ("C1_oracle_evaluation_cut_on_grid", "#000000", "C1 oracle on grid"),
        ("C1f_oracle_evaluation_cut_fine_grid", "#555555", "C1f oracle fine grid"),
    )
    width = 0.16
    for offset, (key, colour, label) in enumerate(series):
        axis.bar(
            position + (offset - 2) * width,
            [per_env[e][key]["risk_controlled"] for e in environments],
            width,
            color=colour,
            label=label,
        )
    axis.set_xticks(position)
    axis.set_xticklabels(labels, fontsize=6.5)
    axis.set_ylabel(f"risk-controlled repair recall at eps={PRIMARY_EPSILON}")
    axis.set_title(
        "SGV15: where the achievable repair recall is lost  "
        f"(mean cost of the {len(CUT_GRID)}-point grid: "
        f"{resolution['mean_grid_resolution_gap']:+.4f})"
    )
    axis.legend(fontsize=6.5, ncol=3)
    _finish(figure, FIGURE_DIR / "cut_decomposition.png")

    # 10. the 2x2.
    figure, axis = plt.subplots(figsize=(9.5, 4.0))
    corners = (
        (X_SOURCE_RANK_SOURCE_CUT, "#8c8c8c", "source rank / source cut"),
        (X_ADAPTED_RANK_SOURCE_CUT, "#d68910", "adapted rank / source cut"),
        (X_SOURCE_RANK_CERTIFIED_CUT, "#2ca02c", "source rank / certified cut"),
        (X_ADAPTED_RANK_CERTIFIED_CUT, "#1f77b4", "adapted rank / certified cut"),
    )
    width = 0.2
    cells_2x2 = two_by_two["per_environment"]
    for offset, (key, colour, label) in enumerate(corners):
        axis.bar(
            position + (offset - 1.5) * width,
            [cells_2x2[e][key]["risk_controlled"] for e in environments],
            width,
            color=colour,
            label=label,
        )
    axis.set_xticks(position)
    axis.set_xticklabels(labels, fontsize=6.5)
    axis.set_ylabel(f"risk-controlled repair recall at eps={PRIMARY_EPSILON}")
    axis.set_title("SGV15: which component is load-bearing")
    axis.legend(fontsize=6.5, ncol=2)
    _finish(figure, FIGURE_DIR / "adaptation_certification_decomposition.png")

    # 11. the failure-mode matrix.
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    mode_index = {mode: i for i, mode in enumerate(FAILURE_MODES)}
    grid_values = np.full((len(environments), len(sizes)), np.nan)
    matrix = feasibility["matrix"][M2_CLOPPER]
    for column, size in enumerate(sizes):
        entry = matrix.get(str(size), {})
        for row, name in enumerate(environments):
            if name in entry:
                grid_values[row, column] = mode_index[entry[name]]
    colours = ["#cccccc", "#f0c674", "#2ca02c", "#d62728"]
    from matplotlib.colors import BoundaryNorm, ListedColormap

    axis.imshow(
        grid_values,
        cmap=ListedColormap(colours),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4),
        aspect="auto",
    )
    axis.set_xticks(range(len(sizes)))
    axis.set_xticklabels([str(s) for s in sizes])
    axis.set_yticks(range(len(environments)))
    axis.set_yticklabels(environments, fontsize=6.5)
    axis.set_xlabel("certification labels (N_cert)")
    axis.set_title(f"SGV15: dominant outcome per environment, {METHOD_LABEL[M2_CLOPPER]}")
    axis.grid(visible=False)
    handles = [
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=colours[i], markersize=8, label=mode
        )
        for i, mode in enumerate(FAILURE_MODES)
    ]
    axis.legend(handles=handles, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _finish(figure, FIGURE_DIR / "failure_mode_matrix.png")

    sources = (
        CERTIFICATION_BUDGET_RESULTS,
        BUDGET_ALLOCATION_RESULTS,
        ADAPTATION_BUDGET_RESULTS,
        CERTIFICATION_SAMPLING,
        CUT_DECOMPOSITION,
        ADAPTATION_CERTIFICATION_DECOMPOSITION,
        OUT / "certification_feasibility.json",
        OUT / "grid_resolution_analysis.json",
        SAMPLE_SIZE_ANALYSIS,
        DESIGN_RECORD,
    )
    _write_json_once(
        FIGURE_MANIFEST,
        {
            "artifact": "figure_manifest",
            "schema_version": f"sgv15-figure_manifest-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "synthetic": False,
            "scientific_status": "DEVELOPMENT -- SGV14-exposed environments",
            "figures": {
                _relative(path): {
                    "sha256": file_sha256(path),
                    "derived_from": [_relative(p) for p in sources],
                    "source_sha256": {_relative(p): file_sha256(p) for p in sources},
                }
                for path in written
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {_relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ sections 13/14: the decision


def run_decide() -> int:
    """The six frozen criteria, the outcome taxonomy, and whether SGV16 is justified."""
    started = time.monotonic()
    cells, design = load_cells()
    frozen = load_frozen()
    budgets = cc_read_json(CERTIFICATION_BUDGET_RESULTS)
    allocations = cc_read_json(BUDGET_ALLOCATION_RESULTS)["allocations"]
    sampling = cc_read_json(CERTIFICATION_SAMPLING)
    two_by_two = cc_read_json(ADAPTATION_CERTIFICATION_DECOMPOSITION)
    feasibility = cc_read_json(OUT / "certification_feasibility.json")
    statistics = cc_read_json(STATISTICAL_TESTS)
    selection = cc_read_json(METHOD_SELECTION)
    controls = cc_read_json(CONTROL_RESULTS)
    negatives = cc_read_json(NEGATIVE_TESTS)
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    method, sampler, size = selected_configuration()
    environments = list(design["environments"])
    baseline = budgets["baseline"]

    block = primary_slice(cells, "A", method, sampler=sampler, n_cert=size)
    rows = _per_environment(block)

    safety = [name for name in environments if name in rows and holds_the_bound(rows[name])]
    baseline_safety = [name for name in environments if holds_the_bound(baseline[name])]
    gain = [
        name
        for name in environments
        if name in rows and rows[name]["risk_controlled"] > baseline[name]["risk_controlled"]
    ]
    non_degenerate = [
        name
        for name in environments
        if name in rows and rows[name]["coverage"] >= frozen.coverage_floor
    ]

    representative = sampling["comparison"].get(str(size), {})
    criterion_4 = bool(representative.get("representative_beats_uncertainty_on_endpoint", False))

    corners = two_by_two["per_environment"]
    load_bearing = [
        name
        for name in environments
        if corners[name][X_ADAPTED_RANK_CERTIFIED_CUT]["risk_controlled"]
        > corners[name][X_ADAPTED_RANK_SOURCE_CUT]["risk_controlled"]
    ]

    feasible_allocations = {
        label: entry
        for label, entry in allocations.items()
        if entry["environments_holding_the_bound"] >= BREADTH_REQUIRED
        and entry["environments_beating_the_baseline"] >= BREADTH_REQUIRED
    }

    criteria = {
        "1_safety_recovery": {
            "requirement": design["criteria"]["1_safety_recovery"],
            "required": BREADTH_REQUIRED,
            "observed": len(safety),
            "environments": safety,
            "baseline_observed": len(baseline_safety),
            "met": len(safety) >= BREADTH_REQUIRED,
        },
        "2_operational_gain": {
            "requirement": design["criteria"]["2_operational_gain"],
            "required": BREADTH_REQUIRED,
            "observed": len(gain),
            "environments": gain,
            "met": len(gain) >= BREADTH_REQUIRED,
        },
        "3_non_degeneracy": {
            "requirement": design["criteria"]["3_non_degeneracy"],
            "required": BREADTH_REQUIRED,
            "coverage_floor": frozen.coverage_floor,
            "observed": len(non_degenerate),
            "environments": non_degenerate,
            "met": len(non_degenerate) >= BREADTH_REQUIRED,
        },
        "4_representative_sampling_matters": {
            "requirement": design["criteria"]["4_representative_sampling_matters"],
            "observed": representative,
            "met": criterion_4,
        },
        "5_cut_mechanism_is_load_bearing": {
            "requirement": design["criteria"]["5_cut_mechanism_is_load_bearing"],
            "required": BREADTH_REQUIRED,
            "observed": len(load_bearing),
            "environments": load_bearing,
            "met": len(load_bearing) >= BREADTH_REQUIRED,
        },
        "6_fixed_budget_feasibility": {
            "requirement": design["criteria"]["6_fixed_budget_feasibility"],
            "observed": sorted(feasible_allocations),
            "met": len(feasible_allocations) > 0,
        },
    }
    met = sum(1 for entry in criteria.values() if entry["met"])
    verdict = "SUPPORTED" if met == len(criteria) else "NOT SUPPORTED"

    # The outcome taxonomy of section 14, evaluated in its own fixed order.
    largest = max(CERT_BUDGETS)
    curve = budgets["by_method"][M2_CLOPPER]
    feasible_at = {
        int(s): curve[str(s)]["environments_holding_the_bound"] for s in CERT_BUDGETS if s > 0
    }
    non_degenerate_at = {
        int(s): curve[str(s)]["environments_above_the_coverage_floor"]
        for s in CERT_BUDGETS
        if s > 0
    }
    # Section 14's taxonomy. The verdict rule is frozen in the design record; this mapping is
    # not, so it is implemented directly from the stage brief's own definitions and each branch
    # records the evidence that decided it. "Certification works" is read strictly: a budget only
    # counts if it is SAFE and NON-DEGENERATE at the required breadth, because safety bought by
    # refusing everything is exactly what criterion 3 exists to reject.
    works_within_250 = len(feasible_allocations) > 0
    justified_budgets = [
        budget
        for budget in sorted(feasible_at)
        if feasible_at[budget] >= BREADTH_REQUIRED
        and non_degenerate_at.get(budget, 0) >= BREADTH_REQUIRED
    ]
    sampler_view = sampling["comparison"].get(str(size), {}).get("by_sampler", {})
    random_matches_uncertainty = bool(
        sampler_view.get(CS1_UNIFORM, {}).get("mean_risk_controlled", float("-inf"))
        >= sampler_view.get(CS0_UNCERTAINTY, {}).get("mean_risk_controlled", float("inf"))
    )
    adaptation_alone = allocations.get(f"{TOTAL_BUDGET}+0", {})
    adaptation_alone_is_non_degenerate = bool(
        adaptation_alone.get("environments_above_the_coverage_floor", 0) >= BREADTH_REQUIRED
    )
    # "adaptation works" cannot mean "deploys widely": this stage's deployment constraint is
    # Harm <= epsilon, so a configuration that violates it on most environments has not worked,
    # however much of the ranking it accepts.
    adaptation_alone_is_safe = bool(
        adaptation_alone.get("environments_holding_the_bound", 0) >= BREADTH_REQUIRED
    )
    taxonomy = {
        "A_fixed_250_is_sufficient": bool(works_within_250 and met == len(criteria)),
        "B_certification_works_at_a_larger_budget": bool(justified_budgets),
        "C_adaptation_works_and_acquisition_is_unnecessary": bool(
            random_matches_uncertainty
            and adaptation_alone_is_non_degenerate
            and adaptation_alone_is_safe
            and not justified_budgets
        ),
        "evidence": {
            "allocations_meeting_criteria_1_and_2": sorted(feasible_allocations),
            "budgets_safe_and_non_degenerate_at_breadth": justified_budgets,
            "largest_budget_tested": largest,
            "non_degenerate_environments_at_the_largest_budget": non_degenerate_at.get(largest, 0),
            "breadth_required": BREADTH_REQUIRED,
            "random_matches_uncertainty_certification": random_matches_uncertainty,
            "adaptation_alone_is_non_degenerate": adaptation_alone_is_non_degenerate,
            "adaptation_alone_is_safe": adaptation_alone_is_safe,
            "adaptation_alone_environments_holding_epsilon": adaptation_alone.get(
                "environments_holding_the_bound", 0
            ),
            "adaptation_acquisition_comparison_not_run_here": (
                "the document-random adaptation acquisition arm is registered SECONDARY and was "
                "not run in SGV15; SGV14 measured it and found random matched or beat "
                "q2_uncertainty on 7 of 8 relevant comparisons"
            ),
        },
    }
    if taxonomy["A_fixed_250_is_sufficient"]:
        outcome, label = "A", "a fixed 250-label budget is sufficient"
    elif taxonomy["B_certification_works_at_a_larger_budget"]:
        outcome, label = (
            "B",
            f"certification works from N_cert = {justified_budgets[0] if justified_budgets else 0}"
            f", which is more than the {TOTAL_BUDGET}-label budget SGV13 and SGV14 used",
        )
    elif taxonomy["C_adaptation_works_and_acquisition_is_unnecessary"]:
        outcome, label = "C", "adaptation works but sophisticated acquisition is unnecessary"
    else:
        outcome, label = (
            "D",
            "representative target evidence controls harm but cannot preserve meaningful "
            "coverage at any declared budget",
        )

    ready = bool(verdict == "SUPPORTED" and outcome in ("A", "B"))
    smallest = cc_read_json(SAMPLE_SIZE_ANALYSIS)[
        "smallest_sample_certifying_zero_harm_below_epsilon"
    ]
    empirical_minimum = justified_budgets[0] if justified_budgets else None
    false_certifications = {
        str(budget): entry["environments_with_a_false_certification"]
        for budget, entry in feasibility["by_method"][method].items()
    }
    false_certifications_certified_method = {
        str(budget): entry["environments_with_a_false_certification"]
        for budget, entry in feasibility["by_method"][M2_CLOPPER].items()
    }

    payload = {
        "artifact": "research_decision",
        "schema_version": f"sgv15-research_decision-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git("rev-parse", "HEAD"),
        "synthetic": False,
        "scientific_status": {
            "kind": "DEVELOPMENT",
            "why": design["scientific_status"]["why"],
            "prohibited_language": [
                "externally confirmed",
                "generally validated",
                "safe deployment solved",
                "universal reliability achieved",
            ],
            "confirmatory_stage": design["scientific_status"]["confirmatory_stage"],
        },
        "verdict": verdict,
        "criteria": criteria,
        "criteria_met": met,
        "criteria_total": len(criteria),
        "outcome": {"label": outcome, "summary": label, "taxonomy": taxonomy},
        "selected_method": method,
        "selected_budget": size,
        "selected_sampler": sampler,
        "selected_certification_rule": (
            f"the deepest of the {len(CUT_GRID)} declared acceptance depths whose exact one-sided "
            f"Clopper-Pearson upper bound on the certification sample is at most epsilon, at "
            f"alpha = {CERT_DELTA} split across the grid by union bound"
            if method == M2_CLOPPER
            else design["methods"][method]
        ),
        "selected_allocation": selection.get("selected_allocation"),
        "selection_is_stable": selection["selection_is_stable"],
        "primary_epsilon": PRIMARY_EPSILON,
        "alpha": CERT_DELTA,
        "cut_grid": list(CUT_GRID),
        "capacity_caveats": {
            "capacity_limited_environments": inventory["capacity_limited"],
            "effect_on_the_decision": {
                "criterion_1_excluding_them": len(
                    [e for e in safety if e not in inventory["capacity_limited"]]
                ),
                "criterion_2_excluding_them": len(
                    [e for e in gain if e not in inventory["capacity_limited"]]
                ),
                "criterion_3_excluding_them": len(
                    [e for e in non_degenerate if e not in inventory["capacity_limited"]]
                ),
            },
            "note": (
                "no criterion in this stage turns on a capacity-limited environment: the counts "
                "with and without them fall on the same side of the threshold. That is reported "
                "rather than assumed."
            ),
        },
        "statistical_support": {
            "delta": statistics["primary_endpoint_repair_recall"]["delta"],
            "ci_lower": statistics["primary_endpoint_repair_recall"]["ci_lower"],
            "ci_upper": statistics["primary_endpoint_repair_recall"]["ci_upper"],
            "p_value": statistics["primary_endpoint_repair_recall"]["p_value"],
            "harm_delta": statistics["primary_endpoint_realized_harm"]["delta"],
            "harm_ci_lower": statistics["primary_endpoint_realized_harm"]["ci_lower"],
            "harm_ci_upper": statistics["primary_endpoint_realized_harm"]["ci_upper"],
            "harm_p_value": statistics["primary_endpoint_realized_harm"]["p_value"],
            "surviving_after_holm": statistics["count_surviving"],
            "family_size": statistics["family_size"],
        },
        "nondegeneracy_status": {
            "coverage_floor": frozen.coverage_floor,
            "environments_above_the_floor": len(non_degenerate),
            "mean_coverage": _mean([r["coverage"] for r in rows.values()]),
            "baseline_mean_coverage": _mean([r["coverage"] for r in baseline.values()]),
            "refusal_rate": feasibility["by_method"][method][str(size)]["refusal_rate"],
        },
        "safety_summary": {
            "environments_holding_epsilon": len(safety),
            "environments_violating_epsilon": len(environments) - len(safety),
            "baseline_environments_holding_epsilon": len(baseline_safety),
            "false_certifications_by_budget": false_certifications,
            "false_certifications_by_budget_note": (
                f"the SELECTED method is {method}; these are its counts. The conservative "
                "certified method's counts are reported alongside because the difference between "
                "them is the whole safety argument for a bound."
            ),
            "false_certifications_by_budget_certified_method": (
                false_certifications_certified_method
            ),
            "refusals_by_budget": {
                s: entry["refusal_rate"] for s, entry in feasibility["by_method"][method].items()
            },
        },
        "sample_complexity": {
            "a_priori_minimum_alpha": smallest["alpha"],
            "a_priori_minimum_grid_corrected": smallest["alpha_over_grid"],
            "empirical_minimum_for_breadth": empirical_minimum,
            "environments_holding_the_bound_by_budget": feasible_at,
            "environments_above_the_coverage_floor_by_budget": non_degenerate_at,
        },
        "controls_summary": controls["verdicts"],
        "negative_tests": {
            "passed": negatives["passed"],
            "total": negatives["total"],
            "failed": negatives["failed"],
        },
        "confirmatory_reserve_consumed": False,
        "ready_for_sgv16": ready,
        "reason": (
            f"{verdict} on {met} of {len(criteria)} frozen development criteria; outcome "
            f"{outcome} -- {label}."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DECISION, payload)
    print(
        f"decide: {verdict} -- {met}/{len(criteria)} criteria, outcome {outcome}, "
        f"ready_for_sgv16={ready} -> {_relative(DECISION)}"
    )
    return 0


# ------------------------------------------------------------------ provenance and the cli

DETERMINISM = OUT / "determinism.json"
GRID_RESOLUTION = OUT / "grid_resolution_analysis.json"
CERTIFICATION_FEASIBILITY = OUT / "certification_feasibility.json"
CROSS_FITTED_DIAGNOSTICS = OUT / "cross_fitted_diagnostics.json"

PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    ENVIRONMENT_INVENTORY,
    DESIGN_RECORD,
    SAMPLE_SIZE_ANALYSIS,
    CERTIFICATION_BUDGET_RESULTS,
    BUDGET_ALLOCATION_RESULTS,
    ADAPTATION_BUDGET_RESULTS,
    CERTIFICATION_SAMPLING,
    CUT_RESULTS,
    CUT_DECOMPOSITION,
    GRID_RESOLUTION,
    ADAPTATION_CERTIFICATION_DECOMPOSITION,
    CERTIFICATION_FEASIBILITY,
    RISK_BOUND_RESULTS,
    CONTROL_RESULTS,
    NEGATIVE_TESTS,
    METHOD_SELECTION,
    STATISTICAL_TESTS,
    CROSS_FITTED_DIAGNOSTICS,
    DETERMINISM,
    DECISION,
    FIGURE_MANIFEST,
    CERTIFICATION_ROWS,
    CELL_TABLE,
    DEPLOYMENT_COUNTS,
)

DEPENDENCY_SCRIPTS = UPSTREAM_SCRIPTS

REPORT_SECTIONS = {
    "1. Motivation": (FROZEN_CONFIGURATION, s14.DECISION),
    "2. Difference from SGV9": (DESIGN_RECORD, s9.DECISION),
    "3. SGV14 diagnosis": (s14.DECISION, s14.PRIMARY_RESULTS, s14.CONTROL_RESULTS),
    "4. Research questions": (DESIGN_RECORD,),
    "5. Frozen adaptation model": (FROZEN_CONFIGURATION, RESEARCH_FREEZE),
    "6. Adaptation versus certification labels": (DESIGN_RECORD, ENVIRONMENT_INVENTORY),
    "7. Document-level partition": (ENVIRONMENT_INVENTORY,),
    "8. Capacity constraints": (ENVIRONMENT_INVENTORY,),
    "9. Certification sampling": (CERTIFICATION_SAMPLING, DESIGN_RECORD),
    "10. Risk-bound methods": (DESIGN_RECORD, RISK_BOUND_RESULTS),
    "11. A-priori sample complexity": (SAMPLE_SIZE_ANALYSIS,),
    "12. Experiment A": (CERTIFICATION_BUDGET_RESULTS, CERTIFICATION_FEASIBILITY),
    "13. Experiment B": (BUDGET_ALLOCATION_RESULTS, ADAPTATION_BUDGET_RESULTS),
    "14. Main operational results": (CERTIFICATION_BUDGET_RESULTS, CUT_RESULTS),
    "15. Safety analysis": (CERTIFICATION_FEASIBILITY, CERTIFICATION_BUDGET_RESULTS),
    "16. Certification feasibility": (CERTIFICATION_FEASIBILITY, SAMPLE_SIZE_ANALYSIS),
    "17. Coverage and non-degeneracy": (CERTIFICATION_BUDGET_RESULTS, CERTIFICATION_FEASIBILITY),
    "18. Cut decomposition": (CUT_DECOMPOSITION, CUT_RESULTS),
    "19. Grid-resolution cost": (GRID_RESOLUTION, CUT_DECOMPOSITION),
    "20. Adaptation versus certification": (ADAPTATION_CERTIFICATION_DECOMPOSITION,),
    "21. Sampling strategy analysis": (CERTIFICATION_SAMPLING,),
    "22. Sample-complexity analysis": (SAMPLE_SIZE_ANALYSIS, CERTIFICATION_FEASIBILITY),
    "23. Controls": (CONTROL_RESULTS,),
    "24. Negative tests": (NEGATIVE_TESTS,),
    "25. Statistical analysis": (STATISTICAL_TESTS,),
    "26. Capacity-limited environments": (ENVIRONMENT_INVENTORY, DECISION),
    "27. Limitations": (DECISION, RISK_BOUND_RESULTS, GRID_RESOLUTION),
    "28. Method selection": (METHOD_SELECTION, CROSS_FITTED_DIAGNOSTICS),
    "29. Research decision": (DECISION, DETERMINISM),
    "30. Implications for SGV16": (DECISION, METHOD_SELECTION),
    "Artifacts": (PROVENANCE, TRACEABILITY, FIGURE_MANIFEST),
}


def run_record() -> int:
    """Provenance, the dependency audit and the traceability index."""
    started = time.monotonic()
    missing = [_relative(p) for p in PRODUCED if not p.is_file()]
    if missing:
        raise PhaseError(f"cannot record an incomplete stage; missing {missing}")
    freeze = cc_read_json(RESEARCH_FREEZE)
    inputs = dict(cc_read_json(s14.PROVENANCE)["artifacts"])
    inputs.update(freeze["upstream_script_sha256"])
    figures = cc_read_json(FIGURE_MANIFEST)["figures"]
    artifacts = {_relative(p): file_sha256(p) for p in PRODUCED}
    artifacts.update({path: record["sha256"] for path, record in figures.items()})
    ignored = sorted(path for path in artifacts if _ignored(REPO / path))

    _write_json_once(
        PROVENANCE,
        {
            "artifact": "provenance",
            "schema_version": f"sgv15-provenance-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "stage_title": "SGV15 -- representative target risk certification",
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": _git("rev-parse", "HEAD"),
            "synthetic": False,
            "scientific_status": "DEVELOPMENT",
            "confirmatory_reserve_accessed": False,
            "sgv14_outputs_modified": False,
            "features_added": 0,
            "ranking_models_introduced": 0,
            "models_fitted_note": (
                "SGV15 fits SGV13's frozen a4_joint_refit arm and nothing else. The stage "
                "develops a CUT rule, not a ranking model; the feature universe, model class, "
                "preprocessing, candidate universe and label definitions are SGV13's, unchanged."
            ),
            "risk_library_extensions": {
                "src/ocr_risk/risk/bounds.py": [
                    "clopper_pearson_upper",
                    "beta_binomial_upper",
                ],
                "src/ocr_risk/risk/controller.py": [
                    "ltt_clopper_pearson",
                    "ltt_beta_binomial",
                    "thresholds=",
                    "prior=",
                ],
                "why_in_the_library": (
                    "risk control is defined once under risk/ and imported. A stage-local bound "
                    "would be a second implementation of a project invariant."
                ),
            },
            "inputs": inputs,
            "artifacts": artifacts,
            "table_rows": {
                _relative(path): len(pd.read_parquet(path, columns=[column]))
                for path, column in (
                    (CERTIFICATION_ROWS, "environment"),
                    (CELL_TABLE, "environment"),
                    (DEPLOYMENT_COUNTS, "cell"),
                )
            },
            "dependency_audit": {
                "scripts": {
                    name: file_sha256(REPO / name)
                    for name in DEPENDENCY_SCRIPTS
                    if (REPO / name).is_file()
                },
                "depth": len(DEPENDENCY_SCRIPTS),
                "clean_clone": {
                    "committed": (
                        "nothing. The working tree carries this stage and fourteen untracked "
                        "predecessors; what to commit and in what order is not a decision this "
                        "stage may take."
                    ),
                    "scripts_missing_from_a_clean_clone": sorted(
                        name for name in DEPENDENCY_SCRIPTS if not _tracked(REPO / name)
                    ),
                    "upstream_artifacts_missing_from_a_clean_clone": sorted(
                        path
                        for path in inputs
                        if not _tracked(REPO / path) and not _ignored(REPO / path)
                    ),
                    "outputs_gitignored_but_regenerable": ignored,
                    "new_raw_written_by_this_stage": [],
                },
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        TRACEABILITY,
        {
            "artifact": "traceability",
            "schema_version": f"sgv15-traceability-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "report": "docs/sgv15/target_risk_certification.md",
            "sections": {
                name: [_relative(path) for path in paths] for name, paths in REPORT_SECTIONS.items()
            },
            "rule": (
                "every decimal in the report is read from one of the artifacts its section "
                "names. A number that cannot be traced to one does not belong in the report."
            ),
        },
    )
    print(f"record: {len(artifacts)} artifacts -> {_relative(PROVENANCE)}")
    return 0


PHASES = (
    ("reconstruct", "the section-0 gate on SGV14 being closed"),
    ("freeze", "the frozen SGV14 adaptation configuration"),
    ("partition", "the document-level fitting / certification split and its capacity limits"),
    ("preregister", "the methods, samplers, grid, budgets, criteria and selection plan"),
    ("samplesize", "what each certification budget can certify, before any endpoint"),
    ("experiment-a", "adaptation fixed at 250, certification budgets added"),
    ("experiment-b", "a fixed total of 250 split between adaptation and certification"),
    ("mechanism", "the 2x2, the fine-grid oracle and the two permutation controls"),
    ("crossfit", "the registered secondary cross-fitted arm"),
    ("results", "the two experiments read as endpoints"),
    ("sampling", "the certification sampler comparison"),
    ("decompose", "C0-C3, the grid cost and the adaptation-versus-certification 2x2"),
    ("feasibility", "the failure-mode matrix and the document-clustering audit"),
    ("controls", "the twelve controls"),
    ("negative", "the fifteen falsification tests"),
    ("select", "nested leave-one-environment-out method selection"),
    ("stats", "the pre-registered primary family"),
    ("figures", "the eleven required figures"),
    ("decide", "the machine-readable finding"),
    ("record", "provenance and traceability"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name, help_text in PHASES:
        parser.add_argument(f"--{name}", action="store_true", help=help_text)
    parser.add_argument(
        "--only", default=None, help="restrict an experiment phase to one environment"
    )
    arguments = parser.parse_args()
    selected = [name for name, _ in PHASES if getattr(arguments, name.replace("-", "_"))]
    if not selected:
        parser.print_help()
        return 2
    handlers: dict[str, Any] = {
        "reconstruct": run_reconstruct,
        "freeze": run_freeze,
        "partition": run_partition,
        "preregister": run_preregister,
        "samplesize": run_samplesize,
        "experiment-a": lambda: run_experiment_a(arguments.only),
        "experiment-b": lambda: run_experiment_b(arguments.only),
        "mechanism": lambda: run_mechanism(arguments.only),
        "crossfit": lambda: run_crossfit(arguments.only),
        "results": run_certification_results,
        "sampling": run_sampling,
        "decompose": run_decompose,
        "feasibility": run_feasibility,
        "controls": run_controls,
        "negative": run_negative,
        "select": run_select,
        "stats": run_stats,
        "figures": run_figures,
        "decide": run_decide,
        "record": run_record,
    }
    for name in selected:
        status = handlers[name]()
        if status:
            return int(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
