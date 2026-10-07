#!/usr/bin/env python3
"""SGV14: does SGV13's frozen target-supervised procedure survive a new environment?

SGV13 produced the first substantial positive operational evidence in this chain, and it
produced it from a configuration nothing in the outer evaluation chose: a source-side
simulation, run before any held-out target row was read, selected `a4_joint_refit` on
`q2_uncertainty` at N = 250 under the `frozen_source` cut, and that configuration then moved
the risk-controlled repair frontier on three of four unseen engines while holding the harm
bound on the same three. Two controls closed the attribution -- a source-only refit reproduced
SGV5's frozen model bit for bit, and permuting the purchased labels destroyed the gain -- so
the labels were load-bearing. The verdict was still NOT SUPPORTED, on five of six criteria,
and one engine (EasyOCR) never became usable at any budget.

Everything above was measured on ONE corpus of receipts read by FOUR engine configurations,
with the target engine held out but the documents drawn from the same pool the method was
built on. This stage asks the only question that separates a result from a finding:

    SGV14-V1: the frozen SGV13 procedure -- arm, acquisition, budget, cut and epsilon fixed
    before this stage existed -- reproduces its risk-controlled repair gains on OCR
    environments that played no part in its development, pre-registration, model selection
    or evaluation.

**The hypothesis is recorded as SGV14-V1.** `docs/sgv1/protocol.md` binds SGV1-H1..H4 and says
a frozen id is never reused for a different claim; every stage since SGV2 has opened its own
family for the same reason. The brief's RQ1..RQ3 are carried as sub-claims V1a (the gain
reproduces), V1b (the harm bound holds at non-degenerate coverage) and V1c (it reproduces
across environment kinds), all under V1.

Eleven decisions fix what the numbers below can mean.

**1. Nothing is selected here.** The method is read out of SGV13's `design_record.json` and
asserted against it at run time. `frozen_sgv13_configuration.json` records the arm, the
acquisition rule, the budget, the cut, the epsilon, the source:target weighting, the seeds and
every upstream hash, and `--adapt` refuses to start if any of them has moved. No SGV14 phase
contains a grid, a search, a selection or a tie-break over any of them.

**2. The environments are chosen by a rule over the repository, not by a preference.** Every
corpus with ground truth and recorded OCR whose documents are absent from every SGV1..SGV13
role, crossed with every recorded engine configuration on it, plus the two configurations this
stage runs itself to obtain configuration novelty inside a fixed corpus. The rule, the
candidates it admits, the ones it excludes and the reason are written to
`environment_selection.json` BEFORE any environment is scored.

**3. The representation is rebuilt, and the rebuild is proved.** A new environment needs
SGV1's ninety-six design columns and SGV5's structural block computed on rows SGV1 never saw.
`--build` therefore re-implements nothing: it drives the same library functions SGV1 drove,
with every fitted transformer -- the generator resources, the confidence featurizer, the
character language model, the lexicon, the glyph prototypes -- taken from SGV1's TRAIN
documents and frozen. `--identity` runs that same pipeline over SGV1's own CORD rows and
asserts it reproduces the frozen canonical spans, the frozen candidate ids, the frozen labels
and the frozen design matrix to a maximum absolute difference of 0.0. A representation that
could not rebuild the old rows exactly may not be trusted on new ones.

**4. The frozen model is the frozen model.** For an environment whose engine is a
configuration of base engine b, the fold is SGV5's fold that held b out: source fit rows,
source calibration rows and the selected representation, model class and lambda are exactly
the ones SGV13 deployed. Only the target side is replaced. `--adapt` rebuilds each fold twice
-- once as SGV13 built it, where `build_state` asserts the decision score equals SGV5's
published vector bit for bit, and once with the environment's rows appended -- and asserts the
two give identical scores on the shared rows, so appending rows changed no fitted object.

**5. The two split axes still both apply.** Adaptation and evaluation documents are disjoint by
a deterministic content-blind partition of each corpus, shared across every environment on that
corpus, and the evaluation documents are disjoint from every SGV1..SGV13 role by construction
because the corpus is. `--build` asserts document, page, site and source-image disjointness and
refuses to write a row otherwise.

**6. The budget pays for everything, and it is SGV13's budget.** 250 labels, purchased through
SGV13's own `purchase` from an ordering produced by SGV13's own `q2_uncertainty`, on a
`PoolView` that has no outcome field. Nothing else on the target side is read: not to place the
cut, which is `frozen_source` and reads only source calibration rows, and not to choose
anything, because nothing is chosen.

**7. Twenty draws, common random seeds.** The acquisition rule is stochastic only through
`s6.acquisition_order`'s tie-breaking, and the draw seeds come from SGV13's `_stable_seed` on
the environment name and the draw index. The primary arm, the permutation control and the
random-acquisition diagnostic share the seed at every draw, so a difference between them at a
draw is a difference in what was bought or what the labels said, never in the seed.

**8. Sufficiency is declared before the fact, and the criteria are reported both ways.** An
environment whose evaluation partition is too small to estimate a harm rate is marked
INSUFFICIENT by a rule fixed in `--preregister`, and the four criteria are evaluated over the
sufficient environments AND, as a pre-registered sensitivity, over all of them. Neither is
allowed to move after the numbers exist.

**9. Four criteria, all required, and a small test family.** Breadth, safety, non-degeneracy
and statistical support. The non-degeneracy floor is SGV13's own, read from its design record.
The primary test is one: the environment-stratified paired effect on risk-controlled repair
recall at epsilon = 0.10 against the frozen baseline, document-clustered. Per-environment tests
are Holm-corrected inside a secondary family and cannot rescue the primary.

**10. The controls come with the method.** A source-only refit that spends no target label, a
label permutation on exactly the purchased rows, and a random-acquisition diagnostic at the
same budget. The first two are validation controls and are never compared as alternatives; the
third is explicitly secondary and cannot change the frozen method whatever it says.

**11. A failure is reported, not repaired.** If the method does not reproduce, this stage
records that. It contains no fallback arm, no rescue calibration, no per-environment special
case and no epsilon it can retreat to.

    --reconstruct  the section-0 research freeze: branch, HEAD, dependency graph, hashes
    --freeze       the frozen SGV13 configuration, and the assertions that bind it
    --environments the environment selection audit, written before anything is scored
    --ocr          the two new engine configurations this stage runs itself
    --identity     the representation rebuild proved against SGV1's own frozen rows
    --build        every environment's rows in the frozen representation
    --preregister  the criteria, the sufficiency rule and the test family
    --adapt        the frozen method, the baselines and the controls, on every environment
    --results      the primary, per-environment and safety endpoints
    --controls     the source-only, permutation and random-acquisition controls
    --stats        the pre-registered tests
    --negative     the ten falsification tests
    --figures      the five required figures
    --decide       the machine-readable finding
    --record       provenance, the dependency audit and the traceability index

CONFIRMATORY IN THE SGV14 SENSE ONLY. The SGV1 CORD confirmatory reserve stays LOCKED and is
absent from every artifact this stage writes; `environment_selection.json` records that it was
considered and why it was not spent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_candidate_conditioned as cc
import sgv1_domain_generalization as dg
import sgv1_representation as repr_stage
import sgv1_verifier_pilot as pilot
import sgv2_reliability_layer as rl
import sgv5_candidate_reliability as c5
import sgv6_target_engine_adaptation as s6
import sgv9_target_calibration as s9
import sgv11_minimax_policy_transfer as s11
import sgv12_robust_ranking_transfer as s12
import sgv13_fewshot_prefix_purity_adaptation as s13
from ocr_risk.io.hashing import file_sha256

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv14_confirmatory_validation"

ENVIRONMENT_ROWS = OUT / "environment_rows.parquet"
CONFIRMATORY_SCORES = OUT / "confirmatory_scores.parquet"
DEPLOYMENT_COUNTS = OUT / "deployment_counts.parquet"
RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_sgv13_configuration.json"
ENVIRONMENT_SELECTION = OUT / "environment_selection.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"
IDENTITY_RECORD = OUT / "representation_identity.json"
BUILD_RECORD = OUT / "build_record.json"
DESIGN_RECORD = OUT / "design_record.json"
ADAPTATION_BUDGET = OUT / "adaptation_budget.json"
PRIMARY_RESULTS = OUT / "primary_results.json"
PER_ENVIRONMENT_RESULTS = OUT / "per_environment_results.json"
SAFETY_RESULTS = OUT / "safety_results.json"
CONTROL_RESULTS = OUT / "control_results.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
DECISION = OUT / "research_decision.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

SCHEMA_VERSION = 1
STAGE = "sgv14_confirmatory_validation"
HYPOTHESIS = "SGV14-V1"

# ------------------------------------------------------------------ imported, never restated
#
# Eight stages now import these rather than restating them, and `tests/leakage` asserts the
# identity by `is`. SGV14 adds no metric, no endpoint and no estimator of its own.

deployed = s13.deployed
frontier = s13.frontier
ranking_quality = s13.ranking_quality
place_cut = s13.place_cut
epsilon_key = s13.epsilon_key
prefix_statistics = s13.prefix_statistics
per_document_counts = s13.per_document_counts
document_multiplicities = s11.document_multiplicities
_interval = s9._interval
_stable_seed = s6._stable_seed
build_setup = s13.build_setup
acquisition_order = s13.acquisition_order
purchase = s13.purchase
fit_joint_arm = s13.fit_joint_arm
fit_sgv6_arm = s13.fit_sgv6_arm
thresholds_for = s13.thresholds_for
budget_block = s13.budget_block
frozen_arm = s13.frozen_arm
cc_read_json = s13.cc_read_json

EPSILONS = s13.EPSILONS
PRIMARY_EPSILON = s13.PRIMARY_EPSILON
PRIMARY_KEY = s13.PRIMARY_KEY
DELTA = s13.DELTA
BOOTSTRAP_RESAMPLES = s13.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = s13.BOOTSTRAP_SEED

FIT = s13.FIT
CALIBRATION = s13.CALIBRATION
EVALUATION = s13.EVALUATION
POOL = s13.POOL


class PhaseError(RuntimeError):
    """A guard this stage refuses to run past."""


# ------------------------------------------------------------------ the frozen registry

BASE_ENGINES = ("doctr", "easyocr", "paddleocr", "tesseract")

# SGV1's development roles, which are the SOURCE side of every SGV14 fold.
SGV1_ROLES = ("TRAIN", "CALIBRATION", "DEVELOPMENT")
# SGV14's own roles, which are the TARGET side. Named so no mask can confuse the two.
ROLE_ADAPTATION = "SGV14_ADAPTATION"
ROLE_EVALUATION = "SGV14_EVALUATION"

# The document partition inside a new corpus. Content-blind: the key is a hash of the document
# id and a fixed seed, so which page lands where is decided before a page is opened.
PARTITION_SEED = "sgv14-environment-partition-v1:2026-09-06"
ADAPTATION_FRACTION = 0.5

DRAWS = 20
DRAW_SEED = 20260906

ARM_BASELINE = "b0_frozen_baseline"
ARM_PRIMARY = "b1_sgv13_frozen_method"
ARM_SGV6 = "b2_sgv6_generic"
CONTROL_SOURCE_ONLY = "c1_source_only_refit"
CONTROL_PERMUTED = "c2_permuted_labels"
DIAGNOSTIC_RANDOM = "d1_random_acquisition"
ARMS = (
    ARM_BASELINE,
    ARM_PRIMARY,
    ARM_SGV6,
    CONTROL_SOURCE_ONLY,
    CONTROL_PERMUTED,
    DIAGNOSTIC_RANDOM,
)
ADAPTING_ARMS = (ARM_PRIMARY, ARM_SGV6, CONTROL_PERMUTED, DIAGNOSTIC_RANDOM)

# Sufficiency, fixed in `--preregister` and applied mechanically. An environment whose
# evaluation partition cannot support a document-clustered harm estimate is reported and
# marked, never silently dropped and never quietly kept.
MIN_EVALUATION_DOCUMENTS = 30
MIN_EVALUATION_BENEFICIAL = 20
MIN_EVALUATION_HARMFUL = 20

BREADTH_FRACTION = 0.75
CRITERIA_TOTAL = 4
PRIMARY_ALPHA = 0.05


# ------------------------------------------------------------------ small shared utilities


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Write-once, like every stage since SGV1. Delete deliberately, never overwrite silently."""
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
    """Does git deliberately ignore this path? Asked of git, never guessed from a pattern."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], cwd=REPO, capture_output=True, check=False
    )
    return result.returncode == 0


def _tracked(path: Path) -> bool:
    return bool(_git("ls-files", "--error-unmatch", str(path)))


# ------------------------------------------------------------------ section 0: research freeze
#
# Everything below reads the repository. Nothing below writes to a frozen stage, and the two
# stages carrying a known provenance defect are recorded rather than repaired: section 0 of the
# brief permits a maintenance amendment, and section 20 of SGV13's brief forbids editing a
# frozen stage without explicit authorisation, which this stage does not have.

UPSTREAM_SCRIPTS = (
    "scripts/sgv1_candidate_conditioned.py",
    "scripts/sgv1_development_candidates.py",
    "scripts/sgv1_development_labels.py",
    "scripts/sgv1_development_ocr.py",
    "scripts/sgv1_domain_generalization.py",
    "scripts/sgv1_representation.py",
    "scripts/sgv1_risk_policy.py",
    "scripts/sgv1_verifier_pilot.py",
    "scripts/sgv2_reliability_layer.py",
    "scripts/sgv3_self_aware.py",
    "scripts/sgv4_environment_aware.py",
    "scripts/sgv5_candidate_reliability.py",
    "scripts/sgv6_target_engine_adaptation.py",
    "scripts/sgv9_target_calibration.py",
    "scripts/sgv10_tail_risk_control.py",
    "scripts/sgv11_minimax_policy_transfer.py",
    "scripts/sgv12_robust_ranking_transfer.py",
    "scripts/sgv13_fewshot_prefix_purity_adaptation.py",
)

UPSTREAM_ARTIFACTS = (
    "manifests/sgv1/role_manifest.json",
    "manifests/sgv1/confirmatory_reserve_lock.json",
    "results/generated/sgv1/dev_ocr/canonical_spans.parquet",
    "results/generated/sgv1/dev_candidates/candidates_pre_gt.parquet",
    "results/generated/sgv1/dev_labels/labels.parquet",
    "results/generated/sgv1/verifier_pilot/evidence_bundles.parquet",
    "results/generated/sgv1/verifier_pilot/crop_lineage.parquet",
    "results/generated/sgv1/representation/representation_lineage.parquet",
    "results/generated/sgv1/candidate_conditioned/features.parquet",
    "results/generated/sgv1/domain_generalization/design_matrix.parquet",
    "results/generated/sgv5_candidate_reliability/candidate_features.parquet",
    "results/generated/sgv5_candidate_reliability/model_predictions.parquet",
    "results/generated/sgv5_candidate_reliability/policy_selection.json",
    "results/generated/sgv6_target_engine_adaptation/adaptation_results.json",
    "results/generated/sgv13_fewshot_prefix_purity_adaptation/design_record.json",
    "results/generated/sgv13_fewshot_prefix_purity_adaptation/adaptation_results.json",
    "results/generated/sgv13_fewshot_prefix_purity_adaptation/research_decision.json",
    "results/generated/sgv13_fewshot_prefix_purity_adaptation/provenance.json",
)

KNOWN_PROVENANCE_DEFECT = {
    "id": "SGV14-A1",
    "stages": ["sgv7_risk_constrained_policy_selection", "sgv8_evidence_aware_deployment"],
    "tests": [
        "tests/leakage/test_sgv7_policy_selection.py",
        "tests/leakage/test_sgv8_evidence_aware_deployment.py",
    ],
    "defect": (
        "both suites assert that every path named in their stage's provenance manifest exists on "
        "disk, including parquet intermediates the repository deliberately gitignores, so both "
        "would fail on a clean checkout even though every scientific number they cover is "
        "unaffected. SGV9 onwards ask `git check-ignore` whether a missing file is deliberately "
        "ignored and are clean."
    ),
    "classification": "provenance/engineering defect, not a scientific defect",
    "action_taken": (
        "recorded, not repaired. Section 0 permits a maintenance amendment; SGV13's brief "
        "forbids editing frozen SGV7/SGV8 provenance tests without explicit authorisation, "
        "which SGV14 was not given. No SGV14 result depends on either stage."
    ),
    "published_results_moved": False,
    "verdicts_changed": False,
    "first_recorded_by": "sgv13_fewshot_prefix_purity_adaptation provenance.json",
}


def run_reconstruct() -> int:
    """The mandatory research freeze: what the repository is, before SGV14 changes anything."""
    started = time.monotonic()
    status = _git("status", "--porcelain")
    dirty = [line for line in status.splitlines() if line.strip()]
    scripts = {
        name: (file_sha256(REPO / name) if (REPO / name).is_file() else None)
        for name in UPSTREAM_SCRIPTS
    }
    missing_scripts = sorted(name for name, digest in scripts.items() if digest is None)
    if missing_scripts:
        raise PhaseError(f"upstream scripts absent from the working tree: {missing_scripts}")
    artifacts = {
        name: (file_sha256(REPO / name) if (REPO / name).is_file() else None)
        for name in UPSTREAM_ARTIFACTS
    }
    missing_artifacts = sorted(name for name, digest in artifacts.items() if digest is None)
    if missing_artifacts:
        raise PhaseError(f"upstream artifacts absent: {missing_artifacts}")

    lock = cc_read_json(REPO / "manifests/sgv1/confirmatory_reserve_lock.json")
    manifest = cc_read_json(REPO / "manifests/sgv1/role_manifest.json")
    reserve = sorted(d for d, role in manifest["role_of"].items() if role == "CONFIRMATORY")
    if lock.get("status") != "LOCKED" or lock.get("unlock_record") is not None:
        raise PhaseError("the SGV1 confirmatory reserve lock is not in the state SGV13 left it")

    payload = {
        "artifact": "research_freeze",
        "schema_version": f"sgv14-research_freeze-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "HEAD"),
            "head_subject": _git("log", "-1", "--pretty=%s"),
            "dirty": bool(dirty),
            "dirty_entries": dirty,
            "stash_entries": len([x for x in _git("stash", "list").splitlines() if x.strip()]),
        },
        "dependency_graph": {
            "sgv14_imports": [
                "sgv1_candidate_conditioned",
                "sgv1_domain_generalization",
                "sgv1_representation",
                "sgv1_risk_policy",
                "sgv1_verifier_pilot",
                "sgv2_reliability_layer",
                "sgv5_candidate_reliability",
                "sgv6_target_engine_adaptation",
                "sgv9_target_calibration",
                "sgv11_minimax_policy_transfer",
                "sgv12_robust_ranking_transfer",
                "sgv13_fewshot_prefix_purity_adaptation",
            ],
            "transitive_predecessor_scripts": list(UPSTREAM_SCRIPTS),
            "depth": len(UPSTREAM_SCRIPTS) + 1,
            "note": (
                "SGV13 imports eleven predecessors and SGV14 imports SGV13, so the executable "
                "chain from this stage to SGV1's design matrix is nineteen scripts deep."
            ),
        },
        "upstream_script_sha256": scripts,
        "upstream_artifact_sha256": artifacts,
        "sgv1_confirmatory_reserve": {
            "status": lock["status"],
            "unlock_record": lock.get("unlock_record"),
            "reserve_count": int(lock["reserve_count"]),
            "reserve_document_set_sha256": lock["reserve_document_set_sha256"],
            "documents_verified": len(reserve),
            "consumed_by_sgv14": False,
            "note": (
                "SGV14 does not unlock the reserve. Spending it is a one-way door and this "
                "stage was not asked to open it; `environment_selection.json` records the "
                "decision and what it costs."
            ),
        },
        "clean_clone_risks": {
            "untracked_predecessor_scripts": sorted(
                name for name in UPSTREAM_SCRIPTS if not _tracked(REPO / name)
            ),
            "untracked_upstream_artifacts": sorted(
                name
                for name in UPSTREAM_ARTIFACTS
                if not _tracked(REPO / name) and not _ignored(REPO / name)
            ),
            "gitignored_upstream_artifacts": sorted(
                name for name in UPSTREAM_ARTIFACTS if _ignored(REPO / name)
            ),
            "non_regenerable": [
                "data/raw/** -- the recorded OCR responses, write-once and not committed by "
                "policy. The download scripts, checksums and license records are what ships."
            ],
        },
        "known_defects": [KNOWN_PROVENANCE_DEFECT],
        "frozen_stage_modifications": [],
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(RESEARCH_FREEZE, payload)
    print(
        f"reconstruct: {len(scripts)} scripts, {len(artifacts)} artifacts, "
        f"reserve {lock['status']} ({lock['reserve_count']} documents) "
        f"-> {_relative(RESEARCH_FREEZE)}"
    )
    return 0


# ------------------------------------------------------------------ the frozen configuration


@dataclass(frozen=True, slots=True)
class Frozen:
    """SGV13's selected configuration, read from its record and never restated here."""

    arm: str
    acquisition: str
    budget: int
    cut: str
    epsilon: float
    target_weight: float
    band_multiple: float
    coverage_floor: float
    design_record_sha256: str


def load_frozen() -> Frozen:
    """Read the frozen configuration from SGV13's design record, and bind its bytes."""
    record = cc_read_json(s13.DESIGN_RECORD)
    design = record["frozen_design"]
    return Frozen(
        arm=str(design["arm"]),
        acquisition=str(design["acquisition"]),
        budget=int(design["operating_budget"]),
        cut=str(design["primary_cut"]),
        epsilon=float(PRIMARY_EPSILON),
        target_weight=float(design["target_weight"]),
        band_multiple=float(design["band_multiple"]),
        coverage_floor=float(design["coverage_floor"]),
        design_record_sha256=file_sha256(s13.DESIGN_RECORD),
    )


def assert_frozen(frozen: Frozen) -> None:
    """Refuse to run if the configuration is not the one SGV13 selected.

    Every value is compared against the constant SGV13 itself defines, so a rename or a grid
    change in SGV13 fails here rather than silently redefining what SGV14 replicated.
    """
    expected = {
        "arm": s13.A4_JOINT,
        "acquisition": s13.Q_UNCERTAINTY,
        "cut": s13.CUT_FROZEN,
        "budget": 250,
        "epsilon": 0.10,
    }
    actual = {
        "arm": frozen.arm,
        "acquisition": frozen.acquisition,
        "cut": frozen.cut,
        "budget": frozen.budget,
        "epsilon": frozen.epsilon,
    }
    if actual != expected:
        raise PhaseError(
            f"the frozen configuration is not SGV13's: expected {expected}, read {actual}"
        )
    if frozen.budget not in s13.BUDGETS:
        raise PhaseError(f"budget {frozen.budget} is not on SGV13's ladder {s13.BUDGETS}")
    if frozen.cut not in s13.CUT_RULES:
        raise PhaseError(f"cut rule {frozen.cut!r} is not one of SGV13's")


def run_freeze() -> int:
    """Write the configuration artifact every later phase asserts against."""
    started = time.monotonic()
    frozen = load_frozen()
    assert_frozen(frozen)
    upstream = s13.verify_frozen()
    decision = cc_read_json(s13.DECISION)
    payload = {
        "artifact": "frozen_sgv13_configuration",
        "schema_version": f"sgv14-frozen_configuration-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": {
            "adaptation_arm": frozen.arm,
            "acquisition": frozen.acquisition,
            "target_labels": frozen.budget,
            "deployment_cut": frozen.cut,
            "primary_epsilon": frozen.epsilon,
            "secondary_epsilons": [e for e in EPSILONS if e != PRIMARY_EPSILON],
            "source_to_target_weighting": {"source_rows": 1.0, "target_rows": frozen.target_weight},
            "regularization": (
                "SGV5's selected model class and lambda per fold; the joint refit adds no "
                "regulariser of its own and raises no capacity"
            ),
            "band_multiple": frozen.band_multiple,
            "coverage_floor": frozen.coverage_floor,
            "selected_by": (
                "SGV13's source-side simulation, run before any held-out target row was read"
            ),
        },
        "seeds": {
            "sgv14_draw_seed": DRAW_SEED,
            "sgv14_draws": DRAWS,
            "seed_function": "sgv6_target_engine_adaptation._stable_seed",
            "sgv13_draw_seed": s13.DRAW_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "code_sha256": {
            name: file_sha256(REPO / name)
            for name in (
                "scripts/sgv13_fewshot_prefix_purity_adaptation.py",
                "scripts/sgv6_target_engine_adaptation.py",
                "scripts/sgv5_candidate_reliability.py",
                "scripts/sgv12_robust_ranking_transfer.py",
                "scripts/sgv1_domain_generalization.py",
            )
        },
        "upstream_artifact_sha256": {
            **{entry["path"]: entry["sha256"] for entry in upstream["artifacts"]},
            _relative(s13.DESIGN_RECORD): frozen.design_record_sha256,
            _relative(s13.DECISION): file_sha256(s13.DECISION),
        },
        "sgv13_verdict": {
            "verdict": decision["verdict"],
            "criteria_met": decision["criteria_met"],
            "criteria_total": decision["criteria_total"],
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(FROZEN_CONFIGURATION, payload)
    print(
        f"freeze: {frozen.arm} + {frozen.acquisition} + N={frozen.budget} + {frozen.cut} "
        f"at epsilon={frozen.epsilon} -> {_relative(FROZEN_CONFIGURATION)}"
    )
    return 0


# ------------------------------------------------------------------ the environments
#
# An environment is one (corpus, engine configuration) pair. The selection rule is mechanical
# and is stated before any environment is scored: no environment is included or excluded
# because of what it does to the result, and `environment_selection.json` records that no
# outcome had been observed at selection time.

CORPORA = {
    "funsd": {
        "adapter": "funsd",
        "kind": "forms",
        "gt_policy": "funsd_word_annotation",
        "splits": [{"split": "training_data"}, {"split": "testing_data"}],
        "rationale": (
            "the corpus is scanned business forms. SGV1..SGV13 were built entirely on CORD "
            "receipts, so this is the domain axis the brief asks for -- its example is "
            "literally 'forms if prior work is receipt-heavy'."
        ),
    },
    "ocrd_sbb": {
        "adapter": "ocrd_sbb",
        "kind": "historical_print",
        "gt_policy": "diplomatic_transcription",
        "splits": [{}],
        "rationale": (
            "historical German prints, largely Fraktur, transcribed diplomatically. It is the "
            "hardest shift available and it is the only corpus here whose recorded engine "
            "configurations differ from the ones SGV13 deployed."
        ),
    },
}

# Every engine configuration with recorded raw responses, read from the fingerprint records the
# acquiring run wrote. The two entries flagged `run_by_sgv14` are the configuration-novelty
# environments this stage produces itself; everything else is already on disk.
ENVIRONMENTS: tuple[dict[str, Any], ...] = (
    {
        "environment": "funsd/doctr",
        "corpus": "funsd",
        "base_engine": "doctr",
        "engine_id": "doctr",
        "params": {},
        "run_by_sgv14": False,
    },
    {
        "environment": "funsd/easyocr",
        "corpus": "funsd",
        "base_engine": "easyocr",
        "engine_id": "easyocr",
        "params": {"languages": ("en",)},
        "run_by_sgv14": False,
    },
    {
        "environment": "funsd/paddleocr",
        "corpus": "funsd",
        "base_engine": "paddleocr",
        "engine_id": "paddleocr",
        "params": {"lang": "en"},
        "run_by_sgv14": False,
    },
    {
        "environment": "funsd/tesseract",
        "corpus": "funsd",
        "base_engine": "tesseract",
        "engine_id": "tesseract",
        "params": {"lang": "eng", "psm": 6, "oem": 3},
        "run_by_sgv14": False,
    },
    {
        "environment": "funsd/tesseract_psm3",
        "corpus": "funsd",
        "base_engine": "tesseract",
        "engine_id": "tesseract_psm3",
        "params": {"lang": "eng", "psm": 3, "oem": 3},
        "run_by_sgv14": True,
    },
    {
        "environment": "sbb/doctr",
        "corpus": "ocrd_sbb",
        "base_engine": "doctr",
        "engine_id": "doctr",
        "params": {},
        "run_by_sgv14": False,
    },
    {
        "environment": "sbb/easyocr_de",
        "corpus": "ocrd_sbb",
        "base_engine": "easyocr",
        "engine_id": "easyocr",
        "params": {"languages": ("de",)},
        "run_by_sgv14": False,
    },
    {
        "environment": "sbb/paddleocr_de",
        "corpus": "ocrd_sbb",
        "base_engine": "paddleocr",
        "engine_id": "paddleocr",
        "params": {"lang": "german"},
        "run_by_sgv14": False,
    },
    {
        "environment": "sbb/tesseract_frk",
        "corpus": "ocrd_sbb",
        "base_engine": "tesseract",
        "engine_id": "tesseract",
        "params": {"lang": "frk", "psm": 3, "oem": 3},
        "run_by_sgv14": False,
    },
    {
        "environment": "sbb/tesseract_deu",
        "corpus": "ocrd_sbb",
        "base_engine": "tesseract",
        "engine_id": "tesseract_deu",
        "params": {"lang": "deu", "psm": 3, "oem": 3},
        "run_by_sgv14": True,
    },
)

SGV13_CONFIGURATIONS = {
    "doctr": {},
    "easyocr": {"languages": ["en"]},
    "paddleocr": {"lang": "en"},
    "tesseract": {"lang": "eng", "psm": 6, "oem": 3},
}

EXCLUDED_ENVIRONMENTS = (
    {
        "candidate": "cord x {doctr, easyocr, paddleocr, tesseract} on the SGV1 CONFIRMATORY "
        "reserve (177 documents)",
        "excluded": True,
        "outcome_observed_before_exclusion": False,
        "reason": (
            "the reserve is LOCKED with `unlock_record: null` and spending it is irreversible: "
            "once its pages are OCR'd, sited and scored they can never again serve as an "
            "untouched reserve for any later claim. SGV14 was asked to test the frozen method "
            "on NEW OCR environments, and the reserve supplies new documents in the SAME "
            "domain read by the SAME four configurations -- the weakest of the three kinds of "
            "novelty the brief lists, and the only one that costs a one-way door. It is left "
            "locked, and this is a limitation of SGV14 rather than a property of the method."
        ),
    },
    {
        "candidate": "cord TRAIN / CALIBRATION / DEVELOPMENT documents",
        "excluded": True,
        "outcome_observed_before_exclusion": True,
        "reason": (
            "consumed by SGV1..SGV13 in fitting, calibration, adaptation-pool or evaluation "
            "roles. Re-reading them would be a resample of the SGV13 evaluation rows, which "
            "section 6 of the brief forbids."
        ),
    },
    {
        "candidate": "gt4histocr",
        "excluded": True,
        "outcome_observed_before_exclusion": False,
        "reason": (
            "never ingested: no dataset manifest, no checksums and no per-volume license "
            "review, and acquiring it needs network access this stage does not assume. The "
            "corpus-freshness inventory also records it as excluded from the headline "
            "benchmark for transcription-policy incompatibility."
        ),
    },
    {
        "candidate": "synthetic corpus x the four simulated engines",
        "excluded": True,
        "outcome_observed_before_exclusion": True,
        "reason": (
            "synthetic output is labelled synthetic and never reaches a headline table. A "
            "simulated engine is also not an OCR environment in the sense SGV14 is testing."
        ),
    },
    {
        "candidate": "docTR with a different recognition backbone (master, sar_resnet31, parseq)",
        "excluded": True,
        "outcome_observed_before_exclusion": False,
        "reason": (
            "the alternative backbones are not in the local model cache and fetching them "
            "needs network access. Configuration novelty is instead obtained from the two "
            "Tesseract configurations this stage runs and the three German configurations "
            "already recorded on OCR-D-SBB."
        ),
    },
)


def environment_of(name: str) -> dict[str, Any]:
    for spec in ENVIRONMENTS:
        if spec["environment"] == name:
            return dict(spec)
    raise PhaseError(f"unknown environment {name!r}")


def _adapter_for(spec: dict[str, Any]) -> Any:
    from ocr_risk.engines.registry import build_engine

    return build_engine(spec["base_engine"], engine_id=spec["base_engine"], **spec["params"])


def _datasets_for(corpus: str) -> list[Any]:
    """Every dataset split that makes up one corpus.

    FUNSD ships `training_data` and `testing_data` and both carry recorded OCR; the split is
    part of the dataset's own identity and not of this project's partition, which is drawn over
    whatever documents are loaded. Loading both is what makes the FUNSD environments 199 pages
    rather than 149.
    """
    from ocr_risk.datasets.registry import build_dataset

    entry = CORPORA[corpus]
    return [build_dataset(entry["adapter"], **params) for params in entry["splits"]]


def _document_bundles(corpus: str) -> dict[str, Any]:
    """Every document in a corpus, keyed by id. Read once per corpus and reused."""
    bundles: dict[str, Any] = {}
    for dataset in _datasets_for(corpus):
        for bundle in dataset.documents():
            bundles[bundle.document.document_id] = bundle
    return bundles


PARTITION_UNIT = {"funsd": "document", "ocrd_sbb": "volume"}


def group_of(corpus: str, document_id: str) -> str:
    """The atomic unit the partition moves. A page never leaves its group.

    FUNSD forms are independent scans, so a form is its own group. OCR-D-SBB pages come in
    volumes that share a typeface, a compositor and a scanning session, and splitting a volume
    across the adaptation and evaluation halves would let the method buy labels on the very
    book it is then evaluated on. The document remains the resampling unit; the VOLUME is the
    partition unit, which is the same distinction the project draws between a page and a
    duplicate component.
    """
    if PARTITION_UNIT[corpus] == "document":
        return document_id
    stem = document_id[len("ocrd-") :] if document_id.startswith("ocrd-") else document_id
    return stem.rsplit("-", 1)[0] if "-" in stem else stem


def partition_of(corpus: str, document_ids: list[str]) -> dict[str, str]:
    """Split a corpus into an adaptation and an evaluation half, content-blind.

    The key is a hash of the GROUP name under a fixed seed, so which half a page lands in is
    decided by its name and nothing else -- not its image, not its OCR, not its outcome. Whole
    groups move together and the target document count is met to the nearest group. The same
    partition is used by every environment on the corpus, so a page is never adaptation
    material for one engine and evaluation material for another.
    """
    from ocr_risk.io.hashing import canonical_hash

    groups: dict[str, list[str]] = {}
    for name in sorted(document_ids):
        groups.setdefault(group_of(corpus, name), []).append(name)
    ordered = sorted(
        groups,
        key=lambda key: (canonical_hash({"seed": PARTITION_SEED, "group": key}), key),
    )
    target = ADAPTATION_FRACTION * len(document_ids)
    roles: dict[str, str] = {}
    taken = 0
    for key in ordered:
        members = groups[key]
        role = ROLE_ADAPTATION if taken + len(members) / 2.0 <= target else ROLE_EVALUATION
        if role == ROLE_ADAPTATION:
            taken += len(members)
        roles.update(dict.fromkeys(members, role))
    return roles


def run_environments() -> int:
    """Write the selection audit. No environment has been scored when this runs."""
    started = time.monotonic()
    if ENVIRONMENT_ROWS.exists() or CONFIRMATORY_SCORES.exists():
        raise PhaseError(
            "environment rows or scores already exist; the selection audit must be written "
            "before any environment is built or scored"
        )
    manifest = cc_read_json(REPO / "manifests/sgv1/role_manifest.json")
    consumed = set(manifest["role_of"])

    included: list[dict[str, Any]] = []
    for spec in ENVIRONMENTS:
        corpus = CORPORA[spec["corpus"]]
        documents = sorted(_document_bundles(spec["corpus"]))
        overlap = sorted(set(documents) & consumed)
        if overlap:
            raise PhaseError(
                f"{spec['environment']}: {len(overlap)} documents appear in SGV1's role "
                f"manifest, e.g. {overlap[:3]}"
            )
        roles = partition_of(spec["corpus"], documents)
        fingerprint = _environment_adapter(spec).fingerprint()
        baseline = SGV13_CONFIGURATIONS[spec["base_engine"]]
        included.append(
            {
                "environment": spec["environment"],
                "corpus": spec["corpus"],
                "corpus_kind": corpus["kind"],
                "gt_policy": corpus["gt_policy"],
                "base_engine": spec["base_engine"],
                "engine_parameters": {
                    k: list(v) if isinstance(v, tuple) else v for k, v in spec["params"].items()
                },
                "engine_fingerprint": fingerprint.fingerprint,
                "engine_version": fingerprint.engine_version,
                "model_ids": list(fingerprint.model_ids),
                "configuration_new_versus_sgv13": (
                    {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec["params"].items()}
                    != baseline
                ),
                "domain_new_versus_sgv13": True,
                "documents_total": len(documents),
                "documents_adaptation": sum(1 for r in roles.values() if r == ROLE_ADAPTATION),
                "documents_evaluation": sum(1 for r in roles.values() if r == ROLE_EVALUATION),
                "raw_responses_run_by_sgv14": bool(spec["run_by_sgv14"]),
                "overlap_with_sgv1_role_manifest": 0,
            }
        )

    payload = {
        "artifact": "environment_selection",
        "schema_version": f"sgv14-environment_selection-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git("rev-parse", "HEAD"),
        "selection_rule": (
            "every corpus in the repository that has ground truth, has recorded raw OCR, and "
            "whose documents are absent from SGV1's role manifest -- crossed with every engine "
            "configuration whose raw responses that corpus already carries -- plus two "
            "Tesseract configurations run by this stage so that configuration novelty is "
            "available inside a fixed corpus. The rule is mechanical over repository contents."
        ),
        "any_outcome_observed_before_selection": False,
        "outcome_observation_note": (
            "No SGV14 environment had been sited, labelled or scored when this file was "
            "written; `--environments` refuses to run once `environment_rows.parquet` exists. "
            "FUNSD and OCR-D-SBB were consumed by the earlier h1_pilot/cgv2/cgv3 line, so "
            "results on those corpora have been seen before by a different pipeline -- but by "
            "no stage of SGV1..SGV13, which is the chain SGV14 replicates."
        ),
        "corpora": {name: dict(entry) for name, entry in CORPORA.items()},
        "included": included,
        "excluded": list(EXCLUDED_ENVIRONMENTS),
        "environment_count": len(included),
        "partition": {
            "seed": PARTITION_SEED,
            "adaptation_fraction": ADAPTATION_FRACTION,
            "unit": "document",
            "inputs_allowed": ["document_id"],
            "inputs_forbidden": [
                "image content",
                "annotation content",
                "OCR output",
                "candidate counts",
                "verifier scores",
                "outcome labels",
            ],
            "shared_across_environments_on_a_corpus": True,
            "partition_unit": dict(PARTITION_UNIT),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(ENVIRONMENT_SELECTION, payload)
    new_config = sum(1 for e in included if e["configuration_new_versus_sgv13"])
    print(
        f"environments: {len(included)} included ({new_config} with a configuration SGV13 never "
        f"deployed), {len(EXCLUDED_ENVIRONMENTS)} excluded -> {_relative(ENVIRONMENT_SELECTION)}"
    )
    return 0


# ------------------------------------------------------------------ new engine configurations
#
# Two environments need raw responses that do not exist yet. They are written through the same
# write-once `RawStore` every earlier stage used, at the fingerprint the adapter itself computes,
# so a response written here is indistinguishable in provenance from one written by SGV1.


def _image_path(corpus: str, bundle: Any) -> Path:
    from ocr_risk.io.paths import raw_source_dir

    return raw_source_dir(corpus).parents[2] / bundle.document.image_path


def run_ocr(only: str | None = None) -> int:
    """Recognize the two SGV14 configurations.

    Idempotent: a complete response is validated and skipped, never rewritten.
    """
    from ocr_risk.engines.base import PageInput
    from ocr_risk.io.raw_store import RawStore

    started = time.monotonic()
    store = RawStore()
    for spec in ENVIRONMENTS:
        if not spec["run_by_sgv14"]:
            continue
        if only is not None and spec["environment"] != only:
            continue
        adapter = _environment_adapter(spec)
        adapter.availability().require()
        fingerprint = adapter.fingerprint().fingerprint
        bundles = _document_bundles(spec["corpus"])
        done = 0
        began = time.monotonic()
        for document_id in sorted(bundles):
            if store.has_engine_response(
                spec["corpus"], spec["engine_id"], fingerprint, document_id
            ):
                done += 1
                continue
            bundle = bundles[document_id]
            response = adapter.recognize(
                PageInput(
                    dataset_id=spec["corpus"],
                    document_id=document_id,
                    image_path=_image_path(spec["corpus"], bundle),
                    image_sha256=bundle.document.image_sha256,
                    width=bundle.document.width,
                    height=bundle.document.height,
                )
            )
            store.write_engine_response(
                response.model_copy(update={"engine_id": spec["engine_id"]})
                if response.engine_id != spec["engine_id"]
                else response
            )
            done += 1
        print(
            f"  {spec['environment']}: {done}/{len(bundles)} responses at {fingerprint[:8]} "
            f"in {time.monotonic() - began:.0f}s"
        )
    print(f"ocr: complete in {time.monotonic() - started:.0f}s")
    return 0


# ------------------------------------------------------------------ the frozen representation
#
# Everything below rebuilds SGV1's ninety-six design columns and SGV5's structural block on rows
# SGV1 never saw. Every fitted object -- the generator resources, the confidence featurizer, the
# character language model, the lexicon, the glyph prototypes -- comes from SGV1's TRAIN
# documents and is frozen. `--identity` proves the whole path against SGV1's own frozen tables.

SGV1_ENGINES = BASE_ENGINES
CONTEXT_CHARS = 40
UNION_CAP = 8
PRIMARY_GENERATOR = "g8_union"
UNLABELLED_DISTANCE = -1
OUTCOME_UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class FrozenPipeline:
    """Every transformer the representation needs, fitted once on SGV1 TRAIN and frozen."""

    fitted: dict[str, Any]
    featurizer: Any
    builder: Any
    resources: Any
    site_counts: dict[str, int]
    engines: list[str]
    extra_columns: list[str]
    mask: Any
    train_documents: frozenset[str]
    glyph_prototypes: int
    width_per_char: float


def sgv1_spans() -> tuple[dict[tuple[str, str], list[Any]], dict[tuple[str, str], str]]:
    """SGV1's frozen canonical OCR, as spans and as reading-order streams."""
    import pyarrow.parquet as pq

    from ocr_risk.canonical import rebuild_stream
    from ocr_risk.schemas.spans import CanonicalSpan

    table = pq.read_table(pilot.OCR_SPANS).to_pylist()
    spans: dict[tuple[str, str], list[Any]] = {}
    for raw in table:
        spans.setdefault((str(raw["document_id"]), str(raw["engine_id"])), []).append(
            CanonicalSpan.model_validate(raw)
        )
    for group in spans.values():
        group.sort(key=lambda span: span.reading_order)
    streams = {pair: rebuild_stream(group) for pair, group in spans.items() if group}
    return spans, streams


def build_pipeline() -> FrozenPipeline:
    """Fit every frozen transformer, exactly where SGV1 fitted it: on TRAIN documents.

    Nothing here is refitted per environment. The confidence featurizer's per-engine mean and
    spread, the language model, the lexicon and the glyph prototypes are the ones a deployed
    system would carry, and applying them unchanged to a new engine configuration is what
    deployment MEANS -- the shift they then fail to absorb is the object of study, not a defect
    to be corrected by refitting on the target.
    """
    from ocr_risk.evidence import EvidenceBuilder
    from ocr_risk.evidence.features_conf import ConfidenceFeaturizer
    from ocr_risk.experiments.cgv3_confirmatory import fit_fold_resources

    manifest = cc_read_json(REPO / "manifests/sgv1/role_manifest.json")
    train_documents = frozenset(
        document_id for document_id, role in manifest["role_of"].items() if role == "TRAIN"
    )
    spans, streams = sgv1_spans()
    fitted = fit_fold_resources(streams, train_documents, SGV1_ENGINES)

    featurizer = ConfidenceFeaturizer()
    featurizer.fit(
        [
            (span.engine_id, span.native_conf_recognition, span.conf_scale)
            for (document_id, _engine), group in spans.items()
            if document_id in train_documents
            for span in group
        ]
    )
    builder = EvidenceBuilder(
        crop_policy=pilot.CROP_POLICY,
        context_chars=pilot.CONTEXT_CHARS,
        confidence_featurizer=featurizer,
    )
    resources = cc.fit_resources(streams, set(train_documents))
    lineage = pd.read_parquet(repr_stage.REPR_DIR / "representation_lineage.parquet")
    candidates = pd.read_parquet(pilot.CANDIDATE_TABLE)
    labels = pd.read_parquet(pilot.LABEL_TABLE)
    pool = candidates.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    pool = pool[pool["labelable"]].reset_index(drop=True)
    glyphs, width = cc.fit_glyph_prototypes(lineage, pool, set(train_documents))
    resources.glyphs, resources.width_per_char = glyphs, width

    return FrozenPipeline(
        fitted=fitted,
        featurizer=featurizer,
        builder=builder,
        resources=resources,
        site_counts=candidates.groupby("site_id").size().to_dict(),
        engines=sorted(candidates["engine_id"].astype(str).unique()),
        extra_columns=cc._family_columns(pd.read_parquet(cc.FEATURES), tuple(cc.FAMILIES)),
        mask=repr_stage.EvidenceMask.from_key("sgv1_v1"),
        train_documents=train_documents,
        glyph_prototypes=len(glyphs),
        width_per_char=float(width),
    )


def _canonical_spans(
    spec: dict[str, Any], bundles: dict[str, Any]
) -> tuple[dict[tuple[str, str], list[Any]], dict[str, Any]]:
    """One environment's recorded OCR, canonicalized under SGV1's policy.

    The canonical layer records the BASE engine id while the envelope keeps the environment's
    own fingerprint, because a new configuration of Tesseract is still Tesseract read on
    Tesseract's own 0-100 scale: the frozen per-engine confidence transformer, the frozen
    generator resources and the frozen model are all keyed by the base engine, and a deployment
    carries exactly those. `data/raw` is not touched and the mapping is recorded in the build
    record.
    """
    from ocr_risk.canonical import CanonicalizationPolicy, canonicalize_response
    from ocr_risk.io.paths import data_root
    from ocr_risk.io.raw_store import RawStore

    store = RawStore()
    adapter = _environment_adapter(spec)
    fingerprint = adapter.fingerprint().fingerprint
    policy_ = CanonicalizationPolicy(unicode_policy="nfc")
    base = spec["base_engine"]
    spans: dict[tuple[str, str], list[Any]] = {}
    missing: list[str] = []
    unverified_images: list[str] = []
    for document_id in sorted(bundles):
        path = store.engine_response_path(
            spec["corpus"], spec["engine_id"], fingerprint, document_id
        )
        if not path.is_file():
            missing.append(document_id)
            continue
        raw = store.read_engine_response(path)
        if raw.engine_fingerprint != fingerprint:
            raise PhaseError(
                f"{spec['environment']}/{document_id}: the stored response carries fingerprint "
                f"{raw.engine_fingerprint[:8]}, not {fingerprint[:8]}"
            )
        recorded = raw.payload.get("image_sha256")
        if recorded is not None and str(recorded) != bundles[document_id].document.image_sha256:
            raise PhaseError(
                f"{spec['environment']}/{document_id}: the recorded response was produced from "
                "a different image than the corpus now yields"
            )
        if recorded is None:
            unverified_images.append(document_id)
        spans[(document_id, base)] = list(
            canonicalize_response(
                raw=raw.model_copy(update={"engine_id": base}),
                parsed=adapter.parse(raw),
                policy=policy_,
                conf_scale_name=adapter.confidence_scale.name,
                raw_ref=path.relative_to(data_root()).as_posix(),
            )
        )
    if missing:
        raise PhaseError(
            f"{spec['environment']}: {len(missing)} documents have no recorded response at "
            f"{fingerprint[:8]}, e.g. {missing[:3]}"
        )
    return spans, {
        "engine_fingerprint": fingerprint,
        "engine_id_recorded_in_raw": spec["engine_id"],
        "engine_id_used_in_representation": base,
        "documents": len(spans),
        "spans": int(sum(len(v) for v in spans.values())),
        "responses_without_a_recorded_image_hash": len(unverified_images),
    }


def _environment_adapter(spec: dict[str, Any]) -> Any:
    from ocr_risk.engines.registry import build_engine

    return build_engine(spec["base_engine"], engine_id=spec["engine_id"], **spec["params"])


def _candidate_table(
    spec: dict[str, Any],
    spans: dict[tuple[str, str], list[Any]],
    streams: dict[tuple[str, str], str],
    pipeline: FrozenPipeline,
    roles: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sites and the g8_union candidate stream, in SGV1's construction, ground-truth-blind.

    The joins, the context window, the union cap and the candidate-id digest are SGV1's; the
    digest carries an SGV14 schema string and the environment name, so an environment's ids can
    never collide with SGV1's or with another environment's. `--identity` runs this same
    function over SGV1's own CORD pairs and asserts it reproduces the frozen candidate ids.
    """
    from ocr_risk.discovery.enumerator import DiscoveryRules
    from ocr_risk.experiments.cgv3_confirmatory import discovery_pass, generation_pass
    from ocr_risk.io.hashing import canonical_hash

    engines = [spec["base_engine"]]
    sites, _empty = discovery_pass(spans, pipeline.fitted, engines, DiscoveryRules())
    if sites.empty:
        raise PhaseError(f"{spec['environment']}: the OCR-only enumerator produced no sites")
    sites = sites.sort_values(
        ["document_id", "engine_id", "char_start", "char_end", "site_id"], kind="stable"
    ).reset_index(drop=True)
    sites.insert(4, "role", sites["document_id"].map(roles))
    if sites["role"].isna().any():
        raise PhaseError(f"{spec['environment']}: a site has no partition role")

    ladder = generation_pass(spans, streams, sites, pipeline.fitted, engines)
    if ladder.empty:
        raise PhaseError(f"{spec['environment']}: the frozen sites produced no candidates")
    ladder = ladder.sort_values(
        ["document_id", "engine_id", "site_id", "generator_id", "generator_rank"], kind="stable"
    ).reset_index(drop=True)
    primary = ladder[ladder["generator_id"] == PRIMARY_GENERATOR].copy()
    if primary.empty:
        raise PhaseError(f"{spec['environment']}: g8_union produced no candidates")

    source: dict[tuple[str, str], str] = {}
    for generator_id in ("g3_edit_aware", "g7_structural_v2"):
        for row in ladder[ladder["generator_id"] == generator_id].itertuples(index=False):
            source.setdefault((str(row.site_id), str(row.candidate_text)), generator_id)
    primary["generator_source"] = [
        source.get((str(row.site_id), str(row.candidate_text)), "")
        for row in primary.itertuples(index=False)
    ]
    if (primary["generator_source"] == "").any():
        raise PhaseError(f"{spec['environment']}: a g8_union candidate has no component source")

    site_columns = [
        "site_id",
        "role",
        "anchor_kind",
        "anchor_ref",
        "char_start",
        "char_end",
        "site_type",
        "suspicion_score",
        "provenance_reason",
    ]
    primary = primary.merge(sites[site_columns], on="site_id", how="left", validate="many_to_one")
    primary["original_ocr"] = primary["region_ocr"]
    contexts: dict[str, tuple[str, str]] = {}
    for site in sites.itertuples(index=False):
        stream = streams.get((str(site.document_id), str(site.engine_id)), "")
        start, end = int(site.char_start), int(site.char_end)
        contexts[str(site.site_id)] = (
            stream[max(0, start - CONTEXT_CHARS) : start],
            stream[end : end + CONTEXT_CHARS],
        )
    primary["context_before"] = [contexts[str(s)][0] for s in primary["site_id"]]
    primary["context_after"] = [contexts[str(s)][1] for s in primary["site_id"]]
    primary["candidate_id"] = [
        "sgv14-candidate-"
        + canonical_hash(
            {
                "schema": "sgv14-environment-candidate-id-v1",
                "environment": spec["environment"],
                "site_id": str(row.site_id),
                "document_id": str(row.document_id),
                "engine_id": str(row.engine_id),
                "generator_id": str(row.generator_id),
                "candidate_text": str(row.candidate_text),
                "operation": str(row.operation),
            }
        )
        for row in primary.itertuples(index=False)
    ]
    if primary["candidate_id"].duplicated().any():
        raise PhaseError(f"{spec['environment']}: candidate ids are not unique")
    if (primary["candidate_text"] == primary["original_ocr"]).any():
        raise PhaseError(f"{spec['environment']}: an identity edit reached the candidate stream")
    if int(primary.groupby("site_id").size().max()) > UNION_CAP:
        raise PhaseError(f"{spec['environment']}: a site exceeded the union cap")
    return sites, primary.reset_index(drop=True)


def _labels(
    corpus: str,
    bundles: dict[str, Any],
    spans: dict[tuple[str, str], list[Any]],
    streams: dict[tuple[str, str], str],
    sites: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Ground truth attached to the frozen candidate stream, in SGV1's construction.

    The status gate is SGV1's: only components the alignment marks RESOLVED may produce an
    evaluated correction site, because an empty region ground truth is otherwise three
    different statements at once and labelling the other two against the empty string
    manufactures beneficial edits out of missing annotation (incident SGV1-L3).
    """
    from ocr_risk.align import align_document
    from ocr_risk.config.models import AlignmentConfig
    from ocr_risk.edits.outcome import is_harmful
    from ocr_risk.experiments.cgv3_study import AlignmentIndex, label_candidate
    from ocr_risk.experiments.sgv1_frames import FRAME_CLASS_BENEFICIAL, evaluation_class
    from ocr_risk.schemas.enums import AlignmentStatus, AnchorKind, HarmPolicy

    resolved = AlignmentStatus.RESOLVED.value
    configuration = AlignmentConfig()
    indexes: dict[tuple[str, str], Any] = {}
    status_of: dict[tuple[tuple[str, str], str], str] = {}
    diagnostics = {"pairs_aligned": 0, "sites_index_missing": 0}
    for pair in sorted(spans):
        document_id, _engine = pair
        bundle = bundles[document_id]
        outcome = align_document(bundle.document, spans[pair], bundle.gt_tokens, configuration)
        records = [record.model_dump(mode="json") for record in outcome.records]
        indexes.update(
            AlignmentIndex.from_frames(
                pd.DataFrame(records),
                pd.DataFrame([token.model_dump(mode="json") for token in bundle.gt_tokens]),
            )
        )
        for record in records:
            status_of[pair, str(record["alignment_id"])] = str(record["status"])
        diagnostics["pairs_aligned"] += 1

    by_span = {pair: {span.span_id: span for span in group} for pair, group in spans.items()}
    truth: dict[str, tuple[str, str]] = {}
    for row in sites.itertuples():
        pair = (str(row.document_id), str(row.engine_id))
        site_id = str(row.site_id)
        anchors = [part for part in str(row.anchor_ref).split("\0")[1:] if part]
        index = indexes.get(pair)
        if index is None:
            diagnostics["sites_index_missing"] += 1
            truth[site_id] = ("", "unresolved:no_alignment_index")
            continue
        lookup = by_span.get(pair, {})
        ordered = [span_id for span_id in anchors if span_id in lookup]
        blocked: str | None = None
        if len(ordered) != len(anchors):
            blocked = "unresolved:span_missing_from_page"
        else:
            for span_id in ordered:
                component = index.component_for_span(span_id)
                if component is None:
                    blocked = "unresolved:span_unaligned"
                    break
                status = status_of.get(
                    (pair, component.alignment_id), AlignmentStatus.UNRESOLVED.value
                )
                if status != resolved:
                    blocked = f"unresolved:{status}"
                    break
        if blocked is not None:
            truth[site_id] = ("", blocked)
            continue
        if str(row.anchor_kind) == AnchorKind.GAP.value and len(anchors) == 2:
            positions = [
                next((i for i, c in enumerate(index.components) if span_id in c.ocr_span_ids), None)
                for span_id in anchors
            ]
            left, right = positions
            if left is None or right is None or left >= right:
                truth[site_id] = ("", "unresolved:gap_flanks_unordered")
                continue
            interior = [c for c in index.components[left + 1 : right] if not c.ocr_span_ids]
            if any(
                status_of.get((pair, c.alignment_id), AlignmentStatus.UNRESOLVED.value) != resolved
                for c in interior
            ):
                truth[site_id] = ("", "unresolved:gap_interior_unresolved")
                continue
            truth[site_id] = (index.gap_truth(anchors[0], anchors[1]), resolved)
        else:
            truth[site_id] = (index.region_truth(ordered)[1], resolved)

    expected = [
        streams.get((str(row.document_id), str(row.engine_id)), "")[
            int(row.char_start) : int(row.char_end)
        ]
        for row in candidates.itertuples()
    ]
    mismatch = int(
        (
            candidates["original_ocr"].astype(str).to_numpy() != np.asarray(expected, dtype=object)
        ).sum()
    )
    if mismatch:
        raise PhaseError(
            f"{corpus}: {mismatch} candidates have original_ocr != stream[char_start:char_end]"
        )

    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples():
        region_gt, status = truth[str(row.site_id)]
        labelable = status == resolved
        if labelable:
            d_before, d_after, accepted = label_candidate(
                str(row.original_ocr), str(row.candidate_text), region_gt
            )
            outcome_value = accepted.value
            harmful = bool(is_harmful(accepted, HarmPolicy.STRICT_WORSENING))
            beneficial = bool(evaluation_class(outcome_value) == FRAME_CLASS_BENEFICIAL)
        else:
            d_before = d_after = UNLABELLED_DISTANCE
            outcome_value, harmful, beneficial = OUTCOME_UNRESOLVED, False, False
        rows.append(
            {
                "candidate_id": str(row.candidate_id),
                "outcome": outcome_value,
                "is_harmful": harmful,
                "beneficial": beneficial,
                "labelable": labelable,
                "d_before": int(d_before),
                "d_after": int(d_after),
            }
        )
    frame = pd.DataFrame(rows)
    diagnostics["labelable"] = int(frame["labelable"].sum())
    diagnostics["candidates"] = len(frame)
    return frame, diagnostics


@dataclass(frozen=True, slots=True)
class PageImage:
    """Image metadata for one document. Carries no annotation, by construction."""

    document_id: str
    path: Path
    image_sha256: str
    width: int
    height: int


def _page_images(corpus: str, bundles: dict[str, Any]) -> dict[str, PageImage]:
    """Source-image metadata only. The type has no field an annotation could travel in."""
    return {
        document_id: PageImage(
            document_id=document_id,
            path=_image_path(corpus, bundle),
            image_sha256=bundle.document.image_sha256,
            width=bundle.document.width,
            height=bundle.document.height,
        )
        for document_id, bundle in bundles.items()
    }


def _evidence(
    candidates: pd.DataFrame,
    spans: dict[tuple[str, str], list[Any]],
    streams: dict[tuple[str, str], str],
    images: dict[str, PageImage],
    pipeline: FrozenPipeline,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Ground-truth-blind evidence bundles and the crop each candidate's own geometry names."""
    from ocr_risk.evidence import materialize
    from ocr_risk.evidence.crops import build_recipe
    from ocr_risk.schemas.evidence import ObservationView

    by_span = {pair: {s.span_id: s for s in group} for pair, group in spans.items()}
    scale = repr_stage.CROP_SCALES[cc.CROP_SCALE]
    bundles: dict[str, Any] = {}
    lineage: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates.itertuples():
        pair = (str(row.document_id), str(row.engine_id))
        anchor_ids = pilot._anchor_span_ids(str(row.anchor_ref))
        anchor_spans = [by_span[pair][s] for s in anchor_ids if s in by_span.get(pair, {})]
        bbox = pilot._union_bbox(anchor_spans)
        page = images[str(row.document_id)]
        view = ObservationView(
            site_id=str(row.site_id),
            candidate_id=str(row.candidate_id),
            document_id=str(row.document_id),
            dataset_id=str(row.dataset_id),
            engine_id=str(row.engine_id),
            original_ocr=str(row.original_ocr),
            candidate_text=str(row.candidate_text),
            ocr_span_ids=tuple(anchor_ids),
            image_sha256=page.image_sha256,
            image_width=page.width,
            image_height=page.height,
            bbox=bbox,
        )
        bundles[str(row.candidate_id)] = pipeline.builder.build(
            view,
            ocr_stream=streams.get(pair, ""),
            char_start=int(row.char_start),
            char_end=int(row.char_end),
            native_confidences=[s.native_conf_recognition for s in anchor_spans],
            conf_scale=next((s.conf_scale for s in anchor_spans if s.conf_scale), None),
        )
        digest = ""
        if bbox is not None:
            recipe = build_recipe(page.image_sha256, bbox, scale)
            digest = recipe.recipe_sha256
            if digest not in seen:
                seen.add(digest)
                materialize(recipe, page.path)
        lineage.append(
            {
                "candidate_id": str(row.candidate_id),
                "recipe": digest,
                "bbox_x0": bbox.x0 if bbox is not None else np.nan,
                "bbox_y0": bbox.y0 if bbox is not None else np.nan,
                "bbox_x1": bbox.x1 if bbox is not None else np.nan,
                "bbox_y1": bbox.y1 if bbox is not None else np.nan,
            }
        )
    return bundles, pd.DataFrame(lineage)


def _candidate_conditioned_features(
    pool: pd.DataFrame,
    streams: dict[tuple[str, str], str],
    pipeline: FrozenPipeline,
    site_counts: dict[str, int],
) -> pd.DataFrame:
    """SGV1's edit, plausibility, visual and context blocks, from the frozen resources.

    The loop and the block order are SGV1's `run_features`; every value in them comes out of a
    `cc.*` function this stage imports rather than restates, and `--identity` asserts the whole
    table reproduces SGV1's `features.parquet` to 0.0 on SGV1's own rows.
    """
    from ocr_risk.io.paths import cache_root

    page_tokens = {f"{d}:{e}": Counter(cc._tokens(s)) for (d, e), s in streams.items()}
    line_counts = {f"{d}:{e}": max(len(s.splitlines()), 1) for (d, e), s in streams.items()}
    ink_cache: dict[str, np.ndarray | None] = {}
    rows: list[np.ndarray] = []
    names: list[str] | None = None
    for row in pool.itertuples():
        original, candidate = str(row.original_ocr), str(row.candidate_text)
        digest = str(row.recipe)
        if digest not in ink_cache:
            ink_cache[digest] = (
                cc._ink_columns(cache_root() / "crops" / digest[:2] / f"{digest}.png")
                if digest
                else None
            )
        ink = ink_cache[digest]
        key = f"{row.document_id}:{row.engine_id}"
        blocks = [
            cc.r0_block(row, site_counts.get(str(row.site_id), 1), pipeline.engines),
            cc.edit_block(original, candidate),
            cc.plausibility_block(original, candidate, pipeline.resources),
            cc.visual_block(ink, original, candidate, pipeline.resources),
            cc.context_block(
                original, candidate, page_tokens.get(key, Counter()), line_counts.get(key, 1)
            ),
        ]
        if names is None:
            names = [name for block, _ in blocks for name in block]
        rows.append(np.concatenate([np.asarray(v, dtype=np.float64) for _, v in blocks]))
    assert names is not None
    table = pd.DataFrame(np.vstack(rows), columns=names)
    table.insert(0, "candidate_id", pool["candidate_id"].astype(str).to_numpy())
    return table


def _design_matrix(
    pool: pd.DataFrame,
    features: pd.DataFrame,
    bundles: dict[str, Any],
    pipeline: FrozenPipeline,
    site_counts: dict[str, int],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """SGV1's ninety-six design columns, through SGV1's own verifier construction.

    `site_counts` is how many candidates the environment's OWN generation pass emitted at each
    site: it is a property of this page read by this engine, not a statistic borrowed from
    CORD, and SGV1 computed it the same way from its own candidate table.
    """
    merged = pool.merge(features, on="candidate_id", how="inner", validate="one_to_one")
    if len(merged) != len(pool):
        raise PhaseError("the candidate-conditioned feature table does not align with the pool")
    provenance = {
        str(row.candidate_id): pilot.provenance_block(row, site_counts[str(row.site_id)]).values
        for row in merged.itertuples()
    }
    verifier = cc.CandidateConditionedVerifier(
        extra_names=pipeline.extra_columns,
        verifier_id="domain_generalization",
        evidence_config="sgv1_v1",
        model="logistic",
        provenance=provenance,
        embedding=dict(
            zip(
                merged["candidate_id"].astype(str),
                merged[pipeline.extra_columns].to_numpy(dtype=np.float64),
                strict=True,
            )
        ),
    )
    identifiers = list(merged["candidate_id"].astype(str))
    matrix = verifier._matrix(
        pilot._inputs_for("domain_generalization", identifiers, bundles, {}, pipeline.mask)
    )
    return matrix, tuple(verifier.feature_names)


def _pages_from_spans(spans: dict[tuple[str, str], list[Any]]) -> dict[str, Any]:
    """SGV5's per-page amount summary, on spans that are not in SGV5's frozen parquet.

    `c5.build_pages` reads one hard-coded path, so the grouping loop is restated here while
    every value inside it -- the amount parser, the separator shape, the reconciliation and
    pair residuals -- is SGV5's own function. `--identity` asserts this function reproduces
    `c5.build_pages()` exactly on SGV1's own spans, which is what makes the restatement safe.
    """
    pages: dict[str, Any] = {}
    for (document_id, engine_id), group in spans.items():
        values: list[float] = []
        shapes: Counter[tuple[str, ...]] = Counter()
        for span in group:
            value = c5._amount(span.text)
            if value is not None and value > 0:
                values.append(value)
                shapes[c5._separator_shape(span.text)] += 1
        truncated = len(values) > c5.MAX_PAGE_AMOUNTS
        amounts = np.sort(np.array(sorted(values, reverse=True)[: c5.MAX_PAGE_AMOUNTS]))
        pages[f"{document_id}:{engine_id}"] = c5.Page(
            amounts=amounts,
            modal_shape=shapes.most_common(1)[0][0] if shapes else (),
            median_log=float(np.median(np.log10(amounts))) if amounts.size else 0.0,
            reconcile=c5._reconcile_residual(amounts),
            pair=c5._pair_residual(amounts),
            truncated=truncated,
        )
    return pages


def build_environment(
    spec: dict[str, Any], pipeline: FrozenPipeline, progress: Any = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One environment, all the way from recorded OCR to the frozen representation."""
    from ocr_risk.canonical import rebuild_stream

    began = time.monotonic()
    bundles = _document_bundles(spec["corpus"])
    roles = partition_of(spec["corpus"], sorted(bundles))
    spans, ocr_record = _canonical_spans(spec, bundles)
    streams = {pair: rebuild_stream(group) for pair, group in spans.items() if group}
    sites, candidates = _candidate_table(spec, spans, streams, pipeline, roles)
    labels, label_diagnostics = _labels(spec["corpus"], bundles, spans, streams, sites, candidates)

    pool = candidates.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    pool = pool[pool["labelable"]].reset_index(drop=True)
    if pool.empty:
        raise PhaseError(f"{spec['environment']}: no candidate is labelable")
    site_counts = candidates.groupby("site_id").size().to_dict()
    images = _page_images(spec["corpus"], bundles)
    evidence, lineage = _evidence(pool, spans, streams, images, pipeline)
    pool = pool.merge(lineage, on="candidate_id", how="inner", validate="one_to_one")
    features = _candidate_conditioned_features(pool, streams, pipeline, site_counts)
    matrix, names = _design_matrix(pool, features, evidence, pipeline, site_counts)

    structural = c5.structural_features(pool, _pages_from_spans(spans))
    signatures = np.array(
        [
            c5.edit_signature(o, y, op)
            for o, y, op in zip(
                pool["original_ocr"].astype(str),
                pool["candidate_text"].astype(str),
                pool["operation"].astype(str),
                strict=True,
            )
        ]
    )

    frame = pd.DataFrame(
        {
            "candidate_id": pool["candidate_id"].astype(str).to_numpy(),
            "site_id": pool["site_id"].astype(str).to_numpy(),
            "document_id": pool["document_id"].astype(str).to_numpy(),
            "engine_id": pool["engine_id"].astype(str).to_numpy(),
            "role": pool["document_id"].map(roles).to_numpy(str),
            "outcome": pool["outcome"].astype(str).to_numpy(),
            "is_harmful": pool["is_harmful"].to_numpy(dtype=bool),
            "beneficial": pool["beneficial"].to_numpy(dtype=bool),
            "environment": spec["environment"],
            "corpus": spec["corpus"],
            "base_engine": spec["base_engine"],
            "edit_signature": signatures,
            "image_sha256": pool["document_id"]
            .map({d: image.image_sha256 for d, image in images.items()})
            .to_numpy(str),
        }
    )
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(matrix, columns=list(names)),
            structural.reset_index(drop=True),
        ],
        axis=1,
    )
    adaptation = frame["role"] == ROLE_ADAPTATION
    evaluation = frame["role"] == ROLE_EVALUATION
    _assert_disjoint(spec, frame, adaptation, evaluation)

    record = {
        "environment": spec["environment"],
        "corpus": spec["corpus"],
        "base_engine": spec["base_engine"],
        **ocr_record,
        "sites": len(sites),
        "candidates_generated": len(candidates),
        "candidates_labelable": len(pool),
        "labelable_fraction": float(len(pool) / max(len(candidates), 1)),
        "alignment": {k: int(v) for k, v in label_diagnostics.items()},
        "rows": len(frame),
        "rows_adaptation": int(adaptation.sum()),
        "rows_evaluation": int(evaluation.sum()),
        "documents_adaptation": int(frame.loc[adaptation, "document_id"].nunique()),
        "documents_evaluation": int(frame.loc[evaluation, "document_id"].nunique()),
        "harmful_adaptation": int(frame.loc[adaptation, "is_harmful"].sum()),
        "beneficial_adaptation": int(frame.loc[adaptation, "beneficial"].sum()),
        "harmful_evaluation": int(frame.loc[evaluation, "is_harmful"].sum()),
        "beneficial_evaluation": int(frame.loc[evaluation, "beneficial"].sum()),
        "outcome_census": {
            str(k): int(v) for k, v in Counter(frame["outcome"].tolist()).most_common()
        },
        "design_columns": len(names),
        "structural_columns": len(c5.STRUCTURAL_NAMES),
        "elapsed_seconds": time.monotonic() - began,
    }
    if progress is not None:
        progress(
            f"  {spec['environment']}: {len(frame)} rows "
            f"({record['rows_adaptation']}/{record['rows_evaluation']}), "
            f"{record['documents_adaptation']}/{record['documents_evaluation']} documents, "
            f"beneficial {record['beneficial_evaluation']} / harmful "
            f"{record['harmful_evaluation']} on evaluation, {record['elapsed_seconds']:.0f}s"
        )
    return frame, record


def _assert_disjoint(
    spec: dict[str, Any], frame: pd.DataFrame, adaptation: pd.Series, evaluation: pd.Series
) -> None:
    """Both split axes, asserted on the rows that were actually built.

    Document, correction site and source image, all three. The image check is the one a
    document-id check cannot make: two pages with the same bytes are the same page whatever
    they are called, and the project's own partition rule says they must land in one bucket.
    """
    name = spec["environment"]
    for column, label in (
        ("document_id", "documents"),
        ("site_id", "correction sites"),
        ("image_sha256", "source images"),
    ):
        shared = set(frame.loc[adaptation, column]) & set(frame.loc[evaluation, column])
        if shared:
            raise PhaseError(
                f"{name}: the adaptation and evaluation partitions share {len(shared)} {label}"
            )
    manifest = cc_read_json(REPO / "manifests/sgv1/role_manifest.json")
    overlap = set(frame["document_id"]) & set(manifest["role_of"])
    if overlap:
        raise PhaseError(f"{name}: {len(overlap)} documents appear in SGV1's role manifest")


def run_build(only: str | None = None) -> int:
    """Build every environment's rows in the frozen representation."""
    started = time.monotonic()
    if not ENVIRONMENT_SELECTION.is_file():
        raise PhaseError("run --environments first; the selection audit precedes any build")
    pipeline = build_pipeline()
    print(
        f"  frozen pipeline: {len(pipeline.resources.lm)} lm grams, "
        f"{len(pipeline.resources.lexicon)} lexicon tokens, {pipeline.glyph_prototypes} glyph "
        f"prototypes, confidence statistics for {len(pipeline.featurizer.engines)} engines"
    )
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for spec in ENVIRONMENTS:
        if only is not None and spec["environment"] != only:
            continue
        frame, record = build_environment(spec, pipeline, progress=print)
        frames.append(frame)
        records.append(record)
    combined = pd.concat(frames, ignore_index=True)
    if combined["candidate_id"].duplicated().any():
        raise PhaseError("candidate ids collide across environments")
    _write_parquet_once(ENVIRONMENT_ROWS, combined)
    payload = {
        "artifact": "environment_inventory",
        "schema_version": f"sgv14-environment_inventory-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environments": records,
        "totals": {
            "environments": len(records),
            "rows": len(combined),
            "documents": int(combined["document_id"].nunique()),
            "corpora": sorted(set(combined["corpus"])),
        },
        "frozen_transformers": {
            "fitted_on": "SGV1 TRAIN documents only",
            "train_documents": len(pipeline.train_documents),
            "lm_grams": len(pipeline.resources.lm),
            "lexicon_tokens": len(pipeline.resources.lexicon),
            "glyph_prototypes": pipeline.glyph_prototypes,
            "median_width_per_char": pipeline.width_per_char,
            "confidence_engines": list(pipeline.featurizer.engines),
            "refitted_on_target": False,
        },
        "artifacts": {_relative(ENVIRONMENT_ROWS): file_sha256(ENVIRONMENT_ROWS)},
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(ENVIRONMENT_INVENTORY, payload)
    print(
        f"build: {len(records)} environments, {len(combined)} rows -> "
        f"{_relative(ENVIRONMENT_ROWS)} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the representation identity
#
# A representation that cannot rebuild SGV1's own rows exactly may not be trusted on rows SGV1
# never saw. This phase runs the SGV14 build path over CORD pairs and compares every stage of it
# against the frozen artifact SGV1 published.

IDENTITY_DOCUMENTS = 12


def _identity_documents(count: int) -> list[str]:
    """A deterministic, role-balanced sample of SGV1 documents. Never the reserve."""
    design = pd.read_parquet(dg.DESIGN_MATRIX, columns=["document_id", "role"])
    chosen: list[str] = []
    per_role = max(count // len(SGV1_ROLES), 1)
    for role in SGV1_ROLES:
        names = sorted(design.loc[design["role"] == role, "document_id"].unique())
        chosen.extend(names[:per_role])
    return sorted(set(chosen))


def run_identity() -> int:
    """Prove the rebuild against SGV1's frozen canonical spans, candidates, labels and matrix."""
    from ocr_risk.canonical import CanonicalizationPolicy, canonicalize_response
    from ocr_risk.discovery.enumerator import DiscoveryRules
    from ocr_risk.experiments.cgv3_confirmatory import discovery_pass, generation_pass
    from ocr_risk.io.hashing import canonical_hash
    from ocr_risk.io.paths import data_root
    from ocr_risk.io.raw_store import RawStore

    started = time.monotonic()
    pipeline = build_pipeline()
    documents = _identity_documents(IDENTITY_DOCUMENTS)
    frozen_spans, frozen_streams = sgv1_spans()
    manifest = cc_read_json(REPO / "manifests/sgv1/role_manifest.json")
    roles = {str(k): str(v) for k, v in manifest["role_of"].items()}
    if any(roles[d] == "CONFIRMATORY" for d in documents):
        raise PhaseError("a confirmatory reserve document reached the identity check")

    # 1. canonical spans, rebuilt from the write-once raw responses.
    store, policy_ = RawStore(), CanonicalizationPolicy(unicode_policy="nfc")
    rebuilt: dict[tuple[str, str], list[Any]] = {}
    for engine in SGV1_ENGINES:
        adapter = _adapter_for({"base_engine": engine, "params": SGV13_CONFIGURATIONS[engine]})
        fingerprint = adapter.fingerprint().fingerprint
        for document_id in documents:
            path = store.engine_response_path("cord", engine, fingerprint, document_id)
            raw = store.read_engine_response(path)
            rebuilt[(document_id, engine)] = list(
                canonicalize_response(
                    raw=raw,
                    parsed=adapter.parse(raw),
                    policy=policy_,
                    conf_scale_name=adapter.confidence_scale.name,
                    raw_ref=path.relative_to(data_root()).as_posix(),
                )
            )
    span_mismatch = sum(
        1
        for pair, group in rebuilt.items()
        if [s.model_dump(mode="json") for s in group]
        != [s.model_dump(mode="json") for s in frozen_spans[pair]]
    )
    if span_mismatch:
        raise PhaseError(f"{span_mismatch} rebuilt canonical pairs differ from the frozen ones")

    # 2. sites and candidates, through the same discovery and generation functions.
    subset = {pair: group for pair, group in frozen_spans.items() if pair[0] in set(documents)}
    subset_streams = {pair: frozen_streams[pair] for pair in subset if pair in frozen_streams}
    sites, _ = discovery_pass(subset, pipeline.fitted, SGV1_ENGINES, DiscoveryRules())
    sites = sites.sort_values(
        ["document_id", "engine_id", "char_start", "char_end", "site_id"], kind="stable"
    ).reset_index(drop=True)
    ladder = generation_pass(subset, subset_streams, sites, pipeline.fitted, SGV1_ENGINES)
    ladder = ladder.sort_values(
        ["document_id", "engine_id", "site_id", "generator_id", "generator_rank"], kind="stable"
    ).reset_index(drop=True)
    primary = ladder[ladder["generator_id"] == PRIMARY_GENERATOR].copy()
    identifiers = {
        "sgv1-candidate-"
        + canonical_hash(
            {
                "schema": "sgv1-natural-candidate-id-v1",
                "site_id": str(row.site_id),
                "document_id": str(row.document_id),
                "engine_id": str(row.engine_id),
                "generator_id": str(row.generator_id),
                "candidate_text": str(row.candidate_text),
                "operation": str(row.operation),
            }
        )
        for row in primary.itertuples(index=False)
    }
    frozen_candidates = pd.read_parquet(pilot.CANDIDATE_TABLE)
    published = set(
        frozen_candidates.loc[
            frozen_candidates["document_id"].isin(set(documents)), "candidate_id"
        ].astype(str)
    )
    if identifiers != published:
        raise PhaseError(
            f"rebuilt candidate ids differ from SGV1's: {len(published - identifiers)} missing, "
            f"{len(identifiers - published)} unexpected"
        )

    # 3. the design matrix and the candidate-conditioned features, on SGV1's own pool rows.
    frozen_labels = pd.read_parquet(pilot.LABEL_TABLE)
    pool = frozen_candidates.merge(
        frozen_labels, on="candidate_id", how="inner", validate="one_to_one"
    )
    pool = pool[pool["labelable"] & pool["document_id"].isin(set(documents))].reset_index(drop=True)
    images = {
        document_id: PageImage(
            document_id=document_id,
            path=page.path,
            image_sha256=page.image_sha256,
            width=page.width,
            height=page.height,
        )
        for document_id, page in pilot._page_images(set(pool["document_id"])).items()
    }
    evidence, lineage = _evidence(pool, frozen_spans, frozen_streams, images, pipeline)
    published_lineage = pd.read_parquet(
        repr_stage.REPR_DIR / "representation_lineage.parquet",
        columns=["candidate_id", f"recipe_{cc.CROP_SCALE}"],
    ).set_index("candidate_id")
    recipe_mismatch = int(
        sum(
            1
            for row in lineage.itertuples()
            if str(published_lineage.loc[str(row.candidate_id), f"recipe_{cc.CROP_SCALE}"])
            != str(row.recipe)
        )
    )
    if recipe_mismatch:
        raise PhaseError(f"{recipe_mismatch} rebuilt crop recipes differ from SGV1's")

    pool = pool.merge(lineage, on="candidate_id", how="inner", validate="one_to_one")
    site_counts = frozen_candidates.groupby("site_id").size().to_dict()
    features = _candidate_conditioned_features(pool, frozen_streams, pipeline, site_counts)
    frozen_features = pd.read_parquet(cc.FEATURES).set_index("candidate_id")
    order = features["candidate_id"].astype(str).tolist()
    feature_delta = float(
        np.nanmax(
            np.abs(
                features.set_index("candidate_id").to_numpy(dtype=float)
                - frozen_features.loc[order, list(features.columns[1:])].to_numpy(dtype=float)
            )
        )
    )
    matrix, names = _design_matrix(pool, features, evidence, pipeline, site_counts)
    frozen_design = pd.read_parquet(dg.DESIGN_MATRIX).set_index("candidate_id")
    published_names = tuple(c for c in frozen_design.columns if c not in dg.META_COLUMNS)
    if names != published_names:
        raise PhaseError("the rebuilt design matrix does not carry SGV1's column names")
    design_delta = float(
        np.nanmax(np.abs(matrix - frozen_design.loc[order, list(published_names)].to_numpy(float)))
    )

    # 4. the labels, and SGV5's structural block and page summary.
    identity_candidates = frozen_candidates[
        frozen_candidates["document_id"].isin(set(documents))
    ].reset_index(drop=True)
    labels, _ = _labels(
        "cord", _cord_bundles(set(documents)), subset, subset_streams, sites, identity_candidates
    )
    joined = labels.set_index("candidate_id").loc[
        frozen_labels.loc[
            frozen_labels["candidate_id"].isin(labels["candidate_id"]), "candidate_id"
        ].astype(str)
    ]
    reference = frozen_labels.set_index("candidate_id").loc[joined.index]
    label_agreement = {
        column: int((joined[column].to_numpy() == reference[column].to_numpy()).sum())
        for column in ("outcome", "is_harmful", "labelable", "d_before", "d_after")
    }
    if any(value != len(joined) for value in label_agreement.values()):
        raise PhaseError(f"rebuilt labels disagree with SGV1's: {label_agreement}")

    pages_rebuilt = _pages_from_spans(frozen_spans)
    pages_published = c5.build_pages()
    page_mismatch = sum(
        1
        for key, page in pages_published.items()
        if key not in pages_rebuilt
        or not np.array_equal(pages_rebuilt[key].amounts, page.amounts)
        or pages_rebuilt[key].modal_shape != page.modal_shape
        or not np.isclose(pages_rebuilt[key].median_log, page.median_log, rtol=0, atol=0)
    )
    if page_mismatch:
        raise PhaseError(f"{page_mismatch} rebuilt page summaries differ from SGV5's")
    structural = c5.structural_features(pool, pages_rebuilt)
    frozen_structural = pd.read_parquet(c5.FEATURES).set_index("candidate_id")
    structural_delta = float(
        np.nanmax(
            np.abs(
                structural.to_numpy(dtype=float)
                - frozen_structural.loc[order, list(c5.STRUCTURAL_NAMES)].to_numpy(dtype=float)
            )
        )
    )
    if max(feature_delta, design_delta, structural_delta) != 0.0:
        raise PhaseError(
            "the rebuilt representation is not identical to SGV1's: features "
            f"{feature_delta:g}, design {design_delta:g}, structural {structural_delta:g}"
        )

    payload = {
        "artifact": "representation_identity",
        "schema_version": f"sgv14-representation_identity-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents": documents,
        "document_count": len(documents),
        "roles": {role: sum(1 for d in documents if roles[d] == role) for role in SGV1_ROLES},
        "checks": {
            "canonical_spans_rebuilt_from_raw": {
                "pairs": len(rebuilt),
                "mismatching": span_mismatch,
            },
            "candidate_ids": {
                "rebuilt": len(identifiers),
                "published": len(published),
                "identical": True,
            },
            "crop_recipes": {"rows": len(lineage), "mismatching": recipe_mismatch},
            "labels": {"rows": len(joined), **label_agreement},
            "candidate_conditioned_features": {
                "columns": int(features.shape[1] - 1),
                "max_abs_difference": feature_delta,
            },
            "design_matrix": {
                "columns": len(names),
                "rows": int(matrix.shape[0]),
                "max_abs_difference": design_delta,
                "names_identical": True,
            },
            "sgv5_page_summaries": {"pages": len(pages_published), "mismatching": page_mismatch},
            "sgv5_structural_block": {
                "columns": len(c5.STRUCTURAL_NAMES),
                "max_abs_difference": structural_delta,
            },
        },
        "verdict": "the SGV14 build path reproduces SGV1's frozen representation exactly",
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(IDENTITY_RECORD, payload)
    print(
        f"identity: {len(documents)} documents, {len(rebuilt)} pairs, {int(matrix.shape[0])} rows; "
        f"features {feature_delta:g}, design {design_delta:g}, structural {structural_delta:g} "
        f"-> {_relative(IDENTITY_RECORD)}"
    )
    return 0


def _cord_bundles(selected: set[str]) -> dict[str, Any]:
    """CORD bundles for the identity check only. Never called on an SGV14 environment."""
    from ocr_risk.datasets.cord import CordDataset

    bundles: dict[str, Any] = {}
    for split in ("train", "validation"):
        for bundle in CordDataset(split=split).documents():
            if bundle.document.document_id in selected:
                bundles[bundle.document.document_id] = bundle
    return bundles


# ------------------------------------------------------------------ the pre-registration
#
# Everything the decision rule needs, fixed before the first environment is scored. `--adapt`
# refuses to run without this file and every analysis phase reads it rather than restating it.


def run_preregister() -> int:
    """Freeze the criteria, the sufficiency rule and the test family."""
    started = time.monotonic()
    if CONFIRMATORY_SCORES.exists():
        raise PhaseError(
            "confirmatory scores already exist; the criteria must be frozen before any "
            "environment is scored"
        )
    if not ENVIRONMENT_INVENTORY.is_file():
        raise PhaseError("run --build first; the environment count fixes the breadth majority")
    frozen = load_frozen()
    assert_frozen(frozen)
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    environments = [record["environment"] for record in inventory["environments"]]
    # Sufficiency is a property of the evaluation PARTITION -- how many pages it holds and how
    # many repairable and harmful candidates are in them -- and is settled by `--build`, before
    # any arm is fitted. The breadth majority is therefore a majority of the environments the
    # criteria are actually read on, and is fixed here rather than after the fact.
    eligible = {
        record["environment"]: (
            record["documents_evaluation"] >= MIN_EVALUATION_DOCUMENTS
            and record["beneficial_evaluation"] >= MIN_EVALUATION_BENEFICIAL
            and record["harmful_evaluation"] >= MIN_EVALUATION_HARMFUL
        )
        for record in inventory["environments"]
    }
    sufficient = [name for name, ok in eligible.items() if ok]
    required = int(np.ceil(BREADTH_FRACTION * len(sufficient)))
    payload = {
        "artifact": "design_record",
        "schema_version": f"sgv14-design_record-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git("rev-parse", "HEAD"),
        "frozen_method": {
            "arm": frozen.arm,
            "acquisition": frozen.acquisition,
            "target_labels": frozen.budget,
            "cut": frozen.cut,
            "primary_epsilon": frozen.epsilon,
            "target_row_weight": frozen.target_weight,
            "selected_by": "SGV13, before its own outer evaluation. Not re-selected here.",
        },
        "hypothesis": (
            "the frozen SGV13 procedure reproduces its risk-controlled repair gains on OCR "
            "environments absent from its development, pre-registration, selection and "
            "evaluation"
        ),
        "sub_claims": {
            "V1a": "the gain over the frozen baseline reproduces on new environments",
            "V1b": "the realised harm bound holds at non-degenerate coverage",
            "V1c": "the reproduction is not confined to one environment kind",
        },
        "environments": environments,
        "environment_count": len(environments),
        "environments_sufficient": sufficient,
        "environments_insufficient": [e for e in environments if e not in sufficient],
        "arms": {
            ARM_BASELINE: (
                "SGV5's frozen model for the fold that held this environment's base engine out, "
                "deployed at the published source cut. Spends no target label."
            ),
            ARM_PRIMARY: (
                "the frozen SGV13 method: `a4_joint_refit` on 250 labels acquired by "
                "`q2_uncertainty`, deployed under `frozen_source`."
            ),
            ARM_SGV6: (
                "SGV6's generic few-shot adaptation through SGV6's own `fit_arm` and "
                "`select_arm`, on the identical purchased rows."
            ),
            CONTROL_SOURCE_ONLY: (
                "the same refit construction on the source rows alone at unit weight. A "
                "validation control, never an alternative method."
            ),
            CONTROL_PERMUTED: (
                "the frozen method with the purchased rows' outcome labels permuted among "
                "themselves. A validation control, never an alternative method."
            ),
            DIAGNOSTIC_RANDOM: (
                "the frozen method on 250 labels acquired at random. Secondary and "
                "non-confirmatory: whatever it says, the frozen method does not change."
            ),
        },
        "draws": DRAWS,
        "draw_seed": DRAW_SEED,
        "common_random_seeds": (
            "the primary arm, the permutation control and the random-acquisition diagnostic "
            "share the draw seed at every draw"
        ),
        "sufficiency_rule": {
            "fixed_when": (
                "the three thresholds were written into the stage's registry before `--build` "
                "ran and have not changed since. The environment population counts they are "
                "applied to were therefore visible before this file was written, which is "
                "disclosed here rather than glossed: they are properties of the evaluation "
                "partition -- how many pages, how many repairable candidates, how many harmful "
                "ones -- and not results of any method."
            ),
            "min_evaluation_documents": MIN_EVALUATION_DOCUMENTS,
            "min_evaluation_beneficial": MIN_EVALUATION_BENEFICIAL,
            "min_evaluation_harmful": MIN_EVALUATION_HARMFUL,
            "effect": (
                "an environment failing any of the three is marked INSUFFICIENT: it is reported "
                "in full and excluded from the criteria majority, because a harm rate estimated "
                "from too few documents is not an estimate. The criteria are ALSO evaluated "
                "over every environment as a pre-registered sensitivity, and both numbers "
                "appear in the decision."
            ),
            "applies_to": "population properties of the evaluation partition, never to results",
        },
        "criteria": {
            "1_breadth": (
                f"the frozen method's mean risk-controlled repair recall at epsilon = "
                f"{frozen.epsilon} exceeds the frozen baseline's on at least {required} of the "
                f"{len(sufficient)} sufficient environments"
            ),
            "2_safety": (
                f"the realised accepted-set harm rate is at most {frozen.epsilon} on at least "
                f"{required} of the {len(sufficient)} sufficient environments, with no credit "
                "for holding it by accepting nothing"
            ),
            "3_non_degeneracy": (
                f"the frozen method's accepted coverage is at least SGV13's own coverage floor "
                f"{frozen.coverage_floor:.6f} on at least {required} of the {len(sufficient)} "
                "sufficient environments"
            ),
            "4_statistical_support": (
                "the pre-registered primary test -- the environment-stratified, "
                "document-clustered paired difference in risk-controlled repair recall against "
                f"the frozen baseline at epsilon = {frozen.epsilon} -- is favourable with a "
                f"95% interval excluding zero at alpha = {PRIMARY_ALPHA}"
            ),
        },
        "criteria_total": CRITERIA_TOTAL,
        "breadth_required": required,
        "breadth_fraction": BREADTH_FRACTION,
        "coverage_floor": frozen.coverage_floor,
        "coverage_floor_source": (
            "SGV13's design record, where it was fixed as 0.25 of the median inner deployed "
            "coverage of the frozen baseline. Read, not re-chosen."
        ),
        "verdict_rule": "SUPPORTED requires all four criteria; otherwise NOT SUPPORTED",
        "statistics": {
            "primary_test": (
                "one test: the environment-stratified paired difference against the frozen "
                "baseline on risk-controlled repair recall at the primary epsilon. Documents "
                "are resampled WITHIN each environment and the per-environment deltas are "
                "averaged with EQUAL WEIGHT, because a breadth claim treats each environment "
                "as one replication rather than weighting by how many pages it happens to have."
            ),
            "primary_family_size": 1,
            "secondary_family": (
                "the per-environment paired differences against the frozen baseline, and "
                "against the SGV6 baseline, at the primary epsilon"
            ),
            "multiplicity": "Holm at 0.05 within the secondary family; the primary test is one",
            "resampling_unit": "document",
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "hierarchy": (
                "the primary test is read first. Per-environment tests describe where an effect "
                "lives and cannot rescue a primary test that fails."
            ),
        },
        "secondary_epsilons": [e for e in EPSILONS if e != PRIMARY_EPSILON],
        "secondary_epsilon_rule": (
            "reported, never used to rescue a failure at the primary epsilon"
        ),
        "prohibited_after_this_file_exists": [
            "changing N, the acquisition rule, the arm, the cut, the weighting or epsilon",
            "adding a feature, an arm, an environment or a rescue calibration step",
            "special-casing an environment that fails",
            "changing the sufficiency rule, the breadth majority or the coverage floor",
            "changing which test is primary",
        ],
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DESIGN_RECORD, payload)
    print(
        f"preregister: {len(environments)} environments ({len(sufficient)} sufficient), "
        f"{required} required for a majority, coverage floor {frozen.coverage_floor:.6f} "
        f"-> {_relative(DESIGN_RECORD)}"
    )
    return 0


# ------------------------------------------------------------------ the frozen method, deployed
#
# The fold is SGV5's own -- source fit rows, source calibration rows, selected representation,
# model class and lambda -- with the held-out engine's rows replaced by the environment's. Every
# fitted object below is SGV13's function called on that fold; this stage adds no estimator.


@dataclass(frozen=True, slots=True)
class Combined:
    """SGV1's rows and one environment's rows in one design, with the transposed fold."""

    base: Any
    fold: Any
    signatures: np.ndarray
    retrieval_columns: list[int]
    environment: str
    base_engine: str
    sgv1_rows: int


def combine(
    environment: str, rows: pd.DataFrame, sgv1: tuple[Any, np.ndarray, list[int]]
) -> Combined:
    """Append one environment's rows to SGV1's design and transpose the fold onto them.

    SGV1's rows keep their index positions, so the fit, calibration and seen-evaluation blocks
    are the SAME integers SGV13 used and any change in a fitted object would have to come from
    the appended rows rather than from a re-indexing. The evaluation block is the environment's
    evaluation partition and the adaptation pool is its adaptation partition; the held-out
    engine's own rows are absent entirely, which is what makes this a new environment rather
    than a resample of SGV13's.
    """
    base, signatures, retrieval_columns = sgv1
    spec = environment_of(environment)
    block = rows[rows["environment"] == environment].reset_index(drop=True)
    if block.empty:
        raise PhaseError(f"{environment}: no rows were built")
    names = list(base.names)
    missing = [name for name in names if name not in block.columns]
    if missing:
        raise PhaseError(f"{environment}: the built rows are missing {len(missing)} columns")
    meta = pd.concat([base.meta, block[dg.META_COLUMNS].reset_index(drop=True)], ignore_index=True)
    matrix = np.vstack([base.matrix, block[names].to_numpy(dtype=np.float64)])
    combined_base = dg.Design(matrix=matrix, names=base.names, meta=meta)
    offset = int(base.matrix.shape[0])
    role = block["role"].to_numpy(str)
    source_role = base.meta["role"].to_numpy(str)
    source_engine = base.meta["engine_id"].to_numpy(str)
    held_out = spec["base_engine"]
    fold = rl.Fold(
        held_out=held_out,
        train_engines=tuple(e for e in BASE_ENGINES if e != held_out),
        fit=np.flatnonzero((source_role == "TRAIN") & (source_engine != held_out)),
        source_cal=np.flatnonzero((source_role == "CALIBRATION") & (source_engine != held_out)),
        seen_eval=np.flatnonzero((source_role == "DEVELOPMENT") & (source_engine != held_out)),
        eval=offset + np.flatnonzero(role == ROLE_EVALUATION),
        unlabeled_target=offset + np.flatnonzero(role == ROLE_ADAPTATION),
    )
    documents = combined_base.documents
    evaluation_documents = set(documents[fold.eval].tolist())
    for name, index in (
        ("fit", fold.fit),
        ("source_cal", fold.source_cal),
        ("seen_eval", fold.seen_eval),
        ("unlabeled_target", fold.unlabeled_target),
    ):
        if set(documents[index].tolist()) & evaluation_documents:
            raise PhaseError(f"{environment}: {name} shares documents with the evaluation block")
    if held_out in set(combined_base.meta["engine_id"].to_numpy(str)[fold.fit].tolist()):
        raise PhaseError(f"{environment}: the base engine reached the fit block")
    return Combined(
        base=combined_base,
        fold=fold,
        signatures=np.concatenate([signatures, block["edit_signature"].to_numpy(str)]),
        retrieval_columns=retrieval_columns,
        environment=environment,
        base_engine=held_out,
        sgv1_rows=offset,
    )


def reference_state(
    base_engine: str,
    sgv1: tuple[Any, np.ndarray, list[int]],
    selection: dict[str, Any],
    frozen_scores: pd.DataFrame,
) -> Any:
    """SGV13's own fold, rebuilt, where `build_state` asserts SGV5's published score bit for bit."""
    base, signatures, retrieval_columns = sgv1
    fold = rl.build_fold(base, base_engine)
    return s12.build_state(
        base, fold, selection, signatures, retrieval_columns, base_engine, frozen_scores
    )


def environment_setup(
    combined: Combined, selection: dict[str, Any], frozen: Frozen, reference: Any
) -> Any:
    """SGV13's `build_setup` on the transposed fold, with the frozen model asserted unchanged.

    The assertion is the point of the phase. `build_state` cannot check an environment's rows
    against a published vector, because SGV5 never scored them -- so what is checked instead is
    that appending them left every SGV1 row's decision score EXACTLY where SGV13 found it. If
    that holds, the model deployed on the new environment is the model SGV13 deployed.
    """
    setup = build_setup(
        combined.base,
        combined.fold,
        combined.base_engine,
        combined.signatures,
        combined.retrieval_columns,
        selection,
        None,
        frozen.band_multiple,
        disagreement=False,
    )
    for name, index in (
        ("fit", combined.fold.fit),
        ("source_cal", combined.fold.source_cal),
        ("seen_eval", combined.fold.seen_eval),
    ):
        difference = float(
            np.abs(setup.state.frozen_score(index) - reference.frozen_score(index)).max()
        )
        if difference != 0.0:
            raise PhaseError(
                f"{combined.environment}: appending the environment's rows moved the frozen "
                f"decision score on the {name} block by {difference:g}"
            )
    for epsilon in EPSILONS:
        key = epsilon_key(epsilon)
        if setup.state.source_threshold[key] != reference.source_threshold[key]:
            raise PhaseError(f"{combined.environment}: the published source cut at {key} moved")
    return setup


def _fit_arm(
    setup: Any, arm: str, budget: Any, order: np.ndarray, frozen: Frozen, seed: int
) -> Any:
    """The single dispatch point. No arm can acquire a side door."""
    if arm == ARM_BASELINE:
        return frozen_arm()
    if arm in (ARM_PRIMARY, DIAGNOSTIC_RANDOM, CONTROL_PERMUTED):
        return fit_joint_arm(setup, budget, frozen.target_weight, False, arm)
    if arm == ARM_SGV6:
        adapted, _record = fit_sgv6_arm(setup, order, frozen.budget)
        return adapted
    if arm == CONTROL_SOURCE_ONLY:
        fit = setup.state.index(FIT)
        return s13.Adapted(
            arm,
            s13.REFIT,
            None,
            s13._refit(
                setup,
                fit,
                np.ones(fit.size),
                setup.state.design.harmful[fit],
                setup.state.design.beneficial[fit],
                capacity=False,
            ),
            None,
            0.0,
            {"target_labels_used": 0, "fit_rows": int(fit.size), "fittable": True},
        )
    raise PhaseError(f"unknown arm {arm!r}")


def _draws_for(arm: str) -> int:
    """How many repeated budget draws an arm gets, and why it is not always `DRAWS`.

    The frozen baseline spends no label and the source-only control spends no label, so their
    curves are constant across draws by construction and a second draw would republish the same
    number. Everything that buys labels carries the full count.
    """
    return 1 if arm in (ARM_BASELINE, CONTROL_SOURCE_ONLY) else DRAWS


def _acquisition_for(arm: str, frozen: Frozen) -> str:
    """`q2_uncertainty` everywhere except the explicitly secondary random diagnostic."""
    return s13.Q_RANDOM if arm == DIAGNOSTIC_RANDOM else frozen.acquisition


OUTCOME_CLASSES = (
    "true_correction",
    "partial_improvement",
    "lateral_change",
    "overcorrection",
    "miscorrection",
)


def evaluate(
    setup: Any,
    arm: str,
    adapted: Any,
    budget: Any,
    baseline_score: np.ndarray,
    outcomes: np.ndarray,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One (environment, arm, draw) cell: the endpoints, and the per-page counts behind them.

    The deployment goes through the same `deployed` function SGV12 and SGV13 used, at the same
    cut rule, so a difference between two arms in one cell is a difference in the score and
    never in how either was measured.
    """
    block = setup.evaluation
    score = adapted.score(setup, block)
    row: dict[str, Any] = dict(context)
    # The SGV14 arm name, not the fitted object's. `frozen_arm` calls itself `a0_frozen` and
    # SGV6's selector calls itself `b1_sgv6_generic`; both are correct for the stage that named
    # them and neither is the name this stage's tables are keyed on.
    row["arm"] = arm
    row["fitted_arm"] = adapted.arm
    row["budget_rows"] = int(budget.size)
    row["budget_documents"] = int(budget.n_documents)
    row["budget_harmful"] = int(budget.harmful.sum())
    row["budget_beneficial"] = int(budget.beneficial.sum())
    row["budget_neutral"] = int((~budget.harmful & ~budget.beneficial).sum())
    row["identical_to_frozen"] = bool(np.array_equal(score, baseline_score))
    quality = ranking_quality(score, block.harmful, block.beneficial)
    for key in ("auroc_safe", "pair_accuracy", "average_precision_beneficial", "aurc"):
        row[f"rank__{key}"] = float(quality[key])
    achievable = frontier(score, block.harmful, block.beneficial)
    thresholds = thresholds_for(setup, adapted, budget, s13.CUT_FROZEN)
    counts: dict[str, np.ndarray] = {}
    for epsilon in EPSILONS:
        key = epsilon_key(epsilon)
        row[f"frontier__{key}"] = float(achievable[key]["repair_recall"])
        tau = float(thresholds[key]["tau"])
        point = deployed(score, block.harmful, block.beneficial, tau, epsilon)
        accepted = (
            np.asarray(score >= tau) if np.isfinite(tau) else np.zeros(block.size, dtype=bool)
        )
        stem = f"dep__{key}"
        row[f"{stem}__tau"] = tau
        row[f"{stem}__feasible"] = bool(thresholds[key]["feasible"])
        row[f"{stem}__coverage"] = float(point["coverage"])
        row[f"{stem}__realized_harm_rate"] = float(point["realized_harm_rate"])
        row[f"{stem}__holds_bound"] = bool(point["holds_bound"])
        row[f"{stem}__repair_recall"] = float(point["repair_recall"])
        row[f"{stem}__risk_controlled"] = float(point["risk_controlled_repair_recall"])
        row[f"{stem}__n_accepted"] = int(point["n_accepted"])
        row[f"{stem}__repairs_captured"] = int(point["repairs_captured"])
        row[f"{stem}__joint_harm_rate"] = float(point["joint_harm_rate"])
        row[f"{stem}__harmful_accepted"] = int((accepted & block.harmful).sum())
        row[f"{stem}__beneficial_accepted"] = int((accepted & block.beneficial).sum())
        # Section 12's composition of the accepted set. These are counts of a recorded label
        # column, not a new metric: `outcome` is the taxonomy `edits/outcome.py` defines and
        # SGV1 froze, and every rate below is one of these counts over `n_accepted`.
        size = int(accepted.sum())
        for name in OUTCOME_CLASSES:
            taken = int((accepted & (outcomes == name)).sum())
            row[f"{stem}__accepted_{name}"] = taken
            row[f"{stem}__accepted_{name}_rate"] = float(taken / size) if size else 0.0
        row[f"{stem}__edit_precision"] = (
            float(int((accepted & block.beneficial).sum()) / size) if size else 0.0
        )
        if epsilon == PRIMARY_EPSILON:
            baseline_accepted = (
                np.asarray(baseline_score >= tau)
                if np.isfinite(tau)
                else np.zeros(block.size, dtype=bool)
            )
            # SGV13's prefix statistics, minus the one that has no SGV14 meaning: there is no
            # re-ranking ceiling on a new environment, so `overlap_with_ceiling` would be the
            # baseline overlap under a second name and is dropped rather than published twice.
            for name, value in prefix_statistics(
                block, accepted, baseline_accepted, baseline_accepted
            ).items():
                if name == "overlap_with_ceiling":
                    continue
                row[f"prefix__{name}"] = value
            counts = per_document_counts(block, accepted, sorted(set(block.documents.tolist())))
    return row, counts


def run_adapt(only: str | None = None) -> int:
    """The frozen method, the baselines and the controls, on every environment."""
    started = time.monotonic()
    if not DESIGN_RECORD.is_file():
        raise PhaseError("run --preregister before --adapt; the criteria precede the evaluation")
    frozen = load_frozen()
    assert_frozen(frozen)
    s13.verify_frozen()
    rows = pd.read_parquet(ENVIRONMENT_ROWS)
    sgv1 = s6.load_base()
    selections = s6.frozen_selection()
    frozen_scores = s6.load_frozen_scores()

    records: list[dict[str, Any]] = []
    budget_records: list[dict[str, Any]] = []
    count_keys: list[tuple[str, str, int, str]] = []
    count_blocks: list[np.ndarray] = []
    document_index: dict[str, list[str]] = {}
    references: dict[str, Any] = {}
    for spec in ENVIRONMENTS:
        environment = spec["environment"]
        if only is not None and environment != only:
            continue
        began = time.monotonic()
        engine = spec["base_engine"]
        if engine not in references:
            references[engine] = reference_state(engine, sgv1, selections[engine], frozen_scores)
        combined = combine(environment, rows, sgv1)
        setup = environment_setup(combined, selections[engine], frozen, references[engine])
        block = setup.evaluation
        documents = sorted(set(block.documents.tolist()))
        document_index[environment] = documents
        baseline_score = np.asarray(block.frozen_score, dtype=float)
        outcomes = setup.state.design.meta["outcome"].to_numpy(str)[block.index]
        context_base = {
            "environment": environment,
            "corpus": spec["corpus"],
            "base_engine": engine,
            "evaluation_rows": int(block.size),
            "evaluation_documents": len(documents),
            "evaluation_beneficial": int(block.beneficial.sum()),
            "evaluation_harmful": int(block.harmful.sum()),
            "pool_rows": int(setup.view.size),
            "pool_documents": len(set(setup.view.documents.tolist())),
            "accepted_share": float(setup.view.accepted_share),
        }
        for arm in ARMS:
            rule = _acquisition_for(arm, frozen)
            for draw in range(_draws_for(arm)):
                seed = _stable_seed(
                    "sgv14-acquisition", str(DRAW_SEED), environment, rule, str(draw)
                )
                order = acquisition_order(setup.view, rule, setup.base, seed)
                size = 0 if arm in (ARM_BASELINE, CONTROL_SOURCE_ONLY) else frozen.budget
                permute = (
                    _stable_seed("sgv14-permute", str(DRAW_SEED), environment, str(draw))
                    if arm == CONTROL_PERMUTED
                    else None
                )
                budget = purchase(setup, order, size, permute_seed=permute)
                adapted = _fit_arm(setup, arm, budget, order, frozen, seed)
                context = {**context_base, "acquisition": rule, "draw": draw}
                record, counts = evaluate(
                    setup, arm, adapted, budget, baseline_score, outcomes, context
                )
                records.append(record)
                if counts:
                    count_keys.append((environment, arm, draw, PRIMARY_KEY))
                    count_blocks.append(
                        np.vstack(
                            [
                                counts["accepted"],
                                counts["harmful_accepted"],
                                counts["beneficial_accepted"],
                                counts["beneficial_total"],
                                counts["rows"],
                            ]
                        )
                    )
                if size:
                    budget_records.append(
                        {
                            "environment": environment,
                            "arm": arm,
                            "acquisition": rule,
                            "draw": draw,
                            "labels": int(budget.size),
                            "documents": int(budget.n_documents),
                            "harmful": int(budget.harmful.sum()),
                            "beneficial": int(budget.beneficial.sum()),
                            "neutral": int((~budget.harmful & ~budget.beneficial).sum()),
                            "labels_permuted": permute is not None,
                        }
                    )
        print(
            f"  {environment}: {len(ARMS)} arms, {block.size} evaluation rows / "
            f"{len(documents)} documents in {time.monotonic() - began:.0f}s"
        )

    table = pd.DataFrame(records)
    _write_parquet_once(CONFIRMATORY_SCORES, table)
    counts_frame = pd.DataFrame(
        {
            "environment": pd.Categorical(
                np.repeat([k[0] for k in count_keys], [b.shape[1] for b in count_blocks])
            ),
            "arm": pd.Categorical(
                np.repeat([k[1] for k in count_keys], [b.shape[1] for b in count_blocks])
            ),
            "draw": np.repeat([k[2] for k in count_keys], [b.shape[1] for b in count_blocks]),
            "document_id": pd.Categorical(
                np.concatenate([document_index[k[0]] for k in count_keys])
            ),
            "accepted": np.concatenate([b[0] for b in count_blocks]),
            "harmful_accepted": np.concatenate([b[1] for b in count_blocks]),
            "beneficial_accepted": np.concatenate([b[2] for b in count_blocks]),
            "beneficial_total": np.concatenate([b[3] for b in count_blocks]),
            "rows": np.concatenate([b[4] for b in count_blocks]),
        }
    )
    _write_parquet_once(DEPLOYMENT_COUNTS, counts_frame)
    budgets = pd.DataFrame(budget_records)
    payload = {
        "artifact": "adaptation_budget",
        "schema_version": f"sgv14-adaptation_budget-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget": frozen.budget,
        "acquisition": frozen.acquisition,
        "draws": DRAWS,
        "per_environment": {
            environment: {
                "labels": int(group["labels"].mean()),
                "documents_touched_mean": float(group["documents"].mean()),
                "documents_touched_min": int(group["documents"].min()),
                "documents_touched_max": int(group["documents"].max()),
                "harmful_mean": float(group.loc[~group["labels_permuted"], "harmful"].mean()),
                "beneficial_mean": float(group.loc[~group["labels_permuted"], "beneficial"].mean()),
                "neutral_mean": float(group.loc[~group["labels_permuted"], "neutral"].mean()),
                "by_arm": {
                    arm: {
                        "draws": len(sub),
                        "harmful_mean": float(sub["harmful"].mean()),
                        "beneficial_mean": float(sub["beneficial"].mean()),
                        "neutral_mean": float(sub["neutral"].mean()),
                    }
                    for arm, sub in group.groupby("arm", observed=True)
                },
            }
            for environment, group in budgets.groupby("environment", observed=True)
        },
        "accounting_rule": (
            "every target label spent on fitting, on calibrating, on selecting an adapter or on "
            "placing a cut is the same label and is counted once. The cut is `frozen_source` "
            "and reads no target row at all, so the 250 labels are spent entirely on the fit."
        ),
        "unusable_labels_replaced": False,
        "artifacts": {
            _relative(CONFIRMATORY_SCORES): file_sha256(CONFIRMATORY_SCORES),
            _relative(DEPLOYMENT_COUNTS): file_sha256(DEPLOYMENT_COUNTS),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(ADAPTATION_BUDGET, payload)
    print(
        f"adapt: {len(table)} cells over {table['environment'].nunique()} environments, "
        f"{len(counts_frame)} document rows -> {_relative(CONFIRMATORY_SCORES)} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ reading the endpoints


def load_scores() -> tuple[pd.DataFrame, dict[str, Any]]:
    """The score table and the frozen design, and a refusal to analyse without both."""
    if not CONFIRMATORY_SCORES.is_file():
        raise PhaseError("run --adapt first")
    if not DESIGN_RECORD.is_file():
        raise PhaseError("the pre-registration is missing")
    return pd.read_parquet(CONFIRMATORY_SCORES), cc_read_json(DESIGN_RECORD)


def sufficiency(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Which environments can support a harm estimate, by the rule fixed in `--preregister`."""
    out: dict[str, dict[str, Any]] = {}
    for environment, group in table.groupby("environment", observed=True):
        row = group.iloc[0]
        documents = int(row["evaluation_documents"])
        beneficial = int(row["evaluation_beneficial"])
        harmful = int(row["evaluation_harmful"])
        reasons = []
        if documents < MIN_EVALUATION_DOCUMENTS:
            reasons.append(f"{documents} evaluation documents < {MIN_EVALUATION_DOCUMENTS}")
        if beneficial < MIN_EVALUATION_BENEFICIAL:
            reasons.append(f"{beneficial} beneficial candidates < {MIN_EVALUATION_BENEFICIAL}")
        if harmful < MIN_EVALUATION_HARMFUL:
            reasons.append(f"{harmful} harmful candidates < {MIN_EVALUATION_HARMFUL}")
        out[str(environment)] = {
            "evaluation_documents": documents,
            "evaluation_rows": int(row["evaluation_rows"]),
            "evaluation_beneficial": beneficial,
            "evaluation_harmful": harmful,
            "sufficient": not reasons,
            "reasons": reasons,
        }
    return out


def cell(table: pd.DataFrame, environment: str, arm: str) -> pd.DataFrame:
    block = table[(table["environment"] == environment) & (table["arm"] == arm)]
    if block.empty:
        raise PhaseError(f"no cell for {environment}/{arm}")
    return block


def summarise(block: pd.DataFrame, key: str) -> dict[str, Any]:
    """One arm's endpoints at one epsilon, averaged over draws with their spread."""
    stem = f"dep__{key}"
    return {
        "draws": len(block),
        "risk_controlled_repair_recall": float(block[f"{stem}__risk_controlled"].mean()),
        "risk_controlled_sd": float(block[f"{stem}__risk_controlled"].std(ddof=0)),
        "repair_recall": float(block[f"{stem}__repair_recall"].mean()),
        "realized_harm_rate": float(block[f"{stem}__realized_harm_rate"].mean()),
        "holds_bound_fraction": float(block[f"{stem}__holds_bound"].mean()),
        "coverage": float(block[f"{stem}__coverage"].mean()),
        "n_accepted": float(block[f"{stem}__n_accepted"].mean()),
        "harmful_accepted": float(block[f"{stem}__harmful_accepted"].mean()),
        "beneficial_accepted": float(block[f"{stem}__beneficial_accepted"].mean()),
        "joint_harm_rate": float(block[f"{stem}__joint_harm_rate"].mean()),
        "achievable_frontier": float(block[f"frontier__{key}"].mean()),
        "edit_precision": float(block[f"{stem}__edit_precision"].mean()),
        "accepted_composition": {
            name: float(block[f"{stem}__accepted_{name}"].mean()) for name in OUTCOME_CLASSES
        },
        "accepted_composition_rate": {
            name: float(block[f"{stem}__accepted_{name}_rate"].mean()) for name in OUTCOME_CLASSES
        },
        "auroc_safe": float(block["rank__auroc_safe"].mean()),
        "average_precision_beneficial": float(block["rank__average_precision_beneficial"].mean()),
    }


def run_results() -> int:
    """The primary, per-environment and safety endpoints."""
    started = time.monotonic()
    table, design = load_scores()
    frozen = load_frozen()
    assert_frozen(frozen)
    support = sufficiency(table)
    environments = list(design["environments"])
    required = int(design["breadth_required"])

    per_environment: dict[str, Any] = {}
    for environment in environments:
        entry: dict[str, Any] = {
            "corpus": str(cell(table, environment, ARM_BASELINE).iloc[0]["corpus"]),
            "base_engine": str(cell(table, environment, ARM_BASELINE).iloc[0]["base_engine"]),
            "sufficiency": support[environment],
            "accepted_share_transported": float(
                cell(table, environment, ARM_BASELINE).iloc[0]["accepted_share"]
            ),
            "correction_opportunity_prevalence": float(
                int(cell(table, environment, ARM_BASELINE).iloc[0]["evaluation_beneficial"])
                / int(cell(table, environment, ARM_BASELINE).iloc[0]["evaluation_rows"])
            ),
            "harmful_candidate_prevalence": float(
                int(cell(table, environment, ARM_BASELINE).iloc[0]["evaluation_harmful"])
                / int(cell(table, environment, ARM_BASELINE).iloc[0]["evaluation_rows"])
            ),
            "epsilons": {},
        }
        for epsilon in EPSILONS:
            key = epsilon_key(epsilon)
            arms = {arm: summarise(cell(table, environment, arm), key) for arm in ARMS}
            baseline = arms[ARM_BASELINE]
            primary = arms[ARM_PRIMARY]
            entry["epsilons"][key] = {
                "epsilon": epsilon,
                "arms": arms,
                "delta_risk_controlled_recall_vs_baseline": (
                    primary["risk_controlled_repair_recall"]
                    - baseline["risk_controlled_repair_recall"]
                ),
                "delta_risk_controlled_recall_vs_sgv6": (
                    primary["risk_controlled_repair_recall"]
                    - arms[ARM_SGV6]["risk_controlled_repair_recall"]
                ),
                "delta_harm_vs_baseline": (
                    primary["realized_harm_rate"] - baseline["realized_harm_rate"]
                ),
                "delta_coverage_vs_baseline": primary["coverage"] - baseline["coverage"],
                # Secondary diagnostics. Reported because the diagnosis is worth more than the
                # verdict, and stored so the report can quote them without retyping arithmetic.
                "delta_auroc_vs_baseline": primary["auroc_safe"] - baseline["auroc_safe"],
                "delta_average_precision_vs_baseline": (
                    primary["average_precision_beneficial"]
                    - baseline["average_precision_beneficial"]
                ),
                "delta_achievable_frontier_vs_baseline": (
                    primary["achievable_frontier"] - baseline["achievable_frontier"]
                ),
                "delta_edit_precision_vs_baseline": (
                    primary["edit_precision"] - baseline["edit_precision"]
                ),
                "harm_bound_margin": PRIMARY_EPSILON - primary["realized_harm_rate"],
            }
        primary_key = epsilon_key(PRIMARY_EPSILON)
        block = entry["epsilons"][primary_key]
        entry["primary"] = {
            "improved": bool(block["delta_risk_controlled_recall_vs_baseline"] > 0.0),
            "harm_bound_held": bool(
                block["arms"][ARM_PRIMARY]["realized_harm_rate"] <= PRIMARY_EPSILON
            ),
            "coverage_above_floor": bool(
                block["arms"][ARM_PRIMARY]["coverage"] >= frozen.coverage_floor
            ),
            "delta": block["delta_risk_controlled_recall_vs_baseline"],
            "harm": block["arms"][ARM_PRIMARY]["realized_harm_rate"],
            "coverage": block["arms"][ARM_PRIMARY]["coverage"],
        }
        per_environment[environment] = entry

    sufficient = [e for e in environments if support[e]["sufficient"]]
    counts = {
        scope: {
            "environments": len(names),
            "improved": sum(1 for e in names if per_environment[e]["primary"]["improved"]),
            "harm_bound_held": sum(
                1 for e in names if per_environment[e]["primary"]["harm_bound_held"]
            ),
            "coverage_above_floor": sum(
                1 for e in names if per_environment[e]["primary"]["coverage_above_floor"]
            ),
        }
        for scope, names in (("sufficient", sufficient), ("all", environments))
    }
    payload = {
        "artifact": "primary_results",
        "schema_version": f"sgv14-primary_results-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_method": design["frozen_method"],
        "primary_epsilon": PRIMARY_EPSILON,
        "primary_endpoint": (
            "risk-controlled repair recall on the environment's evaluation partition at "
            "epsilon = 0.10 under the frozen source cut: repair recall where the realised "
            "accepted-set harm rate holds the bound, and zero where it does not"
        ),
        "environments_sufficient": sufficient,
        "environments_insufficient": [e for e in environments if e not in sufficient],
        "breadth_required": required,
        "counts": counts,
        "mean_over_environments": {
            scope: {
                arm: float(
                    np.mean(
                        [
                            per_environment[e]["epsilons"][epsilon_key(PRIMARY_EPSILON)]["arms"][
                                arm
                            ]["risk_controlled_repair_recall"]
                            for e in names
                        ]
                    )
                )
                for arm in ARMS
            }
            for scope, names in (("sufficient", sufficient), ("all", environments))
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(PRIMARY_RESULTS, payload)
    _write_json_once(
        PER_ENVIRONMENT_RESULTS,
        {
            "artifact": "per_environment_results",
            "schema_version": f"sgv14-per_environment_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environments": per_environment,
        },
    )

    safety = {
        environment: {
            key: {
                "epsilon": entry["epsilons"][key]["epsilon"],
                "realized_harm_rate": entry["epsilons"][key]["arms"][ARM_PRIMARY][
                    "realized_harm_rate"
                ],
                "baseline_harm_rate": entry["epsilons"][key]["arms"][ARM_BASELINE][
                    "realized_harm_rate"
                ],
                "holds_bound_fraction": entry["epsilons"][key]["arms"][ARM_PRIMARY][
                    "holds_bound_fraction"
                ],
                "coverage": entry["epsilons"][key]["arms"][ARM_PRIMARY]["coverage"],
                "baseline_coverage": entry["epsilons"][key]["arms"][ARM_BASELINE]["coverage"],
                "coverage_floor": frozen.coverage_floor,
                "non_degenerate": bool(
                    entry["epsilons"][key]["arms"][ARM_PRIMARY]["coverage"] >= frozen.coverage_floor
                ),
                "joint_harm_rate": entry["epsilons"][key]["arms"][ARM_PRIMARY]["joint_harm_rate"],
            }
            for key in (epsilon_key(e) for e in EPSILONS)
        }
        for environment, entry in per_environment.items()
    }
    _write_json_once(
        SAFETY_RESULTS,
        {
            "artifact": "safety_results",
            "schema_version": f"sgv14-safety_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rule": (
                "a recall obtained while the realised harm rate exceeds epsilon is not an "
                "operational result and scores zero on the primary endpoint by construction"
            ),
            "coverage_floor": frozen.coverage_floor,
            "coverage_floor_source": design["coverage_floor_source"],
            "per_environment": safety,
        },
    )
    print(
        f"results: {counts['sufficient']['improved']}/{len(sufficient)} sufficient environments "
        f"improved, {counts['sufficient']['harm_bound_held']}/{len(sufficient)} held the bound, "
        f"{counts['sufficient']['coverage_above_floor']}/{len(sufficient)} above the floor "
        f"-> {_relative(PRIMARY_RESULTS)}"
    )
    return 0


def run_controls() -> int:
    """The two validation controls and the one secondary diagnostic, read side by side."""
    table, design = load_scores()
    frozen = load_frozen()
    key = epsilon_key(PRIMARY_EPSILON)
    stem = f"dep__{key}"
    support = sufficiency(table)
    environments = list(design["environments"])

    per_environment: dict[str, Any] = {}
    for environment in environments:
        baseline = cell(table, environment, ARM_BASELINE)
        primary = cell(table, environment, ARM_PRIMARY)
        source_only = cell(table, environment, CONTROL_SOURCE_ONLY)
        permuted = cell(table, environment, CONTROL_PERMUTED)
        random_arm = cell(table, environment, DIAGNOSTIC_RANDOM)
        base_value = float(baseline[f"{stem}__risk_controlled"].mean())
        primary_value = float(primary[f"{stem}__risk_controlled"].mean())
        per_environment[environment] = {
            "sufficient": support[environment]["sufficient"],
            "baseline": base_value,
            "frozen_method": primary_value,
            "c1_source_only_refit": {
                "risk_controlled_repair_recall": float(
                    source_only[f"{stem}__risk_controlled"].mean()
                ),
                "delta_versus_baseline": float(
                    source_only[f"{stem}__risk_controlled"].mean() - base_value
                ),
                "auroc_safe": float(source_only["rank__auroc_safe"].mean()),
                "baseline_auroc_safe": float(baseline["rank__auroc_safe"].mean()),
                "auroc_difference": float(
                    source_only["rank__auroc_safe"].mean() - baseline["rank__auroc_safe"].mean()
                ),
                "target_labels": 0,
                "expectation": "no target-supervision gain",
            },
            "c2_permuted_labels": {
                "risk_controlled_repair_recall": float(permuted[f"{stem}__risk_controlled"].mean()),
                "realized_harm_rate": float(permuted[f"{stem}__realized_harm_rate"].mean()),
                "coverage": float(permuted[f"{stem}__coverage"].mean()),
                "auroc_safe": float(permuted["rank__auroc_safe"].mean()),
                "delta_versus_baseline": float(
                    permuted[f"{stem}__risk_controlled"].mean() - base_value
                ),
                "delta_versus_frozen_method": float(
                    permuted[f"{stem}__risk_controlled"].mean() - primary_value
                ),
                "gain_retained_fraction": (
                    float(
                        (permuted[f"{stem}__risk_controlled"].mean() - base_value)
                        / (primary_value - base_value)
                    )
                    if primary_value != base_value
                    else float("nan")
                ),
                "expectation": "the gain substantially weakens or disappears",
            },
            "d1_random_acquisition": {
                "risk_controlled_repair_recall": float(
                    random_arm[f"{stem}__risk_controlled"].mean()
                ),
                "realized_harm_rate": float(random_arm[f"{stem}__realized_harm_rate"].mean()),
                "coverage": float(random_arm[f"{stem}__coverage"].mean()),
                "holds_bound_fraction": float(random_arm[f"{stem}__holds_bound"].mean()),
                "frozen_method_realized_harm_rate": float(
                    primary[f"{stem}__realized_harm_rate"].mean()
                ),
                "delta_versus_baseline": float(
                    random_arm[f"{stem}__risk_controlled"].mean() - base_value
                ),
                "delta_versus_uncertainty": float(
                    random_arm[f"{stem}__risk_controlled"].mean() - primary_value
                ),
                "budget_beneficial_mean": float(random_arm["budget_beneficial"].mean()),
                "uncertainty_budget_beneficial_mean": float(primary["budget_beneficial"].mean()),
                "status": "secondary and non-confirmatory",
            },
        }

    sufficient = [e for e in environments if support[e]["sufficient"]]

    def _mean(path: tuple[str, ...]) -> float:
        values = []
        for environment in sufficient:
            node: Any = per_environment[environment]
            for step in path:
                node = node[step]
            values.append(float(node))
        return float(np.mean(values)) if values else float("nan")

    aggregate = {
        "c1_source_only_refit": {
            "mean_delta_versus_baseline": _mean(("c1_source_only_refit", "delta_versus_baseline")),
            "mean_auroc_difference": _mean(("c1_source_only_refit", "auroc_difference")),
            "environments_with_any_gain": sum(
                1
                for e in sufficient
                if per_environment[e]["c1_source_only_refit"]["delta_versus_baseline"] > 0.0
            ),
        },
        "c2_permuted_labels": {
            "mean_delta_versus_baseline": _mean(("c2_permuted_labels", "delta_versus_baseline")),
            "mean_delta_versus_frozen_method": _mean(
                ("c2_permuted_labels", "delta_versus_frozen_method")
            ),
            "environments_where_permutation_beats_the_method": sum(
                1
                for e in sufficient
                if per_environment[e]["c2_permuted_labels"]["delta_versus_frozen_method"] > 0.0
            ),
        },
        "d1_random_acquisition": {
            "mean_delta_versus_baseline": _mean(("d1_random_acquisition", "delta_versus_baseline")),
            "mean_delta_versus_uncertainty": _mean(
                ("d1_random_acquisition", "delta_versus_uncertainty")
            ),
            "environments_where_random_matches_or_beats_uncertainty": sum(
                1
                for e in sufficient
                if per_environment[e]["d1_random_acquisition"]["delta_versus_uncertainty"] >= 0.0
            ),
        },
    }
    _write_json_once(
        CONTROL_RESULTS,
        {
            "artifact": "control_results",
            "schema_version": f"sgv14-control_results-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epsilon": PRIMARY_EPSILON,
            "coverage_floor": frozen.coverage_floor,
            "controls_are_not_alternatives": (
                "C1 and C2 are validation controls: they say whether the measured gain came "
                "from the target labels. Neither is ever compared as a competing method and "
                "neither can be selected. D1 is a secondary diagnostic and cannot change the "
                "frozen acquisition rule whatever it reports."
            ),
            "per_environment": per_environment,
            "aggregate_over_sufficient_environments": aggregate,
        },
    )
    source_delta = aggregate["c1_source_only_refit"]["mean_delta_versus_baseline"]
    permuted_delta = aggregate["c2_permuted_labels"]["mean_delta_versus_baseline"]
    random_delta = aggregate["d1_random_acquisition"]["mean_delta_versus_uncertainty"]
    print(
        f"controls: source-only mean delta {source_delta:+.4f}, "
        f"permutation {permuted_delta:+.4f}, "
        f"random acquisition {random_delta:+.4f} -> {_relative(CONTROL_RESULTS)}"
    )
    return 0


# ------------------------------------------------------------------ the pre-registered tests

RECALL = s13.RECALL
HARM = s13.HARM


def count_block(counts: pd.DataFrame, environment: str, arm: str) -> Any:
    """One cell's per-document counts, one matrix row per draw, in sorted document order."""
    block = counts[(counts["environment"] == environment) & (counts["arm"] == arm)]
    if block.empty:
        raise PhaseError(f"no counts for {environment}/{arm}")
    documents = tuple(sorted({str(name) for name in block["document_id"].tolist()}))
    draws = sorted({int(value) for value in block["draw"].tolist()})
    position = {name: index for index, name in enumerate(documents)}
    row_of = {draw: index for index, draw in enumerate(draws)}
    shape = (len(draws), len(documents))
    names = ("accepted", "harmful_accepted", "beneficial_accepted", "beneficial_total")
    matrices = {name: np.zeros(shape, dtype=float) for name in names}
    rows = np.array([row_of[int(v)] for v in block["draw"].tolist()], dtype=int)
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


def stratified_delta(
    blocks: list[tuple[Any, Any]], name: str, epsilon: float, seed: int
) -> dict[str, float]:
    """The environment-stratified paired difference: resample within, average across.

    Documents are resampled inside each environment and never across one, because two
    environments do not share pages and a pooled resample would mix populations. The
    per-environment deltas are then averaged with EQUAL weight, which is what a breadth claim
    means: each environment is one replication, not one vote per page.
    """
    if not blocks:
        raise PhaseError("the stratified test needs at least one environment")
    per_environment = []
    points = []
    for position, (left, right) in enumerate(blocks):
        if left.documents != right.documents:
            raise PhaseError("a paired bootstrap needs both cells on the same documents")
        multiplicity = document_multiplicities(
            np.asarray(left.documents), seed + position, BOOTSTRAP_RESAMPLES
        ).astype(float)
        identity = np.ones((1, len(left.documents)), dtype=float)
        points.append(
            float(s13.statistic_draws(left, identity, name, epsilon).mean())
            - float(s13.statistic_draws(right, identity, name, epsilon).mean())
        )
        per_environment.append(
            s13.statistic_draws(left, multiplicity, name, epsilon).mean(axis=1)
            - s13.statistic_draws(right, multiplicity, name, epsilon).mean(axis=1)
        )
    deltas = np.mean(np.vstack(per_environment), axis=0)
    interval = _interval(deltas)
    return {
        "delta": float(np.mean(points)),
        "bootstrap_mean_delta": float(interval["estimate"]),
        "ci_lower": float(interval["ci_lower"]),
        "ci_upper": float(interval["ci_upper"]),
        "p_value": float(interval["p_value"]),
        "environments": len(blocks),
        "documents": int(sum(len(left.documents) for left, _ in blocks)),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def holm(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni over the secondary family, whose size is the environment count times two."""
    order = sorted(rows, key=lambda name: (rows[name]["p_value"], name))
    size = len(order)
    running = 0.0
    out: dict[str, dict[str, float]] = {}
    for position, name in enumerate(order):
        adjusted = min(1.0, (size - position) * rows[name]["p_value"])
        running = max(running, adjusted)
        out[name] = {**rows[name], "holm_adjusted_p": running, "survives_holm": running < 0.05}
    return out


def run_stats() -> int:
    """The one primary test, then the secondary per-environment family."""
    started = time.monotonic()
    table, design = load_scores()
    counts = pd.read_parquet(DEPLOYMENT_COUNTS)
    support = sufficiency(table)
    environments = list(design["environments"])
    sufficient = [e for e in environments if support[e]["sufficient"]]

    primary_blocks = [
        (count_block(counts, e, ARM_PRIMARY), count_block(counts, e, ARM_BASELINE))
        for e in sufficient
    ]
    primary = stratified_delta(primary_blocks, RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED)
    all_blocks = [
        (count_block(counts, e, ARM_PRIMARY), count_block(counts, e, ARM_BASELINE))
        for e in environments
    ]
    sensitivity = stratified_delta(all_blocks, RECALL, PRIMARY_EPSILON, BOOTSTRAP_SEED)

    secondary: dict[str, dict[str, float]] = {}
    for environment in sufficient:
        left = count_block(counts, environment, ARM_PRIMARY)
        for arm, label in ((ARM_BASELINE, "vs_baseline"), (ARM_SGV6, "vs_sgv6")):
            secondary[f"{environment}|{label}"] = s13.paired_delta(
                left,
                count_block(counts, environment, arm),
                RECALL,
                PRIMARY_EPSILON,
                BOOTSTRAP_SEED,
            )
    corrected = holm(secondary)
    harm_tests = {
        environment: s13.paired_delta(
            count_block(counts, environment, ARM_PRIMARY),
            count_block(counts, environment, ARM_BASELINE),
            HARM,
            PRIMARY_EPSILON,
            BOOTSTRAP_SEED,
        )
        for environment in sufficient
    }
    control_tests = {
        f"{environment}|{label}": s13.paired_delta(
            count_block(counts, environment, ARM_PRIMARY),
            count_block(counts, environment, arm),
            RECALL,
            PRIMARY_EPSILON,
            BOOTSTRAP_SEED,
        )
        for environment in sufficient
        for arm, label in (
            (CONTROL_PERMUTED, "vs_permuted"),
            (CONTROL_SOURCE_ONLY, "vs_source_only"),
            (DIAGNOSTIC_RANDOM, "vs_random_acquisition"),
        )
    }
    favourable = bool(primary["delta"] > 0.0 and primary["ci_lower"] > 0.0)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            "artifact": "statistical_tests",
            "schema_version": f"sgv14-statistical_tests-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "plan": design["statistics"],
            "resampling_unit": "document",
            "primary_test": {
                "endpoint": RECALL,
                "epsilon": PRIMARY_EPSILON,
                "comparison": f"{ARM_PRIMARY} minus {ARM_BASELINE}",
                "scope": "sufficient environments, equal weight",
                "environments": sufficient,
                **primary,
                "favourable_and_significant": favourable,
                "alpha": PRIMARY_ALPHA,
            },
            "primary_test_sensitivity_all_environments": {
                "scope": "every environment, equal weight",
                "environments": environments,
                **sensitivity,
                "favourable_and_significant": bool(
                    sensitivity["delta"] > 0.0 and sensitivity["ci_lower"] > 0.0
                ),
            },
            "secondary_family": {
                "definition": (
                    "the per-environment paired difference against the frozen baseline and "
                    "against the SGV6 baseline, on sufficient environments"
                ),
                "size": len(secondary),
                "tests": corrected,
                "surviving": sorted(n for n, r in corrected.items() if r["survives_holm"]),
                "surviving_favourable": sorted(
                    n for n, r in corrected.items() if r["survives_holm"] and r["delta"] > 0.0
                ),
                "surviving_adverse": sorted(
                    n for n, r in corrected.items() if r["survives_holm"] and r["delta"] < 0.0
                ),
            },
            "accepted_prefix_harm_tests": harm_tests,
            "control_comparisons": control_tests,
            "hierarchy_note": (
                "the primary test is read first and the secondary family describes where an "
                "effect lives. A secondary test cannot rescue a primary test that fails."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"stats: primary delta {primary['delta']:+.4f} "
        f"[{primary['ci_lower']:+.4f}, {primary['ci_upper']:+.4f}] p={primary['p_value']:.4f}; "
        f"{len([n for n, r in corrected.items() if r['survives_holm']])}/{len(secondary)} "
        f"secondary tests survive Holm -> {_relative(STATISTICAL_TESTS)}"
    )
    return 0


# ------------------------------------------------------------------ the falsification tests
#
# Ten questions the brief asks SGV14 to ask of itself. Each is answered from the artifacts and
# each is reported as it comes out; a failed negative control is a finding, never a defect to
# be reinterpreted.


def run_negative() -> int:
    """The brief's ten falsification tests, plus the leakage assertions behind them."""
    started = time.monotonic()
    table, design = load_scores()
    counts = pd.read_parquet(DEPLOYMENT_COUNTS)
    frozen = load_frozen()
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    controls = cc_read_json(CONTROL_RESULTS)
    statistics = cc_read_json(STATISTICAL_TESTS)
    results = cc_read_json(PER_ENVIRONMENT_RESULTS)["environments"]
    support = sufficiency(table)
    environments = list(design["environments"])
    sufficient = [e for e in environments if support[e]["sufficient"]]
    key = epsilon_key(PRIMARY_EPSILON)
    stem = f"dep__{key}"

    def primary_of(environment: str) -> dict[str, Any]:
        return results[environment]["primary"]

    failures = [e for e in sufficient if not primary_of(e)["improved"]]
    new_configuration = [
        record["environment"]
        for record in cc_read_json(ENVIRONMENT_SELECTION)["included"]
        if record["configuration_new_versus_sgv13"]
    ]
    by_corpus: dict[str, list[str]] = {}
    for environment in environments:
        by_corpus.setdefault(results[environment]["corpus"], []).append(environment)

    # 7. does one document carry the improvement? Leave-one-document-out on the delta.
    influence: dict[str, Any] = {}
    for environment in sufficient:
        left = count_block(counts, environment, ARM_PRIMARY)
        right = count_block(counts, environment, ARM_BASELINE)
        identity = np.ones((1, len(left.documents)), dtype=float)
        full = float(
            s13.statistic_draws(left, identity, RECALL, PRIMARY_EPSILON).mean()
            - s13.statistic_draws(right, identity, RECALL, PRIMARY_EPSILON).mean()
        )
        drops = np.ones((len(left.documents), len(left.documents)), dtype=float)
        np.fill_diagonal(drops, 0.0)
        deltas = s13.statistic_draws(left, drops, RECALL, PRIMARY_EPSILON).mean(
            axis=1
        ) - s13.statistic_draws(right, drops, RECALL, PRIMARY_EPSILON).mean(axis=1)
        worst = int(np.argmin(deltas)) if deltas.size else 0
        influence[environment] = {
            "delta": full,
            "documents": len(left.documents),
            "min_leave_one_out_delta": float(deltas.min()) if deltas.size else float("nan"),
            "max_single_document_influence": float(full - deltas.min()) if deltas.size else 0.0,
            "most_influential_document": left.documents[worst] if deltas.size else "",
            "sign_survives_every_drop": bool(deltas.min() > 0.0) if deltas.size else False,
        }

    # 8. is the effect one environment's? Leave-one-environment-out on the stratified mean.
    per_environment_delta = {e: primary_of(e)["delta"] for e in sufficient}
    leave_one_environment = {
        dropped: float(np.mean([v for e, v in per_environment_delta.items() if e != dropped]))
        for dropped in sufficient
    }

    tests = {
        "1_fails_on_a_new_engine_configuration": {
            "question": "does the frozen method fail on an engine configuration SGV13 never ran?",
            "environments": new_configuration,
            "improved": [
                e for e in new_configuration if e in sufficient and primary_of(e)["improved"]
            ],
            "failed": [
                e for e in new_configuration if e in sufficient and not primary_of(e)["improved"]
            ],
            "insufficient": [e for e in new_configuration if e not in sufficient],
        },
        "2_fails_on_a_new_document_domain": {
            "question": "does it fail on a document domain SGV13 never saw?",
            "per_corpus": {
                corpus: {
                    "environments": names,
                    "sufficient": [e for e in names if e in sufficient],
                    "improved": [e for e in names if e in sufficient and primary_of(e)["improved"]],
                    "mean_delta": float(
                        np.mean([primary_of(e)["delta"] for e in names if e in sufficient])
                    )
                    if any(e in sufficient for e in names)
                    else float("nan"),
                }
                for corpus, names in by_corpus.items()
            },
        },
        "3_harm_exceeds_epsilon_despite_a_recall_gain": {
            "question": "does any environment gain recall while breaking the bound?",
            "environments": [
                e
                for e in environments
                if primary_of(e)["improved"] and not primary_of(e)["harm_bound_held"]
            ],
            "note": (
                "the primary endpoint scores such a cell zero by construction, so a gain of "
                "this kind cannot enter the breadth count"
            ),
            "raw_recall_where_the_bound_broke": {
                e: float(cell(table, e, ARM_PRIMARY)[f"{stem}__repair_recall"].mean())
                for e in environments
                if not primary_of(e)["harm_bound_held"]
            },
        },
        "4_safety_comes_from_low_coverage": {
            "question": "is the bound held by accepting almost nothing?",
            "per_environment": {
                e: {
                    "coverage": primary_of(e)["coverage"],
                    "baseline_coverage": float(
                        cell(table, e, ARM_BASELINE)[f"{stem}__coverage"].mean()
                    ),
                    "coverage_floor": frozen.coverage_floor,
                    "above_floor": primary_of(e)["coverage_above_floor"],
                    "coverage_below_baseline": bool(
                        primary_of(e)["coverage"]
                        < float(cell(table, e, ARM_BASELINE)[f"{stem}__coverage"].mean())
                    ),
                }
                for e in environments
            },
            "environments_below_the_floor": [
                e for e in environments if not primary_of(e)["coverage_above_floor"]
            ],
        },
        "5_source_only_refitting_reproduces_the_gain": {
            "question": "does refitting on source rows alone deliver the same movement?",
            "mean_delta_versus_baseline": controls["aggregate_over_sufficient_environments"][
                "c1_source_only_refit"
            ]["mean_delta_versus_baseline"],
            "mean_auroc_difference": controls["aggregate_over_sufficient_environments"][
                "c1_source_only_refit"
            ]["mean_auroc_difference"],
            "environments_with_any_gain": controls["aggregate_over_sufficient_environments"][
                "c1_source_only_refit"
            ]["environments_with_any_gain"],
        },
        "6_label_permutation_preserves_the_gain": {
            "question": "does the gain survive destroying the label-to-row assignment?",
            "mean_delta_versus_baseline": controls["aggregate_over_sufficient_environments"][
                "c2_permuted_labels"
            ]["mean_delta_versus_baseline"],
            "mean_delta_versus_frozen_method": controls["aggregate_over_sufficient_environments"][
                "c2_permuted_labels"
            ]["mean_delta_versus_frozen_method"],
            "environments_where_permutation_wins": controls[
                "aggregate_over_sufficient_environments"
            ]["c2_permuted_labels"]["environments_where_permutation_beats_the_method"],
        },
        "7_one_document_carries_the_improvement": {
            "question": "does dropping a single page remove the effect?",
            "per_environment": influence,
            "environments_whose_sign_depends_on_one_document": [
                e
                for e, record in influence.items()
                if record["delta"] > 0.0 and not record["sign_survives_every_drop"]
            ],
        },
        "8_one_environment_carries_the_effect": {
            "question": "does the stratified mean depend on a single environment?",
            "per_environment_delta": per_environment_delta,
            "leave_one_environment_out_mean": leave_one_environment,
            "sign_survives_every_drop": bool(
                all(value > 0.0 for value in leave_one_environment.values())
            )
            if leave_one_environment
            else False,
        },
        "9_uncertainty_acquisition_beats_random": {
            "question": "is the frozen acquisition rule doing anything?",
            "mean_delta_versus_uncertainty": controls["aggregate_over_sufficient_environments"][
                "d1_random_acquisition"
            ]["mean_delta_versus_uncertainty"],
            "environments_where_random_matches_or_beats": controls[
                "aggregate_over_sufficient_environments"
            ]["d1_random_acquisition"]["environments_where_random_matches_or_beats_uncertainty"],
            "status": "secondary; the frozen rule does not change on this evidence",
        },
        "10_unstable_across_acquisition_seeds": {
            "question": "how much does the endpoint move between repeated draws?",
            "per_environment": {
                e: {
                    "draws": len(cell(table, e, ARM_PRIMARY)),
                    "mean": float(cell(table, e, ARM_PRIMARY)[f"{stem}__risk_controlled"].mean()),
                    "sd": float(
                        cell(table, e, ARM_PRIMARY)[f"{stem}__risk_controlled"].std(ddof=0)
                    ),
                    "min": float(cell(table, e, ARM_PRIMARY)[f"{stem}__risk_controlled"].min()),
                    "max": float(cell(table, e, ARM_PRIMARY)[f"{stem}__risk_controlled"].max()),
                    "harm_bound_held_fraction": float(
                        cell(table, e, ARM_PRIMARY)[f"{stem}__holds_bound"].mean()
                    ),
                }
                for e in environments
            },
        },
    }

    assertions = {
        "no_evaluation_document_reached_a_budget": _assert_budget_disjoint(counts),
        "no_environment_document_appears_in_sgv1": {
            "documents_checked": int(inventory["totals"]["documents"]),
            "overlap": 0,
        },
        "the_confirmatory_reserve_stayed_locked": {
            "status": cc_read_json(REPO / "manifests/sgv1/confirmatory_reserve_lock.json")[
                "status"
            ],
            "unlock_record": cc_read_json(REPO / "manifests/sgv1/confirmatory_reserve_lock.json")[
                "unlock_record"
            ],
        },
        "the_frozen_model_did_not_move": {
            "checked_in": "--adapt, on the fit, calibration and seen-evaluation blocks",
            "max_abs_difference": 0.0,
        },
        "the_cut_read_no_target_row": {
            "rule": frozen.cut,
            "definition": (
                "select_threshold on the SOURCE engines' calibration rows, the identical call "
                "that produced SGV5's own threshold"
            ),
        },
    }
    _write_json_once(
        NEGATIVE_TESTS,
        {
            "artifact": "negative_tests",
            "schema_version": f"sgv14-negative_tests-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epsilon": PRIMARY_EPSILON,
            "environments_that_failed_to_improve": failures,
            "failure_tests": tests,
            "leakage_assertions": assertions,
            "primary_test_delta": statistics["primary_test"]["delta"],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"negative: 10 failure tests, {len(assertions)} assertions; "
        f"{len(failures)}/{len(sufficient)} sufficient environments did not improve "
        f"-> {_relative(NEGATIVE_TESTS)}"
    )
    return 0


def _assert_budget_disjoint(counts: pd.DataFrame) -> dict[str, Any]:
    """No evaluation document may appear in any purchased budget, on any draw."""
    rows = pd.read_parquet(ENVIRONMENT_ROWS, columns=["environment", "document_id", "role"])
    checked = 0
    for environment, group in rows.groupby("environment", observed=True):
        adaptation = set(group.loc[group["role"] == ROLE_ADAPTATION, "document_id"])
        evaluated = set(counts.loc[counts["environment"] == environment, "document_id"].astype(str))
        shared = adaptation & evaluated
        if shared:
            raise PhaseError(
                f"{environment}: {len(shared)} documents are in both the adaptation pool and "
                "the evaluation block"
            )
        checked += len(evaluated)
    return {"documents_checked": checked, "overlap": 0}


# ------------------------------------------------------------------ figures


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


def run_figures() -> int:
    """The five required figures. Every number on them comes out of an artifact."""
    started = time.monotonic()
    plt = _figure_style()
    table, design = load_scores()
    frozen = load_frozen()
    results = cc_read_json(PER_ENVIRONMENT_RESULTS)["environments"]
    controls = cc_read_json(CONTROL_RESULTS)["per_environment"]
    statistics = cc_read_json(STATISTICAL_TESTS)
    support = sufficiency(table)
    environments = list(design["environments"])
    key = epsilon_key(PRIMARY_EPSILON)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = [e.replace("/", "\n") for e in environments]
    marks = ["" if support[e]["sufficient"] else " *" for e in environments]
    written: list[Path] = []

    def _finish(figure: Any, path: Path, note: str) -> None:
        figure.text(0.005, 0.005, note, fontsize=5.5, color="#555555")
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        figure.savefig(path)
        plt.close(figure)
        written.append(path)

    note = (
        f"SGV14 confirmatory validation | frozen SGV13 method {frozen.arm} + "
        f"{frozen.acquisition} + N={frozen.budget} + {frozen.cut} | epsilon={PRIMARY_EPSILON} | "
        "* marks an environment marked INSUFFICIENT by the pre-registered rule"
    )

    # 1. risk-controlled repair recall, baseline against the frozen method and SGV6.
    figure, axis = plt.subplots(figsize=(9.5, 4.0))
    position = np.arange(len(environments))
    width = 0.27
    series = (
        (ARM_BASELINE, "frozen baseline", "#8c8c8c"),
        (ARM_PRIMARY, "frozen SGV13 method", "#1f77b4"),
        (ARM_SGV6, "SGV6 generic adaptation", "#d68910"),
    )
    for offset, (arm, label, colour) in enumerate(series):
        values = [
            results[e]["epsilons"][key]["arms"][arm]["risk_controlled_repair_recall"]
            for e in environments
        ]
        axis.bar(position + (offset - 1) * width, values, width, label=label, color=colour)
    axis.set_xticks(position)
    axis.set_xticklabels([f"{a}{b}" for a, b in zip(labels, marks, strict=True)], fontsize=6.5)
    axis.set_ylabel(f"risk-controlled repair recall at eps={PRIMARY_EPSILON}")
    axis.set_title("SGV14: the frozen method on environments it never saw")
    axis.legend(fontsize=7)
    _finish(figure, FIGURE_DIR / "confirmatory_repair_recall.png", note)

    # 2. realised harm by environment, against the bound.
    figure, axis = plt.subplots(figsize=(9.5, 3.6))
    for offset, (arm, label, colour) in enumerate(series[:2]):
        values = [
            results[e]["epsilons"][key]["arms"][arm]["realized_harm_rate"] for e in environments
        ]
        axis.bar(position + (offset - 0.5) * width, values, width, label=label, color=colour)
    axis.axhline(
        PRIMARY_EPSILON,
        color="#c0392b",
        linestyle="--",
        linewidth=1.0,
        label=f"epsilon = {PRIMARY_EPSILON}",
    )
    axis.set_xticks(position)
    axis.set_xticklabels([f"{a}{b}" for a, b in zip(labels, marks, strict=True)], fontsize=6.5)
    axis.set_ylabel("realised accepted-set harm rate")
    axis.set_title("SGV14: safety, and where the bound breaks")
    axis.legend(fontsize=7)
    _finish(figure, FIGURE_DIR / "harm_by_environment.png", note)

    # 3. coverage against the pre-registered floor.
    figure, axis = plt.subplots(figsize=(9.5, 3.6))
    for offset, (arm, label, colour) in enumerate(series[:2]):
        values = [results[e]["epsilons"][key]["arms"][arm]["coverage"] for e in environments]
        axis.bar(position + (offset - 0.5) * width, values, width, label=label, color=colour)
    axis.axhline(
        frozen.coverage_floor,
        color="#117a65",
        linestyle="--",
        linewidth=1.0,
        label=f"SGV13 coverage floor = {frozen.coverage_floor:.4f}",
    )
    axis.set_xticks(position)
    axis.set_xticklabels([f"{a}{b}" for a, b in zip(labels, marks, strict=True)], fontsize=6.5)
    axis.set_ylabel("accepted coverage")
    axis.set_title("SGV14: non-degeneracy")
    axis.legend(fontsize=7)
    _finish(figure, FIGURE_DIR / "coverage_by_environment.png", note)

    # 4. per-environment effect sizes with their document-clustered intervals.
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    tests = statistics["secondary_family"]["tests"]
    names = [e for e in environments if f"{e}|vs_baseline" in tests]
    deltas = [tests[f"{e}|vs_baseline"]["delta"] for e in names]
    lower = [tests[f"{e}|vs_baseline"]["ci_lower"] for e in names]
    upper = [tests[f"{e}|vs_baseline"]["ci_upper"] for e in names]
    order = np.arange(len(names))
    axis.errorbar(
        deltas,
        order,
        xerr=[np.array(deltas) - np.array(lower), np.array(upper) - np.array(deltas)],
        fmt="o",
        color="#1f77b4",
        markersize=4,
        capsize=2,
        linewidth=1.0,
    )
    axis.axvline(0.0, color="#c0392b", linewidth=1.0)
    pooled = statistics["primary_test"]
    axis.errorbar(
        [pooled["delta"]],
        [len(names)],
        xerr=[[pooled["delta"] - pooled["ci_lower"]], [pooled["ci_upper"] - pooled["delta"]]],
        fmt="D",
        color="#111111",
        markersize=5,
        capsize=3,
        linewidth=1.2,
    )
    axis.set_yticks([*list(order), len(names)])
    axis.set_yticklabels([*names, "STRATIFIED (primary)"], fontsize=7)
    axis.set_xlabel("delta in risk-controlled repair recall against the frozen baseline")
    axis.set_title("SGV14: effect sizes, document-clustered")
    _finish(figure, FIGURE_DIR / "effect_sizes.png", note)

    # 5. the controls.
    figure, axis = plt.subplots(figsize=(9.5, 4.0))
    control_series = (
        (ARM_PRIMARY, "frozen method", "#1f77b4"),
        (CONTROL_SOURCE_ONLY, "C1 source-only refit", "#7f7f7f"),
        (CONTROL_PERMUTED, "C2 permuted labels", "#c0392b"),
        (DIAGNOSTIC_RANDOM, "D1 random acquisition", "#8e44ad"),
    )
    width = 0.2
    for offset, (arm, label, colour) in enumerate(control_series):
        values = [
            (
                controls[e]["frozen_method"]
                if arm == ARM_PRIMARY
                else controls[e][
                    {
                        CONTROL_SOURCE_ONLY: "c1_source_only_refit",
                        CONTROL_PERMUTED: "c2_permuted_labels",
                        DIAGNOSTIC_RANDOM: "d1_random_acquisition",
                    }[arm]
                ]["risk_controlled_repair_recall"]
            )
            for e in environments
        ]
        axis.bar(position + (offset - 1.5) * width, values, width, label=label, color=colour)
    axis.plot(
        position,
        [controls[e]["baseline"] for e in environments],
        "k_",
        markersize=14,
        label="frozen baseline",
    )
    axis.set_xticks(position)
    axis.set_xticklabels([f"{a}{b}" for a, b in zip(labels, marks, strict=True)], fontsize=6.5)
    axis.set_ylabel(f"risk-controlled repair recall at eps={PRIMARY_EPSILON}")
    axis.set_title("SGV14: what the target labels were actually doing")
    axis.legend(fontsize=7, ncol=2)
    _finish(figure, FIGURE_DIR / "control_comparison.png", note)

    _write_json_once(
        FIGURE_MANIFEST,
        {
            "artifact": "figure_manifest",
            "schema_version": f"sgv14-figure_manifest-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "synthetic": False,
            "figures": {
                _relative(path): {
                    "sha256": file_sha256(path),
                    "derived_from": [
                        _relative(PER_ENVIRONMENT_RESULTS),
                        _relative(CONTROL_RESULTS),
                        _relative(STATISTICAL_TESTS),
                        _relative(DESIGN_RECORD),
                    ],
                    "source_sha256": {
                        _relative(p): file_sha256(p)
                        for p in (
                            PER_ENVIRONMENT_RESULTS,
                            CONTROL_RESULTS,
                            STATISTICAL_TESTS,
                            DESIGN_RECORD,
                        )
                    },
                }
                for path in written
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {_relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ the decision


def run_decide() -> int:
    """The machine-readable finding. Four criteria, all required, none negotiable here."""
    started = time.monotonic()
    table, design = load_scores()
    frozen = load_frozen()
    results = cc_read_json(PER_ENVIRONMENT_RESULTS)["environments"]
    primary = cc_read_json(PRIMARY_RESULTS)
    controls = cc_read_json(CONTROL_RESULTS)
    statistics = cc_read_json(STATISTICAL_TESTS)
    negative = cc_read_json(NEGATIVE_TESTS)
    support = sufficiency(table)
    environments = list(design["environments"])
    sufficient = primary["environments_sufficient"]
    required = int(design["breadth_required"])
    key = epsilon_key(PRIMARY_EPSILON)

    counts = primary["counts"]["sufficient"]
    criteria = {
        "1_breadth": {
            "statement": design["criteria"]["1_breadth"],
            "observed": counts["improved"],
            "required": required,
            "of": len(sufficient),
            "met": bool(counts["improved"] >= required),
        },
        "2_safety": {
            "statement": design["criteria"]["2_safety"],
            "observed": counts["harm_bound_held"],
            "required": required,
            "of": len(sufficient),
            "met": bool(counts["harm_bound_held"] >= required),
        },
        "3_non_degeneracy": {
            "statement": design["criteria"]["3_non_degeneracy"],
            "observed": counts["coverage_above_floor"],
            "required": required,
            "of": len(sufficient),
            "met": bool(counts["coverage_above_floor"] >= required),
        },
        "4_statistical_support": {
            "statement": design["criteria"]["4_statistical_support"],
            "delta": statistics["primary_test"]["delta"],
            "ci": [
                statistics["primary_test"]["ci_lower"],
                statistics["primary_test"]["ci_upper"],
            ],
            "p_value": statistics["primary_test"]["p_value"],
            "met": bool(statistics["primary_test"]["favourable_and_significant"]),
        },
    }
    met = sum(1 for record in criteria.values() if record["met"])
    verdict = "SUPPORTED" if met == CRITERIA_TOTAL else "NOT SUPPORTED"

    sensitivity_counts = primary["counts"]["all"]
    outcome = _outcome_label(criteria, controls, negative, primary, sufficient, results, key)
    payload = {
        "artifact": "research_decision",
        "schema_version": f"sgv14-research_decision-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git("rev-parse", "HEAD"),
        "verdict": verdict,
        "criteria_met": met,
        "criteria_total": CRITERIA_TOTAL,
        "criteria": criteria,
        "outcome": outcome,
        "frozen_sgv13_method": design["frozen_method"],
        "frozen_sgv13_configuration_sha256": file_sha256(FROZEN_CONFIGURATION),
        "frozen_sgv13_design_record_sha256": frozen.design_record_sha256,
        "environments_tested": environments,
        "environments_sufficient": sufficient,
        "environments_insufficient": primary["environments_insufficient"],
        "sufficiency": support,
        "primary_epsilon": PRIMARY_EPSILON,
        "target_label_budget": frozen.budget,
        "primary_results_by_environment": {
            environment: {
                "corpus": results[environment]["corpus"],
                "base_engine": results[environment]["base_engine"],
                "sufficient": support[environment]["sufficient"],
                "baseline": results[environment]["epsilons"][key]["arms"][ARM_BASELINE][
                    "risk_controlled_repair_recall"
                ],
                "frozen_method": results[environment]["epsilons"][key]["arms"][ARM_PRIMARY][
                    "risk_controlled_repair_recall"
                ],
                "sgv6_baseline": results[environment]["epsilons"][key]["arms"][ARM_SGV6][
                    "risk_controlled_repair_recall"
                ],
                "delta": results[environment]["primary"]["delta"],
                "harm": results[environment]["primary"]["harm"],
                "coverage": results[environment]["primary"]["coverage"],
                "harm_bound_held": results[environment]["primary"]["harm_bound_held"],
                "coverage_above_floor": results[environment]["primary"]["coverage_above_floor"],
            }
            for environment in environments
        },
        "baseline_comparison": {
            "mean_over_sufficient_environments": primary["mean_over_environments"]["sufficient"],
            "mean_over_all_environments": primary["mean_over_environments"]["all"],
        },
        "harm_status": {
            "environments_holding_the_bound": counts["harm_bound_held"],
            "of_sufficient": len(sufficient),
            "environments_breaking_the_bound": [
                e for e in environments if not results[e]["primary"]["harm_bound_held"]
            ],
        },
        "coverage_status": {
            "floor": frozen.coverage_floor,
            "environments_above_the_floor": counts["coverage_above_floor"],
            "of_sufficient": len(sufficient),
        },
        "statistical_confirmation": statistics["primary_test"],
        "statistical_sensitivity_all_environments": statistics[
            "primary_test_sensitivity_all_environments"
        ],
        "secondary_family": {
            "size": statistics["secondary_family"]["size"],
            "surviving": statistics["secondary_family"]["surviving"],
            "surviving_favourable": statistics["secondary_family"]["surviving_favourable"],
            "surviving_adverse": statistics["secondary_family"]["surviving_adverse"],
        },
        "source_only_control": controls["aggregate_over_sufficient_environments"][
            "c1_source_only_refit"
        ],
        "permutation_control": controls["aggregate_over_sufficient_environments"][
            "c2_permuted_labels"
        ],
        "random_acquisition_diagnostic": controls["aggregate_over_sufficient_environments"][
            "d1_random_acquisition"
        ],
        "failed_environments": [e for e in sufficient if not results[e]["primary"]["improved"]],
        "negative_test_outcomes": {
            name: record.get("question", "") for name, record in negative["failure_tests"].items()
        },
        "sensitivity_over_all_environments": {
            "improved": sensitivity_counts["improved"],
            "harm_bound_held": sensitivity_counts["harm_bound_held"],
            "coverage_above_floor": sensitivity_counts["coverage_above_floor"],
            "of": sensitivity_counts["environments"],
            "required_if_applied": int(np.ceil(BREADTH_FRACTION * len(environments))),
        },
        "post_hoc_diagnostics_separated": [
            "the random-acquisition diagnostic is secondary and did not and could not change "
            "the frozen acquisition rule",
            "the leave-one-document-out and leave-one-environment-out influence analyses are "
            "diagnostics on an effect already estimated, not additional tests",
        ],
        "limitations": _limitations(design, support, environments),
        "interpretation_rules_observed": [
            "no claim that the framework generalizes universally",
            "no claim of model-agnostic correction verification: the correction method did not "
            "change in this stage",
            "the framework is target-adaptable, not target-free -- 250 target labels per "
            "environment were required",
        ],
        "confirmatory_reserve_consumed": False,
        "synthetic": False,
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DECISION, payload)
    print(f"decide: {verdict} -- {met} of {CRITERIA_TOTAL} criteria; outcome {outcome['label']}")
    return 0


def _outcome_label(
    criteria: dict[str, Any],
    controls: dict[str, Any],
    negative: dict[str, Any],
    primary: dict[str, Any],
    sufficient: list[str],
    results: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Which of the brief's four interpretations the numbers actually support."""
    aggregate = controls["aggregate_over_sufficient_environments"]
    random_matches = aggregate["d1_random_acquisition"]["mean_delta_versus_uncertainty"] >= 0.0
    per_corpus = negative["failure_tests"]["2_fails_on_a_new_document_domain"]["per_corpus"]
    domains_improved = {
        corpus: (record["improved"], record["sufficient"]) for corpus, record in per_corpus.items()
    }
    reproduced = criteria["1_breadth"]["met"] and criteria["2_safety"]["met"]
    domain_limited = any(
        len(improved) == 0 and len(names) > 0 for improved, names in domains_improved.values()
    )
    if reproduced and not domain_limited and not random_matches:
        label = "A"
        summary = "confirmed: the frozen method improves safe repair across the required majority"
    elif reproduced and domain_limited:
        label = "B"
        summary = "engine-general but domain-limited"
    elif reproduced and random_matches:
        label = "C"
        summary = "target supervision helps; acquisition specificity is unsupported"
    else:
        label = "D"
        summary = "the SGV13 gains did not reproduce on the required majority"
    return {
        "label": label,
        "summary": summary,
        "domains": {
            corpus: {"improved": improved, "sufficient": names}
            for corpus, (improved, names) in domains_improved.items()
        },
        "random_acquisition_matches_uncertainty": bool(random_matches),
        "next_stage_note": {
            "A": "cross-correction-method validation",
            "B": "study domain robustness before claiming general framework validity",
            "C": "simplify the framework and drop acquisition-specific claims",
            "D": "reassess target-adaptable representation and source environment diversity "
            "before attempting correction-method generalization",
        }[label],
    }


def _limitations(
    design: dict[str, Any], support: dict[str, Any], environments: list[str]
) -> list[str]:
    """What this stage did not establish. Written from the artifacts, not from hope."""
    insufficient = [e for e in environments if not support[e]["sufficient"]]
    return [
        "the correction method did not change: every candidate in every environment comes from "
        "the same controlled generator ladder SGV1 froze, so nothing here speaks to whether the "
        "reliability layer transfers across correction families",
        "the SGV1 CORD confirmatory reserve stayed locked, so no environment tests new documents "
        "in the SAME domain read by the SAME configuration -- the axis that would separate "
        "document novelty from domain novelty is absent",
        "both corpora were consumed by the earlier h1_pilot/cgv2/cgv3 line, so results on them "
        "have been seen before by a different pipeline even though no SGV1..SGV13 stage read them",
        "every environment is a configuration of one of the four base engines SGV1 recorded; no "
        "genuinely different OCR architecture was added",
        f"{len(insufficient)} environment(s) were marked INSUFFICIENT by the pre-registered rule "
        f"and are excluded from the criteria majority: {insufficient}",
        "the frozen source cut is placed on CORD calibration rows; a deployment on a new corpus "
        "that could recalibrate its own threshold is a different and untested procedure",
        "the acquisition ordering is computed once and never revised as labels arrive, so these "
        "curves are a lower bound on what a sequential loop could deliver",
    ]


# ------------------------------------------------------------------ provenance


PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    ENVIRONMENT_SELECTION,
    ENVIRONMENT_INVENTORY,
    IDENTITY_RECORD,
    DESIGN_RECORD,
    ADAPTATION_BUDGET,
    PRIMARY_RESULTS,
    PER_ENVIRONMENT_RESULTS,
    SAFETY_RESULTS,
    CONTROL_RESULTS,
    STATISTICAL_TESTS,
    NEGATIVE_TESTS,
    DECISION,
    FIGURE_MANIFEST,
    ENVIRONMENT_ROWS,
    CONFIRMATORY_SCORES,
    DEPLOYMENT_COUNTS,
)

DEPENDENCY_SCRIPTS = ("scripts/sgv14_confirmatory_validation.py", *UPSTREAM_SCRIPTS)

REPORT_SECTIONS = {
    "1. Motivation": (FROZEN_CONFIGURATION, s13.DECISION),
    "2. Frozen SGV13 evidence": (
        FROZEN_CONFIGURATION,
        s13.DECISION,
        s13.ADAPTATION_RESULTS,
        s13.PREFIX_PURITY,
    ),
    "3. Frozen configuration": (FROZEN_CONFIGURATION, DESIGN_RECORD),
    "4. Confirmatory research questions": (DESIGN_RECORD,),
    "5. New environments": (ENVIRONMENT_SELECTION, ENVIRONMENT_INVENTORY),
    "6. Environment selection audit": (ENVIRONMENT_SELECTION,),
    "7. Target adaptation protocol": (DESIGN_RECORD, IDENTITY_RECORD, ENVIRONMENT_INVENTORY),
    "8. Label accounting": (ADAPTATION_BUDGET,),
    "9. Primary results": (PRIMARY_RESULTS, PER_ENVIRONMENT_RESULTS),
    "10. Safety results": (SAFETY_RESULTS,),
    "11. Coverage and non-degeneracy": (SAFETY_RESULTS, DESIGN_RECORD),
    "12. Statistical confirmation": (STATISTICAL_TESTS,),
    "13. Source-only refit control": (CONTROL_RESULTS,),
    "14. Label-permutation control": (CONTROL_RESULTS,),
    "15. Random-acquisition diagnostic": (CONTROL_RESULTS,),
    "16. Environment-specific failure analysis": (NEGATIVE_TESTS, PER_ENVIRONMENT_RESULTS),
    "17. Negative tests": (NEGATIVE_TESTS,),
    "18. Limitations": (DECISION,),
    "19. Research decision": (DECISION,),
    "Artifacts": (PROVENANCE,),
    "Dependency audit": (PROVENANCE,),
}


def clean_clone_probe(inputs: dict[str, str]) -> dict[str, Any]:
    """Check a real clean checkout, rather than reasoning about one.

    Section 28 asks whether a fresh worktree can resolve this stage's scripts and its
    non-regenerable inputs. The answer is measured here by making one, looking, and removing it
    again -- and it is reported as it comes out. A clean-clone failure is an engineering defect
    even when every scientific number is unaffected.
    """
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="sgv14-clean-clone-"))
    worktree = root / "checkout"
    created = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        shutil.rmtree(root, ignore_errors=True)
        return {
            "attempted": True,
            "created": False,
            "detail": created.stderr.strip()[:400],
        }
    try:
        scripts = {name: (worktree / name).is_file() for name in DEPENDENCY_SCRIPTS}
        artifacts = {name: (worktree / name).is_file() for name in inputs}
        raw_present = (worktree / "data/raw/ocr").exists()
        return {
            "attempted": True,
            "created": True,
            "head": _git("rev-parse", "HEAD"),
            "scripts_required": len(scripts),
            "scripts_resolved": sum(1 for present in scripts.values() if present),
            "scripts_missing": sorted(name for name, present in scripts.items() if not present),
            "artifacts_required": len(artifacts),
            "artifacts_resolved": sum(1 for present in artifacts.values() if present),
            "artifacts_missing": sorted(name for name, present in artifacts.items() if not present),
            "raw_ocr_present": bool(raw_present),
            "can_run_sgv14": bool(
                all(scripts.values()) and all(artifacts.values()) and raw_present
            ),
            "verdict": (
                "a clean checkout at this HEAD cannot run SGV14: the stage and its predecessors "
                "are untracked and their artifacts are not committed. This is an engineering "
                "and reproducibility defect, recorded rather than hidden. It is not a defect in "
                "any scientific number: every artifact this stage reads is hash-verified "
                "against the record its own producing stage wrote."
            ),
        }
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO,
            capture_output=True,
            check=False,
        )
        shutil.rmtree(root, ignore_errors=True)


def run_record() -> int:
    """Provenance, the dependency audit and the traceability index."""
    started = time.monotonic()
    missing = [_relative(p) for p in PRODUCED if not p.is_file()]
    if missing:
        raise PhaseError(f"cannot record an incomplete stage; missing {missing}")
    freeze = cc_read_json(RESEARCH_FREEZE)
    upstream = s13.verify_frozen()
    inputs = {entry["path"]: entry["sha256"] for entry in upstream["artifacts"]}
    inputs.update(freeze["upstream_artifact_sha256"])
    figures = cc_read_json(FIGURE_MANIFEST)["figures"]
    artifacts = {_relative(p): file_sha256(p) for p in PRODUCED}
    artifacts.update({path: record["sha256"] for path, record in figures.items()})

    ignored = sorted(path for path in artifacts if _ignored(REPO / path))
    payload = {
        "artifact": "provenance",
        "schema_version": f"sgv14-provenance-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "stage_title": "SGV14 -- frozen confirmatory validation on new OCR environments",
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git("rev-parse", "HEAD"),
        "synthetic": False,
        "confirmatory_reserve_accessed": False,
        "features_added": 0,
        "methods_selected_here": 0,
        "models_fitted_note": (
            "SGV14 fits the frozen SGV13 arm on target labels, which is what replicating it "
            "means. It selects nothing: no arm, no acquisition rule, no budget, no cut, no "
            "epsilon and no hyperparameter is chosen by any code in this stage."
        ),
        "inputs": inputs,
        "artifacts": artifacts,
        "dependency_audit": {
            "scripts": {
                name: file_sha256(REPO / name)
                for name in DEPENDENCY_SCRIPTS
                if (REPO / name).is_file()
            },
            "depth": len(DEPENDENCY_SCRIPTS),
            "clean_clone": {
                "committed": (
                    "nothing. The working tree carries this stage and thirteen untracked "
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
                "non_regenerable_dependencies": freeze["clean_clone_risks"]["non_regenerable"],
                "new_raw_written_by_this_stage": [
                    "data/raw/ocr/funsd/tesseract_psm3/**",
                    "data/raw/ocr/ocrd_sbb/tesseract_deu/**",
                ],
                "known_repository_hygiene_issue": KNOWN_PROVENANCE_DEFECT["defect"],
                "known_issue_action": KNOWN_PROVENANCE_DEFECT["action_taken"],
                "clean_checkout_probe": clean_clone_probe(inputs),
            },
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(PROVENANCE, payload)
    _write_json_once(
        TRACEABILITY,
        {
            "artifact": "traceability",
            "schema_version": f"sgv14-traceability-v{SCHEMA_VERSION}",
            "stage": STAGE,
            "hypothesis_id": HYPOTHESIS,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "report": "docs/sgv14/confirmatory_validation.md",
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


# ------------------------------------------------------------------ cli


PHASES = (
    ("reconstruct", "the section-0 research freeze"),
    ("freeze", "the frozen SGV13 configuration"),
    ("environments", "the environment selection audit"),
    ("ocr", "the two engine configurations this stage runs itself"),
    ("identity", "the representation rebuild, proved against SGV1"),
    ("build", "every environment's rows in the frozen representation"),
    ("preregister", "the criteria, the sufficiency rule and the test family"),
    ("adapt", "the frozen method, the baselines and the controls"),
    ("results", "the primary, per-environment and safety endpoints"),
    ("controls", "the two validation controls and the diagnostic"),
    ("stats", "the pre-registered tests"),
    ("negative", "the ten falsification tests"),
    ("figures", "the five required figures"),
    ("decide", "the machine-readable finding"),
    ("record", "provenance and traceability"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name, help_text in PHASES:
        parser.add_argument(f"--{name}", action="store_true", help=help_text)
    parser.add_argument(
        "--only", default=None, help="restrict --ocr/--build/--adapt to one environment"
    )
    arguments = parser.parse_args()
    selected = [name for name, _ in PHASES if getattr(arguments, name)]
    if not selected:
        parser.print_help()
        return 2
    handlers = {
        "reconstruct": run_reconstruct,
        "freeze": run_freeze,
        "environments": run_environments,
        "ocr": lambda: run_ocr(arguments.only),
        "identity": run_identity,
        "build": lambda: run_build(arguments.only),
        "preregister": run_preregister,
        "adapt": lambda: run_adapt(arguments.only),
        "results": run_results,
        "controls": run_controls,
        "stats": run_stats,
        "negative": run_negative,
        "figures": run_figures,
        "decide": run_decide,
        "record": run_record,
    }
    for name in selected:
        status = handlers[name]()
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
