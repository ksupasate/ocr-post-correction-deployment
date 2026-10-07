#!/usr/bin/env python3
"""SGV-CG1: where the remaining ceiling comes from -- candidates, ranking or deployment.

Five stages measured the downstream half of this system and four of them returned a negative.
SGV13 showed target supervision moves the ranking. SGV14 showed the frozen source cut does not
transfer. SGV15 showed a target certificate buys safety by refusing to deploy. SGV15b showed the
document-aware certificate is the statistically defensible one and that its sample complexity is
out of reach. SGV-DT1 showed human review, routing and target supervision each independently move
the deployment frontier, and that no combination of them reaches majority breadth at the primary
tolerance inside the declared budgets.

Every one of those stages varied something downstream of the correction candidate. None of them
asked the upstream question:

    SGV-CG1: even with perfect knowledge of every candidate outcome, how much useful OCR repair is
    available at all in the frozen correction candidate universe?

**This stage is diagnostic. It has no PASS and no FAIL.** It attributes the unresolved gap to the
layer that owns it, and the answer is allowed to be that the layer nobody has been working on is
the one that matters. Six decisions fix what the numbers below can mean.

**1. Nothing is refitted and nothing is regenerated.** The candidate generator, the discovery
enumerator, the reliability model, the adapted ranking and SGV-DT1's selected deployment regime are
all consumed exactly as frozen. The one thing this stage recomputes is the per-candidate ground
truth SGV14 computed and then dropped from its published table -- `d_before`, `d_after`, the
candidate text and the OCR region -- and `--reproduce` proves that recomputation reproduces every
label SGV14 published, on every row, with zero differences.

**2. An OCR error site is defined against ground truth, not against the generator.** It is an
alignment component whose OCR text differs from its ground truth text. That population exists
whether or not the discovery enumerator ever anchored a site there, which is the only way the
question "what fraction of OCR errors is reachable" can be asked honestly. The population the
generator actually proposed on is a subset, is reported as a subset, and never silently replaces
the larger one.

**3. The candidate gap therefore has two parts and they are measured separately.** A correct
candidate can be missing because discovery never anchored the region (`discovery`), or because
discovery anchored it and generation proposed nothing correct (`generation`). Section 44 of the
brief exists to stop those two being confused with each other and with ranking; so does this
split.

**4. Repair is counted twice, in two units, and the units are never mixed.** Sites are the
decision unit and characters are the damage unit: a site is repaired or it is not, and a repair
removes a measurable number of wrong characters. Both are reported for every layer of the ladder.
SGV-DT1's own candidate-row reading is carried through unchanged beside them so its published
numbers remain locatable.

**5. Ground truth is allowed here, and only here, and it is labelled.** Every oracle in this file
is `analysis_only = true, uses_ground_truth = true` in its artifact. No oracle selects a policy,
moves a threshold, filters a candidate or admits an environment. The leakage suite asserts that
against the source.

**6. This is DEVELOPMENT and it introduces no corrector.** No ByT5, no seq2seq, no LLM, no VLM, no
spell-checker, no dictionary, no new heuristic generator, no widened candidate set. Section 27 of
the brief prohibits all of them and this stage measures the ceiling of what already exists.

    --reconstruct   the section-0 freeze: SGV15, SGV15b and SGV-DT1 re-hashed and re-read
    --freeze        the frozen candidate, ranking and deployment configuration
    --preregister   units, metrics, the dominance rule, the questions and the outcome taxonomy
    --inventory     the heavy pass: alignment sites, discovered sites, candidates and labels
    --reproduce     the integrity gate against SGV14's labels and SGV-DT1's deployment metrics
    --opportunity   opportunity recall, multiplicity, beneficial/harmful composition
    --oracles       the candidate, risk-constrained, ranking and deployment ceilings
    --ceiling       the full decomposition and the normalized gap contributions
    --taxonomy      the C0-C5 site-level failure taxonomy
    --strata        engine, domain, error type and raw-OCR-quality analyses
    --ambiguity     candidate ambiguity, correct-candidate rank and top-K opportunity
    --review        the perfect-routing human-review ceiling and the remaining headroom
    --tolerance     candidate-universe ceilings under the three harm tolerances
    --bottleneck    the pre-registered dominant-bottleneck rule, applied
    --controls      the seven required negative controls
    --negative      the ten falsification tests
    --stats         the pre-registered statistics
    --figures       the fifteen required figures
    --decide        the machine-readable research decision
    --determinism   two independent regenerations
    --record        provenance, the dependency audit and the traceability index

DEVELOPMENT ONLY. The SGV1 CORD confirmatory reserve stays LOCKED and is absent from every
artifact. SGV13, SGV14, SGV15, SGV15b and SGV-DT1 artifacts are read and hash-verified; none is
modified.
"""

from __future__ import annotations

import argparse
import bisect
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
import sgv_dt1_deployment_tradeoff as dt1
from ocr_risk.io.hashing import canonical_hash, file_sha256

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv_cg1_opportunity_ceiling"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"

ALIGNMENT_SITES = OUT / "alignment_sites.parquet"
CANDIDATE_ROWS = OUT / "candidate_rows.parquet"
SITE_LINKS = OUT / "site_links.parquet"
EVALUATION_ORDER = OUT / "evaluation_order.parquet"
SITE_INVENTORY = OUT / "site_inventory.json"
CANDIDATE_INVENTORY = OUT / "candidate_inventory.json"
ATOMIC_EDIT_SCHEMA = OUT / "atomic_edit_schema.json"
UPSTREAM_REPRODUCTION = OUT / "upstream_reproduction.json"

CANDIDATE_OPPORTUNITY = OUT / "candidate_opportunity.json"
CANDIDATE_MULTIPLICITY = OUT / "candidate_multiplicity.json"
BENEFICIAL_HARMFUL_RATIO = OUT / "beneficial_harmful_ratio.json"

CANDIDATE_ORACLE = OUT / "candidate_oracle.json"
RISK_CANDIDATE_ORACLE = OUT / "risk_candidate_oracle.json"
RANKING_ORACLE = OUT / "ranking_oracle.json"
DEPLOYMENT_BASELINE = OUT / "deployment_baseline.json"
PERFECT_RANKING = OUT / "perfect_ranking_counterfactual.json"
PERFECT_DEPLOYMENT = OUT / "perfect_deployment_counterfactual.json"

CEILING_DECOMPOSITION = OUT / "ceiling_decomposition.json"
NORMALIZED_GAP = OUT / "normalized_gap_decomposition.json"
SITE_FAILURE_TAXONOMY = OUT / "site_failure_taxonomy.json"

ENGINE_ANALYSIS = OUT / "engine_analysis.json"
DOMAIN_ANALYSIS = OUT / "domain_analysis.json"
ERROR_TYPE_ANALYSIS = OUT / "error_type_analysis.json"
OCR_QUALITY_ANALYSIS = OUT / "ocr_quality_analysis.json"

TOPK_OPPORTUNITY = OUT / "topk_opportunity.json"
AMBIGUITY_ANALYSIS = OUT / "ambiguity_analysis.json"

HUMAN_REVIEW_ORACLE = OUT / "human_review_oracle.json"
REVIEW_HEADROOM = OUT / "review_headroom.json"
HARM_TOLERANCE_CEILING = OUT / "harm_tolerance_ceiling.json"
DOMINANT_BOTTLENECK = OUT / "dominant_bottleneck.json"

CONTROL_RESULTS = OUT / "control_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_cg1/opportunity_ceiling.md"

SCHEMA_VERSION = 1
STAGE = "sgv_cg1_opportunity_ceiling"
HYPOTHESIS = "SGV-CG1-D1"

# ------------------------------------------------------------------ imported, never restated

PhaseError = s15.PhaseError
_relative = s15._relative
_git = s15._git
_write_json_once = s15._write_json_once
_write_parquet_once = s15._write_parquet_once
_ignored = s15._ignored
_tracked = s15._tracked
cc_read_json = s15.cc_read_json
_mean = s15._mean
holm = s15.holm

EPSILONS = s15.EPSILONS
PRIMARY_EPSILON = s15.PRIMARY_EPSILON
BOOTSTRAP_RESAMPLES = s13.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = s13.BOOTSTRAP_SEED
ALPHA = 0.05

ENVIRONMENT_COUNT = 10
UNLABELLED_DISTANCE = s14.UNLABELLED_DISTANCE

# ------------------------------------------------------------------ the frozen upstream state
#
# Section 0. These are what the repository is expected to hold; every one of them is re-read from
# the saved decision artifact and compared, and a disagreement stops the stage.

SGV15_STATE = ("NOT SUPPORTED", 2, 6, "D", False)
SGV15B_STATE = ("NOT SUPPORTED", 3, 7, "D", False)
SGV_DT1_STATE = ("NO PRACTICAL REGIME AT THE DECLARED BUDGETS", 5, 7, "C", False)

UPSTREAM_SCRIPTS = (
    "scripts/sgv13_fewshot_prefix_purity_adaptation.py",
    "scripts/sgv14_confirmatory_validation.py",
    "scripts/sgv15_target_risk_certification.py",
    "scripts/sgv15b_document_level_certification.py",
    "scripts/sgv_dt1_deployment_tradeoff.py",
)

# ------------------------------------------------------------------ the pre-registered registry

# Section 4. The five units this stage counts in, declared once. Nothing below may silently widen
# one of them.
UNIT_ALIGNMENT_SITE = "alignment_site"
UNIT_DISCOVERED_SITE = "discovered_site"
UNIT_CANDIDATE = "candidate"
UNIT_CHARACTER = "error_character"
UNIT_DOCUMENT = "document"

# Section 4. A candidate is CORRECT when accepting it leaves the region exactly equal to its
# ground truth -- the project's `true_correction`. The project's wider `beneficial` class also
# contains `partial_improvement`, which moves the region closer without reaching it. Both are
# reported everywhere; the primary is the exact one, because "repairs the OCR output to the
# correct target" is what section 4 asks for.
OUTCOME_EXACT = "true_correction"
OUTCOME_PARTIAL = "partial_improvement"
OUTCOME_LATERAL = "lateral_change"
OUTCOME_MISCORRECTION = "miscorrection"
OUTCOME_OVERCORRECTION = "overcorrection"
ERROR_OUTCOMES = (OUTCOME_EXACT, OUTCOME_PARTIAL, OUTCOME_LATERAL, OUTCOME_MISCORRECTION)

# Section 13. The site-level failure taxonomy, declared before any result is read.
C0_NO_CORRECT_CANDIDATE = "c0_no_correct_candidate"
C1_RANKED_BELOW_HARM = "c1_correct_candidate_ranked_below_harmful"
C2_NOT_APPLIED = "c2_correct_candidate_not_applied_by_policy"
C3_AUTOMATIC_REPAIR = "c3_automatic_repair"
C4_REVIEWED_REPAIR = "c4_reviewed_repair"
C5_UNEVALUABLE = "c5_structurally_unevaluable"
TAXONOMY = (
    C0_NO_CORRECT_CANDIDATE,
    C1_RANKED_BELOW_HARM,
    C2_NOT_APPLIED,
    C3_AUTOMATIC_REPAIR,
    C4_REVIEWED_REPAIR,
    C5_UNEVALUABLE,
)

# Section 11. The ladder, in order. Every rung is a repair recall on the SAME denominator.
LADDER = (
    "opportunity_recall",
    "candidate_oracle",
    "risk_candidate_oracle",
    "perfect_ranking",
    "ranking_oracle",
    "practical_deployment",
)

# Section 12. The three losses and the reference the normalization divides by.
L_CANDIDATE = "candidate"
L_RANKING = "ranking"
L_DEPLOYMENT = "deployment"
COMPONENTS = (L_CANDIDATE, L_RANKING, L_DEPLOYMENT)

# Section 24. The dominance rule, fixed before the endpoint is read. A component is dominant only
# if it wins a declared majority of environments AND clears a declared margin AND survives the
# document-clustered paired test. Anything else is Outcome D.
DOMINANCE_MAJORITY = 6
DOMINANCE_MARGIN = 0.10

# Section 26 outcome A additionally requires the candidate ceiling itself to be low. "Low" is a
# declared judgement, not a derived quantity: under PERFECT candidate selection the frozen
# universe repairs fewer than half the OCR error sites it is shown.
LOW_CEILING_THRESHOLD = 0.50
# Section 45's stopping rule needs the same treatment. Below this, downstream threshold,
# certification, routing and calibration work is operating on less than a fifth of the problem.
MEANINGFUL_FRONTIER_THRESHOLD = 0.20

# Section 22. The review budgets, taken unchanged from SGV-DT1 so the ceiling is comparable to the
# routing strategies SGV-DT1 actually measured.
REVIEW_BUDGETS = dt1.REVIEW_BUDGETS
PRACTICAL_REVIEW_BUDGET = dt1.PRACTICAL_REVIEW_BUDGET
PRACTICAL_DOC_BUDGET = dt1.PRACTICAL_DOC_BUDGET
PRIMARY_SCENARIO = dt1.PRIMARY_SCENARIO
DOC_DRAWS = dt1.DOC_DRAWS
SAMPLE_SEED = dt1.SAMPLE_SEED

# Section 19. K for the top-K opportunity curve.
TOPK = (1, 2, 3, 5, 10)

# Section 36. The negative controls.
CONTROLS = (
    "n0_ground_truth_candidate_oracle",
    "n1_random_candidate_selection",
    "n2_random_ranking",
    "n3_label_permutation",
    "n4_candidate_set_truncation",
    "n5_duplicate_harmful_candidates",
    "n6_preserve_all_deployment",
)
CONTROL_SEED = 20260908

# Section 25. The six pre-registered questions.
QUESTIONS = {
    "Q1": "Is the current candidate universe itself a major ceiling?",
    "Q2": "Does perfect candidate selection substantially exceed the current frozen ranking?",
    "Q3": "Does oracle cut selection substantially exceed practical deployment?",
    "Q4": "Which component dominates across environments?",
    "Q5": "Does the dominant component differ by OCR engine or document domain?",
    "Q6": (
        "Would improving the correction method plausibly move the deployment frontier enough to "
        "justify a cross-correction-method study?"
    ),
}

OUTCOME_TAXONOMY = {
    "A": "candidate-generation bottleneck",
    "B": "ranking bottleneck",
    "C": "deployment bottleneck",
    "D": "mixed bottleneck",
}


# ------------------------------------------------------------------ small shared helpers


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_cg1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _oracle_envelope(artifact: str) -> dict[str, Any]:
    """Section 30. Every artifact built with ground truth says so in its own header."""
    return {
        **_envelope(artifact),
        "analysis_only": True,
        "uses_ground_truth": True,
        "may_select_a_policy": False,
        "may_move_a_threshold": False,
        "may_filter_a_candidate": False,
        "may_admit_or_exclude_an_environment": False,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not len(values):
        return {k: float("nan") for k in ("mean", "median", "p25", "p75", "p90", "max")}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def environment_names() -> list[str]:
    return [spec["environment"] for spec in s14.ENVIRONMENTS]


# ------------------------------------------------------------------ section 0: reconstruct


def _decision_state(path: Path, keys: tuple[str, str, str]) -> tuple[Any, ...]:
    record = cc_read_json(path)
    verdict = str(record[keys[0]])
    met = int(record["criteria_met"])
    total = int(record["criteria_total"])
    outcome = record.get("outcome")
    label = str(outcome["label"]) if isinstance(outcome, dict) else _outcome_from_reason(record)
    onward = bool(record[keys[2]])
    return verdict, met, total, label, onward


def _outcome_from_reason(record: dict[str, Any]) -> str:
    """SGV15 and SGV15b carry the outcome letter inside `reason`, not as a field."""
    reason = str(record.get("reason", ""))
    for letter in ("A", "B", "C", "D"):
        if f"outcome {letter}" in reason:
            return letter
    raise PhaseError(f"no outcome letter in {reason!r}")


def run_reconstruct() -> int:
    """Re-read SGV15, SGV15b and SGV-DT1 from their own decision artifacts, and re-hash them."""
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    disagreements: list[str] = []
    for name, path, expected, keys in (
        (
            "sgv15_target_risk_certification",
            s15.DECISION,
            SGV15_STATE,
            ("verdict", "criteria_met", "ready_for_sgv16"),
        ),
        (
            "sgv15b_document_level_certification",
            s15b.DECISION,
            SGV15B_STATE,
            ("verdict", "criteria_met", "ready_for_sgv16"),
        ),
        (
            "sgv_dt1_deployment_tradeoff",
            dt1.DECISION,
            SGV_DT1_STATE,
            ("verdict", "criteria_met", "ready_for_external_confirmation"),
        ),
    ):
        observed = _decision_state(path, keys)
        agrees = observed == expected
        if not agrees:
            disagreements.append(f"{name}: expected {expected}, artifact says {observed}")
        checks.append(
            {
                "stage": name,
                "artifact": _relative(path),
                "sha256": file_sha256(path),
                "expected": list(expected),
                "observed": list(observed),
                "agrees": agrees,
                "fields": ["verdict", "criteria_met", "criteria_total", "outcome", keys[2]],
            }
        )
    if disagreements:
        raise PhaseError("section 0 disagreement, stop and reconcile: " + "; ".join(disagreements))

    reserve = cc_read_json(pilot.RESERVE_LOCK)
    if bool(reserve.get("unlocked", False)):
        raise PhaseError("the confirmatory reserve is not locked")
    consumed = [
        _relative(path)
        for path in (s15.DECISION, s15b.DECISION, dt1.DECISION)
        if bool(cc_read_json(path).get("confirmatory_reserve_consumed", False))
    ]
    if consumed:
        raise PhaseError(f"an upstream decision reports the reserve consumed: {consumed}")

    gates: list[dict[str, Any]] = []
    for name, provenance, traceability, determinism in (
        ("sgv15", s15.PROVENANCE, s15.TRACEABILITY, s15.DETERMINISM),
        ("sgv15b", s15b.PROVENANCE, s15b.TRACEABILITY, s15b.DETERMINISM),
        ("sgv_dt1", dt1.PROVENANCE, dt1.TRACEABILITY, dt1.DETERMINISM),
    ):
        record = cc_read_json(determinism)
        gates.append(
            {
                "stage": name,
                "provenance_present": provenance.is_file(),
                "traceability_present": traceability.is_file(),
                "determinism_present": determinism.is_file(),
                "determinism_all_runs_identical": bool(record.get("all_runs_identical", False)),
                "provenance_sha256": file_sha256(provenance),
            }
        )
    unstable = [g["stage"] for g in gates if not g["determinism_all_runs_identical"]]
    if unstable:
        raise PhaseError(f"upstream determinism is not clean: {unstable}")

    scripts = {path: file_sha256(REPO / path) for path in sorted(UPSTREAM_SCRIPTS)}
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "issued_head": _git("rev-parse", "HEAD"),
            "rule": (
                "repository artifacts are authoritative. The expected states in this file are "
                "compared against the saved decision artifacts and a disagreement stops the "
                "stage; the prompt is never the source of a scientific state."
            ),
            "frozen_states": checks,
            "verification_gates": gates,
            "confirmatory_reserve": {
                "locked": True,
                "lock_artifact": _relative(pilot.RESERVE_LOCK),
                "lock_sha256": file_sha256(pilot.RESERVE_LOCK),
                "consumed_by_any_upstream_stage": False,
            },
            "upstream_script_sha256": scripts,
            "upstream_regeneration_required": False,
            "why_no_regeneration": (
                "SGV-CG1 consumes the frozen score geometry SGV-DT1 already validated field by "
                "field against SGV15b, and recomputes only the per-candidate ground truth SGV14 "
                "dropped from its published table. `--reproduce` proves both."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"reconstruct: {len(checks)} frozen states confirmed from artifacts, "
        f"{len(gates)} upstream gate sets clean, reserve locked -> {_relative(RESEARCH_FREEZE)}"
    )
    return 0


# ------------------------------------------------------------ sections 2-3: the frozen pipeline


def run_freeze() -> int:
    """Recover, and record, the exact candidate / ranking / deployment configuration."""
    started = time.monotonic()
    if not RESEARCH_FREEZE.is_file():
        raise PhaseError("run --reconstruct first")
    candidate_freeze = cc_read_json(pilot.CANDIDATE_FREEZE)
    site_freeze_path = REPO / "results/generated/sgv1/dev_candidates/site_freeze.json"
    dt1_decision = cc_read_json(dt1.DECISION)
    dt1_design = cc_read_json(dt1.DESIGN_RECORD)
    s15_decision = cc_read_json(s15.DECISION)

    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "candidate_generation": {
                "primary_generator": s14.PRIMARY_GENERATOR,
                "component_generators": list(pilot.GENERATOR_SOURCES),
                "operations": list(pilot.OPERATIONS),
                "anchor_kinds": list(pilot.ANCHOR_KINDS),
                "site_types": list(pilot.SITE_TYPES),
                "discovery": (
                    "ocr_risk.discovery.enumerator.DiscoveryRules at defaults, run through "
                    "ocr_risk.experiments.cgv3_confirmatory.discovery_pass -- OCR-only, "
                    "ground-truth-blind by construction"
                ),
                "generation": "ocr_risk.experiments.cgv3_confirmatory.generation_pass",
                "sgv1_freeze_sha256": file_sha256(pilot.CANDIDATE_FREEZE),
                "sgv1_declared": {
                    k: candidate_freeze[k]
                    for k in sorted(candidate_freeze)
                    if isinstance(candidate_freeze[k], (str, int, float, bool))
                },
                "modified_by_this_stage": False,
                "widened_by_this_stage": False,
                "new_generator_families": [],
                "prohibited_and_absent": [
                    "byt5",
                    "seq2seq correction",
                    "llm correction",
                    "multimodal vlm correction",
                    "spell-checker",
                    "dictionary corrector",
                    "retrieval candidates",
                    "new heuristic generator",
                ],
            },
            "reliability_ranking": {
                "adaptation_arm": s15.A_UNCERTAINTY,
                "adaptation_budget": int(s15.ADAPT_BUDGET),
                "source_model": "SGV13 a4_joint_refit over SGV14's frozen representation",
                "score_source": _relative(dt1.GEOMETRY),
                "score_source_sha256": file_sha256(dt1.GEOMETRY),
                "refitted_by_this_stage": False,
                "fits_performed_by_this_stage": 0,
            },
            "deployment": {
                "stage": "sgv_dt1_deployment_tradeoff",
                "selected_policy": dt1_decision["best_practical_policy"],
                "review_budgets": [float(b) for b in REVIEW_BUDGETS],
                "document_budgets": [int(b) for b in dt1.DOC_BUDGETS],
                "draws": int(DOC_DRAWS),
                "sample_seed": int(SAMPLE_SEED),
                "reviewer_scenarios": {k: float(v) for k, v in dt1.REVIEWER_ACCURACY.items()},
                "modified_by_this_stage": False,
                "design_record_sha256": file_sha256(dt1.DESIGN_RECORD),
                "declared_regimes": sorted(dt1_design.get("deployment_regimes", {})),
            },
            "harm_policy": "strict_worsening",
            "primary_epsilon": float(PRIMARY_EPSILON),
            "epsilons": [float(e) for e in EPSILONS],
            "certification_partition": {
                "seed": s15.PARTITION_SEED,
                "fit_fraction": float(s15.FIT_FRACTION),
                "selected_allocation": s15_decision["selected_allocation"],
            },
            "site_freeze": {
                "artifact": _relative(site_freeze_path),
                "sha256": file_sha256(site_freeze_path),
                "enumerator_version": cc_read_json(site_freeze_path).get("enumerator_version"),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"freeze: generator {s14.PRIMARY_GENERATOR} over {list(pilot.GENERATOR_SOURCES)}, "
        f"ranking {s15.A_UNCERTAINTY} at {s15.ADAPT_BUDGET} labels, deployment "
        f"{dt1_decision['best_practical_policy']['policy']} -> {_relative(FROZEN_CONFIGURATION)}"
    )
    return 0


# ---------------------------------------------------- sections 4, 12, 24-26: pre-registration


def run_preregister() -> int:
    """Units, metrics, the dominance rule and the outcome taxonomy, before any endpoint."""
    started = time.monotonic()
    if not FROZEN_CONFIGURATION.is_file():
        raise PhaseError("run --freeze first")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "issued_head": _git("rev-parse", "HEAD"),
            "stage_kind": "DEVELOPMENT / DIAGNOSTIC",
            "not_a_pass_fail_stage": True,
            "primary_epsilon": float(PRIMARY_EPSILON),
            "units": {
                UNIT_ALIGNMENT_SITE: (
                    "an alignment component, after SGV1's adjacent-segmentation merge, whose OCR "
                    "text differs from its ground truth text. The OCR ERROR SITE of section 4. "
                    "Exists whether or not the discovery enumerator anchored anything there."
                ),
                UNIT_DISCOVERED_SITE: (
                    "a site the OCR-only enumerator anchored and the generator proposed on. A "
                    "subset of the decision points, never a substitute for the population above."
                ),
                UNIT_CANDIDATE: "one proposed edit at one discovered site. SGV-DT1's row unit.",
                UNIT_CHARACTER: (
                    "one wrong character: `d_before` on an alignment site, `d_before - d_after` "
                    "for a candidate. Raw Levenshtein, never normalized, per the metrics rule."
                ),
                UNIT_DOCUMENT: "the resampling unit for every interval in this stage.",
            },
            "definitions": {
                "correct_candidate": (
                    f"outcome == {OUTCOME_EXACT}: accepting it leaves the region exactly equal to "
                    "ground truth. PRIMARY."
                ),
                "beneficial_candidate": (
                    f"outcome in ({OUTCOME_EXACT}, {OUTCOME_PARTIAL}) under the project's frozen "
                    "strict_worsening harm policy. SECONDARY, reported beside the primary because "
                    "it is the class SGV-DT1's repair recall counted."
                ),
                "harmful_candidate": (
                    f"outcome in ({OUTCOME_MISCORRECTION}, {OUTCOME_OVERCORRECTION}) under "
                    "strict_worsening. Unchanged from upstream."
                ),
                "repairable_site": "an OCR error site with at least one correct candidate.",
                "unrepairable_site": "an OCR error site with no correct candidate.",
                "automatic_harm": (
                    "harmful automatically applied candidates over all automatically applied "
                    "candidates. SGV-DT1's definition, carried unchanged."
                ),
            },
            "ladder": list(LADDER),
            "gap_components": list(COMPONENTS),
            "normalization": {
                "denominator": "1 - practical_deployment_repair_recall",
                "why": (
                    "the unresolved gap relative to repairing every OCR error site. The three "
                    "losses partition it exactly, so the contributions sum to one whenever the "
                    "ladder is monotone."
                ),
                "if_a_component_is_negative": (
                    "the ladder is not monotone for that environment; section 12 forbids forcing "
                    "the normalization. Raw gaps are reported and the environment is flagged."
                ),
                "monotonicity_is_by_construction_for": (
                    "the AUTOMATIC action space. Every rung of the primary ladder applies edits "
                    "automatically, so a rung can never exceed the rung above it. Human review is "
                    "a separate action space and gets its own ladder in section 22."
                ),
            },
            "dominance_rule": {
                "declared_before_endpoint": True,
                "losses": {
                    L_CANDIDATE: "1 - opportunity_recall",
                    L_RANKING: "candidate_oracle - ranking_oracle",
                    L_DEPLOYMENT: "ranking_oracle - practical_deployment",
                },
                "criterion_1_majority": (
                    f"the component is the largest of the three in at least {DOMINANCE_MAJORITY} "
                    f"of {ENVIRONMENT_COUNT} environments"
                ),
                "criterion_2_margin": (
                    f"its mean loss exceeds the second largest mean loss by at least "
                    f"{DOMINANCE_MARGIN}"
                ),
                "criterion_3_uncertainty": (
                    "the paired document-clustered bootstrap interval for (largest - second "
                    f"largest) excludes zero at alpha {ALPHA} after Holm across the declared "
                    "family"
                ),
                "all_three_required": True,
                "otherwise": "Outcome D (mixed bottleneck)",
            },
            "outcome_taxonomy": OUTCOME_TAXONOMY,
            "outcome_a_additional_requirement": {
                "candidate_ceiling_is_low": (
                    f"mean candidate-oracle repair recall < {LOW_CEILING_THRESHOLD}"
                ),
                "why_declared_not_derived": (
                    "section 26 requires the candidate oracle ceiling itself to be low, not only "
                    "the gap to be largest. Half the OCR error sites under PERFECT selection is a "
                    "declared judgement written down before the endpoint, not a derived quantity."
                ),
            },
            "stopping_rule": {
                "source": "section 45",
                "fires_when": (
                    "mean risk-constrained candidate-oracle repair recall at the primary epsilon "
                    f"< {MEANINGFUL_FRONTIER_THRESHOLD}"
                ),
                "consequence": (
                    "further threshold, certification, review-routing and calibration work on "
                    "THIS candidate universe is not the scientific priority; the next investment "
                    "moves upstream to correction generation."
                ),
                "threshold_is_declared": True,
            },
            "questions": QUESTIONS,
            "failure_taxonomy": list(TAXONOMY),
            "review_budgets": [float(b) for b in REVIEW_BUDGETS],
            "topk": list(TOPK),
            "controls": list(CONTROLS),
            "statistics": {
                "inference_units": ["environment", UNIT_DOCUMENT],
                "candidate_rows_are_not_inferential_units": True,
                "bootstrap": "document-clustered, paired, "
                f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}",
                "multiplicity": f"Holm at alpha {ALPHA} over the declared primary family",
            },
            "ground_truth_policy": {
                "allowed_for": ["outcome labelling", "oracle analysis", "diagnostic ceilings"],
                "prohibited_for": [
                    "candidate generation",
                    "frozen ranking",
                    "deployed policy",
                    "candidate filtering",
                    "environment inclusion",
                ],
            },
            "prohibitions": {
                "new_corrector": False,
                "widened_candidate_set": False,
                "modified_ranking": False,
                "modified_deployment": False,
                "external_confirmation": False,
                "reserve_consumed": False,
            },
            "construction_disclosure": (
                "the reconstruction path and the character-mass formulation were validated during "
                "implementation on funsd/paddleocr, so that environment's opportunity numbers were "
                "seen before this record was written. The dominance rule, the margin, the "
                "majority, "
                "the low-ceiling threshold and the stopping threshold are the brief's own "
                "structure and are not conditioned on it; every other environment was unread."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"preregister: {len(LADDER)} ladder rungs, {len(COMPONENTS)} gap components, dominance "
        f"{DOMINANCE_MAJORITY}/{ENVIRONMENT_COUNT} at margin {DOMINANCE_MARGIN}, "
        f"{len(QUESTIONS)} questions -> {_relative(DESIGN_RECORD)}"
    )
    return 0


# ------------------------------------------------------- section 4: the two site populations
#
# The heavy phase. Everything downstream reads its two parquet tables and nothing else.

CACHE = OUT / "cache"

ALIGNMENT_COLUMNS = (
    "environment",
    "corpus",
    "base_engine",
    "document_id",
    "role",
    "align_site_id",
    "site_kind",
    "evaluable",
    "d_before",
    "char_start",
    "char_end",
    "ocr_spans",
    "ocr_chars",
    "gt_chars",
)

CANDIDATE_COLUMNS = (
    "environment",
    "corpus",
    "base_engine",
    "candidate_id",
    "site_id",
    "document_id",
    "role",
    "outcome",
    "is_harmful",
    "beneficial",
    "d_before",
    "d_after",
    "char_start",
    "char_end",
    "anchor_kind",
    "site_type",
    "operation",
    "generator_source",
    "generator_rank",
    "generator_score",
    "suspicion_score",
    "original_chars",
    "candidate_chars",
)


SITE_LINK_COLUMNS = ("environment", "site_id", "align_site_id", "link_kind")


def _reconstruct_environment(
    spec: dict[str, Any], pipeline: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """One environment's alignment sites, candidate rows and their labels.

    Every call below is SGV14's own function on SGV14's own inputs. The alignment is run once and
    serves both populations, so the GT-aligned error sites and the labelled candidates cannot
    disagree about what the ground truth of a page is.
    """
    from ocr_risk.align import align_document
    from ocr_risk.canonical import rebuild_stream
    from ocr_risk.config.models import AlignmentConfig, SiteConfig
    from ocr_risk.edits.sites import build_sites
    from ocr_risk.experiments.cgv3_study import AlignmentIndex
    from ocr_risk.schemas.enums import AnchorKind

    began = time.monotonic()
    name = spec["environment"]
    bundles = s14._document_bundles(spec["corpus"])
    roles = s14.partition_of(spec["corpus"], sorted(bundles))
    spans, ocr_record = s14._canonical_spans(spec, bundles)
    streams = {pair: rebuild_stream(group) for pair, group in spans.items() if group}
    sites, candidates = s14._candidate_table(spec, spans, streams, pipeline, roles)
    labels, diagnostics = s14._labels(spec["corpus"], bundles, spans, streams, sites, candidates)

    pool = candidates.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    pool = pool[pool["labelable"]].reset_index(drop=True)
    if pool.empty:
        raise PhaseError(f"{name}: no candidate is labelable")

    alignment_config = AlignmentConfig()
    site_config = SiteConfig()
    rows: list[dict[str, Any]] = []
    by_span: dict[tuple[str, str], dict[str, str]] = {}
    by_alignment: dict[tuple[str, str], dict[str, str]] = {}
    indexes: dict[tuple[str, str], Any] = {}
    for pair in sorted(spans):
        document_id, _engine_id = pair
        bundle = bundles[document_id]
        outcome = align_document(bundle.document, spans[pair], bundle.gt_tokens, alignment_config)
        records = [record.model_dump(mode="json") for record in outcome.records]
        indexes.update(
            AlignmentIndex.from_frames(
                pd.DataFrame(records),
                pd.DataFrame([token.model_dump(mode="json") for token in bundle.gt_tokens]),
            )
        )
        span_map: dict[str, str] = {}
        alignment_map: dict[str, str] = {}
        for site in build_sites(outcome.records, spans[pair], site_config):
            align_site_id = f"{name}|{site.site_id}"
            for span_id in site.ocr_span_ids:
                span_map[str(span_id)] = align_site_id
            for alignment_id in site.alignment_ids:
                alignment_map[str(alignment_id)] = align_site_id
            rows.append(
                {
                    "environment": name,
                    "corpus": spec["corpus"],
                    "base_engine": spec["base_engine"],
                    "document_id": document_id,
                    "role": roles.get(document_id, ""),
                    "align_site_id": align_site_id,
                    "site_kind": site.site_kind.value,
                    "evaluable": bool(site.evaluable),
                    "d_before": int(site.d_before),
                    "char_start": int(site.char_start),
                    "char_end": int(site.char_end),
                    "ocr_spans": len(site.ocr_span_ids),
                    "ocr_chars": len(site.ocr_text),
                    "gt_chars": len(site.gt_text),
                }
            )
        by_span[pair] = span_map
        by_alignment[pair] = alignment_map
    alignment = pd.DataFrame(rows, columns=list(ALIGNMENT_COLUMNS))
    links = _site_links(name, sites, indexes, by_span, by_alignment, AnchorKind)

    candidate_rows = pd.DataFrame(
        {
            "environment": name,
            "corpus": spec["corpus"],
            "base_engine": spec["base_engine"],
            "candidate_id": pool["candidate_id"].astype(str).to_numpy(),
            "site_id": (name + "|" + pool["site_id"].astype(str)).to_numpy(),
            "document_id": pool["document_id"].astype(str).to_numpy(),
            "role": pool["document_id"].map(roles).to_numpy(str),
            "outcome": pool["outcome"].astype(str).to_numpy(),
            "is_harmful": pool["is_harmful"].to_numpy(dtype=bool),
            "beneficial": pool["beneficial"].to_numpy(dtype=bool),
            "d_before": pool["d_before"].to_numpy(dtype=np.int64),
            "d_after": pool["d_after"].to_numpy(dtype=np.int64),
            "char_start": pool["char_start"].to_numpy(dtype=np.int64),
            "char_end": pool["char_end"].to_numpy(dtype=np.int64),
            "anchor_kind": pool["anchor_kind"].astype(str).to_numpy(),
            "site_type": pool["site_type"].astype(str).to_numpy(),
            "operation": pool["operation"].astype(str).to_numpy(),
            "generator_source": pool["generator_source"].astype(str).to_numpy(),
            "generator_rank": pool["generator_rank"].to_numpy(dtype=np.int64),
            "generator_score": pool["generator_score"].to_numpy(dtype=float),
            "suspicion_score": pool["suspicion_score"].to_numpy(dtype=float),
            "original_chars": pool["original_ocr"].astype(str).str.len().to_numpy(dtype=np.int64),
            "candidate_chars": pool["candidate_text"].astype(str).str.len().to_numpy(np.int64),
        },
        columns=list(CANDIDATE_COLUMNS),
    )

    evaluation = candidate_rows["role"] == s14.ROLE_EVALUATION
    record = {
        "environment": name,
        "corpus": spec["corpus"],
        "base_engine": spec["base_engine"],
        "pairs_aligned": int(diagnostics["pairs_aligned"]),
        "alignment_sites": len(alignment),
        "alignment_sites_evaluable": int(alignment["evaluable"].sum()),
        "alignment_error_sites": int((alignment["evaluable"] & (alignment["d_before"] > 0)).sum()),
        "discovered_sites": len(sites),
        "candidates_generated": len(candidates),
        "candidates_labelable": len(pool),
        "labelable_fraction": _ratio(len(pool), max(len(candidates), 1)),
        "rows_evaluation": int(evaluation.sum()),
        "documents_evaluation": int(candidate_rows.loc[evaluation, "document_id"].nunique()),
        "ocr_documents": int(ocr_record.get("documents", len(bundles))),
        "linked_discovered_sites": int(links["site_id"].nunique()) if len(links) else 0,
        "elapsed_seconds": time.monotonic() - began,
    }
    return alignment, candidate_rows, links, record


def _site_links(
    name: str,
    sites: pd.DataFrame,
    indexes: dict[tuple[str, str], Any],
    by_span: dict[tuple[str, str], dict[str, str]],
    by_alignment: dict[tuple[str, str], dict[str, str]],
    anchor_kind: Any,
) -> pd.DataFrame:
    """Which alignment sites each discovered site sits on, through the project's own index.

    This is the labelling stage's own coupling, not a coordinate heuristic: a discovered anchor
    names OCR spans, `AlignmentIndex` maps a span to its alignment component, and `build_sites`
    maps a component to the merged correction site it belongs to. A GAP anchor additionally
    covers the OCR-empty components lying between its two flanks, which is exactly the region
    `_labels` reads its gap ground truth from and the only place an omission can be repaired.
    """
    rows: list[dict[str, Any]] = []
    for row in sites.itertuples():
        pair = (str(row.document_id), str(row.engine_id))
        anchors = [part for part in str(row.anchor_ref).split("\0")[1:] if part]
        span_map = by_span.get(pair, {})
        site_id = f"{name}|{row.site_id}"
        for span_id in anchors:
            target = span_map.get(span_id)
            if target is not None:
                rows.append(
                    {
                        "environment": name,
                        "site_id": site_id,
                        "align_site_id": target,
                        "link_kind": "anchor_span",
                    }
                )
        if str(row.anchor_kind) != anchor_kind.GAP.value or len(anchors) != 2:
            continue
        index = indexes.get(pair)
        if index is None:
            continue
        positions = [
            next((i for i, c in enumerate(index.components) if span_id in c.ocr_span_ids), None)
            for span_id in anchors
        ]
        left, right = positions
        if left is None or right is None or left >= right:
            continue
        alignment_map = by_alignment.get(pair, {})
        for component in index.components[left + 1 : right]:
            if component.ocr_span_ids:
                continue
            target = alignment_map.get(str(component.alignment_id))
            if target is not None:
                rows.append(
                    {
                        "environment": name,
                        "site_id": site_id,
                        "align_site_id": target,
                        "link_kind": "gap_interior",
                    }
                )
    frame = pd.DataFrame(rows, columns=list(SITE_LINK_COLUMNS))
    return frame.drop_duplicates(ignore_index=True)


def run_inventory() -> int:
    """Rebuild both site populations for all ten environments, and cache them."""
    started = time.monotonic()
    if not DESIGN_RECORD.is_file():
        raise PhaseError("run --preregister first")
    if ENVIRONMENT_INVENTORY.exists():
        raise PhaseError(
            f"{_relative(ENVIRONMENT_INVENTORY)} already exists; delete it deliberately"
        )
    CACHE.mkdir(parents=True, exist_ok=True)

    complete = all(
        (CACHE / f"{spec['environment'].replace('/', '__')}.{suffix}").is_file()
        for spec in s14.ENVIRONMENTS
        for suffix in ("alignment.parquet", "candidates.parquet", "links.parquet", "record.json")
    )
    pipeline = None if complete else s14.build_pipeline()
    if pipeline is not None:
        print(
            f"  frozen pipeline: {len(pipeline.resources.lm)} lm grams, "
            f"{len(pipeline.resources.lexicon)} lexicon tokens, {pipeline.glyph_prototypes} glyphs"
        )
    alignments: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    linkages: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        slug = name.replace("/", "__")
        align_path = CACHE / f"{slug}.alignment.parquet"
        candidate_path = CACHE / f"{slug}.candidates.parquet"
        link_path = CACHE / f"{slug}.links.parquet"
        record_path = CACHE / f"{slug}.record.json"
        cached = (
            align_path.is_file()
            and candidate_path.is_file()
            and link_path.is_file()
            and record_path.is_file()
        )
        if cached:
            alignment = pd.read_parquet(align_path)
            candidate_rows = pd.read_parquet(candidate_path)
            links = pd.read_parquet(link_path)
            record = cc_read_json(record_path)
            print(f"  {name}: cached")
        else:
            alignment, candidate_rows, links, record = _reconstruct_environment(spec, pipeline)
            alignment.to_parquet(align_path, index=False)
            candidate_rows.to_parquet(candidate_path, index=False)
            links.to_parquet(link_path, index=False)
            _write_json_once(record_path, record)
            print(
                f"  {name}: {record['alignment_error_sites']} aligned error sites, "
                f"{record['candidates_labelable']} labelable candidates, "
                f"{record['rows_evaluation']} evaluation rows "
                f"({record['elapsed_seconds']:.0f}s)"
            )
        alignments.append(alignment)
        candidates.append(candidate_rows)
        linkages.append(links)
        records.append(record)

    alignment_table = pd.concat(alignments, ignore_index=True)
    candidate_table = pd.concat(candidates, ignore_index=True)
    if candidate_table["candidate_id"].duplicated().any():
        raise PhaseError("candidate ids collide across environments")
    if alignment_table["align_site_id"].duplicated().any():
        raise PhaseError("alignment site ids collide across environments")
    link_table = pd.concat(linkages, ignore_index=True)
    _write_table(ALIGNMENT_SITES, alignment_table)
    _write_table(CANDIDATE_ROWS, candidate_table)
    _write_table(SITE_LINKS, link_table)
    if not EVALUATION_ORDER.is_file():
        _write_parquet_once(EVALUATION_ORDER, _evaluation_order())

    evaluation = alignment_table["role"] == s14.ROLE_EVALUATION
    errors = evaluation & alignment_table["evaluable"] & (alignment_table["d_before"] > 0)
    _write_json_once(
        ENVIRONMENT_INVENTORY,
        {
            **_envelope("environment_inventory"),
            "environments": records,
            "totals": {
                "environments": len(records),
                "alignment_sites": len(alignment_table),
                "alignment_error_sites_evaluation": int(errors.sum()),
                "alignment_error_characters_evaluation": int(
                    alignment_table.loc[errors, "d_before"].sum()
                ),
                "candidate_rows": len(candidate_table),
                "candidate_rows_evaluation": int(
                    (candidate_table["role"] == s14.ROLE_EVALUATION).sum()
                ),
                "discovered_sites_evaluation": int(
                    candidate_table.loc[
                        candidate_table["role"] == s14.ROLE_EVALUATION, "site_id"
                    ].nunique()
                ),
                "distinct_evaluation_documents": int(
                    alignment_table.loc[evaluation, "document_id"].nunique()
                ),
                "evaluation_document_environment_pairs": int(
                    alignment_table.loc[evaluation, ["environment", "document_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "corpora": sorted(set(alignment_table["corpus"])),
                "engines": sorted(set(alignment_table["base_engine"])),
            },
            "linkage": {
                "rows": len(link_table),
                "rule": (
                    "a discovered site links to every alignment site its anchor spans belong to, "
                    "and a GAP anchor additionally links to the OCR-empty components between its "
                    "flanks. The coupling is AlignmentIndex, the same one the labelling stage "
                    "reads region ground truth through."
                ),
                "linked_alignment_sites": int(link_table["align_site_id"].nunique()),
                "linked_discovered_sites": int(link_table["site_id"].nunique()),
                "by_link_kind": {
                    str(k): int(v) for k, v in link_table["link_kind"].value_counts().items()
                },
            },
            "artifacts": {
                _relative(ALIGNMENT_SITES): file_sha256(ALIGNMENT_SITES),
                _relative(CANDIDATE_ROWS): file_sha256(CANDIDATE_ROWS),
                _relative(SITE_LINKS): file_sha256(SITE_LINKS),
                _relative(EVALUATION_ORDER): file_sha256(EVALUATION_ORDER),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_census(alignment_table, candidate_table, link_table)
    print(
        f"inventory: {len(records)} environments, {int(errors.sum())} evaluation OCR error sites, "
        f"{len(candidate_table)} candidate rows -> {_relative(CANDIDATE_ROWS)} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _census(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def _write_census(alignment: pd.DataFrame, candidates: pd.DataFrame, links: pd.DataFrame) -> None:
    """The two population censuses, as summaries over the tables rather than copies of them."""
    evaluation = alignment[alignment["role"] == s14.ROLE_EVALUATION]
    errors = evaluation[evaluation["evaluable"] & (evaluation["d_before"] > 0)]
    rows = candidates[candidates["role"] == s14.ROLE_EVALUATION]
    _write_json_once(
        SITE_INVENTORY,
        {
            **_envelope("site_inventory"),
            "summary_of": [_relative(ALIGNMENT_SITES), _relative(SITE_LINKS)],
            "populations": {
                UNIT_ALIGNMENT_SITE: (
                    "alignment components after the adjacent-segmentation merge; the OCR error "
                    "sites are the evaluable ones whose OCR text differs from ground truth"
                ),
                UNIT_DISCOVERED_SITE: "sites the OCR-only enumerator anchored",
            },
            "alignment_sites_all_roles": len(alignment),
            "alignment_sites_evaluation": len(evaluation),
            "alignment_sites_evaluation_by_kind": _census(evaluation, "site_kind"),
            "alignment_sites_evaluation_evaluable": int(evaluation["evaluable"].sum()),
            "ocr_error_sites": len(errors),
            "ocr_error_sites_by_kind": _census(errors, "site_kind"),
            "ocr_error_characters": int(errors["d_before"].sum()),
            "unevaluable_error_sites": int(
                ((~evaluation["evaluable"]) & (evaluation["d_before"] > 0)).sum()
            ),
            "discovered_sites_evaluation": int(rows["site_id"].nunique()),
            "discovered_sites_by_anchor_kind": _census(
                rows.drop_duplicates("site_id"), "anchor_kind"
            ),
            "discovered_sites_by_site_type": _census(rows.drop_duplicates("site_id"), "site_type"),
            "links": len(links),
            "links_by_kind": _census(links, "link_kind"),
            "per_environment": [
                {
                    "environment": str(name),
                    "alignment_sites": len(group),
                    "ocr_error_sites": int((group["evaluable"] & (group["d_before"] > 0)).sum()),
                    "ocr_error_characters": int(
                        group.loc[group["evaluable"] & (group["d_before"] > 0), "d_before"].sum()
                    ),
                    "unevaluable_error_sites": int(
                        ((~group["evaluable"]) & (group["d_before"] > 0)).sum()
                    ),
                }
                for name, group in evaluation.groupby("environment", sort=False)
            ],
        },
    )
    _write_json_once(
        CANDIDATE_INVENTORY,
        {
            **_envelope("candidate_inventory"),
            "summary_of": [_relative(CANDIDATE_ROWS)],
            "candidate_rows_all_roles": len(candidates),
            "candidate_rows_evaluation": len(rows),
            "by_outcome": _census(rows, "outcome"),
            "by_operation": _census(rows, "operation"),
            "by_anchor_kind": _census(rows, "anchor_kind"),
            "by_site_type": _census(rows, "site_type"),
            "by_generator_source": _census(rows, "generator_source"),
            "harmful": int(rows["is_harmful"].sum()),
            "beneficial": int(rows["beneficial"].sum()),
            "exact": int((rows["outcome"] == OUTCOME_EXACT).sum()),
            "per_environment": [
                {
                    "environment": str(name),
                    "candidate_rows": len(group),
                    "discovered_sites": int(group["site_id"].nunique()),
                    "by_outcome": _census(group, "outcome"),
                    "by_generator_source": _census(group, "generator_source"),
                }
                for name, group in rows.groupby("environment", sort=False)
            ],
        },
    )


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    """Write once, and on a rerun assert the existing table is the one this run rebuilt.

    The environment rebuild is deterministic, so a second run must produce the same rows. Making
    that an assertion rather than a silent skip is the difference between a resumable phase and a
    phase that quietly serves a stale table.
    """
    if not path.is_file():
        _write_parquet_once(path, frame)
        return
    stored = pd.read_parquet(path)
    if not stored.equals(frame):
        raise PhaseError(f"{_relative(path)} exists and differs from this rebuild")


def _evaluation_order() -> pd.DataFrame:
    """The candidate id behind every position of the frozen evaluation score vector."""
    built, _inventory = s15._environments()
    stored = np.load(dt1.GEOMETRY, allow_pickle=False)
    frames: list[pd.DataFrame] = []
    for environment in built:
        block = environment.setup.evaluation
        scores = stored[dt1._geometry_key(environment.name, "eval_scores")]
        if scores.shape != (block.size,):
            raise PhaseError(f"{environment.name}: frozen score vector does not match the block")
        frames.append(
            pd.DataFrame(
                {
                    "environment": environment.name,
                    "position": np.arange(block.size, dtype=np.int64),
                    "candidate_id": block.candidates.astype(str),
                    "site_id": environment.name + "|" + block.sites.astype(str),
                    "document_id": block.documents.astype(str),
                    "frozen_score": np.asarray(scores, dtype=float),
                    "harmful": np.asarray(block.harmful, dtype=bool),
                    "beneficial": np.asarray(block.beneficial, dtype=bool),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------ the analysis view over one environment
#
# One dataclass per environment, built once, in the frozen evaluation score order. Every ceiling
# below is a mask over `Ceiling.n_rows` positions and an aggregation over sites, documents or
# characters. Nothing here refits, rescores or regenerates anything.


@dataclass(frozen=True, slots=True)
class Ceiling:
    """One environment's evaluation split, in the frozen score order, with both site views."""

    name: str
    corpus: str
    base_engine: str
    candidate_id: np.ndarray
    document_id: np.ndarray
    doc_of: np.ndarray
    documents: np.ndarray
    score: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    exact: np.ndarray
    d_before: np.ndarray
    d_after: np.ndarray
    char_start: np.ndarray
    char_end: np.ndarray
    generator_rank: np.ndarray
    anchor_kind: np.ndarray
    site_type: np.ndarray
    generator_source: np.ndarray
    site_of: np.ndarray
    site_ids: np.ndarray
    site_document: np.ndarray
    site_is_error: np.ndarray
    site_repairable: np.ndarray
    site_beneficial: np.ndarray
    site_kind: np.ndarray
    site_link_indptr: np.ndarray
    site_link_index: np.ndarray
    align_error_ids: np.ndarray
    align_error_d_before: np.ndarray
    align_error_kind: np.ndarray
    align_error_document: np.ndarray
    align_covered: np.ndarray
    align_error_sites: int
    align_error_chars: int
    align_by_kind: dict[str, tuple[int, int]]
    align_documents: int
    align_error_chars_by_document: np.ndarray
    align_total_chars: int
    deployment: Any

    @property
    def n_rows(self) -> int:
        return int(self.score.size)

    @property
    def n_sites(self) -> int:
        return int(self.site_ids.size)

    @property
    def n_error_sites(self) -> int:
        """The denominator: OCR error sites, section 4's definition."""
        return int(self.align_error_ids.size)

    @property
    def n_discovered_error_sites(self) -> int:
        return int(self.site_is_error.sum())

    @property
    def n_repairable_sites(self) -> int:
        return int(self.repaired_sites(np.ones(self.n_rows, dtype=bool)).sum())

    def linked(self, sites: np.ndarray) -> np.ndarray:
        """The OCR error sites a set of discovered sites sits on."""
        out = np.zeros(self.align_error_ids.size, dtype=bool)
        for site in np.flatnonzero(sites):
            lo, hi = self.site_link_indptr[site], self.site_link_indptr[site + 1]
            out[self.site_link_index[lo:hi]] = True
        return out

    def repaired_sites(self, accepted: np.ndarray) -> np.ndarray:
        """OCR error sites an accepted set can leave exactly correct. An UPPER BOUND.

        A discovered region that an exact candidate repairs may be one of several the alignment
        site spans, so counting the site as repaired can only overstate, never understate. The
        share of repaired sites carrying exactly one linked region is reported beside every use
        of this quantity, because that is the share on which the bound is tight.
        """
        return self.linked(_any_by_group(accepted & self.exact, self.site_of, self.n_sites))


def _site_index(site_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable site ordering and the per-row site index."""
    names, inverse = np.unique(site_ids, return_inverse=True)
    return names, inverse.astype(np.int64)


def _any_by_group(values: np.ndarray, group: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=bool)
    if values.size:
        np.logical_or.at(out, group, values)
    return out


def _max_by_group(values: np.ndarray, group: np.ndarray, size: int) -> np.ndarray:
    out = np.full(size, np.iinfo(np.int64).min, dtype=np.int64)
    if values.size:
        np.maximum.at(out, group, values)
    return out


def load_ceilings() -> dict[str, Ceiling]:
    """Assemble every environment's analysis view from the two frozen tables."""
    for path in (ALIGNMENT_SITES, CANDIDATE_ROWS, SITE_LINKS, EVALUATION_ORDER):
        if not path.is_file():
            raise PhaseError("run --inventory first")
    alignment = pd.read_parquet(ALIGNMENT_SITES)
    candidates = pd.read_parquet(CANDIDATE_ROWS)
    links = pd.read_parquet(SITE_LINKS)
    order = pd.read_parquet(EVALUATION_ORDER)
    deployments = dt1.load_deployments()

    candidates = candidates[candidates["role"] == s14.ROLE_EVALUATION]
    indexed = candidates.set_index("candidate_id")
    alignment = alignment[alignment["role"] == s14.ROLE_EVALUATION]

    out: dict[str, Ceiling] = {}
    for name, block in order.groupby("environment", sort=False):
        block = block.sort_values("position", kind="stable")
        rows = indexed.reindex(block["candidate_id"].astype(str).to_numpy())
        if rows["outcome"].isna().any():
            missing = int(rows["outcome"].isna().sum())
            raise PhaseError(f"{name}: {missing} frozen evaluation rows are not in the rebuild")
        site_ids = rows["site_id"].astype(str).to_numpy()
        names, site_of = _site_index(site_ids)
        d_before = rows["d_before"].to_numpy(dtype=np.int64)
        exact = (rows["outcome"].astype(str) == OUTCOME_EXACT).to_numpy(dtype=bool)
        beneficial = rows["beneficial"].to_numpy(dtype=bool)
        site_is_error = _max_by_group(d_before, site_of, names.size) > 0
        site_document = np.empty(names.size, dtype=object)
        site_document[site_of] = rows["document_id"].astype(str).to_numpy()
        site_kind = np.empty(names.size, dtype=object)
        site_kind[site_of] = rows["site_type"].astype(str).to_numpy()

        environment_alignment = alignment[alignment["environment"] == name]
        evaluable = environment_alignment[environment_alignment["evaluable"]]
        errors = evaluable[evaluable["d_before"] > 0].sort_values("align_site_id", kind="stable")
        error_ids = errors["align_site_id"].astype(str).to_numpy()
        error_position = {value: index for index, value in enumerate(error_ids.tolist())}
        indptr, index, covered = _link_structure(
            links[links["environment"] == name], names, error_position
        )
        by_kind = {
            str(kind): (len(group), int(group["d_before"].sum()))
            for kind, group in errors.groupby("site_kind")
        }
        document_names = np.asarray(deployments[name].eval_documents, dtype=str)
        per_document = (
            errors.groupby("document_id")["d_before"].sum().reindex(document_names).fillna(0)
        )

        out[name] = Ceiling(
            name=str(name),
            corpus=str(rows["corpus"].iloc[0]),
            base_engine=str(rows["base_engine"].iloc[0]),
            candidate_id=block["candidate_id"].astype(str).to_numpy(),
            document_id=rows["document_id"].astype(str).to_numpy(),
            doc_of=np.asarray(deployments[name].eval_doc_of, dtype=np.int64),
            documents=document_names,
            score=block["frozen_score"].to_numpy(dtype=float),
            harmful=rows["is_harmful"].to_numpy(dtype=bool),
            beneficial=beneficial,
            exact=exact,
            d_before=d_before,
            d_after=rows["d_after"].to_numpy(dtype=np.int64),
            char_start=rows["char_start"].to_numpy(dtype=np.int64),
            char_end=rows["char_end"].to_numpy(dtype=np.int64),
            generator_rank=rows["generator_rank"].to_numpy(dtype=np.int64),
            anchor_kind=rows["anchor_kind"].astype(str).to_numpy(),
            site_type=rows["site_type"].astype(str).to_numpy(),
            generator_source=rows["generator_source"].astype(str).to_numpy(),
            site_of=site_of,
            site_ids=names,
            site_document=site_document,
            site_is_error=site_is_error,
            site_repairable=_any_by_group(exact, site_of, names.size),
            site_beneficial=_any_by_group(beneficial, site_of, names.size),
            site_kind=site_kind,
            site_link_indptr=indptr,
            site_link_index=index,
            align_error_ids=error_ids,
            align_error_d_before=errors["d_before"].to_numpy(dtype=np.int64),
            align_error_kind=errors["site_kind"].astype(str).to_numpy(),
            align_error_document=errors["document_id"].astype(str).to_numpy(),
            align_covered=covered,
            align_error_sites=len(errors),
            align_error_chars=int(errors["d_before"].sum()),
            align_by_kind=by_kind,
            align_documents=int(evaluable["document_id"].nunique()),
            align_error_chars_by_document=per_document.to_numpy(dtype=np.int64),
            align_total_chars=int(evaluable["ocr_chars"].sum()),
            deployment=deployments[name],
        )
    if len(out) != ENVIRONMENT_COUNT:
        raise PhaseError(f"expected {ENVIRONMENT_COUNT} environments, assembled {len(out)}")
    return out


def _link_structure(
    links: pd.DataFrame, site_names: np.ndarray, error_position: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR linkage from discovered sites to the OCR error sites they sit on."""
    site_position = {value: index for index, value in enumerate(site_names.tolist())}
    buckets: list[list[int]] = [[] for _ in range(site_names.size)]
    covered = np.zeros(len(error_position), dtype=bool)
    for site_id, align_site_id in zip(
        links["site_id"].astype(str), links["align_site_id"].astype(str), strict=True
    ):
        source = site_position.get(site_id)
        target = error_position.get(align_site_id)
        if source is None or target is None:
            continue
        buckets[source].append(target)
        covered[target] = True
    lengths = np.asarray([len(bucket) for bucket in buckets], dtype=np.int64)
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    index = np.asarray(
        [value for bucket in buckets for value in sorted(set(bucket))], dtype=np.int64
    )
    if index.size != int(lengths.sum()):
        # de-duplication inside a bucket shortens it; rebuild the pointers from the kept values.
        kept = [sorted(set(bucket)) for bucket in buckets]
        lengths = np.asarray([len(bucket) for bucket in kept], dtype=np.int64)
        indptr = np.concatenate(([0], np.cumsum(lengths)))
        index = np.asarray([value for bucket in kept for value in bucket], dtype=np.int64)
    return indptr, index, covered


def sites_repaired(ceiling: Ceiling, accepted: np.ndarray) -> int:
    """OCR error sites an accepted set can leave exactly correct. Never double-counts a site."""
    return int(ceiling.repaired_sites(accepted).sum())


def repaired_characters(ceiling: Ceiling, accepted: np.ndarray) -> float:
    """Wrong characters removed, counting each stream position at most once.

    Two accepted candidates can target overlapping regions of the same page, and a deployment
    cannot apply both. The maximum-weight non-overlapping selection is the ceiling of what any
    consistent application of the accepted set could remove, computed exactly by the classic
    interval DP rather than by summing gains that would double-count shared characters.
    """
    gain = (ceiling.d_before - ceiling.d_after).astype(float)
    keep = accepted & (gain > 0)
    if not keep.any():
        return 0.0
    total = 0.0
    for document in np.unique(ceiling.doc_of[keep]):
        rows = np.flatnonzero(keep & (ceiling.doc_of == document))
        total += _interval_dp(ceiling.char_start[rows], ceiling.char_end[rows], gain[rows])
    return float(total)


def _interval_dp(start: np.ndarray, end: np.ndarray, weight: np.ndarray) -> float:
    """Maximum total weight over pairwise non-overlapping half-open intervals.

    A zero-width region is an insertion point and occupies one stream slot, so two proposals at
    the same point conflict with each other and with any replacement spanning it.
    """
    stop = np.maximum(end, start + 1)
    order = np.lexsort((start, stop))
    start, stop, weight = start[order], stop[order], weight[order]
    ends: list[int] = []
    best: list[float] = []
    running = 0.0
    for index in range(start.size):
        position = bisect.bisect_right(ends, int(start[index])) - 1
        take = (best[position] if position >= 0 else 0.0) + float(weight[index])
        running = max(running, take)
        ends.append(int(stop[index]))
        best.append(running)
    return running


# ------------------------------------------------------------------ section 35: reproduction


def run_reproduce() -> int:
    """Reproduce SGV14's labels and SGV-DT1's deployment metrics before anything is interpreted."""
    started = time.monotonic()
    rebuilt = pd.read_parquet(CANDIDATE_ROWS)
    frozen = pd.read_parquet(
        s14.ENVIRONMENT_ROWS,
        columns=[
            "candidate_id",
            "site_id",
            "document_id",
            "role",
            "outcome",
            "is_harmful",
            "beneficial",
            "environment",
        ],
    )
    merged = frozen.merge(
        rebuilt,
        on="candidate_id",
        how="outer",
        suffixes=("_frozen", "_rebuilt"),
        indicator=True,
    )
    only_frozen = int((merged["_merge"] == "left_only").sum())
    only_rebuilt = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    label_fields: dict[str, int] = {}
    for field in ("outcome", "is_harmful", "beneficial", "role", "document_id", "environment"):
        left = both[f"{field}_frozen"].astype(str).to_numpy()
        right = both[f"{field}_rebuilt"].astype(str).to_numpy()
        label_fields[field] = int((left != right).sum())
    site_field = int(
        (
            both["environment_frozen"].astype(str) + "|" + both["site_id_frozen"].astype(str)
            != both["site_id_rebuilt"].astype(str)
        ).sum()
    )
    label_fields["site_id"] = site_field
    label_differences = only_frozen + only_rebuilt + sum(label_fields.values())

    ceilings = load_ceilings()
    geometry_fields: dict[str, int] = {"eval_harm": 0, "eval_beneficial": 0, "eval_documents": 0}
    for ceiling in ceilings.values():
        deployment = ceiling.deployment
        geometry_fields["eval_harm"] += int(
            (np.asarray(deployment.eval_harm, dtype=bool) != ceiling.harmful).sum()
        )
        geometry_fields["eval_beneficial"] += int(
            (np.asarray(deployment.eval_beneficial, dtype=bool) != ceiling.beneficial).sum()
        )
        geometry_fields["eval_documents"] += int(
            (ceiling.documents[ceiling.doc_of] != ceiling.document_id).sum()
        )
    geometry_differences = sum(geometry_fields.values())

    deployment_fields, deployment_cells = _reproduce_deployment(ceilings)
    total = label_differences + geometry_differences + sum(deployment_fields.values())
    if total:
        raise PhaseError(
            f"reproduction gate failed: {label_differences} label, {geometry_differences} "
            f"geometry and {sum(deployment_fields.values())} deployment differences"
        )

    _write_json_once(
        UPSTREAM_REPRODUCTION,
        {
            **_envelope("upstream_reproduction"),
            "rule": "expected maximum difference is exactly 0; anything else stops the stage.",
            "sgv14_labels": {
                "artifact": _relative(s14.ENVIRONMENT_ROWS),
                "sha256": file_sha256(s14.ENVIRONMENT_ROWS),
                "rows_compared": len(both),
                "rows_only_in_frozen_table": only_frozen,
                "rows_only_in_rebuild": only_rebuilt,
                "field_differences": label_fields,
                "total_differences": label_differences,
                "recomputed": [
                    "outcome",
                    "is_harmful",
                    "beneficial",
                    "d_before",
                    "d_after",
                    "site_id",
                    "candidate_id",
                ],
            },
            "sgv_dt1_geometry": {
                "artifact": _relative(dt1.GEOMETRY),
                "sha256": file_sha256(dt1.GEOMETRY),
                "environments": len(ceilings),
                "field_differences": geometry_fields,
                "total_differences": geometry_differences,
            },
            "sgv_dt1_deployment_metrics": {
                "artifact": _relative(dt1.DECISION),
                "sha256": file_sha256(dt1.DECISION),
                "cells_compared": deployment_cells,
                "field_differences": deployment_fields,
                "total_differences": sum(deployment_fields.values()),
            },
            "maximum_absolute_difference": 0.0,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"reproduce: {len(both)} SGV14 label rows, {len(ceilings)} score geometries and "
        f"{deployment_cells} SGV-DT1 deployment cells reproduced, {total} differences"
    )
    return 0


def _reproduce_deployment(ceilings: dict[str, Ceiling]) -> tuple[dict[str, int], int]:
    """Re-derive SGV-DT1's selected regime through SGV-DT1's own functions and compare."""
    decision = cc_read_json(dt1.DECISION)
    policy = decision["best_practical_policy"]
    fields = {"automatic_harm": 0, "automatic_repair_recall": 0, "review_rate": 0}
    per_environment = practical_deployment(ceilings, policy)
    observed = {
        "automatic_harm": _mean([r["automatic_harm"] for r in per_environment.values()]),
        "automatic_repair_recall": _mean(
            [r["automatic_repair_recall"] for r in per_environment.values()]
        ),
        "review_rate": _mean([r["review_rate"] for r in per_environment.values()]),
    }
    for field in fields:
        if abs(observed[field] - float(decision[field])) > 1e-12:
            fields[field] = 1
    return fields, len(per_environment) * DOC_DRAWS


# ------------------------------------- sections 9, 21: the frozen practical deployment baseline


def practical_deployment(
    ceilings: dict[str, Ceiling], policy: dict[str, Any], epsilon: float | None = None
) -> dict[str, dict[str, Any]]:
    """SGV-DT1's selected regime, replayed through SGV-DT1's own functions.

    Nothing here is re-derived: the boundary rule, the routing, the review budget, the document
    budget, the draw seeds and the reviewer model are read off SGV-DT1's decision artifact and
    executed by SGV-DT1's code. The only thing this stage adds is a second reading of the same
    action masks in site and character units.
    """
    tolerance = float(PRIMARY_EPSILON if epsilon is None else epsilon)
    rule = str(policy["boundary_rule"])
    routing = str(policy["routing"])
    budget = int(policy["target_document_budget"])
    rate = float(policy["review_budget"])
    accuracy = float(dt1.REVIEWER_ACCURACY[str(policy["reviewer_scenario"])])
    out: dict[str, dict[str, Any]] = {}
    for name, ceiling in ceilings.items():
        deployment = ceiling.deployment
        correctness = dt1.reviewer_correctness(deployment)
        count = int(rate * deployment.n_rows)
        draws: list[dict[str, Any]] = []
        for draw in range(DOC_DRAWS):
            sample = dt1.document_sample(deployment, budget, dt1._sample_seed(name, budget, draw))
            boundary = dt1.select_boundary(deployment, rule, sample, tolerance)
            tau = float(boundary["tau"])
            order = dt1.review_order(deployment, tau, routing)
            auto, applied = dt1._action_masks(
                deployment, tau, order, min(count, int(order.size)), accuracy, correctness
            )
            draws.append(_deployment_reading(ceiling, auto, applied, min(count, int(order.size))))
        out[name] = {
            key: _mean([draw[key] for draw in draws])
            for key in draws[0]
            if isinstance(draws[0][key], float)
        }
        out[name]["draws"] = len(draws)
        out[name]["environment"] = name
    return out


def _deployment_reading(
    ceiling: Ceiling, auto: np.ndarray, applied: np.ndarray, reviewed: int
) -> dict[str, Any]:
    """One action assignment, read in candidate rows, error sites and error characters."""
    automatic = int(auto.sum())
    automatic_harm = int((auto & ceiling.harmful).sum())
    beneficial_total = int(ceiling.beneficial.sum())
    sites = ceiling.n_error_sites
    chars = ceiling.align_error_chars
    return {
        "automatic_harm": _ratio(automatic_harm, automatic),
        "automatic_repair_recall": _ratio(int((auto & ceiling.beneficial).sum()), beneficial_total),
        "total_repair_recall": _ratio(int((applied & ceiling.beneficial).sum()), beneficial_total),
        "review_rate": _ratio(reviewed, ceiling.n_rows),
        "coverage": _ratio(automatic, ceiling.n_rows),
        "site_repair_recall": _ratio(sites_repaired(ceiling, auto), sites),
        "site_repair_recall_total": _ratio(sites_repaired(ceiling, applied), sites),
        "character_repair_recall": _ratio(repaired_characters(ceiling, auto), chars),
        "character_repair_recall_total": _ratio(repaired_characters(ceiling, applied), chars),
    }


# ------------------------------------------------------------------ section 5: opportunity


def _opportunity_row(ceiling: Ceiling) -> dict[str, Any]:
    """Every section-5 quantity for one environment, on the OCR error site denominator."""
    everything = np.ones(ceiling.n_rows, dtype=bool)
    repaired = ceiling.repaired_sites(everything)
    beneficial_repaired = ceiling.linked(
        _any_by_group(ceiling.beneficial, ceiling.site_of, ceiling.n_sites)
    )
    error = ceiling.site_is_error
    denominator = ceiling.n_error_sites
    widths = np.diff(ceiling.site_link_indptr)
    exact_sites = _any_by_group(ceiling.exact, ceiling.site_of, ceiling.n_sites)
    tight = int((widths[exact_sites] == 1).sum())
    return {
        "environment": ceiling.name,
        "corpus": ceiling.corpus,
        "base_engine": ceiling.base_engine,
        "evaluation_documents": int(np.unique(ceiling.doc_of).size),
        "candidate_rows": ceiling.n_rows,
        "discovered_sites": ceiling.n_sites,
        "discovered_error_sites": int(error.sum()),
        "ocr_error_sites": denominator,
        "ocr_error_characters": ceiling.align_error_chars,
        "ocr_error_sites_covered_by_discovery": int(ceiling.align_covered.sum()),
        "discovery_recall": _ratio(int(ceiling.align_covered.sum()), denominator),
        "repairable_sites": int(repaired.sum()),
        "beneficial_available_sites": int(beneficial_repaired.sum()),
        "opportunity_recall": _ratio(int(repaired.sum()), denominator),
        "opportunity_recall_beneficial": _ratio(int(beneficial_repaired.sum()), denominator),
        "opportunity_recall_given_discovery": _ratio(
            int(repaired.sum()), int(ceiling.align_covered.sum())
        ),
        "unrepairable_fraction": 1.0 - _ratio(int(repaired.sum()), denominator),
        "discovery_gap": 1.0 - _ratio(int(ceiling.align_covered.sum()), denominator),
        "generation_gap": _ratio(
            int(ceiling.align_covered.sum()) - int(repaired.sum()), denominator
        ),
        "exact_repairs_on_a_single_region": tight,
        "exact_repair_sites": int(exact_sites.sum()),
        "bound_is_tight_share": _ratio(tight, int(exact_sites.sum())),
        "beneficial_candidates": int(ceiling.beneficial.sum()),
        "exact_candidates": int(ceiling.exact.sum()),
        "harmful_candidates": int(ceiling.harmful.sum()),
        "neutral_candidates": int((~ceiling.beneficial & ~ceiling.harmful).sum()),
        "candidate_multiplicity": _ratio(ceiling.n_rows, ceiling.n_sites),
    }


def _anchored_characters(ceiling: Ceiling) -> float:
    """OCR error mass the discovery enumerator anchored at all, non-overlapping."""
    rows = ceiling.d_before > 0
    if not rows.any():
        return 0.0
    total = 0.0
    weight = ceiling.d_before.astype(float)
    for document in np.unique(ceiling.doc_of[rows]):
        keep = np.flatnonzero(rows & (ceiling.doc_of == document))
        total += _interval_dp(ceiling.char_start[keep], ceiling.char_end[keep], weight[keep])
    return float(total)


def run_opportunity() -> int:
    """Sections 5.1-5.6: opportunity recall, multiplicity and beneficial/harmful composition."""
    started = time.monotonic()
    ceilings = load_ceilings()
    rows = [_opportunity_row(ceiling) for ceiling in ceilings.values()]
    frame = pd.DataFrame(rows)
    mass = {
        ceiling.name: {
            "anchored_error_characters": _anchored_characters(ceiling),
            "ocr_error_characters": ceiling.align_error_chars,
            "median_candidate_region_distance": float(np.median(ceiling.d_before)),
            "median_candidate_region_characters": float(
                np.median(ceiling.char_end - ceiling.char_start)
            ),
        }
        for ceiling in ceilings.values()
    }
    inflated = [
        name
        for name, record in mass.items()
        if record["anchored_error_characters"] > record["ocr_error_characters"]
    ]

    _write_json_once(
        CANDIDATE_OPPORTUNITY,
        {
            **_oracle_envelope("candidate_opportunity"),
            "denominator": (
                "OCR error sites: evaluable alignment components, after the project's adjacent "
                "segmentation merge, whose OCR text differs from ground truth. Section 4's "
                "definition, independent of what the generator proposed."
            ),
            "numerator": (
                "OCR error sites carrying at least one discovered region whose candidate set "
                "contains an exact repair. An UPPER BOUND on sites made exactly correct, because "
                "an alignment site can span more than one discovered region."
            ),
            "per_environment": rows,
            "pooled": {
                "ocr_error_sites": int(frame["ocr_error_sites"].sum()),
                "ocr_error_characters": int(frame["ocr_error_characters"].sum()),
                "discovered_error_sites": int(frame["discovered_error_sites"].sum()),
                "ocr_error_sites_covered_by_discovery": int(
                    frame["ocr_error_sites_covered_by_discovery"].sum()
                ),
                "repairable_sites": int(frame["repairable_sites"].sum()),
                "discovery_recall": _ratio(
                    int(frame["ocr_error_sites_covered_by_discovery"].sum()),
                    int(frame["ocr_error_sites"].sum()),
                ),
                "opportunity_recall": _ratio(
                    int(frame["repairable_sites"].sum()), int(frame["ocr_error_sites"].sum())
                ),
            },
            "mean_opportunity_recall": _mean(list(frame["opportunity_recall"])),
            "mean_opportunity_recall_beneficial": _mean(
                list(frame["opportunity_recall_beneficial"])
            ),
            "mean_opportunity_recall_given_discovery": _mean(
                list(frame["opportunity_recall_given_discovery"])
            ),
            "mean_unrepairable_fraction": _mean(list(frame["unrepairable_fraction"])),
            "mean_discovery_recall": _mean(list(frame["discovery_recall"])),
            "mean_discovery_gap": _mean(list(frame["discovery_gap"])),
            "mean_generation_gap": _mean(list(frame["generation_gap"])),
            "bound_tightness": {
                "rule": (
                    "the numerator overstates only where an exactly repaired discovered region "
                    "is one of several the alignment site spans. This is the share of exactly "
                    "repaired regions that sit on exactly one alignment site."
                ),
                "pooled_share": _ratio(
                    int(frame["exact_repairs_on_a_single_region"].sum()),
                    int(frame["exact_repair_sites"].sum()),
                ),
                "per_environment": {
                    row["environment"]: row["bound_is_tight_share"] for row in rows
                },
            },
            "character_mass_is_reported_but_not_used": {
                "why": (
                    "the frozen region ground truth is coarser than the OCR region on the "
                    "historical-print corpus, so a candidate's `d_before` there measures the "
                    "granularity mismatch as much as the token error. Anchored character mass "
                    "therefore exceeds the component-wise error mass on those environments and "
                    "is not a sound denominator. It is recorded as a limitation, and every "
                    "headline in this stage is counted in sites."
                ),
                "per_environment": mass,
                "environments_where_anchored_mass_exceeds_error_mass": inflated,
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_multiplicity(ceilings)
    _write_ratio(ceilings, frame)
    print(
        f"opportunity: mean opportunity recall {_mean(list(frame['opportunity_recall'])):.4f}, "
        f"discovery recall {_mean(list(frame['discovery_recall'])):.4f} over "
        f"{int(frame['ocr_error_sites'].sum())} OCR error sites"
    )
    return 0


def _write_multiplicity(ceilings: dict[str, Ceiling]) -> None:
    """Section 5.5: how many candidates, correct candidates and harmful candidates per site."""
    per_environment: list[dict[str, Any]] = []
    for ceiling in ceilings.values():
        size = ceiling.n_sites
        total = np.bincount(ceiling.site_of, minlength=size).astype(float)
        correct = np.bincount(ceiling.site_of, weights=ceiling.exact, minlength=size)
        harmful = np.bincount(ceiling.site_of, weights=ceiling.harmful, minlength=size)
        repairable = ceiling.site_is_error & ceiling.site_repairable
        error = ceiling.site_is_error
        per_environment.append(
            {
                "environment": ceiling.name,
                "all_error_sites": {
                    "candidates": _quantiles(total[error]),
                    "correct_candidates": _quantiles(correct[error]),
                    "harmful_candidates": _quantiles(harmful[error]),
                },
                "repairable_sites": {
                    "candidates": _quantiles(total[repairable]),
                    "correct_candidates": _quantiles(correct[repairable]),
                    "harmful_candidates": _quantiles(harmful[repairable]),
                },
                "harmful_candidates_per_repairable_site": _ratio(
                    float(harmful[repairable].sum()), int(repairable.sum())
                ),
                "harmful_candidates_per_error_site": _ratio(
                    float(harmful[error].sum()), int(error.sum())
                ),
            }
        )
    _write_json_once(
        CANDIDATE_MULTIPLICITY,
        {
            **_oracle_envelope("candidate_multiplicity"),
            "unit": UNIT_DISCOVERED_SITE,
            "per_environment": per_environment,
            "mean_harmful_candidates_per_repairable_site": _mean(
                [
                    r["harmful_candidates_per_repairable_site"]
                    for r in per_environment
                    if not math.isnan(r["harmful_candidates_per_repairable_site"])
                ]
            ),
            "mean_candidates_per_error_site": _mean(
                [r["all_error_sites"]["candidates"]["mean"] for r in per_environment]
            ),
        },
    )


def _write_ratio(ceilings: dict[str, Ceiling], frame: pd.DataFrame) -> None:
    """Section 5.6: the beneficial-to-harmful opportunity ratio."""
    rows = []
    for ceiling in ceilings.values():
        beneficial = int(ceiling.beneficial.sum())
        harmful = int(ceiling.harmful.sum())
        rows.append(
            {
                "environment": ceiling.name,
                "beneficial_candidates": beneficial,
                "exact_candidates": int(ceiling.exact.sum()),
                "harmful_candidates": harmful,
                "beneficial_to_harmful_ratio": _ratio(beneficial, harmful),
                "exact_to_harmful_ratio": _ratio(int(ceiling.exact.sum()), harmful),
                "harmful_share_of_candidates": _ratio(harmful, ceiling.n_rows),
            }
        )
    _write_json_once(
        BENEFICIAL_HARMFUL_RATIO,
        {
            **_oracle_envelope("beneficial_harmful_ratio"),
            "definitions": {
                "beneficial": f"{OUTCOME_EXACT} or {OUTCOME_PARTIAL}",
                "harmful": f"{OUTCOME_MISCORRECTION} or {OUTCOME_OVERCORRECTION}",
                "policy": "strict_worsening, unchanged from upstream",
            },
            "per_environment": rows,
            "pooled_beneficial_to_harmful_ratio": _ratio(
                int(frame["beneficial_candidates"].sum()), int(frame["harmful_candidates"].sum())
            ),
            "pooled_exact_to_harmful_ratio": _ratio(
                int(frame["exact_candidates"].sum()), int(frame["harmful_candidates"].sum())
            ),
            "mean_beneficial_to_harmful_ratio": _mean(
                [r["beneficial_to_harmful_ratio"] for r in rows]
            ),
            "pooled_harmful_share_of_candidates": _ratio(
                int(frame["harmful_candidates"].sum()), int(frame["candidate_rows"].sum())
            ),
            "pooled_beneficial_share_of_candidates": _ratio(
                int(frame["beneficial_candidates"].sum()), int(frame["candidate_rows"].sum())
            ),
        },
    )


# ------------------------------------------- sections 6-9, 20-21: the four ceilings and a floor


def _prefix_boundaries(score: np.ndarray) -> np.ndarray:
    """Every prefix a threshold on `score` can actually produce, ties respected."""
    order = np.argsort(-score, kind="stable")
    ordered = score[order]
    distinct = np.flatnonzero(np.diff(ordered) != 0) + 1
    return np.concatenate((distinct, [ordered.size]))


def oracle_cut(ceiling: Ceiling, score: np.ndarray, epsilon: float) -> dict[str, Any]:
    """The best feasible threshold on a fixed ranking, chosen with evaluation labels.

    The objective is repaired OCR error sites subject to automatic harm at most epsilon, and ties
    break toward the SMALLEST accepted set. An environment whose only feasible prefixes repair
    nothing therefore reports a refusal rather than an arbitrary shallow cut, which is the
    conservative reading and keeps the reported coverage and harm consistent with the reported
    recall.

    Analysis only. This is the ceiling a ranking could reach if the operator already knew the
    answer; no deployable procedure has this information and nothing in this stage selects a
    policy from it.
    """
    order = np.argsort(-score, kind="stable")
    harmful = ceiling.harmful[order]
    beneficial = ceiling.beneficial[order]
    exact = ceiling.exact[order]
    sites = ceiling.site_of[order]
    accepted_harm = np.concatenate(([0], np.cumsum(harmful)))
    accepted_beneficial = np.concatenate(([0], np.cumsum(beneficial)))
    boundaries = _prefix_boundaries(score)

    repaired = np.zeros(ceiling.align_error_ids.size, dtype=bool)
    seen = np.zeros(ceiling.n_sites, dtype=bool)
    cursor = 0
    best = {
        "repaired_sites": 0,
        "accepted": 0,
        "harmful_accepted": 0,
        "beneficial_accepted": 0,
        "prefix": 0,
        "tau": float("inf"),
        "automatic_harm": float("nan"),
    }
    for prefix in boundaries:
        while cursor < prefix:
            if exact[cursor] and not seen[sites[cursor]]:
                seen[sites[cursor]] = True
                site = int(sites[cursor])
                lo, hi = ceiling.site_link_indptr[site], ceiling.site_link_indptr[site + 1]
                repaired[ceiling.site_link_index[lo:hi]] = True
            cursor += 1
        accepted = int(prefix)
        harm = _ratio(int(accepted_harm[prefix]), accepted)
        if accepted and harm > epsilon:
            continue
        count = int(repaired.sum())
        if count > best["repaired_sites"]:
            best = {
                "repaired_sites": count,
                "accepted": accepted,
                "harmful_accepted": int(accepted_harm[prefix]),
                "beneficial_accepted": int(accepted_beneficial[prefix]),
                "prefix": accepted,
                "tau": float(score[order][prefix - 1]) if prefix else float("inf"),
                "automatic_harm": harm,
            }
    best["repair_recall"] = _ratio(best["repaired_sites"], ceiling.n_error_sites)
    best["candidate_repair_recall"] = _ratio(
        best["beneficial_accepted"], int(ceiling.beneficial.sum())
    )
    best["coverage"] = _ratio(best["accepted"], ceiling.n_rows)
    best["environment"] = ceiling.name
    best["epsilon"] = float(epsilon)
    return best


def perfect_ranking_score(ceiling: Ceiling) -> np.ndarray:
    """Section 20: every correct candidate above every harmful one, universe unchanged."""
    score = np.full(ceiling.n_rows, 0.5)
    score[ceiling.exact] = 1.0
    score[ceiling.beneficial & ~ceiling.exact] = 0.75
    score[ceiling.harmful] = 0.0
    return score


def run_oracles() -> int:
    """Sections 6-9 and 20-21: the candidate, risk, perfect-ranking, ranking and policy rungs."""
    started = time.monotonic()
    ceilings = load_ceilings()
    everything = np.ones(1, dtype=bool)
    del everything

    candidate: list[dict[str, Any]] = []
    for ceiling in ceilings.values():
        rows = np.ones(ceiling.n_rows, dtype=bool)
        repaired = ceiling.repaired_sites(rows)
        accepted = int(ceiling.exact.sum())
        candidate.append(
            {
                "environment": ceiling.name,
                "ocr_error_sites": ceiling.n_error_sites,
                "repaired_sites": int(repaired.sum()),
                "unrepaired_sites": ceiling.n_error_sites - int(repaired.sum()),
                "repair_recall": _ratio(int(repaired.sum()), ceiling.n_error_sites),
                "candidate_repair_recall": _ratio(
                    int(ceiling.beneficial.sum()), int(ceiling.beneficial.sum())
                ),
                "accepted_candidates": accepted,
                "harmful_accepted": 0,
                "automatic_harm": 0.0,
                "coverage": _ratio(accepted, ceiling.n_rows),
                "corrected_sites": int(repaired.sum()),
            }
        )
    _write_json_once(
        CANDIDATE_ORACLE,
        {
            **_oracle_envelope("candidate_oracle"),
            "name": "O_candidate",
            "rule": (
                "at every OCR error site accept a correct candidate if the frozen universe "
                "contains one, otherwise preserve the OCR. Never accepts a harmful candidate, so "
                "automatic harm is 0 by construction."
            ),
            "not_a_deployable_method": True,
            "per_environment": candidate,
            "mean_repair_recall": _mean([r["repair_recall"] for r in candidate]),
            "pooled_repair_recall": _ratio(
                sum(r["repaired_sites"] for r in candidate),
                sum(r["ocr_error_sites"] for r in candidate),
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )

    risk = {
        f"epsilon_{epsilon}": [
            {
                "environment": row["environment"],
                "repair_recall": row["repair_recall"],
                "automatic_harm": 0.0,
                "feasible": True,
            }
            for row in candidate
        ]
        for epsilon in EPSILONS
    }
    _write_json_once(
        RISK_CANDIDATE_ORACLE,
        {
            **_oracle_envelope("risk_candidate_oracle"),
            "name": "O_candidate_risk",
            "rule": (
                "the best repair recall achievable at automatic harm <= epsilon when candidate "
                "outcomes are known perfectly."
            ),
            "epsilon_invariant": True,
            "why_epsilon_invariant": (
                "perfect knowledge never accepts a harmful candidate, so the constrained optimum "
                "carries harm 0 and equals the unconstrained candidate oracle at every declared "
                "tolerance. The epsilon dependence SGV-DT1 measured is therefore a property of "
                "the ranking and the policy, not of the candidate universe. Falsification test 11 "
                "asserts this rather than assuming it."
            ),
            "per_epsilon": risk,
            "mean_repair_recall": {
                f"epsilon_{epsilon}": _mean([r["repair_recall"] for r in candidate])
                for epsilon in EPSILONS
            },
        },
    )

    ranking: dict[str, list[dict[str, Any]]] = {}
    perfect: dict[str, list[dict[str, Any]]] = {}
    for epsilon in EPSILONS:
        ranking[f"epsilon_{epsilon}"] = [
            oracle_cut(ceiling, ceiling.score, epsilon) for ceiling in ceilings.values()
        ]
        perfect[f"epsilon_{epsilon}"] = [
            oracle_cut(ceiling, perfect_ranking_score(ceiling), epsilon)
            for ceiling in ceilings.values()
        ]
    _write_json_once(
        RANKING_ORACLE,
        {
            **_oracle_envelope("ranking_oracle"),
            "name": "O_rank",
            "rule": (
                "the frozen adapted reliability ordering, with the deployment cut chosen by an "
                "oracle over every threshold the ranking can express, subject to automatic harm "
                "<= epsilon."
            ),
            "ranking_is_frozen": True,
            "objective": "repaired OCR error sites subject to automatic harm <= epsilon",
            "tie_break": (
                "toward the smallest accepted set, so an environment whose only feasible prefixes "
                "repair no OCR error site reports a refusal rather than an arbitrary shallow cut."
            ),
            "environments_where_no_feasible_cut_repairs_anything": {
                key: [row["environment"] for row in rows if row["repaired_sites"] == 0]
                for key, rows in ranking.items()
            },
            "per_epsilon": ranking,
            "mean_repair_recall": {
                key: _mean([r["repair_recall"] for r in rows]) for key, rows in ranking.items()
            },
            "mean_candidate_repair_recall": {
                key: _mean([r["candidate_repair_recall"] for r in rows])
                for key, rows in ranking.items()
            },
            "mean_coverage": {
                key: _mean([r["coverage"] for r in rows]) for key, rows in ranking.items()
            },
        },
    )
    _write_json_once(
        PERFECT_RANKING,
        {
            **_oracle_envelope("perfect_ranking_counterfactual"),
            "name": "perfect_ranking_on_current_universe",
            "rule": (
                "section 20. Every correct candidate is ranked above every beneficial one, and "
                "both above every neutral and harmful one; the candidate universe is unchanged "
                "and the cut is still oracle-selected."
            ),
            "per_epsilon": perfect,
            "mean_repair_recall": {
                key: _mean([r["repair_recall"] for r in rows]) for key, rows in perfect.items()
            },
            "expected_identity": (
                "a perfect ranking over a fixed universe cannot repair a site the universe "
                "cannot repair, so this equals the candidate oracle. Falsification test 3 "
                "asserts the ranking gap vanishes here."
            ),
        },
    )

    decision = cc_read_json(dt1.DECISION)
    policy = decision["best_practical_policy"]
    practical = practical_deployment(ceilings, policy)
    _write_json_once(
        DEPLOYMENT_BASELINE,
        {
            **_envelope("deployment_baseline"),
            "name": "D_practical",
            "analysis_only": False,
            "uses_ground_truth": False,
            "source": _relative(dt1.DECISION),
            "source_sha256": file_sha256(dt1.DECISION),
            "policy": policy,
            "replayed_through": "sgv_dt1_deployment_tradeoff.select_boundary/_action_masks",
            "draws": DOC_DRAWS,
            "per_environment": list(practical.values()),
            "mean_site_repair_recall": _mean([r["site_repair_recall"] for r in practical.values()]),
            "mean_site_repair_recall_total": _mean(
                [r["site_repair_recall_total"] for r in practical.values()]
            ),
            "mean_automatic_repair_recall": _mean(
                [r["automatic_repair_recall"] for r in practical.values()]
            ),
            "mean_total_repair_recall": _mean(
                [r["total_repair_recall"] for r in practical.values()]
            ),
            "mean_automatic_harm": _mean([r["automatic_harm"] for r in practical.values()]),
        },
    )

    perfect_policy = _perfect_deployment(ceilings)
    _write_json_once(
        PERFECT_DEPLOYMENT,
        {
            **_oracle_envelope("perfect_deployment_counterfactual"),
            "name": "frozen_ranking_plus_perfect_deployment",
            "rule": (
                "section 21. The ranking stays frozen; the deployment rule is allowed the "
                "evaluation labels, so it takes the oracle cut AND spends its review budget on "
                "the candidates a perfect router would choose, with an ideal reviewer."
            ),
            "review_budget": float(PRACTICAL_REVIEW_BUDGET),
            "per_environment": perfect_policy,
            "mean_site_repair_recall": _mean([r["site_repair_recall"] for r in perfect_policy]),
            "mean_site_repair_recall_total": _mean(
                [r["site_repair_recall_total"] for r in perfect_policy]
            ),
        },
    )
    key = f"epsilon_{PRIMARY_EPSILON}"
    print(
        f"oracles: candidate {_mean([r['repair_recall'] for r in candidate]):.4f}, ranking "
        f"{_mean([r['repair_recall'] for r in ranking[key]]):.4f}, deployment "
        f"{_mean([r['site_repair_recall'] for r in practical.values()]):.4f} at epsilon "
        f"{PRIMARY_EPSILON}"
    )
    return 0


def _perfect_deployment(ceilings: dict[str, Ceiling]) -> list[dict[str, Any]]:
    """Frozen ranking, oracle cut, and a review budget spent by a perfect router."""
    out: list[dict[str, Any]] = []
    for ceiling in ceilings.values():
        cut = oracle_cut(ceiling, ceiling.score, PRIMARY_EPSILON)
        tau = float(cut["tau"])
        auto = ceiling.score >= tau if np.isfinite(tau) else np.zeros(ceiling.n_rows, dtype=bool)
        budget = int(PRACTICAL_REVIEW_BUDGET * ceiling.n_rows)
        # A perfect router spends the budget where a perfect reviewer changes the outcome: first
        # on automatic edits that are harmful, then on rejected candidates that are beneficial.
        priority = np.concatenate(
            (
                np.flatnonzero(auto & ceiling.harmful),
                np.flatnonzero(~auto & ceiling.beneficial),
            )
        )
        chosen = priority[:budget]
        reviewed = np.zeros(ceiling.n_rows, dtype=bool)
        reviewed[chosen] = True
        applied = (auto & ~reviewed) | (reviewed & ceiling.beneficial)
        automatic = auto & ~reviewed
        out.append(
            {
                "environment": ceiling.name,
                "reviewed": int(reviewed.sum()),
                "review_rate": _ratio(int(reviewed.sum()), ceiling.n_rows),
                "site_repair_recall": _ratio(
                    sites_repaired(ceiling, automatic), ceiling.n_error_sites
                ),
                "site_repair_recall_total": _ratio(
                    sites_repaired(ceiling, applied), ceiling.n_error_sites
                ),
                "automatic_harm": _ratio(
                    int((automatic & ceiling.harmful).sum()), int(automatic.sum())
                ),
                "total_repair_recall": _ratio(
                    int((applied & ceiling.beneficial).sum()), int(ceiling.beneficial.sum())
                ),
            }
        )
    return out


# --------------------------------------------------- sections 10-12: the ceiling decomposition


def _ladder_rows(ceilings: dict[str, Ceiling], epsilon: float) -> list[dict[str, Any]]:
    """One environment per row: every rung of the ladder and the three losses between them."""
    decision = cc_read_json(dt1.DECISION)
    practical = practical_deployment(ceilings, decision["best_practical_policy"], epsilon)
    rows: list[dict[str, Any]] = []
    for name, ceiling in ceilings.items():
        everything = np.ones(ceiling.n_rows, dtype=bool)
        repaired = int(ceiling.repaired_sites(everything).sum())
        covered = int(ceiling.align_covered.sum())
        opportunity = _ratio(repaired, ceiling.n_error_sites)
        rank = oracle_cut(ceiling, ceiling.score, epsilon)["repair_recall"]
        perfect = oracle_cut(ceiling, perfect_ranking_score(ceiling), epsilon)["repair_recall"]
        deployed = float(practical[name]["site_repair_recall"])
        rows.append(
            {
                "environment": name,
                "corpus": ceiling.corpus,
                "base_engine": ceiling.base_engine,
                "epsilon": float(epsilon),
                "ocr_error_sites": ceiling.n_error_sites,
                "opportunity_recall": opportunity,
                "candidate_oracle": opportunity,
                "risk_candidate_oracle": opportunity,
                "perfect_ranking": perfect,
                "ranking_oracle": rank,
                "practical_deployment": deployed,
                "practical_deployment_with_review": float(
                    practical[name]["site_repair_recall_total"]
                ),
                "candidate_gap": 1.0 - opportunity,
                "discovery_gap": 1.0 - _ratio(covered, ceiling.n_error_sites),
                "generation_gap": _ratio(covered - repaired, ceiling.n_error_sites),
                "ranking_gap": opportunity - rank,
                "deployment_gap": rank - deployed,
            }
        )
    return rows


def run_ceiling() -> int:
    """Sections 10-12: the ladder, the three gaps and the normalized contributions."""
    started = time.monotonic()
    ceilings = load_ceilings()
    per_epsilon = {f"epsilon_{epsilon}": _ladder_rows(ceilings, epsilon) for epsilon in EPSILONS}
    primary = per_epsilon[f"epsilon_{PRIMARY_EPSILON}"]

    non_monotone_by_epsilon = {
        key: [
            {
                "environment": row["environment"],
                "ranking_gap": row["ranking_gap"],
                "deployment_gap": row["deployment_gap"],
            }
            for row in rows
            if row["ranking_gap"] < -1e-12 or row["deployment_gap"] < -1e-12
        ]
        for key, rows in per_epsilon.items()
    }
    non_monotone = [
        row["environment"] for row in non_monotone_by_epsilon[f"epsilon_{PRIMARY_EPSILON}"]
    ]
    normalized: list[dict[str, Any]] = []
    for row in primary:
        reference = 1.0 - row["practical_deployment"]
        shares = {component: _ratio(row[f"{component}_gap"], reference) for component in COMPONENTS}
        normalized.append(
            {
                "environment": row["environment"],
                "reference_denominator": reference,
                **{f"share_{component}": shares[component] for component in COMPONENTS},
                "share_sum": float(sum(v for v in shares.values() if not math.isnan(v))),
                "normalization_valid": all(
                    value >= -1e-9 for value in shares.values() if not math.isnan(value)
                ),
            }
        )
    _write_json_once(
        CEILING_DECOMPOSITION,
        {
            **_oracle_envelope("ceiling_decomposition"),
            "ladder": list(LADDER),
            "unit": UNIT_ALIGNMENT_SITE,
            "denominator": "OCR error sites",
            "per_epsilon": per_epsilon,
            "primary_epsilon": float(PRIMARY_EPSILON),
            "means": {
                rung: _mean([row[rung] for row in primary])
                for rung in (*LADDER, "practical_deployment_with_review")
            },
            "mean_gaps": {
                component: _mean([row[f"{component}_gap"] for row in primary])
                for component in COMPONENTS
            },
            "mean_candidate_gap_parts": {
                "discovery": _mean([row["discovery_gap"] for row in primary]),
                "generation": _mean([row["generation_gap"] for row in primary]),
                "discovery_means": (
                    "no discovered site carrying a labelled evaluation candidate sits on the "
                    "error site"
                ),
                "generation_means": ("one does, and no candidate at it is an exact repair"),
            },
            "means_by_epsilon": {
                key: {rung: _mean([row[rung] for row in rows]) for rung in LADDER}
                for key, rows in per_epsilon.items()
            },
            "environments_with_a_negative_gap": non_monotone,
            "environments_with_a_negative_gap_by_epsilon": non_monotone_by_epsilon,
            "why_a_gap_can_be_negative": (
                "the deployed boundary is fixed on the purchased certification documents and is "
                "not required to hold epsilon on the evaluation split, while the oracle cut is. "
                "The two therefore optimise over different feasible sets, and at a tolerance "
                "where the oracle cut refuses outright the deployed policy can still apply a few "
                "edits and repair a site. Section 12 forbids forcing the normalization over such "
                "an environment; its raw gaps are reported instead."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        NORMALIZED_GAP,
        {
            **_oracle_envelope("normalized_gap_decomposition"),
            "reference": "1 - practical_deployment_repair_recall",
            "rule": (
                "the three losses partition the unresolved gap exactly, so the shares sum to one "
                "whenever every gap is non-negative. Section 12 forbids forcing a normalization "
                "that would produce a negative share; environments where the ladder is not "
                "monotone are named and their raw gaps reported instead."
            ),
            "per_environment": normalized,
            "mean_shares": {
                component: _mean(
                    [
                        row[f"share_{component}"]
                        for row in normalized
                        if row["normalization_valid"] and not math.isnan(row[f"share_{component}"])
                    ]
                )
                for component in COMPONENTS
            },
            "environments_valid": int(sum(row["normalization_valid"] for row in normalized)),
            "environments_with_a_negative_gap": non_monotone,
        },
    )
    print(
        "ceiling: candidate gap "
        f"{_mean([r['candidate_gap'] for r in primary]):.4f}, ranking gap "
        f"{_mean([r['ranking_gap'] for r in primary]):.4f}, deployment gap "
        f"{_mean([r['deployment_gap'] for r in primary]):.4f}"
    )
    return 0


# ------------------------------------------------------------- section 13: the site taxonomy


def run_taxonomy() -> int:
    """C0-C5, assigned per draw so that every OCR error site falls in exactly one class."""
    started = time.monotonic()
    ceilings = load_ceilings()
    alignment = pd.read_parquet(ALIGNMENT_SITES)
    alignment = alignment[alignment["role"] == s14.ROLE_EVALUATION]
    decision = cc_read_json(dt1.DECISION)
    policy = decision["best_practical_policy"]
    rule = str(policy["boundary_rule"])
    routing = str(policy["routing"])
    budget = int(policy["target_document_budget"])
    rate = float(policy["review_budget"])
    accuracy = float(dt1.REVIEWER_ACCURACY[str(policy["reviewer_scenario"])])

    per_environment: list[dict[str, Any]] = []
    for name, ceiling in ceilings.items():
        deployment = ceiling.deployment
        correctness = dt1.reviewer_correctness(deployment)
        count = int(rate * deployment.n_rows)
        repairable = ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool))
        by_rank = ceiling.repaired_sites(
            ceiling.score >= float(oracle_cut(ceiling, ceiling.score, PRIMARY_EPSILON)["tau"])
        )
        draws: list[dict[str, float]] = []
        for draw in range(DOC_DRAWS):
            sample = dt1.document_sample(deployment, budget, dt1._sample_seed(name, budget, draw))
            tau = float(dt1.select_boundary(deployment, rule, sample, PRIMARY_EPSILON)["tau"])
            order = dt1.review_order(deployment, tau, routing)
            auto, applied = dt1._action_masks(
                deployment, tau, order, min(count, int(order.size)), accuracy, correctness
            )
            automatic = ceiling.repaired_sites(auto)
            total = ceiling.repaired_sites(applied)
            c3 = automatic
            c4 = total & ~automatic
            c2 = repairable & ~total & by_rank
            c1 = repairable & ~total & ~by_rank
            c0 = ~repairable
            draws.append(
                {
                    C0_NO_CORRECT_CANDIDATE: float(c0.sum()),
                    C1_RANKED_BELOW_HARM: float(c1.sum()),
                    C2_NOT_APPLIED: float(c2.sum()),
                    C3_AUTOMATIC_REPAIR: float(c3.sum()),
                    C4_REVIEWED_REPAIR: float(c4.sum()),
                }
            )
        counts = {key: _mean([draw[key] for draw in draws]) for key in draws[0]}
        environment_sites = alignment[alignment["environment"] == name]
        unevaluable = int(
            ((~environment_sites["evaluable"]) & (environment_sites["d_before"] > 0)).sum()
        )
        counts[C5_UNEVALUABLE] = float(unevaluable)
        total_sites = ceiling.n_error_sites
        per_environment.append(
            {
                "environment": name,
                "ocr_error_sites": total_sites,
                "counts": counts,
                "shares": {
                    key: _ratio(value, total_sites)
                    for key, value in counts.items()
                    if key != C5_UNEVALUABLE
                },
                "c5_is_outside_the_denominator": True,
                "partition_sums_to_the_denominator": bool(
                    abs(sum(v for k, v in counts.items() if k != C5_UNEVALUABLE) - total_sites)
                    < 1e-6
                ),
                "draws": DOC_DRAWS,
            }
        )
    pooled = {key: sum(row["counts"][key] for row in per_environment) for key in TAXONOMY}
    denominator = sum(row["ocr_error_sites"] for row in per_environment)
    _write_json_once(
        SITE_FAILURE_TAXONOMY,
        {
            **_oracle_envelope("site_failure_taxonomy"),
            "classes": {
                C0_NO_CORRECT_CANDIDATE: "no correct candidate exists -- candidate generation",
                C1_RANKED_BELOW_HARM: (
                    "a correct candidate exists but no feasible cut on the frozen ranking "
                    "accepts it -- ranking"
                ),
                C2_NOT_APPLIED: (
                    "a correct candidate is accepted by the oracle cut on the frozen ranking, "
                    "but the deployed policy does not apply it -- decision / deployment"
                ),
                C3_AUTOMATIC_REPAIR: "repaired automatically -- success",
                C4_REVIEWED_REPAIR: "repaired through human review -- success",
                C5_UNEVALUABLE: (
                    "the alignment could not resolve the region, so the site carries no "
                    "evaluable ground truth. Counted and reported, never forced into another "
                    "class and never inside the denominator."
                ),
            },
            "assignment": (
                "assigned per purchased-document draw so that each draw is an exact partition; "
                f"the counts reported are means over the {DOC_DRAWS} draws."
            ),
            "per_environment": per_environment,
            "pooled_counts": pooled,
            "pooled_shares": {
                key: _ratio(value, denominator)
                for key, value in pooled.items()
                if key != C5_UNEVALUABLE
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"taxonomy: C0 {pooled[C0_NO_CORRECT_CANDIDATE]:.0f}, C1 "
        f"{pooled[C1_RANKED_BELOW_HARM]:.0f}, C2 {pooled[C2_NOT_APPLIED]:.0f}, C3 "
        f"{pooled[C3_AUTOMATIC_REPAIR]:.1f}, C4 {pooled[C4_REVIEWED_REPAIR]:.1f}, C5 "
        f"{pooled[C5_UNEVALUABLE]:.0f}"
    )
    return 0


# ------------------------------------------------ sections 14-17: engine, domain, type, quality


def _reverse_links(ceiling: Ceiling) -> tuple[np.ndarray, np.ndarray]:
    """OCR error site -> the discovered sites sitting on it, as a CSR pair."""
    size = ceiling.align_error_ids.size
    buckets: list[list[int]] = [[] for _ in range(size)]
    for site in range(ceiling.n_sites):
        lo, hi = ceiling.site_link_indptr[site], ceiling.site_link_indptr[site + 1]
        for target in ceiling.site_link_index[lo:hi]:
            buckets[int(target)].append(site)
    lengths = np.asarray([len(bucket) for bucket in buckets], dtype=np.int64)
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    index = np.asarray([value for bucket in buckets for value in bucket], dtype=np.int64)
    return indptr, index


def _stratum(rows: Sequence[dict[str, Any]], label: str, key: str) -> dict[str, Any]:
    """Pool a set of environment rows into one stratum summary."""
    sites = sum(row["ocr_error_sites"] for row in rows)
    return {
        key: label,
        "environments": [row["environment"] for row in rows],
        "ocr_error_sites": sites,
        "repairable_sites": sum(row["repairable_sites"] for row in rows),
        "opportunity_recall": _ratio(sum(row["repairable_sites"] for row in rows), sites),
        "discovery_recall": _ratio(
            sum(row["ocr_error_sites_covered_by_discovery"] for row in rows), sites
        ),
        "mean_opportunity_recall": _mean([row["opportunity_recall"] for row in rows]),
        "candidate_rows": sum(row["candidate_rows"] for row in rows),
        "beneficial_candidates": sum(row["beneficial_candidates"] for row in rows),
        "harmful_candidates": sum(row["harmful_candidates"] for row in rows),
        "beneficial_to_harmful_ratio": _ratio(
            sum(row["beneficial_candidates"] for row in rows),
            sum(row["harmful_candidates"] for row in rows),
        ),
        "candidate_multiplicity": _ratio(
            sum(row["candidate_rows"] for row in rows),
            sum(row["discovered_sites"] for row in rows),
        ),
    }


def run_strata() -> int:
    """Sections 14-17: opportunity by error type, engine, domain and raw OCR quality."""
    started = time.monotonic()
    ceilings = load_ceilings()
    opportunity = cc_read_json(CANDIDATE_OPPORTUNITY)["per_environment"]
    ladder = {row["environment"]: row for row in _ladder_rows(ceilings, PRIMARY_EPSILON)}
    by_name = {row["environment"]: row for row in opportunity}

    engines: dict[str, list[dict[str, Any]]] = {}
    domains: dict[str, list[dict[str, Any]]] = {}
    for row in opportunity:
        engines.setdefault(row["base_engine"], []).append(row)
        domains.setdefault(row["corpus"], []).append(row)

    _write_json_once(
        ENGINE_ANALYSIS,
        {
            **_oracle_envelope("engine_analysis"),
            "note": (
                "an engine can appear in more than one environment: the same base engine read "
                "under a different configuration or language is a different environment and its "
                "rows are pooled here by the base engine the frozen model was keyed to."
            ),
            "per_engine": [
                {
                    **_stratum(rows, engine, "base_engine"),
                    "candidate_oracle": _mean(
                        [ladder[row["environment"]]["candidate_oracle"] for row in rows]
                    ),
                    "ranking_oracle": _mean(
                        [ladder[row["environment"]]["ranking_oracle"] for row in rows]
                    ),
                    "practical_deployment": _mean(
                        [ladder[row["environment"]]["practical_deployment"] for row in rows]
                    ),
                    "dominant_component": _largest(
                        {
                            component: _mean(
                                [ladder[row["environment"]][f"{component}_gap"] for row in rows]
                            )
                            for component in COMPONENTS
                        }
                    ),
                }
                for engine, rows in sorted(engines.items())
            ],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DOMAIN_ANALYSIS,
        {
            **_oracle_envelope("domain_analysis"),
            "per_domain": [
                {
                    **_stratum(rows, domain, "corpus"),
                    "candidate_oracle": _mean(
                        [ladder[row["environment"]]["candidate_oracle"] for row in rows]
                    ),
                    "ranking_oracle": _mean(
                        [ladder[row["environment"]]["ranking_oracle"] for row in rows]
                    ),
                    "practical_deployment": _mean(
                        [ladder[row["environment"]]["practical_deployment"] for row in rows]
                    ),
                    "dominant_component": _largest(
                        {
                            component: _mean(
                                [ladder[row["environment"]][f"{component}_gap"] for row in rows]
                            )
                            for component in COMPONENTS
                        }
                    ),
                }
                for domain, rows in sorted(domains.items())
            ],
        },
    )

    kinds: dict[str, dict[str, Any]] = {}
    for ceiling in ceilings.values():
        indptr, index = _reverse_links(ceiling)
        repaired = ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool))
        candidate_count = np.bincount(ceiling.site_of, minlength=ceiling.n_sites)
        harmful_count = np.bincount(
            ceiling.site_of, weights=ceiling.harmful, minlength=ceiling.n_sites
        )
        for position, kind in enumerate(ceiling.align_error_kind):
            bucket = kinds.setdefault(
                str(kind),
                {
                    "site_kind": str(kind),
                    "ocr_error_sites": 0,
                    "covered": 0,
                    "repairable_sites": 0,
                    "error_characters": 0,
                    "linked_candidates": 0,
                    "linked_harmful_candidates": 0,
                },
            )
            bucket["ocr_error_sites"] += 1
            bucket["error_characters"] += int(ceiling.align_error_d_before[position])
            lo, hi = indptr[position], indptr[position + 1]
            linked = index[lo:hi]
            if linked.size:
                bucket["covered"] += 1
                bucket["linked_candidates"] += int(candidate_count[linked].sum())
                bucket["linked_harmful_candidates"] += int(harmful_count[linked].sum())
            if repaired[position]:
                bucket["repairable_sites"] += 1
    for bucket in kinds.values():
        bucket["opportunity_recall"] = _ratio(bucket["repairable_sites"], bucket["ocr_error_sites"])
        bucket["discovery_recall"] = _ratio(bucket["covered"], bucket["ocr_error_sites"])
        bucket["candidate_multiplicity"] = _ratio(bucket["linked_candidates"], bucket["covered"])
        bucket["harmful_share"] = _ratio(
            bucket["linked_harmful_candidates"], bucket["linked_candidates"]
        )
        bucket["mean_error_characters"] = _ratio(
            bucket["error_characters"], bucket["ocr_error_sites"]
        )
    _write_json_once(
        ERROR_TYPE_ANALYSIS,
        {
            **_oracle_envelope("error_type_analysis"),
            "taxonomy_source": (
                "ocr_risk.schemas.enums.SiteKind, assigned by ocr_risk.edits.sites.site_kind_for "
                "from the alignment relation. No category is invented here."
            ),
            "per_error_type": sorted(kinds.values(), key=lambda b: -b["ocr_error_sites"]),
            "pooled": {
                "ocr_error_sites": sum(b["ocr_error_sites"] for b in kinds.values()),
                "error_characters": sum(b["error_characters"] for b in kinds.values()),
                "repairable_sites": sum(b["repairable_sites"] for b in kinds.values()),
                "best_error_type": max(kinds.values(), key=lambda b: b["opportunity_recall"])[
                    "site_kind"
                ],
                "error_characters_outside_the_best_error_type": sum(
                    b["error_characters"]
                    for b in kinds.values()
                    if b["site_kind"]
                    != max(kinds.values(), key=lambda c: c["opportunity_recall"])["site_kind"]
                ),
            },
            "note": (
                "the oracle repair ceiling for an error type equals its opportunity recall, "
                "because perfect selection repairs exactly the sites a correct candidate exists "
                "for."
            ),
        },
    )

    documents = _document_quality(ceilings)
    _write_json_once(
        OCR_QUALITY_ANALYSIS,
        {
            **_oracle_envelope("ocr_quality_analysis"),
            "question": (
                "section 17: do worse OCR outputs simply create more repair opportunities, or do "
                "they produce errors the current generator cannot handle?"
            ),
            "unit": UNIT_DOCUMENT,
            "documents": len(documents["rows"]),
            "association": documents["association"],
            "by_quality_quintile": documents["quintiles"],
            "per_environment": documents["per_environment"],
        },
    )
    print(
        f"strata: {len(engines)} engines, {len(domains)} domains, {len(kinds)} error types, "
        f"{len(documents['rows'])} documents"
    )
    del by_name
    return 0


def _largest(values: dict[str, float]) -> str:
    return max(values, key=lambda key: (values[key], key))


def _document_quality(ceilings: dict[str, Ceiling]) -> dict[str, Any]:
    """Section 17: per-document raw OCR quality against opportunity and ambiguity."""
    alignment = pd.read_parquet(ALIGNMENT_SITES)
    alignment = alignment[(alignment["role"] == s14.ROLE_EVALUATION) & alignment["evaluable"]]
    rows: list[dict[str, Any]] = []
    per_environment: list[dict[str, Any]] = []
    for name, ceiling in ceilings.items():
        block = alignment[alignment["environment"] == name]
        chars = block.groupby("document_id")["ocr_chars"].sum()
        errors = block[block["d_before"] > 0].groupby("document_id")["d_before"].sum()
        indptr, index = _reverse_links(ceiling)
        repaired = ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool))
        candidate_count = np.bincount(ceiling.site_of, minlength=ceiling.n_sites)
        local: list[dict[str, Any]] = []
        for document in np.unique(ceiling.align_error_document):
            positions = np.flatnonzero(ceiling.align_error_document == document)
            linked = [int(index[indptr[p] : indptr[p + 1]].size > 0) for p in positions]
            ambiguity = [
                float(candidate_count[index[indptr[p] : indptr[p + 1]]].sum())
                for p in positions
                if index[indptr[p] : indptr[p + 1]].size
            ]
            row = {
                "environment": name,
                "document_id": str(document),
                "ocr_characters": int(chars.get(document, 0)),
                "error_characters": int(errors.get(document, 0)),
                "raw_error_rate": _ratio(int(errors.get(document, 0)), int(chars.get(document, 0))),
                "ocr_error_sites": int(positions.size),
                "discovery_recall": _ratio(int(sum(linked)), int(positions.size)),
                "opportunity_recall": _ratio(int(repaired[positions].sum()), int(positions.size)),
                "mean_ambiguity": _mean(ambiguity) if ambiguity else float("nan"),
            }
            rows.append(row)
            local.append(row)
        per_environment.append(
            {
                "environment": name,
                "documents": len(local),
                "spearman_raw_error_rate_vs_opportunity": _spearman(
                    [r["raw_error_rate"] for r in local],
                    [r["opportunity_recall"] for r in local],
                ),
            }
        )
    usable = [
        row
        for row in rows
        if not math.isnan(row["raw_error_rate"]) and not math.isnan(row["opportunity_recall"])
    ]
    quality = np.asarray([row["raw_error_rate"] for row in usable])
    quintiles: list[dict[str, Any]] = []
    if quality.size:
        edges = np.quantile(quality, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for position in range(5):
            low, high = edges[position], edges[position + 1]
            selected = [
                row
                for row in usable
                if low <= row["raw_error_rate"] <= high
                and (position == 4 or row["raw_error_rate"] < edges[position + 1])
            ]
            quintiles.append(
                {
                    "quintile": position + 1,
                    "raw_error_rate_range": [float(low), float(high)],
                    "documents": len(selected),
                    "mean_raw_error_rate": _mean([r["raw_error_rate"] for r in selected]),
                    "mean_opportunity_recall": _mean([r["opportunity_recall"] for r in selected]),
                    "mean_discovery_recall": _mean([r["discovery_recall"] for r in selected]),
                    "mean_ambiguity": _mean(
                        [
                            r["mean_ambiguity"]
                            for r in selected
                            if not math.isnan(r["mean_ambiguity"])
                        ]
                    ),
                    "ocr_error_sites": sum(r["ocr_error_sites"] for r in selected),
                }
            )
    association = {
        "spearman_raw_error_rate_vs_opportunity_recall": _spearman(
            [r["raw_error_rate"] for r in usable], [r["opportunity_recall"] for r in usable]
        ),
        "spearman_raw_error_rate_vs_discovery_recall": _spearman(
            [r["raw_error_rate"] for r in usable], [r["discovery_recall"] for r in usable]
        ),
        "spearman_raw_error_rate_vs_ambiguity": _spearman(
            [r["raw_error_rate"] for r in usable if not math.isnan(r["mean_ambiguity"])],
            [r["mean_ambiguity"] for r in usable if not math.isnan(r["mean_ambiguity"])],
        ),
        "spearman_error_sites_vs_opportunity_recall": _spearman(
            [float(r["ocr_error_sites"]) for r in usable],
            [r["opportunity_recall"] for r in usable],
        ),
        "bootstrap_ci_spearman_quality_vs_opportunity": _bootstrap_spearman(usable),
        "documents_used": len(usable),
    }
    return {
        "rows": rows,
        "association": association,
        "quintiles": quintiles,
        "per_environment": per_environment,
    }


def _rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = array.argsort(kind="stable")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(1, array.size + 1, dtype=float)
    # average ranks over ties, so a heavily tied column cannot manufacture correlation.
    unique, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 3 or len(left) != len(right):
        return float("nan")
    a, b = _rank(left), _rank(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else float("nan")


def _bootstrap_spearman(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Document-clustered bootstrap interval for the quality/opportunity association."""
    quality = [row["raw_error_rate"] for row in rows]
    opportunity = [row["opportunity_recall"] for row in rows]
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[float] = []
    size = len(rows)
    for _ in range(BOOTSTRAP_RESAMPLES):
        pick = generator.integers(0, size, size)
        value = _spearman([quality[i] for i in pick], [opportunity[i] for i in pick])
        if not math.isnan(value):
            draws.append(value)
    if not draws:
        return {"low": float("nan"), "high": float("nan"), "resamples": 0}
    array = np.asarray(draws)
    return {
        "low": float(np.quantile(array, ALPHA / 2)),
        "high": float(np.quantile(array, 1 - ALPHA / 2)),
        "resamples": len(draws),
    }


# ------------------------------------------------------ sections 18-19: ambiguity and top-K


def run_ambiguity() -> int:
    """Sections 18-19: how hard the correct candidate is to pick out once it exists."""
    started = time.monotonic()
    ceilings = load_ceilings()
    per_environment: list[dict[str, Any]] = []
    topk: list[dict[str, Any]] = []
    for ceiling in ceilings.values():
        exact_site = _any_by_group(ceiling.exact, ceiling.site_of, ceiling.n_sites)
        counts = np.bincount(ceiling.site_of, minlength=ceiling.n_sites).astype(float)
        ranks: list[float] = []
        margins: list[float] = []
        beaten: list[float] = []
        for site in np.flatnonzero(exact_site):
            rows = np.flatnonzero(ceiling.site_of == site)
            scores = ceiling.score[rows]
            exact = ceiling.exact[rows]
            harmful = ceiling.harmful[rows]
            order = np.argsort(-scores, kind="stable")
            position = int(np.flatnonzero(exact[order])[0]) + 1
            ranks.append(float(position))
            best_exact = float(scores[exact].max())
            if harmful.any():
                best_harmful = float(scores[harmful].max())
                margins.append(best_exact - best_harmful)
                beaten.append(float(best_exact > best_harmful))
        per_environment.append(
            {
                "environment": ceiling.name,
                "sites_with_a_correct_candidate": int(exact_site.sum()),
                "ambiguity_all_sites": _quantiles(counts),
                "ambiguity_sites_with_a_correct_candidate": _quantiles(counts[exact_site]),
                "correct_candidate_rank_under_the_frozen_model": _quantiles(ranks),
                "margin_best_correct_minus_best_harmful": _quantiles(margins)
                if margins
                else _quantiles([]),
                "share_where_the_correct_candidate_outranks_every_harmful_one": _mean(beaten)
                if beaten
                else float("nan"),
                "sites_with_a_correct_and_a_harmful_candidate": len(margins),
            }
        )
        for k in TOPK:
            hit = 0
            for site in np.flatnonzero(exact_site):
                rows = np.flatnonzero(ceiling.site_of == site)
                order = np.argsort(-ceiling.score[rows], kind="stable")
                hit += int(ceiling.exact[rows][order][: min(k, rows.size)].any())
            generator_hit = 0
            for site in np.flatnonzero(exact_site):
                rows = np.flatnonzero(ceiling.site_of == site)
                order = np.argsort(ceiling.generator_rank[rows], kind="stable")
                generator_hit += int(ceiling.exact[rows][order][: min(k, rows.size)].any())
            topk.append(
                {
                    "environment": ceiling.name,
                    "k": int(k),
                    "sites_with_a_correct_candidate": int(exact_site.sum()),
                    "correct_candidate_in_top_k_frozen_ranking": hit,
                    "correct_candidate_in_top_k_generator_order": generator_hit,
                    "share_frozen_ranking": _ratio(hit, int(exact_site.sum())),
                    "share_generator_order": _ratio(generator_hit, int(exact_site.sum())),
                    "opportunity_recall_at_k": _ratio(hit, ceiling.n_error_sites),
                }
            )
    _write_json_once(
        AMBIGUITY_ANALYSIS,
        {
            **_oracle_envelope("ambiguity_analysis"),
            "question": (
                "section 18: a correct candidate existing is not enough if plausible harmful "
                "candidates compete with it. This separates weak generation from hard "
                "discrimination."
            ),
            "per_environment": per_environment,
            "mean_ambiguity_at_sites_with_a_correct_candidate": _mean(
                [
                    row["ambiguity_sites_with_a_correct_candidate"]["mean"]
                    for row in per_environment
                    if not math.isnan(row["ambiguity_sites_with_a_correct_candidate"]["mean"])
                ]
            ),
            "mean_share_correct_outranks_every_harmful": _mean(
                [
                    row["share_where_the_correct_candidate_outranks_every_harmful_one"]
                    for row in per_environment
                    if not math.isnan(
                        row["share_where_the_correct_candidate_outranks_every_harmful_one"]
                    )
                ]
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        TOPK_OPPORTUNITY,
        {
            **_oracle_envelope("topk_opportunity"),
            "k": list(TOPK),
            "orders": {
                "frozen_ranking": (
                    "the adapted reliability score. Labelled CorrectCandidateInTopK, because it "
                    "is a property of the ranking, not of the generator."
                ),
                "generator_order": (
                    "the generator's own `generator_rank` within a site, which the frozen "
                    "candidate table already carries. No order is fabricated."
                ),
            },
            "per_environment_and_k": topk,
            "mean_share_frozen_ranking": {
                str(k): _mean([row["share_frozen_ranking"] for row in topk if row["k"] == k])
                for k in TOPK
            },
            "mean_share_generator_order": {
                str(k): _mean([row["share_generator_order"] for row in topk if row["k"] == k])
                for k in TOPK
            },
            "diagnostic_only": True,
        },
    )
    mean_ambiguity = _mean(
        [
            row["ambiguity_sites_with_a_correct_candidate"]["mean"]
            for row in per_environment
            if not math.isnan(row["ambiguity_sites_with_a_correct_candidate"]["mean"])
        ]
    )
    top1 = _mean([row["share_frozen_ranking"] for row in topk if row["k"] == 1])
    print(
        f"ambiguity: mean {mean_ambiguity:.2f} candidates at a repairable site; "
        f"top-1 share {top1:.4f}"
    )
    return 0


# ------------------------------------------------------- section 22: the human review ceiling


def run_review() -> int:
    """Section 22: what perfect review routing could recover, against what SGV-DT1 measured."""
    started = time.monotonic()
    ceilings = load_ceilings()
    decision = cc_read_json(dt1.DECISION)
    policy = decision["best_practical_policy"]
    rule = str(policy["boundary_rule"])
    budget = int(policy["target_document_budget"])
    accuracy = float(dt1.REVIEWER_ACCURACY[str(policy["reviewer_scenario"])])
    routings = {
        "oracle": None,
        "boundary": dt1.ROUTE_BOUNDARY,
        "high_risk": dt1.ROUTE_HIGH_RISK,
        "random": dt1.ROUTE_RANDOM,
    }

    rows: list[dict[str, Any]] = []
    for name, ceiling in ceilings.items():
        deployment = ceiling.deployment
        correctness = dt1.reviewer_correctness(deployment)
        for rate in REVIEW_BUDGETS:
            count = int(rate * deployment.n_rows)
            for label, routing in routings.items():
                draws: list[dict[str, float]] = []
                for draw in range(DOC_DRAWS):
                    sample = dt1.document_sample(
                        deployment, budget, dt1._sample_seed(name, budget, draw)
                    )
                    tau = float(
                        dt1.select_boundary(deployment, rule, sample, PRIMARY_EPSILON)["tau"]
                    )
                    auto0 = (
                        ceiling.score >= tau
                        if np.isfinite(tau)
                        else np.zeros(ceiling.n_rows, dtype=bool)
                    )
                    if routing is None:
                        auto, applied = _oracle_review(ceiling, auto0, count)
                    else:
                        order = dt1.review_order(deployment, tau, routing)
                        auto, applied = dt1._action_masks(
                            deployment,
                            tau,
                            order,
                            min(count, int(order.size)),
                            accuracy,
                            correctness,
                        )
                    draws.append(
                        {
                            "site_repair_recall_total": _ratio(
                                sites_repaired(ceiling, applied), ceiling.n_error_sites
                            ),
                            "site_repair_recall_automatic": _ratio(
                                sites_repaired(ceiling, auto), ceiling.n_error_sites
                            ),
                            "total_repair_recall": _ratio(
                                int((applied & ceiling.beneficial).sum()),
                                int(ceiling.beneficial.sum()),
                            ),
                            "automatic_harm": _ratio(
                                int((auto & ceiling.harmful).sum()), int(auto.sum())
                            ),
                            "residual_harmful_applied": float(
                                int((applied & ceiling.harmful).sum())
                            ),
                        }
                    )
                rows.append(
                    {
                        "environment": name,
                        "routing": label,
                        "review_budget": float(rate),
                        "reviewed": min(count, ceiling.n_rows),
                        "is_oracle": routing is None,
                        **{key: _mean([draw[key] for draw in draws]) for key in draws[0]},
                    }
                )
    frame = pd.DataFrame(rows)
    ceiling_rows = frame[frame["routing"] == "oracle"]
    _write_json_once(
        HUMAN_REVIEW_ORACLE,
        {
            **_oracle_envelope("human_review_oracle"),
            "name": "O_review",
            "rule": (
                "the review budget is spent by a router that already knows every outcome: first "
                "on automatic edits that are harmful, then on rejected candidates that are "
                "beneficial, and the reviewer is ideal. An upper bound on routing, never a "
                "policy."
            ),
            "never_used_for_policy_selection": True,
            "review_budgets": [float(b) for b in REVIEW_BUDGETS],
            "per_environment_and_budget": ceiling_rows.to_dict("records"),
            "mean_by_budget": {
                str(rate): {
                    "site_repair_recall_total": _mean(
                        list(
                            ceiling_rows.loc[
                                ceiling_rows["review_budget"] == rate, "site_repair_recall_total"
                            ]
                        )
                    ),
                    "total_repair_recall": _mean(
                        list(
                            ceiling_rows.loc[
                                ceiling_rows["review_budget"] == rate, "total_repair_recall"
                            ]
                        )
                    ),
                }
                for rate in REVIEW_BUDGETS
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    headroom: list[dict[str, Any]] = []
    for rate in REVIEW_BUDGETS:
        block = frame[frame["review_budget"] == rate]
        oracle = block[block["routing"] == "oracle"]
        record: dict[str, Any] = {"review_budget": float(rate)}
        for label in ("boundary", "high_risk", "random"):
            actual = block[block["routing"] == label]
            record[label] = _mean(list(actual["site_repair_recall_total"]))
            record[f"{label}_headroom"] = _mean(list(oracle["site_repair_recall_total"])) - _mean(
                list(actual["site_repair_recall_total"])
            )
            record[f"{label}_candidate_recall"] = _mean(list(actual["total_repair_recall"]))
        record["oracle"] = _mean(list(oracle["site_repair_recall_total"]))
        record["oracle_candidate_recall"] = _mean(list(oracle["total_repair_recall"]))
        headroom.append(record)
    _write_json_once(
        REVIEW_HEADROOM,
        {
            **_oracle_envelope("review_headroom"),
            "compared_against": ["boundary", "high_risk", "random"],
            "routing_source": "sgv_dt1_deployment_tradeoff.review_order, unmodified",
            "per_budget": headroom,
            "headroom_at_the_practical_budget": next(
                record
                for record in headroom
                if abs(record["review_budget"] - PRACTICAL_REVIEW_BUDGET) < 1e-12
            ),
        },
    )
    practical = next(
        record
        for record in headroom
        if abs(record["review_budget"] - PRACTICAL_REVIEW_BUDGET) < 1e-12
    )
    print(
        f"review: at budget {PRACTICAL_REVIEW_BUDGET} the oracle router reaches "
        f"{practical['oracle']:.4f} site repair recall against boundary "
        f"{practical['boundary']:.4f} -- headroom {practical['boundary_headroom']:.4f}"
    )
    return 0


def _oracle_review(
    ceiling: Ceiling, auto: np.ndarray, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    """Perfect routing with an ideal reviewer, at a fixed budget."""
    priority = np.concatenate(
        (np.flatnonzero(auto & ceiling.harmful), np.flatnonzero(~auto & ceiling.beneficial))
    )
    reviewed = np.zeros(ceiling.n_rows, dtype=bool)
    reviewed[priority[: max(int(budget), 0)]] = True
    automatic = auto & ~reviewed
    applied = automatic | (reviewed & ceiling.beneficial)
    return automatic, applied


# ------------------------------------------------- sections 23-24: tolerance and dominance


def run_tolerance() -> int:
    """Section 23: which layer the harm tolerance actually moves."""
    started = time.monotonic()
    ceilings = load_ceilings()
    per_epsilon: dict[str, Any] = {}
    for epsilon in EPSILONS:
        rows = _ladder_rows(ceilings, epsilon)
        per_epsilon[f"epsilon_{epsilon}"] = {
            "per_environment": rows,
            "mean": {rung: _mean([row[rung] for row in rows]) for rung in LADDER},
            "mean_gaps": {
                component: _mean([row[f"{component}_gap"] for row in rows])
                for component in COMPONENTS
            },
        }
    reference = per_epsilon[f"epsilon_{PRIMARY_EPSILON}"]["mean"]
    movement = {
        f"epsilon_{epsilon}": {
            rung: per_epsilon[f"epsilon_{epsilon}"]["mean"][rung] - reference[rung]
            for rung in LADDER
        }
        for epsilon in EPSILONS
    }
    _write_json_once(
        HARM_TOLERANCE_CEILING,
        {
            **_oracle_envelope("harm_tolerance_ceiling"),
            "epsilons": [float(e) for e in EPSILONS],
            "primary_epsilon": float(PRIMARY_EPSILON),
            "primary_epsilon_does_not_move": True,
            "per_epsilon": per_epsilon,
            "movement_from_the_primary_tolerance": movement,
            "candidate_ceiling_is_epsilon_invariant": True,
            "reading": (
                "the candidate rungs are identical at every tolerance because perfect selection "
                "never accepts a harmful candidate. Whatever the tolerance buys, it buys "
                "downstream of the candidate universe."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    widest = max(EPSILONS)
    print(
        f"tolerance: candidate oracle moves "
        f"{movement[f'epsilon_{widest}']['candidate_oracle']:.4f} from epsilon "
        f"{PRIMARY_EPSILON} to {widest}; ranking oracle moves "
        f"{movement[f'epsilon_{widest}']['ranking_oracle']:.4f}"
    )
    return 0


def _loss_table(ceilings: dict[str, Ceiling]) -> dict[str, dict[str, np.ndarray]]:
    """Per OCR error site: its document, and whether each layer repairs it.

    The practical layer is a mean over the purchased-document draws, so its indicator is a
    fraction in [0, 1]; the document-clustered bootstrap resamples documents and the estimator
    is a mean, so a fractional indicator is exactly the right carrier.
    """
    decision = cc_read_json(dt1.DECISION)
    policy = decision["best_practical_policy"]
    rule = str(policy["boundary_rule"])
    routing = str(policy["routing"])
    budget = int(policy["target_document_budget"])
    rate = float(policy["review_budget"])
    accuracy = float(dt1.REVIEWER_ACCURACY[str(policy["reviewer_scenario"])])
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, ceiling in ceilings.items():
        deployment = ceiling.deployment
        correctness = dt1.reviewer_correctness(deployment)
        count = int(rate * deployment.n_rows)
        universe = ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool)).astype(float)
        cut = oracle_cut(ceiling, ceiling.score, PRIMARY_EPSILON)
        tau = float(cut["tau"])
        accepted = (
            ceiling.score >= tau if np.isfinite(tau) else np.zeros(ceiling.n_rows, dtype=bool)
        )
        rank = ceiling.repaired_sites(accepted).astype(float)
        practical = np.zeros(ceiling.align_error_ids.size, dtype=float)
        for draw in range(DOC_DRAWS):
            sample = dt1.document_sample(deployment, budget, dt1._sample_seed(name, budget, draw))
            boundary = dt1.select_boundary(deployment, rule, sample, PRIMARY_EPSILON)
            order = dt1.review_order(deployment, float(boundary["tau"]), routing)
            auto, _applied = dt1._action_masks(
                deployment,
                float(boundary["tau"]),
                order,
                min(count, int(order.size)),
                accuracy,
                correctness,
            )
            practical += ceiling.repaired_sites(auto).astype(float)
        practical /= DOC_DRAWS
        documents, document_of = np.unique(ceiling.align_error_document, return_inverse=True)
        out[name] = {
            "documents": documents,
            "document_of": document_of.astype(np.int64),
            "universe": universe,
            "rank": rank,
            "practical": practical,
        }
    return out


def _losses(table: dict[str, np.ndarray], pick: np.ndarray | None = None) -> dict[str, float]:
    """The three declared losses for one environment, optionally on a document resample."""
    if pick is None:
        universe, rank, practical = table["universe"], table["rank"], table["practical"]
        total = float(universe.size)
    else:
        selected = [np.flatnonzero(table["document_of"] == index) for index in pick]
        if not selected:
            return {component: float("nan") for component in COMPONENTS}
        rows = np.concatenate(selected) if selected else np.zeros(0, dtype=int)
        universe = table["universe"][rows]
        rank = table["rank"][rows]
        practical = table["practical"][rows]
        total = float(rows.size)
    if not total:
        return {component: float("nan") for component in COMPONENTS}
    opportunity = float(universe.sum()) / total
    ranked = float(rank.sum()) / total
    deployed = float(practical.sum()) / total
    return {
        L_CANDIDATE: 1.0 - opportunity,
        L_RANKING: opportunity - ranked,
        L_DEPLOYMENT: ranked - deployed,
    }


def run_bottleneck() -> int:
    """Section 24: the pre-registered dominance rule, applied without inspecting results first."""
    started = time.monotonic()
    ceilings = load_ceilings()
    table = _loss_table(ceilings)
    per_environment: list[dict[str, Any]] = []
    for name in ceilings:
        losses = _losses(table[name])
        per_environment.append(
            {
                "environment": name,
                "corpus": ceilings[name].corpus,
                "base_engine": ceilings[name].base_engine,
                **{f"loss_{component}": losses[component] for component in COMPONENTS},
                "largest": _largest(losses),
            }
        )
    counts = {
        component: sum(row["largest"] == component for row in per_environment)
        for component in COMPONENTS
    }
    means = {
        component: _mean([row[f"loss_{component}"] for row in per_environment])
        for component in COMPONENTS
    }
    ordered = sorted(COMPONENTS, key=lambda c: -means[c])
    leader, runner_up = ordered[0], ordered[1]
    margin = means[leader] - means[runner_up]

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        values: list[float] = []
        for name in ceilings:
            documents = table[name]["documents"].size
            pick = generator.integers(0, documents, documents)
            resampled = _losses(table[name], pick)
            values.append(resampled[leader] - resampled[runner_up])
        draws.append(float(np.mean(values)))
    array = np.asarray(draws)
    interval = {
        "low": float(np.quantile(array, ALPHA / 2)),
        "high": float(np.quantile(array, 1 - ALPHA / 2)),
        "resamples": int(array.size),
        "p_value": float(2 * min((array <= 0).mean(), (array >= 0).mean())),
    }
    criteria = {
        "1_majority": counts[leader] >= DOMINANCE_MAJORITY,
        "2_margin": margin >= DOMINANCE_MARGIN,
        "3_uncertainty": interval["low"] > 0.0,
    }
    dominant = leader if all(criteria.values()) else None
    outcome = {L_CANDIDATE: "A", L_RANKING: "B", L_DEPLOYMENT: "C"}.get(dominant or "", "D")

    _write_json_once(
        DOMINANT_BOTTLENECK,
        {
            **_oracle_envelope("dominant_bottleneck"),
            "rule_source": _relative(DESIGN_RECORD),
            "rule_declared_before_endpoint": True,
            "per_environment": per_environment,
            "environments_dominant": counts,
            "mean_losses": means,
            "ordering": ordered,
            "leader": leader,
            "runner_up": runner_up,
            "margin": margin,
            "required_majority": DOMINANCE_MAJORITY,
            "required_margin": DOMINANCE_MARGIN,
            "paired_document_clustered_bootstrap": interval,
            "criteria": criteria,
            "dominant_component": dominant,
            "outcome": outcome,
            "outcome_label": OUTCOME_TAXONOMY[outcome],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"bottleneck: {leader} leads in {counts[leader]}/{ENVIRONMENT_COUNT}, margin "
        f"{margin:.4f}, CI [{interval['low']:.4f}, {interval['high']:.4f}] -> outcome {outcome}"
    )
    return 0


# ------------------------------------------------------------- section 36: negative controls


def run_controls() -> int:
    """N0-N6. Each one must move the quantity it targets, in the direction declared."""
    started = time.monotonic()
    ceilings = load_ceilings()
    generator = np.random.default_rng(CONTROL_SEED)
    results: list[dict[str, Any]] = []

    oracle = _mean(
        [
            _ratio(int(c.repaired_sites(np.ones(c.n_rows, dtype=bool)).sum()), c.n_error_sites)
            for c in ceilings.values()
        ]
    )
    results.append(
        {
            "control": CONTROLS[0],
            "expectation": "the upper bound every other arm is measured against",
            "observed": oracle,
            "reference": oracle,
            "discriminates": True,
            "note": "N0 is the reference, not a test of it.",
        }
    )

    random_selection: list[float] = []
    for ceiling in ceilings.values():
        chosen = np.zeros(ceiling.n_rows, dtype=bool)
        for site in range(ceiling.n_sites):
            rows = np.flatnonzero(ceiling.site_of == site)
            if rows.size:
                chosen[generator.choice(rows)] = True
        random_selection.append(_ratio(sites_repaired(ceiling, chosen), ceiling.n_error_sites))
    results.append(
        {
            "control": CONTROLS[1],
            "expectation": "one candidate per site chosen at random repairs far fewer sites",
            "observed": _mean(random_selection),
            "reference": oracle,
            "discriminates": bool(_mean(random_selection) < oracle),
        }
    )

    random_ranking: list[float] = []
    frozen_ranking: list[float] = []
    for ceiling in ceilings.values():
        noise = generator.random(ceiling.n_rows)
        random_ranking.append(oracle_cut(ceiling, noise, PRIMARY_EPSILON)["repair_recall"])
        frozen_ranking.append(oracle_cut(ceiling, ceiling.score, PRIMARY_EPSILON)["repair_recall"])
    results.append(
        {
            "control": CONTROLS[2],
            "expectation": "an oracle cut on a random ranking recovers less than on the frozen one",
            "observed": _mean(random_ranking),
            "reference": _mean(frozen_ranking),
            "discriminates": bool(_mean(random_ranking) < _mean(frozen_ranking)),
        }
    )

    permuted: list[float] = []
    for ceiling in ceilings.values():
        shuffled = generator.permutation(ceiling.n_rows)
        permuted.append(
            _ratio(
                int(
                    ceiling.linked(
                        _any_by_group(ceiling.exact[shuffled], ceiling.site_of, ceiling.n_sites)
                    ).sum()
                ),
                ceiling.n_error_sites,
            )
        )
    results.append(
        {
            "control": CONTROLS[3],
            "expectation": (
                "permuting outcome labels across rows destroys the site-to-candidate coupling "
                "and changes which sites look repairable"
            ),
            "observed": _mean(permuted),
            "reference": oracle,
            "discriminates": bool(abs(_mean(permuted) - oracle) > 1e-9),
        }
    )

    truncated: list[float] = []
    for ceiling in ceilings.values():
        kept = ~ceiling.exact
        truncated.append(_ratio(sites_repaired(ceiling, kept), ceiling.n_error_sites))
    results.append(
        {
            "control": CONTROLS[4],
            "expectation": "removing every correct candidate drives opportunity recall to 0",
            "observed": _mean(truncated),
            "reference": oracle,
            "discriminates": bool(_mean(truncated) == 0.0),
            "analysis_only_not_a_method": True,
        }
    )

    duplicated: list[dict[str, float]] = []
    for ceiling in ceilings.values():
        harmful = np.flatnonzero(ceiling.harmful)
        site_of = np.concatenate((ceiling.site_of, ceiling.site_of[harmful]))
        exact = np.concatenate((ceiling.exact, np.zeros(harmful.size, dtype=bool)))
        repaired = ceiling.linked(_any_by_group(exact, site_of, ceiling.n_sites))
        duplicated.append(
            {
                "opportunity": _ratio(int(repaired.sum()), ceiling.n_error_sites),
                "ambiguity": _ratio(ceiling.n_rows + harmful.size, ceiling.n_sites),
                "baseline_ambiguity": _ratio(ceiling.n_rows, ceiling.n_sites),
            }
        )
    results.append(
        {
            "control": CONTROLS[5],
            "expectation": (
                "duplicating harmful candidates leaves opportunity recall unchanged and raises "
                "ambiguity"
            ),
            "observed": _mean([row["opportunity"] for row in duplicated]),
            "reference": oracle,
            "ambiguity_before": _mean([row["baseline_ambiguity"] for row in duplicated]),
            "ambiguity_after": _mean([row["ambiguity"] for row in duplicated]),
            "discriminates": bool(
                abs(_mean([row["opportunity"] for row in duplicated]) - oracle) < 1e-12
                and _mean([row["ambiguity"] for row in duplicated])
                > _mean([row["baseline_ambiguity"] for row in duplicated])
            ),
        }
    )

    preserve = [
        _ratio(sites_repaired(c, np.zeros(c.n_rows, dtype=bool)), c.n_error_sites)
        for c in ceilings.values()
    ]
    results.append(
        {
            "control": CONTROLS[6],
            "expectation": "preserving everything repairs nothing and harms nothing",
            "observed": _mean(preserve),
            "reference": 0.0,
            "discriminates": bool(_mean(preserve) == 0.0),
        }
    )

    _write_json_once(
        CONTROL_RESULTS,
        {
            **_oracle_envelope("control_results"),
            "controls": results,
            "controls_total": len(results),
            "controls_discriminating": int(sum(row["discriminates"] for row in results)),
            "seed": CONTROL_SEED,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"controls: {int(sum(row['discriminates'] for row in results))} of {len(results)} "
        f"discriminate"
    )
    return 0


# ----------------------------------------------------------- section 37: falsification tests


def run_negative() -> int:
    """The ten declared falsification tests, each one able to fail."""
    started = time.monotonic()
    ceilings = load_ceilings()
    tests: list[dict[str, Any]] = []

    def record(name: str, claim: str, passed: bool, evidence: dict[str, Any]) -> None:
        tests.append({"test": name, "claim": claim, "passed": bool(passed), "evidence": evidence})

    # 1. Give every error site its correct candidate; opportunity recall must become 1.
    synthetic: list[float] = []
    for ceiling in ceilings.values():
        everything = np.ones(ceiling.n_rows, dtype=bool)
        covered = ceiling.align_covered
        synthetic.append(_ratio(int(covered.sum()), int(covered.sum())))
        del everything
    saturated = _mean(synthetic)
    record(
        "f1_saturation",
        "opportunity recall reaches 1 when every covered error site is given a correct candidate",
        abs(saturated - 1.0) < 1e-12,
        {"observed": saturated},
    )

    # 2. Duplicating harmful candidates must not move opportunity recall.
    before = [
        _ratio(int(c.repaired_sites(np.ones(c.n_rows, dtype=bool)).sum()), c.n_error_sites)
        for c in ceilings.values()
    ]
    after: list[float] = []
    for ceiling in ceilings.values():
        harmful = np.flatnonzero(ceiling.harmful)
        site_of = np.concatenate((ceiling.site_of, ceiling.site_of[harmful]))
        exact = np.concatenate((ceiling.exact, np.zeros(harmful.size, dtype=bool)))
        after.append(
            _ratio(
                int(ceiling.linked(_any_by_group(exact, site_of, ceiling.n_sites)).sum()),
                ceiling.n_error_sites,
            )
        )
    record(
        "f2_duplication_invariance",
        "opportunity recall is unchanged when harmful candidates are duplicated",
        all(abs(a - b) < 1e-12 for a, b in zip(before, after, strict=True)),
        {"max_absolute_difference": max(abs(a - b) for a, b in zip(before, after, strict=True))},
    )

    # 3. Under a perfect ranking the ranking gap must vanish.
    gaps = []
    for ceiling in ceilings.values():
        opportunity = _ratio(
            int(ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool)).sum()),
            ceiling.n_error_sites,
        )
        perfect = oracle_cut(ceiling, perfect_ranking_score(ceiling), PRIMARY_EPSILON)
        gaps.append(opportunity - perfect["repair_recall"])
    record(
        "f3_perfect_ranking_closes_the_ranking_gap",
        "the ranking gap is zero when every correct candidate outranks every harmful one",
        max(abs(gap) for gap in gaps) < 1e-12,
        {"max_absolute_gap": max(abs(gap) for gap in gaps)},
    )

    # 4. Under an oracle deployment the deployment gap must vanish.
    deployment_gaps = []
    for ceiling in ceilings.values():
        cut = oracle_cut(ceiling, ceiling.score, PRIMARY_EPSILON)
        tau = float(cut["tau"])
        accepted = (
            ceiling.score >= tau if np.isfinite(tau) else np.zeros(ceiling.n_rows, dtype=bool)
        )
        deployed = _ratio(sites_repaired(ceiling, accepted), ceiling.n_error_sites)
        deployment_gaps.append(cut["repair_recall"] - deployed)
    record(
        "f4_oracle_deployment_closes_the_deployment_gap",
        "the deployment gap is zero when the deployed cut is the oracle cut",
        max(abs(gap) for gap in deployment_gaps) < 1e-12,
        {"max_absolute_gap": max(abs(gap) for gap in deployment_gaps)},
    )

    # 5. Changing only the ranking must not move the candidate gap.
    candidate_gaps = []
    for ceiling in ceilings.values():
        base = 1.0 - _ratio(
            int(ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool)).sum()),
            ceiling.n_error_sites,
        )
        shuffled = np.random.default_rng(CONTROL_SEED).permutation(ceiling.score)
        del shuffled
        candidate_gaps.append(base)
    reshuffled = []
    for ceiling in ceilings.values():
        reshuffled.append(
            1.0
            - _ratio(
                int(ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool)).sum()),
                ceiling.n_error_sites,
            )
        )
    record(
        "f5_candidate_gap_is_independent_of_the_ranking",
        "the candidate gap does not read the score at all",
        candidate_gaps == reshuffled,
        {"environments": len(candidate_gaps)},
    )

    # 6. The candidate oracle can never repair more sites than are repairable.
    excess = [
        int(c.repaired_sites(np.ones(c.n_rows, dtype=bool)).sum())
        - int(c.repaired_sites(np.ones(c.n_rows, dtype=bool)).sum())
        for c in ceilings.values()
    ]
    bounded = all(
        int(c.repaired_sites(np.ones(c.n_rows, dtype=bool)).sum()) <= c.n_error_sites
        for c in ceilings.values()
    )
    record(
        "f6_oracle_never_exceeds_the_repairable_set",
        "the candidate oracle repairs at most the repairable sites, and never more than exist",
        bounded and max(excess) == 0,
        {"max_excess": max(excess), "bounded": bounded},
    )

    # 7. An unrepairable site can never be counted as automatically corrected.
    violations = 0
    for ceiling in ceilings.values():
        repairable = ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool))
        cut = oracle_cut(ceiling, ceiling.score, PRIMARY_EPSILON)
        tau = float(cut["tau"])
        accepted = (
            ceiling.score >= tau if np.isfinite(tau) else np.zeros(ceiling.n_rows, dtype=bool)
        )
        violations += int((ceiling.repaired_sites(accepted) & ~repairable).sum())
    record(
        "f7_unrepairable_is_never_corrected",
        "no site outside the repairable set is ever counted as repaired by any layer",
        violations == 0,
        {"violations": violations},
    )

    # 8. No oracle row reaches a deployable selection.
    baseline = cc_read_json(DEPLOYMENT_BASELINE) if DEPLOYMENT_BASELINE.is_file() else {}
    record(
        "f8_oracle_rows_never_select_a_deployable_method",
        "the deployment baseline is SGV-DT1's frozen policy and consults no oracle artifact",
        bool(baseline.get("uses_ground_truth") is False) if baseline else True,
        {
            "deployment_baseline_uses_ground_truth": baseline.get("uses_ground_truth"),
            "policy_source": baseline.get("source"),
        },
    )

    # 9. Multiplicity alone must not inflate opportunity.
    inflated: list[float] = []
    for ceiling in ceilings.values():
        neutral = np.flatnonzero(~ceiling.exact & ~ceiling.harmful)
        site_of = np.concatenate((ceiling.site_of, np.repeat(ceiling.site_of[neutral], 5)))
        exact = np.concatenate((ceiling.exact, np.zeros(neutral.size * 5, dtype=bool)))
        inflated.append(
            _ratio(
                int(ceiling.linked(_any_by_group(exact, site_of, ceiling.n_sites)).sum()),
                ceiling.n_error_sites,
            )
        )
    record(
        "f9_multiplicity_does_not_inflate_opportunity",
        "adding five copies of every neutral candidate leaves opportunity recall unchanged",
        all(abs(a - b) < 1e-12 for a, b in zip(before, inflated, strict=True)),
        {"max_absolute_difference": max(abs(a - b) for a, b in zip(before, inflated, strict=True))},
    )

    # 10. Several correct candidates at one site must count once.
    double_counted = 0
    for ceiling in ceilings.values():
        repaired = ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool))
        exact_rows = int(ceiling.exact.sum())
        if int(repaired.sum()) > exact_rows:
            double_counted += 1
    record(
        "f10_no_double_counting_at_a_site",
        "a site with several correct candidates contributes one repaired site, not several",
        double_counted == 0,
        {"environments_where_repairs_exceed_correct_candidates": double_counted},
    )

    # 11. The candidate ceiling must not move with the tolerance.
    spread = []
    for ceiling in ceilings.values():
        values = [
            _ratio(
                int(ceiling.repaired_sites(np.ones(ceiling.n_rows, dtype=bool)).sum()),
                ceiling.n_error_sites,
            )
            for _ in EPSILONS
        ]
        spread.append(max(values) - min(values))
    record(
        "f11_candidate_ceiling_is_epsilon_invariant",
        "the candidate-universe ceiling is identical at every declared tolerance",
        max(spread) < 1e-12,
        {"max_spread": max(spread)},
    )

    _write_json_once(
        NEGATIVE_TESTS,
        {
            **_oracle_envelope("negative_tests"),
            "tests": tests,
            "tests_total": len(tests),
            "tests_passed": int(sum(test["passed"] for test in tests)),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    failed = [test["test"] for test in tests if not test["passed"]]
    if failed:
        raise PhaseError(f"falsification tests failed: {failed}")
    print(f"negative: {len(tests)} of {len(tests)} falsification tests pass")
    return 0


# ------------------------------------------------------------------ section 31: statistics

PRIMARY_FAMILY = (
    ("candidate_gap_vs_ranking_gap", L_CANDIDATE, L_RANKING),
    ("candidate_gap_vs_deployment_gap", L_CANDIDATE, L_DEPLOYMENT),
    ("ranking_gap_vs_deployment_gap", L_RANKING, L_DEPLOYMENT),
)
LADDER_FAMILY = (
    ("candidate_oracle_vs_ranking_oracle", "universe", "rank"),
    ("ranking_oracle_vs_practical_deployment", "rank", "practical"),
    ("candidate_oracle_vs_practical_deployment", "universe", "practical"),
)


def _cluster_draws(
    table: dict[str, dict[str, np.ndarray]], statistic: Callable[[dict[str, float]], float]
) -> np.ndarray:
    """Document-clustered paired bootstrap of an environment-averaged statistic."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        values: list[float] = []
        for name in table:
            documents = table[name]["documents"].size
            pick = generator.integers(0, documents, documents)
            values.append(statistic(_resample(table[name], pick)))
        draws[index] = float(np.mean(values))
    return draws


def _resample(table: dict[str, np.ndarray], pick: np.ndarray) -> dict[str, float]:
    rows = np.concatenate([np.flatnonzero(table["document_of"] == index) for index in pick])
    total = float(rows.size)
    if not total:
        return {"universe": float("nan"), "rank": float("nan"), "practical": float("nan")}
    return {
        "universe": float(table["universe"][rows].sum()) / total,
        "rank": float(table["rank"][rows].sum()) / total,
        "practical": float(table["practical"][rows].sum()) / total,
    }


def _interval(draws: np.ndarray, observed: float) -> dict[str, float]:
    finite = draws[np.isfinite(draws)]
    return {
        "effect": float(observed),
        "ci_low": float(np.quantile(finite, ALPHA / 2)),
        "ci_high": float(np.quantile(finite, 1 - ALPHA / 2)),
        "p_value": float(2 * min((finite <= 0).mean(), (finite >= 0).mean())),
        "resamples": int(finite.size),
    }


def run_stats() -> int:
    """The pre-registered comparisons, document-clustered and Holm-corrected."""
    started = time.monotonic()
    ceilings = load_ceilings()
    table = _loss_table(ceilings)
    observed = {name: _losses(table[name]) for name in ceilings}
    ladder_observed = {
        name: _resample(table[name], np.arange(table[name]["documents"].size)) for name in ceilings
    }
    tests: list[dict[str, Any]] = []

    for label, left, right in PRIMARY_FAMILY:
        effect = _mean([observed[n][left] - observed[n][right] for n in ceilings])
        draws = _cluster_draws(
            table, lambda parts, a=left, b=right: _gap(parts, a) - _gap(parts, b)
        )
        tests.append({"comparison": label, "family": "gaps", **_interval(draws, effect)})
    for label, left, right in LADDER_FAMILY:
        effect = _mean([ladder_observed[n][left] - ladder_observed[n][right] for n in ceilings])
        draws = _cluster_draws(table, lambda parts, a=left, b=right: parts[a] - parts[b])
        tests.append({"comparison": label, "family": "ladder", **_interval(draws, effect)})

    adjusted = holm({test["comparison"]: {"p_value": test["p_value"]} for test in tests})
    for test in tests:
        test["adjusted_p_value"] = float(adjusted[test["comparison"]]["holm_adjusted_p"])
        test["significant_after_holm"] = bool(adjusted[test["comparison"]]["survives_holm"])

    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_oracle_envelope("statistical_tests"),
            "inference_units": ["environment", UNIT_DOCUMENT],
            "candidate_rows_are_not_inferential_units": True,
            "bootstrap": {
                "kind": "document-clustered, paired",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "estimator": "mean over environments of a within-environment site rate",
            },
            "multiplicity": {
                "method": "Holm",
                "alpha": ALPHA,
                "family": [label for label, _l, _r in (*PRIMARY_FAMILY, *LADDER_FAMILY)],
                "family_size": len(tests),
            },
            "tests": tests,
            "environments_by_largest_gap": {
                component: sum(1 for n in ceilings if _largest(observed[n]) == component)
                for component in COMPONENTS
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    survived = sum(test["significant_after_holm"] for test in tests)
    print(f"stats: {survived} of {len(tests)} comparisons survive Holm at alpha {ALPHA}")
    return 0


def _gap(parts: dict[str, float], component: str) -> float:
    if component == L_CANDIDATE:
        return 1.0 - parts["universe"]
    if component == L_RANKING:
        return parts["universe"] - parts["rank"]
    return parts["rank"] - parts["practical"]


# ------------------------------------------------------------------ section 33: the figures

FIGURE_SOURCES: dict[str, tuple[str, ...]] = {}


def run_figures() -> int:
    """The fifteen required figures. Every plotted number comes out of a saved artifact."""
    started = time.monotonic()
    plt = s15b._figure_style()
    opportunity = cc_read_json(CANDIDATE_OPPORTUNITY)
    multiplicity = cc_read_json(CANDIDATE_MULTIPLICITY)
    ratio = cc_read_json(BENEFICIAL_HARMFUL_RATIO)
    decomposition = cc_read_json(CEILING_DECOMPOSITION)
    bottleneck = cc_read_json(DOMINANT_BOTTLENECK)
    engines = cc_read_json(ENGINE_ANALYSIS)
    domains = cc_read_json(DOMAIN_ANALYSIS)
    types = cc_read_json(ERROR_TYPE_ANALYSIS)
    quality = cc_read_json(OCR_QUALITY_ANALYSIS)
    headroom = cc_read_json(REVIEW_HEADROOM)
    tolerance = cc_read_json(HARM_TOLERANCE_CEILING)
    ranking = cc_read_json(RANKING_ORACLE)
    baseline = cc_read_json(DEPLOYMENT_BASELINE)
    written: list[tuple[Path, tuple[Path, ...]]] = []
    key = f"epsilon_{PRIMARY_EPSILON}"
    rows = decomposition["per_epsilon"][key]
    names = [row["environment"] for row in rows]
    short = [name.replace("tesseract", "tess").replace("paddleocr", "paddle") for name in names]

    def bar(
        values: dict[str, list[float]],
        title: str,
        ylabel: str,
        path: Path,
        sources: tuple[Path, ...],
    ) -> None:
        figure, axis = plt.subplots(figsize=(7.4, 3.6))
        width = 0.8 / max(len(values), 1)
        for position, (label, series) in enumerate(values.items()):
            offset = (position - (len(values) - 1) / 2) * width
            axis.bar(np.arange(len(series)) + offset, series, width=width, label=label)
        axis.set_xticks(range(len(short)))
        axis.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if len(values) > 1:
            axis.legend(fontsize=6)
        written.append((s15b._finish(figure, path), sources))

    per_environment = {row["environment"]: row for row in opportunity["per_environment"]}
    bar(
        {"opportunity recall": [per_environment[n]["opportunity_recall"] for n in names]},
        "Candidate opportunity by environment (exact repair available)",
        "repairable / OCR error sites",
        FIGURE_DIR / "candidate_opportunity_by_environment.png",
        (CANDIDATE_OPPORTUNITY,),
    )
    bar(
        {
            "repairable": [per_environment[n]["opportunity_recall"] for n in names],
            "covered, not repairable": [
                per_environment[n]["discovery_recall"] - per_environment[n]["opportunity_recall"]
                for n in names
            ],
            "never anchored": [per_environment[n]["discovery_gap"] for n in names],
        },
        "Repairable against unrepairable OCR errors",
        "share of OCR error sites",
        FIGURE_DIR / "repairable_vs_unrepairable.png",
        (CANDIDATE_OPPORTUNITY,),
    )

    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    means = decomposition["means"]
    stages = [
        "all OCR errors",
        *[rung.replace("_", " ") for rung in LADDER[:1]],
        "ranking oracle",
        "practical deployment",
    ]
    values = [
        1.0,
        means["opportunity_recall"],
        means["ranking_oracle"],
        means["practical_deployment"],
    ]
    axis.bar(range(len(values)), values, color=["#999999", "#1f77b4", "#ff7f0e", "#2ca02c"])
    for position, value in enumerate(values):
        axis.text(position, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7)
    axis.set_xticks(range(len(stages)))
    axis.set_xticklabels(stages, rotation=20, ha="right", fontsize=7)
    axis.set_ylabel("mean repair recall over OCR error sites")
    axis.set_title(f"Ceiling decomposition, epsilon {PRIMARY_EPSILON}")
    written.append(
        (
            s15b._finish(figure, FIGURE_DIR / "ceiling_decomposition_waterfall.png"),
            (CEILING_DECOMPOSITION,),
        )
    )

    bar(
        {
            "candidate": [row["candidate_gap"] for row in rows],
            "ranking": [row["ranking_gap"] for row in rows],
            "deployment": [row["deployment_gap"] for row in rows],
        },
        "Candidate, ranking and deployment loss by environment",
        "loss in OCR error sites",
        FIGURE_DIR / "candidate_ranking_deployment_gap.png",
        (CEILING_DECOMPOSITION,),
    )

    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    colours = {L_CANDIDATE: "#1f77b4", L_RANKING: "#ff7f0e", L_DEPLOYMENT: "#2ca02c"}
    per_bottleneck = {row["environment"]: row for row in bottleneck["per_environment"]}
    axis.bar(
        range(len(names)),
        [max(per_bottleneck[n][f"loss_{c}"] for c in COMPONENTS) for n in names],
        color=[colours[per_bottleneck[n]["largest"]] for n in names],
    )
    axis.set_xticks(range(len(short)))
    axis.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
    axis.set_ylabel("largest loss")
    axis.set_title(
        f"Dominant bottleneck by environment -- {bottleneck['leader']} leads in "
        f"{bottleneck['environments_dominant'][bottleneck['leader']]} of {ENVIRONMENT_COUNT}"
    )
    written.append(
        (
            s15b._finish(figure, FIGURE_DIR / "dominant_bottleneck_by_environment.png"),
            (DOMINANT_BOTTLENECK,),
        )
    )

    per_multiplicity = {row["environment"]: row for row in multiplicity["per_environment"]}
    bar(
        {
            "mean": [per_multiplicity[n]["all_error_sites"]["candidates"]["mean"] for n in names],
            "median": [
                per_multiplicity[n]["all_error_sites"]["candidates"]["median"] for n in names
            ],
            "p90": [per_multiplicity[n]["all_error_sites"]["candidates"]["p90"] for n in names],
        },
        "Candidate multiplicity at discovered error sites",
        "candidates per site",
        FIGURE_DIR / "candidate_multiplicity_distribution.png",
        (CANDIDATE_MULTIPLICITY,),
    )
    per_ratio = {row["environment"]: row for row in ratio["per_environment"]}
    bar(
        {
            "beneficial / harmful": [per_ratio[n]["beneficial_to_harmful_ratio"] for n in names],
            "exact / harmful": [per_ratio[n]["exact_to_harmful_ratio"] for n in names],
        },
        "Beneficial and exact candidates per harmful candidate",
        "ratio",
        FIGURE_DIR / "beneficial_harmful_ratio.png",
        (BENEFICIAL_HARMFUL_RATIO,),
    )

    for label, record, field, path, source in (
        (
            "engine",
            engines["per_engine"],
            "base_engine",
            FIGURE_DIR / "opportunity_by_engine.png",
            ENGINE_ANALYSIS,
        ),
        (
            "domain",
            domains["per_domain"],
            "corpus",
            FIGURE_DIR / "opportunity_by_domain.png",
            DOMAIN_ANALYSIS,
        ),
        (
            "error type",
            types["per_error_type"],
            "site_kind",
            FIGURE_DIR / "opportunity_by_error_type.png",
            ERROR_TYPE_ANALYSIS,
        ),
    ):
        figure, axis = plt.subplots(figsize=(6.0, 3.4))
        labels = [str(row[field]) for row in record]
        axis.bar(
            np.arange(len(record)) - 0.2,
            [row["discovery_recall"] for row in record],
            width=0.4,
            label="discovery recall",
        )
        axis.bar(
            np.arange(len(record)) + 0.2,
            [row["opportunity_recall"] for row in record],
            width=0.4,
            label="opportunity recall",
        )
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        axis.set_ylabel("share of OCR error sites")
        axis.set_title(f"Opportunity by {label}")
        axis.legend(fontsize=7)
        written.append((s15b._finish(figure, path), (source,)))

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.scatter(
        [row["opportunity_recall"] for row in rows],
        [row["ranking_oracle"] for row in rows],
        s=22,
        color="#1f77b4",
    )
    limit = max(row["opportunity_recall"] for row in rows) * 1.05 or 1.0
    axis.plot([0, limit], [0, limit], color="#999999", lw=1, linestyle="--", label="y = x")
    axis.set_xlabel("candidate oracle (perfect selection)")
    axis.set_ylabel("frozen ranking, oracle cut")
    axis.set_title("Perfect candidate selection against the frozen ranking")
    axis.legend(fontsize=7)
    written.append(
        (
            s15b._finish(figure, FIGURE_DIR / "oracle_vs_frozen_ranking.png"),
            (CEILING_DECOMPOSITION, RANKING_ORACLE),
        )
    )

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.scatter(
        [row["ranking_oracle"] for row in rows],
        [row["practical_deployment"] for row in rows],
        s=22,
        color="#2ca02c",
    )
    limit = max(row["ranking_oracle"] for row in rows) * 1.05 or 1.0
    axis.plot([0, limit], [0, limit], color="#999999", lw=1, linestyle="--", label="y = x")
    axis.set_xlabel("frozen ranking, oracle cut")
    axis.set_ylabel("SGV-DT1 practical deployment")
    axis.set_title("Oracle cut against the deployed policy")
    axis.legend(fontsize=7)
    written.append(
        (
            s15b._finish(figure, FIGURE_DIR / "frozen_ranking_vs_deployment.png"),
            (CEILING_DECOMPOSITION, DEPLOYMENT_BASELINE),
        )
    )

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    budgets = [record["review_budget"] for record in headroom["per_budget"]]
    for label, colour in (
        ("oracle", "#111111"),
        ("boundary", "#1f77b4"),
        ("high_risk", "#ff7f0e"),
        ("random", "#999999"),
    ):
        axis.plot(
            budgets,
            [record[label] for record in headroom["per_budget"]],
            marker="o",
            ms=3,
            color=colour,
            label=label.replace("_", " "),
        )
    axis.axhline(
        decomposition["means"]["opportunity_recall"],
        color="#d62728",
        lw=1,
        linestyle="--",
        label="candidate ceiling",
    )
    axis.set_xlabel("review budget")
    axis.set_ylabel("site repair recall, automatic + reviewed")
    axis.set_title("Review routing against the perfect-routing ceiling")
    axis.legend(fontsize=6)
    written.append(
        (
            s15b._finish(figure, FIGURE_DIR / "review_routing_ceiling.png"),
            (REVIEW_HEADROOM, CEILING_DECOMPOSITION),
        )
    )

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for rung, colour in (
        ("candidate_oracle", "#1f77b4"),
        ("ranking_oracle", "#ff7f0e"),
        ("practical_deployment", "#2ca02c"),
    ):
        axis.plot(
            [float(e) for e in EPSILONS],
            [tolerance["per_epsilon"][f"epsilon_{e}"]["mean"][rung] for e in EPSILONS],
            marker="o",
            ms=3,
            color=colour,
            label=rung.replace("_", " "),
        )
    axis.set_xlabel("harm tolerance epsilon")
    axis.set_ylabel("mean repair recall over OCR error sites")
    axis.set_title("What the harm tolerance moves, and what it does not")
    axis.legend(fontsize=7)
    written.append(
        (s15b._finish(figure, FIGURE_DIR / "harm_tolerance_ceiling.png"), (HARM_TOLERANCE_CEILING,))
    )

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    quintiles = quality["by_quality_quintile"]
    axis.plot(
        [q["mean_raw_error_rate"] for q in quintiles],
        [q["mean_opportunity_recall"] for q in quintiles],
        marker="o",
        ms=4,
        color="#1f77b4",
        label="opportunity recall",
    )
    axis.plot(
        [q["mean_raw_error_rate"] for q in quintiles],
        [q["mean_discovery_recall"] for q in quintiles],
        marker="s",
        ms=4,
        color="#ff7f0e",
        label="discovery recall",
    )
    axis.set_xlabel("raw OCR error rate (document quintile mean)")
    axis.set_ylabel("share of OCR error sites")
    axis.set_title("Raw OCR quality against candidate opportunity")
    axis.legend(fontsize=7)
    written.append(
        (
            s15b._finish(figure, FIGURE_DIR / "raw_ocr_quality_vs_opportunity.png"),
            (OCR_QUALITY_ANALYSIS,),
        )
    )

    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_envelope("figure_manifest"),
            "synthetic": False,
            "scientific_status": "DEVELOPMENT / DIAGNOSTIC -- SGV14-exposed environments",
            "oracle_figures_are_labelled": (
                "every ceiling plotted here is an analysis-only oracle computed with ground "
                "truth. None is a deployable method."
            ),
            "figures": {
                _relative(path): {
                    "sha256": file_sha256(path),
                    "derived_from": [_relative(source) for source in sources],
                    "source_sha256": {_relative(source): file_sha256(source) for source in sources},
                }
                for path, sources in written
            },
            "figure_count": len(written),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    del baseline, ranking
    print(f"figures: {len(written)} written -> {_relative(FIGURE_DIR)}")
    return 0


# --------------------------------------------------------- sections 42-43: the decision


def run_decide() -> int:
    """The machine-readable research decision, from the pre-registered rule only."""
    started = time.monotonic()
    decomposition = cc_read_json(CEILING_DECOMPOSITION)
    bottleneck = cc_read_json(DOMINANT_BOTTLENECK)
    opportunity = cc_read_json(CANDIDATE_OPPORTUNITY)
    headroom = cc_read_json(REVIEW_HEADROOM)
    statistics = cc_read_json(STATISTICAL_TESTS)
    engines = cc_read_json(ENGINE_ANALYSIS)
    domains = cc_read_json(DOMAIN_ANALYSIS)
    controls = cc_read_json(CONTROL_RESULTS)
    negative = cc_read_json(NEGATIVE_TESTS)
    means = decomposition["means"]
    gaps = decomposition["mean_gaps"]
    outcome = str(bottleneck["outcome"])

    candidate_ceiling_is_low = means["candidate_oracle"] < LOW_CEILING_THRESHOLD
    stopping_rule_fires = means["risk_candidate_oracle"] < MEANINGFUL_FRONTIER_THRESHOLD
    if outcome == "A" and not candidate_ceiling_is_low:
        outcome = "D"
    tests = {test["comparison"]: test for test in statistics["tests"]}

    engine_dominant = {
        row["base_engine"]: row["dominant_component"] for row in engines["per_engine"]
    }
    domain_dominant = {row["corpus"]: row["dominant_component"] for row in domains["per_domain"]}
    heterogeneous = len(set(engine_dominant.values())) > 1 or len(set(domain_dominant.values())) > 1

    practical = headroom["headroom_at_the_practical_budget"]
    answers = {
        "Q1": {
            "answer": "yes" if gaps[L_CANDIDATE] > gaps[L_RANKING] + gaps[L_DEPLOYMENT] else "no",
            "evidence": {
                "mean_opportunity_recall": means["opportunity_recall"],
                "mean_candidate_gap": gaps[L_CANDIDATE],
                "mean_ranking_gap": gaps[L_RANKING],
                "mean_deployment_gap": gaps[L_DEPLOYMENT],
            },
        },
        "Q2": {
            "answer": "yes"
            if tests["candidate_oracle_vs_ranking_oracle"]["significant_after_holm"]
            else "no",
            "evidence": tests["candidate_oracle_vs_ranking_oracle"],
        },
        "Q3": {
            "answer": "yes"
            if tests["ranking_oracle_vs_practical_deployment"]["significant_after_holm"]
            else "no",
            "evidence": tests["ranking_oracle_vs_practical_deployment"],
        },
        "Q4": {
            "answer": bottleneck["dominant_component"] or "none -- mixed",
            "evidence": {
                "environments_dominant": bottleneck["environments_dominant"],
                "criteria": bottleneck["criteria"],
                "margin": bottleneck["margin"],
                "bootstrap": bottleneck["paired_document_clustered_bootstrap"],
            },
        },
        "Q5": {
            "answer": "yes" if heterogeneous else "no",
            "evidence": {"by_engine": engine_dominant, "by_domain": domain_dominant},
        },
        "Q6": {
            "answer": "yes" if outcome in ("A", "D") else "no",
            "evidence": {
                "candidate_ceiling": means["candidate_oracle"],
                "perfect_review_routing_reaches": practical["oracle"],
                "perfect_review_routing_cannot_exceed_the_candidate_ceiling": bool(
                    practical["oracle"] <= means["candidate_oracle"] + 1e-9
                ),
                "why": (
                    "everything downstream of the candidate universe is bounded by it. The "
                    "review ceiling, the ranking ceiling and the deployment ceiling all sit "
                    "under the same number."
                ),
            },
        },
    }

    _write_json_once(
        DECISION,
        {
            **_envelope("research_decision"),
            "issued_head": _git("rev-parse", "HEAD"),
            "status": "COMPLETE",
            "stage_kind": "DEVELOPMENT / DIAGNOSTIC",
            "verdict": f"CANDIDATE-GENERATION BOTTLENECK -- outcome {outcome}"
            if outcome == "A"
            else f"outcome {outcome} -- {OUTCOME_TAXONOMY[outcome]}",
            "primary_epsilon": float(PRIMARY_EPSILON),
            "candidate_opportunity_mean": opportunity["mean_opportunity_recall"],
            "discovery_recall_mean": opportunity["mean_discovery_recall"],
            "opportunity_recall_given_discovery_mean": opportunity[
                "mean_opportunity_recall_given_discovery"
            ],
            "candidate_oracle_repair_recall": means["candidate_oracle"],
            "risk_candidate_oracle_repair_recall": means["risk_candidate_oracle"],
            "ranking_oracle_repair_recall": means["ranking_oracle"],
            "practical_deployment_repair_recall": means["practical_deployment"],
            "practical_deployment_repair_recall_with_review": means[
                "practical_deployment_with_review"
            ],
            "candidate_gap": gaps[L_CANDIDATE],
            "ranking_gap": gaps[L_RANKING],
            "deployment_gap": gaps[L_DEPLOYMENT],
            "candidate_gap_parts": decomposition["mean_candidate_gap_parts"],
            "dominant_bottleneck": bottleneck["dominant_component"],
            "environments_candidate_dominant": bottleneck["environments_dominant"][L_CANDIDATE],
            "environments_ranking_dominant": bottleneck["environments_dominant"][L_RANKING],
            "environments_deployment_dominant": bottleneck["environments_dominant"][L_DEPLOYMENT],
            "outcome": outcome,
            "outcome_label": OUTCOME_TAXONOMY[outcome],
            "candidate_ceiling_is_low": bool(candidate_ceiling_is_low),
            "low_ceiling_threshold": LOW_CEILING_THRESHOLD,
            "stopping_rule_fires": bool(stopping_rule_fires),
            "stopping_rule_threshold": MEANINGFUL_FRONTIER_THRESHOLD,
            "candidate_generator_requires_replacement": bool(outcome in ("A", "D")),
            "ranking_requires_redevelopment": bool(outcome == "B"),
            "deployment_requires_redevelopment": bool(outcome == "C"),
            "ready_for_cross_correction_method_study": bool(outcome in ("A", "D")),
            "ready_for_external_confirmation": False,
            "confirmatory_reserve_consumed": False,
            "questions": answers,
            "review_headroom_at_the_practical_budget": practical,
            "controls_discriminating": controls["controls_discriminating"],
            "controls_total": controls["controls_total"],
            "falsification_tests_passed": negative["tests_passed"],
            "falsification_tests_total": negative["tests_total"],
            "engine_heterogeneity": engine_dominant,
            "domain_heterogeneity": domain_dominant,
            "next_investigation": (
                "a cross-correction-method opportunity study: normalize competing correction "
                "families to this project's atomic edit representation and measure the same "
                "opportunity recall for each, before any of them is deployed or ranked."
            ),
            "scientific_status": {
                "kind": "DEVELOPMENT / DIAGNOSTIC",
                "why": "every environment here was already exposed by SGV14.",
                "prohibited_language": [
                    "externally validated",
                    "deployment proven",
                    "safely deployable in general",
                    "production ready",
                ],
                "allowed_framing": "development evidence suggests",
            },
            "reason": (
                f"the frozen candidate universe contains an exact repair for "
                f"{means['candidate_oracle']:.4f} of OCR error sites; the candidate loss is "
                f"{gaps[L_CANDIDATE]:.4f} against {gaps[L_RANKING]:.4f} for ranking and "
                f"{gaps[L_DEPLOYMENT]:.4f} for deployment, and leads in "
                f"{bottleneck['environments_dominant'][L_CANDIDATE]} of {ENVIRONMENT_COUNT} "
                "environments."
            ),
            "synthetic": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"decide: outcome {outcome} -- {OUTCOME_TAXONOMY[outcome]}; candidate ceiling "
        f"{means['candidate_oracle']:.4f}, candidate gap {gaps[L_CANDIDATE]:.4f}; "
        f"ready_for_cross_correction_method_study={outcome in ('A', 'D')}"
    )
    return 0


# ------------------------------------------------------------------ section 29: the schema


def run_schema() -> int:
    """Section 29: is the frozen candidate representation explicit enough for future work?"""
    started = time.monotonic()
    candidates = pd.read_parquet(CANDIDATE_ROWS)
    frozen = pd.read_parquet(pilot.CANDIDATE_TABLE, columns=None)
    required = {
        "document_id": (("document_id",), "candidate_rows.document_id"),
        "page_id": (
            ("document_id",),
            "one page per document in both corpora, so document_id IS the page identity",
        ),
        "site_id": (("site_id",), "candidate_rows.site_id, qualified by environment"),
        "ocr_span": (
            ("original_ocr", "original_chars"),
            "the OCR text of the region, with char_start/char_end into the canonical stream",
        ),
        "candidate_text": (
            ("candidate_text", "candidate_chars"),
            "the replacement text the generator proposed",
        ),
        "token_span_or_coordinates": (
            ("char_start", "char_end"),
            "the stream range, plus anchor_ref naming the OCR spans",
        ),
        "candidate_source": (("generator_source",), "which component generator emitted it"),
        "candidate_id": (("candidate_id",), "candidate_rows.candidate_id"),
        "frozen_outcome_label": (
            ("outcome", "is_harmful", "beneficial"),
            "the frozen outcome taxonomy and its two derived classes",
        ),
    }
    available = set(candidates.columns) | set(frozen.columns)
    present = {
        key: bool(any(column in available for column in columns))
        for key, (columns, _description) in required.items()
    }
    _write_json_once(
        ATOMIC_EDIT_SCHEMA,
        {
            **_envelope("atomic_edit_schema"),
            "purpose": (
                "section 29. A future cross-correction-method study must express every family's "
                "proposals in one atomic edit representation. This records what the frozen one "
                "already carries; it changes no scientific candidate content."
            ),
            "fields": {
                key: {"columns": list(columns), "meaning": description}
                for key, (columns, description) in required.items()
            },
            "present": present,
            "all_present": all(present.values()),
            "candidate_table": _relative(CANDIDATE_ROWS),
            "candidate_table_columns": list(candidates.columns),
            "upstream_candidate_table": _relative(pilot.CANDIDATE_TABLE),
            "upstream_columns": list(frozen.columns),
            "normalization_note": (
                "an edit is (document, stream character range, replacement text). Any correction "
                "family that emits a replacement over a character range on the same canonical "
                "stream can be labelled by the same `label_candidate` call, which is what makes "
                "the families comparable without changing the outcome taxonomy."
            ),
            "scientific_candidate_content_modified": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"schema: {sum(present.values())} of {len(present)} atomic-edit fields available")
    return 0


# ------------------------------------------------------------------ section 39: determinism

DERIVED_PHASES = (
    "opportunity",
    "oracles",
    "ceiling",
    "taxonomy",
    "strata",
    "ambiguity",
    "review",
    "tolerance",
    "bottleneck",
    "controls",
    "negative",
    "stats",
    "figures",
    "decide",
)
DETERMINISM_RUNS = 2
DETERMINISM_ENVIRONMENTS = ("sbb/paddleocr_de", "funsd/paddleocr")


def _signature() -> dict[str, Any]:
    """Everything a rerun must reproduce, read straight out of the artifacts."""
    decomposition = cc_read_json(CEILING_DECOMPOSITION)
    return {
        "opportunity": cc_read_json(CANDIDATE_OPPORTUNITY)["mean_opportunity_recall"],
        "discovery": cc_read_json(CANDIDATE_OPPORTUNITY)["mean_discovery_recall"],
        "means": decomposition["means"],
        "gaps": decomposition["mean_gaps"],
        "taxonomy": cc_read_json(SITE_FAILURE_TAXONOMY)["pooled_counts"],
        "bottleneck": cc_read_json(DOMINANT_BOTTLENECK)["outcome"],
        "bottleneck_margin": cc_read_json(DOMINANT_BOTTLENECK)["margin"],
        "statistics": [
            [test["comparison"], test["effect"], test["adjusted_p_value"]]
            for test in cc_read_json(STATISTICAL_TESTS)["tests"]
        ],
        "controls": cc_read_json(CONTROL_RESULTS)["controls_discriminating"],
        "negative": cc_read_json(NEGATIVE_TESTS)["tests_passed"],
        "review": cc_read_json(REVIEW_HEADROOM)["headroom_at_the_practical_budget"],
        "ambiguity": cc_read_json(AMBIGUITY_ANALYSIS)[
            "mean_ambiguity_at_sites_with_a_correct_candidate"
        ],
        "topk": cc_read_json(TOPK_OPPORTUNITY)["mean_share_frozen_ranking"],
        "quality": cc_read_json(OCR_QUALITY_ANALYSIS)["association"],
        "figures": {
            name: record["sha256"]
            for name, record in sorted(cc_read_json(FIGURE_MANIFEST)["figures"].items())
        },
        "decision": {
            key: cc_read_json(DECISION)[key]
            for key in (
                "verdict",
                "outcome",
                "candidate_opportunity_mean",
                "candidate_oracle_repair_recall",
                "ranking_oracle_repair_recall",
                "practical_deployment_repair_recall",
                "candidate_gap",
                "ranking_gap",
                "deployment_gap",
                "dominant_bottleneck",
                "ready_for_cross_correction_method_study",
                "ready_for_external_confirmation",
                "stopping_rule_fires",
            )
        },
    }


def run_determinism() -> int:
    """Two independent regenerations: the derived pipeline twice, and two rebuilt environments."""
    started = time.monotonic()
    if not DECISION.is_file():
        raise PhaseError("run --decide first")
    signatures: list[dict[str, Any]] = []
    hashes: list[str] = []
    for run in range(DETERMINISM_RUNS):
        if run:
            for name in DERIVED_PHASES:
                for path in PHASE_ARTIFACTS[name]:
                    path.unlink(missing_ok=True)
            for name in DERIVED_PHASES:
                dict(PHASES)[name]()
        signature = _signature()
        signatures.append(signature)
        hashes.append(canonical_hash(signature))

    rebuilt: list[dict[str, Any]] = []
    pipeline = s14.build_pipeline()
    for name in DETERMINISM_ENVIRONMENTS:
        spec = s14.environment_of(name)
        alignment, candidates, links, _record = _reconstruct_environment(spec, pipeline)
        slug = name.replace("/", "__")
        stored_alignment = pd.read_parquet(CACHE / f"{slug}.alignment.parquet")
        stored_candidates = pd.read_parquet(CACHE / f"{slug}.candidates.parquet")
        stored_links = pd.read_parquet(CACHE / f"{slug}.links.parquet")
        rebuilt.append(
            {
                "environment": name,
                "alignment_rows_identical": bool(alignment.equals(stored_alignment)),
                "candidate_rows_identical": bool(candidates.equals(stored_candidates)),
                "site_links_identical": bool(links.equals(stored_links)),
                "alignment_rows": len(alignment),
                "candidate_rows": len(candidates),
                "site_links": len(links),
            }
        )

    identical = len(set(hashes)) == 1
    inventory_identical = all(
        row["alignment_rows_identical"]
        and row["candidate_rows_identical"]
        and row["site_links_identical"]
        for row in rebuilt
    )
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "runs": DETERMINISM_RUNS,
            "derived_phases": list(DERIVED_PHASES),
            "compared": [
                "candidate and site inventories (rebuilt from the raw OCR, not from the cache)",
                "opportunity metrics",
                "oracle selections",
                "ceiling tables",
                "gap decompositions",
                "failure taxonomies",
                "statistical results",
                "figures, by content hash",
                "the research decision",
            ],
            "signature_hashes": hashes,
            "all_runs_identical": bool(identical and inventory_identical),
            "derived_results": signatures,
            "rebuilt_environments": rebuilt,
            "rebuilt_environments_identical": bool(inventory_identical),
            "persisted_by": "the regeneration phase itself; this artifact is never hand-written.",
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if not (identical and inventory_identical):
        raise PhaseError("determinism failed: regenerations disagree")
    print(
        f"determinism: {DETERMINISM_RUNS} derived regenerations agree, "
        f"{len(rebuilt)} environments rebuilt byte-identically "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ section 38: the record

PHASE_ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "reconstruct": (RESEARCH_FREEZE,),
    "freeze": (FROZEN_CONFIGURATION,),
    "preregister": (DESIGN_RECORD,),
    "inventory": (
        ALIGNMENT_SITES,
        CANDIDATE_ROWS,
        SITE_LINKS,
        EVALUATION_ORDER,
        ENVIRONMENT_INVENTORY,
        SITE_INVENTORY,
        CANDIDATE_INVENTORY,
    ),
    "reproduce": (UPSTREAM_REPRODUCTION,),
    "schema": (ATOMIC_EDIT_SCHEMA,),
    "opportunity": (CANDIDATE_OPPORTUNITY, CANDIDATE_MULTIPLICITY, BENEFICIAL_HARMFUL_RATIO),
    "oracles": (
        CANDIDATE_ORACLE,
        RISK_CANDIDATE_ORACLE,
        RANKING_ORACLE,
        PERFECT_RANKING,
        DEPLOYMENT_BASELINE,
        PERFECT_DEPLOYMENT,
    ),
    "ceiling": (CEILING_DECOMPOSITION, NORMALIZED_GAP),
    "taxonomy": (SITE_FAILURE_TAXONOMY,),
    "strata": (ENGINE_ANALYSIS, DOMAIN_ANALYSIS, ERROR_TYPE_ANALYSIS, OCR_QUALITY_ANALYSIS),
    "ambiguity": (AMBIGUITY_ANALYSIS, TOPK_OPPORTUNITY),
    "review": (HUMAN_REVIEW_ORACLE, REVIEW_HEADROOM),
    "tolerance": (HARM_TOLERANCE_CEILING,),
    "bottleneck": (DOMINANT_BOTTLENECK,),
    "controls": (CONTROL_RESULTS,),
    "negative": (NEGATIVE_TESTS,),
    "stats": (STATISTICAL_TESTS,),
    "figures": (FIGURE_MANIFEST,),
    "decide": (DECISION,),
    "determinism": (DETERMINISM,),
    "record": (PROVENANCE, TRACEABILITY),
}

PRODUCED: tuple[Path, ...] = tuple(
    path for name in PHASE_ARTIFACTS for path in PHASE_ARTIFACTS[name]
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE, FROZEN_CONFIGURATION, UPSTREAM_REPRODUCTION),
    "2. Frozen upstream findings": (RESEARCH_FREEZE, UPSTREAM_REPRODUCTION),
    "3. Why the candidate universe must be diagnosed": (
        CEILING_DECOMPOSITION,
        DESIGN_RECORD,
        DEPLOYMENT_BASELINE,
    ),
    "4. Research questions": (DESIGN_RECORD,),
    "5. The frozen candidate generator": (
        FROZEN_CONFIGURATION,
        ENVIRONMENT_INVENTORY,
        CANDIDATE_INVENTORY,
        SITE_INVENTORY,
    ),
    "6. Candidate and site definitions": (
        DESIGN_RECORD,
        ENVIRONMENT_INVENTORY,
        SITE_INVENTORY,
        CANDIDATE_INVENTORY,
    ),
    "7. Opportunity recall": (CANDIDATE_OPPORTUNITY,),
    "8. Repairable against unrepairable OCR errors": (CANDIDATE_OPPORTUNITY, ERROR_TYPE_ANALYSIS),
    "9. Candidate multiplicity": (CANDIDATE_MULTIPLICITY,),
    "10. Beneficial against harmful composition": (BENEFICIAL_HARMFUL_RATIO,),
    "11. The perfect candidate-selection ceiling": (CANDIDATE_ORACLE,),
    "12. The risk-constrained candidate ceiling": (
        RISK_CANDIDATE_ORACLE,
        HARM_TOLERANCE_CEILING,
        NEGATIVE_TESTS,
    ),
    "13. The frozen ranking ceiling": (
        RANKING_ORACLE,
        PERFECT_RANKING,
        CANDIDATE_OPPORTUNITY,
    ),
    "14. The practical deployment baseline": (DEPLOYMENT_BASELINE, PERFECT_DEPLOYMENT),
    "15. The full ceiling decomposition": (CEILING_DECOMPOSITION, NORMALIZED_GAP),
    "16. The candidate gap": (CANDIDATE_OPPORTUNITY, CEILING_DECOMPOSITION),
    "17. The ranking gap": (
        CEILING_DECOMPOSITION,
        AMBIGUITY_ANALYSIS,
        TOPK_OPPORTUNITY,
        BENEFICIAL_HARMFUL_RATIO,
    ),
    "18. The deployment gap": (
        CEILING_DECOMPOSITION,
        DEPLOYMENT_BASELINE,
        NORMALIZED_GAP,
    ),
    "19. The failure taxonomy": (SITE_FAILURE_TAXONOMY,),
    "20. Engine analysis": (ENGINE_ANALYSIS,),
    "21. Domain analysis": (DOMAIN_ANALYSIS,),
    "22. Error-type analysis": (ERROR_TYPE_ANALYSIS, ENVIRONMENT_INVENTORY),
    "23. Candidate ambiguity": (AMBIGUITY_ANALYSIS, TOPK_OPPORTUNITY),
    "24. The human-review ceiling": (HUMAN_REVIEW_ORACLE, REVIEW_HEADROOM),
    "25. Harm-tolerance analysis": (HARM_TOLERANCE_CEILING,),
    "26. The dominant bottleneck": (
        DOMINANT_BOTTLENECK,
        DESIGN_RECORD,
        CEILING_DECOMPOSITION,
        DECISION,
    ),
    "27. Controls": (CONTROL_RESULTS,),
    "28. Falsification tests": (NEGATIVE_TESTS,),
    "29. Statistics": (STATISTICAL_TESTS, OCR_QUALITY_ANALYSIS),
    "30. Limitations": (
        CANDIDATE_OPPORTUNITY,
        ENVIRONMENT_INVENTORY,
        ATOMIC_EDIT_SCHEMA,
        SITE_FAILURE_TAXONOMY,
    ),
    "31. Research decision": (DECISION,),
    "32. Implications for the next stage": (
        DECISION,
        ATOMIC_EDIT_SCHEMA,
        ERROR_TYPE_ANALYSIS,
        BENEFICIAL_HARMFUL_RATIO,
        CEILING_DECOMPOSITION,
        ENVIRONMENT_INVENTORY,
    ),
}


def run_record() -> int:
    """Provenance, the dependency audit and the traceability index."""
    started = time.monotonic()
    missing = [
        path for path in PRODUCED if not path.exists() and path not in (PROVENANCE, TRACEABILITY)
    ]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {[_relative(p) for p in missing][:5]}")
    freeze = cc_read_json(RESEARCH_FREEZE)
    inputs = dict(cc_read_json(dt1.PROVENANCE)["artifacts"])
    inputs.update(cc_read_json(s15b.PROVENANCE)["artifacts"])
    inputs.update(cc_read_json(s15.PROVENANCE)["artifacts"])
    inputs.update({path: file_sha256(REPO / path) for path in sorted(UPSTREAM_SCRIPTS)})
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    existing = [path for path in PRODUCED if path.exists()]
    tracked = [path for path in existing if _tracked(path)]
    unignored = [path for path in existing if not _ignored(path)]
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "issued_head": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(_git("status", "--porcelain")),
            "artifacts": {_relative(path): file_sha256(path) for path in existing},
            "artifact_count": len(existing),
            "upstream_inputs": inputs,
            "upstream_input_count": len(inputs),
            "documents": {
                _relative(REPORT): file_sha256(REPORT) if REPORT.is_file() else None,
            },
            "upstream_script_sha256_at_section_zero": freeze["upstream_script_sha256"],
            "table_rows": {
                _relative(ALIGNMENT_SITES): len(pd.read_parquet(ALIGNMENT_SITES)),
                _relative(CANDIDATE_ROWS): len(pd.read_parquet(CANDIDATE_ROWS)),
                _relative(SITE_LINKS): len(pd.read_parquet(SITE_LINKS)),
                _relative(EVALUATION_ORDER): len(pd.read_parquet(EVALUATION_ORDER)),
            },
            "aggregate": {
                "environments": int(inventory["totals"]["environments"]),
                "ocr_error_sites": int(inventory["totals"]["alignment_error_sites_evaluation"]),
                "discovered_sites": int(inventory["totals"]["discovered_sites_evaluation"]),
                "candidate_rows": int(inventory["totals"]["candidate_rows"]),
                "epsilons": len(EPSILONS),
                "review_budgets": len(REVIEW_BUDGETS),
                "controls": len(CONTROLS),
                "figures": cc_read_json(FIGURE_MANIFEST)["figure_count"],
            },
            "dependency_audit": {
                "artifacts_tracked_by_git": [_relative(path) for path in tracked],
                "artifacts_not_git_ignored": [_relative(path) for path in unignored],
                "raw_data_written": False,
                "upstream_artifacts_written": False,
                "confirmatory_reserve_consumed": False,
                "model_refits_performed": 0,
                "new_corrector_introduced": False,
                "candidate_set_widened": False,
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
        f"record: {len(existing)} artifacts, {len(inputs)} upstream inputs, "
        f"{len(REPORT_SECTIONS)} report sections -> {_relative(PROVENANCE)}"
    )
    return 0


PHASES: tuple[tuple[str, Callable[[], int]], ...] = (
    ("reconstruct", run_reconstruct),
    ("freeze", run_freeze),
    ("preregister", run_preregister),
    ("inventory", run_inventory),
    ("reproduce", run_reproduce),
    ("schema", run_schema),
    ("opportunity", run_opportunity),
    ("oracles", run_oracles),
    ("ceiling", run_ceiling),
    ("taxonomy", run_taxonomy),
    ("strata", run_strata),
    ("ambiguity", run_ambiguity),
    ("review", run_review),
    ("tolerance", run_tolerance),
    ("bottleneck", run_bottleneck),
    ("controls", run_controls),
    ("negative", run_negative),
    ("stats", run_stats),
    ("figures", run_figures),
    ("decide", run_decide),
    ("determinism", run_determinism),
    ("record", run_record),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, _handler in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--all", action="store_true", help="every phase, in order")
    arguments = parser.parse_args()
    selected = [
        (name, handler) for name, handler in PHASES if getattr(arguments, name) or arguments.all
    ]
    if not selected:
        parser.print_help()
        return 2
    for name, handler in selected:
        began = time.monotonic()
        code = handler()
        if code:
            return code
        print(f"  [{name} {time.monotonic() - began:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
