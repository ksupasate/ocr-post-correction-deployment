#!/usr/bin/env python3
"""SGV-RL1: can correction reliability be represented without generator identity?

SGV-HY1 closed with outcome A: routing the sites only the image proposer finds to the frozen
image-conditioned corrector materially expands the exact-repair candidate universe, and the two
correction paths repair almost disjoint sets of error sites. That leaves the candidate universe
heterogeneous, and SGV-XR1 already established why that is a problem for reliability: the frozen
verifier's representation carries generator-source and operation columns that a new generator does
not have, so it cannot validly score Qwen3-VL candidates zero-shot. That was a statement about the
REPRESENTATION, not about reliability. This stage asks the question that follows:

    can harm and benefit be predicted from source-independent properties of the OCR text, the
    proposed edit, its local context, the page geometry and the document image -- so that useful
    ranking survives a change of correction generator?

**1. Nothing is regenerated.** Every candidate, every label and every proposal stratum is read
from SGV-HY1's frozen tables and hash-verified. No OCR correction is decoded here.

**2. The primary representation never sees who proposed the edit.** Generator source, corrector
source, proposal stratum, candidate rank and per-site candidate counts are carried for splitting
and reporting only; the scoring path cannot read them, and a hard invariance test asserts it.

**3. Ground truth enters only as the label.** Every feature is computed from OCR output, page
pixels and frozen candidate text. Fitted resources -- the character language model, the lexicon,
the glyph prototypes, the confidence and geometry statistics -- come from SGV14 ADAPTATION
documents, which no RL1 candidate lives on.

**4. Both split axes always apply.** The benchmark is matched-source, so holding out an engine
does not hold out the page: every leave-one-environment-out fold also holds out the documents.

    --reconstruct   section 0: CG1, XC1, GEN1, XR1, LP1 and HY1 re-read from their artifacts
    --freeze        upstream hashes, the frozen HY1 configuration, the HY1 candidate freeze
    --preregister   questions, representations, splits, models, criteria, outcome rules
    --population    U5 / U4 / U3, deduplication, group ids, the candidate inventory
    --resources     GT-blind resources fitted on ADAPTATION documents only
    --features      the feature audit, the missing-feature policy and R1-R4
    --splits        the split and group registries
    --sourceleak    the diagnostic source classifier on the generator-agnostic matrix
    --fit           every pre-registered representation x model x split, scored out of fold
    --discriminate  harm and benefit AUROC, average precision, per-stratum discrimination
    --transfer      leave-generator-out, leave-environment-out, the transfer matrix, stress
    --prefix        score-ranked prefixes and the diagnostic oracle risk frontier
    --ablate        the visual-evidence effect, the feature ablations, the source-identity gap
    --controls      random ranking, label permutation, counterfactual source, score invariance
    --stats         document-clustered paired bootstrap, Holm within the frozen families
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen HY1 candidates
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / REPRESENTATION ONLY. No deployment threshold is selected, no certification is run,
no human-review allocation is optimized and no reliability model is adapted to a target. The
oracle risk frontier is analysis-only and is never an operating point.
`ready_for_external_confirmation` is false by construction and the confirmatory reserve stays
LOCKED.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_candidate_conditioned as cc
import sgv_hy1_hybrid_candidate_generation as hy1
from ocr_risk.io.hashing import canonical_hash, file_sha256

gen1 = hy1.gen1
lp1 = hy1.lp1
xr1 = hy1.xr1
xc1 = hy1.xc1
s14 = hy1.s14
s15 = hy1.s15
s13 = hy1.s15.s13 if hasattr(hy1.s15, "s13") else hy1.gen1.s13
cg1 = hy1.cg1
dt1 = hy1.dt1
s5 = None  # bound lazily by _model_family(); sgv5 imports a large module tree.

REPO = hy1.REPO
OUT = REPO / "results/generated/sgv_rl1_generator_agnostic_reliability"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
HY1_CANDIDATE_FREEZE = OUT / "hy1_candidate_freeze.json"
DESIGN_RECORD = OUT / "design_record.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"

CANDIDATE_POPULATION = OUT / "candidate_population.parquet"
POPULATION_INVENTORY = OUT / "candidate_population_inventory.json"
GROUP_REGISTRY = OUT / "group_registry.json"

FITTED_RESOURCES = OUT / "fitted_resources.json"
FEATURE_MATRIX = OUT / "feature_matrix.parquet"
FEATURE_AUDIT = OUT / "feature_audit.json"
FEATURE_REGISTRY = OUT / "feature_registry.json"
MISSING_FEATURE_POLICY = OUT / "missing_feature_policy.json"
R0_LEGACY = OUT / "r0_legacy_representation.json"
R1_REPRESENTATION = OUT / "r1_core_representation.parquet"
R2_REPRESENTATION = OUT / "r2_plausibility_representation.parquet"
R3_REPRESENTATION = OUT / "r3_visual_representation.parquet"
R4_REPRESENTATION = OUT / "r4_source_aware_representation.parquet"

SPLIT_REGISTRY = OUT / "split_registry.json"
IMPLICIT_SOURCE_LEAKAGE = OUT / "implicit_source_leakage.json"

MODEL_REGISTRY = OUT / "model_registry.json"
HYPERPARAMETER_REGISTRY = OUT / "hyperparameter_registry.json"
TRAINING_REGISTRY = OUT / "training_registry.json"
SCORES = OUT / "reliability_scores.parquet"

MIXED_SOURCE_RESULTS = OUT / "mixed_source_results.json"
HARM_DISCRIMINATION = OUT / "harm_discrimination.json"
BENEFIT_DISCRIMINATION = OUT / "benefit_discrimination.json"
AVERAGE_PRECISION = OUT / "average_precision.json"
SOURCE_STRATIFIED = OUT / "source_stratified_results.json"
PROPOSAL_STRATIFIED = OUT / "proposal_stratified_results.json"
ERROR_TYPE_ANALYSIS = OUT / "error_type_analysis.json"
ENGINE_ANALYSIS = OUT / "engine_analysis.json"
DOMAIN_ANALYSIS = OUT / "domain_analysis.json"

LEAVE_GENERATOR_OUT = OUT / "leave_generator_out_results.json"
LEAVE_ENVIRONMENT_OUT = OUT / "leave_environment_out_results.json"
GENERATOR_ENVIRONMENT_STRESS = OUT / "generator_environment_stress.json"
TRANSFER_MATRIX = OUT / "cross_generator_transfer_matrix.json"

PREFIX_RANKING = OUT / "prefix_ranking.json"
ORACLE_RISK_FRONTIER = OUT / "oracle_risk_frontier.json"

VISUAL_ABLATION = OUT / "visual_evidence_ablation.json"
FEATURE_ABLATION = OUT / "feature_ablation.json"
SOURCE_IDENTITY_GAP = OUT / "source_identity_gap.json"

COUNTERFACTUAL_SOURCE = OUT / "counterfactual_source_test.json"
RANDOM_RANKING_CONTROL = OUT / "random_ranking_control.json"
LABEL_PERMUTATION_CONTROL = OUT / "label_permutation_control.json"
CONTROL_RESULTS = OUT / "control_results.json"

STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"
RELIABILITY_DECOMPOSITION = OUT / "reliability_decomposition.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_rl1/generator_agnostic_reliability.md"

SCHEMA_VERSION = 1
STAGE = "sgv_rl1_generator_agnostic_reliability"
HYPOTHESIS = "SGV-RL1-D1"
STAGE_KIND = "DEVELOPMENT / REPRESENTATION"

# ------------------------------------------------------------------ imported, never restated

PhaseError = gen1.PhaseError
_relative = gen1._relative
_git = gen1._git
_write_json_once = gen1._write_json_once
_write_parquet_once = gen1._write_parquet_once
cc_read_json = gen1.cc_read_json
_ratio = gen1._ratio
_slug = gen1._slug
_require = gen1._require
_forbid = gen1._forbid

BOOTSTRAP_RESAMPLES = gen1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = gen1.BOOTSTRAP_SEED
ALPHA = gen1.ALPHA
EPSILONS = (0.05, 0.10, 0.20)
epsilon_key = s15.epsilon_key
PRIMARY_EPSILON = float(s15.PRIMARY_EPSILON)

# ------------------------------------------------------------------ the frozen design

# Candidate populations, read from HY1's frozen arms. U5 is primary: the broadest heterogeneous
# frozen candidate set, carrying both correction paths and both exact repairs and harmful
# candidates, and it does not presume that H3 or H4 routing is optimal -- HY1 did not establish
# specialized routing.
U5 = "u5_dual_union"
U4 = "u4_specialized_routing"
U3 = "u3_hybrid_image"
POPULATIONS = (U5, U4, U3)
PRIMARY_POPULATION = U5
ARM_OF_POPULATION = {U5: hy1.H5, U4: hy1.H4, U3: hy1.H3}

C0 = hy1.C0
C1 = hy1.C1
CORRECTORS = (C0, C1)

# Representation ladder. R0 is the frozen legacy verifier representation, reported as a historical
# within-distribution reference only: SGV-XR1 measured that it is undefined on a new generator.
R0 = "r0_legacy"
R1 = "r1_core"
R2 = "r2_plausibility"
R3 = "r3_visual"
R4 = "r4_source_aware"
REPRESENTATIONS = (R1, R2, R3, R4)
PRIMARY_REPRESENTATION = R3
AGNOSTIC_REPRESENTATIONS = (R1, R2, R3)

# Feature families, keyed by the prefix the feature names actually carry.
FAM_TEXT = "txt_"
FAM_EDIT = "edit_"
FAM_CONF = "conf_"
FAM_GEOM = "geom_"
FAM_PLAUS = "plaus_"
FAM_CTX = "ctx_"
FAM_VIS = "vis_"
FAM_SOURCE = "src_"
FAMILIES: dict[str, str] = {
    "text": FAM_TEXT,
    "edit": FAM_EDIT,
    "confidence": FAM_CONF,
    "geometry": FAM_GEOM,
    "plausibility": FAM_PLAUS,
    "context": FAM_CTX,
    "visual": FAM_VIS,
    "source": FAM_SOURCE,
}
REPRESENTATION_FAMILIES: dict[str, tuple[str, ...]] = {
    R1: (FAM_TEXT, FAM_EDIT, FAM_CONF, FAM_GEOM),
    R2: (FAM_TEXT, FAM_EDIT, FAM_CONF, FAM_GEOM, FAM_PLAUS, FAM_CTX),
    R3: (FAM_TEXT, FAM_EDIT, FAM_CONF, FAM_GEOM, FAM_PLAUS, FAM_CTX, FAM_VIS),
    R4: (FAM_TEXT, FAM_EDIT, FAM_CONF, FAM_GEOM, FAM_PLAUS, FAM_CTX, FAM_VIS, FAM_SOURCE),
}

# The pre-registered ablation set, all on R3. A3 reproduces R2 by construction, which is the
# point: it is the visual-evidence contrast written twice so the two agree.
A0 = "a0_full_r3"
A1 = "a1_no_plausibility"
A2 = "a2_no_confidence_or_geometry"
A3 = "a3_no_visual"
A4 = "a4_no_edit_shape"
ABLATIONS: dict[str, tuple[str, ...]] = {
    A0: (),
    A1: (FAM_PLAUS,),
    A2: (FAM_CONF, FAM_GEOM),
    A3: (FAM_VIS,),
    A4: (FAM_EDIT,),
}

# Targets. Benefit and harm are not symmetric upstream, so they are two heads, never one binary.
T_HARM = "harm"
T_BENEFIT = "benefit"
TARGETS = (T_HARM, T_BENEFIT)

# Model family. The scientific variable is the representation, so the model table is closed and
# its hyperparameters are the repository's published ones, taken unchanged from SGV5 (which took
# them from the SGV1 phase-3 study). M0 is primary everywhere; M1 is a pre-registered robustness
# arm on the primary splits only, because a 200-iteration boosted fit costs three orders of
# magnitude more than the logistic one and the stage runs every fit twice for determinism.
M0 = "m0_logistic"
M1 = "m1_boosted_balanced"
MODELS = (M0, M1)
PRIMARY_MODEL = M0
FIT_SEED = 20260921
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000
BOOSTING_ITERATIONS = 200

# Splits. Both axes of the repository's isolation rule apply to every one of them that holds out
# an environment: the benchmark is matched-source, so the document partition must hold too.
SPLIT_A = "a_mixed_source_document_held_out"
SPLIT_B = "b_leave_generator_out"
SPLIT_B_DOC = "b_doc_leave_generator_out_document_held_out"
SPLIT_B_OVERLAP = "b_overlap_leave_generator_out_matched_sites"
SPLIT_C = "c_leave_environment_out"
SPLIT_D = "d_generator_environment_stress"
SPLIT_W = "w_within_generator_document_held_out"
SPLITS = (SPLIT_A, SPLIT_B, SPLIT_B_DOC, SPLIT_B_OVERLAP, SPLIT_C, SPLIT_D, SPLIT_W)
PRIMARY_SPLIT = SPLIT_A
PRIMARY_TRANSFER_SPLIT = SPLIT_B
DOCUMENT_FOLDS = 5
SPLIT_SEED = "sgv-rl1-document-folds-v1:2026-09-21"

# Which model runs on which split. Pre-registered before any endpoint was read.
MODEL_SPLITS: dict[str, tuple[str, ...]] = {
    M0: SPLITS,
    M1: (SPLIT_A, SPLIT_B, SPLIT_C),
}
# M1's representation set on split C, where the fold count is ten times the others'.
M1_SPLIT_C_REPRESENTATIONS = (R3,)

# Success criteria, frozen before any RL1 endpoint. The AUROC floors are section 21's; the
# repository has no published reliability floor on a heterogeneous candidate universe, so there is
# no prior convention to prefer.
C1_HARM_FLOOR = 0.70
C1_BENEFIT_FLOOR = 0.65
C2_TRANSFER_HARM_FLOOR = 0.65
C3_TRANSFER_BENEFIT_FLOOR = 0.60
# "neither primary generator direction near random": a direction is near random below this.
NEAR_RANDOM = 0.55
C4_RETENTION_FLOOR = 0.90
# A retention ratio is not identifiable when the source-aware reference is itself barely above
# random; a degenerate denominator may never count as success.
RETENTION_MIN_EXCESS = 0.02
C5_BREADTH_MAJORITY = 6
C6_PREFIX_MARGIN = 0.10
# "R4 materially outperforms R3" for the outcome rule.
SOURCE_GAP_MARGIN = 0.05
# Below this, a mixed-source model gives no useful discrimination at all.
USEFUL_AUROC = 0.60

# Controls.
RANDOM_RANKING_DRAWS = 200
RANDOM_RANKING_SEED = 20260922
PERMUTATION_DRAWS = 20
PERMUTATION_SEED = 20260923
COUNTERFACTUAL_SAMPLE = 500
COUNTERFACTUAL_SEED = 20260924

# Prefix geometry.
PREFIX_FRACTIONS = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)

# Fitted resources come from SGV14 ADAPTATION documents, which carry no RL1 candidate. Capped per
# environment so the glyph pass is bounded; the cap is a deterministic prefix of the sorted
# adaptation document ids, never a content-dependent choice.
RESOURCE_ROLE = s14.ROLE_ADAPTATION
RESOURCE_DOCUMENT_CAP = 40
GLYPH_MIN_SUPPORT = cc.PROTOTYPE_MIN_SUPPORT
LM_ORDER = cc.LM_ORDER

QUESTIONS = {
    "Q1": "can a generator-agnostic representation separate beneficial from harmful candidates?",
    "Q2": "trained without one correction generator, does discrimination survive on it?",
    "Q3": "can one source-agnostic model rank a pooled current-generator and Qwen3-VL universe?",
    "Q4": "does document-image evidence improve reliability beyond textual/structural features?",
    "Q5": "how much apparent performance is attributable to generator identity?",
}

OUTCOME_TAXONOMY = {
    "A": "generator-agnostic reliability representation works",
    "B": "mixed-source ranking works, but a qualifying criterion does not",
    "C": "explicit source identity remains load-bearing",
    "D": "candidate-level reliability remains weak even within mixed-source training",
}
NEXT_STAGE = {
    "A": "NEXT: RISK CALIBRATION / TARGET ADAPTATION ON THE GENERATOR-AGNOSTIC SCORE",
    "B": "NEXT: FEW-SHOT GENERATOR-AWARE ADAPTATION OF A SOURCE-AGNOSTIC REPRESENTATION",
    "C": "NEXT: SOURCE-CONDITIONAL / MIXTURE-OF-EXPERTS RELIABILITY",
    "D": "NEXT: RICHER CANDIDATE VERIFICATION EVIDENCE",
}

# Columns no feature builder and no scoring path may read. The first group is ground truth and
# everything derived from it; the second is generator and proposal identity.
FORBIDDEN_LABEL_COLUMNS = (
    *gen1.FORBIDDEN_INPUT_COLUMNS,
    "outcome",
    "is_harmful",
    "beneficial",
    "exact",
    "d_before",
    "d_after",
    "site_class",
    "labelable",
    "harmful",
    "exact_repair",
    "oracle_threshold",
)
FORBIDDEN_SOURCE_COLUMNS = (
    "corrector_source",
    "candidate_source",
    "proposal_stratum",
    "generator_rank",
    "method",
    "condition",
    "in_p0",
    "in_image",
    "in_text",
    "ht_incremental",
    "p0_suspicion",
    "image_suspicion",
    "text_suspicion",
    "site_candidate_count",
)

# The frozen upstream state RL1 rests on, each value re-read from the stage that produced it.
HY1_EXPECTED: dict[str, Any] = {
    "status": "COMPLETE",
    "outcome": "A",
    "ready_for_reliability_representation_study": True,
    "ready_for_external_confirmation": False,
    "confirmatory_reserve_consumed": False,
    "deployable": False,
    "recommended_next_stage": "NEXT: GENERATOR-AGNOSTIC RELIABILITY REPRESENTATION",
    "falsification_tests_passed": 20,
    "falsification_tests_total": 20,
    "omission_opportunity": 0.0,
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_rl1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "stage_kind": STAGE_KIND,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str) -> dict[str, Any]:
    """Every artifact computed with ground truth says so, and says it selects nothing."""
    return {
        **_envelope(artifact),
        "uses_ground_truth": True,
        "deployable": False,
        "selects_a_deployment_threshold": False,
        "confirmatory_reserve_consumed": False,
    }


def environment_specs() -> list[dict[str, Any]]:
    return hy1.environment_specs()


def _domain(corpus: str) -> str:
    return xr1._domain(corpus)


def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def _mean(values: Iterable[float]) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite else float("nan")


def retention(agnostic: float, source_aware: float) -> float | None:
    """How much of the source-aware model's excess over random the agnostic one keeps.

    Undefined, and never a success, when the source-aware reference is itself at chance: a ratio
    whose denominator is noise would turn "neither model works" into "identity is unnecessary".
    """
    if not (np.isfinite(agnostic) and np.isfinite(source_aware)):
        return None
    excess = source_aware - 0.5
    if excess < RETENTION_MIN_EXCESS:
        return None
    return float((agnostic - 0.5) / excess)


def assign_outcome(
    criteria: dict[str, bool], source_gap: float, source_aware_harm_auroc: float
) -> str:
    """Exactly one outcome, in the frozen precedence D, C, A, B, C, D.

    D fires first when neither the agnostic nor the source-aware representation discriminates on
    the mixed-source population: there is nothing to attribute. C fires when generator identity is
    load-bearing -- retention fails or the source-aware gap is material -- and transfer does not
    hold. A needs all six criteria. B is the widened mixed-source case: the pooled ranking works
    and identity is not necessary, but one qualifying criterion does not hold, and the label names
    which. Anything left is C if identity is load-bearing, else D.
    """
    transfer = criteria["C2"] and criteria["C3"]
    identity_load_bearing = (not criteria["C4"]) or source_gap >= SOURCE_GAP_MARGIN
    if not criteria["C1"] and source_aware_harm_auroc < USEFUL_AUROC:
        return "D"
    if identity_load_bearing and not transfer:
        return "C"
    if all(criteria.values()):
        return "A"
    if criteria["C1"] and criteria["C4"]:
        return "B"
    if identity_load_bearing:
        return "C"
    return "D"


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    """Every upstream file whose content RL1's conclusions rest on: HY1's list plus HY1's own."""
    files = list(hy1._upstream_files())
    files.append(REPO / "scripts/sgv_hy1_hybrid_candidate_generation.py")
    files.append(REPO / "scripts/sgv1_candidate_conditioned.py")
    for path in sorted(hy1.OUT.glob("*.json")):
        files.append(path)
    for path in sorted(hy1.OUT.glob("*.parquet")):
        files.append(path)
    files.append(hy1.REPORT)
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = xr1._check
    checks = list(hy1.upstream_checks())
    decision = cc_read_json(hy1.DECISION)
    for key, value in HY1_EXPECTED.items():
        checks.append(_check(f"sgv_hy1.{key}", decision.get(key), value))
    # HY1's qualitative findings, each as the number it rests on.
    opportunity = cc_read_json(hy1.OPPORTUNITY_RESULTS)["arms"]
    checks.append(
        _check(
            "sgv_hy1.h0_is_the_frozen_current_baseline",
            abs(
                float(opportunity[hy1.H0]["mean"])
                - float(cc_read_json(xr1.DECISION)["baseline_opportunity"])
            )
            < 5e-5,
            True,
        )
    )
    checks.append(
        _check(
            "sgv_hy1.h4_materially_improves_opportunity",
            bool(float(opportunity[hy1.H4]["mean"]) > float(opportunity[hy1.H0]["mean"])),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_hy1.h5_is_the_largest_candidate_opportunity",
            max(REPRESENTATION_ORDERING_GUARD, key=lambda a: float(opportunity[a]["mean"])),
            hy1.H5,
        )
    )
    checks.append(
        _check(
            "sgv_hy1.h3_can_outperform_h4_numerically",
            bool(float(opportunity[hy1.H3]["mean"]) > float(opportunity[hy1.H4]["mean"])),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_hy1.h4_beats_the_matched_text_proposal_hybrid",
            bool(float(opportunity[hy1.H4]["mean"]) > float(opportunity[hy1.HT]["mean"])),
            True,
        )
    )
    translation = cc_read_json(hy1.TRANSLATION_RESULTS)["by_stratum"]
    checks.append(
        _check(
            "sgv_hy1.image_only_translation_exceeds_p0_only",
            bool(
                float(translation[hy1.S_IMAGE]["pooled"]) > float(translation[hy1.S_P0]["pooled"])
            ),
            True,
        )
    )
    overlap = cc_read_json(hy1.CANDIDATE_OVERLAP)["h0_vs_image_branch"]
    checks.append(
        _check(
            "sgv_hy1.current_and_image_repairs_are_nearly_disjoint",
            bool(float(overlap["jaccard"]) < 0.05),
            True,
        )
    )
    burden = cc_read_json(hy1.HARM_EFFICIENCY)
    checks.append(
        _check(
            "sgv_hy1.harm_cost_within_the_pre_registered_comparator",
            bool(burden["within_comparator"]),
            True,
        )
    )
    # HY1 measured H3 above H4 numerically and the difference did not survive Holm, so
    # specialization itself is not established. RL1 must not treat H4 as the settled architecture.
    secondary = cc_read_json(hy1.STATISTICAL_TESTS)["secondary_family"]
    specialization = secondary[f"S2_{hy1.H3}_vs_{hy1.H4}"]
    checks.append(
        _check(
            "sgv_hy1.specialized_routing_is_not_established",
            bool(specialization["survives_holm"]),
            False,
        )
    )
    checks.append(
        _check(
            "sgv_hy1.h3_outperforms_h4_numerically",
            bool(float(specialization["effect"]) > 0.0),
            True,
        )
    )
    checks.append(
        _check("sgv_hy1.determinism", cc_read_json(hy1.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_hy1.report_traceability",
            cc_read_json(hy1.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
            0,
        )
    )
    provenance = cc_read_json(hy1.PROVENANCE)
    checks.append(
        _check("sgv_hy1.upstream_files_moved", len(provenance["upstream_files_moved"]), 0)
    )
    checks.append(_check("sgv_hy1.lp1_unchanged", provenance["lp1_unchanged"], True))
    # The verifier incompatibility this stage exists to answer, as the number that established it.
    compatibility = cc_read_json(xr1.FEATURE_COMPATIBILITY)
    checks.append(
        _check(
            "sgv_xr1.zero_shot_feature_compatible",
            compatibility["zero_shot_feature_compatible"],
            False,
        )
    )
    checks.append(_check("sgv_xr1.verifier_reproduced", compatibility["verifier_reproduced"], True))
    checks.append(
        _check(
            "sgv_xr1.generator_source_is_not_inert",
            compatibility["inert_by_family"]["generator_source"],
            False,
        )
    )
    checks.append(
        _check(
            "sgv_xr1.operation_labels_are_not_inert",
            compatibility["inert_by_family"]["operation"],
            False,
        )
    )
    # The reserve, everywhere it is recorded.
    for path in (*xr1.RESERVE_DECISIONS, xr1.DECISION, lp1.DECISION, hy1.DECISION):
        consumed = cc_read_json(path).get("confirmatory_reserve_consumed")
        checks.append(_check(f"reserve_locked:{_relative(path)}", consumed, False))
    return checks


REPRESENTATION_ORDERING_GUARD = (hy1.H0, hy1.H1, hy1.H2, hy1.H3, hy1.H4, hy1.H5, hy1.HT)


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


def run_freeze() -> int:
    """The research freeze: upstream hashes and the immutable HY1 candidate universe."""
    started = time.monotonic()
    for path in (
        RESEARCH_FREEZE,
        FROZEN_CONFIGURATION,
        HY1_CANDIDATE_FREEZE,
        ENVIRONMENT_INVENTORY,
    ):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an RL1 endpoint already exists; the freeze must precede every one")
    OUT.mkdir(parents=True, exist_ok=True)
    upstream = _upstream_files()
    hashes = {_relative(p): file_sha256(p) for p in upstream}
    hy1_provenance = cc_read_json(hy1.PROVENANCE)["artifacts"]
    moved = sorted(path for path, sha in hy1_provenance.items() if file_sha256(REPO / path) != sha)
    if moved:
        raise PhaseError(f"{len(moved)} HY1 artifacts moved since HY1 issued its record: {moved}")
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "upstream_sha256": hashes,
            "upstream_file_count": len(hashes),
            "hy1_artifacts_rehashed": len(hy1_provenance),
            "hy1_artifacts_moved": moved,
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status": _git("status", "--short"),
                "diff_stat": _git("diff", "--stat"),
            },
            "hy1_is_immutable_input": True,
            "regenerates_a_correction": False,
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "candidates": _relative(hy1.CANDIDATES),
            "proposal_strata": _relative(hy1.PROPOSAL_STRATA),
            "site_truth": _relative(hy1.SITE_TRUTH),
            "error_sites": _relative(hy1.ERROR_SITES),
            "correction_contexts": _relative(hy1.CONTEXTS),
            "crop_inventory": _relative(hy1.CROP_INVENTORY),
            "labelling": cc_read_json(xr1.FROZEN_CONFIGURATION)["labelling"],
            "routing": {arm: hy1.ROUTING[arm] for arm in ARM_OF_POPULATION.values()},
            "budget": {"c0": "the frozen generator's own", "c1": hy1.PRIMARY_K},
            "legacy_verifier": cc_read_json(xr1.FROZEN_CONFIGURATION)["frozen_verifier"],
            "legacy_verifier_incompatibility": _relative(xr1.FEATURE_COMPATIBILITY),
            "resource_role": RESOURCE_ROLE,
            "uses_ground_truth": False,
        },
    )
    candidates = pd.read_parquet(hy1.CANDIDATES)
    strata = pd.read_parquet(hy1.PROPOSAL_STRATA)
    _write_json_once(
        HY1_CANDIDATE_FREEZE,
        {
            **_envelope("hy1_candidate_freeze"),
            "tables": {
                name: {
                    "path": _relative(path),
                    "sha256": file_sha256(path),
                    "rows": int(rows),
                    "matches_hy1_provenance": bool(
                        hy1_provenance.get(_relative(path)) == file_sha256(path)
                    ),
                }
                for name, path, rows in (
                    ("candidates", hy1.CANDIDATES, len(candidates)),
                    ("proposal_strata", hy1.PROPOSAL_STRATA, len(strata)),
                    ("site_truth", hy1.SITE_TRUTH, len(pd.read_parquet(hy1.SITE_TRUTH))),
                    ("error_sites", hy1.ERROR_SITES, len(pd.read_parquet(hy1.ERROR_SITES))),
                    ("contexts", hy1.CONTEXTS, len(pd.read_parquet(hy1.CONTEXTS))),
                )
            },
            "candidate_rows": len(candidates),
            "candidate_digest": canonical_hash(
                candidates.sort_values("candidate_id", kind="stable")["candidate_id"]
                .astype(str)
                .tolist()
            ),
            "label_digest": canonical_hash(
                candidates.sort_values("candidate_id", kind="stable")[
                    ["candidate_id", "outcome", "is_harmful", "beneficial", "exact"]
                ].to_json(orient="records")
            ),
            "never_regenerated_here": True,
            "uses_ground_truth": True,
        },
    )
    specs = environment_specs()
    _write_json_once(
        ENVIRONMENT_INVENTORY,
        {
            **_envelope("environment_inventory"),
            "environments": [
                {
                    "environment": spec["environment"],
                    "corpus": spec["corpus"],
                    "base_engine": spec["base_engine"],
                    "engine_id": spec["engine_id"],
                    "domain": _domain(spec["corpus"]),
                }
                for spec in specs
            ],
            "count": len(specs),
            "uses_ground_truth": False,
        },
    )
    print(
        f"freeze: {len(hashes)} upstream files hashed, {len(candidates)} HY1 candidates frozen "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _labels_exist() -> bool:
    """True once any RL1 artifact written with a label exists."""
    return any(p.exists() for p in (SCORES, MIXED_SOURCE_RESULTS, DECISION))


# ------------------------------------------------------------------ sections 3-8: the design

# Error-site kinds, under the names section 26 uses.
ERROR_KIND_LABEL = {
    "substitution": "substitution",
    "deletion": "omission",
    "segmentation": "segmentation",
    "insertion": "spurious_insertion",
}
ERROR_KINDS = tuple(ERROR_KIND_LABEL.values())

# The pre-registered statistical families. Sizes are fixed here, before any endpoint is read.
PRIMARY_FAMILY = (
    "P1_r3_vs_r1_mixed_source_harm_auroc",
    "P2_r3_vs_r1_leave_generator_out_harm_auroc",
    "P3_r3_vs_r2_visual_evidence_harm_auroc",
    "P4_r3_vs_r4_source_identity_harm_auroc",
)
BENEFIT_FAMILY = (
    "B1_r3_vs_r1_mixed_source_benefit_auroc",
    "B2_r3_vs_r1_leave_generator_out_benefit_auroc",
    "B3_r3_vs_r2_visual_evidence_benefit_auroc",
    "B4_r3_vs_r4_source_identity_benefit_auroc",
)
CONTROL_FAMILY = (
    "K1_r3_vs_random_ranking_harm_auroc",
    "K2_r3_vs_label_permutation_harm_auroc",
    "K3_r3_vs_random_ranking_benefit_auroc",
    "K4_r3_vs_label_permutation_benefit_auroc",
)


def run_preregister() -> int:
    """Sections 3-21: every choice that could otherwise be made after seeing an endpoint."""
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("an RL1 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "questions": QUESTIONS,
            "primary_hypothesis": (
                "harm and benefit are predictable from source-independent properties of the OCR "
                "text, the proposed edit, its local context, the page geometry and the available "
                "visual evidence, such that useful ranking transfers across correction generators"
            ),
            "populations": {
                U5: {
                    "hy1_arm": hy1.H5,
                    "role": "primary",
                    "why": (
                        "the broadest frozen heterogeneous candidate set: both correction paths, "
                        "both exact repairs and harmful candidates, and no presumption that H3 or "
                        "H4 routing is optimal -- HY1 did not establish specialized routing"
                    ),
                },
                U4: {"hy1_arm": hy1.H4, "role": "secondary", "why": "HY1's registered primary arm"},
                U3: {
                    "hy1_arm": hy1.H3,
                    "role": "secondary",
                    "why": "sensitivity to candidate routing and composition",
                },
            },
            "secondary_populations_may_not_select_the_primary_representation": True,
            "targets": {
                T_HARM: "P(harmful candidate); AUROC positive class = harmful",
                T_BENEFIT: "P(beneficial candidate); AUROC positive class = beneficial",
                "not_collapsed": "benefit and harm are not symmetric upstream, so they stay apart",
            },
            "label_source": "the frozen project taxonomy, read from HY1; never redefined here",
            "representations": {
                R0: "the frozen legacy verifier representation; historical reference only",
                R1: "core generator-agnostic: OCR/edit surface, edit shape, confidence, geometry",
                R2: "R1 plus source-independent candidate plausibility and page-context support",
                R3: "R2 plus generator-independent visual evidence from the document crop",
                R4: "R3 plus explicit source identity; a diagnostic comparator, not a proposal",
            },
            "representation_families": {
                name: list(REPRESENTATION_FAMILIES[name]) for name in REPRESENTATIONS
            },
            "ablations": {name: list(drop) for name, drop in ABLATIONS.items()},
            "models": {
                M0: "regularized logistic regression on standardized features",
                M1: "histogram gradient boosting, the repository's published practical verifier",
            },
            "primary_model": PRIMARY_MODEL,
            "model_splits": {name: list(splits) for name, splits in MODEL_SPLITS.items()},
            "m1_split_c_representations": list(M1_SPLIT_C_REPRESENTATIONS),
            "model_selection": (
                "none. Hyperparameters are the repository's published ones, reused unchanged from "
                "SGV5 (which took them from the SGV1 phase-3 study). No grid is searched, no "
                "configuration is tuned per held-out generator, and no held-out endpoint label "
                "reaches a selection decision."
            ),
            "splits": {
                SPLIT_A: (
                    "mixed-source, grouped by document: five content-blind folds over the frozen "
                    "evaluation documents, every source present in train and test"
                ),
                SPLIT_B: (
                    "primary transfer test: train on every candidate whose corrector is not the "
                    "held-out one AND whose site carries no held-out-generator candidate; "
                    "evaluate on the held-out generator's candidates at sites carrying no "
                    "training-generator candidate. Documents are deliberately shared -- the "
                    "question is generator transfer on the same pages -- and B_doc adds the "
                    "document axis"
                ),
                SPLIT_B_DOC: "Split B with the document partition also held out",
                SPLIT_B_OVERLAP: (
                    "stratum-controlled: overlap sites only, where both generators produced "
                    "candidates on the same sites, with the document partition held out"
                ),
                SPLIT_C: (
                    "leave-one-environment-out AND leave-documents-out together. The benchmark is "
                    "matched-source, so holding out an engine does not hold out the page"
                ),
                SPLIT_D: "one generator and one environment held out at once; sparse, secondary",
                SPLIT_W: (
                    "within-generator with the document partition held out: the diagonal of the "
                    "cross-generator transfer matrix, so the off-diagonal transfer cells have a "
                    "same-generator reference measured under the same document discipline"
                ),
            },
            "document_folds": DOCUMENT_FOLDS,
            "split_seed": SPLIT_SEED,
            "group_rule": (
                "the resampling and grouping unit is the document. A document never appears in "
                "training for one environment and evaluation for another, because the same page "
                "is read by every engine"
            ),
            "multi_source_rule": (
                "one atomic edit proposed by both correctors is kept as ONE candidate row with "
                "both sources recorded, never duplicated. Such a candidate is excluded from the "
                "strict leave-one-generator-out test by construction, because its site carries a "
                "candidate from both generators, and is included in mixed-source evaluation"
            ),
            "forbidden_primary_features": list(FORBIDDEN_SOURCE_COLUMNS),
            "forbidden_label_features": list(FORBIDDEN_LABEL_COLUMNS),
            "fitted_resource_scope": (
                f"{RESOURCE_ROLE} documents only, capped at {RESOURCE_DOCUMENT_CAP} per "
                "environment in sorted document order; no RL1 candidate lives on one"
            ),
            "metrics": {
                "primary": "AUROC for harm and for benefit, with the positive class stated",
                "secondary": "average precision, prefix geometry, the oracle risk frontier",
                "calibration": "diagnostic only; no score is called a probability",
            },
            "prefix_fractions": list(PREFIX_FRACTIONS),
            "epsilons": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "oracle_frontier_is_analysis_only": True,
            "criteria": {
                "C1": (
                    f"R3 mixed-source harm AUROC > {C1_HARM_FLOOR} and benefit AUROC > "
                    f"{C1_BENEFIT_FLOOR} on the document-held-out population"
                ),
                "C2": (
                    f"mean leave-one-generator-out harm AUROC >= {C2_TRANSFER_HARM_FLOOR}, with "
                    f"neither primary direction below {NEAR_RANDOM}"
                ),
                "C3": f"mean leave-one-generator-out benefit AUROC >= {C3_TRANSFER_BENEFIT_FLOOR}",
                "C4": (
                    f"R3 retains at least {C4_RETENTION_FLOOR} of R4's excess AUROC above random "
                    "for BOTH harm and benefit; a denominator below "
                    f"{RETENTION_MIN_EXCESS} makes retention unidentifiable and never a success"
                ),
                "C5": (
                    f"R3 beats the source-free single-feature baseline in at least "
                    f"{C5_BREADTH_MAJORITY}/10 environments on harm AUROC"
                ),
                "C6": (
                    f"at harm <= {PRIMARY_EPSILON}, R3 captures at least +{C6_PREFIX_MARGIN} "
                    "absolute RepairRecall over seeded random ranking"
                ),
            },
            "source_free_baseline": (
                "normalized character edit distance between the OCR text and the candidate, used "
                "directly as a harm score. No fitting, no source, computable identically for "
                "every candidate of every generator"
            ),
            "outcome_rule": OUTCOME_TAXONOMY,
            "outcome_margins": {
                "source_gap_material": SOURCE_GAP_MARGIN,
                "useful_auroc": USEFUL_AUROC,
                "near_random": NEAR_RANDOM,
            },
            "statistical_families": {
                "primary": list(PRIMARY_FAMILY),
                "benefit": list(BENEFIT_FAMILY),
                "controls": list(CONTROL_FAMILY),
            },
            "bootstrap": {
                "unit": "document, resampled within environment",
                "paired": True,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "multiplicity": "Holm within each declared family",
            },
            "non_goals": [
                "no deployment threshold is selected",
                "no certification or finite-sample guarantee is computed",
                "no human-review allocation is optimized",
                "no reliability model is adapted to a target environment or generator",
                "no new OCR correction is generated",
                "no larger LLM or VLM is introduced",
                "the confirmatory reserve is never unlocked",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ sections 5-6: population

DEDUP_KEY = (
    "environment",
    "lattice_site_id",
    "char_start",
    "char_end",
    "original_ocr",
    "candidate_text",
)


def _hy1_candidates() -> pd.DataFrame:
    """HY1's frozen candidate table with the GT-blind membership flags its routing rule reads."""
    candidates = pd.read_parquet(hy1.CANDIDATES)
    strata = pd.read_parquet(hy1.PROPOSAL_STRATA)
    return hy1._attach_membership(candidates, strata)


def _error_kind_of_site() -> dict[tuple[str, str], str]:
    """Each proposal site's error-site kind, or ``none`` / ``mixed``, from HY1's frozen links."""
    errors = pd.read_parquet(hy1.ERROR_SITES)
    kind_of = dict(
        zip(
            errors["align_site_id"].astype(str),
            errors["site_kind"].astype(str).map(ERROR_KIND_LABEL),
            strict=True,
        )
    )
    links = pd.read_parquet(lp1.PROPOSAL_ERROR_LINKS)
    links = links[links["align_site_id"].astype(str).isin(kind_of)]
    out: dict[tuple[str, str], set[str]] = {}
    for environment, site, target in zip(
        links["environment"].astype(str),
        links["site_id"].astype(str),
        links["align_site_id"].astype(str),
        strict=True,
    ):
        out.setdefault((environment, site), set()).add(kind_of[target])
    resolved: dict[tuple[str, str], str] = {}
    for key, kinds in out.items():
        resolved[key] = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    return resolved


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per atomic edit, with every proposing source recorded.

    HY1's candidate identity is the atomic edit at a lattice site. When both correctors propose
    the same edit the row is kept once -- duplicating it would inflate the sample with a pair that
    cannot be separated by any feature -- and both sources are carried for diagnostics and for the
    leave-one-generator-out exclusion rule.
    """
    ordered = frame.sort_values(["corrector_source", "candidate_id"], kind="stable")
    grouped = ordered.groupby(list(DEDUP_KEY), sort=False)
    sources = grouped["corrector_source"].apply(lambda s: "|".join(sorted(set(s.astype(str)))))
    counts = grouped.size()
    first = grouped.head(1).copy()
    index = pd.MultiIndex.from_frame(first[list(DEDUP_KEY)])
    first["corrector_sources"] = sources.reindex(index).to_numpy()
    first["source_count"] = counts.reindex(index).to_numpy()
    first["multi_source"] = first["source_count"] > 1
    if int(first["multi_source"].sum()) != int((counts > 1).sum()):
        raise PhaseError("the multi-source count disagrees with the deduplication groups")
    return first.reset_index(drop=True)


def build_population(name: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """One frozen RL1 candidate population, deduplicated, with its group ids and its labels."""
    arm = ARM_OF_POPULATION[name]
    block = hy1.arm_candidates(candidates, arm)
    block = deduplicate(block)
    kinds = _error_kind_of_site()
    truth = pd.read_parquet(hy1.SITE_TRUTH)
    site_class = dict(
        zip(
            zip(truth["environment"].astype(str), truth["site_id"].astype(str), strict=True),
            truth["site_class"].astype(str),
            strict=True,
        )
    )
    specs = {spec["environment"]: spec for spec in environment_specs()}
    keys = list(
        zip(block["environment"].astype(str), block["lattice_site_id"].astype(str), strict=True)
    )
    out = pd.DataFrame(
        {
            "population": name,
            "candidate_id": block["candidate_id"].astype(str),
            "environment": block["environment"].astype(str),
            "corpus": block["corpus"].astype(str),
            "base_engine": block["base_engine"].astype(str),
            "domain": [_domain(specs[e]["corpus"]) for e in block["environment"].astype(str)],
            "document_id": block["document_id"].astype(str),
            "site_id": block["lattice_site_id"].astype(str),
            "proposal_stratum": block["proposal_stratum"].astype(str),
            "corrector_source": block["corrector_source"].astype(str),
            "corrector_sources": block["corrector_sources"].astype(str),
            "candidate_sources": block["candidate_source"].astype(str),
            "source_count": block["source_count"].astype(int),
            "multi_source": block["multi_source"].astype(bool),
            "generator_rank": block["generator_rank"].astype(int),
            "anchor_kind": block["anchor_kind"].astype(str),
            "edit_kind": block["edit_kind"].astype(str),
            "char_start": block["char_start"].astype(int),
            "char_end": block["char_end"].astype(int),
            "original_ocr": block["original_ocr"].astype(str),
            "candidate_text": block["candidate_text"].astype(str),
            "error_kind": [kinds.get(k, "none") for k in keys],
            "site_class": [site_class.get(k, "unknown") for k in keys],
            "outcome": block["outcome"].astype(str),
            "is_harmful": block["is_harmful"].astype(bool),
            "beneficial": block["beneficial"].astype(bool),
            "exact": block["exact"].astype(bool),
            "d_before": block["d_before"].astype(int),
            "d_after": block["d_after"].astype(int),
        }
    )
    out["site_group"] = out["environment"] + "|" + out["site_id"]
    out["edit_group"] = [
        canonical_hash(
            {
                "environment": e,
                "site": s,
                "start": int(a),
                "end": int(b),
                "original": o,
                "candidate": c,
            }
        )
        for e, s, a, b, o, c in zip(
            out["environment"],
            out["site_id"],
            out["char_start"],
            out["char_end"],
            out["original_ocr"],
            out["candidate_text"],
            strict=True,
        )
    ]
    return out.sort_values(["environment", "candidate_id"], kind="stable").reset_index(drop=True)


def document_folds() -> dict[str, int]:
    """Content-blind document folds: ordered by corpus then by a seeded hash, then round robin.

    Round robin inside the corpus rather than a hash modulus, so both corpora are represented in
    every fold. With 49 evaluation documents a modulus can and does empty a corpus from a fold,
    and a fold with no historical page would make the per-fold numbers incomparable.
    """
    population = pd.read_parquet(CANDIDATE_POPULATION)
    frame = population[population["population"] == PRIMARY_POPULATION]
    pairs = sorted(
        {
            (str(corpus), str(document))
            for corpus, document in zip(frame["corpus"], frame["document_id"], strict=True)
        }
    )
    assignment: dict[str, int] = {}
    for corpus in sorted({c for c, _d in pairs}):
        documents = sorted(
            (d for c, d in pairs if c == corpus),
            key=lambda d: (canonical_hash({"seed": SPLIT_SEED, "document": d}), d),
        )
        for position, document in enumerate(documents):
            assignment[document] = position % DOCUMENT_FOLDS
    return assignment


def run_population() -> int:
    """Sections 5-7: U5, U4 and U3 built from HY1's frozen arms, deduplicated, never regenerated."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (CANDIDATE_POPULATION, POPULATION_INVENTORY, GROUP_REGISTRY):
        _forbid(path)
    candidates = _hy1_candidates()
    frames = [build_population(name, candidates) for name in POPULATIONS]
    population = pd.concat(frames, ignore_index=True)
    _write_parquet_once(CANDIDATE_POPULATION, population)
    inventory: dict[str, Any] = {}
    for name in POPULATIONS:
        block = population[population["population"] == name]
        inventory[name] = {
            "hy1_arm": ARM_OF_POPULATION[name],
            "candidates": len(block),
            "candidates_before_deduplication": int(block["source_count"].sum()),
            "multi_source_candidates": int(block["multi_source"].sum()),
            "sites": int(block["site_id"].nunique()),
            "documents": int(block["document_id"].nunique()),
            "environments": int(block["environment"].nunique()),
            "harmful": int(block["is_harmful"].sum()),
            "beneficial": int(block["beneficial"].sum()),
            "exact": int(block["exact"].sum()),
            "harmful_prevalence": _ratio(int(block["is_harmful"].sum()), len(block)),
            "beneficial_prevalence": _ratio(int(block["beneficial"].sum()), len(block)),
            "exact_prevalence": _ratio(int(block["exact"].sum()), len(block)),
            "by_corrector": {
                str(source): {
                    "candidates": len(group),
                    "harmful_prevalence": _ratio(int(group["is_harmful"].sum()), len(group)),
                    "beneficial_prevalence": _ratio(int(group["beneficial"].sum()), len(group)),
                    "exact_prevalence": _ratio(int(group["exact"].sum()), len(group)),
                }
                for source, group in block.groupby("corrector_source", sort=True)
            },
            "by_stratum": {
                str(stratum): {
                    "candidates": len(group),
                    "correctors": sorted(set(group["corrector_source"].astype(str))),
                    "harmful_prevalence": _ratio(int(group["is_harmful"].sum()), len(group)),
                    "beneficial_prevalence": _ratio(int(group["beneficial"].sum()), len(group)),
                }
                for stratum, group in block.groupby("proposal_stratum", sort=True)
            },
            "by_outcome": {
                str(k): int(v) for k, v in block["outcome"].value_counts().sort_index().items()
            },
            "by_error_kind": {
                str(k): int(v) for k, v in block["error_kind"].value_counts().sort_index().items()
            },
        }
    primary = population[population["population"] == PRIMARY_POPULATION]
    _write_json_once(
        POPULATION_INVENTORY,
        {
            **_analysis_envelope("candidate_population_inventory"),
            "populations": inventory,
            "primary_population": PRIMARY_POPULATION,
            "deduplication_rule": (
                "one row per (environment, site, span, OCR text, candidate text); every proposing "
                "corrector is recorded on the surviving row and no row is duplicated"
            ),
            "reads_hy1_only": True,
            "regenerates_a_correction": False,
            "environments": sorted(set(primary["environment"].astype(str))),
        },
    )
    folds = document_folds()
    _write_json_once(
        GROUP_REGISTRY,
        {
            **_envelope("group_registry"),
            "unit": "document",
            "rule": (
                "candidates from one document never cross a train/evaluation boundary, and the "
                "same page read by a different engine moves with it. Site and atomic-edit groups "
                "are nested inside the document group, so neither can cross one either"
            ),
            "document_folds": {k: int(v) for k, v in sorted(folds.items())},
            "fold_count": DOCUMENT_FOLDS,
            "fold_sizes": {
                str(fold): int(sum(1 for v in folds.values() if v == fold))
                for fold in range(DOCUMENT_FOLDS)
            },
            "fold_documents_by_corpus": {
                str(fold): {
                    str(corpus): int(
                        primary[
                            (primary["corpus"] == corpus)
                            & primary["document_id"].map(folds).eq(fold)
                        ]["document_id"].nunique()
                    )
                    for corpus in sorted(set(primary["corpus"].astype(str)))
                }
                for fold in range(DOCUMENT_FOLDS)
            },
            "documents": len(folds),
            "sites": int(primary["site_group"].nunique()),
            "atomic_edit_groups": int(primary["edit_group"].nunique()),
            "seed": SPLIT_SEED,
            "uses_ground_truth": False,
        },
    )
    print(
        f"population: {inventory[PRIMARY_POPULATION]['candidates']} primary candidates, "
        f"{inventory[PRIMARY_POPULATION]['multi_source_candidates']} multi-source, "
        f"{len(folds)} documents ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ section 9: GT-blind resources

CONF_QUANTILE_POINTS = 1001
GEOM_IMPUTED = (
    "geom_width",
    "geom_height",
    "geom_aspect",
    "geom_rel_x",
    "geom_rel_y",
    "geom_rel_area",
    "geom_chars_per_width",
)
CONF_IMPUTED = (
    "conf_anchor_z",
    "conf_anchor_percentile",
    "conf_anchor_min_z",
    "conf_line_mean_z",
    "conf_line_min_z",
    "conf_neighbourhood_z",
    "conf_page_mean_z",
    "conf_anchor_minus_page",
)


@dataclass(slots=True)
class Environment:
    """One environment's frozen OCR, indexed the way every feature block reads it."""

    name: str
    spec: dict[str, Any]
    pages: dict[tuple[str, str], Any]
    spans: dict[str, Any]
    page_of_span: dict[str, tuple[str, str]]
    roles: dict[str, str]


def _environment(spec: dict[str, Any]) -> Environment:
    """Canonicalized OCR, page geometry and document roles for one environment, built once."""
    bundles = s14._document_bundles(spec["corpus"])
    roles = s14.partition_of(spec["corpus"], sorted(bundles))
    pages = gen1._page_index(spec)
    spans, _record = s14._canonical_spans(spec, bundles)
    index: dict[str, Any] = {}
    page_of: dict[str, tuple[str, str]] = {}
    for pair, group in spans.items():
        for span in group:
            index[str(span.span_id)] = span
            page_of[str(span.span_id)] = pair
    return Environment(
        name=spec["environment"],
        spec=spec,
        pages=pages,
        spans=index,
        page_of_span=page_of,
        roles={str(k): str(v) for k, v in roles.items()},
    )


def _resource_documents(environment: Environment) -> list[str]:
    """The ADAPTATION documents this environment's resources are fitted on, deterministically."""
    documents = sorted(d for d, role in environment.roles.items() if role == RESOURCE_ROLE)
    return documents[:RESOURCE_DOCUMENT_CAP]


def _span_ink(page: Any, span_id: str, image: Any) -> np.ndarray | None:
    """The anchor region's ink, built by exactly the pipeline the evaluation crops came from."""
    box = gen1.crop_box(page, [span_id])
    if box is None:
        return None
    crop = gen1.render_crop(image, box)
    rectangle = xr1.mask_rectangle(page, [span_id], False, box, crop.size)
    if rectangle is None:
        return None
    array = np.asarray(crop.convert("L"), dtype=np.float64) / 255.0
    a, b, c, d = rectangle
    region = 1.0 - array[b:d, a:c]
    return region if region.size else None


def fit_environment_resources(environment: Environment) -> dict[str, Any]:
    """Language, lexicon, glyph, confidence and geometry resources from ADAPTATION pages only.

    Every one of them is built from the engine's own output and the page pixels, never from ground
    truth, and every one is fitted on documents that carry no RL1 candidate. A deployed system has
    exactly this: unlabelled pages of its own from its own engine.
    """
    from PIL import Image

    documents = _resource_documents(environment)
    engine = environment.spec["base_engine"]
    streams = {
        (document, engine): environment.pages[(document, engine)].stream
        for document in documents
        if (document, engine) in environment.pages
    }
    resources = cc.fit_resources(streams, set(documents))

    confidences: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    aspects: list[float] = []
    rel_x: list[float] = []
    rel_y: list[float] = []
    rel_area: list[float] = []
    per_char: list[float] = []
    glyph_sums: dict[str, np.ndarray] = {}
    glyph_counts: Counter[str] = Counter()
    width_per_char: list[float] = []
    crops = 0
    for document in documents:
        pair = (document, engine)
        page = environment.pages.get(pair)
        if page is None:
            continue
        image = None
        for span_id in page.order:
            span = environment.spans.get(str(span_id))
            if span is None:
                continue
            if span.native_conf_recognition is not None:
                confidences.append(float(span.native_conf_recognition))
            box = page.span_box.get(span_id)
            if box is not None and page.width and page.height:
                width = float(box[2] - box[0])
                height = float(box[3] - box[1])
                widths.append(width / max(page.height, 1.0))
                heights.append(height / max(page.height, 1.0))
                aspects.append(width / max(height, 1e-6))
                rel_x.append((box[0] + box[2]) / 2.0 / max(page.width, 1.0))
                rel_y.append((box[1] + box[3]) / 2.0 / max(page.height, 1.0))
                rel_area.append(width * height / max(page.width * page.height, 1.0))
                per_char.append(max(len(str(span.text).strip()), 1) / max(width, 1e-6))
            text = str(span.text).strip()
            if not text or len(text) > 24:
                continue
            if image is None:
                if not Path(page.image_path).is_file():
                    break
                image = Image.open(page.image_path).convert("RGB")
            ink = _span_ink(page, str(span_id), image)
            if ink is None:
                continue
            crops += 1
            width_per_char.append(ink.shape[1] / max(len(text), 1))
            for position, character in enumerate(text):
                grid = cc._cell_grid(ink, position, len(text))
                if not grid.any():
                    continue
                glyph_sums.setdefault(character, np.zeros_like(grid))
                glyph_sums[character] += grid
                glyph_counts[character] += 1
        if image is not None:
            image.close()
    prototypes = {
        character: (total / glyph_counts[character]).tolist()
        for character, total in sorted(glyph_sums.items())
        if glyph_counts[character] >= GLYPH_MIN_SUPPORT
    }
    conf = np.asarray(confidences, dtype=np.float64)
    return {
        "environment": environment.name,
        "documents": documents,
        "document_count": len(documents),
        "role": RESOURCE_ROLE,
        "lm_order": LM_ORDER,
        "lm": resources.lm,
        "lm_backoff": resources.lm_backoff,
        "lm_grams": len(resources.lm),
        "lexicon": sorted(resources.lexicon),
        "lexicon_tokens": len(resources.lexicon),
        "glyph_prototypes": prototypes,
        "glyph_prototype_characters": sorted(prototypes),
        "glyph_prototype_count": len(prototypes),
        "glyph_crops": crops,
        "width_per_char": float(np.median(width_per_char)) if width_per_char else 1.0,
        "conf_scale": next(
            (
                str(environment.spans[s].conf_scale)
                for s in environment.spans
                if environment.spans[s].conf_scale is not None
            ),
            "unavailable",
        ),
        "conf_spans": int(conf.size),
        "conf_mean": float(conf.mean()) if conf.size else 0.0,
        "conf_std": float(conf.std(ddof=0)) if conf.size else 1.0,
        "conf_quantiles": (
            np.quantile(conf, np.linspace(0.0, 1.0, CONF_QUANTILE_POINTS)).tolist()
            if conf.size
            else []
        ),
        "imputation_median": {
            "geom_width": float(np.median(widths)) if widths else 0.0,
            "geom_height": float(np.median(heights)) if heights else 0.0,
            "geom_aspect": float(np.median(aspects)) if aspects else 0.0,
            "geom_rel_x": float(np.median(rel_x)) if rel_x else 0.5,
            "geom_rel_y": float(np.median(rel_y)) if rel_y else 0.5,
            "geom_rel_area": float(np.median(rel_area)) if rel_area else 0.0,
            "geom_chars_per_width": float(np.median(per_char)) if per_char else 0.0,
        },
    }


def run_resources() -> int:
    """Section 9: every fitted resource, from ADAPTATION documents, before any feature exists."""
    started = time.monotonic()
    _require(CANDIDATE_POPULATION, "population")
    _forbid(FITTED_RESOURCES)
    population = pd.read_parquet(CANDIDATE_POPULATION)
    used = set(population["document_id"].astype(str))
    by_environment: dict[str, Any] = {}
    for spec in environment_specs():
        began = time.monotonic()
        environment = _environment(spec)
        record = fit_environment_resources(environment)
        overlap = sorted(set(record["documents"]) & used)
        if overlap:
            raise PhaseError(
                f"{spec['environment']}: {len(overlap)} resource documents carry RL1 candidates"
            )
        by_environment[spec["environment"]] = record
        print(
            f"  {spec['environment']}: {record['document_count']} documents, "
            f"{record['lm_grams']} grams, {record['lexicon_tokens']} lexicon tokens, "
            f"{record['glyph_prototype_count']} glyph prototypes "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )
        del environment
    _write_json_once(
        FITTED_RESOURCES,
        {
            **_envelope("fitted_resources"),
            "by_environment": by_environment,
            "rule": (
                "every fitted resource comes from SGV14 ADAPTATION documents of the same "
                "environment, capped at the first "
                f"{RESOURCE_DOCUMENT_CAP} in sorted document order, and no RL1 candidate lives "
                "on one of them"
            ),
            "fitted_from_ground_truth": False,
            "fitted_from_rl1_candidates": False,
            "glyph_supervision": "the engine's own reading of the span, never the transcription",
            "totals": {
                "documents": sum(r["document_count"] for r in by_environment.values()),
                "lm_grams": sum(r["lm_grams"] for r in by_environment.values()),
                "lexicon_tokens": sum(r["lexicon_tokens"] for r in by_environment.values()),
                "glyph_prototypes": sum(
                    r["glyph_prototype_count"] for r in by_environment.values()
                ),
                "glyph_crops": sum(r["glyph_crops"] for r in by_environment.values()),
            },
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"resources: {len(by_environment)} environments ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ sections 9-14: features

CONTEXT_CHARS = xc1.CONTEXT_CHARS

TEXT_NAMES: tuple[tuple[str, str], ...] = (
    ("txt_len_original", "characters in the OCR text the edit replaces"),
    ("txt_len_candidate", "characters in the proposed replacement"),
    ("txt_len_delta", "replacement length minus OCR length"),
    ("txt_len_ratio", "replacement length over OCR length"),
    ("txt_region_chars", "width of the edited span in stream characters"),
    ("txt_digits_original", "digits in the OCR text"),
    ("txt_digits_candidate", "digits in the replacement"),
    ("txt_digit_delta", "digits gained or lost by the edit"),
    ("txt_alpha_original", "letters in the OCR text"),
    ("txt_alpha_candidate", "letters in the replacement"),
    ("txt_punct_original", "punctuation and symbols in the OCR text"),
    ("txt_punct_candidate", "punctuation and symbols in the replacement"),
    ("txt_space_original", "whitespace characters in the OCR text"),
    ("txt_space_candidate", "whitespace characters in the replacement"),
    ("txt_upper_share_original", "share of the OCR text that is upper case"),
    ("txt_upper_share_candidate", "share of the replacement that is upper case"),
    ("txt_case_only_change", "1 when the edit changes case alone"),
    ("txt_nonascii_original", "non-ASCII characters in the OCR text"),
    ("txt_nonascii_candidate", "non-ASCII characters in the replacement"),
    ("txt_context_before_len", "characters of OCR context recovered before the span"),
    ("txt_context_after_len", "characters of OCR context recovered after the span"),
    ("txt_line_position", "the span's start within its OCR line, as a fraction"),
    ("txt_document_position", "the span's start within the page stream, as a fraction"),
    ("txt_anchor_token", "1 when the site anchors on one OCR token"),
    ("txt_anchor_token_pair", "1 when the site anchors on two adjacent OCR tokens"),
    ("txt_anchor_gap", "1 when the site anchors on the gap between two OCR tokens"),
)

EDIT_DEFINITIONS: dict[str, str] = {
    "edit_subs": "substituted characters in the OCR-to-candidate alignment",
    "edit_inserts": "inserted characters",
    "edit_deletes": "deleted characters",
    "edit_changed": "substituted plus inserted plus deleted characters",
    "edit_first_change_rel": "position of the first change, relative to the OCR length",
    "edit_changed_fraction": "changed characters over OCR length",
    "edit_confusable_subs": "substitutions inside the frozen OCR confusion classes",
    "edit_confusable_fraction": "confusable substitutions over all substitutions",
    "edit_same_length": "1 when OCR and candidate have the same length",
    "edit_digit_only": "1 when both sides are entirely digits",
    "edit_alpha_only": "1 when both sides are entirely letters",
    "edit_cross_class": "1 when the edit crosses the digit/non-digit boundary",
    "edit_prefix_preserved_frac": "share of leading characters the edit leaves alone",
    "edit_len_ratio": "candidate length over OCR length",
}

CONF_NAMES: tuple[tuple[str, str], ...] = (
    ("conf_anchor_z", "mean anchor-span recognition confidence, z-scored on ADAPTATION pages"),
    ("conf_anchor_percentile", "that mean's percentile in the ADAPTATION confidence distribution"),
    ("conf_anchor_min_z", "lowest anchor-span confidence, z-scored"),
    ("conf_line_mean_z", "mean confidence of every span on the anchor's OCR line, z-scored"),
    ("conf_line_min_z", "lowest confidence on that line, z-scored"),
    ("conf_neighbourhood_z", "mean confidence of the two spans either side, z-scored"),
    ("conf_page_mean_z", "mean confidence of the page, z-scored"),
    ("conf_anchor_minus_page", "anchor confidence minus page confidence, both z-scored"),
    ("conf_n_spans", "how many OCR spans the site anchors on"),
    ("conf_missing", "1 when no anchor span reports a recognition confidence"),
)

GEOM_NAMES: tuple[tuple[str, str], ...] = (
    ("geom_width", "width of the anchor box union, over page height"),
    ("geom_height", "height of the anchor box union, over page height"),
    ("geom_aspect", "width over height of the anchor box union"),
    ("geom_rel_x", "box centre x over page width"),
    ("geom_rel_y", "box centre y over page height"),
    ("geom_rel_area", "box area over page area"),
    ("geom_chars_per_width", "OCR characters per unit box width"),
    ("geom_missing", "1 when no anchor span carries a detector box"),
)

PLAUS_DEFINITIONS: dict[str, str] = {
    "plaus_lm_candidate": "character 3-gram log-probability of the replacement",
    "plaus_lm_original": "character 3-gram log-probability of the OCR text",
    "plaus_lm_delta": "the replacement's log-probability minus the OCR text's",
    "plaus_lexicon_candidate": "1 when a replacement token is in the ADAPTATION lexicon",
    "plaus_lexicon_original": "1 when an OCR token is in the ADAPTATION lexicon",
    "plaus_lexicon_gain": "lexicon membership gained by the edit",
    "plaus_price_candidate": "1 when the replacement matches the frozen price pattern",
    "plaus_price_original": "1 when the OCR text matches the frozen price pattern",
    "plaus_price_gain": "price-pattern membership gained by the edit",
    "plaus_numeric_candidate": "1 when the replacement is entirely numeric",
    "plaus_numeric_original": "1 when the OCR text is entirely numeric",
}

CTX_DEFINITIONS: dict[str, str] = {
    "ctx_page_support_candidate": "occurrences of the replacement's tokens elsewhere on the page",
    "ctx_page_support_original": "occurrences of the OCR tokens elsewhere on the page",
    "ctx_page_support_gain": "page support gained by the edit",
    "ctx_line_tokens": "lines in the page stream",
    "ctx_token_count_candidate": "tokens in the replacement",
    "ctx_token_count_delta": "tokens gained or lost by the edit",
}

VIS_DEFINITIONS: dict[str, str] = {
    "vis_cell_mean_y": "mean ink mass per character cell under the replacement's length",
    "vis_cell_min_y": "lowest ink mass per character cell under the replacement's length",
    "vis_cell_std_y": "spread of ink mass per cell under the replacement's length",
    "vis_valley_y": "1 minus the relative ink at the replacement's cell boundaries",
    "vis_cell_mean_o": "mean ink mass per cell under the OCR text's length",
    "vis_cell_min_o": "lowest ink mass per cell under the OCR text's length",
    "vis_cell_std_o": "spread of ink mass per cell under the OCR text's length",
    "vis_valley_o": "1 minus the relative ink at the OCR text's cell boundaries",
    "vis_valley_gain": "the replacement's boundary evidence minus the OCR text's",
    "vis_cell_std_gain": "the replacement's cell-mass spread minus the OCR text's",
    "vis_width_per_char_y": "crop width per replacement character",
    "vis_width_per_char_o": "crop width per OCR character",
    "vis_width_ratio_y": "that width per character over the ADAPTATION median",
    "vis_glyph_match_y": "prototype match of the ink at the changed cells to the replacement",
    "vis_glyph_match_o": "prototype match of the same ink to the OCR characters",
    "vis_glyph_gain": "the replacement's prototype match minus the OCR text's",
    "vis_glyph_available": "1 when glyph matching applied (equal lengths, prototypes present)",
    "vis_missing": "1 when the site has no usable crop region",
}

SOURCE_NAMES: tuple[tuple[str, str], ...] = (
    ("src_corrector_c0", "1 when the frozen current generator proposed the edit"),
    ("src_corrector_c1", "1 when the image-conditioned corrector proposed the edit"),
    ("src_generator_rank", "the proposing generator's own rank for this candidate"),
    ("src_site_candidate_count", "candidates the same generator proposed at this site"),
    ("src_stratum_p0_only", "1 when only the frozen OCR proposer named this site"),
    ("src_stratum_image_only", "1 when only the image proposer named this site"),
    ("src_stratum_overlap", "1 when both proposers named this site"),
    ("src_p0_suspicion", "the frozen proposer's own suspicion score at this site"),
    ("src_p0_suspicion_missing", "1 when the frozen proposer never named this site"),
    ("src_image_suspicion", "the image proposer's own suspicion score at this site"),
    ("src_image_suspicion_missing", "1 when the image proposer never named this site"),
)

FEATURE_SOURCE = {
    FAM_TEXT: "the frozen OCR stream and the frozen candidate text",
    FAM_EDIT: "the alignment of the OCR text with the candidate text",
    FAM_CONF: "the engine's own recognition confidence, normalized on ADAPTATION pages",
    FAM_GEOM: "the engine's own detector boxes and the page dimensions",
    FAM_PLAUS: "a character language model and lexicon fitted on ADAPTATION pages",
    FAM_CTX: "the page's own OCR stream",
    FAM_VIS: "the frozen site crop and glyph prototypes fitted on ADAPTATION pages",
    FAM_SOURCE: "the frozen HY1 candidate provenance",
}


def _resources_of(record: dict[str, Any]) -> cc.FittedResources:
    """One environment's persisted resources, back in the shape the frozen blocks expect."""
    return cc.FittedResources(
        lm={str(k): float(v) for k, v in record["lm"].items()},
        lm_backoff=float(record["lm_backoff"]),
        lexicon=frozenset(str(t) for t in record["lexicon"]),
        glyphs={
            str(k): np.asarray(v, dtype=np.float64) for k, v in record["glyph_prototypes"].items()
        },
        width_per_char=float(record["width_per_char"]),
        documents=int(record["document_count"]),
    )


def _z(value: float | None, mean: float, std: float) -> float:
    return float((float(value) - mean) / std) if (value is not None and std > 0) else 0.0


def text_block(
    original: str,
    candidate: str,
    row: Any,
    line_extent: tuple[int, int],
    stream_length: int,
    before: str,
    after: str,
) -> list[float]:
    """The OCR/edit surface. Nothing here knows which generator proposed the replacement."""
    start = int(row.char_start)
    line_start, line_end = line_extent
    anchor = str(row.anchor_kind)
    return [
        float(len(original)),
        float(len(candidate)),
        float(len(candidate) - len(original)),
        len(candidate) / max(len(original), 1),
        float(int(row.char_end) - start),
        float(sum(c.isdigit() for c in original)),
        float(sum(c.isdigit() for c in candidate)),
        float(sum(c.isdigit() for c in candidate) - sum(c.isdigit() for c in original)),
        float(sum(c.isalpha() for c in original)),
        float(sum(c.isalpha() for c in candidate)),
        float(sum(not c.isalnum() and not c.isspace() for c in original)),
        float(sum(not c.isalnum() and not c.isspace() for c in candidate)),
        float(sum(c.isspace() for c in original)),
        float(sum(c.isspace() for c in candidate)),
        float(sum(c.isupper() for c in original)) / max(len(original), 1),
        float(sum(c.isupper() for c in candidate)) / max(len(candidate), 1),
        1.0 if (original != candidate and original.lower() == candidate.lower()) else 0.0,
        float(sum(ord(c) > 127 for c in original)),
        float(sum(ord(c) > 127 for c in candidate)),
        float(len(before)),
        float(len(after)),
        (start - line_start) / max(line_end - line_start, 1),
        start / max(stream_length, 1),
        1.0 if anchor == "token" else 0.0,
        1.0 if anchor == "token_pair" else 0.0,
        1.0 if anchor == "gap" else 0.0,
    ]


def confidence_block(
    anchors: Sequence[str],
    page: Any,
    spans: dict[str, Any],
    line_spans: Sequence[str],
    neighbours: Sequence[str],
    page_spans: Sequence[str],
    record: dict[str, Any],
) -> list[float]:
    """The recognizer's own confidence, normalized against this engine's ADAPTATION pages."""
    mean, std = float(record["conf_mean"]), float(record["conf_std"]) or 1.0
    quantiles = np.asarray(record["conf_quantiles"], dtype=np.float64)

    def values(ids: Sequence[str]) -> list[float]:
        out = []
        for span_id in ids:
            span = spans.get(str(span_id))
            if span is not None and span.native_conf_recognition is not None:
                out.append(float(span.native_conf_recognition))
        return out

    anchor_values = values(anchors)
    line_values = values(line_spans)
    neighbour_values = values(neighbours)
    page_values = values(page_spans)
    if not anchor_values:
        return [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(len(anchors)), 1.0]
    anchor_mean = float(np.mean(anchor_values))
    page_mean = float(np.mean(page_values)) if page_values else anchor_mean
    percentile = (
        float(np.searchsorted(quantiles, anchor_mean) / max(quantiles.size - 1, 1))
        if quantiles.size
        else 0.5
    )
    _ = page
    return [
        _z(anchor_mean, mean, std),
        min(max(percentile, 0.0), 1.0),
        _z(min(anchor_values), mean, std),
        _z(float(np.mean(line_values)) if line_values else anchor_mean, mean, std),
        _z(min(line_values) if line_values else anchor_mean, mean, std),
        _z(float(np.mean(neighbour_values)) if neighbour_values else anchor_mean, mean, std),
        _z(page_mean, mean, std),
        _z(anchor_mean, mean, std) - _z(page_mean, mean, std),
        float(len(anchors)),
        0.0,
    ]


def geometry_block(
    anchors: Sequence[str], page: Any, original: str, record: dict[str, Any]
) -> list[float]:
    """The detector's own boxes. Imputation is the environment's ADAPTATION median, never zero."""
    boxes = [page.span_box.get(a) for a in anchors]
    boxes = [b for b in boxes if b is not None]
    median = record["imputation_median"]
    if not boxes or not page.width or not page.height:
        return [float(median[name]) for name in GEOM_IMPUTED] + [1.0]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    width, height = float(x1 - x0), float(y1 - y0)
    return [
        width / max(float(page.height), 1.0),
        height / max(float(page.height), 1.0),
        width / max(height, 1e-6),
        (x0 + x1) / 2.0 / max(float(page.width), 1.0),
        (y0 + y1) / 2.0 / max(float(page.height), 1.0),
        width * height / max(float(page.width) * float(page.height), 1.0),
        max(len(original.strip()), 1) / max(width, 1e-6),
        0.0,
    ]


def source_block(row: Any, site: Any, site_candidate_count: int) -> list[float]:
    """The diagnostic comparator's extra block: who proposed the edit, and how it ranked.

    Never part of R1, R2 or R3. It exists so the stage can measure what generator identity is
    worth, which is the only way to say whether a source-independent score gave anything up.
    """
    corrector = str(row.corrector_source)
    stratum = str(row.proposal_stratum)
    p0 = site.p0_suspicion
    image = site.image_suspicion
    has_p0 = p0 is not None and not (isinstance(p0, float) and math.isnan(p0))
    has_image = image is not None and not (isinstance(image, float) and math.isnan(image))
    return [
        1.0 if corrector == C0 else 0.0,
        1.0 if corrector == C1 else 0.0,
        float(row.generator_rank),
        float(site_candidate_count),
        1.0 if stratum == hy1.S_P0 else 0.0,
        1.0 if stratum == hy1.S_IMAGE else 0.0,
        1.0 if stratum == hy1.S_OVERLAP else 0.0,
        float(p0) if has_p0 else 0.0,
        0.0 if has_p0 else 1.0,
        float(image) if has_image else 0.0,
        0.0 if has_image else 1.0,
    ]


FEATURE_NAMES: tuple[str, ...] = (
    *(name for name, _definition in TEXT_NAMES),
    *EDIT_DEFINITIONS,
    *(name for name, _definition in CONF_NAMES),
    *(name for name, _definition in GEOM_NAMES),
    *PLAUS_DEFINITIONS,
    *CTX_DEFINITIONS,
    *VIS_DEFINITIONS,
    *(name for name, _definition in SOURCE_NAMES),
)
FEATURE_DEFINITIONS: dict[str, str] = {
    **dict(TEXT_NAMES),
    **EDIT_DEFINITIONS,
    **dict(CONF_NAMES),
    **dict(GEOM_NAMES),
    **PLAUS_DEFINITIONS,
    **CTX_DEFINITIONS,
    **VIS_DEFINITIONS,
    **dict(SOURCE_NAMES),
}


def family_of(name: str) -> str:
    for prefix in (
        FAM_TEXT,
        FAM_EDIT,
        FAM_CONF,
        FAM_GEOM,
        FAM_PLAUS,
        FAM_CTX,
        FAM_VIS,
        FAM_SOURCE,
    ):
        if name.startswith(prefix):
            return prefix
    raise PhaseError(f"{name} belongs to no declared feature family")


def columns_for(representation: str, drop: Sequence[str] = ()) -> list[str]:
    """The column names one representation uses, in the frozen order the matrix carries."""
    keep = REPRESENTATION_FAMILIES[representation]
    return [
        name
        for name in FEATURE_NAMES
        if family_of(name) in keep and family_of(name) not in set(drop)
    ]


def _line_extents(page: Any) -> dict[int, tuple[int, int]]:
    return {index: (int(start), int(end)) for index, (start, end) in enumerate(page.lines)}


def _spans_by_line(page: Any) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for span_id in page.order:
        out.setdefault(int(page.span_line[span_id]), []).append(str(span_id))
    return out


def _neighbours(page: Any, anchors: Sequence[str]) -> list[str]:
    order = list(page.order)
    positions = [order.index(a) for a in anchors if a in page.span_line]
    if not positions:
        return []
    out: list[str] = []
    for position in (min(positions) - 1, max(positions) + 1):
        if 0 <= position < len(order):
            out.append(str(order[position]))
    return out


def _ink_of(crop_path: Any, rectangle: Any, cache: dict[str, np.ndarray | None]) -> Any:
    """The anchor region's ink from the frozen site crop, cached per crop file."""
    from PIL import Image

    if crop_path is None or rectangle is None:
        return None
    key = f"{crop_path}"
    if key not in cache:
        with Image.open(crop_path) as opened:
            cache[key] = 1.0 - np.asarray(opened.convert("L"), dtype=np.float64) / 255.0
    page_ink = cache[key]
    if page_ink is None:
        return None
    a, b, c, d = (int(v) for v in rectangle)
    region = page_ink[b:d, a:c]
    return region if region.size else None


def build_features() -> pd.DataFrame:
    """Every feature of every RL1 candidate, computed once, from OCR, pixels and candidate text.

    The loop reads the frozen HY1 candidate table, the frozen proposal lattice, the frozen site
    crops and the ADAPTATION-fitted resources. It never reads an outcome, a label, a distance or a
    ground-truth string; the falsification suite asserts that from the call graph.
    """
    population = pd.read_parquet(CANDIDATE_POPULATION)
    wanted = set(population["candidate_id"].astype(str))
    candidates = _hy1_candidates()
    candidates = candidates[candidates["candidate_id"].astype(str).isin(wanted)]
    strata = pd.read_parquet(hy1.PROPOSAL_STRATA)
    contexts = pd.read_parquet(hy1.CONTEXTS)
    resources = cc_read_json(FITTED_RESOURCES)["by_environment"]
    site_counts = (
        _hy1_candidates()
        .groupby(["environment", "lattice_site_id", "corrector_source"])
        .size()
        .to_dict()
    )
    frames: list[pd.DataFrame] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        block = candidates[candidates["environment"] == name]
        if block.empty:
            continue
        environment = _environment(spec)
        record = resources[name]
        fitted = _resources_of(record)
        sites = strata[strata["environment"] == name].set_index("site_id")
        crop_of = contexts[contexts["environment"] == name].set_index("site_id")
        engine = spec["base_engine"]
        page_tokens: dict[str, Counter[str]] = {}
        line_counts: dict[str, int] = {}
        extents: dict[str, dict[int, tuple[int, int]]] = {}
        by_line: dict[str, dict[int, list[str]]] = {}
        page_spans: dict[str, list[str]] = {}
        for document in sorted(set(block["document_id"].astype(str))):
            page = environment.pages[(document, engine)]
            page_tokens[document] = Counter(cc._tokens(page.stream))
            line_counts[document] = max(len(page.stream.splitlines()), 1)
            extents[document] = _line_extents(page)
            by_line[document] = _spans_by_line(page)
            page_spans[document] = [str(s) for s in page.order]
        ink_cache: dict[str, np.ndarray | None] = {}
        rows: list[np.ndarray] = []
        identifiers: list[str] = []
        for row in block.itertuples(index=False):
            document = str(row.document_id)
            page = environment.pages[(document, engine)]
            site = sites.loc[str(row.lattice_site_id)]
            anchors = gen1.anchors_of(str(site.anchor_ref))
            original, candidate_text = str(row.original_ocr), str(row.candidate_text)
            start, end = int(row.char_start), int(row.char_end)
            before = page.stream[max(0, start - CONTEXT_CHARS) : start]
            after = page.stream[end : end + CONTEXT_CHARS]
            line_index = int(site.line_index)
            extent = extents[document].get(line_index, (start, end))
            crop = crop_of.loc[str(row.lattice_site_id)]
            ink = _ink_of(crop.crop_path, crop.mask_rectangle, ink_cache)
            count = site_counts.get((name, str(row.lattice_site_id), str(row.corrector_source)), 1)
            _names_edit, edit_values = cc.edit_block(original, candidate_text)
            _names_plaus, plaus_values = cc.plausibility_block(original, candidate_text, fitted)
            _names_vis, vis_values = cc.visual_block(ink, original, candidate_text, fitted)
            _names_ctx, ctx_values = cc.context_block(
                original,
                candidate_text,
                page_tokens[document],
                line_counts[document],
            )
            values = [
                *text_block(original, candidate_text, row, extent, len(page.stream), before, after),
                *edit_values,
                *confidence_block(
                    anchors,
                    page,
                    environment.spans,
                    by_line[document].get(line_index, []),
                    _neighbours(page, anchors),
                    page_spans[document],
                    record,
                ),
                *geometry_block(anchors, page, original, record),
                *plaus_values,
                *ctx_values,
                *vis_values,
                *source_block(row, site, int(count)),
            ]
            if len(values) != len(FEATURE_NAMES):
                raise PhaseError(
                    f"{row.candidate_id}: {len(values)} values for {len(FEATURE_NAMES)} features"
                )
            rows.append(np.asarray(values, dtype=np.float64))
            identifiers.append(str(row.candidate_id))
        frame = pd.DataFrame(np.vstack(rows), columns=list(FEATURE_NAMES))
        frame.insert(0, "environment", name)
        frame.insert(0, "candidate_id", identifiers)
        frames.append(frame)
        print(f"  {name}: {len(frame)} candidates ({time.monotonic() - began:.0f}s)", flush=True)
        del environment
    matrix = pd.concat(frames, ignore_index=True)
    if not np.isfinite(matrix[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)).all():
        raise PhaseError("a feature value is not finite")
    return matrix.sort_values(["environment", "candidate_id"], kind="stable").reset_index(drop=True)


# The five audit categories, exactly one per inventoried field.
CAT_AGNOSTIC = "A. generator-agnostic and deployment-available"
CAT_SOURCE = "B. generator/source identity"
CAT_LABEL = "C. downstream outcome leakage"
CAT_UNAVAILABLE = "D. unavailable for one or more candidate sources"
CAT_REVIEW = "E. ambiguous / requires review"

# Fields inventoried and kept out of every primary representation. Each says which category it
# falls in and why it is excluded -- removing a column named "generator" is not an audit.
EXCLUDED_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "name": "engine_id",
        "category": CAT_AGNOSTIC,
        "definition": "which OCR engine read the page",
        "source": "the environment specification",
        "reason": (
            "a property of the reader, not of the candidate, and undefined as a fitted column "
            "under leave-one-environment-out; carried for stratification only"
        ),
    },
    {
        "name": "generator_rank",
        "category": CAT_SOURCE,
        "definition": "the proposing generator's own rank for this candidate",
        "source": "the frozen HY1 candidate table",
        "reason": (
            "a source-specific rank convention: the image corrector is budgeted to its top "
            "candidate, so any rank above zero identifies the frozen current generator exactly"
        ),
    },
    {
        "name": "site_candidate_count",
        "category": CAT_SOURCE,
        "definition": "how many candidates the same generator proposed at this site",
        "source": "the frozen HY1 candidate table",
        "reason": "generator-relative by construction; SGV-XR1 classified it the same way",
    },
    {
        "name": "proposal_stratum",
        "category": CAT_SOURCE,
        "definition": "which proposer named the site: P0 only, image only, or both",
        "source": "HY1's frozen proposal strata",
        "reason": (
            "in the primary population the stratum nearly determines the corrector, because "
            "HY1's routing sends P0-only sites to one generator and image-only sites to the other"
        ),
    },
    {
        "name": "p0_suspicion",
        "category": CAT_SOURCE,
        "definition": "the frozen OCR proposer's suspicion score at this site",
        "source": "HY1's frozen proposal strata",
        "reason": "defined only where P0 proposed, so its missingness is the proposal source",
    },
    {
        "name": "image_suspicion",
        "category": CAT_SOURCE,
        "definition": "the image proposer's suspicion score at this site",
        "source": "HY1's frozen proposal strata",
        "reason": "defined only where the image proposer proposed; same objection",
    },
    {
        "name": "prov_source_g3_edit_aware / prov_source_g7_structural_v2",
        "category": CAT_SOURCE,
        "definition": "the legacy verifier's generator one-hots",
        "source": "the frozen SGV1 verifier representation",
        "reason": "SGV-XR1 measured these as not inert, which is why zero-shot transfer failed",
    },
    {
        "name": "prov_operation_*",
        "category": CAT_UNAVAILABLE,
        "definition": "the legacy verifier's six operation-label one-hots",
        "source": "the frozen SGV1 verifier representation",
        "reason": (
            "the frozen pipeline produces an operation label only for its own generators; "
            "SGV-XR1 recorded them as requiring unavailable metadata"
        ),
    },
    {
        "name": "outcome / is_harmful / beneficial / exact",
        "category": CAT_LABEL,
        "definition": "the frozen project outcome taxonomy and its derived flags",
        "source": "SGV14 labelling",
        "reason": "the target; reading it as a feature would be circular",
    },
    {
        "name": "d_before / d_after",
        "category": CAT_LABEL,
        "definition": "edit distance to the transcription before and after the edit",
        "source": "SGV14 labelling",
        "reason": "computed against ground truth; unavailable at deployment and label-equivalent",
    },
    {
        "name": "ground-truth text",
        "category": CAT_LABEL,
        "definition": "the transcription of the span",
        "source": "the corpus annotation",
        "reason": "never reaches a feature builder; the leakage suite asserts it from the AST",
    },
)


def _missingness(matrix: pd.DataFrame, population: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per feature, how often its family's missing indicator fires, and per corrector."""
    primary = population[population["population"] == PRIMARY_POPULATION]
    joined = matrix.merge(
        primary[["candidate_id", "corrector_source"]], on="candidate_id", how="inner"
    )
    out: dict[str, dict[str, float]] = {}
    for indicator in ("conf_missing", "geom_missing", "vis_missing"):
        block = {"overall": float(joined[indicator].mean())}
        for source, group in joined.groupby("corrector_source", sort=True):
            block[str(source)] = float(group[indicator].mean())
        block["max_absolute_difference_between_correctors"] = float(
            max(
                abs(block[a] - block[b])
                for a in CORRECTORS
                for b in CORRECTORS
                if a in block and b in block
            )
            if all(c in block for c in CORRECTORS)
            else 0.0
        )
        out[indicator] = block
    return out


def run_features() -> int:
    """Sections 9-13: the feature audit, the missing-feature policy and the R1-R4 matrices."""
    started = time.monotonic()
    _require(FITTED_RESOURCES, "resources")
    for path in (
        FEATURE_MATRIX,
        FEATURE_AUDIT,
        FEATURE_REGISTRY,
        MISSING_FEATURE_POLICY,
        R0_LEGACY,
        R1_REPRESENTATION,
        R2_REPRESENTATION,
        R3_REPRESENTATION,
        R4_REPRESENTATION,
    ):
        _forbid(path)
    matrix = build_features()
    _write_parquet_once(FEATURE_MATRIX, matrix)
    population = pd.read_parquet(CANDIDATE_POPULATION)
    for representation, path in (
        (R1, R1_REPRESENTATION),
        (R2, R2_REPRESENTATION),
        (R3, R3_REPRESENTATION),
        (R4, R4_REPRESENTATION),
    ):
        columns = columns_for(representation)
        _write_parquet_once(path, matrix[["candidate_id", "environment", *columns]])
    missingness = _missingness(matrix, population)
    rows: list[dict[str, Any]] = []
    for name in FEATURE_NAMES:
        family = family_of(name)
        is_source = family == FAM_SOURCE
        indicator = {FAM_CONF: "conf_missing", FAM_GEOM: "geom_missing", FAM_VIS: "vis_missing"}
        category = CAT_SOURCE if is_source else (CAT_REVIEW if family == FAM_CONF else CAT_AGNOSTIC)
        rows.append(
            {
                "name": name,
                "family": family,
                "definition": FEATURE_DEFINITIONS[name],
                "source": FEATURE_SOURCE[family],
                "category": category,
                "candidate_sources_available": list(CORRECTORS),
                "deployment_available": True,
                "generator_identity": is_source,
                "label_leakage": False,
                "missingness": missingness.get(indicator.get(family, ""), {}).get("overall", 0.0),
                "allowed_primary": not is_source,
                "reason": (
                    "explicit generator provenance; the diagnostic comparator only"
                    if is_source
                    else (
                        "the engine's native confidence scale differs per environment, so "
                        "cross-source equivalence needed review; resolved by normalizing every "
                        "value against the same environment's ADAPTATION distribution"
                        if family == FAM_CONF
                        else "computed identically for every candidate of every generator"
                    )
                ),
            }
        )
    for excluded in EXCLUDED_FIELDS:
        rows.append(
            {
                **excluded,
                "family": "excluded",
                "candidate_sources_available": (
                    [] if excluded["category"] == CAT_UNAVAILABLE else list(CORRECTORS)
                ),
                "deployment_available": excluded["category"] != CAT_LABEL,
                "generator_identity": excluded["category"] == CAT_SOURCE,
                "label_leakage": excluded["category"] == CAT_LABEL,
                "missingness": None,
                "allowed_primary": False,
            }
        )
    by_category = Counter(str(row["category"]) for row in rows)
    _write_json_once(
        FEATURE_AUDIT,
        {
            **_envelope("feature_audit"),
            "features": rows,
            "inventoried_fields": len(rows),
            "produced_features": len(FEATURE_NAMES),
            "allowed_primary": sum(1 for row in rows if row["allowed_primary"]),
            "by_category": {k: int(v) for k, v in sorted(by_category.items())},
            "by_family": {
                family: sum(1 for name in FEATURE_NAMES if family_of(name) == prefix)
                for family, prefix in FAMILIES.items()
            },
            "frozen_before_any_model": True,
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        FEATURE_REGISTRY,
        {
            **_envelope("feature_registry"),
            "order": list(FEATURE_NAMES),
            "representations": {
                representation: {
                    "columns": columns_for(representation),
                    "column_count": len(columns_for(representation)),
                    "families": list(REPRESENTATION_FAMILIES[representation]),
                    "carries_generator_identity": FAM_SOURCE
                    in REPRESENTATION_FAMILIES[representation],
                }
                for representation in REPRESENTATIONS
            },
            "ablations": {
                name: {
                    "drops": list(drop),
                    "columns": len(columns_for(R3, drop)),
                }
                for name, drop in ABLATIONS.items()
            },
            "matrix_sha256": file_sha256(FEATURE_MATRIX),
            "rows": len(matrix),
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        MISSING_FEATURE_POLICY,
        {
            **_envelope("missing_feature_policy"),
            "rule": (
                "one declared rule per family, fixed before any fit. No slot is filled with zero "
                "unless zero is the semantically correct value and was predeclared"
            ),
            "policies": {
                FAM_CONF: {
                    "when": "no anchor span reports a recognition confidence",
                    "action": "explicit missing indicator; z-scores take 0 (the ADAPTATION mean) "
                    "and the percentile takes 0.5 (the ADAPTATION median)",
                    "zero_is_semantic": True,
                },
                FAM_GEOM: {
                    "when": "no anchor span carries a detector box, or the page has no size",
                    "action": "explicit missing indicator; every value slot takes this "
                    "environment's ADAPTATION median for that feature, never zero",
                    "zero_is_semantic": False,
                },
                FAM_VIS: {
                    "when": "the site has no usable crop region",
                    "action": "SGV1's frozen convention, inherited unchanged: vis_missing fires "
                    "and the evidence slots are zero, which is the semantic value for "
                    "'no ink evidence'",
                    "zero_is_semantic": True,
                },
                FAM_TEXT: {
                    "when": "never",
                    "action": "always computable",
                    "zero_is_semantic": True,
                },
                FAM_EDIT: {
                    "when": "never",
                    "action": "always computable",
                    "zero_is_semantic": True,
                },
                FAM_PLAUS: {
                    "when": "never",
                    "action": "the language model backs off; the lexicon returns absence",
                    "zero_is_semantic": True,
                },
                FAM_CTX: {"when": "never", "action": "always computable", "zero_is_semantic": True},
            },
            "missingness": missingness,
            "missingness_encodes_generator_identity": bool(
                any(
                    block["max_absolute_difference_between_correctors"] > 0.05
                    for block in missingness.values()
                )
            ),
            "note": (
                "every missing indicator is a property of the SITE -- whether the engine gave the "
                "span a confidence, a box, or a croppable region -- so two correctors at the same "
                "site carry the same indicator. Any difference between correctors is a difference "
                "between the site populations they were routed to, and is reported as such"
            ),
            "uses_ground_truth": False,
        },
    )
    compatibility = cc_read_json(xr1.FEATURE_COMPATIBILITY)
    audit = next(iter(compatibility["column_audit"].values()))["columns"]
    status = Counter(str(row["status"]) for row in audit)
    _write_json_once(
        R0_LEGACY,
        {
            **_envelope("r0_legacy_representation"),
            "role": "historical within-distribution reference; never scored on RL1 candidates",
            "source": _relative(xr1.FEATURE_COMPATIBILITY),
            "verifier": cc_read_json(FROZEN_CONFIGURATION)["legacy_verifier"],
            "columns_by_status": {k: int(v) for k, v in sorted(status.items())},
            "undefined_on_a_new_generator": [
                str(row["column"])
                for row in audit
                if row["status"] in ("candidate_source_dependent", "requires_unavailable_metadata")
            ],
            "zero_shot_feature_compatible": compatibility["zero_shot_feature_compatible"],
            "inert_by_family": compatibility["inert_by_family"],
            "why_not_evaluated_here": (
                "SGV-XR1 measured that the frozen verifier's generator-source and operation "
                "columns are not inert: substituting their admissible values moves the frozen "
                "score. A Qwen3-VL candidate has no scientifically valid value for either, so "
                "any number produced by scoring one would be an artifact of the encoding chosen, "
                "not a measurement. RL1 reports R0 as the historical reference and builds its "
                "own representation instead"
            ),
            "uses_ground_truth": False,
        },
    )
    print(
        f"features: {len(matrix)} candidates x {len(FEATURE_NAMES)} features, "
        f"{len(columns_for(R3))} in R3 ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 14-15: splits


@dataclass(slots=True)
class Fold:
    """One train/evaluate partition, with the metadata that says what it holds out."""

    split: str
    name: str
    train: np.ndarray
    test: np.ndarray
    meta: dict[str, Any]


def _folds_column(frame: pd.DataFrame, assignment: dict[str, int] | None = None) -> np.ndarray:
    """Each row's document fold. A document with no assigned fold is an error, not a zero."""
    folds = assignment if assignment is not None else cc_read_json(GROUP_REGISTRY)["document_folds"]
    mapped = frame["document_id"].astype(str).map(folds)
    if mapped.isna().any():
        unknown = sorted(set(frame.loc[mapped.isna(), "document_id"].astype(str)))
        raise PhaseError(f"{len(unknown)} documents carry no fold assignment: {unknown[:4]}")
    return mapped.to_numpy(dtype=np.int64)


def _sites_with(frame: pd.DataFrame, corrector: str, present: bool) -> set[str]:
    direct = frame["corrector_source"].astype(str) == corrector
    listed = frame["corrector_sources"].astype(str).str.contains(corrector, regex=False)
    mask = direct | listed
    return set(frame.loc[mask if present else ~mask, "site_group"].astype(str))


def _other(corrector: str) -> str:
    return C1 if corrector == C0 else C0


def build_folds(
    split: str, frame: pd.DataFrame, assignment: dict[str, int] | None = None
) -> list[Fold]:
    """Every fold of one pre-registered split, as index arrays into ``frame``.

    Whenever an environment is held out the document partition is held out with it: the benchmark
    is matched-source, so the same page is read by every engine and leaving out an engine alone
    would leave the page in training.
    """
    folds = _folds_column(frame, assignment)
    corrector = frame["corrector_source"].astype(str).to_numpy()
    environment = frame["environment"].astype(str).to_numpy()
    stratum = frame["proposal_stratum"].astype(str).to_numpy()
    site = frame["site_group"].astype(str).to_numpy()
    index = np.arange(len(frame))
    out: list[Fold] = []
    if split == SPLIT_A:
        for fold in range(DOCUMENT_FOLDS):
            test = index[folds == fold]
            out.append(
                Fold(split, f"fold_{fold}", index[folds != fold], test, {"document_fold": fold})
            )
        return out
    if split in (SPLIT_B, SPLIT_B_DOC, SPLIT_D):
        for held in CORRECTORS:
            other = _other(held)
            held_sites = _sites_with(frame, held, True)
            other_sites = _sites_with(frame, other, True)
            trainable = (corrector == other) & ~np.isin(site, list(held_sites))
            testable = (corrector == held) & ~np.isin(site, list(other_sites))
            if split == SPLIT_B:
                out.append(
                    Fold(
                        split,
                        f"train_{other}__test_{held}",
                        index[trainable],
                        index[testable],
                        {"held_out_generator": held, "training_generator": other},
                    )
                )
            elif split == SPLIT_B_DOC:
                for fold in range(DOCUMENT_FOLDS):
                    out.append(
                        Fold(
                            split,
                            f"train_{other}__test_{held}__fold_{fold}",
                            index[trainable & (folds != fold)],
                            index[testable & (folds == fold)],
                            {
                                "held_out_generator": held,
                                "training_generator": other,
                                "document_fold": fold,
                            },
                        )
                    )
            else:
                for name in sorted(set(environment.tolist())):
                    out.append(
                        Fold(
                            split,
                            f"train_{other}__test_{held}__{name}",
                            index[trainable & (environment != name)],
                            index[testable & (environment == name)],
                            {"held_out_generator": held, "held_out_environment": name},
                        )
                    )
        return out
    if split == SPLIT_B_OVERLAP:
        overlap = stratum == hy1.S_OVERLAP
        for held in CORRECTORS:
            other = _other(held)
            for fold in range(DOCUMENT_FOLDS):
                out.append(
                    Fold(
                        split,
                        f"train_{other}__test_{held}__fold_{fold}",
                        index[overlap & (corrector == other) & (folds != fold)],
                        index[overlap & (corrector == held) & (folds == fold)],
                        {
                            "held_out_generator": held,
                            "training_generator": other,
                            "document_fold": fold,
                            "stratum": hy1.S_OVERLAP,
                        },
                    )
                )
        return out
    if split == SPLIT_W:
        for held in CORRECTORS:
            for fold in range(DOCUMENT_FOLDS):
                out.append(
                    Fold(
                        split,
                        f"train_{held}__test_{held}__fold_{fold}",
                        index[(corrector == held) & (folds != fold)],
                        index[(corrector == held) & (folds == fold)],
                        {
                            "training_generator": held,
                            "evaluated_generator": held,
                            "document_fold": fold,
                        },
                    )
                )
        return out
    if split == SPLIT_C:
        for name in sorted(set(environment.tolist())):
            for fold in range(DOCUMENT_FOLDS):
                out.append(
                    Fold(
                        split,
                        f"{name}__fold_{fold}",
                        index[(environment != name) & (folds != fold)],
                        index[(environment == name) & (folds == fold)],
                        {"held_out_environment": name, "document_fold": fold},
                    )
                )
        return out
    raise PhaseError(f"unknown split {split!r}")


def run_splits() -> int:
    """Section 14-15: the split registry, with every fold's sizes and what it holds out."""
    started = time.monotonic()
    _require(FEATURE_MATRIX, "features")
    _forbid(SPLIT_REGISTRY)
    population = pd.read_parquet(CANDIDATE_POPULATION)
    primary = population[population["population"] == PRIMARY_POPULATION].reset_index(drop=True)
    registry: dict[str, Any] = {}
    for split in SPLITS:
        folds = build_folds(split, primary)
        registry[split] = {
            "folds": [
                {
                    "fold": fold.name,
                    "train_rows": int(fold.train.size),
                    "test_rows": int(fold.test.size),
                    "train_documents": int(
                        primary.iloc[fold.train]["document_id"].nunique() if fold.train.size else 0
                    ),
                    "test_documents": int(
                        primary.iloc[fold.test]["document_id"].nunique() if fold.test.size else 0
                    ),
                    "train_sites": int(
                        primary.iloc[fold.train]["site_group"].nunique() if fold.train.size else 0
                    ),
                    "shared_documents": len(
                        set(primary.iloc[fold.train]["document_id"])
                        & set(primary.iloc[fold.test]["document_id"])
                    ),
                    "shared_sites": len(
                        set(primary.iloc[fold.train]["site_group"])
                        & set(primary.iloc[fold.test]["site_group"])
                    ),
                    "shared_edit_groups": len(
                        set(primary.iloc[fold.train]["edit_group"])
                        & set(primary.iloc[fold.test]["edit_group"])
                    ),
                    "test_harmful": int(primary.iloc[fold.test]["is_harmful"].sum()),
                    "test_beneficial": int(primary.iloc[fold.test]["beneficial"].sum()),
                    **fold.meta,
                }
                for fold in folds
            ],
            "fold_count": len(folds),
            "holds_out_documents": split != SPLIT_B,
            "holds_out_a_generator": split in (SPLIT_B, SPLIT_B_DOC, SPLIT_B_OVERLAP, SPLIT_D),
            "evaluates_the_training_generator": split == SPLIT_W,
            "holds_out_an_environment": split in (SPLIT_C, SPLIT_D),
        }
    leaks = {
        split: {
            "sites_crossing": sum(int(f["shared_sites"]) for f in registry[split]["folds"]),
            "atomic_edits_crossing": sum(
                int(f["shared_edit_groups"]) for f in registry[split]["folds"]
            ),
        }
        for split in SPLITS
    }
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_analysis_envelope("split_registry"),
            "splits": registry,
            "leakage": leaks,
            "no_site_crosses_a_document_held_out_split": all(
                leaks[split]["sites_crossing"] == 0
                for split in (SPLIT_A, SPLIT_B_DOC, SPLIT_B_OVERLAP, SPLIT_C, SPLIT_W)
            ),
            "split_b_shares_documents_by_design": (
                "Split B asks whether reliability transfers between generators on the same pages, "
                "so it holds out the generator alone. Its sites are disjoint by construction -- a "
                "site carrying candidates from both generators enters neither side -- and "
                "Split B_doc repeats the test with the document partition held out as well"
            ),
            "primary_population": PRIMARY_POPULATION,
        },
    )
    print(
        f"splits: {sum(len(registry[s]['folds']) for s in SPLITS)} folds over {len(SPLITS)} splits "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 11-15: models


def _make_model(kind: str) -> Any:
    """The closed model table. Hyperparameters are the repository's, reused without a search."""
    if kind == M0:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=LOGISTIC_C,
            max_iter=LOGISTIC_MAX_ITER,
            class_weight="balanced",
            random_state=FIT_SEED,
        )
    if kind == M1:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=BOOSTING_ITERATIONS, class_weight="balanced", random_state=FIT_SEED
        )
    raise PhaseError(f"unknown model {kind!r}")


def fit_and_score(kind: str, train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    """One fitted scaler and classifier, applied to the held-out block. P(positive class).

    A fold whose training rows carry a single class cannot produce a score, and a constant 0.5
    would be a measurement rather than an absence, so it returns NaN and the metric layer drops
    it. The scaler is fitted on the training rows alone; the held-out block is never pooled into
    a statistic used to transform it.
    """
    from sklearn.preprocessing import StandardScaler

    if train.size == 0 or test.size == 0 or len(set(labels.tolist())) < 2:
        return np.full(test.shape[0], np.nan)
    scaler = StandardScaler().fit(train)
    model = _make_model(kind).fit(scaler.transform(train), labels.astype(int))
    classes = list(model.classes_)
    if 1 not in classes:
        return np.full(test.shape[0], np.nan)
    return np.asarray(model.predict_proba(scaler.transform(test))[:, classes.index(1)], float)


def _matrix_for(frame: pd.DataFrame, matrix: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    joined = frame[["candidate_id"]].merge(
        matrix[["candidate_id", *columns]], on="candidate_id", how="left", validate="one_to_one"
    )
    values = joined[list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise PhaseError("a feature value reaching a model is not finite")
    return values


def score_split(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    columns: Sequence[str],
    kind: str,
    split: str,
    labels: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Out-of-fold scores for one representation, model and split, both targets."""
    design = _matrix_for(frame, matrix, columns)
    truth = labels or {
        T_HARM: frame["is_harmful"].to_numpy(dtype=bool),
        T_BENEFIT: frame["beneficial"].to_numpy(dtype=bool),
    }
    rows: list[pd.DataFrame] = []
    for fold in build_folds(split, frame):
        if fold.test.size == 0:
            continue
        block = pd.DataFrame(
            {
                "fold": fold.name,
                "candidate_id": frame.iloc[fold.test]["candidate_id"].to_numpy(),
            }
        )
        for target in TARGETS:
            block[f"score_{target}"] = fit_and_score(
                kind, design[fold.train], truth[target][fold.train], design[fold.test]
            )
        for key, value in fold.meta.items():
            block[key] = value
        rows.append(block)
    if not rows:
        return pd.DataFrame(columns=["fold", "candidate_id", "score_harm", "score_benefit"])
    return pd.concat(rows, ignore_index=True)


def _fit_jobs() -> list[dict[str, Any]]:
    """Every pre-registered representation x model x split x population, fixed before any fit."""
    jobs: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for split in MODEL_SPLITS[M0]:
            jobs.append(
                {
                    "population": PRIMARY_POPULATION,
                    "representation": representation,
                    "model": M0,
                    "split": split,
                    "drop": (),
                }
            )
        for split in MODEL_SPLITS[M1]:
            if split == SPLIT_C and representation not in M1_SPLIT_C_REPRESENTATIONS:
                continue
            jobs.append(
                {
                    "population": PRIMARY_POPULATION,
                    "representation": representation,
                    "model": M1,
                    "split": split,
                    "drop": (),
                }
            )
        for population in (U4, U3):
            jobs.append(
                {
                    "population": population,
                    "representation": representation,
                    "model": M0,
                    "split": SPLIT_A,
                    "drop": (),
                }
            )
    for name, drop in ABLATIONS.items():
        if name == A0:
            continue
        jobs.append(
            {
                "population": PRIMARY_POPULATION,
                "representation": name,
                "model": M0,
                "split": SPLIT_A,
                "drop": drop,
            }
        )
    return jobs


def run_fit() -> int:
    """Sections 15-16: every pre-registered fit, scored strictly out of fold."""
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    for path in (SCORES, MODEL_REGISTRY, HYPERPARAMETER_REGISTRY, TRAINING_REGISTRY):
        _forbid(path)
    population = pd.read_parquet(CANDIDATE_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    jobs = _fit_jobs()
    frames: list[pd.DataFrame] = []
    training: list[dict[str, Any]] = []
    for job in jobs:
        began = time.monotonic()
        frame = population[population["population"] == job["population"]].reset_index(drop=True)
        base = job["representation"] if job["representation"] in REPRESENTATIONS else R3
        columns = columns_for(base, job["drop"])
        scores = score_split(frame, matrix, columns, job["model"], job["split"])
        scores.insert(0, "split", job["split"])
        scores.insert(0, "model", job["model"])
        scores.insert(0, "representation", job["representation"])
        scores.insert(0, "population", job["population"])
        frames.append(scores)
        training.append(
            {
                **{k: v for k, v in job.items() if k != "drop"},
                "dropped_families": list(job["drop"]),
                "columns": len(columns),
                "folds": int(scores["fold"].nunique()) if len(scores) else 0,
                "scored_rows": len(scores),
                "unscored_folds": int(scores["score_harm"].isna().any()) if len(scores) else 0,
                "runtime_seconds": round(time.monotonic() - began, 3),
            }
        )
        print(
            f"  {job['population']} {job['representation']} {job['model']} {job['split']}: "
            f"{len(scores)} rows ({time.monotonic() - began:.1f}s)",
            flush=True,
        )
    table = pd.concat(frames, ignore_index=True)
    order = ["population", "representation", "model", "split", "fold", "candidate_id"]
    table = table.sort_values(order, kind="stable").reset_index(drop=True)
    _write_parquet_once(SCORES, table)
    _write_json_once(
        MODEL_REGISTRY,
        {
            **_envelope("model_registry"),
            "models": {
                M0: {
                    "class": "sklearn.linear_model.LogisticRegression",
                    "role": "primary; interpretable and stable across representations",
                    "preprocessing": "sklearn.preprocessing.StandardScaler, fitted on train rows",
                },
                M1: {
                    "class": "sklearn.ensemble.HistGradientBoostingClassifier",
                    "role": "pre-registered robustness arm on the primary splits",
                    "preprocessing": "the same scaler, so the two arms see identical inputs",
                },
            },
            "primary_model": PRIMARY_MODEL,
            "two_heads": "one classifier per target; harm and benefit are never collapsed",
            "no_model_search": True,
            "no_new_architecture": True,
            "fine_tuned_here": False,
            "new_or_larger_model": False,
            "uses_ground_truth": True,
        },
    )
    _write_json_once(
        HYPERPARAMETER_REGISTRY,
        {
            **_envelope("hyperparameter_registry"),
            M0: {
                "C": LOGISTIC_C,
                "max_iter": LOGISTIC_MAX_ITER,
                "class_weight": "balanced",
                "random_state": FIT_SEED,
            },
            M1: {
                "max_iter": BOOSTING_ITERATIONS,
                "class_weight": "balanced",
                "random_state": FIT_SEED,
            },
            "source": (
                "the repository's published configurations, reused unchanged: SGV5's reliability "
                "model table, which took the boosted configuration from the SGV1 phase-3 study "
                "and the logistic configuration from the SGV1 verifier pilot"
            ),
            "selection_scope": "none; no grid was searched and no held-out label chose a value",
            "tuned_per_held_out_generator": False,
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        TRAINING_REGISTRY,
        {
            **_analysis_envelope("training_registry"),
            "jobs": training,
            "job_count": len(training),
            "scored_rows": len(table),
            "fits": int(sum(job["folds"] * len(TARGETS) for job in training)),
            "scaler_scope": "training rows of the fold only",
            "held_out_labels_never_fit": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"fit: {len(training)} jobs, {len(table)} scored rows ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 16-18: metrics


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    from ocr_risk.metrics.discrimination import roc_auc

    mask = np.isfinite(scores)
    if not mask.any():
        return float("nan")
    return float(roc_auc(scores[mask], positive[mask].astype(float)))


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    from ocr_risk.metrics.discrimination import average_precision as ap

    mask = np.isfinite(scores)
    if not mask.any() or positive[mask].sum() == 0:
        return float("nan")
    return float(ap(scores[mask], positive[mask].astype(float)))


def baseline_score(frame: pd.DataFrame) -> np.ndarray:
    """The source-free single-feature comparator: normalized character edit distance.

    No fitting, no source, no label. It is the honest "you could have done this with one line"
    reference the breadth criterion is measured against.
    """
    from ocr_risk.metrics.text import levenshtein

    return np.asarray(
        [
            levenshtein(str(o), str(y)) / max(len(str(o)), 1)
            for o, y in zip(frame["original_ocr"], frame["candidate_text"], strict=True)
        ],
        dtype=np.float64,
    )


def load_scores() -> pd.DataFrame:
    """The score table joined to the population labels, once, for every analysis phase."""
    table = pd.read_parquet(SCORES)
    population = pd.read_parquet(CANDIDATE_POPULATION)
    columns = [
        "population",
        "candidate_id",
        "environment",
        "corpus",
        "base_engine",
        "domain",
        "document_id",
        "site_group",
        "proposal_stratum",
        "corrector_source",
        "multi_source",
        "error_kind",
        "outcome",
        "is_harmful",
        "beneficial",
        "exact",
    ]
    return table.merge(
        population[columns], on=["population", "candidate_id"], validate="many_to_one"
    )


def cell(
    scores: pd.DataFrame,
    population: str,
    representation: str,
    model: str,
    split: str,
) -> pd.DataFrame:
    block = scores[
        (scores["population"] == population)
        & (scores["representation"] == representation)
        & (scores["model"] == model)
        & (scores["split"] == split)
    ]
    return block.reset_index(drop=True)


def discrimination(frame: pd.DataFrame) -> dict[str, Any]:
    """Both targets' AUROC and average precision on one scored block, with its prevalences."""
    out: dict[str, Any] = {"rows": len(frame)}
    for target, label in ((T_HARM, "is_harmful"), (T_BENEFIT, "beneficial")):
        positive = frame[label].to_numpy(dtype=bool)
        score = frame[f"score_{target}"].to_numpy(dtype=np.float64)
        out[f"{target}_auroc"] = auroc(score, positive)
        out[f"{target}_average_precision"] = average_precision(score, positive)
        out[f"{target}_prevalence"] = float(positive.mean()) if positive.size else float("nan")
        out[f"{target}_positives"] = int(positive.sum())
    return out


def by_group(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    return {
        str(name): discrimination(group)
        for name, group in frame.groupby(column, sort=True)
        if len(group)
    }


def pairwise_ranking_accuracy(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    """P(a beneficial candidate outscores a harmful one) inside the same document.

    Restricted to comparable strata rather than pooled across pages, because two candidates on
    different documents are not a decision anyone makes.
    """
    wins = ties = total = 0
    for _document, group in frame.groupby("document_id", sort=True):
        score = group[f"score_{target}"].to_numpy(dtype=np.float64)
        good = group["beneficial"].to_numpy(dtype=bool)
        bad = group["is_harmful"].to_numpy(dtype=bool)
        mask = np.isfinite(score)
        positives = score[mask & good]
        negatives = score[mask & bad]
        if positives.size == 0 or negatives.size == 0:
            continue
        comparison = positives[:, None] - negatives[None, :]
        sign = 1.0 if target == T_BENEFIT else -1.0
        wins += int((sign * comparison > 0).sum())
        ties += int((comparison == 0).sum())
        total += int(comparison.size)
    return {
        "pairs": total,
        "documents_with_both": int(
            sum(
                1
                for _d, g in frame.groupby("document_id", sort=True)
                if g["beneficial"].any() and g["is_harmful"].any()
            )
        ),
        "accuracy": _ratio(wins + 0.5 * ties, total),
    }


def _document_blocks(frame: pd.DataFrame) -> list[tuple[str, list[np.ndarray]]]:
    """Row indices grouped by document inside each environment, in a fixed order."""
    out: list[tuple[str, list[np.ndarray]]] = []
    for environment in sorted(set(frame["environment"].astype(str))):
        block = frame[frame["environment"] == environment]
        documents = sorted(set(block["document_id"].astype(str)))
        out.append(
            (
                environment,
                [
                    np.flatnonzero(
                        (frame["environment"].to_numpy(str) == environment)
                        & (frame["document_id"].to_numpy(str) == document)
                    )
                    for document in documents
                ],
            )
        )
    return out


def paired_auroc_comparison(
    frame: pd.DataFrame, a: str, b: str, label: str, name: str, family: str
) -> dict[str, Any]:
    """Document-clustered paired bootstrap on the AUROC difference between two score columns.

    The resampling unit is the document, drawn with replacement inside each environment, and the
    same draw serves both arms so the difference is paired. Candidate rows are never the
    inferential unit: tokens inside a page are strongly correlated and a row-level interval would
    be dishonestly narrow.
    """
    from ocr_risk.stats.bootstrap import _bootstrap_p_value

    positive = frame[label].to_numpy(dtype=bool)
    score_a = frame[a].to_numpy(dtype=np.float64)
    score_b = frame[b].to_numpy(dtype=np.float64)
    effect = auroc(score_a, positive) - auroc(score_b, positive)
    blocks = _document_blocks(frame)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    picks = {
        environment: generator.integers(
            0, len(documents), size=(BOOTSTRAP_RESAMPLES, len(documents))
        )
        for environment, documents in blocks
    }
    for resample in range(BOOTSTRAP_RESAMPLES):
        chosen: list[np.ndarray] = []
        for environment, documents in blocks:
            for position in picks[environment][resample]:
                chosen.append(documents[position])
        index = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
        draws[resample] = auroc(score_a[index], positive[index]) - auroc(
            score_b[index], positive[index]
        )
    finite = draws[np.isfinite(draws)]
    per_environment = {
        environment: auroc(score_a[index], positive[index]) - auroc(score_b[index], positive[index])
        for environment, index in (
            (name_, np.concatenate(documents)) for name_, documents in blocks
        )
    }
    improved = [v for v in per_environment.values() if np.isfinite(v) and v > 0]
    worsened = [v for v in per_environment.values() if np.isfinite(v) and v < 0]
    return {
        "comparison": name,
        "family": family,
        "effect": float(effect),
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        "p_value": float(_bootstrap_p_value(finite)) if finite.size else float("nan"),
        "environments_improved": len(improved),
        "environments_worsened": len(worsened),
        "per_environment": {k: float(v) for k, v in per_environment.items()},
        "units": len(frame),
        "documents": int(frame["document_id"].nunique()),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


# ------------------------------------------------------------------ section 11: implicit source


def run_sourceleak() -> int:
    """Section 11: how much of the generator survives in a representation that never names it.

    A diagnostic, not a research arm. Removing a column called "generator" is not an audit: what
    matters is whether the remaining columns identify the source anyway, and the honest thing is
    to measure it and record it as a property of the representation. Nothing here removes a
    feature; the pre-registered representations are already frozen.
    """
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    _forbid(IMPLICIT_SOURCE_LEAKAGE)
    population = pd.read_parquet(CANDIDATE_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    frame = population[population["population"] == PRIMARY_POPULATION].reset_index(drop=True)
    target = (frame["corrector_source"].astype(str) == C1).to_numpy(dtype=bool)
    results: dict[str, Any] = {}
    for representation in AGNOSTIC_REPRESENTATIONS:
        columns = columns_for(representation)
        design = _matrix_for(frame, matrix, columns)
        predicted = np.full(len(frame), np.nan)
        for fold in build_folds(SPLIT_A, frame):
            predicted[fold.test] = fit_and_score(
                M0, design[fold.train], target[fold.train], design[fold.test]
            )
        accuracy = float(((predicted >= 0.5) == target)[np.isfinite(predicted)].mean())
        confusion = {
            str(source): {
                "rows": int(mask.sum()),
                "predicted_c1_share": float((predicted[mask] >= 0.5).mean()),
            }
            for source, mask in (
                (name, (frame["corrector_source"].astype(str) == name).to_numpy(dtype=bool))
                for name in CORRECTORS
            )
        }
        single = sorted(
            (
                {
                    "feature": name,
                    "source_auroc": auroc(design[:, position], target),
                }
                for position, name in enumerate(columns)
            ),
            key=lambda row: -abs(float(row["source_auroc"]) - 0.5),
        )
        results[representation] = {
            "columns": len(columns),
            "source_auroc": auroc(predicted, target),
            "source_accuracy": accuracy,
            "majority_class_accuracy": float(max(target.mean(), 1.0 - target.mean())),
            "per_source": confusion,
            "most_source_predictive_features": single[:15],
        }
        print(
            f"  {representation}: source AUROC {results[representation]['source_auroc']:.4f}",
            flush=True,
        )
    _write_json_once(
        IMPLICIT_SOURCE_LEAKAGE,
        {
            **_analysis_envelope("implicit_source_leakage"),
            "target": "corrector_source == c1_image_corrector",
            "population": PRIMARY_POPULATION,
            "split": SPLIT_A,
            "model": M0,
            "results": results,
            "is_a_research_arm": False,
            "removes_a_feature": False,
            "interpretation": (
                "a high source AUROC does not mean the representation encodes generator identity "
                "on purpose. The two generators were routed to different proposal strata, so "
                "their candidates differ in edit shape, length and site geometry, and a model "
                "that reads those differences reads real candidate properties. What it does mean "
                "is that the mixed-source result must not be read as evidence of transfer: the "
                "leave-one-generator-out split is the test that separates the two"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"sourceleak: {len(results)} representations ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ sections 17-22, 26-29


def _score_quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {}
    points = (0.05, 0.25, 0.5, 0.75, 0.95)
    return {
        "mean": float(finite.mean()),
        **{f"q{int(p * 100):02d}": float(np.quantile(finite, p)) for p in points},
    }


def run_discriminate() -> int:
    """Sections 17, 20-22 and 26-29: discrimination pooled and in every declared stratum."""
    started = time.monotonic()
    _require(SCORES, "fit")
    for path in (
        MIXED_SOURCE_RESULTS,
        HARM_DISCRIMINATION,
        BENEFIT_DISCRIMINATION,
        AVERAGE_PRECISION,
        SOURCE_STRATIFIED,
        PROPOSAL_STRATIFIED,
        ERROR_TYPE_ANALYSIS,
        ENGINE_ANALYSIS,
        DOMAIN_ANALYSIS,
    ):
        _forbid(path)
    scores = load_scores()
    population = pd.read_parquet(CANDIDATE_POPULATION)
    primary = population[population["population"] == PRIMARY_POPULATION].reset_index(drop=True)
    baseline = pd.DataFrame(
        {
            "candidate_id": primary["candidate_id"],
            "score_harm": baseline_score(primary),
            "score_benefit": -baseline_score(primary),
        }
    )
    baseline_block = primary.merge(baseline, on="candidate_id", validate="one_to_one")
    mixed: dict[str, Any] = {}
    for population_name in POPULATIONS:
        for model in MODELS:
            for representation in (*REPRESENTATIONS, *(k for k in ABLATIONS if k != A0)):
                block = cell(scores, population_name, representation, model, SPLIT_A)
                if block.empty:
                    continue
                mixed.setdefault(population_name, {}).setdefault(model, {})[representation] = {
                    **discrimination(block),
                    "per_environment": {
                        name: {
                            "harm_auroc": row["harm_auroc"],
                            "benefit_auroc": row["benefit_auroc"],
                            "rows": row["rows"],
                        }
                        for name, row in by_group(block, "environment").items()
                    },
                    "pairwise_ranking_accuracy": pairwise_ranking_accuracy(block, T_BENEFIT),
                }
    baseline_row = {
        **discrimination(baseline_block),
        "per_environment": {
            name: {"harm_auroc": row["harm_auroc"], "benefit_auroc": row["benefit_auroc"]}
            for name, row in by_group(baseline_block, "environment").items()
        },
    }
    primary_cell = cell(scores, PRIMARY_POPULATION, PRIMARY_REPRESENTATION, PRIMARY_MODEL, SPLIT_A)
    breadth = {
        name: {
            "r3_harm_auroc": mixed[PRIMARY_POPULATION][PRIMARY_MODEL][PRIMARY_REPRESENTATION][
                "per_environment"
            ][name]["harm_auroc"],
            "baseline_harm_auroc": baseline_row["per_environment"][name]["harm_auroc"],
        }
        for name in sorted(baseline_row["per_environment"])
    }
    for row in breadth.values():
        row["improved"] = bool(
            np.isfinite(row["r3_harm_auroc"])
            and np.isfinite(row["baseline_harm_auroc"])
            and row["r3_harm_auroc"] > row["baseline_harm_auroc"]
        )
    _write_json_once(
        MIXED_SOURCE_RESULTS,
        {
            **_analysis_envelope("mixed_source_results"),
            "split": SPLIT_A,
            "by_population": mixed,
            "source_free_baseline": baseline_row,
            "baseline_definition": (
                "normalized character edit distance used directly as a harm score, and its "
                "negation as a benefit score"
            ),
            "breadth_vs_baseline": breadth,
            "environments_improved": int(sum(1 for row in breadth.values() if row["improved"])),
            "orientation": (
                "the harm score is P(harmful) and its AUROC counts harmful candidates as the "
                "positive class; the benefit score is P(beneficial) with beneficial positive"
            ),
        },
    )
    harm_rows: dict[str, Any] = {}
    benefit_rows: dict[str, Any] = {}
    precision_rows: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        block = cell(scores, PRIMARY_POPULATION, representation, PRIMARY_MODEL, SPLIT_A)
        if block.empty:
            continue
        summary = discrimination(block)
        for target, store in ((T_HARM, harm_rows), (T_BENEFIT, benefit_rows)):
            label = "is_harmful" if target == T_HARM else "beneficial"
            store[representation] = {
                "auroc": summary[f"{target}_auroc"],
                "average_precision": summary[f"{target}_average_precision"],
                "prevalence": summary[f"{target}_prevalence"],
                "rows": summary["rows"],
                "by_environment": {
                    k: v[f"{target}_auroc"] for k, v in by_group(block, "environment").items()
                },
                "by_corrector": {
                    k: v[f"{target}_auroc"] for k, v in by_group(block, "corrector_source").items()
                },
                "by_proposal_stratum": {
                    k: v[f"{target}_auroc"] for k, v in by_group(block, "proposal_stratum").items()
                },
                "by_error_kind": {
                    k: v[f"{target}_auroc"] for k, v in by_group(block, "error_kind").items()
                },
                "pairwise_ranking_accuracy": pairwise_ranking_accuracy(block, target),
                "score_by_outcome": {
                    str(name): _score_quantiles(group[f"score_{target}"].to_numpy(float))
                    for name, group in block.groupby("outcome", sort=True)
                },
                "score_by_generator": {
                    str(name): _score_quantiles(group[f"score_{target}"].to_numpy(float))
                    for name, group in block.groupby("corrector_source", sort=True)
                },
                "positives": summary[f"{target}_positives"],
                "label_column": label,
            }
        precision_rows[representation] = {
            f"{target}_average_precision": summary[f"{target}_average_precision"]
            for target in TARGETS
        } | {f"{target}_prevalence": summary[f"{target}_prevalence"] for target in TARGETS}
    _write_json_once(
        HARM_DISCRIMINATION,
        {
            **_analysis_envelope("harm_discrimination"),
            "target": T_HARM,
            "positive_class": "harmful candidate, by the frozen project taxonomy",
            "population": PRIMARY_POPULATION,
            "split": SPLIT_A,
            "model": PRIMARY_MODEL,
            "by_representation": harm_rows,
        },
    )
    _write_json_once(
        BENEFIT_DISCRIMINATION,
        {
            **_analysis_envelope("benefit_discrimination"),
            "target": T_BENEFIT,
            "positive_class": "beneficial candidate, by the frozen project taxonomy",
            "population": PRIMARY_POPULATION,
            "split": SPLIT_A,
            "model": PRIMARY_MODEL,
            "by_representation": benefit_rows,
        },
    )
    _write_json_once(
        AVERAGE_PRECISION,
        {
            **_analysis_envelope("average_precision"),
            "by_representation": precision_rows,
            "why": (
                "the classes are imbalanced in both directions -- more than half the candidates "
                "are harmful and under a quarter are beneficial -- so AUROC alone would hide how "
                "much of the ranking is usable at the top"
            ),
            "source_free_baseline": {
                f"{target}_average_precision": baseline_row[f"{target}_average_precision"]
                for target in TARGETS
            },
        },
    )
    _write_json_once(
        SOURCE_STRATIFIED,
        {
            **_analysis_envelope("source_stratified_results"),
            "by_corrector": {
                name: {
                    **row,
                    "candidate_share": _ratio(row["rows"], len(primary_cell)),
                }
                for name, row in by_group(primary_cell, "corrector_source").items()
            },
            "representation": PRIMARY_REPRESENTATION,
            "note": (
                "the two correctors differ sharply in base rate -- the frozen generator's "
                "candidates are mostly harmful and the image corrector's are mostly not -- so a "
                "pooled AUROC is partly a between-source contrast. The within-source rows here "
                "are the part of the ranking that is not"
            ),
        },
    )
    _write_json_once(
        PROPOSAL_STRATIFIED,
        {
            **_analysis_envelope("proposal_stratified_results"),
            "by_stratum": by_group(primary_cell, "proposal_stratum"),
            "representation": PRIMARY_REPRESENTATION,
            "question": (
                "does the representation transfer across localization sources as well as across "
                "correction sources?"
            ),
        },
    )
    error_rows = by_group(primary_cell, "error_kind")
    _write_json_once(
        ERROR_TYPE_ANALYSIS,
        {
            **_analysis_envelope("error_type_analysis"),
            "by_kind": {
                kind: {
                    **error_rows.get(kind, {"rows": 0}),
                    "benefit_identifiable": bool(
                        error_rows.get(kind, {}).get("benefit_positives", 0) > 0
                    ),
                    "harm_identifiable": bool(
                        error_rows.get(kind, {}).get("harm_positives", 0) > 0
                    ),
                }
                for kind in ERROR_KINDS
            },
            "unlinked_and_mixed": {
                kind: error_rows.get(kind, {"rows": 0}) for kind in ("none", "mixed")
            },
            "omission_note": (
                "SGV-HY1 produced no exact omission repair at all, so benefit discrimination on "
                "omission sites is reported only where a beneficial candidate exists, and is "
                "recorded as not identifiable otherwise. No number is invented for it"
            ),
            "representation": PRIMARY_REPRESENTATION,
        },
    )
    _write_json_once(
        ENGINE_ANALYSIS,
        {
            **_analysis_envelope("engine_analysis"),
            "by_environment": by_group(primary_cell, "environment"),
            "by_base_engine": by_group(primary_cell, "base_engine"),
            "representation": PRIMARY_REPRESENTATION,
        },
    )
    _write_json_once(
        DOMAIN_ANALYSIS,
        {
            **_analysis_envelope("domain_analysis"),
            "by_domain": by_group(primary_cell, "domain"),
            "by_corpus": by_group(primary_cell, "corpus"),
            "representation": PRIMARY_REPRESENTATION,
        },
    )
    headline = mixed[PRIMARY_POPULATION][PRIMARY_MODEL][PRIMARY_REPRESENTATION]
    print(
        f"discriminate: R3 harm {headline['harm_auroc']:.4f}, "
        f"benefit {headline['benefit_auroc']:.4f} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 18-19, 23-25


def _direction_rows(block: pd.DataFrame, key: str = "held_out_generator") -> dict[str, Any]:
    """One split's discrimination per held-out generator, plus the mean over directions."""
    directions = {str(name): discrimination(group) for name, group in block.groupby(key, sort=True)}
    out: dict[str, Any] = dict(directions)
    for target in TARGETS:
        values = [row[f"{target}_auroc"] for row in directions.values()]
        finite = _finite(values)
        out[f"mean_{target}_auroc"] = _mean(values)
        out[f"min_{target}_auroc"] = min(finite) if finite else float("nan")
    return out


def run_transfer() -> int:
    """Sections 18-19 and 23-25: transfer across generators, environments and both at once."""
    started = time.monotonic()
    _require(MIXED_SOURCE_RESULTS, "discriminate")
    for path in (
        LEAVE_GENERATOR_OUT,
        LEAVE_ENVIRONMENT_OUT,
        GENERATOR_ENVIRONMENT_STRESS,
        TRANSFER_MATRIX,
    ):
        _forbid(path)
    scores = load_scores()
    generator: dict[str, Any] = {}
    for split in (SPLIT_B, SPLIT_B_DOC, SPLIT_B_OVERLAP):
        for model in MODELS:
            for representation in REPRESENTATIONS:
                block = cell(scores, PRIMARY_POPULATION, representation, model, split)
                if block.empty:
                    continue
                generator.setdefault(split, {}).setdefault(model, {})[representation] = {
                    **_direction_rows(block),
                    "per_environment": {
                        name: {
                            "harm_auroc": row["harm_auroc"],
                            "benefit_auroc": row["benefit_auroc"],
                            "rows": row["rows"],
                        }
                        for name, row in by_group(block, "environment").items()
                    },
                }
    primary_transfer = generator[PRIMARY_TRANSFER_SPLIT][PRIMARY_MODEL][PRIMARY_REPRESENTATION]
    _write_json_once(
        LEAVE_GENERATOR_OUT,
        {
            **_analysis_envelope("leave_generator_out_results"),
            "primary_split": PRIMARY_TRANSFER_SPLIT,
            "by_split": generator,
            "primary": primary_transfer,
            "near_random_threshold": NEAR_RANDOM,
            "confound": (
                "HY1's routing sends P0-only sites to the frozen generator and image-only sites "
                "to the image corrector, so a strict leave-one-generator-out split is also, in "
                "large part, a leave-one-proposal-stratum-out split. The overlap split is the "
                "control for exactly that: it restricts both sides to sites both proposers named"
            ),
            "stratum_controlled_split": SPLIT_B_OVERLAP,
            "document_held_out_split": SPLIT_B_DOC,
        },
    )
    environment_rows: dict[str, Any] = {}
    for model in MODELS:
        for representation in REPRESENTATIONS:
            block = cell(scores, PRIMARY_POPULATION, representation, model, SPLIT_C)
            if block.empty:
                continue
            per_environment = by_group(block, "environment")
            environment_rows.setdefault(model, {})[representation] = {
                **discrimination(block),
                "per_environment": per_environment,
                "mean_harm_auroc": _mean([row["harm_auroc"] for row in per_environment.values()]),
                "mean_benefit_auroc": _mean(
                    [row["benefit_auroc"] for row in per_environment.values()]
                ),
                "by_domain": by_group(block, "domain"),
            }
    _write_json_once(
        LEAVE_ENVIRONMENT_OUT,
        {
            **_analysis_envelope("leave_environment_out_results"),
            "split": SPLIT_C,
            "by_model": environment_rows,
            "both_axes_held_out": (
                "every fold holds out the environment AND the documents. The benchmark is "
                "matched-source, so holding out an engine alone would leave the same page in "
                "training under a different reader"
            ),
            "purpose": "separate generator transfer from engine and domain transfer",
        },
    )
    stress: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        block = cell(scores, PRIMARY_POPULATION, representation, PRIMARY_MODEL, SPLIT_D)
        if block.empty:
            continue
        cells = []
        for (held, name), group in block.groupby(
            ["held_out_generator", "held_out_environment"], sort=True
        ):
            cells.append(
                {
                    "held_out_generator": str(held),
                    "held_out_environment": str(name),
                    **discrimination(group),
                }
            )
        stress[representation] = {
            "cells": cells,
            "pooled": discrimination(block),
            "cells_with_both_classes": int(
                sum(1 for row in cells if np.isfinite(row["harm_auroc"]))
            ),
        }
    _write_json_once(
        GENERATOR_ENVIRONMENT_STRESS,
        {
            **_analysis_envelope("generator_environment_stress"),
            "split": SPLIT_D,
            "by_representation": stress,
            "secondary": True,
            "warning": (
                "cells are small and several carry one class only. A sparse cell is reported as "
                "undefined rather than as a number, and no cell here moves the outcome"
            ),
        },
    )
    matrix_rows: dict[str, Any] = {}
    for representation in AGNOSTIC_REPRESENTATIONS:
        within = cell(scores, PRIMARY_POPULATION, representation, PRIMARY_MODEL, SPLIT_W)
        across = cell(scores, PRIMARY_POPULATION, representation, PRIMARY_MODEL, SPLIT_B)
        cells = []
        for trained in CORRECTORS:
            for evaluated in CORRECTORS:
                if trained == evaluated:
                    group = within[within["training_generator"] == trained]
                    source_split = SPLIT_W
                else:
                    group = across[
                        (across["training_generator"] == trained)
                        & (across["held_out_generator"] == evaluated)
                    ]
                    source_split = SPLIT_B
                cells.append(
                    {
                        "train_source": trained,
                        "test_source": evaluated,
                        "split": source_split,
                        **discrimination(group),
                    }
                )
        matrix_rows[representation] = cells
    _write_json_once(
        TRANSFER_MATRIX,
        {
            **_analysis_envelope("cross_generator_transfer_matrix"),
            "by_representation": matrix_rows,
            "diagonal_split": SPLIT_W,
            "off_diagonal_split": SPLIT_B,
            "note": (
                "the diagonal is measured with the document partition held out, so it is a fair "
                "same-generator reference rather than a resubstitution number; the off-diagonal "
                "holds out the generator and uses site-disjoint blocks"
            ),
        },
    )
    print(
        f"transfer: LOGO harm {primary_transfer['mean_harm_auroc']:.4f}, "
        f"benefit {primary_transfer['mean_benefit_auroc']:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 17-18: prefixes


def prefix_geometry(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, exact: np.ndarray
) -> list[dict[str, Any]]:
    """Coverage, harm and benefit along the safety-ranked list. No threshold is selected."""
    mask = np.isfinite(score)
    order = np.argsort(score[mask], kind="mergesort")
    harm = harmful[mask][order]
    good = beneficial[mask][order]
    repair = exact[mask][order]
    total = harm.size
    repairs = int(exact.sum())
    rows: list[dict[str, Any]] = []
    for fraction in PREFIX_FRACTIONS:
        size = round(fraction * total)
        if size <= 0:
            continue
        rows.append(
            {
                "rank_fraction": float(fraction),
                "coverage": _ratio(size, total),
                "candidates": size,
                "harm_rate": float(harm[:size].mean()),
                "beneficial_rate": float(good[:size].mean()),
                "exact_repairs_captured": int(repair[:size].sum()),
                "repair_recall": _ratio(int(repair[:size].sum()), repairs),
            }
        )
    return rows


def oracle_frontier(
    score: np.ndarray, harmful: np.ndarray, exact: np.ndarray, epsilon: float
) -> dict[str, Any]:
    """The longest safety-ranked prefix whose realized harm rate stays at or below epsilon.

    ORACLE, ANALYSIS ONLY, NOT DEPLOYABLE. The cut is chosen by looking at the labels of the rows
    it accepts, which no deployed system can do. It measures the headroom in the ranking, never an
    operating point, and nothing downstream reads it.
    """
    mask = np.isfinite(score)
    order = np.argsort(score[mask], kind="mergesort")
    harm = harmful[mask][order].astype(np.float64)
    repair = exact[mask][order].astype(np.float64)
    if harm.size == 0:
        return {"epsilon": epsilon, "prefix": 0, "coverage": 0.0, "repair_recall": 0.0}
    cumulative = np.cumsum(harm) / np.arange(1, harm.size + 1)
    admissible = np.flatnonzero(cumulative <= epsilon)
    prefix = int(admissible.max() + 1) if admissible.size else 0
    repairs = float(exact.sum())
    return {
        "epsilon": float(epsilon),
        "prefix": prefix,
        "coverage": _ratio(prefix, int(harm.size)),
        "harm_rate": float(cumulative[prefix - 1]) if prefix else 0.0,
        "exact_repairs_captured": int(repair[:prefix].sum()),
        "repair_recall": _ratio(int(repair[:prefix].sum()), int(repairs)),
        "oracle_threshold": True,
        "analysis_only": True,
        "deployable": False,
    }


def _random_frontier(frame: pd.DataFrame, epsilon: float) -> dict[str, float]:
    """The same frontier under seeded random ranking, averaged over draws."""
    harmful = frame["is_harmful"].to_numpy(dtype=bool)
    exact = frame["exact"].to_numpy(dtype=bool)
    generator = np.random.default_rng(RANDOM_RANKING_SEED)
    recalls: list[float] = []
    coverages: list[float] = []
    for _draw in range(RANDOM_RANKING_DRAWS):
        score = generator.random(len(frame))
        row = oracle_frontier(score, harmful, exact, epsilon)
        recalls.append(float(row["repair_recall"]))
        coverages.append(float(row["coverage"]))
    return {
        "mean_repair_recall": float(np.mean(recalls)),
        "mean_coverage": float(np.mean(coverages)),
        "draws": RANDOM_RANKING_DRAWS,
        "seed": RANDOM_RANKING_SEED,
    }


def run_prefix() -> int:
    """Sections 17-18: prefix geometry and the diagnostic oracle risk frontier."""
    started = time.monotonic()
    _require(LEAVE_GENERATOR_OUT, "transfer")
    for path in (PREFIX_RANKING, ORACLE_RISK_FRONTIER):
        _forbid(path)
    scores = load_scores()
    population = pd.read_parquet(CANDIDATE_POPULATION)
    primary = population[population["population"] == PRIMARY_POPULATION].reset_index(drop=True)
    prefixes: dict[str, Any] = {}
    frontier: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        block = cell(scores, PRIMARY_POPULATION, representation, PRIMARY_MODEL, SPLIT_A)
        if block.empty:
            continue
        harmful = block["is_harmful"].to_numpy(dtype=bool)
        beneficial = block["beneficial"].to_numpy(dtype=bool)
        exact = block["exact"].to_numpy(dtype=bool)
        score = block["score_harm"].to_numpy(dtype=np.float64)
        prefixes[representation] = prefix_geometry(score, harmful, beneficial, exact)
        frontier[representation] = {
            epsilon_key(epsilon): oracle_frontier(score, harmful, exact, epsilon)
            for epsilon in EPSILONS
        }
    random_scores = np.random.default_rng(RANDOM_RANKING_SEED).random(len(primary))
    harmful = primary["is_harmful"].to_numpy(dtype=bool)
    beneficial = primary["beneficial"].to_numpy(dtype=bool)
    exact = primary["exact"].to_numpy(dtype=bool)
    prefixes["random"] = prefix_geometry(random_scores, harmful, beneficial, exact)
    prefixes["source_free_baseline"] = prefix_geometry(
        baseline_score(primary), harmful, beneficial, exact
    )
    frontier["random"] = {
        epsilon_key(epsilon): _random_frontier(primary, epsilon) for epsilon in EPSILONS
    }
    frontier["source_free_baseline"] = {
        epsilon_key(epsilon): oracle_frontier(baseline_score(primary), harmful, exact, epsilon)
        for epsilon in EPSILONS
    }
    key = epsilon_key(PRIMARY_EPSILON)
    margin = float(frontier[PRIMARY_REPRESENTATION][key]["repair_recall"]) - float(
        frontier["random"][key]["mean_repair_recall"]
    )
    _write_json_once(
        PREFIX_RANKING,
        {
            **_analysis_envelope("prefix_ranking"),
            "by_representation": prefixes,
            "fractions": list(PREFIX_FRACTIONS),
            "ranking": "ascending P(harmful); the safest candidates come first",
            "selects_a_deployment_threshold": False,
            "population": PRIMARY_POPULATION,
            "split": SPLIT_A,
            "base_rates": {
                "harm": float(harmful.mean()),
                "benefit": float(beneficial.mean()),
                "exact": float(exact.mean()),
                "exact_repairs": int(exact.sum()),
            },
        },
    )
    _write_json_once(
        ORACLE_RISK_FRONTIER,
        {
            **_analysis_envelope("oracle_risk_frontier"),
            "by_representation": frontier,
            "epsilons": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "label": "oracle threshold -- analysis only -- not deployable",
            "rule": (
                "the longest score-ranked prefix whose realized harm rate is at or below epsilon, "
                "chosen with the labels of the accepted rows. It measures ranking headroom and "
                "cannot be implemented without the labels it reads"
            ),
            "primary_margin_over_random": margin,
            "primary_margin_required": C6_PREFIX_MARGIN,
            "random_comparator_is_near_zero_by_construction": (
                "the population's base harm rate is far above the primary epsilon, so a random "
                "prefix almost never satisfies the constraint. The comparator is reported as it "
                "is measured; it is a fact about the population, not a generous baseline"
            ),
        },
    )
    print(
        f"prefix: R3 repair recall at harm<={PRIMARY_EPSILON} is "
        f"{frontier[PRIMARY_REPRESENTATION][key]['repair_recall']:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 22-25: ablations


def run_ablate() -> int:
    """Sections 22, 23 and 25: the visual effect, the feature ablations, the identity gap."""
    started = time.monotonic()
    _require(PREFIX_RANKING, "prefix")
    for path in (VISUAL_ABLATION, FEATURE_ABLATION, SOURCE_IDENTITY_GAP):
        _forbid(path)
    scores = load_scores()
    mixed = cc_read_json(MIXED_SOURCE_RESULTS)["by_population"][PRIMARY_POPULATION][PRIMARY_MODEL]
    logo = cc_read_json(LEAVE_GENERATOR_OUT)["by_split"][PRIMARY_TRANSFER_SPLIT][PRIMARY_MODEL]
    prefixes = cc_read_json(ORACLE_RISK_FRONTIER)["by_representation"]
    key = epsilon_key(PRIMARY_EPSILON)
    visual = {
        "mixed_source": {
            f"{target}_gain": float(mixed[R3][f"{target}_auroc"])
            - float(mixed[R2][f"{target}_auroc"])
            for target in TARGETS
        },
        "leave_generator_out": {
            f"{target}_gain": float(logo[R3][f"mean_{target}_auroc"])
            - float(logo[R2][f"mean_{target}_auroc"])
            for target in TARGETS
        },
        "prefix": {
            "repair_recall_gain": float(prefixes[R3][key]["repair_recall"])
            - float(prefixes[R2][key]["repair_recall"])
        },
        "r2": {target: float(mixed[R2][f"{target}_auroc"]) for target in TARGETS},
        "r3": {target: float(mixed[R3][f"{target}_auroc"]) for target in TARGETS},
    }
    material = bool(
        visual["mixed_source"]["harm_gain"] > 0.0
        and visual["leave_generator_out"]["harm_gain"] > 0.0
    )
    _write_json_once(
        VISUAL_ABLATION,
        {
            **_analysis_envelope("visual_evidence_ablation"),
            **visual,
            "visual_gain_is_material": material,
            "required_for_success": False,
            "interpretation": (
                "visual evidence is a mechanistic question here, not a success condition. If R3 "
                "does not materially improve on R2, the finding is that image evidence helps "
                "correction GENERATION -- which SGV-GEN1 measured -- but is not required for "
                "correction RELIABILITY, and that is reported as the result rather than rescued"
            ),
        },
    )
    ablation_rows: dict[str, Any] = {}
    for name in ABLATIONS:
        key_name = R3 if name == A0 else name
        block = cell(scores, PRIMARY_POPULATION, key_name, PRIMARY_MODEL, SPLIT_A)
        if block.empty:
            continue
        row = discrimination(block)
        ablation_rows[name] = {
            "dropped_families": list(ABLATIONS[name]),
            "columns": len(columns_for(R3, ABLATIONS[name])),
            **{f"{target}_auroc": row[f"{target}_auroc"] for target in TARGETS},
            **{
                f"{target}_delta_vs_full": row[f"{target}_auroc"]
                - float(mixed[R3][f"{target}_auroc"])
                for target in TARGETS
            },
        }
    _write_json_once(
        FEATURE_ABLATION,
        {
            **_analysis_envelope("feature_ablation"),
            "by_ablation": ablation_rows,
            "reference": A0,
            "set_is_pre_registered": True,
            "combinatorial_search": False,
            "a3_reproduces_r2": bool(
                ablation_rows.get(A3, {}).get("harm_auroc") == mixed[R2]["harm_auroc"]
            ),
            "load_bearing_order": sorted(
                (name for name in ablation_rows if name != A0),
                key=lambda name: float(ablation_rows[name]["harm_delta_vs_full"]),
            ),
        },
    )
    gap = {
        "mixed_source": {
            target: {
                "r3": float(mixed[R3][f"{target}_auroc"]),
                "r4": float(mixed[R4][f"{target}_auroc"]),
                "gap": float(mixed[R4][f"{target}_auroc"]) - float(mixed[R3][f"{target}_auroc"]),
                "retention": retention(
                    float(mixed[R3][f"{target}_auroc"]), float(mixed[R4][f"{target}_auroc"])
                ),
            }
            for target in TARGETS
        },
        "leave_generator_out": {
            target: {
                "r3": float(logo[R3][f"mean_{target}_auroc"]),
                "r4": float(logo[R4][f"mean_{target}_auroc"]),
                "gap": float(logo[R4][f"mean_{target}_auroc"])
                - float(logo[R3][f"mean_{target}_auroc"]),
            }
            for target in TARGETS
        },
    }
    _write_json_once(
        SOURCE_IDENTITY_GAP,
        {
            **_analysis_envelope("source_identity_gap"),
            **gap,
            "retention_floor": C4_RETENTION_FLOOR,
            "retention_minimum_excess": RETENTION_MIN_EXCESS,
            "material_gap_margin": SOURCE_GAP_MARGIN,
            "rule": (
                "retention = (A_R3 - 0.5) / (A_R4 - 0.5). A denominator below the minimum excess "
                "makes the ratio unidentifiable, and an unidentifiable ratio is never a success"
            ),
            "note": (
                "R4 adds generator identity to R3 and nothing else, so the difference is what "
                "identity is worth on this population. Under leave-one-generator-out, R4's "
                "identity columns are constant on the training side and take an unseen value on "
                "the evaluation side, which is exactly the situation SGV-XR1 could not resolve; "
                "the number is reported for that reason and not as a usable model"
            ),
        },
    )
    print(
        f"ablate: visual harm gain {visual['mixed_source']['harm_gain']:+.4f}, "
        f"identity harm gap {gap['mixed_source'][T_HARM]['gap']:+.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 24, 35-36, 39-43


def _permuted_labels(frame: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Labels permuted at the DOCUMENT-cluster level, never row by row.

    A row-level shuffle destroys the within-document correlation structure as well as the signal,
    which makes the control easier to pass than the thing it is controlling for. Whole documents
    exchange their label blocks instead.
    """
    generator = np.random.default_rng(seed)
    documents = sorted(set(frame["document_id"].astype(str)))
    order = generator.permutation(len(documents))
    donor = {documents[i]: documents[int(order[i])] for i in range(len(documents))}
    blocks = {
        document: frame.index[frame["document_id"].astype(str) == document].to_numpy()
        for document in documents
    }
    out: dict[str, np.ndarray] = {}
    for target, column in ((T_HARM, "is_harmful"), (T_BENEFIT, "beneficial")):
        values = frame[column].to_numpy(dtype=bool)
        permuted = values.copy()
        for document in documents:
            source_rows = blocks[donor[document]]
            target_rows = blocks[document]
            if source_rows.size == 0 or target_rows.size == 0:
                continue
            picks = generator.integers(0, source_rows.size, size=target_rows.size)
            permuted[target_rows] = values[source_rows[picks]]
        out[target] = permuted
    return out


def run_controls() -> int:
    """Sections 24, 35, 36, 39 and 40: random ranking, permuted labels, counterfactual source."""
    started = time.monotonic()
    _require(SOURCE_IDENTITY_GAP, "ablate")
    for path in (
        RANDOM_RANKING_CONTROL,
        LABEL_PERMUTATION_CONTROL,
        COUNTERFACTUAL_SOURCE,
        CONTROL_RESULTS,
    ):
        _forbid(path)
    population = pd.read_parquet(CANDIDATE_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    primary = population[population["population"] == PRIMARY_POPULATION].reset_index(drop=True)
    harmful = primary["is_harmful"].to_numpy(dtype=bool)
    beneficial = primary["beneficial"].to_numpy(dtype=bool)
    generator = np.random.default_rng(RANDOM_RANKING_SEED)
    draws = {target: [] for target in TARGETS}
    for _draw in range(RANDOM_RANKING_DRAWS):
        score = generator.random(len(primary))
        draws[T_HARM].append(auroc(score, harmful))
        draws[T_BENEFIT].append(auroc(score, beneficial))
    observed = cc_read_json(MIXED_SOURCE_RESULTS)["by_population"][PRIMARY_POPULATION][
        PRIMARY_MODEL
    ]
    random_row = {
        target: {
            "mean_auroc": float(np.mean(draws[target])),
            "min_auroc": float(np.min(draws[target])),
            "max_auroc": float(np.max(draws[target])),
            "draws": RANDOM_RANKING_DRAWS,
            "expected": 0.5,
            "r3_auroc": float(observed[R3][f"{target}_auroc"]),
            "r3_minus_random": float(observed[R3][f"{target}_auroc"])
            - float(np.mean(draws[target])),
        }
        for target in TARGETS
    }
    _write_json_once(
        RANDOM_RANKING_CONTROL,
        {
            **_analysis_envelope("random_ranking_control"),
            "by_target": random_row,
            "seed": RANDOM_RANKING_SEED,
            "near_random": bool(
                all(abs(random_row[target]["mean_auroc"] - 0.5) < 0.02 for target in TARGETS)
            ),
        },
    )
    permutation_rows: list[dict[str, Any]] = []
    columns = columns_for(R3)
    for draw in range(PERMUTATION_DRAWS):
        labels = _permuted_labels(primary, PERMUTATION_SEED + draw)
        permuted = score_split(primary, matrix, columns, M0, SPLIT_A, labels)
        joined = permuted.merge(
            primary[["candidate_id"]].assign(position=np.arange(len(primary))),
            on="candidate_id",
            validate="one_to_one",
        )
        position = joined["position"].to_numpy(dtype=np.int64)
        permutation_rows.append(
            {
                "draw": draw,
                **{
                    f"{target}_auroc_against_permuted": auroc(
                        joined[f"score_{target}"].to_numpy(dtype=np.float64),
                        labels[target][position],
                    )
                    for target in TARGETS
                },
                **{
                    f"{target}_auroc_against_true": auroc(
                        joined[f"score_{target}"].to_numpy(dtype=np.float64),
                        (harmful if target == T_HARM else beneficial)[position],
                    )
                    for target in TARGETS
                },
            }
        )
    permutation = {
        target: {
            "mean_auroc_against_permuted_labels": _mean(
                row[f"{target}_auroc_against_permuted"] for row in permutation_rows
            ),
            "mean_auroc_against_true_labels": _mean(
                row[f"{target}_auroc_against_true"] for row in permutation_rows
            ),
            "r3_auroc": float(observed[R3][f"{target}_auroc"]),
            "discrimination_destroyed": bool(
                _mean(row[f"{target}_auroc_against_permuted"] for row in permutation_rows)
                < float(observed[R3][f"{target}_auroc"]) - 0.05
            ),
        }
        for target in TARGETS
    }
    _write_json_once(
        LABEL_PERMUTATION_CONTROL,
        {
            **_analysis_envelope("label_permutation_control"),
            "by_target": permutation,
            "draws": PERMUTATION_DRAWS,
            "seed": PERMUTATION_SEED,
            "unit": "document cluster",
            "rows": permutation_rows,
            "representation": R3,
        },
    )
    counterfactual = _counterfactual_source(primary, matrix)
    _write_json_once(COUNTERFACTUAL_SOURCE, counterfactual)
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "random_ranking": {target: random_row[target]["mean_auroc"] for target in TARGETS},
            "label_permutation": {
                target: permutation[target]["mean_auroc_against_permuted_labels"]
                for target in TARGETS
            },
            "counterfactual_source": {
                "r3_max_absolute_score_change": counterfactual["r3"]["max_absolute_score_change"],
                "r4_max_absolute_score_change": counterfactual["r4"]["max_absolute_score_change"],
            },
            "all_controls_behave": bool(
                cc_read_json(RANDOM_RANKING_CONTROL)["near_random"]
                and all(permutation[target]["discrimination_destroyed"] for target in TARGETS)
                and counterfactual["r3"]["max_absolute_score_change"] == 0.0
            ),
        },
    )
    print(
        f"controls: random {random_row[T_HARM]['mean_auroc']:.4f}, permuted "
        f"{permutation[T_HARM]['mean_auroc_against_permuted_labels']:.4f}, counterfactual R3 "
        f"{counterfactual['r3']['max_absolute_score_change']:g} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _counterfactual_source(primary: pd.DataFrame, matrix: pd.DataFrame) -> dict[str, Any]:
    """Relabel a candidate's source, change nothing else, and re-score it.

    R3 must be exactly unchanged -- not approximately, exactly -- because its feature matrix does
    not contain a source column at all. R4 may change, and if it did not, the source-identity
    comparator would be measuring nothing.
    """
    sample = min(COUNTERFACTUAL_SAMPLE, len(primary))
    picks = np.sort(
        np.random.default_rng(COUNTERFACTUAL_SEED).choice(len(primary), sample, replace=False)
    )
    out: dict[str, Any] = {
        **_analysis_envelope("counterfactual_source_test"),
        "sampled_candidates": int(sample),
        "seed": COUNTERFACTUAL_SEED,
        "rule": (
            "hold every candidate and evidence feature fixed and swap the source metadata: the "
            "corrector one-hots, the generator rank, the per-site candidate count, the proposal "
            "stratum and both suspicion columns"
        ),
    }
    swapped = matrix.copy()
    identifiers = primary.iloc[picks]["candidate_id"].astype(str)
    rows = swapped["candidate_id"].astype(str).isin(set(identifiers))
    c0 = swapped.loc[rows, "src_corrector_c0"].to_numpy(dtype=np.float64)
    swapped.loc[rows, "src_corrector_c0"] = swapped.loc[rows, "src_corrector_c1"].to_numpy(
        dtype=np.float64
    )
    swapped.loc[rows, "src_corrector_c1"] = c0
    p0 = swapped.loc[rows, "src_stratum_p0_only"].to_numpy(dtype=np.float64)
    swapped.loc[rows, "src_stratum_p0_only"] = swapped.loc[rows, "src_stratum_image_only"].to_numpy(
        dtype=np.float64
    )
    swapped.loc[rows, "src_stratum_image_only"] = p0
    suspicion = swapped.loc[rows, "src_p0_suspicion"].to_numpy(dtype=np.float64)
    swapped.loc[rows, "src_p0_suspicion"] = swapped.loc[rows, "src_image_suspicion"].to_numpy(
        dtype=np.float64
    )
    swapped.loc[rows, "src_image_suspicion"] = suspicion
    missing = swapped.loc[rows, "src_p0_suspicion_missing"].to_numpy(dtype=np.float64)
    swapped.loc[rows, "src_p0_suspicion_missing"] = swapped.loc[
        rows, "src_image_suspicion_missing"
    ].to_numpy(dtype=np.float64)
    swapped.loc[rows, "src_image_suspicion_missing"] = missing
    swapped.loc[rows, "src_generator_rank"] = 0.0
    swapped.loc[rows, "src_site_candidate_count"] = 1.0
    for name, representation in (("r3", R3), ("r4", R4)):
        columns = columns_for(representation)
        before = score_split(primary, matrix, columns, M0, SPLIT_A)
        after = score_split(primary, swapped, columns, M0, SPLIT_A)
        joined = before.merge(after, on=["fold", "candidate_id"], suffixes=("_before", "_after"))
        joined = joined[joined["candidate_id"].astype(str).isin(set(identifiers))]
        difference = max(
            float(
                np.nanmax(
                    np.abs(
                        joined[f"score_{target}_before"].to_numpy(dtype=np.float64)
                        - joined[f"score_{target}_after"].to_numpy(dtype=np.float64)
                    )
                )
            )
            for target in TARGETS
        )
        out[name] = {
            "representation": representation,
            "carries_source_columns": FAM_SOURCE in REPRESENTATION_FAMILIES[representation],
            "compared_rows": len(joined),
            "max_absolute_score_change": difference,
            "expected": "exactly 0" if name == "r3" else "may be non-zero",
            "as_expected": bool(difference == 0.0 if name == "r3" else difference > 0.0),
        }
    return out


# ------------------------------------------------------------------ section 34: statistics


def _paired_frame(scores: pd.DataFrame, a: str, b: str, split: str, model: str) -> pd.DataFrame:
    """The rows both arms scored, once, with each arm's score beside the other's."""
    left = cell(scores, PRIMARY_POPULATION, a, model, split)
    right = cell(scores, PRIMARY_POPULATION, b, model, split)
    columns = ["candidate_id", "fold", "score_harm", "score_benefit"]
    keep = [*columns, "environment", "document_id", "is_harmful", "beneficial"]
    joined = left[keep].merge(
        right[columns],
        on=["candidate_id", "fold"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    return joined.reset_index(drop=True)


def run_stats() -> int:
    """Section 34: the frozen families, document-clustered, paired, Holm-corrected."""
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    scores = load_scores()
    population = pd.read_parquet(CANDIDATE_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    primary_population = population[population["population"] == PRIMARY_POPULATION].reset_index(
        drop=True
    )

    def comparison(a: str, b: str, split: str, target: str, name: str, family: str) -> Any:
        frame = _paired_frame(scores, a, b, split, PRIMARY_MODEL)
        label = "is_harmful" if target == T_HARM else "beneficial"
        return paired_auroc_comparison(
            frame, f"score_{target}_a", f"score_{target}_b", label, name, family
        )

    primary = {
        PRIMARY_FAMILY[0]: comparison(R3, R1, SPLIT_A, T_HARM, PRIMARY_FAMILY[0], "primary"),
        PRIMARY_FAMILY[1]: comparison(
            R3, R1, PRIMARY_TRANSFER_SPLIT, T_HARM, PRIMARY_FAMILY[1], "primary"
        ),
        PRIMARY_FAMILY[2]: comparison(R3, R2, SPLIT_A, T_HARM, PRIMARY_FAMILY[2], "primary"),
        PRIMARY_FAMILY[3]: comparison(R3, R4, SPLIT_A, T_HARM, PRIMARY_FAMILY[3], "primary"),
    }
    benefit = {
        BENEFIT_FAMILY[0]: comparison(R3, R1, SPLIT_A, T_BENEFIT, BENEFIT_FAMILY[0], "benefit"),
        BENEFIT_FAMILY[1]: comparison(
            R3, R1, PRIMARY_TRANSFER_SPLIT, T_BENEFIT, BENEFIT_FAMILY[1], "benefit"
        ),
        BENEFIT_FAMILY[2]: comparison(R3, R2, SPLIT_A, T_BENEFIT, BENEFIT_FAMILY[2], "benefit"),
        BENEFIT_FAMILY[3]: comparison(R3, R4, SPLIT_A, T_BENEFIT, BENEFIT_FAMILY[3], "benefit"),
    }
    reference = cell(scores, PRIMARY_POPULATION, R3, PRIMARY_MODEL, SPLIT_A)
    control_frame = reference[
        ["candidate_id", "environment", "document_id", "is_harmful", "beneficial"]
    ].copy()
    control_frame["score_harm_r3"] = reference["score_harm"].to_numpy(dtype=np.float64)
    control_frame["score_benefit_r3"] = reference["score_benefit"].to_numpy(dtype=np.float64)
    order = np.random.default_rng(RANDOM_RANKING_SEED).random(len(control_frame))
    control_frame["score_harm_random"] = order
    control_frame["score_benefit_random"] = order
    permuted = score_split(
        primary_population,
        matrix,
        columns_for(R3),
        M0,
        SPLIT_A,
        _permuted_labels(primary_population, PERMUTATION_SEED),
    )
    control_frame = control_frame.merge(
        permuted[["candidate_id", "score_harm", "score_benefit"]].rename(
            columns={"score_harm": "score_harm_permuted", "score_benefit": "score_benefit_permuted"}
        ),
        on="candidate_id",
        validate="one_to_one",
    )
    controls = {
        CONTROL_FAMILY[0]: paired_auroc_comparison(
            control_frame,
            "score_harm_r3",
            "score_harm_random",
            "is_harmful",
            CONTROL_FAMILY[0],
            "controls",
        ),
        CONTROL_FAMILY[1]: paired_auroc_comparison(
            control_frame,
            "score_harm_r3",
            "score_harm_permuted",
            "is_harmful",
            CONTROL_FAMILY[1],
            "controls",
        ),
        CONTROL_FAMILY[2]: paired_auroc_comparison(
            control_frame,
            "score_benefit_r3",
            "score_benefit_random",
            "beneficial",
            CONTROL_FAMILY[2],
            "controls",
        ),
        CONTROL_FAMILY[3]: paired_auroc_comparison(
            control_frame,
            "score_benefit_r3",
            "score_benefit_permuted",
            "beneficial",
            CONTROL_FAMILY[3],
            "controls",
        ),
    }
    adjusted = s15.holm(primary)
    benefit_adjusted = s15.holm(benefit)
    controls_adjusted = s15.holm(controls)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "bootstrap": {
                "unit": "document, resampled within environment",
                "paired": True,
                "statistic": "pooled AUROC difference between two score columns",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "alpha": ALPHA,
            },
            "multiplicity": "Holm-Bonferroni inside each declared family",
            "primary_family": adjusted,
            "benefit_family": benefit_adjusted,
            "control_family": controls_adjusted,
            "family_sizes": {
                "primary": len(PRIMARY_FAMILY),
                "benefit": len(BENEFIT_FAMILY),
                "controls": len(CONTROL_FAMILY),
            },
            "declared_before_any_endpoint": True,
            "surviving_primary": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "surviving_benefit": sorted(
                k for k, v in benefit_adjusted.items() if v["survives_holm"]
            ),
            "row_is_never_the_inferential_unit": True,
        },
    )
    print(
        f"stats: {len(adjusted)} primary tests, "
        f"{sum(1 for v in adjusted.values() if v['survives_holm'])} survive Holm "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 43-44: decision


def criteria_from_artifacts() -> dict[str, Any]:
    """Every success criterion, re-derived from the persisted artifacts and nothing else."""
    mixed = cc_read_json(MIXED_SOURCE_RESULTS)
    pooled = mixed["by_population"][PRIMARY_POPULATION][PRIMARY_MODEL]
    logo = cc_read_json(LEAVE_GENERATOR_OUT)["by_split"][PRIMARY_TRANSFER_SPLIT][PRIMARY_MODEL]
    gap = cc_read_json(SOURCE_IDENTITY_GAP)["mixed_source"]
    frontier = cc_read_json(ORACLE_RISK_FRONTIER)
    key = epsilon_key(PRIMARY_EPSILON)
    transfer = logo[PRIMARY_REPRESENTATION]
    directions = {
        name: transfer[name]
        for name in transfer
        if name in CORRECTORS and isinstance(transfer[name], dict)
    }
    harm_directions = [float(row["harm_auroc"]) for row in directions.values()]
    c1 = bool(
        float(pooled[R3]["harm_auroc"]) > C1_HARM_FLOOR
        and float(pooled[R3]["benefit_auroc"]) > C1_BENEFIT_FLOOR
    )
    c2 = bool(
        float(transfer["mean_harm_auroc"]) >= C2_TRANSFER_HARM_FLOOR
        and all(v >= NEAR_RANDOM for v in harm_directions)
    )
    c3 = bool(float(transfer["mean_benefit_auroc"]) >= C3_TRANSFER_BENEFIT_FLOOR)
    retentions = {target: gap[target]["retention"] for target in TARGETS}
    c4 = bool(
        all(
            retentions[target] is not None and retentions[target] >= C4_RETENTION_FLOOR
            for target in TARGETS
        )
    )
    improved = int(mixed["environments_improved"])
    c5 = bool(improved >= C5_BREADTH_MAJORITY)
    margin = float(frontier["primary_margin_over_random"])
    c6 = bool(margin >= C6_PREFIX_MARGIN)
    return {
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6},
        "pooled": pooled,
        "transfer": transfer,
        "directions": directions,
        "gap": gap,
        "retentions": retentions,
        "environments_improved": improved,
        "prefix_margin": margin,
        "frontier": frontier["by_representation"],
        "epsilon_key": key,
    }


def _dominant_mechanism(criteria: dict[str, bool], gap: float) -> str:
    if criteria["C2"] and criteria["C3"] and criteria["C4"]:
        return "candidate evidence itself is sufficient"
    if criteria["C1"] and not (criteria["C2"] and criteria["C3"]):
        return "candidate evidence works, but each new generator needs supervision"
    if gap >= SOURCE_GAP_MARGIN or not criteria["C4"]:
        return "generator source remains predictive and load-bearing"
    return "available candidate evidence is too weak"


def run_decide() -> int:
    """Section 44: the frozen outcome rule, applied to the persisted artifacts."""
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    criteria = state["criteria"]
    pooled = state["pooled"]
    transfer = state["transfer"]
    gap_harm = float(state["gap"][T_HARM]["gap"])
    outcome = assign_outcome(criteria, gap_harm, float(pooled[R4]["harm_auroc"]))
    failed = sorted(name for name, value in criteria.items() if not value)
    inventory = cc_read_json(POPULATION_INVENTORY)["populations"][PRIMARY_POPULATION]
    leakage = cc_read_json(IMPLICIT_SOURCE_LEAKAGE)["results"][PRIMARY_REPRESENTATION]
    visual = cc_read_json(VISUAL_ABLATION)
    negative = cc_read_json(FALSIFICATION)
    frontier = state["frontier"][PRIMARY_REPRESENTATION]
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
            "primary_population": PRIMARY_POPULATION,
            "primary_representation": PRIMARY_REPRESENTATION,
            "primary_model": PRIMARY_MODEL,
            "candidate_count": int(inventory["candidates"]),
            "beneficial_prevalence": float(inventory["beneficial_prevalence"]),
            "harmful_prevalence": float(inventory["harmful_prevalence"]),
            "multi_source_candidates": int(inventory["multi_source_candidates"]),
            **{
                f"{name.split('_')[0]}_harm_auroc": float(pooled[name]["harm_auroc"])
                for name in REPRESENTATIONS
            },
            **{
                f"{name.split('_')[0]}_benefit_auroc": float(pooled[name]["benefit_auroc"])
                for name in REPRESENTATIONS
            },
            "mixed_source_harm_auroc": float(pooled[PRIMARY_REPRESENTATION]["harm_auroc"]),
            "mixed_source_benefit_auroc": float(pooled[PRIMARY_REPRESENTATION]["benefit_auroc"]),
            "current_to_qwen_harm_auroc": float(state["directions"][C1]["harm_auroc"]),
            "qwen_to_current_harm_auroc": float(state["directions"][C0]["harm_auroc"]),
            "current_to_qwen_benefit_auroc": float(state["directions"][C1]["benefit_auroc"]),
            "qwen_to_current_benefit_auroc": float(state["directions"][C0]["benefit_auroc"]),
            "mean_leave_generator_out_harm_auroc": float(transfer["mean_harm_auroc"]),
            "mean_leave_generator_out_benefit_auroc": float(transfer["mean_benefit_auroc"]),
            "source_identity_harm_gain": gap_harm,
            "source_identity_benefit_gain": float(state["gap"][T_BENEFIT]["gap"]),
            "source_identity_harm_retention": state["retentions"][T_HARM],
            "source_identity_benefit_retention": state["retentions"][T_BENEFIT],
            "visual_harm_gain": float(visual["mixed_source"]["harm_gain"]),
            "visual_benefit_gain": float(visual["mixed_source"]["benefit_gain"]),
            "visual_gain_is_material": bool(visual["visual_gain_is_material"]),
            "source_predictability": float(leakage["source_auroc"]),
            **{
                f"oracle_repair_recall_at_harm_{epsilon_key(e).replace('.', '')}": float(
                    frontier[epsilon_key(e)]["repair_recall"]
                )
                for e in EPSILONS
            },
            "oracle_prefix_margin_over_random": float(state["prefix_margin"]),
            "environments_improved": int(state["environments_improved"]),
            "criteria_thresholds": {
                "C1_harm_floor": C1_HARM_FLOOR,
                "C1_benefit_floor": C1_BENEFIT_FLOOR,
                "C2_transfer_harm_floor": C2_TRANSFER_HARM_FLOOR,
                "C2_near_random": NEAR_RANDOM,
                "C3_transfer_benefit_floor": C3_TRANSFER_BENEFIT_FLOOR,
                "C4_retention_floor": C4_RETENTION_FLOOR,
                "C4_retention_minimum_excess": RETENTION_MIN_EXCESS,
                "C5_breadth_majority": C5_BREADTH_MAJORITY,
                "C5_environments": len(environment_specs()),
                "C6_prefix_margin": C6_PREFIX_MARGIN,
                "source_gap_material": SOURCE_GAP_MARGIN,
                "useful_auroc": USEFUL_AUROC,
            },
            "criterion_C1": criteria["C1"],
            "criterion_C2": criteria["C2"],
            "criterion_C3": criteria["C3"],
            "criterion_C4": criteria["C4"],
            "criterion_C5": criteria["C5"],
            "criterion_C6": criteria["C6"],
            "dominant_reliability_mechanism": _dominant_mechanism(criteria, gap_harm),
            "outcome": outcome,
            "outcome_label": OUTCOME_TAXONOMY[outcome],
            "unmet_criteria": failed,
            "ready_for_reliability_adaptation": outcome in ("A", "B"),
            "ready_for_external_confirmation": False,
            "confirmatory_reserve_consumed": False,
            "deployable": False,
            "selects_a_deployment_threshold": False,
            "falsification_tests_passed": int(negative.get("passed", 0)),
            "falsification_tests_total": int(negative.get("total", 0)),
            "recommended_next_stage": NEXT_STAGE[outcome],
            "reason": reason,
            "hypothesis_id": HYPOTHESIS,
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: outcome {outcome} -- {NEXT_STAGE[outcome]}")
    return 0


def run_decompose() -> int:
    """Section 43: where the remaining loss sits, in RL1's own ladder and nowhere else."""
    started = time.monotonic()
    _require(DECISION, "decide")
    _forbid(RELIABILITY_DECOMPOSITION)
    decision = cc_read_json(DECISION)
    frontier = cc_read_json(ORACLE_RISK_FRONTIER)["by_representation"][PRIMARY_REPRESENTATION]
    prefixes = cc_read_json(PREFIX_RANKING)
    inventory = cc_read_json(POPULATION_INVENTORY)["populations"][PRIMARY_POPULATION]
    key = epsilon_key(PRIMARY_EPSILON)
    perfect = 1.0
    ranking = float(frontier[key]["repair_recall"])
    _write_json_once(
        RELIABILITY_DECOMPOSITION,
        {
            **_analysis_envelope("reliability_decomposition"),
            "ladder": [
                {
                    "rung": "heterogeneous candidates",
                    "value": float(inventory["exact_prevalence"]),
                    "meaning": "share of the frozen candidate universe that is an exact repair",
                },
                {
                    "rung": "representation / discrimination",
                    "value": float(decision["mixed_source_harm_auroc"]),
                    "meaning": "mixed-source harm AUROC of the proposed representation",
                },
                {
                    "rung": "generator transfer",
                    "value": float(decision["mean_leave_generator_out_harm_auroc"]),
                    "meaning": "the same quantity with the evaluated generator held out",
                },
                {
                    "rung": "oracle harm-constrained prefix",
                    "value": ranking,
                    "meaning": (
                        f"share of exact repairs inside the longest prefix whose harm rate stays "
                        f"at or below {PRIMARY_EPSILON}; oracle, analysis only"
                    ),
                },
                {
                    "rung": "threshold and deployment",
                    "value": None,
                    "meaning": "unresolved here by construction; no threshold is selected in RL1",
                },
            ],
            "gaps": {
                "representation_discrimination_gap": float(
                    perfect - float(decision["mixed_source_harm_auroc"])
                ),
                "generator_transfer_gap": float(
                    float(decision["mixed_source_harm_auroc"])
                    - float(decision["mean_leave_generator_out_harm_auroc"])
                ),
                "ranking_headroom_at_primary_epsilon": float(perfect - ranking),
            },
            "random_ranking_reference": prefixes["by_representation"]["random"][-1],
            "not_mixed_with_upstream": (
                "HY1's localization and generation losses are a different decomposition on a "
                "different denominator and are never added to these rungs"
            ),
            "threshold_and_deployment_unresolved": True,
        },
    )
    print(
        f"decompose: ranking headroom {perfect - ranking:.4f} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ section 49: falsification


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
    """Section 49: twenty tests, each of which would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    population = pd.read_parquet(CANDIDATE_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    scores = load_scores()
    primary = population[population["population"] == PRIMARY_POPULATION].reset_index(drop=True)
    freeze = cc_read_json(HY1_CANDIDATE_FREEZE)
    splits = cc_read_json(SPLIT_REGISTRY)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    record(
        "t01_hy1_candidate_hashes_reproduce_exactly",
        all(block["matches_hy1_provenance"] for block in freeze["tables"].values()),
        {k: v["sha256"][:12] for k, v in freeze["tables"].items()},
    )
    candidates = _hy1_candidates()
    rebuilt = hy1.arm_candidates(candidates, ARM_OF_POPULATION[PRIMARY_POPULATION])
    record(
        "t02_u5_reproduces_the_frozen_h5_candidate_universe",
        int(primary["source_count"].sum()) == len(rebuilt)
        and set(primary["candidate_id"]) <= set(rebuilt["candidate_id"].astype(str)),
        {"hy1_rows": len(rebuilt), "rl1_rows": len(primary)},
    )
    labels = candidates.set_index("candidate_id")
    joined = primary.set_index("candidate_id")
    shared = joined.index.intersection(labels.index)
    record(
        "t03_candidate_labels_reproduce_exactly",
        bool(
            (joined.loc[shared, "is_harmful"] == labels.loc[shared, "is_harmful"]).all()
            and (joined.loc[shared, "beneficial"] == labels.loc[shared, "beneficial"]).all()
            and (joined.loc[shared, "outcome"] == labels.loc[shared, "outcome"]).all()
        ),
        {"compared": len(shared)},
    )
    identity = [
        name
        for representation in AGNOSTIC_REPRESENTATIONS
        for name in columns_for(representation)
        if family_of(name) == FAM_SOURCE
    ]
    record("t04_no_agnostic_representation_carries_source_identity", not identity, identity)
    counterfactual = cc_read_json(COUNTERFACTUAL_SOURCE)
    record(
        "t05_r3_scores_are_invariant_to_source_relabeling",
        counterfactual["r3"]["max_absolute_score_change"] == 0.0,
        counterfactual["r3"],
    )
    record(
        "t06_r4_scores_can_respond_to_source_metadata",
        counterfactual["r4"]["max_absolute_score_change"] > 0.0,
        counterfactual["r4"],
    )
    # Section 38 licenses generator metadata for splitting, stratified reporting and diagnostics.
    # The path under test is the one that builds the matrix a model sees and scores it, which is
    # `columns_for` -> `_matrix_for` -> `fit_and_score`; `build_folds` is the licensed reader and
    # is deliberately excluded, and test 05 proves the score is invariant to it anyway.
    scoring_names = _names_in("_matrix_for") | _names_in("fit_and_score") | _names_in("columns_for")
    leaked_source = scoring_names & set(FORBIDDEN_SOURCE_COLUMNS)
    record(
        "t07_the_scoring_path_never_reads_generator_metadata",
        not leaked_source,
        {
            "leaked": sorted(leaked_source),
            "licensed_reader": "build_folds, for splitting and stratified reporting only",
            "fold_builder_reads": sorted(_names_in("build_folds") & set(FORBIDDEN_SOURCE_COLUMNS)),
        },
    )
    feature_names = (
        _names_in("build_features")
        | _names_in("text_block")
        | _names_in("confidence_block")
        | _names_in("geometry_block")
    )
    leaked_labels = feature_names & set(FORBIDDEN_LABEL_COLUMNS)
    record(
        "t08_no_ground_truth_field_can_enter_a_representation",
        not leaked_labels,
        sorted(leaked_labels),
    )
    folds = build_folds(PRIMARY_TRANSFER_SPLIT, primary)
    crossing = [
        fold.name
        for fold in folds
        if set(primary.iloc[fold.train]["corrector_source"])
        & set(primary.iloc[fold.test]["corrector_source"])
    ]
    record("t09_held_out_generator_candidates_never_enter_fitting", not crossing, crossing)
    document_crossing = {
        split: sum(int(row["shared_documents"]) for row in splits["splits"][split]["folds"])
        for split in (SPLIT_A, SPLIT_B_DOC, SPLIT_B_OVERLAP, SPLIT_C, SPLIT_W)
    }
    record(
        "t10_held_out_document_candidates_never_enter_fitting",
        all(value == 0 for value in document_crossing.values()),
        document_crossing,
    )
    record(
        "t11_candidate_duplicates_cannot_cross_split_groups",
        all(
            sum(int(row["shared_edit_groups"]) for row in splits["splits"][split]["folds"]) == 0
            for split in SPLITS
        ),
        splits["leakage"],
    )
    multi = primary[primary["multi_source"]]
    multi_in_transfer = set(multi["candidate_id"]) & set(
        cell(scores, PRIMARY_POPULATION, R3, PRIMARY_MODEL, PRIMARY_TRANSFER_SPLIT)["candidate_id"]
    )
    record(
        "t12_multi_source_candidates_follow_the_frozen_rule",
        len(multi_in_transfer) == 0 and int(primary["source_count"].sum()) > len(primary),
        {
            "multi_source": int(primary["multi_source"].sum()),
            "in_strict_transfer": len(multi_in_transfer),
        },
    )
    random_control = cc_read_json(RANDOM_RANKING_CONTROL)["by_target"]
    record(
        "t13_random_ranking_is_near_random",
        all(abs(random_control[t]["mean_auroc"] - 0.5) < 0.02 for t in TARGETS),
        {t: random_control[t]["mean_auroc"] for t in TARGETS},
    )
    permutation = cc_read_json(LABEL_PERMUTATION_CONTROL)["by_target"]
    record(
        "t14_label_permutation_destroys_discrimination",
        all(permutation[t]["discrimination_destroyed"] for t in TARGETS),
        {t: permutation[t]["mean_auroc_against_permuted_labels"] for t in TARGETS},
    )
    block = cell(scores, PRIMARY_POPULATION, R3, PRIMARY_MODEL, SPLIT_A)
    forward = auroc(block["score_harm"].to_numpy(float), block["is_harmful"].to_numpy(bool))
    flipped = auroc(-block["score_harm"].to_numpy(float), block["is_harmful"].to_numpy(bool))
    record(
        "t15_flipping_the_score_orientation_flips_auroc",
        abs((forward + flipped) - 1.0) < 1e-9,
        {"forward": forward, "flipped": flipped},
    )
    r2_block = cell(scores, PRIMARY_POPULATION, R2, PRIMARY_MODEL, SPLIT_A)
    a3_block = cell(scores, PRIMARY_POPULATION, A3, PRIMARY_MODEL, SPLIT_A)
    merged = r2_block[["candidate_id", "score_harm", "score_benefit"]].merge(
        a3_block[["candidate_id", "score_harm", "score_benefit"]],
        on="candidate_id",
        suffixes=("_r2", "_a3"),
        validate="one_to_one",
    )
    difference = float(
        max(
            np.abs(
                merged[f"score_{target}_r2"].to_numpy(float)
                - merged[f"score_{target}_a3"].to_numpy(float)
            ).max()
            for target in TARGETS
        )
    )
    record(
        "t16_removing_visual_features_reproduces_r2",
        difference == 0.0 and columns_for(R3, (FAM_VIS,)) == columns_for(R2),
        {"max_absolute_score_difference": difference},
    )
    stripped = score_split(primary, matrix, columns_for(R4, (FAM_SOURCE,)), M0, SPLIT_A)
    r3_block = cell(scores, PRIMARY_POPULATION, R3, PRIMARY_MODEL, SPLIT_A)
    merged = r3_block[["candidate_id", "score_harm", "score_benefit"]].merge(
        stripped[["candidate_id", "score_harm", "score_benefit"]],
        on="candidate_id",
        suffixes=("_r3", "_stripped"),
        validate="one_to_one",
    )
    stripped_difference = float(
        max(
            np.abs(
                merged[f"score_{target}_r3"].to_numpy(float)
                - merged[f"score_{target}_stripped"].to_numpy(float)
            ).max()
            for target in TARGETS
        )
    )
    record(
        "t17_removing_source_identity_from_r4_reproduces_r3",
        stripped_difference == 0.0,
        {"max_absolute_score_difference": stripped_difference},
    )
    missing = cc_read_json(MISSING_FEATURE_POLICY)
    indicator_auroc = {
        name: auroc(
            matrix.merge(primary[["candidate_id"]], on="candidate_id")[name].to_numpy(float),
            (
                primary.merge(matrix[["candidate_id"]], on="candidate_id")[
                    "corrector_source"
                ].astype(str)
                == C1
            ).to_numpy(bool),
        )
        for name in ("conf_missing", "geom_missing", "vis_missing")
    }
    record(
        "t18_missing_indicators_do_not_encode_the_generator",
        all(
            (not np.isfinite(value)) or abs(value - 0.5) < 0.05
            for value in indicator_auroc.values()
        ),
        {"source_auroc": indicator_auroc, "documented": missing["missingness"]},
    )
    oracle_names = (
        _names_in("fit_and_score") | _names_in("score_split") | _names_in("build_features")
    )
    record(
        "t19_oracle_thresholds_cannot_influence_a_fitted_score",
        not (oracle_names & {"oracle_frontier", "ORACLE_RISK_FRONTIER", "epsilon", "EPSILONS"}),
        sorted(oracle_names & {"oracle_frontier", "ORACLE_RISK_FRONTIER", "epsilon", "EPSILONS"}),
    )
    record(
        "t20_changing_the_harm_epsilon_cannot_alter_a_candidate_score",
        bool(
            not (
                (_names_in("fit_and_score") | _names_in("run_fit"))
                & {"PRIMARY_EPSILON", "EPSILONS", "epsilon_key"}
            )
        ),
        sorted(
            (_names_in("fit_and_score") | _names_in("run_fit"))
            & {"PRIMARY_EPSILON", "EPSILONS", "epsilon_key"}
        ),
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


# ------------------------------------------------------------------ section 47: figures

FIGURE_NOTE = (
    "ANALYSIS ONLY -- development evidence computed with ground truth; not a deployable method"
)
SURFACE = lp1.SURFACE
REPRESENTATION_COLOURS = {
    R1: "#2a78d6",
    R2: "#eb6834",
    R3: "#1baf7a",
    R4: "#eda100",
    "random": "#e87ba4",
    "source_free_baseline": "#008300",
}
REPRESENTATION_LABELS = {R1: "R1 core", R2: "R2 +plaus", R3: "R3 +visual", R4: "R4 +source"}
FIGURES = (
    "mixed_source_harm_auroc.png",
    "mixed_source_benefit_auroc.png",
    "cross_generator_harm_transfer.png",
    "cross_generator_benefit_transfer.png",
    "cross_generator_transfer_matrix.png",
    "source_aware_vs_agnostic.png",
    "visual_evidence_effect.png",
    "feature_ablation.png",
    "harm_coverage_curve.png",
    "benefit_coverage_curve.png",
    "oracle_risk_frontier.png",
    "reliability_by_generator.png",
    "reliability_by_proposal_source.png",
    "reliability_by_environment.png",
    "reliability_by_error_type.png",
    "score_distribution_by_outcome.png",
    "score_distribution_by_generator.png",
    "implicit_source_predictability.png",
    "reliability_decomposition.png",
)


def run_figures() -> int:
    """The nineteen required figures, every value read from a persisted artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    _require(FALSIFICATION, "negative")
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
            "svg.hashsalt": "rl1",
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

    mixed = cc_read_json(MIXED_SOURCE_RESULTS)
    pooled = mixed["by_population"][PRIMARY_POPULATION][PRIMARY_MODEL]
    labels = [REPRESENTATION_LABELS[r] for r in REPRESENTATIONS]
    colours = [REPRESENTATION_COLOURS[r] for r in REPRESENTATIONS]
    for target, name, caption in (
        (T_HARM, FIGURES[0], "mixed-source harm AUROC per representation"),
        (T_BENEFIT, FIGURES[1], "mixed-source benefit AUROC per representation"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        lp1._bars(ax, labels, [pooled[r][f"{target}_auroc"] for r in REPRESENTATIONS], colours)
        ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{target} AUROC (document-held-out, out of fold)")
        ax.set_title(f"Mixed-source {target} discrimination")
        save(fig, name, [MIXED_SOURCE_RESULTS], caption)

    logo = cc_read_json(LEAVE_GENERATOR_OUT)["by_split"][PRIMARY_TRANSFER_SPLIT][PRIMARY_MODEL]
    for target, name, caption in (
        (T_HARM, FIGURES[2], "leave-one-generator-out harm AUROC per direction"),
        (T_BENEFIT, FIGURES[3], "leave-one-generator-out benefit AUROC per direction"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        lp1._grouped(
            ax,
            [f"held out {c}" for c in CORRECTORS],
            {
                r: [float(logo[r][c][f"{target}_auroc"]) for c in CORRECTORS]
                for r in REPRESENTATIONS
            },
            REPRESENTATION_COLOURS,
            REPRESENTATION_LABELS,
        )
        ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{target} AUROC on the held-out generator")
        ax.set_title(f"Cross-generator {target} transfer")
        save(fig, name, [LEAVE_GENERATOR_OUT], caption)

    matrix_rows = cc_read_json(TRANSFER_MATRIX)["by_representation"][PRIMARY_REPRESENTATION]
    grid = np.full((len(CORRECTORS), len(CORRECTORS)), np.nan)
    for row in matrix_rows:
        grid[CORRECTORS.index(row["train_source"]), CORRECTORS.index(row["test_source"])] = row[
            "harm_auroc"
        ]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    image = ax.imshow(grid, cmap="viridis", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(CORRECTORS)), [c.split("_")[0] for c in CORRECTORS])
    ax.set_yticks(range(len(CORRECTORS)), [c.split("_")[0] for c in CORRECTORS])
    ax.set_xlabel("evaluated generator")
    ax.set_ylabel("training generator")
    ax.grid(visible=False)
    for i in range(len(CORRECTORS)):
        for j in range(len(CORRECTORS)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title(f"{PRIMARY_REPRESENTATION}: harm AUROC by train and test generator")
    save(fig, FIGURES[4], [TRANSFER_MATRIX], "cross-generator harm transfer matrix")

    gap = cc_read_json(SOURCE_IDENTITY_GAP)["mixed_source"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lp1._grouped(
        ax,
        list(TARGETS),
        {
            R3: [float(gap[t]["r3"]) for t in TARGETS],
            R4: [float(gap[t]["r4"]) for t in TARGETS],
        },
        REPRESENTATION_COLOURS,
        REPRESENTATION_LABELS,
    )
    ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
    ax.set_ylabel("AUROC (mixed source)")
    ax.set_title("Source-agnostic R3 against the source-aware reference R4")
    save(fig, FIGURES[5], [SOURCE_IDENTITY_GAP], "source-aware versus source-agnostic AUROC")

    visual = cc_read_json(VISUAL_ABLATION)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lp1._bars(
        ax,
        ["mixed harm", "mixed benefit", "LOGO harm", "LOGO benefit"],
        [
            float(visual["mixed_source"]["harm_gain"]),
            float(visual["mixed_source"]["benefit_gain"]),
            float(visual["leave_generator_out"]["harm_gain"]),
            float(visual["leave_generator_out"]["benefit_gain"]),
        ],
        [REPRESENTATION_COLOURS[R3]] * 4,
    )
    ax.axhline(0.0, color=xc1.INK_SECONDARY, linewidth=0.8)
    ax.set_ylabel("AUROC gain of R3 over R2")
    ax.set_title("What the document image adds to reliability")
    save(fig, FIGURES[6], [VISUAL_ABLATION], "visual-evidence effect, R3 minus R2")

    ablation = cc_read_json(FEATURE_ABLATION)["by_ablation"]
    names = [name for name in ABLATIONS if name in ablation]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    lp1._bars(
        ax,
        [name.split("_", 1)[0] for name in names],
        [float(ablation[name]["harm_delta_vs_full"]) for name in names],
        [REPRESENTATION_COLOURS[R1]] * len(names),
    )
    ax.axhline(0.0, color=xc1.INK_SECONDARY, linewidth=0.8)
    ax.set_ylabel("harm AUROC minus full R3")
    ax.set_title("Which evidence families are load-bearing")
    save(fig, FIGURES[7], [FEATURE_ABLATION], "pre-registered feature ablations on R3")

    prefixes = cc_read_json(PREFIX_RANKING)["by_representation"]
    for target, key, name, caption in (
        (T_HARM, "harm_rate", FIGURES[8], "coverage against harm rate along the ranked list"),
        (
            T_BENEFIT,
            "beneficial_rate",
            FIGURES[9],
            "coverage against beneficial rate along the ranked list",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for series in (*REPRESENTATIONS, "random", "source_free_baseline"):
            rows = prefixes.get(series)
            if not rows:
                continue
            ax.plot(
                [row["coverage"] for row in rows],
                [row[key] for row in rows],
                marker="o",
                markersize=3,
                linewidth=1.3,
                color=REPRESENTATION_COLOURS.get(series, xc1.INK_SECONDARY),
                label=REPRESENTATION_LABELS.get(series, series),
            )
        ax.set_xlabel("coverage (share of the ranked candidate list accepted)")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(f"Ranking geometry: coverage against {target}")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, name, [PREFIX_RANKING], caption)

    frontier = cc_read_json(ORACLE_RISK_FRONTIER)["by_representation"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lp1._grouped(
        ax,
        [f"harm <= {e}" for e in EPSILONS],
        {
            r: [float(frontier[r][epsilon_key(e)]["repair_recall"]) for e in EPSILONS]
            for r in REPRESENTATIONS
        },
        REPRESENTATION_COLOURS,
        REPRESENTATION_LABELS,
    )
    ax.set_ylabel("exact-repair recall inside the admissible prefix")
    ax.set_title("Oracle risk frontier -- analysis only, not deployable")
    save(fig, FIGURES[10], [ORACLE_RISK_FRONTIER], "oracle harm-constrained repair recall")

    harm = cc_read_json(HARM_DISCRIMINATION)["by_representation"][PRIMARY_REPRESENTATION]
    benefit = cc_read_json(BENEFIT_DISCRIMINATION)["by_representation"][PRIMARY_REPRESENTATION]
    for column, name, title, caption in (
        ("by_corrector", FIGURES[11], "by correction generator", "reliability by generator"),
        (
            "by_proposal_stratum",
            FIGURES[12],
            "by proposal source",
            "reliability by proposal source",
        ),
        ("by_environment", FIGURES[13], "by environment", "reliability by environment"),
        ("by_error_kind", FIGURES[14], "by error type", "reliability by error type"),
    ):
        groups = sorted(set(harm[column]) | set(benefit[column]))
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        lp1._grouped(
            ax,
            [g.replace("_", " ") for g in groups],
            {
                T_HARM: [float(harm[column].get(g, float("nan"))) for g in groups],
                T_BENEFIT: [float(benefit[column].get(g, float("nan"))) for g in groups],
            },
            {T_HARM: REPRESENTATION_COLOURS[R2], T_BENEFIT: REPRESENTATION_COLOURS[R3]},
            {T_HARM: "harm AUROC", T_BENEFIT: "benefit AUROC"},
        )
        ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{PRIMARY_REPRESENTATION} discrimination {title}")
        ax.tick_params(axis="x", labelrotation=30)
        save(fig, name, [HARM_DISCRIMINATION, BENEFIT_DISCRIMINATION], caption)

    for key, name, caption in (
        ("score_by_outcome", FIGURES[15], "harm-score distribution by frozen outcome"),
        ("score_by_generator", FIGURES[16], "harm-score distribution by correction generator"),
    ):
        groups = sorted(harm[key])
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        positions = np.arange(len(groups))
        for offset, point in ((-0.22, "q25"), (0.0, "q50"), (0.22, "q75")):
            ax.scatter(
                positions + offset,
                [float(harm[key][g].get(point, float("nan"))) for g in groups],
                s=18,
                label=point,
            )
        for position, group in zip(positions, groups, strict=True):
            low = float(harm[key][group].get("q05", float("nan")))
            high = float(harm[key][group].get("q95", float("nan")))
            ax.plot([position, position], [low, high], color=xc1.GRID, linewidth=6, zorder=0)
        ax.set_xticks(positions, [g.replace("_", " ") for g in groups])
        ax.set_ylabel("P(harmful) from R3")
        ax.set_title(caption)
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, name, [HARM_DISCRIMINATION], caption)

    leakage = cc_read_json(IMPLICIT_SOURCE_LEAKAGE)["results"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lp1._bars(
        ax,
        [REPRESENTATION_LABELS[r] for r in AGNOSTIC_REPRESENTATIONS],
        [float(leakage[r]["source_auroc"]) for r in AGNOSTIC_REPRESENTATIONS],
        [REPRESENTATION_COLOURS[r] for r in AGNOSTIC_REPRESENTATIONS],
    )
    ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
    ax.set_ylabel("AUROC of a diagnostic classifier predicting the generator")
    ax.set_title("How much of the generator survives in a representation that never names it")
    save(fig, FIGURES[17], [IMPLICIT_SOURCE_LEAKAGE], "implicit source predictability")

    ladder = cc_read_json(RELIABILITY_DECOMPOSITION)["ladder"]
    rungs = [row for row in ladder if row["value"] is not None]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    lp1._bars(
        ax,
        [row["rung"].replace(" / ", "\n") for row in rungs],
        [float(row["value"]) for row in rungs],
        [
            REPRESENTATION_COLOURS[R1],
            REPRESENTATION_COLOURS[R2],
            REPRESENTATION_COLOURS[R3],
            REPRESENTATION_COLOURS[R4],
        ][: len(rungs)],
    )
    ax.set_ylabel("value on each rung's own scale")
    ax.set_title("Reliability decomposition after hybrid integration")
    ax.tick_params(axis="x", labelrotation=15)
    save(fig, FIGURES[18], [RELIABILITY_DECOMPOSITION], "the RL1 reliability ladder")

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


# ------------------------------------------------------------------ sections 50-52: record

DERIVED_OUTPUTS = (
    "CANDIDATE_POPULATION",
    "POPULATION_INVENTORY",
    "GROUP_REGISTRY",
    "FEATURE_MATRIX",
    "FEATURE_AUDIT",
    "FEATURE_REGISTRY",
    "MISSING_FEATURE_POLICY",
    "R0_LEGACY",
    "R1_REPRESENTATION",
    "R2_REPRESENTATION",
    "R3_REPRESENTATION",
    "R4_REPRESENTATION",
    "SPLIT_REGISTRY",
    "IMPLICIT_SOURCE_LEAKAGE",
    "MODEL_REGISTRY",
    "HYPERPARAMETER_REGISTRY",
    "TRAINING_REGISTRY",
    "SCORES",
    "MIXED_SOURCE_RESULTS",
    "HARM_DISCRIMINATION",
    "BENEFIT_DISCRIMINATION",
    "AVERAGE_PRECISION",
    "SOURCE_STRATIFIED",
    "PROPOSAL_STRATIFIED",
    "ERROR_TYPE_ANALYSIS",
    "ENGINE_ANALYSIS",
    "DOMAIN_ANALYSIS",
    "LEAVE_GENERATOR_OUT",
    "LEAVE_ENVIRONMENT_OUT",
    "GENERATOR_ENVIRONMENT_STRESS",
    "TRANSFER_MATRIX",
    "PREFIX_RANKING",
    "ORACLE_RISK_FRONTIER",
    "VISUAL_ABLATION",
    "FEATURE_ABLATION",
    "SOURCE_IDENTITY_GAP",
    "RANDOM_RANKING_CONTROL",
    "LABEL_PERMUTATION_CONTROL",
    "COUNTERFACTUAL_SOURCE",
    "CONTROL_RESULTS",
    "STATISTICAL_TESTS",
    "RELIABILITY_DECOMPOSITION",
    "FALSIFICATION",
    "DECISION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("population", run_population),
        ("features", run_features),
        ("splits", run_splits),
        ("sourceleak", run_sourceleak),
        ("fit", run_fit),
        ("discriminate", run_discriminate),
        ("transfer", run_transfer),
        ("prefix", run_prefix),
        ("ablate", run_ablate),
        ("controls", run_controls),
        ("stats", run_stats),
        ("negative", run_negative),
        ("decide", run_decide),
        ("decompose", run_decompose),
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
    """Every derived phase, re-run from the frozen HY1 candidates into a directory of its own."""
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
    """Section 50: two independent regenerations from the frozen HY1 candidate universe."""
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
                "fit": FIT_SEED,
                "split": SPLIT_SEED,
                "bootstrap": BOOTSTRAP_SEED,
                "random_ranking": RANDOM_RANKING_SEED,
                "label_permutation": PERMUTATION_SEED,
                "counterfactual": COUNTERFACTUAL_SEED,
            },
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
    HY1_CANDIDATE_FREEZE,
    DESIGN_RECORD,
    ENVIRONMENT_INVENTORY,
    FITTED_RESOURCES,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE, DESIGN_RECORD),
    "2. Frozen HY1 Result": (HY1_CANDIDATE_FREEZE, POPULATION_INVENTORY, hy1.DECISION),
    "3. Reliability Problem After Hybrid Integration": (
        R0_LEGACY,
        POPULATION_INVENTORY,
        hy1.DECISION,
        xr1.DECISION,
    ),
    "4. Research Questions": (DESIGN_RECORD,),
    "5. Non-Goals": (DESIGN_RECORD,),
    "6. Frozen Candidate Universe": (
        POPULATION_INVENTORY,
        GROUP_REGISTRY,
        HY1_CANDIDATE_FREEZE,
    ),
    "7. Outcome Taxonomy": (DESIGN_RECORD,),
    "8. Why the Legacy Representation Is Incompatible": (
        R0_LEGACY,
        xr1.FEATURE_COMPATIBILITY,
    ),
    "9. Feature Audit": (FEATURE_AUDIT, FEATURE_REGISTRY, MISSING_FEATURE_POLICY),
    "10. Generator-Identity Leakage": (IMPLICIT_SOURCE_LEAKAGE, FEATURE_AUDIT),
    "11. Generator-Agnostic Core Representation": (
        FEATURE_REGISTRY,
        FITTED_RESOURCES,
        FEATURE_AUDIT,
    ),
    "12. Candidate-Plausibility Representation": (
        FEATURE_REGISTRY,
        FITTED_RESOURCES,
        FEATURE_ABLATION,
    ),
    "13. Visual Reliability Evidence": (
        FEATURE_REGISTRY,
        FITTED_RESOURCES,
        MISSING_FEATURE_POLICY,
        VISUAL_ABLATION,
    ),
    "14. Source-Aware Reference": (FEATURE_REGISTRY, SOURCE_IDENTITY_GAP),
    "15. Model Family": (
        MODEL_REGISTRY,
        HYPERPARAMETER_REGISTRY,
        TRAINING_REGISTRY,
        MIXED_SOURCE_RESULTS,
    ),
    "16. Split Design": (SPLIT_REGISTRY, GROUP_REGISTRY),
    "17. Mixed-Source Evaluation": (MIXED_SOURCE_RESULTS, STATISTICAL_TESTS),
    "18. Leave-Generator-Out Evaluation": (LEAVE_GENERATOR_OUT, STATISTICAL_TESTS),
    "19. Leave-Environment-Out Evaluation": (
        LEAVE_ENVIRONMENT_OUT,
        LEAVE_GENERATOR_OUT,
    ),
    "20. Harm Discrimination": (HARM_DISCRIMINATION,),
    "21. Benefit Discrimination": (BENEFIT_DISCRIMINATION,),
    "22. Average Precision": (AVERAGE_PRECISION,),
    "23. Cross-Generator Transfer Matrix": (TRANSFER_MATRIX,),
    "24. Current-Generator to Qwen Transfer": (LEAVE_GENERATOR_OUT, TRANSFER_MATRIX),
    "25. Qwen to Current-Generator Transfer": (LEAVE_GENERATOR_OUT, TRANSFER_MATRIX),
    "26. Source-Aware versus Source-Agnostic": (SOURCE_IDENTITY_GAP, STATISTICAL_TESTS),
    "27. Implicit Source Predictability": (IMPLICIT_SOURCE_LEAKAGE,),
    "28. Visual Evidence Effect": (VISUAL_ABLATION, STATISTICAL_TESTS),
    "29. Feature Ablations": (FEATURE_ABLATION,),
    "30. Prefix Ranking": (PREFIX_RANKING, ORACLE_RISK_FRONTIER),
    "31. Oracle Risk Frontier": (ORACLE_RISK_FRONTIER,),
    "32. Proposal-Source Analysis": (PROPOSAL_STRATIFIED, LEAVE_GENERATOR_OUT),
    "33. Corrector-Source Analysis": (SOURCE_STRATIFIED, POPULATION_INVENTORY),
    "34. Substitution": (ERROR_TYPE_ANALYSIS,),
    "35. Omission": (ERROR_TYPE_ANALYSIS,),
    "36. Segmentation": (ERROR_TYPE_ANALYSIS,),
    "37. Spurious Insertion": (ERROR_TYPE_ANALYSIS,),
    "38. Environment Analysis": (ENGINE_ANALYSIS, MIXED_SOURCE_RESULTS),
    "39. Engine Analysis": (ENGINE_ANALYSIS,),
    "40. Domain Analysis": (DOMAIN_ANALYSIS,),
    "41. Random-Ranking Control": (RANDOM_RANKING_CONTROL, CONTROL_RESULTS, STATISTICAL_TESTS),
    "42. Label-Permutation Control": (
        LABEL_PERMUTATION_CONTROL,
        CONTROL_RESULTS,
        STATISTICAL_TESTS,
    ),
    "43. Counterfactual Source Test": (COUNTERFACTUAL_SOURCE, CONTROL_RESULTS),
    "44. Falsification Tests": (FALSIFICATION, DETERMINISM),
    "45. Statistics": (STATISTICAL_TESTS,),
    "46. Reliability Decomposition": (
        RELIABILITY_DECOMPOSITION,
        ORACLE_RISK_FRONTIER,
        DECISION,
    ),
    "47. Limitations": (
        DESIGN_RECORD,
        POPULATION_INVENTORY,
        IMPLICIT_SOURCE_LEAKAGE,
        LEAVE_GENERATOR_OUT,
        ERROR_TYPE_ANALYSIS,
        GENERATOR_ENVIRONMENT_STRESS,
        DECISION,
    ),
    "48. Research Decision": (DECISION, STATISTICAL_TESTS),
    "49. Implications for Deployment": (DECISION, ORACLE_RISK_FRONTIER, PREFIX_RANKING),
    "50. Next Stage": (
        DECISION,
        LEAVE_GENERATOR_OUT,
        SOURCE_IDENTITY_GAP,
        TRANSFER_MATRIX,
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
            "hy1_unchanged": not cc_read_json(HY1_CANDIDATE_FREEZE)["tables"]
            or all(
                block["matches_hy1_provenance"]
                for block in cc_read_json(HY1_CANDIDATE_FREEZE)["tables"].values()
            ),
            "decision_rederived_identically": rederived,
            "regenerates_a_correction": False,
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
        "population": run_population,
        "resources": run_resources,
        "features": run_features,
        "splits": run_splits,
        "sourceleak": run_sourceleak,
        "fit": run_fit,
        "discriminate": run_discriminate,
        "transfer": run_transfer,
        "prefix": run_prefix,
        "ablate": run_ablate,
        "controls": run_controls,
        "stats": run_stats,
        "negative": run_negative,
        "decide": run_decide,
        "decompose": run_decompose,
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
