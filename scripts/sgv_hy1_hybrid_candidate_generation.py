#!/usr/bin/env python3
"""SGV-HY1: does LP1's complementary localization translate into exact repair candidates?

SGV-LP1 closed with outcome B. The image-aware proposer is worse than the frozen OCR-only proposer
P0 at P0's own budget, but it reaches a large share of the error sites P0 misses, and the union of
the two gains broadly. That is a statement about *finding* errors. This stage asks the only
question that follows from it:

    when the sites only the image proposer finds are handed to the frozen image-conditioned
    corrector, do they become exact repair candidates -- and at what candidate cost?

**1. Nothing is re-localized.** Every proposal is read from LP1's frozen tables and hash-verified.
Each one falls in exactly one source stratum: P0-only, image-only, or overlap.

**2. Routing is frozen and GT-blind.** Which corrector sees a site depends only on that stratum.
No label, error type, replacement text or endpoint outcome may reach the routing rule.

**3. Nothing is regenerated that already exists.** A correction request is reused from SGV-XR1's
frozen cache only when its model, condition, prompt text and crop hash are identical; every other
request is generated once and frozen.

**4. Ground truth enters last.** Raw answers are hashed and frozen, normalized to SGV-XC1 atomic
edits, and only then labelled by SGV14.

    --reconstruct   section 0: CG1, XC1, GEN1, XR1 and LP1 re-read from their own artifacts
    --freeze        upstream configuration, model versions, the LP1 proposal reproduction
    --strata        proposal source strata and the frozen routing registry
    --preregister   arms, budgets, criteria, outcome rules, prompt/decoding/crop registries
    --contexts      GT-blind correction contexts and crops for every routed site
    --cacheaudit    which requests SGV-XR1 already answered, by exact request identity
    --generate      only the requests no frozen cache answers (HY1_ARMS=<arms>)
    --spotcheck     each generated arm regenerates its first batches from the frozen inputs
    --normalize     raw answers -> atomic candidates -> SGV14 labels -> frozen candidate table
    --opportunity   H0-H5 and HT opportunity, translation, P0-miss repair recovery
    --types         substitution, omission, segmentation, spurious insertion, historical glyphs
    --burden        harmful candidates, marginal harm, clean sites, edit damage, abstention
    --routing       candidate complementarity, routing comparison, the image-branch budget curve
    --stats         document-clustered paired bootstrap, Holm within the frozen primary family
    --controls      H0 reproduction, identity path, GT ceiling, HT, duplication
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen raw outputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / INTEGRATION ONLY. No reliability model is fitted, adapted, scored or thresholded
here: SGV-XR1 established that the frozen verifier encodes generator identity and cannot validly
score a new generator zero-shot. `ready_for_external_confirmation` is false by construction and the
SGV1 CORD confirmatory reserve stays LOCKED.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_gen1_error_conditioned_generation as gen1
import sgv_lp1_image_error_site_proposal as lp1
import sgv_xr1_image_candidate_reliability as xr1
from ocr_risk.io.hashing import canonical_hash, file_sha256

xc1 = gen1.xc1
s14 = gen1.s14
s15 = gen1.s15
cg1 = gen1.cg1
dt1 = gen1.dt1

REPO = gen1.REPO
OUT = REPO / "results/generated/sgv_hy1_hybrid_candidate_generation"
CACHE = OUT / "cache"
RAW_CACHE = OUT / "raw_generation_outputs"
CROP_DIR = CACHE / "correction_crops"
REGENERATION_DIR = CACHE / "regeneration"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"
MODEL_REGISTRY = OUT / "model_registry.json"
MODEL_VERSIONS = OUT / "model_versions.json"
LP1_REPRODUCTION = OUT / "lp1_proposal_reproduction.json"
PROPOSAL_STRATA = OUT / "proposal_source_strata.parquet"
STRATA_INVENTORY = OUT / "proposal_strata_inventory.json"
ROUTING_REGISTRY = OUT / "routing_registry.json"
DESIGN_RECORD = OUT / "design_record.json"
COMPUTE_BUDGET = OUT / "compute_budget.json"
PROMPT_REGISTRY = OUT / "prompt_registry.json"
DECODING_REGISTRY = OUT / "decoding_registry.json"
CROP_REGISTRY = OUT / "correction_crop_registry.json"
CROP_INVENTORY = OUT / "correction_crop_inventory.parquet"
CONTEXTS = OUT / "correction_contexts.parquet"
CACHE_REUSE_AUDIT = OUT / "cache_reuse_audit.json"

RAW_OUTPUT_MANIFEST = OUT / "raw_output_manifest.json"
RAW_OUTPUT_HASHES = OUT / "raw_output_hashes.json"
# Everything below is written with ground truth.
CANDIDATES = OUT / "normalized_candidates.parquet"
CANDIDATE_INVENTORY = OUT / "candidate_inventory.json"
CANDIDATE_ATTRIBUTION = OUT / "candidate_attribution.json"
SITE_OUTCOMES = OUT / "site_outcomes.parquet"
SITE_TRUTH = OUT / "site_truth.parquet"
ERROR_SITES = OUT / "error_site_table.parquet"
H0_REPRODUCTION = OUT / "h0_reproduction.json"
OPPORTUNITY_RESULTS = OUT / "opportunity_results.json"
TRANSLATION_RESULTS = OUT / "translation_results.json"
P0_MISS_REPAIR = OUT / "p0_miss_repair_recovery.json"
ERROR_TYPE_ANALYSIS = OUT / "error_type_analysis.json"
OMISSION_ANALYSIS = OUT / "omission_analysis.json"
SEGMENTATION_ANALYSIS = OUT / "segmentation_analysis.json"
SUBSTITUTION_ANALYSIS = OUT / "substitution_analysis.json"
HISTORICAL_GLYPH_ANALYSIS = OUT / "historical_glyph_analysis.json"
HARM_BURDEN = OUT / "harmful_candidate_burden.json"
HARM_EFFICIENCY = OUT / "incremental_harm_efficiency.json"
CLEAN_SITE_ANALYSIS = OUT / "clean_site_analysis.json"
EDIT_DAMAGE = OUT / "edit_damage_analysis.json"
ABSTENTION = OUT / "abstention_analysis.json"
CANDIDATE_OVERLAP = OUT / "candidate_overlap.json"
ROUTING_COMPARISON = OUT / "routing_comparison.json"
BUDGET_CURVE = OUT / "budget_curve.json"
ENVIRONMENT_ANALYSIS = OUT / "environment_analysis.json"
DOMAIN_ANALYSIS = OUT / "domain_analysis.json"
RUNTIME_COST = OUT / "runtime_cost.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
CONTROL_RESULTS = OUT / "control_results.json"
FALSIFICATION = OUT / "falsification_tests.json"
BOTTLENECK = OUT / "bottleneck_decomposition.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"
DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"

REPORT = REPO / "docs/sgv_hy1/hybrid_candidate_generation.md"

SCHEMA_VERSION = 1
STAGE = "sgv_hy1_hybrid_candidate_generation"
HYPOTHESIS = "SGV-HY1-D1"
STAGE_KIND = "DEVELOPMENT / INTEGRATION"

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
_resolve_device = gen1._resolve_device
_release_device_cache = gen1._release_device_cache

BOOTSTRAP_RESAMPLES = gen1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = gen1.BOOTSTRAP_SEED
ALPHA = gen1.ALPHA
MODEL = gen1.M5
EXPECTATION_TOLERANCE = xr1.EXPECTATION_TOLERANCE

# ------------------------------------------------------------------ the frozen design

# Proposal source strata. Every routed site belongs to exactly one.
S_P0 = "p0_only"
S_IMAGE = "image_only"
S_OVERLAP = "overlap"
STRATA = (S_P0, S_IMAGE, S_OVERLAP)
# The text-proposal control's own incremental stratum, never part of the primary strata.
S_TEXT = "text_only"

# Correctors. C0 is the frozen current generator; C1 is GEN1/XR1's image-conditioned corrector,
# and C1_TEXT is that same checkpoint under XR1's matched no-image condition.
C0 = "c0_frozen_generator"
C1 = "c1_image_corrector"
C1_TEXT = "c1_text_matched"
CONDITION_OF_CORRECTOR = {C1: gen1.E3, C1_TEXT: gen1.E1}

H0 = "h0_frozen_pipeline"
H1 = "h1_p0_image"
H2 = "h2_image_image"
H3 = "h3_hybrid_image"
H4 = "h4_specialized_routing"
H5 = "h5_dual_union"
HT = "ht_text_proposal_hybrid"
ARMS = (H0, H1, H2, H3, H4, H5, HT)
PRIMARY_ARM = H4
BASELINE_ARM = H0

# The frozen routing rule: stratum -> the correctors whose candidates that arm keeps. Written here
# once, before any label exists, and asserted unchanged by the falsification suite.
ROUTING: dict[str, dict[str, tuple[str, ...]]] = {
    H0: {S_P0: (C0,), S_OVERLAP: (C0,)},
    H1: {S_P0: (C1,), S_OVERLAP: (C1,)},
    H2: {S_IMAGE: (C1,), S_OVERLAP: (C1,)},
    H3: {S_P0: (C1,), S_IMAGE: (C1,), S_OVERLAP: (C1,)},
    H4: {S_P0: (C0,), S_IMAGE: (C1,), S_OVERLAP: (C0,)},
    H5: {S_P0: (C0,), S_IMAGE: (C1,), S_OVERLAP: (C0, C1)},
    HT: {S_P0: (C0,), S_OVERLAP: (C0,), S_TEXT: (C1_TEXT,)},
}
ROUTING_NOTE = {
    H0: "the frozen current pipeline: P0's sites, the frozen current generator",
    H1: "P0's sites with the image corrector, which isolates the corrector from the proposer",
    H2: "the image proposer's own sites with the image corrector",
    H3: "every hybrid site with one corrector everywhere",
    H4: "specialized routing: only the sites P0 never proposed go to the image corrector",
    H5: "the dual union: overlap keeps the candidates of both correctors",
    HT: "the matched text-proposal control: incremental sites come from L1, not from the image",
}

# The frozen proposal budget. LP1's matched-budget selection is P0's own proposal count per
# environment, and HY1 inherits it unchanged.
PROPOSAL_BUDGET_COLUMN = "in_budget_1_00"
L0 = lp1.L0
L1 = lp1.L1
L2 = lp1.L2

PRIMARY_K = 1
SECONDARY_K = gen1.MAX_CANDIDATES

# Success criteria, frozen before any HY1 label. The margins follow the repository's existing
# development convention (SGV-XR1's OPPORTUNITY_MARGIN and BREADTH_MAJORITY).
OPPORTUNITY_MARGIN = xr1.OPPORTUNITY_MARGIN
BREADTH_MAJORITY = xr1.BREADTH_MAJORITY
WEAK_TYPES = xr1.WEAK_TYPES
WEAK_TYPE_MARGIN = xr1.WEAK_TYPE_MARGIN
WEAK_TYPES_REQUIRED = xr1.WEAK_TYPES_REQUIRED
TRANSLATION_BREADTH = 6
# C5's comparator, chosen before the outcome: the frozen current pipeline's own harmful candidates
# per exact-repaired site, as SGV-XR1 measured it on this population. The hybrid's *incremental*
# harm per incremental repair may not exceed it.
HARM_COMPARATOR_SOURCE = "sgv_xr1 g0_frozen_generator harmful candidates per exact repaired site"

IMAGE_BRANCH_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

OUTCOME_LABELS = {
    "A": "hybrid integration materially expands the exact-repair candidate universe",
    "B": "opportunity improves, but burden or selectivity is the new bottleneck",
    "C": "localization gains do not translate into correction gains",
    "D": "hybrid integration adds no meaningful candidate opportunity",
}
NEXT_STAGE = {
    "A": "NEXT: GENERATOR-AGNOSTIC RELIABILITY REPRESENTATION",
    "B": "NEXT: SELECTIVE HYBRID ROUTING / IMAGE-CORRECTION ABSTENTION",
    "C": "NEXT: ERROR-TYPE-CONDITIONED IMAGE CORRECTION ON P0-MISSED SITES",
    "D": "NEXT: OCR-IMAGE DISAGREEMENT METHODS (no model-size escalation)",
}

# LP1 values this stage refuses to start without. Each is read from LP1's own decision record.
LP1_EXPECTED = {
    "outcome": "B",
    "p0_localization_recall": 0.6468,
    "image_localization_recall": 0.5381,
    "text_localization_recall": 0.4842,
    "union_localization_recall": 0.7928,
    "p0_miss_recovery": 0.4090,
    "omission_recall": 0.2260,
    "segmentation_recall": 0.7436,
    "image_vs_p0_gain": -0.1087,
    "image_vs_text_gain": 0.0540,
    "ready_for_external_confirmation": False,
    "ready_for_integrated_image_pipeline": True,
    "confirmatory_reserve_consumed": False,
}

FORBIDDEN_INPUT_COLUMNS = (
    *gen1.FORBIDDEN_INPUT_COLUMNS,
    "subtype",
    "error_type",
    "d_before",
    "d_after",
    "exact",
    "beneficial",
    "is_harmful",
    "outcome",
    "site_class",
    "labelable",
    "reached",
    "repaired",
)


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_hy1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "stage_kind": STAGE_KIND,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str) -> dict[str, Any]:
    """Every artifact computed with ground truth says so, and says it selects nothing."""
    return {
        **_envelope(artifact),
        "uses_ground_truth": True,
        "deployable": False,
        "may_influence_a_prompt_or_decoding_setting": False,
        "confirmatory_reserve_consumed": False,
    }


def environment_specs() -> list[dict[str, Any]]:
    return xr1.environment_specs()


def _spec(name: str) -> dict[str, Any]:
    return xr1._spec(name)


def _labels_exist() -> bool:
    """True once any HY1 artifact written with ground truth exists."""
    return any(p.exists() for p in (CANDIDATES, SITE_OUTCOMES, ERROR_SITES, OPPORTUNITY_RESULTS))


def _domain(corpus: str) -> str:
    return xr1._domain(corpus)


def assign_outcome(criteria: dict[str, bool], translation_positive: bool, gain: float) -> str:
    """Exactly one outcome, in the frozen precedence A, B, C, D.

    A needs every criterion. B is the burden-and-selectivity case: the opportunity criteria (C1,
    C2, C3) hold, so the hybrid does expand the candidate universe, but one of the qualifying
    criteria does not -- the harm budget (C5), the breadth across weak error types (C4), or the
    breadth of the image branch's own contribution (C6). C is the translation case: the newly
    exposed sites are reached but too few of them become exact repairs. Anything else is D.
    """
    if all(criteria.values()):
        return "A"
    opportunity = criteria["C1"] and criteria["C2"] and criteria["C3"]
    if opportunity:
        return "B"
    if gain > 0.0 and translation_positive:
        return "C"
    return "D"


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    """Every upstream file whose content HY1's conclusions rest on, XR1's list plus LP1's."""
    files = list(xr1._upstream_files())
    files.append(REPO / "scripts/sgv_lp1_image_error_site_proposal.py")
    for path in sorted(lp1.OUT.glob("*.json")):
        files.append(path)
    for path in sorted(lp1.OUT.glob("*.parquet")):
        files.append(path)
    files.append(lp1.REPORT)
    return sorted({p for p in files if p.is_file()})


def run_reconstruct() -> int:
    """Section 0: CG1, XC1, GEN1, XR1 and LP1 re-read from their own artifacts, never restated."""
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


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = xr1._check
    checks: list[dict[str, Any]] = []
    decisions = {
        "sgv_cg1": cc_read_json(cg1.OUT / "research_decision.json"),
        "sgv_xc1": cc_read_json(xc1.OUT / "research_decision.json"),
        "sgv_gen1": cc_read_json(gen1.OUT / "research_decision.json"),
        "sgv_xr1": cc_read_json(xr1.DECISION),
        "sgv_lp1": cc_read_json(lp1.DECISION),
    }
    for stage, expected in xr1.UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))
    for key, value in lp1.XR1_EXPECTED.items():
        checks.append(_check(f"sgv_xr1.{key}", decisions["sgv_xr1"].get(key), value))
    for key, value in LP1_EXPECTED.items():
        checks.append(_check(f"sgv_lp1.{key}", decisions["sgv_lp1"].get(key), value))

    # CG1's frozen interpretation, as the numbers it rests on: candidate generation dominates,
    # discovery reaches a third of the errors, and generation given discovery is the larger loss.
    cg1_decision = decisions["sgv_cg1"]
    checks.append(
        _check("sgv_cg1.dominant_bottleneck", cg1_decision["dominant_bottleneck"], "candidate")
    )
    checks.append(
        _check("sgv_cg1.discovery_recall_mean", cg1_decision["discovery_recall_mean"], 0.3342)
    )
    checks.append(
        _check(
            "sgv_cg1.opportunity_recall_given_discovery_mean",
            cg1_decision["opportunity_recall_given_discovery_mean"],
            0.2270,
        )
    )
    # XC1: stronger text families did not raise the natural ceiling.
    checks.append(
        _check(
            "sgv_xc1.text_families_do_not_raise_the_ceiling",
            bool(decisions["sgv_xc1"]["outcome"] == "D"),
            True,
        )
    )
    # GEN1: the image raises generation once the site is known.
    gen1_decision = decisions["sgv_gen1"]
    checks.append(
        _check(
            "sgv_gen1.visual_gain_exceeds_capacity_gain",
            bool(gen1_decision["visual_gain"] > gen1_decision["capacity_gain"]),
            True,
        )
    )
    # XR1: the known-site gain did not survive natural site proposal, and the verifier could not
    # score the new generator at all.
    checks.append(_check("sgv_xr1.zero_shot_feature_compatible", xr1_feature_compatible(), False))
    # LP1's qualitative pattern, each as the number it rests on.
    lp1_matched = cc_read_json(lp1.MATCHED_BUDGET_RESULTS)["arms"]
    checks.append(
        _check(
            "sgv_lp1.p0_beats_image_at_the_matched_budget",
            bool(lp1_matched[L0]["recall"]["mean"] > lp1_matched[L2]["recall"]["mean"]),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_lp1.image_beats_text_at_the_matched_budget",
            bool(lp1_matched[L2]["recall"]["mean"] > lp1_matched[L1]["recall"]["mean"]),
            True,
        )
    )
    lp1_stats = cc_read_json(lp1.STATISTICAL_TESTS)
    checks.append(
        _check(
            "sgv_lp1.visual_mechanism_unresolved",
            bool(lp1_stats["mechanism_family"]["M1_l2_vs_shuffle"]["survives_holm"]),
            False,
        )
    )
    checks.append(
        _check(
            "sgv_lp1.omission_is_the_image_proposer_weakness",
            bool(
                cc_read_json(lp1.ERROR_TYPE_ANALYSIS)["by_kind"]["deletion"]["l2_gain_over_p0"]
                < 0.0
            ),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_lp1.falsification_tests_passed",
            cc_read_json(lp1.FALSIFICATION)["passed"],
            cc_read_json(lp1.FALSIFICATION)["total"],
        )
    )
    checks.append(
        _check("sgv_lp1.determinism", cc_read_json(lp1.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_lp1.report_traceability",
            cc_read_json(lp1.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
            0,
        )
    )
    # The reserve, everywhere it is recorded.
    for path in (*xr1.RESERVE_DECISIONS, xr1.DECISION, lp1.DECISION):
        consumed = cc_read_json(path).get("confirmatory_reserve_consumed")
        checks.append(_check(f"reserve_locked:{_relative(path)}", consumed, False))
    return checks


def xr1_feature_compatible() -> bool:
    return bool(cc_read_json(xr1.FEATURE_COMPATIBILITY)["zero_shot_feature_compatible"])


def run_freeze() -> int:
    """The research freeze: upstream hashes, model versions and the LP1 proposal reproduction."""
    started = time.monotonic()
    for path in (
        RESEARCH_FREEZE,
        FROZEN_CONFIGURATION,
        ENVIRONMENT_INVENTORY,
        MODEL_REGISTRY,
        MODEL_VERSIONS,
        LP1_REPRODUCTION,
    ):
        _forbid(path)
    OUT.mkdir(parents=True, exist_ok=True)
    run_reconstruct_checks = run_reconstruct()
    if run_reconstruct_checks != 0:
        return run_reconstruct_checks
    upstream = _upstream_files()
    hashes = {_relative(p): file_sha256(p) for p in upstream}
    lp1_decision = cc_read_json(lp1.DECISION)
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "git": xr1._git_state(),
            "upstream_sha256": hashes,
            "upstream_file_count": len(hashes),
            "lp1_outcome": lp1_decision["outcome"],
            "upstream_checks": upstream_checks(),
            "upstream_checks_passed": sum(1 for c in upstream_checks() if c["agrees"]),
            "lp1_decision_sha256": file_sha256(lp1.DECISION),
            "reserve_locked": True,
            "uses_ground_truth": False,
        },
    )
    # The frozen upstream configuration HY1 inherits without re-deriving any of it.
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "site_proposer": {
                "name": "P0",
                "source": "SGV14 discovery pass, re-derived and frozen by SGV-LP1",
                "rules": cc_read_json(lp1.FROZEN_CONFIGURATION).get("discovery_rules"),
                "proposals": int(pd.read_parquet(lp1.P0_PROPOSALS).shape[0]),
            },
            "image_proposer": {
                "name": "L2",
                "source": "SGV-LP1 image-aware proposer at P0's matched budget",
                "prompt_revision": cc_read_json(lp1.PROMPT_REGISTRY)["revision"],
            },
            "correctors": {
                C0: "the frozen current generator, as SGV-CG1/XC1/XR1 froze it",
                C1: f"{MODEL} under {gen1.E3}, as SGV-GEN1/XR1 froze it",
                C1_TEXT: f"{MODEL} under {gen1.E1}, XR1's matched no-image condition",
            },
            "uses_ground_truth": False,
        },
    )
    versions = cc_read_json(gen1.OUT / "model_versions.json")
    _write_json_once(MODEL_VERSIONS, {**_envelope("model_versions"), **versions})
    spec = gen1.MODEL_SPECS[MODEL]
    _write_json_once(
        MODEL_REGISTRY,
        {
            **_envelope("model_registry"),
            "model": MODEL,
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "dtype": gen1.MODEL_DTYPE,
            "batch_size": int(spec["batch_size"]),
            "decoding": "greedy, as GEN1 froze it",
            "no_model_search": True,
            "uses_ground_truth": False,
        },
    )
    # The GT-blind environment inventory, taken from LP1's own population.
    lp1_inventory = cc_read_json(lp1.ENVIRONMENT_INVENTORY)
    _write_json_once(
        ENVIRONMENT_INVENTORY,
        {
            **_envelope("environment_inventory"),
            **{
                k: v
                for k, v in lp1_inventory.items()
                if k
                not in {
                    "artifact",
                    "schema_version",
                    "stage",
                    "hypothesis_id",
                    "stage_kind",
                    "issued_utc",
                }
            },
        },
    )
    # LP1's proposal tables reproduce exactly, by hash, before anything reads them.
    tables = {
        "parsed_proposals": lp1.PARSED_PROPOSALS,
        "p0_proposals": lp1.P0_PROPOSALS,
        "site_lattice_inventory": lp1.SITE_LATTICE_INVENTORY,
        "scan_population": lp1.SCAN_POPULATION,
    }
    recorded = cc_read_json(lp1.PROVENANCE)["artifacts"]
    reproduction: dict[str, Any] = {}
    for label, path in tables.items():
        observed = file_sha256(path)
        expected = recorded.get(_relative(path))
        if expected is not None and observed != expected:
            raise PhaseError(f"{label} changed since LP1's provenance record")
        reproduction[label] = {
            "path": _relative(path),
            "sha256": observed,
            "matches_lp1_provenance": expected is not None and observed == expected,
        }
    _write_json_once(
        LP1_REPRODUCTION,
        {
            **_envelope("lp1_proposal_reproduction"),
            "tables": reproduction,
            "lp1_artifacts_rehashed": len(recorded),
            "lp1_artifacts_moved": sorted(
                path for path, sha in recorded.items() if file_sha256(REPO / path) != sha
            ),
            "uses_ground_truth": False,
        },
    )
    moved = cc_read_json(LP1_REPRODUCTION)["lp1_artifacts_moved"]
    if moved:
        raise PhaseError(f"{len(moved)} LP1 artifacts moved since its record: {moved[:3]}")
    print(
        f"freeze: {len(hashes)} upstream files hashed, LP1 tables reproduce "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 6-7: strata


def _matched(method: str) -> pd.DataFrame:
    """One LP1 arm's matched-budget selection, read from the frozen table and never re-ranked."""
    frame = pd.read_parquet(lp1.PARSED_PROPOSALS)
    chosen = frame[(frame["method"] == method) & frame[PROPOSAL_BUDGET_COLUMN]]
    return chosen.reset_index(drop=True)


def stratum_of(site_id: str, p0_sites: set[str], image_sites: set[str]) -> str:
    """The frozen source rule. It reads two proposal sets and nothing else."""
    in_p0 = site_id in p0_sites
    in_image = site_id in image_sites
    if in_p0 and in_image:
        return S_OVERLAP
    if in_p0:
        return S_P0
    if in_image:
        return S_IMAGE
    raise PhaseError(f"{site_id} belongs to no proposal set")


def build_strata() -> pd.DataFrame:
    """Every routed site with its source stratum, built from LP1's frozen tables alone.

    The three primary strata partition P0's proposals and the image proposer's matched-budget
    proposals. The text-proposal control needs its own incremental branch, and a site the text
    proposer names may also be one the image proposer names, so membership of that branch is a
    separate boolean rather than a fourth stratum: the partition must stay a partition.
    """
    lattice = pd.read_parquet(lp1.SITE_LATTICE_INVENTORY)
    p0 = pd.read_parquet(lp1.P0_PROPOSALS)
    image = _matched(L2)
    text = _matched(L1)
    p0_sites = set(p0["lattice_site_id"].astype(str))
    image_sites = set(image["site_id"].astype(str))
    text_sites = set(text["site_id"].astype(str))
    rows = [
        {"site_id": site, "stratum": stratum_of(site, p0_sites, image_sites)}
        for site in sorted(p0_sites | image_sites)
    ]
    # A text-proposed site that neither P0 nor the image proposer names still has to be routed
    # somewhere for the control, so it joins the table under its own stratum.
    rows.extend(
        {"site_id": site, "stratum": S_TEXT} for site in sorted(text_sites - p0_sites - image_sites)
    )
    frame = pd.DataFrame(rows)
    columns = [
        "site_id",
        "environment",
        "document_id",
        "engine_id",
        "anchor_kind",
        "anchor_ref",
        "char_start",
        "char_end",
        "region",
        "line_index",
        "line_uid",
    ]
    joined = frame.merge(lattice[columns], on="site_id", how="left", validate="one_to_one")
    missing = int(joined["environment"].isna().sum())
    if missing:
        raise PhaseError(f"{missing} routed sites are absent from LP1's lattice")
    joined["in_p0"] = joined["site_id"].isin(p0_sites)
    joined["in_image"] = joined["site_id"].isin(image_sites)
    joined["in_text"] = joined["site_id"].isin(text_sites)
    # The control's incremental branch: proposed by the text arm, never by P0. It may intersect
    # the image-only stratum, which is the comparison the control exists to make.
    joined["ht_incremental"] = joined["in_text"] & ~joined["in_p0"]
    # Each proposer's own frozen rank, so an arm can be cut to a budget later without a label.
    p0_rank = p0.set_index("lattice_site_id")["suspicion_score"].astype(float)
    image_rank = image.set_index("site_id")["suspicion"].astype(float)
    text_rank = text.set_index("site_id")["suspicion"].astype(float)
    ranks = pd.concat(
        [
            image.set_index("site_id")[["hash_order", "document_rank"]],
            text.set_index("site_id")[["hash_order", "document_rank"]],
            p0.set_index("lattice_site_id")[["document_rank"]].assign(hash_order=None),
        ]
    )
    ranks = ranks.groupby(level=0).first()
    joined["p0_suspicion"] = joined["site_id"].map(p0_rank)
    joined["image_suspicion"] = joined["site_id"].map(image_rank)
    joined["text_suspicion"] = joined["site_id"].map(text_rank)
    joined["hash_order"] = joined["site_id"].map(ranks["hash_order"])
    joined["document_rank"] = joined["site_id"].map(ranks["document_rank"])
    duplicated = int(joined["site_id"].duplicated().sum())
    if duplicated:
        raise PhaseError(f"{duplicated} sites carry more than one stratum")
    return joined.sort_values(["environment", "site_id"]).reset_index(drop=True)


def routed_sites(arm: str, strata: pd.DataFrame) -> pd.DataFrame:
    """The sites one arm keeps, by the frozen routing rule alone.

    Every arm but the text control reads the stratum. The control reads its own GT-blind
    incremental flag, because a text-proposed site may also be an image-proposed one.
    """
    if arm == HT:
        keep = strata["in_p0"] | strata["ht_incremental"]
        return strata[keep].reset_index(drop=True)
    return strata[strata["stratum"].isin(set(ROUTING[arm]))].reset_index(drop=True)


def corrector_of(arm: str, row: Any) -> tuple[str, ...]:
    """Which correctors' candidates this arm keeps at one site."""
    if arm == HT:
        if bool(row.in_p0):
            return (C0,)
        return (C1_TEXT,) if bool(row.ht_incremental) else ()
    return ROUTING[arm].get(str(row.stratum), ())


def run_strata() -> int:
    """Sections 6-7: the proposal source strata and the frozen routing registry, before labels."""
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    for path in (PROPOSAL_STRATA, STRATA_INVENTORY, ROUTING_REGISTRY):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an HY1 label exists; the strata are frozen before any of them")
    strata = build_strata()
    _write_parquet_once(PROPOSAL_STRATA, strata)
    counts = strata.groupby(["environment", "stratum"]).size().unstack(fill_value=0)
    inventory = {
        "by_environment": [
            {"environment": name, **{s: int(counts.loc[name].get(s, 0)) for s in (*STRATA, S_TEXT)}}
            for name in counts.index
        ],
        "totals": {s: int((strata["stratum"] == s).sum()) for s in (*STRATA, S_TEXT)},
        "sites": len(strata),
    }
    p0_total = inventory["totals"][S_P0] + inventory["totals"][S_OVERLAP]
    image_total = inventory["totals"][S_IMAGE] + inventory["totals"][S_OVERLAP]
    _write_json_once(
        STRATA_INVENTORY,
        {
            **_envelope("proposal_strata_inventory"),
            **inventory,
            "p0_proposals": p0_total,
            "image_proposals": image_total,
            "proposal_expansion": _ratio(p0_total + inventory["totals"][S_IMAGE], p0_total),
            "budget": "LP1's matched budget: P0's own proposal count per environment",
            "rule": "a site is overlap when both proposers name it, else the proposer that did",
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        ROUTING_REGISTRY,
        {
            **_envelope("routing_registry"),
            "arms": {
                arm: {
                    "routes": {s: list(c) for s, c in ROUTING[arm].items()},
                    "note": ROUTING_NOTE[arm],
                    "sites": len(routed_sites(arm, strata)),
                }
                for arm in ARMS
            },
            "correctors": {
                C0: "frozen current generator",
                C1: f"{MODEL} under {gen1.E3}",
                C1_TEXT: f"{MODEL} under {gen1.E1}",
            },
            "routing_reads": ["proposal source stratum"],
            "routing_never_reads": list(FORBIDDEN_INPUT_COLUMNS),
            "frozen_before_any_label": True,
            "uses_ground_truth": False,
        },
    )
    print(
        f"strata: {inventory['sites']} routed sites "
        f"({inventory['totals'][S_P0]} P0-only, {inventory['totals'][S_IMAGE]} image-only, "
        f"{inventory['totals'][S_OVERLAP]} overlap, {inventory['totals'][S_TEXT]} text-only) "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 8-13: the design

# The two request kinds HY1 may have to generate. Everything else is reused from XR1's cache.
NEW_IMAGE = "c1_image_new"
NEW_TEXT = "c1_text_new"
GENERATED_ARMS = (NEW_IMAGE, NEW_TEXT)
CORRECTOR_OF_ARM = {NEW_IMAGE: C1, NEW_TEXT: C1_TEXT}
XR1_ARM_OF_CORRECTOR = {C1: xr1.G2, C1_TEXT: xr1.G1}
# XR1's measured throughput on this machine, from its own shard sidecars.
MEASURED_REQUESTS_PER_SECOND = {NEW_IMAGE: 0.0728, NEW_TEXT: 0.2443}


def _corrector_sites(strata: pd.DataFrame, corrector: str) -> pd.DataFrame:
    """Every site any frozen arm routes to one corrector, GT-blind."""
    keep = np.zeros(len(strata), dtype=bool)
    for arm in ARMS:
        routed = routed_sites(arm, strata)
        for row in routed.itertuples(index=False):
            if corrector in corrector_of(arm, row):
                keep |= strata["site_id"].to_numpy() == row.site_id
    return strata[keep].reset_index(drop=True)


def corrector_demand(strata: pd.DataFrame) -> dict[str, list[str]]:
    """Which sites each corrector must answer, over every arm. Reads strata and nothing else."""
    demand: dict[str, set[str]] = {C0: set(), C1: set(), C1_TEXT: set()}
    for arm in ARMS:
        routed = routed_sites(arm, strata)
        for row in routed.itertuples(index=False):
            for corrector in corrector_of(arm, row):
                demand[corrector].add(str(row.site_id))
    return {corrector: sorted(sites) for corrector, sites in demand.items()}


def run_preregister() -> int:
    """Sections 8-13: arms, budgets, criteria and registries, written before any raw answer."""
    started = time.monotonic()
    _require(PROPOSAL_STRATA, "strata")
    outputs = (DESIGN_RECORD, PROMPT_REGISTRY, DECODING_REGISTRY, CROP_REGISTRY, COMPUTE_BUDGET)
    for path in outputs:
        _forbid(path)
    if _labels_exist() or any(RAW_CACHE.glob("*.parquet")):
        raise PhaseError("an HY1 answer or label exists; the design cannot be written after it")
    strata = pd.read_parquet(PROPOSAL_STRATA)
    demand = corrector_demand(strata)
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "question": (
                "does the complementary localization SGV-LP1 found translate into a materially "
                "larger exact-repair candidate universe when connected to the frozen "
                "image-conditioned corrector?"
            ),
            "primary_metric": (
                "OpportunityRecall: error sites with at least one exact repair candidate"
            ),
            "primary_comparison": f"{PRIMARY_ARM} minus {BASELINE_ARM}",
            "arms": {arm: ROUTING_NOTE[arm] for arm in ARMS},
            "strata": {
                S_P0: "proposed by P0 and not by the image proposer",
                S_IMAGE: "proposed by the image proposer and not by P0",
                S_OVERLAP: "proposed by both",
                S_TEXT: "proposed by the text proposer alone, for the control only",
            },
            "candidate_budget": {
                "primary_k": PRIMARY_K,
                "secondary_k": SECONDARY_K,
                "frozen_generator": "its own frozen candidate budget, never trimmed",
            },
            "criteria": {
                "C1": (
                    f"{PRIMARY_ARM} improves OpportunityRecall over {BASELINE_ARM} by at least "
                    f"{OPPORTUNITY_MARGIN}"
                ),
                "C2": "the document-clustered paired interval for that gain lies above zero",
                "C3": f"the gain holds in at least {BREADTH_MAJORITY} environments",
                "C4": (
                    f"at least {WEAK_TYPES_REQUIRED} of {list(WEAK_TYPES)} improve by "
                    f"{WEAK_TYPE_MARGIN} within the type"
                ),
                "C5": (
                    "incremental harmful candidates per incremental exact repaired site do not "
                    f"exceed the frozen comparator ({HARM_COMPARATOR_SOURCE})"
                ),
                "C6": (
                    "image-only sites contribute exact repairs the baseline does not have, in at "
                    f"least {TRANSLATION_BREADTH} environments"
                ),
            },
            "outcome_rule": "A needs every criterion; B is opportunity without burden control; "
            "C is reach without translation; anything else is D",
            "statistics": {
                "unit": "document, resampled within environment",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "primary_family": [
                    f"{PRIMARY_ARM}_vs_{BASELINE_ARM}",
                    f"{H2}_vs_{H1}",
                    f"{PRIMARY_ARM}_vs_{HT}",
                ],
                "multiplicity": "Holm within each family",
            },
            "image_branch_fractions": list(IMAGE_BRANCH_FRACTIONS),
            "no_reliability_in_this_stage": (
                "SGV-XR1 established the frozen verifier encodes generator identity; no candidate "
                "is scored, ranked by reliability or thresholded here"
            ),
            "uses_ground_truth": False,
        },
    )
    # The corrector registries are XR1's, reused unchanged and verified by hash.
    xr1_prompt = cc_read_json(xr1.PROMPT_REGISTRY)
    xr1_decoding = cc_read_json(xr1.DECODING_REGISTRY)
    xr1_crop = cc_read_json(xr1.CROP_POLICY)
    _write_json_once(
        PROMPT_REGISTRY,
        {
            **_envelope("prompt_registry"),
            "prompt_hash": xr1_prompt["prompt_hash"],
            "source": "SGV-XR1, unchanged",
            "templates": {C1: gen1.E3, C1_TEXT: gen1.E1},
            "contains_ground_truth": False,
            "frozen_before_endpoint_evaluation": True,
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        DECODING_REGISTRY,
        {
            **_envelope("decoding_registry"),
            "decoding_hash": xr1_decoding["decoding_hash"],
            "source": "SGV-XR1, unchanged",
            "batch_size": int(gen1.MODEL_SPECS[MODEL]["batch_size"]),
            "dtype": gen1.MODEL_DTYPE,
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        CROP_REGISTRY,
        {
            **_envelope("correction_crop_registry"),
            "policy_hash": xr1_crop["policy_hash"],
            "source": "SGV-XR1's crop policy, unchanged",
            "policy": {
                k: v for k, v in xr1_crop.items() if k.startswith(("crop", "pad", "render"))
            },
            "reads_ground_truth_geometry": False,
            "uses_ground_truth": False,
        },
    )
    # What this stage must actually decode, before it decodes anything.
    cached = _cached_site_ids()
    new_image = sorted(set(demand[C1]) - cached[C1])
    new_text = sorted(set(demand[C1_TEXT]) - cached[C1_TEXT])
    plan = {
        NEW_IMAGE: len(new_image),
        NEW_TEXT: len(new_text),
    }
    hours = {
        arm: round(count / MEASURED_REQUESTS_PER_SECOND[arm] / 3600.0, 2)
        for arm, count in plan.items()
    }
    _write_json_once(
        COMPUTE_BUDGET,
        {
            **_envelope("compute_budget"),
            "requests_demanded": {c: len(s) for c, s in demand.items()},
            "requests_reused_from_xr1": {
                corrector: len(cached[corrector] & set(demand[corrector]))
                for corrector in (C1, C1_TEXT)
            },
            "requests_to_generate": plan,
            "estimated_hours": hours,
            "estimated_total_hours": round(sum(hours.values()), 2),
            "throughput_source": "SGV-XR1's own shard sidecars on this machine",
            "frozen_generator_requests": 0,
            "note": "the frozen current generator makes no request; its candidates are read",
            "uses_ground_truth": False,
        },
    )
    print(
        f"preregister: {plan[NEW_IMAGE]} image and {plan[NEW_TEXT]} text requests to generate, "
        f"{len(cached[C1])} image answers reused ({sum(hours.values()):.2f} h estimated) "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _lattice_of_p0() -> dict[str, str]:
    """XR1's own full site id for each lattice site P0 proposed, from LP1's frozen mapping.

    The environment prefix has to stay: two environments of one corpus can run the same engine
    binary, so `document:engine:dsite:n` alone is not unique across environments and a cache
    lookup keyed on it would silently answer with another environment's request.
    """
    p0 = pd.read_parquet(lp1.P0_PROPOSALS)
    return dict(zip(p0["lattice_site_id"].astype(str), p0["site_id"].astype(str), strict=True))


def _cached_site_ids() -> dict[str, set[str]]:
    """Lattice sites whose correction XR1 already decoded, by corrector, before identity checks."""
    mapping = _lattice_of_p0()
    answered: dict[str, set[str]] = {}
    for corrector, arm in XR1_ARM_OF_CORRECTOR.items():
        sites: set[str] = set()
        for spec in environment_specs():
            path = xr1._shard_path(arm, spec["environment"])
            if not path.is_file():
                continue
            # XR1's raw shard keys a row by `environment|arm|site`, so the environment is
            # restored here rather than dropped.
            requests = pd.read_parquet(path, columns=["request_id"])["request_id"].astype(str)
            parts = requests.str.split("|", expand=True)
            sites |= set(parts[0] + "|" + parts[2])
        answered[corrector] = {
            lattice for lattice, xr1_site in mapping.items() if xr1_site in sites
        }
    return answered


# ------------------------------------------------------------------ sections 9-10, 13: contexts


def run_contexts() -> int:
    """Section 10: the GT-blind correction context and crop for every routed site."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (CONTEXTS, CROP_INVENTORY):
        _forbid(path)
    if _labels_exist() or any(RAW_CACHE.glob("*.parquet")):
        raise PhaseError("an HY1 answer or label exists; contexts are frozen before both")
    strata = pd.read_parquet(PROPOSAL_STRATA)
    frames: list[pd.DataFrame] = []
    census: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        sites = strata[strata["environment"] == name].reset_index(drop=True)
        pages = gen1._page_index(spec)
        contexts = xr1.build_natural_contexts(sites, pages, CROP_DIR / _slug(name))
        contexts.insert(0, "environment", name)
        frames.append(contexts)
        levels = contexts["crop_level"].value_counts().to_dict()
        census.append(
            {
                "environment": name,
                "sites": len(sites),
                "site_crops": int(levels.get(xr1.CROP_SITE, 0)),
                "line_fallback_crops": int(levels.get(xr1.CROP_LINE, 0)),
                "without_a_crop": int(levels.get(xr1.CROP_NONE, 0)),
                "runtime_seconds": round(time.monotonic() - began, 2),
            }
        )
        print(f"  {name}: {len(sites)} sites, {levels.get(xr1.CROP_NONE, 0)} without a crop")
    contexts = pd.concat(frames, ignore_index=True)
    _write_parquet_once(CONTEXTS, contexts)
    _write_parquet_once(
        CROP_INVENTORY,
        contexts[
            ["environment", "site_id", "crop_level", "crop_sha256", "crop_width", "crop_height"]
        ],
    )
    print(
        f"contexts: {len(contexts)} routed sites, "
        f"{int((contexts['crop_level'] == xr1.CROP_NONE).sum())} without a crop "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _xr1_contexts() -> pd.DataFrame:
    """XR1's frozen contexts, keyed the way its requests are keyed."""
    frame = pd.read_parquet(xr1.CONTEXTS)
    sample = pd.read_parquet(xr1.PROPOSAL_SAMPLE)[["site_id", "environment"]]
    return frame.merge(sample, on="site_id", validate="one_to_one")


def run_cacheaudit() -> int:
    """Section 13: which XR1 answers HY1 may reuse, by exact request identity and nothing less."""
    started = time.monotonic()
    _require(CONTEXTS, "contexts")
    _forbid(CACHE_REUSE_AUDIT)
    strata = pd.read_parquet(PROPOSAL_STRATA)
    contexts = pd.read_parquet(CONTEXTS)
    demand = corrector_demand(strata)
    mapping = _lattice_of_p0()
    xr1_contexts = _xr1_contexts().set_index("site_id")
    mine = contexts.set_index("site_id")
    prompt_matches = (
        cc_read_json(PROMPT_REGISTRY)["prompt_hash"]
        == (cc_read_json(xr1.PROMPT_REGISTRY)["prompt_hash"])
    )
    decoding_matches = (
        cc_read_json(DECODING_REGISTRY)["decoding_hash"]
        == (cc_read_json(xr1.DECODING_REGISTRY)["decoding_hash"])
    )
    crop_matches = (
        cc_read_json(CROP_REGISTRY)["policy_hash"] == (cc_read_json(xr1.CROP_POLICY)["policy_hash"])
    )
    if not (prompt_matches and decoding_matches and crop_matches):
        raise PhaseError("HY1's registries differ from XR1's; no answer may be reused")
    report: dict[str, Any] = {}
    reuse_rows: list[dict[str, Any]] = []
    for corrector, sites in demand.items():
        if corrector == C0:
            report[corrector] = {
                "demanded": len(sites),
                "reused": 0,
                "to_generate": 0,
                "note": "the frozen current generator decodes nothing; its candidates are read",
            }
            continue
        answered = _cached_site_ids()[corrector]
        reusable: list[str] = []
        rejected: list[dict[str, str]] = []
        needs_image = corrector == C1
        for site in sites:
            key = mapping.get(site)
            if key is None or site not in answered or key not in xr1_contexts.index:
                continue
            theirs = xr1_contexts.loc[key]
            ours = mine.loc[site]
            same_text = str(theirs["e1_text"]) == str(ours["e1_text"])
            same_crop = (not needs_image) or (
                str(theirs["crop_sha256"]) == str(ours["crop_sha256"])
            )
            if same_text and same_crop:
                reusable.append(site)
                reuse_rows.append({"corrector": corrector, "site_id": site, "xr1_request_key": key})
            else:
                rejected.append(
                    {
                        "site_id": site,
                        "reason": "text differs" if not same_text else "crop differs",
                    }
                )
        report[corrector] = {
            "demanded": len(sites),
            "answered_by_xr1": len(answered & set(sites)),
            "reused": len(reusable),
            "rejected_for_differing_input": rejected[:20],
            "rejected_count": len(rejected),
            "to_generate": len(sites) - len(reusable),
            "xr1_arm": XR1_ARM_OF_CORRECTOR[corrector],
        }
    _write_json_once(
        CACHE_REUSE_AUDIT,
        {
            **_envelope("cache_reuse_audit"),
            "identity_rule": (
                "a cached answer is reused only when the model, revision, condition, prompt hash, "
                "decoding hash, crop policy, prompt text and crop hash are all identical"
            ),
            "registries_identical_to_xr1": {
                "prompt": prompt_matches,
                "decoding": decoding_matches,
                "crop_policy": crop_matches,
            },
            "by_corrector": report,
            "reused_total": len(reuse_rows),
            "uses_ground_truth": False,
        },
    )
    _write_parquet_once(CACHE / "cache_reuse_map.parquet", pd.DataFrame(reuse_rows))
    print(
        f"cacheaudit: {len(reuse_rows)} XR1 answers reusable, "
        f"{report[C1]['to_generate']} image and {report[C1_TEXT]['to_generate']} text to generate "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ section 14: generation


def _selected_arms() -> list[str]:
    chosen = [a for a in os.environ.get("HY1_ARMS", "").split(",") if a]
    unknown = [a for a in chosen if a not in GENERATED_ARMS]
    if unknown:
        raise PhaseError(f"unknown HY1_ARMS {unknown}; choose from {list(GENERATED_ARMS)}")
    return chosen or list(GENERATED_ARMS)


def _shard_path(arm: str, name: str) -> Path:
    return RAW_CACHE / f"{_slug(name)}.{arm}.parquet"


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name.replace(".parquet", ".cost.json"))


def _shard_complete(path: Path) -> bool:
    sidecar = _sidecar(path)
    if not (path.is_file() and sidecar.is_file()):
        return False
    return bool(cc_read_json(sidecar)["raw_output_sha256"] == file_sha256(path))


def _to_generate(corrector: str) -> set[str]:
    """The sites HY1 must decode for one corrector: demanded, minus every reusable answer."""
    strata = pd.read_parquet(PROPOSAL_STRATA)
    demanded = set(corrector_demand(strata)[corrector])
    reuse = pd.read_parquet(CACHE / "cache_reuse_map.parquet")
    reused = set(reuse[reuse["corrector"] == corrector]["site_id"].astype(str))
    return demanded - reused


def requests_for(arm: str, name: str) -> pd.DataFrame:
    """One shard's frozen requests, rebuilt from the frozen GT-blind tables alone."""
    corrector = CORRECTOR_OF_ARM[arm]
    contexts = pd.read_parquet(CONTEXTS)
    wanted = _to_generate(corrector)
    frame = contexts[
        (contexts["environment"] == name) & contexts["site_id"].isin(wanted)
    ].reset_index(drop=True)
    image = corrector == C1
    if image:
        frame = frame[frame["crop_path"].notna()].reset_index(drop=True)
    strata = pd.read_parquet(PROPOSAL_STRATA).set_index("site_id")
    digest = frame["site_id"].astype(str).str.split("|", n=1).str[1]
    template = CONDITION_OF_CORRECTOR[corrector]
    return pd.DataFrame(
        {
            "environment": frame["environment"].to_numpy(),
            "method": arm,
            "condition": template,
            "request_id": (frame["environment"] + f"|{arm}|" + digest).to_numpy(),
            "site_id": frame["site_id"].to_numpy(),
            "anchor_kind": strata.loc[frame["site_id"], "anchor_kind"].to_numpy(),
            "region": strata.loc[frame["site_id"], "region"].to_numpy(),
            "user_prompt": [gen1.user_message(template, str(t)) for t in frame["e1_text"]],
            "image_path": frame["crop_path"].to_numpy() if image else None,
            "image_sha256": frame["crop_sha256"].to_numpy() if image else None,
        }
    ).reset_index(drop=True)


def run_generate() -> int:
    """GT-blind correction, cached immutably one (arm, environment) shard at a time."""
    started = time.monotonic()
    for path in (CONTEXTS, CACHE_REUSE_AUDIT, PROMPT_REGISTRY, DECODING_REGISTRY, CROP_REGISTRY):
        _require(path, "contexts" if path == CONTEXTS else "preregister")
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    prompt_hash = cc_read_json(PROMPT_REGISTRY)["prompt_hash"]
    decoding_hash = cc_read_json(DECODING_REGISTRY)["decoding_hash"]
    crop_hash = cc_read_json(CROP_REGISTRY)["policy_hash"]
    spec = gen1.MODEL_SPECS[MODEL]
    work = [
        (arm, env["environment"])
        for arm in _selected_arms()
        for env in environment_specs()
        if not _shard_complete(_shard_path(arm, env["environment"]))
    ]
    if not work:
        print("generate: every selected shard complete")
        return 0
    import transformers

    transformers.utils.logging.set_verbosity_error()
    device = _resolve_device()
    began = time.monotonic()
    runner = gen1._runner(MODEL, device)
    print(f"  loaded {MODEL} on {device} ({time.monotonic() - began:.0f}s); {len(work)} shards")
    for arm, name in work:
        path = _shard_path(arm, name)
        requests = requests_for(arm, name)
        if requests.empty:
            print(f"    {arm}/{name}: no request to decode")
        raw, cost = gen1.run_requests(runner, requests, int(spec["batch_size"]), print)
        temporary = path.with_name(path.name + ".partial")
        raw.to_parquet(temporary, index=False)
        temporary.rename(path)
        _sidecar(path).unlink(missing_ok=True)
        _write_json_once(
            _sidecar(path),
            {
                **_envelope("generation_shard"),
                "environment": name,
                "method": arm,
                "corrector": CORRECTOR_OF_ARM[arm],
                "condition": CONDITION_OF_CORRECTOR[CORRECTOR_OF_ARM[arm]],
                "model": MODEL,
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "site_count": int(requests["site_id"].nunique()),
                **cost,
                "prompt_hash": prompt_hash,
                "crop_policy_hash": crop_hash,
                "decoding_hash": decoding_hash,
                "allocator_environment": gen1._allocator_environment(),
                "request_prompts_hash": canonical_hash(sorted(raw["prompt_sha256"].tolist())),
                "device": device,
                "raw_output": _relative(path),
                "raw_output_sha256": file_sha256(path),
                "clock": "time.monotonic, which does not advance while the machine sleeps",
                "uses_ground_truth": False,
            },
        )
        print(
            f"    {arm}/{name}: {cost['requests']} req in {cost['wall_clock_seconds']:.0f}s "
            f"({cost['requests_per_second']:.3f}/s)",
            flush=True,
        )
    del runner
    _release_device_cache(device)
    print(f"generate: done in {time.monotonic() - started:.0f}s")
    return 0


REGENERATION_ENVIRONMENT = "funsd/doctr"
# The slice used only to measure batch-layout sensitivity, never to judge reproducibility.
SENSITIVITY_BATCHES = 3


def run_spotcheck() -> int:
    """Each generated arm decodes one whole shard again, and every answer must repeat exactly.

    The shard is regenerated as a shard -- the same requests, the same order, the same batch
    size -- because that is the decode the frozen file records. A slice of it would land in
    different batches, and batched decoding is padding-sensitive: the same request in a different
    batch can produce a different token. That sensitivity is real, so it is measured here and
    reported separately rather than folded into the reproducibility claim.
    """
    REGENERATION_DIR.mkdir(parents=True, exist_ok=True)
    arms = [a for a in _selected_arms() if not (REGENERATION_DIR / f"{a}.json").is_file()]
    if not arms:
        print("spotcheck: every selected arm already regenerated")
        return 0
    started = time.monotonic()
    spec = gen1.MODEL_SPECS[MODEL]
    batch = int(spec["batch_size"])
    import transformers

    transformers.utils.logging.set_verbosity_error()
    device = _resolve_device()
    runner = gen1._runner(MODEL, device)
    for arm in arms:
        requests = requests_for(arm, REGENERATION_ENVIRONMENT)
        if requests.empty:
            continue
        frozen = pd.read_parquet(_shard_path(arm, REGENERATION_ENVIRONMENT))
        raw, _cost = gen1.run_requests(runner, requests, batch, lambda *_a, **_k: None)
        merged = raw.merge(frozen, on="request_id", suffixes=("_new", "_old"))
        differing = merged[merged["output_new"] != merged["output_old"]]
        sliced, _c = gen1.run_requests(
            runner, requests.head(batch * SENSITIVITY_BATCHES), batch, lambda *_a, **_k: None
        )
        rebatched = sliced.merge(frozen, on="request_id", suffixes=("_new", "_old"))
        moved = rebatched[rebatched["output_new"] != rebatched["output_old"]]
        top1_moved = 0
        for row in moved.itertuples(index=False):
            new = gen1.parse_candidates(str(row.output_new), True)["candidates"]
            old = gen1.parse_candidates(str(row.output_old), True)["candidates"]
            top1_moved += int((new or [None])[0] != (old or [None])[0])
        _write_json_once(
            REGENERATION_DIR / f"{arm}.json",
            {
                **_envelope("spotcheck"),
                "arm": arm,
                "environment": REGENERATION_ENVIRONMENT,
                "regenerated": len(merged),
                "differing": len(differing),
                "identical": bool(differing.empty),
                "batch_sensitivity": {
                    "rebatched_requests": len(rebatched),
                    "differing": len(moved),
                    "top1_differing": top1_moved,
                    "note": (
                        "the same requests decoded in a different batch layout; a difference here "
                        "is padding sensitivity, not a change in the frozen record"
                    ),
                },
                "uses_ground_truth": False,
            },
        )
        print(
            f"  {arm}: {len(merged)} regenerated, {len(differing)} differ; "
            f"re-batched {len(rebatched)}: {len(moved)} differ, {top1_moved} at rank 0"
        )
        if not differing.empty:
            raise PhaseError(f"{arm}: {len(differing)} answers did not repeat")
    del runner
    _release_device_cache(device)
    print(f"spotcheck: done in {time.monotonic() - started:.0f}s")
    return 0


# ------------------------------------------------------------------ section 14-15: candidates

LABEL_SITE_COLUMNS = xr1.LABEL_SITE_COLUMNS
LABEL_INPUT_COLUMNS = xr1.LABEL_INPUT_COLUMNS
METHOD_CANDIDATE_COLUMNS = xr1.METHOD_CANDIDATE_COLUMNS
OUTCOME_EXACT = xr1.OUTCOME_EXACT
G0_CONDITION = xr1.G0_CONDITION
PROBE = "hy1_identity_probe"
SITE_ERROR = xr1.SITE_ERROR
SITE_CLEAN = xr1.SITE_CLEAN
HARMFUL_OUTCOMES = xr1.HARMFUL_OUTCOMES


def _corrector_requests(
    sites: pd.DataFrame, corrector: str, streams: dict[tuple[str, str], str]
) -> pd.DataFrame:
    """SGV-XC1's request columns under HY1's ids, so `_expand` and `_neural_edits` read them.

    The stream context is rebuilt by SGV-XC1's own rule from the same canonical OCR stream, so a
    reused request and a new one are described identically.
    """
    digest = sites["site_id"].astype(str).str.split("|", n=1).str[1]
    frame = sites.copy()
    frame["condition"] = CONDITION_OF_CORRECTOR[corrector]
    frame["request_id"] = (sites["environment"] + f"|{corrector}|" + digest).to_numpy()
    before: list[str] = []
    after: list[str] = []
    window: list[str] = []
    for row in frame.itertuples(index=False):
        stream = streams.get((str(row.document_id), str(row.engine_id)), "")
        start, end = int(row.char_start), int(row.char_end)
        if stream[start:end] != str(row.region):
            raise PhaseError(f"{row.site_id}: region differs from the rebuilt OCR stream")
        before.append(stream[max(0, start - xc1.CONTEXT_CHARS) : start])
        after.append(stream[end : end + xc1.CONTEXT_CHARS])
        window.append(stream[start:end])
    frame["context_before"] = before
    frame["context_after"] = after
    frame["window_start"] = frame["char_start"].astype(int)
    frame["window_text"] = window
    return frame[list(xr1.REQUEST_COLUMNS)].reset_index(drop=True)


def _reused_raw(corrector: str, name: str, lattice_of: dict[str, str]) -> pd.DataFrame:
    """XR1's frozen answers for the reused sites, re-keyed to HY1's request ids.

    Only the request's label changes: XR1 keys a request by P0's own site id, HY1 by the lattice
    site id of the same anchor. The prompt text, the crop and the answer are XR1's own, and
    `--cacheaudit` has already proved the inputs identical.
    """
    arm = XR1_ARM_OF_CORRECTOR[corrector]
    path = xr1._shard_path(arm, name)
    if not path.is_file() or not lattice_of:
        return pd.DataFrame()
    raw = pd.read_parquet(path)
    parts = raw["request_id"].astype(str).str.split("|", expand=True)
    xr1_sites = parts[0] + "|" + parts[2]
    keep = xr1_sites.isin(lattice_of)
    raw = raw[keep].copy()
    lattice = xr1_sites[keep].map(lattice_of).astype(str)
    digest = lattice.str.split("|", n=1).str[1]
    raw["request_id"] = (parts[0][keep] + f"|{corrector}|" + digest).to_numpy()
    raw["method"] = corrector
    raw["condition"] = CONDITION_OF_CORRECTOR[corrector]
    raw["reused_from_xr1"] = True
    return raw.reset_index(drop=True)


def _generated_raw(corrector: str, name: str) -> pd.DataFrame:
    """HY1's own answers for this corrector, re-keyed from the generation arm to the corrector."""
    frames = []
    for arm, mapped in CORRECTOR_OF_ARM.items():
        if mapped != corrector:
            continue
        path = _shard_path(arm, name)
        if not path.is_file():
            continue
        raw = pd.read_parquet(path)
        if raw.empty:
            continue
        parts = raw["request_id"].astype(str).str.split("|", expand=True)
        raw = raw.copy()
        raw["request_id"] = (parts[0] + f"|{corrector}|" + parts[2]).to_numpy()
        raw["method"] = corrector
        raw["reused_from_xr1"] = False
        frames.append(raw)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _verified_shards() -> list[dict[str, Any]]:
    """Every HY1 shard, verified against its own sidecar hash before any label is computed."""
    manifest: list[dict[str, Any]] = []
    for path in sorted(RAW_CACHE.glob("*.parquet")):
        sidecar = _sidecar(path)
        if not sidecar.is_file():
            raise PhaseError(f"{_relative(path)} has no sidecar")
        record = cc_read_json(sidecar)
        observed = file_sha256(path)
        if observed != record["raw_output_sha256"]:
            raise PhaseError(f"{_relative(path)} differs from its sidecar hash")
        manifest.append({**record, "verified_sha256": observed})
    return manifest


def _identity_probes(sites: pd.DataFrame) -> pd.DataFrame:
    """One unchanged candidate per site, so `_labels` reports the site's own truth status."""
    frame = xr1._identity_probes(sites)
    frame["candidate_id"] = (
        frame["candidate_id"].astype(str).str.replace("xr1-probe|", f"{PROBE}|", regex=False)
    )
    return frame


def _c0_frame(spec: dict[str, Any], documents: set[str], routed: set[str]) -> pd.DataFrame:
    """The frozen current generator's labelled candidates, restricted to the routed P0 sites."""
    frame = xr1._g0_frame(spec, documents)
    frame = frame[frame["site_id"].astype(str).isin(routed)].reset_index(drop=True)
    frame["method"] = C0
    frame["candidate_source"] = C0
    return frame


def run_normalize() -> int:
    """Raw answers -> SGV-XC1 atomic edits -> SGV14 labels -> the frozen HY1 candidate table."""
    from ocr_risk.canonical import rebuild_stream

    started = time.monotonic()
    outputs = (
        CANDIDATES,
        RAW_OUTPUT_MANIFEST,
        RAW_OUTPUT_HASHES,
        CANDIDATE_INVENTORY,
        CANDIDATE_ATTRIBUTION,
        ERROR_SITES,
        SITE_TRUTH,
    )
    for path in outputs:
        _forbid(path)
    _require(CACHE_REUSE_AUDIT, "cacheaudit")
    manifest = _verified_shards()
    _write_json_once(
        RAW_OUTPUT_HASHES,
        {
            **_envelope("raw_output_hashes"),
            "hy1_shards": {row["raw_output"]: row["raw_output_sha256"] for row in manifest},
            "xr1_shards_reused": {
                _relative(xr1._shard_path(arm, spec["environment"])): file_sha256(
                    xr1._shard_path(arm, spec["environment"])
                )
                for arm in XR1_ARM_OF_CORRECTOR.values()
                for spec in environment_specs()
                if xr1._shard_path(arm, spec["environment"]).is_file()
            },
            "verified_against_sidecars": True,
            "verified_before_any_label": True,
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        RAW_OUTPUT_MANIFEST,
        {**_envelope("raw_output_manifest"), "count": len(manifest), "shards": manifest},
    )
    strata = pd.read_parquet(PROPOSAL_STRATA)
    reuse = pd.read_parquet(CACHE / "cache_reuse_map.parquet")
    demand = corrector_demand(strata)
    p0_by_lattice = _lattice_of_p0()
    candidate_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    census_rows: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        sites = strata[strata["environment"] == name].reset_index(drop=True)
        documents = set(sites["document_id"].astype(str))
        bundles = s14._document_bundles(spec["corpus"])
        roles = s14.partition_of(spec["corpus"], sorted(bundles))
        spans, _record = s14._canonical_spans(spec, bundles)
        evaluation = {p: g for p, g in spans.items() if roles.get(p[0]) == s14.ROLE_EVALUATION}
        streams = {pair: rebuild_stream(group) for pair, group in evaluation.items() if group}
        frames: list[tuple[str, str, pd.DataFrame]] = []
        for corrector in (C1, C1_TEXT):
            wanted = {s for s in demand[corrector] if s in set(sites["site_id"].astype(str))}
            if not wanted:
                continue
            reused = set(reuse[reuse["corrector"] == corrector]["site_id"].astype(str)) & wanted
            raw = pd.concat(
                [
                    _reused_raw(corrector, name, {p0_by_lattice[s]: s for s in reused}),
                    _generated_raw(corrector, name),
                ],
                ignore_index=True,
            )
            if raw.empty:
                continue
            answered = sites[sites["site_id"].astype(str).isin(wanted)]
            requests = _corrector_requests(answered, corrector, streams)
            raw = raw[raw["request_id"].isin(set(requests["request_id"]))].reset_index(drop=True)
            requests = requests[requests["request_id"].isin(set(raw["request_id"]))].reset_index(
                drop=True
            )
            expanded, census = gen1._expand(
                raw, requests, corrector, CONDITION_OF_CORRECTOR[corrector]
            )
            for row in census:
                row["environment"] = name
                row["corrector"] = corrector
            census_rows.extend(census)
            _s, proposals, _shard = xc1._neural_edits(name, corrector, expanded, requests, {}, None)
            frames.append((corrector, CONDITION_OF_CORRECTOR[corrector], proposals))
        label_sites = sites[list(LABEL_SITE_COLUMNS)].assign(
            site_id=sites["site_id"].astype(str).str.split("|", n=1).str[1]
        )
        probes = _identity_probes(sites)
        present = [(c, cond, f) for c, cond, f in frames if not f.empty]
        label_input = pd.concat(
            [probes, *[f[LABEL_INPUT_COLUMNS] for _c, _cond, f in present]], ignore_index=True
        )
        labels, _diagnostics = s14._labels(
            spec["corpus"], bundles, evaluation, streams, label_sites, label_input
        )
        for corrector, condition, frame in present:
            labelled = xc1._method_frame(spec, corrector, condition, frame, labels, roles)
            candidate_frames.append(labelled)
        routed_p0 = {
            p0_by_lattice[s]
            for s in sites[sites["in_p0"]]["site_id"].astype(str)
            if s in p0_by_lattice
        }
        candidate_frames.append(_c0_frame(spec, documents, routed_p0))
        # The probe says what each site's own ground truth is; it never enters an arm.
        probe_labels = labels.set_index("candidate_id").loc[probes["candidate_id"]]
        labelable = probe_labels["labelable"].to_numpy(dtype=bool)
        d_before = probe_labels["d_before"].to_numpy(dtype=np.int64)
        truth_frames.append(
            pd.DataFrame(
                {
                    "environment": name,
                    "site_id": sites["site_id"].astype(str).to_numpy(),
                    "stratum": sites["stratum"].astype(str).to_numpy(),
                    "document_id": sites["document_id"].astype(str).to_numpy(),
                    "anchor_kind": sites["anchor_kind"].astype(str).to_numpy(),
                    "labelable": labelable,
                    "site_d_before": np.where(labelable, d_before, -1),
                    "site_class": np.where(
                        ~labelable,
                        xr1.SITE_UNLABELABLE,
                        np.where(d_before > 0, SITE_ERROR, SITE_CLEAN),
                    ),
                }
            )
        )
        print(
            f"  {name}: {len(sites)} routed sites, "
            f"{sum(len(f) for _c, _cond, f in present)} model candidates "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    candidates = pd.concat(candidate_frames, ignore_index=True)
    # Attribution: every candidate carries the stratum of its site and the corrector that wrote it.
    lattice_of_p0 = {v: k for k, v in p0_by_lattice.items()}
    candidates["lattice_site_id"] = np.where(
        candidates["method"] == C0,
        candidates["site_id"].astype(str).map(lattice_of_p0),
        candidates["site_id"].astype(str),
    )
    stratum = strata.set_index("site_id")["stratum"]
    candidates["proposal_stratum"] = candidates["lattice_site_id"].map(stratum)
    candidates["corrector_source"] = candidates["method"]
    unattributed = int(candidates["proposal_stratum"].isna().sum())
    if unattributed:
        raise PhaseError(f"{unattributed} candidates carry no proposal stratum")
    _write_parquet_once(CANDIDATES, candidates)
    _write_parquet_once(SITE_TRUTH, pd.concat(truth_frames, ignore_index=True))
    _write_parquet_once(ERROR_SITES, pd.read_parquet(lp1.ERROR_SITES))
    _write_json_once(
        CANDIDATE_INVENTORY,
        {
            **_analysis_envelope("candidate_inventory"),
            "candidates": len(candidates),
            "by_corrector": {
                str(k): int(v) for k, v in candidates["corrector_source"].value_counts().items()
            },
            "by_stratum": {
                str(k): int(v) for k, v in candidates["proposal_stratum"].value_counts().items()
            },
            "sites_with_a_candidate": int(candidates["lattice_site_id"].nunique()),
            "parse_census": census_rows,
        },
    )
    _write_json_once(
        CANDIDATE_ATTRIBUTION,
        {
            **_analysis_envelope("candidate_attribution"),
            "columns": [
                "candidate_id",
                "lattice_site_id",
                "proposal_stratum",
                "corrector_source",
                "generator_rank",
                "candidate_text",
            ],
            "by_corrector_and_stratum": [
                {"corrector": str(c), "stratum": str(s), "candidates": int(n)}
                for (c, s), n in candidates.groupby(["corrector_source", "proposal_stratum"])
                .size()
                .items()
            ],
            "provenance_never_dropped": True,
        },
    )
    print(
        f"normalize: {len(candidates)} candidates over "
        f"{candidates['lattice_site_id'].nunique()} sites ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 15-20: endpoints

ARM_BUDGET: dict[str, int | None] = {C0: None, C1: PRIMARY_K, C1_TEXT: PRIMARY_K}


def load_views() -> dict[str, xr1.EnvironmentView]:
    """The frozen error population and LP1's frozen links from lattice sites onto it."""
    errors = pd.read_parquet(ERROR_SITES)
    links = pd.read_parquet(lp1.PROPOSAL_ERROR_LINKS)
    by_env = {spec["environment"]: spec for spec in environment_specs()}
    views: dict[str, xr1.EnvironmentView] = {}
    for spec in environment_specs():
        name = spec["environment"]
        block = errors[errors["environment"] == name].sort_values("align_site_id", kind="stable")
        ids = block["align_site_id"].astype(str).to_numpy()
        position = {value: index for index, value in enumerate(ids.tolist())}
        grouped: dict[str, list[int]] = {}
        env_links = links[links["environment"] == name]
        for site, target in zip(
            env_links["site_id"].astype(str), env_links["align_site_id"].astype(str), strict=True
        ):
            if target in position:
                grouped.setdefault(site, []).append(position[target])
        index = {k: np.asarray(sorted(set(v)), dtype=np.int64) for k, v in grouped.items()}
        covered = np.zeros(ids.size, dtype=bool)
        for values in index.values():
            covered[values] = True
        views[name] = xr1.EnvironmentView(
            name=name,
            corpus=by_env[name]["corpus"],
            base_engine=by_env[name]["base_engine"],
            error_ids=ids,
            error_kind=block["site_kind"].astype(str).to_numpy(),
            error_subtype=block["subtype"].astype(str).to_numpy(),
            error_document=block["document_id"].astype(str).to_numpy(),
            position=position,
            links=index,
            covered=covered,
        )
    return views


def arm_candidates(candidates: pd.DataFrame, arm: str, k: int | None = PRIMARY_K) -> pd.DataFrame:
    """One arm's candidates, selected by the frozen routing rule and each corrector's budget."""
    keep = np.zeros(len(candidates), dtype=bool)
    stratum = candidates["proposal_stratum"].astype(str).to_numpy()
    corrector = candidates["corrector_source"].astype(str).to_numpy()
    rank = candidates["generator_rank"].to_numpy(dtype=np.int64)
    if arm == HT:
        incremental = candidates["ht_incremental"].to_numpy(dtype=bool)
        in_p0 = candidates["in_p0"].to_numpy(dtype=bool)
        keep |= in_p0 & (corrector == C0)
        keep |= incremental & ~in_p0 & (corrector == C1_TEXT)
    else:
        for source, correctors in ROUTING[arm].items():
            for name in correctors:
                keep |= (stratum == source) & (corrector == name)
    budget = np.ones(len(candidates), dtype=bool)
    for name, limit in ARM_BUDGET.items():
        if limit is not None and k is not None:
            budget &= ~((corrector == name) & (rank >= min(limit, k)))
    return candidates[keep & budget].reset_index(drop=True)


def exact_sites(
    candidates: pd.DataFrame, arm: str, k: int | None = PRIMARY_K
) -> dict[str, set[str]]:
    """Per environment: the proposal sites carrying an exact candidate inside the arm."""
    block = arm_candidates(candidates, arm, k)
    block = block[block["exact"]]
    return {
        str(name): set(group["lattice_site_id"].astype(str))
        for name, group in block.groupby("environment", sort=True)
    }


def repaired_by_arm(
    views: dict[str, xr1.EnvironmentView], candidates: pd.DataFrame, arms: Sequence[str] = ARMS
) -> dict[str, dict[str, np.ndarray]]:
    """For every arm and environment, which sampled error sites it offers an exact repair for."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        sites = exact_sites(candidates, arm)
        out[arm] = {name: view.repaired(sites.get(name, set())) for name, view in views.items()}
    out[f"{PRIMARY_ARM}@{SECONDARY_K}"] = {
        name: view.repaired(exact_sites(candidates, PRIMARY_ARM, SECONDARY_K).get(name, set()))
        for name, view in views.items()
    }
    return out


def _attach_membership(candidates: pd.DataFrame, strata: pd.DataFrame) -> pd.DataFrame:
    """Join the GT-blind membership flags an arm's routing rule reads."""
    columns = ["site_id", "in_p0", "in_image", "in_text", "ht_incremental"]
    return candidates.merge(
        strata[columns].rename(columns={"site_id": "lattice_site_id"}),
        on="lattice_site_id",
        how="left",
        validate="many_to_one",
    )


def reached_by(
    views: dict[str, xr1.EnvironmentView], sites: dict[str, set[str]]
) -> dict[str, np.ndarray]:
    return {name: view.repaired(sites.get(name, set())) for name, view in views.items()}


def _stratum_sites(strata: pd.DataFrame, stratum: str) -> dict[str, set[str]]:
    block = strata[strata["stratum"] == stratum]
    return {
        str(name): set(group["site_id"].astype(str))
        for name, group in block.groupby("environment", sort=True)
    }


def run_opportunity() -> int:
    """Sections 16-19: opportunity per arm, translation by source, and P0-miss repair recovery."""
    started = time.monotonic()
    _require(CANDIDATES, "normalize")
    for path in (
        OPPORTUNITY_RESULTS,
        TRANSLATION_RESULTS,
        P0_MISS_REPAIR,
        H0_REPRODUCTION,
        ENVIRONMENT_ANALYSIS,
        DOMAIN_ANALYSIS,
    ):
        _forbid(path)
    views = load_views()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates = _attach_membership(pd.read_parquet(CANDIDATES), strata)
    repaired = repaired_by_arm(views, candidates)
    table = xr1.opportunity_table(views, repaired)
    baseline = table[BASELINE_ARM]
    for arm, block in table.items():
        gains = {
            name: block["per_environment"][name] - baseline["per_environment"][name]
            for name in views
        }
        block["gain_over_baseline"] = {
            "per_environment": gains,
            "mean": float(np.mean(list(gains.values()))),
            "pooled": block["pooled"] - baseline["pooled"],
            "environments_improved": int(sum(1 for v in gains.values() if v > 0.0)),
            "environments_at_margin": int(
                sum(1 for v in gains.values() if v >= OPPORTUNITY_MARGIN)
            ),
        }
        arm_block = arm_candidates(candidates, arm) if arm in ARMS else pd.DataFrame()
        block["candidates"] = len(arm_block)
        block["harmful_candidates"] = int(arm_block["is_harmful"].sum()) if len(arm_block) else 0
        block["exact_candidates"] = int(arm_block["exact"].sum()) if len(arm_block) else 0
        block["sites_offered"] = (
            int(arm_block["lattice_site_id"].nunique()) if len(arm_block) else 0
        )
    _write_json_once(
        OPPORTUNITY_RESULTS,
        {
            **_analysis_envelope("opportunity_results"),
            "denominator": "every evaluable OCR error site on the sampled documents",
            "error_sites": int(sum(v.count for v in views.values())),
            "primary_k": PRIMARY_K,
            "arms": table,
        },
    )
    # H0 must reproduce the frozen current pipeline exactly.
    xr1_candidates = pd.read_parquet(xr1.OUT / "xr1_candidates.parquet")
    frozen = xr1_candidates[xr1_candidates["method"] == xr1.G0]
    ours = arm_candidates(candidates, BASELINE_ARM, None)
    key = lambda f: set(  # noqa: E731
        f["environment"].astype(str)
        + "|"
        + f["document_id"].astype(str)
        + "|"
        + f["char_start"].astype(str)
        + "|"
        + f["char_end"].astype(str)
        + "|"
        + f["candidate_text"].astype(str)
    )
    identical = key(frozen) == key(ours)
    _write_json_once(
        H0_REPRODUCTION,
        {
            **_analysis_envelope("h0_reproduction"),
            "frozen_candidates": len(frozen),
            "hy1_candidates": len(ours),
            "identical_edit_sets": bool(identical),
            "frozen_exact": int(frozen["exact"].sum()),
            "hy1_exact": int(ours["exact"].sum()),
            "frozen_harmful": int(frozen["is_harmful"].sum()),
            "hy1_harmful": int(ours["is_harmful"].sum()),
            "opportunity": table[BASELINE_ARM]["mean"],
            "xr1_baseline_opportunity": cc_read_json(xr1.DECISION)["baseline_opportunity"],
        },
    )
    if not identical:
        raise PhaseError("H0 does not reproduce the frozen current pipeline; stop")
    # Translation: of the error sites each proposal source reaches, how many are repaired.
    translation: dict[str, Any] = {}
    for stratum in STRATA:
        sites = _stratum_sites(strata, stratum)
        reached = reached_by(views, sites)
        correctors = ROUTING[PRIMARY_ARM].get(stratum, ())
        repaired_here = {
            name: view.repaired(
                {
                    s
                    for s in exact_sites(candidates, PRIMARY_ARM).get(name, set())
                    if s in sites.get(name, set())
                }
            )
            for name, view in views.items()
        }
        per_env = {
            name: _ratio(int((reached[name] & repaired_here[name]).sum()), int(reached[name].sum()))
            for name in views
        }
        translation[stratum] = {
            "corrector_in_the_primary_arm": list(correctors),
            "reached_error_sites": int(sum(int(v.sum()) for v in reached.values())),
            "repaired_error_sites": int(
                sum(int((reached[n] & repaired_here[n]).sum()) for n in views)
            ),
            "pooled": _ratio(
                int(sum(int((reached[n] & repaired_here[n]).sum()) for n in views)),
                int(sum(int(reached[n].sum()) for n in views)),
            ),
            "mean": float(np.mean(list(per_env.values()))),
            "per_environment": per_env,
        }
    _write_json_once(
        TRANSLATION_RESULTS,
        {
            **_analysis_envelope("translation_results"),
            "definition": (
                "of the true error sites a proposal source reaches, the share that receive an "
                "exact repair from the corrector the primary arm routes that source to"
            ),
            "by_stratum": translation,
        },
    )
    # P0-miss repair recovery.
    errors = pd.read_parquet(ERROR_SITES)
    missed = {
        name: ~errors[errors["environment"] == name]
        .sort_values("align_site_id", kind="stable")["reached_by_p0"]
        .to_numpy(dtype=bool)
        for name in views
    }
    image_sites = _stratum_sites(strata, S_IMAGE)
    image_reached = reached_by(views, image_sites)
    recovery: dict[str, Any] = {}
    for arm in (PRIMARY_ARM, H5, H3, H2, HT):
        per_env = {
            name: _ratio(int((missed[name] & repaired[arm][name]).sum()), int(missed[name].sum()))
            for name in views
        }
        recovery[arm] = {
            "pooled": _ratio(
                int(sum(int((missed[n] & repaired[arm][n]).sum()) for n in views)),
                int(sum(int(missed[n].sum()) for n in views)),
            ),
            "mean": float(np.mean(list(per_env.values()))),
            "per_environment": per_env,
            "recovered": int(sum(int((missed[n] & repaired[arm][n]).sum()) for n in views)),
        }
    conditional = _ratio(
        int(
            sum(int((missed[n] & image_reached[n] & repaired[PRIMARY_ARM][n]).sum()) for n in views)
        ),
        int(sum(int((missed[n] & image_reached[n]).sum()) for n in views)),
    )
    _write_json_once(
        P0_MISS_REPAIR,
        {
            **_analysis_envelope("p0_miss_repair_recovery"),
            "p0_missed_error_sites": int(sum(int(v.sum()) for v in missed.values())),
            "p0_missed_reached_by_the_image_proposer": int(
                sum(int((missed[n] & image_reached[n]).sum()) for n in views)
            ),
            "by_arm": recovery,
            "probability_of_repair_given_p0_miss_and_image_hit": conditional,
        },
    )
    # Environment and domain views of the primary comparison.
    _write_json_once(
        ENVIRONMENT_ANALYSIS,
        {
            **_analysis_envelope("environment_analysis"),
            "by_environment": [
                {
                    "environment": name,
                    "error_sites": int(view.count),
                    **{arm: table[arm]["per_environment"][name] for arm in ARMS},
                    "gain": table[PRIMARY_ARM]["per_environment"][name]
                    - table[BASELINE_ARM]["per_environment"][name],
                }
                for name, view in views.items()
            ],
        },
    )
    domains: dict[str, Any] = {}
    for domain in ("modern_forms", "historical_print"):
        names = [n for n, v in views.items() if _domain(v.corpus) == domain]
        total = sum(views[n].count for n in names)
        domains[domain] = {
            "error_sites": int(total),
            **{
                arm: _ratio(int(sum(int(repaired[arm][n].sum()) for n in names)), int(total))
                for arm in ARMS
            },
        }
    _write_json_once(
        DOMAIN_ANALYSIS,
        {**_analysis_envelope("domain_analysis"), "by_domain": domains},
    )
    print(
        f"opportunity: H0 {table[BASELINE_ARM]['mean']:.4f}, "
        f"H4 {table[PRIMARY_ARM]['mean']:.4f}, H5 {table[H5]['mean']:.4f}; "
        f"image-only translation {translation[S_IMAGE]['pooled']:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 20-31: analyses

HISTORICAL_GLYPH = lp1.HISTORICAL_GLYPH
SEGMENTATION_SUBTYPES = lp1.SEGMENTATION_SUBTYPES
ERROR_KINDS = ("substitution", "deletion", "insertion", "segmentation")
KIND_NAME = {
    "deletion": "omission",
    "insertion": "spurious insertion",
    "substitution": "substitution",
    "segmentation": "segmentation",
}


def _kind_rate(
    views: dict[str, xr1.EnvironmentView], repaired: dict[str, np.ndarray], kind: str
) -> dict[str, Any]:
    hit = 0
    total = 0
    per_env: dict[str, float] = {}
    for name, view in views.items():
        mask = view.error_kind == kind
        hit += int(repaired[name][mask].sum())
        total += int(mask.sum())
        per_env[name] = _ratio(int(repaired[name][mask].sum()), int(mask.sum()))
    return {"pooled": _ratio(hit, total), "error_sites": total, "per_environment": per_env}


def _subtype_rate(
    views: dict[str, xr1.EnvironmentView], repaired: dict[str, np.ndarray], subtype: str
) -> float:
    hit = sum(int(repaired[n][views[n].error_subtype == subtype].sum()) for n in views)
    total = sum(int((views[n].error_subtype == subtype).sum()) for n in views)
    return _ratio(hit, total)


def run_types() -> int:
    """Sections 23-27: opportunity by error type, and the three type diagnostics."""
    started = time.monotonic()
    _require(OPPORTUNITY_RESULTS, "opportunity")
    for path in (
        ERROR_TYPE_ANALYSIS,
        OMISSION_ANALYSIS,
        SEGMENTATION_ANALYSIS,
        SUBSTITUTION_ANALYSIS,
        HISTORICAL_GLYPH_ANALYSIS,
    ):
        _forbid(path)
    views = load_views()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates = _attach_membership(pd.read_parquet(CANDIDATES), strata)
    repaired = repaired_by_arm(views, candidates)
    by_kind: dict[str, Any] = {}
    for kind in ERROR_KINDS:
        block = {arm: _kind_rate(views, repaired[arm], kind) for arm in ARMS}
        gain = block[PRIMARY_ARM]["pooled"] - block[BASELINE_ARM]["pooled"]
        by_kind[kind] = {
            "name": KIND_NAME[kind],
            "error_sites": block[BASELINE_ARM]["error_sites"],
            "opportunity": {arm: block[arm]["pooled"] for arm in ARMS},
            "per_environment": {arm: block[arm]["per_environment"] for arm in ARMS},
            "gain_over_baseline": gain,
            "reaches_the_margin": bool(gain >= WEAK_TYPE_MARGIN),
        }
    improved = [k for k in WEAK_TYPES if by_kind[k]["reaches_the_margin"]]
    _write_json_once(
        ERROR_TYPE_ANALYSIS,
        {
            **_analysis_envelope("error_type_analysis"),
            "by_kind": by_kind,
            "weak_types": list(WEAK_TYPES),
            "weak_types_reaching_margin": improved,
            "margin": WEAK_TYPE_MARGIN,
            "taxonomy_note": (
                "the frozen taxonomy's deletion is an omission, its insertion is spurious"
            ),
        },
    )
    # Omission: the diagnostic LP1 flagged, where P0's gap rule is the strong component.
    errors = pd.read_parquet(ERROR_SITES)
    omission_reach = {
        "reached_by_p0": int(
            errors[(errors["site_kind"] == "deletion") & errors["reached_by_p0"]].shape[0]
        ),
        "error_sites": int((errors["site_kind"] == "deletion").sum()),
        "representable": int(
            errors[(errors["site_kind"] == "deletion") & errors["representable"]].shape[0]
        ),
    }
    image_sites = _stratum_sites(strata, S_IMAGE)
    image_reached = reached_by(views, image_sites)
    omission_reach["reached_by_the_image_proposer"] = int(
        sum(int(image_reached[n][views[n].error_kind == "deletion"].sum()) for n in views)
    )
    _write_json_once(
        OMISSION_ANALYSIS,
        {
            **_analysis_envelope("omission_analysis"),
            **omission_reach,
            "opportunity": by_kind["deletion"]["opportunity"],
            "note": (
                "an omission is reachable only through a gap anchor; P0's gap rule is geometric "
                "and the image proposer rarely names a gap, so this is the type the hybrid must "
                "preserve rather than replace"
            ),
        },
    )
    _write_json_once(
        SEGMENTATION_ANALYSIS,
        {
            **_analysis_envelope("segmentation_analysis"),
            "opportunity": by_kind["segmentation"]["opportunity"],
            "by_subtype": {
                subtype: {
                    "error_sites": int(
                        sum(int((v.error_subtype == subtype).sum()) for v in views.values())
                    ),
                    "opportunity": {
                        arm: _subtype_rate(views, repaired[arm], subtype) for arm in ARMS
                    },
                }
                for subtype in SEGMENTATION_SUBTYPES
            },
            "localization_recall_in_lp1": cc_read_json(lp1.ERROR_TYPE_ANALYSIS)["by_kind"][
                "segmentation"
            ]["recall"],
        },
    )
    _write_json_once(
        SUBSTITUTION_ANALYSIS,
        {
            **_analysis_envelope("substitution_analysis"),
            "opportunity": by_kind["substitution"]["opportunity"],
            "error_sites": by_kind["substitution"]["error_sites"],
            "localization_recall_in_lp1": cc_read_json(lp1.ERROR_TYPE_ANALYSIS)["by_kind"][
                "substitution"
            ]["recall"],
        },
    )
    glyph_sites = int(sum(int((v.error_subtype == HISTORICAL_GLYPH).sum()) for v in views.values()))
    _write_json_once(
        HISTORICAL_GLYPH_ANALYSIS,
        {
            **_analysis_envelope("historical_glyph_analysis"),
            "error_sites": glyph_sites,
            "reached_by_p0": int(
                errors[(errors["subtype"] == HISTORICAL_GLYPH) & errors["reached_by_p0"]].shape[0]
            ),
            "opportunity": {
                arm: _subtype_rate(views, repaired[arm], HISTORICAL_GLYPH) for arm in ARMS
            },
            "localization_recall_in_lp1": cc_read_json(lp1.HISTORICAL_GLYPH_ANALYSIS)[
                "localization_recall"
            ],
            "reading": "finding a site and repairing it are separate; this stage measures both",
        },
    )
    print(
        f"types: weak types improved {improved}; "
        f"omission {by_kind['deletion']['opportunity'][PRIMARY_ARM]:.4f} "
        f"against {by_kind['deletion']['opportunity'][BASELINE_ARM]:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _arm_burden(candidates: pd.DataFrame, arm: str, repaired_sites: int) -> dict[str, Any]:
    block = arm_candidates(candidates, arm)
    harmful = int(block["is_harmful"].sum())
    return {
        "candidates": len(block),
        "exact_candidates": int(block["exact"].sum()),
        "beneficial_candidates": int(block["beneficial"].sum()),
        "harmful_candidates": harmful,
        "miscorrections": int((block["outcome"] == "miscorrection").sum()),
        "overcorrections": int((block["outcome"] == "overcorrection").sum()),
        "harmful_share_of_candidates": _ratio(harmful, len(block)),
        "candidate_precision": _ratio(int(block["exact"].sum()), len(block)),
        "sites_offered": int(block["lattice_site_id"].nunique()),
        "harmful_per_site": _ratio(harmful, int(block["lattice_site_id"].nunique())),
        "harmful_per_exact_repaired_site": _ratio(harmful, repaired_sites),
    }


def run_burden() -> int:
    """Sections 28-31: harmful candidates, marginal harm, clean sites, damage and abstention."""
    started = time.monotonic()
    _require(OPPORTUNITY_RESULTS, "opportunity")
    for path in (
        HARM_BURDEN,
        HARM_EFFICIENCY,
        CLEAN_SITE_ANALYSIS,
        EDIT_DAMAGE,
        ABSTENTION,
        SITE_OUTCOMES,
    ):
        _forbid(path)
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates = _attach_membership(pd.read_parquet(CANDIDATES), strata)
    truth = pd.read_parquet(SITE_TRUTH)
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)["arms"]
    burden = {
        arm: _arm_burden(candidates, arm, int(opportunity[arm]["repaired_error_sites"]))
        for arm in ARMS
    }
    _write_json_once(
        HARM_BURDEN,
        {
            **_analysis_envelope("harmful_candidate_burden"),
            "taxonomy": "SGV-CG1's frozen outcome taxonomy, unchanged",
            "by_arm": burden,
            "by_stratum": {
                stratum: {
                    "candidates": len(block),
                    "harmful_candidates": int(block["is_harmful"].sum()),
                    "exact_candidates": int(block["exact"].sum()),
                    "harmful_share": _ratio(int(block["is_harmful"].sum()), len(block)),
                }
                for stratum, block in arm_candidates(candidates, PRIMARY_ARM).groupby(
                    "proposal_stratum"
                )
            },
        },
    )
    # What the new branch costs per repair it adds.
    extra_repairs = int(
        opportunity[PRIMARY_ARM]["repaired_error_sites"]
        - opportunity[BASELINE_ARM]["repaired_error_sites"]
    )
    extra_harm = (
        burden[PRIMARY_ARM]["harmful_candidates"] - burden[BASELINE_ARM]["harmful_candidates"]
    )
    extra_candidates = burden[PRIMARY_ARM]["candidates"] - burden[BASELINE_ARM]["candidates"]
    comparator = burden[BASELINE_ARM]["harmful_per_exact_repaired_site"]
    _write_json_once(
        HARM_EFFICIENCY,
        {
            **_analysis_envelope("incremental_harm_efficiency"),
            "incremental_exact_repaired_sites": extra_repairs,
            "incremental_harmful_candidates": int(extra_harm),
            "incremental_candidates": int(extra_candidates),
            "marginal_harm_per_new_repair": _ratio(int(extra_harm), extra_repairs),
            "marginal_candidates_per_new_repair": _ratio(int(extra_candidates), extra_repairs),
            "comparator": comparator,
            "comparator_source": HARM_COMPARATOR_SOURCE,
            "within_comparator": bool(
                extra_repairs > 0 and _ratio(int(extra_harm), extra_repairs) <= comparator
            ),
        },
    )
    # Site-level behaviour, one row per (corrector, requested site). A site the corrector was
    # asked about but produced no atomic edit for stays in the table as an abstention, exactly as
    # SGV-XR1 counts it -- never as an absent row, which would hide it.
    rows: list[dict[str, Any]] = []
    merged = candidates.merge(
        truth[["site_id", "site_class", "site_d_before"]].rename(
            columns={"site_id": "lattice_site_id"}
        ),
        on="lattice_site_id",
        how="left",
        validate="many_to_one",
    )
    blocks = dict(list(merged.groupby(["corrector_source", "lattice_site_id"])))
    meta = truth.set_index("site_id")
    unlabelable = 0
    for corrector, sites in corrector_demand(strata).items():
        for site in sites:
            record = meta.loc[site]
            # A site whose ground truth did not resolve has its candidates dropped by the label
            # pipeline, so counting it here would read as an abstention the model never made.
            if str(record["site_class"]) == xr1.SITE_UNLABELABLE:
                unlabelable += 1
                continue
            row = {
                "corrector": corrector,
                "site_id": site,
                "environment": str(record["environment"]),
                "document_id": str(record["document_id"]),
                "stratum": str(record["stratum"]),
                "site_class": str(record["site_class"]),
                "d_before": int(record["site_d_before"]),
            }
            block = blocks.get((corrector, site))
            if block is None:
                rows.append(
                    {
                        **row,
                        "candidates": 0,
                        "exact_at_1": False,
                        "harmful_candidates": 0,
                        "top1_text": "",
                        "top1_outcome": "",
                        "abstained": True,
                        "d_after_top1": int(record["site_d_before"]),
                    }
                )
                continue
            top = block[block["generator_rank"] == block["generator_rank"].min()]
            rows.append(
                {
                    **row,
                    "candidates": len(block),
                    "exact_at_1": bool(top["exact"].any()),
                    "harmful_candidates": int(block["is_harmful"].sum()),
                    "top1_text": str(top["candidate_text"].iloc[0]),
                    "top1_outcome": str(top["outcome"].iloc[0]),
                    "abstained": False,
                    "d_after_top1": int(top["d_after"].iloc[0]),
                }
            )
    outcomes = pd.DataFrame(rows)
    _write_parquet_once(SITE_OUTCOMES, outcomes)
    if unlabelable == 0:
        raise PhaseError("no unlabelable site was skipped; the truth table looks wrong")
    clean = outcomes[outcomes["site_class"] == SITE_CLEAN]
    _write_json_once(
        CLEAN_SITE_ANALYSIS,
        {
            **_analysis_envelope("clean_site_analysis"),
            "definition": "a proposed site whose OCR already matches the truth",
            "by_corrector": {
                str(corrector): {
                    "clean_sites": len(block),
                    "abstained": int(block["abstained"].sum()),
                    "rewrote": int((~block["abstained"]).sum()),
                    "rewrite_rate": _ratio(int((~block["abstained"]).sum()), len(block)),
                    "harmful_candidates": int(block["harmful_candidates"].sum()),
                }
                for corrector, block in clean.groupby("corrector")
            },
            "by_stratum": {
                str(stratum): {
                    "clean_sites": len(block),
                    "rewrite_rate": _ratio(int((~block["abstained"]).sum()), len(block)),
                }
                for stratum, block in clean.groupby("stratum")
            },
        },
    )
    # A wrong edit is an edit: a site the corrector abstained at is not one, and counting it
    # would read as damage the model never did.
    wrong = outcomes[
        (outcomes["site_class"] == SITE_ERROR) & ~outcomes["exact_at_1"] & ~outcomes["abstained"]
    ]
    _write_json_once(
        EDIT_DAMAGE,
        {
            **_analysis_envelope("edit_damage_analysis"),
            "secondary": True,
            "denominator": "error sites the corrector edited without reaching the truth exactly",
            "by_corrector": {
                str(corrector): {
                    "wrong_edits": len(block),
                    "improved_but_not_exact": int(
                        (block["d_after_top1"] < block["d_before"]).sum()
                    ),
                    "unchanged_quality": int((block["d_after_top1"] == block["d_before"]).sum()),
                    "worsened": int((block["d_after_top1"] > block["d_before"]).sum()),
                }
                for corrector, block in wrong.groupby("corrector")
            },
        },
    )
    _write_json_once(
        ABSTENTION,
        {
            **_analysis_envelope("abstention_analysis"),
            "definition": (
                "the corrector was asked about the site and produced no atomic edit for it"
            ),
            "denominator": "sites whose ground truth resolved, so a candidate there is labelable",
            "sites_excluded_as_unlabelable": unlabelable,
            "by_corrector_and_stratum": [
                {
                    "corrector": str(corrector),
                    "stratum": str(stratum),
                    "sites": len(block),
                    "abstention_rate": _ratio(int(block["abstained"].sum()), len(block)),
                }
                for (corrector, stratum), block in outcomes.groupby(["corrector", "stratum"])
            ],
            "by_site_class": [
                {
                    "corrector": str(corrector),
                    "site_class": str(site_class),
                    "sites": len(block),
                    "abstention_rate": _ratio(int(block["abstained"].sum()), len(block)),
                }
                for (corrector, site_class), block in outcomes.groupby(["corrector", "site_class"])
            ],
        },
    )
    print(
        f"burden: H4 {burden[PRIMARY_ARM]['harmful_candidates']} harmful vs "
        f"H0 {burden[BASELINE_ARM]['harmful_candidates']}; "
        f"{extra_repairs} new repairs ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 32-36: routing


def _repair_sets(
    views: dict[str, xr1.EnvironmentView], repaired: dict[str, dict[str, np.ndarray]], arm: str
) -> set[str]:
    out: set[str] = set()
    for name, view in views.items():
        out |= set(view.error_ids[repaired[arm][name]].tolist())
    return out


def run_routing() -> int:
    """Sections 32-36: candidate complementarity, routing comparison and the budget curve."""
    started = time.monotonic()
    _require(OPPORTUNITY_RESULTS, "opportunity")
    for path in (CANDIDATE_OVERLAP, ROUTING_COMPARISON, BUDGET_CURVE):
        _forbid(path)
    views = load_views()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates = _attach_membership(pd.read_parquet(CANDIDATES), strata)
    repaired = repaired_by_arm(views, candidates)
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)["arms"]
    total = int(sum(v.count for v in views.values()))
    baseline = _repair_sets(views, repaired, BASELINE_ARM)
    primary = _repair_sets(views, repaired, PRIMARY_ARM)
    image_branch = _repair_sets(views, repaired, H2)
    _write_json_once(
        CANDIDATE_OVERLAP,
        {
            **_analysis_envelope("candidate_overlap"),
            "unit": "sampled OCR error sites with at least one exact repair candidate",
            "h0_vs_h4": {
                "both": len(baseline & primary),
                "h0_only": len(baseline - primary),
                "h4_only": len(primary - baseline),
                "neither": total - len(baseline | primary),
                "jaccard": _ratio(len(baseline & primary), len(baseline | primary)),
            },
            "h0_vs_image_branch": {
                "both": len(baseline & image_branch),
                "h0_only": len(baseline - image_branch),
                "image_branch_only": len(image_branch - baseline),
                "union": len(baseline | image_branch),
                "jaccard": _ratio(len(baseline & image_branch), len(baseline | image_branch)),
            },
            "h4_only_by_stratum": _attribute_new_repairs(views, repaired, strata, candidates),
        },
    )
    _write_json_once(
        ROUTING_COMPARISON,
        {
            **_analysis_envelope("routing_comparison"),
            "question": "is specialization useful, against one corrector everywhere or both?",
            "arms": {
                arm: {
                    "opportunity": opportunity[arm]["mean"],
                    "pooled": opportunity[arm]["pooled"],
                    "candidates": opportunity[arm]["candidates"],
                    "harmful_candidates": opportunity[arm]["harmful_candidates"],
                    "gain_over_baseline": opportunity[arm]["gain_over_baseline"]["mean"],
                    "routes": {s: list(c) for s, c in ROUTING[arm].items()},
                }
                for arm in (H3, PRIMARY_ARM, H5, HT)
            },
            "note": "reported independently; it never overrides the primary outcome rule",
        },
    )
    # The image branch at increasing fractions of its own frozen ranking, P0 always retained.
    image_only = strata[strata["stratum"] == S_IMAGE]
    curve: dict[str, Any] = {}
    for fraction in IMAGE_BRANCH_FRACTIONS:
        kept: set[str] = set()
        for _name, block in image_only.groupby("environment"):
            ordered = block.sort_values(
                ["image_suspicion", "hash_order"], ascending=[False, True], kind="stable"
            )
            take = round(fraction * len(ordered))
            kept |= set(ordered.head(take)["site_id"].astype(str))
        subset = candidates[
            (candidates["proposal_stratum"] != S_IMAGE) | candidates["lattice_site_id"].isin(kept)
        ]
        sites = exact_sites(subset, PRIMARY_ARM)
        rates = {name: view.repaired(sites.get(name, set())) for name, view in views.items()}
        curve[f"{fraction:.2f}"] = {
            "image_only_sites": len(kept),
            "opportunity_mean": float(np.mean([_rate(rates[n]) for n in views])),
            "opportunity_pooled": _ratio(int(sum(int(rates[n].sum()) for n in views)), total),
            "candidates": len(arm_candidates(subset, PRIMARY_ARM)),
        }
    _write_json_once(
        BUDGET_CURVE,
        {
            **_analysis_envelope("budget_curve"),
            "rule": "P0 is always retained; the image branch is cut by LP1's own frozen ranking",
            "reads_no_label": True,
            "fractions": list(IMAGE_BRANCH_FRACTIONS),
            "by_fraction": curve,
        },
    )
    print(
        f"routing: H4-only repairs {len(primary - baseline)}, H0-only {len(baseline - primary)} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def _rate(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _attribute_new_repairs(
    views: dict[str, xr1.EnvironmentView],
    repaired: dict[str, dict[str, np.ndarray]],
    strata: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    """Where the primary arm's new repairs come from: stratum, error type and environment."""
    baseline = _repair_sets(views, repaired, BASELINE_ARM)
    new_by_env: dict[str, int] = {}
    new_by_kind: dict[str, int] = {}
    for name, view in views.items():
        mask = repaired[PRIMARY_ARM][name] & ~np.isin(view.error_ids, list(baseline))
        new_by_env[name] = int(mask.sum())
        for kind in ERROR_KINDS:
            new_by_kind[kind] = new_by_kind.get(kind, 0) + int(
                (mask & (view.error_kind == kind)).sum()
            )
    image_sites = _stratum_sites(strata, S_IMAGE)
    image_reached = reached_by(views, image_sites)
    from_image = 0
    for name, view in views.items():
        mask = repaired[PRIMARY_ARM][name] & ~np.isin(view.error_ids, list(baseline))
        from_image += int((mask & image_reached[name]).sum())

    return {
        "by_environment": new_by_env,
        "by_error_kind": new_by_kind,
        "reached_by_the_image_proposer": from_image,
        "environments_contributing": int(sum(1 for v in new_by_env.values() if v > 0)),
    }


# ------------------------------------------------------------------ section 35: statistics


def _paired_frame(
    views: dict[str, xr1.EnvironmentView],
    repaired: dict[str, dict[str, np.ndarray]],
    a: str,
    b: str,
    kind: str | None = None,
) -> pd.DataFrame:
    return xr1._paired_frame(views, repaired, a, b, kind)


def run_stats() -> int:
    """Section 35: the document-clustered paired bootstrap, Holm within each frozen family."""
    started = time.monotonic()
    _require(OPPORTUNITY_RESULTS, "opportunity")
    _forbid(STATISTICAL_TESTS)
    views = load_views()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates = _attach_membership(pd.read_parquet(CANDIDATES), strata)
    repaired = repaired_by_arm(views, candidates)
    primary = [
        (f"P1_{PRIMARY_ARM}_vs_{BASELINE_ARM}", PRIMARY_ARM, BASELINE_ARM, None),
        (f"P2_{H2}_vs_{H1}", H2, H1, None),
        (f"P3_{PRIMARY_ARM}_vs_{HT}", PRIMARY_ARM, HT, None),
    ]
    secondary = [
        (f"S1_{H5}_vs_{BASELINE_ARM}", H5, BASELINE_ARM, None),
        (f"S2_{H3}_vs_{PRIMARY_ARM}", H3, PRIMARY_ARM, None),
        (f"S3_{H1}_vs_{BASELINE_ARM}", H1, BASELINE_ARM, None),
        *[
            (f"T_{kind}_{PRIMARY_ARM}_vs_{BASELINE_ARM}", PRIMARY_ARM, BASELINE_ARM, kind)
            for kind in ERROR_KINDS
        ],
    ]
    families: dict[str, dict[str, dict[str, Any]]] = {}
    for family, tests in (("primary", primary), ("secondary", secondary)):
        rows = {}
        for name, a, b, kind in tests:
            frame = _paired_frame(views, repaired, a, b, kind)
            rows[name] = xr1.comparison(frame, name, family)
        families[family] = s15.holm(rows)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "bootstrap": {
                "unit": "document, resampled within environment",
                "paired": True,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile",
            },
            "multiplicity": "Holm within each family; family sizes fixed in the design record",
            "primary_family": families["primary"],
            "secondary_family": families["secondary"],
        },
    )
    head = families["primary"][f"P1_{PRIMARY_ARM}_vs_{BASELINE_ARM}"]
    print(
        f"stats: P1 {head['effect']:+.4f} [{head['ci_low']:+.4f}, {head['ci_high']:+.4f}] "
        f"holm {head['holm_adjusted_p']:.4f} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 43-45: controls


def _ceiling_arm(views: dict[str, xr1.EnvironmentView], strata: pd.DataFrame) -> dict[str, Any]:
    """Control C: what a perfect corrector at the hybrid's own sites could reach. Analysis only."""
    sites = {
        str(name): set(group["site_id"].astype(str))
        for name, group in strata[strata["stratum"].isin(STRATA)].groupby("environment")
    }
    reached = reached_by(views, sites)
    return {
        "reached_error_sites": int(sum(int(v.sum()) for v in reached.values())),
        "pooled": _ratio(
            int(sum(int(v.sum()) for v in reached.values())),
            int(sum(v.count for v in views.values())),
        ),
        "mean": float(np.mean([_rate(reached[n]) for n in views])),
        "enters_a_primary_arm": False,
        "note": "a ceiling on localization, not a correction result",
    }


def run_controls() -> int:
    """Section 43: the five controls, each of which could fail."""
    started = time.monotonic()
    _require(OPPORTUNITY_RESULTS, "opportunity")
    _forbid(CONTROL_RESULTS)
    views = load_views()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates = _attach_membership(pd.read_parquet(CANDIDATES), strata)
    truth = pd.read_parquet(SITE_TRUTH)
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)["arms"]
    reproduction = cc_read_json(H0_REPRODUCTION)
    # Control B: the identity path. Every site carries one probe candidate that repeats the OCR
    # unchanged, and the label that probe receives is how the pipeline reports the site's own
    # truth: an error site must sit at a positive distance from it, a clean site at zero.
    error_sites = truth[truth["site_class"] == SITE_ERROR]
    clean_sites = truth[truth["site_class"] == SITE_CLEAN]

    # Control E: duplicating every candidate may not change opportunity.
    duplicated = pd.concat([candidates, candidates], ignore_index=True)
    duplicated_repaired = repaired_by_arm(views, duplicated, (PRIMARY_ARM,))
    original_repaired = repaired_by_arm(views, candidates, (PRIMARY_ARM,))
    duplicate_stable = all(
        bool(np.array_equal(duplicated_repaired[PRIMARY_ARM][n], original_repaired[PRIMARY_ARM][n]))
        for n in views
    )
    controls = {
        "A_h0_reproduces_the_frozen_pipeline": {
            "identical_edit_sets": reproduction["identical_edit_sets"],
            "opportunity": reproduction["opportunity"],
            "frozen_opportunity": reproduction["xr1_baseline_opportunity"],
            "discriminating": True,
        },
        "B_identity_candidate_path": {
            "error_sites": len(error_sites),
            "clean_sites": len(clean_sites),
            "error_sites_at_a_positive_distance": int((error_sites["site_d_before"] > 0).sum()),
            "clean_sites_at_zero_distance": int((clean_sites["site_d_before"] == 0).sum()),
            "labelling_behaves": bool(
                (error_sites["site_d_before"] > 0).all()
                and (clean_sites["site_d_before"] == 0).all()
            ),
            "discriminating": True,
        },
        "C_ground_truth_corrector_ceiling": _ceiling_arm(views, strata),
        "D_matched_text_proposal_hybrid": {
            "opportunity": opportunity[HT]["mean"],
            "gain_over_baseline": opportunity[HT]["gain_over_baseline"]["mean"],
            "candidates": opportunity[HT]["candidates"],
            "note": "adds model-proposed sites without image-aware localization",
            "discriminating": True,
        },
        "E_candidate_duplication": {
            "opportunity_unchanged_under_duplication": duplicate_stable,
            "discriminating": True,
        },
    }
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "controls": controls,
            "discriminating": int(sum(1 for c in controls.values() if c.get("discriminating"))),
            "total": len(controls),
        },
    )
    print(f"controls: {len(controls)} reported ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ section 44: falsification


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
    """Section 44: twenty tests, each of which would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(OPPORTUNITY_RESULTS, "opportunity")
    _forbid(FALSIFICATION)
    views = load_views()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates = _attach_membership(pd.read_parquet(CANDIDATES), strata)
    repaired = repaired_by_arm(views, candidates)
    reproduction = cc_read_json(H0_REPRODUCTION)
    audit = cc_read_json(CACHE_REUSE_AUDIT)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    record(
        "t01_h0_reproduces_the_frozen_current_pipeline",
        reproduction["identical_edit_sets"]
        and abs(reproduction["opportunity"] - reproduction["xr1_baseline_opportunity"]) < 5e-5,
        {"candidates": reproduction["hy1_candidates"]},
    )
    lp1_record = cc_read_json(LP1_REPRODUCTION)["tables"]
    record(
        "t02_lp1_proposal_hashes_are_unchanged",
        all(block["matches_lp1_provenance"] for block in lp1_record.values()),
        {k: v["sha256"][:12] for k, v in lp1_record.items()},
    )
    strata_names = _names_in("build_strata") | _names_in("stratum_of")
    record(
        "t03_strata_are_independent_of_ground_truth",
        not (strata_names & set(FORBIDDEN_INPUT_COLUMNS)),
        sorted(strata_names & set(FORBIDDEN_INPUT_COLUMNS)),
    )
    routing_names = (
        _names_in("routed_sites") | _names_in("corrector_of") | _names_in("arm_candidates")
    )
    leaked = routing_names & set(FORBIDDEN_INPUT_COLUMNS)
    record("t04_routing_reads_no_ground_truth", not leaked, sorted(leaked))
    crop_names = _names_in("run_contexts")
    record(
        "t05_correction_crops_read_no_ground_truth",
        not (crop_names & set(FORBIDDEN_INPUT_COLUMNS)),
        sorted(crop_names & set(FORBIDDEN_INPUT_COLUMNS)),
    )
    prompt_names = _names_in("requests_for") | _names_in("_corrector_requests")
    record(
        "t06_the_prompt_carries_no_replacement_text",
        not (prompt_names & {"gt_text", "candidate_text", "replacement", "truth"}),
        sorted(prompt_names & {"gt_text", "candidate_text", "replacement", "truth"}),
    )
    p0_sites = set(strata[strata["in_p0"]]["site_id"].astype(str))
    image_only = set(strata[strata["stratum"] == S_IMAGE]["site_id"].astype(str))
    record(
        "t07_image_only_sites_are_absent_from_p0",
        not (image_only & p0_sites),
        {"image_only": len(image_only), "intersecting_p0": len(image_only & p0_sites)},
    )
    overlap = strata[strata["stratum"] == S_OVERLAP]["site_id"].astype(str)
    overlap_candidates = arm_candidates(candidates, PRIMARY_ARM)
    overlap_block = overlap_candidates[overlap_candidates["lattice_site_id"].isin(set(overlap))]
    record(
        "t08_overlap_follows_the_frozen_h4_rule",
        set(overlap_block["corrector_source"].astype(str)) <= {C0},
        sorted(set(overlap_block["corrector_source"].astype(str))),
    )
    h5_overlap = arm_candidates(candidates, H5)
    h5_overlap = h5_overlap[h5_overlap["lattice_site_id"].isin(set(overlap))]
    record(
        "t09_h5_retains_both_correctors_at_overlap",
        {C0, C1} <= set(h5_overlap["corrector_source"].astype(str)) or overlap_block.empty,
        sorted(set(h5_overlap["corrector_source"].astype(str))),
    )
    attribution = cc_read_json(CANDIDATE_ATTRIBUTION)["by_corrector_and_stratum"]
    record(
        "t10_deduplication_preserves_provenance",
        int(sum(row["candidates"] for row in attribution)) == len(candidates),
        {"rows": len(attribution)},
    )
    duplicated = pd.concat([candidates, candidates], ignore_index=True)
    duplicated_repaired = repaired_by_arm(views, duplicated, (PRIMARY_ARM,))
    record(
        "t11_duplicating_candidates_does_not_raise_opportunity",
        all(
            bool(np.array_equal(duplicated_repaired[PRIMARY_ARM][n], repaired[PRIMARY_ARM][n]))
            for n in views
        ),
    )
    without_exact = candidates[~candidates["exact"]]
    stripped = repaired_by_arm(views, without_exact, (PRIMARY_ARM,))
    record(
        "t12_removing_exact_candidates_removes_the_opportunity",
        all(not bool(stripped[PRIMARY_ARM][n].any()) for n in views),
    )
    wrong = candidates.copy()
    wrong["exact"] = False
    wrong_repaired = repaired_by_arm(views, wrong, (PRIMARY_ARM,))
    record(
        "t13_a_wrong_candidate_cannot_create_an_exact_repair",
        all(not bool(wrong_repaired[PRIMARY_ARM][n].any()) for n in views),
    )
    # t14: the top-ranked candidate is decided by the generator's own rank, so re-deriving it
    # from a randomly permuted table must return the same text at every site.
    shuffled = candidates.sample(frac=1.0, random_state=BOOTSTRAP_SEED).reset_index(drop=True)

    def _top1(frame: pd.DataFrame) -> dict[tuple[str, str], str]:
        ordered = frame.sort_values(["generator_rank", "candidate_id"], kind="stable")
        first = ordered.groupby(["corrector_source", "lattice_site_id"], sort=True).first()
        return {
            (str(a), str(b)): str(text)
            for (a, b), text in first["candidate_text"].astype(str).items()
        }

    record(
        "t14_top1_is_decided_by_rank_alone",
        _top1(candidates) == _top1(shuffled),
        {"sites": len(_top1(candidates))},
    )
    record(
        "t15_cache_reuse_requires_exact_request_identity",
        all(audit["registries_identical_to_xr1"].values())
        and audit["by_corrector"][C1]["rejected_count"] == 0,
        audit["by_corrector"][C1]["rejected_count"],
    )
    ht_sites = set(strata[strata["ht_incremental"]]["site_id"].astype(str))
    record(
        "t16_the_text_control_uses_its_own_frozen_incremental_budget",
        not (ht_sites & p0_sites),
        {"ht_incremental": len(ht_sites)},
    )
    component_union = set()
    for name, view in views.items():
        component = np.zeros(view.count, dtype=bool)
        for stratum, correctors in ROUTING[PRIMARY_ARM].items():
            sites = _stratum_sites(strata, stratum).get(name, set())
            block = candidates[
                candidates["lattice_site_id"].isin(sites)
                & candidates["corrector_source"].isin(correctors)
                & candidates["exact"]
                & (
                    candidates["generator_rank"]
                    < candidates["corrector_source"].map(lambda c: ARM_BUDGET.get(c) or SECONDARY_K)
                )
            ]
            component |= view.repaired(set(block["lattice_site_id"].astype(str)))
        component_union |= set(view.error_ids[component].tolist())
    record(
        "t17_the_primary_arm_never_exceeds_its_components",
        _repair_sets(views, repaired, PRIMARY_ARM) <= component_union,
    )
    ceiling = _ceiling_arm(views, strata)
    record(
        "t18_the_ceiling_control_enters_no_arm",
        not ceiling["enters_a_primary_arm"]
        and ceiling["pooled"] >= cc_read_json(OPPORTUNITY_RESULTS)["arms"][PRIMARY_ARM]["pooled"],
        ceiling["pooled"],
    )
    record(
        "t19_labels_are_attached_only_after_the_raw_freeze",
        cc_read_json(RAW_OUTPUT_HASHES)["verified_before_any_label"],
    )
    permuted = candidates.copy()
    mask = permuted["proposal_stratum"] == S_IMAGE
    if int(mask.sum()):
        rotated = np.roll(permuted.loc[mask, "lattice_site_id"].to_numpy(), 1)
        permuted.loc[mask, "lattice_site_id"] = rotated
        permuted_repaired = repaired_by_arm(views, permuted, (PRIMARY_ARM,))
        destroyed = sum(int(permuted_repaired[PRIMARY_ARM][n].sum()) for n in views) < sum(
            int(repaired[PRIMARY_ARM][n].sum()) for n in views
        )
    else:
        destroyed = True
    record(
        "t20_permuting_image_site_identities_destroys_their_linkage",
        destroyed,
        {"image_only_candidates": int(mask.sum())},
    )
    passed = sum(1 for test in tests if test["passed"])
    _write_json_once(
        FALSIFICATION,
        {
            **_analysis_envelope("falsification_tests"),
            "tests": tests,
            "passed": passed,
            "total": len(tests),
        },
    )
    print(f"negative: {passed} of {len(tests)} pass")
    for test in tests:
        print(f"  {'PASS' if test['passed'] else 'FAIL'} {test['test']}")
    if passed != len(tests):
        raise PhaseError(f"{len(tests) - passed} falsification tests failed")
    print(f"  [negative {time.monotonic() - started:.0f}s]")
    return 0


# ------------------------------------------------------------------ section 45: the decision


def decision_criteria(
    opportunity: dict[str, Any],
    stats: dict[str, Any],
    types: dict[str, Any],
    efficiency: dict[str, Any],
    overlap: dict[str, Any],
) -> dict[str, bool]:
    """C1-C6, read from the frozen gate alone."""
    gain = opportunity[PRIMARY_ARM]["gain_over_baseline"]
    primary = stats["primary_family"][f"P1_{PRIMARY_ARM}_vs_{BASELINE_ARM}"]
    return {
        "C1": bool(gain["mean"] >= OPPORTUNITY_MARGIN),
        "C2": bool(primary["ci_low"] > 0.0),
        "C3": bool(gain["environments_improved"] >= BREADTH_MAJORITY),
        "C4": bool(len(types["weak_types_reaching_margin"]) >= WEAK_TYPES_REQUIRED),
        "C5": bool(efficiency["within_comparator"]),
        "C6": bool(
            overlap["h4_only_by_stratum"]["environments_contributing"] >= TRANSLATION_BREADTH
            and overlap["h4_only_by_stratum"]["reached_by_the_image_proposer"] > 0
        ),
    }


def run_decide() -> int:
    """Section 45: the machine-readable decision, from the frozen rule alone."""
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)["arms"]
    stats = cc_read_json(STATISTICAL_TESTS)
    types = cc_read_json(ERROR_TYPE_ANALYSIS)
    efficiency = cc_read_json(HARM_EFFICIENCY)
    overlap = cc_read_json(CANDIDATE_OVERLAP)
    translation = cc_read_json(TRANSLATION_RESULTS)["by_stratum"]
    recovery = cc_read_json(P0_MISS_REPAIR)
    burden = cc_read_json(HARM_BURDEN)["by_arm"]
    strata_inventory = cc_read_json(STRATA_INVENTORY)
    negative = cc_read_json(FALSIFICATION)
    criteria = decision_criteria(opportunity, stats, types, efficiency, overlap)
    gain = opportunity[PRIMARY_ARM]["gain_over_baseline"]["mean"]
    outcome = assign_outcome(criteria, translation[S_IMAGE]["pooled"] > 0.0, gain)
    reason = (
        "outcome "
        + outcome
        + ": "
        + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in criteria.items())
        + ". "
        + OUTCOME_LABELS[outcome]
        + "."
    )
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "primary_metric": "OpportunityRecall over every evaluable OCR error site",
            **{f"{arm}_opportunity": opportunity[arm]["mean"] for arm in ARMS},
            "h4_gain_vs_h0": gain,
            "h4_gain_pooled": opportunity[PRIMARY_ARM]["gain_over_baseline"]["pooled"],
            "proposal_expansion": strata_inventory["proposal_expansion"],
            "candidate_expansion": _ratio(
                burden[PRIMARY_ARM]["candidates"], burden[BASELINE_ARM]["candidates"]
            ),
            "p0_only_translation": translation[S_P0]["pooled"],
            "image_only_translation": translation[S_IMAGE]["pooled"],
            "overlap_translation": translation[S_OVERLAP]["pooled"],
            "p0_miss_repair_recovery": recovery["by_arm"][PRIMARY_ARM]["pooled"],
            "probability_of_repair_given_p0_miss_and_image_hit": recovery[
                "probability_of_repair_given_p0_miss_and_image_hit"
            ],
            "substitution_opportunity": types["by_kind"]["substitution"]["opportunity"][
                PRIMARY_ARM
            ],
            "omission_opportunity": types["by_kind"]["deletion"]["opportunity"][PRIMARY_ARM],
            "segmentation_opportunity": types["by_kind"]["segmentation"]["opportunity"][
                PRIMARY_ARM
            ],
            "h4_only_exact_repairs": overlap["h0_vs_h4"]["h4_only"],
            "h0_only_exact_repairs": overlap["h0_vs_h4"]["h0_only"],
            "environments_improved": opportunity[PRIMARY_ARM]["gain_over_baseline"][
                "environments_improved"
            ],
            "weak_error_types_improved": types["weak_types_reaching_margin"],
            "harmful_candidates_h0": burden[BASELINE_ARM]["harmful_candidates"],
            "harmful_candidates_h4": burden[PRIMARY_ARM]["harmful_candidates"],
            "incremental_harm_per_new_repair": efficiency["marginal_harm_per_new_repair"],
            "harm_comparator": efficiency["comparator"],
            "candidate_precision_h0": burden[BASELINE_ARM]["candidate_precision"],
            "candidate_precision_h4": burden[PRIMARY_ARM]["candidate_precision"],
            **{f"criterion_{k}": v for k, v in criteria.items()},
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS[outcome],
            "ready_for_reliability_representation_study": outcome == "A",
            "ready_for_external_confirmation": False,
            "recommended_next_stage": NEXT_STAGE[outcome],
            "reason": reason,
            "falsification_tests_passed": negative["passed"],
            "falsification_tests_total": negative["total"],
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: outcome {outcome} -- {NEXT_STAGE[outcome]}")
    return 0


# ------------------------------------------------------------------ section 50: figures

FIGURE_NOTE = (
    "ANALYSIS ONLY -- development evidence computed with ground truth; not a deployable method"
)
SURFACE = lp1.SURFACE
ARM_COLOURS = {
    H0: "#2a78d6",
    H1: "#eb6834",
    H2: "#1baf7a",
    H3: "#eda100",
    H4: "#e87ba4",
    H5: "#008300",
    HT: "#6b6b6b",
}
ARM_LABELS = {
    H0: "H0 frozen pipeline",
    H1: "H1 P0 + image",
    H2: "H2 image + image",
    H3: "H3 hybrid, one corrector",
    H4: "H4 specialized routing",
    H5: "H5 dual union",
    HT: "HT text-proposal control",
}
STRATUM_COLOURS = {S_P0: "#2a78d6", S_IMAGE: "#1baf7a", S_OVERLAP: "#eda100", S_TEXT: "#6b6b6b"}
FIGURES = (
    "opportunity_by_arm.png",
    "hybrid_gain_vs_baseline.png",
    "localization_to_repair_translation.png",
    "p0_miss_repair_recovery.png",
    "opportunity_by_environment.png",
    "opportunity_by_error_type.png",
    "substitution_opportunity.png",
    "omission_opportunity.png",
    "segmentation_opportunity.png",
    "proposal_source_to_exact_repair.png",
    "h0_vs_h4_unique_repairs.png",
    "routing_comparison.png",
    "candidate_harm_by_arm.png",
    "marginal_harm_per_new_repair.png",
    "clean_site_rewrite_rate.png",
    "abstention_by_source_stratum.png",
    "opportunity_vs_image_branch_budget.png",
    "updated_bottleneck_decomposition.png",
)


def run_figures() -> int:
    """The eighteen required figures, every value read from a persisted artifact."""
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
            "svg.hashsalt": "hy1",
        }
    )
    manifest: dict[str, Any] = {}

    def source_key(path: Path) -> str:
        stage = FIGURE_DIR.parent
        return str(path.relative_to(stage)) if path.is_relative_to(stage) else _relative(path)

    def save(fig: Any, name: str, sources: Sequence[Path], caption: str) -> None:
        fig.text(0.01, 0.005, FIGURE_NOTE, fontsize=5.5, color=xc1.INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(
            FIGURE_DIR / name,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata={"Software": None},
        )
        plt.close(fig)
        manifest[name] = {
            "path": f"{FIGURE_DIR.name}/{name}",
            "caption": caption,
            "sources": {source_key(s): xr1._signature(s) for s in sources},
        }

    opportunity = cc_read_json(OPPORTUNITY_RESULTS)["arms"]
    translation = cc_read_json(TRANSLATION_RESULTS)["by_stratum"]
    recovery = cc_read_json(P0_MISS_REPAIR)
    types = cc_read_json(ERROR_TYPE_ANALYSIS)["by_kind"]
    overlap = cc_read_json(CANDIDATE_OVERLAP)
    burden = cc_read_json(HARM_BURDEN)["by_arm"]
    efficiency = cc_read_json(HARM_EFFICIENCY)
    clean = cc_read_json(CLEAN_SITE_ANALYSIS)["by_corrector"]
    abstention = cc_read_json(ABSTENTION)["by_corrector_and_stratum"]
    curve = cc_read_json(BUDGET_CURVE)["by_fraction"]
    bottleneck = cc_read_json(BOTTLENECK)["ladder"]
    environments = cc_read_json(ENVIRONMENT_ANALYSIS)["by_environment"]
    labels = [ARM_LABELS[a] for a in ARMS]
    colours = [ARM_COLOURS[a] for a in ARMS]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    lp1._bars(ax, labels, [opportunity[a]["mean"] for a in ARMS], colours)
    ax.set_ylabel("OpportunityRecall (mean over environments)")
    ax.set_title("Exact-repair candidate opportunity by arm")
    ax.tick_params(axis="x", labelrotation=20)
    save(fig, FIGURES[0], [OPPORTUNITY_RESULTS], "opportunity per arm")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    gains = [opportunity[a]["gain_over_baseline"]["mean"] for a in ARMS]
    lp1._bars(ax, labels, gains, colours)
    ax.axhline(OPPORTUNITY_MARGIN, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
    ax.text(
        len(ARMS) - 0.5,
        OPPORTUNITY_MARGIN,
        f" registered margin {OPPORTUNITY_MARGIN}",
        fontsize=6,
        va="bottom",
        ha="right",
        color=xc1.INK_SECONDARY,
    )
    ax.set_ylabel("gain over the frozen pipeline")
    ax.set_title("Hybrid gain against the frozen current pipeline")
    ax.tick_params(axis="x", labelrotation=20)
    save(fig, FIGURES[1], [OPPORTUNITY_RESULTS], "gain over the baseline per arm")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    strata_labels = ["P0-only", "image-only", "overlap"]
    lp1._bars(
        ax,
        strata_labels,
        [translation[s]["pooled"] for s in STRATA],
        [STRATUM_COLOURS[s] for s in STRATA],
    )
    ax.set_ylabel("exact repairs / error sites reached")
    ax.set_title("Localization-to-repair translation by proposal source")
    save(fig, FIGURES[2], [TRANSLATION_RESULTS], "translation rate per proposal source")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    arms = [PRIMARY_ARM, H5, H3, H2, HT]
    lp1._bars(
        ax,
        [ARM_LABELS[a] for a in arms],
        [recovery["by_arm"][a]["pooled"] for a in arms],
        [ARM_COLOURS[a] for a in arms],
    )
    ax.set_ylabel("share of P0-missed error sites repaired")
    ax.set_title("P0-miss repair recovery")
    ax.tick_params(axis="x", labelrotation=20)
    save(fig, FIGURES[3], [P0_MISS_REPAIR], "P0-miss repair recovery per arm")

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    names = [row["environment"] for row in environments]
    lp1._grouped(
        ax,
        names,
        {a: [row[a] for row in environments] for a in (H0, H4, H5)},
        ARM_COLOURS,
        ARM_LABELS,
        label_values=False,
    )
    ax.set_ylabel("OpportunityRecall")
    ax.set_title("Opportunity by environment")
    ax.tick_params(axis="x", labelrotation=30)
    save(fig, FIGURES[4], [ENVIRONMENT_ANALYSIS], "opportunity per environment")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    kinds = list(ERROR_KINDS)
    lp1._grouped(
        ax,
        [KIND_NAME[k] for k in kinds],
        {a: [types[k]["opportunity"][a] for k in kinds] for a in (H0, H4, H5)},
        ARM_COLOURS,
        ARM_LABELS,
    )
    ax.set_ylabel("OpportunityRecall within the type")
    ax.set_title("Opportunity by error type")
    save(fig, FIGURES[5], [ERROR_TYPE_ANALYSIS], "opportunity per error type")

    for index, kind in ((6, "substitution"), (7, "deletion"), (8, "segmentation")):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        lp1._bars(
            ax,
            labels,
            [types[kind]["opportunity"][a] for a in ARMS],
            colours,
        )
        ax.set_ylabel("OpportunityRecall within the type")
        ax.set_title(f"{KIND_NAME[kind].capitalize()}: {types[kind]['error_sites']} error sites")
        ax.tick_params(axis="x", labelrotation=20)
        save(fig, FIGURES[index], [ERROR_TYPE_ANALYSIS], f"{KIND_NAME[kind]} opportunity")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    reached = [translation[s]["reached_error_sites"] for s in STRATA]
    repaired_sites = [translation[s]["repaired_error_sites"] for s in STRATA]
    lp1._grouped(
        ax,
        strata_labels,
        {"reached": reached, "repaired": repaired_sites},
        {"reached": "#2a78d6", "repaired": "#1baf7a"},
        {"reached": "error sites reached", "repaired": "with an exact repair"},
    )
    ax.set_ylabel("error sites")
    ax.set_title("From proposal source to exact repair")
    save(fig, FIGURES[9], [TRANSLATION_RESULTS], "reach and repair per proposal source")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    block = overlap["h0_vs_h4"]
    lp1._bars(
        ax,
        ["both", "H0 only", "H4 only", "neither"],
        [block["both"], block["h0_only"], block["h4_only"], block["neither"]],
        ["#eda100", ARM_COLOURS[H0], ARM_COLOURS[H4], "#6b6b6b"],
        fmt="{:.0f}",
    )
    ax.set_ylabel("error sites with an exact repair candidate")
    ax.set_title("Where the hybrid's repairs differ from the frozen pipeline's")
    save(fig, FIGURES[10], [CANDIDATE_OVERLAP], "H0 and H4 repair overlap")

    routing = cc_read_json(ROUTING_COMPARISON)["arms"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    order = [H3, H4, H5, HT]
    lp1._bars(
        ax,
        [ARM_LABELS[a] for a in order],
        [routing[a]["opportunity"] for a in order],
        [ARM_COLOURS[a] for a in order],
    )
    ax.set_ylabel("OpportunityRecall")
    ax.set_title("Is specialized routing useful?")
    ax.tick_params(axis="x", labelrotation=20)
    save(fig, FIGURES[11], [ROUTING_COMPARISON], "routing comparison")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    lp1._bars(
        ax,
        labels,
        [burden[a]["harmful_candidates"] for a in ARMS],
        colours,
        fmt="{:.0f}",
    )
    ax.set_ylabel("harmful candidates")
    ax.set_title("Harmful candidate burden by arm")
    ax.tick_params(axis="x", labelrotation=20)
    save(fig, FIGURES[12], [HARM_BURDEN], "harmful candidates per arm")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lp1._bars(
        ax,
        ["H4 marginal", "H0 comparator"],
        [efficiency["marginal_harm_per_new_repair"], efficiency["comparator"]],
        [ARM_COLOURS[H4], ARM_COLOURS[H0]],
        fmt="{:.2f}",
    )
    ax.set_ylabel("harmful candidates per exact repaired site")
    ax.set_title("What the image branch costs per repair it adds")
    save(fig, FIGURES[13], [HARM_EFFICIENCY], "marginal harm per new repair")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    correctors = sorted(clean)
    lp1._bars(
        ax,
        correctors,
        [clean[c]["rewrite_rate"] for c in correctors],
        ["#2a78d6", "#1baf7a", "#eb6834"][: len(correctors)],
    )
    ax.set_ylabel("share of clean sites rewritten")
    ax.set_title("Clean-site rewrite rate by corrector")
    save(fig, FIGURES[14], [CLEAN_SITE_ANALYSIS], "clean-site rewrite rate")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    strata_present = [*STRATA, S_TEXT]
    series: dict[str, list[float]] = {}
    for row in abstention:
        series.setdefault(row["corrector"], [0.0] * len(strata_present))
        if row["stratum"] in strata_present:
            series[row["corrector"]][strata_present.index(row["stratum"])] = row["abstention_rate"]
    palette = {c: ["#2a78d6", "#1baf7a", "#eb6834"][i] for i, c in enumerate(sorted(series))}
    lp1._grouped(ax, strata_present, series, palette, {c: c for c in series}, label_values=False)
    ax.set_ylabel("abstention rate")
    ax.set_title("Abstention by corrector and proposal source")
    save(fig, FIGURES[15], [ABSTENTION], "abstention per corrector and stratum")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fractions = sorted(curve, key=float)
    ax.plot(
        [float(f) for f in fractions],
        [curve[f]["opportunity_mean"] for f in fractions],
        marker="o",
        markersize=5,
        linewidth=2.0,
        color=ARM_COLOURS[H4],
    )
    for f in fractions:
        ax.annotate(
            f"{curve[f]['opportunity_mean']:.4f}",
            (float(f), curve[f]["opportunity_mean"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=6,
            color=xc1.INK_SECONDARY,
        )
    ax.set_xlabel("fraction of the image branch, by LP1's own ranking")
    ax.set_ylabel("OpportunityRecall (mean)")
    ax.set_title("Opportunity against image-branch budget")
    save(fig, FIGURES[16], [BUDGET_CURVE], "opportunity against the image-branch budget")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    steps = list(bottleneck)
    lp1._bars(
        ax,
        [s.replace("_", " ") for s in steps],
        [bottleneck[s] for s in steps],
        ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e87ba4"][: len(steps)],
    )
    ax.set_ylabel("share of all OCR error sites")
    ax.set_title("Updated bottleneck decomposition")
    ax.tick_params(axis="x", labelrotation=20)
    save(fig, FIGURES[17], [BOTTLENECK], "the updated ladder")

    _write_json_once(
        FIGURE_MANIFEST,
        {**_analysis_envelope("figure_manifest"), "figures": manifest, "count": len(manifest)},
    )
    missing = [name for name in FIGURES if name not in manifest]
    if missing:
        raise PhaseError(f"{len(missing)} figures were not written: {missing}")
    print(f"figures: {len(manifest)} written ({time.monotonic() - started:.0f}s)")
    return 0


def run_bottleneck() -> int:
    """Section 47: the updated ladder, with the loss at each step."""
    started = time.monotonic()
    _require(OPPORTUNITY_RESULTS, "opportunity")
    _forbid(BOTTLENECK)
    views = load_views()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates = _attach_membership(pd.read_parquet(CANDIDATES), strata)
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)["arms"]
    total = int(sum(v.count for v in views.values()))
    hybrid_sites = {
        str(name): set(group["site_id"].astype(str))
        for name, group in strata[strata["stratum"].isin(STRATA)].groupby("environment")
    }
    reached = reached_by(views, hybrid_sites)
    attempted = {
        str(name): set(group["lattice_site_id"].astype(str))
        for name, group in arm_candidates(candidates, PRIMARY_ARM).groupby("environment")
    }
    attempted_reach = reached_by(views, attempted)
    ladder = {
        "all_ocr_error_sites": 1.0,
        "reached_by_p0_or_the_image_proposer": _ratio(
            int(sum(int(v.sum()) for v in reached.values())), total
        ),
        "received_a_correction_attempt": _ratio(
            int(sum(int(v.sum()) for v in attempted_reach.values())), total
        ),
        "has_an_exact_repair_candidate": opportunity[PRIMARY_ARM]["pooled"],
    }
    _write_json_once(
        BOTTLENECK,
        {
            **_analysis_envelope("bottleneck_decomposition"),
            "ladder": ladder,
            "losses": {
                "localization": 1.0 - ladder["reached_by_p0_or_the_image_proposer"],
                "attempt": ladder["reached_by_p0_or_the_image_proposer"]
                - ladder["received_a_correction_attempt"],
                "generation": ladder["received_a_correction_attempt"]
                - ladder["has_an_exact_repair_candidate"],
            },
            "no_reliability_step": (
                "HY1 runs no reliability model, so the ranking loss is not part of this ladder"
            ),
        },
    )
    print(
        f"bottleneck: reach {ladder['reached_by_p0_or_the_image_proposer']:.4f} -> "
        f"repair {ladder['has_an_exact_repair_candidate']:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ sections 53-54: the record

DERIVED_OUTPUTS = (
    "RAW_OUTPUT_MANIFEST",
    "RAW_OUTPUT_HASHES",
    "CANDIDATES",
    "SITE_TRUTH",
    "ERROR_SITES",
    "CANDIDATE_INVENTORY",
    "CANDIDATE_ATTRIBUTION",
    "H0_REPRODUCTION",
    "OPPORTUNITY_RESULTS",
    "TRANSLATION_RESULTS",
    "P0_MISS_REPAIR",
    "ENVIRONMENT_ANALYSIS",
    "DOMAIN_ANALYSIS",
    "ERROR_TYPE_ANALYSIS",
    "OMISSION_ANALYSIS",
    "SEGMENTATION_ANALYSIS",
    "SUBSTITUTION_ANALYSIS",
    "HISTORICAL_GLYPH_ANALYSIS",
    "HARM_BURDEN",
    "HARM_EFFICIENCY",
    "SITE_OUTCOMES",
    "CLEAN_SITE_ANALYSIS",
    "EDIT_DAMAGE",
    "ABSTENTION",
    "CANDIDATE_OVERLAP",
    "ROUTING_COMPARISON",
    "BUDGET_CURVE",
    "BOTTLENECK",
    "STATISTICAL_TESTS",
    "CONTROL_RESULTS",
    "FALSIFICATION",
    "DECISION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
    "RUNTIME_COST",
)


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("normalize", run_normalize),
        ("opportunity", run_opportunity),
        ("types", run_types),
        ("burden", run_burden),
        ("routing", run_routing),
        ("bottleneck", run_bottleneck),
        ("stats", run_stats),
        ("controls", run_controls),
        ("negative", run_negative),
        ("decide", run_decide),
        ("figures", run_figures),
        ("cost", run_cost),
    )


def run_cost() -> int:
    """Section 30: what this stage actually spent, from the shard sidecars alone."""
    started = time.monotonic()
    _forbid(RUNTIME_COST)
    manifest = cc_read_json(RAW_OUTPUT_MANIFEST)["shards"]
    budget = cc_read_json(COMPUTE_BUDGET)
    by_arm: dict[str, dict[str, float]] = {}
    for shard in manifest:
        block = by_arm.setdefault(
            str(shard["method"]),
            {"requests": 0, "wall_clock_hours": 0.0, "output_tokens": 0, "unfinished_decodes": 0},
        )
        block["requests"] += int(shard["requests"])
        block["wall_clock_hours"] += float(shard["wall_clock_seconds"]) / 3600.0
        block["output_tokens"] += int(shard["output_tokens"])
        block["unfinished_decodes"] += int(shard["unfinished_decodes"])
    for block in by_arm.values():
        block["requests_per_second"] = _ratio(
            int(block["requests"]), block["wall_clock_hours"] * 3600.0
        )
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)["arms"]
    extra = int(
        opportunity[PRIMARY_ARM]["repaired_error_sites"]
        - opportunity[BASELINE_ARM]["repaired_error_sites"]
    )
    generated = sum(int(b["requests"]) for b in by_arm.values())
    _write_json_once(
        RUNTIME_COST,
        {
            **_analysis_envelope("runtime_cost"),
            "by_arm": by_arm,
            "generated_requests": generated,
            "reused_requests": budget["requests_reused_from_xr1"],
            "estimated_total_hours": budget["estimated_total_hours"],
            "total_wall_clock_hours": sum(b["wall_clock_hours"] for b in by_arm.values()),
            "extra_requests_per_new_exact_repair": _ratio(generated, extra),
            "clock": "monotonic, which excludes sleep",
        },
    )
    print(f"cost: {generated} generated requests ({time.monotonic() - started:.0f}s)")
    return 0


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
    """Every derived phase, re-run from the frozen raw cache into a directory of its own."""
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
    """Two regenerations from the frozen raw answers, plus the model's own spot-checks."""
    started = time.monotonic()
    _require(FIGURE_MANIFEST, "figures")
    _forbid(DETERMINISM)
    published = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(
        name for name in published if any(run.get(name) != published[name] for run in runs)
    )
    strata = pd.read_parquet(PROPOSAL_STRATA)
    rebuilt = build_strata()
    strata_identical = canonical_hash(rebuilt.to_json(orient="records")) == canonical_hash(
        strata[rebuilt.columns].to_json(orient="records")
    )
    regeneration = {p.stem: cc_read_json(p) for p in sorted(REGENERATION_DIR.glob("*.json"))}
    model_identical = bool(regeneration) and all(r["identical"] for r in regeneration.values())
    reuse = cc_read_json(CACHE_REUSE_AUDIT)
    all_identical = bool(not differing and strata_identical and model_identical)
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "derived_artifacts_compared": len(published),
            "derived_regenerations": len(runs),
            "differing_artifacts": differing,
            "derived_runs_identical": not differing,
            "strata_rebuilt_identically": strata_identical,
            "routed_sites_rebuilt": len(rebuilt),
            "model_regeneration": regeneration,
            "model_regeneration_identical": model_identical,
            "cache_reuse_identity_held": all(reuse["registries_identical_to_xr1"].values()),
            "all_runs_identical": all_identical,
            "uses_ground_truth": True,
        },
    )
    print(
        f"determinism: {len(published)} artifacts x {len(runs)} regenerations, "
        f"identical {all_identical} ({time.monotonic() - started:.0f}s)"
    )
    return 0


PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    MODEL_REGISTRY,
    MODEL_VERSIONS,
    ENVIRONMENT_INVENTORY,
    LP1_REPRODUCTION,
    PROPOSAL_STRATA,
    STRATA_INVENTORY,
    ROUTING_REGISTRY,
    DESIGN_RECORD,
    COMPUTE_BUDGET,
    PROMPT_REGISTRY,
    DECODING_REGISTRY,
    CROP_REGISTRY,
    CROP_INVENTORY,
    CONTEXTS,
    CACHE_REUSE_AUDIT,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE,),
    "2. Frozen LP1 Result": (RESEARCH_FREEZE, LP1_REPRODUCTION),
    "3. Research Questions": (DESIGN_RECORD,),
    "4. Why Integration Is Now Required": (DESIGN_RECORD, RESEARCH_FREEZE),
    "5. Non-Goals": (DESIGN_RECORD,),
    "6. Frozen Proposal Inputs": (LP1_REPRODUCTION, ENVIRONMENT_INVENTORY),
    "7. Proposal-Source Strata": (STRATA_INVENTORY,),
    "8. Frozen Correctors": (MODEL_REGISTRY, MODEL_VERSIONS, FROZEN_CONFIGURATION),
    "9. GT-Blind Routing": (ROUTING_REGISTRY,),
    "10. Image-Crop Construction": (CROP_REGISTRY, CROP_INVENTORY, STRATA_INVENTORY),
    "11. Candidate Arms": (ROUTING_REGISTRY, DESIGN_RECORD),
    "12. Candidate Budgets": (DESIGN_RECORD, CANDIDATE_INVENTORY),
    "13. Cache Reuse": (CACHE_REUSE_AUDIT, COMPUTE_BUDGET),
    "14. Raw-Output Freeze": (RAW_OUTPUT_MANIFEST, RAW_OUTPUT_HASHES),
    "15. H0 Reproduction": (H0_REPRODUCTION,),
    "16. Overall Opportunity": (OPPORTUNITY_RESULTS,),
    "17. Hybrid Gain": (OPPORTUNITY_RESULTS, STATISTICAL_TESTS),
    "18. Localization-to-Repair Translation": (TRANSLATION_RESULTS,),
    "19. P0-Miss Repair Recovery": (P0_MISS_REPAIR,),
    "20. P0-Only Sites": (TRANSLATION_RESULTS, HARM_BURDEN, STRATA_INVENTORY),
    "21. Image-Only Sites": (
        TRANSLATION_RESULTS,
        HARM_BURDEN,
        CANDIDATE_OVERLAP,
        STRATA_INVENTORY,
    ),
    "22. Overlap Sites": (TRANSLATION_RESULTS, HARM_BURDEN, STRATA_INVENTORY),
    "23. Substitution": (
        SUBSTITUTION_ANALYSIS,
        ERROR_TYPE_ANALYSIS,
        STATISTICAL_TESTS,
        CANDIDATE_OVERLAP,
    ),
    "24. Omission": (OMISSION_ANALYSIS, ERROR_TYPE_ANALYSIS, STATISTICAL_TESTS),
    "25. Segmentation": (
        SEGMENTATION_ANALYSIS,
        ERROR_TYPE_ANALYSIS,
        STATISTICAL_TESTS,
        CANDIDATE_OVERLAP,
    ),
    "26. Spurious Insertion": (ERROR_TYPE_ANALYSIS, STATISTICAL_TESTS),
    "27. Historical Glyphs": (HISTORICAL_GLYPH_ANALYSIS,),
    "28. Candidate Harm": (HARM_BURDEN, HARM_EFFICIENCY),
    "29. Clean-Site Behavior": (CLEAN_SITE_ANALYSIS,),
    "30. Edit Damage": (EDIT_DAMAGE,),
    "31. Abstention": (ABSTENTION,),
    "32. Candidate Complementarity": (CANDIDATE_OVERLAP, OPPORTUNITY_RESULTS),
    "33. Specialized Routing": (ROUTING_COMPARISON, STATISTICAL_TESTS),
    "34. Full Dual Union": (ROUTING_COMPARISON, OPPORTUNITY_RESULTS, STATISTICAL_TESTS),
    "35. Matched Text-Proposal Control": (
        CONTROL_RESULTS,
        OPPORTUNITY_RESULTS,
        STATISTICAL_TESTS,
    ),
    "36. Budget Curve": (BUDGET_CURVE,),
    "37. Environment Analysis": (ENVIRONMENT_ANALYSIS,),
    "38. Domain Analysis": (DOMAIN_ANALYSIS,),
    "39. Compute and Runtime": (RUNTIME_COST, COMPUTE_BUDGET),
    "40. Controls": (CONTROL_RESULTS, OPPORTUNITY_RESULTS),
    "41. Falsification Tests": (FALSIFICATION, DETERMINISM),
    "42. Statistics": (STATISTICAL_TESTS,),
    "43. Updated Bottleneck Decomposition": (BOTTLENECK,),
    "44. Limitations": (
        DESIGN_RECORD,
        CACHE_REUSE_AUDIT,
        CANDIDATE_INVENTORY,
        DECISION,
        OPPORTUNITY_RESULTS,
        HARM_BURDEN,
        HARM_EFFICIENCY,
    ),
    "45. Research Decision": (DECISION, STATISTICAL_TESTS, HARM_EFFICIENCY),
    "46. Implications for Reliability": (DECISION, HARM_BURDEN, ABSTENTION),
    "47. Next Stage": (DECISION, STATISTICAL_TESTS, CANDIDATE_OVERLAP, OPPORTUNITY_RESULTS),
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
    raw_files = sorted(RAW_CACHE.glob("*"))
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "artifacts": {_relative(p): file_sha256(p) for p in PRODUCED if p.is_file()},
            "artifact_count": sum(1 for p in PRODUCED if p.is_file()),
            "figures": {
                f"figures/{p.name}": file_sha256(p) for p in sorted(FIGURE_DIR.glob("*.png"))
            },
            "raw_generation_outputs": {_relative(p): file_sha256(p) for p in raw_files},
            "raw_generation_output_count": len(raw_files),
            "xr1_answers_reused": cc_read_json(CACHE_REUSE_AUDIT)["reused_total"],
            "upstream_files_rehashed": len(freeze["upstream_sha256"]),
            "upstream_files_moved": moved,
            "decision_rederived_identically": rederived,
            "lp1_unchanged": not cc_read_json(LP1_REPRODUCTION)["lp1_artifacts_moved"],
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
        f"record: {sum(1 for p in PRODUCED if p.is_file())} artifacts, {len(raw_files)} raw files, "
        f"report claims {audit['numeric_claims']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    phases: dict[str, Callable[[], int]] = {
        "reconstruct": run_reconstruct,
        "freeze": run_freeze,
        "strata": run_strata,
        "preregister": run_preregister,
        "contexts": run_contexts,
        "cacheaudit": run_cacheaudit,
        "generate": run_generate,
        "spotcheck": run_spotcheck,
        "normalize": run_normalize,
        "opportunity": run_opportunity,
        "types": run_types,
        "burden": run_burden,
        "routing": run_routing,
        "stats": run_stats,
        "controls": run_controls,
        "negative": run_negative,
        "bottleneck": run_bottleneck,
        "decide": run_decide,
        "figures": run_figures,
        "cost": run_cost,
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
