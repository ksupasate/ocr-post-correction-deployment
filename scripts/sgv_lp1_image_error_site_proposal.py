#!/usr/bin/env python3
"""SGV-LP1: can page-image evidence recover OCR error locations the frozen GT-blind proposer misses?

SGV-XR1 closed with outcome C. Image-conditioned generation beat the same model without the image
at GT-blind sites, but the frozen OCR-only proposer P0 reached only about two thirds of the real
OCR errors, so the image generator was never offered the rest. This stage moves the image one step
upstream and asks a localization question only:

    can Qwen3-VL-4B, reading every OCR line of the sampled pages, point at error locations -- at a
    budget matched to P0's -- better than P0, and better than the same weights without the image?

**1. No proposal comes from P0's sites.** The image proposer scans every OCR line of XR1's 57
content-blind documents. It marks positions on a GT-blind lattice of the frozen anchor forms
(one token, the gap after a token, an adjacent token pair), each with a stable id.

**2. The model only points.** It returns lattice ids and a ranking score. It never writes a
correction, and any text it emits anyway is recorded as protocol leakage and never used.

**3. Ground truth enters last.** Every raw answer is hashed, parsed and frozen, and the matched-
budget selection is fixed, before the first label is computed. A proposal reaches an OCR error
exactly when the frozen AlignmentIndex / SGV-CG1 site links say it does.

    --reconstruct   section 0: CG1, XC1, GEN1 and XR1 re-read from their own artifacts
    --freeze        upstream configuration, model versions, GT-blind environment inventory
    --lattice       P0 re-derived, the GT-blind site lattice and the scan population
    --preregister   arms, budgets, criteria, outcome rules, prompt/decoding/image registries
    --contexts      line prompts, line crops and the image-shuffle donors
    --envcheck      the frozen checkpoint regenerates XR1's spot-check under this OS
    --generate      GT-blind inference into an immutable per-shard cache (LP1_ARMS=<arms>)
    --spotcheck     each arm regenerates its first batches from the frozen inputs
    --parse         raw answers -> validated lattice proposals and the matched-budget selection
    --link          ground truth enters: error sites, links, lattice ceiling, P0 reproduction
    --localize      recall, precision, burden, miss recovery, error types, union, curves
    --controls      image shuffle and random ranking
    --stats         document-clustered paired bootstrap, Holm within each frozen family
    --negative      the falsification suite
    --figures       every figure from persisted artifacts
    --decide        the frozen outcome rule
    --determinism   every derived phase twice from the frozen raw outputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / MECHANISM ONLY. `ready_for_external_confirmation` is false by construction. The SGV1
CORD confirmatory reserve stays LOCKED. SGV13, SGV14, SGV15, SGV15b, SGV-DT1, SGV-CG1, SGV-XC1,
SGV-GEN1 and SGV-XR1 artifacts are read and hash-verified; none is modified.
"""

from __future__ import annotations

import argparse
import ast
import functools
import json
import os
import platform
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_gen1_error_conditioned_generation as gen1
import sgv_xr1_image_candidate_reliability as xr1
from ocr_risk.io.hashing import canonical_hash, file_sha256

xc1 = gen1.xc1
s14 = gen1.s14
s15 = gen1.s15
cg1 = gen1.cg1
dt1 = gen1.dt1

REPO = gen1.REPO
OUT = REPO / "results/generated/sgv_lp1_image_error_site_proposal"
CACHE = OUT / "cache"
# Raw answers live apart from every derived table, so a GT-blind phase never reads a label beside
# them.
RAW_CACHE = OUT / "raw_generation_outputs"
CROP_DIR = CACHE / "line_crops"
REGENERATION_DIR = CACHE / "regeneration"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"
MODEL_REGISTRY = OUT / "model_registry.json"
MODEL_VERSIONS = OUT / "model_versions.json"
SAMPLING_REGISTRY = OUT / "sampling_registry.json"
SITE_LATTICE_REGISTRY = OUT / "site_lattice_registry.json"
SITE_LATTICE_INVENTORY = OUT / "site_lattice_inventory.parquet"
P0_PROPOSALS = OUT / "p0_proposals.parquet"
SCAN_POPULATION = OUT / "scan_population.parquet"
DESIGN_RECORD = OUT / "design_record.json"
COMPUTE_BUDGET = OUT / "compute_budget.json"
PROMPT_REGISTRY = OUT / "prompt_registry.json"
DECODING_REGISTRY = OUT / "decoding_registry.json"
IMAGE_PREPROCESSING_REGISTRY = OUT / "image_preprocessing_registry.json"
CONTROL_CONTEXTS = OUT / "control_contexts.parquet"
DECODING_ENVIRONMENT_CHECK = OUT / "decoding_environment_check.json"

RAW_OUTPUT_MANIFEST = OUT / "raw_output_manifest.json"
RAW_OUTPUT_HASHES = OUT / "raw_output_hashes.json"
PARSED_PROPOSALS = OUT / "parsed_proposals.parquet"
PROPOSAL_COMPLIANCE = OUT / "proposal_compliance.json"
# Everything below is written with ground truth.
ERROR_SITES = OUT / "error_site_table.parquet"
PROPOSAL_ERROR_LINKS = OUT / "proposal_error_links.parquet"
LATTICE_CEILING = OUT / "lattice_ceiling.json"
NATURAL_RESULTS = OUT / "natural_localization_results.json"
MATCHED_BUDGET_RESULTS = OUT / "matched_budget_results.json"
RECALL_BUDGET_CURVE = OUT / "recall_budget_curve.json"
P0_MISS_RECOVERY = OUT / "p0_miss_recovery.json"
ERROR_TYPE_ANALYSIS = OUT / "error_type_analysis.json"
OMISSION_ANALYSIS = OUT / "omission_analysis.json"
SEGMENTATION_ANALYSIS = OUT / "segmentation_analysis.json"
HISTORICAL_GLYPH_ANALYSIS = OUT / "historical_glyph_analysis.json"
PROPOSAL_PRECISION = OUT / "proposal_precision.json"
PROPOSAL_BURDEN = OUT / "proposal_burden.json"
PROPOSAL_OVERLAP = OUT / "proposal_overlap.json"
UNION_LOCALIZATION = OUT / "union_localization.json"
ENVIRONMENT_ANALYSIS = OUT / "environment_analysis.json"
DOMAIN_ANALYSIS = OUT / "domain_analysis.json"
BOTTLENECK = OUT / "bottleneck_decomposition.json"
RUNTIME_COST = OUT / "runtime_cost.json"
IMAGE_SHUFFLE = OUT / "image_shuffle_control.json"
RANDOM_RANKING = OUT / "random_ranking_control.json"
CONTROL_RESULTS = OUT / "control_results.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"
DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"

REPORT = REPO / "docs/sgv_lp1/image_error_site_proposal.md"

SCHEMA_VERSION = 1
STAGE = "sgv_lp1_image_error_site_proposal"
HYPOTHESIS = "SGV-LP1-D1"
STAGE_KIND = "DEVELOPMENT / MECHANISM"

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

# Names a request builder may never read. GEN1's list, plus every column LP1's evaluation adds.
FORBIDDEN_INPUT_COLUMNS = (
    *gen1.FORBIDDEN_INPUT_COLUMNS,
    "subtype",
    "error_type",
    "d_before",
    "site_class",
    "labelable",
    "link_kind",
    "covered_by_p0",
    "reached",
)


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_lp1-{artifact}-v{SCHEMA_VERSION}",
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
    return gen1.environment_specs()


def _spec(name: str) -> dict[str, Any]:
    return gen1._spec(name)


def _labels_exist() -> bool:
    """True once any LP1 raw answer or label exists: the design phases refuse to run after it."""
    return any(RAW_CACHE.glob("*.parquet")) or _ground_truth_attached()


def _ground_truth_attached() -> bool:
    """Whether --link has run. An existence check only: it reads no label, and it lets every
    GT-blind phase refuse to run after ground truth, without that phase naming a label table."""
    return ERROR_SITES.exists() or PROPOSAL_ERROR_LINKS.exists()


# ------------------------------------------------------------------ section 0: reconstruct
#
# What the brief says the upstream stages found, confirmed from their own artifacts at four decimal
# places. A disagreement stops the stage before anything is designed.

XR1_EXPECTED: dict[str, Any] = {
    "status": "COMPLETE",
    "outcome": "C",
    "judged_arm": "g3_union_g0_g2",
    "site_proposal_recall_sample": 0.6468,
    "site_proposal_recall_population": 0.6491,
    "baseline_opportunity": 0.0590,
    "text_generator_opportunity": 0.0304,
    "image_generator_opportunity": 0.0692,
    "union_opportunity": 0.1228,
    "visual_translation_gain": 0.0388,
    "gain_vs_current": 0.0102,
    "zero_shot_feature_compatible": False,
    "falsification_tests_passed": 15,
    "falsification_tests_total": 15,
    "ready_for_external_confirmation": False,
    "confirmatory_reserve_consumed": False,
    "recommended_next_stage": "NEXT: IMAGE-AWARE ERROR-SITE PROPOSAL / LOCALIZATION",
}
XR1_IMAGE_HEADROOM = 0.0650
RESERVE_DECISIONS = (*xr1.RESERVE_DECISIONS, xr1.DECISION)
UPSTREAM_SCRIPTS = (
    *xr1.UPSTREAM_SCRIPTS,
    "scripts/sgv_xr1_image_candidate_reliability.py",
)
# The shared code LP1 calls into, frozen by hash like every upstream artifact.
UPSTREAM_MODULES = (
    "src/ocr_risk/discovery/enumerator.py",
    "src/ocr_risk/discovery/views.py",
    "src/ocr_risk/experiments/cgv3_confirmatory.py",
    "src/ocr_risk/experiments/cgv3_study.py",
    "src/ocr_risk/edits/sites.py",
)


def _xr1_files() -> list[Path]:
    """Every XR1 scientific file LP1 may read: artifacts, raw shards, spot-checks, figures."""
    files = [p for p in sorted(xr1.OUT.glob("*")) if p.is_file()]
    files += sorted(p for p in xr1.RAW_CACHE.glob("*") if p.is_file())
    files += sorted(xr1.REGENERATION_DIR.glob("*.json"))
    files += sorted(xr1.FIGURE_DIR.glob("*.png"))
    return files


def _upstream_files() -> list[Path]:
    """XR1's own frozen upstream set, XR1 itself, and the shared modules LP1 calls."""
    files = list(xr1._upstream_files()) + _xr1_files()
    files += [REPO / script for script in UPSTREAM_SCRIPTS]
    files += [xr1.REPORT] + [REPO / module for module in UPSTREAM_MODULES]
    return sorted(set(files))


def run_reconstruct() -> int:
    """Section 0: every upstream state the brief names, re-read from its own artifacts."""
    started = time.monotonic()
    _forbid(RESEARCH_FREEZE)
    _check = xr1._check
    checks: list[dict[str, Any]] = []
    decisions = {
        "sgv_cg1": cc_read_json(cg1.OUT / "research_decision.json"),
        "sgv_xc1": cc_read_json(xc1.OUT / "research_decision.json"),
        "sgv_gen1": cc_read_json(gen1.OUT / "research_decision.json"),
        "sgv_xr1": cc_read_json(xr1.DECISION),
    }
    for stage, expected in xr1.UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))
    for key, value in XR1_EXPECTED.items():
        checks.append(_check(f"sgv_xr1.{key}", decisions["sgv_xr1"].get(key), value))

    # GEN1's qualitative pattern, each as the number it rests on: the image helps most, capacity
    # less, wider text context little, image with the stronger model is the best known-site cell.
    exact = cc_read_json(gen1.OUT / "exact_generation_results.json")["cells"]
    checks.append(
        _check(
            "sgv_gen1.matched_no_image_exact_generation",
            exact[xr1.GEN1_MATCHED_CELL]["exact_generation"],
            0.0940,
        )
    )
    gen1_decision = decisions["sgv_gen1"]
    checks.append(
        _check(
            "sgv_gen1.visual_gain_exceeds_capacity_gain",
            bool(gen1_decision["visual_gain"] > gen1_decision["capacity_gain"]),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_gen1.context_gain_is_small",
            bool(abs(gen1_decision["context_gain"]) < 0.05),
            True,
        )
    )
    checks.append(
        _check(
            "sgv_gen1.best_cell_is_image_conditioned",
            gen1_decision["best_gen1_cell"],
            xr1.GEN1_BEST_CELL,
        )
    )
    # XR1's qualitative findings.
    oracle = cc_read_json(xr1.PROPOSAL_MISS_ORACLE)
    checks.append(
        _check(
            "sgv_xr1.image_localization_headroom",
            oracle["by_condition"]["e3_line_image"]["mean_localization_headroom"],
            XR1_IMAGE_HEADROOM,
        )
    )
    shuffle = cc_read_json(xr1.IMAGE_SHUFFLE)
    checks.append(
        _check(
            "sgv_xr1.image_beats_text_on_control_subsample",
            bool(shuffle["exact_top1_rate"][xr1.G2] > shuffle["exact_top1_rate"][xr1.G1]),
            True,
        )
    )
    xr1_negative = cc_read_json(xr1.FALSIFICATION)
    checks.append(
        _check(
            "sgv_xr1.all_falsification_tests_pass",
            bool(all(t["passed"] for t in xr1_negative["tests"])),
            True,
        )
    )

    # Reports complete, traceability clean, determinism clean, decisions re-derived. Each report
    # is re-audited here by GEN1's auditor under its own section map: "traceability clean" is a
    # measurement made now, not a field copied from the stage's own record.
    audits: dict[str, dict[str, int]] = {}
    for stage, module in (("sgv_cg1", cg1), ("sgv_xc1", xc1), ("sgv_gen1", gen1), ("sgv_xr1", xr1)):
        record = gen1.audit_report(module.REPORT, module.REPORT_SECTIONS)
        audits[stage] = {
            "numeric_claims": int(record["numeric_claims"]),
            "untraceable_numeric_claims": int(record["untraceable_numeric_claims"]),
        }
        checks.append(
            _check(f"{stage}.untraceable_numeric_claims", record["untraceable_numeric_claims"], 0.0)
        )
        determinism = cc_read_json(module.OUT / "determinism.json")
        identical = xr1._find(determinism, "all_runs_identical")
        if identical is None:
            identical = xr1._find(determinism, "identical")
        checks.append(_check(f"{stage}.determinism_identical", bool(identical), True))
    for stage, module in (("sgv_gen1", gen1), ("sgv_xr1", xr1)):
        provenance = cc_read_json(module.OUT / "provenance.json")
        checks.append(
            _check(
                f"{stage}.decision_rederived_identically",
                bool(all(provenance["decision_rederived_identically"])),
                True,
            )
        )
    # XR1's own record, re-verified file by file against the hashes it issued.
    xr1_provenance = cc_read_json(xr1.PROVENANCE)
    for group in ("artifacts", "raw_generation_outputs", "figures"):
        table = xr1_provenance[group]
        moved = [k for k, sha in table.items() if file_sha256(REPO / k) != sha]
        checks.append(_check(f"sgv_xr1.{group}_rehash_unchanged", not moved, True))
    checks.append(
        _check(
            "sgv_xr1.upstream_files_moved_at_record",
            not xr1_provenance["upstream_files_moved"],
            True,
        )
    )
    checks.append(
        _check(
            "sgv_xr1.report_rehash_unchanged",
            file_sha256(REPO / xr1_provenance["report"]["path"])
            == xr1_provenance["report"]["sha256"],
            True,
        )
    )
    for path in RESERVE_DECISIONS:
        consumed = cc_read_json(path).get("confirmatory_reserve_consumed")
        checks.append(_check(f"reserve_locked:{_relative(path)}", consumed, False))

    failures = [c for c in checks if not c["agrees"]]
    if failures:
        raise PhaseError(f"{len(failures)} upstream checks disagree: {failures[:4]}")
    files = _upstream_files()
    missing = [_relative(p) for p in files if not p.is_file()]
    if missing:
        raise PhaseError(f"{len(missing)} upstream files are missing, e.g. {missing[:3]}")
    hashes = {_relative(p): file_sha256(p) for p in files}
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "checks": checks,
            "checks_agreeing": len(checks) - len(failures),
            "checks_total": len(checks),
            "report_audits": audits,
            "upstream_decisions": {
                stage: {
                    key: record.get(key)
                    for key in (
                        "outcome",
                        "outcome_label",
                        "status",
                        "recommended_next_stage",
                        "ready_for_external_confirmation",
                        "confirmatory_reserve_consumed",
                    )
                }
                for stage, record in decisions.items()
            },
            "repository_state": xr1._git_state(),
            "xr1_script_sha256": file_sha256(
                REPO / "scripts/sgv_xr1_image_candidate_reliability.py"
            ),
            "upstream_file_count": len(hashes),
            "upstream_sha256": hashes,
            "note": (
                "SGV-XC1, SGV-GEN1 and SGV-XR1 are not committed; no commit was authorized. They "
                "are frozen by content hash here and must re-hash unchanged at --record. LP1 "
                "imports XR1 and GEN1 helpers; both scripts are in this hash set."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"reconstruct: {len(checks)} of {len(checks)} checks agree; {len(hashes)} files hashed")
    return 0


# ------------------------------------------------------------------ the frozen configuration

GEN1_MODEL_VERSIONS = gen1.OUT / "model_versions.json"
GEN1_PLATFORM = "macOS-26.6.2-arm64-arm-64bit"


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "pillow", "numpy", "pandas"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_freeze() -> int:
    """The upstream configuration held fixed, the checkpoint re-hashed, and a GT-blind inventory."""
    from ocr_risk.config.models import AlignmentConfig, SiteConfig
    from ocr_risk.discovery.enumerator import DiscoveryRules
    from ocr_risk.schemas.enums import AnchorKind

    started = time.monotonic()
    _require(RESEARCH_FREEZE, "reconstruct")
    for path in (FROZEN_CONFIGURATION, MODEL_REGISTRY, MODEL_VERSIONS):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an LP1 answer or label exists; the configuration precedes both")
    frozen_kinds = {k.value for k in AnchorKind}
    lattice_kinds = {TOKEN, GAP, TOKEN_PAIR}
    if not lattice_kinds <= frozen_kinds or frozen_kinds - lattice_kinds != set(
        EXCLUDED_ANCHOR_KINDS
    ):
        raise PhaseError("the frozen AnchorKind vocabulary differs from the lattice's forms")
    rules = DiscoveryRules()
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "environments": [
                {k: spec[k] for k in ("environment", "corpus", "base_engine", "engine_id")}
                for spec in environment_specs()
            ],
            "p0": {
                "definition": (
                    "SGV14's OCR-only discovery pass: "
                    "ocr_risk.discovery.enumerator.enumerate_sites "
                    "under the frozen pipeline's fold resources and DiscoveryRules(); read from "
                    "SGV-XC1's frozen-localization request cache and re-derived here"
                ),
                "rules": {
                    field: getattr(rules, field)
                    for field in sorted(getattr(rules, "__dataclass_fields__", {}))
                    if isinstance(getattr(rules, field), (int, float, str, bool))
                },
                "request_cache": _relative(xc1.CACHE),
            },
            "anchor_kinds": [k.value for k in AnchorKind],
            "anchor_kinds_in_lattice": [TOKEN, GAP, TOKEN_PAIR],
            "anchor_kinds_excluded": EXCLUDED_ANCHOR_KINDS,
            "alignment_config": AlignmentConfig().model_dump(mode="json"),
            "site_config": SiteConfig().model_dump(mode="json"),
            "site_link_definition": (
                "SGV-CG1 _site_links: an anchor span maps through AlignmentIndex to its merged "
                "correction site; a GAP anchor also covers the OCR-empty alignment components "
                "between its two flanks. The only definition of 'a proposal reaches an error'."
            ),
            "error_site_definition": (
                "SGV-CG1 section 4 via SGV-XC1 _alignment_population: an evaluable alignment site "
                "whose OCR text differs from its ground truth, after the adjacent-component merge"
            ),
            "document_sample": {
                "source": _relative(xr1.PROPOSAL_SAMPLE),
                "sha256": file_sha256(xr1.PROPOSAL_SAMPLE),
                "rule": cc_read_json(xr1.DESIGN_RECORD)["sample"]["rule"],
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )

    # The checkpoint, re-hashed file by file against what GEN1 recorded when it was frozen.
    spec = gen1.MODEL_SPECS[MODEL]
    snapshot = gen1._snapshot(MODEL)
    if snapshot is None:
        raise PhaseError(f"{MODEL}: no local snapshot")
    recorded = cc_read_json(GEN1_MODEL_VERSIONS)
    frozen_files = recorded["models"][MODEL]["checkpoint_files"]
    observed = {name: file_sha256(snapshot / name) for name in sorted(frozen_files)}
    differing = sorted(name for name, sha in frozen_files.items() if observed.get(name) != sha)
    if differing:
        raise PhaseError(f"{MODEL}: checkpoint files differ from GEN1's record: {differing}")
    packages = _package_versions()
    for name in ("torch", "transformers"):
        if packages[name] != recorded["packages"][name]:
            raise PhaseError(
                f"{name} {packages[name]} differs from GEN1's {recorded['packages'][name]}"
            )
    _write_json_once(
        MODEL_VERSIONS,
        {
            **_envelope("model_versions"),
            "model": MODEL,
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "snapshot": str(snapshot),
            "checkpoint_files": observed,
            "checkpoint_identical_to_gen1_record": True,
            "card_license": recorded["models"][MODEL].get("card_license"),
            "inference_dtype": gen1.MODEL_DTYPE,
            "packages": packages,
            "packages_identical_to_gen1_record": {
                name: packages[name] == recorded["packages"].get(name)
                for name in ("torch", "transformers")
            },
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gen1_platform": recorded["hardware"]["platform"],
            "operating_system_changed_since_gen1": platform.platform() != GEN1_PLATFORM,
            "note": (
                "the operating system was updated between SGV-XR1 and SGV-LP1. Weights and "
                "libraries are unchanged; --envcheck measures whether decoding is too. Every LP1 "
                "arm runs under this one environment, so no LP1 comparison crosses it."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        MODEL_REGISTRY,
        {
            **_envelope("model_registry"),
            "model": MODEL,
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "kind": spec["kind"],
            "batch_size": int(spec["batch_size"]),
            "same_checkpoint_as": ["SGV-GEN1 m5", "SGV-XR1 g1/g2"],
            "model_search": False,
            "fine_tuned_here": False,
            "larger_model_introduced": False,
            "arms": {
                L1: "the checkpoint with the line's OCR tokens and no image",
                L2: "the checkpoint with the same text and the line's page crop",
                C3: "the checkpoint with the same text and another document's line crop",
            },
            "deployable": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"freeze: configuration, {len(observed)} checkpoint files re-hashed, model registry")
    return 0


# ------------------------------------------------------------------ the GT-blind site lattice
#
# The lattice is every place the frozen representation can anchor a repair on these pages: each
# OCR token, and for each pair of stream-adjacent tokens on the same OCR line, the gap between them
# and the pair itself. These are exactly the anchors P0's rules can fire on (its line-boundary rule
# is off), so P0's proposals are a subset of the lattice and every arm proposes from one space.

TOKEN, GAP, TOKEN_PAIR = "token", "gap", "token_pair"
LATTICE_VERSION = "lp1-lattice-v1"
# The one frozen anchor kind the lattice leaves out, and why. It is declared in the enum and used
# nowhere else in the repository: no discovery rule emits it, SGV-CG1's links and SGV14's labels
# never read it, so it has no frozen representation to reach an error through.
EXCLUDED_ANCHOR_KINDS = {
    "token_interior": (
        "declared in AnchorKind but produced, linked and labelled nowhere upstream; P0 never "
        "emits it; a within-token offset would be a coordinate the model had to invent; and the "
        "span it sits in is already a token site"
    )
}


def anchor_ref(kind: str, *span_ids: str) -> str:
    """The frozen anchor reference, as OcrPageView.anchor_ref writes it (asserted at --lattice)."""
    return "\0".join((kind, *span_ids))


def anchor_key(refs: pd.Series) -> pd.Series:
    """A join key for anchor references. pandas hashes object strings as C strings, which stop at
    the NUL the frozen reference uses as its separator, so every token anchor would collide; `|`
    never occurs in a span id, so this key is exact."""
    return refs.astype(str).str.replace("\0", "|", regex=False)


def page_lines(page: Any) -> dict[int, list[str]]:
    """Each OCR line's spans in stream order."""
    lines: dict[int, list[str]] = {}
    for span_id in page.order:
        lines.setdefault(int(page.span_line[span_id]), []).append(span_id)
    return lines


def local_token(index: int) -> str:
    return f"T{index:02d}"


def local_gap(index: int) -> str:
    return f"G{index:02d}"


def local_pair(index: int) -> str:
    return f"{local_token(index)}+{local_token(index + 1)}"


def page_lattice(page: Any, environment: str, document_id: str, engine_id: str) -> pd.DataFrame:
    """Every lattice site on one page, with its line and the id the model sees for it."""
    line_of = {s: int(page.span_line[s]) for s in page.order}
    position = {s: i for i, s in enumerate(page.order)}
    index_in_line: dict[str, int] = {}
    for spans in page_lines(page).values():
        for i, span_id in enumerate(spans):
            index_in_line[span_id] = i
    rows: list[dict[str, Any]] = []
    for span_id in page.order:
        rows.append(
            {
                "anchor_kind": TOKEN,
                "anchor_ref": anchor_ref(TOKEN, span_id),
                "char_start": int(page.span_start[span_id]),
                "char_end": int(page.span_end[span_id]),
                "line_index": line_of[span_id],
                "local_id": local_token(index_in_line[span_id]),
            }
        )
    for left, right in zip(page.order[:-1], page.order[1:], strict=True):
        if line_of[left] != line_of[right]:
            continue
        if position[right] != position[left] + 1:
            continue
        i = index_in_line[left]
        rows.append(
            {
                "anchor_kind": GAP,
                "anchor_ref": anchor_ref(GAP, left, right),
                "char_start": int(page.span_end[left]),
                "char_end": int(page.span_start[right]),
                "line_index": line_of[left],
                "local_id": local_gap(i),
            }
        )
        rows.append(
            {
                "anchor_kind": TOKEN_PAIR,
                "anchor_ref": anchor_ref(TOKEN_PAIR, left, right),
                "char_start": int(page.span_start[left]),
                "char_end": int(page.span_end[right]),
                "line_index": line_of[left],
                "local_id": local_pair(i),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["char_start", "anchor_kind", "anchor_ref"], kind="stable"
    )
    frame = frame.reset_index(drop=True)
    frame.insert(0, "environment", environment)
    frame.insert(
        1,
        "site_id",
        [f"{environment}|{document_id}:{engine_id}:lsite:{i:05d}" for i in range(len(frame))],
    )
    frame.insert(2, "document_id", document_id)
    frame.insert(3, "engine_id", engine_id)
    frame["region"] = [
        page.stream[s:e] for s, e in zip(frame["char_start"], frame["char_end"], strict=True)
    ]
    frame["line_uid"] = [f"{environment}|{document_id}|L{i:04d}" for i in frame["line_index"]]
    frame["anchor_key"] = anchor_key(frame["anchor_ref"])
    return frame


def _quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def render_line(page: Any, spans: Sequence[str], gaps: set[int]) -> str:
    """The OCR line as the model reads it: `T00="The" G00 T01="qu1ck" ...`, JSON-quoted.

    Gnn appears between Tnn and the next token exactly when that gap is a lattice site, so every
    id the model sees names a site, and an omission has a visible place to be marked.
    """
    parts: list[str] = []
    for i, span in enumerate(spans):
        if i and (i - 1) in gaps:
            parts.append(local_gap(i - 1))
        parts.append(
            f"{local_token(i)}={_quoted(page.stream[page.span_start[span] : page.span_end[span]])}"
        )
    return " ".join(parts)


def line_box(page: Any, spans: Sequence[str]) -> tuple[int, int, int, int] | None:
    """The line's OCR boxes, padded by GEN1's crop rule and clamped to the page. No GT sizes it."""
    boxes = [page.span_box[s] for s in spans if page.span_box.get(s) is not None]
    if not boxes:
        return None
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad = max(gen1.CROP_MIN_PAD, gen1.CROP_PAD_FRACTION * (y1 - y0))
    left, top = max(0, int(np.floor(x0 - pad))), max(0, int(np.floor(y0 - pad)))
    right = min(int(page.width), int(np.ceil(x1 + pad)))
    bottom = min(int(page.height), int(np.ceil(y1 + pad)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def scan_lines(
    page: Any,
    lattice: pd.DataFrame,
    environment: str,
    document_id: str,
    engine_id: str,
    document_rank: int,
) -> pd.DataFrame:
    """One row per OCR line: the model's text, its valid ids and the lattice sites they name."""
    rows: list[dict[str, Any]] = []
    by_line = {int(k): g for k, g in lattice.groupby("line_index", sort=True)}
    for line_index, spans in sorted(page_lines(page).items()):
        sites = by_line[line_index].sort_values(["char_start", "anchor_kind"], kind="stable")
        box = line_box(page, spans)
        rows.append(
            {
                "environment": environment,
                "line_uid": f"{environment}|{document_id}|L{line_index:04d}",
                "document_id": document_id,
                "engine_id": engine_id,
                "document_rank": int(document_rank),
                "line_index": int(line_index),
                "tokens": len(spans),
                "lattice_sites": len(sites),
                "line_text": render_line(
                    page,
                    spans,
                    {int(i[1:]) for i in sites["local_id"].astype(str) if i.startswith("G")},
                ),
                "span_ids": list(spans),
                "local_ids": sites["local_id"].astype(str).tolist(),
                "site_ids": sites["site_id"].astype(str).tolist(),
                "crop_box": list(box) if box is not None else None,
                "page_image_path": page.image_path,
                "page_image_sha256": page.image_sha256,
            }
        )
    return pd.DataFrame(rows)


def rederive_p0(spec: dict[str, Any], documents: set[str], pipeline: Any) -> pd.DataFrame:
    """P0's sites on the sampled pages, from SGV14's own discovery call, with suspicion scores."""
    from ocr_risk.discovery.enumerator import DiscoveryRules
    from ocr_risk.experiments.cgv3_confirmatory import discovery_pass

    bundles = s14._document_bundles(spec["corpus"])
    spans, _record = s14._canonical_spans(spec, bundles)
    chosen = {pair: group for pair, group in spans.items() if pair[0] in documents}
    sites, _empty = discovery_pass(chosen, pipeline.fitted, [spec["base_engine"]], DiscoveryRules())
    sites = sites.sort_values(
        ["document_id", "engine_id", "char_start", "char_end", "site_id"], kind="stable"
    ).reset_index(drop=True)
    sites["site_id"] = spec["environment"] + "|" + sites["site_id"].astype(str)
    sites.insert(0, "environment", spec["environment"])
    return sites


def map_p0_to_lattice(
    p0: pd.DataFrame, lattice: pd.DataFrame, ranks: dict[str, int]
) -> tuple[pd.DataFrame, int, int]:
    """Each P0 site's lattice site: the same anchor on the same page, the same stream range."""
    p0 = p0.assign(
        document_id=p0["document_id"].astype(str), anchor_key=anchor_key(p0["anchor_ref"])
    )
    joined = p0.merge(
        lattice[
            ["document_id", "anchor_key", "site_id", "char_start", "char_end", "line_index"]
        ].rename(
            columns={
                "site_id": "lattice_site_id",
                "char_start": "lattice_start",
                "char_end": "lattice_end",
            }
        ),
        on=["document_id", "anchor_key"],
        how="left",
        validate="one_to_one",
    )
    outside = int(joined["lattice_site_id"].isna().sum())
    shifted = int(
        (
            (joined["char_start"] != joined["lattice_start"])
            | (joined["char_end"] != joined["lattice_end"])
        ).sum()
    )
    mapped = joined.drop(columns=["lattice_start", "lattice_end"])
    mapped["line_index"] = mapped["line_index"].fillna(-1).astype(int)
    mapped["document_rank"] = mapped["document_id"].map(ranks).astype(int)
    return mapped, outside, shifted


def run_lattice() -> int:
    """P0 re-derived and checked, the lattice, the scan population and the GT-blind inventory."""
    from ocr_risk.discovery.views import OcrPageView
    from ocr_risk.schemas.enums import AnchorKind

    started = time.monotonic()
    _require(FROZEN_CONFIGURATION, "freeze")
    outputs = (
        SITE_LATTICE_INVENTORY,
        SCAN_POPULATION,
        P0_PROPOSALS,
        SITE_LATTICE_REGISTRY,
        SAMPLING_REGISTRY,
        ENVIRONMENT_INVENTORY,
    )
    for path in outputs:
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an LP1 answer or label exists; the lattice precedes both")
    sample = pd.read_parquet(xr1.PROPOSAL_SAMPLE)
    pipeline = s14.build_pipeline()
    lattice_frames: list[pd.DataFrame] = []
    line_frames: list[pd.DataFrame] = []
    p0_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        mine = sample[sample["environment"] == name]
        ranks = mine.groupby("document_id")["document_rank"].first().astype(int).to_dict()
        documents = set(ranks)
        pages = gen1._page_index(spec)
        chosen = {pair: page for pair, page in pages.items() if pair[0] in documents}
        if {pair[0] for pair in chosen} != documents:
            raise PhaseError(f"{name}: a sampled document has no OCR page")
        env_lattice: list[pd.DataFrame] = []
        for (document_id, engine_id), page in sorted(chosen.items()):
            lattice = page_lattice(page, name, document_id, engine_id)
            env_lattice.append(lattice)
            line_frames.append(
                scan_lines(page, lattice, name, document_id, engine_id, ranks[document_id])
            )
        lattice = pd.concat(env_lattice, ignore_index=True)
        lattice_frames.append(lattice)

        # P0, re-derived by SGV14's call and required to equal SGV-XC1's frozen cache exactly.
        p0 = rederive_p0(spec, documents, pipeline)
        cached = xr1._p0_requests(name)
        cached = cached[cached["document_id"].astype(str).isin(documents)]
        key = ["site_id", "anchor_kind", "anchor_ref", "char_start", "char_end"]
        mine_key = p0[key].astype(str).sort_values(key).reset_index(drop=True)
        cache_key = cached[key].astype(str).sort_values(key).reset_index(drop=True)
        reproduced = bool(mine_key.equals(cache_key))
        if not reproduced:
            raise PhaseError(f"{name}: the re-derived P0 differs from SGV-XC1's frozen cache")
        # Every P0 anchor is a lattice site: the same anchor reference, the same stream range.
        p0, outside, shifted = map_p0_to_lattice(p0, lattice, ranks)
        if outside or shifted:
            raise PhaseError(f"{name}: {outside} P0 anchors outside the lattice, {shifted} shifted")
        p0_frames.append(p0)
        # The frozen anchor reference, written by the frozen view on a real page.
        document_id, engine_id = sorted(chosen)[0]
        view = OcrPageView.from_spans(
            tuple(
                s14._canonical_spans(spec, s14._document_bundles(spec["corpus"]))[0][
                    (document_id, engine_id)
                ]
            )
        )
        first = view.tokens[0]
        checks.append(
            {
                "environment": name,
                "p0_reproduces_xc1_cache": reproduced,
                "p0_sites": len(p0),
                "p0_outside_lattice": outside,
                "anchor_ref_matches_frozen_view": view.anchor_ref(AnchorKind.TOKEN, first)
                == anchor_ref(TOKEN, first.span_id),
            }
        )
        lines = pd.concat([f for f in line_frames if f["environment"].iloc[0] == name])
        inventory.append(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "base_engine": spec["base_engine"],
                "documents": len(documents),
                "pages": len(chosen),
                "ocr_lines": len(lines),
                "ocr_tokens": int(lines["tokens"].sum()),
                "lines_without_a_box": int(lines["crop_box"].isna().sum()),
                "page_images_present": int(
                    sum(Path(p.image_path).is_file() for p in chosen.values())
                ),
                "lattice_sites": len(lattice),
                "lattice_sites_by_kind": {
                    str(k): int(v) for k, v in lattice["anchor_kind"].value_counts().items()
                },
                "p0_proposals": len(p0),
                "p0_proposals_by_kind": {
                    str(k): int(v) for k, v in p0["anchor_kind"].value_counts().items()
                },
            }
        )
        print(
            f"  {name}: {len(lines)} lines, {len(lattice)} lattice sites, {len(p0)} P0 "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )

    lattice = pd.concat(lattice_frames, ignore_index=True)
    lines = pd.concat(line_frames, ignore_index=True)
    p0 = pd.concat(p0_frames, ignore_index=True)
    for frame, what in ((lattice, "lattice"), (lines, "scan population"), (p0, "P0 table")):
        leaked = [c for c in FORBIDDEN_INPUT_COLUMNS if c in frame.columns]
        if leaked:
            raise PhaseError(f"the {what} carries truth columns: {leaked}")
    if not all(c["anchor_ref_matches_frozen_view"] for c in checks):
        raise PhaseError("the lattice's anchor reference differs from the frozen OcrPageView's")
    _write_parquet_once(SITE_LATTICE_INVENTORY, lattice)
    _write_parquet_once(SCAN_POPULATION, lines)
    _write_parquet_once(P0_PROPOSALS, p0)
    interleaved = int(
        sum(
            max(0, row.tokens - 1) - sum(1 for i in row.local_ids if str(i).startswith("G"))
            for row in lines.itertuples(index=False)
        )
    )
    _write_json_once(
        SITE_LATTICE_REGISTRY,
        {
            **_envelope("site_lattice_registry"),
            "version": LATTICE_VERSION,
            "forms": {
                TOKEN: "one OCR span, by its stream range; the model sees Tnn",
                GAP: (
                    "the boundary between two stream-adjacent spans on the same OCR line, "
                    "(left.end, right.start); the model sees Gnn, the gap after Tnn"
                ),
                TOKEN_PAIR: (
                    "two stream-adjacent spans on the same OCR line and the slice between them; "
                    "the model names it by listing Tnn and Tnn+1 together"
                ),
            },
            "not_in_the_lattice": {
                "line_boundary_pairs": (
                    "a gap or pair across two OCR lines. The frozen boundary rule is off, so P0 "
                    "cannot propose one either; measured as a diagnostic at --link"
                ),
                "line_start_or_end_gaps": (
                    "a gap with one flank. The frozen representation has no such anchor: a GAP "
                    "names two spans, and SGV-CG1's link reads the components between them"
                ),
                "longer_spans": "three or more tokens; no frozen anchor form covers them",
                **EXCLUDED_ANCHOR_KINDS,
            },
            "same_line_rule": (
                "two spans share a line when their canonical line ids are equal and not missing, "
                "and are adjacent when consecutive in stream order: P0's own condition"
            ),
            "site_id": "{environment}|{document}:{engine}:lsite:{index}, ordered like P0's ids",
            "sites": len(lattice),
            "sites_by_kind": {
                str(k): int(v) for k, v in lattice["anchor_kind"].value_counts().items()
            },
            "lines": len(lines),
            "tokens": int(lines["tokens"].sum()),
            "same_line_token_pairs_not_stream_adjacent": interleaved,
            "p0_checks": checks,
            "p0_sites": len(p0),
            "p0_subset_of_lattice": True,
            "built_without_ground_truth": True,
            "inventory": {
                "path": _relative(SITE_LATTICE_INVENTORY),
                "sha256": file_sha256(SITE_LATTICE_INVENTORY),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        SAMPLING_REGISTRY,
        {
            **_envelope("sampling_registry"),
            "documents": (
                "SGV-XR1's content-blind sample, reused unchanged so every comparison is paired"
            ),
            "document_source": {
                "path": _relative(xr1.PROPOSAL_SAMPLE),
                "sha256": file_sha256(xr1.PROPOSAL_SAMPLE),
            },
            "document_rule": cc_read_json(xr1.DESIGN_RECORD)["sample"]["rule"],
            "line_rule": "every OCR line of every sampled page; no line is selected or excluded",
            "reads_ground_truth": False,
            "reads_p0_proposals_to_choose_lines": False,
            "by_environment": [
                {
                    "environment": row["environment"],
                    "documents": sorted(
                        {
                            str(d)
                            for d in lines[lines["environment"] == row["environment"]][
                                "document_id"
                            ]
                        }
                    ),
                    "lines": row["ocr_lines"],
                }
                for row in inventory
            ],
            "documents_total": int(sum(r["documents"] for r in inventory)),
            "lines_total": len(lines),
            "scan_population": {
                "path": _relative(SCAN_POPULATION),
                "sha256": file_sha256(SCAN_POPULATION),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        ENVIRONMENT_INVENTORY,
        {
            **_envelope("environment_inventory"),
            "environments": inventory,
            "documents": int(sum(r["documents"] for r in inventory)),
            "ocr_lines": int(sum(r["ocr_lines"] for r in inventory)),
            "ocr_tokens": int(sum(r["ocr_tokens"] for r in inventory)),
            "lattice_sites": len(lattice),
            "p0_proposals": len(p0),
            "reads_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"lattice: {len(lattice)} sites on {len(lines)} lines; P0 reproduced ({len(p0)} sites)")
    return 0


# ------------------------------------------------------------------ the pre-registered design
#
# Everything from here to the end of this block is written to `design_record.json` and the three
# registries by `--preregister`, which refuses to run once any LP1 answer or label exists.

L0 = "l0_frozen_p0"
L1 = "l1_text_proposer"
L2 = "l2_image_proposer"
L3 = "l3_union_p0_l2"
C3 = "c3_image_shuffle"
C4 = "c4_random_ranking"
MODEL_ARMS = (L1, L2, C3)
IMAGE_ARMS = (L2, C3)
ARM_LABELS = {
    L0: "L0 frozen P0",
    L1: "L1 text-only proposer",
    L2: "L2 image proposer",
    L3: "L3 P0 + L2",
    C3: "C3 shuffled image",
    C4: "C4 random lattice sites",
}

LP1_SEED = 20260920
SHUFFLE_LINES_PER_ENVIRONMENT = 40
RANDOM_DRAWS = 200
MAX_PROPOSALS_PER_LINE = 8
BUDGET_FRACTIONS = (0.25, 0.5, 1.0, 1.5, 2.0)
PRIMARY_FRACTION = 1.0

# The localization gate. Margins are fixed against XR1's measured state on these same pages: P0
# reaches 0.6468 of their OCR errors, so 0.3532 is the proposal loss an image proposer could win
# back; XR1's oracle estimate of what the image generator could add there is 0.0650.
RECALL_MARGIN = 0.10
BREADTH_ENVIRONMENT_MARGIN = 0.05
BREADTH_MAJORITY = 6
MISS_RECOVERY_TARGET = 0.25
# The frozen taxonomy calls an omission a "deletion" and a spurious insertion an "insertion".
WEAK_TYPES = ("substitution", "deletion", "segmentation")
WEAK_TYPE_NAMES = {
    "substitution": "substitution",
    "deletion": "omission",
    "segmentation": "segmentation",
}
WEAK_TYPE_MARGIN = 0.10
WEAK_TYPES_REQUIRED = 2
SHUFFLE_RETENTION_MAX = 0.5

OUTCOME_LABELS = {
    "A": "image-aware localization materially works",
    "B": "image localization is complementary but not a standalone replacement",
    "C": "image-aware localization does not materially improve P0",
    "D": "recall improves only through excessive false proposals",
}
NEXT_STAGE = {
    "A": "IMAGE-AWARE SITE PROPOSAL + IMAGE-CONDITIONED CORRECTION INTEGRATION",
    "B": "HYBRID SITE PROPOSAL + IMAGE-CONDITIONED GENERATION",
    "C": "OCR-IMAGE DISAGREEMENT / SPECIALIZED LOCALIZATION MECHANISMS (no model-size escalation)",
    "D": "SELECTIVE IMAGE-AWARE LOCALIZATION / PROPOSAL RANKING",
}

# The proposal contract, revised twice on format and counts alone -- no probe read a label.
# Revision 0 returned about one proposal per line, scores almost all 0.9 and some three-token spans:
# too few for P0's budget, a ranking left to tie-breaks, spans no frozen anchor can hold. Revision 1
# asked for a ranked list with anchored scores and one position per proposal, and no gap was ever
# proposed. Revision 2 shows each lattice gap's id inline, as the brief's own input example does.
PROMPT_REVISION = 2
SYSTEM_PROMPT = (
    "You check one line of OCR output from a scanned document page for recognition errors.\n"
    "The line is given in reading order as tokens T00, T01, T02 and so on, each with the text the "
    "OCR engine produced, and between each two tokens the id of the gap between them, G00, G01 "
    "and so on: Gnn is the gap after Tnn.\n"
    "A position is one of:\n"
    "- a token, Tnn: its text looks misrecognized;\n"
    "- a gap, Gnn: characters or a word look missing there;\n"
    "- two adjacent tokens, Tnn and the next token: one word looks wrongly split into two tokens, "
    "or two words wrongly joined.\n"
    "List the positions that could be OCR errors, most suspicious first, at most "
    f"{MAX_PROPOSALS_PER_LINE}. Include a position whenever it might be wrong, even if you are "
    "unsure, and give each a suspicion score: about 0.9 when it is almost certainly an error, "
    "about 0.5 when it is doubtful, about 0.2 when it is only slightly suspicious. Leave out "
    "positions you are confident are correct.\n"
    "Each proposal names exactly one position. Report only positions: never write a correction, "
    "an explanation or the corrected line.\n"
    'Answer with JSON only, for example {"proposals": [{"site_ids": ["T01"], "suspicion": 0.9}, '
    '{"site_ids": ["G02"], "suspicion": 0.4}, {"site_ids": ["T03", "T04"], "suspicion": 0.2}]}\n'
    'If every position looks correct, answer {"proposals": []}.'
)
USER_TEMPLATES = {
    "text": "OCR line (tokens in order):\n{text}",
    "image": "The image shows this line on the page.\nOCR line (tokens in order):\n{text}",
}
TEMPLATE_OF_ARM = {L1: "text", L2: "image", C3: "image"}
PROMPT_HISTORY = [
    {
        "revision": 0,
        "change": "positions that look wrong; free-form score; no count or shape rule",
        "probe": "24 funsd/doctr lines, L1 and L2, format and counts only",
        "observed": (
            "every answer parsed; about one proposal per line; scores almost all 0.9; some "
            "proposals listed three tokens"
        ),
        "reason_for_revision": (
            "the matched budget is P0's proposal count, about three per line, and a ranking needs "
            "scores that vary; a three-token span has no frozen anchor form"
        ),
    },
    {
        "revision": 1,
        "change": "ranked list, at most eight, anchored scores, exactly one position per proposal",
        "probe": "32 lines of funsd/doctr and sbb/tesseract_frk, L1 and L2, format and counts only",
        "observed": (
            "every answer but one parsed; about 2.4 to 2.5 proposals per line; tokens and a few "
            "pairs, but not a single gap from either arm"
        ),
        "reason_for_revision": (
            "a gap was only described by rule (Gnn follows Tnn), never shown, so omissions -- "
            "reachable only through a gap -- had no visible place to be marked"
        ),
    },
    {
        "revision": 2,
        "change": "gap ids shown inline between the tokens they separate; wording otherwise kept",
        "probe": "the same 32 lines, L1 and L2, format and counts only",
        "observed": (
            "every answer parsed; about three proposals per line; both arms now propose gaps "
            "(24 and 17 of their proposals); one invalid id"
        ),
    },
]

# Decoding: GEN1's greedy rule and batch size; the output cap is LP1's own, from the line's size.
DECODING_REVISION = 0
CAP_BASE_TOKENS = 32
CAP_TOKENS_PER_PROPOSAL = 20
MAX_NEW_TOKENS_RULE = (
    f"{CAP_BASE_TOKENS} + {CAP_TOKENS_PER_PROPOSAL} x min(lattice sites on the widest line of the "
    f"batch, {MAX_PROPOSALS_PER_LINE})"
)

# The line crop: GEN1's padding rule around the whole line's OCR boxes, rendered 96 px high with the
# width capped at 1,536 px. At GEN1's 64 x 1,024 the 10th-percentile line would render about 30 px
# high; at 96 x 1,536 about 45 px, and the probe measured no loss of throughput.
IMAGE_PREPROCESSING_REVISION = 0
LINE_CROP_HEIGHT = 96
LINE_CROP_MAX_WIDTH = 1536
CROP_LINE = "line"
CROP_NONE = "unavailable"

# Measured on the frozen checkpoint with the frozen revision-2 prompt: 16 lines per arm, batch 8,
# on this machine while another project's Python jobs held five to nine cores. The revision-2
# probe's own timing (0.018 lines per second) ran beside this stage's own tests and is not used.
MEASURED_LINES_PER_SECOND: dict[str, float] = {L1: 0.0482, L2: 0.0481, C3: 0.0481}
MEASUREMENT_CONDITIONS = (
    "16 lines per arm under the frozen revision-2 prompt, batch 8, after the model was loaded; "
    "another project's Python jobs held five to nine cores throughout, so the rate is the one "
    "this run should expect"
)


def cap_tokens(widest_sites: int) -> int:
    return CAP_BASE_TOKENS + CAP_TOKENS_PER_PROPOSAL * min(widest_sites, MAX_PROPOSALS_PER_LINE)


def order_key(*parts: str) -> str:
    """A content-blind order: the SHA-256 of the seed and the parts."""
    return xr1.order_key(str(LP1_SEED), *parts)


def assign_outcome(
    criteria: dict[str, bool], union: dict[str, bool], natural: dict[str, bool]
) -> str:
    """Exactly one outcome, in the frozen precedence A, B, D, C.

    A needs every localization criterion. B is the complementary case: P0 united with L2's matched-
    budget proposals gains the margin broadly, at no more than twice P0's proposal count. D is the
    selectivity case: L2's natural output gains the margin over P0 only by proposing more than P0.
    Anything else is C.
    """
    if all(criteria[k] for k in ("C1", "C2", "C3", "C4", "C5")):
        return "A"
    if union["U1"] and union["U2"]:
        return "B"
    if natural["N1"] and natural["N2"]:
        return "D"
    return "C"


def _compute_plan(lines_by_arm: dict[str, int]) -> dict[str, Any]:
    hours = {
        arm: round(count / MEASURED_LINES_PER_SECOND[arm] / 3600.0, 2)
        for arm, count in lines_by_arm.items()
    }
    return {
        "requests": lines_by_arm,
        "estimated_hours": hours,
        "estimated_total_hours": round(sum(hours.values()), 2),
    }


def run_preregister() -> int:
    """Section 31-36, 44-48: the design, the registries and the compute budget, before any label."""
    started = time.monotonic()
    _require(SITE_LATTICE_REGISTRY, "lattice")
    outputs = (
        DESIGN_RECORD,
        PROMPT_REGISTRY,
        DECODING_REGISTRY,
        IMAGE_PREPROCESSING_REGISTRY,
        COMPUTE_BUDGET,
    )
    for path in outputs:
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an LP1 answer or label exists; the design cannot be written after it")
    if set(MEASURED_LINES_PER_SECOND) != set(MODEL_ARMS):
        raise PhaseError("measured throughput is missing for an arm; run the probe first")
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    lines = pd.read_parquet(SCAN_POPULATION)
    budgets = {r["environment"]: int(r["p0_proposals"]) for r in inventory["environments"]}
    prompt_hash = canonical_hash({"system": SYSTEM_PROMPT, "user": USER_TEMPLATES})
    decoding_hash = canonical_hash(
        {
            "revision": DECODING_REVISION,
            "rule": [CAP_BASE_TOKENS, CAP_TOKENS_PER_PROPOSAL, MAX_PROPOSALS_PER_LINE],
            "settings": "gen1._generation_settings",
            "batch": int(gen1.MODEL_SPECS[MODEL]["batch_size"]),
            "dtype": gen1.MODEL_DTYPE,
        }
    )
    image_hash = canonical_hash(
        {
            "revision": IMAGE_PREPROCESSING_REVISION,
            "height": LINE_CROP_HEIGHT,
            "max_width": LINE_CROP_MAX_WIDTH,
            "pad": [gen1.CROP_MIN_PAD, gen1.CROP_PAD_FRACTION],
            "resample": "LANCZOS",
            "format": "PNG RGB",
        }
    )
    requests = {
        L1: len(lines),
        L2: int(lines["crop_box"].notna().sum()),
        C3: SHUFFLE_LINES_PER_ENVIRONMENT * len(budgets),
    }
    _write_json_once(
        PROMPT_REGISTRY,
        {
            **_envelope("prompt_registry"),
            "revision": PROMPT_REVISION,
            "system": SYSTEM_PROMPT,
            "user_templates": USER_TEMPLATES,
            "template_of_arm": TEMPLATE_OF_ARM,
            "matched_between_l1_and_l2": (
                "the same system prompt, OCR text, token ids, output schema, decoding and cap; the "
                "image template adds one sentence naming the image, and the image itself"
            ),
            "output_contract": {
                "schema": {"proposals": [{"site_ids": ["<id>"], "suspicion": "<0..1>"}]},
                "max_proposals_per_line": MAX_PROPOSALS_PER_LINE,
                "position_grammar": {
                    TOKEN: "one id Tnn of the line",
                    GAP: "one id Gnn, the gap after Tnn, present when Tnn and Tnn+1 are adjacent",
                    TOKEN_PAIR: "two ids Tnn and Tnn+1, adjacent",
                },
                "forbidden": "replacement text, explanations, the corrected line",
                "abstention": '{"proposals": []}',
                "score_semantics": "ranking_only",
            },
            "revision_history": PROMPT_HISTORY,
            "contains_ground_truth": False,
            "contains_evaluation_examples": False,
            "prompt_hash": prompt_hash,
            "frozen_before_endpoint_evaluation": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DECODING_REGISTRY,
        {
            **_envelope("decoding_registry"),
            "revision": DECODING_REVISION,
            "settings": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "repetition_penalty": 1.0,
                "source": "gen1._generation_settings",
            },
            "max_new_tokens_rule": MAX_NEW_TOKENS_RULE,
            "batch_size": int(gen1.MODEL_SPECS[MODEL]["batch_size"]),
            "ordering": "requests sorted by prompt length, ties by request id",
            "dtype": gen1.MODEL_DTYPE,
            "allocator_caps": {L1: 0.95, L2: 1.1, C3: 1.1},
            "allocator_note": "the Metal cap changes memory placement, not outputs (GEN1)",
            "decoding_hash": decoding_hash,
            "frozen_before_endpoint_evaluation": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        IMAGE_PREPROCESSING_REGISTRY,
        {
            **_envelope("image_preprocessing_registry"),
            "revision": IMAGE_PREPROCESSING_REVISION,
            "box": (
                "the union of the line's OCR span boxes, padded by the larger of "
                f"{gen1.CROP_MIN_PAD} px and {gen1.CROP_PAD_FRACTION} of its height, clamped to "
                "the "
                "page; built from OCR geometry alone"
            ),
            "render": {
                "height": LINE_CROP_HEIGHT,
                "max_width": LINE_CROP_MAX_WIDTH,
                "aspect": "kept",
                "resample": "LANCZOS",
                "format": "PNG, RGB",
            },
            "fallback": (
                "no crop when the line has no box or the page image is missing: no L2 request"
            ),
            "why_not_gen1_site_crop": (
                "GEN1's 64 x 1,024 px render is for a site; a whole line is far wider, and at that "
                "size the 10th-percentile line renders about 30 px high"
            ),
            "shuffle_control": (
                "a line crop of another document of the same environment, same render"
            ),
            "policy_hash": image_hash,
            "frozen_before_endpoint_evaluation": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    plan = _compute_plan(requests)
    _write_json_once(
        COMPUTE_BUDGET,
        {
            **_envelope("compute_budget"),
            "measured_lines_per_second": MEASURED_LINES_PER_SECOND,
            "measurement": MEASUREMENT_CONDITIONS,
            **plan,
            "xr1_estimated_machine_hours": cc_read_json(xr1.DESIGN_RECORD)["compute_plan"][
                "estimated_total_hours"
            ],
            "within_envelope": (
                "the same order of machine time as SGV-XR1, so XR1's documents are scanned whole "
                "and every comparison stays paired; no subset was needed"
            ),
            "resumable": "one immutable shard per (arm, environment), skipped once complete",
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "question": (
                "Can visual evidence improve GT-blind error-site coverage at a matched proposal "
                "budget?"
            ),
            "hypotheses": {
                "primary": (
                    "page-image evidence lets the frozen Qwen3-VL-4B locate OCR errors on a "
                    "GT-blind lattice better than P0 at P0's own proposal budget"
                ),
                "mechanism": (
                    "the gain comes from the line's own pixels, not from the model or the text"
                ),
                "secondary": "image localization reaches errors P0 misses (complementarity)",
                "status": "research hypotheses under test, not assumptions",
            },
            "population": {
                "documents": "SGV-XR1's 57 content-blind documents, reused so every result pairs",
                "lines": "every OCR line of every sampled page",
                "lattice": _relative(SITE_LATTICE_REGISTRY),
                "error_sites": (
                    "SGV-CG1's definition: evaluable alignment sites whose OCR text differs from "
                    "ground truth, on the sampled documents; read only at --link"
                ),
            },
            "arms": {
                L0: "P0, SGV14's frozen OCR-only discovery pass, re-derived and checked",
                L1: "Qwen3-VL-4B with the line's tokens and ids, no image",
                L2: "the same weights, text and contract, plus the line's crop",
                L3: (
                    "P0 united with L2's matched-budget proposals, deduplicated on the lattice site"
                ),
            },
            "controls": {
                "C0": "the lattice representation ceiling, with ground truth after generation",
                "C1": "P0 (L0), the primary baseline",
                "C2": "the text-only proposer (L1), the matched image-removal control",
                C3: (
                    f"the first {SHUFFLE_LINES_PER_ENVIRONMENT} lines of each environment in a "
                    "content-blind order, with the line crop of another document of the same "
                    "environment; correct text and ids"
                ),
                C4: (
                    f"{RANDOM_DRAWS} draws of admissible lattice sites, uniformly without "
                    "replacement, at L2's matched-budget count per environment"
                ),
                "no_centre_mask": (
                    "XR1's crop-centre mask needs a known site; LP1 has none, so no mask geometry "
                    "can be pre-registered without ground truth"
                ),
            },
            "proposal_contract": {
                "unit": "one lattice site; duplicates within a line keep the highest score",
                "valid": "ids of the line only, in the position grammar of the prompt registry",
                "invalid_handling": (
                    "an invalid id, an impossible span, a duplicate or an unparsed answer is "
                    "counted in proposal_compliance.json and contributes no proposal"
                ),
                "leakage_handling": (
                    "any field other than site_ids and suspicion is protocol leakage: counted, and "
                    "its content discarded; the named position is kept"
                ),
                "score": "ranking_only; a missing or out-of-range score ranks as 0",
                "ranking_key": (
                    "score descending, then the model's own order within the line, then a "
                    "content-blind hash of the site id (page order would favour early pages)"
                ),
            },
            "budget": {
                "primary": "per environment, exactly P0's proposal count on the sampled documents",
                "p0_budget_by_environment": budgets,
                "when_an_arm_proposes_fewer": "all of its proposals, and the fill is reported",
                "fractions": list(BUDGET_FRACTIONS),
                "p0_below_its_count": "P0 ranked by its own frozen suspicion score",
                "selection_reads_labels": False,
            },
            "metrics": {
                "localization_recall": (
                    "OCR error sites reached by at least one proposal / evaluable OCR error sites; "
                    "per environment, averaged over the ten (pooled reported beside it)"
                ),
                "proposal_precision": "proposals linked to an OCR error site / proposals",
                "false_proposal_rate": "1 - proposal precision; no correction is applied here",
                "p0_miss_recovery": (
                    "OCR error sites P0 does not reach that the arm reaches / those P0 misses, "
                    "pooled over environments"
                ),
                "burden": "proposals per document, per page, per line and per 1,000 OCR tokens",
                "union": "P0 and L2-only, both, neither, Jaccard, incremental proposals and errors",
            },
            "gate": {
                "C1": f"L2 matched-budget recall minus P0's at least {RECALL_MARGIN}",
                "C2": (
                    f"that difference at least {BREADTH_ENVIRONMENT_MARGIN} in at least "
                    f"{BREADTH_MAJORITY} of 10 environments"
                ),
                "C3": f"L2 matched-budget P0-miss recovery at least {MISS_RECOVERY_TARGET}",
                "C4": (
                    f"L2 minus P0 matched-budget recall at least {WEAK_TYPE_MARGIN}, pooled, in at "
                    f"least {WEAK_TYPES_REQUIRED} of substitution, omission and segmentation"
                ),
                "C5": (
                    "L2 minus L1 matched-budget recall with a percentile interval above zero, "
                    "and on the shuffle subset the shuffled image keeps at most "
                    f"{SHUFFLE_RETENTION_MAX} of "
                    "L2's gain over L1"
                ),
                "U1": f"P0 + L2 recall minus P0's at least {RECALL_MARGIN}",
                "U2": (
                    f"that difference at least {BREADTH_ENVIRONMENT_MARGIN} in at least "
                    f"{BREADTH_MAJORITY} of 10 environments; the union holds at most twice P0's "
                    "proposals by construction"
                ),
                "N1": f"L2 natural-output recall minus P0's at least {RECALL_MARGIN}",
                "N2": "L2's natural output holds more proposals than P0's",
                "margin_justification": {
                    "p0_recall_on_these_pages": 0.6468,
                    "proposal_loss_on_these_pages": 0.3532,
                    "xr1_image_localization_headroom": XR1_IMAGE_HEADROOM,
                    "reading": (
                        "0.10 is a material share of the 0.3532 P0 leaves unreached; a 0.25 miss "
                        "recovery is one quarter of the errors P0 never offers to a generator"
                    ),
                },
            },
            "outcome_rule": {
                "precedence": "A, then B, then D, then C",
                **{k: f"{OUTCOME_LABELS[k]} -> NEXT: {NEXT_STAGE[k]}" for k in OUTCOME_LABELS},
                "ready_for_integrated_image_pipeline": "true for A and B only",
                "ready_for_external_confirmation": False,
            },
            "statistics": {
                "unit": "document cluster, resampled within environment; paired",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile",
                "alpha": ALPHA,
                "primary_family": [
                    "P1 L2 minus P0, matched-budget recall",
                    "P2 L2 minus L1, matched-budget recall",
                    "P3 L2 matched-budget P0-miss recovery minus 0.25",
                ],
                "mechanism_family": ["M1 L2 minus C3, subset-matched recall on the shuffle subset"],
                "multiplicity": "Holm within each family",
            },
            "non_goals": [
                "correction generation; LP1 ends at site proposal",
                "any reliability model, adaptation, threshold, calibration or certification",
                "external confirmation or the confirmatory reserve",
                "a larger or different model, or any model search",
            ],
            "downstream_estimate": (
                "the bottleneck ladder's last step uses GEN1/XR1 rates as a labelled estimate only"
            ),
            "budgets_changed_after_labels": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"preregister: design, registries, compute budget ({plan['estimated_total_hours']} h)")
    return 0


# ------------------------------------------------------------------ GT-blind contexts

LINE_CONTEXTS = OUT / "line_contexts.parquet"
LINE_CONTEXT_COLUMNS = (
    "line_uid",
    "text_prompt",
    "image_prompt",
    "crop_level",
    "crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
)


def render_line_crop(image: Any, box: tuple[int, int, int, int]) -> Any:
    """The frozen line render: LINE_CROP_HEIGHT high, aspect kept, width capped."""
    from PIL import Image

    region = image.crop(box)
    width, height = region.size
    target_w, target_h = max(1, round(width * LINE_CROP_HEIGHT / height)), LINE_CROP_HEIGHT
    if target_w > LINE_CROP_MAX_WIDTH:
        target_w, target_h = (
            LINE_CROP_MAX_WIDTH,
            max(1, round(height * LINE_CROP_MAX_WIDTH / width)),
        )
    return region.resize((target_w, target_h), Image.Resampling.LANCZOS)


def build_line_contexts(lines: pd.DataFrame, crop_dir: Path) -> pd.DataFrame:
    """Both user prompts and the line crop for every scanned line. Reads OCR, boxes and pixels."""
    from PIL import Image

    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    image: tuple[str, Any] | None = None
    for row in lines.itertuples(index=False):
        record: dict[str, Any] = {
            "line_uid": str(row.line_uid),
            "text_prompt": USER_TEMPLATES["text"].format(text=row.line_text),
            "image_prompt": USER_TEMPLATES["image"].format(text=row.line_text),
            "crop_level": CROP_NONE,
            "crop_path": None,
            "crop_sha256": None,
            "crop_width": None,
            "crop_height": None,
        }
        box = row.crop_box
        has_box = box is not None and not (isinstance(box, float) and np.isnan(box))
        if has_box and Path(str(row.page_image_path)).is_file():
            if image is None or image[0] != row.page_image_path:
                image = (str(row.page_image_path), Image.open(row.page_image_path).convert("RGB"))
            crop = render_line_crop(image[1], tuple(int(v) for v in box))  # type: ignore[arg-type]
            target = crop_dir / f"{canonical_hash({'line': str(row.line_uid)})[:32]}.png"
            if not target.is_file():
                crop.save(target, format="PNG")
            record.update(
                crop_level=CROP_LINE,
                crop_path=str(target),
                crop_sha256=xr1._crop_record(crop),
                crop_width=int(crop.size[0]),
                crop_height=int(crop.size[1]),
            )
        rows.append(record)
    return pd.DataFrame(rows, columns=list(LINE_CONTEXT_COLUMNS))


def shuffle_subset(line_uids: Sequence[str]) -> list[str]:
    """The control subsample: the first lines of an environment in a content-blind order."""
    ordered = sorted(line_uids, key=lambda uid: order_key("shuffle", uid))
    return ordered[:SHUFFLE_LINES_PER_ENVIRONMENT]


def shuffle_donors(lines: pd.DataFrame, chosen: Sequence[str]) -> dict[str, str]:
    """Each chosen line's donor: a cropped line of another document, content-blind order."""
    cropped = lines[lines["crop_path"].notna()]
    donors: dict[str, str] = {}
    for uid in chosen:
        document = str(lines.loc[lines["line_uid"] == uid, "document_id"].iloc[0])
        candidates = cropped[cropped["document_id"].astype(str) != document]["line_uid"].astype(str)
        if candidates.empty:
            continue
        donors[uid] = min(candidates, key=lambda c: order_key("donor", uid, c))
    return donors


def run_contexts() -> int:
    """Line prompts and crops for every line; the image-shuffle subsample and its donors."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (LINE_CONTEXTS, CONTROL_CONTEXTS):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an LP1 answer or label exists; contexts are frozen before both")
    lines = pd.read_parquet(SCAN_POPULATION)
    frames: list[pd.DataFrame] = []
    controls: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        mine = lines[lines["environment"] == name].reset_index(drop=True)
        contexts = build_line_contexts(mine, CROP_DIR / _slug(name))
        frames.append(contexts)
        joined = mine.merge(contexts, on="line_uid", validate="one_to_one")
        chosen = shuffle_subset(joined["line_uid"].astype(str).tolist())
        donors = shuffle_donors(joined, chosen)
        by_uid = joined.set_index("line_uid")
        for uid in chosen:
            donor = donors.get(uid)
            if donor is None or by_uid.loc[uid, "crop_path"] is None:
                continue
            controls.append(
                {
                    "line_uid": uid,
                    "environment": name,
                    "arm": C3,
                    "document_id": str(by_uid.loc[uid, "document_id"]),
                    "donor_line_uid": donor,
                    "donor_document_id": str(by_uid.loc[donor, "document_id"]),
                    "crop_path": str(by_uid.loc[donor, "crop_path"]),
                    "crop_sha256": str(by_uid.loc[donor, "crop_sha256"]),
                    "own_crop_sha256": str(by_uid.loc[uid, "crop_sha256"]),
                }
            )
        print(f"  {name}: {len(contexts)} line contexts, {len(donors)} shuffle donors", flush=True)
    contexts = pd.concat(frames, ignore_index=True)
    control = pd.DataFrame(controls)
    own = control["document_id"] == control["donor_document_id"]
    if bool(own.any()) or bool((control["crop_sha256"] == control["own_crop_sha256"]).any()):
        raise PhaseError("a shuffled line received a crop of its own document")
    _write_parquet_once(LINE_CONTEXTS, contexts)
    _write_parquet_once(CONTROL_CONTEXTS, control)
    print(
        f"contexts: {len(contexts)} lines, {len(control)} shuffle requests "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ inference

REGENERATION_ENVIRONMENT = "funsd/doctr"
REGENERATION_BATCHES = 2


def _selected_arms() -> list[str]:
    chosen = [a for a in os.environ.get("LP1_ARMS", "").split(",") if a]
    unknown = [a for a in chosen if a not in MODEL_ARMS]
    if unknown:
        raise PhaseError(f"unknown LP1_ARMS {unknown}; choose from {list(MODEL_ARMS)}")
    return chosen or list(MODEL_ARMS)


def _shard_path(arm: str, name: str) -> Path:
    return RAW_CACHE / f"{_slug(name)}.{arm}.parquet"


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name.replace(".parquet", ".cost.json"))


def _shard_complete(path: Path) -> bool:
    sidecar = _sidecar(path)
    if not (path.is_file() and sidecar.is_file()):
        return False
    return bool(cc_read_json(sidecar)["raw_output_sha256"] == file_sha256(path))


def requests_for(arm: str, name: str) -> pd.DataFrame:
    """One shard's frozen requests, rebuilt from the frozen GT-blind tables alone."""
    lines = pd.read_parquet(SCAN_POPULATION)
    lines = lines[lines["environment"] == name][
        ["environment", "line_uid", "document_id", "lattice_sites"]
    ]
    contexts = pd.read_parquet(LINE_CONTEXTS)
    joined = lines.merge(contexts, on="line_uid", validate="one_to_one")
    if arm == C3:
        control = pd.read_parquet(CONTROL_CONTEXTS)
        control = control[control["arm"] == C3][["line_uid", "crop_path", "crop_sha256"]]
        joined = joined.drop(columns=["crop_path", "crop_sha256"]).merge(
            control, on="line_uid", validate="one_to_one"
        )
    elif arm == L2:
        joined = joined[joined["crop_path"].notna()]
    image = arm in IMAGE_ARMS
    return pd.DataFrame(
        {
            "environment": joined["environment"].to_numpy(),
            "method": arm,
            "request_id": (joined["line_uid"] + f"|{arm}").to_numpy(),
            "line_uid": joined["line_uid"].to_numpy(),
            "document_id": joined["document_id"].to_numpy(),
            "lattice_sites": joined["lattice_sites"].to_numpy(dtype=np.int64),
            "user_prompt": (joined["image_prompt"] if image else joined["text_prompt"]).to_numpy(),
            "image_path": joined["crop_path"].to_numpy() if image else None,
            "image_sha256": joined["crop_sha256"].to_numpy() if image else None,
        }
    ).reset_index(drop=True)


def run_line_requests(
    runner: Any,
    requests: pd.DataFrame,
    batch_size: int,
    progress: Callable[[str], None],
    *,
    limit_batches: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """GEN1's batching loop with LP1's cap: raw answers exactly as decoded, plus cost accounting."""
    frame = requests.reset_index(drop=True)
    prompts = [str(p) for p in frame["user_prompt"]]
    images = [
        None if p is None or (isinstance(p, float) and np.isnan(p)) else str(p)
        for p in frame["image_path"]
    ]
    ids = frame["request_id"].astype(str).tolist()
    sites = frame["lattice_sites"].to_numpy(dtype=np.int64)
    order = sorted(range(len(frame)), key=lambda i: (len(prompts[i]), ids[i]))
    began = time.monotonic()
    rows: list[dict[str, Any]] = []
    batches = 0
    for start in range(0, len(order), batch_size):
        if limit_batches is not None and batches >= limit_batches:
            break
        chunk = order[start : start + batch_size]
        cap = cap_tokens(int(sites[chunk].max()))
        outputs = runner.generate([(SYSTEM_PROMPT, prompts[i], images[i]) for i in chunk], cap)
        for i, (text, prompt_tokens, output_tokens, finished) in zip(chunk, outputs, strict=True):
            rows.append(
                {
                    "environment": str(frame.at[i, "environment"]),
                    "method": str(frame.at[i, "method"]),
                    "request_id": ids[i],
                    "line_uid": str(frame.at[i, "line_uid"]),
                    "output": text,
                    "prompt_sha256": canonical_hash(
                        {
                            "system": SYSTEM_PROMPT,
                            "user": prompts[i],
                            "image": frame.at[i, "image_sha256"],
                        }
                    ),
                    "prompt_tokens": int(prompt_tokens),
                    "output_tokens": int(output_tokens),
                    "finished": bool(finished),
                    "max_new_tokens": int(cap),
                }
            )
        batches += 1
        if batches % 10 == 0:
            rate = len(rows) / max(time.monotonic() - began, 1e-9)
            progress(f"    {len(rows)}/{len(frame)} ({rate:.3f}/s)")
    elapsed = time.monotonic() - began
    raw = pd.DataFrame(rows)
    cost = {
        "requests": len(raw),
        "wall_clock_seconds": elapsed,
        "requests_per_second": _ratio(len(raw), elapsed),
        "prompt_tokens": int(raw["prompt_tokens"].sum()) if len(raw) else 0,
        "output_tokens": int(raw["output_tokens"].sum()) if len(raw) else 0,
        "unfinished_decodes": int((~raw["finished"]).sum()) if len(raw) else 0,
        "batch_size": batch_size,
        "batches": batches,
        "ordering": "prompt length, then request id",
        "max_new_tokens_rule": MAX_NEW_TOKENS_RULE,
    }
    return raw, cost


def run_generate() -> int:
    """GT-blind inference, cached immutably one (arm, environment) shard at a time.

    Each shard is written atomically with its provenance sidecar the moment it exists, so an
    interrupted run keeps every finished shard and a rerun skips each one whose file matches its
    sidecar hash.
    """
    started = time.monotonic()
    for path in (
        LINE_CONTEXTS,
        CONTROL_CONTEXTS,
        PROMPT_REGISTRY,
        DECODING_REGISTRY,
        IMAGE_PREPROCESSING_REGISTRY,
    ):
        _require(path, "contexts" if path in (LINE_CONTEXTS, CONTROL_CONTEXTS) else "preregister")
    if _ground_truth_attached():
        raise PhaseError("labels exist; generation must precede every label")
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    prompt_hash = cc_read_json(PROMPT_REGISTRY)["prompt_hash"]
    decoding_hash = cc_read_json(DECODING_REGISTRY)["decoding_hash"]
    image_hash = cc_read_json(IMAGE_PREPROCESSING_REGISTRY)["policy_hash"]
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
    print(
        f"  loaded {MODEL} on {device} ({time.monotonic() - began:.0f}s); {len(work)} shards",
        flush=True,
    )
    for arm, name in work:
        path = _shard_path(arm, name)
        requests = requests_for(arm, name)
        raw, cost = run_line_requests(runner, requests, int(spec["batch_size"]), print)
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
                "model": MODEL,
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "document_ids": sorted({str(d) for d in requests["document_id"]}),
                "line_count": int(requests["line_uid"].nunique()),
                "line_ids_sha256": canonical_hash(sorted(requests["line_uid"].astype(str))),
                **cost,
                "prompt_hash": prompt_hash,
                "decoding_hash": decoding_hash,
                "image_preprocessing_hash": image_hash if arm in IMAGE_ARMS else None,
                "allocator_environment": gen1._allocator_environment(),
                "request_prompts_hash": canonical_hash(sorted(raw["prompt_sha256"].tolist())),
                "device": device,
                "platform": platform.platform(),
                "raw_output": _relative(path),
                "raw_output_sha256": file_sha256(path),
                "clock": "time.monotonic, which does not advance while the machine sleeps",
                "uses_ground_truth": False,
            },
        )
        print(
            f"    {arm}/{name}: {cost['requests']} lines in {cost['wall_clock_seconds']:.0f}s "
            f"({cost['requests_per_second']:.3f}/s)",
            flush=True,
        )
    del runner
    _release_device_cache(device)
    print(f"generate: done in {time.monotonic() - started:.0f}s")
    return 0


def run_spotcheck() -> int:
    """Each selected arm regenerates its first batches, and every answer must repeat exactly."""
    REGENERATION_DIR.mkdir(parents=True, exist_ok=True)
    arms = [a for a in _selected_arms() if not (REGENERATION_DIR / f"{a}.json").is_file()]
    if not arms:
        print("spotcheck: every selected arm already checked")
        return 0
    device = _resolve_device()
    runner = gen1._runner(MODEL, device)
    spec = gen1.MODEL_SPECS[MODEL]
    for arm in arms:
        path = _shard_path(arm, REGENERATION_ENVIRONMENT)
        if not _shard_complete(path):
            raise PhaseError(f"{arm}: generate {REGENERATION_ENVIRONMENT} before the spot-check")
        stored = pd.read_parquet(path).set_index("request_id")
        requests = requests_for(arm, REGENERATION_ENVIRONMENT)
        raw, _cost = run_line_requests(
            runner, requests, int(spec["batch_size"]), print, limit_batches=REGENERATION_BATCHES
        )
        differing = int(
            sum(
                str(row.output) != str(stored.loc[str(row.request_id), "output"])
                for row in raw.itertuples(index=False)
            )
        )
        _write_json_once(
            REGENERATION_DIR / f"{arm}.json",
            {
                **_envelope("model_regeneration"),
                "arm": arm,
                "environment": REGENERATION_ENVIRONMENT,
                "batches": REGENERATION_BATCHES,
                "requests": len(raw),
                "differing_outputs": differing,
                "identical": differing == 0,
                "allocator_environment": gen1._allocator_environment(),
                "stored_shard_allocator": cc_read_json(_sidecar(path)).get("allocator_environment"),
            },
        )
        print(f"  {arm}: {len(raw)} regenerated, {differing} differ", flush=True)
    del runner
    _release_device_cache(device)
    return 0


def run_envcheck() -> int:
    """Does the updated OS decode the frozen checkpoint exactly as it did for SGV-XR1?

    XR1's own spot-check requests -- its first two batches of G1 and G2 on funsd/doctr, rebuilt
    from XR1's frozen tables -- are decoded again and compared with XR1's stored answers. The
    result is recorded either way; no LP1 comparison depends on it, because every LP1 arm runs here.
    """
    started = time.monotonic()
    _require(MODEL_VERSIONS, "freeze")
    _forbid(DECODING_ENVIRONMENT_CHECK)
    device = _resolve_device()
    runner = gen1._runner(MODEL, device)
    spec = gen1.MODEL_SPECS[MODEL]
    arms: dict[str, Any] = {}
    for arm in (xr1.G1, xr1.G2):
        stored = pd.read_parquet(xr1._shard_path(arm, xr1.REGENERATION_ENVIRONMENT))
        stored = stored.set_index("request_id")
        requests = xr1.requests_for(arm, xr1.REGENERATION_ENVIRONMENT)
        raw, _cost = gen1.run_requests(
            runner,
            requests,
            int(spec["batch_size"]),
            print,
            limit_batches=xr1.REGENERATION_BATCHES,
        )
        differing = int(
            sum(
                str(row.output) != str(stored.loc[str(row.request_id), "output"])
                for row in raw.itertuples(index=False)
            )
        )
        arms[arm] = {
            "requests": len(raw),
            "differing_outputs": differing,
            "identical": differing == 0,
        }
        print(f"  XR1 {arm}: {len(raw)} regenerated under this OS, {differing} differ", flush=True)
    del runner
    _release_device_cache(device)
    _write_json_once(
        DECODING_ENVIRONMENT_CHECK,
        {
            **_envelope("decoding_environment_check"),
            "platform": platform.platform(),
            "xr1_platform": GEN1_PLATFORM,
            "environment": xr1.REGENERATION_ENVIRONMENT,
            "batches": xr1.REGENERATION_BATCHES,
            "arms": arms,
            "identical_to_xr1": all(v["identical"] for v in arms.values()),
            "allocator_environment": gen1._allocator_environment(),
            "reads_ground_truth": False,
            "note": (
                "XR1's frozen requests and stored answers are read, never written. Every LP1 arm "
                "decodes under this platform, so no LP1 comparison crosses the OS update."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    return 0


# ------------------------------------------------------------------ parse: raw -> lattice proposals
#
# GT-blind. Every raw answer becomes zero or more lattice sites under the frozen position grammar,
# and every arm's matched-budget selection is fixed here, before a single label exists.

PARSE_STRICT = "strict"
PARSE_LENIENT = "lenient"
PARSE_TRUNCATED = "truncated"
PARSE_INVALID = "invalid"
PARSE_EMPTY = "empty"
ALLOWED_FIELDS = frozenset({"site_ids", "suspicion"})
_ID = re.compile(r"^\s*([TG])0*(\d+)\s*$")
_OBJECT = re.compile(r"\{[^{}]*\}", re.S)


def parse_answer(text: str, finished: bool) -> tuple[str, list[Any], bool]:
    """The answer's proposal list, its parse status, and whether prose surrounds the JSON.

    ``strict``: the whole answer is one JSON object with a ``proposals`` list. ``lenient``: such
    an object after stripping a code fence or surrounding prose. ``truncated``: an unfinished
    decode whose complete proposal objects are kept and nothing else. Otherwise ``invalid``, or
    ``empty`` when nothing was produced.
    """
    body = text.strip()
    if not body:
        return PARSE_EMPTY, [], False
    try:
        payload = json.loads(body)
        if isinstance(payload, dict) and isinstance(payload.get("proposals"), list):
            return PARSE_STRICT, list(payload["proposals"]), False
    except json.JSONDecodeError:
        pass
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(body[start : end + 1])
            if isinstance(payload, dict) and isinstance(payload.get("proposals"), list):
                prose = bool(re.sub(r"```(?:json)?", "", body[:start] + body[end + 1 :]).strip())
                return PARSE_LENIENT, list(payload["proposals"]), prose
        except json.JSONDecodeError:
            pass
    if not finished and '"proposals"' in body:
        items = []
        for match in _OBJECT.finditer(body[body.index('"proposals"') :]):
            try:
                items.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
        return PARSE_TRUNCATED, items, False
    return PARSE_INVALID, [], False


def position_of(site_ids: Any, local_ids: set[str], tokens: int) -> tuple[str | None, str]:
    """A proposal's lattice local id under the position grammar, or why it has none.

    One Tnn is a token; one Gnn is the gap after Tnn when that gap is in the lattice; two token ids
    naming Tnn and Tnn+1, in either order, are the adjacent pair. Leading zeros are normalized.
    """
    if not isinstance(site_ids, list) or not site_ids:
        return None, "empty_ids"
    parsed = []
    for raw in site_ids:
        match = _ID.match(str(raw))
        if match is None:
            return None, "invalid_id"
        parsed.append((match.group(1), int(match.group(2))))
    if len(parsed) == 1:
        kind, index = parsed[0]
        local = local_token(index) if kind == "T" else local_gap(index)
        return (local, "ok") if local in local_ids else (None, "invalid_id")
    if len(parsed) == 2 and all(kind == "T" for kind, _i in parsed):
        low, high = sorted(index for _k, index in parsed)
        if high >= tokens:
            return None, "invalid_id"
        if high != low + 1:
            return None, "impossible_span"
        local = local_pair(low)
        return (local, "ok") if local in local_ids else (None, "impossible_span")
    return None, "impossible_span"


def score_of(value: Any) -> tuple[float, str]:
    """The ranking score: a number in [0, 1], or 0 with the reason it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0, "missing_score" if value is None else "invalid_score"
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        return 0.0, "invalid_score"
    return number, "ok"


def parse_line(
    output: str, finished: bool, local_ids: Sequence[str], site_ids: Sequence[str], tokens: int
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    """One line's answer as distinct lattice proposals, in the model's order, with its census."""
    status, items, prose = parse_answer(output, finished)
    census: dict[str, int] = {
        "items": len(items),
        "not_an_object": 0,
        "invalid_id": 0,
        "impossible_span": 0,
        "empty_ids": 0,
        "duplicate": 0,
        "over_cap": 0,
        "extra_fields": 0,
        "prose_outside_json": int(prose),
        "missing_score": 0,
        "invalid_score": 0,
    }
    site_of = dict(zip(local_ids, site_ids, strict=True))
    valid = set(local_ids)
    kept: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            census["not_an_object"] += 1
            continue
        extra = sorted(set(item) - ALLOWED_FIELDS)
        if extra:
            census["extra_fields"] += 1
        local, reason = position_of(item.get("site_ids"), valid, tokens)
        if local is None:
            census[reason] += 1
            continue
        score, score_status = score_of(item.get("suspicion"))
        if score_status != "ok":
            census[score_status] += 1
        if local in kept:
            census["duplicate"] += 1
            kept[local]["suspicion"] = max(kept[local]["suspicion"], score)
            continue
        if len(kept) >= MAX_PROPOSALS_PER_LINE:
            census["over_cap"] += 1
            continue
        kept[local] = {
            "local_id": local,
            "site_id": site_of[local],
            "suspicion": score,
            "within_line_rank": len(kept),
            "leakage": bool(extra) or prose,
        }
    return list(kept.values()), census, status


def select_budget(frame: pd.DataFrame, budget: int, key: Sequence[str]) -> np.ndarray:
    """The first `budget` rows under the frozen ranking key; reads no label."""
    order = frame.sort_values(list(key), kind="stable").index
    chosen = np.zeros(len(frame), dtype=bool)
    positions = frame.index.get_indexer(order[: max(0, budget)])
    chosen[positions] = True
    return chosen


RANK_KEY = ("neg_suspicion", "within_line_rank", "hash_order")
P0_RANK_KEY = ("neg_suspicion", "hash_order")


def _budget_columns(
    frame: pd.DataFrame, budgets: dict[str, int], key: Sequence[str]
) -> pd.DataFrame:
    """Per-environment budget flags for every frozen fraction, from the ranking key alone."""
    out = frame.copy()
    for fraction in BUDGET_FRACTIONS:
        column = budget_column(fraction)
        out[column] = False
        for name, block in out.groupby("environment", sort=True):
            size = round(fraction * budgets[str(name)])
            flags = select_budget(block, size, key)
            out.loc[block.index, column] = flags
    return out


def budget_column(fraction: float) -> str:
    return f"in_budget_{fraction:.2f}".replace(".", "_")


def _verified_shards() -> list[dict[str, Any]]:
    """Every raw shard's bytes against its sidecar, before anything is parsed."""
    manifest: list[dict[str, Any]] = []
    for arm in MODEL_ARMS:
        for spec in environment_specs():
            path = _shard_path(arm, spec["environment"])
            if not _shard_complete(path):
                raise PhaseError(f"{_relative(path)} is missing or does not match its sidecar")
            sidecar = cc_read_json(_sidecar(path))
            manifest.append(
                {
                    "environment": spec["environment"],
                    "method": arm,
                    "raw_output": _relative(path),
                    "raw_output_sha256": sidecar["raw_output_sha256"],
                    "sidecar": _relative(_sidecar(path)),
                    "requests": int(sidecar["requests"]),
                    "line_count": int(sidecar["line_count"]),
                    "prompt_hash": sidecar["prompt_hash"],
                    "decoding_hash": sidecar["decoding_hash"],
                }
            )
    return manifest


def run_parse() -> int:
    """Raw answers -> hashes -> lattice proposals -> matched-budget selections. GT-blind."""
    started = time.monotonic()
    outputs = (RAW_OUTPUT_MANIFEST, RAW_OUTPUT_HASHES, PARSED_PROPOSALS, PROPOSAL_COMPLIANCE)
    for path in outputs:
        _forbid(path)
    if _ground_truth_attached():
        raise PhaseError("labels exist; every proposal must be frozen before the first label")
    manifest = _verified_shards()
    _write_json_once(
        RAW_OUTPUT_HASHES,
        {
            **_envelope("raw_output_hashes"),
            "hashes": {row["raw_output"]: row["raw_output_sha256"] for row in manifest},
            "verified_against_sidecars": True,
            "verified_before_any_label": True,
        },
    )
    _write_json_once(
        RAW_OUTPUT_MANIFEST,
        {**_envelope("raw_output_manifest"), "shards": manifest, "count": len(manifest)},
    )
    lines = pd.read_parquet(SCAN_POPULATION).set_index("line_uid")
    budgets = cc_read_json(DESIGN_RECORD)["budget"]["p0_budget_by_environment"]
    rows: list[dict[str, Any]] = []
    census_rows: list[dict[str, Any]] = []
    for arm in MODEL_ARMS:
        for spec in environment_specs():
            name = spec["environment"]
            raw = pd.read_parquet(_shard_path(arm, name))
            for answer in raw.itertuples(index=False):
                line = lines.loc[str(answer.line_uid)]
                proposals, census, status = parse_line(
                    str(answer.output),
                    bool(answer.finished),
                    list(line["local_ids"]),
                    list(line["site_ids"]),
                    int(line["tokens"]),
                )
                census_rows.append(
                    {
                        "method": arm,
                        "environment": name,
                        "line_uid": str(answer.line_uid),
                        "status": status,
                        "proposals": len(proposals),
                        **census,
                    }
                )
                for proposal in proposals:
                    rows.append(
                        {
                            "method": arm,
                            "environment": name,
                            "line_uid": str(answer.line_uid),
                            "document_id": str(line["document_id"]),
                            "document_rank": int(line["document_rank"]),
                            "line_index": int(line["line_index"]),
                            **proposal,
                        }
                    )
    model = pd.DataFrame(rows)
    p0 = pd.read_parquet(P0_PROPOSALS)
    p0_rows = pd.DataFrame(
        {
            "method": L0,
            "environment": p0["environment"].astype(str),
            "line_uid": [
                f"{e}|{d}|L{i:04d}"
                for e, d, i in zip(
                    p0["environment"], p0["document_id"], p0["line_index"], strict=True
                )
            ],
            "document_id": p0["document_id"].astype(str),
            "document_rank": p0["document_rank"].astype(int),
            "line_index": p0["line_index"].astype(int),
            "local_id": None,
            "site_id": p0["lattice_site_id"].astype(str),
            "suspicion": p0["suspicion_score"].astype(float),
            "within_line_rank": 0,
            "leakage": False,
        }
    )
    frames = []
    for arm, block in pd.concat([p0_rows, model], ignore_index=True).groupby("method", sort=False):
        block = block.reset_index(drop=True).assign(
            neg_suspicion=lambda f: -f["suspicion"].astype(float),
            hash_order=lambda f: [order_key("rank", s) for s in f["site_id"].astype(str)],
        )
        key = P0_RANK_KEY if arm == L0 else RANK_KEY
        frames.append(_budget_columns(block, budgets, key))
    parsed = pd.concat(frames, ignore_index=True)
    # The shuffle subset's own budget: P0's proposals on the subset's lines, per environment.
    subset = set(pd.read_parquet(CONTROL_CONTEXTS)["line_uid"].astype(str))
    on_subset = parsed["line_uid"].isin(subset)
    subset_budgets = (
        parsed[(parsed["method"] == L0) & on_subset].groupby("environment").size().to_dict()
    )
    parsed["on_shuffle_subset"] = on_subset
    parsed["in_subset_budget"] = False
    for arm in (L0, L1, L2, C3):
        for name in budgets:
            block = parsed[(parsed["method"] == arm) & on_subset & (parsed["environment"] == name)]
            if block.empty:
                continue
            key = P0_RANK_KEY if arm == L0 else RANK_KEY
            flags = select_budget(block, int(subset_budgets.get(name, 0)), key)
            parsed.loc[block.index, "in_subset_budget"] = flags
    leaked = [c for c in FORBIDDEN_INPUT_COLUMNS if c in parsed.columns]
    if leaked:
        raise PhaseError(f"the proposal table carries truth columns: {leaked}")
    _write_parquet_once(PARSED_PROPOSALS, parsed)
    census = pd.DataFrame(census_rows)
    _write_json_once(
        PROPOSAL_COMPLIANCE,
        {
            **_compliance_record(census, parsed, budgets, subset_budgets, started),
            "parsed_proposals_sha256": file_sha256(PARSED_PROPOSALS),
        },
    )
    print(f"parse: {len(model)} model proposals and {len(p0_rows)} P0 proposals, selections fixed")
    return 0


def _compliance_record(
    census: pd.DataFrame,
    parsed: pd.DataFrame,
    budgets: dict[str, int],
    subset_budgets: dict[str, int],
    started: float,
) -> dict[str, Any]:
    counters = [
        "items",
        "not_an_object",
        "invalid_id",
        "impossible_span",
        "empty_ids",
        "duplicate",
        "over_cap",
        "extra_fields",
        "prose_outside_json",
        "missing_score",
        "invalid_score",
        "proposals",
    ]
    by_arm: dict[str, Any] = {}
    for arm, block in census.groupby("method", sort=False):
        valid = block[block["status"].isin([PARSE_STRICT, PARSE_LENIENT, PARSE_TRUNCATED])]
        by_arm[str(arm)] = {
            "lines_requested": len(block),
            "parse_status": {str(k): int(v) for k, v in block["status"].value_counts().items()},
            "valid_json_share": _ratio(int(block["status"].eq(PARSE_STRICT).sum()), len(block)),
            **{c: int(block[c].sum()) for c in counters},
            "zero_proposal_lines": int((valid["proposals"] == 0).sum()),
            "zero_proposal_line_rate": _ratio(int((valid["proposals"] == 0).sum()), len(valid)),
            "lines_with_leakage": int(
                ((block["extra_fields"] > 0) | (block["prose_outside_json"] > 0)).sum()
            ),
        }
    fill = {}
    for arm, block in parsed.groupby("method", sort=False):
        column = budget_column(PRIMARY_FRACTION)
        fill[str(arm)] = {
            name: {
                "budget": int(budgets[name]),
                "selected": int(block[(block["environment"] == name) & block[column]].shape[0]),
                "natural": int((block["environment"] == name).sum()),
            }
            for name in budgets
        }
    return {
        **_envelope("proposal_compliance"),
        "by_arm": by_arm,
        "matched_budget_fill": fill,
        "shuffle_subset_budgets": {k: int(v) for k, v in subset_budgets.items()},
        "rules": {
            "unit": "one distinct lattice site per line; a duplicate keeps the highest score",
            "cap": f"at most {MAX_PROPOSALS_PER_LINE} per line, in the model's order",
            "leakage": "extra fields or prose around the JSON: counted, content discarded",
            "invalid": "invalid ids, impossible spans, non-objects: counted, no proposal",
        },
        "reads_ground_truth": False,
        "runtime_seconds": time.monotonic() - started,
    }


# ------------------------------------------------------------------ link: ground truth enters

ERROR_KINDS = ("substitution", "deletion", "segmentation", "insertion")
KIND_NAMES = {**WEAK_TYPE_NAMES, "insertion": "spurious insertion"}
CROSS_LINE = "cross_line_gap"


def _bare(site_ids: pd.Series) -> pd.Series:
    return site_ids.astype(str).str.split("|", n=1).str[1]


def cross_line_gaps(page: Any, document_id: str, engine_id: str) -> pd.DataFrame:
    """Stream-adjacent span pairs on different OCR lines: a diagnostic, never a proposal."""
    rows = [
        {
            "site_id": f"{document_id}:{engine_id}:xline:{i:05d}",
            "document_id": document_id,
            "engine_id": engine_id,
            "anchor_kind": GAP,
            "anchor_ref": anchor_ref(GAP, left, right),
        }
        for i, (left, right) in enumerate(zip(page.order[:-1], page.order[1:], strict=True))
        if int(page.span_line[left]) != int(page.span_line[right])
    ]
    columns = ["site_id", "document_id", "engine_id", "anchor_kind", "anchor_ref"]
    return pd.DataFrame(rows, columns=columns)


def run_link() -> int:
    """Error sites, every lattice site's links, the ceiling and P0's reproduction of XR1."""
    from ocr_risk.schemas.enums import AnchorKind

    started = time.monotonic()
    _require(PARSED_PROPOSALS, "parse")
    for path in (ERROR_SITES, PROPOSAL_ERROR_LINKS, LATTICE_CEILING):
        _forbid(path)
    registry = cc_read_json(SITE_LATTICE_REGISTRY)
    if file_sha256(SITE_LATTICE_INVENTORY) != registry["inventory"]["sha256"]:
        raise PhaseError("the lattice changed after it was frozen")
    lattice = pd.read_parquet(SITE_LATTICE_INVENTORY)
    p0 = pd.read_parquet(P0_PROPOSALS)
    sample = pd.read_parquet(xr1.PROPOSAL_SAMPLE)
    xr1_errors = pd.read_parquet(xr1.ERROR_SITES)
    xr1_coverage = pd.read_parquet(xr1.P0_COVERAGE)
    error_frames: list[pd.DataFrame] = []
    link_frames: list[pd.DataFrame] = []
    ceiling_rows: list[dict[str, Any]] = []
    reproduction: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        documents = set(sample[sample["environment"] == name]["document_id"].astype(str))
        bundles = s14._document_bundles(spec["corpus"])
        roles = s14.partition_of(spec["corpus"], sorted(bundles))
        spans, _record = s14._canonical_spans(spec, bundles)
        chosen = {pair: group for pair, group in spans.items() if pair[0] in documents and group}
        alignment, indexes, by_span, by_alignment, objects = xc1._alignment_population(
            spec, chosen, bundles, roles
        )
        errors = alignment[alignment["evaluable"] & (alignment["d_before"] > 0)]
        error_ids = set(errors["align_site_id"].astype(str))
        texts = {
            f"{name}|{site.site_id}": (str(site.ocr_text), str(site.gt_text))
            for _p, site in objects
        }
        kinds = errors["site_kind"].astype(str).tolist()
        ocr = [texts[str(a)][0] for a in errors["align_site_id"]]
        gt = [texts[str(a)][1] for a in errors["align_site_id"]]
        table = pd.DataFrame(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "base_engine": spec["base_engine"],
                "domain": xr1._domain(spec["corpus"]),
                "align_site_id": errors["align_site_id"].astype(str).to_numpy(),
                "document_id": errors["document_id"].astype(str).to_numpy(),
                "site_kind": kinds,
                "d_before": errors["d_before"].to_numpy(dtype=np.int64),
                "subtype": [
                    gen1.substitution_subtype(o, g)
                    if k == "substitution"
                    else gen1.segmentation_subtype(o, g)
                    if k == "segmentation"
                    else k
                    for o, g, k in zip(ocr, gt, kinds, strict=True)
                ],
            }
        )
        mine = lattice[lattice["environment"] == name]
        sites = mine.assign(site_id=_bare(mine["site_id"]))[
            ["site_id", "document_id", "engine_id", "anchor_kind", "anchor_ref"]
        ]
        links = cg1._site_links(name, sites, indexes, by_span, by_alignment, AnchorKind)
        links = links.assign(is_error=links["align_site_id"].astype(str).isin(error_ids))
        link_frames.append(links)

        # P0 through XR1's own call on P0's own rows, and through its lattice sites: equal.
        own = p0[p0["environment"] == name]
        own_links = cg1._site_links(
            name,
            own.assign(site_id=_bare(own["site_id"])),
            indexes,
            by_span,
            by_alignment,
            AnchorKind,
        )
        reached_direct = set(own_links["align_site_id"].astype(str)) & error_ids
        via_lattice = links[links["site_id"].isin(set(own["lattice_site_id"].astype(str)))]
        reached_lattice = set(via_lattice["align_site_id"].astype(str)) & error_ids
        theirs = xr1_errors[xr1_errors["environment"] == name]
        columns = ["align_site_id", "site_kind", "d_before", "subtype"]
        same_errors = bool(
            table[columns]
            .astype(str)
            .sort_values("align_site_id")
            .reset_index(drop=True)
            .equals(theirs[columns].astype(str).sort_values("align_site_id").reset_index(drop=True))
        )
        covered = xr1_coverage[
            (xr1_coverage["environment"] == name) & xr1_coverage["in_sampled_documents"]
        ]
        xr1_reached = set(covered[covered["covered_by_p0"]]["align_site_id"].astype(str))
        reproduction.append(
            {
                "environment": name,
                "error_sites": len(table),
                "error_table_identical_to_xr1": same_errors,
                "p0_reached": len(reached_direct),
                "p0_reach_identical_to_xr1": reached_direct == xr1_reached,
                "p0_reach_through_lattice_identical": reached_lattice == reached_direct,
            }
        )
        if not (
            same_errors and reached_direct == xr1_reached and reached_lattice == reached_direct
        ):
            raise PhaseError(f"{name}: P0 or the error table does not reproduce SGV-XR1")

        representable = set(links[links["is_error"]]["align_site_id"].astype(str))
        pages = gen1._page_index(spec)
        extra = pd.concat(
            [cross_line_gaps(pages[pair], pair[0], pair[1]) for pair in sorted(chosen)],
            ignore_index=True,
        )
        extra_links = cg1._site_links(name, extra, indexes, by_span, by_alignment, AnchorKind)
        with_cross = representable | (set(extra_links["align_site_id"].astype(str)) & error_ids)
        table["representable"] = table["align_site_id"].isin(representable)
        table["representable_with_cross_line_gaps"] = table["align_site_id"].isin(with_cross)
        table["reached_by_p0"] = table["align_site_id"].isin(reached_direct)
        error_frames.append(table)
        ceiling_rows.append(
            {
                "environment": name,
                "error_sites": len(table),
                "representable": len(representable),
                "ceiling": _ratio(len(representable), len(table)),
                "representable_with_cross_line_gaps": len(with_cross),
                "ceiling_with_cross_line_gaps": _ratio(len(with_cross), len(table)),
                "by_kind": {
                    kind: {
                        "error_sites": int((table["site_kind"] == kind).sum()),
                        "representable": int(
                            table[table["site_kind"] == kind]["representable"].sum()
                        ),
                    }
                    for kind in ERROR_KINDS
                },
            }
        )
        print(
            f"  {name}: {len(table)} error sites, {len(representable)} representable, "
            f"P0 reaches {len(reached_direct)} ({time.monotonic() - began:.0f}s)",
            flush=True,
        )

    errors = pd.concat(error_frames, ignore_index=True)
    links = pd.concat(link_frames, ignore_index=True)
    _write_parquet_once(ERROR_SITES, errors)
    _write_parquet_once(PROPOSAL_ERROR_LINKS, links)
    by_kind = {
        kind: {
            "error_sites": int((errors["site_kind"] == kind).sum()),
            "representable": int(errors[errors["site_kind"] == kind]["representable"].sum()),
            "ceiling": _ratio(
                int(errors[errors["site_kind"] == kind]["representable"].sum()),
                int((errors["site_kind"] == kind).sum()),
            ),
            "name": KIND_NAMES[kind],
        }
        for kind in ERROR_KINDS
    }
    _write_json_once(
        LATTICE_CEILING,
        {
            **_analysis_envelope("lattice_ceiling"),
            "definition": (
                "OCR error sites linked to at least one lattice site by SGV-CG1's frozen links / "
                "evaluable OCR error sites on the sampled documents"
            ),
            "error_sites": len(errors),
            "representable": int(errors["representable"].sum()),
            "ceiling_pooled": _ratio(int(errors["representable"].sum()), len(errors)),
            "ceiling_mean": float(np.mean([r["ceiling"] for r in ceiling_rows])),
            "by_kind": by_kind,
            "by_environment": ceiling_rows,
            "cross_line_gap_diagnostic": {
                "representable": int(errors["representable_with_cross_line_gaps"].sum()),
                "ceiling_pooled": _ratio(
                    int(errors["representable_with_cross_line_gaps"].sum()), len(errors)
                ),
                "note": (
                    "gaps between the last token of a line and the first of the next. No arm can "
                    "propose one (P0's boundary rule is off and LP1 scans one line at a time); "
                    "measured, not proposed, and never used to change the lattice"
                ),
            },
            "p0_reproduction": reproduction,
            "enters_a_primary_proposal": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"link: {len(errors)} error sites, {len(links)} links; ceiling "
        f"{_ratio(int(errors['representable'].sum()), len(errors)):.4f}"
    )
    return 0


# ------------------------------------------------------------------ localization metrics


@dataclass(frozen=True)
class EnvironmentErrors:
    """One environment's OCR error sites and which lattice sites reach each, by frozen links."""

    name: str
    domain: str
    ids: np.ndarray
    kind: np.ndarray
    subtype: np.ndarray
    document: np.ndarray
    representable: np.ndarray
    p0: np.ndarray
    links: dict[str, np.ndarray]

    @property
    def count(self) -> int:
        return int(self.ids.size)

    def reach(self, sites: set[str]) -> np.ndarray:
        """Which error sites at least one of `sites` reaches."""
        mask = np.zeros(self.count, dtype=bool)
        for site in sites:
            hit = self.links.get(site)
            if hit is not None:
                mask[hit] = True
        return mask

    def linked(self, sites: set[str]) -> int:
        """How many of `sites` reach at least one error site."""
        return sum(1 for site in sites if site in self.links)


def load_errors() -> dict[str, EnvironmentErrors]:
    errors = pd.read_parquet(ERROR_SITES)
    links = pd.read_parquet(PROPOSAL_ERROR_LINKS)
    links = links[links["is_error"]]
    views: dict[str, EnvironmentErrors] = {}
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
        views[name] = EnvironmentErrors(
            name=name,
            domain=xr1._domain(spec["corpus"]),
            ids=ids,
            kind=block["site_kind"].astype(str).to_numpy(),
            subtype=block["subtype"].astype(str).to_numpy(),
            document=block["document_id"].astype(str).to_numpy(),
            representable=block["representable"].to_numpy(dtype=bool),
            p0=block["reached_by_p0"].to_numpy(dtype=bool),
            links={k: np.asarray(sorted(set(v)), dtype=np.int64) for k, v in grouped.items()},
        )
    return views


Selections = dict[str, dict[str, set[str]]]
NATURAL = "natural"
MATCHED = "matched"


def arm_key(arm: str, budget: str | float) -> str:
    return f"{arm}@{budget}"


def build_selections(parsed: pd.DataFrame, environments: Sequence[str]) -> Selections:
    """Every arm's proposal set per environment: natural, matched budget, each fraction, union."""

    def pick(method: str, column: str | None = None) -> dict[str, set[str]]:
        rows = parsed[parsed["method"] == method]
        if column is not None:
            rows = rows[rows[column].astype(bool)]
        found = {str(e): set(g["site_id"].astype(str)) for e, g in rows.groupby("environment")}
        return {name: found.get(name, set()) for name in environments}

    matched = budget_column(PRIMARY_FRACTION)
    out: Selections = {}
    for arm in (L0, L1, L2):
        out[arm_key(arm, NATURAL)] = pick(arm)
        out[arm_key(arm, MATCHED)] = pick(arm, matched)
        for fraction in BUDGET_FRACTIONS:
            out[arm_key(arm, fraction)] = pick(arm, budget_column(fraction))
    for budget in (NATURAL, MATCHED):
        out[arm_key(L3, budget)] = {
            name: out[arm_key(L0, NATURAL)][name] | out[arm_key(L2, budget)][name]
            for name in environments
        }
    for arm in (L0, L1, L2, C3):
        out[arm_key(arm, "subset")] = pick(arm, "in_subset_budget")
    return out


def _rate(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def recall_of(
    views: dict[str, EnvironmentErrors],
    chosen: dict[str, set[str]],
    mask: Callable[[EnvironmentErrors], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Localization recall: per environment, the mean over environments, and pooled."""
    per_environment: dict[str, float] = {}
    reached = total = 0
    for name, view in views.items():
        keep = np.ones(view.count, dtype=bool) if mask is None else mask(view)
        hit = view.reach(chosen.get(name, set()))[keep]
        per_environment[name] = _rate(hit)
        reached += int(hit.sum())
        total += int(keep.sum())
    return {
        "mean": float(np.mean(list(per_environment.values()))),
        "pooled": _ratio(reached, total),
        "reached": reached,
        "error_sites": total,
        "per_environment": per_environment,
    }


def precision_of(
    views: dict[str, EnvironmentErrors], chosen: dict[str, set[str]]
) -> dict[str, Any]:
    """Site-proposal precision: proposals linked to an OCR error site over proposals."""
    per_environment: dict[str, float] = {}
    linked = proposals = 0
    for name, view in views.items():
        sites = chosen.get(name, set())
        hit = view.linked(sites)
        per_environment[name] = _ratio(hit, len(sites))
        linked += hit
        proposals += len(sites)
    return {
        "proposals": proposals,
        "error_linked_proposals": linked,
        "unlinked_proposals": proposals - linked,
        "precision_pooled": _ratio(linked, proposals),
        "false_proposal_rate": _ratio(proposals - linked, proposals),
        "precision_mean": float(np.mean(list(per_environment.values()))),
        "per_environment": per_environment,
    }


def miss_recovery_of(
    views: dict[str, EnvironmentErrors], chosen: dict[str, set[str]], kind: str | None = None
) -> dict[str, Any]:
    """Of the error sites P0 does not reach, the share `chosen` reaches. Pooled is primary."""
    per_environment: dict[str, float] = {}
    recovered = missed = 0
    for name, view in views.items():
        keep = ~view.p0 if kind is None else (~view.p0 & (view.kind == kind))
        hit = view.reach(chosen.get(name, set()))[keep]
        per_environment[name] = _rate(hit)
        recovered += int(hit.sum())
        missed += int(keep.sum())
    return {
        "pooled": _ratio(recovered, missed),
        "recovered": recovered,
        "p0_missed": missed,
        "mean": float(np.mean(list(per_environment.values()))),
        "per_environment": per_environment,
    }


def burden_of(chosen: dict[str, set[str]], inventory: dict[str, Any]) -> dict[str, Any]:
    rows = {r["environment"]: r for r in inventory["environments"]}
    proposals = sum(len(v) for v in chosen.values())
    totals = {
        k: sum(int(r[k]) for r in rows.values())
        for k in ("documents", "pages", "ocr_lines", "ocr_tokens")
    }
    return {
        "proposals": proposals,
        "per_document": _ratio(proposals, totals["documents"]),
        "per_page": _ratio(proposals, totals["pages"]),
        "per_line": _ratio(proposals, totals["ocr_lines"]),
        "per_1000_ocr_tokens": 1000.0 * _ratio(proposals, totals["ocr_tokens"]),
        "per_environment": {
            name: {
                "proposals": len(chosen.get(name, set())),
                "per_1000_ocr_tokens": 1000.0
                * _ratio(len(chosen.get(name, set())), int(rows[name]["ocr_tokens"])),
            }
            for name in rows
        },
    }


def _gain(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    per_environment = {
        n: a["per_environment"][n] - b["per_environment"][n] for n in a["per_environment"]
    }
    return {
        "mean": a["mean"] - b["mean"],
        "pooled": a["pooled"] - b["pooled"],
        "per_environment": per_environment,
        "environments_at_breadth_margin": int(
            sum(v >= BREADTH_ENVIRONMENT_MARGIN for v in per_environment.values())
        ),
        "environments_improved": int(sum(v > 0 for v in per_environment.values())),
    }


def _kind_recall(
    views: dict[str, EnvironmentErrors], chosen: dict[str, set[str]], kind: str
) -> dict[str, Any]:
    return recall_of(views, chosen, lambda v: v.kind == kind)


def _subtype_recall(
    views: dict[str, EnvironmentErrors], chosen: dict[str, set[str]], subtype: str
) -> dict[str, Any]:
    return recall_of(views, chosen, lambda v: v.subtype == subtype)


SEGMENTATION_SUBTYPES = (
    "missing_space_merge",
    "spurious_space_split",
    "boundary_with_character_errors",
    "boundary_placement",
)
HISTORICAL_GLYPH = "historical_glyph"
REPORTED = (
    (L0, NATURAL),
    (L1, MATCHED),
    (L2, MATCHED),
    (L3, MATCHED),
    (L1, NATURAL),
    (L2, NATURAL),
    (L3, NATURAL),
)
# XR1's measured rates, used only for the bottleneck ladder's labelled estimate.
XR1_REPAIR_SHARE_OF_REACHED = 0.1035
GEN1_KNOWN_SITE_EXACT = 0.2080


def _population_rows(
    views: dict[str, EnvironmentErrors], parsed: pd.DataFrame
) -> list[dict[str, Any]]:
    """Section 38: every denominator, per environment."""
    inventory = {r["environment"]: r for r in cc_read_json(ENVIRONMENT_INVENTORY)["environments"]}
    compliance = pd.DataFrame()
    rows = []
    for name, view in views.items():
        row = inventory[name]
        block = parsed[parsed["environment"] == name]
        rows.append(
            {
                "environment": name,
                "documents": row["documents"],
                "pages": row["pages"],
                "ocr_lines": row["ocr_lines"],
                "ocr_tokens": row["ocr_tokens"],
                "lattice_sites": row["lattice_sites"],
                "evaluable_error_sites": view.count,
                "representable_error_sites": int(view.representable.sum()),
                "p0_proposals": row["p0_proposals"],
                "l1_valid_proposals": int((block["method"] == L1).sum()),
                "l2_valid_proposals": int((block["method"] == L2).sum()),
                "matched_budget": row["p0_proposals"],
                "l1_matched_budget_proposals": int(
                    ((block["method"] == L1) & block[budget_column(PRIMARY_FRACTION)]).sum()
                ),
                "l2_matched_budget_proposals": int(
                    ((block["method"] == L2) & block[budget_column(PRIMARY_FRACTION)]).sum()
                ),
            }
        )
    del compliance
    return rows


def run_localize() -> int:
    """Sections 15-32 and 38-39: every localization artifact from the frozen selections."""
    started = time.monotonic()
    _require(PROPOSAL_ERROR_LINKS, "link")
    outputs = (
        NATURAL_RESULTS,
        MATCHED_BUDGET_RESULTS,
        RECALL_BUDGET_CURVE,
        P0_MISS_RECOVERY,
        ERROR_TYPE_ANALYSIS,
        OMISSION_ANALYSIS,
        SEGMENTATION_ANALYSIS,
        HISTORICAL_GLYPH_ANALYSIS,
        PROPOSAL_PRECISION,
        PROPOSAL_BURDEN,
        PROPOSAL_OVERLAP,
        UNION_LOCALIZATION,
        ENVIRONMENT_ANALYSIS,
        DOMAIN_ANALYSIS,
        BOTTLENECK,
        RUNTIME_COST,
    )
    for path in outputs:
        _forbid(path)
    views = load_errors()
    parsed = pd.read_parquet(PARSED_PROPOSALS)
    chosen = build_selections(parsed, list(views))
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    recall = {arm_key(a, b): recall_of(views, chosen[arm_key(a, b)]) for a, b in REPORTED}
    precision = {arm_key(a, b): precision_of(views, chosen[arm_key(a, b)]) for a, b in REPORTED}
    burden = {arm_key(a, b): burden_of(chosen[arm_key(a, b)], inventory) for a, b in REPORTED}
    p0 = recall[arm_key(L0, NATURAL)]

    def block(arm: str, budget: str) -> dict[str, Any]:
        key = arm_key(arm, budget)
        return {
            "recall": recall[key],
            "gain_over_p0": _gain(recall[key], p0),
            "proposal_precision": precision[key]["precision_pooled"],
            "error_linked_proposals": precision[key]["error_linked_proposals"],
            "unlinked_proposals": precision[key]["unlinked_proposals"],
            "proposals": precision[key]["proposals"],
            "per_1000_ocr_tokens": burden[key]["per_1000_ocr_tokens"],
        }

    _write_json_once(
        NATURAL_RESULTS,
        {
            **_analysis_envelope("natural_localization_results"),
            "note": (
                "each arm's whole valid output under the frozen prompt; secondary to the budget"
            ),
            "arms": {arm: block(arm, NATURAL) for arm in (L0, L1, L2, L3)},
            "error_sites": p0["error_sites"],
        },
    )
    fill = cc_read_json(PROPOSAL_COMPLIANCE)["matched_budget_fill"]
    _write_json_once(
        MATCHED_BUDGET_RESULTS,
        {
            **_analysis_envelope("matched_budget_results"),
            "budget": "per environment, P0's proposal count on the sampled documents",
            "arms": {
                L0: block(L0, NATURAL),
                L1: block(L1, MATCHED),
                L2: block(L2, MATCHED),
                L3: block(L3, MATCHED),
            },
            "image_vs_text": _gain(recall[arm_key(L2, MATCHED)], recall[arm_key(L1, MATCHED)]),
            "fill": {
                arm: {
                    "budget": sum(v["budget"] for v in fill[arm].values()),
                    "selected": sum(v["selected"] for v in fill[arm].values()),
                    "natural": sum(v["natural"] for v in fill[arm].values()),
                    "environments_filled": int(
                        sum(v["selected"] == v["budget"] for v in fill[arm].values())
                    ),
                }
                for arm in (L0, L1, L2)
            },
            "error_sites": p0["error_sites"],
        },
    )
    curve = {}
    for arm in (L0, L1, L2):
        curve[arm] = {
            f"{fraction:.2f}": {
                "recall": recall_of(views, chosen[arm_key(arm, fraction)]),
                "proposals": sum(len(v) for v in chosen[arm_key(arm, fraction)].values()),
                "precision": precision_of(views, chosen[arm_key(arm, fraction)])[
                    "precision_pooled"
                ],
            }
            for fraction in BUDGET_FRACTIONS
        }
    _write_json_once(
        RECALL_BUDGET_CURVE,
        {
            **_analysis_envelope("recall_budget_curve"),
            "fractions_of_p0": list(BUDGET_FRACTIONS),
            "p0_budget": {r["environment"]: r["p0_proposals"] for r in inventory["environments"]},
            "arms": curve,
            "note": (
                "an arm below a budget keeps all of its proposals, so its curve flattens there; P0 "
                "above 1.0 keeps all of its own"
            ),
        },
    )
    recovery = {
        arm_key(a, b): miss_recovery_of(views, chosen[arm_key(a, b)])
        for a, b in ((L1, MATCHED), (L2, MATCHED), (L1, NATURAL), (L2, NATURAL))
    }
    missed_ids = {n: set(v.ids[~v.p0]) for n, v in views.items()}
    p0_reached_ids = {n: set(v.ids[v.p0]) for n, v in views.items()}
    _write_json_once(
        P0_MISS_RECOVERY,
        {
            **_analysis_envelope("p0_miss_recovery"),
            "definition": "error sites P0 does not reach that the arm reaches / those P0 misses",
            "p0_missed": int(sum(len(v) for v in missed_ids.values())),
            "arms": recovery,
            "by_kind": {
                kind: {
                    arm_key(L2, MATCHED): miss_recovery_of(
                        views, chosen[arm_key(L2, MATCHED)], kind
                    ),
                    arm_key(L1, MATCHED): miss_recovery_of(
                        views, chosen[arm_key(L1, MATCHED)], kind
                    ),
                }
                for kind in ERROR_KINDS
            },
            "target": MISS_RECOVERY_TARGET,
            "missed_set_excludes_p0_reached": all(
                not (missed_ids[n] & p0_reached_ids[n]) for n in views
            ),
        },
    )
    kinds = {
        kind: {
            "error_sites": int(sum((v.kind == kind).sum() for v in views.values())),
            "name": KIND_NAMES[kind],
            "recall": {
                arm_key(a, b): _kind_recall(views, chosen[arm_key(a, b)], kind)["pooled"]
                for a, b in REPORTED
            },
        }
        for kind in ERROR_KINDS
    }
    for row in kinds.values():
        row["l2_gain_over_p0"] = (
            row["recall"][arm_key(L2, MATCHED)] - row["recall"][arm_key(L0, NATURAL)]
        )
        row["l1_gain_over_p0"] = (
            row["recall"][arm_key(L1, MATCHED)] - row["recall"][arm_key(L0, NATURAL)]
        )
        row["union_gain_over_p0"] = (
            row["recall"][arm_key(L3, MATCHED)] - row["recall"][arm_key(L0, NATURAL)]
        )
    weak_reached = [k for k in WEAK_TYPES if kinds[k]["l2_gain_over_p0"] >= WEAK_TYPE_MARGIN]
    _write_json_once(
        ERROR_TYPE_ANALYSIS,
        {
            **_analysis_envelope("error_type_analysis"),
            "pooled": "recall pooled over the sampled error sites of each type",
            "by_kind": kinds,
            "weak_types": list(WEAK_TYPES),
            "weak_types_reaching_margin": weak_reached,
            "margin": WEAK_TYPE_MARGIN,
            "taxonomy_note": (
                "the frozen taxonomy's deletion is an omission, its insertion spurious"
            ),
        },
    )
    omission = {
        arm_key(a, b): {
            "all": _kind_recall(views, chosen[arm_key(a, b)], "deletion")["pooled"],
            "representable": recall_of(
                views, chosen[arm_key(a, b)], lambda v: (v.kind == "deletion") & v.representable
            )["pooled"],
        }
        for a, b in REPORTED
    }
    _write_json_once(
        OMISSION_ANALYSIS,
        {
            **_analysis_envelope("omission_analysis"),
            "omission_sites": kinds["deletion"]["error_sites"],
            "representable_omission_sites": int(
                sum(((v.kind == "deletion") & v.representable).sum() for v in views.values())
            ),
            "recall": omission,
            "p0_miss_recovery": {
                arm_key(L2, MATCHED): miss_recovery_of(
                    views, chosen[arm_key(L2, MATCHED)], "deletion"
                ),
            },
            "note": (
                "an omission is reachable only through a gap site; the model never learns which"
            ),
        },
    )
    segmentation = {
        subtype: {
            "error_sites": int(sum((v.subtype == subtype).sum() for v in views.values())),
            "recall": {
                arm_key(a, b): _subtype_recall(views, chosen[arm_key(a, b)], subtype)["pooled"]
                for a, b in REPORTED
            },
        }
        for subtype in SEGMENTATION_SUBTYPES
    }
    _write_json_once(
        SEGMENTATION_ANALYSIS,
        {
            **_analysis_envelope("segmentation_analysis"),
            "by_subtype": segmentation,
            "taxonomy": "GEN1's segmentation_subtype, decided from the OCR and truth strings",
        },
    )
    xr1_types = cc_read_json(xr1.ERROR_TYPE_ANALYSIS)["historical_glyph"]
    _write_json_once(
        HISTORICAL_GLYPH_ANALYSIS,
        {
            **_analysis_envelope("historical_glyph_analysis"),
            "error_sites": int(sum((v.subtype == HISTORICAL_GLYPH).sum() for v in views.values())),
            "localization_recall": {
                arm_key(a, b): _subtype_recall(views, chosen[arm_key(a, b)], HISTORICAL_GLYPH)[
                    "pooled"
                ]
                for a, b in REPORTED
            },
            "xr1_correction_on_the_same_sites": {
                "g2_top1_repaired": xr1_types[xr1.G2]["repaired"],
                "g2_top5_repaired": xr1_types[f"{xr1.G2}@5"]["repaired"],
                "source": _relative(xr1.ERROR_TYPE_ANALYSIS),
            },
            "reading": "finding a site is measured here; XR1 measured correcting it",
        },
    )
    _write_json_once(
        PROPOSAL_PRECISION,
        {
            **_analysis_envelope("proposal_precision"),
            "arms": precision,
            "definition": (
                "proposals linked to an OCR error site / proposals; no correction applied"
            ),
        },
    )
    _write_json_once(
        PROPOSAL_BURDEN,
        {
            **_analysis_envelope("proposal_burden"),
            "arms": burden,
            "population": {k: inventory[k] for k in ("documents", "ocr_lines", "ocr_tokens")},
        },
    )
    overlap = {}
    for budget in (MATCHED, NATURAL):
        both = p0_only = image_only = neither = 0
        for name, view in views.items():
            a = view.p0
            b = view.reach(chosen[arm_key(L2, budget)][name])
            both += int((a & b).sum())
            p0_only += int((a & ~b).sum())
            image_only += int((~a & b).sum())
            neither += int((~a & ~b).sum())
        union = both + p0_only + image_only
        overlap[budget] = {
            "both": both,
            "p0_only": p0_only,
            "image_only": image_only,
            "neither": neither,
            "jaccard": _ratio(both, union),
            "union_reached": union,
            "union_recall_pooled": _ratio(union, both + p0_only + image_only + neither),
        }
    _write_json_once(
        PROPOSAL_OVERLAP,
        {
            **_analysis_envelope("proposal_overlap"),
            "p0_vs_l2": overlap,
            "unit": "sampled OCR error sites reached by each arm",
        },
    )
    union_rows = {}
    for budget in (MATCHED, NATURAL):
        key = arm_key(L3, budget)
        p0_sites = chosen[arm_key(L0, NATURAL)]
        l2_sites = chosen[arm_key(L2, budget)]
        union_rows[budget] = {
            "recall": recall[key],
            "gain_over_p0": _gain(recall[key], p0),
            "proposals": precision[key]["proposals"],
            "p0_proposals": sum(len(v) for v in p0_sites.values()),
            "incremental_proposals": sum(len(l2_sites[n] - p0_sites[n]) for n in views),
            "incremental_error_sites": recall[key]["reached"] - p0["reached"],
            "attribution": {
                "p0_only": sum(len(p0_sites[n] - l2_sites[n]) for n in views),
                "l2_only": sum(len(l2_sites[n] - p0_sites[n]) for n in views),
                "both": sum(len(p0_sites[n] & l2_sites[n]) for n in views),
            },
            "burden_ratio_to_p0": _ratio(
                precision[key]["proposals"], sum(len(v) for v in p0_sites.values())
            ),
        }
    _write_json_once(
        UNION_LOCALIZATION,
        {
            **_analysis_envelope("union_localization"),
            "union": union_rows,
            "is_production_method": False,
            "note": "reported independently; never overrides the primary outcome except as rule B",
        },
    )
    _write_json_once(
        ENVIRONMENT_ANALYSIS,
        {
            **_analysis_envelope("environment_analysis"),
            "population": _population_rows(views, parsed),
            "recall_by_environment": {k: v["per_environment"] for k, v in recall.items()},
        },
    )
    domains = {}
    for domain in ("modern_forms", "historical_print"):
        mine = {n: v for n, v in views.items() if v.domain == domain}
        domains[domain] = {
            "error_sites": int(sum(v.count for v in mine.values())),
            "recall": {
                arm_key(a, b): recall_of(mine, chosen[arm_key(a, b)])["pooled"] for a, b in REPORTED
            },
        }
    _write_json_once(
        DOMAIN_ANALYSIS,
        {
            **_analysis_envelope("domain_analysis"),
            "by_domain": domains,
            "domains": {"modern_forms": "FUNSD", "historical_print": "SBB"},
        },
    )
    ceiling = cc_read_json(LATTICE_CEILING)
    union_pooled = recall[arm_key(L3, MATCHED)]["pooled"]
    _write_json_once(
        BOTTLENECK,
        {
            **_analysis_envelope("bottleneck_decomposition"),
            "ladder": {
                "all_ocr_error_sites": 1.0,
                "representable_by_the_lattice": ceiling["ceiling_pooled"],
                "reached_by_p0": p0["pooled"],
                "reached_by_l2_matched_budget": recall[arm_key(L2, MATCHED)]["pooled"],
                "reached_by_p0_plus_l2": union_pooled,
            },
            "estimate_not_a_result": {
                "correctable_share_of_all_errors_estimate": union_pooled
                * XR1_REPAIR_SHARE_OF_REACHED,
                "rate_used": XR1_REPAIR_SHARE_OF_REACHED,
                "rate_source": "SGV-XR1 G2 repair share of P0-reached error sites",
                "upper_reference": GEN1_KNOWN_SITE_EXACT,
                "caveat": (
                    "assumes the image generator repairs newly reached sites at XR1's rate; no "
                    "correction was generated in LP1"
                ),
            },
        },
    )
    sidecars = [
        cc_read_json(_sidecar(_shard_path(a, s["environment"])))
        for a in MODEL_ARMS
        for s in environment_specs()
    ]
    runtime = {
        arm: {
            "requests": int(sum(s["requests"] for s in sidecars if s["method"] == arm)),
            "wall_clock_hours": sum(s["wall_clock_seconds"] for s in sidecars if s["method"] == arm)
            / 3600.0,
            "output_tokens": int(sum(s["output_tokens"] for s in sidecars if s["method"] == arm)),
            "unfinished_decodes": int(
                sum(s["unfinished_decodes"] for s in sidecars if s["method"] == arm)
            ),
        }
        for arm in MODEL_ARMS
    }
    for row in runtime.values():
        row["lines_per_second"] = _ratio(row["requests"], row["wall_clock_hours"] * 3600.0)
    _write_json_once(
        RUNTIME_COST,
        {
            **_analysis_envelope("runtime_cost"),
            "arms": runtime,
            "total_wall_clock_hours": sum(r["wall_clock_hours"] for r in runtime.values()),
            "estimated_total_hours": cc_read_json(COMPUTE_BUDGET)["estimated_total_hours"],
            "clock": "monotonic, which excludes sleep",
        },
    )
    print(
        f"localize: P0 {p0['mean']:.4f}, L1 {recall[arm_key(L1, MATCHED)]['mean']:.4f}, "
        f"L2 {recall[arm_key(L2, MATCHED)]['mean']:.4f} (matched), union "
        f"{recall[arm_key(L3, MATCHED)]['mean']:.4f} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ controls


def _stable_seed(*parts: str) -> int:
    return int(canonical_hash({"seed": LP1_SEED, "parts": list(parts)})[:16], 16)


def random_draws(
    views: dict[str, EnvironmentErrors],
    lattice_sites: dict[str, list[str]],
    counts: dict[str, int],
    label: str,
) -> np.ndarray:
    """Mean-over-environment recall of RANDOM_DRAWS uniform draws of admissible lattice sites."""
    values = np.zeros(RANDOM_DRAWS)
    for draw in range(RANDOM_DRAWS):
        rates = []
        for name, view in views.items():
            pool = lattice_sites[name]
            size = min(int(counts.get(name, 0)), len(pool))
            rng = np.random.default_rng(_stable_seed(label, name, str(draw)))
            picked = {pool[i] for i in rng.choice(len(pool), size=size, replace=False)}
            rates.append(_rate(view.reach(picked)))
        values[draw] = float(np.mean(rates))
    return values


def run_controls() -> int:
    """C3 image shuffle on its subsample, C4 random lattice ranking, and the control summary."""
    started = time.monotonic()
    _require(NATURAL_RESULTS, "localize")
    for path in (IMAGE_SHUFFLE, RANDOM_RANKING, CONTROL_RESULTS):
        _forbid(path)
    views = load_errors()
    parsed = pd.read_parquet(PARSED_PROPOSALS)
    chosen = build_selections(parsed, list(views))
    lattice = pd.read_parquet(SITE_LATTICE_INVENTORY)
    control = pd.read_parquet(CONTROL_CONTEXTS)
    subset_lines = set(control["line_uid"].astype(str))
    on_subset = lattice[lattice["line_uid"].isin(subset_lines)]
    subset_sites = {
        name: set(on_subset[on_subset["environment"] == name]["site_id"].astype(str))
        for name in views
    }

    def on_lines(view: EnvironmentErrors) -> np.ndarray:
        return view.reach(subset_sites[view.name])

    shuffle: dict[str, Any] = {}
    for budget, suffix in (("subset_budget", "subset"), ("natural", NATURAL)):
        rows = {}
        for arm in (L0, L1, L2, C3):
            if suffix == "subset":
                sites = chosen[arm_key(arm, "subset")]
            else:
                picked = parsed[(parsed["method"] == arm) & parsed["on_shuffle_subset"]]
                sites = {
                    name: set(picked[picked["environment"] == name]["site_id"].astype(str))
                    for name in views
                }
            rows[arm] = {
                "recall": recall_of(views, sites, on_lines),
                "proposals": sum(len(v) for v in sites.values()),
            }
        l1 = rows[L1]["recall"]["mean"]
        l2 = rows[L2]["recall"]["mean"]
        c3 = rows[C3]["recall"]["mean"]
        gain = l2 - l1
        survivors = reached = 0
        for name, view in views.items():
            keep = on_lines(view)
            by_l2 = view.reach(rows_sites(chosen, parsed, L2, suffix, name))[keep]
            by_c3 = view.reach(rows_sites(chosen, parsed, C3, suffix, name))[keep]
            reached += int(by_l2.sum())
            survivors += int((by_l2 & by_c3).sum())
        shuffle[budget] = {
            "arms": rows,
            "l2_minus_l1": gain,
            "l2_minus_c3": l2 - c3,
            "retention_of_visual_gain": (c3 - l1) / gain if gain > 0 else None,
            "l2_reached_error_sites": reached,
            "l2_reached_surviving_under_shuffle": survivors,
            "survival_share": _ratio(survivors, reached),
        }
    subset_errors = int(sum(on_lines(v).sum() for v in views.values()))
    _write_json_once(
        IMAGE_SHUFFLE,
        {
            **_analysis_envelope("image_shuffle_control"),
            "subset_lines": len(subset_lines),
            "subset_error_sites": subset_errors,
            "subset_error_definition": (
                "error sites some lattice site on a subsample line reaches, by the frozen links"
            ),
            "primary": "subset_budget",
            "results": shuffle,
            "retention_max": SHUFFLE_RETENTION_MAX,
            "donor_never_own_document": bool(
                (control["document_id"] != control["donor_document_id"]).all()
            ),
            "donor_crop_never_own_crop": bool(
                (control["crop_sha256"] != control["own_crop_sha256"]).all()
            ),
        },
    )
    pools = {
        name: sorted(lattice[lattice["environment"] == name]["site_id"].astype(str))
        for name in views
    }
    l2_counts = {n: len(v) for n, v in chosen[arm_key(L2, MATCHED)].items()}
    p0_counts = {n: len(v) for n, v in chosen[arm_key(L0, NATURAL)].items()}
    at_l2 = random_draws(views, pools, l2_counts, "c4-at-l2-matched")
    at_p0 = random_draws(views, pools, p0_counts, "c4-at-p0-budget")
    recall_p0 = recall_of(views, chosen[arm_key(L0, NATURAL)])["mean"]
    recall_l2 = recall_of(views, chosen[arm_key(L2, MATCHED)])["mean"]

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "p2_5": float(np.percentile(values, 2.5)),
            "p97_5": float(np.percentile(values, 97.5)),
        }

    _write_json_once(
        RANDOM_RANKING,
        {
            **_analysis_envelope("random_ranking_control"),
            "draws": RANDOM_DRAWS,
            "rule": "uniform admissible lattice sites without replacement, per environment, seeded",
            "at_l2_matched_count": {**summary(at_l2), "proposals": sum(l2_counts.values())},
            "at_p0_budget": {**summary(at_p0), "proposals": sum(p0_counts.values())},
            "p0_recall": recall_p0,
            "l2_matched_recall": recall_l2,
            "enters_method_selection": False,
        },
    )
    ceiling = cc_read_json(LATTICE_CEILING)
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "C0_lattice_ceiling": ceiling["ceiling_pooled"],
            "C1_p0_recall": recall_p0,
            "C2_text_only_recall": recall_of(views, chosen[arm_key(L1, MATCHED)])["mean"],
            "C3_shuffle_retention": shuffle["subset_budget"]["retention_of_visual_gain"],
            "C4_random_recall_at_p0_budget": float(at_p0.mean()),
            "C4_random_recall_at_l2_count": float(at_l2.mean()),
            "p0_reproduces_xr1": all(
                r["p0_reach_identical_to_xr1"] and r["error_table_identical_to_xr1"]
                for r in ceiling["p0_reproduction"]
            ),
        },
    )
    print(
        f"controls: shuffle retention {shuffle['subset_budget']['retention_of_visual_gain']}, "
        f"random {at_p0.mean():.4f} at P0's budget ({time.monotonic() - started:.0f}s)"
    )
    return 0


def rows_sites(
    chosen: Selections, parsed: pd.DataFrame, arm: str, suffix: str, name: str
) -> set[str]:
    """An arm's sites on the shuffle subsample, under its subset budget or its natural output."""
    if suffix == "subset":
        return chosen[arm_key(arm, "subset")].get(name, set())
    rows = parsed[
        (parsed["method"] == arm) & parsed["on_shuffle_subset"] & (parsed["environment"] == name)
    ]
    return set(rows["site_id"].astype(str))


# ------------------------------------------------------------------ statistics

PRIMARY_FAMILY = ("P1_l2_vs_p0", "P2_l2_vs_l1", "P3_miss_recovery_vs_target")
MECHANISM_FAMILY = ("M1_l2_vs_shuffle",)


def reach_frame(
    views: dict[str, EnvironmentErrors],
    a: dict[str, set[str]],
    b: dict[str, set[str]] | float,
    mask: Callable[[EnvironmentErrors], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Paired per-error-site indicators: reached by A, reached by B (or a constant target)."""
    frames = []
    for name, view in views.items():
        keep = np.ones(view.count, dtype=bool) if mask is None else mask(view)
        left = view.reach(a.get(name, set()))[keep].astype(float)
        right = (
            np.full(int(keep.sum()), float(b))
            if isinstance(b, float)
            else view.reach(b.get(name, set()))[keep].astype(float)
        )
        frames.append(
            pd.DataFrame(
                {"environment": name, "document_id": view.document[keep], "a": left, "b": right}
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_stats() -> int:
    """Document-clustered paired bootstrap; Holm within each pre-registered family."""
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    views = load_errors()
    parsed = pd.read_parquet(PARSED_PROPOSALS)
    chosen = build_selections(parsed, list(views))
    p0 = chosen[arm_key(L0, NATURAL)]
    l1 = chosen[arm_key(L1, MATCHED)]
    l2 = chosen[arm_key(L2, MATCHED)]
    comparison = xr1.comparison
    primary = {
        PRIMARY_FAMILY[0]: comparison(reach_frame(views, l2, p0), PRIMARY_FAMILY[0], "primary"),
        PRIMARY_FAMILY[1]: comparison(reach_frame(views, l2, l1), PRIMARY_FAMILY[1], "primary"),
        PRIMARY_FAMILY[2]: comparison(
            reach_frame(views, l2, MISS_RECOVERY_TARGET, lambda v: ~v.p0),
            PRIMARY_FAMILY[2],
            "primary",
            pooled=True,
        ),
    }
    lattice = pd.read_parquet(SITE_LATTICE_INVENTORY)
    subset_lines = set(pd.read_parquet(CONTROL_CONTEXTS)["line_uid"].astype(str))
    on_subset = lattice[lattice["line_uid"].isin(subset_lines)]
    subset_sites = {
        name: set(on_subset[on_subset["environment"] == name]["site_id"].astype(str))
        for name in views
    }
    mechanism = {
        MECHANISM_FAMILY[0]: comparison(
            reach_frame(
                views,
                chosen[arm_key(L2, "subset")],
                chosen[arm_key(C3, "subset")],
                lambda v: v.reach(subset_sites[v.name]),
            ),
            MECHANISM_FAMILY[0],
            "mechanism",
        )
    }
    secondary = {
        "union_vs_p0": comparison(
            reach_frame(views, chosen[arm_key(L3, MATCHED)], p0), "union_vs_p0", "secondary"
        ),
        "l2_natural_vs_p0": comparison(
            reach_frame(views, chosen[arm_key(L2, NATURAL)], p0), "l2_natural_vs_p0", "secondary"
        ),
        "l1_vs_p0": comparison(reach_frame(views, l1, p0), "l1_vs_p0", "secondary"),
    }
    for kind in ERROR_KINDS:
        name = f"{kind}_l2_vs_p0"
        secondary[name] = comparison(
            reach_frame(views, l2, p0, functools.partial(_is_kind, kind=kind)),
            name,
            "secondary",
            pooled=True,
        )
    name = f"{HISTORICAL_GLYPH}_l2_vs_p0"
    secondary[name] = comparison(
        reach_frame(views, l2, p0, lambda v: v.subtype == HISTORICAL_GLYPH),
        name,
        "secondary",
        pooled=True,
    )
    if tuple(primary) != PRIMARY_FAMILY or tuple(mechanism) != MECHANISM_FAMILY:
        raise PhaseError("a family differs from the one pre-registered")
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
            "primary_family": s15.holm(primary),
            "mechanism_family": s15.holm(mechanism),
            "secondary_family": s15.holm(secondary),
            "p3_note": (
                "P3 compares each P0-missed error site's reach by L2 with the 0.25 target, pooled; "
                "its effect is the miss recovery minus 0.25"
            ),
        },
    )
    print(
        "stats: "
        + ", ".join(
            f"{k} {v['effect']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
            for k, v in primary.items()
        )
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


def _is_kind(view: EnvironmentErrors, kind: str) -> np.ndarray:
    return view.kind == kind


# ------------------------------------------------------------------ falsification suite

# Names that carry ground truth, an error label or an evaluation outcome. No function that builds
# a request, a lattice, a crop or a selection may read any of them.
GT_NAMES = frozenset(
    {
        "gt_text",
        "region_gt",
        "gt_length",
        "gt_tokens",
        "gt_span",
        "site_kind",
        "true_error_type",
        "error_type",
        "subtype",
        "align_site_id",
        "d_before",
        "labelable",
        "is_error",
        "reached_by_p0",
        "representable",
        "covered_by_p0",
        "outcome",
        "exact",
        "beneficial",
        "harmful",
        "error_site_label",
        "ERROR_SITES",
        "PROPOSAL_ERROR_LINKS",
        "LATTICE_CEILING",
        "load_errors",
        "_alignment_population",
        "_site_links",
    }
)
P0_NAMES = frozenset({"P0_PROPOSALS", "_p0_requests", "rederive_p0", "map_p0_to_lattice", "L0"})
REQUEST_BUILDERS = (
    "page_lines",
    "page_lattice",
    "scan_lines",
    "render_line",
    "line_box",
    "_quoted",
    "render_line_crop",
    "build_line_contexts",
    "shuffle_subset",
    "shuffle_donors",
    "requests_for",
    "run_line_requests",
    "run_contexts",
    "run_generate",
)
SELECTION_FUNCTIONS = (
    "select_budget",
    "_budget_columns",
    "parse_line",
    "parse_answer",
    "position_of",
    "score_of",
    "run_parse",
)


@functools.lru_cache(maxsize=1)
def _module_tree() -> ast.Module:
    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def names_in(function_name: str) -> set[str]:
    """Every name, attribute and string constant a module-level function's body mentions."""
    for node in _module_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            found: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    found.add(child.id)
                elif isinstance(child, ast.Attribute):
                    found.add(child.attr)
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    found.add(child.value)
            return found
    raise PhaseError(f"no module-level function {function_name}")


def run_negative() -> int:
    """The falsification suite: each test would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    tests: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: Any) -> None:
        tests.append({"test": name, "passed": bool(passed), "evidence": evidence})

    ceiling = cc_read_json(LATTICE_CEILING)
    registry = cc_read_json(SITE_LATTICE_REGISTRY)
    add(
        "t01_p0_reproduces_its_frozen_xr1_coverage",
        all(r["p0_reproduces_xc1_cache"] for r in registry["p0_checks"])
        and all(
            r["p0_reach_identical_to_xr1"] and r["error_table_identical_to_xr1"]
            for r in ceiling["p0_reproduction"]
        ),
        {
            "environments": len(ceiling["p0_reproduction"]),
            "p0_reached": sum(r["p0_reached"] for r in ceiling["p0_reproduction"]),
        },
    )
    rebuilt = _rebuild_lattice()
    stored = pd.read_parquet(SITE_LATTICE_INVENTORY)
    add(
        "t02_the_lattice_is_identical_before_and_after_ground_truth",
        file_sha256(SITE_LATTICE_INVENTORY) == registry["inventory"]["sha256"]
        and canonical_hash(rebuilt.to_json(orient="records"))
        == canonical_hash(stored[rebuilt.columns].to_json(orient="records")),
        {"sites": len(stored), "rebuilt_sites": len(rebuilt)},
    )
    tables = {
        "scan_population": pd.read_parquet(SCAN_POPULATION),
        "line_contexts": pd.read_parquet(LINE_CONTEXTS),
        "control_contexts": pd.read_parquet(CONTROL_CONTEXTS),
        "lattice": stored,
    }
    truth_columns = {
        k: [c for c in FORBIDDEN_INPUT_COLUMNS if c in v.columns] for k, v in tables.items()
    }
    leaks = {name: sorted(names_in(name) & GT_NAMES) for name in REQUEST_BUILDERS}
    add(
        "t03_no_ground_truth_column_enters_request_construction",
        not any(leaks.values()) and not any(truth_columns.values()),
        {"builders_checked": len(leaks), "gt_names_found": leaks, "truth_columns": truth_columns},
    )
    type_names = {"site_kind", "true_error_type", "error_type", "subtype"}
    add(
        "t04_no_error_type_enters_request_construction",
        not any(names_in(n) & type_names for n in REQUEST_BUILDERS),
        {"names_checked": sorted(type_names)},
    )
    prompts = _rebuilt_prompt_hashes()
    add(
        "t05_no_replacement_text_enters_a_request",
        prompts["differing"] == 0
        and not any(names_in(n) & {"gt_text", "region_gt", "gt_tokens"} for n in REQUEST_BUILDERS),
        prompts,
    )
    lines = tables["scan_population"]
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    p0_leaks = {name: sorted(names_in(name) & P0_NAMES) for name in REQUEST_BUILDERS}
    l2_lines = _shard_lines(L2)
    add(
        "t06_p0_does_not_choose_the_lines_the_image_model_scans",
        not any(p0_leaks.values())
        and len(lines) == inventory["ocr_lines"]
        and l2_lines == set(lines[lines["crop_box"].notna()]["line_uid"].astype(str)),
        {"scan_lines": len(lines), "l2_lines": len(l2_lines), "p0_names_found": p0_leaks},
    )
    l1_lines = _shard_lines(L1)
    add(
        "t07_l1_and_l2_scan_exactly_the_same_lines",
        l1_lines == l2_lines,
        {"l1_lines": len(l1_lines), "l2_lines": len(l2_lines)},
    )
    contexts = tables["line_contexts"].merge(
        tables["scan_population"][["line_uid", "line_text"]], on="line_uid", validate="one_to_one"
    )
    same_text = all(
        text == USER_TEMPLATES["text"].format(text=line)
        and image == USER_TEMPLATES["image"].format(text=line)
        for text, image, line in zip(
            contexts["text_prompt"], contexts["image_prompt"], contexts["line_text"], strict=True
        )
    )
    add(
        "t08_l1_and_l2_receive_the_same_text_and_site_ids",
        same_text,
        {"lines": len(contexts)},
    )
    parsed = pd.read_parquet(PARSED_PROPOSALS)
    model = parsed[parsed["method"].isin(MODEL_ARMS)]
    by_line = lines.set_index("line_uid")["site_ids"].to_dict()
    outside = int(
        sum(
            str(s) not in set(by_line[str(u)])
            for u, s in zip(model["line_uid"], model["site_id"], strict=True)
        )
    )
    add(
        "t09_every_model_proposal_names_an_admissible_site_of_its_own_line",
        outside == 0,
        {"model_proposals": len(model), "outside_their_line": outside},
    )
    views = load_errors()
    chosen = build_selections(parsed, list(views))
    l2 = chosen[arm_key(L2, NATURAL)]
    base = recall_of(views, l2)
    duplicates = _duplicated_answers_reparse(lines)
    add(
        "t10_duplicating_proposals_does_not_raise_recall",
        duplicates["answers_checked"] > 0
        and duplicates["same_sites"] == duplicates["answers_checked"],
        {**duplicates, "recall": base["pooled"]},
    )
    lattice_sites = {
        n: stored[stored["environment"] == n]["site_id"].astype(str).tolist() for n in views
    }
    rng = np.random.default_rng(_stable_seed("t11"))
    added = {
        n: l2[n]
        | {lattice_sites[n][i] for i in rng.choice(len(lattice_sites[n]), size=50, replace=False)}
        for n in views
    }
    grown = recall_of(views, added)
    add(
        "t11_adding_proposals_never_reduces_recall",
        all(grown["per_environment"][n] >= base["per_environment"][n] for n in views),
        {"before": base["pooled"], "after": grown["pooled"]},
    )
    stripped = {n: {s for s in v if s not in views[n].links} for n, v in l2.items()}
    add(
        "t12_removing_every_error_linked_proposal_leaves_no_recall",
        recall_of(views, stripped)["reached"] == 0 and base["reached"] > 0,
        {"reached_before": base["reached"], "reached_after": recall_of(views, stripped)["reached"]},
    )
    random = cc_read_json(RANDOM_RANKING)
    add(
        "t13_random_lattice_sites_do_worse_than_p0_at_its_budget",
        random["at_p0_budget"]["p97_5"] < random["p0_recall"],
        {"random_p97_5": random["at_p0_budget"]["p97_5"], "p0": random["p0_recall"]},
    )
    control = tables["control_contexts"]
    add(
        "t14_a_shuffled_line_never_receives_its_own_document",
        bool((control["document_id"] != control["donor_document_id"]).all())
        and bool((control["crop_sha256"] != control["own_crop_sha256"]).all()),
        {"shuffled_lines": len(control)},
    )
    selection_leaks = {name: sorted(names_in(name) & GT_NAMES) for name in SELECTION_FUNCTIONS}
    budgets = cc_read_json(DESIGN_RECORD)["budget"]["p0_budget_by_environment"]
    again = []
    for arm, block in parsed.groupby("method", sort=False):
        key = P0_RANK_KEY if arm == L0 else RANK_KEY
        flags = _budget_columns(
            block.drop(columns=[budget_column(f) for f in BUDGET_FRACTIONS]), budgets, key
        )
        again.append(
            bool(
                flags[budget_column(PRIMARY_FRACTION)].equals(
                    block[budget_column(PRIMARY_FRACTION)]
                )
            )
        )
    compliance = cc_read_json(PROPOSAL_COMPLIANCE)
    add(
        "t15_matched_budget_selection_reads_no_label",
        not any(selection_leaks.values())
        and all(again)
        and file_sha256(PARSED_PROPOSALS) == compliance["parsed_proposals_sha256"],
        {"gt_names_found": selection_leaks, "selections_rederived": len(again)},
    )
    recovery = cc_read_json(P0_MISS_RECOVERY)
    add(
        "t16_p0_miss_recovery_counts_only_sites_p0_missed",
        bool(recovery["missed_set_excludes_p0_reached"])
        and recovery["p0_missed"] == sum(int((~v.p0).sum()) for v in views.values()),
        {"p0_missed": recovery["p0_missed"]},
    )
    union = cc_read_json(UNION_LOCALIZATION)["union"][MATCHED]
    attribution = union["attribution"]
    add(
        "t17_union_attribution_survives_deduplication",
        attribution["p0_only"] + attribution["l2_only"] + attribution["both"] == union["proposals"],
        {"union_proposals": union["proposals"], **attribution},
    )
    ceiling_leaks = {
        name: sorted(names_in(name) & {"LATTICE_CEILING", "ERROR_SITES", "PROPOSAL_ERROR_LINKS"})
        for name in (*SELECTION_FUNCTIONS, *REQUEST_BUILDERS)
    }
    add(
        "t18_the_ceiling_control_cannot_enter_a_proposal_set",
        not any(ceiling_leaks.values()) and not bool(ceiling["enters_a_primary_proposal"]),
        {"functions_checked": len(ceiling_leaks)},
    )
    _write_json_once(
        FALSIFICATION,
        {
            **_analysis_envelope("falsification_tests"),
            "tests": tests,
            "passed": sum(t["passed"] for t in tests),
            "total": len(tests),
        },
    )
    print(f"negative: {sum(t['passed'] for t in tests)} of {len(tests)} pass")
    for t in tests:
        print(f"  {'PASS' if t['passed'] else 'FAIL'} {t['test']}")
    print(f"  [negative {time.monotonic() - started:.0f}s]")
    return 0


def _duplicated_answers_reparse(lines: pd.DataFrame) -> dict[str, int]:
    """Every L2 answer re-parsed with its proposal list doubled must name the same lattice sites:
    a duplicate becomes no second proposal, so it cannot add recall or spend budget."""
    by_line = lines.set_index("line_uid")
    checked = same = 0
    for spec in environment_specs():
        raw = pd.read_parquet(_shard_path(L2, spec["environment"]))
        for row in raw.itertuples(index=False):
            line = by_line.loc[str(row.line_uid)]
            args = (list(line["local_ids"]), list(line["site_ids"]), int(line["tokens"]))
            _status, items, _prose = parse_answer(str(row.output), bool(row.finished))
            if not items:
                continue
            original = {
                p["site_id"] for p in parse_line(str(row.output), bool(row.finished), *args)[0]
            }
            doubled = json.dumps({"proposals": items + items})
            again = {p["site_id"] for p in parse_line(doubled, True, *args)[0]}
            checked += 1
            same += int(original == again)
    return {"answers_checked": checked, "same_sites": same}


def _rebuild_lattice() -> pd.DataFrame:
    """The lattice rebuilt now from the OCR pages alone, in the stored column order."""
    sample = pd.read_parquet(xr1.PROPOSAL_SAMPLE)
    frames = []
    for spec in environment_specs():
        name = spec["environment"]
        documents = set(sample[sample["environment"] == name]["document_id"].astype(str))
        pages = gen1._page_index(spec)
        for (document_id, engine_id), page in sorted(pages.items()):
            if document_id in documents:
                frames.append(page_lattice(page, name, document_id, engine_id))
    return pd.concat(frames, ignore_index=True)


def _shard_lines(arm: str) -> set[str]:
    return {
        str(u)
        for spec in environment_specs()
        for u in pd.read_parquet(_shard_path(arm, spec["environment"]))["line_uid"]
    }


def _rebuilt_prompt_hashes() -> dict[str, int]:
    """Every raw answer's prompt hash, rebuilt from the frozen GT-blind tables alone."""
    rebuilt = differing = 0
    for arm in MODEL_ARMS:
        for spec in environment_specs():
            requests = requests_for(arm, spec["environment"]).set_index("request_id")
            raw = pd.read_parquet(_shard_path(arm, spec["environment"]))
            for row in raw.itertuples(index=False):
                request = requests.loc[str(row.request_id)]
                digest = canonical_hash(
                    {
                        "system": SYSTEM_PROMPT,
                        "user": str(request["user_prompt"]),
                        "image": request["image_sha256"],
                    }
                )
                rebuilt += 1
                differing += int(digest != str(row.prompt_sha256))
    return {"prompts_rebuilt": rebuilt, "differing": differing}


# ------------------------------------------------------------------ the decision


def localization_criteria(
    matched: dict[str, Any],
    natural: dict[str, Any],
    stats: dict[str, Any],
    recovery: dict[str, Any],
    types: dict[str, Any],
    shuffle: dict[str, Any],
    union: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool]]:
    """C1-C5, the union rule U1-U2 and the natural-output rule N1-N2, from the frozen gate."""
    gain = matched["arms"][L2]["gain_over_p0"]
    retention = shuffle["results"]["subset_budget"]["retention_of_visual_gain"]
    criteria = {
        "C1": gain["mean"] >= RECALL_MARGIN,
        "C2": gain["environments_at_breadth_margin"] >= BREADTH_MAJORITY,
        "C3": recovery["arms"][arm_key(L2, MATCHED)]["pooled"] >= MISS_RECOVERY_TARGET,
        "C4": len(types["weak_types_reaching_margin"]) >= WEAK_TYPES_REQUIRED,
        "C5": bool(
            stats["primary_family"]["P2_l2_vs_l1"]["ci_low"] > 0
            and retention is not None
            and retention <= SHUFFLE_RETENTION_MAX
        ),
    }
    union_gain = union["union"][MATCHED]["gain_over_p0"]
    union_rule = {
        "U1": union_gain["mean"] >= RECALL_MARGIN,
        "U2": union_gain["environments_at_breadth_margin"] >= BREADTH_MAJORITY,
    }
    natural_gain = natural["arms"][L2]["gain_over_p0"]
    natural_rule = {
        "N1": natural_gain["mean"] >= RECALL_MARGIN,
        "N2": natural["arms"][L2]["proposals"] > natural["arms"][L0]["proposals"],
    }
    return criteria, union_rule, natural_rule


def run_decide() -> int:
    """The machine-readable decision, from the frozen rule alone."""
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    matched = cc_read_json(MATCHED_BUDGET_RESULTS)
    natural = cc_read_json(NATURAL_RESULTS)
    stats = cc_read_json(STATISTICAL_TESTS)
    recovery = cc_read_json(P0_MISS_RECOVERY)
    types = cc_read_json(ERROR_TYPE_ANALYSIS)
    shuffle = cc_read_json(IMAGE_SHUFFLE)
    union = cc_read_json(UNION_LOCALIZATION)
    precision = cc_read_json(PROPOSAL_PRECISION)["arms"]
    burden = cc_read_json(PROPOSAL_BURDEN)["arms"]
    negative = cc_read_json(FALSIFICATION)
    criteria, union_rule, natural_rule = localization_criteria(
        matched, natural, stats, recovery, types, shuffle, union
    )
    outcome = assign_outcome(criteria, union_rule, natural_rule)
    arms = matched["arms"]
    kind_recall = {k: v["recall"][arm_key(L2, MATCHED)] for k, v in types["by_kind"].items()}
    retention = shuffle["results"]["subset_budget"]["retention_of_visual_gain"]
    reason = (
        f"outcome {outcome}: "
        + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in criteria.items())
        + "; union "
        + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in union_rule.items())
        + "; natural "
        + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in natural_rule.items())
        + f". {OUTCOME_LABELS[outcome]}."
    )
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "primary_metric": "matched-budget localization recall, mean over environments",
            "primary_proposal_budget": "P0's proposal count per environment on the sampled pages",
            "p0_localization_recall": arms[L0]["recall"]["mean"],
            "text_localization_recall": arms[L1]["recall"]["mean"],
            "image_localization_recall": arms[L2]["recall"]["mean"],
            "union_localization_recall": arms[L3]["recall"]["mean"],
            "image_vs_text_gain": matched["image_vs_text"]["mean"],
            "image_vs_p0_gain": arms[L2]["gain_over_p0"]["mean"],
            "p0_miss_recovery": recovery["arms"][arm_key(L2, MATCHED)]["pooled"],
            "proposal_precision_p0": precision[arm_key(L0, NATURAL)]["precision_pooled"],
            "proposal_precision_text": precision[arm_key(L1, MATCHED)]["precision_pooled"],
            "proposal_precision_image": precision[arm_key(L2, MATCHED)]["precision_pooled"],
            "proposal_burden_p0": burden[arm_key(L0, NATURAL)]["per_1000_ocr_tokens"],
            "proposal_burden_text": burden[arm_key(L1, MATCHED)]["per_1000_ocr_tokens"],
            "proposal_burden_image": burden[arm_key(L2, MATCHED)]["per_1000_ocr_tokens"],
            "substitution_recall": kind_recall["substitution"],
            "omission_recall": kind_recall["deletion"],
            "segmentation_recall": kind_recall["segmentation"],
            "environments_improved": arms[L2]["gain_over_p0"]["environments_at_breadth_margin"],
            "weak_error_types_improved": types["weak_types_reaching_margin"],
            "image_shuffle_retained_gain": retention,
            "lattice_ceiling": cc_read_json(LATTICE_CEILING)["ceiling_pooled"],
            **{f"criterion_{k}": v for k, v in criteria.items()},
            "union_rule": union_rule,
            "natural_output_rule": natural_rule,
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS[outcome],
            "ready_for_integrated_image_pipeline": outcome in ("A", "B"),
            "ready_for_external_confirmation": False,
            "recommended_next_stage": f"NEXT: {NEXT_STAGE[outcome]}",
            "reason": reason,
            "falsification_tests_passed": negative["passed"],
            "falsification_tests_total": negative["total"],
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: outcome {outcome} -- NEXT: {NEXT_STAGE[outcome]}")
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = (
    "ANALYSIS ONLY -- development evidence computed with ground truth; not a deployable method"
)
# The validated categorical palette (dataviz reference instance), in fixed slot order and keyed to
# the arm, so an arm keeps its colour in every figure. Three slots sit below 3:1 contrast on the
# surface, so every bar carries a visible value label and the report carries the table view.
SURFACE = "#fcfcfb"
ARM_COLOURS = {
    L0: "#2a78d6",
    L1: "#eb6834",
    L2: "#1baf7a",
    L3: "#eda100",
    C3: "#e87ba4",
    C4: "#008300",
}
FIGURES = (
    "matched_budget_localization_recall.png",
    "natural_localization_recall.png",
    "localization_recall_by_environment.png",
    "localization_recall_by_error_type.png",
    "p0_miss_recovery.png",
    "proposal_precision_vs_recall.png",
    "recall_vs_proposal_budget.png",
    "proposal_burden_by_method.png",
    "p0_vs_image_unique_errors.png",
    "union_localization.png",
    "omission_localization.png",
    "segmentation_localization.png",
    "historical_glyph_localization.png",
    "image_shuffle_effect.png",
    "lattice_ceiling.png",
    "updated_bottleneck_decomposition.png",
)


def _bars(
    ax: Any,
    labels: Sequence[str],
    values: Sequence[float],
    colours: Sequence[str],
    fmt: str = "{:.4f}",
) -> None:
    """Thin bars with a surface gap between them and a value label at each bar's end."""
    positions = np.arange(len(labels))
    ax.bar(positions, values, width=0.62, color=colours, edgecolor=SURFACE, linewidth=1.0)
    for x, value in zip(positions, values, strict=True):
        ax.text(
            x,
            value,
            " " + fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=xc1.INK_SECONDARY,
        )
    ax.set_xticks(positions, labels)


def _grouped(
    ax: Any,
    groups: Sequence[str],
    series: dict[str, Sequence[float]],
    colours: dict[str, str],
    labels: dict[str, str],
    label_values: bool = True,
) -> None:
    """Grouped bars: one colour per arm in fixed order, a legend, values at bar ends."""
    width = 0.8 / max(len(series), 1)
    positions = np.arange(len(groups))
    for j, (key, values) in enumerate(series.items()):
        offset = positions + (j - (len(series) - 1) / 2) * width
        ax.bar(
            offset,
            values,
            width=width,
            color=colours[key],
            edgecolor=SURFACE,
            linewidth=1.0,
            label=labels[key],
        )
        if label_values:
            for x, value in zip(offset, values, strict=True):
                ax.text(
                    x,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    color=xc1.INK_SECONDARY,
                )
    ax.set_xticks(positions, groups)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))


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
            "svg.hashsalt": "lp1",
        }
    )
    manifest: dict[str, Any] = {}

    def source_key(path: Path) -> str:
        # Stage files relative to the stage directory, upstream files to the repository, so
        # `--determinism`'s regenerations into a sandbox directory compare equal.
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

    matched = cc_read_json(MATCHED_BUDGET_RESULTS)["arms"]
    natural = cc_read_json(NATURAL_RESULTS)["arms"]
    labels4 = [ARM_LABELS[a] for a in (L0, L1, L2, L3)]
    colours4 = [ARM_COLOURS[a] for a in (L0, L1, L2, L3)]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _bars(ax, labels4, [matched[a]["recall"]["mean"] for a in (L0, L1, L2, L3)], colours4)
    ax.set_ylabel("localization recall (mean over environments)")
    ax.set_title("Matched budget: P0's proposal count per environment")
    save(fig, FIGURES[0], [MATCHED_BUDGET_RESULTS], "matched-budget localization recall per arm")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _bars(ax, labels4, [natural[a]["recall"]["mean"] for a in (L0, L1, L2, L3)], colours4)
    for x, arm in enumerate((L0, L1, L2, L3)):
        ax.text(
            x,
            0.0,
            f"{natural[arm]['proposals']:,} proposals",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=SURFACE,
        )
    ax.set_ylabel("localization recall (mean over environments)")
    ax.set_title("Natural output: every valid proposal under the frozen prompt")
    save(fig, FIGURES[1], [NATURAL_RESULTS], "natural-output localization recall per arm")

    environments = [spec["environment"] for spec in environment_specs()]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    _grouped(
        ax,
        [xr1._short(n) for n in environments],
        {
            a: [matched[a]["recall"]["per_environment"][n] for n in environments]
            for a in (L0, L1, L2)
        },
        ARM_COLOURS,
        ARM_LABELS,
        label_values=False,
    )
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_ylabel("matched-budget localization recall")
    ax.set_title("By environment")
    save(fig, FIGURES[2], [MATCHED_BUDGET_RESULTS], "per-environment matched-budget recall")

    types = cc_read_json(ERROR_TYPE_ANALYSIS)["by_kind"]
    keys = {
        L0: arm_key(L0, NATURAL),
        L1: arm_key(L1, MATCHED),
        L2: arm_key(L2, MATCHED),
        L3: arm_key(L3, MATCHED),
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _grouped(
        ax,
        [KIND_NAMES[k] for k in ERROR_KINDS],
        {a: [types[k]["recall"][keys[a]] for k in ERROR_KINDS] for a in (L0, L1, L2, L3)},
        ARM_COLOURS,
        ARM_LABELS,
    )
    ax.set_ylabel("localization recall, pooled")
    ax.set_title("By error type (matched budget; P0 at its own)")
    save(fig, FIGURES[3], [ERROR_TYPE_ANALYSIS], "recall by error type")

    recovery = cc_read_json(P0_MISS_RECOVERY)["arms"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = [arm_key(L1, MATCHED), arm_key(L2, MATCHED), arm_key(L1, NATURAL), arm_key(L2, NATURAL)]
    _bars(
        ax,
        ["L1 matched", "L2 matched", "L1 natural", "L2 natural"],
        [recovery[k]["pooled"] for k in order],
        [ARM_COLOURS[L1], ARM_COLOURS[L2], ARM_COLOURS[L1], ARM_COLOURS[L2]],
    )
    ax.axhline(MISS_RECOVERY_TARGET, color=xc1.INK, linewidth=0.8)
    ax.text(3.4, MISS_RECOVERY_TARGET, " target", va="center", fontsize=6.5, color=xc1.INK)
    ax.set_ylabel("share of P0-missed error sites reached")
    ax.set_title("P0-miss recovery")
    save(fig, FIGURES[4], [P0_MISS_RECOVERY], "recovery of error sites P0 never reaches")

    precision = cc_read_json(PROPOSAL_PRECISION)["arms"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm, budget in REPORTED:
        key = arm_key(arm, budget)
        rec = (
            cc_read_json(NATURAL_RESULTS)["arms"][arm]["recall"]["mean"]
            if budget == NATURAL
            else matched[arm]["recall"]["mean"]
        )
        ax.scatter(
            precision[key]["precision_pooled"],
            rec,
            s=40,
            color=ARM_COLOURS[arm],
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            f"{ARM_LABELS[arm].split(' ')[0]} {budget}",
            (precision[key]["precision_pooled"], rec),
            xytext=(4, -9) if budget == NATURAL else (4, 3),
            textcoords="offset points",
            fontsize=6.5,
            color=xc1.INK,
        )
    ax.set_xlabel("site-proposal precision (pooled)")
    ax.set_ylabel("localization recall (mean over environments)")
    ax.set_title("Precision against recall")
    save(
        fig,
        FIGURES[5],
        [PROPOSAL_PRECISION, NATURAL_RESULTS, MATCHED_BUDGET_RESULTS],
        "precision and recall of each arm",
    )

    curve = cc_read_json(RECALL_BUDGET_CURVE)["arms"]
    random = cc_read_json(RANDOM_RANKING)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm in (L0, L1, L2):
        xs = [float(f) for f in curve[arm]]
        ys = [curve[arm][f]["recall"]["mean"] for f in curve[arm]]
        ax.plot(
            xs,
            ys,
            color=ARM_COLOURS[arm],
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=ARM_LABELS[arm],
        )
        ax.annotate(
            f"{ys[-1]:.3f}",
            (xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=6.5,
            color=xc1.INK_SECONDARY,
            va="center",
        )
    ax.scatter(
        [1.0],
        [random["at_p0_budget"]["mean"]],
        s=36,
        color=ARM_COLOURS[C4],
        edgecolor=SURFACE,
        linewidth=1.0,
        zorder=3,
        label=ARM_LABELS[C4],
    )
    ax.set_xlabel("proposal budget (multiple of P0's count)")
    ax.set_ylabel("localization recall (mean over environments)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Recall against proposal budget")
    save(fig, FIGURES[6], [RECALL_BUDGET_CURVE, RANDOM_RANKING], "recall-budget curve")

    burden = cc_read_json(PROPOSAL_BURDEN)["arms"]
    burden_keys = [
        arm_key(L0, NATURAL),
        arm_key(L1, NATURAL),
        arm_key(L2, NATURAL),
        arm_key(L1, MATCHED),
        arm_key(L2, MATCHED),
        arm_key(L3, MATCHED),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _bars(
        ax,
        ["P0", "L1 natural", "L2 natural", "L1 matched", "L2 matched", "P0 + L2"],
        [burden[k]["per_1000_ocr_tokens"] for k in burden_keys],
        [ARM_COLOURS[a] for a in (L0, L1, L2, L1, L2, L3)],
        fmt="{:.1f}",
    )
    ax.set_ylabel("proposals per 1,000 OCR tokens")
    ax.set_title("Proposal burden")
    save(fig, FIGURES[7], [PROPOSAL_BURDEN], "proposals per 1,000 OCR tokens")

    overlap = cc_read_json(PROPOSAL_OVERLAP)["p0_vs_l2"]
    parts = (("both", L0), ("p0_only", L1), ("image_only", L2), ("neither", C3))
    part_names = {
        "both": "both",
        "p0_only": "P0 only",
        "image_only": "image only",
        "neither": "neither",
    }
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, budget in enumerate((MATCHED, NATURAL)):
        left = 0.0
        for part, colour_arm in parts:
            value = overlap[budget][part]
            ax.barh(
                i,
                value,
                left=left,
                color=ARM_COLOURS[colour_arm],
                edgecolor=SURFACE,
                linewidth=1.0,
                label=part_names[part] if i == 0 else None,
            )
            left += value
    ax.set_yticks([0, 1], ["L2 matched", "L2 natural"])
    ax.set_xlabel("sampled OCR error sites")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Which error sites P0 and the image proposer reach")
    save(fig, FIGURES[8], [PROPOSAL_OVERLAP], "overlap of reached error sites")

    union = cc_read_json(UNION_LOCALIZATION)["union"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _bars(
        ax,
        ["P0", "L2 matched", "P0 + L2 matched", "P0 + L2 natural"],
        [
            matched[L0]["recall"]["mean"],
            matched[L2]["recall"]["mean"],
            union[MATCHED]["recall"]["mean"],
            union[NATURAL]["recall"]["mean"],
        ],
        [ARM_COLOURS[L0], ARM_COLOURS[L2], ARM_COLOURS[L3], ARM_COLOURS[L3]],
    )
    ax.set_ylabel("localization recall (mean over environments)")
    ax.set_title("P0 united with the image proposer")
    save(fig, FIGURES[9], [UNION_LOCALIZATION, MATCHED_BUDGET_RESULTS], "union localization")

    omission = cc_read_json(OMISSION_ANALYSIS)["recall"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    _grouped(
        ax,
        ["P0", "L1 matched", "L2 matched", "P0 + L2"],
        {
            "all": [omission[keys[a]]["all"] for a in (L0, L1, L2, L3)],
            "representable": [omission[keys[a]]["representable"] for a in (L0, L1, L2, L3)],
        },
        {"all": ARM_COLOURS[L0], "representable": ARM_COLOURS[L1]},
        {"all": "all omissions", "representable": "representable omissions"},
    )
    ax.set_ylabel("omission localization recall")
    ax.set_title("Omissions (reachable only through gap sites)")
    save(fig, FIGURES[10], [OMISSION_ANALYSIS], "omission localization")

    segmentation = cc_read_json(SEGMENTATION_ANALYSIS)["by_subtype"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _grouped(
        ax,
        [s.replace("_", " ") for s in SEGMENTATION_SUBTYPES],
        {
            a: [segmentation[s]["recall"][keys[a]] for s in SEGMENTATION_SUBTYPES]
            for a in (L0, L1, L2)
        },
        ARM_COLOURS,
        ARM_LABELS,
    )
    ax.tick_params(axis="x", labelrotation=15)
    ax.set_ylabel("localization recall, pooled")
    ax.set_title("Segmentation errors by subtype")
    save(fig, FIGURES[11], [SEGMENTATION_ANALYSIS], "segmentation localization")

    glyph = cc_read_json(HISTORICAL_GLYPH_ANALYSIS)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _bars(ax, labels4, [glyph["localization_recall"][keys[a]] for a in (L0, L1, L2, L3)], colours4)
    ax.set_ylabel("localization recall, pooled")
    ax.set_title("Historical glyphs: found, before any correction")
    save(fig, FIGURES[12], [HISTORICAL_GLYPH_ANALYSIS], "historical-glyph localization")

    shuffle = cc_read_json(IMAGE_SHUFFLE)["results"]["subset_budget"]["arms"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _bars(
        ax,
        [ARM_LABELS[a] for a in (L0, L1, L2, C3)],
        [shuffle[a]["recall"]["mean"] for a in (L0, L1, L2, C3)],
        [ARM_COLOURS[a] for a in (L0, L1, L2, C3)],
    )
    ax.set_ylabel("recall on the shuffle subsample")
    ax.set_title("Correct image against another document's image")
    save(fig, FIGURES[13], [IMAGE_SHUFFLE], "image-shuffle control")

    ceiling = cc_read_json(LATTICE_CEILING)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _bars(
        ax,
        ["all", *[KIND_NAMES[k] for k in ERROR_KINDS]],
        [ceiling["ceiling_pooled"], *[ceiling["by_kind"][k]["ceiling"] for k in ERROR_KINDS]],
        [ARM_COLOURS[L0]] * 5,
    )
    ax.set_ylabel("share of error sites the lattice can represent")
    ax.set_title("Lattice representation ceiling")
    save(fig, FIGURES[14], [LATTICE_CEILING], "representation ceiling by error type")

    ladder = cc_read_json(BOTTLENECK)
    steps = ladder["ladder"]
    estimate = ladder["estimate_not_a_result"]["correctable_share_of_all_errors_estimate"]
    names = [
        "all OCR errors",
        "representable",
        "reached by P0",
        "reached by L2",
        "reached by P0 + L2",
    ]
    values = [
        steps[k]
        for k in (
            "all_ocr_error_sites",
            "representable_by_the_lattice",
            "reached_by_p0",
            "reached_by_l2_matched_budget",
            "reached_by_p0_plus_l2",
        )
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.barh(range(len(values)), values, color=ARM_COLOURS[L0], edgecolor=SURFACE, linewidth=1.0)
    ax.barh(
        len(values), estimate, color=SURFACE, edgecolor=ARM_COLOURS[L3], linewidth=1.2, hatch="//"
    )
    for y, value in enumerate([*values, estimate]):
        ax.text(value, y, f" {value:.4f}", va="center", fontsize=6.5, color=xc1.INK_SECONDARY)
    ax.set_yticks(range(len(values) + 1), [*names, "correctable (ESTIMATE, not measured)"])
    ax.invert_yaxis()
    ax.set_xlabel("share of the sampled OCR error sites")
    ax.set_title("Updated bottleneck ladder")
    save(fig, FIGURES[15], [BOTTLENECK], "bottleneck ladder; the last rung is an estimate")

    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_analysis_envelope("figure_manifest"),
            "figures": manifest,
            "count": len(manifest),
            "illustrative_values": False,
            "note": FIGURE_NOTE,
            "palette": {"colours": ARM_COLOURS, "validated": "dataviz validate_palette, light"},
        },
    )
    print(f"figures: {len(manifest)} written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ determinism

DERIVED_OUTPUTS = (
    "RAW_OUTPUT_MANIFEST",
    "RAW_OUTPUT_HASHES",
    "PARSED_PROPOSALS",
    "PROPOSAL_COMPLIANCE",
    "ERROR_SITES",
    "PROPOSAL_ERROR_LINKS",
    "LATTICE_CEILING",
    "NATURAL_RESULTS",
    "MATCHED_BUDGET_RESULTS",
    "RECALL_BUDGET_CURVE",
    "P0_MISS_RECOVERY",
    "ERROR_TYPE_ANALYSIS",
    "OMISSION_ANALYSIS",
    "SEGMENTATION_ANALYSIS",
    "HISTORICAL_GLYPH_ANALYSIS",
    "PROPOSAL_PRECISION",
    "PROPOSAL_BURDEN",
    "PROPOSAL_OVERLAP",
    "UNION_LOCALIZATION",
    "ENVIRONMENT_ANALYSIS",
    "DOMAIN_ANALYSIS",
    "BOTTLENECK",
    "RUNTIME_COST",
    "IMAGE_SHUFFLE",
    "RANDOM_RANKING",
    "CONTROL_RESULTS",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "DECISION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)
DETERMINISM_CONTEXT_ENVIRONMENTS = ("funsd/doctr", "sbb/doctr")


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("parse", run_parse),
        ("link", run_link),
        ("localize", run_localize),
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
        globals().update(saved)


def run_determinism() -> int:
    """Two regenerations from the raw outputs, the GT-blind inputs rebuilt, the model rerun."""
    started = time.monotonic()
    _require(FIGURE_MANIFEST, "figures")
    _forbid(DETERMINISM)
    published = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(
        name for name in published if any(run.get(name) != published[name] for run in runs)
    )
    lattice = pd.read_parquet(SITE_LATTICE_INVENTORY)
    rebuilt = _rebuild_lattice()
    lattice_identical = canonical_hash(rebuilt.to_json(orient="records")) == canonical_hash(
        lattice[rebuilt.columns].to_json(orient="records")
    )
    lines = pd.read_parquet(SCAN_POPULATION)
    contexts = pd.read_parquet(LINE_CONTEXTS).set_index("line_uid")
    rebuilt_contexts = []
    for name in DETERMINISM_CONTEXT_ENVIRONMENTS:
        mine = lines[lines["environment"] == name].reset_index(drop=True)
        again = build_line_contexts(mine, CACHE / "determinism" / "crops" / _slug(name))
        again = again.set_index("line_uid")
        rebuilt_contexts.append(
            {
                "environment": name,
                "lines": len(again),
                "prompts_identical": bool(
                    (again["image_prompt"] == contexts.loc[again.index, "image_prompt"]).all()
                    and (again["text_prompt"] == contexts.loc[again.index, "text_prompt"]).all()
                ),
                "crops_identical": bool(
                    (
                        again["crop_sha256"].fillna("")
                        == contexts.loc[again.index, "crop_sha256"].fillna("")
                    ).all()
                ),
            }
        )
    regeneration = {p.stem: cc_read_json(p) for p in sorted(REGENERATION_DIR.glob("*.json"))}
    model_identical = bool(regeneration) and all(r["identical"] for r in regeneration.values())
    all_identical = bool(
        not differing
        and lattice_identical
        and all(r["prompts_identical"] and r["crops_identical"] for r in rebuilt_contexts)
        and model_identical
    )
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "derived_artifacts_compared": len(published),
            "derived_regenerations": len(runs),
            "differing_artifacts": differing,
            "derived_runs_identical": not differing,
            "lattice_rebuilt_identically": lattice_identical,
            "lattice_sites_rebuilt": len(rebuilt),
            "context_rebuild": rebuilt_contexts,
            "model_regeneration": regeneration,
            "model_regeneration_identical": model_identical,
            "all_runs_identical": all_identical,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"determinism: {len(published)} artifacts x {len(runs)} regenerations, "
        f"identical {all_identical}"
    )
    return 0


# ------------------------------------------------------------------ the record

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE,),
    "2. Frozen XR1 Result": (RESEARCH_FREEZE,),
    "3. Research Question": (DESIGN_RECORD,),
    "4. Why Site Proposal Is Now the Bottleneck": (RESEARCH_FREEZE, DESIGN_RECORD),
    "5. Non-Goals": (DESIGN_RECORD,),
    "6. Development Population": (SAMPLING_REGISTRY, ENVIRONMENT_INVENTORY),
    "7. GT-Blind Scan Design": (SAMPLING_REGISTRY, DESIGN_RECORD, ENVIRONMENT_INVENTORY),
    "8. Site Lattice": (SITE_LATTICE_REGISTRY, ENVIRONMENT_INVENTORY),
    "9. Representation Ceiling": (LATTICE_CEILING,),
    "10. Frozen P0 Baseline": (SITE_LATTICE_REGISTRY, FROZEN_CONFIGURATION, ENVIRONMENT_INVENTORY),
    "11. Image-Aware Proposal Model": (
        MODEL_REGISTRY,
        MODEL_VERSIONS,
        IMAGE_PREPROCESSING_REGISTRY,
        DECODING_ENVIRONMENT_CHECK,
    ),
    "12. Text-Only Matched Arm": (PROMPT_REGISTRY, MODEL_REGISTRY),
    "13. Proposal Output Contract": (PROMPT_REGISTRY, DECODING_REGISTRY, PROPOSAL_COMPLIANCE),
    "14. Raw-Output Freeze": (RAW_OUTPUT_MANIFEST, RAW_OUTPUT_HASHES),
    "15. P0 Reproduction": (LATTICE_CEILING, SITE_LATTICE_REGISTRY, FALSIFICATION),
    "16. Natural Localization Recall": (NATURAL_RESULTS,),
    "17. Matched-Budget Localization Recall": (
        MATCHED_BUDGET_RESULTS,
        STATISTICAL_TESTS,
        PROPOSAL_COMPLIANCE,
    ),
    "18. Visual Localization Effect": (MATCHED_BUDGET_RESULTS, STATISTICAL_TESTS),
    "19. Improvement over P0": (MATCHED_BUDGET_RESULTS, STATISTICAL_TESTS),
    "20. P0-Miss Recovery": (P0_MISS_RECOVERY, STATISTICAL_TESTS),
    "21. Proposal Precision": (PROPOSAL_PRECISION,),
    "22. Proposal Burden": (PROPOSAL_BURDEN,),
    "23. Recall-Budget Curve": (RECALL_BUDGET_CURVE, RANDOM_RANKING),
    "24. Substitution": (ERROR_TYPE_ANALYSIS, STATISTICAL_TESTS),
    "25. Omission": (OMISSION_ANALYSIS, ERROR_TYPE_ANALYSIS, STATISTICAL_TESTS),
    "26. Segmentation": (SEGMENTATION_ANALYSIS, ERROR_TYPE_ANALYSIS, STATISTICAL_TESTS),
    "27. Spurious Insertion": (ERROR_TYPE_ANALYSIS, STATISTICAL_TESTS),
    "28. Historical Glyphs": (HISTORICAL_GLYPH_ANALYSIS, STATISTICAL_TESTS),
    "29. Environment Analysis": (ENVIRONMENT_ANALYSIS, MATCHED_BUDGET_RESULTS),
    "30. Domain Analysis": (DOMAIN_ANALYSIS,),
    "31. P0-Image Complementarity": (PROPOSAL_OVERLAP,),
    "32. Union Localization": (UNION_LOCALIZATION, STATISTICAL_TESTS),
    "33. Image-Shuffle Control": (IMAGE_SHUFFLE, STATISTICAL_TESTS),
    "34. Random-Ranking Control": (RANDOM_RANKING,),
    "35. Falsification Tests": (FALSIFICATION, DETERMINISM),
    "36. Statistics": (STATISTICAL_TESTS,),
    "37. Compute and Runtime": (RUNTIME_COST, COMPUTE_BUDGET),
    "38. Limitations": (
        DESIGN_RECORD,
        LATTICE_CEILING,
        PROPOSAL_COMPLIANCE,
        DECODING_ENVIRONMENT_CHECK,
        DECISION,
    ),
    "39. Updated Bottleneck Decomposition": (BOTTLENECK, LATTICE_CEILING),
    "40. Research Decision": (DECISION,),
    "41. Next Stage": (DECISION,),
}
PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    MODEL_REGISTRY,
    MODEL_VERSIONS,
    ENVIRONMENT_INVENTORY,
    SAMPLING_REGISTRY,
    SITE_LATTICE_REGISTRY,
    SITE_LATTICE_INVENTORY,
    P0_PROPOSALS,
    SCAN_POPULATION,
    DESIGN_RECORD,
    COMPUTE_BUDGET,
    PROMPT_REGISTRY,
    DECODING_REGISTRY,
    IMAGE_PREPROCESSING_REGISTRY,
    LINE_CONTEXTS,
    CONTROL_CONTEXTS,
    DECODING_ENVIRONMENT_CHECK,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)


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
        p for p, sha in freeze["upstream_sha256"].items() if file_sha256(REPO / p) != sha
    )
    if moved:
        raise PhaseError(f"{len(moved)} upstream files moved since the freeze: {moved[:4]}")
    rederived: list[bool] = []
    for run in range(2):
        sandbox = CACHE / "record" / f"decision_{run}.json"
        sandbox.parent.mkdir(parents=True, exist_ok=True)
        sandbox.unlink(missing_ok=True)
        saved = globals()["DECISION"]
        globals()["DECISION"] = sandbox
        try:
            run_decide()
        finally:
            globals()["DECISION"] = saved
        rederived.append(
            canonical_hash(gen1._stable_view(cc_read_json(sandbox)))
            == canonical_hash(gen1._stable_view(cc_read_json(DECISION)))
        )
    audit = gen1.audit_report(REPORT, REPORT_SECTIONS)
    if audit["untraceable_numeric_claims"]:
        raise PhaseError(
            f"{audit['untraceable_numeric_claims']} untraceable claims, "
            f"e.g. {audit['untraceable'][:3]}"
        )
    artifacts = {
        _relative(p): file_sha256(p)
        for p in PRODUCED
        if p.is_file() and p not in (PROVENANCE, TRACEABILITY)
    }
    shards = {_relative(p): file_sha256(p) for p in sorted(RAW_CACHE.glob("*"))}
    figures = {_relative(p): file_sha256(p) for p in sorted(FIGURE_DIR.glob("*.png"))}
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "issued_head": _git("rev-parse", "HEAD"),
            "lp1_starting_state": freeze["repository_state"]["head"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "raw_generation_output_count": len(shards),
            "raw_generation_outputs": shards,
            "figures": figures,
            "upstream_files_rehashed": len(freeze["upstream_sha256"]),
            "upstream_files_moved": moved,
            "xr1_script_unchanged": file_sha256(
                REPO / "scripts/sgv_xr1_image_candidate_reliability.py"
            )
            == freeze["xr1_script_sha256"],
            "decision_rederived_identically": rederived,
            "report": {"path": _relative(REPORT), "sha256": file_sha256(REPORT)},
            "dependency_audit": {
                "confirmatory_reserve_consumed": False,
                "external_confirmation_run": False,
                "reliability_model_used": False,
                "reliability_model_retrained_or_adapted": False,
                "certification_run": False,
                "thresholds_tuned": False,
                "correction_generated": False,
                "new_or_larger_model": False,
                "models_fine_tuned": 0,
                "external_api_calls": 0,
                "upstream_artifacts_written": False,
                "deployable": False,
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
                "every number in the report is read from one of the artifacts its section names, "
                "and from the field that means what the sentence says"
            ),
            "audit": {k: audit[k] for k in ("numeric_claims", "untraceable_numeric_claims")},
            "audited_classes": list(audit["by_class"]),
            "sections": {
                name: [_relative(p) for p in paths] for name, paths in REPORT_SECTIONS.items()
            },
            "untraceable": audit["untraceable"],
        },
    )
    print(
        f"record: {len(artifacts)} artifacts, {len(shards)} raw files, report claims "
        f"{audit['numeric_claims']}"
    )
    return 0


# ------------------------------------------------------------------ entry point

PHASES: tuple[tuple[str, Callable[[], int]], ...] = (
    ("reconstruct", run_reconstruct),
    ("freeze", run_freeze),
    ("lattice", run_lattice),
    ("preregister", run_preregister),
    ("contexts", run_contexts),
    ("envcheck", run_envcheck),
    ("generate", run_generate),
    ("spotcheck", run_spotcheck),
    *_derived_phases(),
    ("determinism", run_determinism),
    ("record", run_record),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    for name, _run in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(argv)
    chosen = [(name, run) for name, run in PHASES if getattr(args, name)]
    if not chosen:
        parser.print_help()
        return 2
    for name, run in chosen:
        began = time.monotonic()
        code = run()
        print(f"  [{name} {time.monotonic() - began:.0f}s]", flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
