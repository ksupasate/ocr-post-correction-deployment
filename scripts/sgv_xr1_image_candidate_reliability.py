#!/usr/bin/env python3
"""SGV-XR1: does GEN1's visual gain survive GT-blind proposal, and can the frozen verifier rank it?

SGV-GEN1 closed with outcome B: once the true location of a real OCR error is given, Qwen3-VL-4B
with a crop of the page writes the exact repair far more often than the same weights without it.
That experiment handed the model the location, so it says nothing yet about a real candidate
generator. This stage asks the two questions that stand between that result and any use of it:

    Q1. attached to the frozen GT-blind site proposer (P0), does the image-conditioned generator
        raise the natural candidate opportunity?
    Q2. only if useful candidates are produced: can the frozen reliability model rank them without
        refitting?

**1. No ground-truth location in any primary arm.** P0 is SGV14's OCR-only discovery pass, read
from SGV-XC1's cache of its frozen-localization requests. Every proposed site is sent, error and
clean alike; ground truth enters only after the raw answers are hashed.

**2. The generator is GEN1's, unchanged.** The same checkpoint, system prompt, user templates,
decoding rule, batch size and crop rule. The crop is built from the proposed site's own OCR boxes,
so the only difference from GEN1 is where the site comes from.

**3. Generation and reliability are separate questions with separate gates.** Gate A is judged
from the candidates alone. The reliability transfer is zero-shot through the frozen adapted
verifier, and only if a pre-registered feature-compatibility gate finds that the frozen verifier can
score a candidate from a generator it has never seen.

    --reconstruct   section 0: the upstream stages' frozen states, re-read and checked
    --freeze        the upstream configuration and the GT-blind environment inventory
    --preregister   arms, sample, margins, criteria, gates, outcome rules and the registries
    --compat        the frozen verifier rebuilt exactly, and its feature-compatibility gate
    --proposals     P0's sites, the content-blind document sample and the control subsample
    --contexts      E1 text, GT-blind crops, donor crops and masked crops
    --generate      GT-blind inference into an immutable per-shard cache (XR1_ARMS=<arms>)
    --spotcheck     each arm regenerates its first batches from the frozen inputs

DEVELOPMENT / MECHANISM ONLY. `ready_for_external_confirmation` is false by construction. The SGV1
CORD confirmatory reserve stays LOCKED. SGV13, SGV14, SGV15, SGV15b, SGV-DT1, SGV-CG1, SGV-XC1 and
SGV-GEN1 artifacts are read and hash-verified; none is modified.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import os
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

import sgv_gen1_error_conditioned_generation as gen1
from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.metrics.discrimination import roc_auc

xc1 = gen1.xc1
s14 = gen1.s14
s15 = gen1.s15
cg1 = gen1.cg1
dt1 = gen1.dt1
s13 = dt1.s13
pilot = s14.pilot

REPO = gen1.REPO
OUT = REPO / "results/generated/sgv_xr1_image_candidate_reliability"
CACHE = OUT / "cache"
# Raw answers live apart from every derived table, so a GT-blind phase can never read a label
# written beside them.
RAW_CACHE = OUT / "raw_generation_outputs"
CROP_DIR = CACHE / "crops"
MASK_DIR = CACHE / "masked_crops"
REGENERATION_DIR = CACHE / "regeneration"
COMPAT_DIR = CACHE / "compat"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"
DESIGN_RECORD = OUT / "design_record.json"
SITE_PROPOSER_REGISTRY = OUT / "site_proposer_registry.json"
SITE_PROPOSAL_INVENTORY = OUT / "site_proposal_inventory.json"
MODEL_REGISTRY = OUT / "model_registry.json"
PROMPT_REGISTRY = OUT / "prompt_registry.json"
DECODING_REGISTRY = OUT / "decoding_registry.json"
CROP_POLICY = OUT / "crop_policy.json"
FEATURE_COMPATIBILITY = OUT / "feature_compatibility.json"
VERIFIER_REPRODUCTION = OUT / "frozen_verifier_reproduction.json"

# The GT-blind proposal tables. Inference reads these and nothing else.
PROPOSAL_SAMPLE = OUT / "proposal_sample.parquet"
CONTEXTS = OUT / "proposal_contexts.parquet"
CONTROL_CONTEXTS = OUT / "control_contexts.parquet"

REPORT = REPO / "docs/sgv_xr1/image_candidate_reliability.md"

SCHEMA_VERSION = 1
STAGE = "sgv_xr1_image_candidate_reliability"
HYPOTHESIS = "SGV-XR1-D1"
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
ENVIRONMENT_COUNT = gen1.ENVIRONMENT_COUNT
REQUEST_COLUMNS = gen1.REQUEST_COLUMNS
FORBIDDEN_INPUT_COLUMNS = gen1.FORBIDDEN_INPUT_COLUMNS
EPSILONS = s15.EPSILONS
PRIMARY_EPSILON = float(s15.PRIMARY_EPSILON)
epsilon_key = s15.epsilon_key

# ------------------------------------------------------------------ section 0: the upstream state
#
# What the brief says the upstream stages found, confirmed from their own artifacts at four
# decimal places. A disagreement stops the stage before anything is designed.

EXPECTATION_TOLERANCE = 5e-5
GEN1_BEST_CELL = f"{gen1.M5}::{gen1.E3}"
GEN1_MATCHED_CELL = f"{gen1.M5}::{gen1.E1}"
UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_cg1": {"outcome": "A", "candidate_opportunity_mean": 0.0960},
    "sgv_xc1": {
        "outcome": "D",
        "baseline_opportunity": 0.0960,
        "best_method": xc1.M2,
        "best_method_opportunity": 0.0178,
        "ready_for_reliability_transfer_study": False,
    },
    "sgv_gen1": {
        "outcome": "B",
        "status": "COMPLETE",
        "best_gen1_cell": GEN1_BEST_CELL,
        "best_multimodal_exact_generation": 0.2080,
        "best_text_exact_generation": 0.1027,
        "capacity_gain": 0.0600,
        "context_gain": 0.0080,
        "visual_gain": 0.1140,
        "top1_gain": 0.1733,
        "ready_for_reliability_transfer": True,
        "ready_for_external_confirmation": False,
    },
}
RESERVE_DECISIONS = (
    s14.OUT / "research_decision.json",
    s15.OUT / "research_decision.json",
    xc1.s15b.OUT / "research_decision.json",
    dt1.OUT / "development_decision.json",
    cg1.OUT / "research_decision.json",
    xc1.OUT / "research_decision.json",
    gen1.OUT / "research_decision.json",
)
UPSTREAM_REPORTS = (
    REPO / "docs/sgv_cg1/opportunity_ceiling.md",
    REPO / "docs/sgv_xc1/cross_correction_opportunity.md",
    gen1.REPORT,
)
UPSTREAM_SCRIPTS = (
    *gen1.UPSTREAM_SCRIPTS,
    "scripts/sgv_gen1_error_conditioned_generation.py",
)

# ------------------------------------------------------------------ the pre-registered design
#
# Everything from here to the end of this block is written to `design_record.json` before a single
# proposal is sampled, by a phase that refuses to run once a raw answer or a label exists.

G0 = "g0_frozen_generator"
G1 = "g1_text_matched"
G2 = "g2_image_conditioned"
G3 = "g3_union_g0_g2"
X_IMG = "x_image_shuffled"
X_MASK = "x_crop_masked"
MODEL = gen1.M5
GENERATED_ARMS = (G1, G2)
CONTROL_ARMS = (X_IMG, X_MASK)
MODEL_ARMS = (*GENERATED_ARMS, *CONTROL_ARMS)
# Which frozen GEN1 user template each arm is shown. Every image arm reads the E3 template with
# the E1 text, so only the pixels differ between G2 and its two controls.
TEMPLATE_OF_ARM = {G1: gen1.E1, G2: gen1.E3, X_IMG: gen1.E3, X_MASK: gen1.E3}
CONDITION_OF_ARM = {G1: gen1.E1, G2: gen1.E3, X_IMG: X_IMG, X_MASK: X_MASK}
PRIMARY_K = 1
SECONDARY_K = gen1.MAX_CANDIDATES

# The content-blind sample, sized from GEN1's measured throughput before any XR1 answer existed.
XR1_SEED = 20260914
SITE_TARGET = 300
MIN_DOCUMENTS = 5
CONTROL_PER_ENVIRONMENT = 60
MEASURED_REQUESTS_PER_SECOND = {G1: 0.2789, G2: 0.2287, X_IMG: 0.2287, X_MASK: 0.2287}

# Gate A. The natural-opportunity margin, justified in the design record against the baseline, the
# proposer's reach and GEN1's known-site headroom.
OPPORTUNITY_MARGIN = 0.05
BREADTH_MAJORITY = 6
WEAK_TYPES = ("substitution", "deletion", "segmentation")
WEAK_TYPE_MARGIN = 0.05
WEAK_TYPES_REQUIRED = 2
SHUFFLE_RETENTION_MAX = 0.5

# Gate R. Evaluated only if the feature-compatibility gate passes.
RELIABILITY_AUROC_FLOOR = 0.60
RELIABILITY_FRONTIER_SHARE = 0.5
CLEAN_ACCEPT_MAX = 0.10

# The GT-blind crop, GEN1's rule unchanged, plus the two pre-registered fallbacks and the mask.
CROP_SITE = "site"
CROP_LINE = "line"
CROP_NONE = "unavailable"
MASK_MIN_WIDTH = 4.0

# The frozen verifier's provenance vocabularies, read from SGV1's verifier module.
SOURCE_COLUMNS = tuple(f"prov_source_{s}" for s in pilot.GENERATOR_SOURCES)
OPERATION_COLUMNS = tuple(f"prov_operation_{o}" for o in pilot.OPERATIONS)
SCORE_COLUMNS = ("prov_generator_score", "prov_generator_score_missing")
COUNT_COLUMN = "prov_site_candidate_count"

OUTCOME_LABELS = {
    "A": "generation and reliability both transfer",
    "B": "generation works, reliability does not transfer",
    "C": "the oracle visual gain does not survive natural site proposal",
    "D": "natural gain exists but candidate harm is prohibitive",
}
NEXT_STAGE = {
    "A": "IMAGE-CONDITIONED RELIABILITY ADAPTATION + DEPLOYMENT DEVELOPMENT",
    "B": "TARGET-AWARE RELIABILITY ADAPTATION FOR IMAGE-GENERATED CANDIDATES",
    "C": "IMAGE-AWARE ERROR-SITE PROPOSAL / LOCALIZATION",
    "D": "SELECTIVE IMAGE-CANDIDATE GENERATION / ABSTENTION MECHANISM",
}


def assign_outcome(criteria: dict[str, bool], reliability_pass: bool) -> str:
    """Section 66, exactly one outcome.

    D takes precedence over C: a natural gain with breadth whose burden fails C4 is the
    harm-prohibitive case, not a failed translation. Any other failed generation criterion is C.
    """
    natural_gain = criteria["C1"] and criteria["C2"]
    if natural_gain and not criteria["C4"]:
        return "D"
    if not all(criteria.values()):
        return "C"
    return "A" if reliability_pass else "B"


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_xr1-{artifact}-v{SCHEMA_VERSION}",
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
    """True once any XR1 label or raw answer exists: the design phases refuse to run after it."""
    return any(OUT.glob("*candidates*.parquet")) or any(RAW_CACHE.glob("*.parquet"))


# ------------------------------------------------------------------ section 0: reconstruct


def _find(node: Any, key: str) -> Any:
    """The first value stored under `key` anywhere in a JSON tree, or None."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _check(label: str, observed: Any, expected: Any) -> dict[str, Any]:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        agrees = observed == expected
    else:
        agrees = observed is not None and abs(float(observed) - float(expected)) <= (
            EXPECTATION_TOLERANCE
        )
    return {"check": label, "observed": observed, "expected": expected, "agrees": bool(agrees)}


def _git_state() -> dict[str, Any]:
    """Section 1: the exact repository state this stage starts from."""
    check = subprocess.run(
        ["git", "diff", "--check"], cwd=REPO, capture_output=True, text=True, check=False
    )
    status = [line for line in _git("status", "--short").splitlines() if line.strip()]
    return {
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short_lines": len(status),
        "status_short": status,
        "diff_stat": _git("diff", "--stat").splitlines(),
        "diff_name_only": [line for line in _git("diff", "--name-only").splitlines() if line],
        "diff_check_clean": check.returncode == 0 and not check.stdout.strip(),
        "committed_by_this_stage": False,
    }


def _upstream_files() -> list[Path]:
    """Every upstream artifact this stage reads, hashed so `--record` can prove none moved."""
    files: list[Path] = []
    for root in (cg1.OUT, xc1.OUT, gen1.OUT, dt1.OUT):
        files += [
            p
            for p in sorted(root.glob("*"))
            if p.is_file() and p.suffix in (".json", ".parquet", ".npz")
        ]
    files += sorted(xc1.CACHE.glob("*.parquet")) + sorted(xc1.CACHE.glob("*.json"))
    files += sorted(p for p in gen1.RAW_CACHE.glob("*") if p.is_file())
    files += [s14.ENVIRONMENT_ROWS, s15.CERTIFICATION_ROWS, s15.ENVIRONMENT_INVENTORY]
    files += [REPO / script for script in UPSTREAM_SCRIPTS]
    files += list(UPSTREAM_REPORTS)
    return files


def run_reconstruct() -> int:
    """Section 0: every upstream state the brief names, re-read from its own artifacts."""
    started = time.monotonic()
    _forbid(RESEARCH_FREEZE)
    checks: list[dict[str, Any]] = []
    decisions = {
        "sgv_cg1": cc_read_json(cg1.OUT / "research_decision.json"),
        "sgv_xc1": cc_read_json(xc1.OUT / "research_decision.json"),
        "sgv_gen1": cc_read_json(gen1.OUT / "research_decision.json"),
    }
    for stage, expected in UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))

    exact = cc_read_json(gen1.OUT / "exact_generation_results.json")["cells"]
    checks.append(
        _check(
            "sgv_gen1.matched_no_image_exact_generation",
            exact[GEN1_MATCHED_CELL]["exact_generation"],
            0.0940,
        )
    )
    checks.append(
        _check(
            "sgv_gen1.best_cell_harmful_share_of_candidates",
            decisions["sgv_gen1"]["candidate_burden"]["harmful_share_of_candidates"],
            0.4272,
        )
    )
    damage = cc_read_json(gen1.OUT / "edit_damage_analysis.json")["cells"][GEN1_BEST_CELL]
    checks.append(
        _check(
            "sgv_gen1.best_cell_wrong_top1_worsens_share",
            damage["top1_worsens_share_of_wrong"],
            0.4108,
        )
    )
    # SGV-XC1's qualitative findings, each read as the number it rests on.
    xc1_decision = decisions["sgv_xc1"]
    checks.append(
        _check("sgv_xc1.m2_discovers_more", bool(xc1_decision["discovery_delta"] > 0), True)
    )
    checks.append(
        _check("sgv_xc1.m2_generates_less", bool(xc1_decision["generation_delta"] < 0), True)
    )
    oracle = xc1_decision["oracle_localization"]
    headroom = [float(row["mean_localization_headroom"]) for row in oracle.values()]
    checks.append(
        _check("sgv_xc1.oracle_localization_adds_little", bool(max(headroom) < 0.05), True)
    )
    checks.append(
        _check(
            "sgv_xc1.oracle_localization_ceiling",
            max(float(row["mean_oracle_localization"]) for row in oracle.values()),
            0.0974,
        )
    )
    negative = cc_read_json(gen1.OUT / "negative_tests.json")
    f1 = next(t for t in negative["tests"] if t["test"].startswith("f1_"))
    population = f1["evidence"]["population"]
    checks.append(
        _check(
            "sgv_xc1.representation_encodes_every_labelable_repair",
            population["repaired_error_sites"] == population["expressible_labelable_error_sites"],
            True,
        )
    )

    # Reports complete, traceability clean, determinism clean, decisions re-derived.
    # Each report is re-audited here by GEN1's auditor under its own section map, so "traceability
    # clean" is a measurement made now rather than a field copied from the stage's own record.
    audits: dict[str, dict[str, int]] = {}
    for stage, module in (("sgv_cg1", cg1), ("sgv_xc1", xc1), ("sgv_gen1", gen1)):
        record = gen1.audit_report(module.REPORT, module.REPORT_SECTIONS)
        audits[stage] = {
            "numeric_claims": int(record["numeric_claims"]),
            "untraceable_numeric_claims": int(record["untraceable_numeric_claims"]),
        }
        checks.append(
            _check(f"{stage}.untraceable_numeric_claims", record["untraceable_numeric_claims"], 0.0)
        )
        root = module.OUT
        determinism = cc_read_json(root / "determinism.json")
        identical = _find(determinism, "all_runs_identical")
        if identical is None:
            identical = _find(determinism, "identical")
        checks.append(_check(f"{stage}.determinism_identical", bool(identical), True))
    provenance = cc_read_json(gen1.OUT / "provenance.json")
    checks.append(
        _check(
            "sgv_gen1.decision_rederived_identically",
            bool(all(provenance["decision_rederived_identically"])),
            True,
        )
    )
    for report in UPSTREAM_REPORTS:
        checks.append(_check(f"report_present:{_relative(report)}", report.is_file(), True))
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
            "repository_state": _git_state(),
            "upstream_file_count": len(hashes),
            "upstream_sha256": hashes,
            "note": (
                "SGV-XC1 and SGV-GEN1 are not committed; no commit was authorized. They are frozen "
                "by content hash here and must re-hash unchanged at --record."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"reconstruct: {len(checks)} of {len(checks)} checks agree; {len(hashes)} files hashed")
    return 0


# ------------------------------------------------------------------ freeze


def _p0_requests(name: str) -> pd.DataFrame:
    """P0's frozen proposals for one environment: SGV-XC1's cached frozen-localization requests."""
    frame = pd.read_parquet(xc1.CACHE / f"{_slug(name)}.requests.parquet")
    frame = frame[frame["condition"] == xc1.COND_FROZEN].reset_index(drop=True)
    return frame[list(REQUEST_COLUMNS)]


def run_freeze() -> int:
    """The upstream configuration this stage holds fixed, and a GT-blind environment inventory."""
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "reconstruct")
    for path in (FROZEN_CONFIGURATION, ENVIRONMENT_INVENTORY):
        _forbid(path)
    gen1_prompt = cc_read_json(gen1.PROMPT_REGISTRY)
    gen1_decoding = cc_read_json(gen1.DECODING_REGISTRY)
    gen1_versions = cc_read_json(gen1.MODEL_VERSIONS)
    inventory: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        requests = _p0_requests(name)
        pages = gen1._page_index(spec)
        documents = set(requests["document_id"].astype(str))
        evaluated = [page for pair, page in pages.items() if pair[0] in documents]
        inventory.append(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "engine_id": spec["engine_id"],
                "base_engine": spec["base_engine"],
                "language": xc1.language_of(spec["corpus"]),
                "evaluation_documents_with_proposals": len(documents),
                "proposed_sites": len(requests),
                "proposed_sites_by_anchor_kind": {
                    str(k): int(v) for k, v in requests["anchor_kind"].value_counts().items()
                },
                "ocr_spans": int(sum(len(page.span_box) for page in evaluated)),
                "ocr_spans_with_a_box": int(
                    sum(sum(v is not None for v in page.span_box.values()) for page in evaluated)
                ),
                "page_images_present": int(
                    sum(Path(page.image_path).is_file() for page in evaluated)
                ),
                "pages": len(evaluated),
            }
        )
        print(f"  {name}: {len(requests)} proposed sites on {len(documents)} documents", flush=True)
    _write_json_once(
        ENVIRONMENT_INVENTORY,
        {
            **_envelope("environment_inventory"),
            "environments": inventory,
            "proposed_sites": int(sum(r["proposed_sites"] for r in inventory)),
            "reads_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "site_proposer": {
                "function": "sgv14_confirmatory_validation._candidate_table -> discovery_pass",
                "requests": "sgv_xc1_cross_correction_opportunity._requests_frozen_localization",
                "cache": _relative(xc1.CACHE),
                "condition": xc1.COND_FROZEN,
            },
            "generator": {
                "model": MODEL,
                "repo_id": gen1.MODEL_SPECS[MODEL]["repo_id"],
                "revision": gen1.MODEL_SPECS[MODEL]["revision"],
                "batch_size": int(gen1.MODEL_SPECS[MODEL]["batch_size"]),
                "gen1_model_version": gen1_versions["models"][MODEL]
                if "models" in gen1_versions
                else gen1_versions.get(MODEL),
                "gen1_prompt_hash": gen1_prompt["prompt_hash"],
                "gen1_prompt_revision": gen1_prompt["revision"],
                "gen1_decoding_hash": gen1_decoding["decoding_hash"],
                "gen1_decoding_revision": gen1_decoding["revision"],
            },
            "labelling": {
                "atomic_edits": "sgv_xc1_cross_correction_opportunity._neural_edits (region path)",
                "labels": "sgv14_confirmatory_validation._labels",
                "frame": "sgv_xc1_cross_correction_opportunity._method_frame",
                "links": "sgv_cg1_opportunity_ceiling._site_links",
                "exact_repair_outcome": gen1.OUTCOME_EXACT,
                "harm_policy": "STRICT_WORSENING",
            },
            "frozen_verifier": {
                "environments": "sgv15_target_risk_certification._environments",
                "adaptation": "sgv15_target_risk_certification.fit_adaptation (a4 joint refit)",
                "acquisition": s15.A_UNCERTAINTY,
                "adaptation_budget": int(s15.ADAPT_BUDGET),
                "published_geometry": _relative(dt1.GEOMETRY),
                "operating_cut": "sgv15_target_risk_certification.frozen_source_tau",
            },
            "g0_candidates": _relative(xc1.METHOD_CANDIDATES),
            "g0_links": _relative(xc1.OUT / "method_links.parquet"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"freeze: {len(inventory)} environments inventoried")
    return 0


# ------------------------------------------------------------------ preregister


def _estimated_hours(sites: int, control_sites: int) -> dict[str, float]:
    return {
        G1: sites / MEASURED_REQUESTS_PER_SECOND[G1] / 3600.0,
        G2: sites / MEASURED_REQUESTS_PER_SECOND[G2] / 3600.0,
        X_IMG: control_sites / MEASURED_REQUESTS_PER_SECOND[X_IMG] / 3600.0,
        X_MASK: control_sites / MEASURED_REQUESTS_PER_SECOND[X_MASK] / 3600.0,
    }


def _crop_policy() -> dict[str, Any]:
    policy = {
        "site_crop": (
            "GEN1's crop_box: the union of the proposed site's anchor-span OCR boxes and one OCR "
            "neighbour on each side within the same line, padded by max("
            f"{gen1.CROP_MIN_PAD} px, {gen1.CROP_PAD_FRACTION} x box height), clamped to the page"
        ),
        "gap_sites": "the two flanking spans are the anchors, so a gap crop is their padded box",
        "line_fallback": (
            "when no anchor or neighbour carries a box: the union of every OCR box on the "
            "anchors' lines, padded and clamped by the same rule"
        ),
        "unavailable": "no OCR box on the site's lines, or no page image: no image request",
        "resize": (
            f"height {gen1.CROP_HEIGHT} px, aspect kept, width capped at {gen1.CROP_MAX_WIDTH} px, "
            "Lanczos -- GEN1's render_crop"
        ),
        "masked_crop": (
            "the same crop with its informative centre filled by the crop's own per-channel "
            "median: the union of the anchor boxes at a span site; at a gap site the interval "
            "between the two flanking boxes over their joint height, widened to "
            f"{MASK_MIN_WIDTH} px when narrower. Unavailable when an anchor has no box or a gap's "
            "flanks lie on different lines."
        ),
        "shuffled_crop": (
            "the crop of another sampled site of the same environment, never the same document, "
            f"through a permutation seeded by {XR1_SEED}"
        ),
        "manual_repair": False,
        "uses_ground_truth": False,
        "expanded_to_contain_a_ground_truth_character": False,
    }
    return {**policy, "policy_hash": canonical_hash(policy)}


def run_preregister() -> int:
    """Arms, sample, margins, criteria, gates and outcome rules, frozen before any proposal."""
    started = time.monotonic()
    _require(FROZEN_CONFIGURATION, "freeze")
    for path in (
        DESIGN_RECORD,
        SITE_PROPOSER_REGISTRY,
        MODEL_REGISTRY,
        PROMPT_REGISTRY,
        DECODING_REGISTRY,
        CROP_POLICY,
    ):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an XR1 answer or label exists; the design precedes both")
    inventory = cc_read_json(ENVIRONMENT_INVENTORY)
    counts = pd.concat(
        [
            _p0_requests(spec["environment"])
            .groupby("document_id")
            .size()
            .rename("sites")
            .reset_index()
            .assign(environment=spec["environment"])
            for spec in environment_specs()
        ],
        ignore_index=True,
    )
    planned = sample_plan(counts)
    planned_sites = int(sum(v["sites"] for v in planned.values()))
    control_sites = CONTROL_PER_ENVIRONMENT * len(planned)
    hours = _estimated_hours(planned_sites, control_sites)
    cg1_decision = cc_read_json(cg1.OUT / "research_decision.json")
    gen1_decision = cc_read_json(gen1.OUT / "research_decision.json")
    baseline = float(cg1_decision["candidate_opportunity_mean"])
    discovery = float(cg1_decision["discovery_recall_mean"])
    known_site = float(gen1_decision["best_multimodal_exact_generation"])

    design = {
        **_envelope("design_record"),
        "questions": {
            "Q1": (
                "Can the image-conditioned generation benefit found in SGV-GEN1 be converted into "
                "a real, GT-blind candidate generator?"
            ),
            "Q2": (
                "Only if useful candidates are produced: can the existing frozen reliability model "
                "rank those image-conditioned candidates without retraining?"
            ),
        },
        "hypotheses": {
            "primary": (
                "the visual generation advantage observed under oracle localization remains useful "
                "when Qwen3-VL-4B is applied only at the GT-blind sites the frozen proposer offers"
            ),
            "secondary": (
                "the frozen SGV reliability model retains enough discriminatory power on "
                "image-generated candidates to separate useful from harmful edits without "
                "retraining"
            ),
            "status": "research hypotheses under test, not assumptions",
        },
        "arms": {
            G0: "the frozen current generator (SGV14's g8_union stream at P0's sites), read from "
            "SGV-XC1's labelled table; every candidate it emits",
            G1: f"{MODEL} under GEN1's frozen E1 contract at P0's sites: the OCR line, no image",
            G2: f"{MODEL} under GEN1's frozen E3 contract at P0's sites: the E1 text plus a "
            "GT-blind crop",
            G3: "G0 united with G2's top-1 candidates, deduplicated on the atomic edit with every "
            "source kept in candidate_sources -- a development arm, not a production method",
        },
        "why_g1_is_the_matched_comparator": (
            "GEN1's visual effect was measured inside one set of weights, M5 with and without "
            "the crop. G1 is those weights without the image, so G2 minus G1 differs only in "
            "the pixels and in the image tokens the prompt carries."
        ),
        "controls": {
            X_IMG: "G2's request with the crop of another document's proposed site",
            X_MASK: "G2's request with the site's informative crop centre masked",
            "subsample": f"the first {CONTROL_PER_ENVIRONMENT} sampled sites of each environment "
            "in a content-blind order",
            "deployable": False,
        },
        "candidate_budget": {
            "primary_k": PRIMARY_K,
            "secondary_k": SECONDARY_K,
            "rule": (
                "G1 and G2 decode GEN1's five-candidate list once; the primary arm is its "
                "top-ranked candidate, K=5 is a prefix of the same list reported as secondary "
                "headroom. G0 keeps its own frozen budget. No budget is searched."
            ),
        },
        "population": {
            "proposals": "every P0 site on the sampled evaluation documents, error and clean",
            "error_sites": (
                "SGV-CG1's section 4 definition, unchanged: evaluable alignment sites whose OCR "
                "text differs from ground truth, after the adjacent-component merge"
            ),
            "site_classes": {
                "error": "a labelable P0 site whose region differs from its ground truth",
                "clean": "a labelable P0 site whose region equals its ground truth",
                "unlabelable": "a P0 site whose region ground truth the alignment cannot resolve",
            },
        },
        "sample": {
            "unit": "evaluation document, drawn per environment",
            "order": f"sha256 of '{XR1_SEED}|environment|document_id' -- content-blind",
            "rule": (
                f"documents in that order until the environment holds at least {SITE_TARGET} "
                f"proposed sites and {MIN_DOCUMENTS} documents; every proposal on a taken document "
                "is sampled"
            ),
            "seed": XR1_SEED,
            "site_target": SITE_TARGET,
            "min_documents": MIN_DOCUMENTS,
            "control_subsample": (
                f"the first {CONTROL_PER_ENVIRONMENT} sampled sites of each environment in the "
                f"order sha256 of '{XR1_SEED}|control|site_id'"
            ),
            "planned_sites": planned_sites,
            "planned_documents": int(sum(v["documents"] for v in planned.values())),
            "planned_by_environment": planned,
            "why_a_sample": (
                "all 66,934 proposals through two image-model arms would take about 150 hours on "
                "this machine; the user fixed a budget of about twelve hours before any answer"
            ),
            "reads_ground_truth": False,
        },
        "compute_plan": {
            "estimated_hours": hours,
            "estimated_total_hours": float(sum(hours.values())),
            "measured_requests_per_second": MEASURED_REQUESTS_PER_SECOND,
            "source": "GEN1's measured M5 throughput at E1 and E3, batch 8",
        },
        "metrics": {
            "site_proposal_recall": (
                "OCR error sites linked to at least one P0 site / OCR error sites"
            ),
            "conditional_exact_generation": (
                "error P0 sites whose top-1 candidate is an exact repair / error P0 sites"
            ),
            "opportunity_recall": (
                "OCR error sites linked to a P0 site carrying an exact candidate within the arm's "
                "budget / OCR error sites -- SGV-CG1's and SGV-XC1's quantity"
            ),
            "candidate_precision": "exact labelled candidates / labelled candidates",
            "harmful_burden": "harmful candidates, per proposed site, and as a share of candidates",
            "clean_rewrite_rate": (
                "clean P0 sites whose top-1 answer proposes a change / clean sites"
            ),
            "overcorrection_rate": (
                "clean P0 sites whose top-1 candidate is labelled overcorrection / clean sites"
            ),
            "abstention_rate": "valid answers with no proposal / requested sites",
            "visual_translation_gain": "OpportunityRecall(G2) - OpportunityRecall(G1), both K=1",
            "gain_vs_current": "OpportunityRecall(G2) - OpportunityRecall(G0)",
            "union_gain": "OpportunityRecall(G3) - OpportunityRecall(G0)",
            "averaging": "each rate is computed per environment and averaged over the ten",
        },
        "generation_gate": {
            "margin": OPPORTUNITY_MARGIN,
            "margin_justification": {
                "baseline_opportunity": baseline,
                "proposer_discovery_recall_sgv_cg1": discovery,
                "gen1_known_site_top1": known_site,
                "translation_if_gen1_held_at_reached_sites": discovery * known_site,
                "reading": (
                    "0.05 is about half the baseline opportunity, and about 0.7 of what GEN1's "
                    "known-site rate would give if it held at every error site SGV-CG1's generator "
                    "reaches: a gain that requires most of GEN1's visual gain to survive."
                ),
            },
            "C1": (
                f"G2 or G3 raises OpportunityRecall over G0 by at least {OPPORTUNITY_MARGIN} "
                "(mean over environments, K=1 for G2)"
            ),
            "C2": (
                f"the same arm reaches that margin in at least {BREADTH_MAJORITY} of "
                f"{ENVIRONMENT_COUNT} environments"
            ),
            "C3": (
                f"the same arm raises OpportunityRecall by at least {WEAK_TYPE_MARGIN} pooled "
                f"in at least {WEAK_TYPES_REQUIRED} of {list(WEAK_TYPES)}"
            ),
            "C4": (
                "the gain is not bought with harm: the arm's harmful candidates per repaired "
                "error site -- for G3 the marginal harmful candidates per marginal repaired site "
                "-- do not exceed G0's harmful candidates per repaired error site"
            ),
            "C5": (
                "the image is load-bearing: the visual translation gain's percentile interval lies "
                "above zero, and on the control subsample the shuffled crop keeps at most "
                f"{SHUFFLE_RETENTION_MAX} of G2's top-1 exact-site gain over G1"
            ),
            "same_arm": "C1, C2, C3 and C4 must hold for one arm, G2 or G3",
            "evaluated_before_any_reliability_number": True,
        },
        "feature_compatibility_gate": {
            "rule": (
                "the frozen verifier can score a new generator's candidates zero-shot only if "
                "every column of its model is computable for them under the frozen feature "
                "definitions, and every column whose value would depend on metadata the frozen "
                "pipeline produces only for its own generators -- the generator source, the "
                "operation label -- is inert: substituting every admissible value on the M0 "
                "evaluation rows of every environment changes no frozen score (max |difference| = "
                "0). A column computed from such metadata, such as the retrieval block through the "
                "edit signature, is dependent whenever its metadata column is not inert."
            ),
            "admissible_values": {
                "generator_source": [*pilot.GENERATOR_SOURCES, "none"],
                "operation": list(pilot.OPERATIONS),
            },
            "missing_generator_score": (
                "computable by the frozen convention (score 0, missing flag 1); its training "
                "support is reported"
            ),
            "if_false": (
                "zero_shot_feature_compatible = false; no zero-shot score is computed, R1-R4 are "
                "not evaluable and reliability_gate_pass = false"
            ),
            "reads_an_xr1_candidate": False,
        },
        "reliability_gate": {
            "verifier": (
                "the frozen adapted verifier of every environment -- SGV13's a4 joint refit on 250 "
                "uncertainty-acquired target labels, as SGV-DT1 and SGV-CG1 deployed it -- applied "
                "without refitting, recalibrating or adding a feature"
            ),
            "R1": (
                f"harm and benefit AUROC on G2 candidates at least {RELIABILITY_AUROC_FLOOR}, each "
                "with a document-clustered interval above 0.5"
            ),
            "R2": (
                f"the oracle RepairRecall at harm <= {PRIMARY_EPSILON} on G2 candidates is at "
                f"least {RELIABILITY_FRONTIER_SHARE} of G0's on the same sample"
            ),
            "R3": (
                f"harm AUROC at least {RELIABILITY_AUROC_FLOOR} in at least {BREADTH_MAJORITY} of "
                f"{ENVIRONMENT_COUNT} environments"
            ),
            "R4": (
                f"AUROC of G2's beneficial error-site candidates over its clean-site candidates at "
                f"least {RELIABILITY_AUROC_FLOOR}, and the frozen cut accepts at most "
                f"{CLEAN_ACCEPT_MAX} of G2's clean-site candidates"
            ),
            "oracle_cut": "analysis only; locates a diagnostic prefix, never a deployable cut",
        },
        "secondary_adaptation_diagnostic": {
            "pre_registered": False,
            "reason": (
                "no adaptation arm is added. It would need G2 answers on the adaptation "
                "documents as well, and the order the brief sets -- zero-shot transfer first, "
                "adaptation in a later stage -- is kept. It cannot be added after G2 labels exist."
            ),
        },
        "outcome_rule": {
            **{k: f"{OUTCOME_LABELS[k]} -> {NEXT_STAGE[k]}" for k in OUTCOME_LABELS},
            "precedence": (
                "D if C1 and C2 hold but C4 fails; otherwise C if any of C1-C5 fails; otherwise A "
                "if the reliability gate passes, else B"
            ),
            "ready_for_reliability_adaptation": "true for A and B only",
            "ready_for_external_confirmation": False,
        },
        "union_complementarity": (
            "reported independently: whether G0 united with G2 is substantially stronger than "
            "either alone. It never overrides the outcome."
        ),
        "statistics": {
            "inference_unit": "document cluster, within environment",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile",
            "generation_family": ["P1_visual_translation", "P2_gain_vs_current", "P3_union_gain"],
            "reliability_family": "only if the compatibility gate passes",
            "multiplicity": "Holm within each family",
            "alpha": ALPHA,
        },
        "non_goals": [
            "external confirmation, or any use of the confirmatory reserve",
            "a new or larger correction model",
            "threshold tuning, calibration, certification or human-review optimization",
            "a new reliability model, feature or adaptation in the primary analysis",
            "a change to the site proposer",
        ],
        "primary_epsilon": PRIMARY_EPSILON,
        "ready_for_external_confirmation": False,
        "confirmatory_reserve_consumed": False,
        "environments": [r["environment"] for r in inventory["environments"]],
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_once(DESIGN_RECORD, design)

    rules = s14.DiscoveryRules() if hasattr(s14, "DiscoveryRules") else None
    if rules is None:
        from ocr_risk.discovery.enumerator import DiscoveryRules

        rules = DiscoveryRules()
    _write_json_once(
        SITE_PROPOSER_REGISTRY,
        {
            **_envelope("site_proposer_registry"),
            "name": "P0 -- frozen site proposer",
            "mechanism": (
                "SGV14's OCR-only discovery pass (ocr_risk.experiments.cgv3_confirmatory."
                "discovery_pass) with the frozen default DiscoveryRules, run through SGV14's "
                "_candidate_table on every evaluation page, one request per discovered site "
                "(SGV-XC1's _requests_frozen_localization)"
            ),
            "rules": {
                "confidence_floor": rules.conf_floor,
                "lexical_min_length": rules.lexical_min_length,
                "require_lexical_neighbour": rules.require_lexical_neighbour,
                "gap_mad_k": rules.gap_mad_k,
                "gap_min_pixels": rules.gap_min_pixels,
                "boundary_rule_enabled": rules.enable_boundary_rule,
            },
            "proposal_types": {
                "token": "a low-confidence or lexically implausible OCR token",
                "gap": (
                    "two adjacent OCR tokens whose box gap exceeds the page's median plus "
                    f"{rules.gap_mad_k} x MAD of within-line gaps -- text may be missing"
                ),
                "token_pair": "two tokens proposed together for a merge",
            },
            "gap_anchor_rule": "the two flanking OCR spans; the region is the separator between",
            "span_anchor_rule": (
                "the OCR span or spans of the site; the region is their stream text"
            ),
            "ocr_geometry_used": "OCR token boxes, for the gap rule's page-local gap statistic",
            "document_context_used": (
                "page-local gap statistics, per-engine confidence normalisation and the lexicon "
                "fitted on SGV1 TRAIN; no ground truth"
            ),
            "deterministic_ordering": "document, engine, char_start, char_end, site_id",
            "request_context": f"{gen1.CONTEXT_CHARS} OCR characters each side",
            "proposed_sites_evaluation": int(inventory["proposed_sites"]),
            "proposed_sites_by_environment": {
                r["environment"]: r["proposed_sites"] for r in inventory["environments"]
            },
            "source_code_sha256": {
                path: file_sha256(REPO / path)
                for path in (
                    "src/ocr_risk/discovery/enumerator.py",
                    "src/ocr_risk/experiments/cgv3_confirmatory.py",
                    "scripts/sgv14_confirmatory_validation.py",
                    "scripts/sgv_xc1_cross_correction_opportunity.py",
                )
            },
            "request_tables_sha256": {
                _slug(spec["environment"]): file_sha256(
                    xc1.CACHE / f"{_slug(spec['environment'])}.requests.parquet"
                )
                for spec in environment_specs()
            },
            "redesigned_here": False,
            "uses_ground_truth": False,
        },
    )

    gen1_prompt = cc_read_json(gen1.PROMPT_REGISTRY)
    gen1_decoding = cc_read_json(gen1.DECODING_REGISTRY)
    prompt_hash = canonical_hash({"system": gen1.SYSTEM_PROMPT, "user": gen1.USER_TEMPLATES})
    if prompt_hash != gen1_prompt["prompt_hash"]:
        raise PhaseError("the imported prompt differs from GEN1's frozen prompt registry")
    _write_json_once(
        PROMPT_REGISTRY,
        {
            **_envelope("prompt_registry"),
            "source": _relative(gen1.PROMPT_REGISTRY),
            "revision": gen1.PROMPT_REVISION,
            "system": gen1.SYSTEM_PROMPT,
            "user_templates": gen1.USER_TEMPLATES,
            "template_of_arm": TEMPLATE_OF_ARM,
            "prompt_hash": prompt_hash,
            "identical_to_gen1": True,
            "contains_ground_truth": False,
            "note": (
                "GEN1's system prompt tells the model the marked position is a real OCR error. "
                "At a clean proposed site that is false, and the contract is kept anyway: changing "
                "it would change the generator this stage exists to test."
            ),
        },
    )
    _write_json_once(
        DECODING_REGISTRY,
        {
            **_envelope("decoding_registry"),
            "source": _relative(gen1.DECODING_REGISTRY),
            "revision": gen1.DECODING_REVISION,
            "max_new_tokens_rule": gen1.MAX_NEW_TOKENS_RULE,
            "batch_size": int(gen1.MODEL_SPECS[MODEL]["batch_size"]),
            "dtype": gen1.MODEL_DTYPE,
            "decoding_hash": gen1_decoding["decoding_hash"],
            "identical_to_gen1": True,
            "greedy": True,
        },
    )
    _write_json_once(
        MODEL_REGISTRY,
        {
            **_envelope("model_registry"),
            "model": MODEL,
            **dict(gen1.MODEL_SPECS[MODEL]),
            "arms": {arm: CONDITION_OF_ARM[arm] for arm in MODEL_ARMS},
            "g0": "the frozen generator's labelled candidates, read from SGV-XC1",
            "new_or_larger_model": False,
            "fine_tuned_here": False,
        },
    )
    _write_json_once(CROP_POLICY, {**_envelope("crop_policy"), **_crop_policy()})
    print(
        f"preregister: {planned_sites} planned sites on "
        f"{design['sample']['planned_documents']} documents, "
        f"about {design['compute_plan']['estimated_total_hours']:.1f} h of generation"
    )
    return 0


# ------------------------------------------------------------------ the content-blind sample


def order_key(*parts: str) -> str:
    """The content-blind order: a hash of the stage seed and identifiers, never of content."""
    return hashlib.sha256("|".join((str(XR1_SEED), *parts)).encode()).hexdigest()


def choose_documents(documents: Sequence[str], sites: Sequence[int], environment: str) -> list[str]:
    """Documents in hash order until SITE_TARGET proposals and MIN_DOCUMENTS documents are held."""
    ordered = sorted(zip(documents, sites, strict=True), key=lambda d: order_key(environment, d[0]))
    taken: list[str] = []
    total = 0
    for document, count in ordered:
        if total >= SITE_TARGET and len(taken) >= MIN_DOCUMENTS:
            break
        taken.append(str(document))
        total += int(count)
    return taken


def sample_plan(counts: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """What the rule takes in every environment, from P0's per-document counts alone."""
    plan: dict[str, dict[str, Any]] = {}
    for name, block in counts.groupby("environment", sort=True):
        chosen = choose_documents(
            block["document_id"].astype(str).tolist(), block["sites"].tolist(), str(name)
        )
        sites = int(block[block["document_id"].isin(chosen)]["sites"].sum())
        plan[str(name)] = {"documents": len(chosen), "sites": sites}
    return plan


def control_subsample(site_ids: Sequence[str]) -> set[str]:
    ordered = sorted(site_ids, key=lambda s: order_key("control", str(s)))
    return set(ordered[:CONTROL_PER_ENVIRONMENT])


def run_proposals() -> int:
    """P0's evaluation proposals, the document sample and the control subsample. GT-blind."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (PROPOSAL_SAMPLE, SITE_PROPOSAL_INVENTORY):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an XR1 answer or label exists; the sample precedes both")
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        requests = _p0_requests(name)
        bundles = s14._document_bundles(spec["corpus"])
        roles = s14.partition_of(spec["corpus"], sorted(bundles))
        wrong = sorted(
            {d for d in requests["document_id"].astype(str) if roles.get(d) != s14.ROLE_EVALUATION}
        )
        if wrong:
            raise PhaseError(f"{name}: {len(wrong)} proposal documents are not evaluation-role")
        per_document = requests.groupby("document_id").size()
        chosen = choose_documents(
            per_document.index.astype(str).tolist(), per_document.tolist(), name
        )
        rank = {d: i for i, d in enumerate(chosen)}
        sampled = requests[requests["document_id"].isin(chosen)].copy()
        controls = control_subsample(sampled["site_id"].astype(str).tolist())
        sampled["document_rank"] = sampled["document_id"].map(rank).astype(int)
        sampled["in_control_subsample"] = sampled["site_id"].isin(controls)
        sampled = sampled.sort_values(["document_rank", "char_start", "site_id"], kind="stable")
        frames.append(sampled.reset_index(drop=True))
        rows.append(
            {
                "environment": name,
                "evaluation_documents_with_proposals": int(per_document.size),
                "proposed_sites": len(requests),
                "proposed_by_anchor_kind": {
                    str(k): int(v) for k, v in requests["anchor_kind"].value_counts().items()
                },
                "sampled_documents": len(chosen),
                "sampled_sites": len(sampled),
                "sampled_by_anchor_kind": {
                    str(k): int(v) for k, v in sampled["anchor_kind"].value_counts().items()
                },
                "control_sites": len(controls),
                "documents": [
                    {"document_id": d, "sites": int(per_document[d]), "rank": rank[d]}
                    for d in chosen
                ],
            }
        )
        print(f"  {name}: {len(sampled)} sites on {len(chosen)} documents", flush=True)
    sample = pd.concat(frames, ignore_index=True)
    leaked = [c for c in FORBIDDEN_INPUT_COLUMNS if c in sample.columns]
    if leaked:
        raise PhaseError(f"the proposal sample carries truth columns: {leaked}")
    planned = cc_read_json(DESIGN_RECORD)["sample"]["planned_by_environment"]
    for row in rows:
        plan = planned[row["environment"]]
        if (row["sampled_sites"], row["sampled_documents"]) != (plan["sites"], plan["documents"]):
            raise PhaseError(f"{row['environment']}: the sample differs from the frozen plan")
    _write_parquet_once(PROPOSAL_SAMPLE, sample)
    _write_json_once(
        SITE_PROPOSAL_INVENTORY,
        {
            **_envelope("site_proposal_inventory"),
            "environments": rows,
            "proposed_sites": int(sum(r["proposed_sites"] for r in rows)),
            "sampled_sites": len(sample),
            "sampled_documents": int(sum(r["sampled_documents"] for r in rows)),
            "control_sites": int(sample["in_control_subsample"].sum()),
            "sample_table": {
                "path": _relative(PROPOSAL_SAMPLE),
                "sha256": file_sha256(PROPOSAL_SAMPLE),
            },
            "matches_frozen_plan": True,
            "reads_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"proposals: {len(sample)} sampled sites -> {_relative(PROPOSAL_SAMPLE)}")
    return 0


# ------------------------------------------------------------------ GT-blind crops


def _padded(
    box: tuple[float, float, float, float] | None, page: Any
) -> tuple[int, int, int, int] | None:
    """GEN1's padding and clamping rule, applied to a box."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    pad = max(gen1.CROP_MIN_PAD, gen1.CROP_PAD_FRACTION * (y1 - y0))
    left = max(0, int(np.floor(x0 - pad)))
    top = max(0, int(np.floor(y0 - pad)))
    right = min(page.width, int(np.ceil(x1 + pad)))
    bottom = min(page.height, int(np.ceil(y1 + pad)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def line_box(page: Any, anchors: Sequence[str]) -> tuple[float, float, float, float] | None:
    """The union of every OCR box on the anchors' lines -- the pre-registered line fallback."""
    lines = {page.span_line[a] for a in anchors if a in page.span_line}
    boxes = [
        page.span_box[s]
        for s in page.order
        if page.span_line[s] in lines and page.span_box.get(s) is not None
    ]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def site_crop(page: Any, anchors: Sequence[str]) -> tuple[tuple[int, int, int, int] | None, str]:
    """GEN1's site crop, else the line fallback, else unavailable. Never widened toward truth."""
    box = gen1.crop_box(page, anchors)
    if box is not None:
        return box, CROP_SITE
    fallback = _padded(line_box(page, anchors), page)
    if fallback is not None:
        return fallback, CROP_LINE
    return None, CROP_NONE


def mask_rectangle(
    page: Any,
    anchors: Sequence[str],
    gap: bool,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """The crop's informative centre, in crop pixels. Built from OCR boxes alone."""
    ordered = sorted(anchors, key=lambda a: page.order.index(a))
    boxes = [page.span_box.get(a) for a in ordered]
    if not boxes or any(b is None for b in boxes):
        return None
    if not gap:
        x0 = min(b[0] for b in boxes if b is not None)
        y0 = min(b[1] for b in boxes if b is not None)
        x1 = max(b[2] for b in boxes if b is not None)
        y1 = max(b[3] for b in boxes if b is not None)
    else:
        if len(boxes) != 2:
            return None
        left_box, right_box = boxes
        assert left_box is not None and right_box is not None
        if not (left_box[1] < right_box[3] and right_box[1] < left_box[3]):
            return None
        x0, x1 = left_box[2], right_box[0]
        if x1 - x0 < MASK_MIN_WIDTH:
            centre = (x0 + x1) / 2.0
            x0, x1 = centre - MASK_MIN_WIDTH / 2.0, centre + MASK_MIN_WIDTH / 2.0
        y0, y1 = min(left_box[1], right_box[1]), max(left_box[3], right_box[3])
    left, top, right, bottom = box
    scale_x = size[0] / max(right - left, 1)
    scale_y = size[1] / max(bottom - top, 1)
    a = max(0, int(np.floor((x0 - left) * scale_x)))
    b = max(0, int(np.floor((y0 - top) * scale_y)))
    c = min(size[0], int(np.ceil((x1 - left) * scale_x)))
    d = min(size[1], int(np.ceil((y1 - top) * scale_y)))
    if c <= a or d <= b:
        return None
    return a, b, c, d


def apply_mask(crop: Any, rectangle: tuple[int, int, int, int]) -> Any:
    """Fill the rectangle with the crop's own per-channel median. Dimensions are kept."""
    from PIL import Image

    pixels = np.asarray(crop.convert("RGB"), dtype=np.uint8).copy()
    fill = np.median(pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    a, b, c, d = rectangle
    pixels[b:d, a:c, :] = fill
    return Image.fromarray(pixels, mode="RGB")


def _crop_record(crop: Any) -> str:
    return canonical_hash({"size": list(crop.size), "rgb": canonical_hash(crop.tobytes().hex())})


def donor_map(sites: pd.DataFrame, name: str) -> dict[str, str]:
    """A seeded permutation of the environment's cropped sites that never stays on one document."""
    ordered = sites.sort_values("site_id", kind="stable").reset_index(drop=True)
    ids = ordered["site_id"].astype(str).tolist()
    documents = ordered["document_id"].astype(str).tolist()
    seed = int(order_key("donor", name)[:16], 16)
    permutation = np.random.default_rng(seed).permutation(len(ids)).tolist()
    mapping: dict[str, str] = {}
    for position, site in enumerate(ids):
        for step in range(len(ids)):
            candidate = permutation[(position + step) % len(ids)]
            if documents[candidate] != documents[position]:
                mapping[site] = ids[candidate]
                break
    return mapping


def build_natural_contexts(
    sites: pd.DataFrame, pages: dict[tuple[str, str], Any], crop_dir: Path
) -> pd.DataFrame:
    """E1 text and the GT-blind crop for every sampled proposal. Reads OCR, boxes and pixels."""
    from PIL import Image

    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    image: tuple[str, Any] | None = None
    for row in sites.itertuples(index=False):
        page = pages[str(row.document_id), str(row.engine_id)]
        anchors = gen1.anchors_of(str(row.anchor_ref))
        gap = str(row.anchor_kind) == "gap"
        start, end = int(row.char_start), int(row.char_end)
        if page.stream[start:end] != str(row.region):
            raise PhaseError(f"{row.site_id}: region differs from the rebuilt OCR stream")
        box, level = site_crop(page, anchors)
        record: dict[str, Any] = {
            "site_id": str(row.site_id),
            "e1_text": gen1.line_text(page, anchors, start, end, gap, wide=False),
            "crop_level": level if Path(page.image_path).is_file() else CROP_NONE,
            "crop_path": None,
            "crop_sha256": None,
            "crop_box": list(box) if box is not None else None,
            "crop_width": None,
            "crop_height": None,
            "mask_rectangle": None,
            "page_image_sha256": page.image_sha256,
        }
        if box is not None and Path(page.image_path).is_file():
            if image is None or image[0] != page.image_path:
                image = (page.image_path, Image.open(page.image_path).convert("RGB"))
            crop = gen1.render_crop(image[1], box)
            target = crop_dir / f"{canonical_hash({'site': str(row.site_id)})[:32]}.png"
            if not target.is_file():
                crop.save(target, format="PNG")
            rectangle = mask_rectangle(page, anchors, gap, box, crop.size)
            record.update(
                crop_path=str(target),
                crop_sha256=_crop_record(crop),
                crop_width=int(crop.size[0]),
                crop_height=int(crop.size[1]),
                mask_rectangle=list(rectangle) if rectangle is not None else None,
            )
        rows.append(record)
    return pd.DataFrame(rows)


def run_contexts() -> int:
    """E1 text and crops for every sampled proposal; donor and masked crops for the controls."""
    from PIL import Image

    started = time.monotonic()
    _require(PROPOSAL_SAMPLE, "proposals")
    for path in (CONTEXTS, CONTROL_CONTEXTS):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an XR1 answer or label exists; contexts are frozen before both")
    sample = pd.read_parquet(PROPOSAL_SAMPLE)
    frames: list[pd.DataFrame] = []
    controls: list[dict[str, Any]] = []
    census: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        sites = sample[sample["environment"] == name].reset_index(drop=True)
        pages = gen1._page_index(spec)
        contexts = build_natural_contexts(sites, pages, CROP_DIR / _slug(name))
        frames.append(contexts)
        joined = sites.merge(contexts, on="site_id", validate="one_to_one")
        cropped = joined[joined["crop_path"].notna()]
        donors = donor_map(cropped, name)
        by_site = cropped.set_index("site_id")
        counts = {X_IMG: 0, X_MASK: 0}
        mask_dir = MASK_DIR / _slug(name)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for row in joined[joined["in_control_subsample"]].itertuples(index=False):
            site = str(row.site_id)
            if row.crop_path is None or pd.isna(row.crop_path):
                continue
            donor = donors.get(site)
            if donor is not None:
                controls.append(
                    {
                        "site_id": site,
                        "arm": X_IMG,
                        "donor_site_id": donor,
                        "crop_path": str(by_site.loc[donor, "crop_path"]),
                        "crop_sha256": str(by_site.loc[donor, "crop_sha256"]),
                    }
                )
                counts[X_IMG] += 1
            rectangle = row.mask_rectangle
            if rectangle is not None and not (isinstance(rectangle, float) and np.isnan(rectangle)):
                crop = Image.open(str(row.crop_path)).convert("RGB")
                masked = apply_mask(crop, tuple(int(v) for v in rectangle))  # type: ignore[arg-type]
                target = mask_dir / f"{canonical_hash({'site': site, 'mask': True})[:32]}.png"
                if not target.is_file():
                    masked.save(target, format="PNG")
                controls.append(
                    {
                        "site_id": site,
                        "arm": X_MASK,
                        "donor_site_id": None,
                        "crop_path": str(target),
                        "crop_sha256": _crop_record(masked),
                    }
                )
                counts[X_MASK] += 1
        census.append(
            {
                "environment": name,
                "sites": len(sites),
                "crops_site_level": int((contexts["crop_level"] == CROP_SITE).sum()),
                "crops_line_fallback": int((contexts["crop_level"] == CROP_LINE).sum()),
                "image_unavailable": int((contexts["crop_level"] == CROP_NONE).sum()),
                "control_sites": int(sites["in_control_subsample"].sum()),
                "shuffled_crops": counts[X_IMG],
                "masked_crops": counts[X_MASK],
                "seconds": time.monotonic() - began,
            }
        )
        print(
            f"  {name}: {len(sites)} sites, {int(contexts['crop_path'].notna().sum())} crops, "
            f"{counts[X_IMG]} shuffled, {counts[X_MASK]} masked "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    contexts_all = pd.concat(frames, ignore_index=True)
    control_frame = pd.DataFrame(controls)
    for frame, label in ((contexts_all, "contexts"), (control_frame, "control contexts")):
        leaked = [c for c in FORBIDDEN_INPUT_COLUMNS if c in frame.columns]
        if leaked:
            raise PhaseError(f"the {label} table carries truth columns: {leaked}")
    _write_parquet_once(CONTEXTS, contexts_all)
    _write_parquet_once(CONTROL_CONTEXTS, control_frame)
    policy = cc_read_json(CROP_POLICY)
    _write_json_once(
        CACHE / "context_census.json",
        {
            **_envelope("context_census"),
            "crop_policy_hash": policy["policy_hash"],
            "census": census,
            "contexts_table": {"path": _relative(CONTEXTS), "sha256": file_sha256(CONTEXTS)},
            "control_table": {
                "path": _relative(CONTROL_CONTEXTS),
                "sha256": file_sha256(CONTROL_CONTEXTS),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"contexts: {len(contexts_all)} sites, {int(contexts_all['crop_path'].notna().sum())} "
        f"crops, {len(control_frame)} control requests"
    )
    return 0


# ------------------------------------------------------------------ generate


def _selected_arms() -> list[str]:
    chosen = [a for a in os.environ.get("XR1_ARMS", "").split(",") if a]
    unknown = [a for a in chosen if a not in MODEL_ARMS]
    if unknown:
        raise PhaseError(f"unknown XR1_ARMS {unknown}; choose from {list(MODEL_ARMS)}")
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
    sample = pd.read_parquet(PROPOSAL_SAMPLE)
    sites = sample[sample["environment"] == name]
    contexts = pd.read_parquet(CONTEXTS)
    if arm in CONTROL_ARMS:
        chosen = pd.read_parquet(CONTROL_CONTEXTS)
        chosen = chosen[chosen["arm"] == arm][["site_id", "crop_path", "crop_sha256"]]
        joined = sites.merge(contexts[["site_id", "e1_text"]], on="site_id").merge(
            chosen, on="site_id", validate="one_to_one"
        )
    else:
        joined = sites.merge(contexts, on="site_id", validate="one_to_one")
        if arm == G2:
            joined = joined[joined["crop_path"].notna()]
    image = arm != G1
    digest = joined["site_id"].astype(str).str.split("|", n=1).str[1]
    return pd.DataFrame(
        {
            "environment": joined["environment"].to_numpy(),
            "method": arm,
            "condition": CONDITION_OF_ARM[arm],
            "request_id": (joined["environment"] + f"|{arm}|" + digest).to_numpy(),
            "site_id": joined["site_id"].to_numpy(),
            "anchor_kind": joined["anchor_kind"].to_numpy(),
            "region": joined["region"].to_numpy(),
            "user_prompt": [
                gen1.user_message(TEMPLATE_OF_ARM[arm], str(t)) for t in joined["e1_text"]
            ],
            "image_path": joined["crop_path"].to_numpy() if image else None,
            "image_sha256": joined["crop_sha256"].to_numpy() if image else None,
        }
    ).reset_index(drop=True)


def run_generate() -> int:
    """GT-blind inference, cached immutably one (arm, environment) shard at a time.

    Each shard is written atomically with its provenance sidecar the moment it exists, so an
    interrupted run keeps every finished shard and a rerun skips each one whose file matches its
    sidecar hash.
    """
    started = time.monotonic()
    for path in (CONTEXTS, PROMPT_REGISTRY, DECODING_REGISTRY, CROP_POLICY):
        _require(path, "contexts" if path == CONTEXTS else "preregister")
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    prompt_hash = cc_read_json(PROMPT_REGISTRY)["prompt_hash"]
    decoding_hash = cc_read_json(DECODING_REGISTRY)["decoding_hash"]
    crop_hash = cc_read_json(CROP_POLICY)["policy_hash"]
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
                "condition": CONDITION_OF_ARM[arm],
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
            f"({cost['requests_per_second']:.2f}/s)",
            flush=True,
        )
    del runner
    _release_device_cache(device)
    print(f"generate: done in {time.monotonic() - started:.0f}s")
    return 0


REGENERATION_ENVIRONMENT = "funsd/doctr"
REGENERATION_BATCHES = 2


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
        raw, _cost = gen1.run_requests(
            runner,
            requests,
            int(spec["batch_size"]),
            print,
            limit_batches=REGENERATION_BATCHES,
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


# ------------------------------------------------------------------ sections 25-26: the verifier


def column_audit(names: Sequence[str]) -> list[dict[str, str]]:
    """Section 25: what each frozen design column needs before a new generator's edit is scored."""
    rows: list[dict[str, str]] = []
    for name in names:
        if name in SOURCE_COLUMNS:
            status, note = (
                "candidate_source_dependent",
                "one-hot over the frozen generators; a new generator has no category",
            )
        elif name in OPERATION_COLUMNS:
            status, note = (
                "requires_unavailable_metadata",
                "the operation label is emitted by the frozen generators; no frozen function "
                "derives it from an original and a replacement",
            )
        elif name.startswith("retr_"):
            status, note = (
                "requires_unavailable_metadata",
                "retrieval reads the edit signature, which carries the operation label",
            )
        elif name in SCORE_COLUMNS:
            status, note = (
                "computable_by_frozen_convention",
                "a generator with no score takes the frozen missing-score convention",
            )
        elif name == COUNT_COLUMN:
            status, note = (
                "computable_generator_relative",
                "the count of the new generator's own candidates at the site",
            )
        else:
            status, note = "computable", "a frozen function of the OCR, the edit or the page"
        rows.append({"column": name, "status": status, "note": note})
    return rows


def _substituted(matrix: np.ndarray, names: Sequence[str], values: dict[str, float]) -> np.ndarray:
    out = matrix.copy()
    position = {n: i for i, n in enumerate(names)}
    for column, value in values.items():
        if column in position:
            out[:, position[column]] = value
    return out


def substitution_families() -> dict[str, dict[str, dict[str, float]]]:
    """Every admissible value of each column a new generator cannot take from its definition."""
    sources: dict[str, dict[str, float]] = {}
    for label in (*pilot.GENERATOR_SOURCES, "none"):
        sources[label] = {c: float(c == f"prov_source_{label}") for c in SOURCE_COLUMNS}
    operations = {
        op: {c: float(c == f"prov_operation_{op}") for c in OPERATION_COLUMNS}
        for op in pilot.OPERATIONS
    }
    return {"generator_source": sources, "operation": operations}


def run_compat() -> int:
    """The frozen verifier rebuilt exactly, then the pre-registered feature-compatibility gate.

    No XR1 candidate is read. Every environment's adapted verifier is refitted by SGV15's own call
    and must reproduce SGV-DT1's published scores and cut bit for bit; then the columns a new
    generator could not take from a frozen definition are substituted on the M0 evaluation rows
    to see whether the frozen score depends on them.
    """
    from sgv1_domain_generalization import Design

    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (FEATURE_COMPATIBILITY, VERIFIER_REPRODUCTION):
        _forbid(path)
    deployments = dt1.load_deployments()
    order = pd.read_parquet(cg1.EVALUATION_ORDER)
    families = substitution_families()
    reproduction: list[dict[str, Any]] = []
    invariance: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    COMPAT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        built, _inventory = s15._environments(name)
        environment = built[0]
        seed = dt1._stable_seed(
            "sgv15-adapt", str(s15.SAMPLE_SEED), environment.name, s15.A_UNCERTAINTY
        )
        adapted = s15.fit_adaptation(environment, s15.A_UNCERTAINTY, s15.ADAPT_BUDGET, seed)
        stored = deployments[name]
        block = environment.setup.evaluation
        taus = [s15.frozen_source_tau(environment, adapted.model, float(e)) for e in EPSILONS]
        published = order[order["environment"] == name].sort_values("position")
        score_gap = float(np.abs(adapted.eval_scores - stored.eval_scores).max())
        tau_gap = float(
            max(
                abs(a - b) if np.isfinite(a) and np.isfinite(b) else float(a != b)
                for a, b in zip(taus, [stored.source_tau[float(e)] for e in EPSILONS], strict=True)
            )
        )
        ids_equal = bool(
            np.array_equal(
                block.candidates.astype(str), published["candidate_id"].astype(str).to_numpy()
            )
        )
        labels_equal = bool(
            np.array_equal(block.harmful, stored.eval_harm)
            and np.array_equal(block.beneficial, stored.eval_beneficial)
        )
        state = environment.setup.state
        model = adapted.model.model
        lam = float(environment.setup.lambda_)
        index = state.extended.block(block.index)
        names = list(state.design.names)
        matrix = np.asarray(state.design.matrix[index], dtype=float)
        meta = state.design.meta.iloc[index].reset_index(drop=True)
        local = np.arange(matrix.shape[0])
        base = np.asarray(model.utility(Design(matrix, tuple(names), meta), local, lam))
        vector_gap = float(np.abs(base - adapted.eval_scores).max())
        used = [names[i] for i in model.columns]
        reproduction.append(
            {
                "environment": name,
                "evaluation_rows": int(block.size),
                "candidate_ids_identical": ids_equal,
                "labels_identical": labels_equal,
                "score_max_abs_difference": score_gap,
                "direct_scoring_max_abs_difference": vector_gap,
                "operating_cut_max_abs_difference": tau_gap,
                "feature_matrix_sha256": canonical_hash(matrix.round(12).tolist()),
                "model_kind": str(model.kind),
                "representation": str(model.representation),
                "model_columns": len(used),
                "identical": bool(
                    ids_equal
                    and labels_equal
                    and score_gap == 0.0
                    and tau_gap == 0.0
                    and vector_gap == 0.0
                ),
            }
        )
        audits[name] = column_audit(used)
        for family, substitutions in families.items():
            touched = [
                c
                for c in (SOURCE_COLUMNS if family == "generator_source" else OPERATION_COLUMNS)
                if c in used
            ]
            for label, values in substitutions.items():
                changed = np.asarray(
                    model.utility(
                        Design(_substituted(matrix, names, values), tuple(names), meta), local, lam
                    )
                )
                difference = np.abs(changed - base)
                invariance.append(
                    {
                        "environment": name,
                        "family": family,
                        "value": label,
                        "columns_in_model": touched,
                        "max_abs_difference": float(difference.max()),
                        "rows_changed": int((difference > 0).sum()),
                        "rows": int(difference.size),
                        "share_changed": float((difference > 0).mean()),
                    }
                )
        fit = state.index(s13.FIT) if hasattr(s13, "FIT") else state.fold.fit
        missing_position = names.index("prov_generator_score_missing")
        missing_support = float(np.mean(state.design.matrix[fit, missing_position] > 0.5))
        audits[name] = {
            "columns": audits[name],
            "missing_generator_score_share_in_fit": missing_support,
        }
        print(
            f"  {name}: scores {score_gap:g}, cut {tau_gap:g}, ids {ids_equal}, labels "
            f"{labels_equal} ({time.monotonic() - began:.0f}s)",
            flush=True,
        )
        del built, environment, adapted
    inert = {
        family: all(r["max_abs_difference"] == 0.0 for r in invariance if r["family"] == family)
        for family in families
    }
    undefined = sorted(
        {
            row["column"]
            for audit in audits.values()
            for row in audit["columns"]
            if row["status"] == "undefined"
        }
    )
    compatible = bool(all(inert.values()) and not undefined)
    reproduced = bool(all(r["identical"] for r in reproduction))
    _write_json_once(
        VERIFIER_REPRODUCTION,
        {
            **_envelope("frozen_verifier_reproduction"),
            "environments": reproduction,
            "all_identical": reproduced,
            "reference": _relative(dt1.GEOMETRY),
            "candidate_order_reference": _relative(cg1.EVALUATION_ORDER),
            "verifier": "SGV13 a4 joint refit, 250 uncertainty-acquired labels, SGV15's call",
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        FEATURE_COMPATIBILITY,
        {
            **_envelope("feature_compatibility"),
            "rule": cc_read_json(DESIGN_RECORD)["feature_compatibility_gate"]["rule"],
            "column_audit": audits,
            "invariance": invariance,
            "inert_by_family": inert,
            "undefined_columns": undefined,
            "zero_shot_feature_compatible": compatible,
            "reads_an_xr1_candidate": False,
            "verifier_reproduced": reproduced,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if not reproduced:
        raise PhaseError("the frozen verifier did not reproduce; the reliability study stops")
    print(f"compat: verifier reproduced {reproduced}; zero_shot_feature_compatible {compatible}")
    return 0


# ------------------------------------------------------------------ analysis outputs
#
# Everything below reads the raw answers only after their hashes are checked against the shard
# sidecars, and reads ground truth only inside `--normalize`. Every artifact it writes carries
# `uses_ground_truth = true` and selects nothing.

RAW_OUTPUT_MANIFEST = OUT / "raw_output_manifest.json"
RAW_OUTPUT_HASHES = OUT / "raw_output_hashes.json"
CANDIDATES = OUT / "xr1_candidates.parquet"
SITE_OUTCOMES = OUT / "proposal_site_outcomes.parquet"
PROPOSAL_LINKS = OUT / "proposal_links.parquet"
ERROR_SITES = OUT / "error_site_table.parquet"
P0_COVERAGE = OUT / "p0_coverage.parquet"
CANDIDATE_INVENTORY = OUT / "candidate_inventory.json"
CANDIDATE_NORMALIZATION = OUT / "candidate_normalization.json"
GENERATION_RESULTS = OUT / "generation_results.json"
OPPORTUNITY_RESULTS = OUT / "opportunity_results.json"
CLEAN_SITE_RESULTS = OUT / "clean_site_results.json"
ERROR_TYPE_ANALYSIS = OUT / "error_type_analysis.json"
ENGINE_ANALYSIS = OUT / "engine_analysis.json"
DOMAIN_ANALYSIS = OUT / "domain_analysis.json"
CANDIDATE_OVERLAP = OUT / "candidate_overlap.json"
UNION_RESULTS = OUT / "union_results.json"
IMAGE_SHUFFLE = OUT / "image_shuffle_control.json"
CROP_MASK = OUT / "crop_mask_control.json"
PROPOSAL_MISS_ORACLE = OUT / "proposal_miss_oracle.json"
ZERO_SHOT_SCORES = OUT / "zero_shot_scores.json"
RELIABILITY_DISCRIMINATION = OUT / "reliability_discrimination.json"
PREFIX_PURITY = OUT / "prefix_purity.json"
RISK_FRONTIER = OUT / "risk_frontier.json"
FROZEN_CUT_TRANSFER = OUT / "frozen_cut_transfer.json"
DISTRIBUTION_SHIFT = OUT / "candidate_distribution_shift.json"
CLEAN_SITE_RELIABILITY = OUT / "clean_site_reliability.json"
CEILING = OUT / "ceiling_decomposition.json"
CONTROL_RESULTS = OUT / "control_results.json"
FALSIFICATION = OUT / "falsification_tests.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"
DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"

LABEL_SITE_COLUMNS = xc1.LABEL_SITE_COLUMNS
LABEL_INPUT_COLUMNS = gen1.LABEL_INPUT_COLUMNS
METHOD_CANDIDATE_COLUMNS = xc1.METHOD_CANDIDATE_COLUMNS
OUTCOME_EXACT = gen1.OUTCOME_EXACT
G0_CONDITION = xc1.COND_FROZEN
PROBE = "xr1_identity_probe"
SITE_ERROR = "error"
SITE_CLEAN = "clean"
SITE_UNLABELABLE = "unlabelable"
HARMFUL_OUTCOMES = ("overcorrection", "miscorrection")
# The arms a site-outcome row is written for, and the denominator each one is judged on.
OUTCOME_ARMS = (G0, G1, G2, G3, X_IMG, X_MASK)


def _domain(corpus: str) -> str:
    return "modern_forms" if corpus == "funsd" else "historical_print"


def _arm_requests(sites: pd.DataFrame, arm: str) -> pd.DataFrame:
    """SGV-XC1's request columns under one arm's ids, so `_expand` and `_neural_edits` read them."""
    digest = sites["site_id"].astype(str).str.split("|", n=1).str[1]
    frame = sites.copy()
    frame["condition"] = CONDITION_OF_ARM[arm]
    frame["request_id"] = (sites["environment"] + f"|{arm}|" + digest).to_numpy()
    return frame[list(REQUEST_COLUMNS)].reset_index(drop=True)


def _identity_probes(sites: pd.DataFrame) -> pd.DataFrame:
    """One unchanged candidate per site: how `_labels` reports each site's truth status.

    The probe proposes nothing -- its text is the OCR region -- so its label says only whether
    the site's ground truth resolved and how far the OCR is from it. It never enters an arm.
    """
    local = sites["site_id"].astype(str).str.split("|", n=1).str[1]
    frame = pd.DataFrame(
        {
            "candidate_id": ("xr1-probe|" + sites["site_id"].astype(str)).to_numpy(),
            "site_id": local.to_numpy(),
            "document_id": sites["document_id"].astype(str).to_numpy(),
            "engine_id": sites["engine_id"].astype(str).to_numpy(),
            "original_ocr": sites["region"].astype(str).to_numpy(),
            "candidate_text": sites["region"].astype(str).to_numpy(),
            "char_start": sites["char_start"].astype(int).to_numpy(),
            "char_end": sites["char_end"].astype(int).to_numpy(),
            "anchor_kind": sites["anchor_kind"].astype(str).to_numpy(),
            "edit_kind": np.where(
                sites["anchor_kind"].astype(str) == "gap", xc1.EDIT_GAP, xc1.EDIT_REGION
            ),
            "generator_rank": 0,
        }
    )
    return frame[[c for c in LABEL_INPUT_COLUMNS if c in frame.columns]]


def _g0_frame(spec: dict[str, Any], documents: set[str]) -> pd.DataFrame:
    """The frozen generator's labelled candidates on the sampled documents, with their text.

    Labels and ids are SGV-CG1's, so every G0 row keys into the frozen verifier's published score
    vector; the replacement text comes from SGV-XC1's cache of the same SGV14 candidate stream.
    """
    name = spec["environment"]
    rows = pd.read_parquet(cg1.CANDIDATE_ROWS)
    rows = rows[
        (rows["environment"] == name)
        & (rows["role"] == s14.ROLE_EVALUATION)
        & rows["document_id"].astype(str).isin(documents)
    ].reset_index(drop=True)
    cache = pd.read_parquet(xc1.CACHE / f"{_slug(name)}.m0_candidates.parquet").set_index(
        "candidate_id"
    )
    missing = sorted(set(rows["candidate_id"].astype(str)) - set(cache.index.astype(str)))
    if missing:
        raise PhaseError(f"{name}: {len(missing)} G0 candidates have no cached text")
    text = cache.loc[rows["candidate_id"].astype(str)]
    anchor = rows["anchor_kind"].astype(str)
    return pd.DataFrame(
        {
            "environment": name,
            "corpus": spec["corpus"],
            "base_engine": spec["base_engine"],
            "method": G0,
            "condition": G0_CONDITION,
            "candidate_id": rows["candidate_id"].astype(str).to_numpy(),
            "site_id": rows["site_id"].astype(str).to_numpy(),
            "document_id": rows["document_id"].astype(str).to_numpy(),
            "role": rows["role"].astype(str).to_numpy(),
            "edit_kind": np.where(anchor == "gap", xc1.EDIT_GAP, xc1.EDIT_REGION),
            "anchor_kind": anchor.to_numpy(),
            "char_start": rows["char_start"].to_numpy(dtype=np.int64),
            "char_end": rows["char_end"].to_numpy(dtype=np.int64),
            "original_chars": rows["original_chars"].to_numpy(dtype=np.int64),
            "candidate_chars": rows["candidate_chars"].to_numpy(dtype=np.int64),
            "original_ocr": text["original_ocr"].astype(str).to_numpy(),
            "candidate_text": text["candidate_text"].astype(str).to_numpy(),
            "generator_rank": rows["generator_rank"].to_numpy(dtype=np.int64),
            "candidate_source": G0,
            "outcome": rows["outcome"].astype(str).to_numpy(),
            "is_harmful": rows["is_harmful"].to_numpy(dtype=bool),
            "beneficial": rows["beneficial"].to_numpy(dtype=bool),
            "exact": (rows["outcome"].astype(str) == OUTCOME_EXACT).to_numpy(dtype=bool),
            "d_before": rows["d_before"].to_numpy(dtype=np.int64),
            "d_after": rows["d_after"].to_numpy(dtype=np.int64),
        },
        columns=list(METHOD_CANDIDATE_COLUMNS),
    )


def union_candidates(g0: pd.DataFrame, g2: pd.DataFrame) -> pd.DataFrame:
    """G3: G0 united with G2's top-1 candidates, deduplicated on the atomic edit.

    Two candidates are the same edit when they propose the same text at the same site; their
    labels then agree by construction. Every source that proposed the edit is kept.
    """
    top = g2[g2["generator_rank"] < PRIMARY_K]
    both = pd.concat([g0, top], ignore_index=True)
    key = ["site_id", "candidate_text"]
    sources = both.groupby(key, sort=True)["candidate_source"].agg(
        lambda s: ",".join(sorted(set(s)))
    )
    disagree = both.groupby(key)["outcome"].nunique()
    if int((disagree > 1).sum()):
        raise PhaseError("one atomic edit carries two labels; the union cannot be formed")
    kept = (
        both.sort_values([*key, "generator_rank", "candidate_source"], kind="stable")
        .drop_duplicates(key, keep="first")
        .reset_index(drop=True)
    )
    kept["candidate_sources"] = [
        sources.loc[(s, t)] for s, t in zip(kept["site_id"], kept["candidate_text"], strict=True)
    ]
    kept["method"] = G3
    kept["condition"] = "union"
    kept["candidate_id"] = [
        canonical_hash({"arm": G3, "site": s, "text": t})
        for s, t in zip(kept["site_id"], kept["candidate_text"], strict=True)
    ]
    return kept


def _verified_shards() -> list[dict[str, Any]]:
    """Section 15: every raw shard's bytes against its sidecar, before any label is computed."""
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
                    "raw_output_sha256": file_sha256(path),
                    "sidecar": _relative(_sidecar(path)),
                    "requests": int(sidecar["requests"]),
                    "site_count": int(sidecar["site_count"]),
                    "issued_utc": sidecar["issued_utc"],
                }
            )
    return manifest


def run_normalize() -> int:
    """Raw answers -> hashes -> SGV-XC1 atomic edits -> SGV14 labels -> SGV-CG1 links.

    Every arm passes through the same parse, `_neural_edits` and `_labels` call, and ground truth
    is first read here, after every raw shard has been verified against its sidecar hash.
    """
    from ocr_risk.canonical import rebuild_stream
    from ocr_risk.schemas.enums import AnchorKind

    started = time.monotonic()
    outputs = (
        CANDIDATES,
        SITE_OUTCOMES,
        PROPOSAL_LINKS,
        ERROR_SITES,
        P0_COVERAGE,
        RAW_OUTPUT_MANIFEST,
        RAW_OUTPUT_HASHES,
        CANDIDATE_INVENTORY,
        CANDIDATE_NORMALIZATION,
    )
    for path in outputs:
        _forbid(path)
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
    sample = pd.read_parquet(PROPOSAL_SAMPLE)
    xc1_alignment = pd.read_parquet(xc1.ALIGNMENT_SITES)
    candidate_frames: list[pd.DataFrame] = []
    census_rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    truth_frames: list[pd.DataFrame] = []
    link_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    alignment_checks: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        sites = sample[sample["environment"] == name].reset_index(drop=True)
        documents = set(sites["document_id"].astype(str))
        frames: list[tuple[str, str, pd.DataFrame]] = []
        for arm in MODEL_ARMS:
            raw = pd.read_parquet(_shard_path(arm, name))
            requests = _arm_requests(sites, arm)
            expanded, census = gen1._expand(raw, requests, arm, CONDITION_OF_ARM[arm])
            for row in census:
                row["environment"] = name
            census_rows.extend(census)
            _s, proposals, shard = xc1._neural_edits(name, arm, expanded, requests, {}, None)
            frames.append((arm, CONDITION_OF_ARM[arm], proposals))
            shards.append(
                {"environment": name, "method": arm, **shard, "proposals": len(proposals)}
            )

        bundles = s14._document_bundles(spec["corpus"])
        roles = s14.partition_of(spec["corpus"], sorted(bundles))
        spans, _record = s14._canonical_spans(spec, bundles)
        evaluation = {p: g for p, g in spans.items() if roles.get(p[0]) == s14.ROLE_EVALUATION}
        streams = {pair: rebuild_stream(group) for pair, group in evaluation.items() if group}
        label_sites = sites[list(LABEL_SITE_COLUMNS)].assign(
            site_id=sites["site_id"].astype(str).str.split("|", n=1).str[1]
        )
        probes = _identity_probes(sites)
        present = [(a, c, f) for a, c, f in frames if not f.empty]
        label_input = pd.concat(
            [probes, *[f[LABEL_INPUT_COLUMNS] for _a, _c, f in present]], ignore_index=True
        )
        labels, _diagnostics = s14._labels(
            spec["corpus"], bundles, evaluation, streams, label_sites, label_input
        )
        for arm, condition, frame in present:
            labelled = xc1._method_frame(spec, arm, condition, frame, labels, roles)
            candidate_frames.append(labelled)
            shards.append(
                {
                    "environment": name,
                    "method": arm,
                    "labelled_candidates": len(labelled),
                    "unlabelable_removed": len(frame) - len(labelled),
                }
            )
        candidate_frames.append(_g0_frame(spec, documents))

        probe_labels = labels.set_index("candidate_id").loc[probes["candidate_id"]]
        labelable = probe_labels["labelable"].to_numpy(dtype=bool)
        d_before = probe_labels["d_before"].to_numpy(dtype=np.int64)
        truth_frames.append(
            pd.DataFrame(
                {
                    "site_id": sites["site_id"].astype(str).to_numpy(),
                    "labelable": labelable,
                    "site_d_before": np.where(labelable, d_before, -1),
                    "site_class": np.where(
                        ~labelable,
                        SITE_UNLABELABLE,
                        np.where(d_before > 0, SITE_ERROR, SITE_CLEAN),
                    ),
                }
            )
        )

        # The error-site population and every P0 link, on every evaluation page of the
        # environment, so proposal recall is also known outside the sample.
        alignment, indexes, by_span, by_alignment, objects = xc1._alignment_population(
            spec, evaluation, bundles, roles
        )
        reference = xc1_alignment[
            (xc1_alignment["environment"] == name) & (xc1_alignment["role"] == s14.ROLE_EVALUATION)
        ]
        mine = alignment[alignment["role"] == s14.ROLE_EVALUATION]
        key = ["align_site_id", "site_kind", "evaluable", "d_before"]
        agree = bool(
            mine[key]
            .sort_values("align_site_id")
            .reset_index(drop=True)
            .equals(reference[key].sort_values("align_site_id").reset_index(drop=True))
        )
        alignment_checks.append({"environment": name, "identical_to_sgv_xc1": agree})
        if not agree:
            raise PhaseError(f"{name}: the rebuilt alignment differs from SGV-XC1's")
        all_p0 = _p0_requests(name)
        p0_sites = all_p0.assign(site_id=all_p0["site_id"].astype(str).str.split("|", n=1).str[1])
        links = cg1._site_links(name, p0_sites, indexes, by_span, by_alignment, AnchorKind)
        errors = mine[mine["evaluable"] & (mine["d_before"] > 0)]
        texts = {
            f"{name}|{site.site_id}": (str(site.ocr_text), str(site.gt_text))
            for _pair, site in objects
        }
        covered = set(links["align_site_id"].astype(str))
        coverage_frames.append(
            pd.DataFrame(
                {
                    "environment": name,
                    "align_site_id": errors["align_site_id"].astype(str).to_numpy(),
                    "document_id": errors["document_id"].astype(str).to_numpy(),
                    "site_kind": errors["site_kind"].astype(str).to_numpy(),
                    "covered_by_p0": errors["align_site_id"].astype(str).isin(covered).to_numpy(),
                    "in_sampled_documents": errors["document_id"]
                    .astype(str)
                    .isin(documents)
                    .to_numpy(),
                }
            )
        )
        sampled_errors = errors[errors["document_id"].astype(str).isin(documents)]
        ocr = [texts[str(a)][0] for a in sampled_errors["align_site_id"]]
        gt = [texts[str(a)][1] for a in sampled_errors["align_site_id"]]
        kinds = sampled_errors["site_kind"].astype(str).tolist()
        error_frames.append(
            pd.DataFrame(
                {
                    "environment": name,
                    "corpus": spec["corpus"],
                    "base_engine": spec["base_engine"],
                    "align_site_id": sampled_errors["align_site_id"].astype(str).to_numpy(),
                    "document_id": sampled_errors["document_id"].astype(str).to_numpy(),
                    "site_kind": kinds,
                    "d_before": sampled_errors["d_before"].to_numpy(dtype=np.int64),
                    "subtype": [
                        gen1.substitution_subtype(o, g)
                        if k == "substitution"
                        else gen1.segmentation_subtype(o, g)
                        if k == "segmentation"
                        else k
                        for o, g, k in zip(ocr, gt, kinds, strict=True)
                    ],
                    "ocr_chars": [len(o) for o in ocr],
                    "gt_chars": [len(g) for g in gt],
                }
            )
        )
        sampled_ids = set(sites["site_id"].astype(str))
        link_frames.append(links[links["site_id"].astype(str).isin(sampled_ids)])
        print(
            f"  {name}: {len(label_input)} labels, {len(links)} links, {len(errors)} error sites "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )

    candidates = pd.concat(candidate_frames, ignore_index=True)
    g3 = union_candidates(
        candidates[candidates["method"] == G0], candidates[candidates["method"] == G2]
    )
    candidates = pd.concat(
        [candidates.assign(candidate_sources=candidates["candidate_source"]), g3],
        ignore_index=True,
    )
    truth = pd.concat(truth_frames, ignore_index=True)
    links = pd.concat(link_frames, ignore_index=True)
    census = pd.DataFrame(census_rows)
    outcomes = site_outcomes(sample, truth, candidates, census, links, pd.concat(error_frames))
    _write_parquet_once(CANDIDATES, candidates)
    _write_parquet_once(SITE_OUTCOMES, outcomes)
    _write_parquet_once(PROPOSAL_LINKS, links)
    _write_parquet_once(ERROR_SITES, pd.concat(error_frames, ignore_index=True))
    _write_parquet_once(P0_COVERAGE, pd.concat(coverage_frames, ignore_index=True))
    _write_json_once(
        RAW_OUTPUT_MANIFEST,
        {**_analysis_envelope("raw_output_manifest"), "shards": manifest, "count": len(manifest)},
    )
    _write_json_once(
        CANDIDATE_NORMALIZATION,
        {
            **_analysis_envelope("candidate_normalization"),
            "path": (
                "raw answer -> parse_candidates -> one row per listed candidate -> SGV-XC1 "
                "_neural_edits (identity and duplicates dropped, rank kept) -> SGV14 _labels -> "
                "SGV-XC1 _method_frame; G0 read from SGV-CG1's labelled table"
            ),
            "shards": shards,
            "parse_status": {
                arm: {
                    str(k): int(v)
                    for k, v in census[census["method"] == arm]["parse_status"]
                    .value_counts()
                    .items()
                }
                for arm in MODEL_ARMS
            },
            "alignment_reproduction": alignment_checks,
            "union": {
                "g3_candidates": int((candidates["method"] == G3).sum()),
                "attribution": {
                    str(k): int(v)
                    for k, v in candidates[candidates["method"] == G3]["candidate_sources"]
                    .value_counts()
                    .items()
                },
                "rule": "deduplicated on (site, replacement text); every proposing source kept",
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(CANDIDATE_INVENTORY, _inventory_record(candidates, truth, started))
    print(f"normalize: {len(candidates)} labelled candidates, {len(outcomes)} site outcomes")
    return 0


def _inventory_record(
    candidates: pd.DataFrame, truth: pd.DataFrame, started: float
) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm, block in candidates.groupby("method", sort=True):
        by_arm[str(arm)] = {
            "candidates": len(block),
            "exact": int(block["exact"].sum()),
            "beneficial": int(block["beneficial"].sum()),
            "harmful": int(block["is_harmful"].sum()),
            "by_outcome": {str(k): int(v) for k, v in block["outcome"].value_counts().items()},
            "sites_with_a_candidate": int(block["site_id"].nunique()),
        }
    return {
        **_analysis_envelope("candidate_inventory"),
        "by_arm": by_arm,
        "site_classes": {str(k): int(v) for k, v in truth["site_class"].value_counts().items()},
        "sampled_sites": len(truth),
        "runtime_seconds": time.monotonic() - started,
    }


def site_outcomes(
    sample: pd.DataFrame,
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
    census: pd.DataFrame,
    links: pd.DataFrame,
    errors: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (arm, sampled proposal): the table every site-level analysis reads.

    Primary arms are judged on every sampled proposal, the two controls on the control subsample.
    A site an arm never received a request for -- an image arm at a site with no crop -- stays in
    the table as unrequested, never as an absent row.
    """
    kinds = links.merge(errors[["align_site_id", "site_kind"]], on="align_site_id", how="inner")
    linked_kinds = kinds.groupby("site_id")["site_kind"].agg(lambda s: ",".join(sorted(set(s))))
    meta = sample[
        [
            "environment",
            "site_id",
            "document_id",
            "anchor_kind",
            "in_control_subsample",
            "region",
        ]
    ].merge(truth, on="site_id", validate="one_to_one")
    by_env = {spec["environment"]: spec for spec in environment_specs()}
    meta["corpus"] = meta["environment"].map(lambda n: by_env[n]["corpus"])
    meta["base_engine"] = meta["environment"].map(lambda n: by_env[n]["base_engine"])
    meta["domain"] = meta["corpus"].map(_domain)
    meta["linked_error_kinds"] = meta["site_id"].map(linked_kinds).fillna("")
    frame = candidates.copy()
    frame["exact_at_1"] = frame["exact"] & (frame["generator_rank"] < 1)
    frame["exact_at_5"] = frame["exact"] & (frame["generator_rank"] < SECONDARY_K)
    frame["overcorrection"] = frame["outcome"] == "overcorrection"
    frame["miscorrection"] = frame["outcome"] == "miscorrection"
    grouped = frame.groupby(["method", "site_id"], sort=True)
    aggregate = grouped.agg(
        exact_at_1=("exact_at_1", "any"),
        exact_at_5=("exact_at_5", "any"),
        exact_any=("exact", "any"),
        candidates=("candidate_id", "size"),
        harmful_candidates=("is_harmful", "sum"),
        beneficial_candidates=("beneficial", "sum"),
        exact_candidates=("exact", "sum"),
        overcorrections=("overcorrection", "sum"),
        miscorrections=("miscorrection", "sum"),
    ).reset_index()
    top = (
        frame.sort_values(["method", "site_id", "generator_rank"], kind="stable")
        .drop_duplicates(["method", "site_id"])[
            ["method", "site_id", "candidate_text", "outcome", "is_harmful", "exact", "d_after"]
        ]
        .rename(
            columns={
                "candidate_text": "top1_text",
                "outcome": "top1_outcome",
                "is_harmful": "top1_harmful",
                "exact": "top1_exact",
                "d_after": "top1_d_after",
            }
        )
    )
    requested = (
        census.groupby(["method", "site_id"], sort=True)
        .agg(
            parse_status=("parse_status", "first"),
            valid_output=("valid_output", "any"),
            abstained=("abstained", "any"),
            proposals=("proposals", "sum"),
        )
        .reset_index()
    )
    rows: list[pd.DataFrame] = []
    for arm in OUTCOME_ARMS:
        base = meta if arm not in CONTROL_ARMS else meta[meta["in_control_subsample"]]
        block = base.assign(method=arm).merge(
            aggregate[aggregate["method"] == arm], on=["method", "site_id"], how="left"
        )
        block = block.merge(top[top["method"] == arm], on=["method", "site_id"], how="left")
        if arm in MODEL_ARMS:
            block = block.merge(
                requested[requested["method"] == arm], on=["method", "site_id"], how="left"
            )
            block["requested"] = block["parse_status"].notna()
        else:
            # G0 runs at every proposal; G3 inherits G0's exposure.
            block["requested"] = True
            block["parse_status"] = "frozen_generator"
            block["valid_output"] = True
            block["proposals"] = block["candidates"].fillna(0)
            block["abstained"] = block["candidates"].fillna(0) == 0
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    for column in (
        "exact_at_1",
        "exact_at_5",
        "exact_any",
        "top1_harmful",
        "top1_exact",
        "valid_output",
        "abstained",
    ):
        out[column] = out[column].astype("boolean").fillna(False).astype(bool)
    for column in (
        "candidates",
        "harmful_candidates",
        "beneficial_candidates",
        "exact_candidates",
        "overcorrections",
        "miscorrections",
        "proposals",
    ):
        out[column] = out[column].fillna(0).astype(np.int64)
    out["top1_outcome"] = out["top1_outcome"].fillna("")
    out["rewrote"] = out["requested"] & (out["top1_outcome"] != "")
    return out


# ------------------------------------------------------------------ the opportunity view


@dataclass(frozen=True, slots=True)
class EnvironmentView:
    """One environment's sampled error sites and the P0 links onto them."""

    name: str
    corpus: str
    base_engine: str
    error_ids: np.ndarray
    error_kind: np.ndarray
    error_subtype: np.ndarray
    error_document: np.ndarray
    position: dict[str, int]
    links: dict[str, np.ndarray]
    covered: np.ndarray

    @property
    def count(self) -> int:
        return int(self.error_ids.size)

    def repaired(self, sites: set[str]) -> np.ndarray:
        """The error sites reachable from a set of proposal sites."""
        out = np.zeros(self.count, dtype=bool)
        for site in sites:
            index = self.links.get(site)
            if index is not None:
                out[index] = True
        return out


def load_views() -> dict[str, EnvironmentView]:
    errors = pd.read_parquet(ERROR_SITES)
    links = pd.read_parquet(PROPOSAL_LINKS)
    by_env = {spec["environment"]: spec for spec in environment_specs()}
    views: dict[str, EnvironmentView] = {}
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
        views[name] = EnvironmentView(
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


def exact_sites(candidates: pd.DataFrame, arm: str, k: int | None) -> dict[str, set[str]]:
    """Per environment: the proposal sites carrying an exact candidate within the arm's budget."""
    block = candidates[(candidates["method"] == arm) & candidates["exact"]]
    if k is not None:
        block = block[block["generator_rank"] < k]
    return {
        str(name): set(group["site_id"].astype(str))
        for name, group in block.groupby("environment", sort=True)
    }


ARM_BUDGETS: dict[str, int | None] = {G0: None, G1: PRIMARY_K, G2: PRIMARY_K, G3: None}


def repaired_by_arm(
    views: dict[str, EnvironmentView], candidates: pd.DataFrame
) -> dict[str, dict[str, np.ndarray]]:
    """For every arm and environment, which sampled error sites it offers an exact repair for."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm, k in (*ARM_BUDGETS.items(), (f"{G1}@5", SECONDARY_K), (f"{G2}@5", SECONDARY_K)):
        source = arm.split("@")[0]
        sites = exact_sites(candidates, source, k)
        out[arm] = {name: view.repaired(sites.get(name, set())) for name, view in views.items()}
    return out


def _rate(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def opportunity_table(
    views: dict[str, EnvironmentView], repaired: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    """Opportunity per arm: per environment, the mean over environments, and pooled."""
    table: dict[str, Any] = {}
    for arm, per_env in repaired.items():
        rates = {name: _rate(per_env[name]) for name in views}
        table[arm] = {
            "per_environment": rates,
            "mean": float(np.mean(list(rates.values()))),
            "pooled": _ratio(
                int(sum(per_env[n].sum() for n in views)), int(sum(v.count for v in views.values()))
            ),
            "repaired_error_sites": int(sum(per_env[n].sum() for n in views)),
        }
    return table


def run_opportunity() -> int:
    """Sections 16-22: proposal recall, exact generation, opportunity, burden, clean sites."""
    started = time.monotonic()
    _require(CANDIDATES, "normalize")
    for path in (
        GENERATION_RESULTS,
        OPPORTUNITY_RESULTS,
        CLEAN_SITE_RESULTS,
        ERROR_TYPE_ANALYSIS,
        ENGINE_ANALYSIS,
        DOMAIN_ANALYSIS,
        CANDIDATE_OVERLAP,
        UNION_RESULTS,
    ):
        _forbid(path)
    candidates = pd.read_parquet(CANDIDATES)
    outcomes = pd.read_parquet(SITE_OUTCOMES)
    coverage = pd.read_parquet(P0_COVERAGE)
    views = load_views()
    repaired = repaired_by_arm(views, candidates)
    opportunity = opportunity_table(views, repaired)
    total = int(sum(v.count for v in views.values()))

    recall = {name: _rate(view.covered) for name, view in views.items()}
    full = {
        str(name): _rate(block["covered_by_p0"].to_numpy(dtype=bool))
        for name, block in coverage.groupby("environment", sort=True)
    }
    primary = [G0, G1, G2, G3]
    generation: dict[str, Any] = {}
    for arm in primary:
        block = outcomes[outcomes["method"] == arm]
        errors_block = block[block["site_class"] == SITE_ERROR]
        per_env = {
            str(name): _rate(group["top1_exact"].to_numpy(dtype=bool))
            for name, group in errors_block[errors_block["requested"]].groupby(
                "environment", sort=True
            )
        }
        arm_candidates = candidates[candidates["method"] == arm]
        covered = {name: int(view.covered.sum()) for name, view in views.items()}
        success = {name: _ratio(int(repaired[arm][name].sum()), covered[name]) for name in views}
        generation[arm] = {
            "requested_sites": int(block["requested"].sum()),
            "requested_error_sites": int(errors_block["requested"].sum()),
            "conditional_exact_generation": {
                "per_environment": per_env,
                "mean": float(np.mean(list(per_env.values()))) if per_env else 0.0,
                "definition": "error P0 sites whose top-1 candidate is exact / requested error "
                "P0 sites; for G0 and G3 any candidate ranked first by the frozen order",
            },
            "generation_success_given_reached": {
                "per_environment": success,
                "mean": float(np.mean(list(success.values()))),
            },
            "candidates": len(arm_candidates),
            "candidate_precision": _ratio(int(arm_candidates["exact"].sum()), len(arm_candidates)),
            "harmful_candidates": int(arm_candidates["is_harmful"].sum()),
            "harmful_share": _ratio(int(arm_candidates["is_harmful"].sum()), len(arm_candidates)),
            "harmful_per_requested_site": _ratio(
                int(arm_candidates["is_harmful"].sum()), int(block["requested"].sum())
            ),
            "overcorrections": int((arm_candidates["outcome"] == "overcorrection").sum()),
            "miscorrections": int((arm_candidates["outcome"] == "miscorrection").sum()),
            "candidates_per_requested_site": _ratio(
                len(arm_candidates), int(block["requested"].sum())
            ),
            "harmful_per_repaired_error_site": _ratio(
                int(arm_candidates["is_harmful"].sum()), opportunity[arm]["repaired_error_sites"]
            ),
        }
    for arm in (G1, G2):
        block = outcomes[(outcomes["method"] == arm) & outcomes["requested"]]
        generation[arm]["abstention_rate"] = _rate(block["abstained"].to_numpy(dtype=bool))
        generation[arm]["valid_share"] = _rate(block["valid_output"].to_numpy(dtype=bool))
        generation[arm]["abstention_by_site_class"] = {
            str(k): _rate(g["abstained"].to_numpy(dtype=bool))
            for k, g in block.groupby("site_class", sort=True)
        }
        generation[arm]["abstention_by_error_type"] = {
            kind: _rate(
                block[block["linked_error_kinds"].str.split(",").map(lambda s, k=kind: k in s)][
                    "abstained"
                ].to_numpy(dtype=bool)
            )
            for kind in ("substitution", "deletion", "insertion", "segmentation")
        }
    _write_json_once(
        GENERATION_RESULTS,
        {
            **_analysis_envelope("generation_results"),
            "site_proposal_recall": {
                "sample_per_environment": recall,
                "sample_mean": float(np.mean(list(recall.values()))),
                "population_per_environment": full,
                "population_mean": float(np.mean(list(full.values()))),
                "identical_across_g1_and_g2": True,
                "note": "P0 is shared by every arm, so proposal recall is one number per page set",
            },
            "arms": generation,
            "error_sites_sampled": total,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    translation = {
        "visual_translation_gain": opportunity[G2]["mean"] - opportunity[G1]["mean"],
        "gain_vs_current": opportunity[G2]["mean"] - opportunity[G0]["mean"],
        "union_gain": opportunity[G3]["mean"] - opportunity[G0]["mean"],
        "visual_translation_gain_at_5": opportunity[f"{G2}@5"]["mean"]
        - opportunity[f"{G1}@5"]["mean"],
    }
    per_env_gain = {
        arm: {
            name: opportunity[arm]["per_environment"][name]
            - opportunity[G0]["per_environment"][name]
            for name in views
        }
        for arm in (G2, G3)
    }
    _write_json_once(
        OPPORTUNITY_RESULTS,
        {
            **_analysis_envelope("opportunity_results"),
            "opportunity": opportunity,
            "translation": translation,
            "gain_over_g0_per_environment": per_env_gain,
            "environments_reaching_margin": {
                arm: int(sum(v >= OPPORTUNITY_MARGIN for v in gains.values()))
                for arm, gains in per_env_gain.items()
            },
            "margin": OPPORTUNITY_MARGIN,
            "error_sites_sampled": total,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    clean: dict[str, Any] = {}
    for arm in (G0, G1, G2):
        block = outcomes[(outcomes["method"] == arm) & (outcomes["site_class"] == SITE_CLEAN)]
        asked = block[block["requested"]]
        per_env = {
            str(name): _rate(group["rewrote"].to_numpy(dtype=bool))
            for name, group in asked.groupby("environment", sort=True)
        }
        clean[arm] = {
            "clean_sites": len(block),
            "requested_clean_sites": len(asked),
            "rewrite_rate": _rate(asked["rewrote"].to_numpy(dtype=bool)),
            "rewrite_rate_per_environment": per_env,
            "rewrite_rate_mean": float(np.mean(list(per_env.values()))) if per_env else 0.0,
            "overcorrection_rate": _rate(
                (asked["top1_outcome"] == "overcorrection").to_numpy(dtype=bool)
            ),
            "abstention_rate": _rate(asked["abstained"].to_numpy(dtype=bool)),
            "harmful_candidates_at_clean_sites": int(asked["harmful_candidates"].sum()),
        }
    _write_json_once(
        CLEAN_SITE_RESULTS,
        {
            **_analysis_envelope("clean_site_results"),
            "arms": clean,
            "site_classes": {
                str(k): int(v)
                for k, v in outcomes[outcomes["method"] == G2]["site_class"].value_counts().items()
            },
            "note": (
                "a clean site is a labelable proposal whose OCR region equals its ground truth; "
                "any change there is an overcorrection by the frozen taxonomy"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )

    kinds = ("substitution", "deletion", "insertion", "segmentation")
    by_type: dict[str, Any] = {}
    for arm, per_env in repaired.items():
        rows: dict[str, Any] = {}
        for kind in kinds:
            hit = sum(int(per_env[n][views[n].error_kind == kind].sum()) for n in views)
            size = sum(int((views[n].error_kind == kind).sum()) for n in views)
            rows[kind] = {"repaired": hit, "error_sites": size, "opportunity": _ratio(hit, size)}
        by_type[arm] = rows
    glyph: dict[str, Any] = {}
    for arm, per_env in repaired.items():
        hit = sum(
            int(per_env[n][views[n].error_subtype == "historical_glyph"].sum()) for n in views
        )
        size = sum(int((views[n].error_subtype == "historical_glyph").sum()) for n in views)
        glyph[arm] = {"repaired": hit, "error_sites": size, "opportunity": _ratio(hit, size)}
    weak_gain = {
        arm: {
            kind: by_type[arm][kind]["opportunity"] - by_type[G0][kind]["opportunity"]
            for kind in WEAK_TYPES
        }
        for arm in (G2, G3)
    }
    _write_json_once(
        ERROR_TYPE_ANALYSIS,
        {
            **_analysis_envelope("error_type_analysis"),
            "by_arm": by_type,
            "weak_type_gain_over_g0": weak_gain,
            "weak_types_reaching_margin": {
                arm: [k for k, v in gains.items() if v >= WEAK_TYPE_MARGIN]
                for arm, gains in weak_gain.items()
            },
            "historical_glyph": glyph,
            "error_sites_by_type": {
                kind: int(sum(int((v.error_kind == kind).sum()) for v in views.values()))
                for kind in kinds
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )

    engines: dict[str, Any] = {}
    domains: dict[str, Any] = {}
    for arm, per_env in repaired.items():
        by_engine: dict[str, list[np.ndarray]] = {}
        by_domain: dict[str, list[np.ndarray]] = {}
        for name, view in views.items():
            by_engine.setdefault(view.base_engine, []).append(per_env[name])
            by_domain.setdefault(_domain(view.corpus), []).append(per_env[name])
        engines[arm] = {
            engine: {
                "opportunity": _rate(np.concatenate(v)),
                "error_sites": int(sum(x.size for x in v)),
            }
            for engine, v in by_engine.items()
        }
        domains[arm] = {
            domain: {
                "opportunity": _rate(np.concatenate(v)),
                "error_sites": int(sum(x.size for x in v)),
            }
            for domain, v in by_domain.items()
        }
    _write_json_once(
        ENGINE_ANALYSIS,
        {
            **_analysis_envelope("engine_analysis"),
            "by_environment": {arm: opportunity[arm]["per_environment"] for arm in opportunity},
            "by_base_engine": engines,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DOMAIN_ANALYSIS,
        {
            **_analysis_envelope("domain_analysis"),
            "by_domain": domains,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    shared = only_g0 = only_g2 = 0
    by_kind_overlap: dict[str, dict[str, int]] = {}
    for name, view in views.items():
        a, b = repaired[G0][name], repaired[G2][name]
        shared += int((a & b).sum())
        only_g0 += int((a & ~b).sum())
        only_g2 += int((~a & b).sum())
        for kind in kinds:
            mask = view.error_kind == kind
            row = by_kind_overlap.setdefault(kind, {"shared": 0, "g0_only": 0, "g2_only": 0})
            row["shared"] += int((a & b & mask).sum())
            row["g0_only"] += int((a & ~b & mask).sum())
            row["g2_only"] += int((~a & b & mask).sum())
    union_size = shared + only_g0 + only_g2
    _write_json_once(
        CANDIDATE_OVERLAP,
        {
            **_analysis_envelope("candidate_overlap"),
            "unit": "sampled OCR error site repaired within each arm's budget (G2 at K=1)",
            "shared": shared,
            "g0_only": only_g0,
            "g2_only": only_g2,
            "union": union_size,
            "jaccard": _ratio(shared, union_size),
            "by_error_type": by_kind_overlap,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    union = candidates[candidates["method"] == G3]
    best_single = max(opportunity[G0]["mean"], opportunity[G2]["mean"])
    _write_json_once(
        UNION_RESULTS,
        {
            **_analysis_envelope("union_results"),
            "g3_opportunity": opportunity[G3],
            "g0_opportunity_mean": opportunity[G0]["mean"],
            "g2_opportunity_mean": opportunity[G2]["mean"],
            "union_minus_best_single": opportunity[G3]["mean"] - best_single,
            "union_minus_g0": opportunity[G3]["mean"] - opportunity[G0]["mean"],
            "candidates": len(union),
            "harmful_candidates": int(union["is_harmful"].sum()),
            "attribution": {
                str(k): int(v) for k, v in union["candidate_sources"].value_counts().items()
            },
            "is_production_method": False,
            "note": "complementarity is recorded and never overrides the outcome (section 47)",
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "opportunity: "
        + ", ".join(f"{arm} {opportunity[arm]['mean']:.4f}" for arm in (G0, G1, G2, G3))
        + f"; visual translation {translation['visual_translation_gain']:+.4f}"
    )
    return 0


# ------------------------------------------------------------------ sections 37-42: controls


def _paired_control_sites(outcomes: pd.DataFrame, control: str) -> pd.DataFrame:
    """Control-subsample sites that G1, G2 and the control arm were all asked about."""
    wide = outcomes[outcomes["in_control_subsample"]].pivot_table(
        index=["environment", "site_id", "document_id", "site_class"],
        columns="method",
        values=["requested", "top1_exact", "top1_text"],
        aggfunc="first",
    )
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    keep = (
        wide[f"requested__{G1}"].fillna(False).astype(bool)
        & wide[f"requested__{G2}"].fillna(False).astype(bool)
        & wide[f"requested__{control}"].fillna(False).astype(bool)
    )
    return wide[keep].reset_index(drop=True)


def control_effect(outcomes: pd.DataFrame, control: str) -> dict[str, Any]:
    """Section 37/38: what the control does to G2's exact top-1 sites, on paired sites."""
    paired = _paired_control_sites(outcomes, control)
    g1 = paired[f"top1_exact__{G1}"].fillna(False).astype(bool).to_numpy()
    g2 = paired[f"top1_exact__{G2}"].fillna(False).astype(bool).to_numpy()
    ctrl = paired[f"top1_exact__{control}"].fillna(False).astype(bool).to_numpy()
    gain = _rate(g2) - _rate(g1)
    retention = (_rate(ctrl) - _rate(g1)) / gain if gain > 0 else None
    same_text = (
        paired[f"top1_text__{control}"].astype(str) == paired[f"top1_text__{G2}"].astype(str)
    ).to_numpy()
    survived = int((g2 & ctrl & same_text).sum())
    per_env = {
        str(name): {
            "sites": len(group),
            G1: _rate(group[f"top1_exact__{G1}"].fillna(False).astype(bool).to_numpy()),
            G2: _rate(group[f"top1_exact__{G2}"].fillna(False).astype(bool).to_numpy()),
            control: _rate(group[f"top1_exact__{control}"].fillna(False).astype(bool).to_numpy()),
        }
        for name, group in paired.groupby("environment", sort=True)
    }
    return {
        "paired_sites": len(paired),
        "paired_error_sites": int((paired["site_class"] == SITE_ERROR).sum()),
        "exact_top1_rate": {G1: _rate(g1), G2: _rate(g2), control: _rate(ctrl)},
        "exact_top1_sites": {G1: int(g1.sum()), G2: int(g2.sum()), control: int(ctrl.sum())},
        "control_minus_g2": _rate(ctrl) - _rate(g2),
        "g2_minus_g1": gain,
        "retention_of_visual_gain": retention,
        "g2_exact_sites_surviving_unchanged": survived,
        "survival_share": _ratio(survived, int(g2.sum())),
        "per_environment": per_env,
        "unit": "control-subsample proposal requested by G1, G2 and the control",
    }


def _control_request_counts() -> dict[str, int]:
    """How many control-subsample sites received each control, from the frozen GT-blind tables."""
    sample = pd.read_parquet(PROPOSAL_SAMPLE)
    controls = pd.read_parquet(CONTROL_CONTEXTS)
    subsample = int(sample["in_control_subsample"].sum())
    masked = int((controls["arm"] == X_MASK).sum())
    return {
        "control_subsample": subsample,
        "shuffled_crops": int((controls["arm"] == X_IMG).sum()),
        "masked_crops": masked,
        "mask_unavailable": subsample - masked,
        "sampled_sites_with_a_site_level_crop": int(
            (pd.read_parquet(CONTEXTS)["crop_level"] == CROP_SITE).sum()
        ),
    }


def run_controls() -> int:
    """Sections 37-41: the image shuffle, the crop mask, duplication and removal."""
    started = time.monotonic()
    _require(SITE_OUTCOMES, "normalize")
    for path in (IMAGE_SHUFFLE, CROP_MASK, CONTROL_RESULTS):
        _forbid(path)
    outcomes = pd.read_parquet(SITE_OUTCOMES)
    candidates = pd.read_parquet(CANDIDATES)
    views = load_views()
    shuffle = control_effect(outcomes, X_IMG)
    mask = control_effect(outcomes, X_MASK)
    _write_json_once(
        IMAGE_SHUFFLE,
        {
            **_analysis_envelope("image_shuffle_control"),
            **shuffle,
            "expectation": "exact generation declines when the crop shows another document",
            "deployable_arm": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        CROP_MASK,
        {
            **_analysis_envelope("crop_mask_control"),
            **mask,
            "expectation": "exact generation declines when the crop's informative centre is hidden",
            "deployable_arm": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    duplicated = pd.concat(
        [
            candidates,
            candidates[candidates["method"] == G2].assign(
                candidate_id=lambda f: f["candidate_id"] + "|dup"
            ),
        ],
        ignore_index=True,
    )
    before = repaired_by_arm(views, candidates)
    after = repaired_by_arm(views, duplicated)
    duplication = all(
        np.array_equal(before[G2][n], after[G2][n]) and np.array_equal(before[G3][n], after[G3][n])
        for n in views
    )
    removed = candidates[~((candidates["method"] == G2) & candidates["exact"])]
    stripped = repaired_by_arm(views, removed)
    g2_only = sum(int((before[G2][n] & ~before[G0][n]).sum()) for n in views)
    removal = {
        "g2_repairs_after_removal": int(sum(stripped[G2][n].sum() for n in views)),
        "g2_repairs_before": int(sum(before[G2][n].sum() for n in views)),
        "g2_only_repairs": g2_only,
        "exact": bool(all(stripped[G2][n].sum() == 0 for n in views)),
    }
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "image_shuffle": {
                k: shuffle[k]
                for k in (
                    "paired_sites",
                    "control_minus_g2",
                    "retention_of_visual_gain",
                    "survival_share",
                )
            },
            "crop_mask": {
                k: mask[k]
                for k in (
                    "paired_sites",
                    "control_minus_g2",
                    "retention_of_visual_gain",
                    "survival_share",
                )
            },
            "duplication": {"opportunity_unchanged": bool(duplication)},
            "control_requests": _control_request_counts(),
            "removal": removal,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"controls: shuffle {shuffle['control_minus_g2']:+.4f}, "
        f"mask {mask['control_minus_g2']:+.4f} "
        f"on {shuffle['paired_sites']} / {mask['paired_sites']} paired sites"
    )
    return 0


def run_oracle() -> int:
    """Section 42: GEN1's image answers at the error sites P0 never reaches. Analysis only.

    No new generation: SGV-GEN1 already gave Qwen3-VL-4B the true location of a content-blind
    sample of error sites, and those that P0 misses are its proposal-miss sample. None of these
    rows can enter an XR1 arm.
    """
    started = time.monotonic()
    _require(P0_COVERAGE, "normalize")
    _forbid(PROPOSAL_MISS_ORACLE)
    coverage = pd.read_parquet(P0_COVERAGE)
    outcomes = pd.read_parquet(gen1.SITE_OUTCOMES)
    outcomes = outcomes[outcomes["environment"].isin(set(coverage["environment"]))]
    covered = coverage.set_index("align_site_id")["covered_by_p0"]
    rows: dict[str, Any] = {}
    for condition in (gen1.E3, gen1.E1):
        cell = outcomes[(outcomes["method"] == gen1.M5) & (outcomes["condition"] == condition)]
        cell = cell.assign(reached=cell["align_site_id"].map(covered))
        if cell["reached"].isna().any():
            raise PhaseError("a GEN1 site is missing from the P0 coverage table")
        per_env: dict[str, Any] = {}
        for name, group in cell.groupby("environment", sort=True):
            env = coverage[coverage["environment"] == name]
            unreached_share = 1.0 - _rate(env["covered_by_p0"].to_numpy(dtype=bool))
            missed = group[~group["reached"].astype(bool)]
            reached = group[group["reached"].astype(bool)]
            exact_missed = _rate(missed["exact_at_1"].to_numpy(dtype=bool))
            per_env[str(name)] = {
                "gen1_sites": len(group),
                "gen1_sites_unreached": len(missed),
                "exact_at_1_unreached": exact_missed,
                "exact_at_1_reached": _rate(reached["exact_at_1"].to_numpy(dtype=bool)),
                "population_unreached_share": unreached_share,
                "localization_headroom": unreached_share * exact_missed,
            }
        rows[condition] = {
            "per_environment": per_env,
            "mean_localization_headroom": float(
                np.mean([v["localization_headroom"] for v in per_env.values()])
            ),
            "exact_at_1_unreached_pooled": _rate(
                cell[~cell["reached"].astype(bool)]["exact_at_1"].to_numpy(dtype=bool)
            ),
            "exact_at_1_reached_pooled": _rate(
                cell[cell["reached"].astype(bool)]["exact_at_1"].to_numpy(dtype=bool)
            ),
            "gen1_sites_unreached": int((~cell["reached"].astype(bool)).sum()),
            "gen1_sites": len(cell),
        }
    _write_json_once(
        PROPOSAL_MISS_ORACLE,
        {
            **_analysis_envelope("proposal_miss_oracle"),
            "oracle_localization": True,
            "source": _relative(gen1.SITE_OUTCOMES),
            "by_condition": rows,
            "population_proposal_recall_mean": float(
                np.mean(
                    [
                        _rate(g["covered_by_p0"].to_numpy(dtype=bool))
                        for _n, g in coverage.groupby("environment")
                    ]
                )
            ),
            "definition": (
                "localization headroom = the share of the population's error sites no P0 site "
                "reaches, times GEN1's exact top-1 rate at the unreached error sites of its sample"
            ),
            "enters_a_primary_arm": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "oracle: localization headroom "
        f"{rows[gen1.E3]['mean_localization_headroom']:.4f} (image), "
        f"{rows[gen1.E1]['mean_localization_headroom']:.4f} (text)"
    )
    return 0


# ------------------------------------------------------------------ sections 24-36: reliability


def oracle_prefix(score: np.ndarray, harmful: np.ndarray, epsilon: float) -> np.ndarray:
    """Analysis only: the deepest score prefix whose realised harm is at most epsilon.

    Cuts fall between distinct scores, so tied candidates are accepted or refused together.
    Uses the evaluation labels to place the cut and is never a deployable rule.
    """
    values = np.asarray(score, dtype=float)
    harm = np.asarray(harmful, dtype=bool)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-values, kind="stable")
    ranked = values[order]
    cumulative = np.cumsum(harm[order])
    ends = np.flatnonzero(np.r_[ranked[1:] != ranked[:-1], True])
    rates = cumulative[ends] / (ends + 1)
    valid = np.flatnonzero(rates <= epsilon)
    if valid.size == 0:
        return np.zeros(values.size, dtype=bool)
    return values >= ranked[ends[valid[-1]]]


def _g0_scored(candidates: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """G0 on the sampled documents with the frozen verifier's published score attached."""
    order = pd.read_parquet(cg1.EVALUATION_ORDER)[["candidate_id", "frozen_score"]]
    g0 = candidates[candidates["method"] == G0].merge(
        order, on="candidate_id", how="left", validate="one_to_one"
    )
    if g0["frozen_score"].isna().any():
        raise PhaseError("a sampled G0 candidate has no published frozen score")
    classes = outcomes[outcomes["method"] == G0][["site_id", "site_class", "linked_error_kinds"]]
    return g0.merge(classes, on="site_id", how="left", validate="many_to_one")


def ranking_block(frame: pd.DataFrame, share: float | None = None) -> dict[str, Any]:
    """Discrimination and the oracle frontier for one scored block of candidates."""
    score = frame["frozen_score"].to_numpy(dtype=float)
    harm = frame["is_harmful"].to_numpy(dtype=bool)
    benefit = frame["beneficial"].to_numpy(dtype=bool)
    quality = s15.ranking_quality(score, harm, benefit)
    achievable = s15.frontier(score, harm, benefit)
    out: dict[str, Any] = {
        "candidates": int(score.size),
        "harmful": int(harm.sum()),
        "beneficial": int(benefit.sum()),
        "harm_auroc": float(quality["auroc_safe"]),
        "benefit_auroc": float(roc_auc(score, benefit.astype(float)))
        if 0 < int(benefit.sum()) < score.size
        else float("nan"),
        "repair_recall_at_harm": {
            epsilon_key(e): float(achievable[epsilon_key(e)]["repair_recall"]) for e in EPSILONS
        },
    }
    if share is not None:
        take = int(np.clip(round(share * score.size), 0, score.size))
        prefix = np.argsort(-score, kind="stable")[:take]
        out["matched_coverage_share"] = float(share)
        out["prefix_harm_rate"] = _rate(harm[prefix])
        out["prefix_beneficial_rate"] = _rate(benefit[prefix])
        out["block_harm_rate"] = _rate(harm)
        out["block_beneficial_rate"] = _rate(benefit)
    return out


def run_reliability() -> int:
    """Sections 24-36. R0 always; R1-R3 only if the compatibility gate passed."""
    started = time.monotonic()
    for path in (CANDIDATES, FEATURE_COMPATIBILITY, VERIFIER_REPRODUCTION):
        _require(path, "normalize" if path == CANDIDATES else "compat")
    outputs = (
        ZERO_SHOT_SCORES,
        RELIABILITY_DISCRIMINATION,
        PREFIX_PURITY,
        RISK_FRONTIER,
        FROZEN_CUT_TRANSFER,
        DISTRIBUTION_SHIFT,
        CLEAN_SITE_RELIABILITY,
    )
    for path in outputs:
        _forbid(path)
    compatibility = cc_read_json(FEATURE_COMPATIBILITY)
    compatible = bool(compatibility["zero_shot_feature_compatible"])
    if compatible:
        raise PhaseError(
            "the compatibility gate passed; zero-shot scoring of new candidates through the "
            "frozen representation must be implemented before this phase can run"
        )
    candidates = pd.read_parquet(CANDIDATES)
    outcomes = pd.read_parquet(SITE_OUTCOMES)
    deployments = dt1.load_deployments()
    g0 = _g0_scored(candidates, outcomes)
    not_evaluable = {
        "evaluable": False,
        "reason": "zero_shot_feature_compatible = false (feature_compatibility.json)",
    }
    per_env: dict[str, Any] = {}
    cut: dict[str, Any] = {}
    for name, block in g0.groupby("environment", sort=True):
        deployment = deployments[str(name)]
        tau = float(deployment.source_tau[PRIMARY_EPSILON])
        # The frozen cut's acceptance share on the environment's full evaluation block: the
        # coverage the verifier was deployed at, used to match prefixes.
        share = float(np.mean(deployment.eval_scores >= tau)) if np.isfinite(tau) else 0.0
        per_env[str(name)] = ranking_block(block, share)
        score = block["frozen_score"].to_numpy(dtype=float)
        accepted = score >= tau if np.isfinite(tau) else np.zeros(score.size, dtype=bool)
        clean = (block["site_class"] == SITE_CLEAN).to_numpy()
        cut[str(name)] = {
            "tau": tau,
            "accepted": int(accepted.sum()),
            "coverage": _rate(accepted),
            "harm_rate_accepted": _rate(block["is_harmful"].to_numpy(dtype=bool)[accepted]),
            "beneficial_accepted": int(block["beneficial"].to_numpy(dtype=bool)[accepted].sum()),
            "clean_site_candidates": int(clean.sum()),
            "clean_site_accept_rate": _rate(accepted[clean]),
        }
    pooled = ranking_block(g0)
    harm_by_env = {k: v["harm_auroc"] for k, v in per_env.items()}
    by_type = {
        kind: ranking_block(
            g0[g0["linked_error_kinds"].str.split(",").map(lambda s, k=kind: k in s)]
        )
        for kind in ("substitution", "deletion", "insertion", "segmentation")
    }
    populations = {
        "R0": {"arm": G0, "scored": len(g0), "score_source": _relative(cg1.EVALUATION_ORDER)},
        "R1": {"arm": G1, **not_evaluable},
        "R2": {"arm": G2, **not_evaluable},
        "R3": {"arm": G3, **not_evaluable},
    }
    _write_json_once(
        ZERO_SHOT_SCORES,
        {
            **_analysis_envelope("zero_shot_scores"),
            "zero_shot_feature_compatible": compatible,
            "compatibility_summary": {
                family: {
                    "environments_changed": len(
                        {
                            r["environment"]
                            for r in compatibility["invariance"]
                            if r["family"] == family and r["max_abs_difference"] > 0
                        }
                    ),
                    "environments_tested": len(
                        {r["environment"] for r in compatibility["invariance"]}
                    ),
                    "max_abs_difference": max(
                        r["max_abs_difference"]
                        for r in compatibility["invariance"]
                        if r["family"] == family
                    ),
                    "max_share_changed": max(
                        r["share_changed"]
                        for r in compatibility["invariance"]
                        if r["family"] == family
                    ),
                }
                for family in compatibility["inert_by_family"]
            },
            "populations": populations,
            "note": (
                "G0's scores are the frozen verifier's published scores, reproduced exactly in "
                "frozen_verifier_reproduction.json. No new candidate was scored."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        RELIABILITY_DISCRIMINATION,
        {
            **_analysis_envelope("reliability_discrimination"),
            "R0": {"pooled": pooled, "per_environment": per_env, "by_error_type": by_type},
            "R0_harm_auroc_by_environment": harm_by_env,
            "R1": not_evaluable,
            "R2": not_evaluable,
            "R3": not_evaluable,
            "orientation": (
                "higher score = safer; harm AUROC = P(a not-harmful candidate outranks a "
                "harmful one)"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    rng = np.random.default_rng(XR1_SEED)
    random_prefix: dict[str, Any] = {}
    for name, block in g0.groupby("environment", sort=True):
        shuffled = block.assign(frozen_score=rng.permutation(block["frozen_score"].to_numpy()))
        random_prefix[str(name)] = ranking_block(
            shuffled, per_env[str(name)]["matched_coverage_share"]
        )
    _write_json_once(
        PREFIX_PURITY,
        {
            **_analysis_envelope("prefix_purity"),
            "R0": {
                name: {
                    k: row[k]
                    for k in (
                        "matched_coverage_share",
                        "prefix_harm_rate",
                        "prefix_beneficial_rate",
                        "block_harm_rate",
                        "block_beneficial_rate",
                    )
                }
                for name, row in per_env.items()
            },
            "R0_random_ranking": {
                name: {k: row[k] for k in ("prefix_harm_rate", "prefix_beneficial_rate")}
                for name, row in random_prefix.items()
            },
            "matched_coverage": (
                "the share of each environment's evaluation block the frozen cut accepts"
            ),
            "R1": not_evaluable,
            "R2": not_evaluable,
            "R3": not_evaluable,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        RISK_FRONTIER,
        {
            **_analysis_envelope("risk_frontier"),
            "R0": {
                "pooled": pooled["repair_recall_at_harm"],
                "per_environment": {n: r["repair_recall_at_harm"] for n, r in per_env.items()},
            },
            "primary": f"RepairRecall at harm <= {PRIMARY_EPSILON}, oracle cut, analysis only",
            "R1": not_evaluable,
            "R2": not_evaluable,
            "R3": not_evaluable,
            "deployable": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        FROZEN_CUT_TRANSFER,
        {
            **_analysis_envelope("frozen_cut_transfer"),
            "R0": cut,
            "cut": f"the published frozen source cut at epsilon {PRIMARY_EPSILON}, unchanged",
            "R1": not_evaluable,
            "R2": not_evaluable,
            "R3": not_evaluable,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(DISTRIBUTION_SHIFT, _distribution_shift(candidates, outcomes, g0, started))
    clean = g0[g0["site_class"] == SITE_CLEAN]
    error = g0[g0["site_class"] == SITE_ERROR]
    _write_json_once(
        CLEAN_SITE_RELIABILITY,
        {
            **_analysis_envelope("clean_site_reliability"),
            "R2": not_evaluable,
            "R0_reference": {
                "clean_site_candidates": len(clean),
                "error_site_candidates": len(error),
                "clean_score_quantiles": _score_quantiles(clean["frozen_score"].tolist()),
                "error_score_quantiles": _score_quantiles(error["frozen_score"].tolist()),
                "beneficial_vs_clean_auroc": _beneficial_vs_clean(g0),
                "frozen_cut_clean_accept_rate": {
                    n: r["clean_site_accept_rate"] for n, r in cut.items()
                },
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"reliability: R0 harm AUROC {pooled['harm_auroc']:.4f}; R1-R3 not evaluable")
    return 0


def _score_quantiles(values: Sequence[float]) -> dict[str, float]:
    """The spread of a set of frozen scores: the deciles the figure draws, and the mean."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {key: float("nan") for key in ("p10", "p25", "median", "p75", "p90", "mean")}
    p10, p25, median, p75, p90 = np.percentile(array, [10, 25, 50, 75, 90])
    return {
        "p10": float(p10),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "p90": float(p90),
        "mean": float(array.mean()),
    }


def _beneficial_vs_clean(frame: pd.DataFrame) -> float:
    """AUROC of beneficial error-site candidates over clean-site candidates (R4's statistic)."""
    positive = frame[(frame["site_class"] == SITE_ERROR) & frame["beneficial"]]
    negative = frame[frame["site_class"] == SITE_CLEAN]
    if positive.empty or negative.empty:
        return float("nan")
    scores = np.r_[
        positive["frozen_score"].to_numpy(float), negative["frozen_score"].to_numpy(float)
    ]
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return float(roc_auc(scores, labels))


def _edit_profile(frame: pd.DataFrame) -> dict[str, Any]:
    original = frame["original_ocr"].astype(str)
    candidate = frame["candidate_text"].astype(str)
    gap = frame["edit_kind"].astype(str) == xc1.EDIT_GAP
    delta = candidate.str.len() - original.str.len()
    return {
        "candidates": len(frame),
        "edit_type_share": {
            "insertion_at_gap": _rate((gap & (candidate.str.len() > 0)).to_numpy()),
            "deletion": _rate(((~gap) & (candidate.str.len() == 0)).to_numpy()),
            "replacement": _rate(((~gap) & (candidate.str.len() > 0)).to_numpy()),
        },
        "length_delta_quantiles": xc1._quantiles(delta.astype(float).tolist()),
        "candidate_length_quantiles": xc1._quantiles(candidate.str.len().astype(float).tolist()),
        "digit_change_share": _rate(
            (original.str.count(r"\d") != candidate.str.count(r"\d")).to_numpy()
        ),
        "whitespace_change_share": _rate(
            (original.str.count(r"\s") != candidate.str.count(r"\s")).to_numpy()
        ),
        "anchor_kind_share": {
            str(k): float(v) for k, v in frame["anchor_kind"].value_counts(normalize=True).items()
        },
        "beneficial_prevalence": _rate(frame["beneficial"].to_numpy(dtype=bool)),
        "harmful_prevalence": _rate(frame["is_harmful"].to_numpy(dtype=bool)),
        "exact_prevalence": _rate(frame["exact"].to_numpy(dtype=bool)),
        "clean_site_prevalence": _rate((frame["site_class"] == SITE_CLEAN).to_numpy()),
    }


def _distribution_shift(
    candidates: pd.DataFrame, outcomes: pd.DataFrame, g0: pd.DataFrame, started: float
) -> dict[str, Any]:
    """Section 31: how the new candidates differ from the ones the verifier was fitted on."""
    classes = outcomes[outcomes["method"] == G0][["site_id", "site_class"]]
    profiles: dict[str, Any] = {}
    for arm in (G0, G1, G2):
        block = candidates[candidates["method"] == arm].merge(classes, on="site_id", how="left")
        if arm != G0:
            block = block[block["generator_rank"] < PRIMARY_K]
        profiles[arm] = _edit_profile(block)
    profiles[G0]["score_quantiles"] = xc1._quantiles(g0["frozen_score"].tolist())
    return {
        **_analysis_envelope("candidate_distribution_shift"),
        "profiles": profiles,
        "g1_g2_scope": "top-1 candidates, the primary budget",
        "score_distribution_new_arms": "not computed: the compatibility gate failed",
        "runtime_seconds": time.monotonic() - started,
    }


# ------------------------------------------------------------------ section 43: the ceiling ladder


def run_ceiling() -> int:
    """Section 43: all errors -> reached by P0 -> exact candidate -> frozen-ranked prefix."""
    started = time.monotonic()
    _require(CLEAN_SITE_RELIABILITY, "reliability")
    _forbid(CEILING)
    candidates = pd.read_parquet(CANDIDATES)
    outcomes = pd.read_parquet(SITE_OUTCOMES)
    views = load_views()
    repaired = repaired_by_arm(views, candidates)
    g0 = _g0_scored(candidates, outcomes)
    compatible = bool(cc_read_json(FEATURE_COMPATIBILITY)["zero_shot_feature_compatible"])
    rows: dict[str, Any] = {}
    for name, view in views.items():
        reach = _rate(view.covered)
        block = g0[g0["environment"] == name]
        accepted = oracle_prefix(
            block["frozen_score"].to_numpy(float),
            block["is_harmful"].to_numpy(bool),
            PRIMARY_EPSILON,
        )
        chosen = set(block[accepted & block["exact"].to_numpy(bool)]["site_id"].astype(str))
        risk_controlled = _rate(view.repaired(chosen))
        row: dict[str, Any] = {"error_sites": view.count, "reached_by_p0": reach}
        for arm in (G0, G2, G3):
            opportunity = _rate(repaired[arm][name])
            row[arm] = {
                "opportunity": opportunity,
                "proposal_loss": 1.0 - reach,
                "generation_loss": reach - opportunity,
            }
        row[G0]["risk_controlled_repair"] = risk_controlled
        row[G0]["reliability_ranking_loss"] = row[G0]["opportunity"] - risk_controlled
        for arm in (G2, G3):
            row[arm]["risk_controlled_repair"] = None
            row[arm]["reliability_ranking_loss"] = None
        rows[name] = row

    def mean(path: Callable[[dict[str, Any]], Any]) -> float | None:
        values = [path(r) for r in rows.values()]
        return None if any(v is None for v in values) else float(np.mean(values))

    summary = {
        "reached_by_p0": mean(lambda r: r["reached_by_p0"]),
        **{
            arm: {
                key: mean(lambda r, a=arm, k=key: r[a][k])
                for key in (
                    "opportunity",
                    "proposal_loss",
                    "generation_loss",
                    "risk_controlled_repair",
                    "reliability_ranking_loss",
                )
            }
            for arm in (G0, G2, G3)
        },
    }
    _write_json_once(
        CEILING,
        {
            **_analysis_envelope("ceiling_decomposition"),
            "ladder": [
                "all OCR error sites on the sampled documents",
                "reached by the frozen GT-blind proposer",
                "an exact candidate exists within the arm's budget",
                "the frozen verifier's ranking keeps it inside the risk-controlled prefix",
            ],
            "per_environment": rows,
            "mean": summary,
            "risk_controlled_prefix": (
                f"the deepest frozen-score prefix of the arm's candidates with realised harm <= "
                f"{PRIMARY_EPSILON}, placed with evaluation labels -- analysis only"
            ),
            "reliability_loss_for_new_arms": (
                "not measurable zero-shot" if not compatible else "measured"
            ),
            "losses_are_not_merged": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"ceiling: reached {summary['reached_by_p0']:.4f}; G0 {summary[G0]['opportunity']:.4f} -> "
        f"{summary[G0]['risk_controlled_repair']:.4f}; G2 {summary[G2]['opportunity']:.4f}; "
        f"G3 {summary[G3]['opportunity']:.4f}"
    )
    return 0


# ------------------------------------------------------------------ section 60: statistics


def _paired_frame(
    views: dict[str, EnvironmentView],
    repaired: dict[str, dict[str, np.ndarray]],
    a: str,
    b: str,
    kind: str | None = None,
) -> pd.DataFrame:
    frames = []
    for name, view in views.items():
        mask = np.ones(view.count, dtype=bool) if kind is None else view.error_kind == kind
        frames.append(
            pd.DataFrame(
                {
                    "environment": name,
                    "document_id": view.error_document[mask],
                    "a": repaired[a][name][mask].astype(float),
                    "b": repaired[b][name][mask].astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _site_frame(
    outcomes: pd.DataFrame, a: str, b: str, column: str, site_class: str
) -> pd.DataFrame:
    """Proposal-site units that both arms were asked about, for a site-level comparison."""
    left = outcomes[
        (outcomes["method"] == a) & outcomes["requested"] & (outcomes["site_class"] == site_class)
    ]
    right = outcomes[
        (outcomes["method"] == b) & outcomes["requested"] & (outcomes["site_class"] == site_class)
    ]
    joined = left[["environment", "document_id", "site_id", column]].merge(
        right[["site_id", column]], on="site_id", suffixes=("_a", "_b")
    )
    return pd.DataFrame(
        {
            "environment": joined["environment"],
            "document_id": joined["document_id"],
            "a": joined[f"{column}_a"].astype(float),
            "b": joined[f"{column}_b"].astype(float),
        }
    )


def comparison(frame: pd.DataFrame, name: str, family: str, pooled: bool = False) -> dict[str, Any]:
    """One document-clustered paired bootstrap comparison: effect, interval and p-value."""
    from ocr_risk.stats.bootstrap import _bootstrap_p_value

    effects, draws = gen1._cluster_draws({name: frame}, pooled=pooled)
    values = draws[name]
    finite = values[np.isfinite(values)]
    per_env = (frame["a"] - frame["b"]).groupby(frame["environment"]).mean()
    return {
        "comparison": name,
        "family": family,
        "effect": float(effects[name]),
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        "p_value": float(_bootstrap_p_value(finite)) if finite.size else float("nan"),
        "environments_improved": int((per_env > 0).sum()),
        "environments_worsened": int((per_env < 0).sum()),
        "units": len(frame),
        "pooled": pooled,
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def run_stats() -> int:
    """Section 60: the generation family, Holm-corrected, and its secondaries."""
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    candidates = pd.read_parquet(CANDIDATES)
    outcomes = pd.read_parquet(SITE_OUTCOMES)
    views = load_views()
    repaired = repaired_by_arm(views, candidates)
    primary = {
        "P1_visual_translation": comparison(
            _paired_frame(views, repaired, G2, G1), "P1_visual_translation", "generation"
        ),
        "P2_gain_vs_current": comparison(
            _paired_frame(views, repaired, G2, G0), "P2_gain_vs_current", "generation"
        ),
        "P3_union_gain": comparison(
            _paired_frame(views, repaired, G3, G0), "P3_union_gain", "generation"
        ),
    }
    adjusted = s15.holm(primary)
    secondary_rows: dict[str, dict[str, Any]] = {
        "visual_translation_at_5": comparison(
            _paired_frame(views, repaired, f"{G2}@5", f"{G1}@5"),
            "visual_translation_at_5",
            "secondary_generation",
        ),
        "conditional_exact_g2_vs_g1": comparison(
            _site_frame(outcomes, G2, G1, "top1_exact", SITE_ERROR),
            "conditional_exact_g2_vs_g1",
            "secondary_generation",
        ),
        "clean_rewrite_g2_vs_g1": comparison(
            _site_frame(outcomes, G2, G1, "rewrote", SITE_CLEAN),
            "clean_rewrite_g2_vs_g1",
            "secondary_generation",
        ),
    }
    for kind in WEAK_TYPES:
        for arm in (G2, G3):
            key = f"{kind}_{arm}_vs_g0"
            secondary_rows[key] = comparison(
                _paired_frame(views, repaired, arm, G0, kind),
                key,
                "secondary_error_type",
                pooled=True,
            )
    secondary = s15.holm(secondary_rows)
    controls = {}
    for control in CONTROL_ARMS:
        paired = _paired_control_sites(outcomes, control)
        frame = pd.DataFrame(
            {
                "environment": paired["environment"],
                "document_id": paired["document_id"],
                "a": paired[f"top1_exact__{control}"].fillna(False).astype(float),
                "b": paired[f"top1_exact__{G2}"].fillna(False).astype(float),
            }
        )
        controls[f"{control}_minus_g2"] = comparison(frame, f"{control}_minus_g2", "controls")
    controls = s15.holm(controls)
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
            "generation_family": adjusted,
            "secondary": secondary,
            "controls": controls,
            "reliability_family": {
                "evaluable": False,
                "reason": "zero_shot_feature_compatible = false",
            },
            "multiplicity": "Holm within each family; the generation family is P1-P3",
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "stats: "
        + ", ".join(
            f"{k} {v['effect']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] "
            f"p_adj {v['holm_adjusted_p']:.4f}"
            for k, v in adjusted.items()
        )
    )
    return 0


# ------------------------------------------------------------------ section 59: falsification

GT_NAMES = frozenset(
    {
        "region_gt",
        "gt_text",
        "gt_tokens",
        "d_before",
        "d_after",
        "outcome",
        "is_harmful",
        "beneficial",
        "labelable",
        "site_kind",
        "align_site_id",
        "site_class",
        "truth",
        "_labels",
        "_alignment_population",
    }
)
GENERATION_BUILDERS = (
    "order_key",
    "choose_documents",
    "sample_plan",
    "control_subsample",
    "run_proposals",
    "line_box",
    "site_crop",
    "mask_rectangle",
    "apply_mask",
    "donor_map",
    "build_natural_contexts",
    "run_contexts",
    "requests_for",
    "run_generate",
)
RELIABILITY_ARTIFACT_NAMES = frozenset(
    {
        "ZERO_SHOT_SCORES",
        "RELIABILITY_DISCRIMINATION",
        "PREFIX_PURITY",
        "RISK_FRONTIER",
        "FROZEN_CUT_TRANSFER",
        "CLEAN_SITE_RELIABILITY",
        "CEILING",
    }
)


@functools.lru_cache(maxsize=1)
def _module_tree() -> ast.Module:
    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def _names_in(function_name: str) -> set[str]:
    """Every name, attribute and string key a function of this module mentions, from its source."""
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            names: set[str] = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name):
                    names.add(inner.id)
                elif isinstance(inner, ast.Attribute):
                    names.add(inner.attr)
                elif isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    names.add(inner.value)
            return names
    raise PhaseError(f"{function_name} is not a function of this module")


def run_negative() -> int:
    """Section 59: the fifteen falsification tests, each with the evidence it rests on."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    tests: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: Any) -> None:
        tests.append(
            {"test": name, "passed": bool(passed), "applicable": True, "evidence": evidence}
        )

    _errors, arms = xc1.load_arms()
    m0 = [
        arms[spec["environment"], xc1.M0, xc1.COND_FROZEN].opportunity_recall()
        for spec in environment_specs()
    ]
    baseline = float(cc_read_json(xc1.DECISION)["baseline_opportunity"])
    add(
        "f1_m0_reproduces_the_frozen_opportunity_baseline",
        abs(float(np.mean(m0)) - baseline) <= 1e-12,
        {"recomputed": float(np.mean(m0)), "published": baseline},
    )
    reproduction = cc_read_json(VERIFIER_REPRODUCTION)
    add(
        "f2_m0_reliability_reproduces_the_frozen_verifier",
        bool(reproduction["all_identical"]),
        {
            "environments": len(reproduction["environments"]),
            "max_score_difference": max(
                r["score_max_abs_difference"] for r in reproduction["environments"]
            ),
        },
    )
    shuffle = cc_read_json(IMAGE_SHUFFLE)
    rates = shuffle["exact_top1_rate"]
    add(
        "f3_correct_image_outperforms_the_shuffled_image",
        rates[G2] > rates[X_IMG],
        {"g2": rates[G2], "shuffled": rates[X_IMG], "paired_sites": shuffle["paired_sites"]},
    )
    controls = cc_read_json(CONTROL_RESULTS)
    add(
        "f4_duplicating_g2_candidates_does_not_raise_opportunity",
        bool(controls["duplication"]["opportunity_unchanged"]),
        controls["duplication"],
    )
    candidates = pd.read_parquet(CANDIDATES)
    views = load_views()
    # The union is rebuilt from G0 and the reduced G2, as `--normalize` built it; removing G2's
    # exact rows must then return every arm to what it repairs without them.
    g2_left = candidates[(candidates["method"] == G2) & ~candidates["exact"]]
    removed = pd.concat(
        [
            candidates[~candidates["method"].isin([G2, G3])],
            g2_left,
            union_candidates(candidates[candidates["method"] == G0], g2_left),
        ],
        ignore_index=True,
    )
    base = repaired_by_arm(views, candidates)
    stripped = repaired_by_arm(views, removed)
    union_back_to_g0 = all(np.array_equal(stripped[G3][n], base[G0][n]) for n in views)
    add(
        "f5_removing_exact_g2_candidates_reduces_opportunity_exactly",
        bool(controls["removal"]["exact"]) and union_back_to_g0,
        {**controls["removal"], "union_returns_to_g0": union_back_to_g0},
    )
    leaks = {name: sorted(_names_in(name) & GT_NAMES) for name in GENERATION_BUILDERS}
    add(
        "f6_clean_site_labels_are_not_used_during_generation",
        not any(leaks.values()),
        {"builders_checked": len(leaks), "ground_truth_names_found": leaks},
    )
    sample = pd.read_parquet(PROPOSAL_SAMPLE)
    names = {spec["environment"] for spec in environment_specs()}
    scoped = set(sample[sample["environment"].isin(names)]["site_id"].astype(str))
    p0_ids = set()
    for name in names:
        p0_ids |= set(_p0_requests(name)["site_id"].astype(str))
    add(
        "f7_ground_truth_locations_never_enter_the_primary_proposal",
        scoped <= p0_ids and not [c for c in FORBIDDEN_INPUT_COLUMNS if c in sample.columns],
        {
            "sampled_sites": len(scoped),
            "outside_p0": len(scoped - p0_ids),
            "truth_columns": [c for c in FORBIDDEN_INPUT_COLUMNS if c in sample.columns],
        },
    )
    checked = differing = 0
    for arm in MODEL_ARMS:
        for spec in environment_specs():
            name = spec["environment"]
            requests = requests_for(arm, name).set_index("request_id")
            raw = pd.read_parquet(_shard_path(arm, name))
            for row in raw.itertuples(index=False):
                request = requests.loc[str(row.request_id)]
                rebuilt = canonical_hash(
                    {
                        "system": gen1.SYSTEM_PROMPT,
                        "user": str(request["user_prompt"]),
                        "image": request["image_sha256"],
                    }
                )
                checked += 1
                differing += int(rebuilt != str(row.prompt_sha256))
    add(
        "f8_ground_truth_never_enters_a_prompt",
        differing == 0 and not any(leaks.values()),
        {"prompts_rebuilt_from_gt_blind_tables": checked, "prompts_differing": differing},
    )
    arm_sites = set(candidates[candidates["method"].isin(MODEL_ARMS)]["site_id"].astype(str))
    gen1_sites = set(pd.read_parquet(gen1.SITE_OUTCOMES)["site_id"].astype(str))
    add(
        "f9_proposal_miss_oracle_rows_cannot_enter_a_primary_arm",
        arm_sites <= set(sample["site_id"].astype(str)) and not (arm_sites & gen1_sites),
        {"arm_sites": len(arm_sites), "shared_with_gen1_oracle_sites": len(arm_sites & gen1_sites)},
    )
    g3 = candidates[candidates["method"] == G3]
    expected = pd.concat(
        [
            candidates[candidates["method"] == G0],
            candidates[(candidates["method"] == G2) & (candidates["generator_rank"] < PRIMARY_K)],
        ]
    ).drop_duplicates(["site_id", "candidate_text"])
    attribution = bool(
        len(g3) == len(expected)
        and g3["candidate_sources"].astype(str).str.len().gt(0).all()
        and set(",".join(g3["candidate_sources"]).split(",")) <= {G0, G2}
    )
    add(
        "f10_union_attribution_survives_deduplication",
        attribution,
        {
            "union_rows": len(g3),
            "expected_rows": len(expected),
            "sources": sorted(set(",".join(g3["candidate_sources"]).split(","))),
        },
    )
    compat = cc_read_json(FEATURE_COMPATIBILITY)
    normalization = cc_read_json(CANDIDATE_NORMALIZATION)
    add(
        "f11_reliability_features_are_computed_before_outcome_labels",
        compat["issued_utc"] <= normalization["issued_utc"]
        and not compat["reads_an_xr1_candidate"],
        {"compatibility": compat["issued_utc"], "labels": normalization["issued_utc"]},
    )
    purity = cc_read_json(PREFIX_PURITY)
    frozen_gain = [
        purity["R0"][n]["block_harm_rate"] - purity["R0"][n]["prefix_harm_rate"]
        for n in purity["R0"]
    ]
    random_gain = [
        purity["R0"][n]["block_harm_rate"] - purity["R0_random_ranking"][n]["prefix_harm_rate"]
        for n in purity["R0"]
    ]
    add(
        "f12_random_ranking_destroys_the_prefix_purity_advantage",
        float(np.mean(frozen_gain)) > float(np.mean(random_gain))
        and abs(float(np.mean(random_gain))) < abs(float(np.mean(frozen_gain))),
        {
            "frozen_advantage_mean": float(np.mean(frozen_gain)),
            "random_advantage_mean": float(np.mean(random_gain)),
        },
    )
    outcomes = pd.read_parquet(SITE_OUTCOMES)
    g0 = _g0_scored(candidates, outcomes)
    rng = np.random.default_rng(XR1_SEED)
    permuted = g0.assign(
        is_harmful=g0.groupby("environment")["is_harmful"].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
    )
    permuted_auroc = ranking_block(permuted)["harm_auroc"]
    add(
        "f13_label_permutation_destroys_reliability_discrimination",
        abs(permuted_auroc - 0.5) < 0.05,
        {
            "permuted_harm_auroc": permuted_auroc,
            "observed_harm_auroc": ranking_block(g0)["harm_auroc"],
        },
    )
    scores = g0["frozen_score"].to_numpy(dtype=float).copy()
    oracle_prefix(scores, g0["is_harmful"].to_numpy(bool), PRIMARY_EPSILON)
    add(
        "f14_the_oracle_cut_cannot_influence_the_frozen_score",
        bool(np.array_equal(scores, g0["frozen_score"].to_numpy(dtype=float)))
        and "frozen_score" not in _names_in("oracle_prefix"),
        {"scores_unchanged_after_the_cut": True, "reads": sorted(_names_in("oracle_prefix"))},
    )
    shards = cc_read_json(RAW_OUTPUT_MANIFEST)["shards"]
    last_shard = max(row["issued_utc"] for row in shards)
    cut = cc_read_json(FROZEN_CUT_TRANSFER)
    touches = {
        name: sorted(_names_in(name) & RELIABILITY_ARTIFACT_NAMES) for name in GENERATION_BUILDERS
    }
    add(
        "f15_frozen_cut_evaluation_does_not_affect_generation",
        last_shard <= cut["issued_utc"] and not any(touches.values()),
        {
            "last_shard": last_shard,
            "frozen_cut_issued": cut["issued_utc"],
            "generation_reads": touches,
        },
    )
    passed = sum(t["passed"] for t in tests)
    _write_json_once(
        FALSIFICATION,
        {
            **_analysis_envelope("falsification_tests"),
            "tests": tests,
            "passed": passed,
            "total": len(tests),
            "note": (
                "f3 is a substantive expectation, not a pipeline invariant: it fails if the image "
                "is not load-bearing under natural proposals, and a failure is reported as a result"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"negative: {passed} of {len(tests)} pass")
    for test in tests:
        print(f"  {'PASS' if test['passed'] else 'FAIL'} {test['test']}")
    return 0


# ------------------------------------------------------------------ sections 44-48, 65-67: decide


def generation_criteria(
    opportunity: dict[str, Any],
    types: dict[str, Any],
    generation: dict[str, Any],
    stats: dict[str, Any],
    shuffle: dict[str, Any],
) -> tuple[str, dict[str, bool], dict[str, dict[str, bool]]]:
    """C1-C5 for G2 and G3, and the arm the outcome is judged on."""
    by_arm: dict[str, dict[str, bool]] = {}
    base_harm = generation[G0]["harmful_candidates"]
    base_repaired = opportunity["opportunity"][G0]["repaired_error_sites"]
    base_ratio = _ratio(base_harm, base_repaired)
    for arm in (G2, G3):
        gain = opportunity["opportunity"][arm]["mean"] - opportunity["opportunity"][G0]["mean"]
        harm = generation[arm]["harmful_candidates"]
        repaired = opportunity["opportunity"][arm]["repaired_error_sites"]
        if arm == G3:
            extra = repaired - base_repaired
            ratio = _ratio(harm - base_harm, extra) if extra > 0 else float("inf")
        else:
            ratio = _ratio(harm, repaired) if repaired > 0 else float("inf")
        by_arm[arm] = {
            "C1": gain >= OPPORTUNITY_MARGIN,
            "C2": opportunity["environments_reaching_margin"][arm] >= BREADTH_MAJORITY,
            "C3": len(types["weak_types_reaching_margin"][arm]) >= WEAK_TYPES_REQUIRED,
            "C4": ratio <= base_ratio,
        }
    passing = [a for a in (G2, G3) if all(by_arm[a].values())]
    gaining = [a for a in (G2, G3) if by_arm[a]["C1"] and by_arm[a]["C2"]]
    arm = passing[0] if passing else gaining[0] if gaining else G3
    visual = stats["generation_family"]["P1_visual_translation"]
    retention = shuffle["retention_of_visual_gain"]
    c5 = bool(
        visual["ci_low"] > 0
        and shuffle["g2_minus_g1"] > 0
        and retention is not None
        and retention <= SHUFFLE_RETENTION_MAX
    )
    return arm, {**by_arm[arm], "C5": c5}, by_arm


def run_decide() -> int:
    """Sections 65-67: the machine-readable decision, from the frozen rule alone."""
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)
    generation = cc_read_json(GENERATION_RESULTS)["arms"]
    recall = cc_read_json(GENERATION_RESULTS)["site_proposal_recall"]
    types = cc_read_json(ERROR_TYPE_ANALYSIS)
    stats = cc_read_json(STATISTICAL_TESTS)
    shuffle = cc_read_json(IMAGE_SHUFFLE)
    clean = cc_read_json(CLEAN_SITE_RESULTS)["arms"]
    compat = cc_read_json(FEATURE_COMPATIBILITY)
    ceiling = cc_read_json(CEILING)["mean"]
    union = cc_read_json(UNION_RESULTS)
    negative = cc_read_json(FALSIFICATION)
    arm, criteria, by_arm = generation_criteria(opportunity, types, generation, stats, shuffle)
    compatible = bool(compat["zero_shot_feature_compatible"])
    reliability_pass = False
    outcome = assign_outcome(criteria, reliability_pass)
    generation_pass = all(criteria.values())
    table = opportunity["opportunity"]
    reason = (
        f"outcome {outcome}: generation criteria on {arm} -- "
        + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in criteria.items())
        + f"; zero_shot_feature_compatible {compatible}, so the reliability gate "
        + ("could be evaluated" if compatible else "is not evaluable and does not pass")
        + f". {OUTCOME_LABELS[outcome]}."
    )
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "primary_generator": G2,
            "judged_arm": arm,
            "primary_epsilon": PRIMARY_EPSILON,
            "baseline_opportunity": table[G0]["mean"],
            "text_generator_opportunity": table[G1]["mean"],
            "image_generator_opportunity": table[G2]["mean"],
            "union_opportunity": table[G3]["mean"],
            "visual_translation_gain": opportunity["translation"]["visual_translation_gain"],
            "gain_vs_current": opportunity["translation"]["gain_vs_current"],
            "union_gain": opportunity["translation"]["union_gain"],
            "environments_generation_improved": opportunity["environments_reaching_margin"][arm],
            "weak_error_types_improved": types["weak_types_reaching_margin"][arm],
            "site_proposal_recall_sample": recall["sample_mean"],
            "site_proposal_recall_population": recall["population_mean"],
            "clean_site_rewrite_rate": clean[G2]["rewrite_rate"],
            "harmful_candidate_burden": {
                "harmful_candidates": generation[G2]["harmful_candidates"],
                "harmful_share": generation[G2]["harmful_share"],
                "harmful_per_requested_site": generation[G2]["harmful_per_requested_site"],
                "harmful_per_repaired_error_site": generation[G2][
                    "harmful_per_repaired_error_site"
                ],
                "g0_harmful_per_repaired_error_site": generation[G0][
                    "harmful_per_repaired_error_site"
                ],
            },
            "image_mechanism_load_bearing": criteria["C5"],
            "zero_shot_feature_compatible": compatible,
            "frozen_verifier_harm_auc": None,
            "frozen_verifier_benefit_auc": None,
            "zero_shot_risk_frontier": None,
            "frozen_cut_transfer": None,
            "environments_reliability_transfers": None,
            "reliability_not_evaluable_reason": None
            if compatible
            else "the frozen verifier encodes generator identity and operation labels that a new "
            "generator does not have (feature_compatibility.json)",
            "proposal_loss": ceiling[G2]["proposal_loss"],
            "generation_loss": ceiling[G2]["generation_loss"],
            "reliability_loss": ceiling[G2]["reliability_ranking_loss"],
            "g0_ceiling": ceiling[G0],
            "criteria": criteria,
            "criteria_by_arm": by_arm,
            "generation_gate_pass": generation_pass,
            "reliability_gate_pass": reliability_pass,
            "union_complementarity": {
                "union_minus_best_single": union["union_minus_best_single"],
                "note": "reported independently; never overrides the outcome",
            },
            "falsification_tests_passed": negative["passed"],
            "falsification_tests_total": negative["total"],
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS[outcome],
            "ready_for_reliability_adaptation": outcome in ("A", "B"),
            "ready_for_external_confirmation": False,
            "confirmatory_reserve_consumed": False,
            "recommended_next_stage": f"NEXT: {NEXT_STAGE[outcome]}",
            "reason": reason,
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: outcome {outcome} -- NEXT: {NEXT_STAGE[outcome]}")
    return 0


# ------------------------------------------------------------------ section 53: figures

CONTEXT_CENSUS = CACHE / "context_census.json"
FIGURE_NOTE = (
    "ANALYSIS ONLY -- development evidence computed with ground truth; not a deployable method"
)
ARM_COLOURS = {
    G0: "#8a8984",
    G1: "#9ec5f4",
    G2: "#184f95",
    G3: "#3987e5",
    X_IMG: "#d9822b",
    X_MASK: "#b3413b",
}
ARM_LABELS = {
    G0: "G0 frozen generator",
    G1: "G1 Qwen3-VL-4B, no image",
    G2: "G2 Qwen3-VL-4B + crop",
    G3: "G3 union G0 + G2",
    X_IMG: "shuffled crop",
    X_MASK: "masked crop",
}
TABLE_KEYS = {
    "xr1_candidates.parquet": ["method", "candidate_id"],
    "proposal_site_outcomes.parquet": ["method", "site_id"],
    "proposal_links.parquet": ["site_id", "align_site_id", "link_kind"],
    "error_site_table.parquet": ["align_site_id"],
    "p0_coverage.parquet": ["align_site_id"],
}


def _signature(path: Path) -> str:
    """A content hash that ignores clock fields, so two regenerations compare exactly."""
    if path.suffix == ".json":
        return canonical_hash(gen1._stable_view(cc_read_json(path)))
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        keys = TABLE_KEYS.get(path.name, list(frame.columns))
        frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        return canonical_hash(frame.to_json(orient="records", default_handler=str))
    return file_sha256(path)


def _short(name: str) -> str:
    return name.replace("funsd/", "F ").replace("sbb/", "S ").replace("tesseract", "tess")


def run_figures() -> int:
    """The seventeen required figures, every value read from a persisted artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
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
            "grid.color": xc1.GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "xr1",
        }
    )
    manifest: dict[str, Any] = {}

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
            "sources": {source_key(s): _signature(s) for s in sources},
        }

    def source_key(path: Path) -> str:
        # Stage files relative to the stage directory, upstream files to the repository, so
        # `--determinism`'s regenerations into a sandbox directory compare equal.
        stage = FIGURE_DIR.parent
        return str(path.relative_to(stage)) if path.is_relative_to(stage) else _relative(path)

    opportunity = cc_read_json(OPPORTUNITY_RESULTS)
    table = opportunity["opportunity"]
    environments = [spec["environment"] for spec in environment_specs()]
    arms = (G0, G1, G2, G3)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, arm in enumerate(arms):
        ax.bar(i, table[arm]["mean"], color=ARM_COLOURS[arm], width=0.6)
        values = list(table[arm]["per_environment"].values())
        ax.scatter(
            np.full(len(values), i) + np.linspace(-0.18, 0.18, len(values)),
            values,
            s=9,
            color=xc1.INK,
            zorder=3,
        )
    ax.set_xticks(range(len(arms)), [ARM_LABELS[a] for a in arms], rotation=12)
    ax.set_ylabel("natural opportunity recall")
    ax.set_title("Natural opportunity by generator (mean over environments; dots = environments)")
    save(
        fig,
        "natural_opportunity_by_generator.png",
        [OPPORTUNITY_RESULTS],
        "opportunity recall per arm",
    )

    generation = cc_read_json(GENERATION_RESULTS)["arms"]
    known = cc_read_json(gen1.OUT / "exact_generation_results.json")["cells"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    groups = [
        (
            "known site, exact top-1 (GEN1)",
            known[GEN1_MATCHED_CELL]["exact_generation"],
            known[GEN1_BEST_CELL]["exact_generation"],
        ),
        (
            "proposed error site, exact top-1 (XR1)",
            generation[G1]["conditional_exact_generation"]["mean"],
            generation[G2]["conditional_exact_generation"]["mean"],
        ),
        ("natural opportunity (XR1)", table[G1]["mean"], table[G2]["mean"]),
    ]
    for i, (_label, text, image) in enumerate(groups):
        ax.bar(
            i - 0.18, text, width=0.34, color=ARM_COLOURS[G1], label="no image" if i == 0 else None
        )
        ax.bar(
            i + 0.18,
            image,
            width=0.34,
            color=ARM_COLOURS[G2],
            label="with crop" if i == 0 else None,
        )
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], rotation=8)
    ax.set_ylabel("rate")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("From known-site generation to natural opportunity, Qwen3-VL-4B")
    save(
        fig,
        "known_site_vs_natural_translation.png",
        [gen1.OUT / "exact_generation_results.json", GENERATION_RESULTS, OPPORTUNITY_RESULTS],
        "GEN1's oracle-located rate against XR1's rates at GT-blind proposals",
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, name in enumerate(environments):
        a, b = table[G1]["per_environment"][name], table[G2]["per_environment"][name]
        ax.plot([i, i], [a, b], color=xc1.GRID, zorder=1)
        ax.scatter(i, a, color=ARM_COLOURS[G1], zorder=2, label=ARM_LABELS[G1] if i == 0 else None)
        ax.scatter(i, b, color=ARM_COLOURS[G2], zorder=3, label=ARM_LABELS[G2] if i == 0 else None)
    ax.set_xticks(
        range(len(environments)), [_short(n) for n in environments], rotation=40, ha="right"
    )
    ax.set_ylabel("natural opportunity recall")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Image against text at the same GT-blind proposals")
    save(
        fig,
        "image_vs_text_natural_generation.png",
        [OPPORTUNITY_RESULTS],
        "G2 and G1 per environment",
    )

    types = cc_read_json(ERROR_TYPE_ANALYSIS)
    kinds = ("substitution", "deletion", "insertion", "segmentation")
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for j, arm in enumerate(arms):
        values = [types["by_arm"][arm][k]["opportunity"] for k in kinds]
        ax.bar(
            np.arange(len(kinds)) + (j - 1.5) * 0.2,
            values,
            width=0.2,
            color=ARM_COLOURS[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(
        range(len(kinds)), ["substitution", "omission", "spurious insertion", "segmentation"]
    )
    ax.set_ylabel("opportunity recall, pooled")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Natural opportunity by error type")
    save(
        fig,
        "opportunity_by_error_type.png",
        [ERROR_TYPE_ANALYSIS],
        "pooled opportunity per error type",
    )

    clean = cc_read_json(CLEAN_SITE_RESULTS)["arms"]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    measures = ("rewrite_rate", "overcorrection_rate", "abstention_rate")
    for j, arm in enumerate((G0, G1, G2)):
        ax.bar(
            np.arange(len(measures)) + (j - 1) * 0.26,
            [clean[arm][m] for m in measures],
            width=0.26,
            color=ARM_COLOURS[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(range(len(measures)), ["rewrites a clean site", "overcorrection", "abstains"])
    ax.set_ylabel("share of requested clean proposals")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Behaviour at clean proposed sites")
    save(
        fig,
        "clean_site_rewrite_rate.png",
        [CLEAN_SITE_RESULTS],
        "clean-site rewrite, overcorrection and abstention",
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for axis, key, label in (
        (axes[0], "harmful_per_requested_site", "harmful candidates per requested site"),
        (axes[1], "harmful_share", "harmful share of candidates"),
    ):
        axis.bar(
            range(len(arms)),
            [generation[a][key] for a in arms],
            color=[ARM_COLOURS[a] for a in arms],
        )
        axis.set_xticks(range(len(arms)), ["G0", "G1", "G2", "G3"])
        axis.set_title(label, fontsize=8)
    fig.suptitle("Harmful candidate burden", fontsize=9)
    save(
        fig,
        "harmful_candidate_burden.png",
        [GENERATION_RESULTS],
        "harmful candidates per site and share",
    )

    overlap = cc_read_json(CANDIDATE_OVERLAP)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bottom = np.zeros(len(kinds))
    for key, colour, label in (
        ("shared", xc1.INK_SECONDARY, "both"),
        ("g0_only", ARM_COLOURS[G0], "G0 only"),
        ("g2_only", ARM_COLOURS[G2], "G2 only"),
    ):
        values = np.array([overlap["by_error_type"][k][key] for k in kinds], dtype=float)
        ax.bar(range(len(kinds)), values, bottom=bottom, color=colour, label=label)
        bottom += values
    ax.set_xticks(
        range(len(kinds)), ["substitution", "omission", "spurious insertion", "segmentation"]
    )
    ax.set_ylabel("repaired error sites")
    ax.legend(frameon=False)
    ax.set_title("Which generator repairs which error sites")
    save(fig, "g0_g2_unique_repairs.png", [CANDIDATE_OVERLAP], "shared and unique repairs")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for arm in (G0, G2, G3):
        ax.plot(
            range(len(environments)),
            [table[arm]["per_environment"][n] for n in environments],
            marker="o",
            color=ARM_COLOURS[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(
        range(len(environments)), [_short(n) for n in environments], rotation=40, ha="right"
    )
    ax.set_ylabel("natural opportunity recall")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("The union candidate universe per environment")
    save(
        fig, "union_opportunity.png", [OPPORTUNITY_RESULTS, UNION_RESULTS], "G0, G2 and their union"
    )

    oracle = cc_read_json(PROPOSAL_MISS_ORACLE)["by_condition"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    image = oracle[gen1.E3]["per_environment"]
    ax.bar(
        np.arange(len(environments)) - 0.2,
        [image[n]["population_unreached_share"] for n in environments],
        width=0.4,
        color=xc1.GRID,
        label="error sites no P0 site reaches",
    )
    ax.bar(
        np.arange(len(environments)) + 0.2,
        [image[n]["localization_headroom"] for n in environments],
        width=0.4,
        color=ARM_COLOURS[G2],
        label="of which GEN1's image arm repairs (estimate)",
    )
    ax.set_xticks(
        range(len(environments)), [_short(n) for n in environments], rotation=40, ha="right"
    )
    ax.set_ylabel("share of error sites")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Proposal-miss headroom (oracle diagnostic)")
    save(
        fig,
        "proposal_miss_headroom.png",
        [PROPOSAL_MISS_ORACLE],
        "unreached share and localization headroom",
    )

    reliability = cc_read_json(RELIABILITY_DISCRIMINATION)
    for name, key, title in (
        ("frozen_verifier_harm_auc.png", "harm_auroc", "Frozen verifier harm AUROC"),
        ("frozen_verifier_benefit_auc.png", "benefit_auroc", "Frozen verifier benefit AUROC"),
    ):
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        values = [reliability["R0"]["per_environment"][n][key] for n in environments]
        ax.bar(range(len(environments)), values, color=ARM_COLOURS[G0])
        ax.axhline(0.5, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
        ax.set_xticks(
            range(len(environments)), [_short(n) for n in environments], rotation=40, ha="right"
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("AUROC on G0 candidates")
        ax.set_title(f"{title}: G0 (R0) only -- G1, G2, G3 not scorable zero-shot")
        save(
            fig, name, [RELIABILITY_DISCRIMINATION], f"{key} on the frozen candidates of the sample"
        )

    frontier = cc_read_json(RISK_FRONTIER)["R0"]["per_environment"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    keys = [epsilon_key(e) for e in EPSILONS]
    for n in environments:
        ax.plot(
            [float(e) for e in EPSILONS],
            [frontier[n][k] for k in keys],
            marker="o",
            linewidth=1.0,
            label=_short(n),
        )
    ax.set_xlabel("harm tolerance epsilon")
    ax.set_ylabel("repair recall at the oracle cut")
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Oracle harm-repair frontier of the frozen ranking, G0 (R0)")
    save(fig, "prefix_harm_vs_repair.png", [RISK_FRONTIER], "repair recall at each harm tolerance")

    purity = cc_read_json(PREFIX_PURITY)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for j, (label, source, key, colour) in enumerate(
        (
            ("all G0 candidates", "R0", "block_harm_rate", xc1.GRID),
            ("frozen-ranked prefix", "R0", "prefix_harm_rate", ARM_COLOURS[G0]),
            (
                "randomly ranked prefix",
                "R0_random_ranking",
                "prefix_harm_rate",
                ARM_COLOURS[X_IMG],
            ),
        )
    ):
        ax.bar(
            np.arange(len(environments)) + (j - 1) * 0.27,
            [purity[source][n][key] for n in environments],
            width=0.27,
            color=colour,
            label=label,
        )
    ax.set_xticks(
        range(len(environments)), [_short(n) for n in environments], rotation=40, ha="right"
    )
    ax.set_ylabel("harmful share")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Prefix purity at the frozen cut's coverage, G0 (R0)")
    save(
        fig,
        "matched_coverage_prefix_purity.png",
        [PREFIX_PURITY],
        "harmful share of matched-coverage prefixes",
    )

    cut = cc_read_json(FROZEN_CUT_TRANSFER)["R0"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    axes[0].bar(
        range(len(environments)), [cut[n]["coverage"] for n in environments], color=ARM_COLOURS[G0]
    )
    axes[1].bar(
        range(len(environments)),
        [cut[n]["harm_rate_accepted"] for n in environments],
        color=ARM_COLOURS[G0],
    )
    axes[1].axhline(PRIMARY_EPSILON, color=xc1.INK_SECONDARY, linewidth=0.8, linestyle="--")
    for axis, title in ((axes[0], "coverage"), (axes[1], "harm rate of accepted")):
        axis.set_xticks(
            range(len(environments)),
            [_short(n) for n in environments],
            rotation=60,
            ha="right",
            fontsize=6,
        )
        axis.set_title(title, fontsize=8)
    fig.suptitle("The frozen cut, transported unchanged, on G0's sampled candidates", fontsize=9)
    save(
        fig,
        "frozen_cut_transfer.png",
        [FROZEN_CUT_TRANSFER],
        "coverage and accepted harm at the frozen cut",
    )

    shift = cc_read_json(DISTRIBUTION_SHIFT)["profiles"]
    measures_shift = (
        ("insertion at a gap", lambda p: p["edit_type_share"]["insertion_at_gap"]),
        ("deletion", lambda p: p["edit_type_share"]["deletion"]),
        ("replacement", lambda p: p["edit_type_share"]["replacement"]),
        ("beneficial", lambda p: p["beneficial_prevalence"]),
        ("harmful", lambda p: p["harmful_prevalence"]),
        ("at a clean site", lambda p: p["clean_site_prevalence"]),
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for j, arm in enumerate((G0, G1, G2)):
        ax.bar(
            np.arange(len(measures_shift)) + (j - 1) * 0.26,
            [f(shift[arm]) for _l, f in measures_shift],
            width=0.26,
            color=ARM_COLOURS[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(range(len(measures_shift)), [m[0] for m in measures_shift], rotation=15)
    ax.set_ylabel("share of candidates")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("How the new candidates differ from the frozen generator's")
    save(
        fig,
        "candidate_distribution_shift.png",
        [DISTRIBUTION_SHIFT],
        "edit types and label prevalence per arm",
    )

    clean_reliability = cc_read_json(CLEAN_SITE_RELIABILITY)["R0_reference"]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for j, (_label, key) in enumerate(
        (
            ("error-site candidates", "error_score_quantiles"),
            ("clean-site candidates", "clean_score_quantiles"),
        )
    ):
        q = clean_reliability[key]
        ax.plot([j, j], [q["p10"], q["p90"]], color=ARM_COLOURS[G0], linewidth=6, alpha=0.4)
        ax.scatter(j, q["median"], color=xc1.INK, zorder=3)
    ax.set_xticks([0, 1], ["G0 at error sites", "G0 at clean sites"])
    ax.set_ylabel("frozen score (p10-p90, median)")
    ax.set_title("Frozen scores at clean and error sites, G0 (R0); G2 not scorable")
    save(
        fig,
        "clean_site_reliability.png",
        [CLEAN_SITE_RELIABILITY],
        "score distributions by site class",
    )

    ceiling = cc_read_json(CEILING)["mean"]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    for i, arm in enumerate((G0, G2, G3)):
        row = ceiling[arm]
        parts = [
            ("proposal loss", row["proposal_loss"], xc1.GRID),
            ("generation loss", row["generation_loss"], ARM_COLOURS[G1]),
        ]
        if row["reliability_ranking_loss"] is not None:
            parts += [
                ("reliability-ranking loss", row["reliability_ranking_loss"], ARM_COLOURS[X_IMG]),
                (
                    "inside the risk-controlled prefix",
                    row["risk_controlled_repair"],
                    ARM_COLOURS[G2],
                ),
            ]
        else:
            parts += [
                ("exact candidate, not scorable zero-shot", row["opportunity"], ARM_COLOURS[G3])
            ]
        left = 0.0
        for label, value, colour in parts:
            ax.barh(
                i,
                value,
                left=left,
                color=colour,
                label=None if label in ax.get_legend_handles_labels()[1] else label,
            )
            left += value
    ax.set_yticks(range(3), [ARM_LABELS[a] for a in (G0, G2, G3)])
    ax.set_xlabel("share of OCR error sites on the sampled documents")
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Updated ceiling decomposition")
    save(
        fig,
        "updated_ceiling_decomposition.png",
        [CEILING],
        "proposal, generation and reliability losses",
    )

    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_analysis_envelope("figure_manifest"),
            "figures": manifest,
            "count": len(manifest),
            "illustrative_values": False,
            "note": FIGURE_NOTE,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(manifest)} written")
    return 0


# ------------------------------------------------------------------ section 62: determinism

DERIVED_OUTPUTS = (
    "CANDIDATES",
    "SITE_OUTCOMES",
    "PROPOSAL_LINKS",
    "ERROR_SITES",
    "P0_COVERAGE",
    "RAW_OUTPUT_MANIFEST",
    "RAW_OUTPUT_HASHES",
    "CANDIDATE_INVENTORY",
    "CANDIDATE_NORMALIZATION",
    "GENERATION_RESULTS",
    "OPPORTUNITY_RESULTS",
    "CLEAN_SITE_RESULTS",
    "ERROR_TYPE_ANALYSIS",
    "ENGINE_ANALYSIS",
    "DOMAIN_ANALYSIS",
    "CANDIDATE_OVERLAP",
    "UNION_RESULTS",
    "IMAGE_SHUFFLE",
    "CROP_MASK",
    "CONTROL_RESULTS",
    "PROPOSAL_MISS_ORACLE",
    "ZERO_SHOT_SCORES",
    "RELIABILITY_DISCRIMINATION",
    "PREFIX_PURITY",
    "RISK_FRONTIER",
    "FROZEN_CUT_TRANSFER",
    "DISTRIBUTION_SHIFT",
    "CLEAN_SITE_RELIABILITY",
    "CEILING",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
    "DECISION",
)
DETERMINISM_ENVIRONMENTS = ("funsd/doctr", "sbb/doctr")


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("normalize", run_normalize),
        ("opportunity", run_opportunity),
        ("controls", run_controls),
        ("oracle", run_oracle),
        ("reliability", run_reliability),
        ("ceiling", run_ceiling),
        ("stats", run_stats),
        ("negative", run_negative),
        ("figures", run_figures),
        ("decide", run_decide),
    )


def _current_signatures() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in DERIVED_OUTPUTS:
        path = globals()[name]
        if isinstance(path, Path) and path.is_file():
            out[path.name] = _signature(path)
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
    """Section 62: two regenerations from the raw outputs, the inputs rebuilt, the model rerun."""
    started = time.monotonic()
    _require(DECISION, "decide")
    _forbid(DETERMINISM)
    published = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(
        name for name in published if any(run.get(name) != published[name] for run in runs)
    )
    sample = pd.read_parquet(PROPOSAL_SAMPLE)
    rebuilt_sample = []
    for spec in environment_specs():
        name = spec["environment"]
        requests = _p0_requests(name)
        per_document = requests.groupby("document_id").size()
        chosen = choose_documents(
            per_document.index.astype(str).tolist(), per_document.tolist(), name
        )
        frozen = sorted(set(sample[sample["environment"] == name]["document_id"].astype(str)))
        rebuilt_sample.append({"environment": name, "identical": sorted(chosen) == frozen})
    contexts = pd.read_parquet(CONTEXTS).set_index("site_id")
    rebuilt_contexts = []
    for name in DETERMINISM_ENVIRONMENTS:
        sites = sample[sample["environment"] == name].reset_index(drop=True)
        again = build_natural_contexts(
            sites, gen1._page_index(_spec(name)), CACHE / "determinism" / "crops" / _slug(name)
        ).set_index("site_id")
        rebuilt_contexts.append(
            {
                "environment": name,
                "sites": len(again),
                "text_identical": bool(
                    (again["e1_text"] == contexts.loc[again.index, "e1_text"]).all()
                ),
                "crops_identical": bool(
                    (
                        again["crop_sha256"].fillna("")
                        == contexts.loc[again.index, "crop_sha256"].fillna("")
                    ).all()
                ),
            }
        )
    regeneration = {
        path.stem: cc_read_json(path) for path in sorted(REGENERATION_DIR.glob("*.json"))
    }
    model_identical = bool(regeneration) and all(r["identical"] for r in regeneration.values())
    all_identical = bool(
        not differing
        and all(r["identical"] for r in rebuilt_sample)
        and all(r["text_identical"] and r["crops_identical"] for r in rebuilt_contexts)
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
            "sample_rebuild": rebuilt_sample,
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


# ------------------------------------------------------------------ section 61, 63: the record

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE,),
    "2. Frozen SGV-GEN1 Result": (RESEARCH_FREEZE,),
    "3. Research Questions": (DESIGN_RECORD,),
    "4. Why Oracle Localization Must Be Removed": (DESIGN_RECORD, RESEARCH_FREEZE),
    "5. Non-Goals": (DESIGN_RECORD,),
    "6. Frozen Site Proposer": (
        SITE_PROPOSER_REGISTRY,
        SITE_PROPOSAL_INVENTORY,
        ENVIRONMENT_INVENTORY,
    ),
    "7. GT-Blind Crop Construction": (
        CROP_POLICY,
        CONTEXT_CENSUS,
        CONTROL_RESULTS,
        SITE_PROPOSAL_INVENTORY,
    ),
    "8. Candidate Generators": (MODEL_REGISTRY, PROMPT_REGISTRY, DECODING_REGISTRY, DESIGN_RECORD),
    "9. Raw-Output Freeze": (RAW_OUTPUT_MANIFEST, RAW_OUTPUT_HASHES),
    "10. Baseline Reproduction": (FALSIFICATION, VERIFIER_REPRODUCTION, CANDIDATE_NORMALIZATION),
    "11. Natural Site Proposal Recall": (GENERATION_RESULTS,),
    "12. Natural Exact Generation": (GENERATION_RESULTS, RESEARCH_FREEZE),
    "13. Opportunity Recall": (OPPORTUNITY_RESULTS, STATISTICAL_TESTS),
    "14. Image vs Text Translation": (OPPORTUNITY_RESULTS, STATISTICAL_TESTS, RESEARCH_FREEZE),
    "15. Weak Error Types": (ERROR_TYPE_ANALYSIS, STATISTICAL_TESTS),
    "16. Historical Glyphs": (ERROR_TYPE_ANALYSIS,),
    "17. Clean-Site Behavior": (CLEAN_SITE_RESULTS, STATISTICAL_TESTS, GENERATION_RESULTS),
    "18. Harmful Candidate Burden": (GENERATION_RESULTS, CANDIDATE_INVENTORY),
    "19. Generator Complementarity": (CANDIDATE_OVERLAP,),
    "20. Union Candidate Universe": (UNION_RESULTS, OPPORTUNITY_RESULTS),
    "21. Image-Shuffle Control": (IMAGE_SHUFFLE, CROP_MASK, STATISTICAL_TESTS),
    "22. Proposal-Miss Oracle": (PROPOSAL_MISS_ORACLE,),
    "23. Generation Gate": (
        DECISION,
        OPPORTUNITY_RESULTS,
        ERROR_TYPE_ANALYSIS,
        GENERATION_RESULTS,
        IMAGE_SHUFFLE,
        STATISTICAL_TESTS,
    ),
    "24. Frozen Reliability Model": (VERIFIER_REPRODUCTION, FROZEN_CONFIGURATION),
    "25. Feature Compatibility": (FEATURE_COMPATIBILITY, ZERO_SHOT_SCORES),
    "26. Reliability Reproduction": (VERIFIER_REPRODUCTION,),
    "27. Zero-Shot Harm Ranking": (RELIABILITY_DISCRIMINATION, ZERO_SHOT_SCORES),
    "28. Zero-Shot Benefit Ranking": (RELIABILITY_DISCRIMINATION,),
    "29. Prefix Purity": (PREFIX_PURITY,),
    "30. Risk-Controlled Oracle Frontier": (RISK_FRONTIER,),
    "31. Frozen Cut Transfer": (FROZEN_CUT_TRANSFER,),
    "32. Clean-Site Reliability": (CLEAN_SITE_RELIABILITY,),
    "33. Candidate Distribution Shift": (DISTRIBUTION_SHIFT,),
    "34. Engine Analysis": (ENGINE_ANALYSIS,),
    "35. Domain Analysis": (DOMAIN_ANALYSIS,),
    "36. Updated Ceiling Decomposition": (CEILING,),
    "37. Controls": (CONTROL_RESULTS, IMAGE_SHUFFLE, CROP_MASK),
    "38. Falsification Tests": (FALSIFICATION, DETERMINISM),
    "39. Statistics": (STATISTICAL_TESTS,),
    "40. Limitations": (
        DESIGN_RECORD,
        SITE_PROPOSAL_INVENTORY,
        FEATURE_COMPATIBILITY,
        CONTEXT_CENSUS,
        DECISION,
    ),
    "41. Research Decision": (DECISION,),
    "42. Next Stage": (DECISION,),
}
PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    ENVIRONMENT_INVENTORY,
    DESIGN_RECORD,
    SITE_PROPOSER_REGISTRY,
    SITE_PROPOSAL_INVENTORY,
    MODEL_REGISTRY,
    PROMPT_REGISTRY,
    DECODING_REGISTRY,
    CROP_POLICY,
    FEATURE_COMPATIBILITY,
    VERIFIER_REPRODUCTION,
    PROPOSAL_SAMPLE,
    CONTEXTS,
    CONTROL_CONTEXTS,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)


def run_record() -> int:
    """Provenance, the upstream audit, the decision re-derivation and the traceability index."""
    started = time.monotonic()
    for path in (PROVENANCE, TRACEABILITY):
        _forbid(path)
    missing = [p for p in PRODUCED if not p.exists() and p not in (PROVENANCE, TRACEABILITY)]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {[_relative(p) for p in missing][:5]}")
    if not cc_read_json(DETERMINISM)["all_runs_identical"]:
        raise PhaseError("determinism did not pass; the record will not be issued")
    freeze = cc_read_json(RESEARCH_FREEZE)["upstream_sha256"]
    moved = sorted(p for p, sha in freeze.items() if file_sha256(REPO / p) != sha)
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
    shards = {_relative(p): file_sha256(p) for p in sorted(RAW_CACHE.glob("*.parquet"))}
    figures = {_relative(p): file_sha256(p) for p in sorted(FIGURE_DIR.glob("*.png"))}
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "issued_head": _git("rev-parse", "HEAD"),
            "xr1_starting_state": cc_read_json(RESEARCH_FREEZE)["repository_state"]["head"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "raw_generation_output_count": len(shards),
            "raw_generation_outputs": shards,
            "figures": figures,
            "upstream_files_rehashed": len(freeze),
            "upstream_files_moved": moved,
            "decision_rederived_identically": rederived,
            "report": {"path": _relative(REPORT), "sha256": file_sha256(REPORT)},
            "dependency_audit": {
                "confirmatory_reserve_consumed": False,
                "external_confirmation_run": False,
                "reliability_adaptation_started": False,
                "new_reliability_model_trained": False,
                "reliability_features_added": 0,
                "new_or_larger_correction_model": False,
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
        f"record: {len(artifacts)} artifacts, {len(shards)} raw shards, report claims "
        f"{audit['numeric_claims']}"
    )
    return 0


# ------------------------------------------------------------------ entry point

PHASES: tuple[tuple[str, Callable[[], int]], ...] = (
    ("reconstruct", run_reconstruct),
    ("freeze", run_freeze),
    ("preregister", run_preregister),
    ("compat", run_compat),
    ("proposals", run_proposals),
    ("contexts", run_contexts),
    ("generate", run_generate),
    ("spotcheck", run_spotcheck),
    ("normalize", run_normalize),
    ("opportunity", run_opportunity),
    ("controls", run_controls),
    ("oracle", run_oracle),
    ("reliability", run_reliability),
    ("ceiling", run_ceiling),
    ("stats", run_stats),
    ("negative", run_negative),
    ("figures", run_figures),
    ("decide", run_decide),
    ("determinism", run_determinism),
    ("record", run_record),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    for name, _function in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    arguments = parser.parse_args(argv)
    selected = [(n, f) for n, f in PHASES if getattr(arguments, n)]
    if not selected:
        parser.print_help()
        return 2
    for name, function in selected:
        began = time.monotonic()
        code = function()
        print(f"  [{name} {time.monotonic() - began:.0f}s]", flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PhaseError as error:
        print(f"PHASE ERROR: {error}", file=sys.stderr)
        sys.exit(1)
