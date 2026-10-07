#!/usr/bin/env python3
"""SGV-PF1: scaling the deployment frontier with an expanded labelled page pool.

SGV-DEP1 closed with outcome D: no conservative policy -- full automation or human-in-the-loop
triage -- was deployable or practical within the thirty-odd labelled pages the generator universe
held, and SGV-TH2 had projected that a page-corrected boundary needs about a hundred. This stage
is the first to extend the labelled page pool instead of re-reading it, and asks:

    how many labelled pages does a new correction generator need before safe, useful automation
    becomes possible -- and when it does not, is the ranking or the calibration evidence short?

**1. This is a data-generation stage.** New candidate generation runs on pages the generator
universe never contained: SGV14's frozen OCR of 161 clean development pages (133 FUNSD forms, 28
OCR-D-SBB historical pages), each read by one of the ten frozen engine configurations. No OCR is
re-run: those pages were already recognized by SGV14 and their canonical spans are reused. What is
new is the frozen hybrid proposer and the frozen image corrector run on them, and the labels the
frozen labelling function then attaches from each corpus's own human transcription.

**2. The chain is the frozen chain, proved on the old pages first.** Every step -- the lattice,
P0, the line prompts and crops, the matched budget, the proposal strata, the H5 routing, the
correction contexts and prompts, normalization, labelling, U5 deduplication, RK1 grades, RL1's R3
and RK3's R5 features -- calls the upstream stage's own functions. `--replay` runs PF1's chain over
the 49 frozen U5 pages from their frozen raw answers and requires it to rebuild the frozen tables
exactly before a new page is decoded or labelled.

**3. The pool is a census, fixed before any answer.** Every clean page is taken; nothing is chosen
by content. The engine reading, the fold and the pool/test split are content-blind hash orders,
written with the page registry before the first decode.

**4. The analysis is RK4's and DEP1's, with more pages.** RK4's frozen source fit, arms and cutoff
rules, and DEP1's four policies, over a target page pool of about 130 pages and a fresh sealed test
block no earlier analysis has read.

    --reconstruct   section 0: DEP1, RK4, RK3, RL1, HY1, LP1 and XR1 re-read from their artifacts
    --freeze        upstream hashes, the frozen generator configuration, the reserve lock
    --pool          the page census, engine readings, folds and splits, before any decode
    --lattice       P0 re-derived, the GT-blind lattice and the scan population on the new pages
    --lines         line prompts and line crops for the image proposer
    --plan          the generation plan: shards, the matched budget rule, the compute estimate
    --replay        PF1's GT-blind chain over the frozen U5 pages, required to rebuild them
    --replay-labels PF1's labelled chain over U5: candidates, links, U5, grades, R5, exactly
    --generate      GT-blind inference into immutable per-shard caches (PF1_ARMS=<arms>)
    --parse         raw proposer answers -> lattice proposals -> the matched-budget selection
    --strata        proposal strata and the frozen H5 routing
    --contexts      correction contexts and crops for every routed site
    --preregister   questions, blocks, budgets, arms, policies, definitions, outcome rules
    --normalize     ground truth enters: atomic edits, SGV14 labels, the candidate table
    --link          evaluable OCR error sites and every lattice site's links
    --population    RL1's U5 rule and RK1's grades on the new pages; page annotation status
    --features      R3 and RK3's new families from the frozen builders
    --splits        every direction's blocks and every draw's purchase
    --adapt         A0, M1 and B2 by budget, draw and purchase scheme
    --policies      DEP1's four policies, ranking metrics and the analysis-only oracle
    --controls      random scores, target-label permutation, A0 against RK4
    --curves        label-budget curves per set, arm and scheme
    --frontier      N* per direction, policy and flag, masked by the random control
    --diversity     RQ2: stability, worst-case harm, cross-document robustness, stratified pages
    --bottleneck    RQ3: the test-oracle cutoff against the rules, and the pool diagnosis
    --stats         exact sign tests over paired draws, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --spotcheck     one whole shard per generation arm regenerated and compared
    --determinism   every derived phase twice, and the labelled chain once more
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / DATA GENERATION + DEPLOYMENT METHODOLOGY. Nothing is certified, nothing is
production-ready, and no deployment claim is made unless the pre-registered safety criteria hold.
The confirmatory reserve stays LOCKED and `ready_for_external_confirmation` is false by
construction.
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_dep1_deployment_frontier as dep1
from ocr_risk.io.hashing import canonical_hash, file_sha256

rk4 = dep1.rk4
rk3 = dep1.rk3
th1 = dep1.th1
th2 = dep1.th2
rk2 = dep1.rk2
rk1 = dep1.rk1
ds1 = dep1.ds1
rl1 = dep1.rl1
s15 = dep1.s15
hy1 = rl1.hy1
lp1 = hy1.lp1
xr1 = hy1.xr1
gen1 = hy1.gen1
xc1 = gen1.xc1
s14 = gen1.s14
cg1 = gen1.cg1

REPO = rk4.REPO
OUT = REPO / "results/generated/sgv_pf1"
CACHE = OUT / "cache"
# Raw answers live apart from every derived table, so a GT-blind phase never reads a label beside
# them.
RAW_CACHE = OUT / "raw_generation_outputs"
LINE_CROP_DIR = CACHE / "line_crops"
SITE_CROP_DIR = CACHE / "correction_crops"
REPLAY_DIR = CACHE / "replay"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
PAGE_POOL = OUT / "page_pool.parquet"
PAGE_REGISTRY = OUT / "page_registry.json"
SITE_LATTICE = OUT / "site_lattice.parquet"
SCAN_POPULATION = OUT / "scan_population.parquet"
P0_PROPOSALS = OUT / "p0_proposals.parquet"
LATTICE_INVENTORY = OUT / "lattice_inventory.json"
LINE_CONTEXTS = OUT / "line_contexts.parquet"
GENERATION_PLAN = OUT / "generation_plan.json"
REPLAY = OUT / "chain_replay.json"
LABEL_REPLAY = OUT / "label_replay.json"
PARSED_PROPOSALS = OUT / "parsed_proposals.parquet"
PROPOSAL_COMPLIANCE = OUT / "proposal_compliance.json"
PROPOSAL_STRATA = OUT / "proposal_strata.parquet"
ROUTING_REGISTRY = OUT / "routing_registry.json"
CORRECTION_CONTEXTS = OUT / "correction_contexts.parquet"
DESIGN_RECORD = OUT / "design_record.json"
# Everything below is written with ground truth.
CANDIDATES = OUT / "normalized_candidates.parquet"
SITE_TRUTH = OUT / "site_truth.parquet"
ERROR_SITES = OUT / "error_site_table.parquet"
PROPOSAL_ERROR_LINKS = OUT / "proposal_error_links.parquet"
CANDIDATE_INVENTORY = OUT / "candidate_inventory.json"
POPULATION = OUT / "population.parquet"
POPULATION_INVENTORY = OUT / "population_inventory.json"
PAGE_ANNOTATION = OUT / "page_annotation.parquet"

REPORT = REPO / "docs/sgv_pf1/page_frontier_scaling.md"

SCHEMA_VERSION = 1
STAGE = "sgv_pf1_page_frontier_scaling"
HYPOTHESIS = "SGV-PF1-D1"
STAGE_KIND = "DEVELOPMENT / DATA GENERATION + DEPLOYMENT METHODOLOGY"

PhaseError = rk4.PhaseError
_relative = rk4._relative
_git = rk4._git
_write_json_once = rk4._write_json_once
_write_parquet_once = rk4._write_parquet_once
cc_read_json = rk4.cc_read_json
_require = rk4._require
_forbid = rk4._forbid
_share = rk4._share
_slug = gen1._slug

# ------------------------------------------------------------------ the frozen generation design

POOL_SEED = "sgv-pf1-page-pool-v1:2026-09-26"
FOLD_SEED = "sgv-pf1-page-folds-v1:2026-09-26"
FOLDS = 5
POOL_FOLDS = (0, 1, 2)
TEST_FOLDS = (3, 4)
SPLIT_POOL = "adaptation_pool"
SPLIT_TEST = "test"
# A shard is at most this many pages of one environment, so an interrupted run loses little.
CHUNK_PAGES = 4

MODEL = hy1.MODEL
PROPOSER = "pf1_image_proposer"
CORRECTOR = "pf1_image_corrector"
GENERATION_ARMS = (PROPOSER, CORRECTOR)
# The frozen method each generation arm runs, and the upstream arm it is identical to.
UPSTREAM_ARM = {PROPOSER: lp1.L2, CORRECTOR: hy1.C1}
PRIMARY_ARM_OF_HY1 = hy1.H5
C0 = hy1.C0
C1 = hy1.C1
BUDGET_COLUMN = hy1.PROPOSAL_BUDGET_COLUMN
# Throughput measured on this machine by the upstream stages' own shard sidecars.
LP1_LINES_PER_SECOND = 0.0702
HY1_REQUESTS_PER_SECOND = 0.218

GENERATOR_SOURCES = f"{C0}|{C1}"
ANNOTATION_SOURCE = {
    "funsd": "FUNSD word-level transcription (human annotation shipped with the corpus)",
    "ocrd_sbb": "OCR-D-SBB PAGE-XML ground truth (human transcription shipped with the corpus)",
}

UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_dep1": {
        "path": dep1.DECISION,
        "outcome": "D",
        "recommended_next_stage": dep1.NEXT_STAGE["D"],
        "ready_for_external_confirmation": False,
        "deployment_claim_permitted": False,
    },
    "sgv_rk4": {
        "path": rk4.DECISION,
        "outcome": "B",
        "recommended_next_stage": "NEXT: EXPAND THE LABELLED PAGE POOL FOR THE ADAPTED RANKER",
        "ready_for_external_confirmation": False,
        "deployment_claim_permitted": False,
    },
    "sgv_rk3": {"path": rk3.DECISION, "outcome": "B", "deployment_claim_permitted": False},
    "sgv_rl1": {"path": rl1.DECISION, "outcome": "B"},
    "sgv_hy1": {"path": hy1.DECISION, "outcome": "A"},
    "sgv_lp1": {"path": lp1.DECISION, "outcome": "B"},
    "sgv_xr1": {"path": xr1.DECISION, "outcome": "C"},
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_pf1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "stage_kind": STAGE_KIND,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str) -> dict[str, Any]:
    return {
        **_envelope(artifact),
        "uses_ground_truth": True,
        "production_ready": False,
        "certified": False,
        "confirmatory_reserve_consumed": False,
    }


def _labels_exist() -> bool:
    """Whether ground truth has reached this stage. An existence check that reads no label."""
    return any(p.exists() for p in (CANDIDATES, SITE_TRUTH, ERROR_SITES, PROPOSAL_ERROR_LINKS))


def environment_specs() -> list[dict[str, Any]]:
    return gen1.environment_specs()


def _spec(name: str) -> dict[str, Any]:
    return gen1._spec(name)


@contextmanager
def rebound(bindings: Sequence[tuple[Any, str, Any]]) -> Iterator[None]:
    """Point upstream module globals at PF1's tables for one call, and always restore them.

    The upstream feature builders read their inputs from module-level paths. Rebinding those paths
    runs the upstream code itself -- not a copy of it -- on PF1's pages.
    """
    saved = [(module, name, getattr(module, name)) for module, name, _value in bindings]
    try:
        for module, name, value in bindings:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    """Every upstream artifact PF1 reads, and the code it calls, frozen by hash."""
    files: list[Path] = [
        dep1.DECISION,
        rk4.DECISION,
        rk4.SPLIT_REGISTRY,
        rk4.CELL_SCORES,
        rk4.ADAPTATION_REGISTRY,
        rk3.DECISION,
        rk3.FEATURE_MATRIX,
        rk1.RANKING_POPULATION,
        rl1.DECISION,
        rl1.FITTED_RESOURCES,
        rl1.FEATURE_MATRIX,
        rl1.CANDIDATE_POPULATION,
        rl1.GROUP_REGISTRY,
        hy1.DECISION,
        hy1.PROPOSAL_STRATA,
        hy1.CONTEXTS,
        hy1.CANDIDATES,
        hy1.SITE_TRUTH,
        hy1.ERROR_SITES,
        hy1.CACHE / "cache_reuse_map.parquet",
        hy1.PROMPT_REGISTRY,
        hy1.DECODING_REGISTRY,
        hy1.CROP_REGISTRY,
        lp1.DECISION,
        lp1.SITE_LATTICE_INVENTORY,
        lp1.SCAN_POPULATION,
        lp1.P0_PROPOSALS,
        lp1.LINE_CONTEXTS,
        lp1.PARSED_PROPOSALS,
        lp1.ERROR_SITES,
        lp1.PROPOSAL_ERROR_LINKS,
        lp1.PROMPT_REGISTRY,
        lp1.DECODING_REGISTRY,
        lp1.IMAGE_PREPROCESSING_REGISTRY,
        lp1.DESIGN_RECORD,
        xr1.DECISION,
        xr1.PROPOSAL_SAMPLE,
        xr1.CONTEXTS,
        xr1.PROMPT_REGISTRY,
        xr1.DECODING_REGISTRY,
        xr1.CROP_POLICY,
        cg1.CANDIDATE_ROWS,
        rl1.FEATURE_REGISTRY,
        REPO / "manifests/sgv1/role_manifest.json",
        REPO / "manifests/sgv1/confirmatory_reserve_lock.json",
    ]
    for spec in environment_specs():
        slug = _slug(spec["environment"])
        files.extend(
            [
                xc1.CACHE / f"{slug}.m0_candidates.parquet",
                xc1.CACHE / f"{slug}.requests.parquet",
                lp1._shard_path(lp1.L2, spec["environment"]),
                xr1._shard_path(xr1.G2, spec["environment"]),
                hy1._shard_path(hy1.NEW_IMAGE, spec["environment"]),
            ]
        )
    files.extend(
        REPO / f"scripts/{name}.py"
        for name in (
            "sgv14_confirmatory_validation",
            "sgv_cg1_opportunity_ceiling",
            "sgv_xc1_cross_correction_opportunity",
            "sgv_gen1_error_conditioned_generation",
            "sgv_xr1_image_candidate_reliability",
            "sgv_lp1_image_error_site_proposal",
            "sgv_hy1_hybrid_candidate_generation",
            "sgv_rl1_generator_agnostic_reliability",
            "sgv_rk1_learning_to_rank",
            "sgv_rk2_fewshot_ranker_adaptation",
            "sgv_rk3_risk_aware_ranking",
            "sgv_rk4_generator_adaptive_ranking",
            "sgv_th1_threshold_calibration",
            "sgv_th2_page_budget_frontier",
            "sgv_ds1_deployment_synthesis",
            "sgv_dep1_deployment_frontier",
        )
    )
    files.extend(REPO / path for path in lp1.UPSTREAM_MODULES)
    return [p for p in files if p.is_file()]


def upstream_checks() -> list[dict[str, Any]]:
    """What the brief says the upstream stages found, read from their own decision records."""
    checks: list[dict[str, Any]] = []
    for stage, expected in UPSTREAM_EXPECTED.items():
        record = cc_read_json(expected["path"])
        for key, value in expected.items():
            if key == "path":
                continue
            checks.append(
                {
                    "check": f"{stage}.{key}",
                    "expected": value,
                    "observed": record.get(key),
                    "agrees": record.get(key) == value,
                }
            )
    lock = cc_read_json(REPO / "manifests/sgv1/confirmatory_reserve_lock.json")
    checks.append(
        {
            "check": "confirmatory_reserve.status",
            "expected": "LOCKED",
            "observed": lock["status"],
            "agrees": lock["status"] == "LOCKED",
        }
    )
    return checks


def run_reconstruct() -> int:
    started = time.monotonic()
    _forbid(RESEARCH_FREEZE)
    checks = upstream_checks()
    failed = [c for c in checks if not c["agrees"]]
    if failed:
        raise PhaseError(f"{len(failed)} upstream checks failed: {[c['check'] for c in failed]}")
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {_relative(p): file_sha256(p) for p in _upstream_files()}
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "upstream_checks": checks,
            "upstream_sha256": hashes,
            "upstream_file_count": len(hashes),
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status": _git("status", "--short"),
                "diff_stat": _git("diff", "--stat"),
            },
            "regenerates_ocr": False,
            "generates_new_candidates": True,
            "modifies_an_upstream_artifact": False,
            "uses_ground_truth": False,
        },
    )
    print(
        f"reconstruct: {len(checks)} upstream checks pass, {len(hashes)} files frozen "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def generation_hashes() -> dict[str, str]:
    """The frozen proposer and corrector configuration, recomputed from the upstream constants."""
    proposer_prompt = canonical_hash({"system": lp1.SYSTEM_PROMPT, "user": lp1.USER_TEMPLATES})
    proposer_decoding = canonical_hash(
        {
            "revision": lp1.DECODING_REVISION,
            "rule": [lp1.CAP_BASE_TOKENS, lp1.CAP_TOKENS_PER_PROPOSAL, lp1.MAX_PROPOSALS_PER_LINE],
            "settings": "gen1._generation_settings",
            "batch": int(gen1.MODEL_SPECS[MODEL]["batch_size"]),
            "dtype": gen1.MODEL_DTYPE,
        }
    )
    proposer_image = canonical_hash(
        {
            "revision": lp1.IMAGE_PREPROCESSING_REVISION,
            "height": lp1.LINE_CROP_HEIGHT,
            "max_width": lp1.LINE_CROP_MAX_WIDTH,
            "pad": [gen1.CROP_MIN_PAD, gen1.CROP_PAD_FRACTION],
            "resample": "LANCZOS",
            "format": "PNG RGB",
        }
    )
    corrector_prompt = canonical_hash({"system": gen1.SYSTEM_PROMPT, "user": gen1.USER_TEMPLATES})
    return {
        "proposer_prompt_hash": proposer_prompt,
        "proposer_decoding_hash": proposer_decoding,
        "proposer_image_hash": proposer_image,
        "corrector_prompt_hash": corrector_prompt,
        "corrector_decoding_hash": cc_read_json(hy1.DECODING_REGISTRY)["decoding_hash"],
        "corrector_crop_hash": xr1._crop_policy()["policy_hash"],
    }


def run_freeze() -> int:
    """The generator configuration is LP1's and HY1's, byte for byte, or nothing is decoded."""
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "reconstruct")
    _forbid(FROZEN_CONFIGURATION)
    if _labels_exist():
        raise PhaseError("a PF1 label exists; the freeze must precede every one")
    hashes = generation_hashes()
    frozen = {
        "proposer_prompt_hash": cc_read_json(lp1.PROMPT_REGISTRY)["prompt_hash"],
        "proposer_decoding_hash": cc_read_json(lp1.DECODING_REGISTRY)["decoding_hash"],
        "proposer_image_hash": cc_read_json(lp1.IMAGE_PREPROCESSING_REGISTRY)["policy_hash"],
        "corrector_prompt_hash": cc_read_json(hy1.PROMPT_REGISTRY)["prompt_hash"],
        "corrector_decoding_hash": cc_read_json(xr1.DECODING_REGISTRY)["decoding_hash"],
        "corrector_crop_hash": cc_read_json(hy1.CROP_REGISTRY)["policy_hash"],
    }
    differing = sorted(k for k in hashes if hashes[k] != frozen[k])
    if differing:
        raise PhaseError(f"the generator configuration differs from LP1/HY1: {differing}")
    spec = gen1.MODEL_SPECS[MODEL]
    snapshot = gen1._snapshot(MODEL)
    if snapshot is None:
        raise PhaseError(f"{MODEL}: no local snapshot")
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "model": MODEL,
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "batch_size": int(spec["batch_size"]),
            "dtype": gen1.MODEL_DTYPE,
            "snapshot_present": True,
            "generation_hashes": hashes,
            "identical_to_upstream": {k: hashes[k] == frozen[k] for k in hashes},
            "proposer": {
                "method": lp1.L2,
                "system_prompt": lp1.SYSTEM_PROMPT,
                "user_template": lp1.USER_TEMPLATES[lp1.TEMPLATE_OF_ARM[lp1.L2]],
                "max_proposals_per_line": lp1.MAX_PROPOSALS_PER_LINE,
                "line_crop": {"height": lp1.LINE_CROP_HEIGHT, "max_width": lp1.LINE_CROP_MAX_WIDTH},
            },
            "corrector": {
                "corrector": C1,
                "condition": hy1.CONDITION_OF_CORRECTOR[C1],
                "primary_k": hy1.PRIMARY_K,
            },
            "current_generator": {
                "corrector": C0,
                "source": (
                    "SGV-CG1's frozen candidate rows and SGV-XC1's cached candidate text, which "
                    "cover every SGV14 page of both roles; nothing is decoded"
                ),
            },
            "routing": {
                "arm": PRIMARY_ARM_OF_HY1,
                "routes": {s: list(c) for s, c in hy1.ROUTING[PRIMARY_ARM_OF_HY1].items()},
                "why": "RL1's U5, the population every later stage ranks, is HY1's dual union",
            },
            "allocator_environment": {
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.1",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.8",
            },
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"freeze: generator configuration identical to LP1/HY1 ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the page census


def u5_documents() -> set[str]:
    """The generator universe's own pages: RL1's group registry, read by document id alone."""
    return set(cc_read_json(rl1.GROUP_REGISTRY)["document_folds"])


def resource_documents() -> dict[str, set[str]]:
    """Per corpus, every page any environment's RL1/RK3 resources were fitted on."""
    records = cc_read_json(rl1.FITTED_RESOURCES)["by_environment"]
    out: dict[str, set[str]] = {}
    for spec in environment_specs():
        out.setdefault(spec["corpus"], set()).update(
            str(d) for d in records[spec["environment"]]["documents"]
        )
    return out


def order_key(*parts: str) -> str:
    return str(canonical_hash({"seed": POOL_SEED, "parts": list(parts)}))


def assign_readings(
    documents: Sequence[str], available: dict[str, set[str]], environments: Sequence[str]
) -> dict[str, str]:
    """One engine reading per page: hash order, round robin over the corpus's environments.

    A page goes to the next environment in rotation that has an OCR reading of it; the rotation
    advances past the one chosen. It reads document ids and reading availability, nothing else.
    """
    ordered = sorted(documents, key=lambda d: (order_key("reading", d), d))
    rotation = sorted(environments)
    pointer = 0
    out: dict[str, str] = {}
    for document in ordered:
        for step in range(len(rotation)):
            name = rotation[(pointer + step) % len(rotation)]
            if document in available[name]:
                out[document] = name
                pointer = (pointer + step + 1) % len(rotation)
                break
        else:
            raise PhaseError(f"{document} has no OCR reading in any environment")
    return out


def assign_folds(groups: dict[str, str]) -> dict[str, int]:
    """Content-blind folds over partition groups: hash order, each group to the smallest fold.

    A group is SGV14's partition atom -- a FUNSD form, or an OCR-D-SBB volume -- so pages of one
    volume never straddle the pool and the test block.
    """
    members: dict[str, list[str]] = {}
    for document, group in sorted(groups.items()):
        members.setdefault(group, []).append(document)
    ordered = sorted(members, key=lambda g: (str(canonical_hash({"seed": FOLD_SEED, "g": g})), g))
    sizes = [0] * FOLDS
    out: dict[str, int] = {}
    for group in ordered:
        fold = min(range(FOLDS), key=lambda f: (sizes[f], f))
        sizes[fold] += len(members[group])
        out.update(dict.fromkeys(members[group], fold))
    return out


def split_of(fold: int) -> str:
    return SPLIT_POOL if fold in POOL_FOLDS else SPLIT_TEST


def run_pool() -> int:
    """Every clean page, its one engine reading, its fold and its split. No answer exists yet."""
    started = time.monotonic()
    _require(FROZEN_CONFIGURATION, "freeze")
    for path in (PAGE_POOL, PAGE_REGISTRY):
        _forbid(path)
    if _labels_exist() or any(RAW_CACHE.glob("*.parquet")):
        raise PhaseError("an answer or label exists; the page census precedes both")
    u5 = u5_documents()
    fitted = resource_documents()
    rows: list[dict[str, Any]] = []
    census: dict[str, Any] = {}
    for corpus in sorted({s["corpus"] for s in environment_specs()}):
        specs = [s for s in environment_specs() if s["corpus"] == corpus]
        bundles = s14._document_bundles(corpus)
        roles = s14.partition_of(corpus, sorted(bundles))
        available: dict[str, set[str]] = {}
        images: dict[str, str] = {}
        for spec in specs:
            pages = gen1._page_index(spec)
            available[spec["environment"]] = {d for d, _e in pages}
            for (document, _engine), page in pages.items():
                images.setdefault(document, str(page.image_sha256))
        read = set().union(*available.values())
        excluded_u5 = read & u5
        excluded_fitted = (read - u5) & fitted.get(corpus, set())
        clean = sorted(read - u5 - fitted.get(corpus, set()))
        readings = assign_readings(clean, available, [s["environment"] for s in specs])
        groups = {d: s14.group_of(corpus, d) for d in clean}
        folds = assign_folds(groups)
        for document in clean:
            name = readings[document]
            spec = _spec(name)
            rows.append(
                {
                    "page_id": document,
                    "document_id": document,
                    "corpus": corpus,
                    "domain": xr1._domain(corpus),
                    "environment": name,
                    "ocr_engine": spec["engine_id"],
                    "base_engine": spec["base_engine"],
                    "sgv14_role": roles.get(document, ""),
                    "partition_group": groups[document],
                    "image_sha256": images[document],
                    "fold": int(folds[document]),
                    "split": split_of(int(folds[document])),
                    "generator_sources": GENERATOR_SOURCES,
                    "annotation_source": ANNOTATION_SOURCE[corpus],
                    "annotation_status": "ground truth on disk; not yet attached to any candidate",
                    "ocr_regenerated": False,
                }
            )
        census[corpus] = {
            "documents_read_by_any_environment": len(read),
            "excluded_generator_universe": len(excluded_u5),
            "excluded_resource_pages": len(excluded_fitted),
            "clean_pages": len(clean),
            "by_environment": {
                name: sum(1 for d in clean if readings[d] == name) for name in sorted(available)
            },
            "by_split": {
                split: sum(1 for d in clean if split_of(folds[d]) == split)
                for split in (SPLIT_POOL, SPLIT_TEST)
            },
            "partition_groups": len(set(groups.values())),
        }
    frame = pd.DataFrame(rows).sort_values(["environment", "page_id"], kind="stable")
    ranks = {
        name: sorted(block["page_id"], key=lambda d: (order_key("rank", name, d), d))
        for name, block in frame.groupby("environment", sort=True)
    }
    frame["document_rank"] = [
        ranks[e].index(d) for e, d in zip(frame["environment"], frame["page_id"], strict=True)
    ]
    frame["chunk"] = frame["document_rank"] // CHUNK_PAGES
    frame = frame.reset_index(drop=True)
    if frame["image_sha256"].duplicated().any():
        raise PhaseError("two pool pages share a page image")
    _write_parquet_once(PAGE_POOL, frame)
    _write_json_once(
        PAGE_REGISTRY,
        {
            **_envelope("page_registry"),
            "pages": len(frame),
            "census": census,
            "by_split": frame["split"].value_counts().sort_index().to_dict(),
            "by_environment": frame["environment"].value_counts().sort_index().to_dict(),
            "rules": {
                "eligible": (
                    "every FUNSD or OCR-D-SBB page SGV14 read, less the 49 generator-universe "
                    "pages and every page any environment's RL1 resources were fitted on"
                ),
                "never_eligible": "CORD, which holds the LOCKED confirmatory reserve",
                "selection": "a census: every eligible page is taken, none by content",
                "reading": "one engine reading per page, hash order, round robin over engines",
                "fold": "hash order over partition groups, each to the smallest of five folds",
                "split": {SPLIT_POOL: list(POOL_FOLDS), SPLIT_TEST: list(TEST_FOLDS)},
                "document_rank": "hash order within the environment",
                "chunk": f"{CHUNK_PAGES} pages of one environment per generation shard",
            },
            "seeds": {"pool": POOL_SEED, "fold": FOLD_SEED},
            "reads": ["document ids", "reading availability", "page image hashes", "SGV14 roles"],
            "reads_content": False,
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"pool: {len(frame)} pages ({census}) ({time.monotonic() - started:.0f}s)".replace("'", "")
    )
    return 0


def load_pool() -> pd.DataFrame:
    return pd.read_parquet(PAGE_POOL)


# ------------------------------------------------------------------ the GT-blind lattice


def build_lattice(
    pages: pd.DataFrame, pipeline: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """LP1's lattice, scan population and P0 table for the given (environment, page, rank) rows.

    It calls LP1's own `page_lattice`, `scan_lines`, `rederive_p0` and `map_p0_to_lattice`, so
    the same page yields the same table whichever stage asks.
    """
    lattice_frames: list[pd.DataFrame] = []
    line_frames: list[pd.DataFrame] = []
    p0_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for name, block in pages.groupby("environment", sort=True):
        spec = _spec(str(name))
        ranks = dict(zip(block["document_id"].astype(str), block["document_rank"], strict=True))
        documents = set(ranks)
        index = gen1._page_index(spec)
        chosen = {pair: page for pair, page in index.items() if pair[0] in documents}
        if {pair[0] for pair in chosen} != documents:
            raise PhaseError(f"{name}: a page has no OCR reading")
        env_lattice: list[pd.DataFrame] = []
        env_lines: list[pd.DataFrame] = []
        for (document_id, engine_id), page in sorted(chosen.items()):
            lattice = lp1.page_lattice(page, str(name), document_id, engine_id)
            env_lattice.append(lattice)
            env_lines.append(
                lp1.scan_lines(page, lattice, str(name), document_id, engine_id, ranks[document_id])
            )
        lattice = pd.concat(env_lattice, ignore_index=True)
        p0 = lp1.rederive_p0(spec, documents, pipeline)
        p0, outside, shifted = lp1.map_p0_to_lattice(p0, lattice, ranks)
        if outside or shifted:
            raise PhaseError(f"{name}: {outside} P0 anchors outside the lattice, {shifted} shifted")
        lines = pd.concat(env_lines, ignore_index=True)
        lattice_frames.append(lattice)
        line_frames.append(lines)
        p0_frames.append(p0)
        inventory.append(
            {
                "environment": str(name),
                "pages": len(chosen),
                "ocr_lines": len(lines),
                "ocr_tokens": int(lines["tokens"].sum()),
                "lines_without_a_box": int(lines["crop_box"].isna().sum()),
                "lattice_sites": len(lattice),
                "p0_proposals": len(p0),
            }
        )
    return (
        pd.concat(lattice_frames, ignore_index=True),
        pd.concat(line_frames, ignore_index=True),
        pd.concat(p0_frames, ignore_index=True),
        inventory,
    )


def p0_matches_xc1(p0: pd.DataFrame, pool: pd.DataFrame) -> dict[str, Any]:
    """On SGV14 EVALUATION pages, the re-derived P0 must equal SGV-XC1's frozen request cache."""
    key = ["site_id", "anchor_kind", "anchor_ref", "char_start", "char_end"]
    out: dict[str, Any] = {}
    for name, block in pool.groupby("environment", sort=True):
        evaluation = set(block[block["sgv14_role"] == s14.ROLE_EVALUATION]["document_id"])
        if not evaluation:
            continue
        mine = p0[(p0["environment"] == name) & p0["document_id"].astype(str).isin(evaluation)]
        cached = xr1._p0_requests(str(name))
        cached = cached[cached["document_id"].astype(str).isin(evaluation)]
        left = mine[key].astype(str).sort_values(key).reset_index(drop=True)
        right = cached[key].astype(str).sort_values(key).reset_index(drop=True)
        out[str(name)] = {"pages": len(evaluation), "sites": len(left), "equal": left.equals(right)}
    return out


def run_lattice() -> int:
    """P0 re-derived by SGV14's own call, LP1's lattice and scan population, on the new pages."""
    started = time.monotonic()
    _require(PAGE_POOL, "pool")
    for path in (SITE_LATTICE, SCAN_POPULATION, P0_PROPOSALS, LATTICE_INVENTORY):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("a PF1 label exists; the lattice precedes every one")
    pool = load_pool()
    pipeline = s14.build_pipeline()
    lattice, lines, p0, inventory = build_lattice(pool, pipeline)
    reproduction = p0_matches_xc1(p0, pool)
    if not all(r["equal"] for r in reproduction.values()):
        raise PhaseError(f"the re-derived P0 differs from SGV-XC1's cache: {reproduction}")
    for frame, what in ((lattice, "lattice"), (lines, "scan population"), (p0, "P0 table")):
        leaked = [c for c in lp1.FORBIDDEN_INPUT_COLUMNS if c in frame.columns]
        if leaked:
            raise PhaseError(f"the {what} carries truth columns: {leaked}")
    _write_parquet_once(SITE_LATTICE, lattice)
    _write_parquet_once(SCAN_POPULATION, lines)
    _write_parquet_once(P0_PROPOSALS, p0)
    _write_json_once(
        LATTICE_INVENTORY,
        {
            **_envelope("lattice_inventory"),
            "by_environment": inventory,
            "pages": int(pool["page_id"].nunique()),
            "ocr_lines": len(lines),
            "lattice_sites": len(lattice),
            "p0_proposals": len(p0),
            "p0_budget_by_environment": {r["environment"]: r["p0_proposals"] for r in inventory},
            "p0_reproduces_xc1_on_evaluation_pages": reproduction,
            "lattice_version": lp1.LATTICE_VERSION,
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"lattice: {len(lines)} lines, {len(lattice)} lattice sites, {len(p0)} P0 proposals "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def build_contexts_for_lines(lines: pd.DataFrame, crop_root: Path) -> pd.DataFrame:
    """LP1's `build_line_contexts`, one environment at a time into its own crop directory."""
    frames = [
        lp1.build_line_contexts(block.reset_index(drop=True), crop_root / _slug(str(name)))
        for name, block in lines.groupby("environment", sort=True)
    ]
    return pd.concat(frames, ignore_index=True)


def run_lines() -> int:
    """The proposer's prompt and line crop for every scanned line. Reads OCR, boxes, pixels."""
    started = time.monotonic()
    _require(SCAN_POPULATION, "lattice")
    _forbid(LINE_CONTEXTS)
    if _labels_exist():
        raise PhaseError("a PF1 label exists; line contexts precede every one")
    contexts = build_contexts_for_lines(pd.read_parquet(SCAN_POPULATION), LINE_CROP_DIR)
    _write_parquet_once(LINE_CONTEXTS, contexts)
    print(
        f"lines: {len(contexts)} line contexts, {int(contexts['crop_path'].notna().sum())} with a "
        f"crop ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the generation plan


def shards() -> list[tuple[str, str, int]]:
    """Every (arm, environment, chunk) shard the plan names, in execution order."""
    pool = load_pool()
    pairs = sorted(
        {(str(e), int(c)) for e, c in zip(pool["environment"], pool["chunk"], strict=True)}
    )
    return [(arm, name, chunk) for arm in GENERATION_ARMS for name, chunk in pairs]


def shard_path(arm: str, name: str, chunk: int) -> Path:
    return RAW_CACHE / f"{_slug(name)}.{arm}.c{chunk:02d}.parquet"


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name.replace(".parquet", ".cost.json"))


def shard_complete(path: Path) -> bool:
    sidecar = _sidecar(path)
    if not (path.is_file() and sidecar.is_file()):
        return False
    return bool(cc_read_json(sidecar)["raw_output_sha256"] == file_sha256(path))


def chunk_pages(name: str, chunk: int) -> set[str]:
    pool = load_pool()
    block = pool[(pool["environment"] == name) & (pool["chunk"] == chunk)]
    return set(block["document_id"].astype(str))


def run_plan() -> int:
    """The generation plan: shards, budgets, routing and the compute estimate, before any decode."""
    started = time.monotonic()
    for path, phase in ((LINE_CONTEXTS, "lines"), (LATTICE_INVENTORY, "lattice")):
        _require(path, phase)
    _forbid(GENERATION_PLAN)
    if any(RAW_CACHE.glob("*.parquet")):
        raise PhaseError("an answer exists; the plan precedes every decode")
    inventory = cc_read_json(LATTICE_INVENTORY)
    contexts = pd.read_parquet(LINE_CONTEXTS)
    proposer_requests = int(contexts["crop_path"].notna().sum())
    # The corrector's demand is the image proposer's matched budget, which equals P0's count per
    # environment; the upper bound is every image proposal being new.
    corrector_bound = int(inventory["p0_proposals"])
    hours = {
        PROPOSER: proposer_requests / LP1_LINES_PER_SECOND / 3600.0,
        CORRECTOR: corrector_bound / HY1_REQUESTS_PER_SECOND / 3600.0,
    }
    _write_json_once(
        GENERATION_PLAN,
        {
            **_envelope("generation_plan"),
            "arms": {
                PROPOSER: {
                    "upstream_method": lp1.L2,
                    "unit": "one request per OCR line with a line crop",
                    "requests": proposer_requests,
                    "lines_without_a_crop": int(contexts["crop_path"].isna().sum()),
                },
                CORRECTOR: {
                    "upstream_method": f"{C1} under {hy1.CONDITION_OF_CORRECTOR[C1]}",
                    "unit": "one request per routed site with a crop",
                    "sites": "the image proposer's matched-budget sites (image-only and overlap)",
                    "upper_bound_requests": corrector_bound,
                },
            },
            "matched_budget": {
                "rule": "LP1's: per environment, P0's own proposal count on the same pages",
                "column": BUDGET_COLUMN,
                "p0_budget_by_environment": inventory["p0_budget_by_environment"],
            },
            "routing": {
                "arm": PRIMARY_ARM_OF_HY1,
                "routes": {s: list(c) for s, c in hy1.ROUTING[PRIMARY_ARM_OF_HY1].items()},
            },
            "shards": [
                {"arm": a, "environment": n, "chunk": c, "pages": sorted(chunk_pages(n, c))}
                for a, n, c in shards()
            ],
            "batching": (
                "within a shard, requests sorted by prompt length then request id, batch "
                f"size {gen1.MODEL_SPECS[MODEL]['batch_size']}: the upstream stages' rule. A bf16 "
                "decode is reproducible for a fixed shard, not invariant to re-batching"
            ),
            "estimated_hours": {
                **{k: round(v, 2) for k, v in hours.items()},
                "total": round(sum(hours.values()), 2),
            },
            "throughput_source": {
                PROPOSER: "LP1's L2 image-arm sidecars",
                CORRECTOR: "HY1's image-corrector sidecars",
            },
            "resumable": "a shard is written atomically with its sidecar; a rerun skips it",
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"plan: {proposer_requests} proposer requests, <= {corrector_bound} corrector requests, "
        f"about {sum(hours.values()):.1f} h ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ inference


def proposer_request_frame(lines: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    """LP1's L2 request recipe: one request per line with a crop, the image template, the crop."""
    joined = lines[["environment", "line_uid", "document_id", "lattice_sites"]].merge(
        contexts, on="line_uid", validate="one_to_one"
    )
    joined = joined[joined["crop_path"].notna()]
    return pd.DataFrame(
        {
            "environment": joined["environment"].to_numpy(),
            "method": lp1.L2,
            "request_id": (joined["line_uid"] + f"|{lp1.L2}").to_numpy(),
            "line_uid": joined["line_uid"].to_numpy(),
            "document_id": joined["document_id"].to_numpy(),
            "lattice_sites": joined["lattice_sites"].to_numpy(dtype=np.int64),
            "user_prompt": joined["image_prompt"].to_numpy(),
            "image_path": joined["crop_path"].to_numpy(),
            "image_sha256": joined["crop_sha256"].to_numpy(),
        }
    ).reset_index(drop=True)


def proposer_requests(name: str, pages: set[str]) -> pd.DataFrame:
    """One proposer shard's requests, rebuilt from the frozen GT-blind tables alone."""
    lines = pd.read_parquet(SCAN_POPULATION)
    lines = lines[(lines["environment"] == name) & lines["document_id"].astype(str).isin(pages)]
    return proposer_request_frame(lines, pd.read_parquet(LINE_CONTEXTS))


def prompt_identity(requests: pd.DataFrame, system: str) -> pd.Series:
    """Each request's prompt hash, exactly as the upstream runners record it in a raw shard."""
    return pd.Series(
        [
            canonical_hash({"system": system, "user": str(u), "image": i})
            for u, i in zip(requests["user_prompt"], requests["image_sha256"], strict=True)
        ],
        index=requests["request_id"].astype(str).to_numpy(),
    )


def prompts_match(requests: pd.DataFrame, system: str, raw: pd.DataFrame) -> dict[str, Any]:
    """Rebuilt request prompts against the prompt hashes a frozen raw shard recorded."""
    rebuilt = prompt_identity(requests, system)
    frozen = raw.set_index(raw["request_id"].astype(str))["prompt_sha256"].astype(str)
    shared = sorted(set(rebuilt.index) & set(frozen.index))
    equal = int(sum(rebuilt[k] == frozen[k] for k in shared))
    return {
        "rebuilt_requests": len(rebuilt),
        "frozen_answers": len(frozen),
        "shared": len(shared),
        "identical_prompts": equal,
        "equal": bool(len(shared) == len(frozen) == len(rebuilt) and equal == len(shared)),
    }


def corrector_requests(name: str, pages: set[str]) -> pd.DataFrame:
    """One corrector shard's requests: HY1's recipe over the routed C1 sites of the pages."""
    contexts = pd.read_parquet(CORRECTION_CONTEXTS)
    strata = pd.read_parquet(PROPOSAL_STRATA)
    demanded = routed_to(strata, C1)
    frame = contexts[
        (contexts["environment"] == name)
        & contexts["site_id"].isin(demanded)
        & contexts["document_id"].astype(str).isin(pages)
        & contexts["crop_path"].notna()
    ].reset_index(drop=True)
    return hy1_style_requests(frame, strata, hy1.NEW_IMAGE)


def routed_to(strata: pd.DataFrame, corrector: str) -> set[str]:
    """The sites H5 -- U5's arm -- routes to one corrector, from the stratum alone."""
    routes = hy1.ROUTING[PRIMARY_ARM_OF_HY1]
    chosen = {stratum for stratum, correctors in routes.items() if corrector in correctors}
    return set(strata[strata["stratum"].isin(chosen)]["site_id"].astype(str))


def hy1_style_requests(frame: pd.DataFrame, strata: pd.DataFrame, arm: str) -> pd.DataFrame:
    """HY1's `requests_for` body: the same template, prompt, crop and request id recipe."""
    by_site = strata.set_index("site_id")
    digest = frame["site_id"].astype(str).str.split("|", n=1).str[1]
    template = hy1.CONDITION_OF_CORRECTOR[C1]
    return pd.DataFrame(
        {
            "environment": frame["environment"].to_numpy(),
            "method": arm,
            "condition": template,
            "request_id": (frame["environment"] + f"|{arm}|" + digest).to_numpy(),
            "site_id": frame["site_id"].to_numpy(),
            "anchor_kind": by_site.loc[frame["site_id"], "anchor_kind"].to_numpy(),
            "region": by_site.loc[frame["site_id"], "region"].to_numpy(),
            "user_prompt": [gen1.user_message(template, str(t)) for t in frame["e1_text"]],
            "image_path": frame["crop_path"].to_numpy(),
            "image_sha256": frame["crop_sha256"].to_numpy(),
        }
    ).reset_index(drop=True)


def _selected_arms() -> list[str]:
    chosen = [a for a in os.environ.get("PF1_ARMS", "").split(",") if a]
    unknown = [a for a in chosen if a not in GENERATION_ARMS]
    if unknown:
        raise PhaseError(f"unknown PF1_ARMS {unknown}; choose from {list(GENERATION_ARMS)}")
    return chosen or list(GENERATION_ARMS)


def _decode(runner: Any, arm: str, requests: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    batch = int(gen1.MODEL_SPECS[MODEL]["batch_size"])
    if arm == PROPOSER:
        return lp1.run_line_requests(runner, requests, batch, print)
    return gen1.run_requests(runner, requests, batch, print)


def run_generate() -> int:
    """GT-blind inference, one (arm, environment, chunk) shard at a time, each written once.

    Each shard is written atomically with its provenance sidecar the moment it exists, so an
    interrupted run keeps every finished shard and a rerun skips each one whose bytes match its
    sidecar hash. The corrector's shards need the parsed proposals and the routing first.
    """
    started = time.monotonic()
    _require(GENERATION_PLAN, "plan")
    if _labels_exist():
        raise PhaseError("labels exist; generation must precede every label")
    arms = _selected_arms()
    if CORRECTOR in arms:
        for path, phase in ((PROPOSAL_STRATA, "strata"), (CORRECTION_CONTEXTS, "contexts")):
            _require(path, phase)
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    hashes = cc_read_json(FROZEN_CONFIGURATION)["generation_hashes"]
    work = [
        (arm, name, chunk)
        for arm, name, chunk in shards()
        if arm in arms and not shard_complete(shard_path(arm, name, chunk))
    ]
    if not work:
        print("generate: every selected shard complete")
        return 0
    import transformers

    transformers.utils.logging.set_verbosity_error()
    device = gen1._resolve_device()
    began = time.monotonic()
    runner = gen1._runner(MODEL, device)
    spec = gen1.MODEL_SPECS[MODEL]
    print(f"  loaded {MODEL} on {device} ({time.monotonic() - began:.0f}s); {len(work)} shards")
    for position, (arm, name, chunk) in enumerate(work, start=1):
        path = shard_path(arm, name, chunk)
        pages = chunk_pages(name, chunk)
        requests = (
            proposer_requests(name, pages) if arm == PROPOSER else corrector_requests(name, pages)
        )
        if requests.empty:
            raw = pd.DataFrame(columns=["request_id", "output", "prompt_sha256"])
            cost: dict[str, Any] = {"requests": 0, "wall_clock_seconds": 0.0}
        else:
            raw, cost = _decode(runner, arm, requests)
        temporary = path.with_name(path.name + ".partial")
        raw.to_parquet(temporary, index=False)
        temporary.rename(path)
        _sidecar(path).unlink(missing_ok=True)
        _write_json_once(
            _sidecar(path),
            {
                **_envelope("generation_shard"),
                "arm": arm,
                "upstream_method": UPSTREAM_ARM[arm],
                "environment": name,
                "chunk": chunk,
                "pages": sorted(pages),
                "model": MODEL,
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                **cost,
                "generation_hashes": hashes,
                "allocator_environment": gen1._allocator_environment(),
                "request_ids_sha256": canonical_hash(sorted(requests["request_id"].astype(str)))
                if len(requests)
                else None,
                "request_prompts_hash": canonical_hash(sorted(raw["prompt_sha256"].astype(str)))
                if len(raw)
                else None,
                "device": device,
                "platform": platform.platform(),
                "raw_output": _relative(path),
                "raw_output_sha256": file_sha256(path),
                "clock": "time.monotonic, which does not advance while the machine sleeps",
                "uses_ground_truth": False,
            },
        )
        rate = cost.get("requests_per_second")
        print(
            f"    [{position}/{len(work)}] {arm}/{name}/c{chunk:02d}: {cost['requests']} requests "
            f"in {cost['wall_clock_seconds']:.0f}s" + (f" ({rate:.3f}/s)" if rate else ""),
            flush=True,
        )
    del runner
    gen1._release_device_cache(device)
    print(f"generate: done in {time.monotonic() - started:.0f}s")
    return 0


# ------------------------------------------------------------------ proposals, strata, contexts


def parse_answers(
    raw: pd.DataFrame, lines: pd.DataFrame, p0: pd.DataFrame, budgets: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LP1's `run_parse` body for its two primary sources: the frozen P0 and the image proposer.

    Every answer goes through LP1's own `parse_line`; the matched budget is LP1's `_budget_columns`
    under LP1's ranking keys. Nothing here reads a label.
    """
    by_line = lines.set_index("line_uid")
    rows: list[dict[str, Any]] = []
    census_rows: list[dict[str, Any]] = []
    for answer in raw.itertuples(index=False):
        line = by_line.loc[str(answer.line_uid)]
        proposals, census, status = lp1.parse_line(
            str(answer.output),
            bool(answer.finished),
            list(line["local_ids"]),
            list(line["site_ids"]),
            int(line["tokens"]),
        )
        census_rows.append(
            {
                "method": lp1.L2,
                "environment": str(line["environment"]),
                "line_uid": str(answer.line_uid),
                "status": status,
                "proposals": len(proposals),
                **census,
            }
        )
        rows.extend(
            {
                "method": lp1.L2,
                "environment": str(line["environment"]),
                "line_uid": str(answer.line_uid),
                "document_id": str(line["document_id"]),
                "document_rank": int(line["document_rank"]),
                "line_index": int(line["line_index"]),
                **proposal,
            }
            for proposal in proposals
        )
    p0_rows = pd.DataFrame(
        {
            "method": lp1.L0,
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
    for arm, block in pd.concat([p0_rows, pd.DataFrame(rows)], ignore_index=True).groupby(
        "method", sort=False
    ):
        block = block.reset_index(drop=True).assign(
            neg_suspicion=lambda f: -f["suspicion"].astype(float),
            hash_order=lambda f: [lp1.order_key("rank", s) for s in f["site_id"].astype(str)],
        )
        key = lp1.P0_RANK_KEY if arm == lp1.L0 else lp1.RANK_KEY
        frames.append(lp1._budget_columns(block, budgets, key))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(census_rows)


def verified_shards(arm: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Every shard of one arm, verified against its sidecar hash, concatenated in plan order."""
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    for shard_arm, name, chunk in shards():
        if shard_arm != arm:
            continue
        path = shard_path(arm, name, chunk)
        if not shard_complete(path):
            raise PhaseError(f"{_relative(path)} is missing or does not match its sidecar")
        sidecar = cc_read_json(_sidecar(path))
        manifest.append(
            {
                "arm": arm,
                "environment": name,
                "chunk": chunk,
                "raw_output": _relative(path),
                "raw_output_sha256": sidecar["raw_output_sha256"],
                "requests": int(sidecar["requests"]),
            }
        )
        frame = pd.read_parquet(path)
        if len(frame):
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), manifest


def run_parse() -> int:
    """Raw proposer answers -> hashes -> lattice proposals -> the matched budget. GT-blind."""
    started = time.monotonic()
    for path in (PARSED_PROPOSALS, PROPOSAL_COMPLIANCE):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("labels exist; every proposal must be frozen before the first label")
    raw, manifest = verified_shards(PROPOSER)
    lines = pd.read_parquet(SCAN_POPULATION)
    budgets = cc_read_json(LATTICE_INVENTORY)["p0_budget_by_environment"]
    parsed, census = parse_answers(raw, lines, pd.read_parquet(P0_PROPOSALS), budgets)
    leaked = [c for c in lp1.FORBIDDEN_INPUT_COLUMNS if c in parsed.columns]
    if leaked:
        raise PhaseError(f"the proposal table carries truth columns: {leaked}")
    _write_parquet_once(PARSED_PROPOSALS, parsed)
    image = parsed[parsed["method"] == lp1.L2]
    _write_json_once(
        PROPOSAL_COMPLIANCE,
        {
            **_envelope("proposal_compliance"),
            "raw_shards": manifest,
            "answers": len(raw),
            "status": census["status"].value_counts().sort_index().to_dict(),
            "census": {
                c: int(census[c].sum())
                for c in census.columns
                if c not in ("method", "environment", "line_uid", "status")
            },
            "image_proposals": len(image),
            "image_in_budget": int(image[BUDGET_COLUMN].sum()),
            "p0_proposals": int((parsed["method"] == lp1.L0).sum()),
            "budget_by_environment": budgets,
            "in_budget_by_environment": image.groupby("environment")[BUDGET_COLUMN]
            .sum()
            .astype(int)
            .to_dict(),
            "parsed_proposals_sha256": file_sha256(PARSED_PROPOSALS),
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"parse: {len(image)} image proposals, {int(image[BUDGET_COLUMN].sum())} in budget "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


STRATA_COLUMNS = (
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
)


def build_strata_frame(
    lattice: pd.DataFrame, p0: pd.DataFrame, parsed: pd.DataFrame
) -> pd.DataFrame:
    """HY1's `build_strata` over P0 and the image proposer's matched-budget selection.

    The source rule is HY1's own `stratum_of`. The text-proposal control is not part of the U5
    population and is not run, so its flags are false and its score is missing.
    """
    image = parsed[(parsed["method"] == lp1.L2) & parsed[BUDGET_COLUMN]]
    p0_sites = set(p0["lattice_site_id"].astype(str))
    image_sites = set(image["site_id"].astype(str))
    frame = pd.DataFrame(
        [
            {"site_id": site, "stratum": hy1.stratum_of(site, p0_sites, image_sites)}
            for site in sorted(p0_sites | image_sites)
        ]
    )
    joined = frame.merge(
        lattice[list(STRATA_COLUMNS)], on="site_id", how="left", validate="one_to_one"
    )
    if joined["environment"].isna().any():
        raise PhaseError("a routed site is absent from the lattice")
    joined["in_p0"] = joined["site_id"].isin(p0_sites)
    joined["in_image"] = joined["site_id"].isin(image_sites)
    joined["in_text"] = False
    joined["ht_incremental"] = False
    p0_rank = p0.set_index("lattice_site_id")["suspicion_score"].astype(float)
    image_rank = image.set_index("site_id")["suspicion"].astype(float)
    ranks = pd.concat(
        [
            image.set_index("site_id")[["hash_order", "document_rank"]],
            p0.set_index("lattice_site_id")[["document_rank"]].assign(hash_order=None),
        ]
    )
    ranks = ranks.groupby(level=0).first()
    joined["p0_suspicion"] = joined["site_id"].map(p0_rank)
    joined["image_suspicion"] = joined["site_id"].map(image_rank)
    joined["text_suspicion"] = np.nan
    joined["hash_order"] = joined["site_id"].map(ranks["hash_order"])
    joined["document_rank"] = joined["site_id"].map(ranks["document_rank"])
    if joined["site_id"].duplicated().any():
        raise PhaseError("a site carries more than one stratum")
    return joined.sort_values(["environment", "site_id"]).reset_index(drop=True)


def run_strata() -> int:
    """The proposal source strata and the frozen H5 routing, before any label."""
    started = time.monotonic()
    _require(PARSED_PROPOSALS, "parse")
    for path in (PROPOSAL_STRATA, ROUTING_REGISTRY):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("a PF1 label exists; the strata are frozen before any of them")
    strata = build_strata_frame(
        pd.read_parquet(SITE_LATTICE),
        pd.read_parquet(P0_PROPOSALS),
        pd.read_parquet(PARSED_PROPOSALS),
    )
    _write_parquet_once(PROPOSAL_STRATA, strata)
    counts = strata["stratum"].value_counts().to_dict()
    _write_json_once(
        ROUTING_REGISTRY,
        {
            **_envelope("routing_registry"),
            "arm": PRIMARY_ARM_OF_HY1,
            "routes": {s: list(c) for s, c in hy1.ROUTING[PRIMARY_ARM_OF_HY1].items()},
            "sites_by_stratum": {s: int(counts.get(s, 0)) for s in hy1.STRATA},
            "corrector_demand": {C0: len(routed_to(strata, C0)), C1: len(routed_to(strata, C1))},
            "not_decoded": (
                "HY1's other arms also send P0-only sites to the image corrector; U5 never keeps "
                "those candidates, so they are not decoded here"
            ),
            "routing_reads": ["proposal source stratum"],
            "frozen_before_any_label": True,
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"strata: {len(strata)} routed sites {counts} ({time.monotonic() - started:.0f}s)")
    return 0


def build_correction_contexts(strata: pd.DataFrame, crop_root: Path) -> pd.DataFrame:
    """XR1's `build_natural_contexts` for every routed site, one environment at a time."""
    frames: list[pd.DataFrame] = []
    for name, sites in strata.groupby("environment", sort=True):
        sites = sites.reset_index(drop=True)
        pages = gen1._page_index(_spec(str(name)))
        contexts = xr1.build_natural_contexts(sites, pages, crop_root / _slug(str(name)))
        contexts.insert(0, "environment", str(name))
        contexts.insert(2, "document_id", sites["document_id"].astype(str).to_numpy())
        frames.append(contexts)
    return pd.concat(frames, ignore_index=True)


def run_contexts() -> int:
    """The GT-blind correction context and crop for every routed site (HY1's section 10)."""
    started = time.monotonic()
    _require(PROPOSAL_STRATA, "strata")
    _forbid(CORRECTION_CONTEXTS)
    if _labels_exist():
        raise PhaseError("a PF1 label exists; contexts are frozen before every one")
    contexts = build_correction_contexts(pd.read_parquet(PROPOSAL_STRATA), SITE_CROP_DIR)
    _write_parquet_once(CORRECTION_CONTEXTS, contexts)
    print(
        f"contexts: {len(contexts)} routed sites, "
        f"{int((contexts['crop_level'] == xr1.CROP_NONE).sum())} without a crop "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ ground truth enters

C1_CONDITION = hy1.CONDITION_OF_CORRECTOR[C1]


def c0_frame(
    spec: dict[str, Any], documents: set[str], routed: set[str], rows: pd.DataFrame
) -> pd.DataFrame:
    """XR1's `_g0_frame` on pages of either SGV14 role, restricted to the routed P0 sites.

    SGV-CG1's frozen labelled rows and SGV-XC1's cached text of the same candidate stream cover
    every SGV14 page. XR1 kept only EVALUATION pages because its sample held nothing else; the
    columns, ids and labels are otherwise read exactly as XR1 reads them.
    """
    name = spec["environment"]
    rows = rows[
        (rows["environment"] == name) & rows["document_id"].astype(str).isin(documents)
    ].reset_index(drop=True)
    cache = pd.read_parquet(xc1.CACHE / f"{_slug(name)}.m0_candidates.parquet").set_index(
        "candidate_id"
    )
    missing = sorted(set(rows["candidate_id"].astype(str)) - set(cache.index.astype(str)))
    if missing:
        raise PhaseError(f"{name}: {len(missing)} current-generator candidates have no text")
    text = cache.loc[rows["candidate_id"].astype(str)]
    anchor = rows["anchor_kind"].astype(str)
    frame = pd.DataFrame(
        {
            "environment": name,
            "corpus": spec["corpus"],
            "base_engine": spec["base_engine"],
            "method": C0,
            "condition": xr1.G0_CONDITION,
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
            "candidate_source": C0,
            "outcome": rows["outcome"].astype(str).to_numpy(),
            "is_harmful": rows["is_harmful"].to_numpy(dtype=bool),
            "beneficial": rows["beneficial"].to_numpy(dtype=bool),
            "exact": (rows["outcome"].astype(str) == xr1.OUTCOME_EXACT).to_numpy(dtype=bool),
            "d_before": rows["d_before"].to_numpy(dtype=np.int64),
            "d_after": rows["d_after"].to_numpy(dtype=np.int64),
        },
        columns=list(xr1.METHOD_CANDIDATE_COLUMNS),
    )
    return frame[frame["site_id"].astype(str).isin(routed)].reset_index(drop=True)


def normalize_environment(
    spec: dict[str, Any],
    sites: pd.DataFrame,
    raw: pd.DataFrame,
    p0: pd.DataFrame,
    c0_rows: pd.DataFrame,
) -> tuple[list[pd.DataFrame], pd.DataFrame, list[dict[str, Any]]]:
    """HY1's `run_normalize` body for one environment under H5: the image corrector's answers
    through GEN1's `_expand`, XC1's `_neural_edits` and SGV14's `_labels`; the current
    generator's frozen labelled candidates at the routed P0 sites; one identity probe per site.
    """
    from ocr_risk.canonical import rebuild_stream

    name = spec["environment"]
    documents = set(sites["document_id"].astype(str))
    bundles = s14._document_bundles(spec["corpus"])
    roles = s14.partition_of(spec["corpus"], sorted(bundles))
    spans, _record = s14._canonical_spans(spec, bundles)
    chosen = {pair: group for pair, group in spans.items() if pair[0] in documents}
    streams = {pair: rebuild_stream(group) for pair, group in chosen.items() if group}
    wanted = routed_to(sites, C1)
    census: list[dict[str, Any]] = []
    proposals = pd.DataFrame()
    if wanted and len(raw):
        answered = sites[sites["site_id"].astype(str).isin(wanted)]
        requests = hy1._corrector_requests(answered, C1, streams)
        raw = raw[raw["request_id"].isin(set(requests["request_id"]))].reset_index(drop=True)
        requests = requests[requests["request_id"].isin(set(raw["request_id"]))].reset_index(
            drop=True
        )
        expanded, census = gen1._expand(raw, requests, C1, C1_CONDITION)
        for row in census:
            row["environment"] = name
            row["corrector"] = C1
        _s, proposals, _shard = xc1._neural_edits(name, C1, expanded, requests, {}, None)
    label_sites = sites[list(hy1.LABEL_SITE_COLUMNS)].assign(
        site_id=sites["site_id"].astype(str).str.split("|", n=1).str[1]
    )
    probes = hy1._identity_probes(sites)
    inputs = [probes]
    if not proposals.empty:
        inputs.append(proposals[hy1.LABEL_INPUT_COLUMNS])
    labels, _diagnostics = s14._labels(
        spec["corpus"], bundles, chosen, streams, label_sites, pd.concat(inputs, ignore_index=True)
    )
    frames: list[pd.DataFrame] = []
    if not proposals.empty:
        frames.append(xc1._method_frame(spec, C1, C1_CONDITION, proposals, labels, roles))
    by_lattice = dict(
        zip(p0["lattice_site_id"].astype(str), p0["site_id"].astype(str), strict=True)
    )
    routed_p0 = {by_lattice[s] for s in routed_to(sites, C0) if s in by_lattice}
    frames.append(c0_frame(spec, documents, routed_p0, c0_rows))
    probe_labels = labels.set_index("candidate_id").loc[probes["candidate_id"]]
    labelable = probe_labels["labelable"].to_numpy(dtype=bool)
    d_before = probe_labels["d_before"].to_numpy(dtype=np.int64)
    truth = pd.DataFrame(
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
                np.where(d_before > 0, hy1.SITE_ERROR, hy1.SITE_CLEAN),
            ),
        }
    )
    return frames, truth, census


def attribute(candidates: pd.DataFrame, strata: pd.DataFrame, p0: pd.DataFrame) -> pd.DataFrame:
    """Every candidate's lattice site, proposal stratum and corrector, as HY1 attributes them."""
    lattice_of_p0 = dict(
        zip(p0["site_id"].astype(str), p0["lattice_site_id"].astype(str), strict=True)
    )
    candidates = candidates.copy()
    candidates["lattice_site_id"] = np.where(
        candidates["method"] == C0,
        candidates["site_id"].astype(str).map(lattice_of_p0),
        candidates["site_id"].astype(str),
    )
    candidates["proposal_stratum"] = candidates["lattice_site_id"].map(
        strata.set_index("site_id")["stratum"]
    )
    candidates["corrector_source"] = candidates["method"]
    if candidates["proposal_stratum"].isna().any():
        raise PhaseError("a candidate carries no proposal stratum")
    return candidates


def normalize_all(
    strata: pd.DataFrame, raw_c1: pd.DataFrame, p0: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Every environment's labelled candidates and site truth, attributed."""
    c0_rows = pd.read_parquet(cg1.CANDIDATE_ROWS)
    candidate_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    census: list[dict[str, Any]] = []
    for name, sites in strata.groupby("environment", sort=True):
        began = time.monotonic()
        spec = _spec(str(name))
        raw = raw_c1[raw_c1["request_id"].astype(str).str.startswith(f"{name}|")]
        frames, truth, rows = normalize_environment(
            spec, sites.reset_index(drop=True), raw.reset_index(drop=True), p0, c0_rows
        )
        candidate_frames.extend(frames)
        truth_frames.append(truth)
        census.extend(rows)
        print(
            f"  {name}: {len(sites)} routed sites, {sum(len(f) for f in frames)} candidates "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    candidates = attribute(pd.concat(candidate_frames, ignore_index=True), strata, p0)
    return candidates, pd.concat(truth_frames, ignore_index=True), census


def rekeyed_corrector_answers() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """PF1's corrector answers, re-keyed from the generation arm to the corrector (HY1's rule)."""
    raw, manifest = verified_shards(CORRECTOR)
    parts = raw["request_id"].astype(str).str.split("|", expand=True)
    raw = raw.copy()
    raw["request_id"] = (parts[0] + f"|{C1}|" + parts[2]).to_numpy()
    raw["method"] = C1
    return raw, manifest


def run_normalize() -> int:
    """Raw answers -> atomic edits -> SGV14 labels -> the frozen PF1 candidate table."""
    started = time.monotonic()
    for path, phase in ((DESIGN_RECORD, "preregister"), (PROPOSAL_STRATA, "strata")):
        _require(path, phase)
    for path in (CANDIDATES, SITE_TRUTH, CANDIDATE_INVENTORY):
        _forbid(path)
    raw, manifest = rekeyed_corrector_answers()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    candidates, truth, census = normalize_all(strata, raw, pd.read_parquet(P0_PROPOSALS))
    _write_parquet_once(CANDIDATES, candidates)
    _write_parquet_once(SITE_TRUTH, truth)
    parse = pd.DataFrame(census)
    _write_json_once(
        CANDIDATE_INVENTORY,
        {
            **_analysis_envelope("candidate_inventory"),
            "raw_shards": manifest,
            "candidates": len(candidates),
            "by_corrector": candidates["corrector_source"].value_counts().to_dict(),
            "by_stratum": candidates["proposal_stratum"].value_counts().to_dict(),
            "sites_with_a_candidate": int(candidates["lattice_site_id"].nunique()),
            "corrector_answers": len(raw),
            "parse_status": parse["parse_status"].value_counts().to_dict() if len(parse) else {},
            "site_classes": truth["site_class"].value_counts().to_dict(),
        },
    )
    print(
        f"normalize: {len(candidates)} candidates over {candidates['lattice_site_id'].nunique()} "
        f"sites ({time.monotonic() - started:.0f}s)"
    )
    return 0


def link_environment(
    spec: dict[str, Any], lattice: pd.DataFrame, p0: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LP1's `run_link` body for one environment: its error sites and every lattice site's links."""
    from ocr_risk.schemas.enums import AnchorKind

    name = spec["environment"]
    documents = set(lattice["document_id"].astype(str))
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
        f"{name}|{site.site_id}": (str(site.ocr_text), str(site.gt_text)) for _p, site in objects
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
    sites = lattice.assign(site_id=lp1._bare(lattice["site_id"]))[
        ["site_id", "document_id", "engine_id", "anchor_kind", "anchor_ref"]
    ]
    links = cg1._site_links(name, sites, indexes, by_span, by_alignment, AnchorKind)
    links = links.assign(is_error=links["align_site_id"].astype(str).isin(error_ids))
    own_links = cg1._site_links(
        name,
        p0.assign(site_id=lp1._bare(p0["site_id"])),
        indexes,
        by_span,
        by_alignment,
        AnchorKind,
    )
    reached = set(own_links["align_site_id"].astype(str)) & error_ids
    representable = set(links[links["is_error"]]["align_site_id"].astype(str))
    pages = gen1._page_index(spec)
    extra = pd.concat(
        [lp1.cross_line_gaps(pages[pair], pair[0], pair[1]) for pair in sorted(chosen)],
        ignore_index=True,
    )
    extra_links = cg1._site_links(name, extra, indexes, by_span, by_alignment, AnchorKind)
    with_cross = representable | (set(extra_links["align_site_id"].astype(str)) & error_ids)
    table["representable"] = table["align_site_id"].isin(representable)
    table["representable_with_cross_line_gaps"] = table["align_site_id"].isin(with_cross)
    table["reached_by_p0"] = table["align_site_id"].isin(reached)
    return table, links


def link_all(lattice: pd.DataFrame, p0: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    links: list[pd.DataFrame] = []
    for name, block in lattice.groupby("environment", sort=True):
        table, link = link_environment(
            _spec(str(name)), block.reset_index(drop=True), p0[p0["environment"] == name]
        )
        tables.append(table)
        links.append(link)
    return pd.concat(tables, ignore_index=True), pd.concat(links, ignore_index=True)


def run_link() -> int:
    """Ground truth enters: every evaluable OCR error site and every lattice site's links."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (ERROR_SITES, PROPOSAL_ERROR_LINKS):
        _forbid(path)
    errors, links = link_all(pd.read_parquet(SITE_LATTICE), pd.read_parquet(P0_PROPOSALS))
    _write_parquet_once(ERROR_SITES, errors)
    _write_parquet_once(PROPOSAL_ERROR_LINKS, links)
    print(
        f"link: {len(errors)} error sites, {int(errors['representable'].sum())} representable, "
        f"{len(links)} links ({time.monotonic() - started:.0f}s)"
    )
    return 0


POPULATION_NAME = "pf1_new_pages"


def build_population_frame(
    candidates_path: Path,
    strata_path: Path,
    truth_path: Path,
    errors_path: Path,
    links_path: Path,
) -> pd.DataFrame:
    """RL1's U5 construction, run by RL1's own `build_population` on PF1's tables."""
    candidates = hy1._attach_membership(
        pd.read_parquet(candidates_path), pd.read_parquet(strata_path)
    )
    with rebound(
        [
            (hy1, "SITE_TRUTH", truth_path),
            (hy1, "ERROR_SITES", errors_path),
            (lp1, "PROPOSAL_ERROR_LINKS", links_path),
        ]
    ):
        return rl1.build_population(rl1.U5, candidates)


def with_ranking_columns(
    frame: pd.DataFrame, folds: dict[str, int], blocks: dict[int, str]
) -> pd.DataFrame:
    """RK1's `attach_blocks` with the page's own fold and block, and RK1's graded relevance."""
    mapped = frame["document_id"].astype(str).map(folds)
    if mapped.isna().any():
        raise PhaseError("a candidate's page carries no fold")
    frame = frame.copy()
    frame["fold"] = mapped.to_numpy(dtype=np.int64)
    frame["block"] = [blocks[int(f)] for f in frame["fold"]]
    frame["site_key"] = frame["site_group"]
    frame["grade"] = rk1.relevance(frame)
    return frame


PF1_BLOCK = {fold: f"pf1_{split_of(fold)}" for fold in range(FOLDS)}


def run_population() -> int:
    """PF1's U5-style population with RK1's grades, and each page's annotation status."""
    started = time.monotonic()
    for path, phase in ((CANDIDATES, "normalize"), (ERROR_SITES, "link")):
        _require(path, phase)
    for path in (POPULATION, POPULATION_INVENTORY, PAGE_ANNOTATION):
        _forbid(path)
    pool = load_pool()
    frame = build_population_frame(
        CANDIDATES, PROPOSAL_STRATA, SITE_TRUTH, ERROR_SITES, PROPOSAL_ERROR_LINKS
    ).assign(population=POPULATION_NAME)
    folds = dict(zip(pool["document_id"].astype(str), pool["fold"].astype(int), strict=True))
    frame = with_ranking_columns(frame, folds, PF1_BLOCK)
    frame = frame.sort_values(["environment", "candidate_id"], kind="stable").reset_index(drop=True)
    _write_parquet_once(POPULATION, frame)
    errors = pd.read_parquet(ERROR_SITES)
    per_page = []
    for row in pool.itertuples(index=False):
        mine = frame[frame["document_id"] == row.document_id]
        per_page.append(
            {
                "page_id": row.page_id,
                "environment": row.environment,
                "split": row.split,
                "candidates": len(mine),
                "sites": int(mine["site_group"].nunique()),
                "c0_candidates": int(mine["corrector_sources"].str.contains(C0).sum()),
                "c1_candidates": int(mine["corrector_sources"].str.contains(C1).sum()),
                "evaluable_error_sites": int((errors["document_id"] == row.document_id).sum()),
                "annotation_status": "labelled: corpus transcription attached by SGV14's labeller",
            }
        )
    annotation = pd.DataFrame(per_page)
    _write_parquet_once(PAGE_ANNOTATION, annotation)
    _write_json_once(
        POPULATION_INVENTORY,
        {
            **_analysis_envelope("population_inventory"),
            "candidates": len(frame),
            "sites": int(frame["site_group"].nunique()),
            "pages": int(frame["document_id"].nunique()),
            "pages_without_a_candidate": int((annotation["candidates"] == 0).sum()),
            "by_grade": frame["grade"].value_counts().sort_index().to_dict(),
            "by_corrector_sources": frame["corrector_sources"].value_counts().to_dict(),
            "by_split": frame.groupby("block")["candidate_id"].size().to_dict(),
            "harmful_prevalence": float(frame["is_harmful"].mean()),
            "exact_prevalence": float(frame["exact"].mean()),
            "evaluable_error_sites": len(errors),
            "rule": "RL1's U5: HY1's H5 arm, top-1 image corrector, deduplicated by RL1's rule",
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"population: {len(frame)} candidates, {frame['site_group'].nunique()} sites, "
        f"{frame['document_id'].nunique()} pages ({time.monotonic() - started:.0f}s)"
    )
    return 0


def build_feature_matrix(
    population: pd.DataFrame,
    population_path: Path,
    candidates_path: Path,
    strata_path: Path,
    contexts_path: Path,
) -> pd.DataFrame:
    """R3 by RL1's own `build_features` and the new families by RK3's own `build_new_features`.

    Both read their inputs from module-level paths, which are pointed at PF1's tables for the
    call. The fitted resources stay RL1's frozen ADAPTATION resources, which no PF1 page touched.
    """
    with rebound(
        [
            (rl1, "CANDIDATE_POPULATION", population_path),
            (hy1, "CANDIDATES", candidates_path),
            (hy1, "PROPOSAL_STRATA", strata_path),
            (hy1, "CONTEXTS", contexts_path),
        ]
    ):
        r3 = rl1.build_features()
        new = rk3.build_new_features(rk3.observation_view(population))
    columns = rk3.r3_columns()
    matrix = r3[["candidate_id", "environment", *columns]].merge(
        new.drop(columns=["environment"]), on="candidate_id", how="inner", validate="one_to_one"
    )
    if len(matrix) != population["candidate_id"].nunique():
        raise PhaseError("the feature matrix does not cover every candidate")
    return matrix.sort_values("candidate_id", kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ the pre-registered analysis
#
# Everything from here to the end of this block is written to `design_record.json` by
# `--preregister`, which refuses to run once any PF1 page carries a label.

FEATURE_MATRIX = OUT / "feature_matrix.parquet"
FEATURE_REGISTRY = OUT / "feature_registry.json"
SPLIT_REGISTRY = OUT / "split_registry.json"
PURCHASE_REGISTRY = OUT / "purchase_registry.json"
CELL_SCORES = OUT / "cell_scores.parquet"
ADAPTATION_REGISTRY = OUT / "adaptation_registry.json"
POLICY_OUTCOMES = OUT / "policy_outcomes.parquet"
RANKING_OUTCOMES = OUT / "ranking_outcomes.parquet"
POLICY_REGISTRY = OUT / "policy_registry.json"
CONTROL_RESULTS = OUT / "control_results.json"
CURVES = OUT / "label_budget_curves.json"
FRONTIER = OUT / "annotation_frontier.json"
DIVERSITY = OUT / "page_diversity.json"
BOTTLENECK = OUT / "bottleneck_diagnosis.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"
DECISION = OUT / "research_decision.json"
SPOTCHECK = OUT / "generation_spotcheck.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

PAGE_SEED = "sgv-pf1-page-order-v1:2026-09-26"
CONTROL_SEED = 20261002
PERMUTATION_SEED = 20261003
BOOTSTRAP_RESAMPLES = rk4.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rk4.BOOTSTRAP_SEED

DIRECTIONS = rk4.DIRECTIONS
DIRECTION_SHORT = rk4.DIRECTION_SHORT
SOURCE_OF = rk4.SOURCE_OF
TARGET_OF = rk4.TARGET_OF
POOL = rk4.POOL
BUDGETS = ("0", "5", "10", "25", "50", "100", POOL)
ADAPTED_BUDGETS = BUDGETS[1:]
DRAWS = rk4.DRAWS
CALIBRATION_EVERY = rk4.CALIBRATION_EVERY
RK4_POOL_CAP = {rk4.D_G2Q: 31, rk4.D_Q2G: 30}

A0 = rk4.A0
M1 = rk4.M1
B2 = rk4.B2
ARMS = (A0, M1, B2)
ADAPTED_ARMS = (M1, B2)
PRIMARY_ARM = M1
ARM_LABEL = {
    A0: "RK3's risk-aware ranker fitted on the source generator only (budget 0)",
    M1: "RK4's M1: RK3's ranker refitted on the source rows plus the purchased target pages",
    B2: "RK2's LambdaMART on R3, fitted on the purchased target pages alone",
}

SCHEME_RANDOM = "random"
SCHEME_STRATIFIED = "stratified"
SCHEMES = (SCHEME_RANDOM, SCHEME_STRATIFIED)
STRATIFIED_BUDGETS = ("10", "25", "50", "100")
SCHEME_LABEL = {
    SCHEME_RANDOM: "pages bought in a seeded hash order, RK4's purchase rule",
    SCHEME_STRATIFIED: (
        "pages bought round robin over the ten engine environments, each environment's pages in "
        "the same seeded order: the most diverse purchase a page count allows"
    ),
}

SET_TEST = "test"
SET_EXPOSED = "exposed_test"
SET_CALIBRATION = "calibration"

EPSILON = dep1.EPSILON
ETA = dep1.ETA
DELTA = dep1.DELTA
P0 = dep1.P0
P1 = dep1.P1
P2 = dep1.P2
P3 = dep1.P3
POLICIES = dep1.POLICIES
FULL_AUTOMATION = dep1.FULL_AUTOMATION
THREE_WAY = dep1.THREE_WAY
CONSERVATIVE = (P1, P2)
VIOLATION_CEILING = dep1.VIOLATION_CEILING
WORKING_FLOOR = dep1.WORKING_FLOOR
HARM_TOLERANCE = dep1.HARM_TOLERANCE
REVIEW_REDUCTION_FLOOR = dep1.REVIEW_REDUCTION_FLOOR
USEFUL_COVERAGE_FLOOR = 0.05
USEFUL_SENSITIVITY = (0.02, 0.05, 0.10)
LOST_REPAIR_CEILING = 0.10
ORACLE_COVERAGE_FLOOR = 0.05
ORACLE_SENSITIVITY = (0.02, 0.05, 0.10)
PERMUTATION_BUDGET = "100"
PERMUTATION_DRAWS = 5

QUESTIONS = {
    "RQ1": (
        "how do ranking quality, threshold reliability and safe automation coverage change as "
        "the labelled page budget grows from 0 to the full pool?"
    ),
    "RQ2": (
        "does a more diverse set of labelled pages improve calibration stability, worst-case "
        "harm control and cross-document robustness at a matched page count?"
    ),
    "RQ3": (
        "when safe, useful automation is not reached, is the ranking short (even a cutoff read "
        "off the test labels cannot automate safely) or the calibration evidence short (the "
        "ranking could, but a cutoff chosen without test labels does not)?"
    ),
}
HYPOTHESES = {
    "H1_th2_projection": (
        "SGV-TH2 projected that a page-corrected boundary works at about 100 labelled pages in "
        "current_to_qwen: P1 is safe and useful from at most 100 pages in that direction"
    ),
    "H2_diversity": (
        "stratified page purchase lowers the full-automation violation share at 50 pages"
    ),
    "H3_evidence": (
        "the gap between the test-oracle cutoff's coverage and P1's narrows from 25 to 100 "
        "pages, as it must if calibration evidence is what is short"
    ),
}
OUTCOME_TAXONOMY = {
    "A": "a conservative policy is safe and useful within the measured budgets in both directions",
    "B": "a conservative policy is safe and useful within the measured budgets in one direction",
    "C": (
        "no conservative policy is safe and useful, and at the full pool the ranking could "
        "automate safely in at least one direction: calibration evidence is the bottleneck"
    ),
    "D": (
        "no conservative policy is safe and useful, and even test-oracle cutoffs cannot automate "
        "safely in either direction: ranking capability is the bottleneck"
    ),
}
NEXT_STAGE = {
    "A": "NEXT: CONFIRM THE PAGE-BUDGETED DEPLOYMENT FRONTIER ON UNSEEN DOCUMENTS",
    "B": "NEXT: DIRECTION-SPECIFIC ONBOARDING FOR THE GENERATOR THAT REACHED THE FRONTIER",
    "C": "NEXT: PAGE-EFFICIENT CALIBRATION EVIDENCE BEFORE ANY NEW RANKER",
    "D": "NEXT: TOP-OF-LIST PURITY OF THE ADAPTED RANKER BEFORE MORE LABELLED PAGES",
}
PRIMARY_FAMILY = (
    "T1_p1_safe_coverage_100_vs_25_g2q",
    "T1_p1_safe_coverage_100_vs_25_q2g",
    "T2_harm_auroc_100_vs_10_g2q",
    "T2_harm_auroc_100_vs_10_q2g",
    "T3_p0_violation_stratified_vs_random_50",
    "T4_oracle_gap_100_vs_25",
)
FAMILY_SPEC: dict[str, dict[str, Any]] = {
    PRIMARY_FAMILY[0]: {
        "directions": [rk4.D_G2Q],
        "metric": "safe_coverage",
        "policy": P1,
        "left": ("100", SCHEME_RANDOM),
        "right": ("25", SCHEME_RANDOM),
    },
    PRIMARY_FAMILY[1]: {
        "directions": [rk4.D_Q2G],
        "metric": "safe_coverage",
        "policy": P1,
        "left": ("100", SCHEME_RANDOM),
        "right": ("25", SCHEME_RANDOM),
    },
    PRIMARY_FAMILY[2]: {
        "directions": [rk4.D_G2Q],
        "metric": "harm_auroc",
        "policy": None,
        "left": ("100", SCHEME_RANDOM),
        "right": ("10", SCHEME_RANDOM),
    },
    PRIMARY_FAMILY[3]: {
        "directions": [rk4.D_Q2G],
        "metric": "harm_auroc",
        "policy": None,
        "left": ("100", SCHEME_RANDOM),
        "right": ("10", SCHEME_RANDOM),
    },
    PRIMARY_FAMILY[4]: {
        "directions": list(rk4.DIRECTIONS),
        "metric": "accept_violation",
        "policy": P0,
        "left": ("50", SCHEME_STRATIFIED),
        "right": ("50", SCHEME_RANDOM),
    },
    PRIMARY_FAMILY[5]: {
        "directions": list(rk4.DIRECTIONS),
        "metric": "oracle_gap",
        "policy": P1,
        "left": ("100", SCHEME_RANDOM),
        "right": ("25", SCHEME_RANDOM),
    },
}


def minimal_budget(flags: dict[str, bool]) -> str | None:
    """The smallest measured budget from which the flag holds at every larger measured budget."""
    chosen: str | None = None
    for label in reversed(BUDGETS):
        if label not in flags:
            continue
        if not flags[label]:
            break
        chosen = label
    return chosen


def deployable(violation_share: float | None, working_share: float | None) -> bool:
    """TH1's definition for full automation, unchanged."""
    return dep1.deployable(violation_share, working_share)


def useful(
    violation_share: float | None,
    working_share: float | None,
    coverage: float | None,
    floor: float = USEFUL_COVERAGE_FLOOR,
) -> bool:
    """Full automation that is deployable and accepts a pre-registered share of the decisions."""
    return bool(
        deployable(violation_share, working_share) and coverage is not None and coverage >= floor
    )


def practical(
    accept_violation_share: float | None,
    reject_violation_share: float | None,
    review_reduction: float | None,
    lost_repair_share: float | None,
    floor: float = REVIEW_REDUCTION_FLOOR,
) -> bool:
    """DEP1's practical definition, plus a ceiling on the reachable repairs the reject band loses.

    SGV-DEP1 found its reject constraint nearly vacuous where exact repairs are rare: rejecting
    everything already met it. The added ceiling makes discarding the repairs fail.
    """
    return bool(
        dep1.practical(accept_violation_share, reject_violation_share, review_reduction, floor)
        and lost_repair_share is not None
        and lost_repair_share <= LOST_REPAIR_CEILING
    )


def assign_outcome(frontier: dict[str, str | None], diagnosis: dict[str, str]) -> str:
    """A: both directions reach the frontier; B: one; C: evidence-limited somewhere; D: neither."""
    reached = [d for d in DIRECTIONS if frontier[d] is not None]
    if len(reached) == len(DIRECTIONS):
        return "A"
    if reached:
        return "B"
    if any(diagnosis[d] == "evidence_limited" for d in DIRECTIONS):
        return "C"
    return "D"


def diagnose(reached: bool, oracle_coverage: float | None, floor: float) -> str:
    """RQ3 per direction at the full pool."""
    if reached:
        return "reached"
    if oracle_coverage is not None and oracle_coverage >= floor:
        return "evidence_limited"
    return "ranking_limited"


def run_preregister() -> int:
    """The analysis design, written once, before any PF1 page carries a label."""
    started = time.monotonic()
    _require(GENERATION_PLAN, "plan")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("a PF1 label exists; the design cannot be written after it")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "questions": QUESTIONS,
            "hypotheses": HYPOTHESES,
            "scope": {
                "data_generation": (
                    "new candidate generation on 172 clean development pages with SGV14's frozen "
                    "OCR; no OCR is re-run; no confirmatory page is read"
                ),
                "labels": (
                    "each corpus's shipped human transcription, attached by SGV14's frozen "
                    "labelling function; no new human annotation is collected"
                ),
            },
            "population": {
                "old_pages": "RK3's U5 ranking population, unchanged, with its RL1 folds",
                "new_pages": "PF1's pages through the replayed chain, RL1's U5 rule, RK1's grades",
                "features": "R5: RL1's R3 plus RK3's 53 label-free features, frozen resources",
                "generator_identity_in_the_model": False,
            },
            "blocks": {
                "source_fit": "RK4's source fit, unchanged: U5 fit-block rows of the source only",
                "source_calibration": "RK4's: U5 threshold-block source rows, budget 0 only",
                "pool": (
                    "target rows on RK4's pool pages (U5 folds 0-2) plus target rows on PF1's "
                    "adaptation-pool pages"
                ),
                "test": "PRIMARY: target rows on PF1's test pages, never read by any earlier stage",
                "exposed_test": "SECONDARY: RK4's own sealed test block, for continuity only",
            },
            "budgets": list(BUDGETS),
            "draws": DRAWS,
            "page_seed": PAGE_SEED,
            "purchase": {
                "rule": "RK4's: the first N pages of a draw's order; every third is calibration",
                "calibration_every": CALIBRATION_EVERY,
                "schemes": SCHEME_LABEL,
                "stratified_budgets": list(STRATIFIED_BUDGETS),
                "stratified_arms": [M1],
            },
            "arms": ARM_LABEL,
            "primary_arm": PRIMARY_ARM,
            "policies": dep1.POLICY_LABEL,
            "rules": {k: list(v) for k, v in dep1.POLICY_RULES.items()},
            "conservative_policies": list(CONSERVATIVE),
            "epsilon": EPSILON,
            "eta": ETA,
            "delta": DELTA,
            "definitions": {
                "deployable": (
                    f"TH1's: accept violation share <= {VIOLATION_CEILING} and working share >= "
                    f"{WORKING_FLOOR} over the draws"
                ),
                "useful": (
                    f"deployable, and median acceptance coverage >= {USEFUL_COVERAGE_FLOOR} of "
                    f"the test decisions (sensitivity {list(USEFUL_SENSITIVITY)})"
                ),
                "practical": (
                    f"accept and reject violation shares each <= {VIOLATION_CEILING}, median "
                    f"review reduction >= {REVIEW_REDUCTION_FLOOR} and median lost-repair share "
                    f"<= {LOST_REPAIR_CEILING}; DEP1's definition without the last clause is "
                    "reported as a sensitivity"
                ),
                "lost_repair_share": (
                    "error sites whose decision is an exact repair and falls in the reject band, "
                    "over error sites whose decision is an exact repair in any band"
                ),
                "frontier": (
                    "the smallest measured budget from which the flag holds at every larger "
                    "measured budget (N*), masked where seeded random scores also pass"
                ),
                "safe_and_useful": "P1 useful, or P2 practical",
                "oracle": (
                    "ANALYSIS ONLY: the loosest cutoff meeting epsilon on the test labels "
                    "themselves; never an operating point"
                ),
                "diagnosis": (
                    "at the full pool: reached; else evidence-limited when M1's median oracle "
                    f"coverage >= {ORACLE_COVERAGE_FLOOR} (sensitivity "
                    f"{list(ORACLE_SENSITIVITY)}); else ranking-limited"
                ),
            },
            "metrics": {
                "RQ1": [
                    "harm AUROC",
                    "benefit AUROC",
                    "top-1 exact share",
                    "violation and working shares per policy",
                    "calibration-to-test harm error",
                    "median safe coverage",
                    "automated, review, lost and total repair recall",
                ],
                "RQ2": [
                    "IQR over draws of the acceptance coverage (cutoff stability)",
                    "spread over draws of the realized test harm",
                    "worst draw and worst page harm",
                    "safe coverage and harm by corpus and engine at the pool",
                    "stratified versus random purchase at 10, 25, 50 and 100 pages",
                ],
                "RQ3": [
                    "median oracle coverage and its gap to P0 and P1",
                    "the per-direction diagnosis at the pool",
                ],
            },
            "outcome_taxonomy": OUTCOME_TAXONOMY,
            "next_stage": NEXT_STAGE,
            "statistical_plan": {
                "family": {
                    k: {**v, "left": list(v["left"]), "right": list(v["right"])}
                    for k, v in FAMILY_SPEC.items()
                },
                "test": "exact sign test over draws paired by index (a budget's pages are a prefix "
                "of a larger budget's in the same draw); direction-stratified bootstrap interval",
                "resamples": BOOTSTRAP_RESAMPLES,
                "correction": "Holm within the six-test family",
            },
            "controls": {
                "random_scores": "seeded uniform scores through every policy; flags masked",
                "label_permutation": (
                    f"M1 at {PERMUTATION_BUDGET} pages, {PERMUTATION_DRAWS} draws, target grades "
                    "permuted within pages"
                ),
                "a0_reproduces_rk4": "A0's scores on RK4's test rows equal RK4's exactly",
            },
            "not_measurable": {
                "budgets_above_the_pool": [250, 500],
                "why": "the clean page census holds 172 pages; no projection is made",
            },
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"preregister: design frozen before any PF1 label ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ features


def run_features() -> int:
    """R3 and RK3's new families for every PF1 candidate, from the frozen builders."""
    started = time.monotonic()
    _require(POPULATION, "population")
    for path in (FEATURE_MATRIX, FEATURE_REGISTRY):
        _forbid(path)
    population = pd.read_parquet(POPULATION)
    matrix = build_feature_matrix(
        population, POPULATION, CANDIDATES, PROPOSAL_STRATA, CORRECTION_CONTEXTS
    )
    _write_parquet_once(FEATURE_MATRIX, matrix)
    columns = rk3.columns_for(rk3.R5)
    _write_json_once(
        FEATURE_REGISTRY,
        {
            **_envelope("feature_registry"),
            "rows": len(matrix),
            "r5_columns": len(columns),
            "source_columns_in_r5": [c for c in columns if c.startswith(rl1.FAM_SOURCE)],
            "constant_columns": sorted(c for c in columns if matrix[c].nunique() <= 1),
            "builders": ["rl1.build_features", "rk3.build_new_features"],
            "resources": "RL1's frozen ADAPTATION resources; no PF1 page was fitted on",
            "label_fields_read": [],
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"features: {len(matrix)} candidates x {len(columns)} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the study


@dataclass(slots=True)
class Study:
    """The combined population and every direction's blocks, as index arrays into it."""

    population: pd.DataFrame
    matrix: pd.DataFrame
    blocks: dict[str, Any]
    exposed: dict[str, np.ndarray]
    environment_of: dict[str, str]


def load_study() -> Study:
    """U5 first in RK3's own order, then PF1's pages; RK4's blocks extended by PF1's pages."""
    u5 = rk3.load_population()
    new = pd.read_parquet(POPULATION)[list(u5.columns)]
    if set(u5["candidate_id"]) & set(new["candidate_id"]):
        raise PhaseError("a PF1 candidate id collides with a U5 one")
    population = pd.concat([u5, new], ignore_index=True)
    matrix = pd.concat(
        [pd.read_parquet(rk3.FEATURE_MATRIX), pd.read_parquet(FEATURE_MATRIX)], ignore_index=True
    )
    old = rk4.build_blocks(u5)
    has = rk3.membership(population)
    block = population["block"].astype(str).to_numpy()
    documents = population["document_id"].astype(str).to_numpy()
    blocks: dict[str, Any] = {}
    exposed: dict[str, np.ndarray] = {}
    for direction in DIRECTIONS:
        b = old[direction]
        target, source = TARGET_OF[direction], SOURCE_OF[direction]
        pool = np.concatenate(
            [b.pool, np.flatnonzero(has[target] & (block == PF1_BLOCK[POOL_FOLDS[0]]))]
        )
        test = np.flatnonzero(has[target] & (block == PF1_BLOCK[TEST_FOLDS[0]]))
        blocks[direction] = rk4.Blocks(
            direction=direction,
            source=source,
            target=target,
            source_fit=b.source_fit,
            source_calibration=b.source_calibration,
            pool=pool,
            test=test,
            source_test=np.flatnonzero(has[source] & (block == PF1_BLOCK[TEST_FOLDS[0]])),
            pages=sorted(set(documents[pool].tolist())),
        )
        exposed[direction] = b.test
    environment_of = (
        population.sort_values("environment", kind="stable")
        .groupby("document_id")["environment"]
        .first()
        .astype(str)
        .to_dict()
    )
    return Study(population, matrix, blocks, exposed, environment_of)


def page_order(
    pages: Sequence[str], draw: int, scheme: str, environment_of: dict[str, str]
) -> list[str]:
    """A draw's purchase order. Random: a seeded hash order. Stratified: that same order dealt
    round robin over the environments, in a seeded environment order. It reads page ids only."""
    ordered = sorted(
        pages,
        key=lambda page: (
            canonical_hash({"seed": PAGE_SEED, "draw": int(draw), "page": page}),
            page,
        ),
    )
    if scheme == SCHEME_RANDOM:
        return ordered
    queues: dict[str, list[str]] = {}
    for page in ordered:
        queues.setdefault(environment_of[page], []).append(page)
    rotation = sorted(
        queues,
        key=lambda e: (canonical_hash({"seed": PAGE_SEED, "draw": int(draw), "environment": e}), e),
    )
    out: list[str] = []
    while any(queues.values()):
        for environment in rotation:
            if queues[environment]:
                out.append(queues[environment].pop(0))
    return out


def budget_pages(label: str, pool_pages: int) -> int:
    return pool_pages if label == POOL else min(int(label), pool_pages)


def purchase(
    pages: Sequence[str], draw: int, label: str, scheme: str, environment_of: dict[str, str]
) -> tuple[list[str], list[str]]:
    """The fit pages and the held-back calibration pages a budget buys, in purchase order."""
    bought = page_order(pages, draw, scheme, environment_of)[: budget_pages(label, len(pages))]
    fit = [p for i, p in enumerate(bought) if i % CALIBRATION_EVERY != CALIBRATION_EVERY - 1]
    calibration = [
        p for i, p in enumerate(bought) if i % CALIBRATION_EVERY == CALIBRATION_EVERY - 1
    ]
    return fit, calibration


def _digest(population: pd.DataFrame, index: np.ndarray) -> str:
    return str(canonical_hash(population.iloc[index]["candidate_id"].astype(str).tolist()))


def cells() -> list[tuple[str, str, str, str, int]]:
    """Every (direction, arm, scheme, budget, draw) the design fits, in execution order."""
    out: list[tuple[str, str, str, str, int]] = []
    for direction in DIRECTIONS:
        out.append((direction, A0, SCHEME_RANDOM, "0", 0))
        for label in ADAPTED_BUDGETS:
            for draw in range(DRAWS):
                out.extend((direction, arm, SCHEME_RANDOM, label, draw) for arm in ADAPTED_ARMS)
        for label in STRATIFIED_BUDGETS:
            for draw in range(DRAWS):
                out.append((direction, M1, SCHEME_STRATIFIED, label, draw))
    return out


def run_splits() -> int:
    """Every direction's blocks and every draw's purchase, frozen before any fit."""
    started = time.monotonic()
    _require(FEATURE_MATRIX, "features")
    for path in (SPLIT_REGISTRY, PURCHASE_REGISTRY):
        _forbid(path)
    study = load_study()
    population = study.population
    documents = population["document_id"].astype(str).to_numpy()
    directions: dict[str, Any] = {}
    purchases: list[dict[str, Any]] = []
    for direction, b in study.blocks.items():
        fitting = set(documents[np.concatenate([b.source_fit, b.pool])].tolist())
        tested = set(documents[b.test].tolist())
        directions[direction] = {
            "source_generator": b.source,
            "target_generator": b.target,
            "source_fit": {
                "rows": int(b.source_fit.size),
                "pages": len(set(documents[b.source_fit])),
            },
            "source_calibration": {
                "rows": int(b.source_calibration.size),
                "pages": len(set(documents[b.source_calibration])),
            },
            "pool": {
                "rows": int(b.pool.size),
                "pages": len(b.pages),
                "u5_pages": len(set(documents[b.pool]) & u5_documents()),
                "pf1_pages": len(set(documents[b.pool]) - u5_documents()),
            },
            "test": {
                "rows": int(b.test.size),
                "pages": len(tested),
                "sites": int(population.iloc[b.test]["site_group"].nunique()),
            },
            "exposed_test": {"rows": int(study.exposed[direction].size)},
            "isolation": {"pages_fitting_vs_test": len(fitting & tested)},
            "test_digest": _digest(population, b.test),
        }
        if fitting & tested:
            raise PhaseError(f"{direction}: a test page is also a fitting page")
        for _d, arm, scheme, label, draw in cells():
            if _d != direction or arm in (A0, B2):
                continue
            fit_pages, cal_pages = purchase(b.pages, draw, label, scheme, study.environment_of)
            purchases.append(
                {
                    "direction": direction,
                    "scheme": scheme,
                    "budget": label,
                    "draw": draw,
                    "fit_pages": len(fit_pages),
                    "calibration_pages": len(cal_pages),
                    "fit_digest": _digest(population, rk4.rows_on(population, b.pool, fit_pages)),
                    "calibration_digest": _digest(
                        population, rk4.rows_on(population, b.pool, cal_pages)
                    ),
                    "environments_bought": len(
                        {study.environment_of[p] for p in fit_pages + cal_pages}
                    ),
                }
            )
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_envelope("split_registry"),
            "directions": directions,
            "budgets": list(BUDGETS),
            "pool_pages": {d: len(b.pages) for d, b in study.blocks.items()},
            "rk4_pool_cap": RK4_POOL_CAP,
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(PURCHASE_REGISTRY, {**_envelope("purchase_registry"), "rows": purchases})
    print(
        f"splits: pool pages { ({d: len(b.pages) for d, b in study.blocks.items()}) }, "
        f"{len(purchases)} purchases ({time.monotonic() - started:.0f}s)"
    )
    return 0


def run_adapt() -> int:
    """Every cell's test, exposed-test and calibration scores."""
    started = time.monotonic()
    _require(PURCHASE_REGISTRY, "splits")
    for path in (CELL_SCORES, ADAPTATION_REGISTRY):
        _forbid(path)
    study = load_study()
    population = study.population
    ids = population["candidate_id"].astype(str).to_numpy()
    frozen = {
        (r["direction"], r["scheme"], r["budget"], r["draw"]): r
        for r in cc_read_json(PURCHASE_REGISTRY)["rows"]
    }
    sealed = {d: r["test_digest"] for d, r in cc_read_json(SPLIT_REGISTRY)["directions"].items()}
    frames: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []

    def emit(
        key: tuple[str, str, str, str, int], kind: str, rows: np.ndarray, values: np.ndarray
    ) -> None:
        direction, arm, scheme, label, draw = key
        frames.append(
            pd.DataFrame(
                {
                    "direction": direction,
                    "arm": arm,
                    "scheme": scheme,
                    "budget": label,
                    "draw": int(draw),
                    "evaluation_set": kind,
                    "candidate_id": ids[rows],
                    "score": values,
                }
            )
        )

    for direction, b in study.blocks.items():
        began = time.monotonic()
        if _digest(population, b.test) != sealed[direction]:
            raise PhaseError(f"{direction}: the test block differs from the sealed one")
        context = rk4.arm_context(population, study.matrix, b)
        zero = context["zero_shot"]
        key = (direction, A0, SCHEME_RANDOM, "0", 0)
        emit(key, SET_TEST, b.test, zero[b.test])
        emit(key, SET_EXPOSED, study.exposed[direction], zero[study.exposed[direction]])
        emit(key, SET_CALIBRATION, b.source_calibration, zero[b.source_calibration])
        registry.append(
            {
                "direction": direction,
                "arm": A0,
                "scheme": SCHEME_RANDOM,
                "budget": "0",
                "draw": 0,
                "trainable": bool(context["zero_shot_model"].trees),
                "calibration": "source",
            }
        )
        for _d, arm, scheme, label, draw in cells():
            if _d != direction or arm == A0:
                continue
            fit_pages, cal_pages = purchase(b.pages, draw, label, scheme, study.environment_of)
            fit_rows = rk4.rows_on(population, b.pool, fit_pages)
            cal_rows = rk4.rows_on(population, b.pool, cal_pages)
            record = frozen[(direction, scheme, label, draw)]
            if (
                _digest(population, fit_rows) != record["fit_digest"]
                or _digest(population, cal_rows) != record["calibration_digest"]
            ):
                raise PhaseError(f"{direction} {scheme} {label} {draw}: not the frozen purchase")
            fitted = rk4.fit_arm(arm, fit_rows, context)
            key = (direction, arm, scheme, label, draw)
            test_scores = fitted.score(b.test)
            emit(key, SET_TEST, b.test, test_scores)
            emit(key, SET_EXPOSED, study.exposed[direction], fitted.score(study.exposed[direction]))
            emit(
                key,
                SET_CALIBRATION,
                cal_rows,
                fitted.score(cal_rows) if cal_rows.size else np.empty(0),
            )
            registry.append(
                {
                    "direction": direction,
                    "arm": arm,
                    "scheme": scheme,
                    "budget": label,
                    "draw": draw,
                    "trainable": bool(fitted.trainable and not rk2._constant(test_scores)),
                    "calibration": "target",
                    "fit_pages": len(fit_pages),
                    "calibration_pages": len(cal_pages),
                    "fit_rows": int(fit_rows.size),
                    "calibration_rows": int(cal_rows.size),
                    **fitted.detail,
                }
            )
        print(f"  {direction}: {time.monotonic() - began:.0f}s", flush=True)
    table = pd.concat(frames, ignore_index=True).sort_values(
        ["direction", "arm", "scheme", "budget", "draw", "evaluation_set", "candidate_id"],
        kind="stable",
    )
    _write_parquet_once(CELL_SCORES, table.reset_index(drop=True))
    _write_json_once(
        ADAPTATION_REGISTRY,
        {
            **_analysis_envelope("adaptation_registry"),
            "arms": ARM_LABEL,
            "cells": registry,
            "fits": len(registry),
            "scored_rows": len(table),
            "untrainable": [
                f"{r['direction']}|{r['arm']}|{r['scheme']}|{r['budget']}|{r['draw']}"
                for r in registry
                if not r["trainable"]
            ],
            "source_columns_in_the_model": [
                c for c in rk3.columns_for(rk3.R5) if c.startswith(rl1.FAM_SOURCE)
            ],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"adapt: {len(registry)} fits, {len(table)} scored rows ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ policies on the test pages

LABELS = rk4.LABELS


def error_population() -> pd.DataFrame:
    """PF1's evaluable OCR error sites with their keys and each page's split."""
    errors = pd.read_parquet(ERROR_SITES)
    split = load_pool().set_index("document_id")["split"]
    errors = errors.copy()
    errors["block"] = (
        errors["document_id"].astype(str).map(split).map({SPLIT_POOL: "pool", SPLIT_TEST: "test"})
    )
    if errors["block"].isna().any():
        raise PhaseError("an error site's page carries no split")
    errors["error_key"] = (
        errors["environment"].astype(str) + "|" + errors["align_site_id"].astype(str)
    )
    return errors


def site_to_errors(errors: pd.DataFrame, links: pd.DataFrame) -> dict[str, set[str]]:
    """DS1's `site_to_errors` over PF1's own links."""
    known = set(errors["error_key"])
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


@dataclass(slots=True)
class Evaluation:
    """Labelled score frames per cell and set, and the recall context of each test block."""

    cells: dict[tuple[str, str, str, str, int], dict[str, pd.DataFrame]]
    registry: dict[tuple[str, str, str, str, int], dict[str, Any]]
    context: dict[str, tuple[set[str], dict[str, set[str]], pd.DataFrame]]


def load_evaluation() -> Evaluation:
    study_population = pd.concat(
        [rk3.load_population(), pd.read_parquet(POPULATION)[list(rk3.load_population().columns)]],
        ignore_index=True,
    )
    labels = study_population.set_index("candidate_id")[[*LABELS, "corpus"]]
    cell_frames: dict[tuple[str, str, str, str, int], dict[str, pd.DataFrame]] = {}
    for key, group in pd.read_parquet(CELL_SCORES).groupby(
        ["direction", "arm", "scheme", "budget", "draw", "evaluation_set"], sort=True
    ):
        direction, arm, scheme, budget, draw, kind = key
        joined = group[["candidate_id", "score"]].join(labels, on="candidate_id", how="inner")
        if len(joined) != len(group):
            raise PhaseError(f"{key}: a scored candidate has no label row")
        cell_frames.setdefault((str(direction), str(arm), str(scheme), str(budget), int(draw)), {})[
            str(kind)
        ] = joined.assign(rank_score=joined["score"], safety=joined["score"]).reset_index(drop=True)
    registry = {
        (r["direction"], r["arm"], r["scheme"], r["budget"], r["draw"]): r
        for r in cc_read_json(ADAPTATION_REGISTRY)["cells"]
    }
    errors = error_population()
    test_errors = errors[errors["block"] == "test"].reset_index(drop=True)
    mapping = site_to_errors(errors, pd.read_parquet(PROPOSAL_ERROR_LINKS))
    old = ds1.load_error_population()
    old_test = old[old["block"] == "test"].reset_index(drop=True)
    context = {
        SET_TEST: (set(test_errors["error_key"].astype(str)), mapping, test_errors),
        SET_EXPOSED: (set(old_test["error_key"].astype(str)), ds1.site_to_errors(old), old_test),
    }
    return Evaluation(cell_frames, registry, context)


def cell_key(
    direction: str, arm: str, scheme: str, budget: str, draw: int
) -> tuple[str, str, str, str, int]:
    """Budget zero is the zero-shot ranker for every arm and scheme."""
    if budget == "0" or arm == A0:
        return direction, A0, SCHEME_RANDOM, "0", 0
    return direction, arm, scheme, budget, draw


def repaired_sites(
    decisions: pd.DataFrame, context: tuple[set[str], dict[str, set[str]], pd.DataFrame]
) -> set[str]:
    keys, mapping, errors = context
    _ = keys
    return ds1._repaired_flags(decisions, decisions, errors, mapping)


def policy_row(
    policy: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    context: tuple[set[str], dict[str, set[str]], pd.DataFrame],
    usable: bool,
) -> dict[str, Any]:
    """DEP1's `evaluate_policy`, plus the lost-repair share and the worst page's harm."""
    keys, mapping, errors = context
    row = dep1.evaluate_policy(policy, calibration, test, keys, mapping, errors, usable)
    b = dep1.bands(policy, calibration, test, usable)
    auto = repaired_sites(test[b["accept"]], context)
    review = repaired_sites(test[b["review"]], context)
    lost = repaired_sites(test[b["reject"]], context) - auto - review
    reachable = auto | review | lost
    row["lost_repair_share"] = len(lost) / len(reachable) if reachable else None
    row.update(th2.page_harm(test[b["accept"]], EPSILON))
    return row


def ranking_row(
    test: pd.DataFrame, usable: bool, context: tuple[set[str], dict[str, set[str]], pd.DataFrame]
) -> dict[str, Any]:
    """RK1's ranking metrics and the ANALYSIS-ONLY test-oracle cutoff's outcome."""
    metrics = rk1.ranking_metrics(test) if usable else {}
    decisions = rk4.decisions(test)
    oracle = rk2.oracle_accepted(decisions, EPSILON) if usable else decisions.head(0)
    keys = context[0]
    return {
        "decisions": len(decisions),
        "harm_auroc": metrics.get("harm_auroc"),
        "benefit_auroc": metrics.get("benefit_auroc"),
        "top1": metrics.get("top1"),
        "trainable": usable,
        "oracle_accepted": len(oracle),
        "oracle_coverage": _share(len(oracle), len(decisions)) or 0.0,
        "oracle_harm": _share(int(oracle["is_harmful"].sum()), len(oracle)),
        "oracle_automated_recall": len(repaired_sites(oracle, context)) / max(len(keys), 1),
    }


def run_policies() -> int:
    """Every policy on every cell, on the fresh test pages and on RK4's exposed test block."""
    started = time.monotonic()
    _require(ADAPTATION_REGISTRY, "adapt")
    for path in (POLICY_OUTCOMES, RANKING_OUTCOMES, POLICY_REGISTRY):
        _forbid(path)
    evaluation = load_evaluation()
    policy_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for key, cell in sorted(evaluation.cells.items()):
        direction, arm, scheme, budget, draw = key
        usable = bool(evaluation.registry[key]["trainable"])
        calibration = rk4.decisions(cell.get(SET_CALIBRATION, cell[SET_TEST].head(0)))
        for kind in (SET_TEST, SET_EXPOSED):
            frame = cell[kind]
            context = evaluation.context[kind]
            base = {
                "direction": direction,
                "arm": arm,
                "scheme": scheme,
                "budget": budget,
                "draw": draw,
                "evaluation_set": kind,
            }
            ranking_rows.append(base | ranking_row(frame, usable, context))
            decisions = rk4.decisions(frame)
            policy_rows.extend(
                base | policy_row(policy, calibration, decisions, context, usable)
                for policy in POLICIES
            )
    table = pd.DataFrame(policy_rows)
    ranking = pd.DataFrame(ranking_rows)
    order = ["direction", "arm", "scheme", "budget", "draw", "evaluation_set"]
    _write_parquet_once(
        POLICY_OUTCOMES, table.sort_values([*order, "policy"], kind="stable").reset_index(drop=True)
    )
    _write_parquet_once(
        RANKING_OUTCOMES, ranking.sort_values(order, kind="stable").reset_index(drop=True)
    )
    _write_json_once(
        POLICY_REGISTRY,
        {
            **_analysis_envelope("policy_registry"),
            "policies": dep1.POLICY_LABEL,
            "cells": len(evaluation.cells),
            "rows": len(table),
            "recall_denominator": {k: len(v[0]) for k, v in evaluation.context.items()},
            "cutoffs_chosen_without_test_labels": True,
            "oracle_is_analysis_only": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    elapsed = time.monotonic() - started
    print(f"policies: {len(table)} outcomes on {len(evaluation.cells)} cells ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ controls


def _random_generator(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(
        int(canonical_hash({"seed": CONTROL_SEED, "cell": [str(p) for p in parts]})[:8], 16)
    )


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    """DEP1's per-budget summary over draws, plus the stability and worst-case quantities."""
    summary = dep1.summarize(rows)
    accepting = rows[rows["accepted"] > 0]
    coverage = rows["acceptance_coverage"].dropna().to_numpy(np.float64)
    harm = accepting["harm"].dropna().to_numpy(np.float64)
    pages = accepting["worst_page_harm"].dropna().to_numpy(np.float64)
    lost = rows["lost_repair_share"].dropna().to_numpy(np.float64)
    summary.update(
        {
            "median_lost_repair_share": float(np.median(lost)) if lost.size else None,
            "coverage_iqr": (
                float(np.percentile(coverage, 75) - np.percentile(coverage, 25))
                if coverage.size
                else None
            ),
            "harm_spread": float(harm.std(ddof=0)) if harm.size else None,
            "worst_draw_harm": float(harm.max()) if harm.size else None,
            "harm_p90": float(np.percentile(harm, 90)) if harm.size else None,
            "median_worst_page_harm": float(np.median(pages)) if pages.size else None,
            "worst_page_harm": float(pages.max()) if pages.size else None,
            "draws_with_a_page_over_epsilon": int((rows["pages_exceeding"] > 0).sum()),
        }
    )
    return summary


def flags(summary: dict[str, Any], policy: str) -> dict[str, bool]:
    if policy in FULL_AUTOMATION:
        out = {
            "deployable": deployable(summary["accept_violation_share"], summary["working_share"])
        }
        for floor in USEFUL_SENSITIVITY:
            out[f"useful_at_{floor}"] = useful(
                summary["accept_violation_share"],
                summary["working_share"],
                summary["median_acceptance_coverage"],
                floor,
            )
        return out
    out = {
        f"practical_at_{floor}": practical(
            summary["accept_violation_share"],
            summary["reject_violation_share"],
            summary["median_review_reduction"],
            summary["median_lost_repair_share"],
            floor,
        )
        for floor in dep1.REVIEW_REDUCTION_SENSITIVITY
    }
    out["practical_dep1_definition"] = dep1.practical(
        summary["accept_violation_share"],
        summary["reject_violation_share"],
        summary["median_review_reduction"],
    )
    return out


PRIMARY_FLAG = {
    P0: f"useful_at_{USEFUL_COVERAGE_FLOOR}",
    P1: f"useful_at_{USEFUL_COVERAGE_FLOOR}",
    P2: f"practical_at_{REVIEW_REDUCTION_FLOOR}",
    P3: f"practical_at_{REVIEW_REDUCTION_FLOOR}",
}


def run_controls() -> int:
    """Random scores through every policy; target-label permutation; A0 against RK4."""
    started = time.monotonic()
    _require(POLICY_REGISTRY, "policies")
    _forbid(CONTROL_RESULTS)
    evaluation = load_evaluation()
    context = evaluation.context[SET_TEST]
    rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for label in ADAPTED_BUDGETS:
            for draw in range(DRAWS):
                cell = evaluation.cells[(direction, M1, SCHEME_RANDOM, label, draw)]
                generator = _random_generator(direction, label, draw)
                test = rk4.decisions(cell[SET_TEST])
                test = test.assign(safety=generator.random(len(test)))
                calibration = rk4.decisions(cell[SET_CALIBRATION])
                calibration = calibration.assign(safety=generator.random(len(calibration)))
                for policy in POLICIES:
                    rows.append(
                        {"direction": direction, "budget": label, "draw": draw}
                        | policy_row(policy, calibration, test, context, True)
                    )
    table = pd.DataFrame(rows)
    random_summary: dict[str, Any] = {}
    passing: list[str] = []
    for direction in DIRECTIONS:
        for policy in POLICIES:
            for label in ADAPTED_BUDGETS:
                block = table[
                    (table["direction"] == direction)
                    & (table["policy"] == policy)
                    & (table["budget"] == label)
                ]
                summary = summarize(block) | flags(summarize(block), policy)
                random_summary.setdefault(direction, {}).setdefault(policy, {})[label] = summary
                if summary[PRIMARY_FLAG[policy]]:
                    passing.append(f"{direction}|{policy}|{label}")
    permutation = permutation_control()
    reproduction = a0_reproduction()
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "random": random_summary,
            "random_passing_cells": sorted(passing),
            "random_any_violation_share_overall": float(table["any_violation"].mean()),
            "permutation": permutation,
            "a0_reproduces_rk4": reproduction,
            "seeds": {"random": CONTROL_SEED, "permutation": rk4.PERMUTATION_SEED},
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"controls: random scores pass in {len(passing)} cells; A0 reproduces RK4 "
        f"{reproduction['equal']} ({time.monotonic() - started:.0f}s)"
    )
    return 0


def permutation_control() -> dict[str, Any]:
    """M1 refitted with the purchased target grades permuted within each page."""
    study = load_study()
    population = study.population
    documents = population["document_id"].astype(str).to_numpy()
    labels = population.set_index("candidate_id")[[*LABELS, "corpus"]]
    ids = population["candidate_id"].astype(str).to_numpy()
    real = pd.read_parquet(RANKING_OUTCOMES)
    real_policy = pd.read_parquet(POLICY_OUTCOMES)
    errors = error_population()
    context_test = (
        set(errors[errors["block"] == "test"]["error_key"]),
        site_to_errors(errors, pd.read_parquet(PROPOSAL_ERROR_LINKS)),
        errors[errors["block"] == "test"].reset_index(drop=True),
    )
    out: dict[str, Any] = {}
    for direction, b in study.blocks.items():
        context = rk4.arm_context(population, study.matrix, b)
        rows = []
        for draw in range(PERMUTATION_DRAWS):
            fit_pages, cal_pages = purchase(
                b.pages, draw, PERMUTATION_BUDGET, SCHEME_RANDOM, study.environment_of
            )
            fit_rows = rk4.rows_on(population, b.pool, fit_pages)
            cal_rows = rk4.rows_on(population, b.pool, cal_pages)
            shuffled = rk4.shuffled_within_pages(context["grades"], documents, fit_rows, draw)
            permuted = dict(
                context, grades=shuffled, utility=rk3.utility_of(shuffled, rk3.RISK_UTILITY)
            )
            fitted = rk4.fit_arm(M1, fit_rows, permuted)

            def frame(rows_: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
                joined = pd.DataFrame({"candidate_id": ids[rows_], "score": scores}).join(
                    labels, on="candidate_id"
                )
                return joined.assign(rank_score=joined["score"], safety=joined["score"])

            test = frame(b.test, fitted.score(b.test))
            calibration = rk4.decisions(frame(cal_rows, fitted.score(cal_rows)))
            usable = not rk2._constant(test["score"].to_numpy(np.float64))
            ranking = ranking_row(test, usable, context_test)
            decisions = rk4.decisions(test)
            p0 = policy_row(P0, calibration, decisions, context_test, usable)
            p1 = policy_row(P1, calibration, decisions, context_test, usable)
            mine = real[
                (real["direction"] == direction)
                & (real["arm"] == M1)
                & (real["scheme"] == SCHEME_RANDOM)
                & (real["budget"] == PERMUTATION_BUDGET)
                & (real["draw"] == draw)
                & (real["evaluation_set"] == SET_TEST)
            ].iloc[0]
            real_p0 = real_policy[
                (real_policy["direction"] == direction)
                & (real_policy["arm"] == M1)
                & (real_policy["scheme"] == SCHEME_RANDOM)
                & (real_policy["budget"] == PERMUTATION_BUDGET)
                & (real_policy["draw"] == draw)
                & (real_policy["evaluation_set"] == SET_TEST)
                & (real_policy["policy"] == P0)
            ].iloc[0]
            rows.append(
                {
                    "draw": draw,
                    "permuted_harm_auroc": ranking["harm_auroc"],
                    "real_harm_auroc": _plain(mine["harm_auroc"]),
                    "permuted_oracle_coverage": ranking["oracle_coverage"],
                    "real_oracle_coverage": _plain(mine["oracle_coverage"]),
                    "permuted_p0_safe_coverage": p0["safe_coverage"],
                    "real_p0_safe_coverage": _plain(real_p0["safe_coverage"]),
                    "permuted_p1_accepted": p1["accepted"],
                }
            )
        frame_rows = pd.DataFrame(rows)
        out[direction] = {
            "draws": rows,
            "median_permuted_oracle_coverage": float(
                frame_rows["permuted_oracle_coverage"].median()
            ),
            "median_real_oracle_coverage": float(frame_rows["real_oracle_coverage"].median()),
            "median_permuted_harm_auroc": float(frame_rows["permuted_harm_auroc"].median()),
            "median_real_harm_auroc": float(frame_rows["real_harm_auroc"].median()),
        }
    return out


def a0_reproduction() -> dict[str, Any]:
    """A0 is RK4's zero-shot ranker: identical scores on RK4's own test rows."""
    ours = pd.read_parquet(CELL_SCORES)
    ours = ours[(ours["arm"] == A0) & (ours["evaluation_set"] == SET_EXPOSED)]
    theirs = pd.read_parquet(rk4.CELL_SCORES)
    theirs = theirs[(theirs["arm"] == A0) & (theirs["evaluation_set"] == "test")]
    joined = ours.merge(theirs, on=["direction", "candidate_id"], suffixes=("_pf1", "_rk4"))
    difference = float(np.abs(joined["score_pf1"] - joined["score_rk4"]).max())
    return {
        "rows": len(joined),
        "rk4_rows": len(theirs),
        "max_absolute_difference": difference,
        "equal": bool(len(joined) == len(theirs) == len(ours) and difference == 0.0),
    }


def _plain(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


# ------------------------------------------------------------------ curves and the frontier


def policy_cell_rows(
    table: pd.DataFrame, direction: str, arm: str, scheme: str, budget: str, policy: str, kind: str
) -> pd.DataFrame:
    """One budget's draws for one policy; budget 0 is the zero-shot ranker."""
    d, a, sc, b, _draw = cell_key(direction, arm, scheme, budget, 0)
    return table[
        (table["direction"] == d)
        & (table["arm"] == a)
        & (table["scheme"] == sc)
        & (table["budget"] == b)
        & (table["policy"] == policy)
        & (table["evaluation_set"] == kind)
    ]


def ranking_cell_rows(
    table: pd.DataFrame, direction: str, arm: str, scheme: str, budget: str, kind: str
) -> pd.DataFrame:
    d, a, sc, b, _draw = cell_key(direction, arm, scheme, budget, 0)
    return table[
        (table["direction"] == d)
        & (table["arm"] == a)
        & (table["scheme"] == sc)
        & (table["budget"] == b)
        & (table["evaluation_set"] == kind)
    ]


def ranking_summary(rows: pd.DataFrame) -> dict[str, Any]:
    def median(column: str) -> float | None:
        values = rows[column].dropna().to_numpy(np.float64)
        return float(np.median(values)) if values.size else None

    return {
        "draws": len(rows),
        "median_harm_auroc": median("harm_auroc"),
        "median_benefit_auroc": median("benefit_auroc"),
        "median_top1": median("top1"),
        "median_oracle_coverage": median("oracle_coverage"),
        "median_oracle_automated_recall": median("oracle_automated_recall"),
        "oracle_coverage_iqr": (
            float(
                np.percentile(rows["oracle_coverage"], 75)
                - np.percentile(rows["oracle_coverage"], 25)
            )
            if len(rows)
            else None
        ),
        "decisions": int(rows["decisions"].iloc[0]) if len(rows) else None,
    }


def scheme_budgets(scheme: str) -> tuple[str, ...]:
    return BUDGETS if scheme == SCHEME_RANDOM else ("0", *STRATIFIED_BUDGETS)


def run_curves() -> int:
    """Label-budget curves: ranking, every policy's summary and flags, per set, arm and scheme."""
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(CURVES)
    table = pd.read_parquet(POLICY_OUTCOMES)
    ranking = pd.read_parquet(RANKING_OUTCOMES)
    curves: dict[str, Any] = {}
    for kind in (SET_TEST, SET_EXPOSED):
        for direction in DIRECTIONS:
            for arm in ADAPTED_ARMS:
                for scheme in SCHEMES:
                    if scheme == SCHEME_STRATIFIED and arm != M1:
                        continue
                    for budget in scheme_budgets(scheme):
                        entry = {
                            "ranking": ranking_summary(
                                ranking_cell_rows(ranking, direction, arm, scheme, budget, kind)
                            ),
                            "policies": {},
                        }
                        for policy in POLICIES:
                            rows = policy_cell_rows(
                                table, direction, arm, scheme, budget, policy, kind
                            )
                            summary = summarize(rows)
                            entry["policies"][policy] = summary | flags(summary, policy)
                        curves.setdefault(kind, {}).setdefault(direction, {}).setdefault(
                            arm, {}
                        ).setdefault(scheme, {})[budget] = entry
    pool_pages = cc_read_json(SPLIT_REGISTRY)["pool_pages"]
    _write_json_once(
        CURVES,
        {
            **_analysis_envelope("label_budget_curves"),
            "curves": curves,
            "budgets": list(BUDGETS),
            "pool_pages": pool_pages,
            "rk4_pool_cap": RK4_POOL_CAP,
            "zero_budget_note": "budget 0 is A0 with RK4's source calibration block, every arm",
            "primary": {"evaluation_set": SET_TEST, "arm": M1, "scheme": SCHEME_RANDOM},
        },
    )
    primary = {
        d: curves[SET_TEST][d][M1][SCHEME_RANDOM][POOL]["policies"][P1]["median_safe_coverage"]
        for d in DIRECTIONS
    }
    print(
        f"curves: P1 median safe coverage at the pool {primary} ({time.monotonic() - started:.0f}s)"
    )
    return 0


def frontier_of(
    curves: dict[str, Any],
    direction: str,
    arm: str,
    scheme: str,
    policy: str,
    flag: str,
    chance: set[str],
) -> tuple[str | None, str | None]:
    """N* masked by the random control, and N* before the mask."""
    block = curves[SET_TEST][direction][arm][scheme]
    raw = {b: bool(block[b]["policies"][policy][flag]) for b in block}
    masked = {b: v and f"{direction}|{policy}|{b}" not in chance for b, v in raw.items()}
    return minimal_budget(masked), minimal_budget(raw)


def run_frontier() -> int:
    """The annotation frontier: N* per direction for every policy, flag and arm."""
    started = time.monotonic()
    _require(CURVES, "curves")
    _forbid(FRONTIER)
    curves = cc_read_json(CURVES)["curves"]
    chance = set(cc_read_json(CONTROL_RESULTS)["random_passing_cells"])
    n_star: dict[str, Any] = {}
    unmasked: dict[str, Any] = {}
    for direction in DIRECTIONS:
        for arm in ADAPTED_ARMS:
            for policy in POLICIES:
                names = (
                    ["deployable", *(f"useful_at_{f}" for f in USEFUL_SENSITIVITY)]
                    if policy in FULL_AUTOMATION
                    else [
                        *(f"practical_at_{f}" for f in dep1.REVIEW_REDUCTION_SENSITIVITY),
                        "practical_dep1_definition",
                    ]
                )
                for name in names:
                    masked, raw = frontier_of(
                        curves, direction, arm, SCHEME_RANDOM, policy, name, chance
                    )
                    n_star.setdefault(direction, {}).setdefault(arm, {}).setdefault(policy, {})[
                        name
                    ] = masked
                    unmasked.setdefault(direction, {}).setdefault(arm, {}).setdefault(policy, {})[
                        name
                    ] = raw
    primary: dict[str, Any] = {}
    for direction in DIRECTIONS:
        p1 = n_star[direction][M1][P1][PRIMARY_FLAG[P1]]
        p2 = n_star[direction][M1][P2][PRIMARY_FLAG[P2]]
        reached = [b for b in (p1, p2) if b is not None]
        primary[direction] = {
            "p1_useful": p1,
            "p2_practical": p2,
            "frontier": min(reached, key=BUDGETS.index) if reached else None,
        }
    h1 = n_star[rk4.D_G2Q][M1][P1][PRIMARY_FLAG[P1]]
    _write_json_once(
        FRONTIER,
        {
            **_analysis_envelope("annotation_frontier"),
            "n_star": n_star,
            "n_star_before_the_random_control": unmasked,
            "random_passing_cells": sorted(chance),
            "primary": primary,
            "h1_th2_projection": {
                "claim": HYPOTHESES["H1_th2_projection"],
                "n_star": h1,
                "holds": bool(h1 is not None and BUDGETS.index(h1) <= BUDGETS.index("100")),
            },
            "definitions": cc_read_json(DESIGN_RECORD)["definitions"],
            "measured_budgets": list(BUDGETS),
            "pool_pages": cc_read_json(SPLIT_REGISTRY)["pool_pages"],
        },
    )
    print(f"frontier: {primary} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ RQ2 and RQ3


BREAKDOWN_POLICIES = (P0, P1)


def breakdown(evaluation: Evaluation, direction: str) -> dict[str, Any]:
    """At the pool, M1's accepts by corpus and by engine environment, and by test page."""
    out: dict[str, Any] = {}
    for policy in BREAKDOWN_POLICIES:
        groups: dict[str, dict[str, list[float]]] = {}
        exceeding: list[int] = []
        eligible: list[int] = []
        for draw in range(DRAWS):
            cell = evaluation.cells[(direction, M1, SCHEME_RANDOM, POOL, draw)]
            test = rk4.decisions(cell[SET_TEST])
            calibration = rk4.decisions(cell[SET_CALIBRATION])
            b = dep1.bands(policy, calibration, test, True)
            accepted = test[b["accept"]]
            for column in ("corpus", "environment"):
                for value, block in test.groupby(column, sort=True):
                    mine = accepted[accepted[column] == value]
                    entry = groups.setdefault(f"{column}={value}", {"coverage": [], "harm": []})
                    entry["coverage"].append(len(mine) / max(len(block), 1))
                    if len(mine):
                        entry["harm"].append(float(mine["is_harmful"].mean()))
            per = accepted.groupby("document_id")["is_harmful"].agg(["sum", "size"])
            per = per[per["size"] >= th2.WORST_PAGE_MIN_ACCEPTED]
            eligible.append(len(per))
            exceeding.append(int(((per["sum"] / per["size"]) > EPSILON + HARM_TOLERANCE).sum()))
        out[policy] = {
            "groups": {
                name: {
                    "median_coverage": float(np.median(v["coverage"])),
                    "median_harm": float(np.median(v["harm"])) if v["harm"] else None,
                    "accepting_draws": len(v["harm"]),
                }
                for name, v in sorted(groups.items())
            },
            "pages_with_enough_accepts_median": float(np.median(eligible)),
            "pages_over_epsilon_median": float(np.median(exceeding)),
        }
    return out


def run_diversity() -> int:
    """RQ2: calibration stability, worst-case harm and cross-document robustness."""
    started = time.monotonic()
    _require(FRONTIER, "frontier")
    _forbid(DIVERSITY)
    curves = cc_read_json(CURVES)["curves"][SET_TEST]
    purchases = pd.DataFrame(cc_read_json(PURCHASE_REGISTRY)["rows"])
    evaluation = load_evaluation()
    stability: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    fields = (
        "accept_violation_share",
        "working_share",
        "median_acceptance_coverage",
        "coverage_iqr",
        "harm_spread",
        "worst_draw_harm",
        "median_worst_page_harm",
        "draws_with_a_page_over_epsilon",
    )
    for direction in DIRECTIONS:
        for budget in BUDGETS:
            entry = curves[direction][M1][SCHEME_RANDOM][budget]["policies"]
            stability.setdefault(direction, {})[budget] = {
                policy: {f: entry[policy][f] for f in fields} for policy in FULL_AUTOMATION
            }
        for budget in STRATIFIED_BUDGETS:
            row: dict[str, Any] = {}
            for scheme in SCHEMES:
                entry = curves[direction][M1][scheme][budget]["policies"]
                bought = purchases[
                    (purchases["direction"] == direction)
                    & (purchases["scheme"] == scheme)
                    & (purchases["budget"] == budget)
                ]["environments_bought"]
                row[scheme] = {
                    "median_environments_bought": float(bought.median()),
                    **{policy: {f: entry[policy][f] for f in fields} for policy in FULL_AUTOMATION},
                }
            comparison.setdefault(direction, {})[budget] = row
    robustness = {direction: breakdown(evaluation, direction) for direction in DIRECTIONS}
    _write_json_once(
        DIVERSITY,
        {
            **_analysis_envelope("page_diversity"),
            "stability_by_budget": stability,
            "stratified_vs_random": comparison,
            "cross_document_robustness_at_the_pool": robustness,
            "arm": M1,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"diversity: {len(STRATIFIED_BUDGETS)} matched budgets compared "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def run_bottleneck() -> int:
    """RQ3: the test-oracle cutoff against the rules, per budget, and the pool diagnosis."""
    started = time.monotonic()
    _require(DIVERSITY, "diversity")
    _forbid(BOTTLENECK)
    curves = cc_read_json(CURVES)["curves"][SET_TEST]
    frontier = cc_read_json(FRONTIER)["primary"]
    policy = pd.read_parquet(POLICY_OUTCOMES)
    ranking = pd.read_parquet(RANKING_OUTCOMES)
    by_budget: dict[str, Any] = {}
    diagnosis: dict[str, Any] = {}
    for direction in DIRECTIONS:
        for budget in BUDGETS:
            entry = curves[direction][M1][SCHEME_RANDOM][budget]
            gaps = oracle_gaps(policy, ranking, direction, budget, P1)
            by_budget.setdefault(direction, {})[budget] = {
                "median_harm_auroc": entry["ranking"]["median_harm_auroc"],
                "median_top1": entry["ranking"]["median_top1"],
                "median_oracle_coverage": entry["ranking"]["median_oracle_coverage"],
                "p0_median_safe_coverage": entry["policies"][P0]["median_safe_coverage"],
                "p1_median_safe_coverage": entry["policies"][P1]["median_safe_coverage"],
                "median_oracle_gap_p1": float(np.median(gaps)) if gaps.size else None,
            }
        oracle = by_budget[direction][POOL]["median_oracle_coverage"]
        reached = frontier[direction]["frontier"] is not None
        diagnosis[direction] = {
            "primary": diagnose(reached, oracle, ORACLE_COVERAGE_FLOOR),
            "sensitivity": {str(f): diagnose(reached, oracle, f) for f in ORACLE_SENSITIVITY},
            "pool_oracle_coverage": oracle,
            "reached": reached,
        }
    _write_json_once(
        BOTTLENECK,
        {
            **_analysis_envelope("bottleneck_diagnosis"),
            "by_budget": by_budget,
            "diagnosis": diagnosis,
            "rule": cc_read_json(DESIGN_RECORD)["definitions"]["diagnosis"],
            "oracle_is_analysis_only": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"bottleneck: { ({d: v['primary'] for d, v in diagnosis.items()}) } "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def oracle_gaps(
    policy: pd.DataFrame, ranking: pd.DataFrame, direction: str, budget: str, rule: str
) -> np.ndarray:
    """Per draw: the test-oracle coverage minus the rule's safe coverage."""
    rows = policy_cell_rows(policy, direction, M1, SCHEME_RANDOM, budget, rule, SET_TEST)
    oracle = ranking_cell_rows(ranking, direction, M1, SCHEME_RANDOM, budget, SET_TEST)
    joined = rows.merge(oracle[["draw", "oracle_coverage"]], on="draw")
    return (joined["oracle_coverage"] - joined["safe_coverage"]).to_numpy(np.float64)


# ------------------------------------------------------------------ statistics


def _metric(
    policy: pd.DataFrame,
    ranking: pd.DataFrame,
    direction: str,
    spec: dict[str, Any],
    side: str,
    draw: int,
) -> float:
    budget, scheme = spec[side]
    if spec["metric"] == "harm_auroc":
        rows = ranking_cell_rows(ranking, direction, M1, scheme, budget, SET_TEST)
        rows = rows[rows["draw"] == draw]
        return (
            float(rows.iloc[0]["harm_auroc"])
            if len(rows) and pd.notna(rows.iloc[0]["harm_auroc"])
            else float("nan")
        )
    if spec["metric"] == "oracle_gap":
        rows = policy_cell_rows(policy, direction, M1, scheme, budget, spec["policy"], SET_TEST)
        oracle = ranking_cell_rows(ranking, direction, M1, scheme, budget, SET_TEST)
        joined = rows.merge(oracle[["draw", "oracle_coverage"]], on="draw")
        joined = joined[joined["draw"] == draw]
        if joined.empty:
            return float("nan")
        return float(joined.iloc[0]["oracle_coverage"] - joined.iloc[0]["safe_coverage"])
    rows = policy_cell_rows(policy, direction, M1, scheme, budget, spec["policy"], SET_TEST)
    rows = rows[rows["draw"] == draw]
    if rows.empty or pd.isna(rows.iloc[0][spec["metric"]]):
        return float("nan")
    return float(rows.iloc[0][spec["metric"]])


def draw_comparison(
    name: str, spec: dict[str, Any], policy: pd.DataFrame, ranking: pd.DataFrame
) -> dict[str, Any]:
    """Paired over draws (and directions): exact sign test, direction-stratified bootstrap."""
    a: list[float] = []
    b: list[float] = []
    strata_of: list[str] = []
    for direction in spec["directions"]:
        for draw in range(DRAWS):
            left = _metric(policy, ranking, direction, spec, "left", draw)
            right = _metric(policy, ranking, direction, spec, "right", draw)
            if np.isfinite(left) and np.isfinite(right):
                a.append(left)
                b.append(right)
                strata_of.append(direction)
    left_values, right_values = np.asarray(a), np.asarray(b)
    labels = np.asarray(strata_of)
    shares = spec["metric"] == "accept_violation"

    def effect(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        return float(x.mean() - y.mean()) if shares else float(np.median(x - y))

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero(labels == d) for d in sorted(set(labels.tolist()))]
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for resample in range(BOOTSTRAP_RESAMPLES):
        index = (
            np.concatenate([generator.choice(s, s.size, replace=True) for s in strata])
            if strata
            else np.empty(0, dtype=np.int64)
        )
        draws[resample] = effect(left_values[index], right_values[index])
    p, positive, negative = th1.sign_test(left_values - right_values)
    finite = draws[np.isfinite(draws)]
    return {
        "comparison": name,
        "directions": list(spec["directions"]),
        "metric": spec["metric"],
        "policy": spec["policy"],
        "left": list(spec["left"]),
        "right": list(spec["right"]),
        "units": int(left_values.size),
        "effect": effect(left_values, right_values) if left_values.size else None,
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else None,
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else None,
        "p_value": p,
        "draws_left_higher": positive,
        "draws_right_higher": negative,
        "ties": int(left_values.size - positive - negative),
    }


def run_stats() -> int:
    started = time.monotonic()
    _require(BOTTLENECK, "bottleneck")
    _forbid(STATISTICAL_TESTS)
    policy = pd.read_parquet(POLICY_OUTCOMES)
    ranking = pd.read_parquet(RANKING_OUTCOMES)
    family = {
        name: draw_comparison(name, FAMILY_SPEC[name], policy, ranking) for name in PRIMARY_FAMILY
    }
    adjusted = s15.holm(family)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "family": adjusted,
            "family_size": len(adjusted),
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "plan": cc_read_json(DESIGN_RECORD)["statistical_plan"],
        },
    )
    surviving = sum(1 for v in adjusted.values() if v["survives_holm"])
    print(f"stats: {surviving}/{len(adjusted)} survive Holm ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ falsification


def toy_policy_frames() -> tuple[
    pd.DataFrame, pd.DataFrame, tuple[set[str], dict[str, set[str]], pd.DataFrame]
]:
    """Hand-built decisions for the lost-repair share: six test sites, four of them exact.

    On the toy calibration, RK1's plug-in rule accepts safety >= 7.0 (above it no harm; at 6.0
    harm is 0.25) and the mirrored rule rejects safety <= 2.0 (below it no exact repair; at 3.0
    the exact share is 0.25). The exact test decisions sit at 9.0 (accepted), 5.0 (review), 2.0
    and 1.0 (rejected): four reachable repairs, two lost, a share of 0.5.
    """
    calibration = pd.DataFrame(
        {
            "safety": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5],
            "is_harmful": [False, False, False, True, True, True, False, False, False, False],
            "exact": [True, True, True, False, False, False, True, False, False, False],
            "site_key": [f"c|{i}" for i in range(10)],
            "document_id": [f"cal{i % 3}" for i in range(10)],
        }
    )
    test = pd.DataFrame(
        {
            "safety": [9.0, 5.0, 4.0, 2.0, 1.0, 0.5],
            "is_harmful": [False, False, True, False, False, True],
            "exact": [True, True, False, True, True, False],
            "beneficial": [True, True, False, True, True, False],
            "site_key": [f"t|{i}" for i in range(6)],
            "document_id": ["p0", "p0", "p1", "p1", "p2", "p2"],
            "candidate_id": [f"k{i}" for i in range(6)],
        }
    )
    keys = {f"e|{i}" for i in range(6)}
    mapping = {f"t|{i}": {f"e|{i}"} for i in range(6)}
    errors = pd.DataFrame({"error_key": sorted(keys)})
    return calibration, test, (keys, mapping, errors)


def _labelled_test_cell() -> tuple[
    pd.DataFrame, pd.DataFrame, tuple[set[str], dict[str, set[str]], pd.DataFrame]
]:
    evaluation = load_evaluation()
    cell = evaluation.cells[(rk4.D_G2Q, M1, SCHEME_RANDOM, POOL, 0)]
    return (
        rk4.decisions(cell[SET_CALIBRATION]),
        rk4.decisions(cell[SET_TEST]),
        evaluation.context[SET_TEST],
    )


def run_negative() -> int:
    """The falsification suite: each test would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    pool = load_pool()
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    pages = set(pool["document_id"].astype(str))
    record(
        "f01_pool_excludes_the_generator_universe",
        not pages & u5_documents(),
        len(pages & u5_documents()),
    )
    fitted = set().union(*resource_documents().values())
    record("f02_pool_excludes_resource_pages", not pages & fitted, len(pages & fitted))
    manifest = cc_read_json(REPO / "manifests/sgv1/role_manifest.json")
    lock = cc_read_json(REPO / "manifests/sgv1/confirmatory_reserve_lock.json")
    record(
        "f03_no_confirmatory_page",
        not pages & set(manifest["role_of"]) and lock["status"] == "LOCKED",
        {"cord_pages": len(pages & set(manifest["role_of"])), "lock": lock["status"]},
    )
    record(
        "f04_one_reading_per_page",
        pool["page_id"].is_unique and pool["image_sha256"].is_unique,
        {"pages": len(pool), "environments": int(pool["environment"].nunique())},
    )
    refolded: dict[str, int] = {}
    for _corpus, block in pool.groupby("corpus", sort=True):
        refolded.update(
            assign_folds(dict(zip(block["document_id"], block["partition_group"], strict=True)))
        )
    record(
        "f05_folds_are_content_blind",
        all(refolded[d] == f for d, f in zip(pool["document_id"], pool["fold"], strict=True)),
        "folds recomputed within each corpus from partition-group ids alone, as --pool assigns",
    )
    study = load_study()
    documents = study.population["document_id"].astype(str).to_numpy()
    leaks = 0
    for b in study.blocks.values():
        tested = set(documents[b.test].tolist())
        leaks += len(tested & set(documents[b.source_fit].tolist()))
        for scheme in SCHEMES:
            for label in scheme_budgets(scheme)[1:]:
                for draw in range(DRAWS):
                    fit, cal = purchase(b.pages, draw, label, scheme, study.environment_of)
                    leaks += len(tested & set(fit + cal))
    record("f06_test_pages_never_fit_or_calibrate", leaks == 0, leaks)
    prefix = True
    for b in study.blocks.values():
        for draw in range(DRAWS):
            previous: list[str] = []
            for label in ADAPTED_BUDGETS:
                fit, cal = purchase(b.pages, draw, label, SCHEME_RANDOM, study.environment_of)
                bought = page_order(b.pages, draw, SCHEME_RANDOM, study.environment_of)[
                    : len(fit) + len(cal)
                ]
                prefix &= bought[: len(previous)] == previous and set(bought) == set(fit + cal)
                previous = bought
    record("f07_purchases_are_prefixes", prefix, "every budget buys a prefix of the next")
    toy = page_order(
        ["a1", "a2", "a3", "b1", "b2", "c1"],
        0,
        SCHEME_STRATIFIED,
        {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "c1": "C"},
    )
    record(
        "f08_stratified_order_round_robins",
        len({p[0] for p in toy[:3]}) == 3 and len({p[0] for p in toy[3:5]}) == 2,
        toy,
    )
    record(
        "f09_chain_replays_rebuild_u5",
        cc_read_json(REPLAY)["all_equal"] and cc_read_json(LABEL_REPLAY)["all_equal"],
        {
            "gt_blind_tables": len(cc_read_json(REPLAY)["checks"]["gt_blind"]),
            "labelled_tables": len(cc_read_json(LABEL_REPLAY)["checks"]),
        },
    )
    configuration = cc_read_json(FROZEN_CONFIGURATION)
    shard_hashes = [
        cc_read_json(_sidecar(shard_path(a, n, c)))["generation_hashes"] for a, n, c in shards()
    ]
    record(
        "f10_generator_identical_to_lp1_hy1",
        all(configuration["identical_to_upstream"].values())
        and all(h == configuration["generation_hashes"] for h in shard_hashes),
        {"shards": len(shard_hashes)},
    )
    design = cc_read_json(DESIGN_RECORD)["issued_utc"]
    labelled = cc_read_json(CANDIDATE_INVENTORY)["issued_utc"]
    record(
        "f11_design_frozen_before_any_label",
        design <= labelled and DESIGN_RECORD.stat().st_mtime < CANDIDATES.stat().st_mtime,
        {"design_record": design, "first_label": labelled},
    )
    strata = pd.read_parquet(PROPOSAL_STRATA)
    raw, _manifest = verified_shards(CORRECTOR)
    asked = {f"{r.split('|')[0]}|{r.split('|')[2]}" for r in raw["request_id"].astype(str)}
    routed = routed_to(strata, C1)
    contexts = pd.read_parquet(CORRECTION_CONTEXTS)
    cropped = set(contexts[contexts["crop_path"].notna()]["site_id"].astype(str))
    record(
        "f12_corrector_decoded_exactly_the_h5_sites",
        asked == routed & cropped,
        {"requests": len(asked), "routed_with_a_crop": len(routed & cropped)},
    )
    controls = cc_read_json(CONTROL_RESULTS)
    record(
        "f13_a0_reproduces_rk4",
        controls["a0_reproduces_rk4"]["equal"],
        controls["a0_reproduces_rk4"],
    )
    calibration, test, context = _labelled_test_cell()
    plug_in = rk1.choose_threshold(calibration, EPSILON)
    row = policy_row(P0, calibration, test, context, True)
    record("f14_p0_uses_the_plug_in_cutoff", row["accept_cutoff"] == plug_in, {"cutoff": plug_in})
    shuffled = test.assign(
        is_harmful=np.random.default_rng(1).permutation(test["is_harmful"].to_numpy()),
        exact=np.random.default_rng(2).permutation(test["exact"].to_numpy()),
    )
    blind = all(
        policy_row(policy, calibration, test, context, True)["accept_cutoff"]
        == policy_row(policy, calibration, shuffled, context, True)["accept_cutoff"]
        and policy_row(policy, calibration, test, context, True)["reject_cutoff"]
        == policy_row(policy, calibration, shuffled, context, True)["reject_cutoff"]
        for policy in POLICIES
    )
    record(
        "f15_cutoffs_blind_to_test_labels", blind, "test labels permuted; every cutoff unchanged"
    )
    toy_calibration, toy_test, toy_context = toy_policy_frames()
    toy_row = policy_row(P3, toy_calibration, toy_test, toy_context, True)
    record(
        "f16_lost_repair_share_hand_computed",
        toy_row["lost_repair_share"] == 0.5
        and toy_row["accept_cutoff"] == 7.0
        and toy_row["reject_cutoff"] == 2.0,
        {k: toy_row[k] for k in ("accept_cutoff", "reject_cutoff", "lost_repair_share")},
    )
    record(
        "f17_useful_and_practical_hand_computed",
        useful(0.1, 0.5, 0.05)
        and not useful(0.1, 0.5, 0.049)
        and not useful(0.15, 0.9, 0.5)
        and practical(0.1, 0.1, 0.25, 0.1)
        and not practical(0.1, 0.1, 0.9, 0.11)
        and dep1.practical(0.1, 0.1, 0.9),
        "boundaries at 0.05 coverage and 0.1 lost-repair share",
    )
    record(
        "f18_minimal_budget_hand_computed",
        minimal_budget(
            {"0": False, "5": True, "10": False, "25": True, "50": True, "100": True, POOL: True}
        )
        == "25"
        and minimal_budget(dict.fromkeys(BUDGETS, False)) is None,
        "N* is the start of the unbroken run that reaches the pool",
    )
    source_columns = [c for c in rk3.columns_for(rk3.R5) if c.startswith(rl1.FAM_SOURCE)]
    record("f19_no_generator_identity_in_the_model", not source_columns, source_columns)
    population = pd.read_parquet(POPULATION)
    record(
        "f20_grades_are_rk1s",
        bool((population["grade"].to_numpy() == rk1.relevance(population)).all()),
        {"candidates": len(population)},
    )
    u5_ids = set(rk3.load_population()["candidate_id"])
    record(
        "f21_candidates_and_pages_disjoint_from_u5",
        not set(population["candidate_id"]) & u5_ids
        and not set(population["document_id"]) & u5_documents(),
        "PF1 ids and pages never collide with U5",
    )
    frontier = cc_read_json(FRONTIER)
    chance = set(frontier["random_passing_cells"])
    clean = True
    for direction in DIRECTIONS:
        for policy in POLICIES:
            n = frontier["n_star"][direction][M1][policy][PRIMARY_FLAG[policy]]
            if n is None:
                continue
            clean &= not any(
                f"{direction}|{policy}|{b}" in chance for b in BUDGETS[BUDGETS.index(n) :]
            )
    record("f22_random_control_masks_the_frontier", clean, sorted(chance))
    curves = cc_read_json(CURVES)["curves"]
    recomputed = {}
    for direction in DIRECTIONS:
        p1, _raw = frontier_of(curves, direction, M1, SCHEME_RANDOM, P1, PRIMARY_FLAG[P1], chance)
        p2, _raw = frontier_of(curves, direction, M1, SCHEME_RANDOM, P2, PRIMARY_FLAG[P2], chance)
        recomputed[direction] = {"p1_useful": p1, "p2_practical": p2}
    record(
        "f23_frontier_reads_policy_flags_only",
        all(
            recomputed[d][k] == frontier["primary"][d][k]
            for d in DIRECTIONS
            for k in ("p1_useful", "p2_practical")
        ),
        recomputed,
    )
    passed = sum(1 for t in tests if t["passed"])
    _write_json_once(
        FALSIFICATION,
        {
            **_analysis_envelope("falsification_tests"),
            "tests": tests,
            "passed": passed,
            "total": len(tests),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    elapsed = time.monotonic() - started
    print(f"negative: {passed}/{len(tests)} falsification tests pass ({elapsed:.0f}s)")
    return 0 if passed == len(tests) else 1


# ------------------------------------------------------------------ the decision


def criteria_from_artifacts() -> dict[str, Any]:
    frontier = cc_read_json(FRONTIER)
    bottleneck = cc_read_json(BOTTLENECK)["diagnosis"]
    stats = cc_read_json(STATISTICAL_TESTS)
    falsification = cc_read_json(FALSIFICATION)
    primary = frontier["primary"]
    diagnosis = {d: bottleneck[d]["primary"] for d in DIRECTIONS}
    outcome = assign_outcome({d: primary[d]["frontier"] for d in DIRECTIONS}, diagnosis)
    return {
        "outcome": outcome,
        "frontier": primary,
        "diagnosis": diagnosis,
        "h1_th2_projection_holds": frontier["h1_th2_projection"]["holds"],
        "stats_surviving": stats["surviving"],
        "falsification_passed": falsification["passed"],
        "falsification_total": falsification["total"],
    }


def run_decide() -> int:
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    criteria = criteria_from_artifacts()
    outcome = criteria["outcome"]
    all_pass = criteria["falsification_passed"] == criteria["falsification_total"]
    stats = cc_read_json(STATISTICAL_TESTS)["family"]
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "outcome": outcome,
            "outcome_label": OUTCOME_TAXONOMY[outcome],
            "recommended_next_stage": NEXT_STAGE[outcome],
            "criteria": criteria,
            "hypotheses": {
                "H1_th2_projection": criteria["h1_th2_projection_holds"],
                "H2_diversity": bool(
                    "T3_p0_violation_stratified_vs_random_50" in criteria["stats_surviving"]
                    and (stats["T3_p0_violation_stratified_vs_random_50"]["effect"] or 0) < 0
                ),
                "H3_evidence": bool(
                    "T4_oracle_gap_100_vs_25" in criteria["stats_surviving"]
                    and (stats["T4_oracle_gap_100_vs_25"]["effect"] or 0) < 0
                ),
            },
            "pages_generated": int(load_pool()["page_id"].nunique()),
            "human_reviewer": "idealized, not measured",
            "deployment_claim_permitted": bool(outcome in ("A", "B") and all_pass),
            "deployment_claim_scope": (
                "development pages only: a conservative policy met its pre-registered criteria on "
                "fresh development test pages; nothing is certified or production-ready"
            ),
            "ready_for_external_confirmation": False,
            "production_ready": False,
            "certified": False,
            "confirmatory_reserve_consumed": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"decide: outcome {outcome}, next {NEXT_STAGE[outcome]} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = rk1.FIGURE_NOTE + "; development pages, SGV-PF1"
SURFACE = rk1.SURFACE
INK = rk1.INK
INK_SECONDARY = rk1.INK_SECONDARY
GRID = rk1.GRID
POLICY_COLOURS = dep1.POLICY_COLOURS
ARM_COLOURS = {A0: "#8a8a8a", M1: "#2a78d6", B2: "#eb6834"}
ORACLE_COLOUR = "#6b6b6b"
FIGURES = (
    "fig1_label_budget_frontier.png",
    "fig2_ranking_quality.png",
    "fig3_threshold_reliability.png",
    "fig4_bottleneck.png",
    "fig5_triage_tradeoffs.png",
)
FIGURE_CAPTIONS = {
    FIGURES[0]: (
        "Median safe coverage of M1's test decisions by labelled page budget under P0 and P1, "
        "with the analysis-only test-oracle cutoff; the dotted line is RK4's pool."
    ),
    FIGURES[1]: (
        "Median harm AUROC and top-1 exact share on the fresh test pages by labelled page budget, "
        "for M1 and target-only B2 (budget 0 is A0)."
    ),
    FIGURES[2]: (
        "Accept-violation share over 20 draws by page budget for P0 and P1, random against "
        "environment-stratified page purchase; the dotted line is TH1's 0.1 ceiling."
    ),
    FIGURES[3]: (
        "What the ranking could automate at harm 0.1 if the cutoff were read off the test labels, "
        "against what P1 automates with a cutoff chosen on calibration pages."
    ),
    FIGURES[4]: (
        "Three-way triage: the median share of decisions no human reviews and the median share of "
        "reachable repairs the reject band discards, by page budget."
    ),
}


def run_figures() -> int:
    """Five figures, every value read from a persisted artifact."""
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
            "svg.hashsalt": "pf1",
        }
    )
    manifest: dict[str, Any] = {}

    def save(fig: Any, name: str, sources: Sequence[Path], caption: str) -> None:
        fig.text(0.01, 0.005, FIGURE_NOTE, fontsize=5.5, color=INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(
            FIGURE_DIR / name,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata={"Software": None},
        )
        plt.close(fig)
        manifest[name] = {
            "path": f"{FIGURE_DIR.name}/{name}",
            "caption": caption,
            "sources": {s.name: xr1._signature(s) for s in sources},
        }

    curves = cc_read_json(CURVES)
    pools = curves["pool_pages"]
    table = curves["curves"][SET_TEST]

    def pages(direction: str, budgets: Sequence[str] = BUDGETS) -> list[int]:
        return [budget_pages(b, pools[direction]) if b != "0" else 0 for b in budgets]

    def value(entry: dict[str, Any], field: str) -> float:
        return np.nan if entry.get(field) is None else float(entry[field])

    def policy_series(
        direction: str, arm: str, scheme: str, policy: str, field: str
    ) -> list[float]:
        block = table[direction][arm][scheme]
        return [value(block[b]["policies"][policy], field) for b in scheme_budgets(scheme)]

    def ranking_series(direction: str, arm: str, field: str) -> list[float]:
        block = table[direction][arm][SCHEME_RANDOM]
        return [value(block[b]["ranking"], field) for b in BUDGETS]

    # Figure 1: what each labelled page buys in safe automation, against the test-oracle ceiling.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for axis, direction in zip(axes, DIRECTIONS, strict=True):
        x = pages(direction)
        axis.plot(
            x,
            ranking_series(direction, M1, "median_oracle_coverage"),
            color=ORACLE_COLOUR,
            linestyle="--",
            marker="o",
            markersize=3,
            label="test-oracle cutoff (analysis only)",
        )
        for policy in (P0, P1):
            axis.plot(
                x,
                policy_series(direction, M1, SCHEME_RANDOM, policy, "median_safe_coverage"),
                color=POLICY_COLOURS[policy],
                marker="o",
                markersize=3,
                label=dep1.POLICY_SHORT[policy],
            )
        axis.axvline(RK4_POOL_CAP[direction], color=INK_SECONDARY, linewidth=0.8, linestyle=":")
        axis.text(
            RK4_POOL_CAP[direction],
            axis.get_ylim()[1] * 0.95 if axis.get_ylim()[1] else 0.1,
            " RK4 pool",
            fontsize=6,
            color=INK_SECONDARY,
            va="top",
        )
        axis.set_title(DIRECTION_SHORT[direction])
        axis.set_xlabel("labelled target pages")
    axes[0].set_ylabel("median safe coverage of test decisions")
    axes[0].legend(frameon=False, fontsize=6)
    save(
        fig,
        FIGURES[0],
        (CURVES,),
        FIGURE_CAPTIONS[FIGURES[0]],
    )

    # Figure 2: ranking quality by budget and arm.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex="col")
    for column, direction in enumerate(DIRECTIONS):
        x = pages(direction)
        for arm in ADAPTED_ARMS:
            axes[0][column].plot(
                x,
                ranking_series(direction, arm, "median_harm_auroc"),
                color=ARM_COLOURS[arm],
                marker="o",
                markersize=3,
                label=arm,
            )
            axes[1][column].plot(
                x,
                ranking_series(direction, arm, "median_top1"),
                color=ARM_COLOURS[arm],
                marker="o",
                markersize=3,
                label=arm,
            )
        axes[0][column].set_title(DIRECTION_SHORT[direction])
        axes[1][column].set_xlabel("labelled target pages")
    axes[0][0].set_ylabel("median harm AUROC")
    axes[1][0].set_ylabel("median top-1 exact share")
    axes[0][0].legend(frameon=False, fontsize=6)
    save(
        fig,
        FIGURES[1],
        (CURVES,),
        FIGURE_CAPTIONS[FIGURES[1]],
    )

    # Figure 3: threshold reliability, random against stratified purchase.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for axis, direction in zip(axes, DIRECTIONS, strict=True):
        x = pages(direction)
        xs = pages(direction, scheme_budgets(SCHEME_STRATIFIED))
        for policy in (P0, P1):
            axis.plot(
                x,
                policy_series(direction, M1, SCHEME_RANDOM, policy, "accept_violation_share"),
                color=POLICY_COLOURS[policy],
                marker="o",
                markersize=3,
                label=f"{dep1.POLICY_SHORT[policy]}, random pages",
            )
            axis.plot(
                xs,
                policy_series(direction, M1, SCHEME_STRATIFIED, policy, "accept_violation_share"),
                color=POLICY_COLOURS[policy],
                marker="s",
                markersize=3,
                linestyle="--",
                label=f"{dep1.POLICY_SHORT[policy]}, stratified pages",
            )
        axis.axhline(VIOLATION_CEILING, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
        axis.set_title(DIRECTION_SHORT[direction])
        axis.set_xlabel("labelled target pages")
    axes[0].set_ylabel("share of draws whose accepts exceed harm 0.1")
    axes[0].legend(frameon=False, fontsize=6)
    save(
        fig,
        FIGURES[2],
        (CURVES,),
        FIGURE_CAPTIONS[FIGURES[2]],
    )

    # Figure 4: the bottleneck -- oracle coverage against the conservative rule's.
    bottleneck = cc_read_json(BOTTLENECK)["by_budget"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for axis, direction in zip(axes, DIRECTIONS, strict=True):
        positions = np.arange(len(BUDGETS))
        oracle = [value(bottleneck[direction][b], "median_oracle_coverage") for b in BUDGETS]
        p1 = [value(bottleneck[direction][b], "p1_median_safe_coverage") for b in BUDGETS]
        axis.bar(
            positions - 0.2, oracle, width=0.4, color=ORACLE_COLOUR, label="test-oracle cutoff"
        )
        axis.bar(
            positions + 0.2, p1, width=0.4, color=POLICY_COLOURS[P1], label=dep1.POLICY_SHORT[P1]
        )
        axis.set_xticks(positions, [str(v) for v in pages(direction)])
        axis.set_title(DIRECTION_SHORT[direction])
        axis.set_xlabel("labelled target pages")
    axes[0].set_ylabel("median share of test decisions accepted safely")
    axes[0].legend(frameon=False, fontsize=6)
    save(
        fig,
        FIGURES[3],
        (BOTTLENECK,),
        FIGURE_CAPTIONS[FIGURES[3]],
    )

    # Figure 5: three-way triage.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for axis, direction in zip(axes, DIRECTIONS, strict=True):
        x = pages(direction)
        for policy in THREE_WAY:
            axis.plot(
                x,
                policy_series(direction, M1, SCHEME_RANDOM, policy, "median_review_reduction"),
                color=POLICY_COLOURS[policy],
                marker="o",
                markersize=3,
                label=f"{dep1.POLICY_SHORT[policy]}: review reduction",
            )
            axis.plot(
                x,
                policy_series(direction, M1, SCHEME_RANDOM, policy, "median_lost_repair_share"),
                color=POLICY_COLOURS[policy],
                marker="x",
                markersize=3,
                linestyle="--",
                label=f"{dep1.POLICY_SHORT[policy]}: lost-repair share",
            )
        axis.axhline(REVIEW_REDUCTION_FLOOR, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
        axis.set_title(DIRECTION_SHORT[direction])
        axis.set_xlabel("labelled target pages")
    axes[0].set_ylabel("median over draws")
    axes[0].legend(frameon=False, fontsize=6)
    save(
        fig,
        FIGURES[4],
        (CURVES,),
        FIGURE_CAPTIONS[FIGURES[4]],
    )

    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_envelope("figure_manifest"),
            "figures": manifest,
            "note": FIGURE_NOTE,
            "synthetic": False,
        },
    )
    print(f"figures: {len(manifest)} written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ reproducibility


def run_spotcheck() -> int:
    """A whole proposer shard and a whole corrector shard regenerated and compared answer for
    answer. Batched bf16 decoding is reproducible per shard, so whole shards are compared."""
    started = time.monotonic()
    _forbid(SPOTCHECK)
    work = [next(w for w in shards() if w[0] == arm) for arm in GENERATION_ARMS]
    import transformers

    transformers.utils.logging.set_verbosity_error()
    device = gen1._resolve_device()
    runner = gen1._runner(MODEL, device)
    rows: list[dict[str, Any]] = []
    for arm, name, chunk in work:
        frozen = pd.read_parquet(shard_path(arm, name, chunk)).set_index("request_id")
        pages = chunk_pages(name, chunk)
        requests = (
            proposer_requests(name, pages) if arm == PROPOSER else corrector_requests(name, pages)
        )
        again, _cost = _decode(runner, arm, requests)
        again = again.set_index("request_id")
        shared = sorted(set(frozen.index) & set(again.index))
        differing = [r for r in shared if str(frozen.at[r, "output"]) != str(again.at[r, "output"])]
        rows.append(
            {
                "arm": arm,
                "environment": name,
                "chunk": chunk,
                "requests": len(frozen),
                "regenerated": len(again),
                "differing_answers": len(differing),
                "identical": bool(len(shared) == len(frozen) == len(again) and not differing),
            }
        )
    del runner
    gen1._release_device_cache(device)
    _write_json_once(
        SPOTCHECK,
        {
            **_envelope("generation_spotcheck"),
            "generation_cost": generation_cost(),
            "shards": rows,
            "all_identical": all(r["identical"] for r in rows),
            "rule": "whole shards only; a re-batched slice tests batch layout, not reproducibility",
            "allocator_environment": gen1._allocator_environment(),
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"spotcheck: {rows} ({time.monotonic() - started:.0f}s)")
    return 0


def generation_cost() -> dict[str, Any]:
    """Every arm's decoding cost, summed from its shard sidecars.

    The monotonic clock excludes machine sleep but not a paused process or swapping: the first
    proposer shard's time includes both.
    """
    out: dict[str, Any] = {}
    for arm in GENERATION_ARMS:
        records = [cc_read_json(_sidecar(shard_path(a, n, c))) for a, n, c in shards() if a == arm]
        seconds = float(sum(r["wall_clock_seconds"] for r in records))
        requests = int(sum(r["requests"] for r in records))
        out[arm] = {
            "shards": len(records),
            "requests": requests,
            "wall_clock_hours": round(seconds / 3600.0, 2),
            "requests_per_second": round(requests / seconds, 4) if seconds else None,
            "unfinished_decodes": int(sum(r.get("unfinished_decodes", 0) for r in records)),
        }
    return out


DERIVED_OUTPUTS = (
    "SPLIT_REGISTRY",
    "PURCHASE_REGISTRY",
    "CELL_SCORES",
    "ADAPTATION_REGISTRY",
    "POLICY_OUTCOMES",
    "RANKING_OUTCOMES",
    "POLICY_REGISTRY",
    "CONTROL_RESULTS",
    "CURVES",
    "FRONTIER",
    "DIVERSITY",
    "BOTTLENECK",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "DECISION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)
LABELLED_CHAIN_OUTPUTS = ("CANDIDATES", "SITE_TRUTH", "ERROR_SITES", "PROPOSAL_ERROR_LINKS")


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("splits", run_splits),
        ("adapt", run_adapt),
        ("policies", run_policies),
        ("controls", run_controls),
        ("curves", run_curves),
        ("frontier", run_frontier),
        ("diversity", run_diversity),
        ("bottleneck", run_bottleneck),
        ("stats", run_stats),
        ("negative", run_negative),
        ("decide", run_decide),
        ("figures", run_figures),
    )


def _frame_signature(frame: pd.DataFrame) -> str:
    ordered = _round_trip(frame)
    ordered = ordered.sort_values(list(ordered.columns[:3]), kind="stable").reset_index(drop=True)
    return str(canonical_hash(ordered.to_json(orient="records", default_handler=str)))


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


def labelled_chain_signatures() -> dict[str, str]:
    """The labelled chain rebuilt from the frozen raw answers, as frame signatures."""
    raw, _manifest = rekeyed_corrector_answers()
    strata = pd.read_parquet(PROPOSAL_STRATA)
    p0 = pd.read_parquet(P0_PROPOSALS)
    candidates, truth, _census = normalize_all(strata, raw, p0)
    errors, links = link_all(pd.read_parquet(SITE_LATTICE), p0)
    return {
        CANDIDATES.name: _frame_signature(candidates),
        SITE_TRUTH.name: _frame_signature(truth),
        ERROR_SITES.name: _frame_signature(errors),
        PROPOSAL_ERROR_LINKS.name: _frame_signature(links),
    }


def run_determinism() -> int:
    started = time.monotonic()
    _require(FIGURE_MANIFEST, "figures")
    _forbid(DETERMINISM)
    published = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(n for n in published if any(r.get(n) != published[n] for r in runs))
    chain_published = {
        globals()[n].name: _frame_signature(pd.read_parquet(globals()[n]))
        for n in LABELLED_CHAIN_OUTPUTS
    }
    chain_again = labelled_chain_signatures()
    chain_differing = sorted(n for n in chain_published if chain_again.get(n) != chain_published[n])
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "derived_artifacts_compared": len(published),
            "derived_regenerations": len(runs),
            "differing_artifacts": differing,
            "labelled_chain_tables_compared": len(chain_published),
            "labelled_chain_differing": chain_differing,
            "all_derived_artifacts_identical": not differing,
            "all_runs_identical": not differing and not chain_differing,
            "seeds": {
                "page": PAGE_SEED,
                "bootstrap": BOOTSTRAP_SEED,
                "control": CONTROL_SEED,
                "pool": POOL_SEED,
                "fold": FOLD_SEED,
            },
            "outcome": cc_read_json(DECISION)["outcome"],
            "uses_ground_truth": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"determinism: {len(published)} artifacts x {len(runs)} regenerations and "
        f"{len(chain_published)} chain tables, identical {not differing and not chain_differing} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    PAGE_POOL,
    PAGE_REGISTRY,
    SITE_LATTICE,
    SCAN_POPULATION,
    P0_PROPOSALS,
    LATTICE_INVENTORY,
    LINE_CONTEXTS,
    GENERATION_PLAN,
    REPLAY,
    LABEL_REPLAY,
    PARSED_PROPOSALS,
    PROPOSAL_COMPLIANCE,
    PROPOSAL_STRATA,
    ROUTING_REGISTRY,
    CORRECTION_CONTEXTS,
    DESIGN_RECORD,
    CANDIDATES,
    SITE_TRUTH,
    ERROR_SITES,
    PROPOSAL_ERROR_LINKS,
    CANDIDATE_INVENTORY,
    POPULATION,
    POPULATION_INVENTORY,
    PAGE_ANNOTATION,
    FEATURE_MATRIX,
    FEATURE_REGISTRY,
    SPOTCHECK,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE, FROZEN_CONFIGURATION),
    "2. Research Questions and Pre-Registration": (DESIGN_RECORD,),
    "3. A Data-Generation Stage": (PAGE_REGISTRY, GENERATION_PLAN, FROZEN_CONFIGURATION),
    "4. The New Page Pool": (PAGE_REGISTRY, LATTICE_INVENTORY, POPULATION_INVENTORY),
    "5. The Frozen Chain, Replayed": (REPLAY, LABEL_REPLAY),
    "6. Generation and Labelling": (
        GENERATION_PLAN,
        PROPOSAL_COMPLIANCE,
        ROUTING_REGISTRY,
        CANDIDATE_INVENTORY,
        POPULATION_INVENTORY,
        SPOTCHECK,
    ),
    "7. Study Design": (DESIGN_RECORD, SPLIT_REGISTRY, ADAPTATION_REGISTRY),
    "8. RQ1: Ranking Quality": (CURVES, STATISTICAL_TESTS),
    "9. RQ1: Threshold Reliability and Safe Coverage": (CURVES, FRONTIER),
    "10. The Annotation Frontier": (FRONTIER, CURVES, CONTROL_RESULTS),
    "11. RQ2: Page Diversity": (DIVERSITY, STATISTICAL_TESTS, CURVES),
    "12. RQ3: Ranking or Evidence": (BOTTLENECK, CURVES, STATISTICAL_TESTS),
    "13. Three-Way Triage": (CURVES, FRONTIER),
    "14. Controls and Falsification": (CONTROL_RESULTS, FALSIFICATION),
    "15. Statistical Tests": (STATISTICAL_TESTS,),
    "16. Limitations": (DESIGN_RECORD, PAGE_REGISTRY, CURVES, DETERMINISM, SPOTCHECK),
    "17. Research Decision": (DECISION, FRONTIER, BOTTLENECK, FALSIFICATION),
    "18. Next Research Direction": (DECISION, BOTTLENECK, CURVES),
}


def run_record() -> int:
    started = time.monotonic()
    for path in (PROVENANCE, TRACEABILITY):
        _forbid(path)
    missing = [p for p in PRODUCED if not p.exists() and p not in (PROVENANCE, TRACEABILITY)]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {[_relative(p) for p in missing][:5]}")
    if not cc_read_json(DETERMINISM)["all_runs_identical"]:
        raise PhaseError("determinism did not pass; the record will not be issued")
    if not cc_read_json(SPOTCHECK)["all_identical"]:
        raise PhaseError(
            "the generation spot-check did not reproduce; the record will not be issued"
        )
    freeze = cc_read_json(RESEARCH_FREEZE)
    moved = sorted(
        p for p, sha in freeze["upstream_sha256"].items() if file_sha256(REPO / p) != sha
    )
    if moved:
        raise PhaseError(f"{len(moved)} upstream files moved since the freeze: {moved[:4]}")
    global DECISION
    saved = DECISION
    rederived: list[bool] = []
    for run in range(2):
        directory = CACHE / "record" / f"run_{run}"
        directory.mkdir(parents=True, exist_ok=True)
        sandbox = directory / saved.name
        sandbox.unlink(missing_ok=True)
        try:
            DECISION = sandbox
            run_decide()
        finally:
            DECISION = saved
        again = cc_read_json(sandbox)
        published = cc_read_json(saved)
        rederived.append(
            all(
                again[k] == published[k]
                for k in published
                if k not in ("issued_utc", "runtime_seconds")
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
    raw_shards = sorted(RAW_CACHE.glob("*.parquet"))
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "artifacts": {_relative(p): file_sha256(p) for p in PRODUCED if p.is_file()},
            "artifact_count": sum(1 for p in PRODUCED if p.is_file()),
            "raw_generation_shards": {_relative(p): file_sha256(p) for p in raw_shards},
            "figures": {
                f"figures/{p.name}": file_sha256(p) for p in sorted(FIGURE_DIR.glob("*.png"))
            },
            "upstream_files_rehashed": len(freeze["upstream_sha256"]),
            "upstream_files_moved": moved,
            "decision_rederived_identically": rederived,
            "regenerates_ocr": False,
            "generates_new_candidates": True,
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
            "sections": {n: [_relative(p) for p in ps] for n, ps in REPORT_SECTIONS.items()},
            "audit": audit,
            "untraceable": audit["untraceable"],
            "rule": (
                "every number in the report is read from one of the artifacts its section names, "
                "and from the field that means what the sentence says"
            ),
        },
    )
    print(
        f"record: {sum(1 for p in PRODUCED if p.is_file())} artifacts, report claims "
        f"{audit['numeric_claims']}, untraceable {audit['untraceable_numeric_claims']}"
    )
    return 0


# ------------------------------------------------------------------ the replay on U5 pages


def u5_pages() -> pd.DataFrame:
    """LP1's frozen (environment, page, rank) rows: the pages the generator universe holds."""
    sample = pd.read_parquet(xr1.PROPOSAL_SAMPLE)
    ranks = sample.groupby(["environment", "document_id"])["document_rank"].first().reset_index()
    return ranks.astype({"environment": str, "document_id": str, "document_rank": int})


def _round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame as parquet would store it, so a rebuilt table compares like a frozen one."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    buffer.seek(0)
    return pd.read_parquet(buffer)


def _cells(series: pd.Series) -> list[Any]:
    """Comparable cell values: array cells as tuples, missing as None, everything else as is."""
    out: list[Any] = []
    for value in series.tolist():
        if isinstance(value, (list, tuple, np.ndarray)):
            out.append(tuple(np.asarray(value).tolist()))
        elif value is None or (isinstance(value, float) and np.isnan(value)):
            out.append(None)
        else:
            out.append(value)
    return out


def _same(left: pd.DataFrame, right: pd.DataFrame, key: Sequence[str]) -> dict[str, Any]:
    """Two frames equal cell for cell on every shared column, after a parquet round trip."""
    columns = sorted(set(left.columns) & set(right.columns))
    missing = sorted(set(right.columns) ^ set(left.columns))
    a = _round_trip(left[columns]).sort_values(list(key), kind="stable").reset_index(drop=True)
    b = _round_trip(right[columns]).sort_values(list(key), kind="stable").reset_index(drop=True)
    if len(a) != len(b):
        return {"rows": [len(a), len(b)], "equal": False, "columns_only_in_one": missing}
    differing = [c for c in columns if _cells(a[c]) != _cells(b[c])]
    return {
        "rows": len(a),
        "equal": not differing,
        "differing_columns": differing,
        "columns_only_in_one": missing,
    }


def replay_gt_blind() -> dict[str, Any]:
    """PF1's GT-blind chain on the U5 pages, against the frozen LP1 tables."""
    pages = u5_pages()
    pipeline = s14.build_pipeline()
    lattice, lines, p0, _inventory = build_lattice(pages, pipeline)
    frozen_lattice = pd.read_parquet(lp1.SITE_LATTICE_INVENTORY)
    frozen_lines = pd.read_parquet(lp1.SCAN_POPULATION)
    frozen_p0 = pd.read_parquet(lp1.P0_PROPOSALS)
    checks = {
        "lattice": _same(lattice, frozen_lattice, ["site_id"]),
        "scan_population": _same(
            lines.drop(columns=["page_image_path"]),
            frozen_lines.drop(columns=["page_image_path"]),
            ["line_uid"],
        ),
        "p0_proposals": _same(p0, frozen_p0, ["site_id"]),
    }
    contexts = build_contexts_for_lines(lines, REPLAY_DIR / "line_crops")
    frozen_contexts = pd.read_parquet(lp1.LINE_CONTEXTS)
    checks["line_contexts"] = _same(
        contexts.drop(columns=["crop_path"]),
        frozen_contexts.drop(columns=["crop_path"]),
        ["line_uid"],
    )
    requests = proposer_request_frame(lines, contexts)
    raw = pd.concat(
        [pd.read_parquet(lp1._shard_path(lp1.L2, s["environment"])) for s in environment_specs()],
        ignore_index=True,
    )
    checks["proposer_prompts"] = prompts_match(requests, lp1.SYSTEM_PROMPT, raw)
    frozen_parsed = pd.read_parquet(lp1.PARSED_PROPOSALS)
    budgets = cc_read_json(lp1.DESIGN_RECORD)["budget"]["p0_budget_by_environment"]
    parsed, _census = parse_answers(raw, lines, p0, budgets)
    subset = ["on_shuffle_subset", "in_subset_budget"]
    for method in (lp1.L0, lp1.L2):
        checks[f"parsed_{method}"] = _same(
            parsed[parsed["method"] == method],
            frozen_parsed[frozen_parsed["method"] == method].drop(columns=subset),
            ["site_id"],
        )
    strata = build_strata_frame(lattice, p0, parsed)
    frozen_strata = pd.read_parquet(hy1.PROPOSAL_STRATA)
    frozen_strata = frozen_strata[frozen_strata["stratum"].isin(hy1.STRATA)]
    # The text control's own columns, and the tie-break ranks it can lend a P0-only site, are the
    # only fields the control touches; U5 never reads them.
    control = ["in_text", "ht_incremental", "text_suspicion", "hash_order", "document_rank"]
    checks["strata"] = _same(
        strata.drop(columns=control), frozen_strata.drop(columns=control), ["site_id"]
    )
    contexts = build_correction_contexts(strata, REPLAY_DIR / "correction_crops")
    frozen_contexts = pd.read_parquet(hy1.CONTEXTS)
    frozen_contexts = frozen_contexts[frozen_contexts["site_id"].isin(set(strata["site_id"]))]
    checks["correction_contexts"] = _same(
        contexts.drop(columns=["crop_path", "document_id"]),
        frozen_contexts.drop(columns=["crop_path"]),
        ["site_id"],
    )
    checks["corrector_prompts"] = replay_corrector_prompts(strata, contexts)
    return checks


def frozen_corrector_answers(name: str) -> pd.DataFrame:
    """HY1's image-corrector answers for one environment: its own, plus XR1's reused ones."""
    reuse = pd.read_parquet(hy1.CACHE / "cache_reuse_map.parquet")
    reused = set(reuse[reuse["corrector"] == C1]["site_id"].astype(str))
    by_lattice = hy1._lattice_of_p0()
    mapping = {by_lattice[s]: s for s in reused if s in by_lattice and s.startswith(f"{name}|")}
    return pd.concat(
        [hy1._reused_raw(C1, name, mapping), hy1._generated_raw(C1, name)], ignore_index=True
    )


def replay_corrector_prompts(strata: pd.DataFrame, contexts: pd.DataFrame) -> dict[str, Any]:
    """Every rebuilt image-corrector prompt against the prompt hash its frozen answer records."""
    demanded = set(hy1.corrector_demand(strata)[C1])
    frame = contexts[contexts["site_id"].isin(demanded) & contexts["crop_path"].notna()]
    requests = hy1_style_requests(frame.reset_index(drop=True), strata, hy1.NEW_IMAGE)
    requests["request_id"] = requests["request_id"].str.replace(
        f"|{hy1.NEW_IMAGE}|", f"|{C1}|", regex=False
    )
    raw = pd.concat(
        [frozen_corrector_answers(s["environment"]) for s in environment_specs()],
        ignore_index=True,
    )
    return prompts_match(requests, gen1.SYSTEM_PROMPT, raw)


def run_replay() -> int:
    """PF1's chain over the 49 frozen U5 pages must rebuild the frozen tables before any decode."""
    started = time.monotonic()
    _require(FROZEN_CONFIGURATION, "freeze")
    _forbid(REPLAY)
    checks = {"gt_blind": replay_gt_blind()}
    failed = [name for name, row in checks["gt_blind"].items() if not row["equal"]]
    if failed:
        raise PhaseError(f"the chain does not rebuild U5: {failed} {checks}")
    _write_json_once(
        REPLAY,
        {
            **_envelope("chain_replay"),
            "pages": len(u5_pages()),
            "checks": checks,
            "all_equal": True,
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    tables = sum(len(v) for v in checks.values())
    print(f"replay: {tables} tables rebuilt exactly ({time.monotonic() - started:.0f}s)")
    return 0


def _replay_table(name: str, frame: pd.DataFrame) -> Path:
    """A replay intermediate: a scratch table the upstream builders read through rebinding."""
    path = REPLAY_DIR / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def replay_labelled() -> dict[str, Any]:
    """PF1's labelled chain on the U5 pages from their frozen raw answers, against the frozen
    HY1, LP1, RL1, RK1 and RK3 tables. Only U5's own, long-published labels are read."""
    pages = u5_pages()
    pipeline = s14.build_pipeline()
    lattice, lines, p0, _inventory = build_lattice(pages, pipeline)
    raw_l2 = pd.concat(
        [pd.read_parquet(lp1._shard_path(lp1.L2, s["environment"])) for s in environment_specs()],
        ignore_index=True,
    )
    budgets = cc_read_json(lp1.DESIGN_RECORD)["budget"]["p0_budget_by_environment"]
    parsed, _census = parse_answers(raw_l2, lines, p0, budgets)
    strata = build_strata_frame(lattice, p0, parsed)
    contexts = build_correction_contexts(strata, REPLAY_DIR / "correction_crops")
    raw_c1 = pd.concat(
        [frozen_corrector_answers(s["environment"]) for s in environment_specs()],
        ignore_index=True,
    )
    candidates, truth, _rows = normalize_all(strata, raw_c1, p0)
    frozen = pd.read_parquet(hy1.CANDIDATES)
    kept = (frozen["corrector_source"] == C0) | (
        (frozen["corrector_source"] == C1)
        & frozen["lattice_site_id"].astype(str).isin(routed_to(strata, C1))
    )
    checks: dict[str, Any] = {
        "candidates": _same(candidates, frozen[kept], ["candidate_id"]),
    }
    frozen_truth = pd.read_parquet(hy1.SITE_TRUTH)
    frozen_truth = frozen_truth[frozen_truth["stratum"].isin(hy1.STRATA)]
    checks["site_truth"] = _same(truth, frozen_truth, ["site_id"])
    errors, links = link_all(lattice, p0)
    checks["error_sites"] = _same(errors, pd.read_parquet(lp1.ERROR_SITES), ["align_site_id"])
    checks["error_links"] = _same(
        links,
        pd.read_parquet(lp1.PROPOSAL_ERROR_LINKS),
        ["environment", "site_id", "align_site_id"],
    )
    paths = {
        "candidates": _replay_table("candidates", candidates),
        "strata": _replay_table("strata", strata),
        "truth": _replay_table("site_truth", truth),
        "errors": _replay_table("error_sites", errors),
        "links": _replay_table("links", links),
        "contexts": _replay_table("contexts", contexts),
    }
    population = build_population_frame(
        paths["candidates"], paths["strata"], paths["truth"], paths["errors"], paths["links"]
    )
    frozen_population = pd.read_parquet(rl1.CANDIDATE_POPULATION)
    frozen_population = frozen_population[frozen_population["population"] == rl1.U5]
    checks["population"] = _same(population, frozen_population, ["candidate_id"])
    folds = {str(k): int(v) for k, v in cc_read_json(rl1.GROUP_REGISTRY)["document_folds"].items()}
    ranked = with_ranking_columns(
        population.assign(population=rk1.POP_U5), folds, {f: ds1.block_of(f) for f in range(FOLDS)}
    )
    frozen_ranked = rk3.load_population()
    checks["ranking_population"] = _same(
        ranked[list(frozen_ranked.columns)], frozen_ranked, ["candidate_id"]
    )
    population_path = _replay_table("population", ranked)
    matrix = build_feature_matrix(
        ranked, population_path, paths["candidates"], paths["strata"], paths["contexts"]
    )
    frozen_matrix = pd.read_parquet(rk3.FEATURE_MATRIX).set_index("candidate_id")
    mine = matrix.set_index("candidate_id").loc[frozen_matrix.index]
    columns = rk3.columns_for(rk3.R5)
    difference = float(
        np.abs(
            mine[columns].to_numpy(np.float64) - frozen_matrix[columns].to_numpy(np.float64)
        ).max()
    )
    checks["feature_matrix"] = {
        "rows": len(frozen_matrix),
        "columns": len(columns),
        "max_absolute_difference": difference,
        "equal": bool(difference == 0.0 and len(matrix) == len(frozen_matrix)),
    }
    return checks


def run_replay_labels() -> int:
    """PF1's labelled chain over U5 must rebuild HY1's candidates, LP1's error sites, RL1's U5,
    RK1's grades and RK3's R5 matrix exactly before a new page is labelled."""
    started = time.monotonic()
    _require(REPLAY, "replay")
    _forbid(LABEL_REPLAY)
    checks = replay_labelled()
    failed = [name for name, row in checks.items() if not row["equal"]]
    if failed:
        raise PhaseError(f"the labelled chain does not rebuild U5: {failed} {checks}")
    _write_json_once(
        LABEL_REPLAY,
        {
            **_analysis_envelope("label_replay"),
            "pages": len(u5_pages()),
            "checks": checks,
            "all_equal": True,
            "reads_new_page_labels": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"replay-labels: {len(checks)} tables rebuilt exactly ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ entry point


def main(argv: Sequence[str] | None = None) -> int:
    phases: dict[str, Callable[[], int]] = {
        "reconstruct": run_reconstruct,
        "freeze": run_freeze,
        "pool": run_pool,
        "lattice": run_lattice,
        "lines": run_lines,
        "plan": run_plan,
        "replay": run_replay,
        "replay-labels": run_replay_labels,
        "generate": run_generate,
        "parse": run_parse,
        "strata": run_strata,
        "contexts": run_contexts,
        "preregister": run_preregister,
        "normalize": run_normalize,
        "link": run_link,
        "population": run_population,
        "features": run_features,
        "splits": run_splits,
        "adapt": run_adapt,
        "policies": run_policies,
        "controls": run_controls,
        "curves": run_curves,
        "frontier": run_frontier,
        "diversity": run_diversity,
        "bottleneck": run_bottleneck,
        "stats": run_stats,
        "negative": run_negative,
        "decide": run_decide,
        "figures": run_figures,
        "spotcheck": run_spotcheck,
        "determinism": run_determinism,
        "record": run_record,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    for name in phases:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(argv)
    chosen = [name for name in phases if getattr(args, name.replace("-", "_"))]
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
